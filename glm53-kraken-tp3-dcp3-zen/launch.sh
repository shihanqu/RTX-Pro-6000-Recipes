#!/usr/bin/env bash
# Start GLM-5.3-Flash (TP3 / DCP3 / MTP3) on three RTX PRO 6000 Blackwell GPUs.
#
#   ./launch.sh            check everything, start the container, wait until it serves
#   ./launch.sh config     print the resolved compose config and exit (no GPU use)
#   ./launch.sh check      run only the pre-start checks (port, GPUs, checkpoint, image)
#   ./launch.sh stop       stop the container (docker stop, 90 s grace)
#   ./launch.sh logs       follow the container logs
#
# Settings come from .env (see .env.example). Environment overrides:
#   READY_TIMEOUT_S=1800   how long to wait for /health (first start compiles kernels: ~10 min)
#   FOLLOW_LOGS=1          follow the logs after the server is ready
set -euo pipefail

REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=tools/common.sh
. "$REPO_DIR/tools/common.sh"

action=${1:-start}
case $action in
  start | config | check | stop | logs) ;;
  *) die "usage: $0 [start|config|check|stop|logs]" ;;
esac

load_env required
require_cmd docker jq curl
docker compose version >/dev/null 2>&1 || die "Docker Compose v2 ('docker compose') is required."

compose() {
  docker compose -p "$PROJECT" -f "$REPO_DIR/compose.yaml" --env-file "$REPO_DIR/.env" "$@"
}

if [[ $action == stop ]]; then
  docker stop -t 90 "$CONTAINER"
  exit 0
elif [[ $action == logs ]]; then
  exec docker logs -f "$CONTAINER"
fi

# ---- settings from .env ----
[[ -n ${GPU_IDS:-} ]] || die "GPU_IDS is not set in .env (three GPU indices, e.g. \"0,1,2\")."
[[ -n ${HOST_PORT:-} ]] || die "HOST_PORT is not set in .env."
[[ -n ${HF_CACHE_DIR:-} ]] || die "HF_CACHE_DIR is not set in .env."
[[ -n ${RUNTIME_CACHE_DIR:-} ]] || die "RUNTIME_CACHE_DIR is not set in .env."
[[ $HOST_PORT =~ ^[0-9]+$ ]] || die "HOST_PORT must be a number."

IFS=', ' read -r -a gpus <<<"$GPU_IDS"
[[ ${#gpus[@]} -eq 3 ]] || die "GPU_IDS must list exactly three GPUs (got '$GPU_IDS')."
for gpu in "${gpus[@]}"; do
  [[ $gpu =~ ^[0-9]+$ ]] || die "GPU_IDS entries must be nvidia-smi indices (got '$gpu')."
done
export GPU_ID_1=${gpus[0]} GPU_ID_2=${gpus[1]} GPU_ID_3=${gpus[2]}

if [[ -z ${CPUSET:-} ]]; then
  CPUSET="0-$(($(nproc) - 1))"
fi
export CPUSET HOST_PORT HF_CACHE_DIR RUNTIME_CACHE_DIR

select_checkpoint
export CHECKPOINT MOE_TP=$CKPT_MOE_TP

if [[ $action == config ]]; then
  compose config
  exit 0
fi

# ---- pre-start checks ----
require_cmd nvidia-smi ss
if ! jq -e --arg repo "$CKPT_REPO" --arg rev "$CKPT_REVISION" --argjson n "$CKPT_FILES" \
  --argjson bytes "$CKPT_BYTES" \
  '(.repo == $repo) and (.revision == $rev) and (.files | length == $n) and (.total_bytes == $bytes)' \
  "$CHECKPOINT_RECEIPT" >/dev/null 2>&1; then
  die "Checkpoint not verified: no matching receipt at $CHECKPOINT_RECEIPT
       (want $CKPT_REPO @ $CKPT_REVISION, $CKPT_FILES files). Run ./download_checkpoint.sh first."
fi
[[ -f $CKPT_SNAPSHOT/config.json ]] || die "Checkpoint snapshot missing: $CKPT_SNAPSHOT"

docker image inspect "$IMAGE" >/dev/null 2>&1 || die "Image not present. Pull it first (12.9 GB):
       docker pull $IMAGE"

if [[ $(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null || true) == running ]]; then
  die "$CONTAINER is already running. Stop it first: ./launch.sh stop"
fi
if ss -ltnH "sport = :$HOST_PORT" | grep -q .; then
  die "Port $HOST_PORT is already in use. Free it or change HOST_PORT in .env."
fi
for gpu in "${gpus[@]}"; do
  name=$(nvidia-smi -i "$gpu" --query-gpu=name --format=csv,noheader 2>/dev/null) ||
    die "GPU $gpu not found by nvidia-smi."
  if [[ $name != *"RTX PRO 6000"* ]]; then
    echo "WARNING: GPU $gpu is '$name'; this recipe was tested on RTX PRO 6000 Blackwell 96 GB." >&2
  fi
  pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)
  if [[ -n $pids ]]; then
    die "GPU $gpu has compute processes ($(echo $pids)); stop them first."
  fi
done
echo "Checks passed: checkpoint $CHECKPOINT ($CKPT_REVISION), image present, port $HOST_PORT free, GPUs $GPU_ID_1,$GPU_ID_2,$GPU_ID_3 idle."
if [[ $action == check ]]; then
  exit 0
fi

# ---- start ----
mkdir -p "$RUNTIME_CACHE_DIR" "$REPO_DIR/logs"
expected_sizes=$(compose config --format json |
  jq -r '.services[].environment.CUDAGRAPH_CAPTURE_SIZES | split(" ") | "[" + join(", ") + "]"')
t0=$SECONDS
compose up -d --force-recreate

logfile="$REPO_DIR/logs/startup-$(date -u +%Y%m%dT%H%M%SZ).log"
deadline=$((SECONDS + ${READY_TIMEOUT_S:-1800}))
echo "Waiting for http://127.0.0.1:$HOST_PORT/health (6-12 min; the first start compiles kernels and takes longest) ..."
while ((SECONDS < deadline)); do
  if curl -fsS --max-time 2 "http://127.0.0.1:$HOST_PORT/health" >/dev/null 2>&1; then
    docker logs "$CONTAINER" >"$logfile" 2>&1 || true
    if ! grep -qE "Using \\['PYNCCL'\\] all-reduce backends.*group 'tp:0'" "$logfile"; then
      die "Expected PyNCCL tensor-parallel all-reduce was not observed in the startup log ($logfile)."
    fi
    if ! grep -qF "$CKPT_FP8_LINE" "$logfile"; then
      die "Expected FP8 decode path ('$CKPT_FP8_LINE') was not observed in the startup log ($logfile)."
    fi
    if ! grep -qF "'cudagraph_capture_sizes': $expected_sizes" "$logfile"; then
      die "CUDA graph ladder $expected_sizes was not observed in the startup log ($logfile)."
    fi
    kv=$(grep -E 'GPU KV cache size:' "$logfile" | tail -1 | sed -E 's/.*(GPU KV cache size:)/\1/' || true)
    echo "Ready after $((SECONDS - t0)) s: http://127.0.0.1:$HOST_PORT/v1  (model: ${SERVED_MODEL_NAME:-GLM-5.3-Flash-Kraken-TP3})"
    echo "CUDA graph sizes: $expected_sizes"
    echo "$kv"
    echo "Startup log: $logfile"
    if [[ ${FOLLOW_LOGS:-0} == 1 ]]; then
      exec docker logs -f --since 1s "$CONTAINER"
    fi
    exit 0
  fi
  state=$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null || echo missing)
  if [[ $state != running ]]; then
    docker logs "$CONTAINER" >"$logfile" 2>&1 || true
    tail -n 80 "$logfile" >&2
    die "Container stopped during startup (state: $state). Full log: $logfile"
  fi
  sleep 5
done
docker logs "$CONTAINER" >"$logfile" 2>&1 || true
tail -n 80 "$logfile" >&2
die "Timed out after ${READY_TIMEOUT_S:-1800} s waiting for the server. Full log: $logfile"
