#!/usr/bin/env bash
# Validate a running server with llm-inference-bench v0.6.2.
#
#   ./validate.sh quick    estonia long-context answer test, 10 runs at concurrency 8   (~5 min)
#   ./validate.sh decode   decode tok/s at concurrency 1 and 12, empty context, 30 s     (~3 min)
#   ./validate.sh full     estonia 30 runs + LAVD 30 runs, both at concurrency 30        (~10 min)
#   ./validate.sh sweep    the standard decode sweep: concurrency 1-32 x context 0-512K  (~40 min)
#
# Results go to results/<mode>-<UTC time>/ (JSON + log). Environment / .env:
#   BENCH_DIR   checkout of llm-inference-bench at tag v0.6.2 (default ./llm-inference-bench)
#   PYTHON      interpreter with httpx, rich and psutil (default python3)
#   HOST        default 127.0.0.1;  PORT default $HOST_PORT;  MODEL default $SERVED_MODEL_NAME
#   KV_BUDGET   KV cache tokens for decode/sweep (default: read from the container log)
set -euo pipefail

REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=tools/common.sh
. "$REPO_DIR/tools/common.sh"

mode=${1:-}
case $mode in
  quick | decode | full | sweep) ;;
  *) die "usage: $0 quick|decode|full|sweep" ;;
esac

load_env optional
require_cmd curl
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-${HOST_PORT:-}}
MODEL=${MODEL:-${SERVED_MODEL_NAME:-GLM-5.3-Flash-Kraken-TP3}}
PYTHON=${PYTHON:-python3}
BENCH_DIR=${BENCH_DIR:-./llm-inference-bench}
[[ $BENCH_DIR == /* ]] || BENCH_DIR="$REPO_DIR/$BENCH_DIR"
bench="$BENCH_DIR/llm_decode_bench.py"
[[ -n $PORT ]] || die "Set HOST_PORT in .env (or PORT=...)."

# The benchmark offers to upgrade itself at startup; we pin v0.6.2 and always feed it
# </dev/null so that prompt can never be answered.
bench_version() { sed -nE 's/^VERSION = "([^"]+)"/\1/p' "$bench" | head -1; }
[[ -f $bench ]] || die "llm-inference-bench not found at $BENCH_DIR. Clone it:
       git clone --branch v0.6.2 --depth 1 https://github.com/local-inference-lab/llm-inference-bench.git \"$BENCH_DIR\""
[[ $(bench_version) == 0.6.2 ]] || die "$bench is version $(bench_version), expected 0.6.2."
"$PYTHON" -c 'import httpx, rich, psutil' 2>/dev/null ||
  die "$PYTHON lacks httpx/rich/psutil. Example: python3 -m venv .venv && .venv/bin/pip install httpx rich psutil && PYTHON=.venv/bin/python $0 $mode"

curl -fsS --max-time 5 "http://$HOST:$PORT/v1/models" | grep -qF "\"$MODEL\"" ||
  die "No server serving model '$MODEL' at http://$HOST:$PORT (is it running? see ./launch.sh)."

kv_budget() {
  if [[ -n ${KV_BUDGET:-} ]]; then
    echo "$KV_BUDGET"
    return
  fi
  local line
  line=$(docker logs "$CONTAINER" 2>&1 | grep -E 'GPU KV cache size:' | tail -1 || true)
  [[ -n $line ]] || die "Could not read the KV cache size from 'docker logs $CONTAINER'; set KV_BUDGET=<tokens>."
  sed -E 's/.*GPU KV cache size: ([0-9,]+) tokens.*/\1/' <<<"$line" | tr -d ,
}

out="$REPO_DIR/results/$mode-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$out"
common=(--host "$HOST" --port "$PORT" --model "$MODEL" --display-mode plain)

run() { # name, args...
  local name=$1
  shift
  echo "[$(date -u +%H:%M:%S)] $name: $out/$name.log"
  local rc=0
  "$PYTHON" -u "$bench" "${common[@]}" "$@" --output "$out/$name.json" </dev/null >"$out/$name.log" 2>&1 || rc=$?
  [[ $(bench_version) == 0.6.2 ]] || die "The benchmark changed version during the run; restore v0.6.2."
  ((rc == 0)) || die "$name failed (exit $rc); see $out/$name.log"
}

case $mode in
  quick)
    run estonia10 --test-profile estonia --profile-concurrency 8 --profile-runs 10 --max-tokens 40000
    ;;
  full)
    run estonia30 --test-profile estonia --profile-concurrency 30 --profile-runs 30 --max-tokens 40000
    run lavd30 --test-profile lavd --profile-concurrency 30 --profile-runs 30 --max-tokens 100000
    ;;
  decode)
    kv=$(kv_budget)
    run decode --concurrency 1,12 --contexts 0 --duration 30 --max-tokens 8192 \
      --decode-warmup-seconds 5 --token-targeting exact --kv-budget "$kv" --dcp-size 3 --no-resume
    ;;
  sweep)
    kv=$(kv_budget)
    run sweep --concurrency 1,2,4,8,12,16,32 --contexts 0,8k,16k,32k,64k,128k,256k,512k \
      --duration 30 --max-tokens 8192 --decode-warmup-seconds 5 --cell-warmup-timeout-seconds 1800 \
      --token-targeting exact --kv-budget "$kv" --dcp-size 3 --no-resume
    ;;
esac
"$PYTHON" -I "$REPO_DIR/tools/summarize.py" "$out"/*.json
