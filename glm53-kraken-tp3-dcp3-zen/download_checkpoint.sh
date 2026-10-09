#!/usr/bin/env bash
# Download and verify the pinned checkpoint into $HF_CACHE_DIR (from .env), then write the
# receipt that launch.sh checks.
#
#   ./download_checkpoint.sh             download + verify (asks you to accept the licence)
#   ./download_checkpoint.sh --verify    only re-verify files already on disk and rewrite the receipt
#
# Default checkpoint (CHECKPOINT=csf-qad-2):
#   local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD @ dec48abd33efa73c3bb7c95b74eee10cad34f9be
#   49 files, 177,750,845,553 bytes (~178 GB), Local Inference Lab License 1.0.
#
# Environment:
#   ACCEPT_LICENSE=yes   skip the interactive licence prompt
#   FETCH_WITH=host|docker
#                        host: run tools/fetch_checkpoint.py with the host's python3 (needs
#                        huggingface_hub); docker: run it inside the pinned serving image (CPU only).
#                        Default: host if huggingface_hub imports, else docker.
#   HF_TOKEN             optional; the repository is public and not gated.
set -euo pipefail

REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=tools/common.sh
. "$REPO_DIR/tools/common.sh"

verify_only=
case ${1:-} in
  "") ;;
  --verify) verify_only=--verify-only ;;
  *) die "usage: $0 [--verify]" ;;
esac

load_env required
[[ -n ${HF_CACHE_DIR:-} ]] || die "HF_CACHE_DIR is not set in .env."
select_checkpoint

cat <<EOF
Checkpoint : $CKPT_REPO
Revision   : $CKPT_REVISION (pinned)
Size       : $CKPT_FILES files, $(human_gb "$CKPT_BYTES") ($CKPT_BYTES bytes)
Destination: $CKPT_SNAPSHOT
Receipt    : $CHECKPOINT_RECEIPT
Licence    : $CKPT_LICENSE
             https://huggingface.co/$CKPT_REPO/blob/$CKPT_REVISION/LICENSE

EOF

if [[ -z $verify_only ]]; then
  if [[ ${ACCEPT_LICENSE:-} != yes ]]; then
    [[ -t 0 ]] || die "Read the licence above, then rerun with ACCEPT_LICENSE=yes (stdin is not a terminal)."
    read -r -p "Have you read and do you accept this licence? Type 'yes' to download: " answer
    [[ $answer == yes ]] || die "Not accepted; nothing downloaded."
  fi
  # Free-space check on the filesystem that will hold the files (5% margin).
  probe=$HF_CACHE_DIR
  while [[ ! -d $probe ]]; do probe=$(dirname -- "$probe"); done
  avail=$(df -PB1 -- "$probe" | awk 'NR == 2 { print $4 }')
  have=0
  [[ -d $CKPT_SNAPSHOT ]] && have=$(du -sLB1 -- "$CKPT_SNAPSHOT" | awk '{ print $1 }')
  need=$((CKPT_BYTES * 105 / 100 - have))
  if ((avail < need)); then
    die "Not enough free space under $probe: $(human_gb "$avail") free, about $(human_gb "$need") needed."
  fi
fi

mkdir -p "$HF_CACHE_DIR" "$(dirname -- "$CHECKPOINT_RECEIPT")"
args=(--repo "$CKPT_REPO" --revision "$CKPT_REVISION" --expect-files "$CKPT_FILES"
  --expect-bytes "$CKPT_BYTES" ${verify_only:+"$verify_only"})

mode=${FETCH_WITH:-}
if [[ -z $mode ]]; then
  if python3 -c 'import huggingface_hub' >/dev/null 2>&1; then mode=host; else mode=docker; fi
fi
case $mode in
  host)
    python3 -I "$REPO_DIR/tools/fetch_checkpoint.py" "${args[@]}" \
      --cache-dir "$HF_CACHE_DIR" --receipt "$CHECKPOINT_RECEIPT"
    ;;
  docker)
    docker image inspect "$IMAGE" >/dev/null 2>&1 || die "Neither huggingface_hub on the host nor the image is available.
       Either: pip install huggingface_hub   or: docker pull $IMAGE"
    receipt_dir=$(cd -- "$(dirname -- "$CHECKPOINT_RECEIPT")" && pwd)
    docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HUB_DISABLE_TELEMETRY=1 \
      ${HF_TOKEN:+-e HF_TOKEN} \
      -v "$HF_CACHE_DIR:/hf" -v "$receipt_dir:/receipt" -v "$REPO_DIR/tools:/tools:ro" \
      --entrypoint /opt/venv/bin/python "$IMAGE" -I /tools/fetch_checkpoint.py "${args[@]}" \
      --cache-dir /hf --receipt "/receipt/$(basename -- "$CHECKPOINT_RECEIPT")"
    ;;
  *) die "FETCH_WITH must be host or docker" ;;
esac
echo "Checkpoint ready. Next: ./launch.sh"
