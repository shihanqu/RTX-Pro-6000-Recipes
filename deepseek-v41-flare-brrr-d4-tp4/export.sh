#!/usr/bin/env bash
set -euo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/tools/common.sh"

select_gpus
require_host_ram
python3 "$RECIPE_DIR/tools/checkpoint.py" "$MODEL_DIR" --sizes-only
mkdir -p "$WORK_DIR"
free_kib=$(df -Pk "$WORK_DIR" | awk 'END { print $4 }')
[[ $free_kib =~ ^[0-9]+$ && $free_kib -ge $((1200 * 1024 * 1024)) ]] ||
    fail 'Provide at least 1,200 GiB free in WORK_DIR after downloading the checkpoint.'
umask 077
mkdir -p "$WORK_DIR/settings" "$WORK_DIR/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/$MODEL_REVISION"
docker image inspect "$PREP_IMAGE" >/dev/null || fail 'Build the preparation image with ./build.sh first.'
compose=(docker compose -f "$RECIPE_DIR/compose.export.yaml")
"${compose[@]}" config --quiet
"${compose[@]}" run --rm --no-deps export
