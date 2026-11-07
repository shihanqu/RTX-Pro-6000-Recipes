#!/usr/bin/env bash
set -euo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/tools/common.sh"

mkdir -p "$MODEL_DIR"
hf download "$MODEL_REPO" --revision "$MODEL_REVISION" --local-dir "$MODEL_DIR"
python3 "$RECIPE_DIR/tools/checkpoint.py" "$MODEL_DIR"
