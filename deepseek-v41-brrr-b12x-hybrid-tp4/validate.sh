#!/usr/bin/env bash
set -euo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/tools/common.sh"
exec "$PYTHON" "$RECIPE_DIR/tools/validate.py" "${1:-quality}"
