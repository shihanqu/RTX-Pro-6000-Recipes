#!/usr/bin/env bash
set -euo pipefail
RECIPE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -f $RECIPE_DIR/.env ]]; then
  set -a
  . "$RECIPE_DIR/.env"
  set +a
fi
PYTHON=${PYTHON:-$RECIPE_DIR/.venv/bin/python}
[[ -x $PYTHON ]] || { echo "Create .venv and install requirements.txt first (see README)." >&2; exit 1; }
export RECIPE_DIR
