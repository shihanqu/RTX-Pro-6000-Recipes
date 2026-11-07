#!/usr/bin/env bash
set -euo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/tools/common.sh"

[[ $HOST_PORT =~ ^[0-9]{1,5}$ ]] || fail 'Invalid HOST_PORT.'
HOST_PORT=$((10#$HOST_PORT))
((HOST_PORT >= 1 && HOST_PORT <= 65535)) || fail 'HOST_PORT must be in 1..65535.'
listeners=$(ss -Hltn "sport = :$HOST_PORT") || fail 'Cannot check port use.'
[[ -z $listeners ]] || fail "Port $HOST_PORT is already occupied. Nothing was stopped."
select_gpus
require_host_ram

if [[ -z ${PACKAGE_DIR:-} ]]; then
    parent="$WORK_DIR/packages/models/deepseek-ai_2FDeepSeek-V4.1-Flash-1017"
    [[ -d $parent ]] || fail 'Export a local package first.'
    mapfile -d '' manifests < <(find "$parent" -mindepth 2 -maxdepth 2 -type f -name model.safetensors -print0)
    [[ ${#manifests[@]} -eq 1 ]] || fail 'Set PACKAGE_DIR to one exported TP4 package directory.'
    PACKAGE_DIR=$(dirname -- "${manifests[0]}")
fi
PACKAGE_DIR=$(cd -- "$PACKAGE_DIR" && pwd -P)
[[ -s $PACKAGE_DIR/model.safetensors ]] || fail 'Missing local package manifest.'
PUBLISHER_PROFILE="$WORK_DIR/settings/development-publisher.json"
PUBLISHER_FINGERPRINT=$(python3 "$RECIPE_DIR/tools/publisher_fingerprint.py" "$PUBLISHER_PROFILE")
mkdir -p "$WORK_DIR/serve-state"
export PACKAGE_DIR PUBLISHER_PROFILE PUBLISHER_FINGERPRINT

compose=(docker compose -f "$RECIPE_DIR/compose.yaml")
"${compose[@]}" config --quiet
"${compose[@]}" up -d --no-build serve
