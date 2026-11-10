#!/usr/bin/env bash
# Shared settings for this recipe only.
RECIPE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
WORK_DIR=${WORK_DIR:-"$RECIPE_DIR/work"}
MODEL_DIR=${MODEL_DIR:-"$WORK_DIR/checkpoint"}
PREP_IMAGE=${PREP_IMAGE:-local/flare-brrr-d4-prepare:972e28e-1f8ed8a3-reuse}
HOST_PORT=${HOST_PORT:-1230}
MODEL_REPO=deepseek-ai/DeepSeek-V4.1-Flash-1017
MODEL_REVISION=1bae0e1088993b9b82de08f59de5be5a34e42170
FLARE_REVISION=972e28e44781d8d0c84d70ae434db92c36076a43
HOST_UID=$(id -u)
HOST_GID=$(id -g)
export RECIPE_DIR WORK_DIR MODEL_DIR PREP_IMAGE HOST_PORT MODEL_REVISION HOST_UID HOST_GID

fail() {
    printf '%s\n' "$*" >&2
    exit 1
}

[[ $WORK_DIR = /* && $MODEL_DIR = /* ]] || fail 'WORK_DIR and MODEL_DIR must be absolute paths.'

select_gpus() {
    local devices gpu inventory busy
    local -A seen=()
    IFS=, read -r -a devices <<< "${GPU_IDS:-}"
    [[ ${#devices[@]} -eq 4 ]] || fail 'Set GPU_IDS to four comma-separated GPU UUIDs.'
    inventory=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader) || fail 'Cannot read GPU inventory.'
    busy=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits) ||
        fail 'Cannot check GPU use.'
    for gpu in "${devices[@]}"; do
        [[ $gpu =~ ^GPU-[0-9a-fA-F-]+$ && -z ${seen[$gpu]:-} ]] || fail "Invalid or repeated GPU UUID: $gpu"
        seen[$gpu]=1
        grep -Fxq -- "$gpu, NVIDIA RTX PRO 6000 Blackwell Workstation Edition" <<< "$inventory" ||
            fail "GPU is not a detected RTX PRO 6000 Blackwell Workstation Edition: $gpu"
        if grep -Fq -- "$gpu," <<< "$busy"; then
            fail "GPU $gpu already has a compute process. Nothing was stopped."
        fi
    done
    export GPU0="${devices[0]}" GPU1="${devices[1]}" GPU2="${devices[2]}" GPU3="${devices[3]}"
    export GPU_IDS
}

require_host_ram() {
    local available
    available=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
    [[ $available =~ ^[0-9]+$ && $available -ge $((200 * 1024 * 1024)) ]] ||
        fail 'At least 200 GiB of available host RAM is required for this RAM-Engram profile.'
}
