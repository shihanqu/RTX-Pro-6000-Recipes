# Shared helpers for launch.sh, download_checkpoint.sh and validate.sh (sourced, not executed).

PROJECT=glm53-kraken-tp3-zen
CONTAINER=glm53-kraken-tp3-zen
IMAGE=azallaza/glm53-kraken-tp3@sha256:44c2d27ec0269e4620895a3ce21f639ee93f9e4d9b935e896ed52a3a6900c486

die() {
  echo "ERROR: $*" >&2
  exit 1
}

# Load $REPO_DIR/.env (KEY=VALUE lines; quote values that contain spaces) into the
# environment. Variables already set in the calling shell win, as they do in docker compose.
load_env() {
  local file="$REPO_DIR/.env" required=${1:-required} line key rhs val
  if [[ ! -f $file ]]; then
    [[ $required == required ]] && die "$file not found. Run: cp .env.example .env  and edit it."
    return 0
  fi
  while IFS= read -r line || [[ -n $line ]]; do
    [[ $line =~ ^[[:space:]]*(#|$) ]] && continue
    [[ $line =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] ||
      die "Cannot parse this line of $file: $line"
    key=${BASH_REMATCH[2]}
    rhs=${BASH_REMATCH[3]}
    [[ -n ${!key+x} ]] && continue
    eval "val=$rhs" || die "Cannot parse the value of $key in $file (quote values with spaces)."
    export "$key=$val"
  done <"$file"
}

# Pinned checkpoints. Sets CKPT_REPO, CKPT_REVISION, CKPT_FILES, CKPT_BYTES, CKPT_MOE_TP,
# CKPT_FP8_LINE (expected startup log line), CKPT_LICENSE.
select_checkpoint() {
  case ${CHECKPOINT:-csf-qad-2} in
    csf-qad-2)
      CKPT_REPO=local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD
      CKPT_REVISION=dec48abd33efa73c3bb7c95b74eee10cad34f9be
      CKPT_FILES=49
      CKPT_BYTES=177750845553
      CKPT_MOE_TP=1
      # Attention and shared experts are MXFP8 in this checkpoint; only the 6 dense FFN stay BF16.
      CKPT_FP8_LINE='GLM-5.3 FP8 decode: 6 BF16 projections'
      CKPT_LICENSE='Local Inference Lab License, Version 1.0 (LicenseRef-LIL-1.0): no re-uploading or mirroring; projects that run it must begin their README with the attribution notice'
      ;;
    default)
      CKPT_REPO=local-inference-lab/GLM-5.3-Flash-NVFP4
      CKPT_REVISION=175ae8ce3b5af842b0d0140dbeb43e9cfc557c49
      CKPT_FILES=60
      CKPT_BYTES=199387241289
      CKPT_MOE_TP=0
      CKPT_FP8_LINE='GLM-5.3 FP8 decode: 96 BF16 projections'
      CKPT_LICENSE='MIT (per the model card; read the LICENSE file in the repository before downloading)'
      ;;
    *)
      die "CHECKPOINT must be csf-qad-2 or default (got '${CHECKPOINT}')"
      ;;
  esac
  CHECKPOINT=${CHECKPOINT:-csf-qad-2}
  CKPT_SNAPSHOT="$HF_CACHE_DIR/hub/models--${CKPT_REPO//\//--}/snapshots/$CKPT_REVISION"
  CHECKPOINT_RECEIPT=${CHECKPOINT_RECEIPT:-$HF_CACHE_DIR/checkpoint-verification.json}
}

require_cmd() {
  local cmd
  for cmd in "$@"; do
    command -v "$cmd" >/dev/null 2>&1 || die "'$cmd' is required but not installed."
  done
}

human_gb() {
  awk -v b="$1" 'BEGIN { printf "%.1f GB", b / 1e9 }'
}
