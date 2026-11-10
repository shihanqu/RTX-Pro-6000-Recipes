#!/usr/bin/env bash
set -euo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/tools/common.sh"

mkdir -p "$WORK_DIR"
src=$(mktemp -d "$WORK_DIR/flare-source.XXXXXXXX")
git -C "$src" init --quiet
git -C "$src" remote add origin https://github.com/NVIDIA/hyperflare-inference.git
git -C "$src" fetch --depth 1 origin "$FLARE_REVISION"
git -C "$src" switch --detach FETCH_HEAD
[[ $(git -C "$src" rev-parse HEAD) == "$FLARE_REVISION" ]] || fail 'Unexpected Flare source revision.'
git -C "$src" apply --check "$RECIPE_DIR/patches/flare-brrr-d4.patch"
git -C "$src" apply "$RECIPE_DIR/patches/flare-brrr-d4.patch"

# The upstream .dockerignore is for its Rust serving image, not Ignition.
cp "$RECIPE_DIR/prepare/Dockerfile" "$src/Dockerfile.recipe"
cp "$RECIPE_DIR/prepare/Dockerfile.dockerignore" "$src/Dockerfile.recipe.dockerignore"
cp "$RECIPE_DIR/patches/flashinfer-route-reuse.patch" "$src/flashinfer-route-reuse.patch"
docker build --platform linux/amd64 --file "$src/Dockerfile.recipe" --tag "$PREP_IMAGE" "$src"
