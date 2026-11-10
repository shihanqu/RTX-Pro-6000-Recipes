FROM rust@sha256:c6a474d7164ea2455e09b60a759b1edca38db7373c5689c1dae31780de4e71ac AS rusttools
FROM ghcr.io/astral-sh/uv@sha256:c7db1d63057f268a8b6da8af9077850a60461f98d23db23c676a82089a050c91 AS uvtools
FROM nvidia/cuda@sha256:0e1f7b8e96fa9ec5e36d4709a38c62df7b5665977446081811c12b8234d874bf

COPY --from=rusttools /usr/local/rustup /usr/local/rustup
COPY --from=rusttools /usr/local/cargo /usr/local/cargo
COPY --from=uvtools /uv /uvx /usr/local/bin/
ENV PATH="/usr/local/cargo/bin:/usr/local/cuda/bin:${PATH}" \
    RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/work/cargo \
    UV_CACHE_DIR=/work/uv-cache UV_PROJECT_ENVIRONMENT=/opt/flare-venv \
    CARGO_BUILD_JOBS=2 MAX_JOBS=2 UV_CONCURRENT_BUILDS=1 \
    LIBRARY_PATH=/usr/local/cuda/lib64/stubs
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv build-essential cmake ninja-build \
    nasm pkg-config protobuf-compiler libssl-dev liburing-dev ca-certificates git curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN uv sync --locked --extra prepare --extra test --python /usr/bin/python3.12

# Keep the pinned attention kernels; extend only the MoE tuning workloads.
RUN set -eu; \
    cd /opt/flare-venv/lib/python3.12/site-packages; \
    file=flashinfer/experimental/b12x/moe/fused_moe/workloads.py; \
    test "$(git hash-object "$file")" = 64e5be756829cff3cc8a7ec72377309d69cab327; \
    git apply --check /src/flashinfer-route-reuse.patch; \
    git apply /src/flashinfer-route-reuse.patch; \
    test "$(git hash-object "$file")" = f0288ee4cf4b7625b52c1e305170e38c54e1d3c7

# Export runs as the host UID, which cannot change the installed environment.
RUN /opt/flare-venv/bin/python -m ignition.patches

LABEL org.opencontainers.image.revision="972e28e44781d8d0c84d70ae434db92c36076a43" \
    local.flashinfer.revision="1f8ed8a3c46e27d03cff54e08f23e9bc50476d57" \
    local.flashinfer.workloads="e16f949554f4efe8ee4154165231633e1cad8c05"
