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
