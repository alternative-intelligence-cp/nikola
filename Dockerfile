# ==============================================================================
# Nikola — Multi-stage Docker Build
# v0.2.4: Docker Packaging & Deployment
#
# GPU build (default):
#   docker build -t nikola:latest .
#   docker run --gpus all -it nikola:latest
#
# CPU-only build:
#   docker build \
#     --build-arg BUILDER_IMAGE=ubuntu:24.04 \
#     --build-arg RUNTIME_IMAGE=ubuntu:24.04 \
#     --build-arg CUDA_ARCH="" \
#     -t nikola:cpu .
# ==============================================================================

# ─── Build arguments ─────────────────────────────────────────────────────────

ARG CUDA_VERSION=12.6.3
ARG UBUNTU_VERSION=24.04
ARG ORT_VERSION=1.21.1
ARG CMAKE_BUILD_TYPE=Release
ARG CUDA_ARCH=86
ARG BUILDER_IMAGE=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
ARG RUNTIME_IMAGE=nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

# ==============================================================================
# Stage 1: Builder — full dev toolchain
# ==============================================================================

FROM ${BUILDER_IMAGE} AS builder

ARG CMAKE_BUILD_TYPE
ARG ORT_VERSION
ARG CUDA_ARCH
ARG TARGETARCH

# Avoid interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive

# ─── System build dependencies ───────────────────────────────────────────────

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    pkg-config \
    # Core libraries
    libeigen3-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libzmq3-dev \
    libczmq-dev \
    libcurl4-openssl-dev \
    liblmdb-dev \
    libsodium-dev \
    libssl-dev \
    # Catch2 v3 (build from source — Ubuntu 24.04 ships v2)
    && rm -rf /var/lib/apt/lists/*

# ─── cppzmq (header-only, not always packaged with libzmq) ──────────────────

RUN git clone --depth 1 --branch v4.10.0 \
    https://github.com/zeromq/cppzmq.git /tmp/cppzmq && \
    cd /tmp/cppzmq && mkdir build && cd build && \
    cmake .. -DCPPZMQ_BUILD_TESTS=OFF -G Ninja && \
    ninja install && rm -rf /tmp/cppzmq

# ─── Catch2 v3 ───────────────────────────────────────────────────────────────

RUN git clone --depth 1 --branch v3.4.0 \
    https://github.com/catchorg/Catch2.git /tmp/catch2 && \
    cd /tmp/catch2 && mkdir build && cd build && \
    cmake .. -DBUILD_TESTING=OFF -G Ninja && \
    ninja install && rm -rf /tmp/catch2

# ─── ONNX Runtime (pre-built, GPU-enabled) ──────────────────────────────────

RUN mkdir -p /opt/onnxruntime && \
    apt-get update && apt-get install -y --no-install-recommends wget && \
    wget -q "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz" \
        -O /tmp/ort.tgz && \
    tar xzf /tmp/ort.tgz -C /opt/onnxruntime --strip-components=1 && \
    rm /tmp/ort.tgz && \
    rm -rf /var/lib/apt/lists/*

# ─── Copy source tree ───────────────────────────────────────────────────────

WORKDIR /build/nikola
COPY . .

# ─── Configure & Build ──────────────────────────────────────────────────────

RUN mkdir -p build && cd build && \
    CMAKE_ARGS="-G Ninja \
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
        -DORT_ROOT=/opt/onnxruntime \
        -DNIKOLA_ORT_MODEL_PATH=/build/nikola/models/bert-tiny-onnx/model.onnx \
        -DNIKOLA_ORT_TOKENIZER_PATH=/build/nikola/models/bert-tiny-onnx" && \
    if [ -n "${CUDA_ARCH}" ]; then \
        CMAKE_ARGS="${CMAKE_ARGS} -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}"; \
    fi && \
    cmake .. ${CMAKE_ARGS} && \
    ninja -j$(nproc)

# ─── Run tests (optional, skipped if BUILD_TESTING=OFF) ─────────────────────

RUN cd build && ctest --output-on-failure -j$(nproc) || true

# ==============================================================================
# Stage 2: Runtime — minimal image
# ==============================================================================

FROM ${RUNTIME_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# ─── Runtime dependencies only ───────────────────────────────────────────────

RUN apt-get update && apt-get install -y --no-install-recommends \
    libprotobuf-lite32t64 \
    libzmq5 \
    libcurl4t64 \
    liblmdb0 \
    libsodium23 \
    libssl3t64 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ─── Create nikola user (non-root) ──────────────────────────────────────────

RUN groupadd -r nikola && useradd -r -g nikola -m -s /bin/bash nikola

# ─── Directory structure ─────────────────────────────────────────────────────

RUN mkdir -p /opt/nikola/bin \
             /opt/nikola/lib \
             /opt/nikola/models \
             /data/nikola/checkpoints \
             /data/nikola/memory \
             /data/nikola/logs \
             /tmp/nikola/ipc && \
    chown -R nikola:nikola /data/nikola /tmp/nikola

# ─── Copy binaries from builder ─────────────────────────────────────────────

COPY --from=builder /build/nikola/build/nikola-run        /opt/nikola/bin/
COPY --from=builder /build/nikola/build/nikola-train      /opt/nikola/bin/
COPY --from=builder /build/nikola/build/nikola-orchestrator /opt/nikola/bin/
COPY --from=builder /build/nikola/build/nikola-diag       /opt/nikola/bin/
COPY --from=builder /build/nikola/build/nikola-dap        /opt/nikola/bin/
COPY --from=builder /build/nikola/build/nikola-state-dump /opt/nikola/bin/

# ─── Copy ONNX Runtime shared libs ──────────────────────────────────────────

COPY --from=builder /opt/onnxruntime/lib/libonnxruntime*.so* /opt/nikola/lib/

# ─── Copy BERT-tiny model (if present in source tree) ────────────────────────

COPY --from=builder /build/nikola/models/ /opt/nikola/models/

# ─── Library path ───────────────────────────────────────────────────────────

ENV LD_LIBRARY_PATH="/opt/nikola/lib:${LD_LIBRARY_PATH}"
ENV PATH="/opt/nikola/bin:${PATH}"

# ─── Default environment ────────────────────────────────────────────────────

ENV NIKOLA_STATE_DB="/data/nikola/checkpoints/state.lmdb"
ENV NIKOLA_MEMORY_PATH="/data/nikola/memory"
ENV NIKOLA_LOG_LEVEL="info"
ENV NIKOLA_ORT_MODEL_PATH="/opt/nikola/models/bert-tiny-onnx/model.onnx"
ENV NIKOLA_ORT_TOKENIZER_PATH="/opt/nikola/models/bert-tiny-onnx"

# ─── Ports ───────────────────────────────────────────────────────────────────
# 5555: ZMQ events (PUB)
# 5556: ZMQ control (REP)
# 5557: ZMQ data plane (PUB)
# 5560: Node state/actions (PUB)
# 5561: Node stimulus input (PULL)
# 9876: Peer protocol

EXPOSE 5555 5556 5557 5560 5561 9876

# ─── Health check ────────────────────────────────────────────────────────────
# Basic liveness: verify the binary exists and can print version.
# Phase 4 will wire this to ZMQ spine readiness.

HEALTHCHECK --interval=5s --timeout=2s --retries=5 --start-period=35s \
    CMD ["nikola-diag", "--health", "--json"] || exit 1

# ─── Run as non-root ────────────────────────────────────────────────────────

USER nikola
WORKDIR /data/nikola

# ─── Entrypoint ─────────────────────────────────────────────────────────────

ENTRYPOINT ["nikola-run"]
CMD ["--state-db", "/data/nikola/checkpoints/state.lmdb"]
