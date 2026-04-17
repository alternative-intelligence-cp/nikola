# Docker Deployment Guide

## Quick Start

```bash
# Build GPU image (requires NVIDIA Container Toolkit)
make docker-build

# Run standalone
docker run --gpus all -it nikola:latest

# Full service stack
make compose-gpu
```

## Build Variants

### GPU (default)

Requires: NVIDIA driver, NVIDIA Container Toolkit, Docker 24+

```bash
make docker-build
# or manually:
docker build -t nikola:latest .
```

Default CUDA architecture: sm_86 (RTX 3090). Override with:

```bash
make docker-build CUDA_ARCH=89   # RTX 4090
make docker-build CUDA_ARCH=90   # H100
```

### CPU-only

No GPU required. Suitable for development, CI, and non-GPU hosts.

```bash
make docker-build-cpu
# or manually:
docker build \
  --build-arg BUILDER_IMAGE=ubuntu:24.04 \
  --build-arg RUNTIME_IMAGE=ubuntu:24.04 \
  --build-arg CUDA_ARCH="" \
  -t nikola:cpu .
```

## Docker Compose

The service orchestration follows a 4-layer dependency hierarchy
(see `include/nikola/system/service_orchestration.hpp`):

| Layer | Service | Role |
|-------|---------|------|
| 0 | `nikola-spine` | ZeroMQ broker (no dependencies) |
| 1 | `nikola-physics` | GPU engine (GPU profile) |
| 2 | `nikola-orchestrator` | Decision & routing |
| 2 | `nikola-memory` | LMDB persistence |
| 3 | `nikola-executor` | Task execution |
| 3 | `nikola-web` | DAP interface |

### Start Services

```bash
# CPU services only
make compose-up

# With GPU physics engine
make compose-gpu

# Check status
make compose-ps

# View logs
make compose-logs

# Graceful shutdown (SIGTERM → WAL flush → exit)
make compose-down
```

### Run Diagnostics

```bash
make compose-diag
```

## Volumes

| Volume | Mount Point | Purpose |
|--------|-------------|---------|
| `nikola-checkpoints` | `/data/nikola/checkpoints` | LMDB state database |
| `nikola-memory` | `/data/nikola/memory` | Memory system data |
| `nikola-logs` | `/data/nikola/logs` | Application logs |
| `ipc-sockets` | `/tmp/nikola/ipc` | IPC socket files |

## ZMQ Ports

| Port | Protocol | Purpose |
|------|----------|---------|
| 5555 | PUB | Events |
| 5556 | REP | Control |
| 5557 | PUB | Data plane |
| 5560 | PUB | Node state/actions |
| 5561 | PULL | Node stimulus input |
| 9876 | TCP | Peer protocol |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NIKOLA_VERSION` | `latest` | Image tag |
| `NIKOLA_LOG_LEVEL` | `info` | Log verbosity |
| `NIKOLA_STATE_DB` | `/data/nikola/checkpoints/state.lmdb` | LMDB path |
| `NIKOLA_MEMORY_PATH` | `/data/nikola/memory` | Memory dir |
| `NIKOLA_ORT_MODEL_PATH` | `/opt/nikola/models/bert-tiny-onnx/model.onnx` | ONNX model |
| `NIKOLA_ORT_TOKENIZER_PATH` | `/opt/nikola/models/bert-tiny-onnx` | Tokenizer dir |
| `ZMQ_CURVE_SERVER` | `0` | Enable CurveZMQ (1=on) |
| `CURVEZMQ_KEYS_PATH` | `./keys` | Host path to CurveZMQ keys |

## Security

- All services run as non-root user `nikola`
- CurveZMQ (Ironhouse) available via `ZMQ_CURVE_SERVER=1`
- Key material mounted read-only from host
- No capabilities added by default
