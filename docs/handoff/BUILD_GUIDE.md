# Nikola — Build Guide

_Last updated: 2026-02-27_

This document describes how to build, test, and run Nikola from scratch
on a fresh Linux machine. Read `GOTCHAS.md` before debugging any surprise
failures — many common build issues are documented there.

---

## System Requirements

| Requirement | Minimum | Tested on |
|-------------|---------|-----------|
| OS | Linux (Ubuntu 22.04+) | Ubuntu 22.04 / AriaxOS |
| Compiler | g++ 13+ or clang++ 17+ | g++ 13.3 |
| C++ Standard | C++23 required | — |
| CMake | 3.20+ | 3.28 |
| CPU | AVX-512 capable | Intel/AMD Zen 4 |
| RAM | 8 GB minimum | 32 GB recommended |
| GPU (optional) | NVIDIA RTX 3090 sm_86 | CUDA 12.0 |

> **ARM / Apple Silicon**: Not tested or supported. The AVX-512 SIMD paths
> will fail to compile. You would need to add NEON fallback paths.

---

## Required System Libraries

Install via apt:

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake ninja-build git \
  libeigen3-dev \
  libprotobuf-dev protobuf-compiler \
  libzmq3-dev libcppzmq-dev \
  liblmdb-dev \
  catch2 \
  pkg-config
```

### Eigen3 v3.4

Must be **3.4 or newer** (the `Eigen3::Eigen` CMake target is required).

```bash
# Verify version:
pkg-config --modversion eigen3
```

If the system version is older than 3.4, install manually:

```bash
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
tar xf eigen-3.4.0.tar.gz
cd eigen-3.4.0 && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
sudo make install
```

---

## Optional: ONNX Runtime (for BERT embeddings)

Nikola uses BERT-tiny for semantic input encoding. Without ONNXRuntime,
the holographic injector falls back to random projections (still functional
but semantically degraded).

The build system looks for:
```
/home/randy/Workspace/SYSTEM/onnxruntime/cpp/include/onnxruntime_cxx_api.h
```

To set a different path:
```bash
cmake .. -DORT_ROOT=/path/to/onnxruntime/cpp
```

**Recommended version:** ONNX Runtime 1.24.2 GPU build (Linux x64).
Download from: https://github.com/microsoft/onnxruntime/releases

---

## Clone and Build

```bash
git clone https://github.com/[OWNER]/nikola.git
cd nikola

mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

This produces:
- `build/nikola-run` — the inference CLI
- `build/libnikola_core.a` — static library
- `build/test_phase*` — individual test binaries

### Debug build

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build . -j$(nproc)
```

### With CUDA

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DNVCC_ARCH=sm_86
cmake --build . -j$(nproc)
```

CUDA is auto-detected via `find_package(CUDAToolkit QUIET)`. If CUDA is
available but you want to disable it: `-DNIKOLA_CUDA=OFF`.

---

## Running Tests

### CRITICAL — Never run `ctest` without a filter

Several test phases in the 40s–60s range have **blocking infrastructure**
that hangs indefinitely. Always filter:

```bash
# Test a specific phase (recommended)
cd build
ctest -R Phase133 -v

# Test a range (safe)
ctest -R "Phase12[5-9]|Phase13[0-5]" -v

# Test everything EXCEPT the known-hanging phases
ctest -R "Phase" --exclude-regex "Phase4[0-9]|Phase5[0-9]|Phase6[0-9]" -v
```

See `GOTCHAS.md` §"ctest hangs" for the full list of phases to avoid.

### Building a specific test target

```bash
cmake --build . --target test_phase133_peer_registry
ctest -R Phase133 -v
```

### Expected baseline

As of 2026-02-27:
- **135 ctest entries** (ctest #1 through #135)
- **~98% pass rate** (2 pre-existing timing-sensitive flakes: Phase43, Phase51)
- All Phase 110+ tests should pass cleanly

---

## Running the Inference CLI

```bash
./nikola-run --prompt "What is consciousness?"
```

### Key flags

| Flag | Description |
|------|-------------|
| `--prompt TEXT` | Single-shot inference |
| `--interactive` | REPL mode (stdin loop) |
| `--stream` | Line-buffered token streaming |
| `--json` | JSON output format |
| `--memory PATH` | Path to LMDB memory database |
| `--ticks N` | Number of simulation ticks per inference |
| `--help` | Full usage |

### Environment variables

```bash
NIKOLA_MODEL_PATH=/path/to/bert_tiny.onnx
NIKOLA_MEMORY_DB=/path/to/memory.lmdb
NIKOLA_LOG_LEVEL=info   # debug | info | warn | error
```

### First light (minimal test)

```bash
# Build the first-light entry point
cmake --build . --target nikola_first_light
./nikola_first_light
```

This runs one tick of the physics engine and prints field energy — useful
for verifying the propagator is functional without full pipeline setup.

---

## Directory Structure

```
nikola/
├── CMakeLists.txt          Primary build file
├── include/nikola/         All public headers
│   ├── autonomy/           Decision loop, metabolic controller
│   ├── cognitive/          Scratchpad, AttentionPrimer, SpectralFilter
│   ├── core/               Config, HolographicInjector, EmitterArray
│   ├── economy/            NeuralMarketplace, NeuralWallet
│   ├── interior/           AffectiveState, WaveMirror, DreamEngine, etc.
│   ├── physics/            Propagator, Hamiltonian, Torus manifold
│   ├── security/           HomeostasisMonitor, PolymorphicDefense, etc.
│   ├── social/             SocialMembrane, PeerRegistry
│   └── spatial/            HilbertScanner, Morton encoding
├── src/                    Implementations (mirrors include/ structure)
├── tests/unit/             All Catch2 unit tests
├── docs/                   Architecture docs, handoff package
├── third_party/            Vendored: kyber, sphincsplus, ORT, opentelemetry
├── proto/                  Protobuf definitions (NikolaState, IRSP)
└── build/                  Build output (git-ignored)
```

---

## Troubleshooting Quick Reference

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `Eigen3 3.4 not found` | Old system Eigen | Install 3.4 manually (see above) |
| `cppzmq not found` | ZMQ dev headers missing | `sudo apt install libcppzmq-dev` |
| `Catch2 not found` | Catch2 v3 not installed | `sudo apt install catch2` |
| Test binary links but crashes | Wrong Catch2 version (v2 vs v3) | Check `GOTCHAS.md` |
| `ctest` hangs indefinitely | Ran without `-R` filter | Kill and re-run with filter |
| GPU tests fail to compile | sm_86 mismatch | Set `-DNVCC_ARCH` to your SM version |
| `onnxruntime_cxx_api.h` not found | ORT not at expected path | Set `-DORT_ROOT` |
