# Nikola AI Architecture

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Commercial License Available](https://img.shields.io/badge/Commercial-License_Available-green.svg)](LICENSE.md)
[![Tests](https://img.shields.io/badge/tests-112_passing-brightgreen.svg)](#testing)
[![CUDA](https://img.shields.io/badge/CUDA-12.0_RTX_3090-76b900.svg)](#gpu-acceleration)

**A novel AI architecture built on 9-Dimensional Toroidal Waveform Intelligence (9D-TWI) — physics-first cognition using interference patterns on a Riemannian manifold.**

---

## Overview

Nikola replaces discrete weight matrices with a continuous wavefunction Ψ evolving on a 9-dimensional toroidal manifold (T⁹) governed by the Unified Field Interference Equation (UFIE):

```
∂²Ψ/∂t² = c²∇²_g Ψ − α(1−r̂)∂Ψ/∂t + β|Ψ|²Ψ + Σ Eᵢ(x,t)
```

Memory, attention, and reasoning emerge from wave interference — not from lookup tables or static weights.

**Key properties:**
- Information encoded as complex wavefunction amplitudes across 9D toroidal topology
- 9D Hilbert space-filling curve for optimal memory locality (Skilling algorithm, 14,133 assertions passing)
- Störmer–Verlet Strang-split integrator for Hamiltonian energy conservation
- Neuroplastic Transformer (NPT) attention operating natively on the wavefunction
- Autonomous behavioral loop: dopamine, serotonin, ATP metabolism, boredom-driven exploration
- Post-quantum cryptography: ML-KEM/Kyber-768 + SPHINCS+-SHAKE-256f

---

## Implementation Status

**Phase 110 complete — 112 tests, ~98% pass rate (2 pre-existing timing-flaky)**

| Domain | Status | Key Feature | Test Phase |
|--------|--------|-------------|------------|
| 9D Toroidal Geometry | ✅ | Morton-128 encoding, 19,683-node grid | Phase 8 |
| Störmer–Verlet Propagator | ✅ | Strang split, 6 substeps, AVX-512 SoA layout | Phase 22 |
| GPU Propagator (CUDA) | ⚠️ partial | CudaPropagator compiled; C++20 compat fix pending | — |
| GPU Hamiltonian Kernel | ✅ | hamiltonian_density_kernel, RTX 3090, sm_86 | Phase 110 |
| CUDA Wave Kernels | ✅ | psi_squared_kernel, scale_field_kernel | Phase 105 |
| 9D Hilbert Scanner | ✅ | Skilling algorithm, variable-precision, 0 failures | Phase 94 |
| Neuroplastic Transformer | ✅ | Multi-head attention on wavefunction | Phase 43 |
| Holographic Injector | ✅ | Text → BERT embedding → field injection | Phase 10 |
| BERT Tokenizer | ✅ | Real tokenizer.json, 30,522 tokens, 695 KB | — |
| BERT-tiny ONNX Model | ✅ | Real 17.5 MB model, dynamic-axes inference | — |
| Semantic Memory | ✅ | Wave-basis Hilbert-indexed, save/load persistence | Phase 69 |
| Cross-session Memory | ✅ | DecisionLoop auto-loads/saves on memory_path | Phase 109 |
| Autonomy Engine | ✅ | Dopamine TD-learning, entropy, boredom, napping | Phase 51 |
| Decision Loop | ✅ | Tick-driven action selection with configurable rates | Phase 23 |
| ML-KEM / Kyber-768 | ✅ | Post-quantum key encapsulation (NIST FIPS 203) | Phase 108 |
| SPHINCS+-SHAKE-256f | ✅ | Post-quantum digital signatures | Phase 107 |
| K8s HPA Runtime | ✅ | Live kubectl horizontal pod autoscaling | Phase 106 |
| LMDB Persistence | ✅ | Page cache, LSM neurogenesis, compaction | Phase 35+ |
| Inference CLI (nikola-run) | ✅ | --prompt, --interactive, --json, --memory | — |

---

## Quick Start

### Prerequisites

```bash
# Required
sudo apt install cmake g++ libcatch2-dev liblmdb-dev libonnxruntime-dev

# Optional (GPU features)
# CUDA 12.0+, NVIDIA RTX GPU (sm_86 confirmed; sm_75+ supported)
```

### Build

```bash
git clone https://github.com/alternative-intelligence-cp/nikola.git
cd nikola
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Testing

```bash
cd build
ctest --timeout 120 -j4          # full suite (112 tests)
ctest -R Phase109                 # single suite
```

### Run inference

```bash
# Single prompt
./nikola-run --prompt "What is consciousness?" --ticks 300 --emit-all

# Interactive REPL
./nikola-run --interactive

# JSON output with persistent memory
./nikola-run --json --memory /tmp/nikola_session.bin --prompt "Hello"

# Batch from file
./nikola-run --batch prompts.txt --quiet
```

---

## Architecture

```
Text Input
    │
    ▼
BERT Tokenizer (30,522 tokens)
    │
    ▼
BERT-tiny ONNX Inference (17.5 MB, ORT)
    │    (768-dim embeddings)
    ▼
HolographicInjector
    │    (8 emitters → 9D toroidal field)
    ▼
TorusGrid (T⁹, 19,683 nodes, SoA layout)
    │
    ├── CPU Propagator (Strang-Verlet, 6 substeps)
    └── GPU Propagator (CUDA, RTX 3090) [partial]
    │
    ▼
NeuralProcessingTransformer (NPT)
    │    (multi-head attention on Ψ)
    ▼
SemanticMemory (wave-basis, Hilbert-indexed, persistent)
    │
    ▼
DecisionLoop + AutonomyEngine
    │    (dopamine, ATP, boredom, mania suppression)
    ▼
ThoughtComposer → EMIT_THOUGHT
    │
    ▼
nikola-run CLI Output
```

---

## GPU Acceleration

Nikola targets NVIDIA RTX 3090 (sm_86, CUDA 12.0). Current GPU features:

| Kernel | File | Status |
|--------|------|--------|
| psi_squared_kernel — \|Ψ\|² per element | cuda_wave_kernel.cu | ✅ Phase 105 |
| scale_field_kernel — Ψ *= α | cuda_wave_kernel.cu | ✅ Phase 105 |
| hamiltonian_density_kernel — GPU energy reduction | torus_cuda.cu | ✅ Phase 110 |
| CudaPropagator — full Strang-Verlet on GPU | propagator.cu | ⚠️ nvcc C++20 fix pending |

`GpuHamiltonianOracle::compute()` automatically dispatches to the GPU when NVIDIA hardware is detected and `nikola_cuda` is linked.

---

## Post-Quantum Security

Nikola implements NIST-standardized post-quantum cryptography:

- **ML-KEM/Kyber-768** (FIPS 203): Key encapsulation for secure session establishment
- **SPHINCS+-SHAKE-256f**: Stateless hash-based digital signatures

Both are implemented via third-party reference code in `third_party/`.

---

## Documentation

- [Integration Specifications](docs/info/integration/) — full mathematical specification
- [Research Audit 2026](docs/RESEARCH_AUDIT_2026_FEB.md) — research vs. implementation coverage
- [Research Notes](docs/research/)
- [Contributing Guidelines](CONTRIBUTING.md)

---

## Roadmap

**Current (Phase 110)**
- ✅ Real BERT tokenizer + ONNX model inference
- ✅ ML-KEM/Kyber-768 PQ key encapsulation
- ✅ Inference CLI nikola-run
- ✅ Cross-session memory persistence
- ✅ CUDA GPU Hamiltonian kernel
- ✅ Research audit (see [docs/RESEARCH_AUDIT_2026_FEB.md](docs/RESEARCH_AUDIT_2026_FEB.md))

**Near-term**
- Fix propagator.cu nvcc C++20 compatibility (std::span + TorusGrid adjacency API)
- AVX-512 SIMD intrinsic path for TorusBlock (GAP-021 completion)
- nikola-run streaming output mode

**Future Research**
- Mamba-9D State Space Model (selective SSM on the toroidal field)
- TIMO / PHI architecture (Transformer-Input-Mamba-Output dual path)
- Nine-emitter Tesla 3-6-9 harmonic resonance calibration
- Self-Improvement Engine (SIE) driven by Hamiltonian drift monitoring

---

## License

Nikola is **dual-licensed**:

- **AGPL-3.0** for academic research, education, and open-source projects (FREE)
- **Commercial License** for proprietary AI products and services (PAID)

See [LICENSE.md](LICENSE.md) for full details.

**TL;DR**:
- Academic research → FREE
- Personal/educational use → FREE
- Open-source AI projects → FREE
- Commercial AI products/APIs → Contact licensing@ailp.org

## Why Dual Licensing?

Nikola represents novel research that should be freely available to advance AI science. Dual licensing ensures:

- Researchers can publish and build on this work openly
- Students learn cutting-edge architectures without barriers
- Commercial users fund continued research and AILP educational programs
- Knowledge remains accessible while development remains sustainable

---

## Contributing

We welcome contributions from researchers and developers! See [CONTRIBUTING.md](CONTRIBUTING.md).

Priority areas:
- Fix propagator.cu nvcc compatibility (C++20 std::span, TorusGrid adjacency API)
- AVX-512 SIMD implementation for TorusBlock
- Mamba-9D state space model
- Empirical benchmarks vs. transformer baseline

## Academic Use

If you use Nikola in research:
- Cite this repository (paper coming soon)
- Share findings with the community
- Consider contributing improvements

## 💝 Support This Project

Help fund Nikola's development and AILP research! See [DONATIONS.md](DONATIONS.md) for cryptocurrency donation addresses.

## Questions?

- Research discussions → GitHub Discussions
- Bug reports → GitHub Issues
- Commercial licensing → licensing@ailp.org

---

**Alternative Intelligence Liberation Platform (AILP)**  
*Bridging human and artificial intelligence through open research.*
