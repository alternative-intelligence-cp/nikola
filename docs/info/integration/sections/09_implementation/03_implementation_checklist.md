# IMPLEMENTATION CHECKLIST

## 28.1 Overview

This checklist MUST be followed file-by-file in order. Do NOT skip steps or deviate.

**!!! NO DEVIATION FROM SPECS FOR ANY REASON !!!**

## 28.2 Foundation Layer

### Setup and Configuration

- [ ] **1.1** Create root `CMakeLists.txt`
  - Set C++23 standard
  - Find packages: ZeroMQ, Protobuf, LMDB, libvirt, CUDA (optional), FFTW3
  - Configure build types: Debug, Release, RelWithDebInfo
  - Enable AVX-512 if available

- [ ] **1.2** Create `proto/neural_spike.proto`
  - Define all message types from Section 10.2
  - Generate C++ code: `protoc --cpp_out=. neural_spike.proto`
  - Verify compilation

- [ ] **1.3** Create `config/nikola.conf`
  - Set paths: state_dir, ingest_dir, archive_dir
  - Set constants: golden_ratio=1.618033988749895, emitter frequencies
  - Set thresholds: resonance_threshold=0.7, dopamine_baseline=0.5

## 28.3 Physics Engine

### Types and Core Structures

- [ ] **2.1** `include/nikola/types/nit.hpp`
  ```cpp
  namespace nikola {
      enum class Nit : int8_t {
          N4 = -4, N3 = -3, N2 = -2, N1 = -1, ZERO = 0,
          P1 = 1, P2 = 2, P3 = 3, P4 = 4
      };

      Nit sum_gate(Nit a, Nit b);
      Nit product_gate(Nit a, Nit b);
      Nit quantize_wave(std::complex<double> wave);
  }
  ```

- [ ] **2.2** `src/types/nit.cpp`
  - Implement all three functions from 2.1
  - Add unit tests in `tests/unit/test_nonary.cpp`
  - **Validation:** Test $1 + (-1) = 0$, $2 \times 3 = 4$ (saturate)

- [ ] **2.3** `include/nikola/types/coord9d.hpp`
  - Define `Coord9D` struct with `std::array<int32_t, 9>`
  - Implement `wrap()` method (GEO-TOPO-01 fix)
  - Implement `distance_to()` for geodesic distance
  - Define hash function for use in `unordered_map`

- [ ] **2.4** `include/nikola/types/torus_node.hpp`
  - Define `TorusNode` struct (256-byte aligned)
  - Include: wavefunction, metric_tensor, resonance_r, state_s
  - **CRITICAL:** Zero padding in constructor (MEM-INIT-01 fix)
  - Verify `sizeof(TorusNode) == 256`

### Emitter Array

- [ ] **2.5** `include/nikola/physics/emitter_array.hpp`
  - Define `EmitterArray` class with phase accumulators
  - Declare sine LUT (16384 samples)
  - Define DDS tick() method

- [ ] **2.6** `src/physics/emitter_array.cpp`
  - Initialize sine LUT in constructor
  - Compute tuning words from frequencies
  - Implement DDS algorithm from Section 4.5
  - **Validation:** Generate 1Hz sine, verify with FFT

### Torus Manifold

- [ ] **2.7** `include/nikola/physics/shvo_grid.hpp`
  - Define `SparseHyperVoxelGrid` class
  - Implement Morton code hashing
  - Define neurogenesis methods

- [ ] **2.8** `src/physics/shvo_grid.cpp`
  - Implement sparse grid using `unordered_map<uint64_t, TorusNode*>`
  - Implement `get_or_create()` with neurogenesis trigger
  - Implement `update_gpu_neighbor_map()` (PHY-MEM-01)

- [ ] **2.9** `include/nikola/physics/torus_manifold.hpp`
  - Define main interface
  - Declare `inject_wave()`, `propagate()`, `find_resonance_peak()`
  - Declare neuroplasticity/neurogenesis methods

- [ ] **2.10** `src/physics/torus_manifold.cpp`
  - Implement wave propagation (UFIE from WP1)
  - Implement neuroplasticity update (Section 3.4)
  - Integrate with ENGS global state (PHY-CUDA-01 fix)
  - **Validation:** Inject two waves, verify interference

### Wave Interference Processor

- [ ] **2.11** `src/physics/wave_engine.cpp`
  - Implement superposition addition
  - Implement heterodyning multiplication
  - Implement spectral cascading (carry mechanism)
  - **Validation:** Test $+3 + +2 = +4$ (saturate), not +5

## 28.4 Cognitive Systems

### Mamba-9D

- [ ] **3.1** `include/nikola/mamba/hilbert_scan.hpp`
  - Define `HilbertMapper` class
  - Declare `encode()` and `decode()` methods

- [ ] **3.2** `src/mamba/hilbert_scan.cpp`
  - Implement Hilbert curve mapping
  - **Validation:** Verify locality preservation

- [ ] **3.3** `include/nikola/mamba/ssm_kernel.hpp`
  - Define `Mamba9D` class with A, B, C matrices
  - Implement Topological State Mapping (WP2)

- [ ] **3.4** `src/mamba/ssm_kernel.cpp`
  - Implement SSM forward pass
  - Derive matrices from metric tensor
  - **Validation:** Test state propagation

### Transformer

- [ ] **3.5** `include/nikola/reasoning/attention.hpp`
  - Define `WaveAttentionLayer`
  - Declare wave correlation methods

- [ ] **3.6** `src/reasoning/wave_attention.cpp`
  - Implement Wave Correlation Attention (WP2)
  - Use complex conjugate product
  - **Validation:** Compare with standard attention

- [ ] **3.7** `src/reasoning/transformer.cpp`
  - Implement full transformer stack
  - Integrate wave attention
  - Add neuroplasticity hooks

### Embedder

- [ ] **3.8** `src/reasoning/embedder.cpp`
  - Implement text → waveform conversion
  - Use character/token encoding
  - **Validation:** Text roundtrip accuracy >90%

## 28.5 Infrastructure

### ZeroMQ Spine

- [ ] **4.1** `src/spine/broker.cpp`
  - Implement message router
  - Add CurveZMQ security (Section 10.3)
  - Implement ZAP authentication

- [ ] **4.2** `src/spine/shadow_spine.cpp`
  - Implement A/B testing infrastructure (WP4)
  - Add voting mechanism
  - Add promotion logic

### Orchestrator and Agents

- [ ] **4.3** `src/orchestrator/smart_router.cpp`
  - Implement tool selection logic
  - Integrate all agents

- [ ] **4.4** `src/agents/*.cpp`
  - Implement Tavily, Firecrawl, Gemini clients
  - Implement Custom HTTP client
  - **Validation:** Test API calls

### Executor

- [ ] **4.5** `src/executor/kvm_executor.cpp`
  - Implement VM lifecycle management
  - Add virtio-serial communication
  - Implement CSVP integration (WP4)

## 28.6 Autonomy

### Neurochemistry

- [ ] **5.1** `src/autonomy/engs.cpp`
  - Implement Extended Neurochemical Gating System
  - Use exponential decay (AUTO-ENGS-01 fix)
  - Integrate with physics kernel (PHY-CUDA-01)

- [ ] **5.2** `src/autonomy/dopamine.cpp`
  - Implement TD learning
  - Add reward mechanisms

- [ ] **5.3** `src/autonomy/boredom.cpp`
  - Implement Shannon entropy calculation
  - Add curiosity triggers

- [ ] **5.4** `src/autonomy/goals.cpp`
  - Implement goal DAG
  - Add completion propagation

### Training and Self-Improvement

- [ ] **5.5** `src/autonomy/trainers.cpp`
  - Implement Bicameral Autonomous Trainers
  - Add auto-training triggers

- [ ] **5.6** `src/autonomy/dream_weave.cpp`
  - Implement counterfactual simulation
  - Add z-score normalization (AUTO-DREAM-01)

- [ ] **5.7** `src/self_improve/adversarial_dojo.cpp`
  - Implement Red Team agent (WP4)
  - Add attack generation

## 28.7 Persistence & Security

### Persistence

- [ ] **6.1** `src/persistence/lsm_dmc.cpp`
  - Implement LSM-DMC (PER-LSM-01 fix)
  - Add compaction worker
  - Add Write-Ahead Log

- [ ] **6.2** `src/persistence/gguf_export.cpp`
  - Implement Hilbert flattening
  - Add Q9_0 quantization
  - **Validation:** Load in llama.cpp

### Security

- [ ] **6.3** `src/security/resonance_firewall.cpp`
  - Implement spectral analysis
  - Load hazard database

- [ ] **6.4** `src/security/csvp.cpp`
  - Implement Code Safety Verification Protocol (WP4)
  - Add static analysis hooks
  - Add physics invariant tests

## 28.8 Multimodal

- [ ] **7.1** `src/multimodal/audio_resonance.cpp`
  - Implement FFT binning
  - Fix spectral dead zone (MM-AUD-01)
  - **Validation:** Process speech sample

- [ ] **7.2** `src/multimodal/visual_cymatics.cpp`
  - Implement holographic RGB encoding (MM-VIS-01)
  - Add phase-based color separation
  - **Validation:** Process test image

## 28.9 Tools and CLI

- [ ] **8.1** `tools/twi-ctl/main.cpp`
  - Implement CLI controller
  - Add all commands from Section 25
  - **Validation:** Test all commands

- [ ] **8.2** `tools/convert_nikola_to_gguf.py`
  - Implement Python export script
  - **Validation:** Export sample state

## 28.10 Testing

- [ ] **9.1** Implement all unit tests
  - Physics invariants
  - Nonary arithmetic
  - Wave interference
  - ENGS homeostasis

- [ ] **9.2** Implement integration tests
  - Search-retrieve-store loop
  - Training cycle
  - Multimodal processing

- [ ] **9.3** Implement benchmarks
  - Wave propagation performance
  - Hilbert scan performance

## 28.11 Final Integration

- [ ] **10.1** Build Docker images
- [ ] **10.2** Run security audit
- [ ] **10.3** Performance testing
- [ ] **10.4** Documentation review
- [ ] **10.5** Grant submission preparation

---

**Total Checklist Items:** ~60
**Estimated Completion:** 12 months (5-person team)

---

**Cross-References:**
- See Section 26 for File Structure
- See Section 27 for Development Roadmap
- See WP1-5 for Remediation details
