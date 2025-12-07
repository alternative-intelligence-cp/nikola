# IMPLEMENTATION CHECKLIST

## 🚨 PHASE 0: PHASE 0 REQUIREMENTS (MANDATORY)

**MUST complete before proceeding to 28.2 Foundation Layer**

### P0 Critical Items (Block Everything)

- [ ] **0.1** Structure-of-Arrays Memory Layout
  - Modify `TorusBlock` to use `alignas(64)` SoA layout
  - Separate arrays for: psi_real, psi_imag, metric_tensor (45 arrays), resonance, state
  - Block size: 19683 nodes (3^9)
  - **Reference:** Phase 0 Requirements §1.2
  - **Validation:** Verify cache hit rate >95% in Laplacian kernel
  - **Effort:** 2 days

- [ ] **0.2** Split-Operator Symplectic Integration
  - Replace Velocity-Verlet with 5-step split-operator
  - Step 1: Half-kick damping (analytical exponential)
  - Step 2: Half-kick forces
  - Step 3: Drift
  - Step 4: Recompute forces
  - Step 5: Half-kick forces + final damping
  - **Reference:** Phase 0 Requirements §2.2-2.3
  - **Validation:** Energy drift <0.01% over 10,000 steps
  - **Effort:** 3 days

- [ ] **0.3** Kahan Summation for Laplacian
  - Implement compensated summation in `compute_laplacian_kahan()`
  - Use compensation variable `c` to track lost low-order bits
  - Apply to ALL accumulation loops in physics kernel
  - **Reference:** Phase 0 Requirements §2.4
  - **Validation:** Memory waves persist >1000 timesteps without vanishing
  - **Effort:** 1 day

### P1 High Priority (Performance Critical)

- [ ] **0.4** AVX-512 Nonary Arithmetic
  - Replace enum-based Nit with `typedef int8_t Nit`
  - Implement `vec_sum_gate(__m512i, __m512i)` using `_mm512_add_epi8` + clamp
  - Implement `vec_product_gate(__m512i, __m512i)` with saturation
  - Remove ALL uses of `std::clamp` in hot loops
  - **Reference:** Phase 0 Requirements §4
  - **Validation:** Processes 64 nits in <10 CPU cycles
  - **Effort:** 2 days

- [ ] **0.5** Lazy Cholesky Decomposition
  - Add cached Cholesky factor `L` to `MetricTensor` class
  - Add `dirty_flag` to track when recomputation needed
  - Implement `recompute_if_needed()` with stability check
  - Rollback plasticity update if Cholesky fails (non-positive-definite)
  - **Reference:** Phase 0 Requirements §5
  - **Validation:** Metric inversion <1% of total compute time
  - **Effort:** 3 days

- [ ] **0.6** Energy Watchdog System
  - Implement `EnergyWatchdog` class with state machine
  - States: Stable, Heating, Critical, Dying
  - Monitor Hamiltonian every 100 timesteps
  - Auto-adjust damping when $\Delta E / E > 0.01$
  - Inject noise if $E < E_{min}$ (stochastic resonance)
  - **Reference:** Phase 0 Requirements §9.1
  - **Validation:** System remains stable for 24-hour continuous run
  - **Effort:** 1 day

### P2 Medium Priority (Optimization)

- [ ] **0.7** Shared Memory IPC (Physics ↔ Persistence)
  - Replace Protocol Buffers serialization with `/dev/shm` segments
  - Physics writes grid to `shm_open("/nikola_snapshot_<id>")`
  - ZeroMQ sends only snapshot_id (8 bytes)
  - Persistence mmaps shared segment
  - **Reference:** Phase 0 Requirements §6.3
  - **Validation:** IPC latency <100ns (vs. μs for Protobuf)
  - **Effort:** 2 days

- [ ] **0.8** Mamba-9D Taylor Approximation
  - Replace matrix exponential with first-order Taylor: $\exp(M) \approx I + M$
  - $A_i = I - \Delta(1-r_i)G_i$
  - Verify timestep constraint: $\Delta < \frac{0.1}{(1-r_{min})\lambda_{max}(G)}$
  - **Reference:** Phase 0 Requirements §3
  - **Validation:** SSM computation <10% of total time
  - **Effort:** 2 days

- [ ] **0.9** Q9_0 Quantization Fix
  - Correct packing: 2 nits per byte (not 5)
  - $packed = (n_1 + 4) \times 9 + (n_2 + 4)$
  - Unpack: $n_1 = (packed / 9) - 4$, $n_2 = (packed \% 9) - 4$
  - **Reference:** Phase 0 Requirements §8
  - **Validation:** Storage density = 4 bits/weight
  - **Effort:** 1 day

### P3 Low Priority (Nice-to-Have)

- [ ] **0.10** Sliding Window DFT for Firewall
  - Replace full FFT with Goertzel Algorithm
  - Monitor specific attack frequencies (10Hz, 50Hz, 100Hz)
  - **Reference:** Phase 0 Requirements §7
  - **Validation:** Firewall latency <1μs per sample
  - **Effort:** 1 day

### Phase 0 Validation Gate

**ALL P0 and P1 items MUST be completed and validated before proceeding to Phase 1.**

**Validation Criteria:**
- [ ] Energy drift <0.01% over 10,000 timesteps
- [ ] Memory waves persist >1000 timesteps
- [ ] Cache hit rate >95% in physics kernel
- [ ] Metric inversion <1% of total compute
- [ ] System stable for 24-hour continuous run
- [ ] IPC latency <100ns (if P2 complete)

**Estimated Total Effort:** 17 days (P0: 6 days, P1: 6 days, P2: 5 days)

---

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
  - Implement `wrap()` method for toroidal topology
  - Implement `distance_to()` for geodesic distance
  - Define hash function for use in `unordered_map`

- [ ] **2.4** `include/nikola/types/torus_node.hpp`
  - Define `TorusNode` struct (256-byte aligned)
  - Include: wavefunction, velocity, acceleration, metric_tensor, resonance_r, state_s
  - **CRITICAL:** Zero padding in constructor for proper initialization
  - Note: velocity and acceleration fields required for Velocity-Verlet integration
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
  - Implement `update_gpu_neighbor_map()` for dynamic topology

- [ ] **2.9** `include/nikola/physics/torus_manifold.hpp`
  - Define main interface
  - Declare `inject_wave()`, `propagate()`, `find_resonance_peak()`
  - Declare neuroplasticity/neurogenesis methods

- [ ] **2.10** `src/physics/torus_manifold.cpp`
  - Implement wave propagation using Unified Field Interference Equation
  - Implement neuroplasticity update (Section 3.4)
  - Integrate with ENGS global state
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
  - Implement Topological State Mapping

- [ ] **3.4** `src/mamba/ssm_kernel.cpp`
  - Implement SSM forward pass
  - Derive matrices from metric tensor
  - **Validation:** Test state propagation

### Transformer

- [ ] **3.5** `include/nikola/reasoning/attention.hpp`
  - Define `WaveAttentionLayer`
  - Declare wave correlation methods

- [ ] **3.6** `src/reasoning/wave_attention.cpp`
  - Implement Wave Correlation Attention
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
  - Implement A/B testing infrastructure
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
  - Implement CSVP integration

## 28.6 Autonomy

### Neurochemistry

- [ ] **5.1** `src/autonomy/engs.cpp`
  - Implement Extended Neurochemical Gating System
  - Use exponential decay for homeostasis
  - Integrate with physics kernel

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
  - Add z-score normalization

- [ ] **5.7** `src/self_improve/adversarial_dojo.cpp`
  - Implement Red Team agent
  - Add attack generation

## 28.7 Persistence & Security

### Persistence

- [ ] **6.1** `src/persistence/lsm_dmc.cpp`
  - Implement LSM-DMC persistence system
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
  - Implement Code Safety Verification Protocol
  - Add static analysis hooks
  - Add physics invariant tests

## 28.8 Multimodal

- [ ] **7.1** `src/multimodal/audio_resonance.cpp`
  - Implement FFT binning
  - Implement dynamic frequency folding
  - **Validation:** Process speech sample

- [ ] **7.2** `src/multimodal/visual_cymatics.cpp`
  - Implement holographic RGB encoding
  - Add phase-based color separation
  - **Validation:** Process test image

## 28.9 Tools and CLI

- [ ] **8.1** `tools/twi-ctl/main.cpp`
  - Implement CLI controller
  - **CRITICAL:** Call `curl_global_init(CURL_GLOBAL_DEFAULT)` at program startup (before any threads)
  - **CRITICAL:** Call `curl_global_cleanup()` at program shutdown (after all threads terminate)
  - Note: libcurl global initialization is NOT thread-safe and must be done once per process
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
- [ ] **10.2** Run security verification
- [ ] **10.3** Performance testing
- [ ] **10.4** Documentation review

---

**Total Checklist Items:** ~60
**Estimated Completion:** 12 months (5-person team)

---

**Cross-References:**
- See Section 26 for File Structure
- See Section 27 for Development Roadmap
