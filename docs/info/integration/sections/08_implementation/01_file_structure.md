# FILE STRUCTURE

## 26.0 Phase 0 Critical Components

**Date:** December 7, 2025  
**Source:** Engineering Report Review and Analysis v0.0.4

The file structure has been updated to include Phase 0 critical components. Key additions:

- `include/nikola/physics/soa_layout.hpp` - Structure-of-Arrays memory layout
- `include/nikola/physics/symplectic_integrator.hpp` - Split-operator integration
- `include/nikola/security/physics_oracle.hpp` - Self-improvement safety
- `src/physics/kernels/wave_propagate_soa.cu` - SoA CUDA kernel
- `tests/validation/` - Phase 0 validation test suite

See: `08_audit_remediation/01_critical_fixes.md` for complete specifications.

---

## 26.1 Complete Directory Organization

```
nikola/
├── CMakeLists.txt                   # Root CMake file
├── README.md                        # Project README
├── LICENSE                          # License file
├── .dockerignore                    # Docker ignore
├── Dockerfile                       # Multi-stage Docker build
├── docker-compose.yml               # Service orchestration
│
├── include/                         # Public headers
│   └── nikola/
│       ├── types/
│       │   ├── nit.hpp              # Balanced nonary type (AVX-512)
│       │   ├── coord9d.hpp          # 9D coordinate
│       │   ├── torus_node.hpp       # Node structure (DEPRECATED - use SoA)
│       │   ├── torus_block.hpp      # ⚡ SoA memory layout (Phase 0)
│       │   └── morton_code.hpp      # ⚡ 128-bit Z-order encoding (Phase 0)
│       ├── physics/
│       │   ├── torus_manifold.hpp   # Main 9D grid
│       │   ├── soa_layout.hpp       # ⚡ Structure-of-Arrays (Phase 0)
│       │   ├── symplectic_integrator.hpp # ⚡ Split-operator (Phase 0)
│       │   ├── kahan_sum.hpp        # ⚡ Compensated summation (Phase 0)
│       │   ├── emitter_array.hpp    # DDS emitters
│       │   ├── wave_engine.hpp      # Interference processor
│       │   ├── shvo_grid.hpp        # Sparse hyper-voxel
│       │   ├── metric.hpp           # Riemannian geometry
│       │   └── metric_cache.hpp     # ⚡ Lazy Cholesky (Phase 0)
│       ├── mamba/
│       │   ├── hilbert_scan.hpp     # Space-filling curve
│       │   ├── ssm_kernel.hpp       # State space model
│       │   └── taylor_approx.hpp    # ⚡ Matrix approximation (Phase 0)
│       ├── reasoning/
│       │   ├── transformer.hpp      # Wave transformer
│       │   ├── attention.hpp        # Wave correlation
│       │   └── embedder.hpp         # Nonary embedder
│       ├── spine/
│       │   ├── broker.hpp           # ZeroMQ router
│       │   ├── component_client.hpp # Client interface
│       │   ├── shadow_spine.hpp     # A/B testing
│       │   └── shared_memory.hpp    # ⚡ Zero-copy IPC (Phase 0)
│       ├── agents/
│       │   ├── tavily.hpp           # Search client
│       │   ├── firecrawl.hpp        # Scrape client
│       │   ├── gemini.hpp           # Translation client
│       │   └── http_client.hpp      # Custom HTTP
│       ├── executor/
│       │   └── kvm_executor.hpp     # VM manager
│       ├── autonomy/
│       │   ├── dopamine.hpp         # Reward system
│       │   ├── engs.hpp             # Extended neurochemistry
│       │   ├── boredom.hpp          # Curiosity
│       │   ├── goals.hpp            # Goal DAG
│       │   └── dream_weave.hpp      # Counterfactual learning
│       ├── persistence/
│       │   ├── dmc.hpp              # Checkpoint manager
│       │   ├── lsm_dmc.hpp          # LSM persistence
│       │   ├── gguf_export.hpp      # GGUF converter
│       │   ├── q9_encoder.hpp       # ⚡ Q9_0 quantization (Phase 0)
│       │   └── identity.hpp         # Identity profile
│       ├── multimodal/
│       │   ├── audio_resonance.hpp  # Audio FFT
│       │   └── visual_cymatics.hpp  # Image processing
│       ├── security/
│       │   ├── resonance_firewall.hpp # Attack detection
│       │   ├── physics_oracle.hpp   # ⚡ Self-improvement safety (Phase 0)
│       │   ├── adversarial_dojo.hpp # ⚡ Red team testing (Phase 0)
│       │   └── csvp.hpp             # Code safety protocol
│       ├── monitoring/
│       │   ├── energy_watchdog.hpp  # ⚡ Energy conservation monitor (Phase 0)
│       │   └── profiler.hpp         # ⚡ Performance profiler (Phase 0)
│       └── self_improve/
│           └── hot_swap.hpp         # Module replacement
│
├── src/                             # Implementation
│   ├── core/
│   │   ├── lib9dtwi.cpp             # Main library
│   │   └── CMakeLists.txt
│   ├── types/
│   │   ├── nit.cpp                  # ⚡ AVX-512 nonary ops (Phase 0)
│   │   ├── coord9d.cpp
│   │   ├── torus_block.cpp          # ⚡ SoA implementation
│   │   ├── morton_code.cpp          # ⚡ 128-bit encoding
│   │   └── CMakeLists.txt
│   ├── physics/
│   │   ├── torus_manifold.cpp
│   │   ├── soa_layout.cpp           # ⚡ SoA refactoring
│   │   ├── symplectic_integrator.cpp # ⚡ 6-step Strang splitting
│   │   ├── kahan_sum.cpp            # ⚡ Compensated summation
│   │   ├── emitter_array.cpp
│   │   ├── wave_engine.cpp
│   │   ├── shvo_grid.cpp
│   │   ├── metric.cpp
│   │   ├── metric_cache.cpp         # ⚡ Lazy Cholesky
│   │   ├── kernels/                 # CUDA kernels
│   │   │   ├── wave_propagate.cu    # Original (DEPRECATED)
│   │   │   ├── wave_propagate_soa.cu # ⚡ SoA coalesced (Phase 0)
│   │   │   └── laplacian_kahan.cu   # ⚡ Kahan CUDA kernel
│   │   └── CMakeLists.txt
│   ├── mamba/
│   │   ├── hilbert_scan.cpp
│   │   ├── ssm_kernel.cpp
│   │   ├── taylor_approx.cpp        # ⚡ First-order matrix approx
│   │   └── CMakeLists.txt
│   ├── reasoning/
│   │   ├── transformer.cpp
│   │   ├── wave_attention.cpp
│   │   ├── embedder.cpp
│   │   └── CMakeLists.txt
│   ├── spine/
│   │   ├── broker.cpp
│   │   ├── component_client.cpp
│   │   ├── shadow_spine.cpp
│   │   ├── shared_memory.cpp        # ⚡ Seqlock IPC
│   │   └── CMakeLists.txt
│   ├── orchestrator/
│   │   ├── smart_router.cpp
│   │   └── CMakeLists.txt
│   ├── agents/
│   │   ├── tavily.cpp
│   │   ├── firecrawl.cpp
│   │   ├── gemini.cpp
│   │   ├── http_client.cpp
│   │   └── CMakeLists.txt
│   ├── executor/
│   │   ├── kvm_executor.cpp
│   │   ├── guest_agent.cpp          # Runs inside VM
│   │   └── CMakeLists.txt
│   ├── autonomy/
│   │   ├── dopamine.cpp
│   │   ├── engs.cpp
│   │   ├── boredom.cpp
│   │   ├── goals.cpp
│   │   ├── trainers.cpp
│   │   ├── dream_weave.cpp
│   │   └── CMakeLists.txt
│   ├── persistence/
│   │   ├── dmc.cpp
│   │   ├── lsm_dmc.cpp
│   │   ├── gguf_export.cpp
│   │   ├── q9_encoder.cpp           # ⚡ Radix-9 encoding
│   │   ├── identity.cpp
│   │   └── CMakeLists.txt
│   ├── multimodal/
│   │   ├── audio_resonance.cpp
│   │   ├── visual_cymatics.cpp
│   │   └── CMakeLists.txt
│   ├── security/
│   │   ├── resonance_firewall.cpp
│   │   ├── physics_oracle.cpp       # ⚡ Mathematical verification
│   │   ├── adversarial_dojo.cpp     # ⚡ Attack testing
│   │   ├── csvp.cpp
│   │   └── CMakeLists.txt
│   ├── monitoring/
│   │   ├── energy_watchdog.cpp      # ⚡ Conservation checks
│   │   ├── profiler.cpp             # ⚡ Performance tracking
│   │   └── CMakeLists.txt
│   ├── self_improve/
│   │   ├── hot_swap.cpp
│   │   └── CMakeLists.txt
│   └── ingestion/
│       ├── sentinel.cpp
│       └── CMakeLists.txt
│
├── tools/                           # Utilities
│   ├── twi-ctl/
│   │   ├── main.cpp                 # CLI controller
│   │   └── CMakeLists.txt
│   ├── validate_phase0/             # ⚡ Phase 0 validation (Phase 0)
│   │   ├── test_energy_conservation.cpp
│   │   ├── test_symplectic.cpp
│   │   ├── test_kahan.cpp
│   │   └── CMakeLists.txt
│   └── convert_nikola_to_gguf.py    # GGUF export script
│
├── proto/                           # Protocol Buffers
│   ├── neural_spike.proto
│   └── CMakeLists.txt
│
├── tests/                           # Test suites
│   ├── validation/                  # ⚡ Phase 0 validation suite (Phase 0)
│   │   ├── test_energy_conservation.cpp
│   │   ├── test_symplectic_property.cpp
│   │   ├── test_kahan_summation.cpp
│   │   ├── test_wave_equation.cpp
│   │   ├── test_boundary_wrapping.cpp
│   │   └── test_numerical_stability.cpp
│   ├── unit/
│   │   ├── test_nit.cpp
│   │   ├── test_coord9d.cpp
│   │   ├── test_emitter_array.cpp
│   │   └── CMakeLists.txt
│   └── integration/
│       ├── test_wave_propagation.cpp
│       ├── test_mamba_ssm.cpp
│       └── CMakeLists.txt
│
├── config/                          # Configuration files
│   ├── default.toml                 # Default system config
│   ├── physics_constants.toml       # Physical parameters
│   ├── hazards.db                   # Resonance firewall patterns
│   └── keys/                        # CurveZMQ keys (generated)
│       ├── public.key
│       └── secret.key
│
└── docs/                            # Documentation
    ├── architecture.md
    ├── api_reference.md
    ├── phase0_validation.md         # ⚡ Phase 0 checklist
    └── integration/                 # This documentation set

## 26.2 Implementation Guide - Mandated Organization

**CRITICAL:** To avoid "creative" organization, the engineering team MUST adhere to this exact directory mapping, which corresponds to architectural layers:

```
/src
  /core
    main.cpp              # Entry point, orchestrator initialization
    config_loader.cpp     # JSON/TOML configuration parsing
    
  /physics                # The 9D Substrate Layer
    torus_grid_soa.hpp    # ⚡ SoA Data Structure (The Substrate)
    integrator.cpp        # ⚡ Symplectic Split-Operator Solver
    ufie_kernels.cu       # CUDA Kernels for Laplacian/Nonlinearity
    kahan_sum.cpp         # ⚡ Compensated Summation
    shvo_grid.cpp         # Sparse Hyper-Voxel Octree logic
    metric.cpp            # Metric tensor operations
    emitter_array.cpp     # Golden ratio DDS emitters
    
  /cognitive              # The Cognitive Processing Layer
    mamba_tsm.cpp         # ⚡ TSM (Topology→Matrix mapper)
    transformer_np.cpp    # Neuroplastic Wave Attention
    hilbert_curve.cpp     # BMI2-optimized Hilbert scanning
    embedder.cpp          # Balanced nonary text encoder
    
  /autonomy               # The Autonomous Systems Layer
    engs_system.cpp       # Neurochemistry state machine
    dream_weave.cpp       # Counterfactual simulation engine
    dopamine.cpp          # Reward/learning modulation
    boredom.cpp           # Curiosity-driven exploration
    
  /infrastructure         # The Communication Backbone
    spine_broker.cpp      # ZeroMQ Router implementation
    kvm_manager.cpp       # Libvirt interface for Executors
    shared_memory.cpp     # ⚡ Seqlock zero-copy IPC
    proto/                # Compiled Protocol Buffers (.pb.cc)
    
  /types                  # The Arithmetic Foundation
    nit_avx512.cpp        # ⚡ Optimized Nonary Arithmetic (AVX-512)
    geometry.hpp          # 9D Coordinate utilities
    morton_code.cpp       # ⚡ 128-bit Z-order encoding
    
  /security               # The Safety and Validation Layer
    physics_oracle.cpp    # ⚡ Mathematical verification sandbox
    adversarial_dojo.cpp  # ⚡ Red team attack testing
    resonance_firewall.cpp # Spectral input filtering
    
  /persistence            # The Memory Durability Layer
    dmc.cpp               # Delta Memory Compression checkpoints
    lsm_dmc.cpp           # Log-Structured Merge persistence
    gguf_export.cpp       # Llama.cpp interoperability
    q9_encoder.cpp        # ⚡ Nonary quantization (Q9_0)
    
  /monitoring             # The Observability Layer
    energy_watchdog.cpp   # ⚡ Runtime conservation checks
    profiler.cpp          # ⚡ Performance tracking
```

### 26.2.1 Phase 0 Implementation Checklist (17-Day Sprint)

**Critical Path - Immediate Engineering Tasks:**

**Days 1-2:** Structure-of-Arrays Refactoring
- [ ] Create `include/nikola/physics/torus_grid_soa.hpp`
- [ ] Implement `TorusGridSoA` with 64-byte aligned vectors
- [ ] Implement 45-component metric tensor storage (upper triangular)
- [ ] Create `TorusNodeProxy` accessor class for API compatibility
- [ ] Refactor all grid access code to use proxy pattern
- [ ] Update CUDA kernels for coalesced memory access
- [ ] Validation: Physics kernel achieves <1ms per step on 27³ grid

**Days 3-5:** Split-Operator Symplectic Integration
- [ ] Create `include/nikola/physics/symplectic_integrator.hpp`
- [ ] Implement 6-step Strang splitting:
  - Half-kick damping (analytical exponential decay)
  - Half-kick conservative force (Laplacian + emitters)
  - Full drift (position update)
  - Nonlinear operator (RK2 implicit)
  - Half-kick force (recompute at new position)
  - Half-kick damping (final decay)
- [ ] Replace all Velocity-Verlet code
- [ ] Implement adaptive timestep monitoring
- [ ] Implement energy watchdog (compute Hamiltonian every 100 steps)
- [ ] Validation: Energy conservation within 0.01% over 24 hours

**Day 6:** Kahan Compensated Summation
- [ ] Create `include/nikola/physics/kahan_sum.hpp`
- [ ] Implement `KahanAccumulator` struct
- [ ] Refactor all Laplacian kernels to use Kahan summation
- [ ] Refactor all wave superposition operations
- [ ] Refactor metric tensor updates
- [ ] Validation: Preserve 10⁻⁶ amplitude waves over 10⁶ timesteps

**Day 7:** 128-bit Morton Code Hashing
- [ ] Create `include/nikola/types/morton_code.hpp`
- [ ] Implement BMI2-optimized bit interleaving
- [ ] Implement collision detection and double-hashing fallback
- [ ] Replace existing 64-bit Morton codes
- [ ] Validation: Zero collisions on 10⁷ random 9D coordinates

**Day 8:** Vectorized Nonary Arithmetic
- [ ] Create `include/nikola/types/nit_avx512.hpp`
- [ ] Implement `vec_sum_gate_avx512()` (64 trits parallel)
- [ ] Implement `vec_product_gate_avx512()` (heterodyning)
- [ ] Refactor all nonary operations to use SIMD
- [ ] Validation: 10x speedup vs scalar implementation

**Days 9-11:** Topological State Mapping (TSM)
- [ ] Create `src/cognitive/mamba_tsm.cpp`
- [ ] Implement `tsm_generate_parameters_kernel()`
- [ ] Extract metric tensor → Matrix A conversion
- [ ] Extract state dimension → Matrix B conversion
- [ ] Integrate with Hilbert curve scanner
- [ ] Validation: Mamba layers dynamically respond to metric changes

**Days 12-14:** Physics Oracle & Adversarial Dojo
- [ ] Create `include/nikola/security/physics_oracle.hpp`
- [ ] Implement 5 verification tests:
  - Energy conservation
  - Symplectic property
  - Wave equation validity
  - Boundary conditions (toroidal wrapping)
  - Numerical stability (NaN/Inf detection)
- [ ] Create `include/nikola/security/adversarial_dojo.hpp`
- [ ] Implement 10+ attack vectors
- [ ] Implement hot-swap protocol with Oracle gate
- [ ] Implement runtime energy watchdog
- [ ] Validation: All tests pass; attacks fail; 24-hour stability

**Days 15-16:** Integration & Testing
- [ ] Run full Phase 0 validation suite
- [ ] Profile memory bandwidth (should saturate DDR5)
- [ ] Profile energy conservation (should be <0.01% drift)
- [ ] Profile Laplacian accuracy (should preserve 10⁻⁶ amplitudes)
- [ ] Fix any identified issues

**Day 17:** Documentation & Handoff
- [ ] Document all Phase 0 implementations
- [ ] Create performance benchmark report
- [ ] Update README with Phase 0 status
- [ ] Tag repository as `v0.0.4-phase0-complete`

**Gate Requirement:** ALL checklist items must pass validation before Phase 1 begins.

**Final Directive:** Do not proceed to higher-level cognitive features until Physics Oracle confirms energy stability for >24 hours of continuous operation.

---

**Cross-References:**
- See `08_phase_0_requirements/01_critical_fixes.md` for detailed specifications
- See `11_appendices/04_hardware_optimization.md` for AVX-512 requirements
- See `09_implementation/03_implementation_checklist.md` for complete task list
│   ├── unit/
│   │   ├── test_nonary.cpp
│   │   ├── test_coord9d.cpp
│   │   ├── test_wave_interference.cpp
│   │   ├── test_hilbert.cpp
│   │   ├── test_engs.cpp
│   │   ├── test_neuroplasticity.cpp
│   │   └── CMakeLists.txt
│   ├── integration/
│   │   ├── test_search_retrieve.cpp
│   │   ├── test_training.cpp
│   │   ├── test_multimodal.cpp
│   │   └── CMakeLists.txt
│   └── benchmarks/
│       ├── bench_propagation.cpp
│       ├── bench_hilbert.cpp
│       └── CMakeLists.txt
│
├── docker/                          # Docker files
│   ├── Dockerfile.base              # Base image
│   ├── Dockerfile.runtime           # Runtime image
│   └── gold-image/                  # VM gold image
│       └── ubuntu-24.04.qcow2
│
├── config/                          # Configuration
│   ├── nikola.conf                  # Main config
│   ├── emitters.conf                # Frequency settings
│   └── security.conf                # Firewall patterns
│
└── docs/                            # Documentation
    ├── architecture.md
    ├── api_reference.md
    └── troubleshooting.md
```

## 26.2 File Manifest

**Total Files:** ~150
**Total Lines of Code (estimated):** ~50,000

**Critical Path Files (Must implement first):**

1. `include/nikola/types/nit.hpp` - Balanced nonary enum
2. `include/nikola/types/torus_node.hpp` - Node structure
3. `include/nikola/physics/torus_manifold.hpp` - Grid
4. `include/nikola/physics/emitter_array.hpp` - DDS
5. `src/physics/wave_engine.cpp` - Interference processor
6. `proto/neural_spike.proto` - Message protocol
7. `src/spine/broker.cpp` - Communication backbone

## 26.3 Key Implementation Files by Subsystem

### Physics Engine (Core)
- `types/nit.hpp/cpp` - Balanced nonary arithmetic
- `physics/torus_manifold.hpp/cpp` - 9D sparse grid
- `physics/emitter_array.hpp/cpp` - Golden ratio DDS
- `physics/wave_engine.cpp` - Superposition/heterodyning
- `physics/shvo_grid.cpp` - Sparse hyper-voxel octree
- `physics/kernels/wave_propagate.cu` - CUDA acceleration

### Cognitive Systems
- `mamba/hilbert_scan.cpp` - Space-filling curve scanner
- `mamba/ssm_kernel.cpp` - State space model
- `reasoning/transformer.cpp` - Neuroplastic transformer
- `reasoning/wave_attention.cpp` - Wave correlation
- `reasoning/embedder.cpp` - Text-to-waveform

### Infrastructure
- `spine/broker.cpp` - ZeroMQ message router
- `spine/shadow_spine.cpp` - A/B testing infrastructure
- `orchestrator/smart_router.cpp` - Tool selection
- `agents/*.cpp` - External API clients
- `executor/kvm_executor.cpp` - Sandboxed execution

### Autonomy
- `autonomy/engs.cpp` - Extended neurochemistry
- `autonomy/dopamine.cpp` - Reward system
- `autonomy/boredom.cpp` - Curiosity-driven learning
- `autonomy/goals.cpp` - Hierarchical goal DAG
- `autonomy/dream_weave.cpp` - Counterfactual simulation
- `autonomy/trainers.cpp` - Autonomous training

### Persistence & Safety
- `persistence/lsm_dmc.cpp` - Log-structured persistence
- `persistence/gguf_export.cpp` - GGUF interoperability
- `security/resonance_firewall.cpp` - Attack detection
- `security/csvp.cpp` - Code safety verification
- `self_improve/adversarial_dojo.cpp` - Red team testing

### Multimodal
- `multimodal/audio_resonance.cpp` - FFT-based audio
- `multimodal/visual_cymatics.cpp` - Holographic vision

---

**Cross-References:**
- See Section 27 for Development Roadmap
- See Section 28 for Implementation Checklist
- See Appendices for build system details
