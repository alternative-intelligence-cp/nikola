# FILE STRUCTURE

## 26.0 Engineering Audit Updates

**Date:** December 7, 2025  
**Source:** Engineering Report Review and Analysis v0.0.4

The file structure has been updated to include Phase 0 audit remediation components. Key additions:

- `include/nikola/physics/soa_layout.hpp` - Structure-of-Arrays memory layout
- `include/nikola/physics/symplectic_integrator.hpp` - Split-operator integration
- `include/nikola/security/physics_oracle.hpp` - Self-improvement safety
- `src/physics/kernels/wave_propagate_soa.cu` - SoA CUDA kernel
- `tests/validation/` - Audit validation test suite

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
│   ├── validate_phase0/             # ⚡ Audit validation (Phase 0)
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
