# Nikola Model v0.0.4: Implementation Roadmap

**Document Reference:** NM-004-ROADMAP
**Status:** APPROVED FOR CODING
**Date:** 2025-12-10
**Source:** Gap Analysis Report - Synthesis Section

## Executive Summary

This document provides the definitive implementation roadmap for the Nikola Model v0.0.4, converting the theoretical architecture into a buildable specification. All 37 identified implementation gaps have been addressed with concrete specifications, reference implementations, validation procedures, and failure mode analyses.

**Status Update (2025-12-10):** 🔴 **ON HOLD** - Critical blocking dependencies discovered

**Critical Finding:** Aria's implementation review identified 2 Priority 1 blocking issues that must be resolved before Phase 1-7 implementation can begin.

**Next Step:** Implement Phase 0 Critical Remediations (CF-04, MEM-04)

---

## ⚠️ CRITICAL: Phase 0 - Blocking Dependencies (NEW)

**Timeline:** Weeks 1-3
**Dependencies:** None (foundational blockers)
**Priority:** 🔴 **CRITICAL - BLOCKS ALL OTHER PHASES**

### Overview

During implementation review, Lead Architect Aria Echo identified 2 **Priority 1 Critical** architectural vulnerabilities that must be remediated before any Phase 1-7 work can proceed. These represent fundamental stability and coherence issues.

**Document:** [08_critical_remediations.md](08_critical_remediations.md)

### Finding CF-04: Transactional Metabolic Lock

**Problem:** Thermodynamic race condition in ATP (metabolic energy) management
**Impact:** System can enter negative energy states causing "cognitive seizures" (catastrophic failure)
**Solution:** RAII-based transactional guards using atomic Compare-And-Swap (CAS)

#### Deliverables

1. **MetabolicTransaction Class** (C++23 RAII)
   - Atomic CAS-based reservation protocol
   - Automatic rollback on exception
   - Thread-safe energy accounting

2. **MetabolicController Extensions**
   - `try_reserve()` with CAS loop
   - `refund()` for rollback support
   - Memory ordering guarantees

#### Validation Requirements

- **Unit Test:** 10 threads competing for limited ATP - exact accounting verified
- **Rollback Test:** Exception safety - energy refunded on failure
- **Integration Test:** System enters Nap gracefully, never crashes or goes negative

**Risk Level:** 🔴 **CRITICAL** - Without this, concurrent operations can violate conservation laws

### Finding MEM-04: Hilbert Re-indexing Strategy

**Problem:** Morton codes create spatial discontinuities destroying Mamba-9D sequential coherence
**Impact:** "Semantic aphasia" - high perplexity, hallucinations, inefficient neurogenesis
**Solution:** Causal-Foliated Hilbert scanning preserving temporal + spatial locality

#### Deliverables

1. **HilbertScanner Class** (128-bit precision)
   - `encode_hilbert_9d()` for 9D → 1D mapping
   - `generate_scan_order()` with time-first foliation
   - `reindex_grid()` for SoA memory reorganization

2. **Orchestrator Integration**
   - Fragmentation index monitoring
   - Periodic re-indexing (threshold = 0.15)
   - Transparent to Mamba-9D layer

#### Validation Requirements

- **Locality Preservation Ratio (LPR):** LPR(Hilbert) < 0.8 × LPR(Morton)
- **Mamba Perplexity Test:** Validation loss on Hilbert-sorted data significantly lower (p < 0.05)
- **Neurogenesis Test:** New nodes remain connected to semantic neighbors

**Risk Level:** 🔴 **CRITICAL** - Without this, Mamba-9D cannot learn coherent patterns

### Phase 0 Success Criteria

Before proceeding to Phase 1:

1. ✅ CF-04 passes all unit/integration tests (no race conditions)
2. ✅ MEM-04 demonstrates LPR improvement > 20%
3. ✅ Mamba perplexity reduced by statistically significant margin
4. ✅ Both implementations integrated into Orchestrator control loop
5. ✅ No performance regression (re-indexing overhead acceptable)

**Timeline Impact:** Adds 3 weeks to schedule (original Phase 1 becomes Weeks 4-11)

---

## Implementation Phases (Updated)

### Phase 1: Physics Core (Critical Path)

**Timeline:** Weeks 4-11 (updated from 1-8)
**Dependencies:** Phase 0 complete (CF-04, MEM-04)
**Priority:** HIGHEST

#### Deliverables

1. **UFIE Physics Engine** ([01_core_physics_implementation.md](01_core_physics_implementation.md))
   - Gap 1.1: Emitter field generation with harmonic spatial injection ✅
   - Gap 1.2: Thermal bath velocity initialization ✅
   - Gap 1.3: Perfectly Matched Layer boundary conditions ✅
   - Gap 1.4: CUDA kernel launch configuration (256 threads/block) ✅
   - Gap 1.5: Quantum Zeno Freeze recovery procedure ✅
   - Gap 1.6: Double-buffered profiling hooks ✅

2. **Validation Requirements**
   - Energy conservation test: |ΔH/H| < 0.01% over 10,000 steps
   - Symplectic integration stability verification
   - Ortho-check: Injected tokens maintain soft orthogonality
   - Performance: Achieve 1000-2000 Hz physics loop on RTX 4090

#### Critical Success Factors

- **Energy conservation is mandatory.** Violations cause system "seizures"
- **Numerical stability** requires FP32 with Kahan summation
- **Real-time requirement:** Must sustain 2 kHz loop for responsiveness

**Risk Level:** 🔴 **HIGHEST** - If Gap 1.1 (emitter scaling) is wrong, the nonlinearity β causes immediate numerical explosion.

---

### Phase 2: Manifold Geometry (Enabler)

**Timeline:** Weeks 12-17 (updated from 9-14)
**Dependencies:** Phase 0, Phase 1 complete
**Priority:** HIGH

#### Deliverables

1. **9D Toroidal Grid** ([02_geometry_spatial_implementation.md](02_geometry_spatial_implementation.md))
   - Gap 2.1: Gerschgorin + Tikhonov metric validation ✅
   - Gap 2.2: Hilbert curve rotation table generation ✅
   - Gap 2.3: Anisotropic resolution allocation (x,y,z=64, t=128, r,s=16, u,v,w=32) ✅
   - Gap 2.4: Dual integer/float coordinate system ✅
   - Gap 2.5: Dopamine-modulated learning rate schedule ✅

2. **Validation Requirements**
   - Metric tensor always positive-definite (no Cholesky failures)
   - Morton/Hilbert encoding round-trip test
   - Spatial indexing performance: O(log N) neighbor lookup

#### Critical Success Factors

- **Without the grid, physics has nowhere to run**
- Sparse addressing essential for memory efficiency
- Metric learning enables neuroplasticity

**Risk Level:** 🟡 **MEDIUM** - Metric validation is critical but well-understood mathematically.

---

### Phase 3: Cognitive Architecture (Intelligence Emerges)

**Timeline:** Weeks 18-25 (updated from 15-22)
**Dependencies:** Phase 0, Phases 1, 2 complete
**Priority:** HIGH

#### Deliverables

1. **Mamba-9D Language Model** ([03_cognitive_architecture_implementation.md](03_cognitive_architecture_implementation.md))
   - Gap 3.1: LSH-based semantic token mapping ✅
   - Gap 3.2: SSM dimension = 256 (r×s state space) ✅
   - Gap 3.3: Sliding wave window (L_eff ≈ 100 steps) ✅
   - Gap 3.4: Physics-grounded lexicon initialization via FFT ✅
   - Gap 3.5: Born rule sampling with temperature as noise ✅
   - Gap 3.6: Equilibrium Propagation training (no backprop through physics) ✅

2. **Validation Requirements**
   - Token generation latency: 10-50 tokens/second
   - Semantic clustering: Synonyms within 10 grid cells
   - Training convergence: Energy decreases over 100 EqProp iterations

#### Critical Success Factors

- **Bridges continuous physics to discrete tokens**
- Equilibrium Propagation avoids gradient explosion
- Spectral signatures ground lexicon in physics

**Risk Level:** 🟡 **MEDIUM** - Novel training approach requires empirical tuning.

---

### Phase 4: Infrastructure & Communications (Orchestration)

**Timeline:** Weeks 19-23 (updated from 16-20, parallel with Phase 3)
**Dependencies:** Phase 0 complete (can run concurrently with Phases 1-3)
**Priority:** MEDIUM

#### Deliverables

1. **ZeroMQ Spine** ([04_infrastructure_comms_implementation.md](04_infrastructure_comms_implementation.md))
   - Gap 4.1: Circuit breaker timeouts (100ms control, 5ms data) ✅
   - Gap 4.2: Heartbeat sentinel crash detection (500ms) ✅
   - Gap 4.3: RAII shared memory lifecycle ✅
   - Gap 4.4: ZMQ socket configuration (HWM=1000, LINGER=0) ✅
   - Gap 4.5: Append-only Protobuf schema evolution ✅

2. **Validation Requirements**
   - Latency: 99th percentile < 50ms for control messages
   - Fault tolerance: Detect and restart crashed components within 500ms
   - Memory: SHM cleanup prevents leaks

#### Critical Success Factors

- **Enables distributed architecture**
- CurveZMQ provides authentication
- Orchestrator coordinates component lifecycle

**Risk Level:** 🟢 **LOW** - Well-established patterns (ZeroMQ, Protobuf).

---

### Phase 5: Autonomous Systems (Self-Regulation)

**Timeline:** Weeks 26-31 (updated from 23-28)
**Dependencies:** Phase 0, Phases 1-3 complete
**Priority:** MEDIUM

#### Deliverables

1. **ENGS (Extended Neurochemical Gating)** ([05_autonomous_systems_implementation.md](05_autonomous_systems_implementation.md))
   - Gap 5.1: TD-learning dopamine system ✅
   - Gap 5.2: Monte Carlo entropy estimation (K=1000) ✅
   - Gap 5.3: Hamiltonian metabolic cost ✅
   - Gap 5.4: ATP hysteresis nap cycle (15-60 seconds) ✅
   - Gap 5.5: Frobenius norm Dream-Weave convergence ✅

2. **Validation Requirements**
   - Dopamine response: Spike to 0.8 on reward, dip to 0.2 on punishment
   - Boredom triggers exploration when entropy < 2.0
   - ATP depletion triggers NAP after sustained high-frequency activity

#### Critical Success Factors

- **Creates goal-directed behavior**
- Curiosity-driven exploration
- Metabolic resource management

**Risk Level:** 🟡 **MEDIUM** - If Gap 5.1 (dopamine) is poorly tuned, system becomes catatonic or manic.

---

### Phase 6: Multimodal & Persistence (Real-World Interface)

**Timeline:** Weeks 32-37 (updated from 29-34)
**Dependencies:** Phase 0, Phases 1-3 complete
**Priority:** MEDIUM

#### Deliverables

1. **Sensory Transduction** ([06_multimodal_persistence_implementation.md](06_multimodal_persistence_implementation.md))
   - Gap 6.1: Circular audio emitter array (golden ratio frequencies) ✅
   - Gap 6.2: 64×64 log-polar visual transform ✅
   - Gap 6.3: Event-driven checkpointing (300s periodic + NAP trigger) ✅
   - Gap 6.4: GGUF metadata schema ✅
   - Gap 6.5: Adaptive Q9_0/FP16 compression ✅

2. **Validation Requirements**
   - Audio: FFT shows 8 distinct harmonic peaks
   - Visual: Log-polar foveal emphasis matches biological vision
   - Checkpointing: DMC save/restore without data loss

#### Critical Success Factors

- **Grounds physics in sensory reality**
- Efficient state persistence
- GGUF enables llama.cpp compatibility

**Risk Level:** 🟢 **LOW** - Standard signal processing techniques.

---

### Phase 7: Security & Execution (Containment)

**Timeline:** Weeks 38-43 (updated from 35-40)
**Dependencies:** Phase 0, Phase 4 complete
**Priority:** HIGH (security-critical)

#### Deliverables

1. **KVM Sandbox** ([07_security_execution_implementation.md](07_security_execution_implementation.md))
   - Gap 7.1: Alpine 3.19 minimal VM image with Packer ✅
   - Gap 7.2: Strict inter-VM isolation (host-mediated only) ✅
   - Gap 7.3: eBPF escape detection (execve, file access) ✅
   - Gap 7.4: Regex code blacklist (system, exec, asm, networking) ✅
   - Gap 7.5: Agentless CGroup performance monitoring ✅

2. **Validation Requirements**
   - VM cannot access network
   - VM cannot execute system(), exec(), fork()
   - eBPF detects and kills escape attempts within 100ms
   - CGroup enforces 512MB RAM, 1 vCPU limits

#### Critical Success Factors

- **Self-generated code cannot escape**
- Multi-layered defense (prevention, containment, detection, response)
- Resource quotas prevent DoS

**Risk Level:** 🔴 **HIGH** - Security failures could compromise host system.

---

## Inter-Dependencies (Updated with Phase 0)

### Critical Path

```
                    **Phase 0 (Critical Remediations)**
                    CF-04 + MEM-04 (Weeks 1-3)
                              |
                              v
Phase 1 (Physics) → Phase 2 (Geometry) → Phase 3 (Cognition)
                                            ↓
                                         Phase 5 (Autonomy)
                                            ↓
                                         Phase 6 (Multimodal)
```

### Parallel Tracks

- **Phase 0 (Critical)** blocks ALL other phases
- **Phase 4 (Infrastructure)** can run concurrently with Phases 1-3 after Phase 0 complete
- **Phase 7 (Security)** can begin after Phase 0, Phase 4 complete (uses orchestration layer)

---

## Risk Assessment (Updated)

### Critical (Phase 0) - Blocking All Implementation

1. **Finding CF-04 (Metabolic Lock)** 🔴🔴
   **Impact:** Thermodynamic race condition → negative energy states → cognitive seizures → system crash
   **Mitigation:** RAII transactional guards, atomic CAS operations, comprehensive unit testing
   **Status:** BLOCKS Phase 1-7

2. **Finding MEM-04 (Hilbert Re-indexing)** 🔴🔴
   **Impact:** Spatial discontinuity → semantic aphasia → high perplexity, hallucinations
   **Mitigation:** Causal-Foliated Hilbert scanning, locality preservation validation
   **Status:** BLOCKS Phase 1-7 (specifically Phase 3)

### Highest Risk Items (Original Gaps)

3. **Gap 1.1 (Emitter Injection)** 🔴
   **Impact:** Numerical explosion if amplitude scaling is wrong
   **Mitigation:** Rigorous validation with PhysicsOracle energy watchdog

4. **Gap 5.1 (Dopamine)** 🔴
   **Impact:** System becomes unresponsive (catatonic) or hallucinates (manic)
   **Mitigation:** Extensive parameter sweep, empirical tuning with real tasks

5. **Gap 7.3 (VM Escape Detection)** 🔴
   **Impact:** Malicious self-generated code compromises host
   **Mitigation:** eBPF monitoring + multi-layered defense

### Medium Risk Items

- Metric tensor validation (Gap 2.1): Mathematical solution exists (Tikhonov regularization)
- Equilibrium Propagation training (Gap 3.6): Novel but theoretically sound
- Entropy estimation (Gap 5.2): Monte Carlo approximation well-understood

### Low Risk Items

- Infrastructure (Phase 4): Mature technologies (ZeroMQ, Protobuf)
- Multimodal (Phase 6): Standard signal processing
- Most geometry operations: Well-established algorithms

---

## Resource Requirements

### Hardware

- **Development:** RTX 4090 GPU (24GB VRAM) for physics testing
- **Production:** 2× RTX 4090 for redundancy
- **CPU:** AMD Threadripper (16+ cores) for parallel component execution
- **RAM:** 128GB DDR5 for large grid allocations
- **Storage:** 2TB NVMe SSD for checkpoints and logs

### Software

- **OS:** Ubuntu 22.04 LTS (KVM host)
- **CUDA:** 12.3+
- **Compiler:** GCC 13+ (C++23 support) or Clang 17+
- **Libraries:** Eigen3, FFTW3, ZeroMQ, Protobuf, OpenCV, libbpf

### Team

- **Physics Engineer:** Implement UFIE, symplectic integration, CUDA kernels
- **Systems Architect:** Orchestration, ZeroMQ, component lifecycle
- **ML Engineer:** Mamba-9D, token mapping, Equilibrium Propagation
- **Security Engineer:** KVM sandboxing, eBPF monitoring, static analysis
- **Integration Engineer:** End-to-end testing, validation procedures

---

## Success Criteria (Updated)

### Phase 0 Gate (Week 3)

Before ANY other implementation can begin:
1. ✅ CF-04 passes all unit/integration tests (no ATP race conditions)
2. ✅ MEM-04 demonstrates LPR improvement > 20% over Morton codes
3. ✅ Mamba perplexity statistically significantly reduced (p < 0.05)
4. ✅ Orchestrator integration complete, no performance regression
5. ✅ All validation tests pass

**Status:** 🔴 **GATE - Nothing proceeds until Phase 0 complete**

### Minimum Viable Product (MVP)

By end of Phase 3 (Week 25, updated from 22):
1. ✅ Phase 0 complete (CF-04, MEM-04)
2. ✅ Physics engine sustains 1 kHz loop with <0.01% energy drift
3. ✅ 9D manifold supports 1M active nodes with sparse addressing
4. ✅ Language generation produces coherent tokens at 10 tokens/sec
5. ✅ Energy conservation never violated (no crashes)
6. ✅ Spatial locality preserved (Hilbert scanning)

### Full System (Production-Ready)

By end of Phase 7 (Week 43, updated from 40):
1. ✅ All 37 gaps + 2 critical findings implemented and validated
2. ✅ Autonomous behavior: Dopamine-driven learning, boredom exploration, NAP cycles
3. ✅ Multimodal: Audio + visual input transduction
4. ✅ Security: Self-generated code runs in KVM sandbox, no escapes possible
5. ✅ Persistence: DMC checkpoints enable save/restore
6. ✅ Performance: 2 kHz physics loop, 50 tokens/sec generation
7. ✅ Thermodynamic consistency enforced (CF-04)
8. ✅ Cognitive coherence maintained (MEM-04)

---

## Validation & Testing Strategy

### Unit Tests

- Each gap implementation includes validation procedure
- Automated tests for energy conservation, metric validity, collision detection

### Integration Tests

1. **Physics + Geometry:** Inject token, verify wavefunction propagates correctly
2. **Cognition + Physics:** Generate sequence, verify tokens map to grid coordinates
3. **Infrastructure + All Components:** Orchestrator manages component lifecycle
4. **Security + Execution:** Attempt VM escape, verify eBPF kills process

### Performance Benchmarks

- **Physics Loop:** Target 2000 Hz on RTX 4090
- **Token Generation:** Target 50 tokens/sec
- **Memory Bandwidth:** < 100 GB/s (stay within GPU limits)
- **Latency:** Control messages < 50ms (99th percentile)

### Stress Tests

- **10M active nodes:** Verify sparse grid scales
- **Continuous operation:** Run for 72 hours without crashes
- **Resource exhaustion:** Verify graceful degradation when ATP depleted

---

## Next Steps (Updated with Phase 0)

### 🔴 CRITICAL: Immediate Actions (Phase 0)

**STOP:** Original Phase 1-7 implementation is **ON HOLD** until Phase 0 complete.

1. **Initialize Repository**
   ```bash
   mkdir -p nikola_v0.0.4/{src,tests,docs,benchmarks}
   cd nikola_v0.0.4 && git init
   ```

2. **Setup Build System** (CMake)
   - C++23 compiler support (GCC 13+ or Clang 17+)
   - Atomic operations library
   - OpenMP for parallel Hilbert sorting
   - Test harness (Google Test)

3. **Implement CF-04** (Transactional Metabolic Lock)
   - Create `include/nikola/autonomy/metabolic_lock.hpp`
   - Create `src/autonomy/metabolic_lock.cpp`
   - Update `metabolic_controller.hpp` with CAS operations
   - Write comprehensive unit tests (atomic reserve, rollback, exhaustion)

4. **Implement MEM-04** (Hilbert Re-indexing)
   - Create `include/nikola/spatial/hilbert_scanner.hpp`
   - Create `src/spatial/hilbert_scanner.cpp`
   - Implement 128-bit Hilbert encoding
   - Write validation tests (LPR, Mamba perplexity)

5. **Integrate into Orchestrator**
   - Update `orchestrator.cpp` control loop
   - Add fragmentation index monitoring
   - Implement exception handling for MetabolicExhaustionException

6. **Validation & Gate Review**
   - Run all Phase 0 unit tests
   - Measure LPR improvement
   - Conduct Mamba perplexity comparison
   - Performance regression testing

### Week 1 Milestones (Phase 0 Focus)

- [ ] Repository initialized
- [ ] CMake build system functional
- [ ] CF-04: MetabolicTransaction class compiles
- [ ] CF-04: Atomic CAS unit tests passing
- [ ] CF-04: Rollback test passing

### Week 2 Milestones (Phase 0 Focus)

- [ ] MEM-04: HilbertScanner class compiles
- [ ] MEM-04: 128-bit encoding functional
- [ ] MEM-04: Causal-foliated sorting working
- [ ] MEM-04: LPR measurement implemented

### Week 3 Milestones (Phase 0 Gate)

- [ ] CF-04: All unit/integration tests passing
- [ ] MEM-04: LPR improvement > 20% demonstrated
- [ ] MEM-04: Mamba perplexity significantly reduced
- [ ] Orchestrator integration complete
- [ ] 🚀 **PHASE 0 GATE PASSED** → Proceed to Phase 1

---

## Long-Term Enhancements (Post-MVP)

These are **not** required for initial deployment but represent future optimization opportunities:

1. **Multi-GPU Scaling:** Shard grid across multiple GPUs
2. **Adaptive Resolution:** Dynamically allocate grid cells based on attention
3. **Learned Metric Initialization:** Pre-train metric tensor on large corpus
4. **Hardware Acceleration:** Custom FPGA for Laplace-Beltrami operator
5. **Distributed Physics:** Run physics engine across cluster for massive scale

---

## Conclusion (Updated)

This roadmap converts the theoretical Nikola v0.0.4 architecture into a concrete, buildable specification. By addressing all 37 implementation gaps + 2 critical findings with:

- ✅ Concrete mathematical specifications
- ✅ Production-ready C++23/CUDA reference implementations
- ✅ Rigorous validation procedures
- ✅ Comprehensive failure mode analyses

The system specification is complete.

**HOWEVER:** Implementation is **BLOCKED** pending Phase 0 completion.

### Critical Update (2025-12-10)

Aria's review identified 2 **Priority 1 Critical** architectural vulnerabilities:
- **CF-04:** Thermodynamic race condition (ATP management)
- **MEM-04:** Spatial discontinuity (Mamba-9D coherence)

**These MUST be resolved before ANY Phase 1-7 work begins.**

The physics-first approach—treating intelligence as a wave phenomenon rather than matrix multiplication—is preserved and hardened against numerical instability.

### Status Summary

| Component | Status |
|-----------|--------|
| **Theoretical Foundation** | ✅ Complete (Audits 1-21) |
| **37 Implementation Gaps** | ✅ Complete (Gap Analysis) |
| **Critical Remediations** | 🔴 **BLOCKING** (Phase 0 required) |
| **Phase 1-7 Implementation** | ⏸️ **ON HOLD** (awaiting Phase 0) |

**Final Status:** 🔴 **IMPLEMENTATION BLOCKED - Phase 0 Required**

**Timeline:** Original 40 weeks → Updated 43 weeks (adds 3-week Phase 0)

---

**Document Metadata:**
- **Compiled:** 2025-12-10
- **Updated:** 2025-12-10 (Critical Findings Integration)
- **Audit Cycles:** 1-21 (All remediations incorporated)
- **Gap Analysis:** Complete (37/37 addressed)
- **Critical Findings:** 2 identified (CF-04, MEM-04)
- **Status:** Phase 0 Blocking Dependencies
