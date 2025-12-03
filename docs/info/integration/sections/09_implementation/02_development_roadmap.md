# DEVELOPMENT ROADMAP

## 27.1 Phase 1: Core Physics Engine (Months 1-3)

**Milestone:** Standing waves propagate correctly in 9D

**Tasks:**

| Week | Task | Deliverable |
|------|------|-------------|
| 1-2 | Implement `Nit` enum and nonary arithmetic | Unit tests pass |
| 3-4 | Implement `TorusNode` structure with metric tensor | Structure defined |
| 5-6 | Implement sparse `TorusManifold` grid (SHVO) | Grid can be created |
| 7-8 | Implement `EmitterArray` with DDS | Emitters generate signals |
| 9-10 | Implement wave propagation kernel | Waves propagate |
| 11-12 | Optimize with AVX-512/CUDA | Performance targets met |

**Validation Criteria:**

- [ ] Nonary addition: $1 + (-1) = 0$
- [ ] Wave superposition creates interference patterns
- [ ] Energy conserved over 1000 time steps
- [ ] Performance: <1ms per physics step (sparse 27³ grid)
- [ ] Toroidal wrapping works correctly (GEO-TOPO-01 fixed)

## 27.2 Phase 2: Logic and Memory (Months 4-6)

**Milestone:** Store text as wave, retrieve via resonance

**Tasks:**

| Week | Task | Deliverable |
|------|------|-------------|
| 13-14 | Implement balanced nonary arithmetic gates | Gates work |
| 15-16 | Build `NonaryEmbedder` (text → wave) | Embedder functional |
| 17-18 | Integrate LMDB storage backend | DB stores/loads nodes |
| 19-20 | Implement search-retrieve-store loop | Basic memory works |
| 21-22 | Implement LSM-DMC persistence (.nik format) | State persists |
| 23-24 | Validate memory accuracy over sessions | Retrieval >90% accurate |

**Validation Criteria:**

- [ ] Text → Waveform → Text roundtrip works
- [ ] Resonance detection finds stored patterns
- [ ] LSM-DMC saves and loads state correctly (PER-LSM-01 fixed)
- [ ] Merkle tree detects corruption
- [ ] Nap consolidation triggers correctly

## 27.3 Phase 3: The Brain (Months 7-9)

**Milestone:** System demonstrates learning

**Tasks:**

| Week | Task | Deliverable |
|------|------|-------------|
| 25-26 | Implement Mamba-9D Hilbert scanner | Scanner works |
| 27-28 | Port Transformer to Wave Correlation | Transformer operational |
| 29-30 | Implement Neuroplasticity (metric updates) | Learning observable |
| 31-32 | Implement Neurogenesis (grid expansion) | Grid grows when needed |
| 33-34 | Build autonomous trainers (BAT) | Training runs automatically |
| 35-36 | Benchmark retrieval accuracy improvements | Accuracy improves >10% |

**Validation Criteria:**

- [ ] Hilbert scan visits all nodes
- [ ] Wave correlation attention works
- [ ] Metric tensor contracts with co-activation
- [ ] New nodes created when saturated
- [ ] Repeated queries answered faster
- [ ] Topological State Mapping functional (WP2)

## 27.4 Phase 4: Integration and Agents (Months 10-11)

**Milestone:** Full autonomous system

**Tasks:**

| Week | Task | Deliverable |
|------|------|-------------|
| 37-38 | Build ZeroMQ Spine with CurveZMQ security | Spine operational |
| 39-40 | Integrate Tavily/Firecrawl/Gemini APIs | Agents work |
| 41-42 | Implement KVM Executor with libvirt | VMs spawn and execute |
| 43-44 | Build twi-ctl CLI controller | CLI functional |
| 45-46 | Implement auto-ingestion pipeline (inotify) | Files ingested automatically |
| 47-48 | Finalize Docker multi-stage build | Docker image builds |

**Validation Criteria:**

- [ ] All components communicate via Spine
- [ ] External tools fetch data correctly
- [ ] Executor runs sandboxed commands safely
- [ ] CLI responds to all commands
- [ ] Files dropped in folder are ingested
- [ ] Shadow Spine Protocol operational (WP4)

## 27.5 Phase 5: Autonomy and Evolution (Month 12)

**Milestone:** Self-improving AGI

**Tasks:**

| Week | Task | Deliverable |
|------|------|-------------|
| 49-50 | Implement ENGS (Dopamine/Serotonin/Norepinephrine) | Neurochemistry works |
| 50 | Implement Boredom/Curiosity and Goal systems | Autonomy functional |
| 51 | Build Resonance Firewall | Security operational |
| 52 | Implement Self-Improvement loop with CSVP | System improves itself |
| 53 | Implement Adversarial Code Dojo | Red Team testing works |
| 54 | Build GGUF export pipeline | GGUF export works |
| 55 | Security hardening and audit | Security checklist complete |
| 56 | Final integration testing | All systems operational |

**Validation Criteria:**

- [ ] Dopamine modulates learning rate correctly
- [ ] Exponential decay achieves homeostasis (AUTO-ENGS-01 fixed)
- [ ] ENGS couples with physics kernel (PHY-CUDA-01 fixed)
- [ ] Boredom triggers curiosity
- [ ] Goals provide structure
- [ ] Firewall blocks known attacks
- [ ] CSVP prevents unsafe code modifications
- [ ] System identifies and patches bottlenecks
- [ ] Dream-Weave counterfactual learning works
- [ ] GGUF file loads in llama.cpp

## 27.6 Remaining Work Items (Post-Launch)

**P1 Priority (150 LOC):**
- [ ] PHY-MEM-01: GPU neighbor map updates after neurogenesis

**P2 Priority (131 LOC):**
- [ ] MM-AUD-01: Dynamic folding limit for audio (1 LOC fix)
- [ ] MM-VIS-01: Holographic RGB encoding refactor (80 LOC)
- [ ] AUTO-DREAM-01: Z-score normalization (50 LOC)

**Total Remaining Effort:** ~281 LOC for 100% defect closure

## 27.7 Timeline Summary

| Phase | Duration | Milestone | Completion |
|-------|----------|-----------|------------|
| Phase 1 | Months 1-3 | Physics Engine | Core functional |
| Phase 2 | Months 4-6 | Memory | Storage works |
| Phase 3 | Months 7-9 | Learning | System learns |
| Phase 4 | Months 10-11 | Integration | Full system |
| Phase 5 | Month 12 | Autonomy | AGI complete |

**Total Development Time:** 12 months (5-person team)
**System Readiness at Launch:** 95% complete

---

**Cross-References:**
- See Section 26 for File Structure
- See Section 28 for Detailed Checklist
- See WP5 for Remediation status
