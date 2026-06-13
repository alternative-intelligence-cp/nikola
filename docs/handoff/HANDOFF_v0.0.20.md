# Nikola v0.0.20 Handoff — Pre-Release Audit & Documentation

_Completed: 2026-04-09_

---

## What Was Completed

### Pre-Release Audit of the v0.0.x Prototype Series

v0.0.20 is the final release in the 0.0.x prototype series. No new features
were added. This release audits the entire codebase, fixes security
vulnerabilities, updates all documentation, and establishes the performance
baseline for the v0.1.0 milestone.

---

## Security Fixes

### CRITICAL: Gate 1 Empty Source Bypass (Fixed)
- **File**: `src/autonomy/evolutionary_orchestrator.cpp`
- Empty `source_code` in `run_cycle()` previously skipped the blacklist scan entirely
- Now returns `SECURITY_REJECTED` — no .so can proceed without source verification
- Tests updated in Phase113 + Phase114 (17 call sites)

### HIGH: JSON Escape Injection (Fixed)
- **File**: `include/nikola/nitpick/specialist_interface.hpp`
- `extract_json_string()` passed invalid escapes through verbatim
- Now handles `\b`, `\f`, `\/`, `\uXXXX` per RFC 7159; drops invalid escapes

### MEDIUM: SCRAM Threshold Mismatch (Fixed)
- **File**: `include/nikola/physics/propagator.hpp`
- `evolve()` default tolerance aligned from 1e-4 to 1e-5 (matches PhysicsOracle)

---

## Documentation Updates

| Document | Changes |
|----------|---------|
| ARCHITECTURE.md | Added §8 Multimodal Input, §9 Nitpick Specialist Integration |
| BUILD_GUIDE.md | Test count → 163, added Nitpick SIE build instructions |
| GOTCHAS.md | 3 new gotchas (Phase142 timeout, nitpickc -c, LMDB dirs) |
| MODULE_REFERENCE.md | Added Multimodal + Nitpick namespaces (158 total headers) |
| **docs/api/README.md** | **New** — Full API reference: 24 namespaces, core types |
| **docs/CHANGELOG_v0.0.x.md** | **New** — Complete v0.0.4–v0.0.20 changelog |
| **docs/AUDIT_v0.0.20.md** | **New** — Comprehensive technical audit report |

---

## Test Suite

| Metric | Value |
|--------|-------|
| Total targets | 163 |
| Pass rate | **163/163 (100%)** |
| Timeout policy | 300s for Phase142 calibration |
| Framework | Catch2 v3.4.0 |

---

## Performance Baseline

| Suite | Wall Time |
|-------|-----------|
| Physics integration | 5.5s |
| Cognitive integration | 15.4s |
| Autonomy integration | 16.2s |
| Multimodal integration | 11.3s |
| Nitpick Specialist integration | 6.8s |
| `libnikola_core.a` | 21 MB |

---

## Nitpick SIE

- 15 .npk files: 7 libraries (`nitpickc -c`) + 8 tests (`nitpickc -I -L -l`)
- **15/15 compile successfully**
- Gap for v0.1.0: Nitpick lacks direct PhysicsOracle and ModuleSwapper integration

---

## Files Changed

13 files: 818 insertions, 48 deletions.

| File | Change |
|------|--------|
| CMakeLists.txt | Phase142 timeout |
| evolutionary_orchestrator.cpp | Gate 1 fix |
| specialist_interface.hpp | JSON escape fix |
| propagator.hpp | SCRAM threshold |
| phase113 test | k_safe_source (14 calls) |
| phase114 test | k_safe_source (3 calls) |
| ARCHITECTURE.md | §8 + §9 |
| BUILD_GUIDE.md | Test count + Nitpick SIE |
| GOTCHAS.md | 3 new entries |
| MODULE_REFERENCE.md | 2 new namespaces |
| docs/api/README.md | New API reference |
| docs/CHANGELOG_v0.0.x.md | New changelog |
| docs/AUDIT_v0.0.20.md | New audit report |

---

## Known Security Gaps (Documented for v0.1.0)

1. Gate 2 null physics_provider — architectural (non-physics modules)
2. Module path symlink/traversal — low risk (CRC check before dlopen)
3. SpecialistInterface PATH race — use absolute path via env var

---

## What's Next: v0.1.0

v0.1.0 is the **first functional self-improvement cycle**. The prototype
series (v0.0.x) established the full architecture:

- Physics engine + cognitive torus + metabolic controller
- 4-gate SIE pipeline + evolutionary orchestrator
- Post-quantum crypto (SPHINCS+, Kyber-768, CurveZMQ)
- LMDB persistence + observability + telemetry
- Multimodal input (audio + visual)
- Nitpick specialist integration (generate → compile → persist)

v0.1.0 will close the loop: specialist generates code → SIE validates →
module hot-swapped → performance measured → training data fed back →
specialist improves. See `META/NIKOLA/ROADMAP/RELEASE_0.1.0.md`.
