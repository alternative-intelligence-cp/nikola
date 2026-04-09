# Nikola Changelog — v0.0.x Prototype Series

_Covering v0.0.4 through v0.0.20_

---

## v0.0.20 — Pre-Release Audit & Documentation (2026-04-09)

Final audit of the 0.0.x prototype series. Feature freeze; bug fixes,
security hardening, documentation expansion, and release preparation for
v0.1.0.

### Security Fixes
- **Gate 1 bypass closed**: Empty source code no longer skips the blacklist
  scan — now returns `SECURITY_REJECTED`. (`evolutionary_orchestrator.cpp`)
- **JSON escape injection fixed**: `specialist_interface.hpp` now handles
  `\b`, `\f`, `\/`, and `\uXXXX` per RFC 7159. Invalid escapes are dropped
  instead of passed through.
- **SCRAM threshold aligned**: `propagator.hpp` default tolerance changed
  from 1e-4 to 1e-5, matching `PhysicsOracle::ORACLE_DRIFT_RATE_SCRAM`.

### Documentation
- Updated all 4 handoff docs (ARCHITECTURE, BUILD_GUIDE, GOTCHAS, MODULE_REFERENCE)
- Created `docs/api/README.md` — comprehensive API reference (158 headers, 24 namespaces)
- Added Multimodal and Aria namespace documentation
- Documented 3 new gotchas (Phase142 timeout, ariac `-c` flag, LMDB directory mode)

### Test Fixes
- Added `TIMEOUT 300` for Phase142 physics calibration tests in CMakeLists.txt
- Updated Phase113 + Phase114 tests to supply source code after Gate 1 fix

### Stats
- 163/163 tests pass
- 10 files modified, 262 insertions, 48 deletions
- 0 open bugs

---

## v0.0.19 — Aria Specialist Integration (2026-04-03)

Phase 145. Connects the Aria specialist model to the SIE pipeline:
`SpecialistInterface` (JSON-Lines subprocess), `AriaCompileValidator`
(ariac subprocess wrapper), `CodeProposalStore` (LMDB persistence).
Added `GENERATE_CODE = 10` to `ActionType`. 74 new tests.

## v0.0.18 — Multimodal Input (2026-04-02)

Audio and visual injection into the cognitive torus. `AudioInput` module
with cymatic transduction, visual frame pipeline with log-polar transform,
`MultimodalEngine` orchestrator. GGUF export and checkpoint management.

## v0.0.17 — Integration & End-to-End Test Framework (2026-03-28)

5 integration test suites (physics, cognitive, autonomy, multimodal, aria).
Established end-to-end testing patterns across the full stack.

## v0.0.16.1 — Mamba S6 SSM Wiring (2026-03-25)

Wired Mamba S6 State Space Model into the live `DecisionLoop` pipeline
for sequence-based action prediction.

## v0.0.16 — Curiosity Engine & Exploration Dynamics (2026-03-25)

Phase 143. Curiosity-driven exploration with novelty scoring, information
gain estimation, and curiosity decay. Drives autonomous investigation of
novel stimuli.

## v0.0.15 — Physics Oracle Calibration Suite (2026-03-22)

GAP-030. Full physics calibration pipeline: quick calibration (5 tests),
long-term stability (drift detection), and oracle validation. Phase 142.

## v0.0.14 — Social/Economic Layer & Peer Foundations (2026-03-18)

Social layer with peer handshake over CurveZMQ Ironhouse. Economic
primitives: ATP-based budgeting, resource accounting. 14 E2E security
tests.

## v0.0.13 — Security Hardening (2026-03-15)

12 end-to-end integration tests for the security posture. SPHINCS+
signature verification, Kyber-768 key encapsulation, CurveZMQ transport
encryption, SCRAM emergency shutdown protocol.

## v0.0.12 — DebugAdapter & Developer Tooling (2026-03-12)

DAP-compatible debug adapter for runtime inspection. Step/continue/
breakpoint support, variable inspection, watch expressions.

## v0.0.11 — Training Pipeline & Corpus Expansion (2026-03-08)

Training data pipeline for self-improvement. Corpus management, data
augmentation, training loop integration with the specialist model.

## v0.0.10 — Aria SIE Integration Layer (2026-03-05)

4-gate Self-Improvement Engine pipeline: ShadowSpine (Gate 0, binary
load + signature), CodePatternBlacklist (Gate 1, source scan),
PhysicsOracle (Gate 2, drift validation), ModuleSwapper (Gate 3, hot
swap). FFI bridge between Aria and C++ runtime.

## v0.0.9 — Cognitive Tuning & First Conversations (2026-03-01)

Cognitive parameter tuning: dopamine/serotonin balance, attention decay
curves, boredom thresholds. First conversational interactions via the
decision loop.

## v0.0.8 — Observability & Profiling (2026-02-26)

ScopeProfiler wiring into all hot paths. TelemetryDaemon for metrics
export. OpenTelemetry-compatible tracing spans.

## v0.0.7 — CUDA Propagator Validation (2026-02-22)

GPU-accelerated physics propagation (RTX 3090, CUDA). Runtime `--gpu`
toggle. Benchmark suite for CPU vs GPU comparison.

## v0.0.6 — LMDB Memory Persistence (2026-02-18)

Phase 137. LMDB-backed persistent memory for cognitive state, episodic
memory, and module metadata. Survives restarts.

## v0.0.5 — Stabilization & Aria SIE Bootstrap (2026-02-14)

Zero-warning build. 137/137 tests pass. Initial Aria SIE integration
layer: 3 modules + README.

## v0.0.4 — Foundation (2026-02-10)

First tagged release. Core architecture: physics engine, cognitive torus,
metabolic controller, decision loop. ATPM memory model. ~100 tests.
