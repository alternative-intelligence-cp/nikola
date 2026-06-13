# Nikola — Roadmap

_Last updated: 2026-02-27_

This document describes where the project is, where it was going, and what
the next steward should continue. It is opinionated. The choices made here
were deliberate — read the rationale before changing direction.

---

## Where We Came From

### Phase 1–9: Foundation (pre-2026)
Core physics, memory layout, 9D geometry, ZMQ infrastructure, dopamine/entropy
autonomy system, security subsystem skeleton, BERT-tiny integration.
The torus was built, the field propagated, the injector worked.

### Phase 10–30: Cognitive Pipeline
Holographic memory resonance, NPT architecture, cognitive generation, query
engine. Nikola could inject text, tick the field, and decode a response.
Not well, but functionally.

### Phase 31–60: Infrastructure and Self-Modification
ZMQ IPC hardening, seqlock memory, failure mode guards, Hebbian update rules,
evolutionary orchestrator scaffolding. The SIE (Self-Improvement Engine)
pipeline was designed: 5 gates, Physics Oracle, HybridVerifier, ShadowSpine,
ModuleSwapper, Rollback.

### Phase 61–100: Filling in the Physics
Latency budgets, SoA alignment, manifold seeder, coordinate semantics,
physics oracle, Hamiltonian density, GPU kernel architecture (pre-working).

### Phase 100–115: Closing Critical Gaps
GPU propagator (Phase 111), hybrid signature verification (Phase 114),
true Mamba-S6 selective scan (Phase 115), AVX-512 SIMD (Phase 116),
streaming output (Phase 117), curiosity engine (Phase 118),
OpenTelemetry tracing (Phase 119).

At Phase 115, the core pipeline was complete end-to-end:
text → BERT → field → torus tick → Hilbert scan → Mamba-9D S6 → NPT → response.

### Phase 120–125: Stub Completion (Cognitive)
Scratchpad hypothesis buffer (Phase 125), AttentionPrimer topic bias tracker
(Phase 126), SpectralFilter 9-band decomposition (Phase 127).

### Phase 128–133: Stub Completion (Security / Economy / Social)
PolymorphicDefense ASLR-style token mutation (Phase 128), HomeostasisMonitor
energy/entropy watchdog (Phase 129), NeuralMarketplace service registry
(Phase 130), SimulatedWallet (Phase 131), SocialMembrane permeability model
(Phase 132), PeerRegistry (Phase 133).

**Phase 133 completes all module stubs. No remaining skeleton code.**

---

## Current State (Phase 133)

| Component | Status | Notes |
|-----------|--------|-------|
| T⁹ physics torus | ✅ Complete | 19,683 nodes, UFIE, Störmer–Verlet |
| HolographicInjector | ✅ Complete | BERT-tiny ONNX, 9 emitters, π·φⁿ |
| CognitiveCore (SSM + NPT) | ✅ Complete | Mamba-S6, 8-head NPT |
| Autonomy loop | ✅ Complete | TD-learning, ATP, boredom |
| SIE (5-gate self-improvement) | ✅ Complete | Phase 114 |
| Security (PQ crypto) | ✅ Complete | MLKEM, SPHINCS+, CurveZMQ |
| Curiosity engine | ✅ Complete | Phase 118 |
| OpenTelemetry tracing | ✅ Complete | Phase 119 |
| AVX-512 SIMD | ✅ Complete | Phase 116 |
| CPU propagator | ✅ Complete | Phase 111 |
| GPU propagator (CUDA) | ⚠️ Compiled | C++20 pending fix, sm_86 |
| All cognitive stubs | ✅ Complete | Phase 125–127 |
| All security stubs | ✅ Complete | Phase 128–129 |
| All economy stubs | ✅ Complete | Phase 130–131 |
| All social stubs | ✅ Complete | Phase 132–133 |
| `nikola-run` CLI | ✅ Working | `--prompt`, `--interactive`, `--stream`, `--json`, `--memory` |
| Test suite | ✅ 135 tests | ~98% pass, 2 pre-existing timing flakes |
| Real blockchain wallet | 🔲 Planned | Simulated only for now |
| Nitpick language port | 🔲 Planned | Long-term |

---

## Near-Term: Ecosystem Integrations

These are the next concrete work items. They were recorded in `TASKS.md`
as the session ended.

### 1. `ScopeProfiler` — performance instrumentation wrapper

Status: **Not implemented.** Referenced in code samples and TASKS.md.

A lightweight RAII scope profiler that wraps OTel spans for the tick loop.
Needed before profiling analysis is possible.

```cpp
// Desired API:
ScopeProfiler p("physics_tick");
// ... ticks ...
// destructor closes span, records duration
```

Attach to: `AutonomyEngine::step()`, `WavePropagator::tick()`,
`CognitiveCore::process()`.

### 2. `TelemetryDaemon` — persistent metrics collection

Status: **Not implemented.**

A background thread that aggregates per-tick metrics and periodically flushes
to a metrics backend (Prometheus, or local files). Required for long-running
inference sessions to capture behavioral drift.

### 3. `DebugAdapter` — DAP-compatible debugger bridge

Status: **Not implemented.**

A Debug Adapter Protocol server that exposes NikolaState and torus field
contents to VS Code debugger at runtime. Would allow visualization of Ψ
and live NikolaState inspection during a running session.

### 4. CUDA propagator C++20 fix

The GPU propagator (`propagator.cu`) was patched in Phase 111 but there is
a remaining issue where nvcc chokes on C++20 features used in shared headers.
The fix is to move the offending constructs to `.cpp` wrappers that are
compiled with the host compiler, keeping `propagator.cu` on C++17.

---

## Mid-Term

### Real LMDB memory persistence

The `--memory` flag accepts a path but the actual memory architecture was
implemented in stubs. The autobiographical memory (`autobiography.hpp`) and
persistence layer need wiring to a real LMDB schema.

Priority: high. Without persistent memory, each session starts from zero.

### Real blockchain wallet

`SimulatedWallet` is deterministic but not on-chain. The `NeuralWallet`
interface was designed to allow swapping in a real Ethereum/Layer-2 wallet
by implementing the virtual interface. The address derivation already happens;
the signing needs to use real ECDSA.

### ChainNode: Nitpick community integration

The economy module was designed to connect to the Nitpick community network
(`REPOS/nitpick-community/`). The `NeuralMarketplace` lists services; the next
step is advertising those services to a real peer discovery mechanism.

### Peer discovery over real network

`PeerRegistry` stores peers by ID and key. Real peer discovery needs a
rendezvous mechanism — DHT or a known bootstrap node list. The CurveZMQ
(`ironhouse.hpp`) channel is ready to use; it needs a peer handshake protocol.

---

## Long-Term

### Nitpick language port

The long-term vision is to port Nikola's cognitive pipeline to the Nitpick
language (`REPOS/nitpick/`), which is under active development as a C-family
language with first-class physics-simulation primitives. The Nitpick runtime
would give Nikola stronger isolation and a more natural substrate.

This is a multi-year effort. The C++ codebase must remain the reference
implementation while Nitpick matures.

### Multi-node distributed torus

The T⁹ is currently single-process. The ZMQ IPC infrastructure (`connect/`)
was designed with multi-node in mind. A distributed Nikola would shard the
torus across physical nodes with boundary synchronization via ZMQ PUSH/PULL.

### Embodiment integration

The multimodal namespace (`include/nikola/multimodal/`) contains stubs for
audio/visual input integration. Connection to a sensory stream would allow
Nikola to operate in a grounded environment rather than purely text-based.

---

## What to Prioritize First

If you've just received this handoff and don't know where to start:

1. **Get a full build and test run passing** on your machine (see BUILD_GUIDE.md)
2. **Run `./nikola-run --interactive`** and have a conversation to verify the
   end-to-end pipeline is alive
3. **Implement `ScopeProfiler`** — it's small and unlocks performance analysis
4. **Wire LMDB memory persistence** — this is what makes Nikola actually learn
   across sessions rather than being stateless
5. **Fix the CUDA C++20 issue** — the GPU path is important for throughput
   once memory is working and sessions run longer

---

## What Not to Change

Some things should be treated as settled design decisions:

- **Emitter spacing** — π·φⁿ, do not change (see GOTCHAS.md)
- **Grid size** — 3⁹, changing this requires extensive rework
- **Integrator** — Störmer–Verlet is symplectic, do not replace with RK4
- **BERT-tiny architecture** — the ONNX model is fixed; the embedding dimension
  (128) propagates through the injector to the emitter coefficients
- **NikolaState fields** — adding/removing fields breaks multiple consumers
- **AGPL license** — specified in HANDOFF.md conditions of transfer

---

## Contact

If you have access to a transcript of prior development sessions, those
contain rationale for many architectural decisions that predates this document.
The META/ workspace (outside the git repo) contains extensive design notes
in `META/NITPICK/` and `META/INFO/ARIA/`.

The project has no public forum or community list yet. That is something
the next steward should establish.
