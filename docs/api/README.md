# Nikola — Public API Reference

_Last updated: 2026-04-09 (v0.0.20)_

This directory documents the public C++ API surface exposed by `libnikola_core`.
All headers live under `include/nikola/` and are organized by namespace.

**Total: 158 public headers across 24 namespaces.**

For architecture context, see `docs/handoff/ARCHITECTURE.md`.
For per-module status and ctest mapping, see `docs/handoff/MODULE_REFERENCE.md`.

---

## Namespace Index

| Namespace | Headers | Primary Responsibility |
|-----------|---------|----------------------|
| `nikola::physics` | 13 | UFIE wave equation, Störmer–Verlet propagator, Hamiltonian |
| `nikola::cognitive` | 20 | HolographicInjector, CognitiveCore (Mamba-S6 + NPT), memory |
| `nikola::autonomy` | 18 | DecisionLoop, dopamine/TD-learning, SIE orchestration |
| `nikola::security` | 12 | Post-quantum crypto, Ironhouse, HomeostasisMonitor |
| `nikola::multimodal` | 8 | Audio/visual input, checkpoint, GGUF export |
| `nikola::aria` | 8 | Aria specialist integration, compile validation |
| `nikola::foundation` | 6 | Complex field, Nit embedding, Vector9D, toroidal grid |
| `nikola::infrastructure` | 17 | CUDA kernels, ZMQ spine, orchestration, error taxonomy |
| `nikola::interior` | 6 | Affective state, curiosity, dreams, autobiography |
| `nikola::economy` | 3 | NeuralMarketplace, SimulatedWallet |
| `nikola::social` | 3 | SocialMembrane, PeerRegistry, PeerHandshake |
| `nikola::spatial` | 3 | HilbertScanner, topology manager |
| `nikola::math` | 4 | Coordinate semantics, Hebbian metric, Voronoi quantizer |
| `nikola::memory` | 2 | Resonance index |
| `nikola::persistence` | 5 | LMDB state store, LSM neurogenesis, GGUF model streamer |
| `nikola::core` | 1 | NikolaConfig, NikolaState |
| `nikola::cli` | 1 | StreamEmitter |
| `nikola::diag` | 4 | DebugAdapter, ScopeProfiler, TelemetryDaemon |
| `nikola::interface` | 1 | FeedbackLoop |
| `nikola::ml` | 1 | TinyTransformer |
| `nikola::system` | 8 | CRC32C, error taxonomy, latency budget, performance policy |
| `nikola::telemetry` | 1 | NikolaTracer (OpenTelemetry) |
| `nikola::validation` | 1 | ConversionValidator |

---

## Core Types

### `NikolaState` (`core/config.hpp`)

The primary runtime state carrier, consumed by virtually every subsystem.

```cpp
struct NikolaState {
    float dopamine;      // [0,1] — current reward signal
    float td_error;      // temporal-difference error, signed
    float atp;           // [0,1] — metabolic energy
    float boredom;       // [0,1] — drives exploration when high
    float entropy;       // field entropy estimate
    float torus_energy;  // Hamiltonian H of the torus
    int   last_action;   // discrete action index (ActionType enum)
    float time;          // simulation time in ticks
};
```

### `ActionType` (`autonomy/decision_loop.hpp`)

```cpp
enum class ActionType : int {
    EMIT_THOUGHT = 0,    // Generate output text
    INJECT_STIMULUS = 1, // Inject new wave packet
    REST = 2,            // Replenish ATP
    EXPLORE = 3,         // Entropy-maximizing injection
    CONSOLIDATE = 4,     // Memory consolidation
    SELF_EVALUATE = 5,   // Introspective wave mirror
    DREAM = 6,           // Low-stimulation replay
    SOCIAL_PROBE = 7,    // Peer interaction
    TRADE = 8,           // Marketplace transaction
    NAP = 9,             // Extended consolidation
    GENERATE_CODE = 10   // Aria specialist code proposal
};
```

---

## Physics API

### `WavePropagator` (`physics/propagator.hpp`)
- `tick(Ψ, dt)` — One full Strang-split Störmer–Verlet step
- `field_energy()` — Total Hamiltonian H
- `reset()` — Zero the field

### `HamiltonianDensity` (`physics/hamiltonian.hpp`)
- Computes kinetic + potential + nonlinear energy density over T⁹ grid
- GPU variant: `physics/gpu_hamiltonian.hpp`

### `PhysicsOracle` (`physics/physics_oracle.hpp`)
- `check_standard_candle(H₀, H_f, SimdLevel)` — Energy conservation
- `check_viscosity_trap(E_actual, E₀, α, t)` — Damping validation
- `check_resonance_attack(|Ψ|_max)` — Amplitude bound
- `check_reversibility(initial[], recovered[])` — Time-reversal

---

## Cognitive API

### `HolographicInjector` (`cognitive/holographic_injector.hpp`)
- Text → BERT-tiny → 128-dim → 9 emitter coefficients → Ψ
- Falls back to random projection when ORT unavailable

### `CognitiveCore` (`cognitive/cognitive_core.hpp`)
- Mamba-S6 selective scan (256 hidden, 16r×16s state)
- NPT 8-head wave-correlation attention
- Input: HilbertScanner traversal of 19,683 nodes

### `QuantumScratchpad` (`cognitive/scratchpad.hpp`)
- `commit(fact)` → `inject(hypothesis)` → `measure_resonance()` → `collapse_if_resonant(0.40)`
- Caps: 128 hypotheses, 512 committed

### `AttentionPrimer` (`cognitive/attention_primer.hpp`)
- `prime(topic, strength, tick)` / `decay(dt)` / conflict detection at 1.20

### `SpectralFilter` (`cognitive/spectral_filter.hpp`)
- 9-band decomposition: DELTA through ULTRA
- `extract_band()`, `reconstruct()`, `bandpass()`, `dominant_band()`

---

## Autonomy API

### `DecisionLoop` (`autonomy/decision_loop.hpp`)
- `tick()` — Full simulation cycle (physics → inject → cognitive → action → emit)
- `on_action(callback)` — Hook for `--stream` output
- 11 action types, multi-armed bandit + TD-learning selection
- GENERATE_CODE fires when boredom > 0.4, ATP ≥ 0.30, cooldown 30s

### `DopamineSystem` (`autonomy/dopamine_system.hpp`)
- TD(0) update: `δ = reward + γ·V(s') − V(s)`

### `MetabolicController` (`autonomy/metabolic_controller.hpp`)
- ATP lifecycle: consumed by actions, replenished by rest
- `MetabolicLock` (RAII) blocks action below ATP threshold

### `EvolutionaryOrchestrator` (`autonomy/evolutionary_orchestrator.hpp`)
- SIE 5-gate loop: PhysicsOracle → HybridVerifier → ShadowSpine → ModuleSwapper → Rollback
- ATP cost: 750 total

### `ModuleSwapper` (`autonomy/module_swapper.hpp`)
- dlopen/dlsym hot-swap of POSIX shared libraries
- Thread-safe RAII load/unload/rollback

---

## Security API

### `HybridVerifier` (`security/hybrid_verifier.hpp`)
- Ed25519 fast-path + SPHINCS+ slow-path signature gate (SIE Gate 0)

### `HomeostasisMonitor` (`security/homeostasis.hpp`)
- Energy = `0.3*dopamine + 0.4*atp + 0.3*(1−boredom)`
- Detects ENERGY_SPIKE/DROP, ENTROPY_SPIKE/DROP
- Auto-lockdown when severity ≥ threshold

### `PolymorphicDefense` (`security/polymorphic_defense.hpp`)
- ASLR-style behavioral token mutation
- `randomize(rate, tick)`, continuous mode via background thread

### Post-Quantum Cryptography
- **SPHINCS+-SHAKE-256f** (`security/sphincs_signer.hpp`) — Digital signatures
- **ML-KEM / Kyber-768** (`security/mlkem_kem.hpp`) — Key encapsulation
- **CurveZMQ Ironhouse** (`security/ironhouse.hpp`) — Authenticated channels

---

## Multimodal API (v0.0.18)

### `AudioInput` (`multimodal/audio_input.hpp`)
- `process_samples(pcm, n)` → Goertzel 8-band → Nit[128]
- DC offset removal, phase coding

### `MultimodalEngine` (`multimodal/multimodal_engine.hpp`)
- `tick_audio_nits()`, `inject_visual()`, checkpoint coordination

---

## Aria Integration API (v0.0.19)

### `AriaCompileValidator` (`aria/compile_validator.hpp`)
- `validate(source_code)` → `CompileResult{success, errors, warnings, elapsed_ms}`
- Uses `ariac` subprocess; path from `$ARIAC_BIN` or default

### `SpecialistInterface` (`aria/specialist_interface.hpp`)
- `start()` / `ask(instruction, context, timeout)` / `stop()`
- JSON-Lines protocol to aria-specialist `server.py`

### `CodeProposalStore` (`aria/code_proposal_store.hpp`)
- LMDB 128MB, `store()` / `load()` / `export_successful()` / `success_rate()`

---

## Social / Economic API

### `SocialMembrane` (`social/membrane.hpp`)
- `permeability = clamp(trust / (dissonance + ε), 0, 1)`
- `filter_incoming(friend, self)` → blended state

### `PeerRegistry` (`social/peer_registry.hpp`)
- FIFO eviction at 128, per-peer SocialMembrane
- `record_interaction(id, resonance, tick)`, `peers_by_trust()`

### `NeuralMarketplace` (`economy/marketplace.hpp`)
- `list_service()`, `browse_services(query)`, `purchase_service()`
- EMA rating blend α=0.30, FNV-1a transaction hash

### `SimulatedWallet` (`economy/wallet.hpp`)
- FNV-1a address derivation, `credit()` / `debit()`
- Not a real blockchain wallet — drop-in replacement surface

---

## Persistence API

### `LmdbStateStore` (`persistence/lmdb_state_store.hpp`)
- 512MB, 5 named databases: state, checkpoint, events, skills, values
- See `docs/architecture/memory_schema.md` for wire formats

### `LmdbMemoryStore` (`cognitive/lmdb_memory_store.hpp`)
- 256MB, semantic wave-field memories keyed by Hilbert index

### `CodeProposalStore` (`aria/code_proposal_store.hpp`)
- 128MB, code proposals with compile results

---

## Telemetry API

### `NikolaTracer` (`telemetry/nikola_tracer.hpp`)
- `setup_ostream_tracer()`, `setup_in_memory_tracer()`, `teardown_tracer()`
- `TickTracer::trace_tick()` — one span per tick with 8 NikolaState attributes

### Metrics (FD 3 JSON-Lines)
- See `docs/architecture/metrics_schema.md` for wire format
- Gauges: tick.energy, tick.dopamine, tick.atp, tick.boredom, tick.entropy, tick.duration
