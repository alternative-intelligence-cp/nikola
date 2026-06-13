# Nikola — Module Reference

_Last updated: 2026-04-09_

One entry per significant module in `include/nikola/`. For complete API docs
see inline header comments. For ctest mapping see column "ctest #".

---

## How to Read This

- **Status**: IMPL = fully implemented; STUB = skeleton + tests, body is a stub
- **ctest #**: the registered test index in the build (use `ctest -R <Name> -v`)
- **Phase**: the implementation phase number (see `META/NITPICK/NIKOLA_PHASE_REGISTRY.md`)
- **NS**: C++ namespace

---

## Physics Namespace — `nikola::physics`

### `propagator.hpp` / `propagator.cu`
| Field | Value |
|-------|-------|
| Status | IMPL |
| Phase | 111 |
| ctest # | 113 (Phase111CudaPropagator) |
| NS | `nikola::physics` |
| Key types | `WavePropagator`, `CudaPropagator` |

The UFIE integrator. `WavePropagator` is the CPU Störmer–Verlet implementation;
`CudaPropagator` is the GPU path compiled into `nikola_cuda` STATIC. The GPU
path requires C++17 (nvcc limitation); the host code is C++23.

Key methods: `tick(Ψ, dt)`, `step()`, `field_energy()`, `reset()`.

### `hamiltonian.hpp` / `gpu_hamiltonian.hpp`
| Field | Value |
|-------|-------|
| Status | IMPL |
| Phase | 110 |
| ctest # | 112 (Phase110TorusCudaKernels) |
| NS | `nikola::physics` |
| Key types | `HamiltonianDensity`, GPU kernel wrappers |

Computes kinetic + potential energy density over the T⁹ grid. The GPU kernel
uses `torus_block_simd.hpp` AVX-512 paths on CPU fallback.

### `wave_function.hpp`
| Field | Value |
|-------|-------|
| Status | IMPL |
| NS | `nikola::physics` |
| Key types | `Ψ` (complex field), `complex_field` |

The field array over 3⁹ = 19,683 nodes. Layout is SoA; see `soa_layout.hpp`.

### `torus_block_simd.hpp`
| Field | Value |
|-------|-------|
| Status | IMPL |
| Phase | 116 |
| Key types | `TorusBlock` with AVX-512 paths |

AVX-512 SIMD for `psi_zero`, `psi_scale`, `psi_add_scaled`, `psi_norm_sq`,
`psi_renormalize`, `metric_scale`. Falls back to scalar automatically when
AVX-512 not available.

### `coordinate_semantics.hpp`
| Field | Value |
|-------|-------|
| Status | IMPL |
| Key types | `DimensionRole`, `CoordSemantics` |

Maps the 9 torus dimensions to conceptual roles:
`(time, space_x, space_y, space_z, emotion, memory, intent, social, meta)`.
Documents the semantic contract for emitter placement.

---

## Cognitive Namespace — `nikola::cognitive`

### `holographic_injector.hpp`
| Field | Value |
|-------|-------|
| Status | IMPL |
| Phase | 10 |
| ctest # | 13 (Phase10Holographic) |
| NS | `nikola::cognitive` |
| Key types | `HolographicInjector`, `EmitterArray` |

Pipeline: text → BERT-tiny → 128-dim embedding → 9 emitter coefficients → Ψ.
When ORT is unavailable, falls back to random projections. Uses `EmitterArray`
with π·φⁿ Hz frequencies for the 9 emitters (irrational spacing required —
**do not change**).

### `cognitive_core.hpp`
| Field | Value |
|-------|-------|
| Status | IMPL |
| Phase | 115 |
| Key types | `CognitiveCore`, `SSMLayer` (Mamba-S6) |

Central cognitive pipeline. Reads from `HilbertScanner` (spatial order),
runs through Mamba-9D S6 SSM (256 hidden dims, input-dependent Δ/B/C with ZOH
discretisation), then NPT (NeuroplasticTransformer) 8-head wave-correlation
attention. Phase 115 upgraded to true selective-scan kernel.

### `neuroplastic_transformer.hpp`
| Field | Value |
|-------|-------|
| Status | IMPL |
| Phase | 11 |
| ctest # | 14 (Phase11CognitiveGen) |
| Key types | `NPT`, `WaveAttentionHead` |

8-head attention where keys/queries are computed as wave correlation
coefficients over the torus field rather than raw dot-products.

### `scratchpad.hpp`
| Field | Value |
|-------|-------|
| Status | IMPL |
| Phase | 125 |
| ctest # | 127 (Phase125Scratchpad) |
| Key types | `QuantumScratchpad`, `HypothesisEntry`, `CommittedEntry` |

Hypothesis-testing working-memory buffer. Workflow: `commit(fact)` →
`inject(hypothesis)` → `measure_resonance()` (Jaccard overlap) →
`collapse_if_resonant(threshold=0.40)`. Caps: 128 hypotheses, 512 committed.
`QuantumScratchpad` is a type alias for backwards compatibility.

### `attention_primer.hpp`
| Field | Value |
|-------|-------|
| Status | IMPL |
| Phase | 126 |
| ctest # | 128 (Phase126AttentionPrimer) |
| Key types | `AttentionPrimer`, `TopicBias` |

Topic-priming attention-bias tracker. `prime(topic, strength, tick)` raises
weight; `decay(dt)` reduces all over time. Conflict detection: combined weight
≥ 1.20 signals competing topics. Focus mode: clears all other topics.

### `spectral_filter.hpp`
| Field | Value |
|-------|-------|
| Status | IMPL |
| Phase | 127 |
| ctest # | 129 (Phase127SpectralFilter) |
| Key types | `SpectralFilter`, `SpectralBand`, `SpectralStats` |

9-band frequency decomposition: DELTA/THETA/ALPHA/BETA_LOW/BETA_HIGH/
GAMMA_LOW/GAMMA_HIGH/RIPPLE/ULTRA. Hann-windowed convolution. `extract_band()`,
`reconstruct()`, `bandpass()`, `dominant_band()`, `compute_stats()`.

### `bpe_tokenizer.hpp`
| Field | Value |
|-------|-------|
| Status | IMPL |
| Phase | ~100 |
| Key types | `BPETokenizer` |

HuggingFace WordPiece tokenizer, vocab_size=30522. Auto-resolves directory
path to `dir/tokenizer.json`. Used by HolographicInjector.

### `relevance_gate.hpp`, `resonance_decoder.hpp`, `query_engine.hpp`
Working memory access gates and response decoding layers. All IMPL,
wired into the CognitiveCore output pipeline.

### `inner_monologue.hpp`
Internal narrative stream — feeds EMIT_THOUGHT actions for `--stream` mode.

### `thought_composer.hpp`, `cognitive_generator.hpp`
Higher-level response assembly from decoded torus field activations.

---

## Autonomy Namespace — `nikola::autonomy`

### `autonomy_engine.hpp`
| Field | Value |
|-------|-------|
| Status | IMPL |
| Key types | `AutonomyEngine` |

Main loop: physics tick → inject stimulus → cognitive pipeline → action
selection → emit. Owns `NikolaState` and drives all subsystems.

### `decision_loop.hpp`
| Field | Value |
|-------|-------|
| Status | IMPL |
| Key types | `DecisionLoop`, `Action`, `ActionType` |

Multi-armed bandit + TD-learning action selector. `on_action(callback)` fires
per selected action (used by `--stream` output in `nikola_run.cpp`).

### `dopamine_system.hpp`
| Field | Value |
|-------|-------|
| Status | IMPL |
| Key types | `DopamineSystem` |

TD-learning reward signal. Updates `NikolaState.dopamine` and `td_error`.
Boredom accumulates when entropy is too low, driving exploration.

### `metabolic_controller.hpp`
| Field | Value |
|-------|-------|
| Status | IMPL |
| Key types | `MetabolicController` |

ATP lifecycle. `NikolaState.atp` consumed by actions, replenished by rest.
`MetabolicLock` (RAII) blocks action while ATP is below threshold.

### `evolutionary_orchestrator.hpp`
| Field | Value |
|-------|-------|
| Status | IMPL |
| Phase | 113 |
| ctest # | 115 (Phase113EvolutionaryOrchestrator) |
| Key types | `EvolutionaryOrchestrator` |

SIE loop controller — drives the 5-gate Self-Improvement Engine:
PhysicsOracle → HybridVerifier → ShadowSpine → ModuleSwapper → Rollback.

### `module_swapper.hpp`
| Field | Value |
|-------|-------|
| Status | IMPL |
| Phase | 112 |
| ctest # | 114 (Phase112ModuleSwapper) |
| Key types | `ModuleSwapper` |

dlopen hot-swap engine. Thread-safe RAII loading/unloading/rollback of POSIX
shared libraries. Driven by `EvolutionaryOrchestrator`.

### `adversarial_dojo_ga.hpp`
Red-team generator for adversarial self-improvement stress-testing.

### `dream_weave.hpp`, `nap_controller.hpp`
Low-stimulation replay loop; consolidation during idle ticks.

---

## Security Namespace — `nikola::security`

### `polymorphic_defense.hpp`
| Field | Value |
|-------|-------|
| Status | IMPL |
| Phase | 128 |
| ctest # | 130 (Phase128PolymorphicDefense) |
| Key types | `PolymorphicDefense`, `ProtectedEntry` |

ASLR-style behavioral token mutation. Entries registered by name → `uint64_t`
token. `randomize(rate, tick)` mutates a fraction; continuous mode via
background thread. `validate_token(name, token)` for gate checking.

### `homeostasis.hpp`
| Field | Value |
|-------|-------|
| Status | IMPL |
| Phase | 129 |
| ctest # | 131 (Phase129HomeostasisMonitor) |
| Key types | `HomeostasisMonitor`, `AnomalyRecord` |

NikolaState energy/entropy watchdog. Computes energy = `0.3*dopamine +
0.4*atp + 0.3*(1−boredom)`. Detects ENERGY_SPIKE/DROP, ENTROPY_SPIKE/DROP.
Auto-lockdown when severity ≥ threshold. Background monitoring thread.

### `hybrid_verifier.hpp`
| Field | Value |
|-------|-------|
| Status | IMPL |
| Phase | 114 |
| ctest # | 116 (Phase114HybridVerifier) |
| Key types | `HybridVerifier` |

Ed25519 fast-path + SPHINCS+ slow-path signature gate. Gate 0 in SIE pipeline.

### `sphincs_signer.hpp`
SPHINCS+ signing wrapper (post-quantum, `third_party/sphincsplus`).

### `mlkem_kem.hpp`
ML-KEM / Kyber key encapsulation (post-quantum, `third_party/kyber`).

### `ironhouse.hpp`
CurveZMQ CURVE-based authenticated channels.

### `escape_detector.hpp`, `bootstrap_manager.hpp`, `code_blacklist.hpp`
VM escape detection, cold-start security bootstrap, blacklisted instruction
pattern scanner.

---

## Economy Namespace — `nikola::economy`

### `marketplace.hpp`
| Field | Value |
|-------|-------|
| Status | IMPL |
| Phase | 130 |
| ctest # | 132 (Phase130NeuralMarketplace) |
| Key types | `NeuralMarketplace`, `ServiceListing`, `Transaction` |

Keyword-searchable service registry + transaction ledger. `list_service()`,
`browse_services(query)`, `purchase_service()`, `execute_service()`,
`rate_service()` (EMA blend α=0.30). FNV-1a deterministic `tx_hash`.

### `wallet.hpp`
| Field | Value |
|-------|-------|
| Status | IMPL |
| Phase | 131 |
| ctest # | 133 (Phase131SimulatedWallet) |
| Key types | `NeuralWallet` (interface), `SimulatedWallet` (impl) |

Simulated on-chain wallet. FNV-1a address derivation from seed. `credit()`,
`debit()` with insufficient-funds guard. Deterministic `sign()`/`verify()`.
**Not a real blockchain wallet** — for future integration see ROADMAP.md.

### `metabolic_calibrator.hpp`
Economy-autonomy bridge: adjusts metabolic parameters based on market signals.

---

## Social Namespace — `nikola::social`

### `membrane.hpp`
| Field | Value |
|-------|-------|
| Status | IMPL |
| Phase | 132 |
| ctest # | 134 (Phase132SocialMembrane) |
| Key types | `SocialMembrane` |

Permeability model: `p = clamp(trust / (dissonance + ε), 0, 1)`.
`filter_incoming(friend, self)` → `self + p*(friend−self)`. Trust/dissonance
updated per interaction. Owned by `PeerInfo` in `PeerRegistry`.

### `peer_registry.hpp`
| Field | Value |
|-------|-------|
| Status | IMPL |
| Phase | 133 |
| ctest # | 135 (Phase133PeerRegistry) |
| Key types | `PeerRegistry`, `PeerInfo` |

Identity store for known peers. FIFO eviction at cap (128). Each `PeerInfo`
owns a `SocialMembrane`. `record_interaction(id, resonance, tick)` triggers
trust update and fires `on_interaction` callback. `peers_by_trust()` descending.

---

## Interior Namespace — `nikola::interior`

### `curiosity.hpp`
| Field | Value |
|-------|-------|
| Status | IMPL |
| Phase | 118 |
| Key types | `CuriosityEngine`, `KnowledgeGap`, `Question` |

Shannon entropy-ranked gap tracking. `register_gap()`, `generate_questions()`,
`identify_knowledge_gaps()`, `measure_information_gain()`, `pursue_interest()`.

### `affective_state.hpp`
Internal mood/valence model. Feeds into `NikolaState.dopamine` and `boredom`.

### `dream_engine.hpp`
Replay-based consolidation during idle/nap cycles.

### `wave_mirror.hpp`
Self-referential field reflection — Nikola observing its own Ψ.

### `autobiography.hpp`
Long-term episodic memory accumulation.

### `internal_dialogue.hpp`
Multi-perspective inner monologue generation.

---

## Spatial Namespace — `nikola::spatial` / `nikola::math`

### `hilbert_scanner.hpp`
| Field | Value |
|-------|-------|
| Status | IMPL |
| ctest # | 2 (HilbertScanner) |
| Key types | `HilbertScanner` |

Skilling algorithm, 9D. Traverses the 19,683-node torus in spatial-locality
order. Used by `CognitiveCore` as the scan order for SSM input.

### `topology_manager.hpp`
Manages T⁹ boundary wrapping (toroidal adjacency). Wraps coordinates mod-3
in all 9 dimensions.

---

## Memory Namespace — `nikola::memory`

### `metabolic_lock.hpp`
| Field | Value |
|-------|-------|
| Status | IMPL |
| ctest # | 1 (MetabolicLock) |
| Key types | `MetabolicLock` (RAII) |

RAII ATP acquisition/release. Blocks action if ATP below threshold.

---

## Core Namespace — `nikola::core`

### `config.hpp`
| Field | Value |
|-------|-------|
| Key types | `NikolaConfig`, `NikolaState` |

`NikolaState` fields: `dopamine`, `td_error`, `atp`, `boredom`, `entropy`,
`torus_energy`, `last_action`, `time`. This struct is the primary runtime
state carrier. **Do not add or remove fields** without updating all downstream
consumers (HomeostasisMonitor, HolographicInjector, AutonomyEngine, tests).

---

## ML / Telemetry

### `ml/` — ONNX Runtime integration, BERT-tiny loader, embedding pipeline
### `telemetry/` — OpenTelemetry wrapper (`nikola_tracer.hpp`, Phase 119)

`nikola_tracer.hpp` provides `setup_ostream_tracer()`,
`setup_in_memory_tracer()`, `teardown_tracer()`, `TickTracer::trace_tick()`.
`nikola_run.cpp` emits one `"nikola.tick"` span per tick with 8 NikolaState
attributes when `--trace` flag is set.

---

## Multimodal Namespace — `nikola::multimodal`

### `audio_input.hpp`
| Field | Value |
|-------|-------|
| Status | IMPL |
| Phase | 144 |
| Key types | `AudioInput`, `AudioBand` |

PCM → Goertzel 8-band frequency analysis → phase-coded Nit[128] embedding.
`process_samples(pcm, n)` → `get_embedding()`. Bands: sub-bass through
brilliance (20Hz–20kHz). DC offset removal built-in.

### `multimodal_engine.hpp`
| Field | Value |
|-------|-------|
| Status | IMPL |
| Phase | 144 |
| Key types | `MultimodalEngine` |

Top-level facade for audio/visual/text injection. `tick_audio_nits()`,
`inject_visual()`, checkpoint management. Owns `CognitiveTorus` reference.

### `audio_emitter.hpp`, `visual_frame_rate.hpp`
Audio emission and visual frame-rate control for output modalities.

### `log_polar_transform.hpp`, `cymatic_transduction.hpp`
Visual input pipeline: pixel → log-polar → cymatic frequency domain.

### `adaptive_quantizer.hpp`, `checkpoint_manager.hpp`, `gguf_exporter.hpp`
Quantization, checkpoint persistence, and GGUF model export utilities.

---

## Nitpick Namespace — `nikola::nitpick`

### `compile_validator.hpp`
| Field | Value |
|-------|-------|
| Status | IMPL |
| Phase | 145 |
| Key types | `NitpickCompileValidator`, `CompileResult` |

C++ subprocess wrapper for `nitpickc` compiler. `validate(source_code)` writes
to tempfile, invokes nitpickc, parses error/warning lines, returns
`CompileResult{success, errors, warnings, raw_output, elapsed_ms}`.
Also contains `extract_code_block()` for parsing model responses.

### `specialist_interface.hpp`
| Field | Value |
|-------|-------|
| Status | IMPL |
| Phase | 145 |
| Key types | `SpecialistInterface`, `SpecialistResult` |

C++ client for nitpick-specialist Python server. Fork/exec `python3 server.py`,
JSON-Lines protocol over stdin/stdout pipes. `start()`, `ask(instruction,
context, timeout)` → `SpecialistResult{ok, response, error}`, `stop()`.

### `code_proposal_store.hpp`
| Field | Value |
|-------|-------|
| Status | IMPL |
| Phase | 145 |
| Key types | `CodeProposalStore`, `CodeProposal` |

LMDB-backed persistence for code proposals. 128MB map size, `"proposals"`
named database. `store(proposal)`, `load(id)`, `count()`,
`count_successful()`, `export_successful()`, `success_rate()`. Magic
prefix `0x4E50524F` ("NPRO").

### `compiler.hpp`, `code_generator.hpp`, `interpreter.hpp`
Nitpick language bridge modules for code generation and interpretation.

### `metaprogramming.hpp`, `native_interface.hpp`
Nitpick metaprogramming support and native FFI interface.

---

## `NikolaState` Quick Reference

```cpp
struct NikolaState {
    float dopamine;      // [0,1] — current reward signal
    float td_error;      // temporal-difference error, signed
    float atp;           // [0,1] — metabolic energy
    float boredom;       // [0,1] — drives exploration when high
    float entropy;       // field entropy estimate
    float torus_energy;  // Hamiltonian H of the torus
    int   last_action;   // discrete action index
    float time;          // simulation time in ticks
};
```

---

## Module Count Summary

| Namespace | IMPL | STUB* | Headers |
|-----------|------|-------|---------|
| physics | ~12 | 0 | ~13 |
| cognitive | ~15 | 0 | ~20 |
| autonomy | ~12 | 0 | ~18 |
| security | ~10 | 0 | ~12 |
| economy | 3 | 0 | 3 |
| social | 3 | 0 | 3 |
| interior | ~6 | 0 | ~6 |
| spatial/math | ~5 | 0 | ~5 |
| memory/core | ~4 | 0 | ~4 |
| multimodal | 8 | 0 | 8 |
| nitpick | 8 | 0 | 8 |

*As of Phase 145 — all stubs promoted to IMPL. No remaining stubs.
**Total public headers: 158.**
