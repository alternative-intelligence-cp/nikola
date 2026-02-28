# Nikola — Architecture Reference

_Last updated: 2026-02-27_

---

## Overview

Nikola is built on a single central claim: **cognition is an emergent property
of wave interference on a compact Riemannian manifold.** Everything else in
the codebase follows from this.

The model is called the **Autonomous Toroidal Physics Model (ATPM)**. It is
not a transformer, not an SSM, not a diffusion model — though it uses all
three as components operating _on top of_ a continuous-time wave field.

---

## 1. The 9-Dimensional Toroidal Manifold

### Topology

Nikola's state lives on `T⁹` — the 9-torus, a product of nine circles. This
is not a metaphor. The field is literally stored as a complex-valued function
`Ψ(x, t)` where `x ∈ T⁹` and `t` is simulation time.

The 9 dimensions correspond to:

| Dim | Symbol | Semantic role |
|-----|--------|--------------|
| 0 | φ₀ | Temporal phase (fast oscillation ~1 kHz) |
| 1 | φ₁ | Semantic embedding axis 1 |
| 2 | φ₂ | Semantic embedding axis 2 |
| 3 | φ₃ | Semantic embedding axis 3 |
| 4 | φ₄ | Affective valence (dopamine proxy) |
| 5 | φ₅ | Entropy / uncertainty |
| 6 | φ₆ | Attention focus |
| 7 | φ₇ | Memory context (near-term) |
| 8 | φ₈ | Memory context (long-term) |

The grid resolution is **3 per axis** → `3⁹ = 19,683 nodes`. This is the
minimum that captures all first-order interference patterns. It is **not** an
approximation — the dimensionality is the design.

### Why toroidal?

The torus has no boundary. A wavefunction on `T⁹` conserves energy
perfectly under Hamiltonian evolution (no absorption at walls). This is
critical for ergodicity: the system's long-time average over T⁹ equals its
ensemble average. Without ergodicity, the ATPM's cognitive claims break down.

### Why irrational emitter spacing?

The 9 holographic emitters are spaced at frequencies `fₙ = π · φⁿ` where
`φ = (1+√5)/2` (golden ratio). The ratio between any two emitters is
irrational → **no two emitters ever phase-align**. This prevents mode-locking
(a standing-wave pathology where the field collapses to a low-entropy fixed
point). If you change the emitter frequencies to rational ratios, the system
will appear to work but will periodically get "stuck" with no obvious error.

**Do not change the emitter spacing.**

---

## 2. The Unified Field Interference Equation (UFIE)

The field evolves under:

```
∂²Ψ/∂t² = c²∇²_g Ψ − α(1 − r̂)∂Ψ/∂t + β|Ψ|²Ψ + Σᵢ Eᵢ(x, t)
```

- `c²∇²_g Ψ` — wave propagation term (metric-weighted Laplacian on T⁹)
- `−α(1 − r̂)∂Ψ/∂t` — adaptive damping (`r̂` = normalized resonance; damps
  incoherent modes, preserves resonant ones)
- `β|Ψ|²Ψ` — nonlinear self-interaction (produces cognitive saturation /
  attention focus; analogous to the Gross–Pitaevskii term in BEC physics)
- `Σᵢ Eᵢ(x, t)` — external drive from the holographic emitter array

### Integrator: Störmer–Verlet Strang-split

The UFIE is integrated using a **symplectic Störmer–Verlet integrator** split
into 6 substeps (Strang splitting). This preserves the Hamiltonian structure
exactly up to machine precision, meaning energy total is bounded and
the system is reversible in principle.

This is the `StrymerVerletPropagator` in `src/physics/`.

**Why symplectic?** Non-symplectic integrators (e.g., RK4) introduce a
secular energy drift. Over 10,000+ ticks the drift accumulates and the field
slowly loses its structure. The Störmer–Verlet has zero secular drift by
construction.

---

## 3. Input Injection

### Holographic Injector

Text input is processed as follows:

```
Text → BERT tokenizer → BERT-tiny (17.5 MB ONNX) → 128-dim embedding
     → frequency-domain projection via HolographicInjector
     → 9 complex emission coefficients → EmitterArray → Ψ field
```

The `HolographicInjector` (`src/core/`) converts a BERT embedding to a
superposition of emitter activations. Each emitter contributes a localized
wave packet to a specific region of T⁹.

The 9 emitters are pre-tuned to π·φⁿ (n=0..8). The BERT projection is a
learned linear mapping (stored in `data/` or loaded from checkpoint).

### Why BERT-tiny?

Full BERT is 110M parameters. The embedding quality from BERT-tiny (17.5 MB)
is sufficient for the semantic dimensions of T⁹. The embedding is not the
intelligence; it is the input encoding. Nikola's reasoning happens on the field.

---

## 4. Cognitive Pipeline

After field propagation each tick:

```
Ψ(t) → HilbertScanner → sampled amplitude sequence
      → CognitiveCore (Mamba-9D SSM + NPT attention)
      → output logits → TokenMapper → decoded text
```

### 9D Hilbert Scanner

The `HilbertScanner` traverses the 19,683-node grid using the Skilling
algorithm for 9D Hilbert curve coordinates. This ordering maximizes spatial
locality — adjacent nodes in the traversal are adjacent in T⁹.

The scanner produces a flat sequence of `N=19,683` complex amplitudes.

### Mamba-9D SSM

The CognitiveCore applies a Mamba-style selective state space model (Mamba S6)
with:
- Hidden state `H = 256`
- State expansion `16r × 16s = 256D state`
- Input gate, forget gate, and selective scan operating on the amplitude sequence

This produces a contextualized sequence of activations that capture temporal
dynamics (short-term context within the tick window).

### Neuroplastic Transformer (NPT)

On top of the SSM, an 8-head transformer with **wave-correlation attention**:
- Attention weights are computed from the _phase angles_ of the complex
  amplitudes, not from dot products of embedding vectors
- The 8 attention heads are tuned to π·φⁿ spectral bands (same irrational
  spacing as the emitters)
- "Neuroplastic" means attention weights are modulated by the current
  dopamine and entropy fields (metabolic state influences cognition directly)

---

## 5. Autonomy Loop

Nikola has a continuous autonomy cycle running independently of inference:

```
NikolaState:
  dopamine (reward signal, TD-learning)
  td_error  (temporal difference error)
  atp       (energy budget — depletes on computation, replenishes on rest)
  boredom   (exploration drive — increases when reward is flat)
  entropy   (field entropy — drives curiosity-seeking when high)
  torus_energy (total field energy)
  last_action  (action taken at previous tick)
  time         (simulation clock tick)
```

### Dopamine / TD-learning

The dopamine signal follows a standard TD(0) update:
```
δ = reward + γ·V(s') − V(s)
dopamine += α·δ
```

High dopamine → strong injection into T⁹ (system is "engaged")  
Low dopamine → weak injection (system is "disengaged")

### ATP Metabolism

ATP depletes proportionally to the emitter power used each tick and
replenishes passively. Boredom increases linearly when dopamine is flat.
When boredom exceeds a threshold, the system spontaneously injects
entropy-maximizing wave packets (random exploration).

### Decision Loop

The `DecisionLoop` (`include/nikola/autonomy/decision_loop.hpp`) is the
primary tick driver. It coordinates:
- Field propagation (physics step)
- Input injection (if available)
- Cognitive processing
- Action selection (from logit distribution)
- State update (dopamine, ATP, boredom)
- Output emission

---

## 6. Memory

Nikola uses two memory layers:

### Wave-basis Semantic Memory

Stored in an LMDB database as serialized wavefunction snapshots. Each
"memory" is literally a saved Ψ state tagged with a semantic key.
Retrieval uses wavefunction inner product (interference) to find the
best-matching stored state.

### Cross-session Memory

On each startup, `DecisionLoop` auto-loads saved state from `memory_path`
(configurable in `NikolaConfig`). On shutdown, it saves the current field
state. This gives Nikola genuine continuity across sessions.

---

## 7. Security Layer

Nikola has a post-quantum cryptographic identity layer:

- **ML-KEM / Kyber-768** (NIST FIPS 203) — post-quantum key encapsulation
- **SPHINCS+-SHAKE-256f** — post-quantum digital signatures
- **HybridVerifier** — combines classical + post-quantum verification
- **HomeostasisMonitor** — energy/entropy watchdog; detects state anomalies
  and optionally triggers lockdown mode
- **PolymorphicDefense** — ASLR-style token mutation to resist behavioral
  fingerprinting

---

## 8. Social / Economic Layer (Simulated)

These modules exist as a design commitment to the multi-agent future:

- **SocialMembrane** — trust-weighted wave filter: `permeability = trust/(dissonance+ε)`
- **PeerRegistry** — roster of known peer Nikola instances with per-peer membranes
- **NeuralMarketplace** — service registry with mock purchase/execute flow
- **SimulatedWallet** — deterministic identity derivation from seed; debit/credit

Real blockchain integration (Polygon CDK) is deferred. The surface area is
designed so a real wallet implementation is a drop-in replacement for
`SimulatedWallet`.

---

## 9. Invariants — Do Not Break These

1. **Emitter frequencies must be `π·φⁿ`** — changing to rational values
   causes mode-locking (silent failure over ~1000 ticks).

2. **Integrator must remain symplectic** — switching to RK4 or Euler causes
   secular energy drift.

3. **Grid size is `3⁹ = 19,683`** — the Hilbert scanner, Morton encoding,
   and many static arrays are sized to this. This is not a parameter.

4. **BERT-tiny model** must remain at the same architecture (4 layers, 128
   hidden, 2 heads, dynamic-axes) — the holographic projection matrix was
   trained against its embedding space.

5. **`NikolaState` fields** (`dopamine`, `td_error`, `atp`, `boredom`,
   `entropy`, `torus_energy`, `last_action`, `time`) — adding/removing fields
   requires updating every module that consumes the state.

6. **Catch2 v3** — the tests use Catch2 v3 API. Catch2 v2 headers will
   compile but produce wrong results (silent test infrastructure breakage).
