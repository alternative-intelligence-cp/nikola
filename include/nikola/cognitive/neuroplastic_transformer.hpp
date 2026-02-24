/**
 * @file cognitive/neuroplastic_transformer.hpp
 * @brief Phase 37-43 — Neuroplastic Transformer (NPT): the reasoning engine.
 *
 * This is NOT TinyTransformer.  They serve opposite roles:
 *   TinyTransformer  — the "ear":  external text → 128-dim semantic vector
 *   NeuroplasticTransformer — the "thinker": live torus state → reasoned
 *                             field modification (runs in the tick loop)
 *
 * Standard transformer attention is insufficient for wave-field reasoning
 * because dot-product similarity discards phase relationships.  The NPT
 * replaces dot-product attention with Wave Correlation:
 *
 *   Correlation(Q, K) = (|Ψ_Q + Ψ_K|² − (|Ψ_Q|² + |Ψ_K|²))
 *                       ─────────────────────────────────────────
 *                            |Ψ_Q|² + |Ψ_K|² + ε
 *
 * Perfectly in-phase  →  Correlation = +1
 * Perfectly anti-phase →  Correlation = −1
 * Orthogonal           →  Correlation =  0
 *
 * Multi-head design: 8 heads, one per golden-ratio emitter frequency band.
 * Each head attends to a different cognitive frequency range, mirroring the
 * spectral structure already present in the torus (π·φⁿ Hz).
 *
 * | Head | Frequency (Hz) | Cognitive Function     |
 * |------|--------------  |------------------------|
 * |  0   | π·φ⁰ ≈  3.14   | Global Context         |
 * |  1   | π·φ¹ ≈  5.08   | Long-term Memory       |
 * |  2   | π·φ² ≈  8.22   | Working Memory         |
 * |  3   | π·φ³ ≈ 13.31   | Logic & Reasoning (lo) |
 * |  4   | π·φ⁴ ≈ 21.53   | Logic & Reasoning (hi) |
 * |  5   | π·φ⁵ ≈ 34.84   | Sensory Integration    |
 * |  6   | π·φ⁶ ≈ 56.37   | Fine Detail            |
 * |  7   | π·φ⁷ ≈ 91.21   | Error Correction       |
 *
 * Phase delivery plan:
 *   Phase 37 — Data structures; 8-head skeleton; forward() stub       ← HERE
 *   Phase 38 — Wave correlation kernel
 *   Phase 39 — Riemannian curvature bias
 *   Phase 40 — Heterodyne value aggregation
 *   Phase 41 — Multi-head merge + full forward pass
 *   Phase 42 — Wire into DecisionLoop (REASON action)
 *   Phase 43 — Hebbian-Riemannian metric update via NPT output
 *
 * Reference: NIKOLA_REQUIREMENTS_AUDIT.md §F "Neuroplastic Transformer (Wave
 *            Correlation Attention)"
 */
#pragma once

#include <nikola/physics/wave_function.hpp>
#include <nikola/foundation/toroidal_grid.hpp>

#include <array>
#include <cmath>
#include <string>
#include <vector>

namespace nikola::cognitive {

// ============================================================================
// Frequency constants — mirror HolographicInjector::emitter_frequencies()
// ============================================================================

inline constexpr double NPT_PHI = 1.6180339887498948482;
inline constexpr double NPT_PI  = 3.1415926535897932385;
inline constexpr size_t NPT_NUM_HEADS = 8;

/**
 * @brief Compute the n-th NPT head frequency: f_n = π · φⁿ
 */
inline double npt_head_frequency(size_t head_idx) noexcept {
    double phi_n = 1.0;
    for (size_t i = 0; i < head_idx; ++i) phi_n *= NPT_PHI;
    return NPT_PI * phi_n;
}

/**
 * @brief All 8 NPT head frequencies in ascending order.
 */
inline std::array<double, NPT_NUM_HEADS> npt_all_frequencies() noexcept {
    std::array<double, NPT_NUM_HEADS> f{};
    for (size_t i = 0; i < NPT_NUM_HEADS; ++i) f[i] = npt_head_frequency(i);
    return f;
}

/**
 * @brief Normalized frequency weights for Riemannian curvature bias.
 *
 * Each weight w_i = f_i / f_max = φⁱ⁻⁷ ∈ (0, 1].
 *
 * Physical rationale: higher-frequency heads attend to finer temporal scales
 * and are therefore more sensitive to local curvature fluctuations.  The
 * lowest-frequency head (global context) gets the smallest boost; the
 * highest-frequency head (error correction) gets the full boost.
 *
 * Example values: head 0 ≈ 0.056, head 3 ≈ 0.236, head 6 ≈ 0.618, head 7 = 1.0
 */
inline std::array<float, NPT_NUM_HEADS> npt_curvature_weights() noexcept {
    const auto freqs = npt_all_frequencies();
    const double f_max = freqs[NPT_NUM_HEADS - 1];
    std::array<float, NPT_NUM_HEADS> w{};
    for (size_t i = 0; i < NPT_NUM_HEADS; ++i)
        w[i] = static_cast<float>(freqs[i] / f_max);
    return w;
}

// ============================================================================
// Phase 38 — Wave Correlation Kernel
// ============================================================================

/**
 * @brief Wave-correlation score between two WaveFunctions.
 *
 * The standard transformer dot-product discards phase; this kernel preserves
 * it by measuring constructive vs. destructive interference:
 *
 *   Correlation(Q, K) = (|Ψ_Q + Ψ_K|² − (|Ψ_Q|² + |Ψ_K|²))
 *                       ─────────────────────────────────────────
 *                            |Ψ_Q|² + |Ψ_K|² + ε
 *
 * Which simplifies to:
 *   = 2 · Re(⟨Ψ_Q | Ψ_K⟩) / (|Ψ_Q|² + |Ψ_K|² + ε)
 *
 * Returns in [−1, +1]:
 *   +1  perfectly in-phase (maximum constructive interference)
 *   −1  perfectly anti-phase (maximum destructive interference)
 *    0  orthogonal fields   (zero interference)
 *
 * @param Q  Query WaveFunction (head's frequency-band projection).
 * @param K  Key WaveFunction (live torus state or another head's projection).
 */
[[nodiscard]]
inline float wave_correlation(const physics::WaveFunction& Q,
                               const physics::WaveFunction& K) noexcept {
    const double re_inner = Q.inner_product_re(K);
    const double norm_QK  = Q.total_probability() + K.total_probability();
    if (norm_QK < 1e-30) return 0.f;  // both vacuum — undefined, return neutral
    return static_cast<float>((2.0 * re_inner) / (norm_QK + 1e-12));
}

/**
 * @brief Softmax over 8 raw correlation scores with temperature scaling.
 *
 * Converts raw wave-correlation values into a proper probability distribution
 * over attention heads.  Numerically stable (max subtraction before exp).
 *
 * @param raw          Array of 8 raw scores (any range).
 * @param temperature  Scaling divisor τ > 0.  τ < 1 sharpens; τ > 1 flattens.
 * @return             Array of 8 positive values summing to 1.0.
 */
[[nodiscard]]
inline std::array<float, NPT_NUM_HEADS>
attention_softmax(const std::array<float, NPT_NUM_HEADS>& raw,
                  float temperature) noexcept {
    const float tau = (temperature > 1e-6f) ? temperature : 1e-6f;
    std::array<float, NPT_NUM_HEADS> out{};
    // Find max for numeric stability
    float max_val = raw[0];
    for (size_t i = 1; i < NPT_NUM_HEADS; ++i)
        if (raw[i] > max_val) max_val = raw[i];
    // Compute exp((x - max) / τ)
    float sum = 0.f;
    for (size_t i = 0; i < NPT_NUM_HEADS; ++i) {
        out[i] = std::exp((raw[i] - max_val) / tau);
        sum   += out[i];
    }
    // Normalise
    const float inv = (sum > 1e-30f) ? (1.f / sum) : (1.f / static_cast<float>(NPT_NUM_HEADS));
    for (size_t i = 0; i < NPT_NUM_HEADS; ++i) out[i] *= inv;
    return out;
}

// ============================================================================
// Phase 39 — Riemannian Curvature Bias
// ============================================================================

/**
 * @brief Apply Riemannian curvature bias to raw wave-correlation scores.
 *
 * The torus resonance field encodes local metric curvature: nodes with lower
 * resonance r_i are actively plastic (high curvature), indicating regions
 * where memory is being consolidated or patterns are being learned.
 *
 * The bias nudges high-frequency heads (which detect fine-scale structure)
 * to attend more strongly when the field is actively reconfiguring:
 *
 *   biased_i = raw_i + alpha * R̄ * w_i
 *
 * where:
 *   R̄   = mean_curvature(torus_wf)  = (1/N) Σ (1 - r_i) ∈ [0, 1]
 *   w_i = npt_curvature_weights()[i] = f_i / f_max ∈ (0, 1]
 *   alpha = scaling coefficient (default 0.5 — mild, non-dominating)
 *
 * The bias is additive and precedes softmax so it acts as a log-prior:
 * it shifts attention toward high-frequency heads during active learning.
 *
 * @param raw      Array of 8 raw wave-correlation scores (pre-softmax).
 * @param R_mean   Mean curvature scalar from torus_wf.mean_curvature().
 * @param weights  Per-head frequency weights from npt_curvature_weights().
 * @param alpha    Bias coefficient.  0.0 = disabled; 0.5 = moderate; 1.0 = full.
 * @return         Biased raw scores (same scale as input, ready for softmax).
 */
[[nodiscard]]
inline std::array<float, NPT_NUM_HEADS>
apply_curvature_bias(const std::array<float, NPT_NUM_HEADS>& raw,
                     float R_mean,
                     const std::array<float, NPT_NUM_HEADS>& weights,
                     float alpha = 0.5f) noexcept {
    std::array<float, NPT_NUM_HEADS> biased{};
    for (size_t i = 0; i < NPT_NUM_HEADS; ++i)
        biased[i] = raw[i] + alpha * R_mean * weights[i];
    return biased;
}

// ============================================================================
// CognitiveBand — descriptive label for each head's spectral role
// ============================================================================

enum class CognitiveBand : uint8_t {
    GLOBAL_CONTEXT       = 0,  ///< Head 0: π·φ⁰ — broadest context window
    LONG_TERM_MEMORY     = 1,  ///< Head 1: π·φ¹ — resonance with stored memories
    WORKING_MEMORY       = 2,  ///< Head 2: π·φ² — recent short-term buffer
    LOGIC_REASONING_LO   = 3,  ///< Head 3: π·φ³ — logical inference (slow)
    LOGIC_REASONING_HI   = 4,  ///< Head 4: π·φ⁴ — logical inference (fast)
    SENSORY_INTEGRATION  = 5,  ///< Head 5: π·φ⁵ — cross-modal binding
    FINE_DETAIL          = 6,  ///< Head 6: π·φ⁶ — high-resolution analysis
    ERROR_CORRECTION     = 7,  ///< Head 7: π·φ⁷ — discrepancy / surprise
};

/// Human-readable name for a CognitiveBand.
inline const char* band_name(CognitiveBand b) noexcept {
    switch (b) {
        case CognitiveBand::GLOBAL_CONTEXT:      return "GLOBAL_CONTEXT";
        case CognitiveBand::LONG_TERM_MEMORY:    return "LONG_TERM_MEMORY";
        case CognitiveBand::WORKING_MEMORY:      return "WORKING_MEMORY";
        case CognitiveBand::LOGIC_REASONING_LO:  return "LOGIC_REASONING_LO";
        case CognitiveBand::LOGIC_REASONING_HI:  return "LOGIC_REASONING_HI";
        case CognitiveBand::SENSORY_INTEGRATION: return "SENSORY_INTEGRATION";
        case CognitiveBand::FINE_DETAIL:         return "FINE_DETAIL";
        case CognitiveBand::ERROR_CORRECTION:    return "ERROR_CORRECTION";
        default:                                 return "UNKNOWN";
    }
}

// ============================================================================
// WaveCorrelationHead — one attention head operating in one frequency band
// ============================================================================

/**
 * @brief One NPT attention head: Q, K, V wave fields + frequency assignment.
 *
 * Each head holds three WaveFunctions representing the Query, Key, and Value
 * projections of the torus state into this head's cognitive band.
 *
 * After Phase 37 (this file), Q/K/V are initialised to vacuum (zero energy)
 * with the same grid configuration as the live torus.  Phases 38-40 will:
 *   - populate Q/K from the torus state via frequency-selective projection
 *   - compute correlation scores
 *   - aggregate V into an output WaveFunction via heterodyning
 *
 * The head is move-only (WaveFunction is move-only).
 *
 * @param grid_n    Nodes-per-dimension of the torus this head is attached to.
 * @param head_idx  Head index 0-7; determines frequency and CognitiveBand.
 */
struct WaveCorrelationHead {
    size_t       head_index;    ///< 0-7
    CognitiveBand band;         ///< Descriptive label
    double       frequency;     ///< π·φ^head_index  (Hz)

    physics::WaveFunction Q;    ///< Query  projection into this band
    physics::WaveFunction K;    ///< Key    projection into this band
    physics::WaveFunction V;    ///< Value  projection into this band

    // Non-copyable (WaveFunction is non-copyable)
    WaveCorrelationHead(const WaveCorrelationHead&)            = delete;
    WaveCorrelationHead& operator=(const WaveCorrelationHead&) = delete;
    WaveCorrelationHead(WaveCorrelationHead&&)                 = default;
    WaveCorrelationHead& operator=(WaveCorrelationHead&&)      = default;

    /**
     * @brief Construct a head seeded to vacuum (zero field).
     *
     * @param idx     Head index 0-7.
     * @param grid_n  Nodes-per-dimension (must match the live torus).
     */
    explicit WaveCorrelationHead(size_t idx, int grid_n = 3)
        : head_index(idx)
        , band(static_cast<CognitiveBand>(idx))
        , frequency(npt_head_frequency(idx))
        , Q(foundation::GridConfig::uniform(grid_n))
        , K(foundation::GridConfig::uniform(grid_n))
        , V(foundation::GridConfig::uniform(grid_n))
    {
        // Seed each QKV field to vacuum (no pilot wave — zero energy, ready to
        // receive projections from the live torus in Phase 38).
        Q.seed_manifold(grid_n, 0, 1, 0.f, 42);
        K.seed_manifold(grid_n, 0, 1, 0.f, 42);
        V.seed_manifold(grid_n, 0, 1, 0.f, 42);
    }
};

// ============================================================================
// AttentionResult — output of one complete NPT forward pass
// ============================================================================

/**
 * @brief Output of a NeuroplasticTransformer::forward() call.
 *
 * Per-head scores and the merged attended output WaveFunction.
 *
 * After Phase 37: output is vacuum (zero energy).
 * After Phase 40: output carries heterodyne-aggregated wave content.
 * After Phase 41: output is the superposition of all 8 head outputs.
 */
struct AttentionResult {
    /// Per-head correlation score (one float per head, in [−1, +1]).
    /// Zero-initialised in Phase 37; populated in Phase 38+.
    std::array<float, NPT_NUM_HEADS> head_scores{};

    /// Merged attended WaveFunction — the field modification the NPT proposes.
    /// In Phase 42 this is injected back into the torus at REASON action time.
    physics::WaveFunction output;

    /// True if the forward pass produced a non-trivial (non-vacuum) result.
    /// Set to false by the stub until Phase 40 populates output.
    bool has_output = false;

    /**
     * @brief Construct with a vacuum output WaveFunction.
     * @param grid_n  Must match the torus the NPT is attached to.
     */
    explicit AttentionResult(int grid_n = 3)
        : output(foundation::GridConfig::uniform(grid_n))
    {
        head_scores.fill(0.f);
        output.seed_manifold(grid_n, 0, 1, 0.f, 0);
    }

    // Move-only
    AttentionResult(const AttentionResult&)            = delete;
    AttentionResult& operator=(const AttentionResult&) = delete;
    AttentionResult(AttentionResult&&)                 = default;
    AttentionResult& operator=(AttentionResult&&)      = default;
};

// ============================================================================
// NeuroplasticTransformer
// ============================================================================

/**
 * @brief 8-head wave-correlation attention over the CognitiveTorus.
 *
 * Architecture position (within the tick loop):
 *
 *   CognitiveTorus (live field)
 *          ↓  read torus state
 *   NeuroplasticTransformer::forward(torus_wf)
 *          ↓  AttentionResult.output (proposed field modification)
 *   CognitiveTorus::inject via REASON action weighting
 *
 * The NPT does NOT own the torus — it receives a const reference to the
 * WaveFunction per forward() call (observer pattern, same as DecisionLoop).
 *
 * Construction: caller provides grid_n to size the 8 QKV WaveFunctions
 * to match the live torus.  Must be reconstructed if torus grid_n changes
 * (rare — only neurogenesis triggers this).
 *
 * Thread safety: NOT thread-safe.  Must be called from the same thread as
 * DecisionLoop::tick().
 */
class NeuroplasticTransformer {
public:
    // ------------------------------------------------------------------ construction

    /**
     * @brief Construct with 8 heads sized to match a torus of grid_n nodes/dim.
     *
     * @param grid_n        Nodes per dimension (e.g. 3 for 3^9 = 19,683 nodes).
     * @param temperature   Softmax temperature τ.  Default 1.0f (neutral).
     * @param curvature_alpha  Strength of Riemannian curvature bias α.
     *                         0.0 = disabled; 0.5 = moderate (default); 1.0 = full.
     */
    explicit NeuroplasticTransformer(int grid_n = 3, float temperature = 1.0f,
                                     float curvature_alpha = 0.5f)
        : grid_n_(grid_n)
        , temperature_(temperature)
        , curvature_alpha_(curvature_alpha)
    {
        heads_.reserve(NPT_NUM_HEADS);
        for (size_t i = 0; i < NPT_NUM_HEADS; ++i)
            heads_.emplace_back(i, grid_n_);
    }

    // Non-copyable (heads hold WaveFunctions)
    NeuroplasticTransformer(const NeuroplasticTransformer&)            = delete;
    NeuroplasticTransformer& operator=(const NeuroplasticTransformer&) = delete;
    NeuroplasticTransformer(NeuroplasticTransformer&&)                 = default;
    NeuroplasticTransformer& operator=(NeuroplasticTransformer&&)      = default;

    // ------------------------------------------------------------------ Phase 41: V projection + forward

    /**
     * @brief Project the torus WaveFunction into each head's frequency band.
     *
     * For each head i, this produces a phase-rotated copy of the torus field:
     *
     *   V_i(x) = torus_wf(x) · e^{i · 2π · f_i · t}
     *
     * where f_i = π·φⁱ (the head's cognitive-band carrier frequency) and
     * t = torus_wf.time() (the current simulation clock).
     *
     * Physical interpretation:
     *   Each head demodulates the torus at its own carrier frequency.  The
     *   torus field contains activity at all cognitive frequencies mixed
     *   together; the phase rotation selectively "highlights" the component
     *   at f_i.  Heads at different frequencies therefore see genuinely
     *   different aspects of the same field.
     *
     *   At t = 0 all rotations are e^{i·0} = 1, so all V_i = torus_wf
     *   (degenerate uniform case — identical to Phase 40 behaviour).
     *
     * Must be called before forward() for per-head differentiation to be
     * active.  Called automatically by forward() on every tick so callers
     * do not need to manage this explicitly.
     *
     * @param torus_wf  Live torus WaveFunction (read-only; cloned per head).
     */
    void project_heads(const physics::WaveFunction& torus_wf) {
        const float t = torus_wf.time();
        for (size_t i = 0; i < NPT_NUM_HEADS; ++i) {
            heads_[i].V = torus_wf.clone();
            const float phase = static_cast<float>(
                2.0 * NPT_PI * heads_[i].frequency * static_cast<double>(t));
            heads_[i].V.phase_rotate_psi(phase);
        }
    }

    /**
     * @brief Attend over the current torus WaveFunction.
     *
     * Phase 41 full pipeline:
     *   1. project_heads(torus_wf)  — V_i = torus_wf · e^{i·2π·f_i·t}
     *   2. raw[i]    = wave_correlation(head.Q, torus_wf)
     *   3. biased[i] = raw[i] + α · R̄ · w[i]
     *   4. scores[i] = softmax(biased, τ)
     *   5. output_ψ  = Σᵢ scores[i] · head.V_i
     *
     * Phase 43-44: Hebbian Q update + K differentiation (see hebbian_update /
     *              k_update) using the effective rates scaled by dopamine.
     *
     * Phase 45: Dopamine-modulated learning rate.
     *   Effective rates for this pass:
     *     η_scale         = 1 + tanh(dopamine − 0.5)
     *     effective_hebb  = hebbian_alpha_ · η_scale
     *     effective_k     = k_alpha_       · η_scale
     *
     *   Regime table:
     *     D = 1.0 (spike)    η_scale ≈ 1.46  hyper-plastic, fastest learning
     *     D = 0.5 (baseline) η_scale = 1.00  standard rate, Phase 43-44 behaviour
     *     D = 0.0 (dip)      η_scale ≈ 0.54  plasticity dampened (aversion lock)
     *
     * Phase 46: Serotonin-modulated metric elasticity (restoring force).
     *   After Hebbian/K updates the heads are pulled back toward vacuum:
     *     λ_s = λ_base · (0.5 + 0.5·tanh(S − 0.5))
     *     Q_i ← Q_i · (1 − λ_s)     K_i ← K_i · (1 − λ_s)
     *
     *   Regime table:
     *     S = 1.0 (exploitation)  λ_s ≈ λ_base       stabilising, resists new change
     *     S = 0.5 (baseline)      λ_s = λ_base/2      moderate elasticity
     *     S = 0.0 (exploration)   λ_s ≈ 0             full plasticity, no damping
     *
     * Phase 47: Norepinephrine arousal modulation (refractive index coupling).
     *   High NE → lower effective temperature → sharper attention (hyper-vigilance,
     *   winner-take-all: one dominant head saturates the softmax).
     *   Low NE  → higher effective temperature → broader multi-head integration
     *   (calm, exploratory, diffuse association).
     *
     *   Mirrors the spec's refractive index formula §5.1:
     *     s_eff = s_local / (1 + N_t)
     *   mapped to NPT attention temperature:
     *     τ_eff = τ / (1 + N_t)
     *
     *   Regime table:
     *     N = 1.0 (panic/stress)  τ_eff = τ/2     sharp, tunnel-vision attention
     *     N = 0.5 (baseline)      τ_eff = τ/1.5   moderate focus
     *     N = 0.0 (deep calm)     τ_eff = τ       full breadth, max integration
     *
     * @param torus_wf       Read-only live torus WaveFunction.
     * @param dopamine       Current dopamine level ∈ [0, 1].  Default 0.5 (baseline)
     *                       — identical to Phase 43-44 behaviour when not provided.
     * @param serotonin      Current serotonin level ∈ [0, 1].  Default 0.5 (baseline)
     *                       — moderate elasticity, identical to Phase 45 behaviour
     *                       when not provided.
     * @param norepinephrine Current norepinephrine level ∈ [0, 1].  Default 0.5
     *                       — τ_eff = τ/1.5, identical to Phase 46 behaviour
     *                       when not provided.
     * @return AttentionResult with softmax head_scores and the heterodyne
     *         output WaveFunction (has_output = true).
     */
    [[nodiscard]]
    AttentionResult forward(const physics::WaveFunction& torus_wf,
                            float dopamine        = 0.5f,
                            float serotonin       = 0.5f,
                            float norepinephrine  = 0.5f) {
        // Step 1: populate per-head V with frequency-band projections
        project_heads(torus_wf);

        // Step 2: raw correlation scores (Q is still vacuum; differentiation
        //         of Q/K arrives in Phase 43 via Hebbian-Riemannian updates)
        std::array<float, NPT_NUM_HEADS> raw{};
        for (size_t i = 0; i < NPT_NUM_HEADS; ++i)
            raw[i] = wave_correlation(heads_[i].Q, torus_wf);

        // Step 3-4: curvature bias + softmax
        const float R_mean  = static_cast<float>(torus_wf.mean_curvature());
        const auto  weights = npt_curvature_weights();
        const auto  biased  = apply_curvature_bias(raw, R_mean, weights, curvature_alpha_);

        // Phase 47: Norepinephrine arousal — modulate effective attention temperature.
        // τ_eff = τ / (1 + N)  mirrors spec §5.1 refractive index formula.
        // High N → lower τ_eff → sharper softmax (hyper-vigilance, focus).
        // Low N  → τ_eff → τ   → broad integration (calm, exploratory).
        const float tau_eff  = temperature_ / (1.0f + std::clamp(norepinephrine, 0.0f, 1.0f));
        last_tau_eff_        = tau_eff;

        AttentionResult result(grid_n_);
        result.head_scores = attention_softmax(biased, tau_eff);

        // Step 5: heterodyne aggregation over per-head V fields
        result.output = heads_[0].V.clone();
        result.output.scale_by(result.head_scores[0]);
        for (size_t i = 1; i < NPT_NUM_HEADS; ++i)
            result.output.add_scaled(heads_[i].V, result.head_scores[i]);

        result.has_output = true;

        // Phase 45: Dopamine-modulated learning rate.
        // η(D) = η_base · (1 + tanh(D − 0.5))  implements spec §A:
        //   High D → faster Q/K updates (reward just happened, encode hard)
        //   Baseline → unmodified Phase 43-44 rates
        //   Low D → plasticity dampened (aversion / punishment lock)
        const float eta_scale   = 1.0f + std::tanh(dopamine - 0.5f);
        last_eta_scale_         = eta_scale;
        const float eff_hebb    = hebbian_alpha_ * eta_scale;
        const float eff_k       = k_alpha_       * eta_scale;

        // Phase 43: Hebbian-Riemannian metric update — Q_i drifts toward output
        hebbian_update(result.output, R_mean, eff_hebb);

        // Phase 44: K-head differentiation — K_i drifts toward the live torus
        k_update(torus_wf, result.head_scores, R_mean, eff_k);

        // Phase 46: Serotonin metric elasticity — restoring force toward vacuum.
        // Applied after Hebbian/K updates so that plasticity and stability
        // act on the same tick: encode first, then damp excess.
        serotonin_elasticity(serotonin);

        return result;
    }

    // ------------------------------------------------------------------ accessors

    /// Number of active attention heads (always NPT_NUM_HEADS = 8).
    size_t num_heads() const noexcept { return heads_.size(); }

    /// Grid size this NPT was constructed for.
    int grid_n() const noexcept { return grid_n_; }

    /// Attention temperature τ.
    float temperature() const noexcept { return temperature_; }

    /// Riemannian curvature bias coefficient α.
    float curvature_alpha() const noexcept { return curvature_alpha_; }

    /// Hebbian learning rate α_hebb (Phase 43).  Default 0.01.
    float hebbian_alpha() const noexcept { return hebbian_alpha_; }

    /// Set the Hebbian learning rate.  0.0 disables Q-head updates entirely.
    void set_hebbian_alpha(float alpha) noexcept { hebbian_alpha_ = alpha; }

    /// K-head learning rate α_k (Phase 44).  Default 0.005 (half of α_hebb).
    float k_alpha() const noexcept { return k_alpha_; }

    /// Set the K-head learning rate.  0.0 disables K-head updates entirely.
    void set_k_alpha(float alpha) noexcept { k_alpha_ = alpha; }

    /// Last computed η_scale = 1 + tanh(D − 0.5) from the most recent forward().
    /// Phase 45 telemetry: 1.0 at baseline, ~1.46 at full spike, ~0.54 at dip.
    float last_eta_scale() const noexcept { return last_eta_scale_; }

    // ── Phase 46 accessors ──────────────────────────────────────────────────

    /// Serotonin elasticity base λ_base (Phase 46).  Default 0.002.
    /// Scales the restoring force: Q/K decay per tick = λ_base·(0.5+0.5·tanh(S−0.5)).
    float serotonin_lambda_base() const noexcept { return serotonin_lambda_base_; }

    /// Set λ_base.  0.0 disables serotonin elasticity entirely.
    void  set_serotonin_lambda_base(float lb) noexcept { serotonin_lambda_base_ = lb; }

    /// Last computed λ_s = λ_base·(0.5+0.5·tanh(S−0.5)) (Phase 46 telemetry).
    float last_lambda_s() const noexcept { return last_lambda_s_; }

    // ── Phase 47 accessors ──────────────────────────────────────────────────

    /// Last computed τ_eff = τ / (1 + N) from the most recent forward() (Phase 47).
    /// Telemetry: equals τ at N=0 (calm), τ/2 at N=1.0 (panic).
    float last_tau_eff() const noexcept { return last_tau_eff_; }

    /// Read-only access to head i (0-7).
    const WaveCorrelationHead& head(size_t i) const { return heads_.at(i); }

    /// Frequency of head i (π·φⁱ Hz).
    double head_frequency(size_t i) const { return heads_.at(i).frequency; }

    /// CognitiveBand of head i.
    CognitiveBand head_band(size_t i) const { return heads_.at(i).band; }

private:
    int   grid_n_;
    float temperature_;
    float curvature_alpha_;
    float hebbian_alpha_         = 0.01f;  ///< Phase 43: Q-head Hebbian learning rate
    float k_alpha_               = 0.005f; ///< Phase 44: K-head differentiation rate
    float last_eta_scale_        = 1.0f;   ///< Phase 45: last dopamine η_scale (telemetry)
    float serotonin_lambda_base_ = 0.002f; ///< Phase 46: elasticity base λ_base
    float last_lambda_s_         = 0.0f;   ///< Phase 46: last computed λ_s (telemetry)
    float last_tau_eff_          = 0.0f;   ///< Phase 47: last computed τ_eff telemetry
    std::vector<WaveCorrelationHead> heads_;

    // ------------------------------------------------------------------ Phase 43: Hebbian-Riemannian update

    /**
     * @brief Update each head's Q toward the attended output field.
     *
     * Implements the Hebbian-Riemannian rule:
     *
     *   Q_i ← (1 − t_i) · Q_i  +  t_i · output
     *
     * where:
     *   t_i = α_hebb · R̄ · w_i
     *   R̄   = mean curvature of the live torus  (=0 → frozen; =1 → fully plastic)
     *   w_i = npt_curvature_weights()[i]         (ascending: high-freq heads update faster)
     *
     * Heads that fired (produced output) update their Q toward what they
     * attended — the Hebbian "fire together, wire together" principle.
     * The Riemannian weighting (R̄ · w_i) ensures learning only happens
     * when the field is actively plastic (non-zero curvature).
     *
     * t_i is clamped to [0, 1] to prevent overshoot.
     *
     * @param output   The attended output WaveFunction from this forward pass.
     * @param R_mean   Mean curvature scalar already computed in forward().
     */
    void hebbian_update(const physics::WaveFunction& output, float R_mean,
                        float effective_alpha) {
        if (effective_alpha < 1e-9f || R_mean < 1e-9f) return;
        const auto weights = npt_curvature_weights();
        for (size_t i = 0; i < NPT_NUM_HEADS; ++i) {
            float t = effective_alpha * R_mean * weights[i];
            if (t < 1e-9f) continue;
            if (t > 1.0f) t = 1.0f;
            heads_[i].Q.scale_by(1.0f - t);
            heads_[i].Q.add_scaled(output, t);
        }
    }

    // ------------------------------------------------------------------ Phase 44: K-head differentiation

    /**
     * @brief Update each head's K toward the live torus input WaveFunction.
     *
     * Implements the complementary K-head rule:
     *
     *   K_i ← (1 − s_i) · K_i  +  s_i · torus_wf
     *
     * where:
     *   s_i = k_alpha · score_i · R̄ · w_i
     *   score_i = softmax attention weight for head i (high-attending heads
     *             reinforce their key more strongly — "the key that worked")
     *   R̄      = mean curvature gate (same signal as Q update)
     *   w_i    = npt_curvature_weights()[i] (ascending, high-freq first)
     *
     * Physical interpretation:
     *   The head that contributes most to the attention output (highest score_i)
     *   also pulls its K closest to the current torus state.  On subsequent
     *   ticks, wave_correlation(Q_i, K_i) reflects genuine pattern learning:
     *   Q has moved toward what the system outputs, K toward what arrived.
     *
     * Together with Q update (Phase 43), this closes the QK co-adaptation loop
     *   Q_i → output WF (what I reason about)
     *   K_i → torus WF (what arrived in the field)
     * implementing the first term of ∂g_ij/∂t = −η·Re(Ψ_Q·Ψ_K*) in field form.
     *
     * s_i is clamped to [0, 1].
     *
     * @param torus_wf  The live torus input used in this forward pass.
     * @param scores    Softmax attention weights from this forward pass.
     * @param R_mean    Mean curvature already computed in forward().
     */
    void k_update(const physics::WaveFunction& torus_wf,
                  const std::array<float, NPT_NUM_HEADS>& scores,
                  float R_mean,
                  float effective_alpha)
    {
        if (effective_alpha < 1e-9f || R_mean < 1e-9f) return;
        const auto weights = npt_curvature_weights();
        for (size_t i = 0; i < NPT_NUM_HEADS; ++i) {
            float s = effective_alpha * scores[i] * R_mean * weights[i];
            if (s < 1e-9f) continue;
            if (s > 1.0f) s = 1.0f;
            heads_[i].K.scale_by(1.0f - s);
            heads_[i].K.add_scaled(torus_wf, s);
        }
    }

    // ------------------------------------------------------------------ Phase 46: Serotonin elasticity

    /**
     * @brief Apply serotonin-modulated restoring force toward vacuum (flat metric).
     *
     * Implements the second term of the full metric update equation from spec §F:
     *
     *   ∂g_ij/∂t  +=  λ(S_t) · (g_ij − δ_ij)
     *
     * In NPT terms: Q_i and K_i are multiplied by (1 − λ_s), pulling them
     * toward the zero (vacuum) WaveFunction each tick.  This elastic decay
     * models structural stiffness — high serotonin means the manifold resists
     * deformation and old patterns are gently erased unless continuously
     * reinforced by Hebbian/K updates.
     *
     * The serotonin-modulated coefficient:
     *   λ_s = λ_base · (0.5 + 0.5 · tanh(S − 0.5))
     *
     * Regime table:
     *   S = 1.0 (exploitation)  λ_s ≈ λ_base       strongest restoring (stability)
     *   S = 0.5 (baseline)      λ_s = λ_base · 0.5  moderate elasticity
     *   S = 0.0 (exploration)   λ_s ≈ 0             full plasticity, no damping
     *
     * Note: λ_base = 0.002 by default, so even at S=1 the decay per tick
     *   is only ~0.2% — slow enough not to interfere with rapid Hebbian
     *   learning but cumulative over many ticks.
     *
     * @param serotonin  Current serotonin level ∈ [0, 1].
     */
    void serotonin_elasticity(float serotonin) {
        const float lambda_s = serotonin_lambda_base_
                               * (0.5f + 0.5f * std::tanh(serotonin - 0.5f));
        last_lambda_s_ = lambda_s;
        if (lambda_s < 1e-9f) return;
        const float decay = 1.0f - lambda_s;
        for (auto& h : heads_) {
            h.Q.scale_by(decay);
            h.K.scale_by(decay);
        }
    }
};

} // namespace nikola::cognitive
