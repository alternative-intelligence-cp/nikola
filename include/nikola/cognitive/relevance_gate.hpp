/**
 * @file cognitive/relevance_gate.hpp
 * @brief Phase 35 — RelevanceGate: input salience filter (RGT).
 *
 * Biological analogy: the Reticular Activating System (RAS).
 * The RAS decides, at a pre-conscious level, what is worth processing.
 * Routine/familiar signals are filtered out before they consume expensive
 * embedding and torus-injection resources.  Novel or energetically loud
 * signals pass through at full amplitude.
 *
 * Architecture position:
 *
 *   External stimulus (ZMQ / file)
 *          ↓
 *   [ RelevanceGate::gate(input_wf, norepinephrine) ]
 *          ↓ SalienceResult.weight  (if > threshold, inject into embedder)
 *   NonaryEmbedder → HolographicInjector → CognitiveTorus
 *
 * Two salience components:
 *
 *   1. Novelty   — cosine distance between the input WaveFunction and the
 *      gate's reference field (a snapshot of the last-updated torus state).
 *      0 = identical to last known state (routine/boring),
 *      1 = completely orthogonal (never seen this before).
 *
 *   2. Urgency   — total probability (energy) of the input WaveFunction.
 *      Loud signals are intrinsically more salient regardless of novelty.
 *      Mapped to [0, 1] via a saturating sqrt transform.
 *
 * Combined salience: s = 0.6 × novelty + 0.4 × urgency
 *
 * Effective threshold: t_eff = base_threshold / (1 + norepinephrine)
 *   - norepinephrine = 0.0 → t_eff = base_threshold  (calm baseline)
 *   - norepinephrine = 1.0 → t_eff = base_threshold / 2  (hypervigilance)
 *   High norepinephrine (stress, alertness) lowers the bar — more gets through.
 *
 * Marginal signals (s < t_eff but > t_eff × 0.5) are not dropped; they are
 * attenuated: the returned weight < 1 so the caller can scale injection
 * amplitude proportionally.  Outright quiet/familiar signals return weight = 0
 * and passes = false.
 *
 * Design notes:
 *   - Header-only, no heap allocation.
 *   - The reference field is stored as a small float array (mean psi per node).
 *     Callers update it via set_reference() whenever the torus state changes
 *     significantly (not every tick — sparse updates are fine).
 *   - Thread-safety: not thread-safe; protect externally if calling across
 *     threads.
 *
 * Reference: NIKOLA_REQUIREMENTS_AUDIT.md §G "Relevance Gating Transformer"
 */
#pragma once

#include <nikola/physics/wave_function.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace nikola::cognitive {

// ============================================================================
// SalienceResult
// ============================================================================

/**
 * @brief Output of a single RelevanceGate evaluation.
 */
struct SalienceResult {
    float novelty   = 0.f;  ///< Cosine distance from reference field [0, 1]
    float urgency   = 0.f;  ///< Amplitude-based energy term [0, 1]
    float salience  = 0.f;  ///< Combined score: 0.6×novelty + 0.4×urgency
    float weight    = 0.f;  ///< Injection weight to apply (0 = drop, 1 = full)
    bool  passes    = false; ///< True if weight > 0 (signal should be processed)
};

// ============================================================================
// RelevanceGate
// ============================================================================

/**
 * @brief Salience filter — decides what is worth injecting into the torus.
 *
 * Usage:
 * @code
 *   RelevanceGate gate;
 *
 *   // After each significant torus state change:
 *   gate.set_reference(torus.wave_function());
 *
 *   // Before embedding any incoming signal:
 *   auto result = gate.gate(incoming_wf, norepinephrine_level);
 *   if (result.passes) {
 *       embedder.embed_scaled(text, result.weight);
 *   }
 * @endcode
 */
class RelevanceGate {
public:
    // ------------------------------------------------------------------ construction

    /**
     * @brief Construct with a salience threshold.
     *
     * @param base_threshold  Minimum salience to pass (pre-norepinephrine).
     *                        Default 0.25f works well for a cold torus
     *                        (novelty = 1, urgency ≈ 0.5 → salience ≈ 0.8).
     * @param attenuation_lo  Lower bound of the marginal zone as a fraction
     *                        of the effective threshold (default 0.5).
     *                        Signals in [t_eff × attenuation_lo, t_eff) get
     *                        a reduced weight rather than being dropped.
     */
    explicit RelevanceGate(float base_threshold  = 0.25f,
                           float attenuation_lo  = 0.5f) noexcept
        : base_threshold_(base_threshold)
        , attenuation_lo_(attenuation_lo)
    {}

    // ------------------------------------------------------------------ reference update

    /**
     * @brief Update the reference field used to compute novelty.
     *
     * Call this whenever the torus has processed enough new information
     * that "what we already know" has changed meaningfully.  Updating
     * every tick is counterproductive — sparse updates are correct.
     *
     * Implementation stores only the squared-magnitude per node
     * (memory-efficient: no complex pairs, just energy density), plus
     * the full complex mean for cosine similarity.
     *
     * @param wf  Current torus WaveFunction.
     */
    void set_reference(const physics::WaveFunction& wf) noexcept {
        const size_t N = wf.num_nodes();
        if (N == 0) {
            ref_real_.clear();
            ref_imag_.clear();
            ref_energy_ = 0.f;
            return;
        }

        const float* pr = wf.grid().psi_real();
        const float* pi = wf.grid().psi_imag();

        ref_real_.assign(pr, pr + N);
        ref_imag_.assign(pi, pi + N);

        ref_energy_ = 0.f;
        for (size_t i = 0; i < N; ++i)
            ref_energy_ += pr[i]*pr[i] + pi[i]*pi[i];
    }

    // ------------------------------------------------------------------ embedding reference

    /**
     * @brief Update the reference from a float embedding vector.
     *
     * Called after a text embedding is computed so the gate can measure
     * novelty of future embeddings at embedding-level cost (no WF needed).
     * Stores a normalised copy of the vector.
     *
     * @param emb  Float embedding vector (any length, typically 128).
     */
    void set_reference_embedding(const std::vector<float>& emb) noexcept {
        ref_emb_ = emb;
        double sq = 0.0;
        for (float v : ref_emb_) sq += static_cast<double>(v) * v;
        ref_emb_norm_ = static_cast<float>(std::sqrt(sq));
    }

    bool has_reference_embedding() const noexcept { return !ref_emb_.empty(); }

    // ------------------------------------------------------------------ embedding-level gate

    /**
     * @brief Evaluate salience directly from a float embedding vector.
     *
     * This is the fast pre-injection path for text signals:
     *   - Urgency: saturating transform on L2 norm of the embedding.
     *   - Novelty: cosine distance from the stored reference embedding.
     *
     * If no embedding reference has been set (cold start), novelty = 1.0.
     *
     * @param emb            Float embedding vector (same space as reference).
     * @param norepinephrine Alertness modulator in [0, 1] (default 0.5).
     *
     * @return SalienceResult identical in semantics to gate(WaveFunction).
     */
    [[nodiscard]]
    SalienceResult gate_embedding(const std::vector<float>& emb,
                                  float norepinephrine = 0.5f) const noexcept {
        SalienceResult result;

        // ── urgency ─────────────────────────────────────────────────────────
        double sq = 0.0;
        for (float v : emb) sq += static_cast<double>(v) * v;
        const float norm = static_cast<float>(std::sqrt(sq));
        // Normalise: typical BERT-Tiny CLS vector has L2 norm ≈ 5-15.
        // Divide by 10 before sqrt-saturation to put it in a useful range.
        result.urgency = std::min(1.f,
            static_cast<float>(std::sqrt(static_cast<double>(norm) / 10.0)));

        // ── novelty ─────────────────────────────────────────────────────────
        result.novelty = compute_novelty_emb(emb, norm);

        // ── combined salience & weight ───────────────────────────────────────
        result.salience = 0.6f * result.novelty + 0.4f * result.urgency;
        apply_threshold(result, norepinephrine);
        return result;
    }

    // ------------------------------------------------------------------ WF-level gate evaluation

    /**
     * @brief Evaluate salience of an incoming signal WaveFunction.
     *
     * @param input_wf       The incoming signal as a WaveFunction.
     *                       Does not need to be the same size as the reference;
     *                       the overlap is computed over the shorter of the two.
     * @param norepinephrine Alertness/stress modulator in [0, 1].
     *                       Higher → lower effective threshold → more passes.
     *                       0.5f is a calm-but-awake baseline.
     *
     * @return SalienceResult with novelty, urgency, salience, weight, passes.
     */
    [[nodiscard]]
    SalienceResult gate(const physics::WaveFunction& input_wf,
                        float norepinephrine = 0.5f) const noexcept {
        SalienceResult result;

        // ── urgency: saturating transform on input energy ──────────────────
        const double input_prob = input_wf.total_probability();
        result.urgency = static_cast<float>(
            std::min(1.0, std::sqrt(input_prob)));

        // ── novelty: cosine distance from reference field ───────────────────
        result.novelty = compute_novelty(input_wf);

        // ── combined salience + threshold ───────────────────────────────────
        result.salience = 0.6f * result.novelty + 0.4f * result.urgency;
        apply_threshold(result, norepinephrine);
        return result;
    }

    // ------------------------------------------------------------------ accessors

    float base_threshold()         const noexcept { return base_threshold_; }
    float attenuation_lo()         const noexcept { return attenuation_lo_; }
    float ref_energy()             const noexcept { return ref_energy_; }
    bool  has_reference()          const noexcept { return !ref_real_.empty(); }

private:
    // ------------------------------------------------------------------ shared threshold logic

    void apply_threshold(SalienceResult& r, float norepinephrine) const noexcept {
        const float ne    = std::clamp(norepinephrine, 0.f, 1.f);
        const float t_eff = base_threshold_ / (1.f + ne);
        const float t_low = t_eff * attenuation_lo_;

        if (r.salience >= t_eff) {
            r.weight = 1.f;
            r.passes = true;
        } else if (r.salience >= t_low) {
            r.weight = (r.salience - t_low) / (t_eff - t_low);
            r.passes = r.weight > 0.f;
        } else {
            r.weight = 0.f;
            r.passes = false;
        }
    }

    // ------------------------------------------------------------------ WF novelty

    /**
     * @brief Cosine distance between input_wf and the stored WF reference.
     */
    float compute_novelty(const physics::WaveFunction& input_wf) const noexcept {
        if (ref_real_.empty() || input_wf.num_nodes() == 0)
            return 1.f;

        const size_t N = std::min(input_wf.num_nodes(), ref_real_.size());
        const float* pr = input_wf.grid().psi_real();
        const float* pi = input_wf.grid().psi_imag();

        double dot_r   = 0.0;
        double dot_i   = 0.0;
        double e_input = 0.0;

        for (size_t i = 0; i < N; ++i) {
            dot_r   += static_cast<double>(pr[i]) * ref_real_[i]
                     + static_cast<double>(pi[i]) * ref_imag_[i];
            dot_i   += static_cast<double>(pi[i]) * ref_real_[i]
                     - static_cast<double>(pr[i]) * ref_imag_[i];
            e_input += static_cast<double>(pr[i])*pr[i]
                     + static_cast<double>(pi[i])*pi[i];
        }

        const double dot_mag = std::sqrt(dot_r*dot_r + dot_i*dot_i);
        const double denom   = std::sqrt(e_input) * std::sqrt(ref_energy_) + 1e-12;
        const float  cos_sim = static_cast<float>(dot_mag / denom);
        return 1.f - std::clamp(cos_sim, 0.f, 1.f);
    }

    // ------------------------------------------------------------------ embedding novelty

    /**
     * @brief Cosine distance between emb and the stored float embedding reference.
     */
    float compute_novelty_emb(const std::vector<float>& emb,
                               float emb_norm) const noexcept {
        if (ref_emb_.empty()) return 1.f;
        if (emb_norm < 1e-9f || ref_emb_norm_ < 1e-9f) return 1.f;

        const size_t N = std::min(emb.size(), ref_emb_.size());
        double dot = 0.0;
        for (size_t i = 0; i < N; ++i)
            dot += static_cast<double>(emb[i]) * ref_emb_[i];

        const float cos_sim = static_cast<float>(
            dot / (static_cast<double>(emb_norm) * ref_emb_norm_));
        return 1.f - std::clamp(cos_sim, 0.f, 1.f);
    }

    // ------------------------------------------------------------------ data
    float base_threshold_;
    float attenuation_lo_;

    // WF-level reference
    std::vector<float> ref_real_;
    std::vector<float> ref_imag_;
    float              ref_energy_ = 0.f;

    // Embedding-level reference
    std::vector<float> ref_emb_;
    float              ref_emb_norm_ = 0.f;
};

} // namespace nikola::cognitive
