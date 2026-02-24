/**
 * @file autonomy/entropy_estimator.hpp
 * @brief Gap 5.2 — Monte Carlo Shannon entropy estimation + boredom regulation.
 *
 * EntropyEstimator:
 *   H ≈ -Σ_{k=1}^K  p_k log₂(p_k)
 *   p_k = |Ψ_k|² / Σ|Ψ_j|²
 *
 *   Samples K = 1000 active nodes → O(K) instead of O(N), tractable at 2 kHz.
 *
 * BoredomRegulator (Phase 49 — AUTO-04 Boredom Singularity Fix, spec §6.2):
 *
 *   ΔB(t) = α_acc · (1 − tanh(k · H(Ψ))) · dt    ← bounded sigmoidal accum
 *         − decay_rate · dt                         ← passive drain
 *
 *   H → 0:  ΔB = α_acc · dt   (max finite rate — singularity eliminated)
 *   H → ∞:  ΔB = 0            (high entropy → no accumulation)
 *   θ_explore = 0.8            (raised from 0.7 stub per spec §6.3)
 *
 * Target range: H ∈ [4, 8] for healthy cognitive state (entropy target = 6.0).
 */

#pragma once

#include <algorithm>   // std::shuffle, std::min
#include <cmath>       // std::log2, std::sqrt
#include <numeric>     // std::iota
#include <random>
#include <span>
#include <vector>

namespace nikola::autonomy {

// ── Gap 5.2 constants ────────────────────────────────────────────────────────

/// Number of nodes sampled per entropy estimate.  Spec: K = 1000.
inline constexpr int   ENTROPY_SAMPLE_SIZE    = 1000;

/// Minimum intensity threshold to count a node as "active".
inline constexpr float ENTROPY_ACTIVE_THRESH  = 1e-6f;

/// Entropy target for a "healthy" cognitive state.  Spec: 6.0.
inline constexpr float ENTROPY_TARGET         = 6.0f;

// ── Phase 49: AUTO-04 Sigmoidal Boredom constants (Boredom Singularity Fix) ──

/// Peak accumulation rate α_acc: ΔB_max = α_acc per second at H = 0.
/// Kept equal to the old BOREDOM_RATE=0.1 so H=0 behaviour is unchanged.
inline constexpr float BOREDOM_ALPHA_ACC      = 0.1f;

/// Sigmoid steepness k: tanh(k·H).  k=0.5 → ΔB ≈ 0 at H ≥ 6 (target).
inline constexpr float BOREDOM_K              = 0.5f;

/// Passive decay rate: constant boredom drain per second (dt units).
/// Allows boredom to fall back to 0 after exploration restores entropy.
inline constexpr float BOREDOM_DECAY_RATE     = 0.01f;

/// Threshold above which the system enters spontaneous exploration.
/// Spec §6.3: "θ_explore ≈ 0.8".  Raised from the original 0.7 stub.
inline constexpr float BOREDOM_EXPLORE_THRESH = 0.8f;

// ── EntropyEstimator ──────────────────────────────────────────────────────────

/**
 * @class EntropyEstimator
 * @brief Computes Shannon entropy of the wavefunction via Monte Carlo sampling.
 *
 * Accepts the SoA (Structure-of-Arrays) layout used by TorusGrid:
 * separate float spans for real and imaginary parts.
 *
 * The RNG is seeded deterministically; pass your own seed for reproducibility.
 */
class EntropyEstimator {
public:
    explicit EntropyEstimator(uint32_t seed = 42u)
        : rng_(seed) {}

    /**
     * @brief Estimate H from SoA psi arrays.
     *
     * @param psi_real  Span of Re(Ψ) values, one per grid node.
     * @param psi_imag  Span of Im(Ψ) values, one per grid node.
     * @return Shannon entropy in bits (log₂ base).  Returns 0.0f if grid empty.
     *
     * Complexity: O(N) to collect active nodes, O(K) for entropy sum.
     */
    [[nodiscard]]
    float estimate(std::span<const float> psi_real,
                   std::span<const float> psi_imag) {
        const std::size_t N = std::min(psi_real.size(), psi_imag.size());

        // 1. Total energy
        float total_energy = 0.0f;
        for (std::size_t i = 0; i < N; ++i) {
            float r = psi_real[i], im = psi_imag[i];
            total_energy += r*r + im*im;   // |Ψ|²
        }
        if (total_energy < 1e-10f) return 0.0f;

        // 2. Collect active indices
        active_.clear();
        for (std::size_t i = 0; i < N; ++i) {
            float r = psi_real[i], im = psi_imag[i];
            if (r*r + im*im > ENTROPY_ACTIVE_THRESH)
                active_.push_back(i);
        }
        if (active_.empty()) return 0.0f;

        // 3. Subsample
        std::shuffle(active_.begin(), active_.end(), rng_);
        const int K = std::min(ENTROPY_SAMPLE_SIZE,
                               static_cast<int>(active_.size()));

        // 4. Shannon entropy
        float entropy = 0.0f;
        for (int k = 0; k < K; ++k) {
            std::size_t idx = active_[k];
            float r = psi_real[idx], im = psi_imag[idx];
            float p = (r*r + im*im) / total_energy;
            if (p > 1e-10f)
                entropy -= p * std::log2(p);
        }
        return entropy;
    }

    /**
     * @brief Convenience overload: single interleaved intensity array.
     *
     * @param intensities  Pre-computed |Ψ|² values per node.
     */
    [[nodiscard]]
    float estimate_from_intensities(std::span<const float> intensities) {
        const std::size_t N = intensities.size();

        float total = 0.0f;
        for (float v : intensities) total += v;
        if (total < 1e-10f) return 0.0f;

        active_.clear();
        for (std::size_t i = 0; i < N; ++i)
            if (intensities[i] > ENTROPY_ACTIVE_THRESH)
                active_.push_back(i);

        if (active_.empty()) return 0.0f;

        std::shuffle(active_.begin(), active_.end(), rng_);
        const int K = std::min(ENTROPY_SAMPLE_SIZE,
                               static_cast<int>(active_.size()));

        float entropy = 0.0f;
        for (int k = 0; k < K; ++k) {
            float p = intensities[active_[k]] / total;
            if (p > 1e-10f)
                entropy -= p * std::log2(p);
        }
        return entropy;
    }

private:
    std::mt19937            rng_;
    std::vector<std::size_t> active_;  // reused buffer — no heap allocation per call
};

// ── BoredomRegulator ──────────────────────────────────────────────────────────

/**
 * @class BoredomRegulator
 * @brief Drives spontaneous exploratory behavior from entropy deficit.
 *
 * @par Phase 49 — AUTO-04 Boredom Singularity Fix (spec §6.2)
 *
 * Replaces the naive linear formula with bounded sigmoidal accumulation:
 *
 *   ΔB(t) = α_acc · (1 − tanh(k · H(Ψ))) · dt
 *
 * Properties:
 *   H → 0  :  tanh(0) = 0  →  ΔB = α_acc · dt   (max finite rate — no singularity)
 *   H → ∞  :  tanh(∞) = 1  →  ΔB = 0            (high entropy → no accumulation)
 *
 * A separate passive decay term (decay_rate · dt) lets boredom drain back
 * to 0 once exploration restores entropy above the target.
 *
 * Integration pattern:
 * @code
 *   BoredomRegulator bored;  // α_acc=0.1, k=0.5, decay=0.01
 *   EntropyEstimator est;
 *   float H = est.estimate(psi_r, psi_i);
 *   bored.update(H, dt);
 *   if (bored.should_explore()) { ... inject_noise_or_new_goal(); ... }
 * @endcode
 */
class BoredomRegulator {
public:
    /// Phase 49: tunable — defaults match spec §6.2 constants.
    explicit BoredomRegulator(float alpha_acc  = BOREDOM_ALPHA_ACC,
                              float k          = BOREDOM_K,
                              float decay_rate = BOREDOM_DECAY_RATE) noexcept
        : alpha_acc_(alpha_acc), k_(k), decay_rate_(decay_rate) {}

    /**
     * @brief Advance boredom using AUTO-04 sigmoidal formula.
     *
     *   ΔB = α_acc · (1 − tanh(k · H)) · dt   — bounded accumulation
     *      − decay_rate · dt                   — passive drain
     *
     * @param entropy  Shannon entropy from EntropyEstimator (bits).
     * @param dt       Elapsed seconds.
     */
    void update(float entropy, float dt) noexcept {
        // Phase 49: AUTO-04 sigmoidal accumulation (spec §6.2)
        last_delta_b_ = alpha_acc_ * (1.0f - std::tanh(k_ * entropy));
        boredom_ += last_delta_b_ * dt;   // bounded: never infinite at H=0
        boredom_ -= decay_rate_   * dt;   // passive drain when entropy is high
        boredom_  = std::clamp(boredom_, 0.0f, 1.0f);
        last_entropy_ = entropy;
    }

    /// True when boredom > θ_explore (spec §6.3: 0.8).
    [[nodiscard]] bool  should_explore() const noexcept {
        return boredom_ > BOREDOM_EXPLORE_THRESH;
    }

    [[nodiscard]] float level()        const noexcept { return boredom_; }
    [[nodiscard]] float last_entropy() const noexcept { return last_entropy_; }

    /// Phase 49 telemetry: last ΔB/dt = α_acc·(1−tanh(k·H)) ∈ [0, α_acc].
    [[nodiscard]] float last_delta_b() const noexcept { return last_delta_b_; }

    /// Phase 49 param accessors (for tests / telemetry).
    [[nodiscard]] float alpha_acc()   const noexcept { return alpha_acc_; }
    [[nodiscard]] float k_param()     const noexcept { return k_; }
    [[nodiscard]] float decay_rate()  const noexcept { return decay_rate_; }

    /// Phase 50: directly subtract from boredom (clamped to 0). Used by
    /// AutonomyEngine to drain boredom after emitting a CuriosityGoal, preventing
    /// immediate re-fire (early Mania Loop guard, spec §9.2).
    void drain(float amount) noexcept {
        boredom_ = std::max(0.0f, boredom_ - amount);
    }

    void reset() noexcept {
        boredom_      = 0.0f;
        last_entropy_ = 0.0f;
        last_delta_b_ = 0.0f;
    }

private:
    float alpha_acc_    = BOREDOM_ALPHA_ACC;
    float k_            = BOREDOM_K;
    float decay_rate_   = BOREDOM_DECAY_RATE;
    float boredom_      = 0.0f;
    float last_entropy_ = 0.0f;
    float last_delta_b_ = 0.0f;   ///< Phase 49 telemetry
};

} // namespace nikola::autonomy
