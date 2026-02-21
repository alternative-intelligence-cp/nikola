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
 * BoredomRegulator:
 *   Low entropy  → boredom rises  → triggers exploration when > 0.7
 *   High entropy → boredom falls
 *   Target range: H ∈ [4, 8] for healthy cognitive state.
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

/// Boredom rise / fall rate per second of simulation.
inline constexpr float BOREDOM_RATE           = 0.1f;

/// Threshold above which the system enters spontaneous exploration.
inline constexpr float BOREDOM_EXPLORE_THRESH = 0.7f;

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
 * Integration pattern:
 * @code
 *   BoredomRegulator bored;
 *   EntropyEstimator est;
 *
 *   // per-frame
 *   float H  = est.estimate(psi_r, psi_i);
 *   bored.update(H, dt);
 *   if (bored.should_explore()) { ... inject_noise_or_new_goal(); ... }
 * @endcode
 */
class BoredomRegulator {
public:
    BoredomRegulator() = default;

    /**
     * @brief Advance boredom toward or away from threshold.
     *
     * @param entropy  Shannon entropy from EntropyEstimator (bits).
     * @param dt       Elapsed seconds.
     */
    void update(float entropy, float dt) noexcept {
        if (entropy < ENTROPY_TARGET)
            boredom_ += BOREDOM_RATE * dt;   // getting bored
        else
            boredom_ -= BOREDOM_RATE * dt;   // engaged

        boredom_ = std::clamp(boredom_, 0.0f, 1.0f);
        last_entropy_ = entropy;
    }

    /// True when sufficiently bored to warrant spontaneous action.
    [[nodiscard]] bool  should_explore() const noexcept {
        return boredom_ > BOREDOM_EXPLORE_THRESH;
    }

    [[nodiscard]] float level()        const noexcept { return boredom_; }
    [[nodiscard]] float last_entropy() const noexcept { return last_entropy_; }

    void reset() noexcept { boredom_ = 0.0f; last_entropy_ = 0.0f; }

private:
    float boredom_      = 0.0f;
    float last_entropy_ = 0.0f;
};

} // namespace nikola::autonomy
