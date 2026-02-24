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

// ── GAP-036: Time-domain logistic calibration (10-minute exploration cycle) ──

/// Slope of the time-domain logistic: k = 0.00656 s⁻¹.
/// Derived from boundary conditions (spec §GAP-036):
///   B(0s) ≈ 0.10 (right after novelty)
///   B(600s) ≈ 0.85 (trigger after 10 idle minutes)
/// Solution: k·T_half = ln(9) ≈ 2.197, 600k − 2.197 = 1.737 → k ≈ 0.00656.
inline constexpr float BOREDOM_K_SEC         = 0.00656f;

/// Inflection point of the time-domain logistic (B = 0.5 at this age in seconds).
/// T_half = ln(9) / k ≈ 335 s.
inline constexpr float BOREDOM_T_HALF_SEC    = 335.0f;

/// k scaled to per-tick for a 1000 Hz physics loop: k_tick = k_sec / 1000.
/// At 1 kHz a tick is 1 ms; in 600 000 ticks (= 600 s) boredom reaches 0.85.
inline constexpr float BOREDOM_K_TICK        = 6.56e-6f;

/// T_half in ticks for a 1000 Hz physics loop: 335 s × 1000 Hz = 335 000 ticks.
inline constexpr float BOREDOM_T_HALF_TICKS  = 335'000.0f;

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
    /// @param time_domain_mode  Phase 53 / GAP-036: when true, use the
    ///   calibrated time-domain logistic B(t)=σ(k·(elapsed−T_half)) instead
    ///   of the Phase 49 entropy-driven ΔB formula.  Default false preserves
    ///   all previous behaviour.
    explicit BoredomRegulator(float alpha_acc       = BOREDOM_ALPHA_ACC,
                              float k               = BOREDOM_K,
                              float decay_rate      = BOREDOM_DECAY_RATE,
                              bool  time_domain_mode = false) noexcept
        : alpha_acc_(alpha_acc), k_(k), decay_rate_(decay_rate)
        , time_domain_mode_(time_domain_mode)
    {
        if (time_domain_mode_) {
            // Pre-compute logistic at elapsed=0: B = 1/(1+exp(k·T_half)).
            // With the spec-calibrated params this evaluates to ≈ 0.10.
            boredom_ = 1.0f / (1.0f + std::exp(BOREDOM_K_SEC * BOREDOM_T_HALF_SEC));
        }
    }

    /**
     * @brief Advance boredom one timestep.
     *
     * **Phase 49 mode** (default, time_domain_mode = false):
     *   ΔB = α_acc · (1 − tanh(k · H)) · dt   — bounded accumulation
     *      − decay_rate · dt                   — passive drain
     *
     * **Phase 53 / GAP-036 mode** (time_domain_mode = true):
     *   Tracks elapsed seconds since last novelty event.
     *   Entropy ≥ ENTROPY_TARGET is treated as novelty and resets the timer.
     *   B(elapsed) = 1 / (1 + exp(−k_sec · (elapsed − T_half_sec)))
     *   where k_sec = BOREDOM_K_SEC = 0.00656, T_half_sec = 335 s.
     *   Boundary conditions (spec §GAP-036): B(0)≈0.10, B(600)≈0.85.
     *
     * @param entropy  Shannon entropy from EntropyEstimator (bits).
     * @param dt       Elapsed seconds.
     */
    void update(float entropy, float dt) noexcept {
        last_entropy_ = entropy;
        if (time_domain_mode_) {
            // --- Phase 53: GAP-036 time-domain logistic ---
            if (entropy >= ENTROPY_TARGET) {
                // Novelty detected: reset elapsed counter.
                elapsed_s_ = 0.0f;
            } else {
                elapsed_s_ += dt;
            }
            boredom_      = 1.0f / (1.0f + std::exp(-BOREDOM_K_SEC
                                    * (elapsed_s_ - BOREDOM_T_HALF_SEC)));
            // Instantaneous rate = logistic derivative = k·B·(1−B).
            last_delta_b_ = BOREDOM_K_SEC * boredom_ * (1.0f - boredom_);
        } else {
            // --- Phase 49: AUTO-04 sigmoidal accumulation (spec §6.2) ---
            last_delta_b_ = alpha_acc_ * (1.0f - std::tanh(k_ * entropy));
            boredom_ += last_delta_b_ * dt;   // bounded: never infinite at H=0
            boredom_ -= decay_rate_   * dt;   // passive drain when entropy is high
            boredom_  = std::clamp(boredom_, 0.0f, 1.0f);
        }
    }

    /// True when boredom > θ_explore (spec §6.3: 0.8).
    [[nodiscard]] bool  should_explore() const noexcept {
        return boredom_ > BOREDOM_EXPLORE_THRESH;
    }

    [[nodiscard]] float level()        const noexcept { return boredom_; }
    [[nodiscard]] float last_entropy() const noexcept { return last_entropy_; }

    /// Phase 49 telemetry: last ΔB/dt = α_acc·(1−tanh(k·H)) ∈ [0, α_acc].
    /// Phase 53 mode: instantaneous logistic derivative k·B·(1−B).
    [[nodiscard]] float last_delta_b() const noexcept { return last_delta_b_; }

    /// Phase 49 param accessors (for tests / telemetry).
    [[nodiscard]] float alpha_acc()   const noexcept { return alpha_acc_; }
    [[nodiscard]] float k_param()     const noexcept { return k_; }
    [[nodiscard]] float decay_rate()  const noexcept { return decay_rate_; }

    /// Phase 53 / GAP-036 accessors.
    [[nodiscard]] bool  is_time_domain_mode()         const noexcept { return time_domain_mode_; }
    [[nodiscard]] float elapsed_since_novelty_s()     const noexcept { return elapsed_s_; }

    /// Phase 50: directly subtract from boredom (clamped to 0). Used by
    /// AutonomyEngine to drain boredom after emitting a CuriosityGoal, preventing
    /// immediate re-fire (early Mania Loop guard, spec §9.2).
    /// In time-domain mode, drain resets the novelty counter to zero.
    void drain(float amount) noexcept {
        if (time_domain_mode_) {
            elapsed_s_ = 0.0f;   // exploration event = novelty reset
            boredom_   = 1.0f / (1.0f + std::exp(BOREDOM_K_SEC * BOREDOM_T_HALF_SEC));
        } else {
            boredom_ = std::max(0.0f, boredom_ - amount);
        }
    }

    void reset() noexcept {
        if (time_domain_mode_) {
            boredom_ = 1.0f / (1.0f + std::exp(BOREDOM_K_SEC * BOREDOM_T_HALF_SEC));
        } else {
            boredom_ = 0.0f;
        }
        last_entropy_ = 0.0f;
        last_delta_b_ = 0.0f;
        elapsed_s_    = 0.0f;
    }

private:
    float alpha_acc_       = BOREDOM_ALPHA_ACC;
    float k_               = BOREDOM_K;
    float decay_rate_      = BOREDOM_DECAY_RATE;
    bool  time_domain_mode_ = false;   ///< Phase 53: GAP-036 logistic mode

    float boredom_      = 0.0f;
    float last_entropy_ = 0.0f;
    float last_delta_b_ = 0.0f;   ///< Phase 49 / 53 telemetry
    float elapsed_s_    = 0.0f;   ///< Phase 53: seconds since last novelty
};

} // namespace nikola::autonomy
