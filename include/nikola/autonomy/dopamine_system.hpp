/**
 * @file autonomy/dopamine_system.hpp
 * @brief Gap 5.1 — Temporal Difference (TD) dopamine prediction-error signal.
 *
 * Implements:
 *   δ_t = (R_t + γ·V(S_{t+1})) - V(S_t)
 *
 * Where V(S) = total system energy (Σ|Ψ|²) and R_t ∈ {-1, 0, +1}.
 *
 * Interpretation:
 *   D > 0.5  → better than expected → spike (increase learning rate)
 *   D < 0.5  → worse than expected  → dip (suppress, explore)
 *   D = 0.5  → no surprise          → baseline (maintain policy)
 */

#pragma once

#include <algorithm>  // std::clamp
#include <cstdint>

namespace nikola::autonomy {

// ── Gap 5.1 constants ────────────────────────────────────────────────────────

/// TD discount factor γ.  Matches spec (0.95).
inline constexpr float DOPAMINE_GAMMA         = 0.95f;

/// Baseline dopamine level (no-surprise state).
inline constexpr float DOPAMINE_BASELINE      = 0.5f;

/// Hebbian learning rate modifier multiplied against the TD error.
inline constexpr float DOPAMINE_LEARNING_RATE = 0.01f;

/// Decay time constant τ (seconds) — exponential drift back to baseline.
inline constexpr float DOPAMINE_TAU_SEC       = 2.0f;

// ── Phase 29: Neural habituation constants ─────────────────────────────────

/// EMA smoothing factor α for familiar_td_ (slow adaptation).
inline constexpr float NOVELTY_EMA_ALPHA    = 0.03f;
/// TD deviation below which a signal is considered "familiar".
inline constexpr float NOVELTY_THRESHOLD    = 0.03f;
/// Per-update multiplicative decay of novelty_factor_ on familiar signals.
inline constexpr float NOVELTY_DECAY        = 0.985f;
/// Per-update additive recovery rate of novelty_factor_ on surprising signals.
inline constexpr float NOVELTY_RECOVERY     = 0.15f;

// ── Reward signal ─────────────────────────────────────────────────────────────

/**
 * @brief External reward signal injected by the CLI / evaluator.
 *
 * Positive | negative | neutral.  Semantically mirrors biological
 * dopamine reward: positive = "better than expected", negative = "worse".
 */
enum class Reward : int8_t {
    NEGATIVE = -1,
    NEUTRAL  =  0,
    POSITIVE = +1,
};

// ── DopamineSystem ────────────────────────────────────────────────────────────

/**
 * @class DopamineSystem
 * @brief Encodes prediction errors as a continuous dopamine level ∈ [0, 1].
 *
 * Thread-safe: no mutable state shared across threads.
 * NOT thread-safe for concurrent writes — serialize externally if needed.
 *
 * Usage:
 * @code
 *   DopamineSystem dopa;
 *   dopa.update(engine.total_energy(), Reward::POSITIVE);
 *   float d = dopa.level();  // [0, 1]
 *   dopa.decay(dt_seconds);
 * @endcode
 */
class DopamineSystem {
public:
    DopamineSystem() = default;

    // ── primary interface ─────────────────────────────────────────────────

    /**
     * @brief Update dopamine based on TD error this timestep.
     *
     * @param total_energy   Σ|Ψ|² from the current physics state.
     * @param reward         External reward signal (+1 / 0 / -1).
     */
    void update(float total_energy, Reward reward = Reward::NEUTRAL) noexcept {
        current_value_ = total_energy;

        // δ_t = R_t + γ·V(S_{t+1}) - V(S_t)
        const float r      = static_cast<float>(static_cast<int8_t>(reward));
        const float td_raw = r + DOPAMINE_GAMMA * current_value_ - prev_value_;

        // Phase 29: neural habituation ─────────────────────────────────────
        // Update familiar_td_ (EMA of raw td_error).
        familiar_td_ = (1.0f - NOVELTY_EMA_ALPHA) * familiar_td_
                     + NOVELTY_EMA_ALPHA * td_raw;

        // Reward/punishment signals always fire at full strength.
        // Only neutral field fluctuations are habituatable.
        if (reward == Reward::NEUTRAL) {
            const float surprise = std::abs(td_raw - familiar_td_);
            if (surprise < NOVELTY_THRESHOLD) {
                novelty_factor_ *= NOVELTY_DECAY;          // habituate
            } else {
                novelty_factor_ += (1.0f - novelty_factor_) * NOVELTY_RECOVERY; // recover
            }
            novelty_factor_ = std::clamp(novelty_factor_, 0.0f, 1.0f);
        }

        // Effective TD: attenuated by habituation for neutral signals.
        const float nf        = (reward == Reward::NEUTRAL) ? novelty_factor_ : 1.0f;
        const float td_effective = td_raw * nf;

        // Dopamine encodes the (possibly habituated) prediction error, clamped to [0, 1]
        dopamine_ = std::clamp(DOPAMINE_BASELINE + td_effective, 0.0f, 1.0f);

        prev_value_ = current_value_;
    }

    /**
     * @brief Passive decay back to baseline.
     * @param dt  Elapsed seconds since last decay call.
     */
    void decay(float dt) noexcept {
        // Exponential drift: dD/dt = (baseline - D) / τ
        dopamine_ += (DOPAMINE_BASELINE - dopamine_) * dt / DOPAMINE_TAU_SEC;
        dopamine_ = std::clamp(dopamine_, 0.0f, 1.0f);
    }

    /**
     * @brief Additive nudge from external coupling (GAP-005 cross-coupling matrix).
     *
     * Applies a signed delta directly to the dopamine level and clamps to [0, 1].
     * Unlike update(), this bypasses TD error computation — it is only for
     * inter-neurochemical coupling terms (M·N cross-coupling step).
     *
     * @param delta  Signed change in dopamine (may be positive or negative).
     */
    void adjust(float delta) noexcept {
        dopamine_ = std::clamp(dopamine_ + delta, 0.0f, 1.0f);
    }

    // ── observers ────────────────────────────────────────────────────────────

    /// Current dopamine level ∈ [0, 1].
    [[nodiscard]] float level()      const noexcept { return dopamine_; }

    /// True when dopamine is above baseline (reward surprise).
    [[nodiscard]] bool  is_spiking() const noexcept { return dopamine_ > DOPAMINE_BASELINE; }

    /// True when dopamine is below baseline (punishment surprise).
    [[nodiscard]] bool  is_dipping() const noexcept { return dopamine_ < DOPAMINE_BASELINE; }

    /// Most recent (habituated) TD error (for telemetry).
    [[nodiscard]] float last_td_error() const noexcept {
        return dopamine_ - DOPAMINE_BASELINE;
    }

    /// Current novelty attenuation scalar ∈ [0, 1].  1 = fully novel, ~0 = habituated.
    [[nodiscard]] float novelty_factor() const noexcept { return novelty_factor_; }

    /// EMA of recent raw TD errors — the "familiar" baseline.
    [[nodiscard]] float familiar_td() const noexcept { return familiar_td_; }

    /// Reset to factory state.
    void reset() noexcept {
        dopamine_       = DOPAMINE_BASELINE;
        prev_value_     = 0.0f;
        current_value_  = 0.0f;
        familiar_td_    = 0.0f;
        novelty_factor_ = 1.0f;
    }

private:
    float dopamine_       = DOPAMINE_BASELINE;
    float prev_value_     = 0.0f;
    float current_value_  = 0.0f;
    float familiar_td_    = 0.0f;    ///< Phase 29: EMA of raw td_errors
    float novelty_factor_ = 1.0f;   ///< Phase 29: habituation scalar ∈ [0, 1]
};

} // namespace nikola::autonomy
