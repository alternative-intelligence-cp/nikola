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
        float r = static_cast<float>(static_cast<int8_t>(reward));
        float td_error = r + DOPAMINE_GAMMA * current_value_ - prev_value_;

        // Dopamine encodes the prediction error, clamped to [0, 1]
        dopamine_ = std::clamp(DOPAMINE_BASELINE + td_error, 0.0f, 1.0f);

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

    // ── observers ────────────────────────────────────────────────────────────

    /// Current dopamine level ∈ [0, 1].
    [[nodiscard]] float level()      const noexcept { return dopamine_; }

    /// True when dopamine is above baseline (reward surprise).
    [[nodiscard]] bool  is_spiking() const noexcept { return dopamine_ > DOPAMINE_BASELINE; }

    /// True when dopamine is below baseline (punishment surprise).
    [[nodiscard]] bool  is_dipping() const noexcept { return dopamine_ < DOPAMINE_BASELINE; }

    /// Most recent TD error (for telemetry).
    [[nodiscard]] float last_td_error() const noexcept {
        return dopamine_ - DOPAMINE_BASELINE;
    }

    /// Reset to factory state.
    void reset() noexcept {
        dopamine_      = DOPAMINE_BASELINE;
        prev_value_    = 0.0f;
        current_value_ = 0.0f;
    }

private:
    float dopamine_      = DOPAMINE_BASELINE;
    float prev_value_    = 0.0f;
    float current_value_ = 0.0f;
};

} // namespace nikola::autonomy
