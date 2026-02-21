/**
 * @file autonomy/nap_controller.hpp
 * @brief Gap 5.4 — ATP hysteresis nap cycle: enter/exit/timeout management.
 *
 * Hysteresis prevents oscillation ("flapping"):
 *   Enter nap:  ATP < 0.15
 *   Exit  nap:  ATP > 0.90
 *   Min  duration: (0.90 - 0.15) / 0.05 ≈ 15 seconds
 *   Max  duration: 60 seconds (emergency arousal)
 *
 * During nap:
 *   - Physics loop continues at reduced rate (DreamWeaveEngine takes over)
 *   - MetabolicSimulator::recharge() called every step
 *   - No external I/O processed
 *
 * Callback hooks allow the Orchestrator to suspend/resume subsystems.
 */

#pragma once

#include <algorithm>   // std::clamp
#include <functional>
#include <string>

namespace nikola::autonomy {

// ── Gap 5.4 constants ────────────────────────────────────────────────────────

/// ATP level that triggers nap entry.  Spec: 0.15.
inline constexpr float NAP_ENTER_THRESHOLD   = 0.15f;

/// ATP level required to exit nap.  Spec: 0.90.
inline constexpr float NAP_EXIT_THRESHOLD    = 0.90f;

/// Hard ceiling on nap duration (seconds). Spec: 60 s.
inline constexpr float NAP_MAX_DURATION_SEC  = 60.0f;

/// Expected minimum nap duration given regen rate: (0.90-0.15)/0.05 ≈ 15 s.
inline constexpr float NAP_MIN_DURATION_SEC  = 15.0f;

// ── NapState ──────────────────────────────────────────────────────────────────

enum class NapState : uint8_t {
    AWAKE,     ///< Normal operation
    NAPPING,   ///< ATP recharge + Dream-Weave consolidation
};

[[nodiscard]] inline const char* nap_state_name(NapState s) noexcept {
    switch (s) {
        case NapState::AWAKE:   return "AWAKE";
        case NapState::NAPPING: return "NAPPING";
    }
    return "UNKNOWN";
}

// ── NapController ─────────────────────────────────────────────────────────────

/**
 * @class NapController
 * @brief Drives nap lifecycle driven by ATP hysteresis.
 *
 * The controller is time-driven: call `update()` every simulation tick.
 * It does NOT own a MetabolicSimulator; the caller owns the ATP and passes it in.
 *
 * Callbacks:
 *   on_enter_nap  — called once when AWAKE → NAPPING
 *   on_exit_nap   — called once when NAPPING → AWAKE
 *   on_nap_tick   — called every tick while napping (for DreamWeave step)
 *
 * Usage:
 * @code
 *   NapController nap;
 *   nap.on_enter_nap = [&]{ orchestrator.suspend_io(); };
 *   nap.on_exit_nap  = [&]{ orchestrator.resume_io();  };
 *   nap.on_nap_tick  = [&](float t){ dream_weave.step(t); };
 *
 *   // Per second:
 *   meta.recharge(dt);          // MetabolicSimulator
 *   nap.update(meta.atp(), t);  // NapController
 * @endcode
 */
class NapController {
public:
    // ── callbacks ────────────────────────────────────────────────────────────

    std::function<void()>        on_enter_nap;           ///< AWAKE → NAPPING
    std::function<void()>        on_exit_nap;            ///< NAPPING → AWAKE
    std::function<void(float)>  on_nap_tick;             ///< called while napping; arg = elapsed_nap_s

    NapController() = default;

    // ── primary interface ─────────────────────────────────────────────────

    /**
     * @brief Drive the nap state machine.
     *
     * @param atp_level     Current normalized ATP ∈ [0, 1].
     * @param current_time  Monotonic wall-clock time (seconds, any epoch).
     */
    void update(float atp_level, float current_time) {
        switch (state_) {
            case NapState::AWAKE:
                if (atp_level < NAP_ENTER_THRESHOLD) {
                    enter_nap_(current_time);
                }
                break;

            case NapState::NAPPING: {
                float elapsed = current_time - nap_start_;
                bool recharged = (atp_level >= NAP_EXIT_THRESHOLD);
                bool timed_out = (elapsed  >= NAP_MAX_DURATION_SEC);

                if (on_nap_tick) on_nap_tick(elapsed);

                if (recharged || timed_out) {
                    last_nap_duration_ = elapsed;
                    last_exit_reason_  = timed_out ? "TIMEOUT" : "RECHARGED";
                    exit_nap_();
                }
                break;
            }
        }
    }

    // ── observers ────────────────────────────────────────────────────────────

    [[nodiscard]] NapState    state()              const noexcept { return state_; }
    [[nodiscard]] bool        is_napping()         const noexcept { return state_ == NapState::NAPPING; }
    [[nodiscard]] float       nap_start_time()     const noexcept { return nap_start_; }
    [[nodiscard]] float       last_nap_duration()  const noexcept { return last_nap_duration_; }
    [[nodiscard]] const char* last_exit_reason()   const noexcept { return last_exit_reason_; }
    [[nodiscard]] uint32_t    nap_count()          const noexcept { return nap_count_; }

    /// Current elapsed nap time (0 if AWAKE).
    [[nodiscard]] float current_nap_elapsed(float current_time) const noexcept {
        if (state_ != NapState::NAPPING) return 0.0f;
        return current_time - nap_start_;
    }

private:
    NapState    state_             = NapState::AWAKE;
    float       nap_start_         = 0.0f;
    float       last_nap_duration_ = 0.0f;
    uint32_t    nap_count_         = 0u;
    const char* last_exit_reason_  = "";

    void enter_nap_(float t) {
        state_     = NapState::NAPPING;
        nap_start_ = t;
        ++nap_count_;
        if (on_enter_nap) on_enter_nap();
    }

    void exit_nap_() {
        state_ = NapState::AWAKE;
        if (on_exit_nap) on_exit_nap();
    }
};

} // namespace nikola::autonomy
