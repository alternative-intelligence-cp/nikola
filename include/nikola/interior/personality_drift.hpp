/**
 * @file interior/personality_drift.hpp
 * @brief v0.2.3 Phase 2 — PersonalityDrift: trait axes that evolve based
 *        on accumulated experience outcomes.
 *
 * Trait axes (bipolar dimensions):
 *   CURIOUS   ↔ FOCUSED      — breadth vs depth of inquiry
 *   CAUTIOUS  ↔ BOLD         — risk tolerance in decisions
 *   VERBOSE   ↔ TERSE        — communication style
 *   PATIENT   ↔ URGENT       — time horizon for goals
 *   ANALYTICAL ↔ INTUITIVE   — reasoning style preference
 *
 * Each trait is a float ∈ [-1.0, +1.0]:
 *   -1.0 = fully left pole (e.g., maximally curious)
 *   +1.0 = fully right pole (e.g., maximally focused)
 *    0.0 = balanced / neutral
 *
 * Drift mechanics:
 *   - Outcomes (success/failure + context) shift traits toward the pole
 *     that was active when the outcome occurred.
 *   - Drift rate bounded: max DRIFT_PER_EVENT per event, max DRIFT_PER_EPOCH total.
 *   - Homeostatic decay: extreme positions slowly regress toward 0
 *     (prevents permanent personality lockup).
 *
 * Influence:
 *   - trait_multiplier(TraitAxis, action_type) → float ∈ [0.5, 1.5]
 *     used as a scoring multiplier in the DecisionLoop.
 *
 * Phase: NIK-PERS-02 (PersonalityDrift, v0.2.3)
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>

namespace nikola::interior {

// ============================================================================
// TraitAxis — bipolar personality dimensions
// ============================================================================

enum class TraitAxis : uint8_t {
    CURIOUS_FOCUSED     = 0,  ///< -1=curious, +1=focused
    CAUTIOUS_BOLD       = 1,  ///< -1=cautious, +1=bold
    VERBOSE_TERSE       = 2,  ///< -1=verbose, +1=terse
    PATIENT_URGENT      = 3,  ///< -1=patient, +1=urgent
    ANALYTICAL_INTUITIVE = 4, ///< -1=analytical, +1=intuitive
    COUNT               = 5
};

[[nodiscard]] inline const char* trait_axis_name(TraitAxis t) noexcept {
    switch (t) {
        case TraitAxis::CURIOUS_FOCUSED:      return "CURIOUS_FOCUSED";
        case TraitAxis::CAUTIOUS_BOLD:        return "CAUTIOUS_BOLD";
        case TraitAxis::VERBOSE_TERSE:        return "VERBOSE_TERSE";
        case TraitAxis::PATIENT_URGENT:       return "PATIENT_URGENT";
        case TraitAxis::ANALYTICAL_INTUITIVE: return "ANALYTICAL_INTUITIVE";
        default:                              return "UNKNOWN";
    }
}

/// Human-readable label for current pole (e.g., "slightly cautious", "very bold")
[[nodiscard]] inline std::string trait_description(TraitAxis t, float value) {
    const char* left = "";
    const char* right = "";
    switch (t) {
        case TraitAxis::CURIOUS_FOCUSED:      left = "curious";    right = "focused";    break;
        case TraitAxis::CAUTIOUS_BOLD:        left = "cautious";   right = "bold";       break;
        case TraitAxis::VERBOSE_TERSE:        left = "verbose";    right = "terse";      break;
        case TraitAxis::PATIENT_URGENT:       left = "patient";    right = "urgent";     break;
        case TraitAxis::ANALYTICAL_INTUITIVE: left = "analytical"; right = "intuitive";  break;
        default:                              left = "?";          right = "?";          break;
    }

    const char* intensity;
    float av = std::abs(value);
    if (av < 0.15f) intensity = "balanced";
    else if (av < 0.4f) intensity = "slightly";
    else if (av < 0.7f) intensity = "moderately";
    else intensity = "very";

    if (av < 0.15f) {
        return std::string(intensity) + " " + left + "/" + right;
    }
    return std::string(intensity) + " " + (value < 0 ? left : right);
}

// ============================================================================
// PersonalityDriftConfig
// ============================================================================

struct PersonalityDriftConfig {
    /// Maximum trait shift per single event.
    float drift_per_event = 0.02f;

    /// Maximum total trait shift per epoch (decay period).
    float drift_per_epoch = 0.10f;

    /// Homeostatic decay rate per second: extreme positions regress toward 0.
    float homeostatic_decay_rate = 0.0005f;

    /// Minimum absolute trait value below which trait is considered "balanced".
    float balanced_threshold = 0.15f;
};

// ============================================================================
// ExperienceOutcome — what happened and in what context
// ============================================================================

struct ExperienceOutcome {
    float success;       ///< ∈ [-1, +1]: -1=failure, 0=neutral, +1=success
    int   action_type;   ///< ActionType enum value that produced this outcome
    float risk_taken;    ///< How risky was the action? ∈ [0, 1]
    float complexity;    ///< How complex was the task? ∈ [0, 1]
};

// ============================================================================
// PersonalitySnapshot
// ============================================================================

struct PersonalitySnapshot {
    std::array<float, static_cast<size_t>(TraitAxis::COUNT)> traits;
    uint64_t total_events = 0;
};

// ============================================================================
// PersonalityDrift
// ============================================================================

class PersonalityDrift {
public:
    static constexpr size_t N_TRAITS = static_cast<size_t>(TraitAxis::COUNT);

    explicit PersonalityDrift(PersonalityDriftConfig cfg = {})
        : cfg_(cfg) {
        traits_.fill(0.0f);
        epoch_drift_.fill(0.0f);
    }

    // ── Core drift ──────────────────────────────────────────────────────

    /**
     * @brief Apply experience outcome to personality traits.
     *
     * Rules:
     *   - Success with bold action   → drift CAUTIOUS_BOLD toward +1 (bold)
     *   - Failure with bold action   → drift CAUTIOUS_BOLD toward -1 (cautious)
     *   - Success with EXPLORE       → drift CURIOUS_FOCUSED toward -1 (curious)
     *   - Success with REASON        → drift ANALYTICAL_INTUITIVE toward -1 (analytical)
     *   - Success with GENERATE_CODE → drift ANALYTICAL_INTUITIVE toward -1 (analytical)
     *   - Complex task success       → drift PATIENT_URGENT toward -1 (patient)
     *   - Quick simple success       → drift PATIENT_URGENT toward +1 (urgent)
     *   - EMIT_THOUGHT chosen often  → drift VERBOSE_TERSE toward -1 (verbose)
     */
    void apply_outcome(const ExperienceOutcome& outcome);

    /**
     * @brief Homeostatic decay: extreme positions regress toward 0.
     * @param dt  Elapsed seconds.
     */
    void decay(float dt);

    /**
     * @brief Reset epoch drift accumulators (call at epoch boundaries, e.g., NAP).
     */
    void reset_epoch();

    // ── Influence on scoring ─────────────────────────────────────────────

    /**
     * @brief Compute a scoring multiplier for an action based on personality.
     *
     * Returns a float ∈ [0.7, 1.3] that the DecisionLoop can multiply
     * against the base score for the given action type.
     *
     * @param action_type  ActionType enum value.
     * @return Multiplier — >1.0 means personality favors this action.
     */
    [[nodiscard]] float action_multiplier(int action_type) const;

    // ── Accessors ────────────────────────────────────────────────────────

    [[nodiscard]] float trait(TraitAxis axis) const {
        return traits_[static_cast<size_t>(axis)];
    }

    void set_trait(TraitAxis axis, float value) {
        traits_[static_cast<size_t>(axis)] = std::clamp(value, -1.0f, 1.0f);
    }

    [[nodiscard]] PersonalitySnapshot snapshot() const {
        PersonalitySnapshot s;
        s.traits = traits_;
        s.total_events = total_events_;
        return s;
    }

    [[nodiscard]] uint64_t total_events() const noexcept { return total_events_; }

    /**
     * @brief Human-readable personality summary.
     */
    [[nodiscard]] std::string describe() const;

    // ── Persistence ──────────────────────────────────────────────────────

    [[nodiscard]] std::string to_json() const;
    bool from_json(const std::string& json);

    // ── Config ───────────────────────────────────────────────────────────

    [[nodiscard]] const PersonalityDriftConfig& config() const noexcept { return cfg_; }

private:
    PersonalityDriftConfig cfg_;
    std::array<float, N_TRAITS> traits_;
    std::array<float, N_TRAITS> epoch_drift_;  ///< Accumulated drift this epoch
    uint64_t total_events_ = 0;

    /// Apply bounded drift to a single trait axis.
    void drift_trait(TraitAxis axis, float direction);
};

} // namespace nikola::interior
