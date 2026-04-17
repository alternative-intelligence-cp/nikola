/**
 * @file interior/preference_engine.hpp
 * @brief v0.2.3 Phase 1 — PreferenceEngine: domain-categorized preference
 *        learning with decay and behavioral influence on action scoring.
 *
 * Preference domains:
 *   TOPICS            — what subjects Nikola gravitates toward
 *   CODE_PATTERNS     — coding style preferences (functional, OOP, terse, etc.)
 *   INTERACTION_STYLES — how Nikola prefers to communicate
 *   DATA_SOURCES      — preferred types of training material
 *   ACTIONS           — bias toward/away from specific action types
 *
 * Learning:
 *   Implicit — DecisionLoop reports chosen actions; preferences strengthen
 *   for domains related to that action.  Explicit — external callers can
 *   directly adjust preferences (teacher feedback).
 *
 * Decay:
 *   Every tick, all preferences drift toward 0.0 at rate DECAY_RATE × dt.
 *   This ensures unused preferences fade, preventing personality fossilisation.
 *
 * Scoring influence:
 *   action_bias(ActionType) returns a small float ∈ [-MAX_BIAS, +MAX_BIAS]
 *   that the DecisionLoop adds to the base score of each candidate action.
 *
 * Persistence:
 *   Serialise/deserialise via to_json()/from_json() for LMDB or file storage.
 *
 * Phase: NIK-PREF-01 (PreferenceEngine, v0.2.3)
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <sstream>
#include <vector>

namespace nikola::interior {

// ============================================================================
// PreferenceDomain
// ============================================================================

enum class PreferenceDomain : uint8_t {
    TOPICS             = 0,
    CODE_PATTERNS      = 1,
    INTERACTION_STYLES = 2,
    DATA_SOURCES       = 3,
    ACTIONS            = 4,
    COUNT              = 5
};

[[nodiscard]] inline const char* domain_name(PreferenceDomain d) noexcept {
    switch (d) {
        case PreferenceDomain::TOPICS:             return "TOPICS";
        case PreferenceDomain::CODE_PATTERNS:      return "CODE_PATTERNS";
        case PreferenceDomain::INTERACTION_STYLES:  return "INTERACTION_STYLES";
        case PreferenceDomain::DATA_SOURCES:        return "DATA_SOURCES";
        case PreferenceDomain::ACTIONS:             return "ACTIONS";
        default:                                   return "UNKNOWN";
    }
}

// ============================================================================
// Preference — single preference entry
// ============================================================================

struct Preference {
    double value    = 0.0;    ///< Affinity ∈ [-1.0, +1.0]; positive = like, negative = dislike
    double strength = 0.0;    ///< Confidence/certainty ∈ [0, ∞); grows with repetition
    uint64_t last_tick = 0;   ///< Last tick this preference was updated
    uint32_t update_count = 0; ///< Total updates to this preference
};

// ============================================================================
// PreferenceEngineConfig
// ============================================================================

struct PreferenceEngineConfig {
    /// Learning rate: how much each learn() call shifts value.
    double learn_rate = 0.05;

    /// Decay rate per second: preferences drift toward 0.0.
    double decay_rate = 0.001;

    /// Maximum absolute bias added to action scores.
    double max_bias = 0.15;

    /// Minimum strength before a preference influences scoring.
    double min_influence_strength = 0.5;

    /// Maximum absolute preference value (clamped).
    double max_value = 1.0;
};

// ============================================================================
// PreferenceStats
// ============================================================================

struct PreferenceStats {
    size_t total_preferences = 0;
    size_t total_updates = 0;
    size_t domains_active = 0;
};

// ============================================================================
// PreferenceEngine
// ============================================================================

class PreferenceEngine {
public:
    explicit PreferenceEngine(PreferenceEngineConfig cfg = {})
        : cfg_(cfg) {}

    // ── Learning ─────────────────────────────────────────────────────────

    /**
     * @brief Learn a preference: adjust value for key in domain.
     * @param domain  Which category this preference belongs to.
     * @param key     The specific preference key (e.g., "concurrency", "verbose").
     * @param delta   Direction: positive = like, negative = dislike.
     * @param tick    Current tick for timestamp tracking.
     */
    void learn(PreferenceDomain domain, const std::string& key,
               double delta, uint64_t tick);

    /**
     * @brief Learn implicitly from a chosen action.
     *
     * Maps action types to relevant preference domains and keys:
     *   EXPLORE       → ACTIONS:"explore" +, TOPICS:"novelty" +
     *   GENERATE_CODE → ACTIONS:"generate_code" +, CODE_PATTERNS:"active" +
     *   EMIT_THOUGHT  → ACTIONS:"emit_thought" +, INTERACTION_STYLES:"expressive" +
     *   REASON        → ACTIONS:"reason" +, TOPICS:"analysis" +
     *   STORE_MEMORY  → ACTIONS:"store_memory" +
     *   PURSUE_GOAL   → ACTIONS:"pursue_goal" +
     *   RECALL_MEMORY → ACTIONS:"recall_memory" +
     *   REQUEST_LOOKUP → ACTIONS:"request_lookup" +, DATA_SOURCES:"external" +
     */
    void learn_from_action(int action_type, uint64_t tick);

    // ── Decay ────────────────────────────────────────────────────────────

    /**
     * @brief Decay all preferences toward 0.0.
     * @param dt  Elapsed seconds since last decay call.
     */
    void decay(double dt);

    // ── Query ────────────────────────────────────────────────────────────

    /**
     * @brief Get preference value for a key in a domain.
     * @return Preference value ∈ [-1.0, +1.0], or 0.0 if not found.
     */
    [[nodiscard]] double query(PreferenceDomain domain,
                               const std::string& key) const;

    /**
     * @brief Get the full Preference struct for a key.
     * @return Pointer to Preference, or nullptr if not found.
     */
    [[nodiscard]] const Preference* get(PreferenceDomain domain,
                                        const std::string& key) const;

    /**
     * @brief Compute scoring bias for an action type.
     *
     * Looks up ACTIONS domain for the action name.  Only returns non-zero
     * if the preference has sufficient strength (≥ min_influence_strength).
     *
     * @return Bias ∈ [-max_bias, +max_bias] to add to the action's base score.
     */
    [[nodiscard]] double action_bias(int action_type) const;

    /**
     * @brief List all preferences in a domain.
     */
    [[nodiscard]] std::vector<std::pair<std::string, Preference>>
        list_domain(PreferenceDomain domain) const;

    /**
     * @brief Get top N strongest preferences across all domains.
     */
    [[nodiscard]] std::vector<std::pair<std::string, Preference>>
        top_preferences(size_t n = 10) const;

    // ── Stats ────────────────────────────────────────────────────────────

    [[nodiscard]] PreferenceStats stats() const;

    // ── Persistence ──────────────────────────────────────────────────────

    /**
     * @brief Serialise all preferences to a JSON string.
     */
    [[nodiscard]] std::string to_json() const;

    /**
     * @brief Deserialise preferences from a JSON string.
     * @return true on success.
     */
    bool from_json(const std::string& json);

    // ── Reset ────────────────────────────────────────────────────────────

    void reset();

    // ── Config access ────────────────────────────────────────────────────

    [[nodiscard]] const PreferenceEngineConfig& config() const noexcept { return cfg_; }

private:
    PreferenceEngineConfig cfg_;

    /// preferences_[domain_index][key] = Preference
    std::map<std::string, Preference> domains_[static_cast<size_t>(PreferenceDomain::COUNT)];

    /// Map action type int to its string name for ACTIONS domain lookup.
    static const char* action_key(int action_type);
};

} // namespace nikola::interior
