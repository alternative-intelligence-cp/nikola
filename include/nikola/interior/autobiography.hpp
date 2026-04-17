#pragma once
/**
 * @file autobiography.hpp
 * @brief Phase 124 — AutobiographicalMemory: personal narrative and identity
 *
 * Creates a continuous sense of self over time by accumulating significant
 * life events, tracking value formation, and measuring skill development.
 * Answers: who am I, what have I experienced, what do I care about, how
 * have I changed?
 *
 * No TorusManifold / Coord9D / QuantumScratchpad dependencies.
 * Uses NikolaState for neurochemical snapshots and Affect for emotional
 * colouring of events.
 *
 * Key constants:
 *  AUTOBIOGRAPHY_MAX_EVENTS          1024   FIFO cap on stored LifeEvents
 *  AUTOBIOGRAPHY_SKILL_LEARN_RATE    0.10   proficiency gain per success
 *  AUTOBIOGRAPHY_SKILL_DECAY         0.02   proficiency decay per failure
 *  AUTOBIOGRAPHY_VALUE_LEARN_RATE    0.10   importance delta per update call
 *  AUTOBIOGRAPHY_SIGNIFICANCE_MIN    0.30   minimum significance to qualify
 *                                           as "most significant"
 *  AUTOBIOGRAPHY_TOP_N               10     default for get_most_significant()
 */

#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <functional>

#include <nikola/autonomy/decision_loop.hpp>
#include <nikola/interior/affective_state.hpp>

namespace nikola::interior {

using nikola::autonomy::NikolaState;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

inline constexpr size_t AUTOBIOGRAPHY_MAX_EVENTS       = 1024;
inline constexpr double AUTOBIOGRAPHY_SKILL_LEARN_RATE = 0.10;
inline constexpr double AUTOBIOGRAPHY_SKILL_DECAY      = 0.02;
inline constexpr double AUTOBIOGRAPHY_VALUE_LEARN_RATE = 0.10;
inline constexpr double AUTOBIOGRAPHY_SIGNIFICANCE_MIN = 0.30;
inline constexpr size_t AUTOBIOGRAPHY_TOP_N            = 10;

// ---------------------------------------------------------------------------
// Tick-based time range
// ---------------------------------------------------------------------------

struct TickRange {
    uint64_t start_tick = 0;
    uint64_t end_tick   = UINT64_MAX;

    bool contains(uint64_t tick) const {
        return tick >= start_tick && tick <= end_tick;
    }
};

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/**
 * @brief A single significant life event.
 */
struct LifeEvent {
    uint64_t    tick          = 0;
    std::string description;
    NikolaState state         = {};       ///< neurochemical snapshot at event
    Affect      dominant_affect = Affect::NEUTRAL;
    double      significance  = 0.5;     ///< importance [0, 1]
    std::vector<std::string> tags;        ///< e.g. "first_success", "milestone"
};

/**
 * @brief Tracked skill with proficiency and practice history.
 */
struct SkillLevel {
    std::string skill_name;
    double      proficiency    = 0.0;    ///< [0, 1] — clamped after each update
    uint64_t    last_tick      = 0;      ///< tick of last practice
    uint64_t    practice_count = 0;
    uint64_t    success_count  = 0;

    double success_rate() const {
        return (practice_count == 0)
            ? 0.0
            : static_cast<double>(success_count) /
              static_cast<double>(practice_count);
    }
};

/**
 * @brief A tracked value — something the system cares about.
 */
struct ValueEntry {
    std::string value_name;
    double      importance    = 0.5;    ///< [0, 1] — clamped after each update
    uint64_t    update_count  = 0;
};

// ---------------------------------------------------------------------------
// AutobiographicalMemory
// ---------------------------------------------------------------------------

class AutobiographicalMemory {
public:
    AutobiographicalMemory() = default;

    // --- Event recording ----------------------------------------------------

    /**
     * @brief Record a significant life event.
     *
     * Buffer is capped at AUTOBIOGRAPHY_MAX_EVENTS (FIFO eviction).
     * significance is clamped to [0, 1].
     */
    void record_event(const std::string&              description,
                      const NikolaState&               state,
                      Affect                           dominant_affect = Affect::NEUTRAL,
                      double                           significance    = 0.5,
                      const std::vector<std::string>&  tags            = {});

    // --- Recall -------------------------------------------------------------

    /**
     * @brief Recall all events within a tick range.
     */
    std::vector<const LifeEvent*> recall_period(const TickRange& range) const;

    /**
     * @brief Recall events whose description or tags contain any of the
     *        space-separated keywords in query (case-insensitive).
     */
    std::vector<const LifeEvent*> recall_by_query(const std::string& query,
                                                   size_t max = 20) const;

    /**
     * @brief Return the top-N most significant events (descending).
     */
    std::vector<const LifeEvent*> get_most_significant(
        size_t count = AUTOBIOGRAPHY_TOP_N) const;

    /**
     * @brief Find events carrying a specific tag (exact match).
     */
    std::vector<const LifeEvent*> find_by_tag(const std::string& tag) const;

    // --- Narrative ----------------------------------------------------------

    /**
     * @brief Generate a human-readable narrative over all events (or a range).
     * @param range   Optional tick range; nullptr = all events.
     * @return        Multi-sentence story string.
     */
    std::string generate_narrative(const TickRange* range = nullptr) const;

    /**
     * @brief Produce an identity summary string.
     *
     * Format: "I am a [top-affect] entity. I value [top-2 values].
     *          My strongest skills are [top-2 skills].
     *          I have recorded [N] events."
     */
    std::string get_identity() const;

    // --- Values -------------------------------------------------------------

    /**
     * @brief Get current value map (name → importance).
     */
    std::map<std::string, double> get_values() const;

    /**
     * @brief Update (or create) a value by name.
     *
     * importance += delta * AUTOBIOGRAPHY_VALUE_LEARN_RATE, clamped to [0,1].
     */
    void update_value(const std::string& value_name, double delta);

    /**
     * @brief Return the highest-importance value name, or "" if none.
     */
    std::string dominant_value() const;

    // --- Skills -------------------------------------------------------------

    /**
     * @brief Get a snapshot of all tracked skills.
     */
    std::vector<SkillLevel> get_skills() const { return skills_; }

    /**
     * @brief Update (or create) a skill entry.
     *
     * success=true  → proficiency += AUTOBIOGRAPHY_SKILL_LEARN_RATE
     * success=false → proficiency -= AUTOBIOGRAPHY_SKILL_DECAY
     * proficiency clamped to [0, 1].
     */
    void update_skill(const std::string& skill_name, bool success,
                      uint64_t tick = 0);

    /**
     * @brief Return the skill with highest proficiency, or "" if none.
     */
    std::string best_skill() const;

    // --- Accessors ----------------------------------------------------------

    const std::vector<LifeEvent>&  events()       const { return events_; }
    const std::vector<ValueEntry>& value_entries() const { return values_; }

    /**
     * @brief Replace events vector (used by NarrativeGrowth for compression).
     */
    void replace_events(std::vector<LifeEvent> new_events) {
        events_ = std::move(new_events);
    }

    size_t event_count() const { return events_.size(); }
    size_t skill_count() const { return skills_.size(); }
    size_t value_count() const { return values_.size(); }

    // --- Stats --------------------------------------------------------------

    struct Stats {
        size_t   total_events      = 0;
        size_t   total_skills      = 0;
        size_t   total_values      = 0;
        double   mean_significance = 0.0;
        Affect   most_common_affect = Affect::NEUTRAL;
    };

    Stats stats() const;

    // --- Callback -----------------------------------------------------------

    using EventCallback = std::function<void(const LifeEvent&)>;
    void on_event_recorded(EventCallback cb) { event_cb_ = std::move(cb); }

    // --- Pure-static helpers ------------------------------------------------

    /**
     * @brief True if any keyword (space-split) appears in text (case-insensitive).
     */
    static bool text_matches(const std::string& text, const std::string& query);

    /**
     * @brief Return the label for an Affect enum value.
     */
    static std::string affect_label(Affect a);

private:
    std::vector<LifeEvent>  events_;
    std::vector<SkillLevel> skills_;
    std::vector<ValueEntry> values_;
    EventCallback           event_cb_;

    // find skill/value by name; return nullptr if absent
    SkillLevel*  find_skill(const std::string& name);
    ValueEntry*  find_value(const std::string& name);
};

} // namespace nikola::interior
