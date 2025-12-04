#pragma once

#include <string>
#include <vector>
#include <map>
#include <cstdint>
#include "nikola/interior/affective_state.hpp"

namespace nikola::interior {

// Forward declaration
class TorusManifold;

/**
 * @brief Autobiographical Memory - Personal Narrative and Identity
 *
 * More than just event logs - this creates a continuous sense of self
 * over time. Answers questions like:
 * - Who am I?
 * - What have I experienced?
 * - What do I care about?
 * - What am I good at?
 * - How have I changed?
 *
 * Identity emerges from accumulated experiences, values, and skills.
 * This isn't a database - it's a narrative structure that gives
 * the system a sense of continuity and purpose.
 *
 * Key capabilities:
 * - Record significant life events
 * - Generate personal narratives ("my story")
 * - Track value formation over time
 * - Measure skill development
 * - Maintain coherent identity across restarts
 *
 * @status STUB - Implementation deferred to Phase 6
 */

struct LifeEvent {
    int64_t timestamp;
    std::string description;
    std::vector<double> wave_signature;  // State during event
    Affect dominant_affect;              // How it felt
    double significance;                 // 0.0-1.0 (how important)
    std::vector<std::string> tags;       // "first_success", "major_failure", etc.
};

struct SkillLevel {
    std::string skill_name;
    double proficiency;       // 0.0-1.0
    int64_t last_practiced;
    int practice_count;
};

class AutobiographicalMemory {
public:
    /**
     * @brief Record a significant life event
     * @param event Event to record
     * @param torus Current torus state
     */
    void record_event(const LifeEvent& event, const TorusManifold& torus);

    /**
     * @brief Recall events from a time period
     * @param range Time range to query
     * @return Events in that period
     */
    std::vector<LifeEvent> recall_period(const TimeRange& range);

    /**
     * @brief Generate narrative summary of experiences
     * @param range Optional time range (default = all time)
     * @return Human-readable story
     */
    std::string generate_narrative(const TimeRange* range = nullptr);

    /**
     * @brief Get current values (what I care about)
     * @return Map of value → importance
     */
    std::map<std::string, double> get_values() const;

    /**
     * @brief Update value based on experience
     * @param value_name Name like "curiosity", "safety", "efficiency"
     * @param delta Change in importance (-1.0 to +1.0)
     */
    void update_value(const std::string& value_name, double delta);

    /**
     * @brief Get all tracked skills and proficiency levels
     * @return Skill inventory
     */
    std::vector<SkillLevel> get_skills() const;

    /**
     * @brief Update skill level based on practice/success
     * @param skill_name Name like "summarization", "math", "debugging"
     * @param success true if practice was successful
     */
    void update_skill(const std::string& skill_name, bool success);

    /**
     * @brief Get identity summary
     * @return Text like "I am a curious AI who values learning and helping"
     */
    std::string get_identity() const;

    /**
     * @brief Get most significant events
     * @param count How many to return
     * @return Top N most significant events
     */
    std::vector<LifeEvent> get_most_significant(int count = 10);

    /**
     * @brief Search events by tag
     * @param tag Tag to search for
     * @return Matching events
     */
    std::vector<LifeEvent> find_by_tag(const std::string& tag);

    /**
     * @brief Get total lifetime stats
     * @return Map of metric → count
     */
    std::map<std::string, uint64_t> get_lifetime_stats() const;

private:
    std::vector<LifeEvent> events_;
    std::map<std::string, double> values_;  // What I care about
    std::vector<SkillLevel> skills_;
    int64_t birth_timestamp_ = 0;  // When system first started
};

} // namespace nikola::interior
