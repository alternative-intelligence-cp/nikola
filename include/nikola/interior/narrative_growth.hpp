/**
 * @file interior/narrative_growth.hpp
 * @brief v0.2.3 Phase 3 — NarrativeGrowth: extends AutobiographicalMemory
 *        with structured narrative, self-reflection during NAP, milestone
 *        detection, and old-entry compression.
 *
 * Sits on top of AutobiographicalMemory (owned externally, e.g., by DecisionLoop)
 * and adds:
 *   - Milestone detection: auto-tag first occurrences and personality shifts
 *   - Self-reflection: NAP callback generates introspective entries
 *   - Compression: old routine entries merged into summaries, milestones kept
 *   - Personality context: record current trait snapshot with events
 *
 * Phase: NIK-NARR-03 (NarrativeGrowth, v0.2.3)
 */

#pragma once

#include <nikola/interior/autobiography.hpp>
#include <nikola/interior/personality_drift.hpp>

#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace nikola::interior {

// ============================================================================
// MilestoneType — categories of significant life events
// ============================================================================

enum class MilestoneType : uint8_t {
    FIRST_ACTION       = 0,  ///< First time an action type was taken
    PERSONALITY_SHIFT  = 1,  ///< A trait crossed a threshold
    SKILL_MASTERY      = 2,  ///< A skill reached high proficiency
    VALUE_FORMATION    = 3,  ///< A new dominant value emerged
    GOAL_COMPLETED     = 4,  ///< A pursued goal succeeded
    NAP_REFLECTION     = 5,  ///< Self-reflection generated during NAP
    CUSTOM             = 6,  ///< Externally triggered milestone
};

[[nodiscard]] inline const char* milestone_type_name(MilestoneType m) noexcept {
    switch (m) {
        case MilestoneType::FIRST_ACTION:      return "FIRST_ACTION";
        case MilestoneType::PERSONALITY_SHIFT:  return "PERSONALITY_SHIFT";
        case MilestoneType::SKILL_MASTERY:      return "SKILL_MASTERY";
        case MilestoneType::VALUE_FORMATION:    return "VALUE_FORMATION";
        case MilestoneType::GOAL_COMPLETED:     return "GOAL_COMPLETED";
        case MilestoneType::NAP_REFLECTION:     return "NAP_REFLECTION";
        case MilestoneType::CUSTOM:             return "CUSTOM";
        default:                               return "UNKNOWN";
    }
}

// ============================================================================
// ReflectionEntry — self-reflection generated during NAP
// ============================================================================

struct ReflectionEntry {
    uint64_t    tick = 0;
    std::string text;
    PersonalitySnapshot personality;
    double      significance = 0.7;
};

// ============================================================================
// CompressionStats
// ============================================================================

struct CompressionStats {
    size_t events_before   = 0;
    size_t events_after    = 0;
    size_t events_removed  = 0;
    size_t milestones_kept = 0;
};

// ============================================================================
// NarrativeGrowthConfig
// ============================================================================

struct NarrativeGrowthConfig {
    /// Trait shift threshold for milestone detection.
    float personality_shift_threshold = 0.3f;

    /// Skill proficiency threshold for mastery milestone.
    double skill_mastery_threshold = 0.8;

    /// Maximum non-milestone events before compression triggers.
    size_t compress_threshold = 800;

    /// Target event count after compression (milestones always kept).
    size_t compress_target = 400;

    /// Minimum significance to survive compression.
    double compress_min_significance = 0.5;
};

// ============================================================================
// NarrativeGrowth
// ============================================================================

class NarrativeGrowth {
public:
    explicit NarrativeGrowth(NarrativeGrowthConfig cfg = {})
        : cfg_(cfg) {}

    // ── Milestone Detection ──────────────────────────────────────────────

    /**
     * @brief Check if an action constitutes a milestone (first occurrence).
     *
     * Tracks which action types have been seen.  The first time a new action
     * type is observed, returns the milestone description (or "" if not new).
     */
    [[nodiscard]] std::string check_first_action(int action_type, uint64_t tick);

    /**
     * @brief Check personality snapshot for trait shift milestones.
     *
     * Compares current traits against last-recorded snapshot.  If any trait
     * has crossed the shift threshold, returns milestone descriptions.
     */
    [[nodiscard]] std::vector<std::string> check_personality_shifts(
        const PersonalitySnapshot& current, uint64_t tick);

    /**
     * @brief Check skills for mastery milestones.
     */
    [[nodiscard]] std::string check_skill_mastery(
        const std::string& skill, double proficiency, uint64_t tick);

    /**
     * @brief Record a milestone event into the given autobiography.
     */
    void record_milestone(AutobiographicalMemory& autobiography,
                          MilestoneType type,
                          const std::string& description,
                          const NikolaState& state,
                          Affect affect,
                          uint64_t tick);

    // ── Self-Reflection (NAP) ────────────────────────────────────────────

    /**
     * @brief Generate a self-reflection entry based on recent events and
     *        current personality.  Called during NAP cycle.
     *
     * The reflection synthesizes:
     *   - Most significant recent events
     *   - Current personality traits
     *   - Dominant values
     *   - Skill progress
     *
     * @param autobiography  The autobiography to reflect on.
     * @param personality    Current personality snapshot.
     * @param recent_ticks   How far back to look (0 = all).
     * @param tick           Current tick for timestamping.
     * @return The generated reflection (also recorded into autobiography).
     */
    ReflectionEntry generate_reflection(
        AutobiographicalMemory& autobiography,
        const PersonalitySnapshot& personality,
        uint64_t recent_ticks,
        uint64_t tick);

    // ── Compression ──────────────────────────────────────────────────────

    /**
     * @brief Check if the autobiography needs compression.
     */
    [[nodiscard]] bool needs_compression(
        const AutobiographicalMemory& autobiography) const;

    /**
     * @brief Compress the autobiography: remove low-significance non-milestone
     *        events, keeping milestones and high-significance entries.
     *
     * Creates a summary entry for the removed batch, recording what was lost.
     *
     * @return Compression statistics.
     */
    CompressionStats compress(AutobiographicalMemory& autobiography,
                              const NikolaState& state,
                              uint64_t tick);

    // ── Observers ────────────────────────────────────────────────────────

    [[nodiscard]] size_t milestone_count() const noexcept { return milestone_count_; }
    [[nodiscard]] size_t reflection_count() const noexcept { return reflection_count_; }
    [[nodiscard]] const std::set<int>& seen_actions() const noexcept { return seen_actions_; }
    [[nodiscard]] const std::set<std::string>& mastered_skills() const noexcept { return mastered_skills_; }

    // ── Config ───────────────────────────────────────────────────────────

    [[nodiscard]] const NarrativeGrowthConfig& config() const noexcept { return cfg_; }

private:
    NarrativeGrowthConfig cfg_;
    std::set<int> seen_actions_;          ///< Action types already seen
    std::set<std::string> mastered_skills_; ///< Skills that hit mastery
    PersonalitySnapshot last_personality_; ///< For trait shift detection
    bool has_baseline_personality_ = false;
    size_t milestone_count_  = 0;
    size_t reflection_count_ = 0;
};

} // namespace nikola::interior
