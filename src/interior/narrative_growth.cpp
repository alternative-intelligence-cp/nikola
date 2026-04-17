/**
 * @file interior/narrative_growth.cpp
 * @brief v0.2.3 Phase 3 — NarrativeGrowth implementation.
 */

#include <nikola/interior/narrative_growth.hpp>
#include <nikola/autonomy/decision_loop.hpp>

#include <algorithm>
#include <cmath>
#include <sstream>

namespace nikola::interior {

using nikola::autonomy::ActionType;
using nikola::autonomy::action_name;

// ============================================================================
// Milestone detection
// ============================================================================

std::string NarrativeGrowth::check_first_action(int action_type, uint64_t /*tick*/) {
    if (action_type == static_cast<int>(ActionType::SILENT)) return {};

    if (seen_actions_.insert(action_type).second) {
        // First time seeing this action type
        std::string name = action_name(static_cast<ActionType>(action_type));
        return "First time performing action: " + name;
    }
    return {};
}

std::vector<std::string> NarrativeGrowth::check_personality_shifts(
    const PersonalitySnapshot& current, uint64_t /*tick*/)
{
    std::vector<std::string> milestones;

    if (!has_baseline_personality_) {
        last_personality_ = current;
        has_baseline_personality_ = true;
        return milestones;
    }

    for (size_t i = 0; i < PersonalityDrift::N_TRAITS; ++i) {
        float delta = current.traits[i] - last_personality_.traits[i];
        if (std::abs(delta) >= cfg_.personality_shift_threshold) {
            auto axis = static_cast<TraitAxis>(i);
            std::string desc = "Personality shift on " +
                std::string(trait_axis_name(axis)) + ": " +
                trait_description(axis, last_personality_.traits[i]) +
                " → " + trait_description(axis, current.traits[i]);
            milestones.push_back(desc);
        }
    }

    if (!milestones.empty()) {
        last_personality_ = current;
    }

    return milestones;
}

std::string NarrativeGrowth::check_skill_mastery(
    const std::string& skill, double proficiency, uint64_t /*tick*/)
{
    if (proficiency >= cfg_.skill_mastery_threshold) {
        if (mastered_skills_.insert(skill).second) {
            return "Achieved mastery in skill: " + skill +
                   " (proficiency " + std::to_string(proficiency) + ")";
        }
    }
    return {};
}

void NarrativeGrowth::record_milestone(
    AutobiographicalMemory& autobiography,
    MilestoneType type,
    const std::string& description,
    const NikolaState& state,
    Affect affect,
    uint64_t tick)
{
    std::string tag = std::string("milestone:") + milestone_type_name(type);
    autobiography.record_event(description, state, affect, 0.9, {tag, "milestone"});
    ++milestone_count_;
}

// ============================================================================
// Self-Reflection (NAP)
// ============================================================================

ReflectionEntry NarrativeGrowth::generate_reflection(
    AutobiographicalMemory& autobiography,
    const PersonalitySnapshot& personality,
    uint64_t recent_ticks,
    uint64_t tick)
{
    std::ostringstream oss;
    oss << "Self-reflection at tick " << tick << ": ";

    // Personality summary
    PersonalityDrift temp_pd;
    for (size_t i = 0; i < PersonalityDrift::N_TRAITS; ++i) {
        temp_pd.set_trait(static_cast<TraitAxis>(i), personality.traits[i]);
    }
    oss << temp_pd.describe() << " ";

    // Recent significant events
    auto top = autobiography.get_most_significant(3);
    if (!top.empty()) {
        oss << "Recent highlights: ";
        for (size_t i = 0; i < top.size(); ++i) {
            if (i > 0) oss << "; ";
            oss << top[i]->description;
        }
        oss << ". ";
    }

    // Identity
    std::string identity = autobiography.get_identity();
    if (!identity.empty()) {
        oss << identity;
    }

    ReflectionEntry entry;
    entry.tick = tick;
    entry.text = oss.str();
    entry.personality = personality;
    entry.significance = 0.7;

    // Record into autobiography
    autobiography.record_event(
        entry.text,
        NikolaState{},
        Affect::NEUTRAL,
        entry.significance,
        {"nap_reflection", "milestone:NAP_REFLECTION", "milestone"}
    );

    ++reflection_count_;
    return entry;
}

// ============================================================================
// Compression
// ============================================================================

bool NarrativeGrowth::needs_compression(
    const AutobiographicalMemory& autobiography) const
{
    return autobiography.event_count() > cfg_.compress_threshold;
}

CompressionStats NarrativeGrowth::compress(
    AutobiographicalMemory& autobiography,
    const NikolaState& state,
    uint64_t tick)
{
    CompressionStats stats;
    const auto& events = autobiography.events();
    stats.events_before = events.size();

    if (!needs_compression(autobiography)) {
        stats.events_after = stats.events_before;
        return stats;
    }

    std::vector<LifeEvent> kept;
    kept.reserve(cfg_.compress_target);

    size_t removed = 0;

    for (const auto& event : events) {
        bool is_milestone = false;
        for (const auto& tag : event.tags) {
            if (tag.find("milestone") != std::string::npos) {
                is_milestone = true;
                break;
            }
        }

        if (is_milestone || event.significance >= cfg_.compress_min_significance) {
            kept.push_back(event);
            if (is_milestone) ++stats.milestones_kept;
        } else {
            ++removed;
        }
    }

    // If still too many, keep only the most significant
    if (kept.size() > cfg_.compress_target) {
        std::sort(kept.begin(), kept.end(),
                  [](const LifeEvent& a, const LifeEvent& b) {
                      return a.significance > b.significance;
                  });
        kept.resize(cfg_.compress_target);
        // Re-sort by tick for chronological order
        std::sort(kept.begin(), kept.end(),
                  [](const LifeEvent& a, const LifeEvent& b) {
                      return a.tick < b.tick;
                  });
    }

    stats.events_removed = stats.events_before - kept.size();
    stats.events_after = kept.size();

    // Add a compression summary event
    if (stats.events_removed > 0) {
        LifeEvent summary;
        summary.tick = tick;
        summary.description = "Compressed autobiography: removed " +
            std::to_string(stats.events_removed) + " routine events, kept " +
            std::to_string(stats.events_after) + " (including " +
            std::to_string(stats.milestones_kept) + " milestones)";
        summary.state = state;
        summary.dominant_affect = Affect::NEUTRAL;
        summary.significance = 0.6;
        summary.tags = {"compression_summary"};
        kept.push_back(summary);
    }

    autobiography.replace_events(std::move(kept));

    return stats;
}

} // namespace nikola::interior
