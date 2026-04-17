/**
 * @file tests/unit/narrative_growth_test.cpp
 * @brief v0.2.3 Phase 3 — NarrativeGrowth unit tests.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <nikola/interior/narrative_growth.hpp>

using namespace nikola::interior;
using Catch::Approx;

// Helper: create a NikolaState with defaults
static NikolaState make_state() { return NikolaState{}; }

// ============================================================================
// §A — Milestone: First Action
// ============================================================================

TEST_CASE("§A-1 first action detected as milestone", "[narrative]") {
    NarrativeGrowth ng;
    auto desc = ng.check_first_action(4, 100);  // EXPLORE
    CHECK(!desc.empty());
    CHECK(desc.find("EXPLORE") != std::string::npos);
}

TEST_CASE("§A-2 repeated action not a milestone", "[narrative]") {
    NarrativeGrowth ng;
    ng.check_first_action(4, 100);
    auto desc = ng.check_first_action(4, 200);
    CHECK(desc.empty());
}

TEST_CASE("§A-3 SILENT never flagged as milestone", "[narrative]") {
    NarrativeGrowth ng;
    auto desc = ng.check_first_action(0, 100);  // SILENT
    CHECK(desc.empty());
}

TEST_CASE("§A-4 different actions each trigger milestone", "[narrative]") {
    NarrativeGrowth ng;
    CHECK(!ng.check_first_action(1, 100).empty());  // EMIT_THOUGHT
    CHECK(!ng.check_first_action(4, 200).empty());  // EXPLORE
    CHECK(!ng.check_first_action(9, 300).empty());  // REASON
    CHECK(ng.seen_actions().size() == 3);
}

// ============================================================================
// §B — Milestone: Personality Shift
// ============================================================================

TEST_CASE("§B-1 first personality snapshot sets baseline", "[narrative]") {
    NarrativeGrowth ng;
    PersonalitySnapshot snap{};
    auto shifts = ng.check_personality_shifts(snap, 100);
    CHECK(shifts.empty());  // baseline only
}

TEST_CASE("§B-2 small shift not detected", "[narrative]") {
    NarrativeGrowth ng;
    PersonalitySnapshot snap1{};
    (void)ng.check_personality_shifts(snap1, 100);

    PersonalitySnapshot snap2{};
    snap2.traits[1] = 0.1f;  // small CAUTIOUS_BOLD shift
    auto shifts = ng.check_personality_shifts(snap2, 200);
    CHECK(shifts.empty());
}

TEST_CASE("§B-3 large shift detected as milestone", "[narrative]") {
    NarrativeGrowth ng;
    PersonalitySnapshot snap1{};
    (void)ng.check_personality_shifts(snap1, 100);

    PersonalitySnapshot snap2{};
    snap2.traits[1] = 0.5f;  // big CAUTIOUS_BOLD shift
    auto shifts = ng.check_personality_shifts(snap2, 200);
    CHECK(shifts.size() == 1);
    CHECK(shifts[0].find("CAUTIOUS_BOLD") != std::string::npos);
}

TEST_CASE("§B-4 multiple shifts detected simultaneously", "[narrative]") {
    NarrativeGrowth ng;
    PersonalitySnapshot snap1{};
    ng.check_personality_shifts(snap1, 100);

    PersonalitySnapshot snap2{};
    snap2.traits[0] = -0.5f;  // CURIOUS_FOCUSED
    snap2.traits[1] = 0.5f;   // CAUTIOUS_BOLD
    auto shifts = ng.check_personality_shifts(snap2, 200);
    CHECK(shifts.size() == 2);
}

// ============================================================================
// §C — Milestone: Skill Mastery
// ============================================================================

TEST_CASE("§C-1 skill below threshold not a milestone", "[narrative]") {
    NarrativeGrowth ng;
    auto desc = ng.check_skill_mastery("reasoning", 0.5, 100);
    CHECK(desc.empty());
}

TEST_CASE("§C-2 skill at threshold is a milestone", "[narrative]") {
    NarrativeGrowth ng;
    auto desc = ng.check_skill_mastery("reasoning", 0.8, 100);
    CHECK(!desc.empty());
    CHECK(desc.find("reasoning") != std::string::npos);
}

TEST_CASE("§C-3 repeated mastery not re-detected", "[narrative]") {
    NarrativeGrowth ng;
    (void)ng.check_skill_mastery("reasoning", 0.9, 100);
    auto desc = ng.check_skill_mastery("reasoning", 0.95, 200);
    CHECK(desc.empty());
}

// ============================================================================
// §D — Record Milestone
// ============================================================================

TEST_CASE("§D-1 record_milestone adds tagged event", "[narrative]") {
    NarrativeGrowth ng;
    AutobiographicalMemory auto_mem;

    ng.record_milestone(auto_mem, MilestoneType::FIRST_ACTION,
                        "First explore", make_state(), Affect::CURIOSITY, 100);

    CHECK(auto_mem.event_count() == 1);
    CHECK(ng.milestone_count() == 1);
    auto events = auto_mem.find_by_tag("milestone");
    CHECK(events.size() == 1);
}

TEST_CASE("§D-2 milestone has high significance", "[narrative]") {
    NarrativeGrowth ng;
    AutobiographicalMemory auto_mem;

    ng.record_milestone(auto_mem, MilestoneType::GOAL_COMPLETED,
                        "Completed first goal", make_state(), Affect::SATISFACTION, 200);

    CHECK(auto_mem.events()[0].significance >= 0.9);
}

// ============================================================================
// §E — Self-Reflection
// ============================================================================

TEST_CASE("§E-1 generate_reflection creates entry", "[narrative]") {
    NarrativeGrowth ng;
    AutobiographicalMemory auto_mem;
    PersonalitySnapshot snap{};

    auto ref = ng.generate_reflection(auto_mem, snap, 0, 1000);
    CHECK(!ref.text.empty());
    CHECK(ref.tick == 1000);
    CHECK(ng.reflection_count() == 1);
}

TEST_CASE("§E-2 reflection recorded in autobiography", "[narrative]") {
    NarrativeGrowth ng;
    AutobiographicalMemory auto_mem;
    PersonalitySnapshot snap{};

    ng.generate_reflection(auto_mem, snap, 0, 1000);
    CHECK(auto_mem.event_count() == 1);
    auto events = auto_mem.find_by_tag("nap_reflection");
    CHECK(events.size() == 1);
}

TEST_CASE("§E-3 reflection includes personality description", "[narrative]") {
    NarrativeGrowth ng;
    AutobiographicalMemory auto_mem;
    PersonalitySnapshot snap{};
    snap.traits[1] = 0.8f;  // bold

    auto ref = ng.generate_reflection(auto_mem, snap, 0, 1000);
    CHECK(ref.text.find("bold") != std::string::npos);
}

// ============================================================================
// §F — Compression
// ============================================================================

TEST_CASE("§F-1 needs_compression false when under threshold", "[narrative]") {
    NarrativeGrowth ng;
    AutobiographicalMemory auto_mem;
    CHECK(!ng.needs_compression(auto_mem));
}

TEST_CASE("§F-2 needs_compression true when over threshold", "[narrative]") {
    NarrativeGrowthConfig cfg;
    cfg.compress_threshold = 10;
    NarrativeGrowth ng(cfg);
    AutobiographicalMemory auto_mem;

    for (int i = 0; i < 15; ++i) {
        auto_mem.record_event("event " + std::to_string(i), make_state(),
                              Affect::NEUTRAL, 0.3);
    }

    CHECK(ng.needs_compression(auto_mem));
}

TEST_CASE("§F-3 compress removes low-significance events", "[narrative]") {
    NarrativeGrowthConfig cfg;
    cfg.compress_threshold = 10;
    cfg.compress_target = 5;
    cfg.compress_min_significance = 0.5;
    NarrativeGrowth ng(cfg);
    AutobiographicalMemory auto_mem;

    // Add low significance events
    for (int i = 0; i < 12; ++i) {
        auto_mem.record_event("routine " + std::to_string(i), make_state(),
                              Affect::NEUTRAL, 0.2);
    }
    // Add high significance events
    for (int i = 0; i < 3; ++i) {
        auto_mem.record_event("important " + std::to_string(i), make_state(),
                              Affect::SATISFACTION, 0.8);
    }

    auto stats = ng.compress(auto_mem, make_state(), 5000);
    CHECK(stats.events_before == 15);
    CHECK(stats.events_removed > 0);
    CHECK(auto_mem.event_count() < 15);
}

TEST_CASE("§F-4 milestones survive compression", "[narrative]") {
    NarrativeGrowthConfig cfg;
    cfg.compress_threshold = 10;
    cfg.compress_target = 5;
    cfg.compress_min_significance = 0.5;
    NarrativeGrowth ng(cfg);
    AutobiographicalMemory auto_mem;

    // Add milestone
    ng.record_milestone(auto_mem, MilestoneType::FIRST_ACTION,
                        "First explore", make_state(), Affect::CURIOSITY, 50);

    // Add low significance events
    for (int i = 0; i < 15; ++i) {
        auto_mem.record_event("routine " + std::to_string(i), make_state(),
                              Affect::NEUTRAL, 0.2);
    }

    ng.compress(auto_mem, make_state(), 5000);

    auto milestones = auto_mem.find_by_tag("milestone");
    CHECK(milestones.size() >= 1);  // milestone survived
}

TEST_CASE("§F-5 compression adds summary event", "[narrative]") {
    NarrativeGrowthConfig cfg;
    cfg.compress_threshold = 10;
    cfg.compress_target = 5;
    NarrativeGrowth ng(cfg);
    AutobiographicalMemory auto_mem;

    for (int i = 0; i < 15; ++i) {
        auto_mem.record_event("event " + std::to_string(i), make_state(),
                              Affect::NEUTRAL, 0.2);
    }

    ng.compress(auto_mem, make_state(), 5000);

    auto summaries = auto_mem.find_by_tag("compression_summary");
    CHECK(summaries.size() == 1);
}

TEST_CASE("§F-6 no compression when not needed", "[narrative]") {
    NarrativeGrowth ng;
    AutobiographicalMemory auto_mem;
    auto_mem.record_event("small", make_state(), Affect::NEUTRAL, 0.5);

    auto stats = ng.compress(auto_mem, make_state(), 100);
    CHECK(stats.events_removed == 0);
    CHECK(auto_mem.event_count() == 1);
}

// ============================================================================
// §G — Milestone Type Names
// ============================================================================

TEST_CASE("§G-1 milestone_type_name returns correct strings", "[narrative]") {
    CHECK(std::string(milestone_type_name(MilestoneType::FIRST_ACTION)) == "FIRST_ACTION");
    CHECK(std::string(milestone_type_name(MilestoneType::PERSONALITY_SHIFT)) == "PERSONALITY_SHIFT");
    CHECK(std::string(milestone_type_name(MilestoneType::SKILL_MASTERY)) == "SKILL_MASTERY");
    CHECK(std::string(milestone_type_name(MilestoneType::VALUE_FORMATION)) == "VALUE_FORMATION");
    CHECK(std::string(milestone_type_name(MilestoneType::GOAL_COMPLETED)) == "GOAL_COMPLETED");
    CHECK(std::string(milestone_type_name(MilestoneType::NAP_REFLECTION)) == "NAP_REFLECTION");
    CHECK(std::string(milestone_type_name(MilestoneType::CUSTOM)) == "CUSTOM");
}

// ============================================================================
// §H — Config
// ============================================================================

TEST_CASE("§H-1 custom config used", "[narrative]") {
    NarrativeGrowthConfig cfg;
    cfg.personality_shift_threshold = 0.5f;
    cfg.skill_mastery_threshold = 0.9;
    NarrativeGrowth ng(cfg);

    CHECK(ng.config().personality_shift_threshold == 0.5f);
    CHECK(ng.config().skill_mastery_threshold == 0.9);
}
