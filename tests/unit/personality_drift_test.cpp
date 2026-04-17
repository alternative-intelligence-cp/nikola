/**
 * @file tests/unit/personality_drift_test.cpp
 * @brief v0.2.3 Phase 2 — PersonalityDrift unit tests.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <nikola/interior/personality_drift.hpp>

using namespace nikola::interior;
using Catch::Approx;

// ============================================================================
// §A — Construction & Defaults
// ============================================================================

TEST_CASE("§A-1 default traits are zero (balanced)", "[personality]") {
    PersonalityDrift pd;
    for (size_t i = 0; i < PersonalityDrift::N_TRAITS; ++i) {
        CHECK(pd.trait(static_cast<TraitAxis>(i)) == 0.0f);
    }
    CHECK(pd.total_events() == 0);
}

TEST_CASE("§A-2 set_trait works", "[personality]") {
    PersonalityDrift pd;
    pd.set_trait(TraitAxis::CAUTIOUS_BOLD, 0.5f);
    CHECK(pd.trait(TraitAxis::CAUTIOUS_BOLD) == Approx(0.5f));
}

TEST_CASE("§A-3 set_trait clamped", "[personality]") {
    PersonalityDrift pd;
    pd.set_trait(TraitAxis::CAUTIOUS_BOLD, 5.0f);
    CHECK(pd.trait(TraitAxis::CAUTIOUS_BOLD) == Approx(1.0f));
    pd.set_trait(TraitAxis::CAUTIOUS_BOLD, -5.0f);
    CHECK(pd.trait(TraitAxis::CAUTIOUS_BOLD) == Approx(-1.0f));
}

// ============================================================================
// §B — Drift from Outcomes
// ============================================================================

TEST_CASE("§B-1 risky success → bolder", "[personality]") {
    PersonalityDrift pd;
    ExperienceOutcome outcome{1.0f, 4, 0.8f, 0.5f};  // success, EXPLORE, high risk
    pd.apply_outcome(outcome);

    CHECK(pd.trait(TraitAxis::CAUTIOUS_BOLD) > 0.0f);  // drifted toward bold
}

TEST_CASE("§B-2 risky failure → more cautious", "[personality]") {
    PersonalityDrift pd;
    ExperienceOutcome outcome{-1.0f, 10, 0.8f, 0.5f};  // failure, high risk
    pd.apply_outcome(outcome);

    CHECK(pd.trait(TraitAxis::CAUTIOUS_BOLD) < 0.0f);  // drifted toward cautious
}

TEST_CASE("§B-3 EXPLORE success → more curious", "[personality]") {
    PersonalityDrift pd;
    ExperienceOutcome outcome{0.8f, 4, 0.2f, 0.3f};  // EXPLORE success
    pd.apply_outcome(outcome);

    CHECK(pd.trait(TraitAxis::CURIOUS_FOCUSED) < 0.0f);  // toward curious
}

TEST_CASE("§B-4 REASON success → more analytical", "[personality]") {
    PersonalityDrift pd;
    ExperienceOutcome outcome{0.8f, 9, 0.3f, 0.6f};  // REASON success
    pd.apply_outcome(outcome);

    CHECK(pd.trait(TraitAxis::ANALYTICAL_INTUITIVE) < 0.0f);  // toward analytical
}

TEST_CASE("§B-5 complex task success → more patient", "[personality]") {
    PersonalityDrift pd;
    ExperienceOutcome outcome{0.8f, 2, 0.3f, 0.8f};  // high complexity success
    pd.apply_outcome(outcome);

    CHECK(pd.trait(TraitAxis::PATIENT_URGENT) < 0.0f);  // toward patient
}

TEST_CASE("§B-6 EMIT_THOUGHT → more verbose", "[personality]") {
    PersonalityDrift pd;
    ExperienceOutcome outcome{0.5f, 1, 0.1f, 0.2f};  // EMIT_THOUGHT
    pd.apply_outcome(outcome);

    CHECK(pd.trait(TraitAxis::VERBOSE_TERSE) < 0.0f);  // toward verbose
}

TEST_CASE("§B-7 drift bounded per event", "[personality]") {
    PersonalityDrift pd;
    ExperienceOutcome outcome{1.0f, 4, 1.0f, 1.0f};  // max everything
    pd.apply_outcome(outcome);

    for (size_t i = 0; i < PersonalityDrift::N_TRAITS; ++i) {
        CHECK(std::abs(pd.trait(static_cast<TraitAxis>(i))) <= pd.config().drift_per_event + 0.001f);
    }
}

TEST_CASE("§B-8 drift bounded per epoch", "[personality]") {
    PersonalityDrift pd;

    // Apply many outcomes in same direction
    for (int i = 0; i < 100; i++) {
        ExperienceOutcome outcome{1.0f, 4, 0.8f, 0.5f};
        pd.apply_outcome(outcome);
    }

    // CAUTIOUS_BOLD should be capped at drift_per_epoch
    float bold = pd.trait(TraitAxis::CAUTIOUS_BOLD);
    CHECK(bold <= pd.config().drift_per_epoch + 0.001f);
}

TEST_CASE("§B-9 reset_epoch allows new drift", "[personality]") {
    PersonalityDrift pd;

    // Fill epoch
    for (int i = 0; i < 100; i++) {
        pd.apply_outcome({1.0f, 4, 0.8f, 0.5f});
    }
    float before = pd.trait(TraitAxis::CAUTIOUS_BOLD);

    pd.reset_epoch();

    // More drift now possible
    for (int i = 0; i < 10; i++) {
        pd.apply_outcome({1.0f, 4, 0.8f, 0.5f});
    }
    float after = pd.trait(TraitAxis::CAUTIOUS_BOLD);
    CHECK(after > before);
}

// ============================================================================
// §C — Homeostatic Decay
// ============================================================================

TEST_CASE("§C-1 extreme positions regress toward zero", "[personality]") {
    PersonalityDrift pd;
    pd.set_trait(TraitAxis::CAUTIOUS_BOLD, 0.8f);

    pd.decay(100.0f);  // 100 seconds of decay

    CHECK(pd.trait(TraitAxis::CAUTIOUS_BOLD) < 0.8f);
    CHECK(pd.trait(TraitAxis::CAUTIOUS_BOLD) > 0.0f);
}

TEST_CASE("§C-2 balanced traits stay balanced", "[personality]") {
    PersonalityDrift pd;
    pd.decay(100.0f);

    for (size_t i = 0; i < PersonalityDrift::N_TRAITS; ++i) {
        CHECK(pd.trait(static_cast<TraitAxis>(i)) == 0.0f);
    }
}

// ============================================================================
// §D — Action Multiplier
// ============================================================================

TEST_CASE("§D-1 balanced personality gives 1.0 multiplier", "[personality]") {
    PersonalityDrift pd;
    for (int a = 0; a <= 11; a++) {
        CHECK(pd.action_multiplier(a) == Approx(1.0f));
    }
}

TEST_CASE("§D-2 curious personality boosts EXPLORE", "[personality]") {
    PersonalityDrift pd;
    pd.set_trait(TraitAxis::CURIOUS_FOCUSED, -0.8f);  // very curious

    CHECK(pd.action_multiplier(4) > 1.0f);  // EXPLORE boosted
}

TEST_CASE("§D-3 analytical personality boosts REASON", "[personality]") {
    PersonalityDrift pd;
    pd.set_trait(TraitAxis::ANALYTICAL_INTUITIVE, -0.8f);  // very analytical

    CHECK(pd.action_multiplier(9) > 1.0f);  // REASON boosted
}

TEST_CASE("§D-4 verbose personality boosts EMIT_THOUGHT", "[personality]") {
    PersonalityDrift pd;
    pd.set_trait(TraitAxis::VERBOSE_TERSE, -0.8f);  // very verbose

    CHECK(pd.action_multiplier(1) > 1.0f);  // EMIT_THOUGHT boosted
}

TEST_CASE("§D-5 bold personality boosts PURSUE_GOAL", "[personality]") {
    PersonalityDrift pd;
    pd.set_trait(TraitAxis::CAUTIOUS_BOLD, 0.8f);  // very bold
    pd.set_trait(TraitAxis::PATIENT_URGENT, 0.8f);  // very urgent

    CHECK(pd.action_multiplier(11) > 1.0f);  // PURSUE_GOAL boosted
}

TEST_CASE("§D-6 multiplier clamped to [0.7, 1.3]", "[personality]") {
    PersonalityDrift pd;
    pd.set_trait(TraitAxis::CURIOUS_FOCUSED, -1.0f);
    pd.set_trait(TraitAxis::CAUTIOUS_BOLD, 1.0f);

    for (int a = 0; a <= 11; a++) {
        float m = pd.action_multiplier(a);
        CHECK(m >= 0.7f);
        CHECK(m <= 1.3f);
    }
}

// ============================================================================
// §E — Description
// ============================================================================

TEST_CASE("§E-1 balanced personality described as balanced", "[personality]") {
    PersonalityDrift pd;
    std::string desc = pd.describe();
    CHECK(desc.find("balanced") != std::string::npos);
}

TEST_CASE("§E-2 strong traits appear in description", "[personality]") {
    PersonalityDrift pd;
    pd.set_trait(TraitAxis::CAUTIOUS_BOLD, 0.8f);

    std::string desc = pd.describe();
    CHECK(desc.find("bold") != std::string::npos);
}

// ============================================================================
// §F — Persistence
// ============================================================================

TEST_CASE("§F-1 JSON round-trip preserves traits", "[personality]") {
    PersonalityDrift pd1;
    pd1.set_trait(TraitAxis::CURIOUS_FOCUSED, -0.3f);
    pd1.set_trait(TraitAxis::CAUTIOUS_BOLD, 0.7f);
    pd1.set_trait(TraitAxis::VERBOSE_TERSE, -0.1f);
    // Apply some events for total_events counter
    pd1.apply_outcome({0.5f, 4, 0.3f, 0.5f});

    std::string json = pd1.to_json();

    PersonalityDrift pd2;
    CHECK(pd2.from_json(json));

    for (size_t i = 0; i < PersonalityDrift::N_TRAITS; ++i) {
        auto axis = static_cast<TraitAxis>(i);
        CHECK(pd2.trait(axis) == Approx(pd1.trait(axis)).margin(0.01f));
    }
}

TEST_CASE("§F-2 from_json on empty returns true", "[personality]") {
    PersonalityDrift pd;
    CHECK(pd.from_json("{}"));
}

// ============================================================================
// §G — Trait Axis Names
// ============================================================================

TEST_CASE("§G-1 trait_axis_name returns correct strings", "[personality]") {
    CHECK(std::string(trait_axis_name(TraitAxis::CURIOUS_FOCUSED)) == "CURIOUS_FOCUSED");
    CHECK(std::string(trait_axis_name(TraitAxis::CAUTIOUS_BOLD)) == "CAUTIOUS_BOLD");
    CHECK(std::string(trait_axis_name(TraitAxis::VERBOSE_TERSE)) == "VERBOSE_TERSE");
    CHECK(std::string(trait_axis_name(TraitAxis::PATIENT_URGENT)) == "PATIENT_URGENT");
    CHECK(std::string(trait_axis_name(TraitAxis::ANALYTICAL_INTUITIVE)) == "ANALYTICAL_INTUITIVE");
}

TEST_CASE("§G-2 trait_description gives meaningful labels", "[personality]") {
    CHECK(trait_description(TraitAxis::CAUTIOUS_BOLD, 0.0f).find("balanced") != std::string::npos);
    CHECK(trait_description(TraitAxis::CAUTIOUS_BOLD, 0.8f).find("bold") != std::string::npos);
    CHECK(trait_description(TraitAxis::CAUTIOUS_BOLD, -0.8f).find("cautious") != std::string::npos);
    CHECK(trait_description(TraitAxis::CAUTIOUS_BOLD, 0.3f).find("slightly") != std::string::npos);
}

// ============================================================================
// §H — Snapshot
// ============================================================================

TEST_CASE("§H-1 snapshot captures current state", "[personality]") {
    PersonalityDrift pd;
    pd.set_trait(TraitAxis::CAUTIOUS_BOLD, 0.5f);
    pd.apply_outcome({0.5f, 4, 0.3f, 0.5f});

    auto snap = pd.snapshot();
    CHECK(snap.traits[static_cast<size_t>(TraitAxis::CAUTIOUS_BOLD)] != 0.0f);
    CHECK(snap.total_events == 1);
}
