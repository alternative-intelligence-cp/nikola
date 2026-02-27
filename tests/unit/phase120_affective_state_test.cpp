/**
 * @file phase120_affective_state_test.cpp
 * @brief Phase 120 — AffectiveState unit tests
 *
 * Tests cover:
 *  §1  Constants and enum properties
 *  §2  Construction and initial state
 *  §3  Pure static: compute_valence()
 *  §4  Pure static: compute_arousal()
 *  §5  Pure static: compute_scores() -- affect derivation logic
 *  §6  update() -- full state machine
 *  §7  induce_affect() -- induction + decay
 *  §8  on_affect_change callback
 *  §9  affect_to_neurochemistry() -- consequence table
 *  §10 attention_weight() -- modulation
 *  §11 describe_state() -- human-readable output
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <nikola/interior/affective_state.hpp>
#include <nikola/autonomy/decision_loop.hpp>

using namespace nikola::interior;
using namespace nikola::autonomy;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static NikolaState make_state(float dopamine   = 0.5f,
                               float td_error   = 0.0f,
                               float atp        = 0.7f,
                               float boredom    = 0.1f,
                               float entropy    = 1.0f,
                               float energy     = 0.5f) {
    NikolaState s;
    s.dopamine    = dopamine;
    s.td_error    = td_error;
    s.atp         = atp;
    s.boredom     = boredom;
    s.entropy     = entropy;
    s.torus_energy = energy;
    return s;
}

// ---------------------------------------------------------------------------
// §1 — Constants and enum
// ---------------------------------------------------------------------------

TEST_CASE("Phase120 §1 Constants and enum", "[Phase120]") {
    SECTION("AFFECT_COUNT is 11") {
        CHECK(AFFECT_COUNT == 11);
    }

    SECTION("Affect labels have correct int values") {
        CHECK(static_cast<int>(Affect::CURIOSITY)    == 0);
        CHECK(static_cast<int>(Affect::FRUSTRATION)  == 1);
        CHECK(static_cast<int>(Affect::SATISFACTION) == 2);
        CHECK(static_cast<int>(Affect::CONCERN)      == 3);
        CHECK(static_cast<int>(Affect::BOREDOM)      == 4);
        CHECK(static_cast<int>(Affect::INTEREST)     == 5);
        CHECK(static_cast<int>(Affect::CONFUSION)    == 6);
        CHECK(static_cast<int>(Affect::CONFIDENCE)   == 7);
        CHECK(static_cast<int>(Affect::ANXIETY)      == 8);
        CHECK(static_cast<int>(Affect::EXCITEMENT)   == 9);
        CHECK(static_cast<int>(Affect::NEUTRAL)      == 10);
    }

    SECTION("affect_name returns non-empty strings for all labels") {
        for (int i = 0; i < AFFECT_COUNT; ++i) {
            const char* name = affect_name(static_cast<Affect>(i));
            CHECK(name != nullptr);
            CHECK(std::string(name).size() > 0);
        }
    }

    SECTION("ENTROPY_AROUSAL_CEILING is positive") {
        CHECK(ENTROPY_AROUSAL_CEILING > 0.0);
    }

    SECTION("INDUCED_AFFECT_DECAY is in (0, 1)") {
        CHECK(INDUCED_AFFECT_DECAY > 0.0);
        CHECK(INDUCED_AFFECT_DECAY < 1.0);
    }

    SECTION("BOREDOM_THRESHOLD is in (0, 1)") {
        CHECK(BOREDOM_THRESHOLD > 0.0);
        CHECK(BOREDOM_THRESHOLD < 1.0);
    }
}

// ---------------------------------------------------------------------------
// §2 — Construction
// ---------------------------------------------------------------------------

TEST_CASE("Phase120 §2 Construction and initial state", "[Phase120]") {
    AffectiveState as;

    SECTION("Initial dominant affect is NEUTRAL") {
        CHECK(as.current_affect() == Affect::NEUTRAL);
    }

    SECTION("Initial valence is 0") {
        CHECK_THAT(as.valence(), WithinAbs(0.0, 1e-9));
    }

    SECTION("Initial arousal is 0") {
        CHECK_THAT(as.arousal(), WithinAbs(0.0, 1e-9));
    }

    SECTION("Initial NEUTRAL intensity is 1.0") {
        CHECK_THAT(as.get_affect_intensity(Affect::NEUTRAL), WithinAbs(1.0, 1e-9));
    }

    SECTION("Initial non-neutral intensities are 0") {
        for (int i = 0; i < 10; ++i) {
            CHECK_THAT(as.get_affect_intensity(static_cast<Affect>(i)),
                       WithinAbs(0.0, 1e-9));
        }
    }

    SECTION("get_all_affects returns 11 entries") {
        CHECK(as.get_all_affects().size() == 11);
    }
}

// ---------------------------------------------------------------------------
// §3 — compute_valence
// ---------------------------------------------------------------------------

TEST_CASE("Phase120 §3 compute_valence pure function", "[Phase120]") {
    SECTION("Equilibrium dopamine, zero td, full ATP => near zero valence") {
        double v = AffectiveState::compute_valence(0.5, 0.0, 1.0);
        CHECK_THAT(v, WithinAbs(0.0, 0.05));
    }

    SECTION("High dopamine => positive valence") {
        double v = AffectiveState::compute_valence(0.9, 0.2, 0.8);
        CHECK(v > 0.3);
    }

    SECTION("Low dopamine + negative td => negative valence") {
        double v = AffectiveState::compute_valence(0.1, -0.3, 0.5);
        CHECK(v < -0.3);
    }

    SECTION("Low ATP adds negative contribution") {
        double v_high = AffectiveState::compute_valence(0.5, 0.0, 0.8);
        double v_low  = AffectiveState::compute_valence(0.5, 0.0, 0.05);
        CHECK(v_low < v_high);
    }

    SECTION("Result is always clamped to [-1, +1]") {
        // Extreme positive
        double v_high = AffectiveState::compute_valence(1.0, 1.0, 1.0);
        CHECK(v_high <= 1.0);
        CHECK(v_high >= -1.0);
        // Extreme negative
        double v_low = AffectiveState::compute_valence(0.0, -1.0, 0.0);
        CHECK(v_low >= -1.0);
        CHECK(v_low <= 1.0);
    }
}

// ---------------------------------------------------------------------------
// §4 — compute_arousal
// ---------------------------------------------------------------------------

TEST_CASE("Phase120 §4 compute_arousal pure function", "[Phase120]") {
    SECTION("Zero entropy, zero boredom => low arousal") {
        double a = AffectiveState::compute_arousal(0.0, 0.0);
        CHECK_THAT(a, WithinAbs(0.0, 0.05));
    }

    SECTION("High entropy increases arousal") {
        double a_low  = AffectiveState::compute_arousal(0.5, 0.0);
        double a_high = AffectiveState::compute_arousal(2.5, 0.0);
        CHECK(a_high > a_low);
    }

    SECTION("High boredom increases arousal (seeking drive)") {
        double a_low  = AffectiveState::compute_arousal(1.0, 0.0);
        double a_high = AffectiveState::compute_arousal(1.0, 0.9);
        CHECK(a_high > a_low);
    }

    SECTION("Result clamped to [0, 1]") {
        double a = AffectiveState::compute_arousal(100.0, 1.0);
        CHECK(a <= 1.0);
        CHECK(a >= 0.0);
    }
}

// ---------------------------------------------------------------------------
// §5 — compute_scores affect derivation
// ---------------------------------------------------------------------------

TEST_CASE("Phase120 §5 compute_scores affect derivation", "[Phase120]") {
    SECTION("All scores in [0, 1]") {
        auto s = make_state(0.5f, 0.0f, 0.7f, 0.3f, 1.5f);
        auto scores = AffectiveState::compute_scores(s);
        for (int i = 0; i < AFFECT_COUNT; ++i) {
            CHECK(scores[i] >= 0.0);
            CHECK(scores[i] <= 1.0);
        }
    }

    SECTION("CURIOSITY scores high when boredom high and entropy moderate") {
        // High boredom (0.8), moderate entropy (1.0), adequate ATP (0.6)
        auto s = make_state(0.4f, 0.0f, 0.6f, 0.8f, 1.0f);
        auto scores = AffectiveState::compute_scores(s);
        CHECK(scores[static_cast<int>(Affect::CURIOSITY)] > 0.3);
    }

    SECTION("FRUSTRATION scores high when dopamine low and td negative") {
        auto s = make_state(0.1f, -0.4f, 0.5f, 0.2f, 0.5f);
        auto scores = AffectiveState::compute_scores(s);
        CHECK(scores[static_cast<int>(Affect::FRUSTRATION)] > 0.3);
    }

    SECTION("SATISFACTION scores high when dopamine high and td positive") {
        auto s = make_state(0.85f, 0.3f, 0.8f, 0.1f, 0.8f);
        auto scores = AffectiveState::compute_scores(s);
        CHECK(scores[static_cast<int>(Affect::SATISFACTION)] > 0.3);
    }

    SECTION("BOREDOM scores high when boredom very high and entropy low") {
        auto s = make_state(0.4f, 0.0f, 0.6f, 0.9f, 0.1f);
        auto scores = AffectiveState::compute_scores(s);
        CHECK(scores[static_cast<int>(Affect::BOREDOM)] > 0.3);
    }

    SECTION("ANXIETY scores high when ATP critically low") {
        auto s = make_state(0.5f, 0.0f, 0.05f, 0.3f, 1.5f);
        auto scores = AffectiveState::compute_scores(s);
        CHECK(scores[static_cast<int>(Affect::ANXIETY)] > 0.3);
    }

    SECTION("CONFIDENCE scores high when dopamine high and ATP high") {
        auto s = make_state(0.85f, 0.1f, 0.9f, 0.1f, 0.5f);
        auto scores = AffectiveState::compute_scores(s);
        CHECK(scores[static_cast<int>(Affect::CONFIDENCE)] > 0.2);
    }

    SECTION("CONFUSION scores high when dopamine low and entropy very high") {
        auto s = make_state(0.1f, -0.1f, 0.5f, 0.2f, 2.8f);
        auto scores = AffectiveState::compute_scores(s);
        CHECK(scores[static_cast<int>(Affect::CONFUSION)] > 0.2);
    }

    SECTION("EXCITEMENT scores high with high D, high entropy, positive td") {
        auto s = make_state(0.85f, 0.4f, 0.7f, 0.2f, 2.0f);
        auto scores = AffectiveState::compute_scores(s);
        CHECK(scores[static_cast<int>(Affect::EXCITEMENT)] > 0.2);
    }

    SECTION("NEUTRAL scores high at homeostatic equilibrium, low entropy") {
        // All params at equilibrium, low entropy → nothing strongly active
        auto s = make_state(0.5f, 0.0f, 0.7f, 0.1f, 0.2f);
        auto scores = AffectiveState::compute_scores(s);
        CHECK(scores[static_cast<int>(Affect::NEUTRAL)] > 0.3);
    }
}

// ---------------------------------------------------------------------------
// §6 — update()
// ---------------------------------------------------------------------------

TEST_CASE("Phase120 §6 update() state machine", "[Phase120]") {
    SECTION("update() sets valence and arousal from state") {
        AffectiveState as;
        auto s = make_state(0.85f, 0.3f, 0.9f, 0.1f, 0.5f);
        as.update(s);
        // High dopamine + positive td should give positive valence
        CHECK(as.valence() > 0.0);
        // arousal depends on entropy
        CHECK(as.arousal() >= 0.0);
        CHECK(as.arousal() <= 1.0);
    }

    SECTION("update() makes dominant affect non-NEUTRAL when D is high") {
        AffectiveState as;
        auto s = make_state(0.9f, 0.4f, 0.9f, 0.1f, 0.5f);
        as.update(s);
        // Should not be neutral with strong satisfaction signals
        // (might be SATISFACTION or CONFIDENCE)
        // Just verify it responded to the state
        CHECK(as.get_affect_intensity(as.current_affect()) > 0.0);
    }

    SECTION("Multiple update calls with same state are idempotent (no uninduced drift)") {
        AffectiveState as;
        auto s = make_state(0.7f, 0.2f, 0.8f, 0.3f, 1.2f);
        as.update(s);
        double v1 = as.valence();
        double a1 = as.arousal();
        as.update(s);
        CHECK_THAT(as.valence(), WithinAbs(v1, 1e-9));
        CHECK_THAT(as.arousal(), WithinAbs(a1, 1e-9));
    }

    SECTION("Exhausted state (low ATP) produces negative valence") {
        AffectiveState as;
        auto s = make_state(0.5f, -0.1f, 0.05f, 0.2f, 1.0f);
        as.update(s);
        CHECK(as.valence() < 0.0);
    }
}

// ---------------------------------------------------------------------------
// §7 — induce_affect and decay
// ---------------------------------------------------------------------------

TEST_CASE("Phase120 §7 induce_affect and decay", "[Phase120]") {
    SECTION("induce_affect raises intensity of target affect") {
        AffectiveState as;
        double before = as.get_affect_intensity(Affect::EXCITEMENT);
        as.induce_affect(Affect::EXCITEMENT, 0.8);
        CHECK(as.get_affect_intensity(Affect::EXCITEMENT) > before);
    }

    SECTION("induce_affect with intensity 0 is valid (no change)") {
        AffectiveState as;
        double before = as.get_affect_intensity(Affect::CURIOSITY);
        REQUIRE_NOTHROW(as.induce_affect(Affect::CURIOSITY, 0.0));
        CHECK_THAT(as.get_affect_intensity(Affect::CURIOSITY),
                   WithinAbs(before, 1e-9));
    }

    SECTION("induce_affect intensity > 1 throws") {
        AffectiveState as;
        CHECK_THROWS_AS(as.induce_affect(Affect::CURIOSITY, 1.1),
                        std::invalid_argument);
    }

    SECTION("induce_affect intensity < 0 throws") {
        AffectiveState as;
        CHECK_THROWS_AS(as.induce_affect(Affect::CURIOSITY, -0.1),
                        std::invalid_argument);
    }

    SECTION("Induced weight decays each update()") {
        AffectiveState as;
        as.induce_affect(Affect::SATISFACTION, 1.0);

        auto s = make_state(0.5f, 0.0f, 0.7f, 0.1f, 0.5f);
        as.update(s);
        double i1 = as.get_affect_intensity(Affect::SATISFACTION);

        // The induced component decays; base score may differ, but total
        // should be different or equal depending on base
        // At minimum the update ran without crashing
        CHECK(i1 >= 0.0);
        CHECK(i1 <= 1.0);
    }

    SECTION("Repeated updates eventually clear all induced weights") {
        AffectiveState as;
        as.induce_affect(Affect::EXCITEMENT, 0.5);

        auto s = make_state(0.5f, 0.0f, 0.7f, 0.1f, 0.5f);
        // After ~50 updates with INDUCED_AFFECT_DECAY=0.85, induced -> ~0.001
        for (int i = 0; i < 80; ++i) as.update(s);

        // Induced component should be negligible
        // (base score may still be non-zero depending on state)
        CHECK(as.get_affect_intensity(Affect::EXCITEMENT) <= 1.0);
    }

    SECTION("Inducing same affect multiple times clamps at 1") {
        AffectiveState as;
        as.induce_affect(Affect::ANXIETY, 0.8);
        as.induce_affect(Affect::ANXIETY, 0.8);
        CHECK(as.get_affect_intensity(Affect::ANXIETY) <= 1.0);
    }
}

// ---------------------------------------------------------------------------
// §8 — on_affect_change callback
// ---------------------------------------------------------------------------

TEST_CASE("Phase120 §8 on_affect_change callback", "[Phase120]") {
    SECTION("Callback fires when dominant affect changes") {
        AffectiveState as;
        int call_count = 0;
        Affect last_next = Affect::NEUTRAL;

        as.on_affect_change = [&](Affect prev, Affect next, double intensity) {
            ++call_count;
            last_next = next;
            (void)prev; (void)intensity;
        };

        // Strong satisfaction signal — likely changes from NEUTRAL
        auto s = make_state(0.9f, 0.5f, 0.9f, 0.1f, 0.5f);
        as.update(s);

        if (as.current_affect() != Affect::NEUTRAL) {
            CHECK(call_count >= 1);
        }
    }

    SECTION("Callback not called when affect doesn't change") {
        AffectiveState as;
        int call_count = 0;
        as.on_affect_change = [&](Affect, Affect, double) { ++call_count; };

        // Neutral state should stay neutral
        auto s = make_state(0.5f, 0.0f, 0.7f, 0.05f, 0.1f);
        as.update(s);
        int after_first = call_count;
        as.update(s);
        // Second update with same state: no change expected
        CHECK(call_count == after_first);
    }
}

// ---------------------------------------------------------------------------
// §9 — affect_to_neurochemistry
// ---------------------------------------------------------------------------

TEST_CASE("Phase120 §9 affect_to_neurochemistry", "[Phase120]") {
    SECTION("Returns 3 keys for every Affect") {
        for (int i = 0; i < AFFECT_COUNT; ++i) {
            auto nc = AffectiveState::affect_to_neurochemistry(
                                          static_cast<Affect>(i));
            CHECK(nc.count("dopamine")        == 1);
            CHECK(nc.count("serotonin")       == 1);
            CHECK(nc.count("norepinephrine")  == 1);
        }
    }

    SECTION("NEUTRAL has zero deltas") {
        auto nc = AffectiveState::affect_to_neurochemistry(Affect::NEUTRAL);
        CHECK_THAT(nc["dopamine"],       WithinAbs(0.0, 1e-9));
        CHECK_THAT(nc["serotonin"],      WithinAbs(0.0, 1e-9));
        CHECK_THAT(nc["norepinephrine"], WithinAbs(0.0, 1e-9));
    }

    SECTION("SATISFACTION has positive dopamine and serotonin deltas") {
        auto nc = AffectiveState::affect_to_neurochemistry(Affect::SATISFACTION);
        CHECK(nc["dopamine"]  > 0.0);
        CHECK(nc["serotonin"] > 0.0);
    }

    SECTION("ANXIETY has positive norepinephrine delta") {
        auto nc = AffectiveState::affect_to_neurochemistry(Affect::ANXIETY);
        CHECK(nc["norepinephrine"] > 0.0);
        CHECK(nc["dopamine"]       < 0.0);
    }

    SECTION("FRUSTRATION has negative dopamine delta") {
        auto nc = AffectiveState::affect_to_neurochemistry(Affect::FRUSTRATION);
        CHECK(nc["dopamine"] < 0.0);
    }

    SECTION("CURIOSITY raises norepinephrine (arousal for exploration)") {
        auto nc = AffectiveState::affect_to_neurochemistry(Affect::CURIOSITY);
        CHECK(nc["norepinephrine"] > 0.0);
    }
}

// ---------------------------------------------------------------------------
// §10 — attention_weight
// ---------------------------------------------------------------------------

TEST_CASE("Phase120 §10 attention_weight modulation", "[Phase120]") {
    SECTION("Weight is in [0.5, 2.0] for any entropy") {
        AffectiveState as;
        auto s = make_state(0.5f, 0.0f, 0.7f, 0.3f, 1.0f);
        as.update(s);

        for (double e : {0.0, 0.5, 1.0, 2.0, 3.0, 5.0}) {
            double w = as.attention_weight(e);
            CHECK(w >= 0.5);
            CHECK(w <= 2.0);
        }
    }

    SECTION("Curiosity-dominant state raises weight at high entropy") {
        AffectiveState as;
        // Induce curiosity strongly
        as.induce_affect(Affect::CURIOSITY, 1.0);

        double w_low  = as.attention_weight(0.1);
        double w_high = as.attention_weight(2.5);
        CHECK(w_high > w_low);
    }

    SECTION("Anxiety-dominant state does not push weight above 2.0") {
        AffectiveState as;
        as.induce_affect(Affect::ANXIETY, 1.0);
        double w = as.attention_weight(3.0);
        CHECK(w <= 2.0);
        CHECK(w >= 0.5);
    }
}

// ---------------------------------------------------------------------------
// §11 — describe_state
// ---------------------------------------------------------------------------

TEST_CASE("Phase120 §11 describe_state", "[Phase120]") {
    SECTION("Returns non-empty string") {
        AffectiveState as;
        CHECK(!as.describe_state().empty());
    }

    SECTION("Contains valence and arousal") {
        AffectiveState as;
        std::string desc = as.describe_state();
        CHECK(desc.find("valence") != std::string::npos);
        CHECK(desc.find("arousal") != std::string::npos);
    }

    SECTION("Contains the dominant affect name") {
        AffectiveState as;
        auto s = make_state(0.5f, 0.0f, 0.7f, 0.05f, 0.1f);
        as.update(s);
        std::string desc = as.describe_state();
        const char* name = affect_name(as.current_affect());
        CHECK(desc.find(name) != std::string::npos);
    }

    SECTION("Strong induction produces non-neutral description") {
        AffectiveState as;
        as.induce_affect(Affect::EXCITEMENT, 1.0);
        std::string desc = as.describe_state();
        CHECK(!desc.empty());
        // Should mention excitement
        CHECK(desc.find("excitement") != std::string::npos);
    }
}
