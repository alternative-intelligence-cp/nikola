/**
 * @file phase124_autobiography_test.cpp
 * @brief Phase 124 — AutobiographicalMemory unit tests
 *
 * §1  Constants
 * §2  text_matches() static
 * §3  affect_label() static
 * §4  TickRange::contains()
 * §5  record_event() — storage and FIFO eviction
 * §6  recall_period()
 * §7  recall_by_query()
 * §8  find_by_tag()
 * §9  get_most_significant()
 * §10 generate_narrative()
 * §11 Values — update_value(), get_values(), dominant_value()
 * §12 Skills — update_skill(), get_skills(), best_skill(), success_rate()
 * §13 get_identity()
 * §14 stats()
 * §15 on_event_recorded callback
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <nikola/interior/autobiography.hpp>
#include <nikola/autonomy/decision_loop.hpp>

using namespace nikola::interior;
using namespace nikola::autonomy;
using Catch::Matchers::WithinAbs;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static NikolaState make_state(float dopamine = 0.5f,
                               float atp      = 0.7f,
                               float entropy  = 0.8f) {
    NikolaState s;
    s.dopamine = dopamine;
    s.atp      = atp;
    s.entropy  = entropy;
    return s;
}

// ---------------------------------------------------------------------------
// §1 — Constants
// ---------------------------------------------------------------------------

TEST_CASE("Phase124 §1 constants", "[Phase124]") {
    CHECK(AUTOBIOGRAPHY_MAX_EVENTS       > 0u);
    CHECK(AUTOBIOGRAPHY_SKILL_LEARN_RATE > 0.0);
    CHECK(AUTOBIOGRAPHY_SKILL_LEARN_RATE < 1.0);
    CHECK(AUTOBIOGRAPHY_SKILL_DECAY      > 0.0);
    CHECK(AUTOBIOGRAPHY_SKILL_DECAY      < 1.0);
    CHECK(AUTOBIOGRAPHY_VALUE_LEARN_RATE > 0.0);
    CHECK(AUTOBIOGRAPHY_VALUE_LEARN_RATE < 1.0);
    CHECK(AUTOBIOGRAPHY_SIGNIFICANCE_MIN >= 0.0);
    CHECK(AUTOBIOGRAPHY_SIGNIFICANCE_MIN <  1.0);
    CHECK(AUTOBIOGRAPHY_TOP_N            > 0u);
}

// ---------------------------------------------------------------------------
// §2 — text_matches()
// ---------------------------------------------------------------------------

TEST_CASE("Phase124 §2 text_matches()", "[Phase124]") {
    SECTION("keyword found in text -> true") {
        CHECK(AutobiographicalMemory::text_matches("reward spike detected", "reward"));
    }

    SECTION("case insensitive") {
        CHECK(AutobiographicalMemory::text_matches("Reward Spike", "reward"));
    }

    SECTION("one of multiple keywords matches -> true") {
        CHECK(AutobiographicalMemory::text_matches("network failure", "reward failure"));
    }

    SECTION("no keyword found -> false") {
        CHECK(!AutobiographicalMemory::text_matches("the sky is blue", "reward"));
    }

    SECTION("empty query -> false") {
        CHECK(!AutobiographicalMemory::text_matches("anything", ""));
    }
}

// ---------------------------------------------------------------------------
// §3 — affect_label()
// ---------------------------------------------------------------------------

TEST_CASE("Phase124 §3 affect_label()", "[Phase124]") {
    SECTION("each affect has a non-empty label") {
        for (int i = 0; i <= static_cast<int>(Affect::NEUTRAL); ++i) {
            auto lbl = AutobiographicalMemory::affect_label(static_cast<Affect>(i));
            CHECK(!lbl.empty());
        }
    }

    SECTION("NEUTRAL -> 'neutral'") {
        CHECK(AutobiographicalMemory::affect_label(Affect::NEUTRAL) == "neutral");
    }

    SECTION("CURIOSITY -> 'curious'") {
        CHECK(AutobiographicalMemory::affect_label(Affect::CURIOSITY) == "curious");
    }

    SECTION("CONFIDENCE -> 'confident'") {
        CHECK(AutobiographicalMemory::affect_label(Affect::CONFIDENCE) == "confident");
    }
}

// ---------------------------------------------------------------------------
// §4 — TickRange
// ---------------------------------------------------------------------------

TEST_CASE("Phase124 §4 TickRange::contains()", "[Phase124]") {
    SECTION("tick in range -> true") {
        TickRange r{10, 50};
        CHECK(r.contains(10));
        CHECK(r.contains(30));
        CHECK(r.contains(50));
    }

    SECTION("tick outside range -> false") {
        TickRange r{10, 50};
        CHECK(!r.contains(9));
        CHECK(!r.contains(51));
    }

    SECTION("default range contains everything") {
        TickRange r;
        CHECK(r.contains(0));
        CHECK(r.contains(UINT64_MAX));
    }
}

// ---------------------------------------------------------------------------
// §5 — record_event()
// ---------------------------------------------------------------------------

TEST_CASE("Phase124 §5 record_event()", "[Phase124]") {
    SECTION("event count increments") {
        AutobiographicalMemory m;
        m.record_event("first event", make_state());
        CHECK(m.event_count() == 1);
        m.record_event("second event", make_state());
        CHECK(m.event_count() == 2);
    }

    SECTION("description stored correctly") {
        AutobiographicalMemory m;
        m.record_event("milestone reached", make_state());
        CHECK(m.events().back().description == "milestone reached");
    }

    SECTION("dominant_affect stored") {
        AutobiographicalMemory m;
        m.record_event("great success", make_state(), Affect::SATISFACTION, 0.9);
        CHECK(m.events().back().dominant_affect == Affect::SATISFACTION);
    }

    SECTION("significance clamped to [0, 1]") {
        AutobiographicalMemory m;
        m.record_event("over", make_state(), Affect::NEUTRAL, 2.5);
        m.record_event("under", make_state(), Affect::NEUTRAL, -1.0);
        CHECK_THAT(m.events()[0].significance, WithinAbs(1.0, 1e-9));
        CHECK_THAT(m.events()[1].significance, WithinAbs(0.0, 1e-9));
    }

    SECTION("tags stored") {
        AutobiographicalMemory m;
        m.record_event("first win", make_state(), Affect::SATISFACTION, 0.8,
                        {"first_success", "milestone"});
        REQUIRE(m.events().back().tags.size() == 2);
        CHECK(m.events().back().tags[0] == "first_success");
        CHECK(m.events().back().tags[1] == "milestone");
    }

    SECTION("FIFO: count never exceeds AUTOBIOGRAPHY_MAX_EVENTS") {
        AutobiographicalMemory m;
        for (size_t i = 0; i < AUTOBIOGRAPHY_MAX_EVENTS + 5; ++i)
            m.record_event("e", make_state());
        CHECK(m.event_count() <= AUTOBIOGRAPHY_MAX_EVENTS);
    }

    SECTION("FIFO: newest events remain after eviction") {
        AutobiographicalMemory m;
        for (size_t i = 0; i < AUTOBIOGRAPHY_MAX_EVENTS; ++i)
            m.record_event("old", make_state());
        m.record_event("newest", make_state());
        CHECK(m.events().back().description == "newest");
    }
}

// ---------------------------------------------------------------------------
// §6 — recall_period()
// ---------------------------------------------------------------------------

TEST_CASE("Phase124 §6 recall_period()", "[Phase124]") {
    AutobiographicalMemory m;
    m.record_event("tick0", make_state());  // tick 0
    m.record_event("tick1", make_state());  // tick 1
    m.record_event("tick2", make_state());  // tick 2
    m.record_event("tick3", make_state());  // tick 3

    SECTION("range covers all -> returns all") {
        auto r = m.recall_period({0, 3});
        CHECK(r.size() == 4);
    }

    SECTION("narrow range returns subset") {
        auto r = m.recall_period({1, 2});
        CHECK(r.size() == 2);
    }

    SECTION("range with no events -> empty") {
        auto r = m.recall_period({100, 200});
        CHECK(r.empty());
    }

    SECTION("returned pointers are non-null") {
        auto r = m.recall_period({0, 10});
        for (const auto* e : r) CHECK(e != nullptr);
    }
}

// ---------------------------------------------------------------------------
// §7 — recall_by_query()
// ---------------------------------------------------------------------------

TEST_CASE("Phase124 §7 recall_by_query()", "[Phase124]") {
    AutobiographicalMemory m;
    m.record_event("reward spike observed",   make_state());
    m.record_event("atp depletion detected",  make_state());
    m.record_event("reward loop completed",   make_state());

    SECTION("keyword matches description") {
        auto r = m.recall_by_query("reward");
        CHECK(r.size() == 2);
    }

    SECTION("keyword matches tag") {
        AutobiographicalMemory m2;
        m2.record_event("something happened", make_state(), Affect::NEUTRAL, 0.5,
                         {"critical_failure"});
        auto r = m2.recall_by_query("critical");
        CHECK(r.size() == 1);
    }

    SECTION("empty query -> empty result") {
        CHECK(m.recall_by_query("").empty());
    }

    SECTION("max limits result") {
        auto r = m.recall_by_query("reward", 1);
        CHECK(r.size() == 1);
    }
}

// ---------------------------------------------------------------------------
// §8 — find_by_tag()
// ---------------------------------------------------------------------------

TEST_CASE("Phase124 §8 find_by_tag()", "[Phase124]") {
    AutobiographicalMemory m;
    m.record_event("event A", make_state(), Affect::NEUTRAL, 0.8, {"milestone"});
    m.record_event("event B", make_state(), Affect::NEUTRAL, 0.6, {"error"});
    m.record_event("event C", make_state(), Affect::NEUTRAL, 0.7, {"milestone", "first"});

    SECTION("finds events with exact tag") {
        auto r = m.find_by_tag("milestone");
        CHECK(r.size() == 2);
    }

    SECTION("tag not present -> empty") {
        CHECK(m.find_by_tag("nonexistent").empty());
    }

    SECTION("each event appears only once even if tag present multiple times") {
        AutobiographicalMemory m2;
        m2.record_event("x", make_state(), Affect::NEUTRAL, 0.5, {"t", "t"});
        auto r = m2.find_by_tag("t");
        // Should dedup — event listed once
        size_t count = 0;
        for (const auto* e : r) if (e->description == "x") ++count;
        CHECK(count == 1);
    }
}

// ---------------------------------------------------------------------------
// §9 — get_most_significant()
// ---------------------------------------------------------------------------

TEST_CASE("Phase124 §9 get_most_significant()", "[Phase124]") {
    AutobiographicalMemory m;
    m.record_event("low",  make_state(), Affect::NEUTRAL, 0.1);
    m.record_event("mid",  make_state(), Affect::NEUTRAL, 0.6);
    m.record_event("high", make_state(), Affect::NEUTRAL, 0.9);

    SECTION("returns events sorted by significance descending") {
        auto r = m.get_most_significant(3);
        REQUIRE(r.size() >= 1);
        CHECK(r[0]->significance >= AUTOBIOGRAPHY_SIGNIFICANCE_MIN);
        for (size_t i = 1; i < r.size(); ++i)
            CHECK(r[i-1]->significance >= r[i]->significance);
    }

    SECTION("count parameter limits result") {
        auto r = m.get_most_significant(1);
        CHECK(r.size() == 1);
    }

    SECTION("events below SIGNIFICANCE_MIN not included") {
        auto r = m.get_most_significant(10);
        for (const auto* e : r)
            CHECK(e->significance >= AUTOBIOGRAPHY_SIGNIFICANCE_MIN);
    }
}

// ---------------------------------------------------------------------------
// §10 — generate_narrative()
// ---------------------------------------------------------------------------

TEST_CASE("Phase124 §10 generate_narrative()", "[Phase124]") {
    SECTION("no events -> placeholder string") {
        AutobiographicalMemory m;
        std::string n = m.generate_narrative();
        CHECK(!n.empty());
        CHECK(n.find("No events") != std::string::npos);
    }

    SECTION("with events -> contains event count") {
        AutobiographicalMemory m;
        m.record_event("first thing", make_state(), Affect::CURIOSITY, 0.8);
        m.record_event("second thing", make_state(), Affect::SATISFACTION, 0.5);
        std::string n = m.generate_narrative();
        CHECK(n.find("2") != std::string::npos);
    }

    SECTION("with range filter -> only includes range events") {
        AutobiographicalMemory m;
        for (int i = 0; i < 5; ++i)
            m.record_event("event " + std::to_string(i), make_state());
        TickRange r{2, 3};
        std::string n = m.generate_narrative(&r);
        CHECK(!n.empty());
        // Should reflect 2 events
        CHECK(n.find("2") != std::string::npos);
    }

    SECTION("contains affect label for significant events") {
        AutobiographicalMemory m;
        m.record_event("great win", make_state(), Affect::SATISFACTION, 0.9);
        std::string n = m.generate_narrative();
        CHECK(n.find("satisfied") != std::string::npos);
    }
}

// ---------------------------------------------------------------------------
// §11 — Values
// ---------------------------------------------------------------------------

TEST_CASE("Phase124 §11 values", "[Phase124]") {
    SECTION("update_value creates new value at 0.5 baseline") {
        AutobiographicalMemory m;
        m.update_value("curiosity", 1.0);
        auto vals = m.get_values();
        REQUIRE(vals.count("curiosity"));
        CHECK(vals.at("curiosity") > 0.5);
    }

    SECTION("repeated updates increase importance") {
        AutobiographicalMemory m;
        m.update_value("safety", 1.0);
        double v1 = m.get_values().at("safety");
        m.update_value("safety", 1.0);
        double v2 = m.get_values().at("safety");
        CHECK(v2 > v1);
    }

    SECTION("negative delta decreases importance") {
        AutobiographicalMemory m;
        m.update_value("safety", -1.0);
        auto vals = m.get_values();
        CHECK(vals.at("safety") < 0.5);
    }

    SECTION("importance clamped to [0, 1]") {
        AutobiographicalMemory m;
        for (int i = 0; i < 50; ++i) m.update_value("x", 10.0);
        CHECK(m.get_values().at("x") <= 1.0);
        for (int i = 0; i < 50; ++i) m.update_value("x", -10.0);
        CHECK(m.get_values().at("x") >= 0.0);
    }

    SECTION("value_count() reflects unique values") {
        AutobiographicalMemory m;
        m.update_value("a", 1.0);
        m.update_value("b", 1.0);
        m.update_value("a", 1.0); // update existing
        CHECK(m.value_count() == 2);
    }

    SECTION("dominant_value() returns highest-importance name") {
        AutobiographicalMemory m;
        m.update_value("low",  -1.0); // < 0.5
        m.update_value("high", 1.0);  // > 0.5
        CHECK(m.dominant_value() == "high");
    }

    SECTION("dominant_value() returns empty when no values") {
        AutobiographicalMemory m;
        CHECK(m.dominant_value().empty());
    }
}

// ---------------------------------------------------------------------------
// §12 — Skills
// ---------------------------------------------------------------------------

TEST_CASE("Phase124 §12 skills", "[Phase124]") {
    SECTION("update_skill creates new skill at 0.0 proficiency") {
        AutobiographicalMemory m;
        m.update_skill("debugging", true);
        REQUIRE(m.skill_count() == 1);
        CHECK(m.get_skills()[0].proficiency > 0.0);
    }

    SECTION("success increases proficiency by SKILL_LEARN_RATE") {
        AutobiographicalMemory m;
        m.update_skill("math", true);
        CHECK_THAT(m.get_skills()[0].proficiency,
                   WithinAbs(AUTOBIOGRAPHY_SKILL_LEARN_RATE, 1e-9));
    }

    SECTION("failure decreases proficiency by SKILL_DECAY") {
        AutobiographicalMemory m;
        m.update_skill("math", true);  // start at learn_rate
        m.update_skill("math", false); // subtract decay
        double expected = AUTOBIOGRAPHY_SKILL_LEARN_RATE - AUTOBIOGRAPHY_SKILL_DECAY;
        CHECK_THAT(m.get_skills()[0].proficiency, WithinAbs(expected, 1e-9));
    }

    SECTION("proficiency clamped to [0, 1]") {
        AutobiographicalMemory m;
        for (int i = 0; i < 50; ++i) m.update_skill("x", true);
        CHECK(m.get_skills()[0].proficiency <= 1.0);
        for (int i = 0; i < 50; ++i) m.update_skill("x", false);
        CHECK(m.get_skills()[0].proficiency >= 0.0);
    }

    SECTION("practice_count and success_count tracked") {
        AutobiographicalMemory m;
        m.update_skill("coding", true);
        m.update_skill("coding", true);
        m.update_skill("coding", false);
        auto skills_snap = m.get_skills();
        REQUIRE(!skills_snap.empty());
        const auto& s = skills_snap[0];
        CHECK(s.practice_count == 3);
        CHECK(s.success_count  == 2);
    }

    SECTION("success_rate() = success / practice") {
        AutobiographicalMemory m;
        m.update_skill("coding", true);
        m.update_skill("coding", false);
        CHECK_THAT(m.get_skills()[0].success_rate(), WithinAbs(0.5, 1e-9));
    }

    SECTION("skill_count() uses unique names") {
        AutobiographicalMemory m;
        m.update_skill("a", true);
        m.update_skill("b", true);
        m.update_skill("a", true); // update existing
        CHECK(m.skill_count() == 2);
    }

    SECTION("best_skill() returns highest-proficiency name") {
        AutobiographicalMemory m;
        m.update_skill("low_skill", true); // proficiency = learn_rate
        for (int i = 0; i < 5; ++i)
            m.update_skill("high_skill", true);
        CHECK(m.best_skill() == "high_skill");
    }

    SECTION("best_skill() returns empty when no skills") {
        AutobiographicalMemory m;
        CHECK(m.best_skill().empty());
    }

    SECTION("last_tick stored on update") {
        AutobiographicalMemory m;
        m.update_skill("coding", true, 42);
        CHECK(m.get_skills()[0].last_tick == 42);
    }
}

// ---------------------------------------------------------------------------
// §13 — get_identity()
// ---------------------------------------------------------------------------

TEST_CASE("Phase124 §13 get_identity()", "[Phase124]") {
    SECTION("no events uses neutral affect") {
        AutobiographicalMemory m;
        std::string id = m.get_identity();
        CHECK(!id.empty());
        CHECK(id.find("neutral") != std::string::npos);
    }

    SECTION("includes event count") {
        AutobiographicalMemory m;
        m.record_event("e1", make_state(), Affect::CURIOSITY, 0.7);
        m.record_event("e2", make_state(), Affect::CURIOSITY, 0.6);
        std::string id = m.get_identity();
        CHECK(id.find("2") != std::string::npos);
    }

    SECTION("includes dominant value when set") {
        AutobiographicalMemory m;
        m.update_value("learning", 1.0);
        std::string id = m.get_identity();
        CHECK(id.find("learning") != std::string::npos);
    }

    SECTION("includes best skill when set") {
        AutobiographicalMemory m;
        m.update_skill("reasoning", true);
        std::string id = m.get_identity();
        CHECK(id.find("reasoning") != std::string::npos);
    }

    SECTION("reflects dominant affect across events") {
        AutobiographicalMemory m;
        for (int i = 0; i < 3; ++i)
            m.record_event("curiosity event", make_state(), Affect::CURIOSITY, 0.5);
        m.record_event("one neutral", make_state(), Affect::NEUTRAL, 0.5);
        std::string id = m.get_identity();
        CHECK(id.find("curious") != std::string::npos);
    }
}

// ---------------------------------------------------------------------------
// §14 — stats()
// ---------------------------------------------------------------------------

TEST_CASE("Phase124 §14 stats()", "[Phase124]") {
    SECTION("fresh instance -> all zeros") {
        AutobiographicalMemory m;
        auto s = m.stats();
        CHECK(s.total_events  == 0);
        CHECK(s.total_skills  == 0);
        CHECK(s.total_values  == 0);
        CHECK_THAT(s.mean_significance, WithinAbs(0.0, 1e-9));
        CHECK(s.most_common_affect == Affect::NEUTRAL);
    }

    SECTION("total_events and mean_significance reflect recorded events") {
        AutobiographicalMemory m;
        m.record_event("e1", make_state(), Affect::NEUTRAL, 0.4);
        m.record_event("e2", make_state(), Affect::NEUTRAL, 0.8);
        auto s = m.stats();
        CHECK(s.total_events == 2);
        CHECK_THAT(s.mean_significance, WithinAbs(0.6, 1e-9));
    }

    SECTION("most_common_affect is correct") {
        AutobiographicalMemory m;
        m.record_event("a", make_state(), Affect::EXCITEMENT, 0.5);
        m.record_event("b", make_state(), Affect::EXCITEMENT, 0.5);
        m.record_event("c", make_state(), Affect::CURIOSITY,  0.5);
        auto s = m.stats();
        CHECK(s.most_common_affect == Affect::EXCITEMENT);
    }

    SECTION("total_skills and total_values tracked") {
        AutobiographicalMemory m;
        m.update_skill("s1", true);
        m.update_skill("s2", true);
        m.update_value("v1", 1.0);
        auto s = m.stats();
        CHECK(s.total_skills == 2);
        CHECK(s.total_values == 1);
    }
}

// ---------------------------------------------------------------------------
// §15 — on_event_recorded callback
// ---------------------------------------------------------------------------

TEST_CASE("Phase124 §15 event callback", "[Phase124]") {
    SECTION("callback fires after record_event") {
        AutobiographicalMemory m;
        bool fired = false;
        std::string captured;
        m.on_event_recorded([&](const LifeEvent& e) {
            fired    = true;
            captured = e.description;
        });
        m.record_event("test event", make_state());
        CHECK(fired);
        CHECK(captured == "test event");
    }

    SECTION("no callback -> record_event does not crash") {
        AutobiographicalMemory m;
        REQUIRE_NOTHROW(m.record_event("silent", make_state()));
    }
}
