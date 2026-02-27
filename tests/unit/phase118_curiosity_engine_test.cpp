/**
 * @file phase118_curiosity_engine_test.cpp
 * @brief Phase 118 — CuriosityEngine intrinsic-motivation implementation tests.
 *
 * TorusManifold is forward-declared in curiosity.hpp but never fully defined
 * in production headers (it is a future integration hook).  We define a
 * minimal empty stub here in the same namespace so that the CuriosityEngine
 * methods that accept it by reference can be exercised without dereferencing
 * it.
 *
 * Test map  (22 test cases)
 * ─────────────────────────
 *  [P118/rate]       exploration_rate default, set/get, clamping
 *  [P118/gaps]       register_gap: add, merge-uncertainty, merge-memories
 *  [P118/gain]       measure_information_gain: unknown domain, known domain, saturation
 *  [P118/questions]  generate_questions: count=0, padded, ranked, callback fires
 *  [P118/pursue]     pursue_interest: empty, returns true, tracks history, reduces uncertainty
 *  [P118/identify]   identify_knowledge_gaps: sorted by descending uncertainty
 *  [P118/interest]   measure_interestingness: unknown > known, familiarity decay
 *  [P118/learning]   start/stop autonomous learning
 *  [P118/stats]      get_stats keys, accumulation
 */

// ── Minimal TorusManifold stub ────────────────────────────────────────────────
// Satisfies the forward declaration in nikola::interior so we can pass a real
// reference to CuriosityEngine methods.
namespace nikola::interior {
class TorusManifold {};   // empty; never dereferenced by CuriosityEngine
}

#include <nikola/interior/curiosity.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <string>
#include <vector>

using nikola::interior::CuriosityEngine;
using nikola::interior::KnowledgeGap;
using nikola::interior::Question;
using nikola::interior::TorusManifold;

// Shared torus stub for all tests
static TorusManifold torus;

// Helper: build a KnowledgeGap
static KnowledgeGap make_gap(const std::string& domain, double uncertainty,
                              int query_count = 0,
                              std::vector<std::string> memories = {})
{
    KnowledgeGap g;
    g.domain           = domain;
    g.uncertainty      = uncertainty;
    g.query_count      = query_count;
    g.related_memories = std::move(memories);
    return g;
}

// ─────────────────────────────────────────────────────────────────────────────
// [P118/rate]  exploration_rate
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase118 exploration rate defaults to 0.3", "[Phase118][P118/rate]")
{
    CuriosityEngine ce;
    REQUIRE_THAT(ce.get_exploration_rate(),
                 Catch::Matchers::WithinAbs(0.3, 1e-9));
}

TEST_CASE("Phase118 set_exploration_rate round-trips", "[Phase118][P118/rate]")
{
    CuriosityEngine ce;
    ce.set_exploration_rate(0.75);
    REQUIRE_THAT(ce.get_exploration_rate(),
                 Catch::Matchers::WithinAbs(0.75, 1e-9));
}

TEST_CASE("Phase118 set_exploration_rate clamps below 0", "[Phase118][P118/rate]")
{
    CuriosityEngine ce;
    ce.set_exploration_rate(-5.0);
    REQUIRE_THAT(ce.get_exploration_rate(), Catch::Matchers::WithinAbs(0.0, 1e-9));
}

TEST_CASE("Phase118 set_exploration_rate clamps above 1", "[Phase118][P118/rate]")
{
    CuriosityEngine ce;
    ce.set_exploration_rate(99.0);
    REQUIRE_THAT(ce.get_exploration_rate(), Catch::Matchers::WithinAbs(1.0, 1e-9));
}

// ─────────────────────────────────────────────────────────────────────────────
// [P118/gaps]  register_gap
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase118 register_gap adds new gap; visible in identify_knowledge_gaps",
          "[Phase118][P118/gaps]")
{
    CuriosityEngine ce;
    ce.register_gap(make_gap("quantum-gravity", 0.9));
    auto gaps = ce.identify_knowledge_gaps(torus);
    REQUIRE(gaps.size() == 1);
    CHECK(gaps[0].domain == "quantum-gravity");
}

TEST_CASE("Phase118 register_gap merges uncertainty on duplicate domain",
          "[Phase118][P118/gaps]")
{
    CuriosityEngine ce;
    ce.register_gap(make_gap("topology", 0.8));
    ce.register_gap(make_gap("topology", 0.4));
    auto gaps = ce.identify_knowledge_gaps(torus);
    REQUIRE(gaps.size() == 1);
    // (0.8 + 0.4) / 2 = 0.6
    CHECK_THAT(gaps[0].uncertainty, Catch::Matchers::WithinAbs(0.6, 1e-9));
}

TEST_CASE("Phase118 register_gap unions related_memories on duplicate",
          "[Phase118][P118/gaps]")
{
    CuriosityEngine ce;
    ce.register_gap(make_gap("biology", 0.7, 0, {"cells"}));
    ce.register_gap(make_gap("biology", 0.5, 0, {"dna", "cells"}));
    auto gaps = ce.identify_knowledge_gaps(torus);
    REQUIRE(gaps.size() == 1);
    // "cells" should appear once, "dna" should appear once
    const auto& mems = gaps[0].related_memories;
    CHECK(std::count(mems.begin(), mems.end(), "cells") == 1);
    CHECK(std::count(mems.begin(), mems.end(), "dna")   == 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// [P118/gain]  measure_information_gain
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase118 measure_information_gain unknown domain returns exploration_rate",
          "[Phase118][P118/gain]")
{
    CuriosityEngine ce;
    ce.set_exploration_rate(0.4);
    double gain = ce.measure_information_gain("dark-matter", torus);
    CHECK_THAT(gain, Catch::Matchers::WithinAbs(0.4, 1e-9));
}

TEST_CASE("Phase118 measure_information_gain known domain reflects uncertainty",
          "[Phase118][P118/gain]")
{
    CuriosityEngine ce;
    ce.register_gap(make_gap("dark-energy", 0.9, 0));
    double gain = ce.measure_information_gain("dark-energy", torus);
    // With query_count=0: saturation = 1/(1+0) = 1.0 → gain ≈ 0.9
    CHECK_THAT(gain, Catch::Matchers::WithinAbs(0.9, 1e-6));
}

TEST_CASE("Phase118 measure_information_gain decreases with rising query_count",
          "[Phase118][P118/gain]")
{
    CuriosityEngine ce;
    ce.register_gap(make_gap("dark-energy", 0.9, 0));
    double gain_fresh = ce.measure_information_gain("dark-energy", torus);

    ce.register_gap(make_gap("dark-energy", 0.9, 10));  // high query_count
    double gain_stale = ce.measure_information_gain("dark-energy", torus);

    CHECK(gain_fresh > gain_stale);
}

// ─────────────────────────────────────────────────────────────────────────────
// [P118/questions]  generate_questions
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase118 generate_questions count=0 returns empty", "[Phase118][P118/questions]")
{
    CuriosityEngine ce;
    ce.register_gap(make_gap("physics", 0.8));
    CHECK(ce.generate_questions(torus, 0).empty());
}

TEST_CASE("Phase118 generate_questions with no gaps returns padded question",
          "[Phase118][P118/questions]")
{
    CuriosityEngine ce;
    auto qs = ce.generate_questions(torus, 3);
    // Should get exactly the padding question (we asked for 3 but have 0 gaps)
    REQUIRE_FALSE(qs.empty());
    CHECK(qs.size() == 1);   // only the exploration pad
}

TEST_CASE("Phase118 generate_questions returns up to count items from registered gaps",
          "[Phase118][P118/questions]")
{
    CuriosityEngine ce;
    for (int i = 0; i < 5; ++i)
        ce.register_gap(make_gap("domain" + std::to_string(i), 0.5 + i * 0.05));
    auto qs = ce.generate_questions(torus, 3);
    CHECK(qs.size() == 3);
}

TEST_CASE("Phase118 generate_questions question text contains gap domain",
          "[Phase118][P118/questions]")
{
    CuriosityEngine ce;
    ce.register_gap(make_gap("superconductivity", 0.7));
    auto qs = ce.generate_questions(torus, 1);
    REQUIRE(qs.size() == 1);
    CHECK(qs[0].text.find("superconductivity") != std::string::npos);
}

TEST_CASE("Phase118 generate_questions increments questions_generated stat",
          "[Phase118][P118/questions]")
{
    CuriosityEngine ce;
    ce.register_gap(make_gap("waves", 0.6));
    ce.generate_questions(torus, 1);
    auto stats = ce.get_stats();
    CHECK(stats.at("questions_generated") >= 1u);
}

TEST_CASE("Phase118 generate_questions fires curiosity_callback for each question",
          "[Phase118][P118/questions]")
{
    CuriosityEngine ce;
    ce.register_gap(make_gap("g1", 0.8));
    ce.register_gap(make_gap("g2", 0.6));

    int call_count = 0;
    ce.set_curiosity_callback([&](const Question&) { ++call_count; });
    ce.generate_questions(torus, 2);
    CHECK(call_count == 2);
}

// ─────────────────────────────────────────────────────────────────────────────
// [P118/pursue]  pursue_interest
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase118 pursue_interest empty topic returns false", "[Phase118][P118/pursue]")
{
    CuriosityEngine ce;
    TorusManifold torus2;
    CHECK_FALSE(ce.pursue_interest("", torus2));
}

TEST_CASE("Phase118 pursue_interest non-empty topic returns true", "[Phase118][P118/pursue]")
{
    CuriosityEngine ce;
    TorusManifold torus2;
    CHECK(ce.pursue_interest("quantum-foam", torus2));
}

TEST_CASE("Phase118 pursue_interest increments topics_pursued stat",
          "[Phase118][P118/pursue]")
{
    CuriosityEngine ce;
    TorusManifold torus2;
    ce.pursue_interest("entropy", torus2);
    ce.pursue_interest("entropy", torus2);
    CHECK(ce.get_stats().at("topics_pursued") == 2u);
}

TEST_CASE("Phase118 pursue_interest reduces gap uncertainty each call",
          "[Phase118][P118/pursue]")
{
    CuriosityEngine ce;
    ce.register_gap(make_gap("fluid-dynamics", 0.8));
    TorusManifold torus2;
    ce.pursue_interest("fluid-dynamics", torus2);
    auto gaps = ce.identify_knowledge_gaps(torus2);
    REQUIRE(gaps.size() == 1);
    CHECK(gaps[0].uncertainty < 0.8);
}

TEST_CASE("Phase118 pursue_interest fires callback with topic-named question",
          "[Phase118][P118/pursue]")
{
    CuriosityEngine ce;
    TorusManifold torus2;
    std::string received;
    ce.set_curiosity_callback([&](const Question& q) { received = q.text; });
    ce.pursue_interest("magnetism", torus2);
    CHECK(received.find("magnetism") != std::string::npos);
}

// ─────────────────────────────────────────────────────────────────────────────
// [P118/identify]  identify_knowledge_gaps
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase118 identify_knowledge_gaps returns gaps sorted by descending uncertainty",
          "[Phase118][P118/identify]")
{
    CuriosityEngine ce;
    ce.register_gap(make_gap("low",  0.2));
    ce.register_gap(make_gap("high", 0.9));
    ce.register_gap(make_gap("mid",  0.5));
    auto gaps = ce.identify_knowledge_gaps(torus);
    REQUIRE(gaps.size() == 3);
    CHECK(gaps[0].uncertainty >= gaps[1].uncertainty);
    CHECK(gaps[1].uncertainty >= gaps[2].uncertainty);
}

// ─────────────────────────────────────────────────────────────────────────────
// [P118/interest]  measure_interestingness
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase118 unknown topic is more interesting than known+familiar",
          "[Phase118][P118/interest]")
{
    CuriosityEngine ce;
    ce.register_gap(make_gap("familiar", 0.5, 50));   // high query_count
    double unknown_score = ce.measure_interestingness("brand-new-topic", torus);
    double known_score   = ce.measure_interestingness("familiar",        torus);
    CHECK(unknown_score > known_score);
}

TEST_CASE("Phase118 measure_interestingness empty topic returns 0",
          "[Phase118][P118/interest]")
{
    CuriosityEngine ce;
    CHECK_THAT(ce.measure_interestingness("", torus),
               Catch::Matchers::WithinAbs(0.0, 1e-9));
}

// ─────────────────────────────────────────────────────────────────────────────
// [P118/learning]  autonomous learning
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase118 is_learning starts false", "[Phase118][P118/learning]")
{
    CuriosityEngine ce;
    CHECK_FALSE(ce.is_learning());
}

TEST_CASE("Phase118 start_autonomous_learning sets is_learning true",
          "[Phase118][P118/learning]")
{
    CuriosityEngine ce;
    TorusManifold torus2;
    ce.start_autonomous_learning(torus2);
    CHECK(ce.is_learning());
}

TEST_CASE("Phase118 stop_autonomous_learning sets is_learning false",
          "[Phase118][P118/learning]")
{
    CuriosityEngine ce;
    TorusManifold torus2;
    ce.start_autonomous_learning(torus2);
    ce.stop_autonomous_learning();
    CHECK_FALSE(ce.is_learning());
}

// ─────────────────────────────────────────────────────────────────────────────
// [P118/stats]  get_stats
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase118 get_stats has expected keys", "[Phase118][P118/stats]")
{
    CuriosityEngine ce;
    auto stats = ce.get_stats();
    CHECK(stats.count("questions_generated") == 1);
    CHECK(stats.count("topics_pursued")      == 1);
    CHECK(stats.count("gaps_tracked")        == 1);
    CHECK(stats.count("interest_history")    == 1);
}

TEST_CASE("Phase118 get_stats gaps_tracked reflects registered gap count",
          "[Phase118][P118/stats]")
{
    CuriosityEngine ce;
    ce.register_gap(make_gap("a", 0.5));
    ce.register_gap(make_gap("b", 0.7));
    CHECK(ce.get_stats().at("gaps_tracked") == 2u);
}
