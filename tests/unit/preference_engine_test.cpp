/**
 * @file tests/unit/preference_engine_test.cpp
 * @brief v0.2.3 Phase 1 — PreferenceEngine unit tests.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <nikola/interior/preference_engine.hpp>

using namespace nikola::interior;
using Catch::Approx;

// ============================================================================
// §A — Basic Learning
// ============================================================================

TEST_CASE("§A-1 learn creates new preference", "[preference]") {
    PreferenceEngine pe;
    pe.learn(PreferenceDomain::TOPICS, "physics", 1.0, 100);

    CHECK(pe.query(PreferenceDomain::TOPICS, "physics") == Approx(0.05));
    auto* p = pe.get(PreferenceDomain::TOPICS, "physics");
    REQUIRE(p != nullptr);
    CHECK(p->strength == Approx(0.05));
    CHECK(p->last_tick == 100);
    CHECK(p->update_count == 1);
}

TEST_CASE("§A-2 repeated learning accumulates", "[preference]") {
    PreferenceEngine pe;
    for (int i = 0; i < 10; i++) {
        pe.learn(PreferenceDomain::TOPICS, "math", 1.0, static_cast<uint64_t>(i));
    }
    CHECK(pe.query(PreferenceDomain::TOPICS, "math") == Approx(0.5));
    auto* p = pe.get(PreferenceDomain::TOPICS, "math");
    REQUIRE(p != nullptr);
    CHECK(p->update_count == 10);
}

TEST_CASE("§A-3 negative learning creates dislike", "[preference]") {
    PreferenceEngine pe;
    pe.learn(PreferenceDomain::TOPICS, "spam", -1.0, 1);
    CHECK(pe.query(PreferenceDomain::TOPICS, "spam") == Approx(-0.05));
}

TEST_CASE("§A-4 value clamped to max_value", "[preference]") {
    PreferenceEngine pe;
    for (int i = 0; i < 100; i++) {
        pe.learn(PreferenceDomain::TOPICS, "favorite", 1.0, static_cast<uint64_t>(i));
    }
    CHECK(pe.query(PreferenceDomain::TOPICS, "favorite") == Approx(1.0));
}

TEST_CASE("§A-5 value clamped to -max_value", "[preference]") {
    PreferenceEngine pe;
    for (int i = 0; i < 100; i++) {
        pe.learn(PreferenceDomain::TOPICS, "hated", -1.0, static_cast<uint64_t>(i));
    }
    CHECK(pe.query(PreferenceDomain::TOPICS, "hated") == Approx(-1.0));
}

TEST_CASE("§A-6 query returns 0 for unknown key", "[preference]") {
    PreferenceEngine pe;
    CHECK(pe.query(PreferenceDomain::TOPICS, "nonexistent") == 0.0);
    CHECK(pe.get(PreferenceDomain::TOPICS, "nonexistent") == nullptr);
}

// ============================================================================
// §B — Implicit Learning from Actions
// ============================================================================

TEST_CASE("§B-1 learn_from_action records in ACTIONS domain", "[preference]") {
    PreferenceEngine pe;
    pe.learn_from_action(4, 10);  // EXPLORE

    CHECK(pe.query(PreferenceDomain::ACTIONS, "explore") > 0.0);
}

TEST_CASE("§B-2 EXPLORE also learns TOPICS:novelty", "[preference]") {
    PreferenceEngine pe;
    pe.learn_from_action(4, 10);  // EXPLORE

    CHECK(pe.query(PreferenceDomain::TOPICS, "novelty") > 0.0);
}

TEST_CASE("§B-3 GENERATE_CODE also learns CODE_PATTERNS:active", "[preference]") {
    PreferenceEngine pe;
    pe.learn_from_action(10, 10);  // GENERATE_CODE

    CHECK(pe.query(PreferenceDomain::ACTIONS, "generate_code") > 0.0);
    CHECK(pe.query(PreferenceDomain::CODE_PATTERNS, "active") > 0.0);
}

TEST_CASE("§B-4 EMIT_THOUGHT also learns INTERACTION_STYLES:expressive", "[preference]") {
    PreferenceEngine pe;
    pe.learn_from_action(1, 10);  // EMIT_THOUGHT

    CHECK(pe.query(PreferenceDomain::INTERACTION_STYLES, "expressive") > 0.0);
}

TEST_CASE("§B-5 REASON also learns TOPICS:analysis", "[preference]") {
    PreferenceEngine pe;
    pe.learn_from_action(9, 10);  // REASON

    CHECK(pe.query(PreferenceDomain::TOPICS, "analysis") > 0.0);
}

TEST_CASE("§B-6 REQUEST_LOOKUP also learns DATA_SOURCES:external", "[preference]") {
    PreferenceEngine pe;
    pe.learn_from_action(3, 10);  // REQUEST_LOOKUP

    CHECK(pe.query(PreferenceDomain::DATA_SOURCES, "external") > 0.0);
}

// ============================================================================
// §C — Decay
// ============================================================================

TEST_CASE("§C-1 decay reduces preference values", "[preference]") {
    PreferenceEngine pe;
    pe.learn(PreferenceDomain::TOPICS, "transient", 1.0, 0);
    double before = pe.query(PreferenceDomain::TOPICS, "transient");

    pe.decay(10.0);  // 10 seconds of decay

    double after = pe.query(PreferenceDomain::TOPICS, "transient");
    CHECK(after < before);
    CHECK(after > 0.0);  // Not fully decayed yet
}

TEST_CASE("§C-2 extended decay removes weak preferences", "[preference]") {
    PreferenceEngineConfig cfg;
    cfg.decay_rate = 0.1;  // Fast decay for testing
    PreferenceEngine pe(cfg);

    pe.learn(PreferenceDomain::TOPICS, "ephemeral", 0.01, 0);
    // Very weak preference

    // Decay aggressively
    for (int i = 0; i < 100; i++) {
        pe.decay(1.0);
    }

    auto stats = pe.stats();
    // Should have been pruned
    CHECK(pe.get(PreferenceDomain::TOPICS, "ephemeral") == nullptr);
}

TEST_CASE("§C-3 strong preferences survive decay", "[preference]") {
    PreferenceEngine pe;
    for (int i = 0; i < 20; i++) {
        pe.learn(PreferenceDomain::TOPICS, "strong", 1.0, static_cast<uint64_t>(i));
    }

    pe.decay(10.0);

    auto* p = pe.get(PreferenceDomain::TOPICS, "strong");
    REQUIRE(p != nullptr);
    CHECK(std::abs(p->value) > 0.0);
}

// ============================================================================
// §D — Action Bias
// ============================================================================

TEST_CASE("§D-1 action_bias returns 0 for unknown action", "[preference]") {
    PreferenceEngine pe;
    CHECK(pe.action_bias(4) == 0.0);  // No preferences learned yet
}

TEST_CASE("§D-2 action_bias returns 0 below min_influence_strength", "[preference]") {
    PreferenceEngine pe;
    pe.learn_from_action(4, 1);  // Single learn → weak strength

    // Strength should be below default min_influence_strength (0.5)
    auto* p = pe.get(PreferenceDomain::ACTIONS, "explore");
    REQUIRE(p != nullptr);
    CHECK(p->strength < pe.config().min_influence_strength);
    CHECK(pe.action_bias(4) == 0.0);
}

TEST_CASE("§D-3 action_bias returns positive for well-learned action", "[preference]") {
    PreferenceEngine pe;
    // Learn explore many times to build strength
    for (int i = 0; i < 20; i++) {
        pe.learn_from_action(4, static_cast<uint64_t>(i));
    }

    double bias = pe.action_bias(4);
    CHECK(bias > 0.0);
    CHECK(bias <= pe.config().max_bias);
}

TEST_CASE("§D-4 action_bias clamped to max_bias", "[preference]") {
    PreferenceEngine pe;
    for (int i = 0; i < 1000; i++) {
        pe.learn_from_action(4, static_cast<uint64_t>(i));
    }

    double bias = pe.action_bias(4);
    CHECK(bias <= pe.config().max_bias);
}

// ============================================================================
// §E — Domain Listing and Top Preferences
// ============================================================================

TEST_CASE("§E-1 list_domain returns all entries", "[preference]") {
    PreferenceEngine pe;
    pe.learn(PreferenceDomain::TOPICS, "math", 1.0, 0);
    pe.learn(PreferenceDomain::TOPICS, "physics", 1.0, 0);
    pe.learn(PreferenceDomain::TOPICS, "art", -1.0, 0);

    auto list = pe.list_domain(PreferenceDomain::TOPICS);
    CHECK(list.size() == 3);
}

TEST_CASE("§E-2 top_preferences sorted by absolute value", "[preference]") {
    PreferenceEngine pe;
    pe.learn(PreferenceDomain::TOPICS, "weak", 0.1, 0);
    for (int i = 0; i < 10; i++) {
        pe.learn(PreferenceDomain::TOPICS, "strong", 1.0, static_cast<uint64_t>(i));
    }

    auto top = pe.top_preferences(2);
    REQUIRE(top.size() == 2);
    CHECK(std::abs(top[0].second.value) >= std::abs(top[1].second.value));
}

// ============================================================================
// §F — Stats
// ============================================================================

TEST_CASE("§F-1 stats tracks totals", "[preference]") {
    PreferenceEngine pe;
    pe.learn(PreferenceDomain::TOPICS, "a", 1.0, 0);
    pe.learn(PreferenceDomain::CODE_PATTERNS, "b", 1.0, 0);
    pe.learn(PreferenceDomain::TOPICS, "a", 1.0, 1);  // Update existing

    auto s = pe.stats();
    CHECK(s.total_preferences == 2);
    CHECK(s.total_updates == 3);
    CHECK(s.domains_active == 2);
}

// ============================================================================
// §G — Persistence (JSON round-trip)
// ============================================================================

TEST_CASE("§G-1 to_json produces valid output", "[preference]") {
    PreferenceEngine pe;
    pe.learn(PreferenceDomain::TOPICS, "math", 1.0, 42);
    pe.learn(PreferenceDomain::ACTIONS, "explore", 0.5, 43);

    std::string json = pe.to_json();
    CHECK(json.find("TOPICS") != std::string::npos);
    CHECK(json.find("math") != std::string::npos);
    CHECK(json.find("ACTIONS") != std::string::npos);
    CHECK(json.find("explore") != std::string::npos);
}

TEST_CASE("§G-2 from_json restores preferences", "[preference]") {
    PreferenceEngine pe1;
    pe1.learn(PreferenceDomain::TOPICS, "physics", 1.0, 100);
    pe1.learn(PreferenceDomain::CODE_PATTERNS, "functional", -0.5, 200);
    for (int i = 0; i < 5; i++) {
        pe1.learn(PreferenceDomain::ACTIONS, "explore", 1.0, static_cast<uint64_t>(300 + i));
    }

    std::string json = pe1.to_json();

    PreferenceEngine pe2;
    CHECK(pe2.from_json(json));

    CHECK(pe2.query(PreferenceDomain::TOPICS, "physics") ==
          Approx(pe1.query(PreferenceDomain::TOPICS, "physics")));
    CHECK(pe2.query(PreferenceDomain::CODE_PATTERNS, "functional") ==
          Approx(pe1.query(PreferenceDomain::CODE_PATTERNS, "functional")));

    auto* p1 = pe1.get(PreferenceDomain::ACTIONS, "explore");
    auto* p2 = pe2.get(PreferenceDomain::ACTIONS, "explore");
    REQUIRE(p1 != nullptr);
    REQUIRE(p2 != nullptr);
    CHECK(p2->value == Approx(p1->value));
    CHECK(p2->strength == Approx(p1->strength));
    CHECK(p2->last_tick == p1->last_tick);
    CHECK(p2->update_count == p1->update_count);
}

TEST_CASE("§G-3 from_json on empty string returns true (no crash)", "[preference]") {
    PreferenceEngine pe;
    CHECK(pe.from_json("{}"));
    CHECK(pe.stats().total_preferences == 0);
}

// ============================================================================
// §H — Reset
// ============================================================================

TEST_CASE("§H-1 reset clears all preferences", "[preference]") {
    PreferenceEngine pe;
    pe.learn(PreferenceDomain::TOPICS, "a", 1.0, 0);
    pe.learn(PreferenceDomain::ACTIONS, "b", 1.0, 0);

    pe.reset();

    CHECK(pe.stats().total_preferences == 0);
    CHECK(pe.query(PreferenceDomain::TOPICS, "a") == 0.0);
}

// ============================================================================
// §I — Domain Names
// ============================================================================

TEST_CASE("§I-1 domain_name returns correct strings", "[preference]") {
    CHECK(std::string(domain_name(PreferenceDomain::TOPICS)) == "TOPICS");
    CHECK(std::string(domain_name(PreferenceDomain::CODE_PATTERNS)) == "CODE_PATTERNS");
    CHECK(std::string(domain_name(PreferenceDomain::INTERACTION_STYLES)) == "INTERACTION_STYLES");
    CHECK(std::string(domain_name(PreferenceDomain::DATA_SOURCES)) == "DATA_SOURCES");
    CHECK(std::string(domain_name(PreferenceDomain::ACTIONS)) == "ACTIONS");
}

// ============================================================================
// §J — Multiple Domains Independent
// ============================================================================

TEST_CASE("§J-1 same key in different domains is independent", "[preference]") {
    PreferenceEngine pe;
    pe.learn(PreferenceDomain::TOPICS, "test", 1.0, 0);
    pe.learn(PreferenceDomain::CODE_PATTERNS, "test", -1.0, 0);

    CHECK(pe.query(PreferenceDomain::TOPICS, "test") == Approx(0.05));
    CHECK(pe.query(PreferenceDomain::CODE_PATTERNS, "test") == Approx(-0.05));
}
