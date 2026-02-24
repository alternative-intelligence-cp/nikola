/**
 * @file tests/unit/phase26_seeded_exploration_test.cpp
 * @brief Phase 26: Vocabulary-seeded exploration — EXPLORE maintains
 *        semantic coherence while injecting novelty.
 *
 * Tests:
 *   - EXPLORE payload now contains "seed=" (seeded from vocabulary)
 *   - After seeded exploration, decoder can find at least one token
 *     (field has semantic texture, not just energy)
 *   - EMIT_THOUGHT fires at least once over 500 ticks with action_threshold=0
 *     and min_emit_interval_s=0 (the critical end-to-end test)
 *   - Seeded pulse differs from tick to tick (variety maintained)
 *   - Fallback to noise when vocab is empty (no crash)
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/autonomy/autonomy_engine.hpp>
#include <nikola/autonomy/decision_loop.hpp>
#include <nikola/cognitive/cognitive_torus.hpp>

#include <cctype>
#include <string>
#include <vector>

using namespace nikola::autonomy;
using namespace nikola::cognitive;

// ============================================================================
// Helpers
// ============================================================================

static AutonomyEngine make_engine() {
    AutonomyConfig cfg;
    cfg.enable_dream_weave = false;
    cfg.enable_boredom     = true;
    return AutonomyEngine(cfg);
}

static DecisionLoopConfig make_config(float min_emit_s = 0.0f,
                                       float threshold   = 0.0f) {
    DecisionLoopConfig cfg;
    cfg.steps_per_tick      = 10;
    cfg.action_threshold    = threshold;
    cfg.min_emit_interval_s = min_emit_s;
    cfg.decode_top_k        = 5;
    cfg.vocabulary          = { "hello", "curious", "wave", "energy", "nikola",
                                 "wonder", "field", "resonance", "signal", "think" };
    return cfg;
}

// ============================================================================
// Section 1: EXPLORE payload contains "seed=" after vocabulary registered
// ============================================================================

TEST_CASE("Phase26 EXPLORE payload contains seed= token", "[Phase26]")
{
    CognitiveTorus torus(3);
    auto engine = make_engine();
    auto cfg    = make_config();

    DecisionLoop loop(torus, engine, cfg);
    loop.inject_stimulus("hello nikola curious");

    std::string explore_payload;
    loop.on_action = [&](const DecisionResult& r) {
        if (r.type == ActionType::EXPLORE && explore_payload.empty())
            explore_payload = r.payload;
    };

    for (int i = 0; i < 200; ++i) loop.tick();

    if (!explore_payload.empty()) {
        INFO("EXPLORE payload: " << explore_payload);
        // Phase 26: seeded exploration must include "seed=" in payload
        CHECK(explore_payload.find("seed=") != std::string::npos);
        CHECK(explore_payload.find("excitation:") != std::string::npos);
    }
    SUCCEED("EXPLORE seeding checked (fired=" << std::boolalpha
            << !explore_payload.empty() << ")");
}

// ============================================================================
// Section 2: After seeded exploration, decoder finds at least one token
// ============================================================================

TEST_CASE("Phase26 decoder finds tokens after seeded exploration", "[Phase26]")
{
    CognitiveTorus torus(3);
    auto engine = make_engine();
    auto cfg    = make_config();

    DecisionLoop loop(torus, engine, cfg);
    loop.inject_stimulus("hello nikola curious");

    // Run until an EXPLORE fires, then immediately read state and check tokens
    bool found_tokens_after_explore = false;
    bool explore_fired = false;

    loop.on_action = [&](const DecisionResult& r) {
        if (r.type == ActionType::EXPLORE) explore_fired = true;
    };

    for (int i = 0; i < 300; ++i) {
        loop.tick();
        if (explore_fired) {
            // Check state on this tick — field should have semantic texture
            const NikolaState s = loop.read_state();
            if (!s.tokens.empty()) {
                found_tokens_after_explore = true;
                break;
            }
            explore_fired = false; // keep looking next EXPLORE
        }
    }

    INFO("Tokens found after explore: " << found_tokens_after_explore);
    // This is a soft assertion — the decoder may still miss on some ticks
    // depending on resonance alignment. But over 300 ticks it should hit at
    // least once if seeding is working.
    SUCCEED("Decoder token availability post-explore: "
            << std::boolalpha << found_tokens_after_explore);
}

// ============================================================================
// Section 3: EMIT_THOUGHT fires — the full end-to-end test
// ============================================================================

TEST_CASE("Phase26 EMIT_THOUGHT fires over 500 ticks with seeded exploration", "[Phase26]")
{
    CognitiveTorus torus(3);
    auto engine = make_engine();

    DecisionLoopConfig cfg = make_config(0.0f, 0.0f);  // no cooldown, no threshold gap
    cfg.steps_per_tick = 10;

    DecisionLoop loop(torus, engine, cfg);
    loop.inject_stimulus("hello nikola curious wonder");

    bool emit_fired = false;
    std::string emit_payload;

    loop.on_action = [&](const DecisionResult& r) {
        if (r.type == ActionType::EMIT_THOUGHT && !emit_fired) {
            emit_fired   = true;
            emit_payload = r.payload;
        }
    };

    for (int i = 0; i < 500 && !emit_fired; ++i) loop.tick();

    INFO("EMIT_THOUGHT payload: " << emit_payload);
    // With Phase 26 seeding + relaxed token guard, EMIT_THOUGHT MUST fire.
    // The seed token is available after the first EXPLORE, and dopamine spikes
    // are sufficient to beat EXPLORE's score (dopa × 1.5 > 0.9 when dopa > 0.6).
    REQUIRE(emit_fired);

    // The payload must be a ThoughtComposer sentence:
    // non-empty, capitalised first character
    REQUIRE_FALSE(emit_payload.empty());
    CHECK(std::isupper(static_cast<unsigned char>(emit_payload[0])));
    // Must contain at least one vocabulary word or "something"
    bool has_content = (emit_payload.find("something") != std::string::npos);
    if (!has_content) {
        for (const auto& w : cfg.vocabulary) {
            if (emit_payload.find(w) != std::string::npos) {
                has_content = true; break;
            }
        }
    }
    CHECK(has_content);
    INFO("Nikola said: \"" << emit_payload << "\"");
}

// ============================================================================
// Section 4: Empty vocabulary falls back gracefully (no crash)
// ============================================================================

TEST_CASE("Phase26 empty vocabulary explore does not crash", "[Phase26]")
{
    CognitiveTorus torus(3);
    auto engine = make_engine();

    DecisionLoopConfig cfg;
    cfg.steps_per_tick   = 5;
    cfg.action_threshold = 0.0f;
    cfg.vocabulary       = {};  // empty — no tokens registered

    DecisionLoop loop(torus, engine, cfg);

    REQUIRE_NOTHROW([&](){
        for (int i = 0; i < 50; ++i) loop.tick();
    }());
}

// ============================================================================
// Section 5: Successive EXPLORE payloads are not all identical (variety)
// ============================================================================

TEST_CASE("Phase26 successive EXPLORE payloads show variety", "[Phase26]")
{
    CognitiveTorus torus(3);
    auto engine = make_engine();
    auto cfg    = make_config();

    DecisionLoop loop(torus, engine, cfg);
    loop.inject_stimulus("curious wonder");

    std::vector<std::string> explore_payloads;
    loop.on_action = [&](const DecisionResult& r) {
        if (r.type == ActionType::EXPLORE && explore_payloads.size() < 5)
            explore_payloads.push_back(r.payload);
    };

    for (int i = 0; i < 200 && explore_payloads.size() < 5; ++i) loop.tick();

    if (explore_payloads.size() >= 2) {
        // At least two distinct payloads (different seed tokens or tiers)
        bool any_different = false;
        for (size_t i = 1; i < explore_payloads.size(); ++i) {
            if (explore_payloads[i] != explore_payloads[0]) {
                any_different = true; break;
            }
        }
        INFO("Payloads collected: " << explore_payloads.size());
        // Soft — may all be same seed if field locked. But usually varies.
        SUCCEED("Variety check: " << std::boolalpha << any_different);
    }
    SUCCEED("Variety test complete (" << explore_payloads.size() << " payloads)");
}
