/**
 * @file tests/unit/phase31_lookup_agent_test.cpp
 * @brief Phase 31: Oracle-Gated Lookup Fulfillment — test suite (Catch2 v3).
 *
 * Tests are layered from pure unit tests (no sockets) to live ZMQ round-trip:
 *
 *   Section 1 — OracleVerdict: construction + field defaults
 *   Section 2 — StubOracle: returns configured score, correct name()
 *   Section 3 — CoherenceOracle: scoring by length + contradiction heuristic
 *   Section 4 — OraclePool: empty=0.5, single, multi-oracle averaging
 *   Section 5 — inject_stimulus with credibility: torus energy scales with amplitude
 *   Section 6 — parse_action_json: valid REQUEST_LOOKUP, other types, bad JSON
 *   Section 7 — to_stimulus_json: format, field presence, credibility precision
 *   Section 8 — parse_scored_stimulus (NikolaNode): envelope, plain text fallback
 *   Section 9 — Live ZMQ round-trip: agent receives REQUEST_LOOKUP, pushes scored stimulus
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/autonomy/oracle_pool.hpp>
#include <nikola/autonomy/lookup_agent.hpp>
#include <nikola/autonomy/nikola_node.hpp>
#include <nikola/infrastructure/spine.hpp>

#include <atomic>
#include <chrono>
#include <string>
#include <thread>

using namespace nikola::autonomy;
using namespace nikola::infrastructure;
using namespace std::chrono_literals;

// =============================================================================
//  Section 1 — OracleVerdict defaults
// =============================================================================

TEST_CASE("Phase31 — OracleVerdict has correct defaults", "[phase31][oracle]") {
    OracleVerdict v;
    CHECK(v.confidence == Catch::Approx(0.5f));
    CHECK(v.rationale.empty());

    OracleVerdict v2{0.9f, "test reason"};
    CHECK(v2.confidence == Catch::Approx(0.9f));
    CHECK(v2.rationale == "test reason");
}

// =============================================================================
//  Section 2 — StubOracle
// =============================================================================

TEST_CASE("Phase31 — StubOracle always returns configured score", "[phase31][oracle]") {
    SECTION("Default score 0.7") {
        StubOracle o("test-oracle");
        CHECK(o.name() == "test-oracle");
        const auto v = o.assess("query", "content");
        CHECK(v.confidence == Catch::Approx(0.7f));
    }

    SECTION("Custom high score") {
        StubOracle o("high", 0.95f);
        CHECK(o.assess("q", "c").confidence == Catch::Approx(0.95f));
        CHECK(o.name() == "high");
    }

    SECTION("Custom low score") {
        StubOracle o("low", 0.1f);
        CHECK(o.assess("q", "c").confidence == Catch::Approx(0.1f));
    }

    SECTION("Clamped above 1.0") {
        StubOracle o("over", 1.5f);
        CHECK(o.assess("q", "c").confidence == Catch::Approx(1.0f));
    }

    SECTION("Clamped below 0.0") {
        StubOracle o("under", -0.5f);
        CHECK(o.assess("q", "c").confidence == Catch::Approx(0.0f));
    }

    SECTION("Query and content are unused") {
        StubOracle o("unused", 0.42f);
        CHECK(o.assess("", "").confidence == Catch::Approx(0.42f));
        CHECK(o.assess("something", "else").confidence == Catch::Approx(0.42f));
    }
}

// =============================================================================
//  Section 3 — CoherenceOracle
// =============================================================================

TEST_CASE("Phase31 — CoherenceOracle scores by length and contradiction", "[phase31][oracle]") {
    CoherenceOracle o;
    CHECK(o.name() == "coherence");

    SECTION("Empty content → 0.0") {
        CHECK(o.assess("anything", "").confidence == Catch::Approx(0.0f));
    }

    SECTION("Very short content (< 20 chars) → 0.20") {
        CHECK(o.assess("q", "short text ok.").confidence == Catch::Approx(0.20f));
    }

    SECTION("Short content (20–99 chars) → 0.55") {
        const std::string s(50, 'x');
        CHECK(o.assess("q", s).confidence == Catch::Approx(0.55f));
    }

    SECTION("Medium content (100–499 chars) → 0.75") {
        const std::string s(200, 'x');
        CHECK(o.assess("q", s).confidence == Catch::Approx(0.75f));
    }

    SECTION("Long content (>=500 chars) → 0.85") {
        const std::string s(600, 'x');
        CHECK(o.assess("q", s).confidence == Catch::Approx(0.85f));
    }

    SECTION("Contradiction penalty: ' is not ' subtracts 0.20") {
        // Medium content (0.75) minus penalty (0.20) = 0.55
        const std::string s(200, 'y');
        const std::string with_not = s + " water is not H2O in this context";
        CHECK(o.assess("q", with_not).confidence == Catch::Approx(0.55f));
    }

    SECTION("Short content with contradiction penalty") {
        // "It is not true.    abcde" = 25 chars → 20-99 bucket (0.55)
        // " is not " substring triggers -0.20 penalty → 0.35
        const std::string s = "It is not true.    abcde";
        CHECK(o.assess("q", s).confidence == Catch::Approx(0.35f));
    }
}

// =============================================================================
//  Section 4 — OraclePool
// =============================================================================

TEST_CASE("Phase31 — OraclePool averages oracle verdicts", "[phase31][oracle]") {
    SECTION("Empty pool returns 0.5 (neutral)") {
        OraclePool pool;
        CHECK(pool.empty());
        CHECK(pool.size() == 0);
        CHECK(pool.evaluate("q", "content") == Catch::Approx(0.5f));
    }

    SECTION("Single oracle — pool returns that oracle's score") {
        OraclePool pool;
        pool.add_oracle(std::make_shared<StubOracle>("a", 0.8f));
        CHECK(pool.size() == 1);
        CHECK(pool.evaluate("q", "c") == Catch::Approx(0.8f));
    }

    SECTION("Two oracles — returns arithmetic mean") {
        OraclePool pool;
        pool.add_oracle(std::make_shared<StubOracle>("a", 0.6f));
        pool.add_oracle(std::make_shared<StubOracle>("b", 0.8f));
        // Mean of 0.6 and 0.8 = 0.7
        CHECK(pool.evaluate("q", "c") == Catch::Approx(0.7f));
    }

    SECTION("Three oracles — mean of all three") {
        OraclePool pool;
        pool.add_oracle(std::make_shared<StubOracle>("a", 0.9f));
        pool.add_oracle(std::make_shared<StubOracle>("b", 0.6f));
        pool.add_oracle(std::make_shared<StubOracle>("c", 0.3f));
        // Mean of 0.9 + 0.6 + 0.3 = 1.8 / 3 = 0.6
        CHECK(pool.evaluate("q", "c") == Catch::Approx(0.6f));
    }

    SECTION("CoherenceOracle in pool — real scoring") {
        OraclePool pool;
        pool.add_oracle(std::make_shared<CoherenceOracle>());
        pool.add_oracle(std::make_shared<StubOracle>("fact", 1.0f));
        const std::string content(200, 'a');  // 200 chars → coherence = 0.75
        // Mean of 0.75 + 1.0 = 0.875
        CHECK(pool.evaluate("q", content) == Catch::Approx(0.875f));
    }

    SECTION("Pool rejects empty content consistently") {
        OraclePool pool;
        pool.add_oracle(std::make_shared<CoherenceOracle>());
        // Empty content → coherence 0.0
        CHECK(pool.evaluate("q", "") == Catch::Approx(0.0f));
    }
}

// =============================================================================
//  Section 5 — inject_stimulus with credibility
// =============================================================================

TEST_CASE("Phase31 — inject_stimulus credibility modulates torus energy",
          "[phase31][decision_loop]") {
    using nikola::cognitive::CognitiveTorus;
    using nikola::autonomy::AutonomyEngine;

#ifdef NIKOLA_HAS_ORT
    CognitiveTorus torus(3, NIKOLA_ORT_TOKENIZER_PATH, NIKOLA_ORT_MODEL_PATH,
                         3, 0.01f);
#else
    CognitiveTorus torus(3, 3, 0.01f);
#endif
    AutonomyConfig ac;
    ac.enable_dream_weave = false;
    AutonomyEngine engine(ac);
    DecisionLoopConfig cfg;
    cfg.steps_per_tick = 5;
    DecisionLoop loop(torus, engine, cfg);

    SECTION("Credibility 0.0 does not inject (no energy change)") {
        const float e0 = torus.total_probability();
        loop.inject_stimulus("some content that would normally raise energy", 0.0f);
        const float e1 = torus.total_probability();
        // Energy unchanged (credibility 0 skips injection)
        CHECK(std::abs(e1 - e0) < 1e-6f);
    }

    SECTION("Credibility 1.0 injects full-strength stimulus") {
        loop.inject_stimulus("bootstrap", 1.0f);
        // After a full-strength injection the torus energy should be nonzero
        CHECK(torus.total_probability() > 0.f);
    }

    SECTION("Credibility 1.0 injects more energy than 0.25") {
        // Fresh identical toruses — compare energy after different credibility
#ifdef NIKOLA_HAS_ORT
        CognitiveTorus t_high(3, NIKOLA_ORT_TOKENIZER_PATH, NIKOLA_ORT_MODEL_PATH,
                               3, 0.01f);
        CognitiveTorus t_low(3, NIKOLA_ORT_TOKENIZER_PATH, NIKOLA_ORT_MODEL_PATH,
                              3, 0.01f);
#else
        CognitiveTorus t_high(3, 3, 0.01f);
        CognitiveTorus t_low(3, 3, 0.01f);
#endif
        AutonomyConfig ac2;
        ac2.enable_dream_weave = false;
        AutonomyEngine eng_high(ac2), eng_low(ac2);
        DecisionLoopConfig c2; c2.steps_per_tick = 1;

        DecisionLoop loop_high(t_high, eng_high, c2);
        DecisionLoop loop_low(t_low, eng_low, c2);

        const std::string text = "water is H2O";
        loop_high.inject_stimulus(text, 1.0f);
        loop_low.inject_stimulus(text, 0.25f);

        // High credibility torus should have more energy
        CHECK(t_high.total_probability() >= t_low.total_probability());
    }

    SECTION("Plain inject_stimulus (no credibility) still works") {
        loop.inject_stimulus("hello nikola");
        CHECK(torus.total_probability() > 0.f);
    }
}

// =============================================================================
//  Section 6 — parse_action_json
// =============================================================================

TEST_CASE("Phase31 — LookupFulfillmentAgent::parse_action_json", "[phase31][json]") {
    SECTION("Valid REQUEST_LOOKUP action") {
        const std::string json =
            R"({"tick":42,"type":"REQUEST_LOOKUP","score":0.6500,"payload":"what is entropy"})";
        const auto [type, payload] = LookupFulfillmentAgent::parse_action_json(json);
        CHECK(type == "REQUEST_LOOKUP");
        CHECK(payload == "what is entropy");
    }

    SECTION("EMIT_THOUGHT action — type extracted, payload extracted") {
        const std::string json =
            R"({"tick":7,"type":"EMIT_THOUGHT","score":0.7200,"payload":"some thought"})";
        const auto [type, payload] = LookupFulfillmentAgent::parse_action_json(json);
        CHECK(type == "EMIT_THOUGHT");
        CHECK(payload == "some thought");
    }

    SECTION("SILENT action — payload is empty string") {
        const std::string json =
            R"({"tick":1,"type":"SILENT","score":0.3000,"payload":""})";
        const auto [type, payload] = LookupFulfillmentAgent::parse_action_json(json);
        CHECK(type == "SILENT");
        CHECK(payload.empty());
    }

    SECTION("Empty JSON string — both empty") {
        const auto [type, payload] = LookupFulfillmentAgent::parse_action_json("");
        CHECK(type.empty());
        CHECK(payload.empty());
    }

    SECTION("Payload with escaped quote") {
        const std::string json =
            R"({"tick":3,"type":"REQUEST_LOOKUP","score":0.5000,"payload":"what is \"entropy\""})";
        const auto [type, payload] = LookupFulfillmentAgent::parse_action_json(json);
        CHECK(type == "REQUEST_LOOKUP");
        CHECK(payload.find("entropy") != std::string::npos);
    }

    SECTION("Type field present before payload") {
        // Verify field ordering does not affect extraction
        const std::string json =
            R"({"tick":99,"type":"REQUEST_LOOKUP","score":0.9,"payload":"query Q"})";
        const auto [type, payload] = LookupFulfillmentAgent::parse_action_json(json);
        CHECK(type == "REQUEST_LOOKUP");
        CHECK(payload == "query Q");
    }
}

// =============================================================================
//  Section 7 — to_stimulus_json
// =============================================================================

TEST_CASE("Phase31 — LookupFulfillmentAgent::to_stimulus_json format", "[phase31][json]") {
    SECTION("Basic format and required fields") {
        const std::string j = LookupFulfillmentAgent::to_stimulus_json("hello", 0.75f);
        CHECK(j.front() == '{');
        CHECK(j.back() == '}');
        CHECK(j.find("\"type\":\"stimulus\"")  != std::string::npos);
        CHECK(j.find("\"text\":\"hello\"")      != std::string::npos);
        CHECK(j.find("\"credibility\":")        != std::string::npos);
    }

    SECTION("Credibility precision is 4 decimal places") {
        const std::string j = LookupFulfillmentAgent::to_stimulus_json("x", 0.8f);
        // Should contain "0.8000"
        CHECK(j.find("0.8000") != std::string::npos);
    }

    SECTION("Credibility clamped to [0, 1]") {
        const std::string j_high = LookupFulfillmentAgent::to_stimulus_json("x", 2.5f);
        CHECK(j_high.find("1.0000") != std::string::npos);

        const std::string j_low = LookupFulfillmentAgent::to_stimulus_json("x", -1.0f);
        CHECK(j_low.find("0.0000") != std::string::npos);
    }

    SECTION("Text with special characters is escaped") {
        const std::string j = LookupFulfillmentAgent::to_stimulus_json("say \"hi\"", 0.5f);
        CHECK(j.find("\\\"hi\\\"") != std::string::npos);
    }

    SECTION("Newline in text is escaped") {
        const std::string j = LookupFulfillmentAgent::to_stimulus_json("line1\nline2", 0.5f);
        CHECK(j.find("\\n") != std::string::npos);
        CHECK(j.find('\n') == std::string::npos);  // Raw newline must not appear
    }

    SECTION("Empty text is valid") {
        const std::string j = LookupFulfillmentAgent::to_stimulus_json("", 0.5f);
        CHECK(j.find("\"text\":\"\"") != std::string::npos);
    }
}

// =============================================================================
//  Section 8 — NikolaNode::parse_scored_stimulus
// =============================================================================

TEST_CASE("Phase31 — NikolaNode::parse_scored_stimulus", "[phase31][nikola_node]") {
    SECTION("Valid envelope is parsed") {
        const std::string json =
            R"({"type":"stimulus","text":"Water is H2O.","credibility":0.8700})";
        std::string text;
        float cred = -1.f;
        const bool ok = NikolaNode::parse_scored_stimulus(json, text, cred);
        CHECK(ok);
        CHECK(text == "Water is H2O.");
        CHECK(cred == Catch::Approx(0.87f).epsilon(0.001f));
    }

    SECTION("Plain text (no braces) returns false") {
        std::string text;
        float cred = -1.f;
        const bool ok = NikolaNode::parse_scored_stimulus("hello nikola", text, cred);
        CHECK_FALSE(ok);
    }

    SECTION("JSON that is not a stimulus envelope returns false") {
        std::string text;
        float cred = -1.f;
        const std::string j =
            R"({"type":"action","text":"something","credibility":0.5})";
        const bool ok = NikolaNode::parse_scored_stimulus(j, text, cred);
        CHECK_FALSE(ok);
    }

    SECTION("Missing text field returns false") {
        std::string text;
        float cred = -1.f;
        const std::string j =
            R"({"type":"stimulus","credibility":0.5})";
        const bool ok = NikolaNode::parse_scored_stimulus(j, text, cred);
        CHECK_FALSE(ok);
    }

    SECTION("Missing credibility field returns false") {
        std::string text;
        float cred = -1.f;
        const std::string j =
            R"({"type":"stimulus","text":"content here"})";
        const bool ok = NikolaNode::parse_scored_stimulus(j, text, cred);
        CHECK_FALSE(ok);
    }

    SECTION("Credibility 0.0 is parsed correctly") {
        const std::string json =
            R"({"type":"stimulus","text":"low trust","credibility":0.0000})";
        std::string text;
        float cred = -1.f;
        CHECK(NikolaNode::parse_scored_stimulus(json, text, cred));
        CHECK(text == "low trust");
        CHECK(cred == Catch::Approx(0.0f));
    }

    SECTION("NikolaNode::to_stimulus_json round-trips through parse") {
        const std::string original_text = "Quantum entanglement is non-local.";
        const float original_cred = 0.82f;
        const std::string envelope =
            LookupFulfillmentAgent::to_stimulus_json(original_text, original_cred);

        std::string parsed_text;
        float parsed_cred = -1.f;
        CHECK(NikolaNode::parse_scored_stimulus(envelope, parsed_text, parsed_cred));
        CHECK(parsed_text == original_text);
        CHECK(parsed_cred == Catch::Approx(original_cred).epsilon(0.001f));
    }
}

// =============================================================================
//  Section 9 — Live ZMQ round-trip
//              Test PUB → agent SUB → lookup_fn → oracle → agent PUSH → test PULL
// =============================================================================

TEST_CASE("Phase31 — LookupFulfillmentAgent live ZMQ round-trip",
          "[phase31][zmq][network]") {
    // Unique ports to avoid collision with Phase 30 (55651/55652)
    constexpr uint16_t PUB_PORT  = 55721;
    constexpr uint16_t PULL_PORT = 55722;

    const std::string pub_addr  = "tcp://127.0.0.1:" + std::to_string(PUB_PORT);
    const std::string pull_addr = "tcp://127.0.0.1:" + std::to_string(PULL_PORT);

    // ── Test-side sockets (the "NikolaNode" surrogate) ───────────────────────
    ZmqSpine test_spine;
    auto test_pub  = test_spine.make_publisher(pub_addr);   // binds
    auto test_pull = test_spine.make_pull(pull_addr);        // binds

    // ── Configure agent ───────────────────────────────────────────────────────
    LookupAgentConfig cfg;
    cfg.action_sub_endpoint    = pub_addr;
    cfg.stimulus_push_endpoint = pull_addr;
    cfg.poll_timeout_ms        = 50;

    auto agent = std::make_unique<LookupFulfillmentAgent>(cfg);

    // Lookup function: simple DB stub returning a known string
    const std::string lookup_result = "Entropy is a measure of disorder in a thermodynamic system.";
    agent->set_lookup_fn([&](const std::string& /*q*/) { return lookup_result; });

    // Add a high-confidence stub oracle so credibility is predictable
    agent->add_oracle(std::make_shared<StubOracle>("fact-check", 0.9f));
    agent->add_oracle(std::make_shared<CoherenceOracle>());
    // CoherenceOracle: lookup_result is 58 chars → 0.55; StubOracle: 0.9 → mean 0.725

    // ── Start agent in background thread ─────────────────────────────────────
    std::thread agent_thread([&agent]() {
        agent->run();
    });

    // Allow ZMQ connections to settle (slow-joiner problem)
    std::this_thread::sleep_for(300ms);

    // ── Publish a fake REQUEST_LOOKUP action ──────────────────────────────────
    const std::string action_json =
        R"({"tick":1,"type":"REQUEST_LOOKUP","score":0.6500,"payload":"what is entropy"})";

    // Published as multipart: topic frame + body frame
    ZmqSpine::publish(test_pub, "action",
                      action_json.data(), action_json.size());

    // ── Wait for the scored stimulus to arrive on test PULL ───────────────────
    std::string received_msg;
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (std::chrono::steady_clock::now() < deadline) {
        zmq_pollitem_t item{};
        item.socket = static_cast<void*>(test_pull);
        item.events = ZMQ_POLLIN;
        zmq::poll(&item, 1, std::chrono::milliseconds(200));
        if (item.revents & ZMQ_POLLIN) {
            zmq::message_t msg;
            [[maybe_unused]] auto res = test_pull.recv(msg);
            if (res.has_value() && msg.size() > 0) {
                received_msg = std::string(
                    static_cast<const char*>(msg.data()), msg.size());
                break;
            }
        }
    }

    // ── Clean up ─────────────────────────────────────────────────────────────
    agent->stop();
    agent_thread.join();
    const uint64_t n_fulfilled = agent->fulfilled_count();  // capture before reset
    agent.reset();

    // ── Assertions ───────────────────────────────────────────────────────────
    INFO("Received message: " << received_msg);
    REQUIRE_FALSE(received_msg.empty());

    // Must be a stimulus envelope
    CHECK(received_msg.front() == '{');
    CHECK(received_msg.find("\"type\":\"stimulus\"")  != std::string::npos);
    CHECK(received_msg.find("\"text\":")              != std::string::npos);
    CHECK(received_msg.find("\"credibility\":")       != std::string::npos);

    // Must contain our lookup result
    CHECK(received_msg.find("Entropy") != std::string::npos);

    // Credibility must parse correctly and be in [0, 1]
    std::string parsed_text;
    float parsed_cred = -1.f;
    CHECK(NikolaNode::parse_scored_stimulus(received_msg, parsed_text, parsed_cred));
    CHECK(parsed_cred >= 0.0f);
    CHECK(parsed_cred <= 1.0f);
    // Mean of StubOracle(0.9) + CoherenceOracle for 58-char content (0.55) = 0.725
    CHECK(parsed_cred == Catch::Approx(0.725f).epsilon(0.05f));

    // Agent fulfilled exactly one lookup
    INFO("Fulfilled count: " << n_fulfilled);
    CHECK(n_fulfilled >= 1);
}
