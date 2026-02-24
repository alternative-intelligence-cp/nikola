/**
 * @file tests/unit/phase30_zmq_spine_test.cpp
 * @brief Phase 30: ZMQ Spine — NikolaNode test suite (Catch2 v3).
 *
 * Tests are layered from pure (no sockets) to live ZMQ round-trip:
 *
 *   Section 1 — action_to_json: format, field presence, payload escaping
 *   Section 2 — state_to_json: format, all fields present
 *   Section 3 — NikolaNodeConfig: default ports, state interval
 *   Section 4 — NikolaNode::run(N): callbacks fire, tick_count advances
 *   Section 5 — ZMQ pub/sub round-trip: subscriber receives versioned frames
 *
 * Section 5 uses TCP loopback (two separate ZmqSpine instances / contexts),
 * handling the ZMQ slow-joiner problem with a 300ms settle delay.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/autonomy/nikola_node.hpp>
#include <nikola/infrastructure/spine.hpp>

#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

using namespace nikola::autonomy;
using namespace nikola::infrastructure;
using namespace std::chrono_literals;

// ─────────────────────────────────────────────────────────────────────────────
//  Section 1 — action_to_json
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase30 — action_to_json produces correct JSON format", "[phase30][json]") {

    SECTION("SILENT action at tick 0") {
        DecisionResult r;
        r.type    = ActionType::SILENT;
        r.score   = 0.0f;
        r.payload = "";
        const std::string json = NikolaNode::action_to_json(r, 0);

        CHECK(json.find("\"tick\":0")              != std::string::npos);
        CHECK(json.find("\"type\":\"SILENT\"")      != std::string::npos);
        CHECK(json.find("\"score\":")              != std::string::npos);
        CHECK(json.find("\"payload\":\"\"")         != std::string::npos);
        // Must open and close with braces
        CHECK(json.front() == '{');
        CHECK(json.back()  == '}');
    }

    SECTION("EMIT_THOUGHT with payload at tick 42") {
        DecisionResult r;
        r.type    = ActionType::EMIT_THOUGHT;
        r.score   = 0.72f;
        r.payload = "Something feels off about time";
        const std::string json = NikolaNode::action_to_json(r, 42);

        CHECK(json.find("\"tick\":42")                  != std::string::npos);
        CHECK(json.find("\"type\":\"EMIT_THOUGHT\"")     != std::string::npos);
        CHECK(json.find("Something feels off about time") != std::string::npos);
    }

    SECTION("Payload with special characters is escaped") {
        DecisionResult r;
        r.type    = ActionType::EXPLORE;
        r.score   = 0.5f;
        r.payload = "she said \"hello\" and left\nnewline";
        const std::string json = NikolaNode::action_to_json(r, 7);

        // Double-quote must be escaped
        CHECK(json.find("\\\"hello\\\"") != std::string::npos);
        // Newline must be escaped
        CHECK(json.find("\\n")           != std::string::npos);
        // The raw character must NOT appear in the json string value
        // (it can appear in the surrounding structure, but payload shouldn't have raw \n)
        const auto payload_start = json.find("\"payload\":");
        CHECK(payload_start != std::string::npos);
        const std::string payload_section = json.substr(payload_start);
        CHECK(payload_section.find('\n') == std::string::npos);
    }

    SECTION("All ActionType names render without crashing") {
        const std::vector<ActionType> types = {
            ActionType::SILENT, ActionType::EMIT_THOUGHT, ActionType::STORE_MEMORY,
            ActionType::REQUEST_LOOKUP, ActionType::EXPLORE, ActionType::NAP,
            ActionType::REFUSE
        };
        for (const ActionType t : types) {
            DecisionResult r;
            r.type = t;
            const std::string json = NikolaNode::action_to_json(r, 0);
            CHECK_FALSE(json.empty());
            CHECK(json.find("\"type\":\"") != std::string::npos);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 2 — state_to_json
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase30 — state_to_json produces correct JSON format", "[phase30][json]") {

    SECTION("All fields are present") {
        NikolaState s;
        s.time         = 21.5f;
        s.torus_energy = 1.35f;
        s.dopamine     = 0.88f;
        s.td_error     = -0.074f;
        s.atp          = 0.99f;
        s.boredom      = 0.50f;
        s.entropy      = 0.71f;
        const std::string json = NikolaNode::state_to_json(s, 42);

        CHECK(json.find("\"tick\":42")    != std::string::npos);
        CHECK(json.find("\"time\":")      != std::string::npos);
        CHECK(json.find("\"energy\":")    != std::string::npos);
        CHECK(json.find("\"dopamine\":") != std::string::npos);
        CHECK(json.find("\"td_error\":") != std::string::npos);
        CHECK(json.find("\"atp\":")      != std::string::npos);
        CHECK(json.find("\"boredom\":") != std::string::npos);
        CHECK(json.find("\"entropy\":") != std::string::npos);
        CHECK(json.front() == '{');
        CHECK(json.back()  == '}');
    }

    SECTION("Numeric values round-trip via fixed precision") {
        NikolaState s;
        s.atp     = 1.0f;
        s.boredom = 0.0f;
        const std::string json = NikolaNode::state_to_json(s, 0);

        // 1.0 with 4 decimal places → "1.0000"
        CHECK(json.find("1.0000") != std::string::npos);
        // 0.0 with 4 decimal places → "0.0000"
        CHECK(json.find("0.0000") != std::string::npos);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 3 — NikolaNodeConfig defaults
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase30 — NikolaNodeConfig has correct defaults", "[phase30][config]") {
    NikolaNodeConfig cfg;

    CHECK(cfg.pub_endpoint          == "tcp://*:5560");
    CHECK(cfg.pull_endpoint         == "tcp://*:5561");
    CHECK(cfg.state_publish_interval == 10);

    // DecisionLoopConfig defaults should be sane
    CHECK(cfg.decision_config.steps_per_tick   > 0);
    CHECK(cfg.decision_config.action_threshold >= 0.0f);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 4 — NikolaNode::run(N) drives callbacks and advances tick_count
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase30 — NikolaNode runs N ticks and fires callbacks", "[phase30][node]") {
    NikolaNodeConfig cfg;
    // Use loopback ports unique to this test so we don't clash with section 5
    cfg.pub_endpoint           = "tcp://127.0.0.1:55641";
    cfg.pull_endpoint          = "tcp://127.0.0.1:55642";
    cfg.state_publish_interval = 1;
    cfg.decision_config.steps_per_tick   = 5;
    cfg.decision_config.action_threshold = 0.05f;
    cfg.decision_config.vocabulary = {
        "hello", "explore", "think", "wonder", "time"
    };
#ifdef NIKOLA_HAS_ORT
    cfg.decision_config.tokenizer_json_path    =
        std::string(NIKOLA_ORT_TOKENIZER_PATH) + "/tokenizer.json";
    cfg.decision_config.transformer_model_path = NIKOLA_ORT_MODEL_PATH;
#endif

    NikolaNode node(cfg);
    node.inject_stimulus("hello");

    // Accumulate actions fired via on_action callback
    std::vector<DecisionResult> fired_actions;
    uint64_t tick_callback_count = 0;

    node.on_action = [&fired_actions](const DecisionResult& r) {
        fired_actions.push_back(r);
    };
    node.on_tick = [&tick_callback_count](const NikolaState&, uint64_t) {
        ++tick_callback_count;
    };

    // Run 10 ticks
    node.run(10);

    // tick_count must be exactly 10
    CHECK(node.tick_count() == 10);

    // on_tick fired exactly 10 times
    CHECK(tick_callback_count == 10);

    // At least one action fired in 10 ticks (STORE_MEMORY usually fires on tick 1)
    CHECK(fired_actions.size() >= 1);

    // All fired actions have valid scores
    for (const auto& r : fired_actions) {
        CHECK(r.score >= 0.0f);
    }

    // last_state() reflects a live system (time advanced)
    const auto& s = node.last_state();
    CHECK(s.time    > 0.0f);
    CHECK(s.atp     > 0.0f);
    CHECK(s.atp     <= 1.0f);
    CHECK(s.dopamine >= 0.0f);
    CHECK(s.dopamine <= 1.0f);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 5 — ZMQ pub/sub round-trip
//              NikolaNode publishes versioned topic frames; subscriber receives
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase30 — ZMQ pub/sub round-trip: subscriber receives versioned frames",
          "[phase30][zmq][network]")
{
    constexpr uint16_t PUB_PORT  = 55651;
    constexpr uint16_t PULL_PORT = 55652;

    const std::string pub_addr  = "tcp://127.0.0.1:" + std::to_string(PUB_PORT);
    const std::string pull_addr = "tcp://127.0.0.1:" + std::to_string(PULL_PORT);

    NikolaNodeConfig cfg;
    cfg.pub_endpoint           = pub_addr;
    cfg.pull_endpoint          = pull_addr;
    cfg.state_publish_interval = 1;   // publish state every tick — maximise messages
    cfg.decision_config.steps_per_tick      = 5;
    cfg.decision_config.action_threshold    = 0.05f;
    cfg.decision_config.min_emit_interval_s = 1.0f;
    cfg.decision_config.vocabulary = {
        "hello", "explore", "think", "wonder", "time",
        "wave", "energy", "field", "curious", "signal"
    };
#ifdef NIKOLA_HAS_ORT
    cfg.decision_config.tokenizer_json_path    =
        std::string(NIKOLA_ORT_TOKENIZER_PATH) + "/tokenizer.json";
    cfg.decision_config.transformer_model_path = NIKOLA_ORT_MODEL_PATH;
#endif

    // ── Start node in background thread ──────────────────────────────────────
    auto node = std::make_unique<NikolaNode>(cfg);
    node->inject_stimulus("hello curious");

    std::thread node_thread([&node]() {
        node->run(80);  // run 80 ticks then exit naturally
    });

    // Allow sockets to bind (ZMQ bind is synchronous, but the thread needs to
    // reach the run() call — constructor finishes before thread starts so
    // sockets are already bound before node_thread begins running)
    std::this_thread::sleep_for(300ms);

    // ── Create subscriber in test thread (separate ZMQ context) ──────────────
    ZmqSpine test_spine;
    auto sub = test_spine.make_subscriber(pub_addr, "nikola.v1");

    // Slow-joiner delay: publisher won't retroactively send; subscriber needs
    // to be registered before the next message is sent.
    std::this_thread::sleep_for(350ms);

    // ── Also push a stimulus via PULL socket ─────────────────────────────────
    auto pusher = test_spine.make_push(pull_addr);
    const std::string stimulus = "curious wonder";
    pusher.send(zmq::buffer(stimulus), zmq::send_flags::none);

    // ── Collect messages for up to 8 seconds ─────────────────────────────────
    struct Message { std::string topic; std::string body; };
    std::vector<Message> received;

    const auto deadline = std::chrono::steady_clock::now() + 8s;
    while (std::chrono::steady_clock::now() < deadline && received.size() < 5) {
        zmq_pollitem_t item{};
        item.socket = static_cast<void*>(sub);
        item.events = ZMQ_POLLIN;

        zmq::poll(&item, 1, std::chrono::milliseconds(300));
        if (item.revents & ZMQ_POLLIN) {
            zmq::message_t topic_msg;
            [[maybe_unused]] auto _t = sub.recv(topic_msg);
            if (topic_msg.more()) {
                zmq::message_t data_msg;
                [[maybe_unused]] auto _d = sub.recv(data_msg);
                received.push_back({
                    std::string(static_cast<const char*>(topic_msg.data()), topic_msg.size()),
                    std::string(static_cast<const char*>(data_msg.data()),  data_msg.size())
                });
            }
        }
    }

    // ── Clean up ─────────────────────────────────────────────────────────────
    node->stop();
    node_thread.join();
    node.reset();  // destroy before ZmqSpine goes out of scope

    // ── Assertions ───────────────────────────────────────────────────────────
    INFO("Messages received: " << received.size());
    REQUIRE(received.size() >= 1);

    bool found_state_message  = false;
    bool topic_version_correct = false;

    for (const auto& m : received) {
        // Topic must start with "nikola.v1." (versioned topic prefix)
        if (m.topic.find("nikola.v1.") != std::string::npos) {
            topic_version_correct = true;
        }

        // State messages contain "tick" and "atp" fields
        if (m.topic.find("nikola.v1.state") != std::string::npos) {
            CHECK(m.body.find("\"tick\"")   != std::string::npos);
            CHECK(m.body.find("\"atp\":")   != std::string::npos);
            CHECK(m.body.find("\"energy\":") != std::string::npos);
            CHECK(m.body.front() == '{');
            CHECK(m.body.back()  == '}');
            found_state_message = true;
        }

        // Action messages contain "tick" and "type" fields
        if (m.topic.find("nikola.v1.action") != std::string::npos) {
            CHECK(m.body.find("\"tick\"")   != std::string::npos);
            CHECK(m.body.find("\"type\":\"") != std::string::npos);
            CHECK(m.body.find("\"score\":") != std::string::npos);
        }
    }

    CHECK(topic_version_correct);
    CHECK(found_state_message);
}
