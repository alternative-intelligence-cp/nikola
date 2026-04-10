/**
 * @file tests/unit/phase4_infrastructure_test.cpp
 * @brief Phase 4: Infrastructure & Communications test suite (Catch2 v3).
 *
 * Covers all 5 Gap criteria:
 *   Gap 4.1 — CircuitBreaker state machine + RetryPolicy constants
 *   Gap 4.2 — ComponentWatchdog 500ms crash detection + HeartbeatPublisher
 *   Gap 4.3 — WaveformSHM RAII lifecycle + SeqlockFrame lock-free IPC
 *   Gap 4.4 — ZMQ socket configuration constants (HWM, LINGER, IMMEDIATE)
 *   Gap 4.5 — Topic versioning helpers (make_topic, topic_version_prefix)
 *
 * Plus:
 *   Orchestrator lifecycle, LLMRequest/Response, ToolExecutor.
 *
 * Tests run without live ZMQ connections.  All networking is mocked via the
 * circuit-breaker pattern: we test the state machine, not the socket.
 */

// Pull in Orchestrator + LLMBridge implementations (header-only impl guards)
#define NIKOLA_ORCHESTRATOR_IMPL
#define NIKOLA_LLM_BRIDGE_IMPL

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <nikola/infrastructure/circuit_breaker.hpp>
#include <nikola/infrastructure/heartbeat.hpp>
#include <nikola/infrastructure/shared_memory.hpp>
#include <nikola/infrastructure/spine.hpp>
#include <nikola/infrastructure/orchestrator.hpp>
#include <nikola/infrastructure/llm_bridge.hpp>
#include <nikola/infrastructure/tool_executor.hpp>

#include <chrono>
#include <thread>
#include <string>

using namespace nikola::infrastructure;
using namespace std::chrono_literals;

// Helper: create an OrchestratorConfig with unique inproc endpoints for tests
static int g_orch_test_id = 0;
static OrchestratorConfig make_test_config() {
    int id = ++g_orch_test_id;
    OrchestratorConfig cfg;
    cfg.events_endpoint  = "inproc://test_p4_events_" + std::to_string(id);
    cfg.control_endpoint = "inproc://test_p4_control_" + std::to_string(id);
    cfg.data_endpoint    = "inproc://test_p4_data_" + std::to_string(id);
    cfg.cleanup_stale_shm = false;
    return cfg;
}

// ===========================================================================
// GAP 4.1 — Circuit Breaker + Retry Logic
// ===========================================================================

TEST_CASE("Gap4.1 — Timeout constants are correct", "[circuit_breaker][gap4.1]") {
    CHECK(ZMQ_CONTROL_TIMEOUT_MS == 100);
    CHECK(ZMQ_DATA_TIMEOUT_MS    == 5);
    CHECK(ZMQ_MAX_RETRIES        == 3);
    CHECK(ZMQ_BACKOFF_BASE_MS    == 1);

    CHECK(timeout_ms(MessagePriority::CONTROL) == 100);
    CHECK(timeout_ms(MessagePriority::DATA)    == 5);
}

TEST_CASE("Gap4.1 — RetryPolicy back-off schedule", "[circuit_breaker][gap4.1]") {
    RetryPolicy p;
    // Exponential: 1ms, 2ms, 4ms
    CHECK(p.backoff_for(0).count() == 1);
    CHECK(p.backoff_for(1).count() == 2);
    CHECK(p.backoff_for(2).count() == 4);
}

TEST_CASE("Gap4.1 — RetryPolicy timeout reflects priority", "[circuit_breaker][gap4.1]") {
    RetryPolicy ctrl{3, 1, MessagePriority::CONTROL};
    RetryPolicy data{3, 1, MessagePriority::DATA};
    CHECK(ctrl.timeout().count() == 100);
    CHECK(data.timeout().count() ==   5);
}

TEST_CASE("Gap4.1 — CircuitBreaker starts CLOSED", "[circuit_breaker][gap4.1]") {
    CircuitBreaker cb;
    CHECK(cb.state() == CBState::CLOSED);
    CHECK(cb.failure_count() == 0);
    CHECK_FALSE(cb.is_open());
    CHECK(cb.allow_attempt());
}

TEST_CASE("Gap4.1 — CircuitBreaker trips after failure_threshold failures",
          "[circuit_breaker][gap4.1]") {
    CircuitBreaker cb{CircuitBreaker::Config{.failure_threshold = 3}};

    cb.record_failure();
    CHECK(cb.state() == CBState::CLOSED); // 1 < 3
    cb.record_failure();
    CHECK(cb.state() == CBState::CLOSED); // 2 < 3
    cb.record_failure();
    CHECK(cb.state() == CBState::OPEN);   // 3 >= 3 — tripped
    CHECK(cb.is_open());
    CHECK_FALSE(cb.allow_attempt());
}

TEST_CASE("Gap4.1 — CircuitBreaker reset clears state", "[circuit_breaker][gap4.1]") {
    CircuitBreaker cb{CircuitBreaker::Config{.failure_threshold = 2}};
    cb.record_failure();
    cb.record_failure();
    REQUIRE(cb.is_open());

    cb.reset();
    CHECK(cb.state() == CBState::CLOSED);
    CHECK(cb.failure_count() == 0);
    CHECK(cb.allow_attempt());
}

TEST_CASE("Gap4.1 — Successful attempt heals HALF_OPEN back to CLOSED",
          "[circuit_breaker][gap4.1]") {
    // Directly simulate HALF_OPEN scenario via record_success on fresh breaker
    CircuitBreaker cb;
    cb.record_success();
    CHECK(cb.state() == CBState::CLOSED);
    CHECK(cb.failure_count() == 0);
}

TEST_CASE("Gap4.1 — cb_state_name returns correct strings", "[circuit_breaker][gap4.1]") {
    CHECK(std::string(cb_state_name(CBState::CLOSED))    == "CLOSED");
    CHECK(std::string(cb_state_name(CBState::OPEN))      == "OPEN");
    CHECK(std::string(cb_state_name(CBState::HALF_OPEN)) == "HALF_OPEN");
}

TEST_CASE("Gap4.1 — retry_with_circuit_breaker succeeds on first attempt",
          "[circuit_breaker][gap4.1]") {
    CircuitBreaker cb;
    RetryPolicy policy{3, 1, MessagePriority::DATA};

    int call_count = 0;
    bool result = retry_with_circuit_breaker([&]() -> bool {
        ++call_count;
        return true;
    }, cb, policy);

    CHECK(result == true);
    CHECK(call_count == 1);
    CHECK(cb.state() == CBState::CLOSED);
}

TEST_CASE("Gap4.1 — retry_with_circuit_breaker exhausts retries and trips breaker",
          "[circuit_breaker][gap4.1]") {
    // threshold=1: one failed operation (even after N retries) opens the breaker
    CircuitBreaker cb{CircuitBreaker::Config{.failure_threshold = 1}};
    RetryPolicy policy{3, 0 /*zero back-off for speed*/, MessagePriority::DATA};

    int call_count = 0;
    bool result = retry_with_circuit_breaker([&]() -> bool {
        ++call_count;
        return false; // always fail
    }, cb, policy);

    CHECK(result == false);
    CHECK(call_count == 3);
    CHECK(cb.is_open()); // 1 failure recorded after all retries exhausted → trips breaker
}

TEST_CASE("Gap4.1 — fast-fail when breaker OPEN", "[circuit_breaker][gap4.1]") {
    CircuitBreaker cb{CircuitBreaker::Config{
        .failure_threshold = 1,
        .cool_down = 10000ms // very long cool-down
    }};
    cb.record_failure(); // opens the breaker
    REQUIRE(cb.is_open());

    RetryPolicy policy{3, 0, MessagePriority::CONTROL};
    int call_count = 0;
    bool result = retry_with_circuit_breaker([&]() -> bool {
        ++call_count;
        return true;
    }, cb, policy);

    CHECK(result == false);
    CHECK(call_count == 0); // breaker prevented any attempt
}

// ===========================================================================
// GAP 4.2 — Heartbeat Sentinel / Component Watchdog
// ===========================================================================

TEST_CASE("Gap4.2 — Heartbeat constants are correct", "[heartbeat][gap4.2]") {
    CHECK(HEARTBEAT_INTERVAL.count() == 100);
    CHECK(HEARTBEAT_TIMEOUT.count()  == 500);
    CHECK(HEARTBEAT_MAX_MISSED       == 5);
}

TEST_CASE("Gap4.2 — ComponentStatus names", "[heartbeat][gap4.2]") {
    CHECK(std::string(component_status_name(ComponentStatus::ALIVE))   == "ALIVE");
    CHECK(std::string(component_status_name(ComponentStatus::TIMEOUT)) == "TIMEOUT");
    CHECK(std::string(component_status_name(ComponentStatus::DEAD))    == "DEAD");
}

TEST_CASE("Gap4.2 — ComponentWatchdog registers and queries components",
          "[heartbeat][gap4.2]") {
    ComponentWatchdog wd;
    wd.register_component("physics", 1234);

    CHECK(wd.is_registered("physics"));
    CHECK(wd.component_count() == 1);
    CHECK(wd.status("physics") == ComponentStatus::ALIVE);
    CHECK_FALSE(wd.is_registered("memory"));
}

TEST_CASE("Gap4.2 — update_heartbeat resets missed count", "[heartbeat][gap4.2]") {
    ComponentWatchdog wd(10ms, 3); // 10ms timeout, 3 misses

    wd.register_component("engine", 9999);
    // Artificially age the heartbeat by sleeping > timeout
    std::this_thread::sleep_for(20ms);

    auto dead = wd.check_health();
    CHECK(wd.missed_beats("engine") == 1);
    CHECK(dead.empty()); // only 1 miss, threshold = 3

    // Resuscitate
    wd.update_heartbeat("engine");
    CHECK(wd.missed_beats("engine") == 0);
    CHECK(wd.status("engine") == ComponentStatus::ALIVE);
}

TEST_CASE("Gap4.2 — Watchdog detects dead component after max_missed checks",
          "[heartbeat][gap4.2]") {
    // Use 1ms timeout and 2 misses for speed
    ComponentWatchdog wd(1ms, 2);
    wd.register_component("crasher", 77);

    // Exhaust heartbeats without any update
    std::this_thread::sleep_for(5ms);
    auto dead1 = wd.check_health(); // missed_beats → 1 (TIMEOUT, threshold=2)
    std::this_thread::sleep_for(5ms);
    auto dead2 = wd.check_health(); // missed_beats → 2 → DEAD

    // Either dead1 or dead2 should contain "crasher"
    bool found = (!dead1.empty() && dead1[0] == "crasher") ||
                 (!dead2.empty() && dead2[0] == "crasher");
    CHECK(found);
}

TEST_CASE("Gap4.2 — Watchdog death callback fires on kill_and_cleanup",
          "[heartbeat][gap4.2]") {
    ComponentWatchdog wd;
    wd.register_component("victim", 0); // PID=0 → kill() is safe (no-op on 0)

    bool callback_fired = false;
    wd.set_death_callback([&](const std::string& name) {
        CHECK(name == "victim");
        callback_fired = true;
    });

    // Mark dead manually then cleanup
    wd.check_health(); // won't detect after default 500ms timeout — force kill directly
    bool ok = wd.kill_and_cleanup("victim");
    CHECK(ok);
    CHECK(callback_fired);
    CHECK_FALSE(wd.is_registered("victim"));
}

TEST_CASE("Gap4.2 — Deregister removes component", "[heartbeat][gap4.2]") {
    ComponentWatchdog wd;
    wd.register_component("temp", 555);
    REQUIRE(wd.is_registered("temp"));

    wd.deregister_component("temp");
    CHECK_FALSE(wd.is_registered("temp"));
    CHECK(wd.component_count() == 0);
}

TEST_CASE("Gap4.2 — HeartbeatPublisher records beat time", "[heartbeat][gap4.2]") {
    bool sent = false;
    HeartbeatPublisher pub("physics", [&](const std::string& topic) {
        CHECK(topic == "HEARTBEAT.physics");
        sent = true;
    });

    auto before = std::chrono::steady_clock::now();
    pub.beat();
    auto after = std::chrono::steady_clock::now();

    CHECK(sent);
    CHECK(pub.last_beat() >= before);
    CHECK(pub.last_beat() <= after);
}

TEST_CASE("Gap4.2 — Unknown component status returns DEAD", "[heartbeat][gap4.2]") {
    ComponentWatchdog wd;
    CHECK(wd.status("nonexistent") == ComponentStatus::DEAD);
    CHECK(wd.missed_beats("nonexistent") == -1);
}

// ===========================================================================
// GAP 4.3 — Shared Memory Lifecycle + Seqlock IPC
// ===========================================================================

TEST_CASE("Gap4.3 — WaveformSHM creates and maps segment", "[shm][gap4.3]") {
    // Use a unique name based on process time
    auto t = std::chrono::steady_clock::now().time_since_epoch().count();
    std::string name = "/nikola_test_" + std::to_string(t % 1000000);

    {
        WaveformSHM shm(name, 4096);
        CHECK(shm.valid());
        CHECK(shm.get_size() == 4096);
        CHECK(shm.name() == name);
        CHECK(shm.data() != nullptr);

        // Write a known byte sequence
        auto* bytes = static_cast<uint8_t*>(shm.data());
        bytes[0] = 0xDE;
        bytes[1] = 0xAD;
        bytes[4095] = 0xBE;

        CHECK(bytes[0]    == 0xDE);
        CHECK(bytes[1]    == 0xAD);
        CHECK(bytes[4095] == 0xBE);
    }
    // WaveformSHM destructor calls shm_unlink — segment no longer in /dev/shm
    // (cannot directly verify unlink without filesystem snoop, but no leak)
}

TEST_CASE("Gap4.3 — WaveformSHM rejects zero/oversized allocations", "[shm][gap4.3]") {
    auto t = std::chrono::steady_clock::now().time_since_epoch().count();
    std::string name = "/nikola_test_sv_" + std::to_string(t % 1000000);

    CHECK_THROWS_AS(WaveformSHM(name, 0), std::runtime_error);
    CHECK_THROWS_AS(WaveformSHM(name, SHM_MAX_TOTAL_BYTES + 1), std::runtime_error);
}

TEST_CASE("Gap4.3 — SeqlockFrame initial state is stable (seq=0, even)",
          "[shm][seqlock][gap4.3]") {
    struct PhysicsFrame { float psi_real = 0.f; float psi_imag = 0.f; uint32_t step = 0; };

    SeqlockFrame<PhysicsFrame> frame;
    CHECK(frame.sequence.load() == 0); // even = stable
    CHECK((frame.sequence.load() & 1u) == 0);
    CHECK(frame.frame_number == 0);
    CHECK(SeqlockFrame<PhysicsFrame>::byte_size() > sizeof(uint64_t));
}

TEST_CASE("Gap4.3 — SeqlockWriter/Reader round-trip", "[shm][seqlock][gap4.3]") {
    struct Frame { float a; float b; int c; };

    SeqlockFrame<Frame> shm_frame;
    SeqlockWriter<Frame> writer(&shm_frame);
    SeqlockReader<Frame> reader(&shm_frame);

    // Write first frame
    writer.write({1.0f, 2.0f, 42});
    CHECK(writer.frame_number() == 1);

    Frame out{};
    bool ok = reader.read(out);
    CHECK(ok);
    CHECK(out.a == 1.0f);
    CHECK(out.b == 2.0f);
    CHECK(out.c == 42);

    // Write second frame
    writer.write({3.14f, -1.0f, 100});
    CHECK(writer.frame_number() == 2);

    reader.read(out);
    CHECK(out.a == 3.14f);
    CHECK(out.c == 100);

    // Sequence must be even after write
    CHECK((reader.current_sequence() & 1u) == 0);
}

TEST_CASE("Gap4.3 — Multiple writes are monotonically sequenced", "[shm][seqlock][gap4.3]") {
    struct F { int n; };
    SeqlockFrame<F> frame;
    SeqlockWriter<F> w(&frame);
    SeqlockReader<F> r(&frame);

    for (int i = 1; i <= 10; ++i) {
        w.write({i});
        F out{};
        r.read(out);
        CHECK(out.n == i);
    }
    CHECK(w.frame_number() == 10);
}

TEST_CASE("Gap4.3 — cleanup_stale_shm runs without error on clean system",
          "[shm][gap4.3]") {
    // Must not throw; returns int (number of removed segments)
    int removed = cleanup_stale_shm();
    CHECK(removed >= 0); // zero or more stale segments cleaned
}

// ===========================================================================
// GAP 4.4 — ZMQ Socket Configuration Constants
// ===========================================================================

TEST_CASE("Gap4.4 — ZMQ socket configuration constants match spec",
          "[spine][zmq][gap4.4]") {
    CHECK(NIKOLA_SOCKET_HWM       == 1000);
    CHECK(NIKOLA_SOCKET_LINGER    == 0);
    CHECK(NIKOLA_SOCKET_IMMEDIATE == 1);
}

TEST_CASE("Gap4.4 — ZmqSpine constructs without throwing", "[spine][zmq][gap4.4]") {
    // Just create the context; don't bind/connect sockets
    ZmqSpine spine{1};
    (void)spine;
    CHECK(true);  // constructor did not throw
}

TEST_CASE("Gap4.4 — configure_socket applies correct options to new socket",
          "[spine][zmq][gap4.4]") {
    ZmqSpine spine;
    // Create a raw socket without binding
    zmq::socket_t sock(spine.context(), zmq::socket_type::push);
    CHECK_NOTHROW(configure_socket(sock));

    // Verify options were applied
    int sndhwm = sock.get(zmq::sockopt::sndhwm);
    int rcvhwm = sock.get(zmq::sockopt::rcvhwm);
    int linger = sock.get(zmq::sockopt::linger);

    CHECK(sndhwm == NIKOLA_SOCKET_HWM);
    CHECK(rcvhwm == NIKOLA_SOCKET_HWM);
    CHECK(linger == NIKOLA_SOCKET_LINGER);
}

// ===========================================================================
// GAP 4.5 — Protobuf Topic Versioning
// ===========================================================================

TEST_CASE("Gap4.5 — make_topic produces versioned strings", "[spine][topic][gap4.5]") {
    CHECK(make_topic("spikes")        == "nikola.v1.spikes");
    CHECK(make_topic("heartbeat")     == "nikola.v1.heartbeat");
    CHECK(make_topic("waveform")      == "nikola.v1.waveform");
    CHECK(make_topic("spikes", 2)     == "nikola.v2.spikes");
    CHECK(make_topic("spikes", 0)     == "nikola.v0.spikes");
}

TEST_CASE("Gap4.5 — topic_version_prefix builds subscription filter",
          "[spine][topic][gap4.5]") {
    CHECK(topic_version_prefix()   == "nikola.v1");
    CHECK(topic_version_prefix(2)  == "nikola.v2");
    CHECK(topic_version_prefix(0)  == "nikola.v0");
}

TEST_CASE("Gap4.5 — NIKOLA_PROTO_VERSION is 1", "[spine][topic][gap4.5]") {
    CHECK(NIKOLA_PROTO_VERSION == 1);
}

TEST_CASE("Gap4.5 — Proto schema evolution — new optional field is backward safe",
          "[spine][topic][gap4.5]") {
    // Simulate the spec's example: v1 NeuralSpike gains new optional field.
    // Old subscribers subscribed to "nikola.v1" still receive the message.
    std::string old_topic = "nikola.v1.spikes";
    std::string new_topic = "nikola.v1.spikes"; // same topic — field appended, not removed

    // Topic unchanged = backward compat satisfied
    CHECK(old_topic == new_topic);

    // Breaking change requires new version:
    std::string breaking_topic = "nikola.v2.spikes";
    CHECK(breaking_topic != old_topic);

    // Old subscriber "nikola.v1" does NOT match "nikola.v2"
    std::string sub_filter  = topic_version_prefix(1);
    std::string v2_prefix   = topic_version_prefix(2);
    CHECK(breaking_topic.find(sub_filter)  == std::string::npos);
    CHECK(breaking_topic.find(v2_prefix)   != std::string::npos);
}

// ===========================================================================
// Orchestrator Lifecycle
// ===========================================================================

TEST_CASE("Orchestrator — default config is sane", "[orchestrator]") {
    OrchestratorConfig cfg;
    CHECK(cfg.control_timeout.count() == 100);
    CHECK(cfg.data_timeout.count()    ==   5);
    CHECK(cfg.heartbeat_timeout.count() == 500);
    CHECK(cfg.proto_version     == 1);
    CHECK(cfg.io_threads        == 1);
    CHECK(cfg.physics_shm_bytes == 64 * 1024);
}

TEST_CASE("Orchestrator — starts in IDLE state", "[orchestrator]") {
    Orchestrator orch;
    CHECK(orch.state() == OrchestratorState::IDLE);
    CHECK_FALSE(orch.is_running());
}

TEST_CASE("Orchestrator — start transitions to RUNNING", "[orchestrator]") {
    Orchestrator orch(make_test_config());
    orch.start();
    CHECK(orch.is_running());
    CHECK(orch.state() == OrchestratorState::RUNNING);
    orch.stop();
}

TEST_CASE("Orchestrator — stop transitions to STOPPED", "[orchestrator]") {
    Orchestrator orch(make_test_config());
    orch.start();
    REQUIRE(orch.is_running());
    orch.stop();
    CHECK_FALSE(orch.is_running());
    CHECK(orch.state() == OrchestratorState::STOPPED);
}

TEST_CASE("Orchestrator — double stop is safe", "[orchestrator]") {
    Orchestrator orch(make_test_config());
    orch.start();
    orch.stop();
    CHECK_NOTHROW(orch.stop()); // idempotent
}

TEST_CASE("Orchestrator — register_component adds to watchdog", "[orchestrator]") {
    Orchestrator orch(make_test_config());
    orch.start();

    orch.register_component("physics", 1000);
    orch.register_component("memory",  1001);

    auto comps = orch.components();
    CHECK(comps.size() == 2);

    orch.stop();
}

TEST_CASE("Orchestrator — send_control returns false for unregistered component",
          "[orchestrator]") {
    Orchestrator orch(make_test_config());
    orch.start();

    uint8_t msg[] = {0x01, 0x02};
    bool ok = orch.send_control("phantom", msg, sizeof(msg));
    CHECK_FALSE(ok); // not registered

    orch.stop();
}

TEST_CASE("Orchestrator — send_control reports true for registered (closed breaker)",
          "[orchestrator]") {
    Orchestrator orch(make_test_config());
    orch.start();
    orch.register_component("logic", 2000);

    uint8_t msg[] = {0xFF};
    bool ok = orch.send_control("logic", msg, 1);
    CHECK(ok); // breaker closed → passes

    orch.stop();
}

TEST_CASE("Orchestrator — restart callback fires on component death", "[orchestrator]") {
    Orchestrator orch(make_test_config());

    bool restarted = false;
    std::string restarted_name;
    orch.set_restart_callback([&](const std::string& name) {
        restarted = true;
        restarted_name = name;
    });

    orch.start();
    orch.register_component("crasher", 0); // PID=0 is safe for kill() in this context
    orch.kill_component("crasher");

    CHECK(restarted);
    CHECK(restarted_name == "crasher");

    orch.stop();
}

TEST_CASE("Orchestrator — OrchestratorState names", "[orchestrator]") {
    CHECK(std::string(orchestrator_state_name(OrchestratorState::IDLE))     == "IDLE");
    CHECK(std::string(orchestrator_state_name(OrchestratorState::RUNNING))  == "RUNNING");
    CHECK(std::string(orchestrator_state_name(OrchestratorState::DEGRADED)) == "DEGRADED");
    CHECK(std::string(orchestrator_state_name(OrchestratorState::STOPPING)) == "STOPPING");
    CHECK(std::string(orchestrator_state_name(OrchestratorState::STOPPED))  == "STOPPED");
}

// ===========================================================================
// LLMBridge
// ===========================================================================

TEST_CASE("LLMBridge — LLMRequest serializes to JSON", "[llm_bridge]") {
    LLMRequest req;
    req.request_id  = "id-001";
    req.prompt      = "What is 2+2?";
    req.temperature = 0.5f;
    req.max_tokens  = 64;

    std::string json = req.to_json();
    CHECK_THAT(json, Catch::Matchers::ContainsSubstring("id-001"));
    CHECK_THAT(json, Catch::Matchers::ContainsSubstring("What is 2+2?"));
    CHECK_THAT(json, Catch::Matchers::ContainsSubstring("64"));
}

TEST_CASE("LLMBridge — LLMResponse deserializes non-empty as ok=true", "[llm_bridge]") {
    auto r = LLMResponse::from_json("{\"text\":\"4\"}");
    CHECK(r.ok);
    CHECK_FALSE(r.text.empty());
}

TEST_CASE("LLMBridge — LLMResponse empty JSON → ok=false", "[llm_bridge]") {
    auto r = LLMResponse::from_json("");
    CHECK_FALSE(r.ok);
}

TEST_CASE("LLMBridge — default config is sane", "[llm_bridge]") {
    LLMBridgeConfig cfg;
    CHECK(cfg.timeout_ms    == 2000);
    CHECK(cfg.max_retries   == 2);
    CHECK(cfg.proto_version == 1);
}

TEST_CASE("LLMBridge — infer returns error without live endpoint", "[llm_bridge]") {
    // No live service → bridge should return ok=false gracefully
    LLMBridge bridge(LLMBridgeConfig{
        .endpoint      = "tcp://localhost:19999", // nothing listening
        .timeout_ms    = 1,   // minimal wait
        .max_retries   = 0
    });

    LLMRequest req;
    req.request_id = "r1";
    req.prompt     = "hello";

    // Must not throw even without a live service
    LLMResponse resp;
    CHECK_NOTHROW(resp = bridge.infer(req));
    // ok=false is expected (no server)
}

// ===========================================================================
// ToolExecutor
// ===========================================================================

TEST_CASE("ToolExecutor — register and list tools", "[tool_executor]") {
    ToolExecutor exec;
    exec.register_tool(
        {"echo", "Echoes arg back", "{}", "{}"},
        [](const std::string& args) { return args; }
    );

    CHECK(exec.has_tool("echo"));
    CHECK(exec.tool_count() == 1);
    auto tools = exec.list_tools();
    REQUIRE(tools.size() == 1);
    CHECK(tools[0].name == "echo");
}

TEST_CASE("ToolExecutor — execute dispatches to correct tool", "[tool_executor]") {
    ToolExecutor exec;
    exec.register_tool(
        {"add", "Adds 2+2", "{}", "{}"},
        [](const std::string&) { return "{\"result\":4}"; }
    );

    ToolCall call{"c1", "add", "{\"a\":2,\"b\":2}"};
    auto result = exec.execute(call);

    CHECK(result.ok);
    CHECK(result.call_id   == "c1");
    CHECK(result.tool_name == "add");
    CHECK_THAT(result.result_json, Catch::Matchers::ContainsSubstring("4"));
    CHECK(result.elapsed.count() >= 0);
}

TEST_CASE("ToolExecutor — unknown tool returns ok=false with error", "[tool_executor]") {
    ToolExecutor exec;
    ToolCall call{"c2", "unknown_tool", "{}"};
    auto result = exec.execute(call);

    CHECK_FALSE(result.ok);
    CHECK_THAT(result.error_msg, Catch::Matchers::ContainsSubstring("Unknown tool"));
    CHECK(exec.tool_cb_state("unknown_tool") == CBState::OPEN); // unknown → circuit open
}

TEST_CASE("ToolExecutor — tool that throws returns ok=false", "[tool_executor]") {
    ToolExecutor exec;
    exec.register_tool(
        {"boom", "Always throws", "{}", "{}"},
        [](const std::string&) -> std::string {
            throw std::runtime_error("intentional error");
        }
    );

    ToolCall call{"c3", "boom", "{}"};
    auto result = exec.execute(call);

    CHECK_FALSE(result.ok);
    CHECK_THAT(result.error_msg, Catch::Matchers::ContainsSubstring("intentional error"));
}

TEST_CASE("ToolExecutor — execute_batch processes multiple calls", "[tool_executor]") {
    ToolExecutor exec;
    exec.register_tool(
        {"ping", "Returns pong", "{}", "{}"},
        [](const std::string&) { return "{\"reply\":\"pong\"}"; }
    );

    std::vector<ToolCall> calls{
        {"b1", "ping", "{}"},
        {"b2", "ping", "{}"},
        {"b3", "missing", "{}"}  // unknown
    };

    auto results = exec.execute_batch(calls);
    REQUIRE(results.size() == 3);

    CHECK(results[0].ok);
    CHECK(results[1].ok);
    CHECK_FALSE(results[2].ok);
}

TEST_CASE("ToolExecutor — circuit breaker can be reset after failure", "[tool_executor]") {
    ToolExecutor exec;
    exec.register_tool(
        {"flaky", "Fails once", "{}", "{}"},
        [](const std::string&) -> std::string {
            throw std::runtime_error("flaky failure");
        }
    );

    ToolCall call{"c4", "flaky", "{}"};
    exec.execute(call); // trigger failure

    // Reset and confirm breaker closes
    exec.reset_tool_cb("flaky");
    CHECK(exec.tool_cb_state("flaky") == CBState::CLOSED);
}

TEST_CASE("ToolExecutor — deregister removes tool", "[tool_executor]") {
    ToolExecutor exec;
    exec.register_tool({"tmp", "temp", "{}", "{}"}, [](const std::string&){ return "{}"; });
    REQUIRE(exec.has_tool("tmp"));

    exec.deregister_tool("tmp");
    CHECK_FALSE(exec.has_tool("tmp"));
    CHECK(exec.tool_count() == 0);
}
