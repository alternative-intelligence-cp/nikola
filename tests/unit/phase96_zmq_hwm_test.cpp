/**
 * @file tests/unit/phase96_zmq_hwm_test.cpp
 * @brief Phase 96: ZMQ HWM Backpressure — GAP-039 (Catch2 v3).
 *
 * Validates the canonical socket configuration (HWM / LINGER / IMMEDIATE) and
 * the backpressure enforcement behaviour specified in §3.4 of the ZeroMQ Spine
 * design and Gap 4.4 of the Infrastructure & Communications implementation spec:
 *
 *   SNDHWM = RCVHWM = 1000  — drop messages when queue overflows (real-time safe)
 *   LINGER  = 0             — close() returns immediately; discard unsent messages
 *   IMMEDIATE = 1           — only queue messages when a peer connection exists
 *
 * Test sections:
 *
 *   Section 1 — Compile-time constants: correct numeric values
 *   Section 2 — configure_socket(): sockopt readback matches constants
 *   Section 3 — HWM enforcement: DONTWAIT send returns nullopt when queue full
 *   Section 4 — Round-trip integrity: N messages sent → N messages received in order
 *   Section 5 — LINGER=0: socket close completes in < 100 ms even with pending msgs
 *   Section 6 — IMMEDIATE=1: queuing blocked when no peer is connected
 *   Section 7 — ZmqSpine factory: make_push / make_pull return pre-configured sockets
 *   Section 8 — ZmqSpine versioned publish helpers: topic format is correct
 *
 * All tests are entirely self-contained (own ZMQ contexts, ipc:///tmp/nikola_p96_*
 * endpoints) — they do not depend on a running Nikola process.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/infrastructure/spine.hpp>

#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

using namespace nikola::infrastructure;
using namespace std::chrono_literals;

// ─────────────────────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build a unique ipc endpoint string using a tag.
static std::string ipc(const char* tag) {
    return std::string("ipc:///tmp/nikola_p96_") + tag + ".ipc";
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 1 — Compile-time constants
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase96 — compile-time socket constants have correct values",
          "[phase96][constants]")
{
    // Canonical values required by the implementation spec (Gap 4.4)
    STATIC_CHECK(NIKOLA_SOCKET_HWM       == 1000);
    STATIC_CHECK(NIKOLA_SOCKET_LINGER    == 0);
    STATIC_CHECK(NIKOLA_SOCKET_IMMEDIATE == 1);

    // Proto topic version is 1
    STATIC_CHECK(NIKOLA_PROTO_VERSION == 1);

    // Topic prefix is "nikola"
    STATIC_CHECK(NIKOLA_TOPIC_PREFIX == "nikola");
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 2 — configure_socket() sockopt readback
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase96 — configure_socket() applies correct sockopt values",
          "[phase96][configure_socket]")
{
    zmq::context_t ctx(1);

    SECTION("PUSH socket") {
        zmq::socket_t sock(ctx, zmq::socket_type::push);
        configure_socket(sock);

        CHECK(sock.get(zmq::sockopt::sndhwm)    == NIKOLA_SOCKET_HWM);
        CHECK(sock.get(zmq::sockopt::rcvhwm)    == NIKOLA_SOCKET_HWM);
        CHECK(sock.get(zmq::sockopt::linger)    == NIKOLA_SOCKET_LINGER);
        CHECK(sock.get(zmq::sockopt::immediate) == NIKOLA_SOCKET_IMMEDIATE);
    }

    SECTION("PULL socket") {
        zmq::socket_t sock(ctx, zmq::socket_type::pull);
        configure_socket(sock);

        CHECK(sock.get(zmq::sockopt::sndhwm) == NIKOLA_SOCKET_HWM);
        CHECK(sock.get(zmq::sockopt::rcvhwm) == NIKOLA_SOCKET_HWM);
        CHECK(sock.get(zmq::sockopt::linger) == NIKOLA_SOCKET_LINGER);
    }

    SECTION("PUB socket") {
        zmq::socket_t sock(ctx, zmq::socket_type::pub);
        configure_socket(sock);

        CHECK(sock.get(zmq::sockopt::sndhwm) == NIKOLA_SOCKET_HWM);
        CHECK(sock.get(zmq::sockopt::linger) == NIKOLA_SOCKET_LINGER);
    }

    SECTION("SUB socket") {
        zmq::socket_t sock(ctx, zmq::socket_type::sub);
        configure_socket(sock);

        CHECK(sock.get(zmq::sockopt::rcvhwm) == NIKOLA_SOCKET_HWM);
        CHECK(sock.get(zmq::sockopt::linger) == NIKOLA_SOCKET_LINGER);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 3 — HWM enforcement: DONTWAIT returns nullopt when queue saturated
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase96 — HWM backpressure: DONTWAIT send returns nullopt when queue full",
          "[phase96][hwm][backpressure]")
{
    // Use a smaller HWM (TEST_HWM) to keep the test fast and deterministic.
    // The production value (NIKOLA_SOCKET_HWM = 1000) is tested in Section 2.
    constexpr int TEST_HWM   = 5;
    constexpr int BURST_SIZE = 200;  // far more than 2 × TEST_HWM

    const std::string ep = ipc("hwm_bp");

    zmq::context_t ctx(1);

    // PULL (receiver) — bound, TEST_HWM capacity
    zmq::socket_t puller(ctx, zmq::socket_type::pull);
    puller.set(zmq::sockopt::rcvhwm, TEST_HWM);
    puller.set(zmq::sockopt::linger, 0);
    puller.bind(ep);

    // PUSH (sender) — connected, TEST_HWM capacity
    zmq::socket_t pusher(ctx, zmq::socket_type::push);
    pusher.set(zmq::sockopt::sndhwm,    TEST_HWM);
    pusher.set(zmq::sockopt::linger,    0);
    pusher.set(zmq::sockopt::immediate, 1);
    pusher.connect(ep);

    // Allow the IPC connection to fully establish
    std::this_thread::sleep_for(10ms);

    // Blast BURST_SIZE small messages using non-blocking sends.
    // The combined send+receive queue holds at most ~2×TEST_HWM messages.
    int sent   = 0;
    int eagain = 0;
    const std::string payload = "p";

    for (int i = 0; i < BURST_SIZE; ++i) {
        auto result = pusher.send(zmq::buffer(payload),
                                  zmq::send_flags::dontwait);
        if (result.has_value()) {
            ++sent;
        } else {
            ++eagain;  // nullopt ≡ EAGAIN (would-block)
        }
    }

    INFO("sent=" << sent << "  eagain=" << eagain
         << "  TEST_HWM=" << TEST_HWM);

    // At least some messages must have been accepted
    CHECK(sent > 0);

    // Total must account for all attempted sends
    CHECK(sent + eagain == BURST_SIZE);

    // Backpressure MUST have kicked in: EAGAIN count must be non-zero.
    // Queue capacity = sndhwm + rcvhwm = 2 × TEST_HWM = 10.  We blasted 200.
    CHECK(eagain > 0);

    // Sanity bound: the queue can hold at most 4×TEST_HWM messages
    // (accounting for ZMQ's internal OS-level buffering).
    CHECK(sent <= 4 * TEST_HWM);

    // ── Allow IPC transit: messages travel from PUSH send buffer → PULL recv buffer ──
    std::this_thread::sleep_for(30ms);

    // ── Now drain the queue and verify what we can receive ───────────────────
    int drained = 0;
    while (true) {
        zmq::message_t msg;
        auto res = puller.recv(msg, zmq::recv_flags::dontwait);
        if (!res.has_value()) break;
        ++drained;
    }
    INFO("drained=" << drained);
    CHECK(drained == sent);  // every sent message must be drainable
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 4 — Round-trip integrity through a HWM queue
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase96 — round-trip: N messages sent ≤ HWM arrive intact and in order",
          "[phase96][roundtrip]")
{
    constexpr int N  = 50;   // well below NIKOLA_SOCKET_HWM
    const std::string ep = ipc("rtrip");

    zmq::context_t ctx(1);

    zmq::socket_t puller(ctx, zmq::socket_type::pull);
    configure_socket(puller);
    puller.bind(ep);

    zmq::socket_t pusher(ctx, zmq::socket_type::push);
    configure_socket(pusher);
    pusher.connect(ep);

    std::this_thread::sleep_for(10ms);

    // Send N numbered messages (blocking — queue not yet full)
    for (int i = 0; i < N; ++i) {
        const std::string msg = "msg:" + std::to_string(i);
        pusher.send(zmq::buffer(msg), zmq::send_flags::none);
    }

    // Receive all N messages
    std::vector<std::string> received;
    received.reserve(N);

    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (static_cast<int>(received.size()) < N &&
           std::chrono::steady_clock::now() < deadline) {
        zmq::message_t msg;
        auto res = puller.recv(msg, zmq::recv_flags::dontwait);
        if (res.has_value()) {
            received.emplace_back(static_cast<const char*>(msg.data()), msg.size());
        } else {
            std::this_thread::sleep_for(1ms);
        }
    }

    REQUIRE(static_cast<int>(received.size()) == N);

    // Verify ordering and content
    for (int i = 0; i < N; ++i) {
        const std::string expected = "msg:" + std::to_string(i);
        CHECK(received[static_cast<std::size_t>(i)] == expected);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 5 — LINGER=0 fast close
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase96 — LINGER=0: socket close completes in < 100 ms",
          "[phase96][linger]")
{
    SECTION("PUSH socket with pending messages closes instantly") {
        const std::string ep = ipc("linger");

        zmq::context_t ctx(1);

        // Receiver — holds the connection but does not read
        zmq::socket_t puller(ctx, zmq::socket_type::pull);
        puller.set(zmq::sockopt::rcvhwm, 1000);
        puller.set(zmq::sockopt::linger, 0);
        puller.bind(ep);

        // Sender — LINGER=0
        zmq::socket_t pusher(ctx, zmq::socket_type::push);
        pusher.set(zmq::sockopt::sndhwm, 1000);
        pusher.set(zmq::sockopt::linger, 0);
        pusher.connect(ep);

        std::this_thread::sleep_for(10ms);

        // Push some messages without blocking
        const std::string payload = "linger_test";
        for (int i = 0; i < 10; ++i) {
            pusher.send(zmq::buffer(payload), zmq::send_flags::dontwait);
        }

        // Time the close
        const auto t0 = std::chrono::steady_clock::now();
        pusher.close();
        const auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count();

        INFO("pusher.close() took " << elapsed_ms << " ms");
        CHECK(elapsed_ms < 100);
    }

    SECTION("Linger sockopt reads back 0 before any connection") {
        zmq::context_t ctx(1);
        zmq::socket_t sock(ctx, zmq::socket_type::push);
        sock.set(zmq::sockopt::linger, 0);
        CHECK(sock.get(zmq::sockopt::linger) == 0);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 6 — IMMEDIATE=1: no queueing to unconnected peers
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase96 — IMMEDIATE=1: DONTWAIT send fails when no peer is connected",
          "[phase96][immediate]")
{
    zmq::context_t ctx(1);

    // PUSH socket with IMMEDIATE=1 and no peer connected
    zmq::socket_t pusher(ctx, zmq::socket_type::push);
    pusher.set(zmq::sockopt::sndhwm,    NIKOLA_SOCKET_HWM);
    pusher.set(zmq::sockopt::linger,    NIKOLA_SOCKET_LINGER);
    pusher.set(zmq::sockopt::immediate, 1);   // fail-fast: no queueing

    // No connect() call — there is no peer

    auto result = pusher.send(zmq::buffer(std::string("no_peer")),
                               zmq::send_flags::dontwait);

    // IMMEDIATE=1 means ZMQ won't accept the message for queuing when no peer
    // exists, so DONTWAIT send must return nullopt (EAGAIN)
    CHECK_FALSE(result.has_value());
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 7 — ZmqSpine factory
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase96 — ZmqSpine factory creates pre-configured sockets",
          "[phase96][spine][factory]")
{
    SECTION("make_pull returns configured PULL socket (bound)") {
        const std::string ep = ipc("spine_pull");
        ZmqSpine spine;
        auto sock = spine.make_pull(ep);

        // Socket must reflect canonical config
        CHECK(sock.get(zmq::sockopt::rcvhwm) == NIKOLA_SOCKET_HWM);
        CHECK(sock.get(zmq::sockopt::linger) == NIKOLA_SOCKET_LINGER);
    }

    SECTION("make_push returns configured PUSH socket (connected)") {
        // We need a bound endpoint to connect to
        const std::string ep = ipc("spine_push_pull");

        ZmqSpine spine_server;
        auto pull_sock = spine_server.make_pull(ep);   // binds

        ZmqSpine spine_client;
        auto push_sock = spine_client.make_push(ep);   // connects

        CHECK(push_sock.get(zmq::sockopt::sndhwm)    == NIKOLA_SOCKET_HWM);
        CHECK(push_sock.get(zmq::sockopt::linger)    == NIKOLA_SOCKET_LINGER);
        CHECK(push_sock.get(zmq::sockopt::immediate) == NIKOLA_SOCKET_IMMEDIATE);
    }

    SECTION("make_publisher + make_subscriber round-trip smoke test") {
        const std::string pub_ep = ipc("spine_pub");

        ZmqSpine pub_spine;
        auto pub_sock = pub_spine.make_publisher(pub_ep);

        ZmqSpine sub_spine;
        auto sub_sock = sub_spine.make_subscriber(pub_ep, "nikola.v1");

        // Slow-joiner: ZMQ must propagate the subscription filter from SUB→PUB
        // before any message arrives.  500ms is conservative but reliable.
        std::this_thread::sleep_for(500ms);

        std::string topic_str;
        std::string data_str;
        bool received = false;

        // Publish repeatedly (every 100ms) until the subscriber picks one up or
        // 5 seconds elapses.  This sidesteps any residual slow-joiner race.
        const auto dl = std::chrono::steady_clock::now() + 5s;
        while (!received && std::chrono::steady_clock::now() < dl) {
            ZmqSpine::publish(pub_sock, "test", "hello", 5);

            zmq_pollitem_t item{};
            item.socket = static_cast<void*>(sub_sock);
            item.events = ZMQ_POLLIN;
            zmq::poll(&item, 1, std::chrono::milliseconds(100));

            if (item.revents & ZMQ_POLLIN) {
                zmq::message_t t_msg, d_msg;
                [[maybe_unused]] auto _t = sub_sock.recv(t_msg);
                if (t_msg.more()) {
                    [[maybe_unused]] auto _d = sub_sock.recv(d_msg);
                    topic_str = std::string(static_cast<const char*>(t_msg.data()), t_msg.size());
                    data_str  = std::string(static_cast<const char*>(d_msg.data()), d_msg.size());
                    received  = true;
                }
            }
        }

        CHECK(received);
        CHECK(topic_str.rfind("nikola.v1.", 0) == 0);  // starts with "nikola.v1."
        CHECK(data_str == "hello");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 8 — Versioned topic helpers
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase96 — make_topic() and topic_version_prefix() produce correct strings",
          "[phase96][topic]")
{
    SECTION("make_topic with default version") {
        const std::string t = make_topic("spikes");
        CHECK(t == "nikola.v1.spikes");
    }

    SECTION("make_topic with explicit version 2") {
        const std::string t = make_topic("state", 2);
        CHECK(t == "nikola.v2.state");
    }

    SECTION("topic_version_prefix default") {
        const std::string p = topic_version_prefix();
        CHECK(p == "nikola.v1");
    }

    SECTION("topic_version_prefix version 3") {
        const std::string p = topic_version_prefix(3);
        CHECK(p == "nikola.v3");
    }

    SECTION("make_topic format: prefix + vN + subsystem") {
        const std::string t = make_topic("action", 1);
        CHECK(t.substr(0, 6) == "nikola");
        CHECK(t.find(".v1.") != std::string::npos);
        CHECK(t.find("action") != std::string::npos);
    }
}
