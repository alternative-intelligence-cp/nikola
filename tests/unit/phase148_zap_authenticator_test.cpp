// ============================================================
// tests/unit/phase148_zap_authenticator_test.cpp
// Phase 148 — v0.1.11: ZAP Authenticator + CurveZMQ Integration
//
// Validates the ZAP handler thread, whitelist enforcement, and
// end-to-end CurveZMQ authenticated message delivery.
// ============================================================

#include <catch2/catch_test_macros.hpp>

#include <nikola/security/ironhouse.hpp>
#include <nikola/security/zap_authenticator.hpp>
#include <nikola/infrastructure/spine.hpp>

#include <chrono>
#include <cstring>
#include <string>
#include <thread>

using namespace nikola::security;
using namespace nikola::infrastructure;
using namespace std::chrono_literals;

// ────────────────────────────────────────────────────────────────────────────
// §1  ZapWhitelist basics (from ironhouse.hpp, but tested here for coverage)
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase148 — ZapWhitelist starts empty", "[zap][phase148]") {
    ZapWhitelist wl;
    CHECK(wl.empty());
    CHECK(wl.size() == 0);
}

TEST_CASE("Phase148 — ZapWhitelist add and check", "[zap][phase148]") {
    auto kp = generate_ironhouse_keypair();
    ZapWhitelist wl;
    wl.add_key(kp.pub());
    CHECK(wl.size() == 1);
    CHECK(wl.is_authorized(kp.pub()));
}

TEST_CASE("Phase148 — ZapWhitelist deny unknown key", "[zap][phase148]") {
    auto kp = generate_ironhouse_keypair();
    auto other = generate_ironhouse_keypair();
    ZapWhitelist wl;
    wl.add_key(kp.pub());
    CHECK_FALSE(wl.is_authorized(other.pub()));
}

TEST_CASE("Phase148 — ZapWhitelist remove key", "[zap][phase148]") {
    auto kp = generate_ironhouse_keypair();
    ZapWhitelist wl;
    wl.add_key(kp.pub());
    CHECK(wl.is_authorized(kp.pub()));
    wl.remove_key(kp.pub());
    CHECK_FALSE(wl.is_authorized(kp.pub()));
    CHECK(wl.empty());
}

TEST_CASE("Phase148 — ZapWhitelist clear empties all keys", "[zap][phase148]") {
    ZapWhitelist wl;
    for (int i = 0; i < 5; ++i) {
        wl.add_key(generate_ironhouse_keypair().pub());
    }
    CHECK(wl.size() == 5);
    wl.clear();
    CHECK(wl.empty());
}

// ────────────────────────────────────────────────────────────────────────────
// §2  ZapAuthenticator lifecycle
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase148 — ZapAuthenticator starts and stops cleanly", "[zap][phase148]") {
    zmq::context_t ctx(1);
    ZapWhitelist wl;
    ZapAuthenticator auth(ctx, wl);

    CHECK_FALSE(auth.is_running());
    auth.start();
    CHECK(auth.is_running());
    auth.stop();
    CHECK_FALSE(auth.is_running());
}

TEST_CASE("Phase148 — ZapAuthenticator double start is idempotent", "[zap][phase148]") {
    zmq::context_t ctx(1);
    ZapWhitelist wl;
    ZapAuthenticator auth(ctx, wl);

    auth.start();
    CHECK_NOTHROW(auth.start());
    CHECK(auth.is_running());
    auth.stop();
}

TEST_CASE("Phase148 — ZapAuthenticator double stop is safe", "[zap][phase148]") {
    zmq::context_t ctx(1);
    ZapWhitelist wl;
    ZapAuthenticator auth(ctx, wl);

    auth.start();
    auth.stop();
    CHECK_NOTHROW(auth.stop());
}

TEST_CASE("Phase148 — ZapAuthenticator destructor stops thread", "[zap][phase148]") {
    zmq::context_t ctx(1);
    ZapWhitelist wl;
    {
        ZapAuthenticator auth(ctx, wl);
        auth.start();
        CHECK(auth.is_running());
    } // destructor should stop + join
    // If we get here without hanging, destructor worked
    CHECK(true);
}

// ────────────────────────────────────────────────────────────────────────────
// §3  CurveZMQ Ironhouse helpers
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase148 — generate_ironhouse_keypair produces valid keys", "[ironhouse][phase148]") {
    auto kp = generate_ironhouse_keypair();
    CHECK(kp.valid());
    CHECK(IronhouseKeypair::is_valid_z85(kp.pub()));
    CHECK(IronhouseKeypair::is_valid_z85(kp.sec()));
}

TEST_CASE("Phase148 — keypairs are unique", "[ironhouse][phase148]") {
    auto kp1 = generate_ironhouse_keypair();
    auto kp2 = generate_ironhouse_keypair();
    CHECK(kp1.pub() != kp2.pub());
    CHECK(kp1.sec() != kp2.sec());
}

TEST_CASE("Phase148 — KeyTier rotation periods", "[ironhouse][phase148]") {
    CHECK(rotation_period_seconds(KeyTier::SPINE_BROKER) == 0);
    CHECK(rotation_period_seconds(KeyTier::CORE_COMPONENT) == 2592000);
    CHECK(rotation_period_seconds(KeyTier::EPHEMERAL_AGENT) == 86400);
}

// ────────────────────────────────────────────────────────────────────────────
// §4  End-to-end CurveZMQ authenticated message via ZAP (TCP only)
// ────────────────────────────────────────────────────────────────────────────

static int g_curve_port = 18800;

TEST_CASE("Phase148 — CurveZMQ PUSH/PULL with ZAP auth — authorized client",
          "[zap][ironhouse][e2e][phase148]") {
    zmq::context_t ctx(1);
    std::string endpoint = "tcp://127.0.0.1:" + std::to_string(g_curve_port++);

    // Generate keypairs
    auto server_kp = generate_ironhouse_keypair();
    auto client_kp = generate_ironhouse_keypair();

    // Whitelist: authorize the client
    ZapWhitelist wl;
    wl.add_key(client_kp.pub());

    // Start ZAP authenticator
    ZapAuthenticator auth(ctx, wl);
    auth.start();
    std::this_thread::sleep_for(50ms); // let ZAP thread bind

    // Server: PULL socket with CurveZMQ server role
    zmq::socket_t server(ctx, zmq::socket_type::pull);
    configure_socket(server);
    configure_curve_server(server, server_kp);
    server.bind(endpoint);

    // Client: PUSH socket with CurveZMQ client role
    zmq::socket_t client(ctx, zmq::socket_type::push);
    configure_socket(client);
    configure_curve_client(client, client_kp, server_kp.pub());
    client.connect(endpoint);

    std::this_thread::sleep_for(200ms); // CurveZMQ handshake

    // Send a message
    const std::string payload = "AUTHENTICATED_MSG";
    zmq::message_t send_msg(payload.data(), payload.size());
    auto rc = client.send(send_msg, zmq::send_flags::none);
    REQUIRE(rc.has_value());

    // Receive with timeout
    zmq::pollitem_t items[] = {{server.handle(), 0, ZMQ_POLLIN, 0}};
    zmq::poll(items, 1, 2000ms);

    REQUIRE(items[0].revents & ZMQ_POLLIN);
    zmq::message_t recv_msg;
    auto r = server.recv(recv_msg, zmq::recv_flags::none);
    REQUIRE(r.has_value());

    std::string received(static_cast<const char*>(recv_msg.data()), recv_msg.size());
    CHECK(received == payload);

    // ZAP handler should have processed at least one auth request
    CHECK(auth.requests_processed() >= 1);

    auth.stop();
}

TEST_CASE("Phase148 — CurveZMQ denies unauthorized client",
          "[zap][ironhouse][e2e][phase148]") {
    zmq::context_t ctx(1);
    std::string endpoint = "tcp://127.0.0.1:" + std::to_string(g_curve_port++);

    auto server_kp = generate_ironhouse_keypair();
    auto client_kp = generate_ironhouse_keypair();

    // Whitelist is EMPTY — no clients authorized
    ZapWhitelist wl;

    ZapAuthenticator auth(ctx, wl);
    auth.start();
    std::this_thread::sleep_for(50ms);

    zmq::socket_t server(ctx, zmq::socket_type::pull);
    configure_socket(server);
    configure_curve_server(server, server_kp);
    server.bind(endpoint);

    zmq::socket_t client(ctx, zmq::socket_type::push);
    configure_socket(client);
    configure_curve_client(client, client_kp, server_kp.pub());
    client.connect(endpoint);

    std::this_thread::sleep_for(300ms); // give time for handshake to fail

    // Try to send — handshake should have been rejected
    const std::string payload = "UNAUTHORIZED_MSG";
    zmq::message_t msg(payload.data(), payload.size());
    try {
        client.send(msg, zmq::send_flags::dontwait);
    } catch (...) {
        // Expected: send may throw if socket is in bad state
    }

    // Server should NOT receive anything within timeout
    zmq::pollitem_t items[] = {{server.handle(), 0, ZMQ_POLLIN, 0}};
    zmq::poll(items, 1, 500ms);
    CHECK_FALSE(items[0].revents & ZMQ_POLLIN);

    // ZAP handler should have processed the rejected auth request
    CHECK(auth.requests_denied() >= 1);

    auth.stop();
}
