/**
 * @file tests/unit/phase98_ironhouse_test.cpp
 * @brief Phase 98: CurveZMQ Ironhouse identity management — GAP-010 (Catch2 v3).
 *
 * Validates the Ironhouse security model as specified in §1.3 of
 * 04_infrastructure/01_zeromq_spine.md and the GAP-010 RESOLUTION block:
 *
 *   "Every single connection is mutually authenticated and encrypted using
 *    Curve25519 cryptography. There are no 'public' endpoints within the spine."
 *
 * Test sections:
 *
 *   Section 1 — Compile-time constants: key lengths (Z85 and binary)
 *   Section 2 — KeyTier rotation period policy
 *   Section 3 — Key generation: valid Z85, randomness, pub ≠ sec
 *   Section 4 — IronhouseKeypair struct validity and helper methods
 *   Section 5 — Socket option readback: configure_curve_server/client
 *   Section 6 — Live CurveZMQ round-trip: encrypted PUSH → PULL
 *   Section 7 — Wrong server key: handshake fails, no messages received
 *   Section 8 — ZapWhitelist: add / remove / deny-by-default / bootstrap mode
 *
 * Self-contained: all ZMQ sockets use ipc:///tmp/nikola_p98_* endpoints.
 */

#include <catch2/catch_test_macros.hpp>

#include <nikola/security/ironhouse.hpp>

#include <chrono>
#include <string>
#include <thread>

#include <zmq.h>    // zmq_pollitem_t, zmq_poll

using namespace nikola::security;
using namespace std::chrono_literals;

// ─────────────────────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────────────────────

static std::string ep(const char* tag) {
    return std::string("ipc:///tmp/nikola_p98_") + tag + ".ipc";
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 1 — Compile-time constants
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase98 — compile-time: key length constants are correct",
          "[phase98][constants]")
{
    STATIC_CHECK(CURVE_KEY_Z85_CHARS  == 40);
    STATIC_CHECK(CURVE_KEY_BIN_BYTES  == 32);
    STATIC_CHECK(CURVE_KEY_Z85_BUFSIZE == 41);

    // Z85 encodes 4 binary bytes as 5 ASCII characters.
    // 32 binary bytes → 32 × (5/4) = 40 Z85 chars.
    STATIC_CHECK(CURVE_KEY_BIN_BYTES * 5 == CURVE_KEY_Z85_CHARS * 4);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 2 — KeyTier rotation period
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase98 — KeyTier: rotation_period_seconds() matches spec",
          "[phase98][tier]")
{
    // Tier 0 (Spine Broker) — manual / emergency only → period = 0
    CHECK(rotation_period_seconds(KeyTier::SPINE_BROKER)    ==       0u);

    // Tier 1 (Core) — ~30 days
    CHECK(rotation_period_seconds(KeyTier::CORE_COMPONENT)  == 2592000u);

    // Tier 2 (Ephemeral) — 24 hours
    CHECK(rotation_period_seconds(KeyTier::EPHEMERAL_AGENT) ==   86400u);

    // Broker rotation period is less than core (emergency only vs. scheduled)
    CHECK(rotation_period_seconds(KeyTier::SPINE_BROKER) <
          rotation_period_seconds(KeyTier::CORE_COMPONENT));
    CHECK(rotation_period_seconds(KeyTier::CORE_COMPONENT) >
          rotation_period_seconds(KeyTier::EPHEMERAL_AGENT));
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 3 — Key generation
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase98 — key generation: generate_ironhouse_keypair() properties",
          "[phase98][keygen]")
{
    SECTION("Does not throw") {
        REQUIRE_NOTHROW(generate_ironhouse_keypair());
    }

    SECTION("Returns valid Z85 keys") {
        const auto kp = generate_ironhouse_keypair();
        CHECK(IronhouseKeypair::is_valid_z85(kp.pub()));
        CHECK(IronhouseKeypair::is_valid_z85(kp.sec()));
    }

    SECTION("Each key is exactly 40 chars") {
        const auto kp = generate_ironhouse_keypair();
        CHECK(kp.pub().size() == CURVE_KEY_Z85_CHARS);
        CHECK(kp.sec().size() == CURVE_KEY_Z85_CHARS);
    }

    SECTION("Public key != private key") {
        const auto kp = generate_ironhouse_keypair();
        CHECK(kp.pub() != kp.sec());
    }

    SECTION("Two calls produce different keypairs") {
        const auto kp1 = generate_ironhouse_keypair();
        const auto kp2 = generate_ironhouse_keypair();
        // Vanishingly unlikely that two random Curve25519 keys are equal
        CHECK(kp1.pub() != kp2.pub());
        CHECK(kp1.sec() != kp2.sec());
    }

    SECTION("valid() returns true for generated keypair") {
        const auto kp = generate_ironhouse_keypair();
        CHECK(kp.valid());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 4 — IronhouseKeypair struct
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase98 — IronhouseKeypair: struct helpers",
          "[phase98][keypair]")
{
    SECTION("Default-constructed: valid() == false") {
        IronhouseKeypair kp;
        CHECK_FALSE(kp.valid());
        CHECK((kp.pub().empty() || kp.pub()[0] == '\0'));
    }

    SECTION("is_valid_z85: accepts the Z85 alphabet") {
        // Known valid Z85 string (40 chars — Z85 alphabet: 0-9, a-z, A-Z, .-:+=^!/*?&<>()[]{}@%$#)
        const std::string valid40(40, 'a');  // 'a' is a valid Z85 char
        CHECK( IronhouseKeypair::is_valid_z85(valid40));
        // Too short
        CHECK_FALSE(IronhouseKeypair::is_valid_z85("short"));
        // Contains illegal char (space)
        const std::string bad(40, ' ');
        CHECK_FALSE(IronhouseKeypair::is_valid_z85(bad));
        // Exactly 40 valid chars
        const std::string good(40, 'a');
        CHECK(IronhouseKeypair::is_valid_z85(good));
    }

    SECTION("pub() and sec() return correct string_views") {
        const auto kp = generate_ironhouse_keypair();
        // pub() points into public_key array
        CHECK(kp.pub() == std::string_view(kp.public_key.data(), 40));
        // sec() points into secret_key array
        CHECK(kp.sec() == std::string_view(kp.secret_key.data(), 40));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 5 — Socket option readback
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase98 — configure_curve_server: sets CURVE_SERVER option",
          "[phase98][socopts]")
{
    zmq::context_t ctx(1);
    auto server_kp = generate_ironhouse_keypair();

    SECTION("CURVE_SERVER flag is set to 1") {
        zmq::socket_t sock(ctx, zmq::socket_type::pull);
        configure_curve_server(sock, server_kp);
        CHECK(sock.get(zmq::sockopt::curve_server) == 1);
    }

    SECTION("CURVE_SECRETKEY reads back non-empty value") {
        zmq::socket_t sock(ctx, zmq::socket_type::pull);
        configure_curve_server(sock, server_kp);
        // ZMQ stores CurveZMQ keys internally as binary (32 bytes)
        const auto readback = sock.get(zmq::sockopt::curve_secretkey);
        CHECK(readback.size() == 40u);
    }
}

TEST_CASE("Phase98 — configure_curve_client: sets CURVE_SERVERKEY / PUBLICKEY / SECRETKEY",
          "[phase98][socopts]")
{
    zmq::context_t ctx(1);
    auto server_kp = generate_ironhouse_keypair();
    auto client_kp = generate_ironhouse_keypair();

    zmq::socket_t sock(ctx, zmq::socket_type::push);
    configure_curve_client(sock, client_kp, server_kp.pub());

    SECTION("CURVE_SERVERKEY reads back non-empty binary key") {
        // ZMQ stores CurveZMQ keys internally as binary (32 bytes)
        auto readback = sock.get(zmq::sockopt::curve_serverkey);
        CHECK(readback.size() == 40u);
    }

    SECTION("CURVE_PUBLICKEY reads back non-empty binary key") {
        auto readback = sock.get(zmq::sockopt::curve_publickey);
        CHECK(readback.size() == 40u);
    }

    SECTION("CURVE_SECRETKEY reads back non-empty binary key") {
        auto readback = sock.get(zmq::sockopt::curve_secretkey);
        CHECK(readback.size() == 40u);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 6 — Live CurveZMQ round-trip
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase98 — live CurveZMQ round-trip: 5 encrypted messages PUSH→PULL",
          "[phase98][live][curve]")
{
    const std::string endpoint = ep("roundtrip");

    auto server_kp = generate_ironhouse_keypair();
    auto client_kp = generate_ironhouse_keypair();

    zmq::context_t ctx(1);

    // PULL is the "server" (binds; accepts encrypted connections)
    zmq::socket_t puller(ctx, zmq::socket_type::pull);
    puller.set(zmq::sockopt::linger, 0);
    configure_curve_server(puller, server_kp);
    puller.bind(endpoint);

    // PUSH is the "client" (connects with correct server public key)
    zmq::socket_t pusher(ctx, zmq::socket_type::push);
    pusher.set(zmq::sockopt::linger, 0);
    configure_curve_client(pusher, client_kp, server_kp.pub());
    pusher.connect(endpoint);

    // Allow CurveZMQ handshake to complete
    std::this_thread::sleep_for(50ms);

    // Send 5 messages
    constexpr int N = 5;
    for (int i = 0; i < N; ++i) {
        const std::string msg = "secure_msg_" + std::to_string(i);
        pusher.send(zmq::buffer(msg), zmq::send_flags::none);
    }

    // Collect all 5 with a 2s deadline
    std::vector<std::string> received;
    received.reserve(N);

    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (static_cast<int>(received.size()) < N &&
           std::chrono::steady_clock::now() < deadline) {
        zmq::message_t msg;
        auto res = puller.recv(msg, zmq::recv_flags::dontwait);
        if (res.has_value()) {
            received.emplace_back(
                static_cast<const char*>(msg.data()), msg.size());
        } else {
            std::this_thread::sleep_for(2ms);
        }
    }

    REQUIRE(static_cast<int>(received.size()) == N);
    for (int i = 0; i < N; ++i) {
        CHECK(received[static_cast<std::size_t>(i)] ==
              "secure_msg_" + std::to_string(i));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 7 — Wrong server key: handshake fails, no messages received
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase98 — wrong server key: CurveZMQ handshake fails silently",
          "[phase98][live][security]")
{
    // The legitimate server PULL socket uses server_kp.
    // The "attacker" PUSH uses evil_kp.pub() as the server key — wrong.
    // Without a valid shared secret, ZMQ cannot complete the Diffie-Hellman
    // handshake, so no plaintext crosses the wire.

    const std::string endpoint = ep("wrongkey");

    auto server_kp = generate_ironhouse_keypair();
    auto evil_kp   = generate_ironhouse_keypair();  // different from server_kp
    auto evil_client_kp = generate_ironhouse_keypair();

    // Sanity: the evil key must not be the server key
    REQUIRE(evil_kp.pub() != server_kp.pub());

    zmq::context_t ctx(1);

    // Legitimate server PULL
    zmq::socket_t puller(ctx, zmq::socket_type::pull);
    puller.set(zmq::sockopt::linger, 0);
    configure_curve_server(puller, server_kp);
    puller.bind(endpoint);

    // Attacker PUSH: knows the endpoint address but NOT the true server pubkey
    zmq::socket_t pusher(ctx, zmq::socket_type::push);
    pusher.set(zmq::sockopt::linger, 0);
    configure_curve_client(pusher, evil_client_kp, evil_kp.pub()); // WRONG server key
    pusher.connect(endpoint);

    // Give "connection" time to fail
    std::this_thread::sleep_for(100ms);

    // Try to send — the message sits in the sender buffer; handshake never completes
    pusher.send(zmq::buffer(std::string("INTERCEPT")), zmq::send_flags::dontwait);

    // Poll PULL for 400ms — must receive NOTHING
    zmq_pollitem_t item{};
    item.socket = static_cast<void*>(puller);
    item.events = ZMQ_POLLIN;
    const int nready = zmq_poll(&item, 1, 400);

    INFO("zmq_poll returned " << nready << " (expected 0 — no messages)");
    CHECK(nready == 0);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 8 — ZapWhitelist
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase98 — ZapWhitelist: deny-by-default and authorization management",
          "[phase98][whitelist]")
{
    const auto kp1 = generate_ironhouse_keypair();
    const auto kp2 = generate_ironhouse_keypair();

    SECTION("Default-constructed: empty and denies all") {
        ZapWhitelist wl;
        CHECK(wl.empty());
        CHECK(wl.size() == 0);
        CHECK_FALSE(wl.is_authorized(kp1.pub()));
    }

    SECTION("add_key / is_authorized") {
        ZapWhitelist wl;
        wl.add_key(kp1.pub());
        CHECK(wl.is_authorized(kp1.pub()));
        CHECK_FALSE(wl.is_authorized(kp2.pub()));
        CHECK(wl.size() == 1);
    }

    SECTION("add_key is idempotent") {
        ZapWhitelist wl;
        wl.add_key(kp1.pub());
        wl.add_key(kp1.pub());
        CHECK(wl.size() == 1);
    }

    SECTION("remove_key revokes authorization") {
        ZapWhitelist wl;
        wl.add_key(kp1.pub());
        wl.add_key(kp2.pub());
        CHECK(wl.size() == 2);
        wl.remove_key(kp1.pub());
        CHECK_FALSE(wl.is_authorized(kp1.pub()));
        CHECK(wl.is_authorized(kp2.pub()));
        CHECK(wl.size() == 1);
    }

    SECTION("remove_key on absent key is a no-op") {
        ZapWhitelist wl;
        REQUIRE_NOTHROW(wl.remove_key(kp1.pub()));
        CHECK(wl.empty());
    }

    SECTION("clear() empties the whitelist (bootstrap mode)") {
        ZapWhitelist wl;
        wl.add_key(kp1.pub());
        wl.add_key(kp2.pub());
        wl.clear();
        CHECK(wl.empty());
        CHECK_FALSE(wl.is_authorized(kp1.pub()));
    }

    SECTION("empty() == true signals TOFU bootstrap mode") {
        ZapWhitelist wl;
        CHECK(wl.empty());   // bootstrap mode active
        wl.add_key(kp1.pub());
        CHECK_FALSE(wl.empty());  // LOCKED mode
        wl.clear();
        CHECK(wl.empty());   // bootstrap mode again
    }

    SECTION("add_key throws on invalid key length") {
        ZapWhitelist wl;
        CHECK_THROWS_AS(wl.add_key("tooshort"),   std::invalid_argument);
        CHECK_THROWS_AS(wl.add_key(std::string(41, 'a')), std::invalid_argument);
        CHECK_THROWS_AS(wl.add_key(""),            std::invalid_argument);
    }

    SECTION("multiple keys, all authorized") {
        ZapWhitelist wl;
        const auto kp3 = generate_ironhouse_keypair();
        wl.add_key(kp1.pub());
        wl.add_key(kp2.pub());
        wl.add_key(kp3.pub());
        CHECK(wl.size() == 3);
        CHECK(wl.is_authorized(kp1.pub()));
        CHECK(wl.is_authorized(kp2.pub()));
        CHECK(wl.is_authorized(kp3.pub()));
    }
}
