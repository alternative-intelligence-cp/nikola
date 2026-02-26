/**
 * @file include/nikola/security/ironhouse.hpp
 * @brief GAP-010: CurveZMQ Ironhouse cryptographic identity management.
 *
 * Spec: §1.3 "The Ironhouse Security Model" and §GAP-010 RESOLUTION in
 *       docs/info/integration/sections/04_infrastructure/01_zeromq_spine.md
 *
 * Ironhouse is a ZeroMQ security pattern where **every connection is mutually
 * authenticated and encrypted** using Curve25519 (CurveZMQ).  Nikola extends
 * the base pattern with:
 *
 *   - Tiered key hierarchy (Tier 0 Broker ↔ Tier 1 Core ↔ Tier 2 Ephemeral)
 *   - In-memory ZAP whitelist (deny-by-default)
 *   - Key generation via libsodium-backed `zmq_curve_keypair()`
 *   - RAII socket configuration helpers
 *
 * All keys are stored in Z85 ASCII encoding (40 chars + NUL) as required by
 * the ZMQ `ZMQ_CURVE_*` socket options.
 *
 * Usage — server side:
 * @code
 *   using namespace nikola::security;
 *   auto server_kp = generate_ironhouse_keypair();
 *   zmq::socket_t server(ctx, zmq::socket_type::push);
 *   configure_curve_server(server, server_kp);
 *   server.bind("tcp://0.0.0.0:5566");
 * @endcode
 *
 * Usage — client side:
 * @code
 *   auto client_kp = generate_ironhouse_keypair();
 *   zmq::socket_t client(ctx, zmq::socket_type::pull);
 *   configure_curve_client(client, client_kp, server_kp.public_key);
 *   client.connect("tcp://localhost:5566");
 * @endcode
 */

#pragma once

#include <zmq.hpp>       // zmq_curve_keypair, zmq::sockopt::curve_*

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>

namespace nikola::security {

// ─────────────────────────────────────────────────────────────────────────────
//  Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Z85-encoded key length (chars, NOT including null terminator).
inline constexpr std::size_t CURVE_KEY_Z85_CHARS = 40;

/// Binary key length in bytes (Curve25519 public or private key).
inline constexpr std::size_t CURVE_KEY_BIN_BYTES = 32;

/// Full Z85 string length including NUL terminator.
inline constexpr std::size_t CURVE_KEY_Z85_BUFSIZE = CURVE_KEY_Z85_CHARS + 1;

// ─────────────────────────────────────────────────────────────────────────────
//  Key tier enum (from §GAP-010 Tiered Key Hierarchy)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Nikola key persistence tier.
 *
 * | Tier | Component          | Rotation             | Persistence |
 * |------|--------------------|----------------------|-------------|
 * |  0   | Spine Broker       | Emergency only       | Maximum     |
 * |  1   | Core components    | Monthly / 2.5e9 ticks | High        |
 * |  2   | Ephemeral agents   | Per-session / daily  | Low (TOFU)  |
 */
enum class KeyTier : uint8_t {
    SPINE_BROKER     = 0,  ///< Root of trust; MitM risk if compromised
    CORE_COMPONENT   = 1,  ///< Physics Engine, Memory, Orchestrator
    EPHEMERAL_AGENT  = 2,  ///< Tools, CLI; Trust-On-First-Use
};

/**
 * @brief Returns the recommended rotation period in seconds for a key tier.
 */
[[nodiscard]] constexpr std::uint32_t rotation_period_seconds(KeyTier tier) noexcept {
    switch (tier) {
    case KeyTier::SPINE_BROKER:    return 0;        // manual / emergency only
    case KeyTier::CORE_COMPONENT:  return 2592000;  // ~30 days
    case KeyTier::EPHEMERAL_AGENT: return 86400;    // 24 hours
    }
    return 86400;
}

// ─────────────────────────────────────────────────────────────────────────────
//  IronhouseKeypair
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief A Curve25519 keypair in Z85 ASCII format (as required by CurveZMQ).
 *
 * Both keys are exactly CURVE_KEY_Z85_CHARS (40) printable characters.
 * The struct guarantees zero-padding of the underlying storage so the C-string
 * pointers are always NUL-terminated.
 */
struct IronhouseKeypair {
    /// Z85-encoded public key (40 chars).
    std::array<char, CURVE_KEY_Z85_BUFSIZE> public_key{};
    /// Z85-encoded private/secret key (40 chars).
    std::array<char, CURVE_KEY_Z85_BUFSIZE> secret_key{};

    /// Convenience: public key as std::string_view.
    [[nodiscard]] std::string_view pub() const noexcept {
        return {public_key.data(), CURVE_KEY_Z85_CHARS};
    }

    /// Convenience: secret key as std::string_view.
    [[nodiscard]] std::string_view sec() const noexcept {
        return {secret_key.data(), CURVE_KEY_Z85_CHARS};
    }

    /// True if both keys are non-empty (not default-constructed).
    [[nodiscard]] bool valid() const noexcept {
        return public_key[0] != '\0' && secret_key[0] != '\0';
    }

    /// True if the input is a valid 40-char Z85 string.
    [[nodiscard]] static bool is_valid_z85(std::string_view s) noexcept {
        if (s.size() != CURVE_KEY_Z85_CHARS) return false;
        // Z85 alphabet: 0–9, a–z, A–Z, .-:+=^!/*?&<>()[]{}@%$#
        static constexpr std::string_view Z85_ALPHABET =
            "0123456789abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            ".-:+=^!/*?&<>()[]{}@%$#";
        for (char c : s) {
            if (Z85_ALPHABET.find(c) == std::string_view::npos) return false;
        }
        return true;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Key generation
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Generate a fresh Curve25519 keypair using ZMQ's built-in CSPRNG.
 *
 * Internally calls `zmq_curve_keypair()` which uses libsodium's
 * `randombytes_buf()` (seeded from `/dev/urandom`).  Keys are returned in Z85
 * encoding as required by `ZMQ_CURVE_PUBLICKEY` / `ZMQ_CURVE_SECRETKEY`.
 *
 * @throws std::runtime_error if key generation fails.
 */
[[nodiscard]] inline IronhouseKeypair generate_ironhouse_keypair() {
    IronhouseKeypair kp;
    const int rc = ::zmq_curve_keypair(kp.public_key.data(), kp.secret_key.data());
    if (rc != 0) {
        throw std::runtime_error(
            "generate_ironhouse_keypair: zmq_curve_keypair() failed "
            "(libsodium not compiled in?)");
    }
    return kp;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Socket configuration helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Apply CurveZMQ server-side options to a socket.
 *
 * Must be called BEFORE bind().
 *
 * Sets:
 *   - ZMQ_CURVE_SERVER = 1   (enable server role)
 *   - ZMQ_CURVE_SECRETKEY    (server's private key)
 *
 * The server's public key must be distributed out-of-band to clients
 * (e.g., file system, bootstrap protocol).
 */
inline void configure_curve_server(zmq::socket_t& sock,
                                    const IronhouseKeypair& server_kp)
{
    sock.set(zmq::sockopt::curve_server, 1);
    sock.set(zmq::sockopt::curve_secretkey, std::string(server_kp.sec()));
}

/**
 * @brief Apply CurveZMQ client-side options to a socket.
 *
 * Must be called BEFORE connect().
 *
 * Sets:
 *   - ZMQ_CURVE_SERVERKEY    (server's public key — obtained out-of-band)
 *   - ZMQ_CURVE_PUBLICKEY    (client's own public key)
 *   - ZMQ_CURVE_SECRETKEY    (client's own private key)
 *
 * @param client_kp        The client's own freshly generated keypair.
 * @param server_public    The Z85 public key of the server to connect to.
 */
inline void configure_curve_client(zmq::socket_t& sock,
                                    const IronhouseKeypair& client_kp,
                                    std::string_view server_public)
{
    sock.set(zmq::sockopt::curve_serverkey, std::string(server_public));
    sock.set(zmq::sockopt::curve_publickey, std::string(client_kp.pub()));
    sock.set(zmq::sockopt::curve_secretkey, std::string(client_kp.sec()));
}

// ─────────────────────────────────────────────────────────────────────────────
//  ZapWhitelist — in-memory authorized-key store
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Thread-unsafe in-memory set of authorized Curve25519 public keys.
 *
 * This is the data structure behind the ZAP handler.  In production, keys are
 * persisted to a permission-locked `whitelist.txt`; here we provide the
 * in-memory equivalent for testing and single-process deployments.
 *
 * Design rules (from §1.3 spec):
 *   - Deny-by-default: `is_authorized()` returns false for any unknown key.
 *   - Keys are stored normalised as 40-char Z85 strings.
 *   - Bootstrap state: when empty, TOFU mode is active (handled externally).
 */
class ZapWhitelist {
public:
    /// Add a public key to the whitelist.  No-op if already present.
    void add_key(std::string_view z85_public) {
        if (z85_public.size() != CURVE_KEY_Z85_CHARS) {
            throw std::invalid_argument(
                "ZapWhitelist::add_key: key must be 40 Z85 chars");
        }
        keys_.emplace(z85_public);
    }

    /// Remove a public key from the whitelist.  No-op if absent.
    void remove_key(std::string_view z85_public) {
        keys_.erase(std::string(z85_public));
    }

    /// Return true if `z85_public` is in the whitelist (deny-by-default).
    [[nodiscard]] bool is_authorized(std::string_view z85_public) const noexcept {
        return keys_.count(std::string(z85_public)) > 0;
    }

    /// Return the number of authorized keys.
    [[nodiscard]] std::size_t size() const noexcept { return keys_.size(); }

    /// Return true when the whitelist is empty (TOFU bootstrap mode).
    [[nodiscard]] bool empty() const noexcept { return keys_.empty(); }

    /// Remove all keys (triggers TOFU bootstrap mode).
    void clear() noexcept { keys_.clear(); }

private:
    std::unordered_set<std::string> keys_;
};

} // namespace nikola::security
