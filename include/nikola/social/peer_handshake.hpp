#pragma once
/**
 * @file peer_handshake.hpp
 * @brief Phase 141 — Peer handshake over CurveZMQ (Ironhouse)
 *
 * Bootstrap-node-list discovery model:
 *   1. Each peer has a known (or preconfigured) ZMQ endpoint.
 *   2. Initiator connects with CurveZMQ (Ironhouse) authentication.
 *   3. Both sides exchange PeerAnnouncement protobufs.
 *   4. On success, each adds the other to its PeerRegistry.
 *
 * Protocol (REQ/REP over CurveZMQ):
 *   Client → Server :  PeerAnnouncement (serialized protobuf)
 *   Server → Client :  PeerAnnouncement (serialized protobuf)
 *
 * See docs/architecture/peer_protocol.md for the full specification.
 */

#include <nikola/security/ironhouse.hpp>
#include <nikola/social/peer_registry.hpp>
#include <zmq.hpp>
#include <string>
#include <string_view>
#include <vector>

namespace nikola::social {

// ---------------------------------------------------------------------------
// HandshakeResult
// ---------------------------------------------------------------------------

struct HandshakeResult {
    bool        success = false;
    std::string peer_id;            ///< Remote peer's self-declared ID
    std::string public_key_z85;     ///< Remote peer's CurveZMQ public key
    std::string error;              ///< Non-empty on failure
};

// ---------------------------------------------------------------------------
// Serialization helpers  (PeerAnnouncement ↔ protobuf wire)
// ---------------------------------------------------------------------------

/// Serialize a PeerAnnouncement to protobuf wire format.
std::string serialize_announcement(
    const std::string& peer_id,
    const std::string& pub_key_z85,
    const std::string& endpoint,
    const std::vector<std::string>& capabilities = {});

/// Deserialize a PeerAnnouncement from protobuf wire format.
bool deserialize_announcement(
    const std::string& data,
    std::string& peer_id,
    std::string& pub_key_z85,
    std::string& endpoint,
    std::vector<std::string>& capabilities);

// ---------------------------------------------------------------------------
// Handshake functions
// ---------------------------------------------------------------------------

/**
 * @brief Initiate handshake (client / connector side).
 *
 * 1. Open REQ socket with CurveZMQ client config
 * 2. Connect to remote_endpoint
 * 3. Send own PeerAnnouncement
 * 4. Receive remote PeerAnnouncement
 * 5. Register remote in registry
 */
HandshakeResult initiate_handshake(
    zmq::context_t& ctx,
    const security::IronhouseKeypair& own_kp,
    std::string_view remote_pub_z85,
    const std::string& remote_endpoint,
    PeerRegistry& registry,
    const std::string& self_id,
    int timeout_ms = 5000);

/**
 * @brief Accept handshake (server / listener side).
 *
 * 1. Open REP socket with CurveZMQ server config
 * 2. Bind to bind_endpoint
 * 3. Receive remote PeerAnnouncement
 * 4. Send own PeerAnnouncement
 * 5. Register remote in registry
 */
HandshakeResult accept_handshake(
    zmq::context_t& ctx,
    const security::IronhouseKeypair& own_kp,
    const std::string& bind_endpoint,
    PeerRegistry& registry,
    const std::string& self_id,
    int timeout_ms = 5000);

} // namespace nikola::social
