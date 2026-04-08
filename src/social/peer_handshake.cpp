// ============================================================
// Phase 141 — Peer Handshake over CurveZMQ
// src/social/peer_handshake.cpp
//
// Implements the bootstrap-node-list handshake protocol using
// Ironhouse (CurveZMQ) for authenticated encryption and
// PeerAnnouncement protobuf for announcement serialization.
// ============================================================

#include <nikola/social/peer_handshake.hpp>
#include <irsp.pb.h>
#include <chrono>

namespace nikola::social {

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

std::string serialize_announcement(
    const std::string& peer_id,
    const std::string& pub_key_z85,
    const std::string& endpoint,
    const std::vector<std::string>& capabilities)
{
    ::nikola::social::PeerAnnouncement msg;
    msg.set_peer_id(peer_id);
    msg.set_public_key_z85(pub_key_z85);
    msg.set_zmq_endpoint(endpoint);
    for (const auto& cap : capabilities)
        msg.add_capabilities(cap);
    msg.set_timestamp(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    return msg.SerializeAsString();
}

bool deserialize_announcement(
    const std::string& data,
    std::string& peer_id,
    std::string& pub_key_z85,
    std::string& endpoint,
    std::vector<std::string>& capabilities)
{
    ::nikola::social::PeerAnnouncement msg;
    if (!msg.ParseFromString(data))
        return false;
    peer_id       = msg.peer_id();
    pub_key_z85   = msg.public_key_z85();
    endpoint      = msg.zmq_endpoint();
    capabilities.clear();
    for (int i = 0; i < msg.capabilities_size(); ++i)
        capabilities.push_back(msg.capabilities(i));
    return true;
}

// ---------------------------------------------------------------------------
// Initiator (client / REQ)
// ---------------------------------------------------------------------------

HandshakeResult initiate_handshake(
    zmq::context_t& ctx,
    const security::IronhouseKeypair& own_kp,
    std::string_view remote_pub_z85,
    const std::string& remote_endpoint,
    PeerRegistry& registry,
    const std::string& self_id,
    int timeout_ms)
{
    HandshakeResult result;
    try {
        zmq::socket_t sock(ctx, zmq::socket_type::req);
        sock.set(zmq::sockopt::rcvtimeo, timeout_ms);
        sock.set(zmq::sockopt::sndtimeo, timeout_ms);
        sock.set(zmq::sockopt::linger, 0);

        security::configure_curve_client(sock, own_kp, remote_pub_z85);
        sock.connect(remote_endpoint);

        // Send own announcement
        auto payload = serialize_announcement(
            self_id, std::string(own_kp.pub()), remote_endpoint);
        zmq::message_t out(payload.data(), payload.size());
        if (!sock.send(out, zmq::send_flags::none)) {
            result.error = "send failed";
            return result;
        }

        // Receive remote announcement
        zmq::message_t in;
        if (!sock.recv(in, zmq::recv_flags::none)) {
            result.error = "recv timeout";
            return result;
        }

        std::string peer_id, pub_key, ep;
        std::vector<std::string> caps;
        std::string wire(static_cast<const char*>(in.data()), in.size());
        if (!deserialize_announcement(wire, peer_id, pub_key, ep, caps)) {
            result.error = "bad announcement";
            return result;
        }

        registry.add_peer(peer_id, pub_key, 0);
        result.success        = true;
        result.peer_id        = peer_id;
        result.public_key_z85 = pub_key;
    } catch (const zmq::error_t& e) {
        result.error = e.what();
    }
    return result;
}

// ---------------------------------------------------------------------------
// Responder (server / REP)
// ---------------------------------------------------------------------------

HandshakeResult accept_handshake(
    zmq::context_t& ctx,
    const security::IronhouseKeypair& own_kp,
    const std::string& bind_endpoint,
    PeerRegistry& registry,
    const std::string& self_id,
    int timeout_ms)
{
    HandshakeResult result;
    try {
        zmq::socket_t sock(ctx, zmq::socket_type::rep);
        sock.set(zmq::sockopt::rcvtimeo, timeout_ms);
        sock.set(zmq::sockopt::sndtimeo, timeout_ms);
        sock.set(zmq::sockopt::linger, 0);

        security::configure_curve_server(sock, own_kp);
        sock.bind(bind_endpoint);

        // Receive remote announcement
        zmq::message_t in;
        if (!sock.recv(in, zmq::recv_flags::none)) {
            result.error = "recv timeout";
            return result;
        }

        std::string peer_id, pub_key, ep;
        std::vector<std::string> caps;
        std::string wire(static_cast<const char*>(in.data()), in.size());
        if (!deserialize_announcement(wire, peer_id, pub_key, ep, caps)) {
            result.error = "bad announcement";
            return result;
        }

        // Send own announcement back
        auto payload = serialize_announcement(
            self_id, std::string(own_kp.pub()), bind_endpoint);
        zmq::message_t out(payload.data(), payload.size());
        if (!sock.send(out, zmq::send_flags::none)) {
            result.error = "send failed";
            return result;
        }

        registry.add_peer(peer_id, pub_key, 0);
        result.success        = true;
        result.peer_id        = peer_id;
        result.public_key_z85 = pub_key;
    } catch (const zmq::error_t& e) {
        result.error = e.what();
    }
    return result;
}

} // namespace nikola::social
