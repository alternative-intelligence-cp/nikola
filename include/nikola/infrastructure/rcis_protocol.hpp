#pragma once
/**
 * @file rcis_protocol.hpp
 * @brief v0.3.5 — RCIS protobuf + ZMQ transport helpers.
 */

#include <nikola/security/ironhouse.hpp>

#include <rcis.pb.h>
#include <zmq.hpp>

#include <chrono>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace nikola::infrastructure::rcis {

inline constexpr std::string_view RCIS_TOPIC_REQUEST  = "rcis.request";
inline constexpr std::string_view RCIS_TOPIC_RESPONSE = "rcis.response";

[[nodiscard]] inline uint64_t now_ns() {
    using namespace std::chrono;
    return static_cast<uint64_t>(duration_cast<nanoseconds>(
        system_clock::now().time_since_epoch()).count());
}

[[nodiscard]] inline bool validate_neural_spike(const ::nikola::spine::NeuralSpike& spike) noexcept {
    return !spike.request_id().empty() && spike.timestamp_ns() > 0;
}

[[nodiscard]] inline bool validate_request(const ::nikola::spine::RCISRequest& req) noexcept {
    if (req.request_id().empty() || req.timestamp_ns() == 0) return false;
    if (req.type() == ::nikola::spine::RCISRequest::REQUEST_UNSPECIFIED) return false;

    if (req.type() == ::nikola::spine::RCISRequest::INJECT_STIMULUS) {
        return !req.stimulus_text().empty();
    }
    if (req.type() == ::nikola::spine::RCISRequest::FORWARD_SPIKE) {
        return req.has_spike() && validate_neural_spike(req.spike());
    }
    return true;
}

[[nodiscard]] inline bool validate_response(const ::nikola::spine::RCISResponse& resp) noexcept {
    if (resp.request_id().empty() || resp.timestamp_ns() == 0) return false;
    if (resp.status() == ::nikola::spine::RCISResponse::STATUS_UNSPECIFIED) return false;
    if (resp.has_spike()) return validate_neural_spike(resp.spike());
    return true;
}

[[nodiscard]] inline ::nikola::spine::RCISRequest make_ping_request(std::string_view request_id) {
    ::nikola::spine::RCISRequest req;
    req.set_request_id(std::string(request_id));
    req.set_timestamp_ns(now_ns());
    req.set_type(::nikola::spine::RCISRequest::PING);
    return req;
}

[[nodiscard]] inline ::nikola::spine::RCISResponse make_ok_response(std::string_view request_id,
                                                                    std::string_view message = "ok") {
    ::nikola::spine::RCISResponse resp;
    resp.set_request_id(std::string(request_id));
    resp.set_timestamp_ns(now_ns());
    resp.set_status(::nikola::spine::RCISResponse::OK);
    resp.set_message(std::string(message));
    return resp;
}

inline std::string serialize(const google::protobuf::MessageLite& msg) {
    std::string out;
    if (!msg.SerializeToString(&out)) {
        throw std::runtime_error("RCIS serialize failed");
    }
    return out;
}

template <typename ProtoT>
inline bool deserialize(const std::string& wire, ProtoT& out) {
    return out.ParseFromString(wire);
}

template <typename ProtoT>
inline void send_proto(zmq::socket_t& sock, const ProtoT& msg, zmq::send_flags flags = zmq::send_flags::none) {
    const std::string wire = serialize(msg);
    zmq::message_t out(wire.data(), wire.size());
    auto ok = sock.send(out, flags);
    if (!ok.has_value()) {
        throw std::runtime_error("RCIS send_proto failed");
    }
}

template <typename ProtoT>
inline bool recv_proto(zmq::socket_t& sock, ProtoT& out,
                       zmq::recv_flags flags = zmq::recv_flags::none) {
    zmq::message_t in;
    auto ok = sock.recv(in, flags);
    if (!ok.has_value()) return false;

    const std::string wire(static_cast<const char*>(in.data()), in.size());
    return deserialize(wire, out);
}

inline void configure_curve_server(zmq::socket_t& sock,
                                   const nikola::security::IronhouseKeypair& kp) {
    nikola::security::configure_curve_server(sock, kp);
}

inline void configure_curve_client(zmq::socket_t& sock,
                                   const nikola::security::IronhouseKeypair& client,
                                   std::string_view server_public_key) {
    if (server_public_key.empty()) {
        throw std::invalid_argument("configure_curve_client: server_public_key must not be empty");
    }
    nikola::security::configure_curve_client(sock, client, server_public_key);
}

} // namespace nikola::infrastructure::rcis
