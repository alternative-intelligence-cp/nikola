/**
 * @file include/nikola/infrastructure/spine.hpp
 * @brief ZeroMQ Communication Spine for the Nikola distributed architecture.
 *
 * Resolves Gap 4.4: ZeroMQ Socket Configuration.
 * Implements Gap 4.5: Protobuf Version / Topic Versioning.
 *
 * Spine responsibilities:
 *   1. Socket factory: apply canonical HWM / LINGER / IMMEDIATE settings to
 *      every socket before it is handed to the caller.
 *   2. Context lifecycle: single zmq::context_t per process, hidden in ZmqSpine.
 *   3. Topic helpers: versioned publish/subscribe for append-only schema evolution.
 *   4. Thin integration with CircuitBreaker and ComponentWatchdog.
 *
 * ZMQ socket configuration (Gap 4.4):
 *   SNDHWM = RCVHWM = 1000   — drop messages if queue overflows (real-time safe)
 *   LINGER = 0               — close immediately; discard unsent (fast shutdown)
 *   IMMEDIATE = 1            — only queue if a peer is connected (fail-fast)
 *
 * Topic versioning (Gap 4.5):
 *   Topic strings: "nikola.v<N>.<subsystem>"
 *   e.g. "nikola.v1.spikes", "nikola.v1.heartbeat"
 *   Breaking schema changes → increment N; old components attach to v<N-1> only.
 *
 * Design note: This header intentionally includes <zmq.hpp>.
 * It is an *internal* infrastructure header; external callers interact with
 * Orchestrator's PIMPL interface (orchestrator.hpp) which does not expose ZMQ.
 */

#pragma once

#include <zmq.hpp>
#include <string>
#include <stdexcept>
#include <string_view>

namespace nikola::infrastructure {

// ---------------------------------------------------------------------------
// Gap 4.4 — Socket option values (prefixed NIKOLA_ to avoid zmq.h macro conflicts)
// ---------------------------------------------------------------------------

inline constexpr int NIKOLA_SOCKET_HWM       = 1000;  ///< High-water mark (send + recv)
inline constexpr int NIKOLA_SOCKET_LINGER    = 0;     ///< Discard unsent on close
inline constexpr int NIKOLA_SOCKET_IMMEDIATE = 1;     ///< Only queue if peer connected

// ---------------------------------------------------------------------------
// Gap 4.5 — Topic versioning
// ---------------------------------------------------------------------------

inline constexpr std::string_view NIKOLA_TOPIC_PREFIX = "nikola";
inline constexpr int              NIKOLA_PROTO_VERSION = 1;  ///< Current schema version

/**
 * @brief Build a versioned ZMQ topic string.
 * @param subsystem  E.g. "spikes", "heartbeat", "waveform"
 * @param version    Schema version; defaults to current.
 * @return           E.g. "nikola.v1.spikes"
 */
[[nodiscard]] inline std::string make_topic(
    std::string_view subsystem,
    int version = NIKOLA_PROTO_VERSION
) {
    return std::string(NIKOLA_TOPIC_PREFIX) + ".v" + std::to_string(version) + "." + std::string(subsystem);
}

/**
 * @brief Returns the subscription prefix for a given version.
 * Subscribers attach to "nikola.v1" to receive all v1 messages.
 */
[[nodiscard]] inline std::string topic_version_prefix(int version = NIKOLA_PROTO_VERSION) {
    return std::string(NIKOLA_TOPIC_PREFIX) + ".v" + std::to_string(version);
}

// ---------------------------------------------------------------------------
// configure_socket()  (Gap 4.4)
// ---------------------------------------------------------------------------

/**
 * @brief Apply the canonical Nikola socket options to any ZMQ socket.
 * Must be called *before* bind()/connect().
 */
inline void configure_socket(zmq::socket_t& sock) {
    // High-water mark: drop if queue overflows (real-time safe)
    sock.set(zmq::sockopt::sndhwm, NIKOLA_SOCKET_HWM);
    sock.set(zmq::sockopt::rcvhwm, NIKOLA_SOCKET_HWM);

    // Fast shutdown — discard pending messages
    sock.set(zmq::sockopt::linger, NIKOLA_SOCKET_LINGER);

    // Only queue messages if a peer is reachable
    sock.set(zmq::sockopt::immediate, NIKOLA_SOCKET_IMMEDIATE);
}

// ---------------------------------------------------------------------------
// ZmqSpine — context + socket factory
// ---------------------------------------------------------------------------

/**
 * @class ZmqSpine
 * @brief Manages the single zmq::context_t and produces pre-configured sockets.
 *
 * Ownership model:
 *   ZmqSpine holds the context; sockets are returned by value (movable).
 *   Destroying ZmqSpine terminates all sockets.
 *
 * Usage:
 *   ZmqSpine spine;
 *   auto pub = spine.make_publisher("tcp://*:5555");
 *   auto sub = spine.make_subscriber("tcp://localhost:5555", "nikola.v1");
 */
class ZmqSpine {
public:
    explicit ZmqSpine(int io_threads = 1)
        : ctx_(io_threads)
    {}

    ZmqSpine(const ZmqSpine&)            = delete;
    ZmqSpine& operator=(const ZmqSpine&) = delete;
    ZmqSpine(ZmqSpine&&)                 = default;
    ZmqSpine& operator=(ZmqSpine&&)      = default;

    // -----------------------------------------------------------------------
    // Publisher (PUB)
    // -----------------------------------------------------------------------

    /**
     * @brief Create a PUB socket bound to `endpoint`.
     * @param endpoint  ZMQ endpoint, e.g. "tcp://*:5555"
     */
    [[nodiscard]] zmq::socket_t make_publisher(const std::string& endpoint) {
        zmq::socket_t sock(ctx_, zmq::socket_type::pub);
        configure_socket(sock);
        sock.bind(endpoint);
        return sock;
    }

    // -----------------------------------------------------------------------
    // Subscriber (SUB)
    // -----------------------------------------------------------------------

    /**
     * @brief Create a SUB socket connected to `endpoint`, filtered by `topic_prefix`.
     * @param endpoint      E.g. "tcp://localhost:5555"
     * @param topic_prefix  E.g. "nikola.v1"  (empty string = subscribe all)
     */
    [[nodiscard]] zmq::socket_t make_subscriber(const std::string& endpoint,
                                                 const std::string& topic_prefix = "") {
        zmq::socket_t sock(ctx_, zmq::socket_type::sub);
        configure_socket(sock);
        sock.connect(endpoint);
        sock.set(zmq::sockopt::subscribe, topic_prefix);
        return sock;
    }

    // -----------------------------------------------------------------------
    // Push / Pull (pipeline)
    // -----------------------------------------------------------------------

    [[nodiscard]] zmq::socket_t make_push(const std::string& endpoint) {
        zmq::socket_t sock(ctx_, zmq::socket_type::push);
        configure_socket(sock);
        sock.connect(endpoint);
        return sock;
    }

    [[nodiscard]] zmq::socket_t make_pull(const std::string& endpoint) {
        zmq::socket_t sock(ctx_, zmq::socket_type::pull);
        configure_socket(sock);
        sock.bind(endpoint);
        return sock;
    }

    // -----------------------------------------------------------------------
    // REQ / REP (request–reply)
    // -----------------------------------------------------------------------

    [[nodiscard]] zmq::socket_t make_req(const std::string& endpoint) {
        zmq::socket_t sock(ctx_, zmq::socket_type::req);
        configure_socket(sock);
        sock.connect(endpoint);
        return sock;
    }

    [[nodiscard]] zmq::socket_t make_rep(const std::string& endpoint) {
        zmq::socket_t sock(ctx_, zmq::socket_type::rep);
        configure_socket(sock);
        sock.bind(endpoint);
        return sock;
    }

    // -----------------------------------------------------------------------
    // Versioned publish / subscribe
    // -----------------------------------------------------------------------

    /**
     * @brief Publish a multi-part message with a versioned topic frame.
     * @param sock       PUB socket
     * @param subsystem  E.g. "spikes"
     * @param payload    Raw bytes
     * @param len        Payload byte count
     */
    static void publish(zmq::socket_t& sock,
                        std::string_view subsystem,
                        const void* payload, std::size_t len,
                        int version = NIKOLA_PROTO_VERSION)
    {
        std::string topic = make_topic(subsystem, version);
        zmq::message_t topic_msg(topic.data(), topic.size());
        zmq::message_t data_msg(payload, len);

        sock.send(topic_msg, zmq::send_flags::sndmore);
        sock.send(data_msg,  zmq::send_flags::none);
    }

    // -----------------------------------------------------------------------
    // Raw context access (for tests / advanced usage)
    // -----------------------------------------------------------------------

    [[nodiscard]] zmq::context_t& context() noexcept { return ctx_; }

private:
    zmq::context_t ctx_;
};

} // namespace nikola::infrastructure
