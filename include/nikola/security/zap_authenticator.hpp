/**
 * @file include/nikola/security/zap_authenticator.hpp
 * @brief ZAP (ZeroMQ Authentication Protocol) handler for CurveZMQ Ironhouse.
 *
 * Implements a background thread that processes ZAP requests on the
 * well-known `inproc://zeromq.zap.01` endpoint.  When a CurveZMQ connection
 * is established, the ZMQ library sends a ZAP request to this handler.
 * The handler checks the client's public key against the ZapWhitelist
 * (from ironhouse.hpp) and replies with 200 (allow) or 400 (deny).
 *
 * ZAP protocol reference:
 *   https://rfc.zeromq.org/spec/27/
 *
 * Frame layout — request (5+ frames):
 *   [0] version   "1.0"
 *   [1] request_id
 *   [2] domain
 *   [3] address
 *   [4] identity
 *   [5] mechanism "CURVE"
 *   [6] credentials (32-byte binary public key for CURVE)
 *
 * Frame layout — reply (5 frames):
 *   [0] version   "1.0"
 *   [1] request_id  (echo back)
 *   [2] status_code "200" or "400"
 *   [3] status_text "OK" or "NOT AUTHORIZED"
 *   [4] user_id     (arbitrary, used for logging)
 *   [5] metadata    (empty)
 *
 * Usage:
 * @code
 *   zmq::context_t ctx(1);
 *   ZapWhitelist whitelist;
 *   whitelist.add_key(server_kp.pub());
 *
 *   ZapAuthenticator auth(ctx, whitelist);
 *   auth.start();  // spawns background thread
 *   // ... create CurveZMQ sockets as normal ...
 *   auth.stop();   // joins thread
 * @endcode
 */

#pragma once

#include <nikola/security/ironhouse.hpp>

#include <zmq.hpp>
#include <atomic>
#include <string>
#include <thread>

namespace nikola::security {

class ZapAuthenticator {
public:
    /**
     * @brief Construct a ZAP authenticator.
     * @param ctx        ZMQ context (must outlive this object).
     * @param whitelist  Reference to the authorized-key store.
     */
    ZapAuthenticator(zmq::context_t& ctx, const ZapWhitelist& whitelist)
        : ctx_(ctx)
        , whitelist_(whitelist)
    {}

    ~ZapAuthenticator() { stop(); }

    ZapAuthenticator(const ZapAuthenticator&)            = delete;
    ZapAuthenticator& operator=(const ZapAuthenticator&) = delete;

    /// Start the authenticator thread.  Idempotent.
    void start() {
        if (running_.load(std::memory_order_acquire)) return;
        running_.store(true, std::memory_order_release);
        thread_ = std::thread([this] { run(); });
    }

    /// Stop the authenticator thread and join.  Idempotent.
    void stop() {
        if (!running_.load(std::memory_order_acquire)) return;
        running_.store(false, std::memory_order_release);
        if (thread_.joinable()) thread_.join();
    }

    [[nodiscard]] bool is_running() const noexcept {
        return running_.load(std::memory_order_acquire);
    }

    /// Number of ZAP requests processed since start().
    [[nodiscard]] uint64_t requests_processed() const noexcept {
        return requests_processed_.load(std::memory_order_relaxed);
    }

    /// Number of denied requests.
    [[nodiscard]] uint64_t requests_denied() const noexcept {
        return requests_denied_.load(std::memory_order_relaxed);
    }

private:
    void run() {
        zmq::socket_t zap(ctx_, zmq::socket_type::rep);
        zap.bind("inproc://zeromq.zap.01");

        // Use a poll timeout so we can check the running_ flag
        zmq::pollitem_t items[] = {{zap.handle(), 0, ZMQ_POLLIN, 0}};

        while (running_.load(std::memory_order_acquire)) {
            zmq::poll(items, 1, std::chrono::milliseconds(100));

            if (!(items[0].revents & ZMQ_POLLIN)) continue;

            // Read all frames of the ZAP request
            std::vector<zmq::message_t> frames;
            bool more = true;
            while (more) {
                zmq::message_t frame;
                auto rc = zap.recv(frame, zmq::recv_flags::none);
                if (!rc) break;
                more = frame.more();
                frames.push_back(std::move(frame));
            }

            // ZAP request must have at least 7 frames for CURVE
            if (frames.size() < 7) continue;

            // Extract fields
            std::string version(static_cast<const char*>(frames[0].data()), frames[0].size());
            std::string request_id(static_cast<const char*>(frames[1].data()), frames[1].size());
            std::string mechanism(static_cast<const char*>(frames[5].data()), frames[5].size());

            // For CURVE, frame[6] is the 32-byte binary public key
            std::string status_code = "400";
            std::string status_text = "NOT AUTHORIZED";
            std::string user_id;

            if (version == "1.0" && mechanism == "CURVE" && frames[6].size() == CURVE_KEY_BIN_BYTES) {
                // Convert binary key to Z85 for whitelist lookup
                char z85_buf[CURVE_KEY_Z85_BUFSIZE];
                const char* z85 = zmq_z85_encode(z85_buf, frames[6].data<uint8_t>(), CURVE_KEY_BIN_BYTES);
                if (z85) {
                    std::string_view z85_key(z85, CURVE_KEY_Z85_CHARS);
                    if (whitelist_.is_authorized(z85_key)) {
                        status_code = "200";
                        status_text = "OK";
                        user_id = std::string(z85_key);
                    }
                }
            }

            if (status_code != "200") {
                requests_denied_.fetch_add(1, std::memory_order_relaxed);
            }
            requests_processed_.fetch_add(1, std::memory_order_relaxed);

            // Send ZAP reply (6 frames)
            auto send_str = [&](const std::string& s, zmq::send_flags flags) {
                zmq::message_t msg(s.data(), s.size());
                zap.send(msg, flags);
            };

            send_str(version,     zmq::send_flags::sndmore);
            send_str(request_id,  zmq::send_flags::sndmore);
            send_str(status_code, zmq::send_flags::sndmore);
            send_str(status_text, zmq::send_flags::sndmore);
            send_str(user_id,     zmq::send_flags::sndmore);
            send_str("",          zmq::send_flags::none);    // metadata (empty)
        }
    }

    zmq::context_t&     ctx_;
    const ZapWhitelist&  whitelist_;
    std::atomic<bool>    running_{false};
    std::thread          thread_;
    std::atomic<uint64_t> requests_processed_{0};
    std::atomic<uint64_t> requests_denied_{0};
};

} // namespace nikola::security
