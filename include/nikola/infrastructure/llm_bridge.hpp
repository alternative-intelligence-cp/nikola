/**
 * @file include/nikola/infrastructure/llm_bridge.hpp
 * @brief LLM Bridge: ZMQ-backed request/response interface to external language models.
 *
 * Provides a clean, dependency-free public API for sending prompts to an
 * external LLM service (e.g. Gemini, GPT-4, local llama.cpp) and receiving
 * structured responses.
 *
 * Transport: ZMQ REQ-REP pair (LLMBridge=REQ, remote service=REP).
 * Serialization: JSON payload (plain string; caller serialises as needed).
 * Timeout: controlled by CircuitBreaker with CONTROL priority (100 ms default).
 *
 * Design: Public header does NOT include zmq.hpp (IMP-04 ABI firewall).
 * The Impl is defined in the NIKOLA_LLM_BRIDGE_IMPL block below for the
 * header-only testing pattern used across Phase 0-4.
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace nikola::infrastructure {

// ---------------------------------------------------------------------------
// LLMRequest / LLMResponse
// ---------------------------------------------------------------------------

/// A single prompt sent to the external language model.
struct LLMRequest {
    std::string  request_id;        ///< Unique ID (caller fills in; e.g. UUID string)
    std::string  prompt;            ///< The prompt text
    float        temperature = 0.7f; ///< Sampling temperature
    int          max_tokens  = 256;  ///< Max tokens to generate
    int          proto_version = 1;  ///< Gap 4.5 — schema version

    /// Serialise to JSON string (minimal implementation for testing).
    [[nodiscard]] std::string to_json() const {
        return "{\"id\":\"" + request_id +
               "\",\"prompt\":\"" + prompt +
               "\",\"temperature\":" + std::to_string(temperature) +
               ",\"max_tokens\":" + std::to_string(max_tokens) +
               ",\"v\":" + std::to_string(proto_version) + "}";
    }
};

/// Response from the external language model.
struct LLMResponse {
    std::string  request_id;
    std::string  text;           ///< Generated text
    int          token_count = 0;
    bool         ok          = false; ///< false = timeout or error
    std::string  error_msg;

    /// Parse minimal JSON from external service.
    [[nodiscard]] static LLMResponse from_json(const std::string& json) {
        LLMResponse r;
        r.ok = !json.empty();
        r.text = json; // trivial — real impl would parse fields
        return r;
    }
};

// ---------------------------------------------------------------------------
// LLMBridgeConfig
// ---------------------------------------------------------------------------

struct LLMBridgeConfig {
    std::string endpoint        = "tcp://localhost:6000"; ///< Remote LLM service
    int         timeout_ms      = 2000;                   ///< Request timeout
    int         max_retries     = 2;                      ///< Retry count on timeout
    int         proto_version   = 1;                      ///< Gap 4.5
    int         io_threads      = 1;
};

// ---------------------------------------------------------------------------
// LLMBridge  (PIMPL)
// ---------------------------------------------------------------------------

/**
 * @class LLMBridge
 * @brief Sends prompts to an external LLM and collects responses over ZMQ REQ-REP.
 *
 * Thread-safety: infer() is NOT thread-safe.  Use one bridge per thread or
 * serialise externally.
 */
class LLMBridge {
public:
    explicit LLMBridge(LLMBridgeConfig config = {});
    ~LLMBridge();

    LLMBridge(const LLMBridge&)            = delete;
    LLMBridge& operator=(const LLMBridge&) = delete;
    LLMBridge(LLMBridge&&) noexcept;
    LLMBridge& operator=(LLMBridge&&) noexcept;

    /// Send a prompt and block until a response arrives or timeout.
    [[nodiscard]] LLMResponse infer(const LLMRequest& request);

    /// Returns true if the underlying ZMQ connection appears healthy.
    [[nodiscard]] bool is_healthy() const noexcept;

    const LLMBridgeConfig& config() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace nikola::infrastructure


// ===========================================================================
// Implementation block — compiled only when requested (header-only pattern)
// ===========================================================================

#ifdef NIKOLA_LLM_BRIDGE_IMPL

#include <nikola/infrastructure/spine.hpp>
#include <nikola/infrastructure/circuit_breaker.hpp>

namespace nikola::infrastructure {

struct LLMBridge::Impl {
    LLMBridgeConfig cfg;
    ZmqSpine        spine;
    CircuitBreaker  cb;

    explicit Impl(LLMBridgeConfig c)
        : cfg(std::move(c))
        , spine(cfg.io_threads)
        , cb(CircuitBreaker::Config{cfg.max_retries, std::chrono::milliseconds(500), "llm"})
    {}
};

LLMBridge::LLMBridge(LLMBridgeConfig config)
    : impl_(std::make_unique<Impl>(std::move(config)))
{}

LLMBridge::~LLMBridge() = default;
LLMBridge::LLMBridge(LLMBridge&&) noexcept = default;
LLMBridge& LLMBridge::operator=(LLMBridge&&) noexcept = default;

LLMResponse LLMBridge::infer(const LLMRequest& request) {
    RetryPolicy policy{impl_->cfg.max_retries, ZMQ_BACKOFF_BASE_MS, MessagePriority::CONTROL};

    LLMResponse response;
    response.request_id = request.request_id;

    bool ok = retry_with_circuit_breaker([&]() -> bool {
        // In a real implementation: create REQ socket, set timeout, send, recv.
        // For now we mark as failed (no live service available in tests).
        return false;
    }, impl_->cb, policy);

    response.ok = ok;
    if (!ok) {
        response.error_msg = "Bridge: no live LLM endpoint (test mode)";
    }
    return response;
}

bool LLMBridge::is_healthy() const noexcept {
    return !impl_->cb.is_open();
}

const LLMBridgeConfig& LLMBridge::config() const noexcept {
    return impl_->cfg;
}

} // namespace nikola::infrastructure
#endif // NIKOLA_LLM_BRIDGE_IMPL
