/**
 * @file include/nikola/infrastructure/circuit_breaker.hpp
 * @brief Circuit Breaker Pattern + ZMQ Retry Logic for Nikola Infrastructure.
 *
 * Resolves Gap 4.1: Message Timeout and Retry Logic.
 *
 * Implements:
 *   - CircuitBreaker: CLOSED → OPEN → HALF_OPEN state machine
 *   - ZmqSocketConfig: Timeout constants (100ms control, 5ms data)
 *   - RetryPolicy: Exponential back-off (1ms, 2ms, 4ms; 3 attempts max)
 *
 * Design principle: This header is intentionally decoupled from <zmq.hpp>.
 * The ZmqReliableSocket adapter is in spine.hpp which owns the ZMQ objects.
 * Circuit-breaker logic is pure C++ so it is fully testable without sockets.
 */

#pragma once

#include <chrono>
#include <atomic>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <thread>

namespace nikola::infrastructure {

// ---------------------------------------------------------------------------
// Gap 4.1 – Timeout constants
// ---------------------------------------------------------------------------

/// Control-plane timeout (heartbeats, commands). 100 ms.
inline constexpr int ZMQ_CONTROL_TIMEOUT_MS = 100;

/// Data-plane timeout (physics spikes, waveform packets). 5 ms.
inline constexpr int ZMQ_DATA_TIMEOUT_MS = 5;

/// Maximum send/receive retries before circuit trips.
inline constexpr int ZMQ_MAX_RETRIES = 3;

/// Base back-off for retry #0. Doubles each attempt (1ms, 2ms, 4ms).
inline constexpr int ZMQ_BACKOFF_BASE_MS = 1;

// ---------------------------------------------------------------------------
// MessagePriority
// ---------------------------------------------------------------------------

/// Differentiates control vs data plane for timeout selection.
enum class MessagePriority {
    CONTROL, ///< Commands, heartbeats, topology updates — 100 ms timeout
    DATA,    ///< Waveform packets, spike trains         —   5 ms timeout
};

/// Returns the timeout in milliseconds for a given priority.
[[nodiscard]] inline int timeout_ms(MessagePriority p) noexcept {
    return p == MessagePriority::CONTROL ? ZMQ_CONTROL_TIMEOUT_MS : ZMQ_DATA_TIMEOUT_MS;
}

// ---------------------------------------------------------------------------
// RetryPolicy
// ---------------------------------------------------------------------------

/// Encapsulates retry configuration.  Values match the spec defaults.
struct RetryPolicy {
    int max_retries   = ZMQ_MAX_RETRIES;       ///< Total attempts allowed
    int backoff_ms    = ZMQ_BACKOFF_BASE_MS;    ///< Initial back-off (doubles each attempt)
    MessagePriority priority = MessagePriority::CONTROL;

    /// Computes back-off duration for attempt index `n` (0-based).
    [[nodiscard]] std::chrono::milliseconds backoff_for(int n) const noexcept {
        // 1ms, 2ms, 4ms for n = 0, 1, 2
        return std::chrono::milliseconds(backoff_ms << n);
    }

    /// Timeout applicable to this policy.
    [[nodiscard]] std::chrono::milliseconds timeout() const noexcept {
        return std::chrono::milliseconds(timeout_ms(priority));
    }
};

// ---------------------------------------------------------------------------
// CircuitBreaker — state machine
// ---------------------------------------------------------------------------

/// Circuit-breaker state.
enum class CBState : uint8_t {
    CLOSED,     ///< Normal operation — messages flow through
    OPEN,       ///< Failed: messages are rejected immediately
    HALF_OPEN,  ///< Recovery probe: one trial message allowed
};

[[nodiscard]] inline const char* cb_state_name(CBState s) noexcept {
    switch (s) {
        case CBState::CLOSED:    return "CLOSED";
        case CBState::OPEN:      return "OPEN";
        case CBState::HALF_OPEN: return "HALF_OPEN";
    }
    return "UNKNOWN";
}

/// Configuration for a CircuitBreaker instance.
/// Extracted to namespace scope to avoid C++ nested-class default-initializer restriction.
struct CircuitBreakerConfig {
    /// Number of consecutive failures to open the breaker.
    int failure_threshold = ZMQ_MAX_RETRIES;

    /// How long to stay OPEN before probing again.
    std::chrono::milliseconds cool_down{500};

    std::string component_name = "unknown";
};

// ---------------------------------------------------------------------------
// CircuitBreaker — state machine
// ---------------------------------------------------------------------------

/**
 * @class CircuitBreaker
 * @brief Tracks per-component send failures and trips after threshold.
 *
 * Thread-safe via std::atomic for state and counters.
 * The breaker is keyed to a single logical endpoint (component name).
 *
 * State transitions:
 *   CLOSED   — failure_count >= failure_threshold → OPEN
 *   OPEN     — cool-down elapsed                  → HALF_OPEN
 *   HALF_OPEN — success                          → CLOSED
 *   HALF_OPEN — failure                          → OPEN  (reset cool-down)
 */
class CircuitBreaker {
public:
    using Config = CircuitBreakerConfig;  ///< Type alias for ergonomic construction

    explicit CircuitBreaker(Config cfg = {})
        : config_(std::move(cfg))
        , state_(CBState::CLOSED)
        , failure_count_(0)
        , opened_at_{}
    {}

    // Non-copyable, non-movable (atomic members prevent move)
    CircuitBreaker(const CircuitBreaker&)            = delete;
    CircuitBreaker& operator=(const CircuitBreaker&) = delete;
    CircuitBreaker(CircuitBreaker&&)                 = delete;
    CircuitBreaker& operator=(CircuitBreaker&&)      = delete;

    // -----------------------------------------------------------------------
    // Query
    // -----------------------------------------------------------------------

    [[nodiscard]] CBState state() const noexcept { return state_.load(std::memory_order_acquire); }
    [[nodiscard]] int     failure_count() const noexcept { return failure_count_.load(std::memory_order_relaxed); }
    [[nodiscard]] bool    is_open() const noexcept { return state() == CBState::OPEN; }

    /**
     * @brief Returns true if a message attempt is permitted.
     * - CLOSED: always permitted.
     * - OPEN: permitted only after cool-down (transitions to HALF_OPEN).
     * - HALF_OPEN: exactly one probe permitted.
     */
    [[nodiscard]] bool allow_attempt() noexcept {
        using clock = std::chrono::steady_clock;
        auto s = state();

        if (s == CBState::CLOSED) return true;

        if (s == CBState::OPEN) {
            if (clock::now() - opened_at_ >= config_.cool_down) {
                // Transition to probe state
                CBState expected = CBState::OPEN;
                if (state_.compare_exchange_strong(expected, CBState::HALF_OPEN,
                        std::memory_order_acq_rel, std::memory_order_relaxed)) {
                    return true; // We own the probe attempt
                }
            }
            return false;
        }

        // HALF_OPEN: allow the single probe
        return s == CBState::HALF_OPEN;
    }

    // -----------------------------------------------------------------------
    // Outcome recording
    // -----------------------------------------------------------------------

    /// Call after a successful send/recv to reset breaker or close it.
    void record_success() noexcept {
        failure_count_.store(0, std::memory_order_relaxed);

        CBState expected_half = CBState::HALF_OPEN;
        if (state_.compare_exchange_strong(expected_half, CBState::CLOSED,
                std::memory_order_acq_rel, std::memory_order_relaxed)) {
            return; // Healed
        }
        // Already CLOSED — nothing to do
        state_.store(CBState::CLOSED, std::memory_order_release);
    }

    /// Call after a failed send/recv attempt.
    void record_failure() noexcept {
        int count = failure_count_.fetch_add(1, std::memory_order_acq_rel) + 1;

        if (count >= config_.failure_threshold) {
            opened_at_ = std::chrono::steady_clock::now();
            state_.store(CBState::OPEN, std::memory_order_release);
        }

        // If we were HALF_OPEN and probe failed, go back to OPEN
        CBState expected_half = CBState::HALF_OPEN;
        state_.compare_exchange_strong(expected_half, CBState::OPEN,
                std::memory_order_acq_rel, std::memory_order_relaxed);
    }

    /// Manually reset (e.g., after component restart).
    void reset() noexcept {
        failure_count_.store(0, std::memory_order_relaxed);
        state_.store(CBState::CLOSED, std::memory_order_release);
    }

    const Config& config() const noexcept { return config_; }

private:
    Config config_;
    std::atomic<CBState> state_;
    std::atomic<int>     failure_count_;
    std::chrono::steady_clock::time_point opened_at_;
};

// ---------------------------------------------------------------------------
// RetryExecutor — helper that drives retries + circuit breaker together
// ---------------------------------------------------------------------------

/**
 * @brief Executes an operation with retry logic and circuit-breaker protection.
 *
 * @tparam Op  Callable() → bool  (true = success)
 * @return     true if operation ultimately succeeded
 */
template<typename Op>
bool retry_with_circuit_breaker(Op&& op, CircuitBreaker& cb, const RetryPolicy& policy) {
    if (!cb.allow_attempt()) {
        return false; // Breaker open — fast-fail
    }

    for (int attempt = 0; attempt < policy.max_retries; ++attempt) {
        bool ok = op();
        if (ok) {
            cb.record_success();
            return true;
        }
        // Back-off before next attempt (skip sleep on last attempt)
        if (attempt + 1 < policy.max_retries) {
            std::this_thread::sleep_for(policy.backoff_for(attempt));
        }
    }

    cb.record_failure();
    return false;
}

} // namespace nikola::infrastructure
