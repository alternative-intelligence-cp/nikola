/**
 * @file include/nikola/infrastructure/heartbeat.hpp
 * @brief Heartbeat Sentinel System for Nikola component lifecycle.
 *
 * Resolves Gap 4.2: Component Crash Recovery.
 *
 * Protocol:
 *   - Each component publishes a HEARTBEAT event every 100 ms.
 *   - ComponentWatchdog maintains a LastSeen map (steady_clock).
 *   - If  Now − LastSeen > 500 ms  → component marked DEAD.
 *   - Dead component: kill(pid, SIGKILL), shm_unlink, trigger restart callback.
 *
 * Design: No ZMQ dependency here. The heartbeat *reception* is driven by the
 * ZmqSpine (spine.hpp), which calls `watchdog.update_heartbeat(name)` on each
 * received HEARTBEAT frame.  ComponentWatchdog is testable with a mock clock.
 */

#pragma once

#include <chrono>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>
#include <csignal>
#include <sys/types.h>

// POSIX
#if defined(__unix__) || defined(__APPLE__)
#  include <signal.h>
#  include <sys/mman.h>   // shm_unlink declaration
#endif

namespace nikola::infrastructure {

// ---------------------------------------------------------------------------
// Constants (Gap 4.2)
// ---------------------------------------------------------------------------

/// How often each component is expected to publish a heartbeat.
inline constexpr auto HEARTBEAT_INTERVAL = std::chrono::milliseconds(100);

/// Dead-detection threshold: silence longer than this → DEAD.
inline constexpr auto HEARTBEAT_TIMEOUT  = std::chrono::milliseconds(500);

/// After this many consecutive missed intervals the component is killed.
inline constexpr int  HEARTBEAT_MAX_MISSED = 5;

// ---------------------------------------------------------------------------
// ComponentStatus
// ---------------------------------------------------------------------------

enum class ComponentStatus : uint8_t {
    ALIVE,   ///< Receiving heartbeats within timeout window
    TIMEOUT, ///< No heartbeat received recently; may still recover
    DEAD,    ///< Missed >= MAX_MISSED beats; kill-and-restart pending
};

[[nodiscard]] inline const char* component_status_name(ComponentStatus s) noexcept {
    switch (s) {
        case ComponentStatus::ALIVE:   return "ALIVE";
        case ComponentStatus::TIMEOUT: return "TIMEOUT";
        case ComponentStatus::DEAD:    return "DEAD";
    }
    return "UNKNOWN";
}

// ---------------------------------------------------------------------------
// ComponentHealth — per-component tracking record
// ---------------------------------------------------------------------------

struct ComponentHealth {
    std::string name;
    pid_t       pid;
    std::chrono::steady_clock::time_point last_heartbeat;
    int         missed_heartbeats = 0;
    ComponentStatus status = ComponentStatus::ALIVE;
};

// ---------------------------------------------------------------------------
// ComponentWatchdog
// ---------------------------------------------------------------------------

/**
 * @class ComponentWatchdog
 * @brief Monitors liveness of registered sub-processes via heartbeat timestamps.
 *
 * Thread-safety: Not internally synchronised; CallerN must serialise if
 * multiple threads drive check_health() and update_heartbeat().
 *
 * Optional death callback lets the Orchestrator restart the component.
 */
class ComponentWatchdog {
public:
    using DeathCallback = std::function<void(const std::string& /*name*/)>;

    explicit ComponentWatchdog(
        std::chrono::milliseconds timeout   = HEARTBEAT_TIMEOUT,
        int                       max_missed = HEARTBEAT_MAX_MISSED
    )
        : timeout_(timeout)
        , max_missed_(max_missed)
    {}

    // -----------------------------------------------------------------------
    // Registration
    // -----------------------------------------------------------------------

    /// Register a new component with its OS PID.
    void register_component(const std::string& name, pid_t pid) {
        components_[name] = ComponentHealth{
            name,
            pid,
            clock::now(),
            0,
            ComponentStatus::ALIVE
        };
    }

    /// Deregister a component (e.g., after clean shutdown).
    void deregister_component(const std::string& name) {
        components_.erase(name);
    }

    /// Set the callback invoked when a component is classified as DEAD.
    void set_death_callback(DeathCallback cb) { on_death_ = std::move(cb); }

    // -----------------------------------------------------------------------
    // Heartbeat update (called by ZmqSpine on each HEARTBEAT frame)
    // -----------------------------------------------------------------------

    void update_heartbeat(const std::string& name) {
        auto it = components_.find(name);
        if (it == components_.end()) return;
        auto& h = it->second;
        h.last_heartbeat    = clock::now();
        h.missed_heartbeats = 0;
        h.status            = ComponentStatus::ALIVE;
    }

    // -----------------------------------------------------------------------
    // Health check — call every 100 ms from watchdog thread
    // -----------------------------------------------------------------------

    /**
     * @brief Evaluate all components against the timeout window.
     * @return Vector of component names now classified as DEAD.
     */
    std::vector<std::string> check_health() {
        std::vector<std::string> dead;
        auto now = clock::now();

        for (auto& [name, h] : components_) {
            auto elapsed = now - h.last_heartbeat;

            if (elapsed > timeout_) {
                ++h.missed_heartbeats;

                if (h.missed_heartbeats >= max_missed_) {
                    h.status = ComponentStatus::DEAD;
                    dead.push_back(name);
                } else {
                    h.status = ComponentStatus::TIMEOUT;
                }
            } else {
                h.status = ComponentStatus::ALIVE;
            }
        }

        return dead;
    }

    // -----------------------------------------------------------------------
    // Kill and cleanup (Gap 4.2)
    // -----------------------------------------------------------------------

    /**
     * @brief Sends SIGKILL to the component's process and unlinks its SHM segment.
     *        Calls the death callback, then removes the component from the registry.
     */
    bool kill_and_cleanup(const std::string& name) {
        auto it = components_.find(name);
        if (it == components_.end()) return false;

        auto& h = it->second;

        // 1. Kill process (best-effort; pid may already be dead)
        if (h.pid > 0) {
            ::kill(h.pid, SIGKILL);
        }

        // 2. Cleanup shared memory segment
        std::string shm_name = "/nikola_" + name;
        ::shm_unlink(shm_name.c_str()); // best-effort, ignore ENOENT

        // 3. Invoke restart callback
        if (on_death_) on_death_(name);

        // 4. Remove from registry (caller restarts and re-registers)
        components_.erase(it);
        return true;
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    [[nodiscard]] bool is_registered(const std::string& name) const {
        return components_.count(name) > 0;
    }

    [[nodiscard]] ComponentStatus status(const std::string& name) const {
        auto it = components_.find(name);
        if (it == components_.end()) return ComponentStatus::DEAD;
        return it->second.status;
    }

    [[nodiscard]] int missed_beats(const std::string& name) const {
        auto it = components_.find(name);
        if (it == components_.end()) return -1;
        return it->second.missed_heartbeats;
    }

    [[nodiscard]] std::size_t component_count() const noexcept {
        return components_.size();
    }

    /// Direct access for testing and inspection.
    [[nodiscard]] const std::unordered_map<std::string, ComponentHealth>& all() const noexcept {
        return components_;
    }

private:
    using clock = std::chrono::steady_clock;

    std::chrono::milliseconds timeout_;
    int max_missed_;
    std::unordered_map<std::string, ComponentHealth> components_;
    DeathCallback on_death_;
};

// ---------------------------------------------------------------------------
// HeartbeatPublisher — lightweight "I am alive" stamp for each sub-process
// ---------------------------------------------------------------------------

/**
 * @brief Lightweight publisher side: sub-process calls `beat()` every 100 ms.
 * In the full system this writes to the ZMQ events socket; here it just records
 * the timestamp so it can be tested independently.
 */
class HeartbeatPublisher {
public:
    using SendFn = std::function<void(const std::string& /*topic*/)>;

    explicit HeartbeatPublisher(std::string name, SendFn send_fn = nullptr)
        : name_(std::move(name))
        , send_fn_(std::move(send_fn))
    {}

    /// Emit a heartbeat.  Calls send_fn if provided (e.g., zmq pub), else no-op.
    void beat() {
        last_beat_ = std::chrono::steady_clock::now();
        if (send_fn_) send_fn_("HEARTBEAT." + name_);
    }

    [[nodiscard]] const std::string& name() const noexcept { return name_; }
    [[nodiscard]] std::chrono::steady_clock::time_point last_beat() const noexcept {
        return last_beat_;
    }

private:
    std::string name_;
    SendFn send_fn_;
    std::chrono::steady_clock::time_point last_beat_{};
};

} // namespace nikola::infrastructure
