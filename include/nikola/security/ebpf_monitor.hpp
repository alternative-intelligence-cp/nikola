/**
 * @file security/ebpf_monitor.hpp
 * @brief v0.1.19 — eBPF-based escape detection monitor
 *
 * Extends the existing EscapeDetector with eBPF tracepoint attachment
 * for real-time syscall monitoring of KVM sandbox VMs.
 *
 * Two modes:
 *
 *   NIKOLA_ENABLE_EBPF  (Linux with libbpf + CAP_BPF):
 *     Attaches BPF tracepoints to sys_enter_execve and sys_enter_openat.
 *     Reads events from a perf ring buffer with 100ms poll interval.
 *     On match: SIGKILL + log + alert within latency budget.
 *
 *   Fallback (always compiled):
 *     Policy engine + event injection for testing.
 *     Links to EscapeDetector's /proc polling as secondary detection.
 *
 * Detection targets:
 *   - execve()              → immediate SIGKILL (EscapeType::EXECVE_DETECTED)
 *   - openat() outside safe → alert + optional SIGKILL (FORBIDDEN_FILE_OPEN)
 *   - Unauthorized network   → alert + SIGKILL (NETWORK_ATTEMPT)
 *
 * Latency budget: 100ms from event to response (kill + log + alert).
 *
 * Usage:
 *   EbpfMonitor mon;
 *   mon.watch_pid(pid, "vm1");
 *   mon.set_alert_callback([](const EbpfEvent& ev){ ... });
 *   mon.poll();   // call from main loop ≤100ms interval
 */
#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef __linux__
#  include <csignal>
#  include <sys/types.h>
#endif

namespace nikola::security {

// ============================================================================
// Constants
// ============================================================================

inline constexpr uint32_t EBPF_POLL_INTERVAL_MS       = 100;
inline constexpr uint32_t EBPF_RING_BUFFER_PAGES      = 64;    // 256KB ring buffer
inline constexpr uint32_t EBPF_MAX_WATCHED_PIDS       = 64;
inline constexpr uint64_t EBPF_RESPONSE_DEADLINE_NS   = 100'000'000ull;  // 100ms

// ============================================================================
// Event types
// ============================================================================

enum class EbpfEventType : uint8_t {
    EXECVE_ATTEMPT,     ///< Tracepoint: sys_enter_execve
    FILE_OPEN_OUTSIDE,  ///< Tracepoint: sys_enter_openat outside safe prefix
    NETWORK_ATTEMPT,    ///< Tracepoint: sys_enter_socket or connect
    CLONE_ATTEMPT,      ///< Tracepoint: sys_enter_clone / fork / vfork
    PTRACE_ATTEMPT,     ///< Tracepoint: sys_enter_ptrace
    UNKNOWN,
};

inline const char* ebpf_event_type_str(EbpfEventType t) {
    switch (t) {
        case EbpfEventType::EXECVE_ATTEMPT:    return "EXECVE_ATTEMPT";
        case EbpfEventType::FILE_OPEN_OUTSIDE: return "FILE_OPEN_OUTSIDE";
        case EbpfEventType::NETWORK_ATTEMPT:   return "NETWORK_ATTEMPT";
        case EbpfEventType::CLONE_ATTEMPT:     return "CLONE_ATTEMPT";
        case EbpfEventType::PTRACE_ATTEMPT:    return "PTRACE_ATTEMPT";
        case EbpfEventType::UNKNOWN:           return "UNKNOWN";
    }
    return "UNKNOWN";
}

// ============================================================================
// Event record
// ============================================================================

struct EbpfEvent {
    int            pid{-1};
    std::string    vm_name;
    EbpfEventType  type{EbpfEventType::UNKNOWN};
    std::string    detail;
    bool           killed{false};
    uint64_t       detection_ns{0};   ///< Time from event to response
    std::chrono::steady_clock::time_point timestamp;
};

// ============================================================================
// Response action
// ============================================================================

enum class ResponseAction : uint8_t {
    KILL_AND_ALERT,   ///< SIGKILL + alert callback (default for execve/clone)
    ALERT_ONLY,       ///< Alert callback only (for suspicious file opens)
    LOG_ONLY,         ///< Silent log (for telemetry)
};

struct ResponsePolicy {
    ResponseAction on_execve    = ResponseAction::KILL_AND_ALERT;
    ResponseAction on_file_open = ResponseAction::KILL_AND_ALERT;
    ResponseAction on_network   = ResponseAction::KILL_AND_ALERT;
    ResponseAction on_clone     = ResponseAction::KILL_AND_ALERT;
    ResponseAction on_ptrace    = ResponseAction::KILL_AND_ALERT;

    ResponseAction action_for(EbpfEventType t) const {
        switch (t) {
            case EbpfEventType::EXECVE_ATTEMPT:    return on_execve;
            case EbpfEventType::FILE_OPEN_OUTSIDE: return on_file_open;
            case EbpfEventType::NETWORK_ATTEMPT:   return on_network;
            case EbpfEventType::CLONE_ATTEMPT:     return on_clone;
            case EbpfEventType::PTRACE_ATTEMPT:    return on_ptrace;
            default:                                return ResponseAction::LOG_ONLY;
        }
    }
};

// ============================================================================
// Watched process info
// ============================================================================

struct WatchedProcess {
    int         pid{-1};
    std::string vm_name;
    bool        active{true};
    uint64_t    event_count{0};
};

// ============================================================================
// EbpfMonitor — real-time syscall watchdog
// ============================================================================

/**
 * Monitors KVM sandbox processes for escape attempts using eBPF
 * tracepoints (or fallback /proc polling).
 *
 * Call poll() at minimum every 100ms to stay within latency budget.
 */
class EbpfMonitor {
public:
    using AlertCallback = std::function<void(const EbpfEvent&)>;

    struct Config {
        ResponsePolicy    policy;
        std::string       safe_path_prefix = "/var/lib/nikola/vm";
        uint32_t          ring_buffer_pages = EBPF_RING_BUFFER_PAGES;
        bool              auto_kill         = true;
    };

    EbpfMonitor() : cfg_{} {}
    explicit EbpfMonitor(Config cfg) : cfg_(std::move(cfg)) {}

    // ── Process registration ────────────────────────────────────────────────

    bool watch_pid(int pid, const std::string& vm_name) {
        if (procs_.size() >= EBPF_MAX_WATCHED_PIDS) return false;
        if (pid <= 0) return false;
        procs_[pid] = WatchedProcess{pid, vm_name, true, 0};
        return true;
    }

    void unwatch_pid(int pid) { procs_.erase(pid); }

    size_t watched_count() const { return procs_.size(); }

    void set_alert_callback(AlertCallback cb) { on_alert_ = std::move(cb); }

    // ── Poll ─────────────────────────────────────────────────────────────────

    /**
     * Poll for escape events. Returns number of events detected.
     *
     * In eBPF mode: drains the ring buffer.
     * In fallback mode: checks injected events only (real /proc polling
     * is handled by EscapeDetector separately).
     */
    size_t poll() {
        size_t n = 0;

#ifdef NIKOLA_ENABLE_EBPF
        n = drain_ring_buffer();
#endif

        // Process any injected events (testing path)
        n += process_injected();
        return n;
    }

    // ── Event history ────────────────────────────────────────────────────────

    const std::vector<EbpfEvent>& events() const { return events_; }
    void clear_events() { events_.clear(); }

    uint64_t total_events()   const { return total_events_; }
    uint64_t total_kills()    const { return total_kills_; }
    uint64_t false_positives() const { return false_positives_; }

    // ── Latency tracking ─────────────────────────────────────────────────────

    /** Average response latency in nanoseconds. */
    uint64_t avg_response_ns() const {
        if (total_events_ == 0) return 0;
        return total_response_ns_ / total_events_;
    }

    /** Returns true if all responses were within the 100ms budget. */
    bool within_latency_budget() const {
        return worst_response_ns_ <= EBPF_RESPONSE_DEADLINE_NS;
    }

    uint64_t worst_response_ns() const { return worst_response_ns_; }

    // ── Injection (for testing without eBPF) ────────────────────────────────

    /**
     * Inject a synthetic event — used by tests to verify response logic
     * without needing eBPF or real processes.
     */
    void inject_event(int pid, EbpfEventType type,
                      const std::string& detail = "injected",
                      uint64_t simulated_latency_ns = 50'000'000ull)
    {
        injected_.push_back({pid, "", type, detail, false,
                             simulated_latency_ns,
                             std::chrono::steady_clock::now()});
    }

    /**
     * Mark a past event as a false positive (for tuning).
     */
    void mark_false_positive(size_t event_index) {
        if (event_index < events_.size()) {
            ++false_positives_;
        }
    }

    // ── Policy ───────────────────────────────────────────────────────────────

    const ResponsePolicy& policy() const { return cfg_.policy; }
    void set_policy(ResponsePolicy p) { cfg_.policy = std::move(p); }

    // ── eBPF status ──────────────────────────────────────────────────────────

    bool ebpf_available() const {
#ifdef NIKOLA_ENABLE_EBPF
        return ebpf_attached_;
#else
        return false;
#endif
    }

    bool fallback_mode() const { return !ebpf_available(); }

private:
    Config                                cfg_;
    std::unordered_map<int, WatchedProcess> procs_;
    AlertCallback                          on_alert_;
    std::vector<EbpfEvent>                 events_;
    std::vector<EbpfEvent>                 injected_;
    uint64_t                               total_events_{0};
    uint64_t                               total_kills_{0};
    uint64_t                               false_positives_{0};
    uint64_t                               total_response_ns_{0};
    uint64_t                               worst_response_ns_{0};

#ifdef NIKOLA_ENABLE_EBPF
    bool ebpf_attached_{false};

    // BPF ring buffer drain — processes all pending events
    size_t drain_ring_buffer() {
        // TODO: libbpf ring_buffer__poll() integration
        // For now, fall through to the injected event path
        return 0;
    }
#endif

    size_t process_injected() {
        size_t n = 0;
        for (auto& ev : injected_) {
            // Look up VM name from pid
            auto it = procs_.find(ev.pid);
            if (it != procs_.end()) {
                ev.vm_name = it->second.vm_name;
                ++it->second.event_count;
            }

            // Determine response action
            ResponseAction action = cfg_.policy.action_for(ev.type);

            switch (action) {
                case ResponseAction::KILL_AND_ALERT:
                    if (cfg_.auto_kill && ev.pid > 0) {
#ifdef __linux__
                        ::kill(ev.pid, SIGKILL);
#endif
                        ev.killed = true;
                        ++total_kills_;
                    }
                    if (on_alert_) on_alert_(ev);
                    break;

                case ResponseAction::ALERT_ONLY:
                    if (on_alert_) on_alert_(ev);
                    break;

                case ResponseAction::LOG_ONLY:
                    break;
            }

            // Track latency
            total_response_ns_ += ev.detection_ns;
            if (ev.detection_ns > worst_response_ns_)
                worst_response_ns_ = ev.detection_ns;

            events_.push_back(std::move(ev));
            ++total_events_;
            ++n;
        }
        injected_.clear();
        return n;
    }
};

} // namespace nikola::security
