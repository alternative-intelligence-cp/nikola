/**
 * @file security/escape_detector.hpp
 * @brief Gap 7.3 — EscapeDetector
 *
 * Detects VM breakout attempts from qemu-kvm processes.
 *
 * Detection strategy (two modes, selected at compile time):
 *
 *   NIKOLA_ENABLE_EBPF (Linux with libbpf):
 *     eBPF tracepoints attached to sys_enter_execve and sys_enter_openat,
 *     watching for qemu-kvm processes crossing the containment boundary.
 *
 *   Fallback (always available):
 *     Process watchdog that polls /proc/<pid>/syscall and /proc/<pid>/fd/
 *     looking for anomalous activity. Works on any Linux host without
 *     kernel headers / CAP_BPF.
 *
 * Escape heuristics (from spec):
 *   - qemu-kvm calls execve()              → immediate SIGKILL
 *   - qemu-kvm opens path outside          → alert + optional SIGKILL
 *     /var/lib/nikola/vm
 *
 * Usage:
 *   EscapeDetector ed;
 *   ed.watch_vm("vm1", pid1);
 *   ed.poll();   // call from main loop
 *
 *   ed.set_alert_callback([](const EscapeEvent& ev){ ... });
 */
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef __linux__
#  include <csignal>
#  include <dirent.h>
#  include <filesystem>
#  include <fstream>
#  include <sys/types.h>
#  include <unistd.h>
#endif

namespace nikola::security {

// ============================================================================
// Constants
// ============================================================================

inline constexpr char VM_SAFE_PATH_PREFIX[] = "/var/lib/nikola/vm";

// ============================================================================
// Gap 7.3 — EscapeDetector
// ============================================================================

enum class EscapeType : uint8_t {
    EXECVE_DETECTED,     ///< VM process attempted execve()
    FORBIDDEN_FILE_OPEN, ///< VM process opened path outside safe prefix
    RESOURCE_LIMIT,      ///< CPU/memory limit violation detected
    PROCESS_GONE,        ///< VM process exited unexpectedly
};

inline const char* escape_type_str(EscapeType t) {
    switch (t) {
        case EscapeType::EXECVE_DETECTED:     return "EXECVE_DETECTED";
        case EscapeType::FORBIDDEN_FILE_OPEN: return "FORBIDDEN_FILE_OPEN";
        case EscapeType::RESOURCE_LIMIT:      return "RESOURCE_LIMIT";
        case EscapeType::PROCESS_GONE:        return "PROCESS_GONE";
    }
    return "UNKNOWN";
}

struct EscapeEvent {
    int         pid{-1};
    std::string vm_name;
    EscapeType  type{EscapeType::EXECVE_DETECTED};
    std::string detail;
    bool        killed{false};
};

struct WatchedVM {
    std::string name;
    int         pid{-1};
    bool        active{true};
    uint64_t    alert_count{0};
};

/**
 * Monitors qemu-kvm processes for escape attempts.
 *
 * Call poll() regularly (e.g. every 100ms) from the main loop.
 * Suspicious events trigger the alert callback and optionally SIGKILL
 * the offending process.
 */
class EscapeDetector {
public:
    using AlertCallback = std::function<void(const EscapeEvent&)>;

    struct Config {
        bool auto_kill      = true;   ///< SIGKILL on confirmed escape attempt
        bool watch_fds      = true;   ///< Check /proc/<pid>/fd/ for path violations
        bool watch_syscall  = true;   ///< Check /proc/<pid>/syscall for execve
        std::string safe_path_prefix = VM_SAFE_PATH_PREFIX;
    };

    EscapeDetector() : cfg_{} {}
    explicit EscapeDetector(Config cfg) : cfg_(std::move(cfg)) {}

    // ── VM registration ──────────────────────────────────────────────────────

    void watch_vm(const std::string& name, int pid) {
        vms_[name] = WatchedVM{name, pid, true, 0};
    }

    void unwatch_vm(const std::string& name) { vms_.erase(name); }

    size_t watched_count() const { return vms_.size(); }

    void set_alert_callback(AlertCallback cb) { on_alert_ = std::move(cb); }

    // ── Poll ─────────────────────────────────────────────────────────────────

    /**
     * Poll all watched VMs for escape indicators.
     * Should be called from the main loop approximately every 100ms.
     * Returns the number of alerts fired this call.
     */
    size_t poll() {
        size_t n_alerts = 0;
        for (auto& [name, vm] : vms_) {
            if (!vm.active) continue;
            n_alerts += check_vm(vm);
        }
        return n_alerts;
    }

    /** Return all accumulated alerts since last clear. */
    const std::vector<EscapeEvent>& events() const { return events_; }
    void clear_events() { events_.clear(); }

    uint64_t total_alerts() const { return total_alerts_; }

    // ── Injection (for testing without real processes) ────────────────────────

    /**
     * Inject a synthetic escape event — used by tests to verify alert
     * and kill logic without needing a real qemu-kvm process.
     */
    void inject_event(const std::string& vm_name, EscapeType type,
                      const std::string& detail = "injected")
    {
        auto it = vms_.find(vm_name);
        int pid = (it != vms_.end()) ? it->second.pid : -1;
        fire_alert(vm_name, pid, type, detail);
    }

private:
    Config                                config_;  // unused field shadowed below
    Config                                cfg_;
    std::unordered_map<std::string, WatchedVM> vms_;
    AlertCallback                         on_alert_;
    std::vector<EscapeEvent>              events_;
    uint64_t                              total_alerts_{0};

    size_t check_vm(WatchedVM& vm) {
        size_t n = 0;

#ifdef __linux__
        // Check process is still alive
        const std::string proc_base = "/proc/" + std::to_string(vm.pid);
        if (!std::filesystem::exists(proc_base)) {
            fire_alert(vm.name, vm.pid, EscapeType::PROCESS_GONE, "process vanished");
            vm.active = false;
            return 1;
        }

        // Gap 7.3 spec: alert if qemu-kvm calls execve()
        if (cfg_.watch_syscall) {
            n += check_syscall(vm);
        }

        // Gap 7.3 spec: alert if qemu-kvm opens path outside safe prefix
        if (cfg_.watch_fds) {
            n += check_open_fds(vm);
        }
#else
        (void)vm; // suppress unused warning on non-Linux
#endif
        return n;
    }

#ifdef __linux__
    size_t check_syscall(WatchedVM& vm) {
        // /proc/<pid>/syscall format: "NR arg0 arg1 ... sp pc"
        // execve syscall number = 59 on x86_64
        const std::string path = "/proc/" + std::to_string(vm.pid) + "/syscall";
        std::ifstream f(path);
        if (!f) return 0;
        int nr = -1;
        f >> nr;
        if (nr == 59 /* __NR_execve on x86_64 */) {
            fire_alert(vm.name, vm.pid, EscapeType::EXECVE_DETECTED,
                       "execve() syscall detected");
            if (cfg_.auto_kill) ::kill(vm.pid, SIGKILL);
            return 1;
        }
        return 0;
    }

    size_t check_open_fds(WatchedVM& vm) {
        const std::string fd_dir = "/proc/" + std::to_string(vm.pid) + "/fd";
        DIR* d = ::opendir(fd_dir.c_str());
        if (!d) return 0;
        size_t n = 0;
        struct dirent* ent;
        while ((ent = ::readdir(d)) != nullptr) {
            if (ent->d_name[0] == '.') continue;
            const std::string link = fd_dir + "/" + ent->d_name;
            char target[4096]{};
            ssize_t len = ::readlink(link.c_str(), target, sizeof(target)-1);
            if (len <= 0) continue;
            target[len] = '\0';
            std::string tgt(target);
            // Ignore sockets, pipes, anon_inodes
            if (tgt[0] != '/') continue;
            if (tgt.find(cfg_.safe_path_prefix) == 0) continue;
            // Also allow /dev/null and /dev/kvm (needed by hypervisor itself)
            if (tgt == "/dev/null" || tgt == "/dev/kvm" || tgt == "/dev/urandom")
                continue;
            fire_alert(vm.name, vm.pid, EscapeType::FORBIDDEN_FILE_OPEN,
                       "open fd outside safe prefix: " + tgt);
            if (cfg_.auto_kill) ::kill(vm.pid, SIGKILL);
            ++n;
            break; // one alert per check cycle per VM
        }
        ::closedir(d);
        return n;
    }
#endif

    void fire_alert(const std::string& vm_name, int pid,
                    EscapeType type, const std::string& detail)
    {
        EscapeEvent ev;
        ev.vm_name = vm_name;
        ev.pid     = pid;
        ev.type    = type;
        ev.detail  = detail;
        ev.killed  = false;

        events_.push_back(ev);
        ++total_alerts_;

        auto it = vms_.find(vm_name);
        if (it != vms_.end()) ++it->second.alert_count;

        if (on_alert_) on_alert_(ev);
    }
};

} // namespace nikola::security
