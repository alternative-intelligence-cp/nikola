/**
 * @file security/mini_vm_manager.hpp
 * @brief v0.2.6 — Mini-VM lifecycle manager for SIE Tier 1 deployment
 *
 * Manages a pool of pre-warmed KVM sandbox VMs for fast deployment
 * of self-generated modules. Provides:
 *   - VM pre-warming: boot N standby VMs at startup
 *   - VM recycling: reset overlay after each use (no full reboot)
 *   - Health monitoring: detect crashed/hung VMs, auto-recycle
 *   - ZMQ spine connection: modules in VMs communicate via spine
 *
 * All pool management and policy logic compiles without NIKOLA_ENABLE_KVM.
 * Actual VM operations delegate to KvmSandbox.
 *
 * Usage:
 *   MiniVMManager pool(config, sandbox);
 *   pool.warm_pool();                          // pre-boot standby VMs
 *   auto lease = pool.acquire("module_001");   // get a warm VM
 *   // ... deploy module, execute, collect results ...
 *   pool.release(lease.vm_name);               // recycle for next use
 */
#pragma once

#include "kvm_sandbox.hpp"
#include "vm_image_manager.hpp"
#include "ebpf_monitor.hpp"

#include <atomic>
#include <chrono>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace nikola::security {

// ============================================================================
// Constants
// ============================================================================

inline constexpr uint32_t MINIVM_DEFAULT_POOL_SIZE       = 4;
inline constexpr uint32_t MINIVM_MAX_POOL_SIZE           = KVM_MAX_VMS;
inline constexpr uint32_t MINIVM_HEALTH_CHECK_INTERVAL_S = 10;
inline constexpr uint32_t MINIVM_MAX_RECYCLES            = 100;
inline constexpr uint32_t MINIVM_BOOT_TIMEOUT_MS         = 15'000;
inline constexpr uint32_t MINIVM_EXEC_TIMEOUT_MS         = 30'000;

// ============================================================================
// Mini-VM States
// ============================================================================

enum class MiniVMState : uint8_t {
    STANDBY,     ///< Booted, idle, ready for lease
    LEASED,      ///< Assigned to a module execution
    EXECUTING,   ///< Module code running inside VM
    RECYCLING,   ///< Overlay being reset for next use
    UNHEALTHY,   ///< Failed health check — needs replacement
    RETIRED,     ///< Max recycles reached — being destroyed
};

inline const char* minivm_state_str(MiniVMState s) {
    switch (s) {
        case MiniVMState::STANDBY:   return "STANDBY";
        case MiniVMState::LEASED:    return "LEASED";
        case MiniVMState::EXECUTING: return "EXECUTING";
        case MiniVMState::RECYCLING: return "RECYCLING";
        case MiniVMState::UNHEALTHY: return "UNHEALTHY";
        case MiniVMState::RETIRED:   return "RETIRED";
    }
    return "UNKNOWN";
}

// ============================================================================
// Lease — returned to caller when acquiring a VM
// ============================================================================

struct VMLease {
    std::string vm_name;              ///< Name of the leased VM
    bool        valid{false};         ///< true if lease was granted
    std::string error;                ///< set if valid==false
    std::chrono::steady_clock::time_point lease_time;
};

// ============================================================================
// Deployment result — full execution report
// ============================================================================

struct DeploymentResult {
    bool        success{false};
    std::string vm_name;
    std::string module_name;
    std::string stdout_data;
    std::string error;
    double      boot_latency_s{0.0};  ///< Time from acquire to code start
    double      exec_latency_s{0.0};  ///< Time from code start to completion
    double      total_latency_s{0.0}; ///< End-to-end
    uint32_t    recycle_count{0};      ///< How many times this VM has been recycled
};

// ============================================================================
// Pool-level VM entry
// ============================================================================

struct MiniVMEntry {
    std::string    vm_name;
    MiniVMState    state{MiniVMState::STANDBY};
    uint32_t       recycle_count{0};
    std::string    current_module;     ///< Module using this VM (if leased)
    std::chrono::steady_clock::time_point last_activity;
    std::chrono::steady_clock::time_point boot_time;
};

// ============================================================================
// Configuration
// ============================================================================

struct MiniVMConfig {
    uint32_t    pool_size          = MINIVM_DEFAULT_POOL_SIZE;
    uint32_t    max_recycles       = MINIVM_MAX_RECYCLES;
    uint32_t    boot_timeout_ms    = MINIVM_BOOT_TIMEOUT_MS;
    uint32_t    exec_timeout_ms    = MINIVM_EXEC_TIMEOUT_MS;
    uint32_t    health_check_s     = MINIVM_HEALTH_CHECK_INTERVAL_S;
    std::string spine_endpoint;    ///< ZMQ endpoint for in-VM modules
    bool        enable_ebpf        = true;  ///< Attach eBPF monitor to VMs
};

// ============================================================================
// MiniVMManager
// ============================================================================

/**
 * Pool of pre-warmed KVM sandbox VMs for fast SIE Tier 1 deployment.
 *
 * Lifecycle:
 *   warm_pool() → acquire() → deploy_and_execute() → release() → [recycle]
 *
 * VMs are recycled by resetting their copy-on-write overlay rather than
 * destroying and recreating them (avoiding the full boot penalty).
 *
 * When NIKOLA_ENABLE_KVM is not defined, all operations succeed in
 * simulation mode for testing pool management logic.
 */
class MiniVMManager {
public:
    using HealthCallback = std::function<void(const std::string& vm_name,
                                               MiniVMState old_state,
                                               MiniVMState new_state,
                                               const std::string& reason)>;

    explicit MiniVMManager(MiniVMConfig cfg, KvmSandbox& sandbox)
        : cfg_(std::move(cfg)), sandbox_(sandbox) {}

    MiniVMManager(MiniVMConfig cfg, KvmSandbox& sandbox, EbpfMonitor& ebpf)
        : cfg_(std::move(cfg)), sandbox_(sandbox), ebpf_(&ebpf) {}

    void set_health_callback(HealthCallback cb) { on_health_ = std::move(cb); }

    // ── Pool warm-up ─────────────────────────────────────────────────────

    /**
     * Pre-boot pool_size VMs into STANDBY state.
     * Returns number of VMs successfully warmed.
     */
    uint32_t warm_pool() {
        uint32_t warmed = 0;
        for (uint32_t i = 0; i < cfg_.pool_size && i < MINIVM_MAX_POOL_SIZE; ++i) {
            std::string name = "minivm_" + std::to_string(next_id_++);
            if (boot_standby_vm(name)) {
                ++warmed;
            }
        }
        return warmed;
    }

    // ── Acquire / Release ────────────────────────────────────────────────

    /**
     * Acquire a warm VM from the pool.
     * Returns a VMLease with valid=true if a VM is available.
     */
    VMLease acquire(const std::string& module_name) {
        VMLease lease;
        lease.lease_time = std::chrono::steady_clock::now();

        // Find a STANDBY VM
        for (auto& [name, entry] : pool_) {
            if (entry.state == MiniVMState::STANDBY) {
                entry.state = MiniVMState::LEASED;
                entry.current_module = module_name;
                entry.last_activity = std::chrono::steady_clock::now();
                ++total_leases_;

                lease.vm_name = name;
                lease.valid   = true;

                // Register PID with eBPF monitor if available
                if (ebpf_) {
                    auto* vm = sandbox_.get_vm(name);
                    if (vm && vm->pid > 0) {
                        ebpf_->watch_pid(vm->pid, name);
                    }
                }

                return lease;
            }
        }

        // No standby VMs — try to boot a new one if under limit
        if (pool_.size() < MINIVM_MAX_POOL_SIZE) {
            std::string name = "minivm_" + std::to_string(next_id_++);
            if (boot_standby_vm(name)) {
                auto& entry = pool_[name];
                entry.state = MiniVMState::LEASED;
                entry.current_module = module_name;
                entry.last_activity = std::chrono::steady_clock::now();
                ++total_leases_;

                lease.vm_name = name;
                lease.valid   = true;
                return lease;
            }
        }

        lease.error = "no VMs available (pool exhausted)";
        ++total_exhaustions_;
        return lease;
    }

    /**
     * Release a leased VM back to the pool. Recycles the overlay for reuse.
     * Returns true if the VM was successfully recycled.
     */
    bool release(const std::string& vm_name) {
        auto it = pool_.find(vm_name);
        if (it == pool_.end()) return false;

        auto& entry = it->second;

        // Unwatch from eBPF
        if (ebpf_) {
            auto* vm = sandbox_.get_vm(vm_name);
            if (vm && vm->pid > 0) {
                ebpf_->unwatch_pid(vm->pid);
            }
        }

        // Check if VM should be retired
        entry.recycle_count++;
        if (entry.recycle_count >= cfg_.max_recycles) {
            transition(entry, MiniVMState::RETIRED, "max recycles reached");
            sandbox_.destroy(vm_name);
            pool_.erase(it);
            ++total_retirements_;

            // Boot a replacement
            std::string name = "minivm_" + std::to_string(next_id_++);
            boot_standby_vm(name);
            return true;
        }

        // Recycle: reset overlay
        entry.state = MiniVMState::RECYCLING;
        entry.current_module.clear();
        entry.last_activity = std::chrono::steady_clock::now();

        bool recycled = recycle_vm(vm_name);
        if (recycled) {
            entry.state = MiniVMState::STANDBY;
            ++total_recycles_;
        } else {
            transition(entry, MiniVMState::UNHEALTHY, "recycle failed");
            ++total_unhealthy_;
        }

        return recycled;
    }

    // ── Deploy & Execute (convenience) ───────────────────────────────────

    /**
     * Full deploy-execute-collect cycle:
     *   acquire → inject code → wait completion → release
     */
    DeploymentResult deploy_and_execute(const std::string& module_name,
                                         const std::string& source_code) {
        DeploymentResult result;
        result.module_name = module_name;

        auto lease = acquire(module_name);
        if (!lease.valid) {
            result.error = lease.error;
            return result;
        }

        result.vm_name = lease.vm_name;
        auto& entry = pool_[lease.vm_name];
        result.recycle_count = entry.recycle_count;

        // Mark executing
        entry.state = MiniVMState::EXECUTING;
        entry.last_activity = std::chrono::steady_clock::now();

        auto exec_start = std::chrono::steady_clock::now();
        result.boot_latency_s = std::chrono::duration<double>(
            exec_start - lease.lease_time).count();

        // Inject code and wait for completion
        sandbox_.inject_code(lease.vm_name, source_code);
        auto exec_result = sandbox_.wait_completion(lease.vm_name,
                                                     cfg_.exec_timeout_ms);

        auto exec_end = std::chrono::steady_clock::now();
        result.exec_latency_s = std::chrono::duration<double>(
            exec_end - exec_start).count();
        result.total_latency_s = std::chrono::duration<double>(
            exec_end - lease.lease_time).count();

        result.success     = exec_result.success;
        result.stdout_data = exec_result.stdout_data;
        if (!exec_result.error.empty())
            result.error = exec_result.error;

        // Release back to pool
        release(lease.vm_name);
        ++total_deploys_;

        return result;
    }

    // ── Health monitoring ────────────────────────────────────────────────

    /**
     * Check all VMs in the pool for health issues.
     * Returns number of unhealthy VMs detected.
     */
    uint32_t health_check() {
        uint32_t unhealthy = 0;
        auto now = std::chrono::steady_clock::now();

        std::vector<std::string> to_replace;

        for (auto& [name, entry] : pool_) {
            if (entry.state == MiniVMState::UNHEALTHY ||
                entry.state == MiniVMState::RETIRED) {
                continue;
            }

            // Check: leased/executing VM stuck for too long?
            if (entry.state == MiniVMState::LEASED ||
                entry.state == MiniVMState::EXECUTING) {
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    now - entry.last_activity).count();
                if (elapsed > static_cast<int64_t>(cfg_.exec_timeout_ms / 1000 + 30)) {
                    transition(entry, MiniVMState::UNHEALTHY, "stuck: " +
                               std::to_string(elapsed) + "s since last activity");
                    to_replace.push_back(name);
                    ++unhealthy;
                    continue;
                }
            }

            // Check: underlying KVM VM in bad state?
            auto* vm = sandbox_.get_vm(name);
            if (vm && vm->state == VMState::FAILED) {
                transition(entry, MiniVMState::UNHEALTHY, "KVM VM failed");
                to_replace.push_back(name);
                ++unhealthy;
            }
        }

        // Replace unhealthy VMs
        for (const auto& name : to_replace) {
            sandbox_.destroy(name);
            pool_.erase(name);
            ++total_unhealthy_;

            std::string new_name = "minivm_" + std::to_string(next_id_++);
            boot_standby_vm(new_name);
        }

        return unhealthy;
    }

    // ── Query ────────────────────────────────────────────────────────────

    size_t  pool_size()     const { return pool_.size(); }
    size_t  standby_count() const {
        size_t n = 0;
        for (const auto& [_, e] : pool_)
            if (e.state == MiniVMState::STANDBY) ++n;
        return n;
    }
    size_t  leased_count()  const {
        size_t n = 0;
        for (const auto& [_, e] : pool_)
            if (e.state == MiniVMState::LEASED ||
                e.state == MiniVMState::EXECUTING) ++n;
        return n;
    }

    uint64_t total_leases()      const { return total_leases_; }
    uint64_t total_recycles()    const { return total_recycles_; }
    uint64_t total_deploys()     const { return total_deploys_; }
    uint64_t total_retirements() const { return total_retirements_; }
    uint64_t total_exhaustions() const { return total_exhaustions_; }
    uint64_t total_unhealthy()   const { return total_unhealthy_; }

    const std::unordered_map<std::string, MiniVMEntry>& pool() const {
        return pool_;
    }

    const MiniVMConfig& config() const { return cfg_; }

private:
    MiniVMConfig  cfg_;
    KvmSandbox&   sandbox_;
    EbpfMonitor*  ebpf_{nullptr};
    HealthCallback on_health_;

    std::unordered_map<std::string, MiniVMEntry> pool_;
    uint32_t next_id_{0};

    // Stats
    std::atomic<uint64_t> total_leases_{0};
    std::atomic<uint64_t> total_recycles_{0};
    std::atomic<uint64_t> total_deploys_{0};
    std::atomic<uint64_t> total_retirements_{0};
    std::atomic<uint64_t> total_exhaustions_{0};
    std::atomic<uint64_t> total_unhealthy_{0};

    // ── Internal helpers ─────────────────────────────────────────────────

    bool boot_standby_vm(const std::string& name) {
        bool created = sandbox_.create_vm(name);
        if (!created) return false;

        if (!sandbox_.boot(name)) return false;

        MiniVMEntry entry;
        entry.vm_name      = name;
        entry.state        = MiniVMState::STANDBY;
        entry.boot_time    = std::chrono::steady_clock::now();
        entry.last_activity = entry.boot_time;
        pool_[name] = std::move(entry);

        return true;
    }

    bool recycle_vm(const std::string& vm_name) {
        // Complete the current VM execution (if any)
        auto* vm = sandbox_.get_vm(vm_name);
        if (!vm) return false;

        // In KVM mode: reset overlay (destroy + recreate without full reboot)
        // In non-KVM mode: just reset state
        if (vm->state == VMState::COMPLETED || vm->state == VMState::FAILED) {
            // Mark as CREATED, then re-boot
            vm->state = VMState::CREATED;
            vm->exit_code = -1;
            vm->stdout_capture.clear();
            vm->pid = -1;

            // Reset overlay (KVM mode will recreate qcow2)
#ifdef NIKOLA_ENABLE_KVM
            sandbox_.remove_overlay(*vm);
            sandbox_.create_overlay(*vm);
#endif
            return sandbox_.boot(vm_name);
        }

        // Already in a good state
        return true;
    }

    void transition(MiniVMEntry& entry, MiniVMState new_state,
                     const std::string& reason) {
        MiniVMState old_state = entry.state;
        entry.state = new_state;
        if (on_health_) {
            on_health_(entry.vm_name, old_state, new_state, reason);
        }
    }
};

} // namespace nikola::security
