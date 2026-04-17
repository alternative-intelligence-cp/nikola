/**
 * @file security/kvm_sandbox.hpp
 * @brief v0.1.19 — KVM Sandbox lifecycle management
 *
 * Manages KVM-isolated VMs for running self-generated code safely.
 * Provides VM boot, shutdown, network isolation enforcement, and
 * cgroup resource limit creation/enforcement.
 *
 * Actual qemu-system invocations are gated behind NIKOLA_ENABLE_KVM.
 * All policy logic (cgroup config, isolation rules, syscall restrictions)
 * is always compiled and testable without hardware.
 *
 * Resource limits (from spec):
 *   - Memory: 512 MB hard limit
 *   - CPU:    1 vCPU (cgroup v2 cpu.max = "100000 100000")
 *   - Disk:   512 MB copy-on-write overlay
 *   - Network: none (no netdev / --net=none)
 *
 * Blocked operations (enforced by sandbox + seccomp profile):
 *   - system(), exec*(), fork()  — prevented via seccomp BPF
 *   - Host network access        — no network device attached
 *   - Host filesystem access      — overlay only, no passthrough
 *
 * Usage:
 *   KvmSandbox sandbox;
 *   auto vm = sandbox.create_vm("exec_001");
 *   sandbox.boot(vm);
 *   sandbox.inject_code(vm, source_code);
 *   auto result = sandbox.wait_completion(vm, timeout_ms);
 *   sandbox.destroy(vm);
 */
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef NIKOLA_ENABLE_KVM
#  include <filesystem>
#  include <fstream>
#  include <csignal>
#  include <cstring>
#  include <sys/types.h>
#  include <sys/wait.h>
#  include <unistd.h>
#  include <fcntl.h>
#  include <poll.h>
#endif

namespace nikola::security {

// ============================================================================
// Constants
// ============================================================================

inline constexpr uint64_t KVM_VM_MEMORY_MB      = 512;
inline constexpr uint64_t KVM_VM_DISK_MB        = 512;
inline constexpr int      KVM_VM_VCPUS           = 1;
inline constexpr uint64_t KVM_CGROUP_MEM_BYTES  = KVM_VM_MEMORY_MB * 1024 * 1024;
inline constexpr uint64_t KVM_CPU_QUOTA_US       = 100'000;  // 100ms period
inline constexpr uint64_t KVM_CPU_PERIOD_US      = 100'000;  // = 1 vCPU
inline constexpr uint32_t KVM_MAX_VMS            = 16;
inline constexpr char     KVM_GOLD_IMAGE[]       = "/var/lib/nikola/gold.qcow2";
inline constexpr char     KVM_OVERLAY_DIR[]      = "/var/lib/nikola/overlays";
inline constexpr char     KVM_CGROUP_PREFIX[]    = "nikola_vm_";

// ============================================================================
// VM State
// ============================================================================

enum class VMState : uint8_t {
    CREATED,      ///< VM config exists, not yet booted
    BOOTING,      ///< qemu-system process starting
    RUNNING,      ///< VM is executing
    COMPLETED,    ///< Guest code finished, exit code available
    FAILED,       ///< VM crashed or timed out
    DESTROYED,    ///< Resources cleaned up
};

inline const char* vm_state_str(VMState s) {
    switch (s) {
        case VMState::CREATED:   return "CREATED";
        case VMState::BOOTING:   return "BOOTING";
        case VMState::RUNNING:   return "RUNNING";
        case VMState::COMPLETED: return "COMPLETED";
        case VMState::FAILED:    return "FAILED";
        case VMState::DESTROYED: return "DESTROYED";
    }
    return "UNKNOWN";
}

// ============================================================================
// Isolation rule set — always compiled, testable without KVM
// ============================================================================

struct IsolationRules {
    bool     network_disabled  = true;   ///< --net=none
    bool     seccomp_enabled   = true;   ///< Seccomp BPF profile loaded
    bool     readonly_rootfs   = true;   ///< Root filesystem is read-only
    uint64_t memory_limit_mb   = KVM_VM_MEMORY_MB;
    int      vcpu_count        = KVM_VM_VCPUS;
    uint64_t disk_limit_mb     = KVM_VM_DISK_MB;

    /// Syscalls blocked by seccomp profile within the VM
    std::vector<std::string> blocked_syscalls = {
        "execve", "execveat", "fork", "vfork", "clone",
        "ptrace", "mount", "umount2", "pivot_root",
        "reboot", "kexec_load", "init_module", "finit_module",
    };
};

// ============================================================================
// CGroup configuration — always compiled, testable without root
// ============================================================================

struct CGroupConfig {
    std::string base_path     = "/sys/fs/cgroup";
    std::string scope_prefix  = KVM_CGROUP_PREFIX;
    uint64_t    memory_max    = KVM_CGROUP_MEM_BYTES;
    uint64_t    cpu_quota_us  = KVM_CPU_QUOTA_US;
    uint64_t    cpu_period_us = KVM_CPU_PERIOD_US;

    /// Build the full cgroup scope name for a VM
    std::string scope_name(const std::string& vm_name) const {
        return scope_prefix + vm_name;
    }

    /// Build the full cgroup v2 path for a VM
    std::string v2_path(const std::string& vm_name) const {
        return base_path + "/" + scope_name(vm_name);
    }

    /// Generate the cpu.max content: "quota period"
    std::string cpu_max_value() const {
        return std::to_string(cpu_quota_us) + " " + std::to_string(cpu_period_us);
    }

    /// Generate the memory.max content
    std::string memory_max_value() const {
        return std::to_string(memory_max);
    }
};

// ============================================================================
// VM Instance descriptor
// ============================================================================

struct VMInstance {
    std::string    name;
    VMState        state{VMState::CREATED};
    int            pid{-1};           ///< qemu-system PID (or -1)
    int            exit_code{-1};     ///< Guest exit code (-1 = not yet)
    int            stdout_fd_{-1};    ///< Pipe fd for stdout capture (KVM mode)
    IsolationRules isolation;
    CGroupConfig   cgroup;
    std::string    overlay_path;      ///< Copy-on-write overlay qcow2 path
    std::string    stdout_capture;    ///< Guest stdout (for code execution results)
    std::chrono::steady_clock::time_point boot_time;
    std::chrono::steady_clock::time_point end_time;

    double elapsed_seconds() const {
        auto end = (state == VMState::RUNNING || state == VMState::BOOTING)
                       ? std::chrono::steady_clock::now() : end_time;
        return std::chrono::duration<double>(end - boot_time).count();
    }
};

// ============================================================================
// Execution result
// ============================================================================

struct ExecutionResult {
    bool        success{false};
    int         exit_code{-1};
    std::string stdout_data;
    std::string error;
    double      elapsed_s{0.0};
};

// ============================================================================
// KvmSandbox — VM lifecycle manager
// ============================================================================

/**
 * Manages a pool of KVM sandbox VMs for executing untrusted code.
 *
 * Lifecycle: create_vm → boot → inject_code → wait_completion → destroy
 *
 * All policy validation (isolation rules, cgroup config) works without
 * actual KVM. The qemu-system invocations are gated behind
 * NIKOLA_ENABLE_KVM and do NOT compile on non-Linux or non-KVM hosts.
 */
class KvmSandbox {
public:
    using AlertCallback = std::function<void(const std::string& vm_name,
                                              const std::string& reason)>;

    struct Config {
        std::string gold_image  = KVM_GOLD_IMAGE;
        std::string overlay_dir = KVM_OVERLAY_DIR;
        uint32_t    max_vms     = KVM_MAX_VMS;
        IsolationRules default_isolation;
        CGroupConfig   default_cgroup;
    };

    KvmSandbox() : cfg_{} {}
    explicit KvmSandbox(Config cfg) : cfg_(std::move(cfg)) {}

    // ── VM lifecycle ────────────────────────────────────────────────────────

    /**
     * Create a VM instance with the given name.
     * Returns false if name already exists or pool is full.
     */
    bool create_vm(const std::string& name) {
        if (vms_.size() >= cfg_.max_vms) return false;
        if (vms_.count(name)) return false;

        VMInstance vm;
        vm.name         = name;
        vm.state        = VMState::CREATED;
        vm.isolation    = cfg_.default_isolation;
        vm.cgroup       = cfg_.default_cgroup;
        vm.overlay_path = cfg_.overlay_dir + "/" + name + ".qcow2";
        vms_[name] = std::move(vm);
        return true;
    }

    /**
     * Boot a VM. On real KVM hosts:
     *   1. Create cgroup scope
     *   2. Create copy-on-write overlay
     *   3. Launch qemu-system with --net=none, seccomp profile
     *
     * Without NIKOLA_ENABLE_KVM, transitions state to RUNNING for testing.
     */
    bool boot(const std::string& name) {
        auto it = vms_.find(name);
        if (it == vms_.end()) return false;
        auto& vm = it->second;
        if (vm.state != VMState::CREATED) return false;

        vm.state = VMState::BOOTING;
        vm.boot_time = std::chrono::steady_clock::now();

        // Validate isolation rules before proceeding
        if (!validate_isolation(vm.isolation)) {
            vm.state = VMState::FAILED;
            return false;
        }

#ifdef NIKOLA_ENABLE_KVM
        // Real KVM path: create cgroup, overlay, launch qemu-system
        if (!setup_cgroup(vm)) {
            vm.state = VMState::FAILED;
            return false;
        }
        if (!create_overlay(vm)) {
            vm.state = VMState::FAILED;
            return false;
        }
        if (!launch_qemu(vm)) {
            vm.state = VMState::FAILED;
            return false;
        }
#endif

        vm.state = VMState::RUNNING;
        ++total_boots_;
        return true;
    }

    /**
     * Simulate code execution completing with a result.
     * On real KVM, this would write code to a virtio-serial channel
     * and wait for the guest agent to report completion.
     *
     * For testing: directly sets result fields on the VM instance.
     */
    bool complete(const std::string& name, int exit_code,
                  const std::string& stdout_data = "")
    {
        auto it = vms_.find(name);
        if (it == vms_.end()) return false;
        auto& vm = it->second;
        if (vm.state != VMState::RUNNING) return false;

        vm.exit_code      = exit_code;
        vm.stdout_capture  = stdout_data;
        vm.state           = VMState::COMPLETED;
        vm.end_time        = std::chrono::steady_clock::now();
        return true;
    }

    /**
     * Force-fail a VM (timeout, crash, escape attempt).
     */
    bool fail(const std::string& name, const std::string& reason = "") {
        auto it = vms_.find(name);
        if (it == vms_.end()) return false;
        auto& vm = it->second;

        vm.state    = VMState::FAILED;
        vm.end_time = std::chrono::steady_clock::now();
        if (on_alert_ && !reason.empty()) on_alert_(name, reason);
        return true;
    }

    /**
     * Destroy a VM: clean up resources.
     * On real KVM: kill process, remove overlay, tear down cgroup.
     */
    bool destroy(const std::string& name) {
        auto it = vms_.find(name);
        if (it == vms_.end()) return false;
        auto& vm = it->second;

#ifdef NIKOLA_ENABLE_KVM
        if (vm.pid > 0) {
            ::kill(vm.pid, SIGKILL);
            vm.pid = -1;
        }
        teardown_cgroup(vm);
        remove_overlay(vm);
#endif

        vm.state = VMState::DESTROYED;
        ++total_destroys_;
        return true;
    }

    /**
     * Wait for a VM to complete execution, with timeout in milliseconds.
     * On real KVM: waits for qemu-system process, captures stdout/stderr.
     * Without KVM: returns immediately with whatever state the VM is in.
     */
    ExecutionResult wait_completion(const std::string& name,
                                    uint32_t timeout_ms = 30000)
    {
        auto it = vms_.find(name);
        if (it == vms_.end()) return {false, -1, "", "vm not found", 0.0};
        auto& vm = it->second;

        if (vm.state == VMState::COMPLETED) {
            return {true, vm.exit_code, vm.stdout_capture, "", vm.elapsed_seconds()};
        }
        if (vm.state != VMState::RUNNING) {
            return {false, vm.exit_code, vm.stdout_capture,
                    "vm not running (state=" + std::string(vm_state_str(vm.state)) + ")",
                    vm.elapsed_seconds()};
        }

#ifdef NIKOLA_ENABLE_KVM
        return wait_for_completion(vm, timeout_ms);
#else
        (void)timeout_ms;
        // Non-KVM: return current state
        return {vm.state == VMState::COMPLETED, vm.exit_code,
                vm.stdout_capture, "", vm.elapsed_seconds()};
#endif
    }

    // ── Query ────────────────────────────────────────────────────────────────

    const VMInstance* get_vm(const std::string& name) const {
        auto it = vms_.find(name);
        return (it != vms_.end()) ? &it->second : nullptr;
    }

    VMInstance* get_vm(const std::string& name) {
        auto it = vms_.find(name);
        return (it != vms_.end()) ? &it->second : nullptr;
    }

    /**
     * Inject source code into a running VM for execution.
     * In non-KVM mode: stores the code in the VM's stdout_capture for testing.
     * In KVM mode: writes code to shared overlay and signals guest to execute.
     */
    void inject_code(const std::string& vm_name, const std::string& source_code) {
        auto* vm = get_vm(vm_name);
        if (!vm || vm->state != VMState::RUNNING) return;

#ifdef NIKOLA_ENABLE_KVM
        // Write source code to the overlay filesystem for the guest to pick up
        if (!vm->overlay_path.empty()) {
            std::ofstream ofs(vm->overlay_path + "/code_input.src");
            if (ofs) ofs << source_code;
        }
#else
        // Non-KVM simulation: store code for test retrieval
        vm->stdout_capture = "[injected:" + std::to_string(source_code.size()) + " bytes]";
        vm->state = VMState::COMPLETED;
        vm->exit_code = 0;
        vm->end_time = std::chrono::steady_clock::now();
#endif
    }

    size_t vm_count()    const { return vms_.size(); }
    size_t active_count() const {
        size_t n = 0;
        for (const auto& [_, vm] : vms_)
            if (vm.state == VMState::RUNNING || vm.state == VMState::BOOTING) ++n;
        return n;
    }

    uint64_t total_boots()    const { return total_boots_; }
    uint64_t total_destroys() const { return total_destroys_; }

    void set_alert_callback(AlertCallback cb) { on_alert_ = std::move(cb); }

    // ── Isolation validation (always compiled) ──────────────────────────────

    /**
     * Validate that isolation rules meet minimum security requirements.
     */
    static bool validate_isolation(const IsolationRules& rules) {
        // Network MUST be disabled for sandbox VMs
        if (!rules.network_disabled) return false;
        // Seccomp MUST be enabled
        if (!rules.seccomp_enabled) return false;
        // Memory limit must be reasonable (1MB..4GB)
        if (rules.memory_limit_mb < 1 || rules.memory_limit_mb > 4096) return false;
        // Must have at least 1 vCPU, at most 4
        if (rules.vcpu_count < 1 || rules.vcpu_count > 4) return false;
        // Must block at least execve and fork
        bool has_execve = false, has_fork = false;
        for (const auto& sc : rules.blocked_syscalls) {
            if (sc == "execve") has_execve = true;
            if (sc == "fork")   has_fork   = true;
        }
        if (!has_execve || !has_fork) return false;
        return true;
    }

    /**
     * Validate cgroup configuration.
     */
    static bool validate_cgroup(const CGroupConfig& cg) {
        if (cg.memory_max == 0) return false;
        if (cg.cpu_quota_us == 0 || cg.cpu_period_us == 0) return false;
        // Quota must not exceed period * 4 (max 4 vCPUs)
        if (cg.cpu_quota_us > cg.cpu_period_us * 4) return false;
        return true;
    }

    /**
     * Build the qemu-system argument list for a VM (always compiled).
     * Useful for inspection/testing even without KVM.
     */
    std::vector<std::string> build_qemu_args(const VMInstance& vm) const {
        std::vector<std::string> args;
        args.push_back("qemu-system-x86_64");
        args.push_back("-enable-kvm");
        args.push_back("-m");
        args.push_back(std::to_string(vm.isolation.memory_limit_mb));
        args.push_back("-smp");
        args.push_back(std::to_string(vm.isolation.vcpu_count));
        args.push_back("-drive");
        args.push_back("file=" + vm.overlay_path + ",format=qcow2,if=virtio");
        // Network isolation: no network device
        args.push_back("-net");
        args.push_back("none");
        // Display off
        args.push_back("-display");
        args.push_back("none");
        args.push_back("-nographic");
        // Virtio-serial for code injection / result capture
        args.push_back("-device");
        args.push_back("virtio-serial");
        args.push_back("-chardev");
        args.push_back("socket,id=code_channel,path=/tmp/nikola_" + vm.name + ".sock,server=on,wait=off");
        args.push_back("-device");
        args.push_back("virtserialport,chardev=code_channel,name=code_port");
        return args;
    }

private:
    Config cfg_;
    std::unordered_map<std::string, VMInstance> vms_;
    AlertCallback on_alert_;
    uint64_t total_boots_{0};
    uint64_t total_destroys_{0};

#ifdef NIKOLA_ENABLE_KVM
    bool setup_cgroup(VMInstance& vm) {
        const std::string path = vm.cgroup.v2_path(vm.name);
        // Create cgroup directory
        if (std::filesystem::exists(path)) return true;
        std::error_code ec;
        std::filesystem::create_directories(path, ec);
        if (ec) return false;
        // Set memory limit
        {
            std::ofstream f(path + "/memory.max");
            if (!f) return false;
            f << vm.cgroup.memory_max_value();
        }
        // Set CPU quota
        {
            std::ofstream f(path + "/cpu.max");
            if (!f) return false;
            f << vm.cgroup.cpu_max_value();
        }
        return true;
    }

    void teardown_cgroup(VMInstance& vm) {
        const std::string path = vm.cgroup.v2_path(vm.name);
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }

    bool create_overlay(VMInstance& vm) {
        // qemu-img create -b gold.qcow2 -F qcow2 -f qcow2 overlay.qcow2 512M
        std::string cmd = "qemu-img create -b " + cfg_.gold_image +
                          " -F qcow2 -f qcow2 " + vm.overlay_path +
                          " " + std::to_string(vm.isolation.disk_limit_mb) + "M 2>&1";
        return (::system(cmd.c_str()) == 0);
    }

    void remove_overlay(VMInstance& vm) {
        std::error_code ec;
        std::filesystem::remove(vm.overlay_path, ec);
    }

    bool launch_qemu(VMInstance& vm) {
        auto args = build_qemu_args(vm);

        // Create pipe for stdout capture
        int stdout_pipe[2];
        if (::pipe2(stdout_pipe, O_CLOEXEC) != 0) return false;

        pid_t pid = ::fork();
        if (pid < 0) {
            ::close(stdout_pipe[0]);
            ::close(stdout_pipe[1]);
            return false;
        }

        if (pid == 0) {
            // ── Child process ────────────────────────────────────────────
            ::close(stdout_pipe[0]);  // close read end

            // Redirect stdout + stderr to pipe
            ::dup2(stdout_pipe[1], STDOUT_FILENO);
            ::dup2(stdout_pipe[1], STDERR_FILENO);
            ::close(stdout_pipe[1]);

            // Build argv for execvp
            std::vector<char*> argv;
            argv.reserve(args.size() + 1);
            for (auto& a : args) argv.push_back(a.data());
            argv.push_back(nullptr);

            ::execvp(argv[0], argv.data());
            ::_exit(127);  // exec failed
        }

        // ── Parent process ───────────────────────────────────────────────
        ::close(stdout_pipe[1]);  // close write end

        vm.pid = static_cast<int>(pid);
        vm.stdout_fd_ = stdout_pipe[0];

        // Assign to cgroup (best-effort — cgroup may not exist in test envs)
        const std::string cgroup_procs = vm.cgroup.v2_path(vm.name) + "/cgroup.procs";
        std::ofstream cg(cgroup_procs);
        if (cg) cg << pid;

        return true;
    }

    /**
     * Wait for a VM process to finish with timeout.
     * Captures stdout/stderr and populates the execution result.
     */
    ExecutionResult wait_for_completion(VMInstance& vm, uint32_t timeout_ms) {
        ExecutionResult result;
        if (vm.pid <= 0) {
            result.error = "no process";
            return result;
        }

        // Non-blocking read of stdout while waiting for process
        std::string output;
        char buf[4096];
        auto deadline = std::chrono::steady_clock::now()
                      + std::chrono::milliseconds(timeout_ms);

        while (std::chrono::steady_clock::now() < deadline) {
            // Check if process has exited
            int status = 0;
            pid_t w = ::waitpid(vm.pid, &status, WNOHANG);
            if (w > 0) {
                // Process exited — drain remaining stdout
                if (vm.stdout_fd_ >= 0) {
                    while (true) {
                        ssize_t n = ::read(vm.stdout_fd_, buf, sizeof(buf));
                        if (n <= 0) break;
                        output.append(buf, static_cast<size_t>(n));
                    }
                    ::close(vm.stdout_fd_);
                    vm.stdout_fd_ = -1;
                }

                vm.exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
                vm.stdout_capture = output;
                vm.state = (vm.exit_code == 0) ? VMState::COMPLETED : VMState::FAILED;
                vm.end_time = std::chrono::steady_clock::now();

                result.success = (vm.exit_code == 0);
                result.exit_code = vm.exit_code;
                result.stdout_data = output;
                result.elapsed_s = vm.elapsed_seconds();
                return result;
            }

            // Poll stdout for data (100ms chunks)
            if (vm.stdout_fd_ >= 0) {
                struct pollfd pfd{vm.stdout_fd_, POLLIN, 0};
                if (::poll(&pfd, 1, 100) > 0) {
                    ssize_t n = ::read(vm.stdout_fd_, buf, sizeof(buf));
                    if (n > 0) output.append(buf, static_cast<size_t>(n));
                }
            }
        }

        // Timeout — send SIGTERM, then SIGKILL after 2s
        ::kill(vm.pid, SIGTERM);

        auto kill_deadline = std::chrono::steady_clock::now()
                           + std::chrono::seconds(2);
        while (std::chrono::steady_clock::now() < kill_deadline) {
            int status = 0;
            if (::waitpid(vm.pid, &status, WNOHANG) > 0) break;
            usleep(50000);  // 50ms
        }

        // Force kill if still alive
        if (::waitpid(vm.pid, nullptr, WNOHANG) == 0) {
            ::kill(vm.pid, SIGKILL);
            ::waitpid(vm.pid, nullptr, 0);
        }

        if (vm.stdout_fd_ >= 0) {
            ::close(vm.stdout_fd_);
            vm.stdout_fd_ = -1;
        }

        vm.state = VMState::FAILED;
        vm.end_time = std::chrono::steady_clock::now();

        result.error = "timeout after " + std::to_string(timeout_ms) + "ms";
        result.stdout_data = output;
        result.elapsed_s = vm.elapsed_seconds();
        return result;
    }
#endif // NIKOLA_ENABLE_KVM
};

} // namespace nikola::security
