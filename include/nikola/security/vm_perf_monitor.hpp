/**
 * @file security/vm_perf_monitor.hpp
 * @brief Gap 7.5 — VMPerformanceMonitor
 *
 * Agentless cgroup-based performance monitoring for KVM sandbox VMs.
 * Reads metrics from the host's cgroup filesystem — does NOT trust
 * the VM to report its own resource consumption.
 *
 * Supports both cgroup v1 (/sys/fs/cgroup/{cpu,memory,blkio}/...)
 * and cgroup v2 (/sys/fs/cgroup/<scope>/cpu.stat, memory.current, etc.)
 *
 * Resource limits (from spec):
 *   MAX CPU   : 1 vCPU  (1,000,000,000 ns / sec)
 *   MAX MEMORY: 512 MB
 *   MAX I/O   : 1 MB/s
 *
 * Falls back gracefully when cgroup paths don't exist (non-KVM hosts)
 * — returns zeroed stats rather than throwing.
 */
#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>

namespace nikola::security {

// ============================================================================
// Constants
// ============================================================================

inline constexpr char    CGROUP_BASE[] = "/sys/fs/cgroup";
inline constexpr uint64_t MAX_CPU_NS_PER_SEC   = 1'000'000'000ull; // 1 vCPU
inline constexpr uint64_t MAX_MEMORY_BYTES     = 512ull * 1024 * 1024; // 512 MB
inline constexpr uint64_t MAX_IO_BYTES_PER_SEC = 1024ull * 1024;       // 1 MB/s

// ============================================================================
// Gap 7.5 — VMPerformanceMonitor
// ============================================================================

struct VMStats {
    uint64_t cpu_usage_ns{0};         ///< Cumulative CPU ns used by VM
    uint64_t memory_usage_bytes{0};   ///< Current RSS bytes
    uint64_t io_read_bytes{0};        ///< Cumulative block I/O read bytes
    uint64_t io_write_bytes{0};       ///< Cumulative block I/O write bytes
    bool     available{false};        ///< False if cgroup paths not found
};

struct ResourceViolation {
    std::string resource;    // "memory", "cpu", "io"
    uint64_t    value{0};
    uint64_t    limit{0};
};

/**
 * Reads per-VM cgroup metrics from the host.
 * Instantiate one per VM; call collect_stats() repeatedly.
 *
 * Cgroup naming convention: "nikola_vm_<vm_name>"
 */
class VMPerformanceMonitor {
public:
    explicit VMPerformanceMonitor(const std::string& vm_name,
                                   std::string cgroup_base = CGROUP_BASE)
        : vm_name_(vm_name)
        , cgroup_base_(std::move(cgroup_base))
        , cgroup_scope_("nikola_vm_" + vm_name)
    {}

    /**
     * Collect current resource metrics from cgroup filesystem.
     * Tries cgroup v2 first, then cgroup v1.
     * Returns zeroed VMStats with available=false if paths don't exist.
     */
    VMStats collect_stats() const {
        VMStats stats;

        // Try cgroup v2 unified hierarchy
        if (try_collect_v2(stats)) {
            stats.available = true;
            return stats;
        }

        // Fall back to cgroup v1
        if (try_collect_v1(stats)) {
            stats.available = true;
            return stats;
        }

        // No cgroup paths — non-KVM host or VM not yet started
        stats.available = false;
        return stats;
    }

    /**
     * Check resource limits. Returns list of violations (empty = OK).
     */
    std::vector<ResourceViolation> check_limits(const VMStats& s) const {
        std::vector<ResourceViolation> viols;

        if (s.memory_usage_bytes > MAX_MEMORY_BYTES)
            viols.push_back({"memory", s.memory_usage_bytes, MAX_MEMORY_BYTES});

        // CPU and I/O are enforced by cgroup settings; we just report here
        // (include rate-check stubs that can be enabled if a previous sample exists)
        if (prev_stats_) {
            const uint64_t io_delta =
                (s.io_read_bytes  - prev_stats_->io_read_bytes) +
                (s.io_write_bytes - prev_stats_->io_write_bytes);
            if (io_delta > MAX_IO_BYTES_PER_SEC)
                viols.push_back({"io", io_delta, MAX_IO_BYTES_PER_SEC});
        }

        return viols;
    }

    /**
     * Collect stats + check limits in one call.
     * Stores previous sample for rate calculations.
     */
    std::vector<ResourceViolation> tick() {
        const VMStats s = collect_stats();
        auto viols = check_limits(s);
        prev_stats_ = s;
        return viols;
    }

    const std::string& vm_name()       const { return vm_name_; }
    const std::string& cgroup_scope()  const { return cgroup_scope_; }

    // ── Direct cgroup path queries (useful for testing) ──────────────────────

    std::string v2_memory_path() const {
        return cgroup_base_ + "/" + cgroup_scope_ + "/memory.current";
    }
    std::string v2_cpu_path() const {
        return cgroup_base_ + "/" + cgroup_scope_ + "/cpu.stat";
    }
    std::string v1_memory_path() const {
        return cgroup_base_ + "/memory/nikola_vm/" + cgroup_scope_
             + "/memory.usage_in_bytes";
    }
    std::string v1_cpu_path() const {
        return cgroup_base_ + "/cpu/nikola_vm/" + cgroup_scope_
             + "/cpuacct.usage";
    }

private:
    std::string vm_name_;
    std::string cgroup_base_;
    std::string cgroup_scope_;
    mutable std::optional<VMStats> prev_stats_;

    // ── cgroup v2 ─────────────────────────────────────────────────────────────

    bool try_collect_v2(VMStats& s) const {
        const std::string scope_dir = cgroup_base_ + "/" + cgroup_scope_;
        if (!std::filesystem::exists(scope_dir)) return false;

        // memory.current
        s.memory_usage_bytes = read_u64(scope_dir + "/memory.current").value_or(0);

        // cpu.stat: usage_usec line
        {
            auto content = read_file(scope_dir + "/cpu.stat");
            if (content) {
                std::istringstream iss(*content);
                std::string key;
                uint64_t val;
                while (iss >> key >> val) {
                    if (key == "usage_usec") {
                        s.cpu_usage_ns = val * 1000; // µs → ns
                        break;
                    }
                }
            }
        }

        // io.stat: sum read/write bytes across all devices
        {
            auto content = read_file(scope_dir + "/io.stat");
            if (content) parse_io_stat_v2(*content, s);
        }

        return true;
    }

    // ── cgroup v1 ─────────────────────────────────────────────────────────────

    bool try_collect_v1(VMStats& s) const {
        const std::string mem_path = v1_memory_path();
        const std::string cpu_path = v1_cpu_path();

        bool any = false;
        if (auto v = read_u64(mem_path)) { s.memory_usage_bytes = *v; any = true; }
        if (auto v = read_u64(cpu_path)) { s.cpu_usage_ns        = *v; any = true; }

        // blkio.throttle.io_service_bytes
        const std::string blkio_path =
            cgroup_base_ + "/blkio/nikola_vm/" + cgroup_scope_
            + "/blkio.throttle.io_service_bytes";
        if (auto content = read_file(blkio_path)) {
            parse_io_stat_v1(*content, s);
            any = true;
        }

        return any;
    }

    // ── Parsing helpers ───────────────────────────────────────────────────────

    static void parse_io_stat_v2(const std::string& data, VMStats& s) {
        // Format: "8:0 rbytes=123 wbytes=456 rios=7 wios=8 ..."
        std::istringstream iss(data);
        std::string token;
        while (iss >> token) {
            if (token.find("rbytes=") == 0)
                s.io_read_bytes  += std::stoull(token.substr(7));
            else if (token.find("wbytes=") == 0)
                s.io_write_bytes += std::stoull(token.substr(7));
        }
    }

    static void parse_io_stat_v1(const std::string& data, VMStats& s) {
        // Format: "8:0 Read 123456\n8:0 Write 654321\nTotal ...\n"
        std::istringstream iss(data);
        std::string line;
        while (std::getline(iss, line)) {
            uint64_t val = 0;
            if (line.find("Read") != std::string::npos) {
                std::istringstream ls(line);
                std::string dev, op; ls >> dev >> op >> val;
                s.io_read_bytes += val;
            } else if (line.find("Write") != std::string::npos) {
                std::istringstream ls(line);
                std::string dev, op; ls >> dev >> op >> val;
                s.io_write_bytes += val;
            }
        }
    }

    // ── File I/O ──────────────────────────────────────────────────────────────

    static std::optional<uint64_t> read_u64(const std::string& path) {
        std::ifstream f(path);
        if (!f) return std::nullopt;
        uint64_t v = 0;
        if (!(f >> v)) return std::nullopt;
        return v;
    }

    static std::optional<std::string> read_file(const std::string& path) {
        std::ifstream f(path);
        if (!f) return std::nullopt;
        std::ostringstream oss;
        oss << f.rdbuf();
        return oss.str();
    }
};

} // namespace nikola::security
