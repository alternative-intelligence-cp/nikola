/**
 * @file security/security_engine.hpp
 * @brief Phase 7 integration facade: combines all security & execution components.
 *
 * SecurityEngine owns and drives:
 *   - VMImageManager     (Gap 7.1 — Alpine gold.qcow2 SHA-256 verification)
 *   - InterVMChannel     (Gap 7.2 — host-mediated inter-VM routing)
 *   - EscapeDetector     (Gap 7.3 — process watchdog + eBPF escape detection)
 *   - CodePatternBlacklist (Gap 7.4 — static analysis of generated source)
 *   - VMPerformanceMonitor (Gap 7.5 — agentless cgroup metric collection)
 *
 * Designed for header-only use (no separate .cpp) behind a Pimpl guard
 * for ABI stability — same pattern as AutonomyEngine and MultimodalEngine.
 *
 * Public API:
 *   engine.verify_image()                  → ImageVerifyResult
 *   engine.scan_code(source)               → ScanResult
 *   engine.route_message(msg)              → VMMessageResult
 *   engine.poll_escape()                   → alert count
 *   engine.collect_vm_stats(name)          → VMStats
 *
 * Impl (behind NIKOLA_SECURITY_ENGINE_IMPL):
 *   Full implementation includes all 5 gap headers.
 */
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <nikola/security/vm_image_manager.hpp>
#include <nikola/security/inter_vm_channel.hpp>
#include <nikola/security/escape_detector.hpp>
#include <nikola/security/code_blacklist.hpp>
#include <nikola/security/vm_perf_monitor.hpp>

namespace nikola::security {

// ============================================================================
// Config
// ============================================================================

struct SecurityConfig {
    VMImageManager::Config image_cfg;
    EscapeDetector::Config escape_cfg;
    bool strict_code_check = true;
};

// ============================================================================
// Snapshot (telemetry)
// ============================================================================

struct SecuritySnapshot {
    uint64_t code_scans{0};
    uint64_t code_rejections{0};
    uint64_t messages_routed{0};
    uint64_t messages_blocked{0};
    uint64_t escape_alerts{0};
    bool     image_verified{false};
};

// ============================================================================
// Forward declaration for Pimpl
// ============================================================================

struct SecurityEngineImpl;

// ============================================================================
// SecurityEngine
// ============================================================================

class SecurityEngine {
public:
    explicit SecurityEngine(SecurityConfig cfg = {});
    ~SecurityEngine();

    SecurityEngine(const SecurityEngine&)            = delete;
    SecurityEngine& operator=(const SecurityEngine&) = delete;
    SecurityEngine(SecurityEngine&&)                 noexcept;
    SecurityEngine& operator=(SecurityEngine&&)      noexcept;

    // ── Gap 7.1 ──────────────────────────────────────────────────────────────

    ImageVerifyResult verify_image();
    void set_expected_image_hex(const std::string& hex);

    // ── Gap 7.2 ──────────────────────────────────────────────────────────────

    void register_vm(const std::string& name, int fd = -1, int pid = -1);
    void unregister_vm(const std::string& name);
    VMMessageResult route_message(const VMMessage& msg);
    void allow_vm_pair(const std::string& from, const std::string& to);

    // ── Gap 7.3 ──────────────────────────────────────────────────────────────

    void watch_vm_escape(const std::string& name, int pid);
    size_t poll_escape();
    const std::vector<EscapeEvent>& escape_events() const;
    void inject_escape_event(const std::string& vm, EscapeType type);

    // ── Gap 7.4 ──────────────────────────────────────────────────────────────

    ScanResult scan_code(const std::string& source);
    bool is_code_safe(const std::string& source);

    // ── Gap 7.5 ──────────────────────────────────────────────────────────────

    VMStats collect_vm_stats(const std::string& vm_name);
    std::vector<ResourceViolation> tick_vm_monitor(const std::string& vm_name);

    // ── Telemetry ─────────────────────────────────────────────────────────────

    SecuritySnapshot snapshot() const;

private:
    std::unique_ptr<SecurityEngineImpl> impl_;
};

// ============================================================================
// Implementation (compiled only with NIKOLA_SECURITY_ENGINE_IMPL)
// ============================================================================

#ifdef NIKOLA_SECURITY_ENGINE_IMPL

struct SecurityEngineImpl {
    SecurityConfig       cfg;
    VMImageManager       image_mgr;
    InterVMChannel       ivm_channel;
    EscapeDetector       escape_det;
    CodePatternBlacklist blacklist;
    SecuritySnapshot     snap;

    // Per-VM monitors live in a map (instantiated on demand)
    std::unordered_map<std::string, VMPerformanceMonitor> perf_monitors;

    explicit SecurityEngineImpl(SecurityConfig c)
        : cfg(std::move(c))
        , image_mgr(cfg.image_cfg)
        , escape_det(cfg.escape_cfg)
    {}
};

// ── Constructor / Destructor ─────────────────────────────────────────────────

SecurityEngine::SecurityEngine(SecurityConfig cfg)
    : impl_(std::make_unique<SecurityEngineImpl>(std::move(cfg)))
{}

SecurityEngine::~SecurityEngine() = default;
SecurityEngine::SecurityEngine(SecurityEngine&&) noexcept = default;
SecurityEngine& SecurityEngine::operator=(SecurityEngine&&) noexcept = default;

// ── Gap 7.1 — VMImageManager ─────────────────────────────────────────────────

ImageVerifyResult SecurityEngine::verify_image() {
    auto r = impl_->image_mgr.verify_integrity();
    impl_->snap.image_verified = r.ok;
    return r;
}

void SecurityEngine::set_expected_image_hex(const std::string& hex) {
    impl_->image_mgr.set_expected_hex(hex);
}

// ── Gap 7.2 — InterVMChannel ─────────────────────────────────────────────────

void SecurityEngine::register_vm(const std::string& name, int fd, int pid) {
    impl_->ivm_channel.register_vm(name, fd, pid);
}

void SecurityEngine::unregister_vm(const std::string& name) {
    impl_->ivm_channel.unregister_vm(name);
}

VMMessageResult SecurityEngine::route_message(const VMMessage& msg) {
    auto r = impl_->ivm_channel.route(msg);
    if (r.status == VMMessageStatus::DELIVERED) ++impl_->snap.messages_routed;
    else                                         ++impl_->snap.messages_blocked;
    return r;
}

void SecurityEngine::allow_vm_pair(const std::string& from, const std::string& to) {
    impl_->ivm_channel.allow(from, to);
}

// ── Gap 7.3 — EscapeDetector ─────────────────────────────────────────────────

void SecurityEngine::watch_vm_escape(const std::string& name, int pid) {
    impl_->escape_det.watch_vm(name, pid);
}

size_t SecurityEngine::poll_escape() {
    size_t n = impl_->escape_det.poll();
    impl_->snap.escape_alerts += n;
    return n;
}

const std::vector<EscapeEvent>& SecurityEngine::escape_events() const {
    return impl_->escape_det.events();
}

void SecurityEngine::inject_escape_event(const std::string& vm, EscapeType type) {
    impl_->escape_det.inject_event(vm, type);
    ++impl_->snap.escape_alerts;
}

// ── Gap 7.4 — CodePatternBlacklist ───────────────────────────────────────────

ScanResult SecurityEngine::scan_code(const std::string& source) {
    ++impl_->snap.code_scans;
    auto r = impl_->blacklist.check(source);
    if (!r.safe) ++impl_->snap.code_rejections;
    return r;
}

bool SecurityEngine::is_code_safe(const std::string& source) {
    return scan_code(source).safe;
}

// ── Gap 7.5 — VMPerformanceMonitor ───────────────────────────────────────────

VMStats SecurityEngine::collect_vm_stats(const std::string& vm_name) {
    auto it = impl_->perf_monitors.find(vm_name);
    if (it == impl_->perf_monitors.end()) {
        impl_->perf_monitors.emplace(vm_name,
            VMPerformanceMonitor(vm_name));
        it = impl_->perf_monitors.find(vm_name);
    }
    return it->second.collect_stats();
}

std::vector<ResourceViolation>
SecurityEngine::tick_vm_monitor(const std::string& vm_name) {
    auto it = impl_->perf_monitors.find(vm_name);
    if (it == impl_->perf_monitors.end()) {
        impl_->perf_monitors.emplace(vm_name,
            VMPerformanceMonitor(vm_name));
        it = impl_->perf_monitors.find(vm_name);
    }
    return it->second.tick();
}

// ── Telemetry ─────────────────────────────────────────────────────────────────

SecuritySnapshot SecurityEngine::snapshot() const { return impl_->snap; }

#endif // NIKOLA_SECURITY_ENGINE_IMPL

} // namespace nikola::security
