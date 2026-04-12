/**
 * @file phase156_security_hardening_test.cpp
 * @brief Phase 156 — v0.1.19 Security Hardening test suite
 *
 * Tests for:
 *   - KVM Sandbox lifecycle and isolation validation
 *   - eBPF Monitor event detection and response policy
 *   - CSVP (Code Safety Verification Protocol) 4-stage pipeline
 *   - Anomaly Detector behavioral profiling and quarantine
 *
 * 24 test cases covering all v0.1.19 security hardening components.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <nikola/security/kvm_sandbox.hpp>
#include <nikola/security/ebpf_monitor.hpp>
#include <nikola/security/csvp.hpp>
#include <nikola/security/anomaly_detector.hpp>
#include <nikola/security/code_blacklist.hpp>
#include <nikola/security/escape_detector.hpp>

#include <set>

using namespace nikola::security;

// ============================================================================
// §1  KVM Sandbox — Constants valid
// ============================================================================

TEST_CASE("§1 KVM constants valid", "[phase156][kvm]") {
    REQUIRE(KVM_VM_MEMORY_MB == 512);
    REQUIRE(KVM_VM_DISK_MB   == 512);
    REQUIRE(KVM_VM_VCPUS     == 1);
    REQUIRE(KVM_CGROUP_MEM_BYTES == 512ull * 1024 * 1024);
    REQUIRE(KVM_CPU_QUOTA_US  == 100'000);
    REQUIRE(KVM_CPU_PERIOD_US == 100'000);
    REQUIRE(KVM_MAX_VMS == 16);
}

// ============================================================================
// §2  KVM Sandbox — Default isolation rules
// ============================================================================

TEST_CASE("§2 KVM default isolation rules", "[phase156][kvm]") {
    IsolationRules rules;
    REQUIRE(rules.network_disabled == true);
    REQUIRE(rules.seccomp_enabled  == true);
    REQUIRE(rules.readonly_rootfs  == true);
    REQUIRE(rules.memory_limit_mb  == 512);
    REQUIRE(rules.vcpu_count       == 1);
    REQUIRE(rules.disk_limit_mb    == 512);
    REQUIRE(rules.blocked_syscalls.size() >= 5);  // execve, execveat, fork, vfork, clone + more

    // Must block execve and fork
    bool has_execve = false, has_fork = false;
    for (const auto& sc : rules.blocked_syscalls) {
        if (sc == "execve") has_execve = true;
        if (sc == "fork")   has_fork   = true;
    }
    REQUIRE(has_execve);
    REQUIRE(has_fork);
}

// ============================================================================
// §3  KVM Sandbox — Isolation validation
// ============================================================================

TEST_CASE("§3 KVM isolation validation", "[phase156][kvm]") {
    // Default rules should pass
    IsolationRules good;
    REQUIRE(KvmSandbox::validate_isolation(good));

    // Network enabled = FAIL
    IsolationRules bad_net = good;
    bad_net.network_disabled = false;
    REQUIRE_FALSE(KvmSandbox::validate_isolation(bad_net));

    // Seccomp disabled = FAIL
    IsolationRules bad_sec = good;
    bad_sec.seccomp_enabled = false;
    REQUIRE_FALSE(KvmSandbox::validate_isolation(bad_sec));

    // No execve in blocked list = FAIL
    IsolationRules bad_syscall = good;
    bad_syscall.blocked_syscalls = {"fork", "ptrace"};
    REQUIRE_FALSE(KvmSandbox::validate_isolation(bad_syscall));

    // Memory out of range = FAIL
    IsolationRules bad_mem = good;
    bad_mem.memory_limit_mb = 0;
    REQUIRE_FALSE(KvmSandbox::validate_isolation(bad_mem));

    // Too many vCPUs = FAIL
    IsolationRules bad_cpu = good;
    bad_cpu.vcpu_count = 8;
    REQUIRE_FALSE(KvmSandbox::validate_isolation(bad_cpu));
}

// ============================================================================
// §4  KVM Sandbox — VM lifecycle
// ============================================================================

TEST_CASE("§4 KVM VM lifecycle", "[phase156][kvm]") {
    KvmSandbox sandbox;

    // Create
    REQUIRE(sandbox.create_vm("test_vm_1"));
    REQUIRE(sandbox.vm_count() == 1);

    auto* vm = sandbox.get_vm("test_vm_1");
    REQUIRE(vm != nullptr);
    REQUIRE(vm->state == VMState::CREATED);

    // Boot
    REQUIRE(sandbox.boot("test_vm_1"));
    vm = sandbox.get_vm("test_vm_1");
    REQUIRE(vm->state == VMState::RUNNING);
    REQUIRE(sandbox.active_count() == 1);
    REQUIRE(sandbox.total_boots() == 1);

    // Complete
    REQUIRE(sandbox.complete("test_vm_1", 0, "OK"));
    vm = sandbox.get_vm("test_vm_1");
    REQUIRE(vm->state == VMState::COMPLETED);
    REQUIRE(vm->exit_code == 0);
    REQUIRE(vm->stdout_capture == "OK");
    REQUIRE(sandbox.active_count() == 0);

    // Destroy
    REQUIRE(sandbox.destroy("test_vm_1"));
    vm = sandbox.get_vm("test_vm_1");
    REQUIRE(vm->state == VMState::DESTROYED);
    REQUIRE(sandbox.total_destroys() == 1);
}

// ============================================================================
// §5  KVM Sandbox — Pool limit
// ============================================================================

TEST_CASE("§5 KVM pool limit", "[phase156][kvm]") {
    KvmSandbox::Config cfg;
    cfg.max_vms = 2;
    KvmSandbox sandbox(cfg);

    REQUIRE(sandbox.create_vm("vm1"));
    REQUIRE(sandbox.create_vm("vm2"));
    REQUIRE_FALSE(sandbox.create_vm("vm3"));  // pool full
    REQUIRE(sandbox.vm_count() == 2);

    // Duplicate name rejected
    REQUIRE_FALSE(sandbox.create_vm("vm1"));
}

// ============================================================================
// §6  KVM Sandbox — CGroup validation
// ============================================================================

TEST_CASE("§6 KVM cgroup validation", "[phase156][kvm]") {
    CGroupConfig good;
    REQUIRE(KvmSandbox::validate_cgroup(good));

    // Zero memory = FAIL
    CGroupConfig bad_mem = good;
    bad_mem.memory_max = 0;
    REQUIRE_FALSE(KvmSandbox::validate_cgroup(bad_mem));

    // Zero quota = FAIL
    CGroupConfig bad_cpu = good;
    bad_cpu.cpu_quota_us = 0;
    REQUIRE_FALSE(KvmSandbox::validate_cgroup(bad_cpu));

    // Excessive quota = FAIL (more than 4 vCPUs)
    CGroupConfig bad_excess = good;
    bad_excess.cpu_quota_us = good.cpu_period_us * 5;
    REQUIRE_FALSE(KvmSandbox::validate_cgroup(bad_excess));

    // Config values
    REQUIRE(good.scope_name("exec1") == "nikola_vm_exec1");
    REQUIRE(good.cpu_max_value() == "100000 100000");
    REQUIRE(good.memory_max_value() == std::to_string(KVM_CGROUP_MEM_BYTES));
}

// ============================================================================
// §7  KVM Sandbox — qemu args generation
// ============================================================================

TEST_CASE("§7 KVM qemu args generation", "[phase156][kvm]") {
    KvmSandbox sandbox;
    sandbox.create_vm("args_test");

    auto* vm = sandbox.get_vm("args_test");
    REQUIRE(vm != nullptr);
    auto args = sandbox.build_qemu_args(*vm);

    // Must contain critical args
    bool has_kvm = false, has_net_none = false, has_nographic = false;
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "-enable-kvm") has_kvm = true;
        if (args[i] == "-net" && i + 1 < args.size() && args[i+1] == "none")
            has_net_none = true;
        if (args[i] == "-nographic") has_nographic = true;
    }
    REQUIRE(has_kvm);
    REQUIRE(has_net_none);
    REQUIRE(has_nographic);
    REQUIRE(args.size() >= 10);
}

// ============================================================================
// §8  KVM Sandbox — VM failure and alert
// ============================================================================

TEST_CASE("§8 KVM VM failure and alert", "[phase156][kvm]") {
    KvmSandbox sandbox;
    sandbox.create_vm("fail_test");
    sandbox.boot("fail_test");

    std::string alert_name, alert_reason;
    sandbox.set_alert_callback([&](const std::string& n, const std::string& r) {
        alert_name = n;
        alert_reason = r;
    });

    REQUIRE(sandbox.fail("fail_test", "escape attempt"));
    auto* vm = sandbox.get_vm("fail_test");
    REQUIRE(vm->state == VMState::FAILED);
    REQUIRE(alert_name == "fail_test");
    REQUIRE(alert_reason == "escape attempt");
}

// ============================================================================
// §9  eBPF Monitor — Constants and defaults
// ============================================================================

TEST_CASE("§9 eBPF constants valid", "[phase156][ebpf]") {
    REQUIRE(EBPF_POLL_INTERVAL_MS == 100);
    REQUIRE(EBPF_RING_BUFFER_PAGES == 64);
    REQUIRE(EBPF_MAX_WATCHED_PIDS == 64);
    REQUIRE(EBPF_RESPONSE_DEADLINE_NS == 100'000'000ull);

    EbpfMonitor mon;
    REQUIRE(mon.fallback_mode() == true);
    REQUIRE(mon.total_events() == 0);
    REQUIRE(mon.total_kills() == 0);
}

// ============================================================================
// §10 eBPF Monitor — Event injection and response
// ============================================================================

TEST_CASE("§10 eBPF event injection and response", "[phase156][ebpf]") {
    EbpfMonitor::Config cfg;
    cfg.auto_kill = false;  // don't try to kill real processes
    EbpfMonitor mon(cfg);

    mon.watch_pid(12345, "sandbox_vm1");
    REQUIRE(mon.watched_count() == 1);

    // Inject execve attempt
    mon.inject_event(12345, EbpfEventType::EXECVE_ATTEMPT, "execve detected", 40'000'000ull);

    std::vector<EbpfEvent> alerts;
    mon.set_alert_callback([&](const EbpfEvent& ev) { alerts.push_back(ev); });

    size_t n = mon.poll();
    REQUIRE(n == 1);
    REQUIRE(mon.total_events() == 1);
    REQUIRE(mon.events().size() == 1);
    REQUIRE(mon.events()[0].type == EbpfEventType::EXECVE_ATTEMPT);
    REQUIRE(mon.events()[0].vm_name == "sandbox_vm1");
    REQUIRE(alerts.size() == 1);
}

// ============================================================================
// §11 eBPF Monitor — Latency tracking
// ============================================================================

TEST_CASE("§11 eBPF latency tracking", "[phase156][ebpf]") {
    EbpfMonitor::Config cfg;
    cfg.auto_kill = false;
    EbpfMonitor mon(cfg);
    mon.watch_pid(100, "vm_lat");

    // Event within budget (50ms)
    mon.inject_event(100, EbpfEventType::EXECVE_ATTEMPT, "test1", 50'000'000ull);
    mon.poll();
    REQUIRE(mon.within_latency_budget());
    REQUIRE(mon.avg_response_ns() == 50'000'000ull);

    // Event right at budget (100ms)
    mon.inject_event(100, EbpfEventType::FILE_OPEN_OUTSIDE, "test2", 100'000'000ull);
    mon.poll();
    REQUIRE(mon.within_latency_budget());

    // Event over budget (150ms)
    mon.inject_event(100, EbpfEventType::NETWORK_ATTEMPT, "test3", 150'000'000ull);
    mon.poll();
    REQUIRE_FALSE(mon.within_latency_budget());
    REQUIRE(mon.worst_response_ns() == 150'000'000ull);
}

// ============================================================================
// §12 eBPF Monitor — Response policy
// ============================================================================

TEST_CASE("§12 eBPF response policy", "[phase156][ebpf]") {
    ResponsePolicy policy;
    REQUIRE(policy.action_for(EbpfEventType::EXECVE_ATTEMPT) == ResponseAction::KILL_AND_ALERT);
    REQUIRE(policy.action_for(EbpfEventType::CLONE_ATTEMPT)  == ResponseAction::KILL_AND_ALERT);
    REQUIRE(policy.action_for(EbpfEventType::UNKNOWN)        == ResponseAction::LOG_ONLY);

    // Custom policy: log-only for file opens
    policy.on_file_open = ResponseAction::LOG_ONLY;
    REQUIRE(policy.action_for(EbpfEventType::FILE_OPEN_OUTSIDE) == ResponseAction::LOG_ONLY);
}

// ============================================================================
// §13 eBPF Monitor — Multiple event types
// ============================================================================

TEST_CASE("§13 eBPF multiple event types", "[phase156][ebpf]") {
    EbpfMonitor::Config cfg;
    cfg.auto_kill = false;
    EbpfMonitor mon(cfg);
    mon.watch_pid(200, "multi_vm");

    mon.inject_event(200, EbpfEventType::EXECVE_ATTEMPT);
    mon.inject_event(200, EbpfEventType::FILE_OPEN_OUTSIDE);
    mon.inject_event(200, EbpfEventType::CLONE_ATTEMPT);
    mon.inject_event(200, EbpfEventType::PTRACE_ATTEMPT);
    mon.inject_event(200, EbpfEventType::NETWORK_ATTEMPT);

    size_t n = mon.poll();
    REQUIRE(n == 5);
    REQUIRE(mon.total_events() == 5);
    REQUIRE(mon.events().size() == 5);

    // Verify all types present
    std::set<EbpfEventType> evt_types;
    for (const auto& ev : mon.events()) evt_types.insert(ev.type);
    REQUIRE(evt_types.size() == 5);
}

// ============================================================================
// §14 CSVP — Constants valid
// ============================================================================

TEST_CASE("§14 CSVP constants valid", "[phase156][csvp]") {
    REQUIRE(CSVP_MAX_SOURCE_BYTES     == 1024 * 1024);
    REQUIRE(CSVP_MAX_FUNCTION_LINES   == 500);
    REQUIRE(CSVP_MAX_NESTING_DEPTH    == 10);
    REQUIRE(CSVP_MAX_ALLOCATION_BYTES == 64 * 1024 * 1024);
}

// ============================================================================
// §15 CSVP — Safe code passes all stages
// ============================================================================

TEST_CASE("§15 CSVP safe code passes", "[phase156][csvp]") {
    CodeSafetyVerifier csvp;

    std::string safe = R"(
#include <cmath>
#include <vector>

double compute(double x) {
    return std::sqrt(x * x + 1.0);
}

int main() {
    std::vector<double> data(100);
    for (int i = 0; i < 100; ++i) {
        data[i] = compute(static_cast<double>(i));
    }
    return 0;
}
)";

    auto result = csvp.verify(safe);
    REQUIRE(result.approved);
    REQUIRE(result.stages_passed == 4);
    REQUIRE(result.violations.empty());
}

// ============================================================================
// §16 CSVP — Stage 1 blacklist catches dangerous patterns
// ============================================================================

TEST_CASE("§16 CSVP stage 1 blacklist", "[phase156][csvp]") {
    CodeSafetyVerifier csvp;

    // system() call
    REQUIRE_FALSE(csvp.is_safe("int main() { system(\"rm -rf /\"); }"));

    // exec family
    REQUIRE_FALSE(csvp.is_safe("int main() { execve(\"/bin/sh\", NULL, NULL); }"));

    // fork
    REQUIRE_FALSE(csvp.is_safe("int main() { fork(); }"));

    // inline asm
    REQUIRE_FALSE(csvp.is_safe("void f() { asm(\"nop\"); }"));

    // Network header
    REQUIRE_FALSE(csvp.is_safe("#include <sys/socket.h>\nint main() { return 0; }"));

    // Verify failed stage is PATTERN_BLACKLIST
    auto result = csvp.verify("int main() { system(\"cmd\"); }");
    REQUIRE_FALSE(result.approved);
    REQUIRE(result.failed_stage == CSVPStage::PATTERN_BLACKLIST);
}

// ============================================================================
// §17 CSVP — Stage 2 structural (goto forbidden)
// ============================================================================

TEST_CASE("§17 CSVP stage 2 goto forbidden", "[phase156][csvp]") {
    CodeSafetyVerifier csvp;

    std::string code_with_goto = R"(
#include <cmath>
int main() {
    goto done;
    done:
    return 0;
}
)";

    auto result = csvp.verify(code_with_goto);
    REQUIRE_FALSE(result.approved);
    REQUIRE(result.failed_stage == CSVPStage::STRUCTURAL);

    bool found_goto = false;
    for (const auto& v : result.violations) {
        if (v.rule_name == "goto_forbidden") found_goto = true;
    }
    REQUIRE(found_goto);
}

// ============================================================================
// §18 CSVP — Stage 3 resource (large allocation)
// ============================================================================

TEST_CASE("§18 CSVP stage 3 large allocation", "[phase156][csvp]") {
    CodeSafetyVerifier csvp;

    // 100 million element array * 8 bytes = 800MB > 64MB limit
    std::string code = R"(
#include <cstdint>
int main() {
    double arr[100000000];
    return 0;
}
)";

    auto result = csvp.verify(code);
    REQUIRE_FALSE(result.approved);
    REQUIRE(result.failed_stage == CSVPStage::RESOURCE);

    bool found_alloc = false;
    for (const auto& v : result.violations) {
        if (v.rule_name == "large_allocation") found_alloc = true;
    }
    REQUIRE(found_alloc);
}

// ============================================================================
// §19 CSVP — Stage 3 resource (unbounded loop)
// ============================================================================

TEST_CASE("§19 CSVP stage 3 unbounded loop", "[phase156][csvp]") {
    CodeSafetyVerifier csvp;

    std::string code = R"(
#include <cmath>
int main() {
    while(true) {
        double x = 1.0;
        x = x + 1.0;
        x = x * 2.0;
    }
}
)";

    auto result = csvp.verify(code);
    REQUIRE_FALSE(result.approved);
    REQUIRE(result.failed_stage == CSVPStage::RESOURCE);

    bool found_loop = false;
    for (const auto& v : result.violations) {
        if (v.rule_name == "unbounded_loop") found_loop = true;
    }
    REQUIRE(found_loop);
}

// ============================================================================
// §20 CSVP — Stage 4 physics (negative energy)
// ============================================================================

TEST_CASE("§20 CSVP stage 4 physics invariant", "[phase156][csvp]") {
    CodeSafetyVerifier csvp;

    std::string code = R"(
#include <cmath>
double compute() {
    double energy = -999.0;
    return energy;
}
)";

    auto result = csvp.verify(code);
    REQUIRE_FALSE(result.approved);
    REQUIRE(result.failed_stage == CSVPStage::PHYSICS_INVARIANT);
}

// ============================================================================
// §21 CSVP — Source size limit
// ============================================================================

TEST_CASE("§21 CSVP source size limit", "[phase156][csvp]") {
    CodeSafetyVerifier::Config cfg;
    cfg.max_source_bytes = 100;
    CodeSafetyVerifier csvp(cfg);

    std::string big(200, 'x');
    auto result = csvp.verify(big);
    REQUIRE_FALSE(result.approved);
}

// ============================================================================
// §22 Anomaly Detector — Module registration and observation
// ============================================================================

TEST_CASE("§22 Anomaly detector registration", "[phase156][anomaly]") {
    AnomalyDetector det;

    REQUIRE(det.register_module("mod1"));
    REQUIRE(det.register_module("mod2"));
    REQUIRE_FALSE(det.register_module("mod1"));  // duplicate
    REQUIRE(det.module_count() == 2);

    REQUIRE(det.unregister_module("mod2"));
    REQUIRE(det.module_count() == 1);

    // Record observations
    BehaviorObservation obs;
    obs.cpu_usage = 0.2;
    obs.memory_usage = 0.3;
    REQUIRE(det.record_observation("mod1", obs));
    REQUIRE_FALSE(det.record_observation("nonexistent", obs));

    auto* prof = det.get_profile("mod1");
    REQUIRE(prof != nullptr);
    REQUIRE(prof->total_observations == 1);
}

// ============================================================================
// §23 Anomaly Detector — Spike detection and quarantine
// ============================================================================

TEST_CASE("§23 Anomaly spike detection and quarantine", "[phase156][anomaly]") {
    AnomalyDetector::Config cfg;
    cfg.min_baseline_samples = 5;
    cfg.quarantine_threshold = 0.8;
    AnomalyDetector det(cfg);

    det.register_module("test_mod");

    // Build a baseline with low CPU usage (0.1-0.15)
    for (int i = 0; i < 20; ++i) {
        BehaviorObservation obs;
        obs.cpu_usage    = 0.10 + 0.005 * (i % 3);  // 0.10-0.11
        obs.memory_usage = 0.20;
        obs.syscall_count = 50;
        obs.message_count = 5;
        det.record_observation("test_mod", obs);
        det.analyze("test_mod");  // update baseline
    }

    REQUIRE_FALSE(det.is_quarantined("test_mod"));

    // Now spike CPU to 0.95 (way above baseline ~0.105)
    BehaviorObservation spike;
    spike.cpu_usage    = 0.95;
    spike.memory_usage = 0.20;
    spike.syscall_count = 50;
    spike.message_count = 5;
    det.record_observation("test_mod", spike);

    auto threats = det.analyze("test_mod");
    REQUIRE(threats.size() >= 1);

    // At least one threat should be a resource spike
    bool found_spike = false;
    for (const auto& t : threats) {
        if (t.type == ThreatType::RESOURCE_SPIKE) {
            found_spike = true;
            REQUIRE(t.severity > 0.0);
            REQUIRE(t.deviation > 3.0);  // beyond 3σ
        }
    }
    REQUIRE(found_spike);
}

// ============================================================================
// §24 Anomaly Detector — Manual quarantine and release
// ============================================================================

TEST_CASE("§24 Anomaly quarantine management", "[phase156][anomaly]") {
    AnomalyDetector det;
    det.register_module("quarantine_test");

    REQUIRE_FALSE(det.is_quarantined("quarantine_test"));
    REQUIRE(det.quarantined_count() == 0);

    // Manual quarantine
    std::string q_name;
    QuarantineReason q_reason = QuarantineReason::NOT_QUARANTINED;
    det.set_quarantine_callback([&](const std::string& n, QuarantineReason r) {
        q_name = n;
        q_reason = r;
    });

    REQUIRE(det.quarantine("quarantine_test", QuarantineReason::ESCAPE_ATTEMPT));
    REQUIRE(det.is_quarantined("quarantine_test"));
    REQUIRE(det.quarantined_count() == 1);
    REQUIRE(q_name == "quarantine_test");
    REQUIRE(q_reason == QuarantineReason::ESCAPE_ATTEMPT);

    auto qlist = det.quarantined_modules();
    REQUIRE(qlist.size() == 1);

    // Release
    REQUIRE(det.release("quarantine_test"));
    REQUIRE_FALSE(det.is_quarantined("quarantine_test"));
    REQUIRE(det.quarantined_count() == 0);

    // Profile reflects quarantine state
    auto* prof = det.get_profile("quarantine_test");
    REQUIRE(prof != nullptr);
    REQUIRE_FALSE(prof->quarantined);
    REQUIRE(prof->quarantine_reason == QuarantineReason::NOT_QUARANTINED);
}
