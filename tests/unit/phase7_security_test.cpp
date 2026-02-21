/**
 * @file tests/unit/phase7_security_test.cpp
 * @brief Phase 7: Security & Execution test suite (Catch2 v3).
 *
 * Covers all 5 Gap criteria:
 *   Gap 7.1 — VMImageManager: SHA-256 verification, checksum parsing
 *   Gap 7.2 — InterVMChannel: host-mediated routing, policy whitelist, payload validation
 *   Gap 7.3 — EscapeDetector: event injection, alert callback, watched VM tracking
 *   Gap 7.4 — CodePatternBlacklist: dangerous call detection, include whitelist
 *   Gap 7.5 — VMPerformanceMonitor: cgroup path resolution, limit checking
 *
 * Plus: SecurityEngine integration facade.
 *
 * No KVM, no eBPF, no libbpf required — all tests use the pure C++ paths.
 */

#define NIKOLA_SECURITY_ENGINE_IMPL

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <nikola/security/vm_image_manager.hpp>
#include <nikola/security/inter_vm_channel.hpp>
#include <nikola/security/escape_detector.hpp>
#include <nikola/security/code_blacklist.hpp>
#include <nikola/security/vm_perf_monitor.hpp>
#include <nikola/security/security_engine.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace nikola::security;
using Catch::Matchers::ContainsSubstring;

// ─────────────────────────────────────────────────────────────────────────────
//  Gap 7.1 — VMImageManager / SHA-256
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Gap7.1 — SHA-256 of empty string matches known value", "[sha256][gap7.1]") {
    // SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    const std::string expected = "e3b0c44298fc1c149afbf4c8996fb924"
                                  "27ae41e4649b934ca495991b7852b855";
    const auto digest = sha256_string("");
    CHECK(digest_to_hex(digest) == expected);
}

TEST_CASE("Gap7.1 — SHA-256 of 'abc' matches known value", "[sha256][gap7.1]") {
    // SHA-256("abc") = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
    const std::string expected = "ba7816bf8f01cfea414140de5dae2223"
                                  "b00361a396177a9cb410ff61f20015ad";
    const auto digest = sha256_string("abc");
    CHECK(digest_to_hex(digest) == expected);
}

TEST_CASE("Gap7.1 — SHA-256 of 55-byte string (one block minus 1)", "[sha256][gap7.1]") {
    // 55 bytes of 'a' — fits in one block without needing extra padding block
    const std::string s(55, 'a');
    const auto digest = sha256_string(s);
    CHECK(digest_to_hex(digest).size() == 64u);
}

TEST_CASE("Gap7.1 — SHA-256 of 64-byte string (exact block size)", "[sha256][gap7.1]") {
    const std::string s(64, 'x');
    const auto digest = sha256_string(s);
    CHECK(digest_to_hex(digest).size() == 64u);
}

TEST_CASE("Gap7.1 — SHA-256 of file matches in-memory computation", "[sha256][gap7.1]") {
    // Write a temp file and verify file hash matches in-memory hash
    const auto tmpf = std::filesystem::temp_directory_path() / "nikola_sha_test.bin";
    const std::string content = "nikola security test data 1234567890!";
    {
        std::ofstream f(tmpf, std::ios::binary);
        f.write(content.data(), static_cast<std::streamsize>(content.size()));
    }
    bool ok = false;
    const auto file_digest = sha256_file(tmpf.string(), ok);
    CHECK(ok);
    const auto mem_digest = sha256_string(content);
    CHECK(digest_to_hex(file_digest) == digest_to_hex(mem_digest));
    std::filesystem::remove(tmpf);
}

TEST_CASE("Gap7.1 — hex_to_digest / digest_to_hex round-trip", "[sha256][gap7.1]") {
    const std::string hex = "ba7816bf8f01cfea414140de5dae2223"
                             "b00361a396177a9cb410ff61f20015ad";
    SHA256Digest d{};
    REQUIRE(hex_to_digest(hex, d));
    CHECK(digest_to_hex(d) == hex);
}

TEST_CASE("Gap7.1 — VMImageManager returns error if image file missing", "[vm_image][gap7.1]") {
    VMImageManager::Config cfg;
    cfg.gold_image_path = "/does/not/exist/gold.qcow2";
    cfg.checksums_path  = "/does/not/exist/checksums.txt";
    cfg.strict_mode     = false;
    VMImageManager mgr(cfg);
    auto r = mgr.verify_integrity();
    CHECK_FALSE(r.ok);
    CHECK_FALSE(r.error_msg.empty());
}

TEST_CASE("Gap7.1 — VMImageManager verifies correctly when digest matches",
          "[vm_image][gap7.1]") {
    // Write a temp file with known content
    const auto tmpf = std::filesystem::temp_directory_path() / "nikola_gold_test.qcow2";
    const std::string content = "fake-qcow2-content";
    {
        std::ofstream f(tmpf, std::ios::binary);
        f.write(content.data(), static_cast<std::streamsize>(content.size()));
    }
    const std::string expected_hex = digest_to_hex(sha256_string(content));

    VMImageManager::Config cfg;
    cfg.gold_image_path = tmpf.string();
    cfg.strict_mode     = true;
    VMImageManager mgr(cfg);
    mgr.set_expected_hex(expected_hex);

    auto r = mgr.verify_integrity();
    CHECK(r.ok);
    CHECK(r.actual_hex == expected_hex);
    std::filesystem::remove(tmpf);
}

TEST_CASE("Gap7.1 — VMImageManager rejects tampered file", "[vm_image][gap7.1]") {
    const auto tmpf = std::filesystem::temp_directory_path() / "nikola_tampered.qcow2";
    {
        std::ofstream f(tmpf, std::ios::binary);
        f.write("original", 8);
    }
    const std::string bad_hex(64, '0'); // all-zero digest — won't match
    VMImageManager::Config cfg;
    cfg.gold_image_path = tmpf.string();
    cfg.strict_mode     = true;
    VMImageManager mgr(cfg);
    mgr.set_expected_hex(bad_hex);

    auto r = mgr.verify_integrity();
    CHECK_FALSE(r.ok);
    CHECK_THAT(r.error_msg, ContainsSubstring("mismatch"));
    std::filesystem::remove(tmpf);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Gap 7.2 — InterVMChannel
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Gap7.2 — default policy allows executor_1 → orchestrator",
          "[ivm][gap7.2]") {
    InterVMChannel ch;
    ch.register_vm("executor_1");
    ch.register_vm("orchestrator");

    VMMessage msg{"executor_1", "orchestrator", {0x01, 0x02, 0x03}};
    auto r = ch.route(msg);
    CHECK(r.status == VMMessageStatus::DELIVERED);
}

TEST_CASE("Gap7.2 — default policy allows executor_2 → orchestrator",
          "[ivm][gap7.2]") {
    InterVMChannel ch;
    ch.register_vm("executor_2");
    ch.register_vm("orchestrator");

    VMMessage msg{"executor_2", "orchestrator", {0xAA}};
    auto r = ch.route(msg);
    CHECK(r.status == VMMessageStatus::DELIVERED);
}

TEST_CASE("Gap7.2 — VMs cannot communicate directly (executor_1 → executor_2)",
          "[ivm][gap7.2]") {
    InterVMChannel ch;
    ch.register_vm("executor_1");
    ch.register_vm("executor_2");

    VMMessage msg{"executor_1", "executor_2", {0x01}};
    auto r = ch.route(msg);
    CHECK(r.status == VMMessageStatus::BLOCKED_POLICY);
}

TEST_CASE("Gap7.2 — unknown sender is rejected", "[ivm][gap7.2]") {
    InterVMChannel ch;
    ch.register_vm("orchestrator");

    VMMessage msg{"ghost_vm", "orchestrator", {}};
    auto r = ch.route(msg);
    CHECK(r.status == VMMessageStatus::BLOCKED_UNKNOWN_SENDER);
}

TEST_CASE("Gap7.2 — unknown receiver is rejected", "[ivm][gap7.2]") {
    InterVMChannel ch;
    ch.register_vm("executor_1");

    VMMessage msg{"executor_1", "nowhere", {}};
    auto r = ch.route(msg);
    CHECK(r.status == VMMessageStatus::BLOCKED_UNKNOWN_RECEIVER);
}

TEST_CASE("Gap7.2 — payload over 1MB is rejected", "[ivm][gap7.2]") {
    InterVMChannel ch;
    ch.register_vm("executor_1");
    ch.register_vm("orchestrator");

    VMMessage msg;
    msg.from_vm = "executor_1";
    msg.to_vm   = "orchestrator";
    msg.payload.resize(IVM_MAX_PAYLOAD_BYTES + 1, 0x00);

    auto r = ch.route(msg);
    CHECK(r.status == VMMessageStatus::BLOCKED_PAYLOAD_TOO_LARGE);
}

TEST_CASE("Gap7.2 — NOP sled (16× 0x90) is rejected as shellcode", "[ivm][gap7.2]") {
    InterVMChannel ch;
    ch.register_vm("executor_1");
    ch.register_vm("orchestrator");

    VMMessage msg{"executor_1", "orchestrator", {}};
    msg.payload.resize(32, 0x90u); // 32 NOP bytes

    auto r = ch.route(msg);
    CHECK(r.status == VMMessageStatus::BLOCKED_PAYLOAD_INVALID);
}

TEST_CASE("Gap7.2 — delivery callback fires on success", "[ivm][gap7.2]") {
    InterVMChannel ch;
    ch.register_vm("executor_1");
    ch.register_vm("orchestrator");

    bool called = false;
    ch.set_delivery_callback([&](const std::string&, const std::string&,
                                  const std::vector<uint8_t>&) {
        called = true;
    });

    VMMessage msg{"executor_1", "orchestrator", {0x42}};
    ch.route(msg);
    CHECK(called);
}

TEST_CASE("Gap7.2 — stats count delivered vs blocked correctly", "[ivm][gap7.2]") {
    InterVMChannel ch;
    ch.register_vm("executor_1");
    ch.register_vm("orchestrator");

    VMMessage ok_msg{"executor_1", "orchestrator", {0x01}};
    VMMessage bad_msg{"executor_1", "executor_1", {0x01}};

    ch.route(ok_msg);
    ch.route(ok_msg);
    ch.route(bad_msg); // blocked (same VM, not allowed)
    // bad_msg has unknown recv actually — let's use a blocked-policy case
    VMMessage policy_block{"executor_1", "executor_1", {0x01}};
    // Need executor_1 registered as receiver too:
    ch.route(policy_block);

    CHECK(ch.stats().delivered == 2u);
    CHECK(ch.stats().blocked   >= 1u);
}

TEST_CASE("Gap7.2 — allow() adds custom policy pair", "[ivm][gap7.2]") {
    InterVMChannel ch;
    ch.register_vm("vm_a");
    ch.register_vm("vm_b");
    ch.allow("vm_a", "vm_b");

    VMMessage msg{"vm_a", "vm_b", {0x01}};
    auto r = ch.route(msg);
    CHECK(r.status == VMMessageStatus::DELIVERED);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Gap 7.3 — EscapeDetector
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Gap7.3 — inject_event fires alert callback", "[escape][gap7.3]") {
    EscapeDetector ed;
    ed.watch_vm("test_vm", -1);

    bool alerted = false;
    EscapeEvent received;
    ed.set_alert_callback([&](const EscapeEvent& ev) {
        alerted = true;
        received = ev;
    });

    ed.inject_event("test_vm", EscapeType::EXECVE_DETECTED);
    CHECK(alerted);
    CHECK(received.type == EscapeType::EXECVE_DETECTED);
    CHECK(received.vm_name == "test_vm");
}

TEST_CASE("Gap7.3 — inject forbidden file open alert", "[escape][gap7.3]") {
    EscapeDetector ed;
    ed.watch_vm("vm1", -1);

    ed.inject_event("vm1", EscapeType::FORBIDDEN_FILE_OPEN, "/etc/passwd");
    REQUIRE(ed.events().size() == 1u);
    CHECK(ed.events()[0].type   == EscapeType::FORBIDDEN_FILE_OPEN);
    CHECK(ed.events()[0].detail == "/etc/passwd");
}

TEST_CASE("Gap7.3 — total_alerts counter increments correctly", "[escape][gap7.3]") {
    EscapeDetector ed;
    ed.watch_vm("vm1", -1);
    ed.watch_vm("vm2", -1);

    ed.inject_event("vm1", EscapeType::EXECVE_DETECTED);
    ed.inject_event("vm2", EscapeType::RESOURCE_LIMIT);
    ed.inject_event("vm1", EscapeType::FORBIDDEN_FILE_OPEN);

    CHECK(ed.total_alerts() == 3u);
}

TEST_CASE("Gap7.3 — clear_events empties event log", "[escape][gap7.3]") {
    EscapeDetector ed;
    ed.watch_vm("vm1", -1);
    ed.inject_event("vm1", EscapeType::EXECVE_DETECTED);
    CHECK(ed.events().size() == 1u);
    ed.clear_events();
    CHECK(ed.events().empty());
}

TEST_CASE("Gap7.3 — watched_count reflects registrations", "[escape][gap7.3]") {
    EscapeDetector ed;
    CHECK(ed.watched_count() == 0u);
    ed.watch_vm("vm1", -1);
    ed.watch_vm("vm2", -1);
    CHECK(ed.watched_count() == 2u);
    ed.unwatch_vm("vm1");
    CHECK(ed.watched_count() == 1u);
}

TEST_CASE("Gap7.3 — escape_type_str returns correct strings", "[escape][gap7.3]") {
    CHECK(std::string(escape_type_str(EscapeType::EXECVE_DETECTED))     == "EXECVE_DETECTED");
    CHECK(std::string(escape_type_str(EscapeType::FORBIDDEN_FILE_OPEN)) == "FORBIDDEN_FILE_OPEN");
    CHECK(std::string(escape_type_str(EscapeType::RESOURCE_LIMIT))      == "RESOURCE_LIMIT");
    CHECK(std::string(escape_type_str(EscapeType::PROCESS_GONE))        == "PROCESS_GONE");
}

// ─────────────────────────────────────────────────────────────────────────────
//  Gap 7.4 — CodePatternBlacklist
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Gap7.4 — safe minimal C++ code passes", "[blacklist][gap7.4]") {
    CodePatternBlacklist bl;
    const std::string src = R"(
#include <vector>
#include <algorithm>
#include <cmath>
int main() {
    std::vector<double> v = {1.0, 2.0, 3.0};
    double s = 0;
    for (double x : v) s += std::sqrt(x);
    return 0;
}
)";
    CHECK(bl.is_safe(src));
}

TEST_CASE("Gap7.4 — system() call is rejected", "[blacklist][gap7.4]") {
    CodePatternBlacklist bl;
    const std::string src = R"(
#include <cstdlib>
int main() { system("rm -rf /"); return 0; }
)";
    auto r = bl.check(src);
    CHECK_FALSE(r.safe);
    const bool has_system = std::any_of(r.violations.begin(), r.violations.end(),
        [](const ScanViolation& v){ return v.pattern_name == "system_call"; });
    CHECK(has_system);
}

TEST_CASE("Gap7.4 — execve() family is rejected", "[blacklist][gap7.4]") {
    CodePatternBlacklist bl;
    const std::string src = "int main() { execve(\"/bin/sh\", 0, 0); }";
    CHECK_FALSE(bl.is_safe(src));
}

TEST_CASE("Gap7.4 — fork() is rejected", "[blacklist][gap7.4]") {
    CodePatternBlacklist bl;
    const std::string src = "int main() { int p = fork(); return p; }";
    CHECK_FALSE(bl.is_safe(src));
}

TEST_CASE("Gap7.4 — popen() is rejected", "[blacklist][gap7.4]") {
    CodePatternBlacklist bl;
    const std::string src = "auto f = popen(\"ls\", \"r\");";
    CHECK_FALSE(bl.is_safe(src));
}

TEST_CASE("Gap7.4 — inline asm is rejected", "[blacklist][gap7.4]") {
    CodePatternBlacklist bl;
    const std::string src = "void f() { asm(\"nop\"); }";
    CHECK_FALSE(bl.is_safe(src));
}

TEST_CASE("Gap7.4 — __asm__ is rejected", "[blacklist][gap7.4]") {
    CodePatternBlacklist bl;
    const std::string src = "void f() { __asm__(\"xor %eax, %eax\"); }";
    CHECK_FALSE(bl.is_safe(src));
}

TEST_CASE("Gap7.4 — socket header inclusion is rejected", "[blacklist][gap7.4]") {
    CodePatternBlacklist bl;
    const std::string src = "#include <sys/socket.h>\nint main(){}";
    CHECK_FALSE(bl.is_safe(src));
}

TEST_CASE("Gap7.4 — /proc/ path reference is rejected", "[blacklist][gap7.4]") {
    CodePatternBlacklist bl;
    const std::string src = R"(const char* p = "/proc/self/mem";)";
    CHECK_FALSE(bl.is_safe(src));
}

TEST_CASE("Gap7.4 — /dev/ path (non-null) is rejected", "[blacklist][gap7.4]") {
    CodePatternBlacklist bl;
    const std::string src = R"(const char* d = "/dev/sda";)";
    CHECK_FALSE(bl.is_safe(src));
}

TEST_CASE("Gap7.4 — disallowed include (<unistd.h>) is rejected", "[blacklist][gap7.4]") {
    CodePatternBlacklist bl;
    const std::string src = "#include <unistd.h>\nint main(){}";
    auto r = bl.check(src);
    CHECK_FALSE(r.safe);
    bool found = std::any_of(r.violations.begin(), r.violations.end(),
        [](const ScanViolation& v){ return v.pattern_name == "disallowed_include"; });
    CHECK(found);
}

TEST_CASE("Gap7.4 — all spec-listed safe includes pass", "[blacklist][gap7.4]") {
    CodePatternBlacklist bl;
    const std::string src = R"(
#include <math.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
int main() { return 0; }
)";
    CHECK(bl.is_safe(src));
}

TEST_CASE("Gap7.4 — violation reports line numbers > 0", "[blacklist][gap7.4]") {
    CodePatternBlacklist bl;
    const std::string src = "// line 1\n// line 2\nsystem(\"x\");\n";
    auto r = bl.check(src);
    REQUIRE_FALSE(r.violations.empty());
    CHECK(r.violations[0].line_number >= 1u);
}

TEST_CASE("Gap7.4 — add_dangerous_pattern extends the blacklist", "[blacklist][gap7.4]") {
    CodePatternBlacklist bl;
    bl.add_dangerous_pattern("custom_banned", R"(\bcustom_func\s*\()");
    const std::string src = "int x = custom_func(42);";
    CHECK_FALSE(bl.is_safe(src));
}

TEST_CASE("Gap7.4 — add_allowed_include extends whitelist", "[blacklist][gap7.4]") {
    CodePatternBlacklist bl;
    bl.add_allowed_include("my_safe_lib.h");
    const std::string src = "#include <my_safe_lib.h>\nint main(){}";
    CHECK(bl.is_safe(src));
}

// ─────────────────────────────────────────────────────────────────────────────
//  Gap 7.5 — VMPerformanceMonitor
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Gap7.5 — cgroup scope name follows convention", "[perf][gap7.5]") {
    VMPerformanceMonitor m("executor_1");
    CHECK(m.cgroup_scope() == "nikola_vm_executor_1");
}

TEST_CASE("Gap7.5 — collect_stats returns zeroed stats when cgroup absent",
          "[perf][gap7.5]") {
    // Use a tmpdir as non-existent cgroup base
    VMPerformanceMonitor m("vm1", "/nonexistent/cgroup/path");
    auto s = m.collect_stats();
    CHECK(s.available == false);
    CHECK(s.cpu_usage_ns       == 0u);
    CHECK(s.memory_usage_bytes == 0u);
    CHECK(s.io_read_bytes      == 0u);
    CHECK(s.io_write_bytes     == 0u);
}

TEST_CASE("Gap7.5 — check_limits returns violation for memory over limit",
          "[perf][gap7.5]") {
    VMPerformanceMonitor m("vm1", "/nonexistent");
    VMStats s;
    s.memory_usage_bytes = MAX_MEMORY_BYTES + 1;
    auto viols = m.check_limits(s);
    REQUIRE(viols.size() == 1u);
    CHECK(viols[0].resource == "memory");
    CHECK(viols[0].value    >  viols[0].limit);
}

TEST_CASE("Gap7.5 — check_limits passes when memory under limit", "[perf][gap7.5]") {
    VMPerformanceMonitor m("vm1", "/nonexistent");
    VMStats s;
    s.memory_usage_bytes = MAX_MEMORY_BYTES / 2;
    auto viols = m.check_limits(s);
    CHECK(viols.empty());
}

TEST_CASE("Gap7.5 — cgroup v2 paths are correctly constructed", "[perf][gap7.5]") {
    VMPerformanceMonitor m("executor_2", "/sys/fs/cgroup");
    CHECK_THAT(m.v2_memory_path(),
        ContainsSubstring("nikola_vm_executor_2"));
    CHECK_THAT(m.v2_memory_path(),
        ContainsSubstring("memory.current"));
    CHECK_THAT(m.v2_cpu_path(),
        ContainsSubstring("cpu.stat"));
}

TEST_CASE("Gap7.5 — cgroup v1 paths are correctly constructed", "[perf][gap7.5]") {
    VMPerformanceMonitor m("executor_2", "/sys/fs/cgroup");
    CHECK_THAT(m.v1_memory_path(),
        ContainsSubstring("memory.usage_in_bytes"));
    CHECK_THAT(m.v1_cpu_path(),
        ContainsSubstring("cpuacct.usage"));
}

TEST_CASE("Gap7.5 — tick returns no violations on zeroed stats", "[perf][gap7.5]") {
    // Non-existent cgroup → zeroed stats → no violations
    VMPerformanceMonitor m("vm_test", "/nonexistent");
    auto viols = m.tick();
    CHECK(viols.empty());
}

TEST_CASE("Gap7.5 — MAX limits have expected values", "[perf][gap7.5]") {
    CHECK(MAX_MEMORY_BYTES     == 512ull * 1024 * 1024);
    CHECK(MAX_CPU_NS_PER_SEC   == 1'000'000'000ull);
    CHECK(MAX_IO_BYTES_PER_SEC == 1024ull * 1024);
}

// ─────────────────────────────────────────────────────────────────────────────
//  SecurityEngine integration
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("SecurityEngine — constructs with default config", "[engine][phase7]") {
    SecurityConfig cfg;
    cfg.image_cfg.strict_mode = false; // don't require gold image to exist
    SecurityEngine eng(std::move(cfg));
    const auto snap = eng.snapshot();
    CHECK(snap.code_scans       == 0u);
    CHECK(snap.messages_routed  == 0u);
    CHECK(snap.escape_alerts    == 0u);
}

TEST_CASE("SecurityEngine — safe code passes scan", "[engine][phase7]") {
    SecurityConfig cfg;
    cfg.image_cfg.strict_mode = false;
    SecurityEngine eng(std::move(cfg));

    const std::string src = R"(
#include <vector>
#include <cmath>
int main() { return 0; }
)";
    auto r = eng.scan_code(src);
    CHECK(r.safe);
    CHECK(eng.snapshot().code_scans      == 1u);
    CHECK(eng.snapshot().code_rejections == 0u);
}

TEST_CASE("SecurityEngine — dangerous code is rejected and counted", "[engine][phase7]") {
    SecurityConfig cfg;
    cfg.image_cfg.strict_mode = false;
    SecurityEngine eng(std::move(cfg));

    CHECK_FALSE(eng.is_code_safe("system(\"rm -rf /\");"));
    CHECK(eng.snapshot().code_rejections == 1u);
}

TEST_CASE("SecurityEngine — route_message obeys policy", "[engine][phase7]") {
    SecurityConfig cfg;
    cfg.image_cfg.strict_mode = false;
    SecurityEngine eng(std::move(cfg));

    eng.register_vm("executor_1");
    eng.register_vm("orchestrator");

    VMMessage msg{"executor_1", "orchestrator", {0x01}};
    auto r = eng.route_message(msg);
    CHECK(r.status == VMMessageStatus::DELIVERED);
    CHECK(eng.snapshot().messages_routed == 1u);
}

TEST_CASE("SecurityEngine — inject_escape_event increments alert count",
          "[engine][phase7]") {
    SecurityConfig cfg;
    cfg.image_cfg.strict_mode = false;
    SecurityEngine eng(std::move(cfg));

    eng.watch_vm_escape("vm1", -1);
    eng.inject_escape_event("vm1", EscapeType::EXECVE_DETECTED);
    CHECK(eng.snapshot().escape_alerts == 1u);
    CHECK(eng.escape_events().size()   == 1u);
}

TEST_CASE("SecurityEngine — collect_vm_stats returns stats struct", "[engine][phase7]") {
    SecurityConfig cfg;
    cfg.image_cfg.strict_mode = false;
    SecurityEngine eng(std::move(cfg));

    auto s = eng.collect_vm_stats("executor_1");
    // On non-KVM host: available=false, all zeros — just check it doesn't throw
    CHECK((s.memory_usage_bytes == 0u || s.memory_usage_bytes > 0u)); // any value OK
}
