/**
 * @file security/security_pipeline.hpp
 * @brief v0.2.6 — Production Security Pipeline
 *
 * End-to-end pipeline for SIE-generated code deployment:
 *   1. SIE generates code → extracted source
 *   2. CSVP verify (4-stage static analysis)
 *   3. CodePatternBlacklist scan (pattern matching)
 *   4. KVM sandbox execution (isolated VM)
 *   5. eBPF monitoring (syscall escape detection)
 *   6. Anomaly detection (behavioral analysis)
 *   7. Deploy or quarantine
 *
 * Every decision is recorded in the audit log with reasoning.
 *
 * All policy logic compiles without NIKOLA_ENABLE_KVM/EBPF.
 * Real KVM/eBPF operations are only attempted when those macros are set.
 */
#pragma once

#include "csvp.hpp"
#include "anomaly_detector.hpp"
#include "kvm_sandbox.hpp"
#include "ebpf_monitor.hpp"
#include "mini_vm_manager.hpp"
#include "code_blacklist.hpp"

#include <chrono>
#include <functional>
#include <string>
#include <vector>

namespace nikola::security {

// ============================================================================
// Pipeline stage enumeration
// ============================================================================

enum class PipelineStage : uint8_t {
    SUBMITTED,         ///< Code received from SIE
    CSVP_VERIFY,       ///< CSVP 4-stage static analysis
    STATIC_ANALYSIS,   ///< clang-tidy + cppcheck (Gate 1.5)
    PATTERN_SCAN,      ///< CodePatternBlacklist scan
    KVM_EXECUTE,       ///< Sandboxed execution in KVM VM
    EBPF_MONITOR,      ///< eBPF escape detection during execution
    ANOMALY_CHECK,     ///< Behavioral anomaly analysis post-execution
    DEPLOYED,          ///< Successfully deployed
    QUARANTINED,       ///< Rejected and isolated
};

inline const char* pipeline_stage_str(PipelineStage s) {
    switch (s) {
        case PipelineStage::SUBMITTED:        return "SUBMITTED";
        case PipelineStage::CSVP_VERIFY:       return "CSVP_VERIFY";
        case PipelineStage::STATIC_ANALYSIS:   return "STATIC_ANALYSIS";
        case PipelineStage::PATTERN_SCAN:      return "PATTERN_SCAN";
        case PipelineStage::KVM_EXECUTE:       return "KVM_EXECUTE";
        case PipelineStage::EBPF_MONITOR:      return "EBPF_MONITOR";
        case PipelineStage::ANOMALY_CHECK:     return "ANOMALY_CHECK";
        case PipelineStage::DEPLOYED:          return "DEPLOYED";
        case PipelineStage::QUARANTINED:       return "QUARANTINED";
    }
    return "UNKNOWN";
}

// ============================================================================
// Pipeline decision record
// ============================================================================

struct PipelineDecision {
    PipelineStage stage;
    bool          passed{false};
    std::string   reason;
    std::chrono::steady_clock::time_point timestamp;
};

// ============================================================================
// Pipeline result — full audit of a single code submission
// ============================================================================

struct PipelineResult {
    std::string                  module_name;
    std::string                  source_hash;    ///< SHA-256 of submitted code
    bool                         deployed{false};
    bool                         quarantined{false};
    PipelineStage                final_stage{PipelineStage::SUBMITTED};
    std::vector<PipelineDecision> decisions;
    std::string                  stdout_data;    ///< VM execution output
    double                       total_time_s{0.0};

    void record(PipelineStage stage, bool passed, const std::string& reason) {
        decisions.push_back({stage, passed, reason,
                             std::chrono::steady_clock::now()});
        final_stage = stage;
    }
};

// ============================================================================
// Pipeline configuration
// ============================================================================

struct PipelineConfig {
    bool     skip_kvm_on_unavailable = true;   ///< Skip KVM if not available
    bool     skip_ebpf_on_unavailable = true;  ///< Skip eBPF if not available
    uint32_t kvm_timeout_ms          = 30'000; ///< VM execution timeout
    double   anomaly_quarantine_threshold = 0.8; ///< Anomaly score for quarantine
};

// ============================================================================
// SecurityPipeline
// ============================================================================

class SecurityPipeline {
public:
    using DeployFn     = std::function<bool(const std::string& module_name,
                                             const std::string& stdout_data)>;
    using QuarantineFn = std::function<void(const std::string& module_name,
                                             const std::string& reason)>;

    /**
     * Construct pipeline with all security components.
     * Components are passed by reference — pipeline does not own them.
     */
    SecurityPipeline(PipelineConfig cfg,
                      CodeSafetyVerifier& csvp,
                      CodePatternBlacklist& blacklist,
                      KvmSandbox& sandbox,
                      EbpfMonitor& ebpf,
                      AnomalyDetector& anomaly)
        : cfg_(std::move(cfg))
        , csvp_(csvp)
        , blacklist_(blacklist)
        , sandbox_(sandbox)
        , ebpf_(ebpf)
        , anomaly_(anomaly)
    {}

    /// Construct with optional MiniVMManager (preferred for pool-based execution)
    SecurityPipeline(PipelineConfig cfg,
                      CodeSafetyVerifier& csvp,
                      CodePatternBlacklist& blacklist,
                      KvmSandbox& sandbox,
                      EbpfMonitor& ebpf,
                      AnomalyDetector& anomaly,
                      MiniVMManager& vm_pool)
        : cfg_(std::move(cfg))
        , csvp_(csvp)
        , blacklist_(blacklist)
        , sandbox_(sandbox)
        , ebpf_(ebpf)
        , anomaly_(anomaly)
        , vm_pool_(&vm_pool)
    {}

    void set_deploy_callback(DeployFn fn)       { on_deploy_ = std::move(fn); }
    void set_quarantine_callback(QuarantineFn fn) { on_quarantine_ = std::move(fn); }

    // ── Main pipeline entry point ────────────────────────────────────────

    /**
     * Run the full security pipeline on a code submission.
     * Returns a complete audit trail of all decisions.
     */
    PipelineResult evaluate(const std::string& module_name,
                             const std::string& source_code) {
        PipelineResult result;
        result.module_name = module_name;
        auto pipeline_start = std::chrono::steady_clock::now();

        // ── Stage 1: CSVP 4-stage static analysis ────────────────────────
        {
            auto csvp_result = csvp_.verify(source_code);
            if (!csvp_result.approved) {
                std::string detail;
                for (const auto& v : csvp_result.violations) {
                    if (!detail.empty()) detail += "; ";
                    detail += v.rule_name + ": " + v.detail;
                }
                result.record(PipelineStage::CSVP_VERIFY, false,
                              "CSVP rejected: " + detail);
                quarantine(result, "CSVP rejection: " + detail);
                finalize(result, pipeline_start);
                return result;
            }
            result.record(PipelineStage::CSVP_VERIFY, true,
                          std::to_string(csvp_result.stages_passed) +
                          "/4 stages passed");
        }

        // ── Stage 2: Pattern blacklist scan ──────────────────────────────
        {
            auto scan = blacklist_.check(source_code);
            if (!scan.safe) {
                std::string patterns;
                for (const auto& v : scan.violations) {
                    if (!patterns.empty()) patterns += ", ";
                    patterns += v.pattern_name;
                }
                result.record(PipelineStage::PATTERN_SCAN, false,
                              "blacklisted patterns: " + patterns);
                quarantine(result, "pattern blacklist: " + patterns);
                finalize(result, pipeline_start);
                return result;
            }
            result.record(PipelineStage::PATTERN_SCAN, true, "no blacklisted patterns");
        }

        // ── Stage 3: KVM sandboxed execution ─────────────────────────────
        ExecutionResult exec_result;
        std::string vm_name;
        {
            if (vm_pool_) {
                // Use MiniVMManager for pooled execution
                auto deploy = vm_pool_->deploy_and_execute(module_name, source_code);
                exec_result.success     = deploy.success;
                exec_result.exit_code   = deploy.success ? 0 : -1;
                exec_result.stdout_data = deploy.stdout_data;
                exec_result.error       = deploy.error;
                exec_result.elapsed_s   = deploy.total_latency_s;
                vm_name = deploy.vm_name;
            } else {
                // Direct KVM sandbox execution
                vm_name = module_name + "_exec";
                bool created = sandbox_.create_vm(vm_name);

                if (!created || !sandbox_.boot(vm_name)) {
                    if (cfg_.skip_kvm_on_unavailable) {
                        result.record(PipelineStage::KVM_EXECUTE, true,
                                      "KVM unavailable — skipped (policy: skip)");
                    } else {
                        result.record(PipelineStage::KVM_EXECUTE, false,
                                      "KVM boot failed");
                        quarantine(result, "KVM boot failure");
                        finalize(result, pipeline_start);
                        return result;
                    }
                } else {
                    sandbox_.inject_code(vm_name, source_code);
                    exec_result = sandbox_.wait_completion(vm_name, cfg_.kvm_timeout_ms);
                }
            }

            if (!exec_result.error.empty() && !exec_result.success) {
                result.record(PipelineStage::KVM_EXECUTE, false,
                              "execution failed: " + exec_result.error);
                if (!vm_name.empty()) sandbox_.destroy(vm_name);
                quarantine(result, "execution failure: " + exec_result.error);
                finalize(result, pipeline_start);
                return result;
            }

            result.stdout_data = exec_result.stdout_data;
            result.record(PipelineStage::KVM_EXECUTE, true,
                          "exit=" + std::to_string(exec_result.exit_code) +
                          " in " + std::to_string(exec_result.elapsed_s) + "s");
        }

        // ── Stage 4: eBPF escape monitoring analysis ─────────────────────
        {
            // Poll any pending eBPF events from the execution
            ebpf_.poll();
            const auto& events = ebpf_.events();

            uint32_t escape_count = 0;
            for (const auto& ev : events) {
                if (ev.vm_name == vm_name) {
                    // Any event from our VM is a potential escape
                    if (ev.type == EbpfEventType::EXECVE_ATTEMPT ||
                        ev.type == EbpfEventType::NETWORK_ATTEMPT ||
                        ev.type == EbpfEventType::PTRACE_ATTEMPT) {
                        ++escape_count;
                    }
                }
            }

            if (escape_count > 0) {
                result.record(PipelineStage::EBPF_MONITOR, false,
                              std::to_string(escape_count) +
                              " escape attempts detected");
                if (!vm_name.empty()) sandbox_.destroy(vm_name);
                quarantine(result, "eBPF: " + std::to_string(escape_count) +
                           " escape attempts");
                finalize(result, pipeline_start);
                return result;
            }

            result.record(PipelineStage::EBPF_MONITOR, true,
                          ebpf_.ebpf_available()
                              ? "no escape attempts (eBPF active)"
                              : "no escape attempts (eBPF: fallback mode)");
        }

        // ── Stage 5: Anomaly detection ───────────────────────────────────
        {
            // Record the execution as an observation
            BehaviorObservation obs;
            obs.duration_s = exec_result.elapsed_s;
            obs.cpu_usage  = 0.1;  // baseline estimate
            obs.timestamp  = std::chrono::steady_clock::now();
            anomaly_.record_observation(module_name, obs);
            auto threats = anomaly_.analyze(module_name);

            // Check if any threat exceeds quarantine threshold
            double max_severity = 0.0;
            std::string worst_detail;
            for (const auto& t : threats) {
                if (t.severity > max_severity) {
                    max_severity = t.severity;
                    worst_detail = t.detail;
                }
            }

            if (max_severity >= cfg_.anomaly_quarantine_threshold) {
                result.record(PipelineStage::ANOMALY_CHECK, false,
                              "anomaly severity " +
                              std::to_string(max_severity) +
                              " exceeds threshold " +
                              std::to_string(cfg_.anomaly_quarantine_threshold));
                anomaly_.quarantine(module_name, QuarantineReason::AUTO_SEVERITY);
                if (!vm_name.empty()) sandbox_.destroy(vm_name);
                quarantine(result, "anomaly: " + worst_detail);
                finalize(result, pipeline_start);
                return result;
            }

            result.record(PipelineStage::ANOMALY_CHECK, true,
                          "max_severity=" + std::to_string(max_severity));
        }

        // ── All stages passed — deploy ───────────────────────────────────
        if (!vm_name.empty()) sandbox_.destroy(vm_name);

        result.deployed = true;
        result.record(PipelineStage::DEPLOYED, true,
                      "all " + std::to_string(result.decisions.size()) +
                      " stages passed");
        ++total_deploys_;

        if (on_deploy_) {
            on_deploy_(module_name, exec_result.stdout_data);
        }

        finalize(result, pipeline_start);
        return result;
    }

    // ── Query ────────────────────────────────────────────────────────────

    uint64_t total_submissions()  const { return total_submissions_; }
    uint64_t total_deployments()  const { return total_deploys_; }
    uint64_t total_quarantines()  const { return total_quarantines_; }

    const std::vector<PipelineResult>& audit_trail() const {
        return audit_trail_;
    }

private:
    PipelineConfig        cfg_;
    CodeSafetyVerifier&   csvp_;
    CodePatternBlacklist& blacklist_;
    KvmSandbox&           sandbox_;
    EbpfMonitor&          ebpf_;
    AnomalyDetector&      anomaly_;
    MiniVMManager*        vm_pool_{nullptr};

    DeployFn     on_deploy_;
    QuarantineFn on_quarantine_;

    uint64_t total_submissions_{0};
    uint64_t total_deploys_{0};
    uint64_t total_quarantines_{0};

    std::vector<PipelineResult> audit_trail_;

    void quarantine(PipelineResult& result, const std::string& reason) {
        result.quarantined = true;
        result.record(PipelineStage::QUARANTINED, false, reason);
        ++total_quarantines_;

        if (on_quarantine_) {
            on_quarantine_(result.module_name, reason);
        }
    }

    void finalize(PipelineResult& result,
                   std::chrono::steady_clock::time_point start) {
        ++total_submissions_;
        result.total_time_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        audit_trail_.push_back(result);
    }
};

} // namespace nikola::security
