/**
 * @file security/anomaly_detector.hpp
 * @brief v0.1.19 — Behavioral anomaly detection and module quarantine
 *
 * Monitors running modules for anomalous behavior — resource usage spikes,
 * unusual syscall patterns, and unexpected communication patterns.
 *
 * This extends HomeostasisMonitor (which tracks energy/entropy of NikolaState)
 * with per-module behavioral profiling and automatic quarantine.
 *
 * Detection layers:
 *   1. Resource anomaly  — CPU/memory usage outside 3σ baseline
 *   2. Syscall anomaly   — unusual syscall frequency or blocked calls
 *   3. Communication anomaly — unexpected message targets or high volume
 *   4. Temporal anomaly   — activity at unusual times or unusual duration
 *
 * On detection:
 *   severity < 0.5  → LOG and continue
 *   severity < 0.8  → ALERT and restrict
 *   severity >= 0.8 → QUARANTINE (isolate module immediately)
 *
 * Usage:
 *   AnomalyDetector detector;
 *   detector.register_module("self_improve_1");
 *   detector.record_observation("self_improve_1", obs);
 *   auto threats = detector.analyze("self_improve_1");
 *   if (detector.is_quarantined("self_improve_1")) { ... }
 */
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nikola::security {

// ============================================================================
// Constants
// ============================================================================

inline constexpr double  ANOMALY_LOG_THRESHOLD        = 0.3;
inline constexpr double  ANOMALY_ALERT_THRESHOLD      = 0.5;
inline constexpr double  ANOMALY_QUARANTINE_THRESHOLD  = 0.8;
inline constexpr size_t  ANOMALY_BASELINE_WINDOW       = 100;   // observations
inline constexpr size_t  ANOMALY_MAX_HISTORY           = 1024;  // per module
inline constexpr double  ANOMALY_SIGMA_MULTIPLIER      = 3.0;   // 3σ rule

// ============================================================================
// Observation record
// ============================================================================

struct BehaviorObservation {
    double      cpu_usage{0.0};       ///< [0, 1] CPU utilization
    double      memory_usage{0.0};    ///< [0, 1] memory utilization
    uint64_t    syscall_count{0};     ///< syscalls in observation window
    uint64_t    message_count{0};     ///< inter-module messages sent
    uint64_t    file_ops{0};          ///< file open/read/write operations
    double      duration_s{0.0};      ///< observation window duration
    std::chrono::steady_clock::time_point timestamp;

    BehaviorObservation() : timestamp(std::chrono::steady_clock::now()) {}
};

// ============================================================================
// Threat classification
// ============================================================================

enum class ThreatType : uint8_t {
    RESOURCE_SPIKE,        ///< CPU or memory usage > 3σ above baseline
    RESOURCE_SUSTAINED,    ///< Elevated usage over multiple windows
    SYSCALL_ANOMALY,       ///< Unusual syscall count / pattern
    COMMUNICATION_FLOOD,   ///< Excessive inter-module messaging
    FILE_ACCESS_BURST,     ///< Unusual number of file operations
    TEMPORAL_ANOMALY,      ///< Activity outside expected patterns
};

inline const char* threat_type_str(ThreatType t) {
    switch (t) {
        case ThreatType::RESOURCE_SPIKE:      return "RESOURCE_SPIKE";
        case ThreatType::RESOURCE_SUSTAINED:  return "RESOURCE_SUSTAINED";
        case ThreatType::SYSCALL_ANOMALY:     return "SYSCALL_ANOMALY";
        case ThreatType::COMMUNICATION_FLOOD: return "COMMUNICATION_FLOOD";
        case ThreatType::FILE_ACCESS_BURST:   return "FILE_ACCESS_BURST";
        case ThreatType::TEMPORAL_ANOMALY:    return "TEMPORAL_ANOMALY";
    }
    return "UNKNOWN";
}

// ============================================================================
// Threat record
// ============================================================================

struct ThreatRecord {
    std::string module_name;
    ThreatType  type;
    double      severity{0.0};   ///< [0, 1]
    double      deviation{0.0};  ///< how many σ above baseline
    std::string detail;
    std::chrono::steady_clock::time_point timestamp;

    ThreatRecord()
        : type(ThreatType::RESOURCE_SPIKE)
        , timestamp(std::chrono::steady_clock::now()) {}
};

// ============================================================================
// Quarantine status
// ============================================================================

enum class QuarantineReason : uint8_t {
    NOT_QUARANTINED,
    AUTO_SEVERITY,       ///< Automatic: severity >= threshold
    MANUAL,              ///< Operator-initiated quarantine
    ESCAPE_ATTEMPT,      ///< Linked to EscapeDetector alert
};

inline const char* quarantine_reason_str(QuarantineReason r) {
    switch (r) {
        case QuarantineReason::NOT_QUARANTINED: return "NOT_QUARANTINED";
        case QuarantineReason::AUTO_SEVERITY:   return "AUTO_SEVERITY";
        case QuarantineReason::MANUAL:          return "MANUAL";
        case QuarantineReason::ESCAPE_ATTEMPT:  return "ESCAPE_ATTEMPT";
    }
    return "UNKNOWN";
}

// ============================================================================
// Module behavioral profile
// ============================================================================

struct ModuleProfile {
    std::string name;
    bool        quarantined{false};
    QuarantineReason quarantine_reason{QuarantineReason::NOT_QUARANTINED};

    // Baseline statistics (running mean/variance via Welford's algorithm)
    struct RunningStats {
        double mean{0.0};
        double m2{0.0};      ///< Sum of squared differences from mean
        size_t count{0};

        void update(double value) {
            ++count;
            double delta = value - mean;
            mean += delta / static_cast<double>(count);
            double delta2 = value - mean;
            m2 += delta * delta2;
        }

        double variance() const {
            if (count < 2) return 0.0;
            return m2 / static_cast<double>(count - 1);
        }

        double stddev() const { return std::sqrt(variance()); }

        /// How many σ is value from the mean?
        double z_score(double value) const {
            double sd = stddev();
            if (sd < 1e-12) return 0.0;
            return (value - mean) / sd;
        }
    };

    RunningStats cpu_stats;
    RunningStats mem_stats;
    RunningStats syscall_stats;
    RunningStats msg_stats;
    RunningStats file_stats;

    std::deque<BehaviorObservation> history;
    std::vector<ThreatRecord>       threats;
    uint64_t total_observations{0};
    uint64_t total_threats{0};

    void record(const BehaviorObservation& obs) {
        cpu_stats.update(obs.cpu_usage);
        mem_stats.update(obs.memory_usage);
        syscall_stats.update(static_cast<double>(obs.syscall_count));
        msg_stats.update(static_cast<double>(obs.message_count));
        file_stats.update(static_cast<double>(obs.file_ops));

        history.push_back(obs);
        if (history.size() > ANOMALY_MAX_HISTORY)
            history.pop_front();
        ++total_observations;
    }
};

// ============================================================================
// AnomalyDetector — behavioral monitoring + quarantine
// ============================================================================

class AnomalyDetector {
public:
    using AlertCallback     = std::function<void(const ThreatRecord&)>;
    using QuarantineCallback = std::function<void(const std::string& module,
                                                   QuarantineReason reason)>;

    struct Config {
        double sigma_multiplier     = ANOMALY_SIGMA_MULTIPLIER;
        double log_threshold        = ANOMALY_LOG_THRESHOLD;
        double alert_threshold      = ANOMALY_ALERT_THRESHOLD;
        double quarantine_threshold = ANOMALY_QUARANTINE_THRESHOLD;
        size_t min_baseline_samples = 10;  // need this many before alerting
    };

    AnomalyDetector() : cfg_{} {}
    explicit AnomalyDetector(Config cfg) : cfg_(std::move(cfg)) {}

    // ── Module registration ─────────────────────────────────────────────────

    bool register_module(const std::string& name) {
        if (modules_.count(name)) return false;
        ModuleProfile prof;
        prof.name = name;
        modules_[name] = std::move(prof);
        return true;
    }

    bool unregister_module(const std::string& name) {
        return modules_.erase(name) > 0;
    }

    size_t module_count() const { return modules_.size(); }

    // ── Observation recording ───────────────────────────────────────────────

    bool record_observation(const std::string& name,
                            const BehaviorObservation& obs)
    {
        auto it = modules_.find(name);
        if (it == modules_.end()) return false;
        it->second.record(obs);
        return true;
    }

    // ── Analysis ────────────────────────────────────────────────────────────

    /**
     * Analyze a module's recent behavior against its baseline.
     * Returns any detected threats.
     */
    std::vector<ThreatRecord> analyze(const std::string& name) {
        std::vector<ThreatRecord> threats;
        auto it = modules_.find(name);
        if (it == modules_.end()) return threats;

        auto& prof = it->second;
        if (prof.total_observations < cfg_.min_baseline_samples) return threats;
        if (prof.history.empty()) return threats;

        const auto& latest = prof.history.back();

        // Check CPU anomaly
        check_metric(prof, "cpu_usage", latest.cpu_usage,
                     prof.cpu_stats, ThreatType::RESOURCE_SPIKE, threats);

        // Check memory anomaly
        check_metric(prof, "memory_usage", latest.memory_usage,
                     prof.mem_stats, ThreatType::RESOURCE_SPIKE, threats);

        // Check syscall anomaly
        check_metric(prof, "syscall_count",
                     static_cast<double>(latest.syscall_count),
                     prof.syscall_stats, ThreatType::SYSCALL_ANOMALY, threats);

        // Check communication flood
        check_metric(prof, "message_count",
                     static_cast<double>(latest.message_count),
                     prof.msg_stats, ThreatType::COMMUNICATION_FLOOD, threats);

        // Check file access burst
        check_metric(prof, "file_ops",
                     static_cast<double>(latest.file_ops),
                     prof.file_stats, ThreatType::FILE_ACCESS_BURST, threats);

        // Check sustained resource usage (3+ consecutive above 2σ)
        if (prof.history.size() >= 3) {
            size_t n = prof.history.size();
            int consecutive_high = 0;
            for (size_t i = n > 3 ? n - 3 : 0; i < n; ++i) {
                double z = prof.cpu_stats.z_score(prof.history[i].cpu_usage);
                if (z > 2.0) ++consecutive_high;
            }
            if (consecutive_high >= 3) {
                ThreatRecord t;
                t.module_name = name;
                t.type        = ThreatType::RESOURCE_SUSTAINED;
                t.severity    = 0.7;
                t.deviation   = 2.0;
                t.detail      = "CPU above 2σ for 3+ consecutive windows";
                threats.push_back(std::move(t));
            }
        }

        // Store threats and handle quarantine
        for (auto& t : threats) {
            prof.threats.push_back(t);
            ++prof.total_threats;

            if (t.severity >= cfg_.quarantine_threshold && !prof.quarantined) {
                quarantine(name, QuarantineReason::AUTO_SEVERITY);
            } else if (t.severity >= cfg_.alert_threshold && on_alert_) {
                on_alert_(t);
            }
        }

        return threats;
    }

    /**
     * Analyze all registered modules.
     */
    std::vector<ThreatRecord> analyze_all() {
        std::vector<ThreatRecord> all;
        for (auto& [name, _] : modules_) {
            auto threats = analyze(name);
            all.insert(all.end(), threats.begin(), threats.end());
        }
        return all;
    }

    // ── Quarantine management ───────────────────────────────────────────────

    bool quarantine(const std::string& name, QuarantineReason reason) {
        auto it = modules_.find(name);
        if (it == modules_.end()) return false;
        it->second.quarantined       = true;
        it->second.quarantine_reason = reason;
        quarantined_.insert(name);
        if (on_quarantine_) on_quarantine_(name, reason);
        return true;
    }

    bool release(const std::string& name) {
        auto it = modules_.find(name);
        if (it == modules_.end()) return false;
        it->second.quarantined       = false;
        it->second.quarantine_reason = QuarantineReason::NOT_QUARANTINED;
        quarantined_.erase(name);
        return true;
    }

    bool is_quarantined(const std::string& name) const {
        return quarantined_.count(name) > 0;
    }

    size_t quarantined_count() const { return quarantined_.size(); }

    std::vector<std::string> quarantined_modules() const {
        return {quarantined_.begin(), quarantined_.end()};
    }

    // ── Query ───────────────────────────────────────────────────────────────

    const ModuleProfile* get_profile(const std::string& name) const {
        auto it = modules_.find(name);
        return (it != modules_.end()) ? &it->second : nullptr;
    }

    uint64_t total_threats_detected() const {
        uint64_t n = 0;
        for (const auto& [_, p] : modules_) n += p.total_threats;
        return n;
    }

    // ── Callbacks ───────────────────────────────────────────────────────────

    void set_alert_callback(AlertCallback cb) { on_alert_ = std::move(cb); }
    void set_quarantine_callback(QuarantineCallback cb) { on_quarantine_ = std::move(cb); }

private:
    Config cfg_;
    std::unordered_map<std::string, ModuleProfile> modules_;
    std::unordered_set<std::string>                quarantined_;
    AlertCallback       on_alert_;
    QuarantineCallback  on_quarantine_;

    void check_metric(ModuleProfile& prof,
                      const std::string& metric_name,
                      double value,
                      const ModuleProfile::RunningStats& stats,
                      ThreatType type,
                      std::vector<ThreatRecord>& threats)
    {
        double z = stats.z_score(value);
        if (z <= cfg_.sigma_multiplier) return;

        // Severity scales from 0.3 (at 3σ) to 1.0 (at 6σ+)
        double severity = std::min(1.0, 0.3 + (z - cfg_.sigma_multiplier) * 0.233);

        ThreatRecord t;
        t.module_name = prof.name;
        t.type        = type;
        t.severity    = severity;
        t.deviation   = z;
        t.detail      = metric_name + " at " +
                        std::to_string(z).substr(0, 4) + "σ (value=" +
                        std::to_string(value).substr(0, 8) + ")";
        threats.push_back(std::move(t));
    }
};

} // namespace nikola::security
