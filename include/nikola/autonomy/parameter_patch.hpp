/**
 * @file autonomy/parameter_patch.hpp
 * @brief v0.2.6 — Tier 2 Parameter Patch Interface
 *
 * Constrained key-value tunable API for safe in-place parameter updates.
 * SIE Tier 2 modifications: no code execution, just tunable parameter
 * adjustments applied during NAP (enforced by NapOrchestrator).
 *
 * Safety guarantees:
 *   - Whitelist of allowed tunables (no arbitrary key injection)
 *   - Hard bounds per tunable (cannot set learning rate to 1000)
 *   - Patches apply ONLY during NAP state (rejected otherwise)
 *   - Automatic rollback: monitor N ticks post-wake, revert if degraded
 *   - Full audit trail of all patch attempts
 *
 * Usage:
 *   ParameterPatchInterface patcher;
 *   patcher.register_tunable("learning_rate", 0.001, {0.0001, 0.1});
 *   patcher.register_tunable("sigma_threshold", 3.0, {1.0, 10.0});
 *
 *   Patch p{{"learning_rate", 0.005}};
 *   auto result = patcher.apply(p, NapState::NAPPING);  // succeeds
 *   auto result = patcher.apply(p, NapState::AWAKE);    // rejected
 *
 *   patcher.begin_monitoring(1000);  // monitor 1000 ticks post-wake
 *   // ... if metric degrades ...
 *   patcher.rollback();              // restore pre-patch values
 */
#pragma once

#include <chrono>
#include <functional>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace nikola::autonomy {

// ============================================================================
// Constants
// ============================================================================

inline constexpr uint32_t PATCH_DEFAULT_MONITOR_TICKS = 1000;
inline constexpr double   PATCH_DEGRADATION_THRESHOLD = 0.05; ///< 5% degradation triggers rollback

// ============================================================================
// Tunable bounds
// ============================================================================

struct TunableBounds {
    double min_val{0.0};
    double max_val{1.0};

    bool contains(double v) const { return v >= min_val && v <= max_val; }
};

// ============================================================================
// Tunable registry entry
// ============================================================================

struct TunableEntry {
    std::string   name;
    double        current_value{0.0};
    double        default_value{0.0};   ///< Original value (for full reset)
    double        pre_patch_value{0.0}; ///< Value before most recent patch
    TunableBounds bounds;
    uint32_t      patch_count{0};       ///< How many times this was patched
};

// ============================================================================
// Patch — a set of key-value changes
// ============================================================================

struct PatchEntry {
    std::string key;
    double      value;
};

struct Patch {
    std::vector<PatchEntry> entries;
    std::string             reason;      ///< SIE rationale for the patch
    std::string             source;      ///< "SIE", "manual", etc.
    std::chrono::steady_clock::time_point timestamp;

    Patch() : timestamp(std::chrono::steady_clock::now()) {}
    explicit Patch(std::vector<PatchEntry> e, std::string r = "",
                   std::string s = "SIE")
        : entries(std::move(e)), reason(std::move(r)),
          source(std::move(s)), timestamp(std::chrono::steady_clock::now()) {}
};

// ============================================================================
// Patch result
// ============================================================================

enum class PatchOutcome : uint8_t {
    APPLIED,            ///< All entries applied successfully
    REJECTED_NOT_NAP,   ///< Rejected: system not in NAP state
    REJECTED_UNKNOWN_KEY, ///< Rejected: key not in whitelist
    REJECTED_OUT_OF_BOUNDS, ///< Rejected: value outside allowed bounds
    REJECTED_EMPTY,     ///< Rejected: no entries in patch
    ROLLED_BACK,        ///< Was applied, then rolled back due to degradation
};

inline const char* patch_outcome_str(PatchOutcome o) {
    switch (o) {
        case PatchOutcome::APPLIED:               return "APPLIED";
        case PatchOutcome::REJECTED_NOT_NAP:      return "REJECTED_NOT_NAP";
        case PatchOutcome::REJECTED_UNKNOWN_KEY:  return "REJECTED_UNKNOWN_KEY";
        case PatchOutcome::REJECTED_OUT_OF_BOUNDS: return "REJECTED_OUT_OF_BOUNDS";
        case PatchOutcome::REJECTED_EMPTY:        return "REJECTED_EMPTY";
        case PatchOutcome::ROLLED_BACK:           return "ROLLED_BACK";
    }
    return "UNKNOWN";
}

struct PatchResult {
    PatchOutcome outcome{PatchOutcome::REJECTED_EMPTY};
    std::string  rejected_key;    ///< Which key caused rejection (if any)
    double       rejected_value{0.0};
    std::string  error;

    bool applied() const { return outcome == PatchOutcome::APPLIED; }
};

// ============================================================================
// Audit entry
// ============================================================================

struct PatchAuditEntry {
    Patch        patch;
    PatchResult  result;
    std::chrono::steady_clock::time_point timestamp;
};

// ============================================================================
// NapState (mirror from nap_controller.hpp to avoid circular include)
// ============================================================================

enum class PatchNapState : uint8_t {
    AWAKE   = 0,
    NAPPING = 1,
};

// ============================================================================
// ParameterPatchInterface
// ============================================================================

class ParameterPatchInterface {
public:
    using MetricFn    = std::function<double()>;  ///< Returns current quality metric
    using RollbackFn  = std::function<void()>;    ///< Called on rollback (optional hook)

    ParameterPatchInterface() = default;

    // ── Tunable registration ─────────────────────────────────────────────

    /**
     * Register a tunable parameter with its default value and bounds.
     * Must be called before any patches referencing this key.
     */
    void register_tunable(const std::string& name, double default_val,
                           TunableBounds bounds) {
        TunableEntry entry;
        entry.name            = name;
        entry.current_value   = default_val;
        entry.default_value   = default_val;
        entry.pre_patch_value = default_val;
        entry.bounds          = bounds;
        tunables_[name] = std::move(entry);
    }

    /**
     * Check if a tunable is registered.
     */
    bool has_tunable(const std::string& name) const {
        return tunables_.count(name) > 0;
    }

    /**
     * Get current value of a tunable. Returns NaN if not found.
     */
    double get(const std::string& name) const {
        auto it = tunables_.find(name);
        if (it == tunables_.end())
            return std::numeric_limits<double>::quiet_NaN();
        return it->second.current_value;
    }

    /**
     * Get all registered tunables.
     */
    const std::unordered_map<std::string, TunableEntry>& tunables() const {
        return tunables_;
    }

    // ── Patch application ────────────────────────────────────────────────

    /**
     * Apply a patch. Only succeeds if system is in NAP state and all
     * keys are whitelisted with values within bounds.
     */
    PatchResult apply(const Patch& patch, PatchNapState nap_state) {
        PatchResult result;

        // Gate 1: must be in NAP
        if (nap_state != PatchNapState::NAPPING) {
            result.outcome = PatchOutcome::REJECTED_NOT_NAP;
            result.error   = "patches only allowed during NAP";
            record_audit(patch, result);
            return result;
        }

        // Gate 2: non-empty
        if (patch.entries.empty()) {
            result.outcome = PatchOutcome::REJECTED_EMPTY;
            result.error   = "empty patch";
            record_audit(patch, result);
            return result;
        }

        // Gate 3: validate all entries before applying any
        for (const auto& e : patch.entries) {
            auto it = tunables_.find(e.key);
            if (it == tunables_.end()) {
                result.outcome       = PatchOutcome::REJECTED_UNKNOWN_KEY;
                result.rejected_key  = e.key;
                result.error         = "unknown tunable: " + e.key;
                record_audit(patch, result);
                return result;
            }
            if (!it->second.bounds.contains(e.value)) {
                result.outcome        = PatchOutcome::REJECTED_OUT_OF_BOUNDS;
                result.rejected_key   = e.key;
                result.rejected_value = e.value;
                result.error          = "value " + std::to_string(e.value) +
                                        " outside bounds [" +
                                        std::to_string(it->second.bounds.min_val) +
                                        ", " +
                                        std::to_string(it->second.bounds.max_val) + "]";
                record_audit(patch, result);
                return result;
            }
        }

        // All valid — save pre-patch values and apply
        for (const auto& e : patch.entries) {
            auto& t = tunables_[e.key];
            t.pre_patch_value = t.current_value;
            t.current_value   = e.value;
            ++t.patch_count;
        }

        last_patch_ = patch;
        has_pending_patch_ = true;
        ++total_applied_;

        result.outcome = PatchOutcome::APPLIED;
        record_audit(patch, result);
        return result;
    }

    // ── Rollback ─────────────────────────────────────────────────────────

    /**
     * Rollback the most recently applied patch.
     * Restores all tunables to their pre-patch values.
     */
    bool rollback() {
        if (!has_pending_patch_) return false;

        for (const auto& e : last_patch_.entries) {
            auto it = tunables_.find(e.key);
            if (it != tunables_.end()) {
                it->second.current_value = it->second.pre_patch_value;
            }
        }

        has_pending_patch_ = false;
        ++total_rollbacks_;

        // Record rollback in audit
        PatchResult rb_result;
        rb_result.outcome = PatchOutcome::ROLLED_BACK;
        record_audit(last_patch_, rb_result);

        if (on_rollback_) on_rollback_();

        return true;
    }

    /**
     * Reset all tunables to their original default values.
     */
    void reset_all() {
        for (auto& [_, t] : tunables_) {
            t.current_value   = t.default_value;
            t.pre_patch_value = t.default_value;
        }
        has_pending_patch_ = false;
    }

    // ── Monitoring ───────────────────────────────────────────────────────

    /**
     * Begin post-wake monitoring. Call this when exiting NAP.
     * The metric_fn should return a quality score (higher = better).
     */
    void begin_monitoring(MetricFn metric_fn,
                           uint32_t monitor_ticks = PATCH_DEFAULT_MONITOR_TICKS) {
        metric_fn_      = std::move(metric_fn);
        monitor_ticks_  = monitor_ticks;
        tick_count_     = 0;
        monitoring_     = true;

        if (metric_fn_)
            baseline_metric_ = metric_fn_();
    }

    /**
     * Call each tick during monitoring period.
     * Returns true if monitoring is still active.
     * Automatically triggers rollback if degradation detected.
     */
    bool monitor_tick() {
        if (!monitoring_) return false;

        ++tick_count_;

        if (tick_count_ >= monitor_ticks_) {
            // Monitoring complete — check final metric
            if (metric_fn_ && has_pending_patch_) {
                double current = metric_fn_();
                double degradation = (baseline_metric_ - current) / baseline_metric_;
                if (degradation > PATCH_DEGRADATION_THRESHOLD) {
                    rollback();
                }
            }
            monitoring_ = false;
            has_pending_patch_ = false;  // committed
            return false;
        }

        // Periodic check (every 10% of monitor window)
        if (tick_count_ % (monitor_ticks_ / 10 + 1) == 0) {
            if (metric_fn_ && has_pending_patch_) {
                double current = metric_fn_();
                double degradation = (baseline_metric_ - current) / baseline_metric_;
                if (degradation > PATCH_DEGRADATION_THRESHOLD * 2.0) {
                    // Severe degradation — immediate rollback
                    rollback();
                    monitoring_ = false;
                    return false;
                }
            }
        }

        return true;
    }

    // ── Callbacks ────────────────────────────────────────────────────────

    void set_rollback_callback(RollbackFn fn) { on_rollback_ = std::move(fn); }

    // ── Query ────────────────────────────────────────────────────────────

    bool     is_monitoring()     const { return monitoring_; }
    bool     has_pending_patch() const { return has_pending_patch_; }
    uint64_t total_applied()     const { return total_applied_; }
    uint64_t total_rollbacks()   const { return total_rollbacks_; }
    uint32_t tick_count()        const { return tick_count_; }

    const std::vector<PatchAuditEntry>& audit_log() const { return audit_log_; }

private:
    std::unordered_map<std::string, TunableEntry> tunables_;

    Patch    last_patch_;
    bool     has_pending_patch_{false};

    // Monitoring state
    MetricFn metric_fn_;
    uint32_t monitor_ticks_{PATCH_DEFAULT_MONITOR_TICKS};
    uint32_t tick_count_{0};
    double   baseline_metric_{0.0};
    bool     monitoring_{false};

    // Callbacks
    RollbackFn on_rollback_;

    // Stats
    uint64_t total_applied_{0};
    uint64_t total_rollbacks_{0};

    // Audit
    std::vector<PatchAuditEntry> audit_log_;

    void record_audit(const Patch& patch, const PatchResult& result) {
        audit_log_.push_back({patch, result, std::chrono::steady_clock::now()});
    }
};

} // namespace nikola::autonomy
