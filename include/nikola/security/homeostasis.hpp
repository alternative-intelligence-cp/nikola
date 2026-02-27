#pragma once
/**
 * @file homeostasis.hpp
 * @brief Phase 129 — HomeostasisMonitor: NikolaState energy/entropy watchdog
 *
 * Monitors system "energy" and entropy derived from NikolaState to detect
 * anomalies — spikes, drops, or sustained drift that would indicate intrusion
 * or destabilisation.
 *
 * Energy model (scalar):
 *   system_energy = 0.3 * dopamine + 0.4 * atp + 0.3 * (1 - boredom)
 *
 * Entropy:
 *   system_entropy = NikolaState::entropy
 *
 * Anomaly triggers:
 *   |current_energy  - baseline_energy|  > energy_tolerance
 *   |current_entropy - baseline_entropy| > entropy_tolerance
 *
 * On anomaly:  AnomalyRecord is appended to history; on_anomaly callback
 *              fires; if severity >= lockdown_threshold → trigger_lockdown()
 *
 * No TorusManifold / Coord9D / Eigen dependencies.
 *
 * Key constants:
 *  HSK_ENERGY_TOLERANCE    0.25   default allowed energy drift
 *  HSK_ENTROPY_TOLERANCE   0.30   default allowed entropy drift
 *  HSK_LOCKDOWN_THRESHOLD  0.80   severity level that triggers auto-lockdown
 *  HSK_MAX_HISTORY         512    FIFO cap on anomaly history
 */

#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include <nikola/autonomy/decision_loop.hpp>

namespace nikola::security {

using nikola::autonomy::NikolaState;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

inline constexpr double  HSK_ENERGY_TOLERANCE    = 0.25;
inline constexpr double  HSK_ENTROPY_TOLERANCE   = 0.30;
inline constexpr double  HSK_LOCKDOWN_THRESHOLD  = 0.80;
inline constexpr size_t  HSK_MAX_HISTORY         = 512;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

enum class AnomalyType {
    ENERGY_SPIKE,    ///< energy >> baseline
    ENERGY_DROP,     ///< energy << baseline
    ENTROPY_SPIKE,   ///< entropy >> baseline
    ENTROPY_DROP,    ///< entropy << baseline
};

struct AnomalyRecord {
    AnomalyType  type;
    double       severity    = 0.0;   ///< [0, 1] — 1 = maximum deviation
    double       delta       = 0.0;   ///< signed deviation from baseline
    uint64_t     tick        = 0;
    std::string  description;
};

// ---------------------------------------------------------------------------
// HomeostasisMonitor
// ---------------------------------------------------------------------------

class HomeostasisMonitor {
public:
    HomeostasisMonitor();
    ~HomeostasisMonitor();

    // --- Baseline -----------------------------------------------------------

    /**
     * @brief Record baseline energy/entropy from an initial stable NikolaState.
     * Must be called before check() for meaningful results.
     */
    void set_baseline(const NikolaState& state);

    bool has_baseline() const { return baseline_set_; }

    // --- Checks -------------------------------------------------------------

    /**
     * @brief Check current state against baseline.
     * Appends anomalies to history.  Returns true if ALL checks pass (no
     * anomaly detected) or if no baseline has been set.
     */
    bool check(const NikolaState& state, uint64_t tick = 0);

    /**
     * @brief Convenience: verify_integrity = check() with no history side-effects.
     * Returns true if no anomaly would be detected.
     */
    bool verify_integrity(const NikolaState& state) const;

    // --- Lockdown -----------------------------------------------------------

    void trigger_lockdown();
    void release_lockdown();
    bool is_locked_down() const { return lockdown_.load(); }

    // --- Continuous monitoring ----------------------------------------------

    /**
     * @brief Start background thread that calls check() every interval_ms.
     * @param state_provider  Callable returning the current NikolaState.
     * @param interval_ms     Polling interval (ms).
     */
    void start_monitoring(std::function<NikolaState()> state_provider,
                          uint64_t interval_ms = 1000);

    /**
     * @brief Stop the background monitoring thread.
     */
    void stop_monitoring();

    bool is_monitoring() const { return running_.load(); }

    // --- Tolerance tuning ---------------------------------------------------

    void set_energy_tolerance (double t) { energy_tolerance_  = t; }
    void set_entropy_tolerance(double t) { entropy_tolerance_ = t; }
    void set_lockdown_threshold(double t) { lockdown_threshold_ = t; }

    double energy_tolerance()  const { return energy_tolerance_;  }
    double entropy_tolerance() const { return entropy_tolerance_; }

    // --- History / stats ----------------------------------------------------

    const std::vector<AnomalyRecord>& anomaly_history() const { return history_; }
    void                              clear_history()         { history_.clear(); }

    size_t check_count()   const { return check_count_;   }
    size_t anomaly_count() const { return anomaly_count_; }

    struct Stats {
        bool   has_baseline      = false;
        double baseline_energy   = 0.0;
        double baseline_entropy  = 0.0;
        size_t total_checks      = 0;
        size_t total_anomalies   = 0;
        bool   locked_down       = false;
        bool   monitoring_active = false;
    };

    Stats stats() const;

    // --- Callback -----------------------------------------------------------

    using AnomalyCallback = std::function<void(const AnomalyRecord&)>;
    void on_anomaly(AnomalyCallback cb) { anomaly_cb_ = std::move(cb); }

    // --- Static helpers -----------------------------------------------------

    /**
     * @brief Compute scalar energy from NikolaState.
     *   energy = 0.3*dopamine + 0.4*atp + 0.3*(1 - boredom), clamped [0,1]
     */
    static double compute_energy(const NikolaState& state) noexcept;

    /**
     * @brief Entropy = NikolaState::entropy, clamped [0,1].
     */
    static double compute_entropy(const NikolaState& state) noexcept;

    /**
     * @brief Normalised severity of a delta: |delta| / tolerance, clamped [0,1].
     */
    static double compute_severity(double delta, double tolerance) noexcept;

private:
    double  baseline_energy_  = 0.0;
    double  baseline_entropy_ = 0.0;
    bool    baseline_set_     = false;

    double  energy_tolerance_   = HSK_ENERGY_TOLERANCE;
    double  entropy_tolerance_  = HSK_ENTROPY_TOLERANCE;
    double  lockdown_threshold_ = HSK_LOCKDOWN_THRESHOLD;

    std::atomic<bool> lockdown_{false};
    std::atomic<bool> running_{false};
    std::thread       monitor_thread_;

    std::vector<AnomalyRecord> history_;
    size_t check_count_   = 0;
    size_t anomaly_count_ = 0;

    AnomalyCallback anomaly_cb_;

    void monitor_loop(std::function<NikolaState()> provider, uint64_t interval_ms);
    void record_anomaly(AnomalyRecord rec);
};

} // namespace nikola::security
