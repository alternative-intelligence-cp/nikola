/**
 * @file homeostasis.cpp
 * @brief Phase 129 — HomeostasisMonitor implementation
 */

#include <nikola/security/homeostasis.hpp>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <sstream>

namespace nikola::security {

// ---------------------------------------------------------------------------
// Static helpers
// ---------------------------------------------------------------------------

double HomeostasisMonitor::compute_energy(const NikolaState& state) noexcept {
    const double raw = 0.3 * static_cast<double>(state.dopamine)
                     + 0.4 * static_cast<double>(state.atp)
                     + 0.3 * (1.0 - static_cast<double>(state.boredom));
    return std::clamp(raw, 0.0, 1.0);
}

double HomeostasisMonitor::compute_entropy(const NikolaState& state) noexcept {
    return static_cast<double>(
        std::clamp(state.entropy, 0.0f, 1.0f));
}

double HomeostasisMonitor::compute_severity(double delta,
                                              double tolerance) noexcept {
    if (tolerance <= 0.0) return 1.0;
    return std::clamp(std::fabs(delta) / tolerance, 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

HomeostasisMonitor::HomeostasisMonitor() = default;

HomeostasisMonitor::~HomeostasisMonitor() {
    stop_monitoring();
}

// ---------------------------------------------------------------------------
// Baseline
// ---------------------------------------------------------------------------

void HomeostasisMonitor::set_baseline(const NikolaState& state) {
    baseline_energy_  = compute_energy(state);
    baseline_entropy_ = compute_entropy(state);
    baseline_set_     = true;
}

// ---------------------------------------------------------------------------
// Private: record anomaly
// ---------------------------------------------------------------------------

void HomeostasisMonitor::record_anomaly(AnomalyRecord rec) {
    ++anomaly_count_;

    // FIFO cap
    if (history_.size() >= HSK_MAX_HISTORY) {
        history_.erase(history_.begin());
    }
    history_.push_back(rec);

    if (anomaly_cb_) anomaly_cb_(rec);

    // Auto-lockdown on severe anomaly
    if (rec.severity >= lockdown_threshold_) {
        trigger_lockdown();
    }
}

// ---------------------------------------------------------------------------
// check()
// ---------------------------------------------------------------------------

bool HomeostasisMonitor::check(const NikolaState& state, uint64_t tick) {
    ++check_count_;

    if (!baseline_set_) return true;

    bool ok = true;

    const double cur_energy  = compute_energy(state);
    const double cur_entropy = compute_entropy(state);

    const double energy_delta  = cur_energy  - baseline_energy_;
    const double entropy_delta = cur_entropy - baseline_entropy_;

    // --- Energy anomaly check ---
    if (std::fabs(energy_delta) > energy_tolerance_) {
        ok = false;
        AnomalyRecord r;
        r.type        = energy_delta > 0.0
                       ? AnomalyType::ENERGY_SPIKE
                       : AnomalyType::ENERGY_DROP;
        r.delta       = energy_delta;
        r.severity    = compute_severity(energy_delta, energy_tolerance_);
        r.tick        = tick;

        std::ostringstream oss;
        oss << "Energy " << (energy_delta > 0 ? "spike" : "drop")
            << " delta=" << energy_delta
            << " (baseline=" << baseline_energy_
            << " current=" << cur_energy << ")";
        r.description = oss.str();

        record_anomaly(r);
    }

    // --- Entropy anomaly check ---
    if (std::fabs(entropy_delta) > entropy_tolerance_) {
        ok = false;
        AnomalyRecord r;
        r.type     = entropy_delta > 0.0
                    ? AnomalyType::ENTROPY_SPIKE
                    : AnomalyType::ENTROPY_DROP;
        r.delta    = entropy_delta;
        r.severity = compute_severity(entropy_delta, entropy_tolerance_);
        r.tick     = tick;

        std::ostringstream oss;
        oss << "Entropy " << (entropy_delta > 0 ? "spike" : "drop")
            << " delta=" << entropy_delta
            << " (baseline=" << baseline_entropy_
            << " current=" << cur_entropy << ")";
        r.description = oss.str();

        record_anomaly(r);
    }

    return ok;
}

// ---------------------------------------------------------------------------
// verify_integrity (no side-effects)
// ---------------------------------------------------------------------------

bool HomeostasisMonitor::verify_integrity(const NikolaState& state) const {
    if (!baseline_set_) return true;

    const double cur_energy  = compute_energy(state);
    const double cur_entropy = compute_entropy(state);

    return std::fabs(cur_energy  - baseline_energy_)  <= energy_tolerance_ &&
           std::fabs(cur_entropy - baseline_entropy_) <= entropy_tolerance_;
}

// ---------------------------------------------------------------------------
// Lockdown
// ---------------------------------------------------------------------------

void HomeostasisMonitor::trigger_lockdown() {
    lockdown_.store(true);
}

void HomeostasisMonitor::release_lockdown() {
    lockdown_.store(false);
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

HomeostasisMonitor::Stats HomeostasisMonitor::stats() const {
    Stats s;
    s.has_baseline      = baseline_set_;
    s.baseline_energy   = baseline_energy_;
    s.baseline_entropy  = baseline_entropy_;
    s.total_checks      = check_count_;
    s.total_anomalies   = anomaly_count_;
    s.locked_down       = lockdown_.load();
    s.monitoring_active = running_.load();
    return s;
}

// ---------------------------------------------------------------------------
// Continuous monitoring
// ---------------------------------------------------------------------------

void HomeostasisMonitor::monitor_loop(std::function<NikolaState()> provider,
                                       uint64_t interval_ms) {
    uint64_t tick = 0;
    while (running_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
        if (running_.load()) {
            const NikolaState st = provider();
            check(st, tick++);
        }
    }
}

void HomeostasisMonitor::start_monitoring(
    std::function<NikolaState()> state_provider,
    uint64_t interval_ms) {
    if (running_.load()) return;

    running_.store(true);
    monitor_thread_ = std::thread(
        &HomeostasisMonitor::monitor_loop, this,
        std::move(state_provider), interval_ms);
}

void HomeostasisMonitor::stop_monitoring() {
    running_.store(false);
    if (monitor_thread_.joinable()) {
        monitor_thread_.join();
    }
}

} // namespace nikola::security
