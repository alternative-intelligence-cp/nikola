/**
 * @file latency_budget.hpp
 * @brief GAP-025: End-to-End Latency Budget Allocation for 1000 Hz Physics Loop
 *
 * @spec FABRICATION-READY — docs/info/integration/sections/02_foundations/
 *       02_wave_interference_physics.md §GAP-025
 *
 * Central constraint: Split-Operator Symplectic Integrator requires physics
 * ticks to complete in ≤1000 μs (CFL condition + Hamiltonian conservation).
 * Exceeding this introduces numerical dispersion → "epileptic resonance"
 * (wavefunction amplitude diverges, destroys encoded memories).
 *
 * Budget hierarchy:
 *   Total:       1000 μs  (1 ms, hard physical limit)
 *   Safety:       100 μs  (10% for OS jitter / interrupt handling)
 *   Allocatable:  900 μs  → distributed across 4 critical-path components
 *
 * Component breakdown (spec §GAP-025 "Critical Path"):
 *   Physics Kernel (wave propagation):   600 μs  (66.6%)
 *   Cognitive Scanner (Mamba-9D SSM):    200 μs  (22.2%)
 *   ENGS Neurochemical Gating:            50 μs   (5.5%)
 *   Infrastructure / IPC (Seqlock+ZMQ):   50 μs   (5.5%)
 *
 * Policy: "Drop or Degrade" — no buffering permitted (breaks symplectic
 * time-reversibility). Under pressure: skip nonlinear soliton step first,
 * then drop the entire frame if still over budget.
 */
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

namespace nikola::physics {

// ---------------------------------------------------------------------------
// Timing constants (all in microseconds unless postfix says _NS)
// ---------------------------------------------------------------------------
inline constexpr double TICK_BUDGET_US         = 1'000.0;  ///< 1 ms hard limit
inline constexpr double TICK_SAFETY_MARGIN_US  =   100.0;  ///< OS jitter reserve
inline constexpr double TICK_ALLOCATABLE_US    =   900.0;  ///< distributed budget

/// Component budgets (μs) — must sum to TICK_ALLOCATABLE_US
inline constexpr double BUDGET_PHYSICS_KERNEL_US  = 600.0;
inline constexpr double BUDGET_SCANNER_US         = 200.0;
inline constexpr double BUDGET_ENGS_US            =  50.0;
inline constexpr double BUDGET_INFRASTRUCTURE_US  =  50.0;

/// Alert thresholds (nanoseconds for direct comparison with clock_gettime)
inline constexpr int64_t TICK_WARNING_NS  =  950'000;  ///< 950 μs
inline constexpr int64_t TICK_CRITICAL_NS = 1'050'000; ///< 1050 μs

/// Same thresholds in microseconds for convenience
inline constexpr double TICK_WARNING_US  =  950.0;
inline constexpr double TICK_CRITICAL_US = 1'050.0;

/// Energy drift thresholds (relative: |ΔH / H|)
inline constexpr double ENERGY_DRIFT_WARNING  = 1e-4;  ///< 0.01%
inline constexpr double ENERGY_DRIFT_CRITICAL = 1e-3;  ///< 0.10%

/// ATP reserve thresholds (percent)
inline constexpr double ATP_WARNING_PCT  = 15.0;
inline constexpr double ATP_CRITICAL_PCT =  5.0;

/// Wavefunction amplitude thresholds (balanced nonary limit ≈ 4.5)
inline constexpr double AMPLITUDE_WARNING  = 4.0;
inline constexpr double AMPLITUDE_CRITICAL = 5.0;

/// Hardware watchdog timeout — if physics thread fails to "pet" within this
/// window the process is assumed deadlocked and SIGALRM fires.
inline constexpr double WATCHDOG_TIMEOUT_US = 2'000.0;  ///< 2 ticks

// ---------------------------------------------------------------------------
// Enumerations
// ---------------------------------------------------------------------------

/// Ordered so that max(a, b) yields the more-severe level.
enum class AlertLevel : int {
    OK       = 0,
    WARNING  = 1,
    CRITICAL = 2,
};

/// Critical-path components of the 1-tick pipeline.
enum class Component {
    PHYSICS_KERNEL,   ///< Wave propagation (Strang splitting)
    SCANNER,          ///< Mamba-9D SSM causal-foliated scan
    ENGS,             ///< Neurochemical gating + parameter broadcast
    INFRASTRUCTURE,   ///< Seqlock IPC + ZeroMQ control signal check
};

/// What to do when the tick is exceeding budget ("Drop or Degrade" policy).
enum class DegradePolicy {
    NO_DEGRADE,      ///< tick ≤ allocatable budget (900 μs) — all steps run
    SKIP_NONLINEAR,  ///< tick ≤ hard limit (1000 μs) — skip soliton step (ĤN)
    DROP_FRAME,      ///< tick > hard limit  — discard entire frame
};

// ---------------------------------------------------------------------------
// TelemetrySnapshot
// ---------------------------------------------------------------------------

/**
 * @brief Snapshot of the three primary Physics Oracle telemetry points.
 *
 * Spec (§GAP-025 "Telemetry Points"):
 *  tick_duration_ns    — monotonic clock delta (start/end of propagate())
 *  energy_drift_ratio  — |ΔH / H| (relative Hamiltonian drift since last tick)
 *  lock_contention_cnt — failed atomic CAS operations in metabolic lock
 *  atp_reserve_pct     — remaining virtual ATP as percentage [0, 100]
 *  amplitude_max       — max |Ψ_i| across all grid nodes this tick
 */
struct TelemetrySnapshot {
    int64_t tick_duration_ns    {0};
    double  energy_drift_ratio  {0.0};
    int     lock_contention_cnt {0};
    double  atp_reserve_pct     {100.0};
    double  amplitude_max       {0.0};

    [[nodiscard]] bool is_sane() const noexcept {
        return tick_duration_ns >= 0
            && energy_drift_ratio >= 0.0
            && lock_contention_cnt >= 0
            && atp_reserve_pct >= 0.0
            && amplitude_max >= 0.0;
    }
};

// ---------------------------------------------------------------------------
// LatencyBudget — stateless helper class (all static methods)
// ---------------------------------------------------------------------------

/**
 * @brief Stateless GAP-025 latency-budget enforcement engine.
 *
 * All methods are pure functions of their inputs.  The class itself holds no
 * state; instantiate-and-call or use as a namespace-in-a-class.
 */
class LatencyBudget {
public:
    LatencyBudget() = delete;

    // ------------------------------------------------------------------
    // Budget queries
    // ------------------------------------------------------------------

    /**
     * @brief Allocated budget for a critical-path component (μs).
     *
     * Spec directly assigns:
     *   PHYSICS_KERNEL  → 600 μs
     *   SCANNER         → 200 μs
     *   ENGS            →  50 μs
     *   INFRASTRUCTURE  →  50 μs
     *                     ------
     *                     900 μs  == TICK_ALLOCATABLE_US
     */
    [[nodiscard]] static constexpr double component_budget_us(Component c) noexcept {
        switch (c) {
            case Component::PHYSICS_KERNEL:  return BUDGET_PHYSICS_KERNEL_US;
            case Component::SCANNER:         return BUDGET_SCANNER_US;
            case Component::ENGS:            return BUDGET_ENGS_US;
            case Component::INFRASTRUCTURE:  return BUDGET_INFRASTRUCTURE_US;
        }
        return 0.0; // unreachable
    }

    /**
     * @brief Sum of all four component budgets — must equal TICK_ALLOCATABLE_US.
     */
    [[nodiscard]] static constexpr double total_component_budget_us() noexcept {
        return BUDGET_PHYSICS_KERNEL_US
             + BUDGET_SCANNER_US
             + BUDGET_ENGS_US
             + BUDGET_INFRASTRUCTURE_US;
    }

    /**
     * @brief Fraction of allocatable budget consumed by component c.
     *
     * E.g. PHYSICS_KERNEL → 600/900 ≈ 0.6667.
     */
    [[nodiscard]] static constexpr double budget_fraction(Component c) noexcept {
        return component_budget_us(c) / TICK_ALLOCATABLE_US;
    }

    /**
     * @brief Hardware watchdog timeout in microseconds (2000 μs = 2 ticks).
     *
     * The physics thread must "pet" the watchdog every tick.  Failure to do
     * so within this window triggers SIGALRM → stack dump → fail-safe restart.
     */
    [[nodiscard]] static constexpr double watchdog_timeout_us() noexcept {
        return WATCHDOG_TIMEOUT_US;
    }

    // ------------------------------------------------------------------
    // Per-metric alerting
    // ------------------------------------------------------------------

    /**
     * @brief Alert level for a tick duration measured in nanoseconds.
     *
     * Thresholds (spec §GAP-025 "Alerting Thresholds"):
     *   OK       : tick_ns ≤ 950,000 ns  (950 μs)
     *   WARNING  : tick_ns ≤ 1,050,000 ns (1050 μs)
     *   CRITICAL : tick_ns >  1,050,000 ns
     */
    [[nodiscard]] static constexpr AlertLevel tick_alert(int64_t tick_ns) noexcept {
        if (tick_ns <= TICK_WARNING_NS)  return AlertLevel::OK;
        if (tick_ns <= TICK_CRITICAL_NS) return AlertLevel::WARNING;
        return AlertLevel::CRITICAL;
    }

    /**
     * @brief Alert level for relative Hamiltonian energy drift |ΔH/H|.
     *
     * Thresholds:
     *   OK       : ratio < 1e-4  (< 0.01%)
     *   WARNING  : ratio < 1e-3  (< 0.10%)
     *   CRITICAL : ratio ≥ 1e-3
     */
    [[nodiscard]] static constexpr AlertLevel energy_drift_alert(double ratio) noexcept {
        if (ratio < ENERGY_DRIFT_WARNING)  return AlertLevel::OK;
        if (ratio < ENERGY_DRIFT_CRITICAL) return AlertLevel::WARNING;
        return AlertLevel::CRITICAL;
    }

    /**
     * @brief Alert level for virtual ATP reserve (percentage 0–100).
     *
     * Thresholds:
     *   OK       : pct > 15%
     *   WARNING  : pct ≥  5%  (and ≤ 15%)
     *   CRITICAL : pct <  5%  → force "Nap" state
     */
    [[nodiscard]] static constexpr AlertLevel atp_alert(double pct) noexcept {
        if (pct > ATP_WARNING_PCT)  return AlertLevel::OK;
        if (pct >= ATP_CRITICAL_PCT) return AlertLevel::WARNING;
        return AlertLevel::CRITICAL;
    }

    /**
     * @brief Alert level for max wavefunction amplitude |Ψ|_max.
     *
     * Thresholds (spec §GAP-030 runtime monitoring + GAP-025 balanced nonary):
     *   OK       : amp < 4.0
     *   WARNING  : amp < 5.0
     *   CRITICAL : amp ≥ 5.0  → SCRAM (hard limit)
     */
    [[nodiscard]] static constexpr AlertLevel amplitude_alert(double amp) noexcept {
        if (amp < AMPLITUDE_WARNING)  return AlertLevel::OK;
        if (amp < AMPLITUDE_CRITICAL) return AlertLevel::WARNING;
        return AlertLevel::CRITICAL;
    }

    // ------------------------------------------------------------------
    // Policy
    // ------------------------------------------------------------------

    /**
     * @brief Degrade policy derived from a measured tick duration (ns).
     *
     * Spec mandate ("Drop or Degrade", §GAP-025):
     *   No buffering.  If the system cannot keep up choose either:
     *     - Degrade precision: skip the nonlinear soliton step (ĤN)
     *     - Drop the frame entirely
     *
     * Boundaries:
     *   tick_ns ≤ 900,000 ns  → NO_DEGRADE   (within allocatable budget)
     *   tick_ns ≤ 1,000,000 ns → SKIP_NONLINEAR (over budget, under hard limit)
     *   tick_ns > 1,000,000 ns → DROP_FRAME
     */
    [[nodiscard]] static constexpr DegradePolicy degrade_policy(int64_t tick_ns) noexcept {
        constexpr int64_t ALLOCATABLE_NS = static_cast<int64_t>(TICK_ALLOCATABLE_US * 1'000.0);
        constexpr int64_t BUDGET_NS      = static_cast<int64_t>(TICK_BUDGET_US     * 1'000.0);
        if (tick_ns <= ALLOCATABLE_NS) return DegradePolicy::NO_DEGRADE;
        if (tick_ns <= BUDGET_NS)      return DegradePolicy::SKIP_NONLINEAR;
        return DegradePolicy::DROP_FRAME;
    }

    // ------------------------------------------------------------------
    // Composite assessment
    // ------------------------------------------------------------------

    /**
     * @brief Overall alert level for a full telemetry snapshot.
     *
     * Returns the maximum (most severe) alert across all four monitored
     * metrics: tick latency, energy drift, ATP reserve, amplitude max.
     */
    [[nodiscard]] static constexpr AlertLevel assess_overall(const TelemetrySnapshot& snap) noexcept {
        auto worst = AlertLevel::OK;
        auto upd = [&](AlertLevel a) noexcept {
            if (static_cast<int>(a) > static_cast<int>(worst)) worst = a;
        };
        upd(tick_alert(snap.tick_duration_ns));
        upd(energy_drift_alert(snap.energy_drift_ratio));
        upd(atp_alert(snap.atp_reserve_pct));
        upd(amplitude_alert(snap.amplitude_max));
        return worst;
    }

    /**
     * @brief True if the snapshot indicates the system is within all safe limits.
     */
    [[nodiscard]] static constexpr bool is_healthy(const TelemetrySnapshot& snap) noexcept {
        return assess_overall(snap) == AlertLevel::OK;
    }

    /**
     * @brief True if any metric is CRITICAL (SCRAM-worthy).
     */
    [[nodiscard]] static constexpr bool requires_scram(const TelemetrySnapshot& snap) noexcept {
        return assess_overall(snap) == AlertLevel::CRITICAL;
    }
};

} // namespace nikola::physics
