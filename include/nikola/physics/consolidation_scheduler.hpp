/**
 * @file consolidation_scheduler.hpp
 * @brief GAP-024: Metric Tensor Consolidation Interval Justification
 *
 * @spec FABRICATION-READY — docs/info/integration/sections/02_foundations/
 *       01_9d_toroidal_geometry.md §GAP-024
 *
 * ### Background
 * Nikola's physics engine maintains a 9D Riemannian metric tensor g_ij(x,t)
 * that must be inverted at every node to evaluate the Laplace-Beltrami
 * operator.  For ~10^7 nodes at 1 kHz this costs ~20 TFLOPS — far beyond
 * any single GPU.
 *
 * ### Timescale Separation
 * The metric is split into two components:
 *   g_ij(t) = g_ij_base + h_ij(t)
 *
 *   Fast path (1 ms):  uses cached Cholesky of g_ij_base.
 *                      Effect of h_ij computed via 1st-order perturbation:
 *                        Γ(g+h) ≈ Γ(g) + δΓ(h)      → ~200 FLOPS/node
 *   Slow path (5 min): sums accumulated h_ij into base, recomputes L.
 *                        O(N·D³) — expensive, but amortized to negligible.
 *
 * ### Adaptive Triggers (ANY triggers consolidation)
 *   1. Time elapsed > T_max (5 min) with ATP ≥ METABOLIC_FLOOR
 *   2. Time elapsed > 2×T_max regardless of ATP (deadline)
 *   3. Perturbation magnitude > ε·‖g_base‖_F  (ε = 0.1 → 10%)
 *   4. System enters Nap state (ideal time — physics loop paused)
 *
 * ### Performance Impact (per spec)
 *   • 90% FLOPS reduction on fast path
 *   • 1 kHz physics loop maintained 99.9% of time
 *   • Geodesic error < 1% (ε = 0.1 constraint)
 */
#pragma once

#include <stdexcept>

namespace nikola::physics {

// ---------------------------------------------------------------------------
// Consolidation constants (spec §GAP-024 "Tuning Parameters")
// ---------------------------------------------------------------------------

/// Maximum consolidation interval: 5 minutes × 60 s/min = 300 s.
/// Ensures eventual consistency even without any perturbation trigger.
inline constexpr double CONSOLIDATION_MAX_INTERVAL_SEC = 300.0;

/// Perturbation threshold ε: if ‖h_ij‖_F > ε·‖g_base‖_F, the first-order
/// approximation error becomes unacceptable.  Forces immediate consolidation.
inline constexpr double CONSOLIDATION_PERTURBATION_LIMIT = 0.1;

/// ATP level below which consolidation is deferred (saves energy).
/// If ATP < 20%, expensive Cholesky recomputation is postponed.
inline constexpr double CONSOLIDATION_METABOLIC_FLOOR = 0.2;

/// Hard-deadline multiplier: even if ATP is low, consolidation is forced
/// once time_since_last > MAX_INTERVAL × DEFERRAL_FACTOR (= 10 min cap).
inline constexpr double CONSOLIDATION_DEFERRAL_FACTOR = 2.0;

/// 90% FLOPS reduction achieved on the 1 ms fast path via perturbation theory.
inline constexpr double CONSOLIDATION_FAST_PATH_FLOPS_REDUCTION = 0.90;

/// Fraction of ticks where 1 kHz loop is maintained (availability target).
inline constexpr double CONSOLIDATION_AVAILABILITY_TARGET = 0.999;

// ---------------------------------------------------------------------------
// ConsolidationScheduler
// ---------------------------------------------------------------------------

/**
 * @brief Adaptive scheduler for base-metric Cholesky consolidation events.
 *
 * Spec §GAP-024 "Adaptive Consolidation Scheduler":
 *   Three triggers — perturbation magnitude, nap opportunity, time deadline.
 *   Workload-adaptive: defers when ATP is below metabolic floor unless the
 *   hard deadline (2× MAX_INTERVAL) is reached.
 *
 * ### Typical usage
 * ```cpp
 * ConsolidationScheduler sched;
 * while (running) {
 *     sched.advance(dt_sec);
 *     double dev = engine.get_max_metric_deviation();
 *     if (sched.should_consolidate(dev, engine.is_napping(), metabolism.get_atp())) {
 *         engine.consolidate_metric();
 *         sched.on_consolidated();
 *     }
 *     engine.tick();
 * }
 * ```
 */
struct ConsolidationScheduler {
    // Configuration (defaults match spec; may be overridden for testing)
    double max_interval_sec       = CONSOLIDATION_MAX_INTERVAL_SEC;
    double perturbation_limit     = CONSOLIDATION_PERTURBATION_LIMIT;
    double metabolic_floor        = CONSOLIDATION_METABOLIC_FLOOR;
    double deferral_factor        = CONSOLIDATION_DEFERRAL_FACTOR;

    // State
    double time_since_last_update = 0.0;   // seconds since last consolidation
    double max_perturbation_norm  = 0.0;   // ‖h_ij‖_F / ‖g_base‖_F ratio

    /**
     * @brief Determine whether to trigger a consolidation event now.
     *
     * Priority order (matches spec):
     *  1. Perturbation > limit     → MUST consolidate immediately (numerical stability)
     *  2. Napping                  → SHOULD consolidate (free compute window)
     *  3. Time > max_interval      → consolidate UNLESS ATP is low AND deadline not reached
     *  4. Time > max×deferral_fac  → FORCE consolidation (hard deadline)
     *
     * @param max_deviation   Current max‖h_ij‖/‖g_base‖ ratio from physics engine
     * @param is_napping      True if the physics loop is in a Nap cycle
     * @param atp_level       Normalised ATP in [0, 1] from MetabolicController
     */
    [[nodiscard]] bool should_consolidate(double max_deviation,
                                          bool   is_napping,
                                          double atp_level) const noexcept {
        // 1. Critical stability: first-order approximation breaking down
        if (max_deviation > perturbation_limit)
            return true;

        // 2. Nap opportunity: expensive work is free during physics pause
        if (is_napping)
            return true;

        // 3. Time-based check with workload-adaptive deferral
        if (time_since_last_update > max_interval_sec) {
            // Defer if low energy — but only up to the hard deadline
            if (atp_level < metabolic_floor) {
                // Hard deadline: cap deferral at deferral_factor × max_interval
                return time_since_last_update >= max_interval_sec * deferral_factor;
            }
            return true;
        }

        return false;
    }

    /**
     * @brief Reset the scheduler after a consolidation event completes.
     * Clears both the elapsed timer and the tracked perturbation norm.
     */
    void on_consolidated() noexcept {
        time_since_last_update = 0.0;
        max_perturbation_norm  = 0.0;
    }

    /**
     * @brief Advance the scheduler clock by dt seconds.
     * Call once per physics tick (dt = 1e-3 s at 1 kHz).
     *
     * @param dt_sec  Elapsed wall time in seconds (must be ≥ 0)
     */
    void advance(double dt_sec) {
        if (dt_sec < 0.0)
            throw std::invalid_argument("ConsolidationScheduler::advance: dt_sec must be >= 0");
        time_since_last_update += dt_sec;
    }

    /**
     * @brief Update the tracked max perturbation norm from the physics engine.
     * Call at the end of each tick to keep the guard informed.
     */
    void update_perturbation(double max_deviation) noexcept {
        if (max_deviation > max_perturbation_norm)
            max_perturbation_norm = max_deviation;
    }

    /**
     * @brief True if we are currently past the time-based trigger threshold
     * (before considering ATP deferral).
     */
    [[nodiscard]] bool is_overdue() const noexcept {
        return time_since_last_update > max_interval_sec;
    }

    /**
     * @brief True if we are past the hard deadline (deferral cap exceeded).
     */
    [[nodiscard]] bool is_past_deadline() const noexcept {
        return time_since_last_update >= max_interval_sec * deferral_factor;
    }
};

} // namespace nikola::physics
