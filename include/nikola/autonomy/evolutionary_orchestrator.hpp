/**
 * @file include/nikola/autonomy/evolutionary_orchestrator.hpp
 * @brief SIE Phase-4 main loop controller.
 *
 * The EvolutionaryOrchestrator (EO) is the coordinator that drives the full
 * Self-Improvement Evolution (SIE) cycle described in:
 *   docs/info/integration/sections/05_autonomous_systems/04_self_improvement.md
 *
 * It wires together the four existing Phase-4 infrastructure components:
 *
 *   1. MetabolicController  — ATP budget enforcement (§7)
 *   2. CodePatternBlacklist — static security scan (§2.1, Gate 1)
 *   3. PhysicsOracle        — energy-conservation validation (§5, Gate 2)
 *   4. ModuleSwapper        — POSIX dlopen hot-swap + one-step rollback (§2.4)
 *
 * Cycle sequence (run_cycle)
 * ─────────────────────────────────────────────────────────────────────────────
 *  1.  Acquire MetabolicLock for the full cycle cost (COMPILE+VERIFY+DEPLOY).
 *      → If ATP is insufficient: return ATP_DENIED immediately.
 *
 *  2.  Gate 1 — Security Scan (CodePatternBlacklist)
 *      If source code is supplied, scan it for blacklisted patterns.
 *      → Rejection: return SECURITY_REJECTED; lock auto-refunds.
 *
 *  3.  Gate 2 — Physics Oracle
 *      If the caller supplies a PhysicsProvider callable, invoke it with the
 *      candidate factory symbol and classify the returned measurements.
 *      → Rejection: return PHYSICS_REJECTED; lock auto-refunds.
 *
 *  4.  Gate 3 — ModuleSwapper::swap_in
 *      Hot-load and promote the validated candidate.
 *      → Failure mapped to appropriate status; lock auto-refunds.
 *
 *  5.  Commit MetabolicLock — deducts ATP from the controller.
 *
 * CycleReport
 * ─────────────────────────────────────────────────────────────────────────────
 * Every call to run_cycle() returns a CycleReport with per-gate outcomes, the
 * module path, and elapsed time.  The orchestrator also maintains a cumulative
 * CycleStats counter.
 *
 * Rollback
 * ─────────────────────────────────────────────────────────────────────────────
 * rollback() delegates directly to ModuleSwapper::rollback().  No ATP is
 * consumed on rollback (rollback is a safety operation, not a luxury).
 *
 * Thread safety
 * ─────────────────────────────────────────────────────────────────────────────
 * run_cycle() and rollback() are individually thread-safe via an internal
 * mutex.  ModuleSwapper has its own internal mutex; the two are not nested.
 *
 * ATP cost constants (spec §7.1)
 * ─────────────────────────────────────────────────────────────────────────────
 *   COMPILE_COST  = 500.0f  — reserved to cover code-generation/compilation
 *   VERIFY_COST   = 200.0f  — covers oracle validation
 *   DEPLOY_COST   =  50.0f  — covers dlopen swap
 *   TOTAL_COST    = 750.0f  — acquired as a single MetabolicLock
 */

#pragma once

#include <nikola/autonomy/metabolic_controller.hpp>
#include <nikola/autonomy/metabolic_lock.hpp>
#include <nikola/autonomy/module_swapper.hpp>
#include <nikola/physics/physics_oracle.hpp>
#include <nikola/security/code_blacklist.hpp>

#include <chrono>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

namespace nikola::autonomy {

// ============================================================================
// CycleStatus
// ============================================================================

/// High-level outcome of a single EO cycle.
enum class CycleStatus : int {
    SUCCESS           = 0, ///< All gates passed; candidate is now active.
    ATP_DENIED        = 1, ///< Insufficient ATP; cycle not started.
    SECURITY_REJECTED = 2, ///< Gate 1 (CodePatternBlacklist) rejected source.
    PHYSICS_REJECTED  = 3, ///< Gate 2 (PhysicsOracle) rejected measurements.
    LOAD_FAILED       = 4, ///< Gate 3: dlopen failed (bad path/missing deps).
    SYMBOL_MISSING    = 5, ///< Gate 3: factory symbol not found in candidate.
    SAME_MODULE       = 6, ///< Gate 3: candidate path identical to active.
};

/// Human-readable label for a CycleStatus.
[[nodiscard]] constexpr std::string_view cycle_status_str(CycleStatus s) noexcept {
    switch (s) {
        case CycleStatus::SUCCESS:           return "SUCCESS";
        case CycleStatus::ATP_DENIED:        return "ATP_DENIED";
        case CycleStatus::SECURITY_REJECTED: return "SECURITY_REJECTED";
        case CycleStatus::PHYSICS_REJECTED:  return "PHYSICS_REJECTED";
        case CycleStatus::LOAD_FAILED:       return "LOAD_FAILED";
        case CycleStatus::SYMBOL_MISSING:    return "SYMBOL_MISSING";
        case CycleStatus::SAME_MODULE:       return "SAME_MODULE";
        default:                             return "UNKNOWN";
    }
}

// ============================================================================
// PhysicsMeasurement
// ============================================================================

/// Measurements that the caller's physics sandbox produces for Gate 2.
///
/// The orchestrator does NOT run physics simulations — it classifies
/// pre-computed measurements through PhysicsOracle.
struct PhysicsMeasurement {
    double H_initial{0.0};       ///< Hamiltonian before propagation.
    double H_final{0.0};         ///< Hamiltonian after propagation.
    double reversibility_l2{0.0};///< L2 norm of (initial − reversed) state.

    /// If true, the orchestrator skips oracle classification and marks the
    /// physics gate as passed unconditionally (useful for modules with no
    /// physics kernel, e.g., pure utility plugins).
    bool skip_oracle{false};
};

// ============================================================================
// CycleReport
// ============================================================================

/// Per-cycle result returned by EvolutionaryOrchestrator::run_cycle().
struct CycleReport {
    CycleStatus status{CycleStatus::ATP_DENIED};

    // Gate outcomes (empty/zero if the gate was not reached).
    bool gate1_security_passed{false};   ///< CodePatternBlacklist result.
    bool gate2_physics_passed{false};    ///< PhysicsOracle result.
    bool gate3_swap_passed{false};       ///< ModuleSwapper result.

    // Physics details (populated if Gate 2 was run).
    double energy_drift_ratio{0.0};      ///< |H_f − H_i| / H_i
    double reversibility_error{0.0};     ///< L2 norm of time-reversal residual.

    // Metadata.
    std::string candidate_path;          ///< Path attempted.
    float atp_consumed{0.0f};            ///< ATP deducted on SUCCESS only.
    double elapsed_ms{0.0};              ///< Wall-clock time for the cycle.

    /// Convenience: true iff status == SUCCESS.
    explicit operator bool() const noexcept {
        return status == CycleStatus::SUCCESS;
    }
};

// ============================================================================
// CycleStats
// ============================================================================

/// Cumulative counters maintained by the orchestrator across all cycles.
struct CycleStats {
    std::size_t total{0};
    std::size_t succeeded{0};
    std::size_t atp_denied{0};
    std::size_t security_rejected{0};
    std::size_t physics_rejected{0};
    std::size_t load_failed{0};
};

// ============================================================================
// EvolutionaryOrchestrator
// ============================================================================

/// SIE Phase-4 loop coordinator.
///
/// Owns a ModuleSwapper and references external MetabolicController,
/// CodePatternBlacklist, and the stateless PhysicsOracle.  Move-only.
class EvolutionaryOrchestrator {
public:
    // ── Types ────────────────────────────────────────────────────────────────

    /// ATP cost constants (spec §7.1).
    static constexpr float COMPILE_COST = 500.0f;
    static constexpr float VERIFY_COST  = 200.0f;
    static constexpr float DEPLOY_COST  =  50.0f;
    static constexpr float TOTAL_COST   = COMPILE_COST + VERIFY_COST + DEPLOY_COST;

    /// Callable that the caller provides to run Gate 2 physics validation.
    ///
    /// Receives the successfully resolved factory symbol pointer from the
    /// candidate module and returns measurements for the oracle to classify.
    /// Return std::nullopt to signal that the candidate could not be measured
    /// (treated as physics failure).
    using PhysicsProvider =
        std::function<std::optional<PhysicsMeasurement>(void* factory_sym)>;

    // ── Construction ─────────────────────────────────────────────────────────

    /// Construct with all required infrastructure references.
    ///
    /// @param controller       ATP budget owner.
    /// @param blacklist        Compiled pattern set for Gate 1.
    /// @param factory_symbol   Name of the factory symbol each candidate must
    ///                         export (forwarded to internal ModuleSwapper).
    explicit EvolutionaryOrchestrator(
        MetabolicController&                     controller,
        const nikola::security::CodePatternBlacklist& blacklist,
        std::string                 factory_symbol = "nikola_module_factory");

    ~EvolutionaryOrchestrator() = default;

    // Non-copyable.
    EvolutionaryOrchestrator(const EvolutionaryOrchestrator&)            = delete;
    EvolutionaryOrchestrator& operator=(const EvolutionaryOrchestrator&) = delete;

    // Movable.
    EvolutionaryOrchestrator(EvolutionaryOrchestrator&&) noexcept;
    EvolutionaryOrchestrator& operator=(EvolutionaryOrchestrator&&) noexcept;

    // ── Primary interface ────────────────────────────────────────────────────

    /// Run one full SIE cycle against the given candidate shared library.
    ///
    /// @param candidate_so_path  Path to the candidate .so file.
    /// @param source_code        C++ source of the candidate (empty = skip
    ///                           Gate 1).  The orchestrator does NOT compile;
    ///                           you supply the pre-compiled .so separately.
    /// @param physics_provider   Callable for Gate 2 (empty = skip physics).
    ///
    /// @returns CycleReport describing per-gate outcomes and telemetry.
    [[nodiscard]] CycleReport run_cycle(
        std::string_view    candidate_so_path,
        const std::string&  source_code       = {},
        PhysicsProvider     physics_provider  = {});

    /// Roll back to the previous module (no ATP charge).
    ///
    /// Delegates to ModuleSwapper::rollback().  Returns false if no previous
    /// module is available.
    bool rollback();

    // ── Inspection ──────────────────────────────────────────────────────────

    [[nodiscard]] bool        has_active()       const noexcept;
    [[nodiscard]] bool        has_previous()     const noexcept;
    [[nodiscard]] void*       active_factory()   const noexcept;
    [[nodiscard]] std::string active_path()      const noexcept;
    [[nodiscard]] CycleStats  stats()            const noexcept;

    /// Direct read-only access to the internal ModuleSwapper.
    [[nodiscard]] const ModuleSwapper& swapper() const noexcept;

private:
    MetabolicController&                          controller_;
    const nikola::security::CodePatternBlacklist& blacklist_;
    ModuleSwapper               swapper_;

    CycleStats          stats_{};
    mutable std::mutex  mtx_;

    /// Map a SwapResult to the corresponding CycleStatus.
    static CycleStatus from_swap_result(SwapResult sr) noexcept;
};

} // namespace nikola::autonomy
