/**
 * @file src/autonomy/evolutionary_orchestrator.cpp
 * @brief EvolutionaryOrchestrator — SIE Phase-4 loop controller.
 *
 * Wires MetabolicController + CodePatternBlacklist + PhysicsOracle +
 * ModuleSwapper into the full SIE validation and deployment cycle.
 *
 * See the header for the authoritative design contract.
 */

#include <nikola/autonomy/evolutionary_orchestrator.hpp>

#include <chrono>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

namespace nikola::autonomy {

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Milliseconds since an arbitrary start point (used for elapsed timing only).
static double now_ms() {
    using clock = std::chrono::steady_clock;
    static const auto start = clock::now();
    return std::chrono::duration<double, std::milli>(clock::now() - start).count();
}

// ── Construction / destruction ────────────────────────────────────────────────

EvolutionaryOrchestrator::EvolutionaryOrchestrator(
    MetabolicController&                          controller,
    const nikola::security::CodePatternBlacklist& blacklist,
    std::string                                   factory_symbol)
    : controller_ {controller}
    , blacklist_  {blacklist}
    , swapper_    {std::move(factory_symbol)} {}

EvolutionaryOrchestrator::EvolutionaryOrchestrator(
    EvolutionaryOrchestrator&& other) noexcept
    : controller_ {other.controller_}
    , blacklist_  {other.blacklist_}
    , swapper_    {std::move(other.swapper_)}
    , stats_      {std::exchange(other.stats_, {})} {}

EvolutionaryOrchestrator&
EvolutionaryOrchestrator::operator=(EvolutionaryOrchestrator&& other) noexcept {
    if (this == &other) return *this;
    // controller_ and blacklist_ are references — can't be rebound.
    // We can only take ownership of the movable members.
    swapper_ = std::move(other.swapper_);
    stats_   = std::exchange(other.stats_, {});
    return *this;
}

// ── Private helpers ────────────────────────────────────────────────────────

CycleStatus
EvolutionaryOrchestrator::from_swap_result(SwapResult sr) noexcept {
    switch (sr) {
        case SwapResult::SUCCESS:           return CycleStatus::SUCCESS;
        case SwapResult::LOAD_FAILED:       return CycleStatus::LOAD_FAILED;
        case SwapResult::SYMBOL_MISSING:    return CycleStatus::SYMBOL_MISSING;
        case SwapResult::SAME_MODULE:       return CycleStatus::SAME_MODULE;
        case SwapResult::VALIDATION_FAILED: return CycleStatus::PHYSICS_REJECTED;
        default:                            return CycleStatus::LOAD_FAILED;
    }
}

// ── run_cycle ─────────────────────────────────────────────────────────────────

CycleReport EvolutionaryOrchestrator::run_cycle(
    std::string_view   candidate_so_path,
    const std::string& source_code,
    PhysicsProvider    physics_provider)
{
    std::lock_guard<std::mutex> lk{mtx_};

    CycleReport report;
    report.candidate_path = std::string{candidate_so_path};
    const double t_start  = now_ms();

    // Helper that finalises the report and updates stats before returning.
    auto finish = [&](CycleStatus status) -> CycleReport& {
        report.status      = status;
        report.elapsed_ms  = now_ms() - t_start;
        ++stats_.total;
        switch (status) {
            case CycleStatus::SUCCESS:           ++stats_.succeeded;          break;
            case CycleStatus::ATP_DENIED:        ++stats_.atp_denied;         break;
            case CycleStatus::SECURITY_REJECTED: ++stats_.security_rejected;  break;
            case CycleStatus::PHYSICS_REJECTED:  ++stats_.physics_rejected;   break;
            case CycleStatus::LOAD_FAILED:
            case CycleStatus::SYMBOL_MISSING:
            case CycleStatus::SAME_MODULE:       ++stats_.load_failed;        break;
        }
        return report;
    };

    // ── Step 1: Acquire MetabolicLock ────────────────────────────────────────
    // The lock constructor throws MetabolicExhaustionException if ATP < TOTAL_COST.
    // We catch it and map to ATP_DENIED.
    std::optional<MetabolicLock> lock;
    try {
        lock.emplace(controller_, TOTAL_COST);
    } catch (const MetabolicExhaustionException&) {
        finish(CycleStatus::ATP_DENIED);
        return report;
    }

    // ── Step 2: Gate 1 — Security Scan ──────────────────────────────────────
    if (!source_code.empty()) {
        const bool safe = blacklist_.is_safe(source_code);
        report.gate1_security_passed = safe;
        if (!safe) {
            finish(CycleStatus::SECURITY_REJECTED);
            return report;   // lock destructs → auto-refund
        }
    } else {
        // No source provided — cannot verify safety.  Reject.
        report.gate1_security_passed = false;
        finish(CycleStatus::SECURITY_REJECTED);
        return report;
    }

    // ── Step 3: Gate 2 — Physics Oracle + Gate 3 swap_in (combined) ─────────
    // We inject the physics check as the ModuleSwapper validator callback so
    // that the .so must be successfully loaded before we can call the provider.
    // This gives Gate 2 access to the actual factory symbol.

    bool physics_gate_passed = true;

    ModuleSwapper::ValidatorFn composite_validator;

    if (physics_provider) {
        composite_validator = [&](void* factory_sym) -> bool {
            auto measurement = physics_provider(factory_sym);

            // Provider returned nullopt → treat as physics failure.
            if (!measurement.has_value()) {
                physics_gate_passed = false;
                return false;
            }

            const auto& m = *measurement;

            if (m.skip_oracle) {
                // Caller opted out of oracle classification.
                report.gate2_physics_passed = true;
                return true;
            }

            // Classify energy conservation (spec §5.1 eq. 1).
            const auto energy_result = physics::PhysicsOracle::check_energy_conservation(
                m.H_initial, m.H_final);

            // Classify time reversibility (spec §5.1 eq. 2).
            // check_reversibility() takes full spans; we have a pre-computed L2
            // scalar, so we apply the published threshold directly.
            const bool rev_passed =
                (m.reversibility_l2 < physics::REVERSIBILITY_ERROR_LIMIT);

            report.energy_drift_ratio   = energy_result.drift_ratio;
            report.reversibility_error  = m.reversibility_l2;

            const bool passed = energy_result.passed && rev_passed;
            report.gate2_physics_passed = passed;
            physics_gate_passed         = passed;
            return passed;
        };
    } else {
        // No physics provider → Gate 2 is skipped (treated as passed).
        report.gate2_physics_passed = true;
    }

    // ── Step 4: Gate 3 — ModuleSwapper hot-swap ──────────────────────────────
    const SwapResult sr = swapper_.swap_in(candidate_so_path,
                                           std::move(composite_validator));

    if (sr == SwapResult::VALIDATION_FAILED && !physics_gate_passed) {
        // The validator already set gate2_physics_passed = false.
        finish(CycleStatus::PHYSICS_REJECTED);
        return report;
    }

    if (sr != SwapResult::SUCCESS) {
        finish(from_swap_result(sr));
        return report;
    }

    report.gate3_swap_passed = true;

    // ── Step 5: Commit — deduct ATP ──────────────────────────────────────────
    lock->commit();
    report.atp_consumed = TOTAL_COST;

    finish(CycleStatus::SUCCESS);
    return report;
}

// ── rollback ──────────────────────────────────────────────────────────────────

bool EvolutionaryOrchestrator::rollback() {
    std::lock_guard<std::mutex> lk{mtx_};
    return swapper_.rollback();
}

// ── Inspection ────────────────────────────────────────────────────────────────

bool EvolutionaryOrchestrator::has_active() const noexcept {
    return swapper_.has_active();
}

bool EvolutionaryOrchestrator::has_previous() const noexcept {
    return swapper_.has_previous();
}

void* EvolutionaryOrchestrator::active_factory() const noexcept {
    return swapper_.active_factory();
}

std::string EvolutionaryOrchestrator::active_path() const noexcept {
    return swapper_.active_path();
}

CycleStats EvolutionaryOrchestrator::stats() const noexcept {
    std::lock_guard<std::mutex> lk{mtx_};
    return stats_;
}

const ModuleSwapper& EvolutionaryOrchestrator::swapper() const noexcept {
    return swapper_;
}

} // namespace nikola::autonomy
