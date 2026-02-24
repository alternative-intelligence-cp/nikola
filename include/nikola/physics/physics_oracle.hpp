/**
 * @file physics_oracle.hpp
 * @brief GAP-030: Physics Oracle Calibration Test Suite
 *
 * @spec SPECIFICATION COMPLETE — docs/info/integration/sections/02_foundations/
 *       02_wave_interference_physics.md §GAP-030
 *
 * The Physics Oracle is the system's runtime watchdog (the "Superego in code").
 * Its purpose is to detect Decoherence — violations of the fundamental physical
 * invariants of the UFIE caused by numerical errors, software bugs, or malicious
 * self-modification.
 *
 * In a system capable of self-improvement (KVM Executor), the Oracle is the
 * final gatekeeper:
 *   - False positive → SCRAM (agent killed)
 *   - False negative → epileptic resonance (manifold corrupted)
 *
 * Quantitative acceptance criteria (Split-Operator Symplectic Integrator):
 *
 * 1. Energy Conservation (Hamiltonian Check — closed system, α=0):
 *      ΔE_rel = |H(t) - H(0)| / |H(0)| < 10⁻⁵ over 10⁶ steps
 *      Standard Candle test: < 10⁻⁶ over 100k steps
 *
 * 2. Symplectic Structure (Liouville / Reversibility Check):
 *      ε_rev = ‖Ψ(0) - Ψ_fwd_bwd(0)‖² < 10⁻¹² (≈ machine epsilon for double)
 *
 * 3. Numerical Viscosity (Damping Check, α > 0):
 *      ε_decay = |E(t) / E_theory(t) − 1| < 0.01%  (1e‑4)
 *      where E_theory(t) = E₀ · exp(−2αt)
 *
 * 4. Runtime SCRAM thresholds (continuously monitored every tick):
 *      |dH/dt|   > 1e‑7 per step  → WARNING
 *      |dH/dt|   > 1e‑5 per step  → SCRAM
 *      |Ψ|_max   > 4.0             → WARNING
 *      |Ψ|_max   > 5.0             → SCRAM (hard limit)
 *      NaN / Inf in Ψ or gij       → SCRAM (immediate)
 *
 * Execution matrix (SIMD fallback tolerance relaxation):
 *   AVX-512 (reference): exact criteria above
 *   AVX2 fallback       : energy tolerance × 5
 *   NEON fallback       : energy tolerance × 10
 *   Scalar fallback     : energy tolerance × 50
 */
#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <span>

namespace nikola::physics {

// ---------------------------------------------------------------------------
// Acceptance criteria constants
// ---------------------------------------------------------------------------

/// Standard Candle: max allowed ΔE_rel over 100,000 steps (α=β=0)
inline constexpr double CANDLE_ENERGY_DRIFT_LIMIT   = 1e-6;

/// Long-run energy conservation: max ΔE_rel over 1,000,000 steps (α=β=0)
inline constexpr double LONGRUN_ENERGY_DRIFT_LIMIT  = 1e-5;

/// Symplectic reversibility: max L2 error after N-step forward + N-step backward
inline constexpr double REVERSIBILITY_ERROR_LIMIT   = 1e-12;

/// Viscosity decay accuracy: |E(t)/E_theory(t) − 1| must be below this value
inline constexpr double VISCOSITY_DECAY_ERROR_LIMIT = 1e-4;  ///< 0.01%

/// Resonance attack: max wavefunction amplitude before saturation
inline constexpr double RESONANCE_AMPLITUDE_LIMIT   = 4.5;  ///< balanced nonary + headroom

// Runtime monitoring thresholds (mirror latency_budget.hpp SCRAM values)
inline constexpr double ORACLE_DRIFT_RATE_WARNING  = 1e-7;  ///< |dH/dt| per step
inline constexpr double ORACLE_DRIFT_RATE_SCRAM    = 1e-5;

inline constexpr double ORACLE_AMPLITUDE_WARNING   = 4.0;
inline constexpr double ORACLE_AMPLITUDE_SCRAM     = 5.0;

/// SIMD-fallback tolerance multipliers (applied to the energy drift limits)
inline constexpr double SIMD_FACTOR_AVX512          = 1.0;
inline constexpr double SIMD_FACTOR_AVX2            = 5.0;
inline constexpr double SIMD_FACTOR_NEON            = 10.0;
inline constexpr double SIMD_FACTOR_SCALAR          = 50.0;

// ---------------------------------------------------------------------------
// Enumerations
// ---------------------------------------------------------------------------

/// SIMD execution level — controls tolerance relaxation in the calibration suite
enum class SimdLevel {
    AVX512,  ///< Full AVX-512 reference (strictest tolerances)
    AVX2,    ///< AVX2 fallback (5× energy tolerance)
    NEON,    ///< ARM NEON fallback (10× energy tolerance)
    SCALAR,  ///< No SIMD (50× energy tolerance)
};

/// Runtime alert level for SCRAM-qualified events (mirrors AlertLevel in latency_budget.hpp)
enum class OracleAlert : int {
    OK       = 0,
    WARNING  = 1,
    SCRAM    = 2,   ///< Trigger SCRAM: dump stack trace, restart physics container
};

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/**
 * @brief Result from energy-conservation check.
 */
struct EnergyCheckResult {
    bool   passed;          ///< true iff drift_ratio < limit
    double drift_ratio;     ///< |H(t) - H(0)| / |H(0)|  (0.0 if H(0)==0)
    double limit_used;      ///< the threshold applied (SIMD-adjusted if applicable)
};

/**
 * @brief Result from symplectic reversibility check.
 */
struct ReversibilityResult {
    bool   passed;          ///< true iff l2_error < REVERSIBILITY_ERROR_LIMIT
    double l2_error;        ///< ‖Ψ(0) - Ψ_fwd_bwd(0)‖²  (L2 squared norm)
};

/**
 * @brief Result from numerical-viscosity / damping-accuracy check.
 */
struct ViscosityResult {
    bool   passed;          ///< true iff decay_error < VISCOSITY_DECAY_ERROR_LIMIT
    double decay_error;     ///< |E(t) / E_theory(t) - 1|
    double E_theory;        ///< analytical reference value at time t
};

/**
 * @brief Result from resonance-attack / amplitude-saturation check.
 */
struct ResonanceResult {
    bool   passed;          ///< true iff amplitude < RESONANCE_AMPLITUDE_LIMIT
    double max_amplitude;   ///< max |Ψ_i| observed
};

// ---------------------------------------------------------------------------
// PhysicsOracle — stateless calibration + runtime-monitoring engine
// ---------------------------------------------------------------------------

/**
 * @brief Stateless Physics Oracle engine (all methods static).
 *
 * Implements the GAP-030 calibration suite as pure functions of their inputs.
 * No physics simulation is contained here — the oracle classifies measurements
 * produced by the live physics kernel.
 */
class PhysicsOracle {
public:
    PhysicsOracle() = delete;

    // ------------------------------------------------------------------
    // SIMD tolerance helpers
    // ------------------------------------------------------------------

    /**
     * @brief Tolerance multiplier for the given SIMD execution level.
     */
    [[nodiscard]] static constexpr double simd_factor(SimdLevel lvl) noexcept {
        switch (lvl) {
            case SimdLevel::AVX512: return SIMD_FACTOR_AVX512;
            case SimdLevel::AVX2:   return SIMD_FACTOR_AVX2;
            case SimdLevel::NEON:   return SIMD_FACTOR_NEON;
            case SimdLevel::SCALAR: return SIMD_FACTOR_SCALAR;
        }
        return SIMD_FACTOR_SCALAR; // unreachable
    }

    /**
     * @brief Effective energy drift limit for a SIMD level.
     *
     * AVX-512 reference → 1e-5 (long-run) / 1e-6 (Standard Candle)
     * Looser fallback levels multiply these by simd_factor().
     */
    [[nodiscard]] static constexpr double effective_candle_limit(SimdLevel lvl) noexcept {
        return CANDLE_ENERGY_DRIFT_LIMIT * simd_factor(lvl);
    }

    [[nodiscard]] static constexpr double effective_longrun_limit(SimdLevel lvl) noexcept {
        return LONGRUN_ENERGY_DRIFT_LIMIT * simd_factor(lvl);
    }

    // ------------------------------------------------------------------
    // Test Case A — Standard Candle (energy conservation, closed system)
    // ------------------------------------------------------------------

    /**
     * @brief Evaluate Standard Candle energy conservation.
     *
     * Spec: Initialize Gaussian soliton, α=0, β=0, run 100k steps.
     * Pass criterion: ΔE_rel = |H_final - H_initial| / |H_initial| < 1e-6
     * (SIMD-adjusted via effective_candle_limit).
     *
     * @param H_initial  Hamiltonian at t=0
     * @param H_final    Hamiltonian after N steps
     * @param lvl        SIMD execution level (relaxes tolerance for fallback paths)
     */
    [[nodiscard]] static constexpr EnergyCheckResult
    check_standard_candle(double H_initial, double H_final,
                          SimdLevel lvl = SimdLevel::AVX512) noexcept {
        const double limit = effective_candle_limit(lvl);
        double drift = 0.0;
        if (H_initial != 0.0) {
            drift = std::abs((H_final - H_initial) / H_initial);
        }
        return {drift < limit, drift, limit};
    }

    // ------------------------------------------------------------------
    // Test Case B — Long-run energy conservation
    // ------------------------------------------------------------------

    /**
     * @brief Evaluate long-run energy conservation over ~1M steps.
     *
     * Spec pass criterion: ΔE_rel < 1e-5 (AVX-512 reference).
     */
    [[nodiscard]] static constexpr EnergyCheckResult
    check_energy_conservation(double H_initial, double H_final,
                               SimdLevel lvl = SimdLevel::AVX512) noexcept {
        const double limit = effective_longrun_limit(lvl);
        double drift = 0.0;
        if (H_initial != 0.0) {
            drift = std::abs((H_final - H_initial) / H_initial);
        }
        return {drift < limit, drift, limit};
    }

    // ------------------------------------------------------------------
    // Test Case C — Viscosity Trap (numerical damping accuracy)
    // ------------------------------------------------------------------

    /**
     * @brief Evaluate numerical viscosity (damping accuracy).
     *
     * Spec: α > 0. Energy must decay as E_theory(t) = E₀ · exp(−2αt).
     * Pass criterion:  |E_actual / E_theory(t) - 1| < 0.01% (1e-4)
     *
     * @param E_actual   Measured energy at time t
     * @param E_0        Initial energy at t=0
     * @param alpha      Damping coefficient (α > 0)
     * @param t          Elapsed simulation time (seconds; dt × steps)
     */
    [[nodiscard]] static ViscosityResult
    check_viscosity_trap(double E_actual, double E_0,
                         double alpha, double t) noexcept {
        // Analytical reference: E₀ · exp(−2αt)
        const double E_theory = E_0 * std::exp(-2.0 * alpha * t);
        double decay_error = 0.0;
        if (E_theory != 0.0) {
            decay_error = std::abs(E_actual / E_theory - 1.0);
        }
        return {decay_error < VISCOSITY_DECAY_ERROR_LIMIT, decay_error, E_theory};
    }

    // ------------------------------------------------------------------
    // Symplectic Reversibility Check
    // ------------------------------------------------------------------

    /**
     * @brief Evaluate symplectic structure preservation by reversibility.
     *
     * Spec: run N steps forward (Δt > 0) then N steps backward (Δt → -Δt).
     * Pass criterion:  ε_rev = ‖Ψ_initial - Ψ_recovered‖² < 1e-12
     *
     * This function accepts two equal-length spans of wavefunction amplitudes
     * (complex magnitudes or real components) and computes the L2 squared norm
     * of their element-wise difference.
     *
     * @param initial    Wavefunction state before any steps
     * @param recovered  Wavefunction state after forward + backward run
     */
    [[nodiscard]] static double
    compute_reversibility_error(std::span<const double> initial,
                                std::span<const double> recovered) noexcept {
        if (initial.size() != recovered.size() || initial.empty()) return 0.0;
        double acc = 0.0;
        for (std::size_t i = 0; i < initial.size(); ++i) {
            const double d = initial[i] - recovered[i];
            acc += d * d;
        }
        return acc;
    }

    [[nodiscard]] static ReversibilityResult
    check_reversibility(std::span<const double> initial,
                        std::span<const double> recovered) noexcept {
        const double err = compute_reversibility_error(initial, recovered);
        return {err < REVERSIBILITY_ERROR_LIMIT, err};
    }

    // ------------------------------------------------------------------
    // Resonance Attack (amplitude saturation)
    // ------------------------------------------------------------------

    /**
     * @brief Evaluate resonance-attack / amplitude-saturation criterion.
     *
     * Spec: drive the system at an eigenmode frequency (β > 0, nonlinear active).
     * Pass criterion: max |Ψ_i| must not exceed 4.5 (balanced nonary + headroom).
     * Ensures the nonlinear soliton term saturates amplitude before overflow.
     *
     * @param psi_max  Maximum wavefunction amplitude observed during the test
     */
    [[nodiscard]] static constexpr ResonanceResult
    check_resonance_attack(double psi_max) noexcept {
        return {psi_max < RESONANCE_AMPLITUDE_LIMIT, psi_max};
    }

    // ------------------------------------------------------------------
    // Runtime monitoring (called every tick by the live Oracle sidecar)
    // ------------------------------------------------------------------

    /**
     * @brief Alert level for a Hamiltonian drift *rate* (|dH/dt| per step).
     *
     * Spec runtime monitoring:
     *   |dH/dt| > 1e-7 → WARNING
     *   |dH/dt| > 1e-5 → SCRAM
     */
    [[nodiscard]] static constexpr OracleAlert
    drift_rate_alert(double abs_dH_dt) noexcept {
        if (abs_dH_dt <= ORACLE_DRIFT_RATE_WARNING) return OracleAlert::OK;
        if (abs_dH_dt <= ORACLE_DRIFT_RATE_SCRAM)   return OracleAlert::WARNING;
        return OracleAlert::SCRAM;
    }

    /**
     * @brief Alert level for max wavefunction amplitude |Ψ|_max.
     *
     * Spec runtime monitoring:
     *   |Ψ|_max > 4.0 → WARNING
     *   |Ψ|_max > 5.0 → SCRAM (hard limit)
     */
    [[nodiscard]] static constexpr OracleAlert
    amplitude_alert(double psi_max) noexcept {
        if (psi_max <= ORACLE_AMPLITUDE_WARNING) return OracleAlert::OK;
        if (psi_max <= ORACLE_AMPLITUDE_SCRAM)   return OracleAlert::WARNING;
        return OracleAlert::SCRAM;
    }

    /**
     * @brief True if a value contains NaN or Inf — triggers immediate SCRAM.
     *
     * Must be called on every element of Ψ and every component of g_ij each tick.
     */
    [[nodiscard]] static constexpr bool is_decoherent(double v) noexcept {
        return !std::isfinite(v);
    }

    /**
     * @brief Scan a wavefunction span for NaN/Inf — O(N) but mandatory per spec.
     * Returns true if ANY element is non-finite (immediate SCRAM).
     */
    [[nodiscard]] static bool
    has_decoherence(std::span<const double> psi) noexcept {
        for (double v : psi) {
            if (!std::isfinite(v)) return true;
        }
        return false;
    }

    // ------------------------------------------------------------------
    // Composite calibration (mirrors spec PhysicsCalibration::TestResult)
    // ------------------------------------------------------------------

    /**
     * @brief Aggregate result from a full Standard Candle run.
     *
     * Mirrors the `TestResult` struct from the spec's `PhysicsCalibration` class.
     * Contains both energy drift and reversibility error for a single test run.
     */
    struct TestResult {
        bool   passed;
        double max_drift;               ///< ΔE_rel
        double reversibility_error;     ///< ε_rev (L2²)
    };

    /**
     * @brief Compose a TestResult from independent energy + reversibility checks.
     *
     * Mirrors spec's PhysicsCalibration::run_standard_candle() result.
     * Both sub-criteria must pass for overall pass.
     */
    [[nodiscard]] static constexpr TestResult
    make_test_result(const EnergyCheckResult& energy,
                     const ReversibilityResult& rev) noexcept {
        return {
            energy.passed && rev.passed,
            energy.drift_ratio,
            rev.l2_error
        };
    }

    /**
     * @brief Analytical energy of a Gaussian soliton in a flat metric.
     *
     * In the Standard Candle test (α=0, β=0, flat metric g=δ_ij), the Hamiltonian
     * is conserved and equals the initial kinetic + potential energy.  This helper
     * provides a reference value for a unit Gaussian with amplitude A, width σ,
     * in a 1D projection (used in test setup to construct reproducible H_initial).
     *
     * E = A² · σ · √(π/2)  (L2 norm squared × shape factor)
     *
     * @param amplitude  Peak amplitude A of the Gaussian
     * @param sigma      Width σ (standard deviation in lattice units)
     */
    [[nodiscard]] static double
    gaussian_soliton_energy(double amplitude, double sigma) noexcept {
        return amplitude * amplitude * sigma * std::sqrt(std::numbers::pi / 2.0);
    }
};

} // namespace nikola::physics
