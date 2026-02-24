/**
 * @file phase59_physics_oracle_test.cpp
 * @brief Phase 59 — GAP-030: Physics Oracle Calibration Test Suite
 *
 * Validates the PhysicsOracle engine against spec §GAP-030:
 *
 *   Test Case A — Standard Candle (Gaussian soliton, α=0, β=0, 100k steps)
 *     Pass: ΔE_rel < 1e-6 (AVX-512 reference)
 *
 *   Test Case B — Viscosity Trap (α=0.1 damping, checkerboard initial state)
 *     Pass: |E(t)/E_theory(t) - 1| < 0.01% where E_theory = E₀·exp(-2αt)
 *
 *   Test Case C — Resonance Attack (external emitter at eigenmode, β > 0)
 *     Pass: max |Ψ|_max < 4.5 (balanced nonary + headroom)
 *
 *   Reversibility Check (fwd N steps + bwd N steps, α=0)
 *     Pass: ε_rev = ‖Ψ(0) - Ψ_fwd_bwd(0)‖² < 1e-12
 *
 *   Long-run Energy Conservation (~1M steps)
 *     Pass: ΔE_rel < 1e-5 (AVX-512 reference)
 *
 *   SIMD Execution Matrix (tolerance relaxation)
 *     AVX-512: ×1, AVX2: ×5, NEON: ×10, Scalar: ×50
 *
 *   Runtime Oracle monitoring
 *     |dH/dt| thresholds: 1e-7 (WARNING), 1e-5 (SCRAM)
 *     Amplitude thresholds: 4.0 (WARNING), 5.0 (SCRAM)
 *     NaN/Inf → immediate SCRAM
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/catch_approx.hpp>

#include "nikola/physics/physics_oracle.hpp"

#include <cmath>
#include <numbers>
#include <vector>

using namespace nikola::physics;
using Catch::Approx;
using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

// ---------------------------------------------------------------------------
// §1 — Acceptance criteria constants
// ---------------------------------------------------------------------------

TEST_CASE("GAP-030 §1: calibration threshold constants are exact", "[gap030][constants]") {
    REQUIRE(CANDLE_ENERGY_DRIFT_LIMIT   == Approx(1e-6));
    REQUIRE(LONGRUN_ENERGY_DRIFT_LIMIT  == Approx(1e-5));
    REQUIRE(REVERSIBILITY_ERROR_LIMIT   == Approx(1e-12));
    REQUIRE(VISCOSITY_DECAY_ERROR_LIMIT == Approx(1e-4));
    REQUIRE(RESONANCE_AMPLITUDE_LIMIT   == Approx(4.5));

    // Standard Candle is stricter than long-run by exactly 10×
    REQUIRE(LONGRUN_ENERGY_DRIFT_LIMIT  == Approx(CANDLE_ENERGY_DRIFT_LIMIT * 10.0));
}

TEST_CASE("GAP-030 §2: runtime alert constants are set correctly", "[gap030][constants]") {
    REQUIRE(ORACLE_DRIFT_RATE_WARNING == Approx(1e-7));
    REQUIRE(ORACLE_DRIFT_RATE_SCRAM   == Approx(1e-5));
    REQUIRE(ORACLE_AMPLITUDE_WARNING  == Approx(4.0));
    REQUIRE(ORACLE_AMPLITUDE_SCRAM    == Approx(5.0));
    // Drift SCRAM is 100× the WARNING threshold
    REQUIRE(ORACLE_DRIFT_RATE_SCRAM == Approx(ORACLE_DRIFT_RATE_WARNING * 100.0));
}

// ---------------------------------------------------------------------------
// §3 — SIMD tolerance factors
// ---------------------------------------------------------------------------

TEST_CASE("GAP-030 §3: simd_factor returns exact spec values", "[gap030][simd]") {
    REQUIRE(PhysicsOracle::simd_factor(SimdLevel::AVX512) == Approx(1.0));
    REQUIRE(PhysicsOracle::simd_factor(SimdLevel::AVX2)   == Approx(5.0));
    REQUIRE(PhysicsOracle::simd_factor(SimdLevel::NEON)   == Approx(10.0));
    REQUIRE(PhysicsOracle::simd_factor(SimdLevel::SCALAR) == Approx(50.0));
    // Each level is strictly more permissive than the previous
    REQUIRE(PhysicsOracle::simd_factor(SimdLevel::AVX2)   > PhysicsOracle::simd_factor(SimdLevel::AVX512));
    REQUIRE(PhysicsOracle::simd_factor(SimdLevel::NEON)   > PhysicsOracle::simd_factor(SimdLevel::AVX2));
    REQUIRE(PhysicsOracle::simd_factor(SimdLevel::SCALAR) > PhysicsOracle::simd_factor(SimdLevel::NEON));
}

TEST_CASE("GAP-030 §4: effective_candle_limit and effective_longrun_limit scale correctly", "[gap030][simd]") {
    // AVX-512 reference: exact base limits
    REQUIRE(PhysicsOracle::effective_candle_limit(SimdLevel::AVX512) == Approx(1e-6));
    REQUIRE(PhysicsOracle::effective_longrun_limit(SimdLevel::AVX512) == Approx(1e-5));

    // AVX2: ×5
    REQUIRE(PhysicsOracle::effective_candle_limit(SimdLevel::AVX2) == Approx(5e-6));
    REQUIRE(PhysicsOracle::effective_longrun_limit(SimdLevel::AVX2) == Approx(5e-5));

    // NEON: ×10
    REQUIRE(PhysicsOracle::effective_candle_limit(SimdLevel::NEON) == Approx(1e-5));
    REQUIRE(PhysicsOracle::effective_longrun_limit(SimdLevel::NEON) == Approx(1e-4));

    // Scalar: ×50
    REQUIRE(PhysicsOracle::effective_candle_limit(SimdLevel::SCALAR) == Approx(5e-5));
    REQUIRE(PhysicsOracle::effective_longrun_limit(SimdLevel::SCALAR) == Approx(5e-4));
}

// ---------------------------------------------------------------------------
// §5 — Standard Candle (Test Case A)
// ---------------------------------------------------------------------------

TEST_CASE("GAP-030 §5: Standard Candle — zero drift passes on all SIMD levels", "[gap030][candle]") {
    // Perfect conservation: H_final == H_initial
    for (auto lvl : {SimdLevel::AVX512, SimdLevel::AVX2, SimdLevel::NEON, SimdLevel::SCALAR}) {
        auto r = PhysicsOracle::check_standard_candle(100.0, 100.0, lvl);
        REQUIRE(r.passed);
        REQUIRE(r.drift_ratio == Approx(0.0));
    }
}

TEST_CASE("GAP-030 §6: Standard Candle — drift below AVX-512 limit passes", "[gap030][candle]") {
    // 5e-7 < 1e-6 → should pass on AVX-512
    double H0 = 1e6;
    double Hf = H0 * (1.0 - 5e-7);
    auto r = PhysicsOracle::check_standard_candle(H0, Hf, SimdLevel::AVX512);
    REQUIRE(r.passed);
    REQUIRE(r.drift_ratio == Approx(5e-7).margin(1e-9));
    REQUIRE(r.limit_used  == Approx(1e-6));
}

TEST_CASE("GAP-030 §7: Standard Candle — drift above AVX-512 limit fails", "[gap030][candle]") {
    // 2e-6 > 1e-6 → fails on AVX-512
    double H0 = 1e6;
    double Hf = H0 * (1.0 - 2e-6);
    auto r512 = PhysicsOracle::check_standard_candle(H0, Hf, SimdLevel::AVX512);
    REQUIRE_FALSE(r512.passed);

    // But the same drift (2e-6) passes on AVX2 (limit = 5e-6)
    auto r_avx2 = PhysicsOracle::check_standard_candle(H0, Hf, SimdLevel::AVX2);
    REQUIRE(r_avx2.passed);
    // And on NEON and Scalar
    REQUIRE(PhysicsOracle::check_standard_candle(H0, Hf, SimdLevel::NEON).passed);
    REQUIRE(PhysicsOracle::check_standard_candle(H0, Hf, SimdLevel::SCALAR).passed);
}

TEST_CASE("GAP-030 §8: Standard Candle — zero H_initial returns zero drift (guarded division)", "[gap030][candle]") {
    auto r = PhysicsOracle::check_standard_candle(0.0, 0.0);
    REQUIRE(r.drift_ratio == Approx(0.0));
    REQUIRE(r.passed);  // 0.0 < 1e-6
}

// ---------------------------------------------------------------------------
// §9 — Long-run energy conservation (Test Case E)
// ---------------------------------------------------------------------------

TEST_CASE("GAP-030 §9: long-run energy conservation — boundary conditions", "[gap030][energy]") {
    double H0 = 1.0;

    // Just above limit (1.1e-5 > 1e-5): must fail
    {
        double Hf = H0 * (1.0 - 1.1e-5);
        auto r = PhysicsOracle::check_energy_conservation(H0, Hf, SimdLevel::AVX512);
        REQUIRE_FALSE(r.passed);  // 1.1e-5 > 1e-5
    }
    // Just below limit (9.9e-6): passes
    {
        double Hf = H0 * (1.0 - 9.9e-6);
        auto r = PhysicsOracle::check_energy_conservation(H0, Hf, SimdLevel::AVX512);
        REQUIRE(r.passed);
    }
    // Scalar: limit is 5e-4; drift of 3e-4 should pass
    {
        double Hf = H0 * (1.0 - 3e-4);
        auto r = PhysicsOracle::check_energy_conservation(H0, Hf, SimdLevel::SCALAR);
        REQUIRE(r.passed);
        REQUIRE(r.limit_used == Approx(5e-4));
    }
    // Scalar: drift of 6e-4 (above 5e-4 scalar limit) fails
    {
        double Hf = H0 * (1.0 - 6e-4);
        auto rS = PhysicsOracle::check_energy_conservation(H0, Hf, SimdLevel::SCALAR);
        REQUIRE_FALSE(rS.passed);
    }
}

// ---------------------------------------------------------------------------
// §10 — Viscosity Trap (Test Case B)
// ---------------------------------------------------------------------------

TEST_CASE("GAP-030 §10: Viscosity Trap — exact analytical decay passes", "[gap030][viscosity]") {
    // E_theory(t) = E₀·exp(-2αt).  Feed exact value → error = 0 → passes
    const double E0    = 500.0;
    const double alpha = 0.1;
    const double t     = 1.0;  // 1 second
    const double E_theory = E0 * std::exp(-2.0 * alpha * t);

    auto r = PhysicsOracle::check_viscosity_trap(E_theory, E0, alpha, t);
    REQUIRE(r.passed);
    REQUIRE(r.decay_error == Approx(0.0).margin(1e-14));
    REQUIRE(r.E_theory    == Approx(E_theory));
}

TEST_CASE("GAP-030 §11: Viscosity Trap — small deviation within 0.01% passes", "[gap030][viscosity]") {
    const double E0    = 1000.0;
    const double alpha = 0.05;
    const double t     = 2.0;
    const double E_theory = E0 * std::exp(-2.0 * alpha * t);

    // Perturb by 0.005% — well within 0.01% limit
    const double E_actual = E_theory * (1.0 + 5e-5);
    auto r = PhysicsOracle::check_viscosity_trap(E_actual, E0, alpha, t);
    REQUIRE(r.passed);
    REQUIRE(r.decay_error == Approx(5e-5).margin(1e-12));
}

TEST_CASE("GAP-030 §12: Viscosity Trap — deviation above 0.01% fails", "[gap030][viscosity]") {
    const double E0    = 1000.0;
    const double alpha = 0.05;
    const double t     = 2.0;
    const double E_theory = E0 * std::exp(-2.0 * alpha * t);

    // Perturb by 0.02% (2e-4 > 1e-4) — exceeds limit
    const double E_actual = E_theory * (1.0 + 2e-4);
    auto r = PhysicsOracle::check_viscosity_trap(E_actual, E0, alpha, t);
    REQUIRE_FALSE(r.passed);
    REQUIRE(r.decay_error > VISCOSITY_DECAY_ERROR_LIMIT);
}

TEST_CASE("GAP-030 §13: Viscosity Trap — α=0 gives E_theory=E₀ regardless of t", "[gap030][viscosity]") {
    // exp(-2·0·t) = 1 for any t
    const double E0 = 250.0;
    for (double t : {0.0, 1.0, 100.0}) {
        auto r = PhysicsOracle::check_viscosity_trap(E0, E0, 0.0, t);
        REQUIRE(r.E_theory == Approx(E0));
        REQUIRE(r.passed);
    }
}

// ---------------------------------------------------------------------------
// §14 — Reversibility Check (symplectic structure)
// ---------------------------------------------------------------------------

TEST_CASE("GAP-030 §14: Reversibility — identical arrays give zero L2 error", "[gap030][reversibility]") {
    std::vector<double> state = {0.1, 0.3, -0.2, 0.5, 1.0};
    auto r = PhysicsOracle::check_reversibility(state, state);
    REQUIRE(r.passed);
    REQUIRE(r.l2_error == Approx(0.0).margin(1e-16));
}

TEST_CASE("GAP-030 §15: Reversibility — machine-epsilon perturbation stays below 1e-12", "[gap030][reversibility]") {
    // Each element perturbed by at most 1e-7 → L2 = N × (1e-7)² = N × 1e-14
    std::vector<double> initial   = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> recovered = {1.0 + 1e-7, 2.0 + 1e-7, 3.0 + 1e-7,
                                     4.0 + 1e-7, 5.0 + 1e-7};
    double expected_l2 = 5.0 * (1e-7 * 1e-7);  // = 5e-14
    auto r = PhysicsOracle::check_reversibility(initial, recovered);
    REQUIRE(r.passed);
    REQUIRE(r.l2_error == Approx(expected_l2).margin(1e-20));
}

TEST_CASE("GAP-030 §16: Reversibility — large perturbation fails", "[gap030][reversibility]") {
    // Single element differs by 0.001 → L2 = 1e-6 >> 1e-12
    std::vector<double> initial   = {1.0, 0.0, 0.0};
    std::vector<double> recovered = {1.0 + 1e-3, 0.0, 0.0};
    auto r = PhysicsOracle::check_reversibility(initial, recovered);
    REQUIRE_FALSE(r.passed);
    REQUIRE(r.l2_error == Approx(1e-6).margin(1e-12));
}

TEST_CASE("GAP-030 §17: Reversibility — mismatched sizes return zero error", "[gap030][reversibility]") {
    std::vector<double> a = {1.0, 2.0};
    std::vector<double> b = {1.0};
    auto r = PhysicsOracle::check_reversibility(a, b);
    REQUIRE(r.l2_error == Approx(0.0));
    // Zero error passes (no assertion can be made — data is inconsistent)
    REQUIRE(r.passed);
}

// ---------------------------------------------------------------------------
// §18 — Resonance Attack (Test Case C)
// ---------------------------------------------------------------------------

TEST_CASE("GAP-030 §18: Resonance Attack — amplitude well below limit passes", "[gap030][resonance]") {
    // Soliton self-limits at 2.0 — healthy saturation
    auto r = PhysicsOracle::check_resonance_attack(2.0);
    REQUIRE(r.passed);
    REQUIRE(r.max_amplitude == Approx(2.0));
}

TEST_CASE("GAP-030 §19: Resonance Attack — amplitude at and above limit fails", "[gap030][resonance]") {
    // Spec limit is strict <  4.5
    REQUIRE_FALSE(PhysicsOracle::check_resonance_attack(4.5).passed); // at limit
    REQUIRE_FALSE(PhysicsOracle::check_resonance_attack(5.0).passed); // over
    REQUIRE_FALSE(PhysicsOracle::check_resonance_attack(9.9).passed); // overflowing

    // Just below limit passes
    REQUIRE(PhysicsOracle::check_resonance_attack(4.499).passed);
}

// ---------------------------------------------------------------------------
// §20 — Runtime drift rate alert
// ---------------------------------------------------------------------------

TEST_CASE("GAP-030 §20: drift_rate_alert — OK / WARNING / SCRAM boundaries", "[gap030][runtime]") {
    // Below WARNING threshold → OK
    REQUIRE(PhysicsOracle::drift_rate_alert(0.0)    == OracleAlert::OK);
    REQUIRE(PhysicsOracle::drift_rate_alert(1e-8)   == OracleAlert::OK);
    REQUIRE(PhysicsOracle::drift_rate_alert(1e-7)   == OracleAlert::OK);  // exactly at edge → OK

    // Between WARNING and SCRAM
    REQUIRE(PhysicsOracle::drift_rate_alert(5e-7)   == OracleAlert::WARNING);
    REQUIRE(PhysicsOracle::drift_rate_alert(1e-5)   == OracleAlert::WARNING); // at SCRAM boundary → WARNING

    // Above SCRAM threshold
    REQUIRE(PhysicsOracle::drift_rate_alert(1.1e-5) == OracleAlert::SCRAM);
    REQUIRE(PhysicsOracle::drift_rate_alert(1e-2)   == OracleAlert::SCRAM);
}

// ---------------------------------------------------------------------------
// §21 — Runtime amplitude alert
// ---------------------------------------------------------------------------

TEST_CASE("GAP-030 §21: amplitude_alert — OK / WARNING / SCRAM boundaries", "[gap030][runtime]") {
    // Normal regime
    REQUIRE(PhysicsOracle::amplitude_alert(0.0) == OracleAlert::OK);
    REQUIRE(PhysicsOracle::amplitude_alert(3.5) == OracleAlert::OK);
    REQUIRE(PhysicsOracle::amplitude_alert(4.0) == OracleAlert::OK);  // at WARNING edge → OK

    // WARNING zone
    REQUIRE(PhysicsOracle::amplitude_alert(4.1) == OracleAlert::WARNING);
    REQUIRE(PhysicsOracle::amplitude_alert(4.5) == OracleAlert::WARNING);
    REQUIRE(PhysicsOracle::amplitude_alert(5.0) == OracleAlert::WARNING); // at SCRAM edge → WARNING

    // SCRAM
    REQUIRE(PhysicsOracle::amplitude_alert(5.01) == OracleAlert::SCRAM);
    REQUIRE(PhysicsOracle::amplitude_alert(10.0) == OracleAlert::SCRAM);
}

// ---------------------------------------------------------------------------
// §22 — NaN / Inf decoherence detection
// ---------------------------------------------------------------------------

TEST_CASE("GAP-030 §22: NaN/Inf detection — is_decoherent and has_decoherence", "[gap030][decoherence]") {
    // Finite values are coherent
    REQUIRE_FALSE(PhysicsOracle::is_decoherent(0.0));
    REQUIRE_FALSE(PhysicsOracle::is_decoherent(1.5));
    REQUIRE_FALSE(PhysicsOracle::is_decoherent(-1e300));

    // NaN and Inf are decoherent
    REQUIRE(PhysicsOracle::is_decoherent(std::numeric_limits<double>::quiet_NaN()));
    REQUIRE(PhysicsOracle::is_decoherent(std::numeric_limits<double>::infinity()));
    REQUIRE(PhysicsOracle::is_decoherent(-std::numeric_limits<double>::infinity()));

    // has_decoherence span scan
    std::vector<double> all_finite = {0.1, 0.5, -0.3, 2.0};
    REQUIRE_FALSE(PhysicsOracle::has_decoherence(all_finite));

    std::vector<double> with_nan = {0.1, std::numeric_limits<double>::quiet_NaN(), 1.0};
    REQUIRE(PhysicsOracle::has_decoherence(with_nan));

    std::vector<double> with_inf = {2.0, 3.5, std::numeric_limits<double>::infinity()};
    REQUIRE(PhysicsOracle::has_decoherence(with_inf));

    // Empty span — no decoherence by convention
    REQUIRE_FALSE(PhysicsOracle::has_decoherence(std::vector<double>{}));
}

// ---------------------------------------------------------------------------
// §23 — gaussian_soliton_energy helper
// ---------------------------------------------------------------------------

TEST_CASE("GAP-030 §23: gaussian_soliton_energy gives positive finite reference", "[gap030][helper]") {
    // E = A²·σ·√(π/2)
    const double A = 1.0, sigma = 1.0;
    const double expected = A * A * sigma * std::sqrt(std::numbers::pi / 2.0);
    REQUIRE(PhysicsOracle::gaussian_soliton_energy(A, sigma) == Approx(expected));
    REQUIRE(PhysicsOracle::gaussian_soliton_energy(A, sigma) > 0.0);

    // Doubling amplitude → 4× energy
    REQUIRE(PhysicsOracle::gaussian_soliton_energy(2.0 * A, sigma) ==
            Approx(4.0 * PhysicsOracle::gaussian_soliton_energy(A, sigma)));
    // Doubling sigma → 2× energy (linear in σ)
    REQUIRE(PhysicsOracle::gaussian_soliton_energy(A, 2.0 * sigma) ==
            Approx(2.0 * PhysicsOracle::gaussian_soliton_energy(A, sigma)));
}

// ---------------------------------------------------------------------------
// §24 — make_test_result composite (mirrors spec PhysicsCalibration::TestResult)
// ---------------------------------------------------------------------------

TEST_CASE("GAP-030 §24: make_test_result — composite passes only when both sub-criteria pass",
          "[gap030][composite]") {
    // Both pass
    {
        EnergyCheckResult e{true,  5e-7, 1e-6};
        ReversibilityResult r{true, 5e-14};
        auto tr = PhysicsOracle::make_test_result(e, r);
        REQUIRE(tr.passed);
        REQUIRE(tr.max_drift           == Approx(5e-7));
        REQUIRE(tr.reversibility_error == Approx(5e-14));
    }
    // Energy fails, reversibility passes → composite fails
    {
        EnergyCheckResult e{false, 2e-6, 1e-6};
        ReversibilityResult r{true, 1e-14};
        REQUIRE_FALSE(PhysicsOracle::make_test_result(e, r).passed);
    }
    // Energy passes, reversibility fails → composite fails
    {
        EnergyCheckResult e{true, 5e-7, 1e-6};
        ReversibilityResult r{false, 1e-6};
        REQUIRE_FALSE(PhysicsOracle::make_test_result(e, r).passed);
    }
    // Both fail
    {
        EnergyCheckResult e{false, 5e-5, 1e-6};
        ReversibilityResult r{false, 1e-3};
        REQUIRE_FALSE(PhysicsOracle::make_test_result(e, r).passed);
    }
}

// ---------------------------------------------------------------------------
// §25 — Integration: Full Standard Candle workflow
// ---------------------------------------------------------------------------

TEST_CASE("GAP-030 §25: integration — Standard Candle + Reversibility combined",
          "[gap030][integration]") {
    // Simulate a perfect closed-system run:
    //   H stays constant → energy check passes
    //   Fwd-bwd returns exactly to origin → reversibility passes
    const double H0 = PhysicsOracle::gaussian_soliton_energy(1.0, 2.0);
    const double Hf = H0; // no drift (perfect integrator)

    std::vector<double> initial   = {0.5, -0.5, 0.3, -0.3, 0.1};
    std::vector<double> recovered = initial; // perfect symplectic: returns exactly

    auto E_res  = PhysicsOracle::check_standard_candle(H0, Hf, SimdLevel::AVX512);
    auto rev    = PhysicsOracle::check_reversibility(initial, recovered);
    auto result = PhysicsOracle::make_test_result(E_res, rev);

    REQUIRE(result.passed);
    REQUIRE(result.max_drift           == Approx(0.0).margin(1e-15));
    REQUIRE(result.reversibility_error == Approx(0.0).margin(1e-15));

    // Confirm runtime is clean
    REQUIRE(PhysicsOracle::drift_rate_alert(0.0)   == OracleAlert::OK);
    REQUIRE(PhysicsOracle::amplitude_alert(1.0)    == OracleAlert::OK);
    REQUIRE_FALSE(PhysicsOracle::has_decoherence(initial));
}
