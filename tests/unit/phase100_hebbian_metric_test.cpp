/**
 * phase100_hebbian_metric_test.cpp
 *
 * Phase 100 — GAP-031: Hebbian-Riemannian Metric Convergence
 *
 * Tests the Eigen3-backed 9×9 metric-tensor plasticity implementation,
 * including the update rule, Lyapunov energy, SPD predicates, Tikhonov
 * projection, and analytical convergence rate.
 *
 * Sections:
 *   1. Compile-time constants
 *   2. compute_correlation(): structure and known values
 *   3. is_spd(): positive and negative examples
 *   4. project_to_spd(): eigenvalue floor and SPD guarantee
 *   5. hebbian_update(): single-step formula correctness + domain errors
 *   6. lyapunov_energy(): formula verification + strict decrease
 *   7. Convergence: g → g* at rate exp(−λt)
 *   8. Discrete stability: λΔt ≤ 1 implies monotone convergence
 *   9. SPD enforcement: pathological input triggers Soft SCRAM
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "nikola/math/hebbian_metric.hpp"

#include <cmath>
#include <complex>

using namespace nikola::math;
using Catch::Approx;

// ─── helpers ─────────────────────────────────────────────────────────────────

static double frob(const MetricTensor& a, const MetricTensor& b)
{
    return (a - b).norm();  // Frobenius norm of difference
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 1 — Compile-time constants
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase100 — compile-time constants", "[phase100][constants]")
{
    // Runtime comparison (constexpr doubles checked for reasonable values)
    CHECK(HM_DT_PHYSICS   == Approx(1.0e-3));
    CHECK(HM_EPSILON_MIN  == Approx(1.0e-6));
    CHECK(HM_FROB_TOL     == Approx(1.0e-4));

    // MetricTensor dimensions
    STATIC_CHECK(MetricTensor::RowsAtCompileTime == 9);
    STATIC_CHECK(MetricTensor::ColsAtCompileTime == 9);

    // WavefunctionVec dimensions
    STATIC_CHECK(WavefunctionVec::RowsAtCompileTime == 9);
    STATIC_CHECK(WavefunctionVec::ColsAtCompileTime == 1);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 2 — compute_correlation(): structure and known values
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase100 — compute_correlation()", "[phase100][correlation]")
{
    SECTION("Zero wavefunction → zero correlation matrix") {
        WavefunctionVec psi = WavefunctionVec::Zero();
        auto C = compute_correlation(psi);
        CHECK(C.isZero(1.0e-12));
    }

    SECTION("Result is symmetric") {
        WavefunctionVec psi;
        for (int i = 0; i < 9; ++i)
            psi(i) = std::complex<double>(i + 1.0, i * 0.5);
        auto C = compute_correlation(psi);
        // C must equal its own transpose
        CHECK(C.isApprox(C.transpose(), 1.0e-12));
    }

    SECTION("Result is positive semi-definite (all eigenvalues >= 0)") {
        WavefunctionVec psi;
        for (int i = 0; i < 9; ++i)
            psi(i) = std::complex<double>(std::cos(i * 0.7), std::sin(i * 0.3));
        auto C = compute_correlation(psi);
        Eigen::SelfAdjointEigenSolver<CorrelationMatrix> es(C);
        CHECK((es.eigenvalues().minCoeff()) >= -1.0e-10);
    }

    SECTION("Real wavefunction: C_ij = Ψ_i * Ψ_j (imaginary part zero)") {
        WavefunctionVec psi = WavefunctionVec::Zero();
        psi(0) = std::complex<double>(3.0, 0.0);
        psi(1) = std::complex<double>(4.0, 0.0);
        auto C = compute_correlation(psi);
        // C_00 = 3*3 = 9, C_11 = 4*4 = 16, C_01 = C_10 = 3*4 = 12
        CHECK(C(0, 0) == Approx(9.0));
        CHECK(C(1, 1) == Approx(16.0));
        CHECK(C(0, 1) == Approx(12.0));
        CHECK(C(1, 0) == Approx(12.0));
    }

    SECTION("Complex wavefunction: C_ij = Re(Ψ_i * conj(Ψ_j))") {
        WavefunctionVec psi = WavefunctionVec::Zero();
        psi(0) = std::complex<double>(1.0, 1.0);  // |Ψ|² = 2
        psi(1) = std::complex<double>(1.0, -1.0); // |Ψ|² = 2
        auto C = compute_correlation(psi);
        // C_01 = Re((1+i)(1+i)) = Re(1 + 2i - 1) = Re(2i) = 0
        CHECK(C(0, 1) == Approx(0.0).margin(1.0e-12));
        CHECK(C(0, 0) == Approx(2.0));
        CHECK(C(1, 1) == Approx(2.0));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 3 — is_spd()
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase100 — is_spd()", "[phase100][spd]")
{
    SECTION("Identity matrix is SPD") {
        CHECK(is_spd(MetricTensor::Identity()) == true);
    }

    SECTION("2*Identity is SPD") {
        CHECK(is_spd(2.0 * MetricTensor::Identity()) == true);
    }

    SECTION("Matrix with one zero eigenvalue is not SPD") {
        MetricTensor g = MetricTensor::Identity();
        g(0, 0) = 0.0;  // zero eigenvalue → not positive definite
        CHECK(is_spd(g) == false);
    }

    SECTION("Matrix with one negative eigenvalue is not SPD") {
        MetricTensor g = MetricTensor::Identity();
        g(0, 0) = -1.0;  // negative eigenvalue
        CHECK(is_spd(g) == false);
    }

    SECTION("Asymmetric matrix is not SPD") {
        MetricTensor g = MetricTensor::Identity();
        g(0, 1) = 1.0;   // break symmetry
        g(1, 0) = 0.0;
        CHECK(is_spd(g) == false);
    }

    SECTION("Diagonal positive matrix is SPD") {
        MetricTensor g = MetricTensor::Zero();
        for (int i = 0; i < 9; ++i)
            g(i, i) = static_cast<double>(i + 1);  // diag(1,2,...,9)
        CHECK(is_spd(g) == true);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 4 — project_to_spd()
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase100 — project_to_spd()", "[phase100][project]")
{
    SECTION("Already-SPD matrix is unchanged within tolerance") {
        MetricTensor g = MetricTensor::Identity();
        auto gp = project_to_spd(g);
        // Projection of identity should be approx identity
        CHECK(frob(g, gp) < HM_FROB_TOL);
        CHECK(is_spd(gp) == true);
    }

    SECTION("Non-SPD matrix is projected to SPD") {
        MetricTensor g = MetricTensor::Identity();
        g(0, 0) = -0.5;  // negative eigenvalue
        CHECK(is_spd(g) == false);
        auto gp = project_to_spd(g);
        CHECK(is_spd(gp) == true);
    }

    SECTION("All eigenvalues of projected matrix >= eps_min") {
        MetricTensor g = MetricTensor::Identity();
        g(3, 3) = -100.0;  // very negative eigenvalue
        auto gp = project_to_spd(g, HM_EPSILON_MIN);
        Eigen::SelfAdjointEigenSolver<MetricTensor> es(gp);
        CHECK(es.eigenvalues().minCoeff() >= HM_EPSILON_MIN - 1.0e-10);
    }

    SECTION("Custom eps floor is respected") {
        MetricTensor g = MetricTensor::Identity();
        g(0, 0) = 0.0;
        const double custom_eps = 1.0e-3;
        auto gp = project_to_spd(g, custom_eps);
        Eigen::SelfAdjointEigenSolver<MetricTensor> es(gp);
        CHECK(es.eigenvalues().minCoeff() >= custom_eps - 1.0e-10);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 5 — hebbian_update(): single-step formula
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase100 — hebbian_update(): single step", "[phase100][update]")
{
    SECTION("With C=0 (no correlation), only relaxation term active") {
        // g_t+1 = g_t + dt * [0 - λ*(g_t - I)]
        //       = g_t + dt * λ * (I - g_t)
        // Starting from g = 2*I → g_t+1 = 2I + dt*λ*(I - 2I) = 2I - dt*λ*I
        const MetricTensor g = 2.0 * MetricTensor::Identity();
        const CorrelationMatrix C = CorrelationMatrix::Zero();
        const double eta = 1.0, lambda = 1.0, dt = 0.01;
        auto gn = hebbian_update(g, C, eta, lambda, dt);
        // Expected: 2I - 0.01*1.0*I = (1.99)*I
        const MetricTensor expected = (2.0 - dt * lambda) * MetricTensor::Identity();
        CHECK(frob(gn, expected) < 1.0e-12);
    }

    SECTION("With g = I and C = I, formula: g_t+1 = I − η·I·dt") {
        // g_t+1 = I + dt*[−η*I − λ*(I − I)] = I − dt*η*I
        const MetricTensor g = MetricTensor::Identity();
        const CorrelationMatrix C = CorrelationMatrix::Identity();
        const double eta = 0.5, lambda = 1.0, dt = 0.001;
        auto gn = hebbian_update(g, C, eta, lambda, dt);
        const MetricTensor expected = (1.0 - dt * eta) * MetricTensor::Identity();
        CHECK(frob(gn, expected) < 1.0e-12);
    }

    SECTION("Throws if eta < 0") {
        CHECK_THROWS_AS(
            hebbian_update(MetricTensor::Identity(),
                           CorrelationMatrix::Zero(),
                           -0.1, 1.0, 0.001),
            std::domain_error);
    }

    SECTION("Throws if lambda <= 0") {
        CHECK_THROWS_AS(
            hebbian_update(MetricTensor::Identity(),
                           CorrelationMatrix::Zero(),
                           0.1, 0.0, 0.001),
            std::domain_error);
    }

    SECTION("Throws if dt <= 0") {
        CHECK_THROWS_AS(
            hebbian_update(MetricTensor::Identity(),
                           CorrelationMatrix::Zero(),
                           0.1, 1.0, 0.0),
            std::domain_error);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 6 — lyapunov_energy(): formula and strict decrease
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase100 — lyapunov_energy()", "[phase100][lyapunov]")
{
    SECTION("At identity with zero correlation: E = 0 (elastic only, g=I → 0)") {
        const MetricTensor g = MetricTensor::Identity();
        const CorrelationMatrix C = CorrelationMatrix::Zero();
        const double E = lyapunov_energy(g, C, 0.5, 1.0);
        // E = (λ/2)||g - I||_F² + η·Tr(g·C) = 0 + 0 = 0
        CHECK(E == Approx(0.0).margin(1.0e-12));
    }

    SECTION("Elastic energy term: E = (λ/2)||g-I||² for C=0") {
        MetricTensor g = 2.0 * MetricTensor::Identity();
        const CorrelationMatrix C = CorrelationMatrix::Zero();
        const double lambda = 2.0;
        // ||g-I||_F² = ||I||_F² = 9  (I is 9x9 identity)
        const double expected = (lambda / 2.0) * 9.0;
        CHECK(lyapunov_energy(g, C, 0.0, lambda) == Approx(expected).margin(1.0e-12));
    }

    SECTION("Energy strictly decreases during Hebbian integration (10 steps)") {
        // Use diagonal C to ensure stability
        CorrelationMatrix C = CorrelationMatrix::Zero();
        for (int i = 0; i < 9; ++i) C(i, i) = 0.05 * (i + 1.0);

        MetricTensor g = 1.5 * MetricTensor::Identity();
        const double eta = 0.1, lambda = 1.0;

        double prev_E = lyapunov_energy(g, C, eta, lambda);
        for (int step = 0; step < 10; ++step) {
            g = hebbian_update(g, C, eta, lambda);
            double E = lyapunov_energy(g, C, eta, lambda);
            CHECK(E <= prev_E + 1.0e-12);  // non-increasing
            prev_E = E;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 7 — Convergence to equilibrium g* = I − (η/λ)·C
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase100 — convergence to g*", "[phase100][convergence]")
{
    SECTION("Analytical prediction: error decays exponentially at rate λ") {
        // Parameters: η=0.05, λ=2.0, dt=0.001
        const double eta = 0.05, lambda = 2.0;

        // Use a small diagonal correlation matrix to keep g* SPD easily
        CorrelationMatrix C = CorrelationMatrix::Zero();
        for (int i = 0; i < 9; ++i) C(i, i) = 0.1 * (i + 1.0);

        const MetricTensor g_star = equilibrium_metric(C, eta, lambda);
        CHECK(is_spd(g_star) == true);  // equilibrium must be SPD for this test

        // Start from identity
        const MetricTensor g0 = MetricTensor::Identity();
        const double error_0  = frob(g0, g_star);

        const int    steps = 2000;  // 2 seconds simulated
        const double t_end = steps * HM_DT_PHYSICS;  // 2.0 s

        auto result = run_hebbian(g0, C, eta, lambda, steps);
        const double error_N = frob(result.g, g_star);

        // Analytical: error_N ≈ error_0 * exp(−λ * t_end)
        //            = error_0 * exp(−2.0 * 2.0) = error_0 * exp(−4.0) ≈ 0.0183
        const double predicted = error_0 * std::exp(-lambda * t_end);
        // Allow 20% relative tolerance (numerical Euler vs exact ODE)
        CHECK(error_N < predicted * 1.20 + 1.0e-8);
        // Also check absolute: error must be much less than initial
        CHECK(error_N < error_0 * 0.05);  // at least 20× reduction
    }

    SECTION("equilibrium_metric() formula: g* = I − (η/λ)·C") {
        const double eta = 0.2, lambda = 4.0;
        CorrelationMatrix C = CorrelationMatrix::Identity();
        const MetricTensor g_star = equilibrium_metric(C, eta, lambda);
        // g* = I - 0.05 * I = 0.95 * I
        const MetricTensor expected = (1.0 - eta / lambda) * MetricTensor::Identity();
        CHECK(frob(g_star, expected) < 1.0e-12);
    }

    SECTION("No Soft SCRAM needed for stable parameters (λΔt = 0.002 << 1)") {
        CorrelationMatrix C = 0.1 * CorrelationMatrix::Identity();
        auto result = run_hebbian(MetricTensor::Identity(), C,
                                   0.1, 2.0, 1000, 0.001);
        CHECK(result.scram_count == 0);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 8 — Discrete stability: λΔt ≤ 1 → monotone convergence
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase100 — discrete stability criterion λΔt ≤ 1",
          "[phase100][stability]")
{
    SECTION("λΔt = 0.5 (stable, r = 0.5): error strictly decreases each step") {
        // With C=0, pure relaxation: g_t+1 = g_t + dt*λ*(I - g_t) = r*g_t + (1-r)*I
        // Starting from g = 2I, convergence toward I is monotone.
        const double lambda = 500.0, dt = 0.001;  // λΔt = 0.5
        MetricTensor g = 2.0 * MetricTensor::Identity();
        const CorrelationMatrix C = CorrelationMatrix::Zero();

        double prev_err = frob(g, MetricTensor::Identity());
        for (int s = 0; s < 20; ++s) {
            g = hebbian_update(g, C, 0.0, lambda, dt);
            double err = frob(g, MetricTensor::Identity());
            CHECK(err < prev_err + 1.0e-12);
            prev_err = err;
        }
    }

    SECTION("Maximum stable λ_max: given dt=0.001, λ_max = 1000") {
        // λΔt = 1000 * 0.001 = 1.0 — the boundary case
        const double lambda = 1000.0, dt = 0.001;
        // r = 1 - λΔt = 0.0 → each step error = 0 (reaches equilibrium in 1 step exactly)
        MetricTensor g = 2.0 * MetricTensor::Identity();
        const CorrelationMatrix C = CorrelationMatrix::Zero();
        auto gn = hebbian_update(g, C, 0.0, lambda, dt);
        // g_t+1 = g_t + 0.001 * (-1000 * (g_t - I)) = g_t + (-1) * (g_t - I) = I
        CHECK(frob(gn, MetricTensor::Identity()) < 1.0e-10);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 9 — SPD enforcement: pathological Soft SCRAM
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase100 — Soft SCRAM SPD enforcement", "[phase100][scram]")
{
    SECTION("Large η/λ → g* non-SPD → Soft SCRAM triggered, result always SPD") {
        // g* = I - (η/λ)*C. With η/λ = 10 and C = I, g* = I - 10*I = -9*I (non-SPD)
        // The integration will push g into non-SPD territory, triggering ScrAM.
        const double eta = 10.0, lambda = 1.0;
        CorrelationMatrix C = 2.0 * CorrelationMatrix::Identity();

        auto result = run_hebbian(MetricTensor::Identity(), C,
                                   eta, lambda, 500, 0.001);

        CHECK(is_spd(result.g) == true);   // final metric is always SPD
        CHECK(result.scram_count > 0);     // at least one projection was needed
    }

    SECTION("project_to_spd() applied to near-singular matrix preserves SPD") {
        MetricTensor g = MetricTensor::Identity() * 1.0e-10;  // near-singular
        auto gp = project_to_spd(g);
        CHECK(is_spd(gp) == true);
        Eigen::SelfAdjointEigenSolver<MetricTensor> es(gp);
        CHECK(es.eigenvalues().minCoeff() >= HM_EPSILON_MIN * 0.9);
    }
}
