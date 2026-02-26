/**
 * @file hebbian_metric.hpp
 * @brief GAP-031: Hebbian-Riemannian metric convergence for the 9D toroidal manifold.
 *
 * Implements the Hebbian-Riemannian plasticity rule governing the evolution of
 * the 9×9 metric tensor g_ij(x,t) that encodes learned geometry.
 *
 * **Update Rule** (§GAP-031, continuous form):
 *   ∂g/∂t = −η(D_t)·Re(Ψ_i·Ψ_j*) + λ(S_t)·(g − I)
 *
 *   Discrete Euler step (Δt = 1 ms):
 *   g_{t+1} = g_t + Δt·[−η·C − λ·(g_t − I)]
 *
 *   where C_ij = Re(Ψ_i·Ψ_j*) is the instantaneous correlation matrix.
 *
 * **Equilibrium**: g* = I − (η/λ)·C
 *
 * **Convergence**: ε(t) = g(t) − g* decays as ε(0)·exp(−λ·t)
 *
 * **SPD Guarantee** ("Geometric Firewall"):
 *   After each update, if Cholesky decomposition fails (non-SPD detected),
 *   Tikhonov regularisation is applied: λ_i ← max(λ_i, ε_min = 1e−6).
 *
 * **Lyapunov energy** (strictly non-increasing):
 *   E(g) = (λ/2)·||g − I||_F² + η·Tr(g·C)
 *
 * Spec: §"Mathematical Proof of Hebbian Metric Convergence (GAP-031)" in
 *       02_foundations/01_9d_toroidal_geometry.md
 *
 * Dependencies: Eigen 3.4 (system-installed at /usr/include/eigen3)
 */
#pragma once

#include <complex>
#include <cmath>
#include <stdexcept>
#include <string>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

namespace nikola::math {

// ─────────────────────────────────────────────────────────────────────────────
//  Type aliases
// ─────────────────────────────────────────────────────────────────────────────

/// 9×9 symmetric positive-definite metric tensor (double precision).
using MetricTensor = Eigen::Matrix<double, 9, 9>;

/// 9-component complex wavefunction vector (one component per dimension).
using WavefunctionVec = Eigen::Matrix<std::complex<double>, 9, 1>;

/// 9×9 real correlation matrix C_ij = Re(Ψ_i · Ψ_j*).
using CorrelationMatrix = Eigen::Matrix<double, 9, 9>;

// ─────────────────────────────────────────────────────────────────────────────
//  Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Physics engine time step (1 kHz = 1 ms).
inline constexpr double HM_DT_PHYSICS  = 1.0e-3;

/// Minimum eigenvalue floor for Tikhonov regularisation (ε_min).
inline constexpr double HM_EPSILON_MIN = 1.0e-6;

/// Frobenius norm convergence criterion for unit-test equality checks.
inline constexpr double HM_FROB_TOL    = 1.0e-4;

// ─────────────────────────────────────────────────────────────────────────────
//  Core mathematical primitives
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Compute the 9×9 correlation matrix C_ij = Re(Ψ_i · Ψ_j*).
 *
 * C is symmetric and positive semi-definite by construction.
 * Physically it encodes the pairwise co-activation strength of the
 * nine wavefunction dimensions.
 */
[[nodiscard]] inline CorrelationMatrix compute_correlation(
    const WavefunctionVec& psi) noexcept
{
    // Outer product Ψ ⊗ Ψ† then take real part
    return (psi * psi.adjoint()).real();
}

/**
 * @brief Check whether a 9×9 matrix is symmetric positive definite (SPD)
 *        by attempting an Eigen LLT Cholesky decomposition.
 *
 * @return true  if decomposition succeeds (matrix is SPD).
 * @return false if decomposition fails (eigenvalue ≤ 0) or matrix is
 *               not numerically symmetric.
 */
[[nodiscard]] inline bool is_spd(const MetricTensor& g) noexcept
{
    // symmetry check
    if ((g - g.transpose()).cwiseAbs().maxCoeff() > 1.0e-10) return false;
    const Eigen::LLT<MetricTensor> llt(g);
    return llt.info() == Eigen::Success;
}

/**
 * @brief Project a matrix into the SPD cone via Tikhonov regularisation.
 *
 * Algorithm (§GAP-031 "Riemannian Projection via Lazy Cholesky"):
 *  1. Symmetrise g ← (g + gᵀ) / 2.
 *  2. Compute eigendecomposition g = Q · Λ · Qᵀ.
 *  3. Floor eigenvalues: λ_i ← max(λ_i, eps).
 *  4. Reconstruct: g_safe = Q · diag(λ') · Qᵀ.
 *
 * @param g    Input (possibly non-SPD) symmetric matrix.
 * @param eps  Minimum eigenvalue (default: HM_EPSILON_MIN = 1e-6).
 * @return SPD matrix guaranteed to have all eigenvalues ≥ eps.
 */
[[nodiscard]] inline MetricTensor project_to_spd(
    const MetricTensor& g,
    double eps = HM_EPSILON_MIN) noexcept
{
    const MetricTensor sym = (g + g.transpose()) * 0.5;
    const Eigen::SelfAdjointEigenSolver<MetricTensor> es(sym);
    auto eigenvalues = es.eigenvalues().cwiseMax(eps);
    return es.eigenvectors() *
           eigenvalues.asDiagonal() *
           es.eigenvectors().transpose();
}

/**
 * @brief Compute one Euler step of the Hebbian-Riemannian plasticity rule.
 *
 *   g_{t+1} = g_t + dt · [−η · C − λ · (g_t − I)]
 *
 * @param g      Current metric tensor.
 * @param C      Correlation matrix at time t (treated as constant / adiabatic).
 * @param eta    Learning rate η (Dopamine gating, range [0, 1]).
 * @param lambda Relaxation rate λ (Serotonin gating, range (0, ∞)).
 * @param dt     Time step (default: HM_DT_PHYSICS = 1 ms).
 * @return Updated metric tensor (NOT projected to SPD — call project_to_spd
 *         separately if the update makes the matrix non-SPD).
 * @throws std::domain_error if eta < 0, lambda ≤ 0, or dt ≤ 0.
 */
[[nodiscard]] inline MetricTensor hebbian_update(
    const MetricTensor&   g,
    const CorrelationMatrix& C,
    double eta,
    double lambda,
    double dt = HM_DT_PHYSICS)
{
    if (eta < 0.0)    throw std::domain_error("hebbian_update: eta must be >= 0");
    if (lambda <= 0.0) throw std::domain_error("hebbian_update: lambda must be > 0");
    if (dt <= 0.0)     throw std::domain_error("hebbian_update: dt must be > 0");

    const MetricTensor I = MetricTensor::Identity();
    const MetricTensor g_dot = -eta * C - lambda * (g - I);
    return g + dt * g_dot;
}

/**
 * @brief Compute the Lyapunov (Geometrodynamic) potential energy.
 *
 *   E(g) = (λ/2) · ||g − I||_F² + η · Tr(g · C)
 *
 * Strictly non-increasing along trajectories of the Hebbian update:
 *   dE/dt = −||ġ||_F² ≤ 0
 *
 * At equilibrium g*, E is minimised and its gradient vanishes.
 */
[[nodiscard]] inline double lyapunov_energy(
    const MetricTensor&      g,
    const CorrelationMatrix& C,
    double eta,
    double lambda) noexcept
{
    const MetricTensor I  = MetricTensor::Identity();
    const MetricTensor dg = g - I;
    const double elastic  = (lambda / 2.0) * dg.squaredNorm();  // (λ/2)||g-I||_F²
    const double interact = eta * (g * C).trace();               // η·Tr(g·C)
    return elastic + interact;
}

// ─────────────────────────────────────────────────────────────────────────────
//  HebbianResult — output of multi-step integration
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @struct HebbianResult
 * @brief Returned by run_hebbian() after N integration steps.
 */
struct HebbianResult {
    MetricTensor g;           ///< Final metric tensor after all steps.
    int    scram_count{0};    ///< Number of SPD-projection (Soft SCRAM) events.
    double final_energy{0.0}; ///< Lyapunov energy at the final step.
};

/**
 * @brief Run N discrete Euler steps of Hebbian-Riemannian plasticity.
 *
 * After each step, if is_spd() returns false (Cholesky failure), the metric
 * is projected back to the SPD cone via project_to_spd() — a "Soft SCRAM".
 * The scram_count in the result records how often this was triggered.
 *
 * @param g0     Initial metric tensor.
 * @param C      Correlation matrix (held constant — adiabatic approximation).
 * @param eta    Learning rate η.
 * @param lambda Relaxation rate λ.
 * @param steps  Number of integration steps.
 * @param dt     Time step per step (default: HM_DT_PHYSICS).
 * @return HebbianResult {final g, scram_count, final_energy}.
 */
[[nodiscard]] inline HebbianResult run_hebbian(
    MetricTensor             g0,
    const CorrelationMatrix& C,
    double eta,
    double lambda,
    int    steps,
    double dt = HM_DT_PHYSICS)
{
    HebbianResult result;
    result.g = g0;
    result.scram_count = 0;

    for (int s = 0; s < steps; ++s) {
        result.g = hebbian_update(result.g, C, eta, lambda, dt);
        if (!is_spd(result.g)) {
            result.g = project_to_spd(result.g);
            ++result.scram_count;
        }
    }
    result.final_energy = lyapunov_energy(result.g, C, eta, lambda);
    return result;
}

/**
 * @brief Compute the analytical equilibrium metric g* = I − (η/λ)·C.
 *
 * @param C      Correlation matrix.
 * @param eta    Learning rate η.
 * @param lambda Relaxation rate λ (must be > 0).
 * @return Equilibrium metric (may need SPD projection if η/λ is large).
 */
[[nodiscard]] inline MetricTensor equilibrium_metric(
    const CorrelationMatrix& C,
    double eta,
    double lambda) noexcept
{
    return MetricTensor::Identity() - (eta / lambda) * C;
}

} // namespace nikola::math
