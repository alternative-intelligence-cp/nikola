/**
 * @file hamiltonian.hpp
 * @brief Energy operator and watchdog for the UFIE physics engine.
 *
 * Implements the discrete Hamiltonian for the Unified Field Interference
 * Equation (UFIE):
 *
 *   H = Σᵢ [ |Vᵢ|²                          (kinetic / velocity energy)
 *           + (c₀/(1+sᵢ))² · Σ_d |∂_d Ψᵢ|²  (gradient / field energy)
 *           + (β/2) · |Ψᵢ|⁴               ]  (nonlinear self-interaction)
 *
 * where:
 *   V = ∂Ψ/∂t      (velocity field)
 *   s = state field (modulates wave speed: c_eff = c₀/(1+s))
 *   β = nonlinearity coefficient
 *
 * The discrete gradient ∂_d Ψ is approximated with centred differences:
 *   ∂_d Ψ(x) ≈ [Ψ(x+e_d) - Ψ(x-e_d)] / (2·h_d)
 * For missing (vacuum) neighbours, the PML ghost value is used.
 *
 * The Hamiltonian also provides an energy watchdog that triggers a Quantum
 * Zeno Freeze (SCRAM) if |ΔH/H| exceeds the configured tolerance.
 *
 * Reference: nikola engineering report, Phase 1, Gap 1.5 (SCRAM recovery),
 *            IMP-03 (bootstrap validation), Section 7.1 (energy check).
 */
#pragma once

#include <nikola/physics/wave_function.hpp>
#include <nikola/foundation/complex_field.hpp>

#include <cmath>
#include <array>
#include <functional>
#include <stdexcept>
#include <limits>

namespace nikola::physics {

// ============================================================================
// Hamiltonian  (energy oracle)
// ============================================================================

/**
 * @brief Computes UFIE Hamiltonian and monitors energy conservation.
 *
 * Usage:
 * @code
 *   Hamiltonian ham;
 *   ham.set_c0(1.0f).set_beta(1.0f);
 *   double H0 = ham.compute(wf);        // baseline
 *   // ... propagate ...
 *   double H1 = ham.compute(wf);        // after N steps
 *   double drift = std::abs(H1 - H0) / H0;   // should be < 0.0001
 * @endcode
 */
class Hamiltonian {
public:
    // ------------------------------------------------------------------ configuration

    Hamiltonian& set_c0(float c0)     { c0_   = c0;   return *this; }
    Hamiltonian& set_beta(float beta) { beta_ = beta; return *this; }

    float c0()   const noexcept { return c0_; }
    float beta() const noexcept { return beta_; }

    // ------------------------------------------------------------------ energy computation

    /**
     * @brief Compute total discrete Hamiltonian H over all active nodes.
     *
     * H = Σᵢ [ |Vᵢ|²
     *         - c_eff(i)² · Re(Ψᵢ* · ∇²Ψᵢ)   (IBP field energy ≥ 0)
     *         + (β/2) · |Ψᵢ|⁴               ]
     *
     * Uses the integration-by-parts (IBP) identity for discrete periodic grids:
     *   Σᵢ |∇Ψᵢ|²_fd  ≡  -Σᵢ Re(Ψᵢ* · ∇²Ψᵢ)
     * so the 18-point Laplacian stencil is the same as in the propagator,
     * guaranteeing that the Störmer-Verlet integrator conserves THIS H.
     *
     * Uses double-precision Kahan accumulation.
     *
     * @param wf  WaveFunction on the grid.
     * @return    Total energy (dimensionless in natural units).
     */
    [[nodiscard]]
    double compute(const WaveFunction& wf) const noexcept {
        const foundation::TorusGrid& g = wf.grid();
        const size_t N = g.num_active_nodes();

        const float* pr = g.psi_real();
        const float* pi = g.psi_imag();
        const float* vr = g.vel_real();
        const float* vi = g.vel_imag();
        const float* sf = g.state_field();
        const bool fast = g.adjacency_valid();

        double H    = 0.0;
        double comp = 0.0;   // Kahan compensation

        for (size_t i = 0; i < N; ++i) {
            const double psi_r = pr[i], psi_i = pi[i];
            const double vel_r = vr[i], vel_i = vi[i];

            // Kinetic term: |V|²
            const double kinetic = vel_r*vel_r + vel_i*vel_i;

            // Effective wave speed
            const double c_eff = c0_ / (1.0 + sf[i]);
            const double c_eff_sq = c_eff * c_eff;

            // IBP field energy: -Re(Ψ* · ∇²Ψ) × c_eff²
            double lap_r, lap_i;
            if (fast)
                laplacian_fast(i, g, pr, pi, psi_r, psi_i, lap_r, lap_i);
            else
                laplacian_slow(i, g, pr, pi, psi_r, psi_i, lap_r, lap_i);
            const double field_energy = c_eff_sq * (-(psi_r*lap_r + psi_i*lap_i));

            // Nonlinear term: (β/2) |Ψ|⁴
            const double psi_sq = psi_r*psi_r + psi_i*psi_i;
            const double nonlinear = 0.5 * beta_ * psi_sq * psi_sq;

            const double node_energy = kinetic + field_energy + nonlinear;

            // Kahan add
            const double y = node_energy - comp;
            const double t = H + y;
            comp = (t - H) - y;
            H = t;
        }

        return H;
    }

    // ------------------------------------------------------------------ energy watchdog

    /**
     * @brief Check energy drift and optionally trigger SCRAM.
     *
     * Computes |ΔH/H₀| and calls the scram_callback if it exceeds tolerance.
     * Returns the fractional drift.
     *
     * @param H0             Reference energy (e.g., initial or previous step).
     * @param H1             Current energy.
     * @param tolerance      Maximum acceptable |ΔH/H₀| (default 1e-4 = 0.01%).
     * @param scram_callback Called when drift exceeds tolerance.  Signature:
     *                       void(double drift, double H0, double H1).
     *                       Default: throws std::runtime_error.
     * @return               Fractional drift |ΔH/H₀|.
     */
    double check_drift(
            double H0, double H1,
            double tolerance = 1e-4,
            std::function<void(double, double, double)> scram_callback = {}) const
    {
        if (H0 == 0.0) return 0.0;  // avoids division by zero at t=0

        const double drift = std::abs(H1 - H0) / std::abs(H0);
        if (drift > tolerance) {
            if (scram_callback) {
                scram_callback(drift, H0, H1);
            } else {
                throw std::runtime_error(
                    "Hamiltonian energy drift exceeded tolerance: |ΔH/H| = "
                    + std::to_string(drift * 100.0) + "%");
            }
        }
        return drift;
    }

    /**
     * @brief Verify system is in a valid initial state (bootstrap gate, IMP-03).
     *
     * Checks: 1) total energy > 0  (Pilot Wave successfully injected)
     *         2) energy is finite  (no NaN/Inf from seeding)
     *
     * @throws std::runtime_error on failure.
     */
    void verify_initial_conditions(const WaveFunction& wf) const {
        const double H = compute(wf);
        if (!std::isfinite(H))
            throw std::runtime_error("Bootstrap failed: Hamiltonian is not finite");
        if (H <= 0.0)
            throw std::runtime_error("Bootstrap failed: Total energy ≤ 0 (vacuum deadlock)");
        if (!wf.is_finite())
            throw std::runtime_error("Bootstrap failed: Field contains NaN/Inf");
    }

    // ------------------------------------------------------------------ per-node contributions (for diagnostics)

    /**
     * @brief Return the kinetic energy density at node i.
     */
    [[nodiscard]]
    double kinetic_at(const WaveFunction& wf, size_t i) const noexcept {
        const TorusGrid& g = wf.grid();
        const float vr = g.vel_real()[i], vi = g.vel_imag()[i];
        return static_cast<double>(vr*vr + vi*vi);
    }

    /**
     * @brief Return the nonlinear energy density at node i.
     */
    [[nodiscard]]
    double nonlinear_at(const WaveFunction& wf, size_t i) const noexcept {
        const TorusGrid& g = wf.grid();
        const float pr = g.psi_real()[i], pi = g.psi_imag()[i];
        const double psi_sq = static_cast<double>(pr*pr + pi*pi);
        return 0.5 * beta_ * psi_sq * psi_sq;
    }

private:
    // ------------------------------------------------------------------ Laplacian helpers (IBP)

    /**
     * @brief 18-point discrete Laplacian at node i (fast path: precomputed adj).
     *
     *   ∇²Ψ(i) = Σ_d [(Ψ⁺ + Ψ⁻ - 2Ψ) / h_d²],  vacuum → PML ghost.
     */
    inline void laplacian_fast(
            size_t i, const foundation::TorusGrid& g,
            const float* pr, const float* pi,
            double psi_r, double psi_i,
            double& lap_r, double& lap_i) const noexcept
    {
        const size_t* nbrs = g.get_neighbors_fast(i);
        lap_r = lap_i = 0.0;
        double cr = 0.0, ci = 0.0;   // Kahan compensation
        for (int d = 0; d < 9; ++d) {
            const double inv_h2 = 1.0 / (g.spacing(d) * g.spacing(d));
            float np_r, np_i, nm_r, nm_i;
            const size_t np = nbrs[2*d];
            if (np != foundation::VACUUM_NODE) { np_r=pr[np]; np_i=pi[np]; }
            else { auto g_=foundation::pml_ghost({(float)psi_r,(float)psi_i}); np_r=g_.real(); np_i=g_.imag(); }
            const size_t nm = nbrs[2*d+1];
            if (nm != foundation::VACUUM_NODE) { nm_r=pr[nm]; nm_i=pi[nm]; }
            else { auto g_=foundation::pml_ghost({(float)psi_r,(float)psi_i}); nm_r=g_.real(); nm_i=g_.imag(); }
            const double v_r = ((double)np_r + nm_r - 2.0*psi_r) * inv_h2;
            const double v_i = ((double)np_i + nm_i - 2.0*psi_i) * inv_h2;
            { double y=v_r-cr; double t=lap_r+y; cr=(t-lap_r)-y; lap_r=t; }
            { double y=v_i-ci; double t=lap_i+y; ci=(t-lap_i)-y; lap_i=t; }
        }
    }

    /// Slow path: same computation via hash-map neighbour lookup.
    inline void laplacian_slow(
            size_t i, const foundation::TorusGrid& g,
            const float* pr, const float* pi,
            double psi_r, double psi_i,
            double& lap_r, double& lap_i) const noexcept
    {
        const auto nbrs = g.get_neighbors(i);
        lap_r = lap_i = 0.0;
        double cr = 0.0, ci = 0.0;
        for (int d = 0; d < 9; ++d) {
            const double inv_h2 = 1.0 / (g.spacing(d) * g.spacing(d));
            float np_r, np_i, nm_r, nm_i;
            const size_t np = nbrs[2*d];
            if (np != foundation::VACUUM_NODE) { np_r=pr[np]; np_i=pi[np]; }
            else { auto g_=foundation::pml_ghost({(float)psi_r,(float)psi_i}); np_r=g_.real(); np_i=g_.imag(); }
            const size_t nm = nbrs[2*d+1];
            if (nm != foundation::VACUUM_NODE) { nm_r=pr[nm]; nm_i=pi[nm]; }
            else { auto g_=foundation::pml_ghost({(float)psi_r,(float)psi_i}); nm_r=g_.real(); nm_i=g_.imag(); }
            const double v_r = ((double)np_r + nm_r - 2.0*psi_r) * inv_h2;
            const double v_i = ((double)np_i + nm_i - 2.0*psi_i) * inv_h2;
            { double y=v_r-cr; double t=lap_r+y; cr=(t-lap_r)-y; lap_r=t; }
            { double y=v_i-ci; double t=lap_i+y; ci=(t-lap_i)-y; lap_i=t; }
        }
    }

    // ------------------------------------------------------------------ data

    float c0_   = DEFAULT_C0;
    float beta_ = DEFAULT_BETA;
};

} // namespace nikola::physics
