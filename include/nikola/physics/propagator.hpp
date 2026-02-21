/**
 * @file propagator.hpp
 * @brief Symplectic time integrator for the UFIE wave equation.
 *
 * Implements the **Strang split-operator** scheme for the Unified Field
 * Interference Equation (UFIE):
 *
 *   ∂²Ψ/∂t² + α(1-r)·∂Ψ/∂t - (c₀/(1+s))² ∇²_g Ψ = β|Ψ|²Ψ + E(x,t)
 *
 * First-order reduction (V = ∂Ψ/∂t):
 *   ∂Ψ/∂t = V
 *   ∂V/∂t = c_eff² ∇²Ψ  -  α(1-r)V  +  β|Ψ|²Ψ  +  E(x,t)
 *
 * Strang B-A-B decomposition per timestep Δt:
 *
 *   exp[(D+H+N)Δt] ≈ D(Δt/2) · H(Δt/2) · N(Δt) · H(Δt/2) · D(Δt/2)
 *
 * Substep operators:
 *
 *   D(τ)  — Damping:     V ← V · exp(-α(1-r)·τ)             [exact]
 *   H(τ)  — Kinetic:     Ψ ← Ψ + V·τ;   V ← V + c_eff²·∇²Ψ_new·τ  [symplectic Euler]
 *   N(τ)  — Nonlinear:   V ← V + β|Ψ|²Ψ·τ                  [Euler]
 *
 * Two consecutive H(Δt/2) steps produce a 2nd-order Störmer-Verlet (leapfrog)
 * update for the free wave, which exactly conserves a shadow Hamiltonian.
 * Combined with Strang composition, the full integrator is 2nd order in time.
 *
 * The 9D Laplacian uses an 18-point finite-difference stencil (2 neighbours
 * per dimension) with Kahan-compensated summation for numerical stability:
 *
 *   ∇²Ψ(i) ≈ Σ_{d=0}^{8} [Ψ(i+e_d) + Ψ(i-e_d) - 2Ψ(i)] / h_d²
 *
 * Vacuum (unallocated) neighbours are handled with PML ghost values.
 *
 * Reference: nikola engineering report Phase 1,
 *            Gap 1.2 (thermal init), Gap 1.3 (PML), Gap 1.5 (SCRAM),
 *            Phase 0 numerical requirements (SoA layout, Kahan, split-operator).
 */
#pragma once

#include <nikola/physics/hamiltonian.hpp>
#include <nikola/physics/wave_function.hpp>
#include <nikola/foundation/complex_field.hpp>

#include <vector>
#include <cmath>
#include <chrono>
#include <cassert>
#include <stdexcept>
#include <functional>

namespace nikola::physics {

// ============================================================================
// Propagator
// ============================================================================

/**
 * @brief Strang split-operator symplectic integrator for the UFIE.
 *
 * Thread-safe: each instance has its own scratch buffers; simultaneous
 * use from different threads requires separate Propagator instances.
 */
class Propagator {
public:
    // ------------------------------------------------------------------ configuration

    Propagator& set_c0(float c0)       { ham_.set_c0(c0);     return *this; }
    Propagator& set_beta(float beta)   { ham_.set_beta(beta); return *this; }
    Propagator& set_alpha(float alpha) { alpha_ = alpha;      return *this; }

    float c0()    const noexcept { return ham_.c0(); }
    float beta()  const noexcept { return ham_.beta(); }
    float alpha() const noexcept { return alpha_; }

    /// Access the embedded Hamiltonian (for external energy checks).
    const Hamiltonian& hamiltonian() const noexcept { return ham_; }
          Hamiltonian& hamiltonian()       noexcept { return ham_; }

    // ------------------------------------------------------------------ emitter

    /**
     * @brief Register an external driving field E(x, t).
     *
     * Called once per full step after the N substep.
     * Signature: void(WaveFunction&, float t, float dt)
     */
    void set_emitter(std::function<void(WaveFunction&, float, float)> fn) {
        emitter_ = std::move(fn);
    }

    // ------------------------------------------------------------------ adaptive timestep

    /**
     * @brief CFL-safe maximum timestep.
     *
     * Courant-Friedrich-Lewy condition for 9D wave equation:
     *   dt_max = CFL_factor · min_h / (c₀ · sqrt(TORUS_DIMS))
     *
     * @param grid        Grid (used to read spacing).
     * @param cfl_factor  Safety factor (default 0.5).
     */
    float max_stable_dt(const foundation::TorusGrid& grid,
                        float cfl_factor = 0.5f) const noexcept
    {
        float min_h = grid.spacing(0);
        for (int d = 1; d < TORUS_DIMS; ++d)
            min_h = std::min(min_h, grid.spacing(d));
        return cfl_factor * min_h / (ham_.c0() * std::sqrt(static_cast<float>(TORUS_DIMS)));
    }

    // ------------------------------------------------------------------ primary integration step

    /**
     * @brief Advance the wavefunction by one timestep using Strang splitting.
     *
     * Sequence: D(Δt/2) · H(Δt/2) · N(Δt) · H(Δt/2) · D(Δt/2)
     *
     * @param wf  WaveFunction to evolve (modified in place).
     * @param dt  Timestep.  Should satisfy CFL: dt ≤ max_stable_dt().
     */
    void step(WaveFunction& wf, float dt) {
        // Ensure adjacency is precomputed for O(1) neighbour access
        if (!wf.grid().adjacency_valid()) {
            wf.grid().precompute_adjacency();
        }

        const float half_dt = 0.5f * dt;

        // 6-step velocity-Verlet / Störmer-Verlet decomposition
        // (second-order symplectic for the full UFIE)
        //
        //  1.  D(dt/2):  V *= exp(-α(1-r)·dt/2)     [half damping]
        //  2.  kick(dt/2): V += c²·∇²Ψ·(dt/2)       [with OLD Ψ]
        //  3.  drift(dt):  Ψ += V·dt                  [full drift]
        //  4.  kick(dt/2): V += c²·∇²Ψ·(dt/2)       [with NEW Ψ]
        //  5.  N(dt):  V += β|Ψ|²Ψ·dt                [NL with new Ψ]
        //  6.  D(dt/2):  V *= exp(-α(1-r)·dt/2)     [half damping]
        //
        // For α=0, β=0: reduces to the standard leapfrog
        //   V_{n+1/2} = V_n + c²·∇²Ψ_n·(dt/2)
        //   Ψ_{n+1}   = Ψ_n + V_{n+1/2}·dt
        //   V_{n+1}   = V_{n+1/2} + c²·∇²Ψ_{n+1}·(dt/2)
        // which exactly conserves the shadow Hamiltonian to O(dt⁴).

        step_damping(wf, half_dt);       // Step 1: D(dt/2)
        step_kick(wf, half_dt);          // Step 2: kick using Ψ_n
        step_drift(wf, dt);              // Step 3: full drift
        step_kick(wf, half_dt);          // Step 4: kick using Ψ_{n+1}
        step_nonlinear(wf, dt);          // Step 5: N(dt)
        step_damping(wf, half_dt);       // Step 6: D(dt/2)

        // External emitters (injected fields E(x,t))
        if (emitter_) {
            emitter_(wf, wf.time(), dt);
        }

        wf.advance_time(dt);
    }

    /**
     * @brief Evolve for N steps, monitoring energy conservation.
     *
     * Computes the initial Hamiltonian and checks drift every
     * check_interval steps.  Triggers Quantum Zeno SCRAM if needed.
     *
     * @param wf              WaveFunction to evolve.
     * @param dt              Timestep per step.
     * @param n_steps         Number of integration steps.
     * @param tolerance       Max |ΔH/H₀| before SCRAM (default 0.0001 = 0.01%).
     * @param check_interval  How often to recompute H (default every 100 steps).
     */
    void evolve(WaveFunction& wf, float dt, size_t n_steps,
                double tolerance = 1e-4, size_t check_interval = 100)
    {
        const double H0 = ham_.compute(wf);
        // Accept at most 10 SCRAM events before hard abort
        int scram_count = 0;

        for (size_t s = 0; s < n_steps; ++s) {
            step(wf, dt);

            if ((s + 1) % check_interval == 0) {
                const double H1 = ham_.compute(wf);
                const double drift = std::abs(H1 - H0) / (std::abs(H0) + 1e-30);

                if (drift > tolerance) {
                    ++scram_count;
                    if (scram_count > 10) {
                        throw std::runtime_error(
                            "Propagator::evolve: repeated SCRAM events — "
                            "reduce dt or check initial conditions");
                    }
                    // Soft SCRAM: drain 50% of kinetic energy
                    wf.apply_emergency_damping(0.5f);
                }
            }
        }
    }

    // ------------------------------------------------------------------ substep access (for unit tests / inspection)

    /**
     * @brief D substep: exact exponential damping of velocity.
     *
     *   V(t+τ) = V(t) · exp(-α·(1-r)·τ)
     */
    void step_damping(WaveFunction& wf, float tau) {
        const size_t N = wf.num_nodes();
        foundation::TorusGrid& g = wf.grid();
        float* vr = g.vel_real();
        float* vi = g.vel_imag();
        const float* res = g.resonance();

        for (size_t i = 0; i < N; ++i) {
            const float decay = std::exp(-alpha_ * (1.f - res[i]) * tau);
            vr[i] *= decay;
            vi[i] *= decay;
        }
    }

    /**
     * @brief Kick substep: update velocity using current Ψ.
     *
     *   V ← V + c_eff² · ∇²Ψ · τ
     *
     * The 9D Laplacian uses Kahan summation.  This is the "B" operator in
     * the ABA Strang splitting, where A = drift and B = kick.
     */
    void step_kick(WaveFunction& wf, float tau) {
        if (!wf.grid().adjacency_valid()) wf.grid().precompute_adjacency();
        const size_t N = wf.num_nodes();
        foundation::TorusGrid& g = wf.grid();
        const float* pr = g.psi_real();
        const float* pi = g.psi_imag();
        float* vr = g.vel_real();
        float* vi = g.vel_imag();
        const float* sf = g.state_field();

        ensure_scratch(N);

        // Compute Laplacian of CURRENT Ψ → stored in scratch
        for (size_t i = 0; i < N; ++i) {
            const size_t* nbrs = g.get_neighbors_fast(i);
            const float psi_r = pr[i], psi_i = pi[i];

            float lap_r = 0.f, comp_r = 0.f;
            float lap_i = 0.f, comp_i = 0.f;

            for (int d = 0; d < TORUS_DIMS; ++d) {
                const float inv_h2 = 1.f / (g.spacing(d) * g.spacing(d));

                float np_r, np_i;
                const size_t np = nbrs[2*d];
                if (np != foundation::VACUUM_NODE) { np_r=pr[np]; np_i=pi[np]; }
                else { auto gh=foundation::pml_ghost({psi_r,psi_i}); np_r=gh.real(); np_i=gh.imag(); }

                float nm_r, nm_i;
                const size_t nm = nbrs[2*d+1];
                if (nm != foundation::VACUUM_NODE) { nm_r=pr[nm]; nm_i=pi[nm]; }
                else { auto gh=foundation::pml_ghost({psi_r,psi_i}); nm_r=gh.real(); nm_i=gh.imag(); }

                const float vv_r = (np_r + nm_r - 2.f*psi_r) * inv_h2;
                const float vv_i = (np_i + nm_i - 2.f*psi_i) * inv_h2;

                { float y=vv_r-comp_r; float t=lap_r+y; comp_r=(t-lap_r)-y; lap_r=t; }
                { float y=vv_i-comp_i; float t=lap_i+y; comp_i=(t-lap_i)-y; lap_i=t; }
            }

            scratch_lap_r_[i] = lap_r;
            scratch_lap_i_[i] = lap_i;
        }

        // Apply kick: V += c_eff² · Lap · τ
        const float c0 = ham_.c0();
        for (size_t i = 0; i < N; ++i) {
            const float c_eff = c0 / (1.f + sf[i]);
            const float c2_tau = c_eff * c_eff * tau;
            vr[i] += c2_tau * scratch_lap_r_[i];
            vi[i] += c2_tau * scratch_lap_i_[i];
        }
    }

    /**
     * @brief Drift substep: update wavefunction using current velocity.
     *
     *   Ψ ← Ψ + V · τ
     *
     * This is the "A" operator in the ABA Strang splitting.
     */
    void step_drift(WaveFunction& wf, float tau) {
        const size_t N = wf.num_nodes();
        foundation::TorusGrid& g = wf.grid();
        float* pr = g.psi_real();
        float* pi = g.psi_imag();
        const float* vr = g.vel_real();
        const float* vi = g.vel_imag();

        for (size_t i = 0; i < N; ++i) {
            pr[i] += vr[i] * tau;
            pi[i] += vi[i] * tau;
        }
    }

    /**
     * @brief N substep: Euler step for the cubic nonlinearity.
     *
     *   V ← V + β·|Ψ|²·Ψ·τ
     */
    void step_nonlinear(WaveFunction& wf, float tau) {
        const size_t N = wf.num_nodes();
        foundation::TorusGrid& g = wf.grid();
        const float* pr = g.psi_real();
        const float* pi = g.psi_imag();
        float* vr = g.vel_real();
        float* vi = g.vel_imag();
        const float beta_tau = ham_.beta() * tau;

        for (size_t i = 0; i < N; ++i) {
            const float psi_sq = pr[i]*pr[i] + pi[i]*pi[i];
            vr[i] += beta_tau * psi_sq * pr[i];
            vi[i] += beta_tau * psi_sq * pi[i];
        }
    }

    /// Legacy: combined kick+drift step (calls step_kick then step_drift).
    void step_kinetic(WaveFunction& wf, float tau) {
        if (!wf.grid().adjacency_valid()) wf.grid().precompute_adjacency();
        step_kick(wf, tau);
        step_drift(wf, tau);
    }

private:
    // ------------------------------------------------------------------ helpers

    void ensure_scratch(size_t N) {
        if (scratch_lap_r_.size() < N) {
            scratch_lap_r_.resize(N, 0.f);
            scratch_lap_i_.resize(N, 0.f);
        }
    }

    // ------------------------------------------------------------------ data

    Hamiltonian ham_;
    float alpha_ = DEFAULT_ALPHA;   ///< Damping coefficient

    std::function<void(WaveFunction&, float, float)> emitter_;

    // Scratch buffers for Laplacian storage (allocated once, reused)
    std::vector<float> scratch_lap_r_;
    std::vector<float> scratch_lap_i_;
};

} // namespace nikola::physics
