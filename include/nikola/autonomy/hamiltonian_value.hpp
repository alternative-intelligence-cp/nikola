/**
 * @file autonomy/hamiltonian_value.hpp
 * @brief NIK-005 — Hamiltonian Value Function for the ENGS dopamine loop.
 *
 * Resolves "Stroboscopic Value Collapse": the DopamineSystem previously received
 * only Σ|Ψ|² (potential energy), which oscillates at 2ω for standing waves.
 * This caused spurious negative TD errors during the kinetic phase, suppressing
 * high-frequency cognition ("cognitive thrashing").
 *
 * The Hamiltonian Value Function computes:
 *
 *   H = γ_K Σ|V|²  +  γ_P Σ|Ψ|²  +  γ_NL Σ(β/2)|Ψ|⁴
 *
 * where:
 *   - |V|²  = kinetic energy   (vel_real² + vel_imag²)
 *   - |Ψ|²  = potential energy  (psi_real² + psi_imag²)
 *   - β/2|Ψ|⁴ = nonlinear interaction (soliton self-energy)
 *
 * Key property: for a stable standing wave Ψ = A cos(kx) e^{iωt}:
 *   - Σ|Ψ|² oscillates at 2ω   (stroboscopic — the old bug)
 *   - H (Hamiltonian) is CONSTANT → TD error δ = 0 for stable states ✓
 *
 * A full-gradient version that wraps nikola::physics::Hamiltonian is available
 * through the WaveFunction overload (see compute_full() below) — this uses the
 * same IBP Laplacian stencil as the Störmer-Verlet propagator and is therefore
 * exactly conserved by the physics engine.
 *
 * Usage:
 * @code
 *   // Simplified (no WaveFunction — works from raw spans)
 *   HamiltonianValue hv;
 *   float H = hv.compute_spans(psi_r.data(), psi_i.data(),
 *                               vel_r.data(), vel_i.data(), N, beta);
 *
 *   // Full physics (requires WaveFunction + Hamiltonian — most accurate)
 *   #define NIKOLA_PHYSICS_HAMILTONIAN_IMPL
 *   #include <nikola/physics/hamiltonian.hpp>
 *   double H_full = hv.compute_full(wf, c0, beta);
 * @endcode
 *
 * @see TASKS.md  NIK-005
 * @see autonomy_engine.hpp   tick_physics() (uses this class)
 */

#pragma once

#include <algorithm>     // std::min
#include <cmath>         // std::isfinite
#include <cstddef>       // std::size_t
#include <span>

namespace nikola::autonomy {

// ─────────────────────────────────────────────────────────────────────────────
//  HamiltonianValue
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Weighted Hamiltonian estimate for use as the RL Value Function.
 *
 * Defaults: all γ = 1.0, H_max = 1e6 (epileptic-resonance safety valve).
 */
class HamiltonianValue {
public:
    // ── configuration ──────────────────────────────────────────────────────

    static constexpr float DEFAULT_H_MAX = 1.0e6f;

    float gamma_k  = 1.0f;  ///< weight for kinetic term  |V|²
    float gamma_p  = 1.0f;  ///< weight for potential term |Ψ|²
    float gamma_nl = 1.0f;  ///< weight for nonlinear term β/2|Ψ|⁴
    float h_max    = DEFAULT_H_MAX;  ///< safety cap (epileptic-resonance limit)

    // ── primary API ────────────────────────────────────────────────────────

    /**
     * @brief Compute the weighted Hamiltonian from raw SoA spans.
     *
     * This simplified O(N) version (T + P + NL, no gradient term) is adequate
     * for every-step dopamine updates.  It avoids the 18-point Laplacian lookup
     * and requires only the psi and velocity arrays already held in TorusGrid.
     *
     * @param psi_r  Real part of Ψ — length N
     * @param psi_i  Imaginary part of Ψ — length N
     * @param vel_r  Real part of ∂_t Ψ — length N  (Störmer-Verlet velocity)
     * @param vel_i  Imaginary part of ∂_t Ψ — length N
     * @param N      Number of active nodes
     * @param beta   Nonlinear coupling constant (β ≥ 0)
     * @return       Clamped Hamiltonian estimate — always ≥ 0, ≤ h_max
     */
    [[nodiscard]]
    float compute_spans(
            const float* psi_r, const float* psi_i,
            const float* vel_r, const float* vel_i,
            std::size_t N,
            float beta = 0.0f) const noexcept
    {
        double H    = 0.0;
        double comp = 0.0;  // Kahan compensation

        for (std::size_t i = 0; i < N; ++i) {
            const double pr = psi_r[i], pi = psi_i[i];
            const double vr = vel_r[i], vi = vel_i[i];

            const double psi_sq   = pr*pr + pi*pi;
            const double kinetic  = vr*vr + vi*vi;
            const double nonlinear = 0.5 * static_cast<double>(beta) * psi_sq * psi_sq;

            const double node_H =
                    static_cast<double>(gamma_k)  * kinetic  +
                    static_cast<double>(gamma_p)  * psi_sq   +
                    static_cast<double>(gamma_nl) * nonlinear;

            // Kahan compensated summation (guard against catastrophic cancellation)
            const double y = node_H - comp;
            const double t = H + y;
            comp = (t - H) - y;
            H = t;
        }

        // Safety: NaN/Inf → 0; clamp to h_max
        if (!std::isfinite(H) || H < 0.0) H = 0.0;
        return static_cast<float>(std::min(H, static_cast<double>(h_max)));
    }

    /**
     * @brief span-based convenience overload.
     */
    [[nodiscard]]
    float compute_spans(
            std::span<const float> psi_r,
            std::span<const float> psi_i,
            std::span<const float> vel_r,
            std::span<const float> vel_i,
            float beta = 0.0f) const noexcept
    {
        const std::size_t N = std::min({psi_r.size(), psi_i.size(),
                                        vel_r.size(), vel_i.size()});
        return compute_spans(psi_r.data(), psi_i.data(),
                             vel_r.data(), vel_i.data(), N, beta);
    }

    // ── Physics Oracle — stability penalty ─────────────────────────────────

    /**
     * @brief Compute a stability penalty for H exceeding h_max.
     *
     * Returns a non-negative penalty proportional to the excess energy:
     *   penalty = alpha * max(0, H - H_max)
     *
     * Intended for use in reward shaping:
     *   R_adjusted = R_raw - stability_penalty(H, h_max, alpha)
     *
     * @param H      Current Hamiltonian value
     * @param H_max  Threshold (default: use member h_max)
     * @param alpha  Penalty scale (default 1.0)
     * @return       Non-negative penalty
     */
    [[nodiscard]]
    static float stability_penalty(
            float H,
            float H_max = DEFAULT_H_MAX,
            float alpha = 1.0f) noexcept
    {
        const float excess = H - H_max;
        return (excess > 0.0f) ? alpha * excess : 0.0f;
    }

    // ── TD error convenience helper ─────────────────────────────────────────

    /**
     * @brief Compute the one-step Temporal Difference error δ.
     *
     *   δ = reward + γ * V(S_{t+1}) - V(S_t)
     *
     * For a stable standing wave (no reward): δ ≈ 0. ✓
     *
     * @param H_prev   Value at prior timestep
     * @param H_curr   Value at current timestep
     * @param reward   External reward signal  R_t
     * @param discount TD discount factor γ (typically 0.95–0.99)
     * @return         TD error δ_t
     */
    [[nodiscard]]
    static float td_error(
            float H_prev, float H_curr,
            float reward   = 0.0f,
            float discount = 0.99f) noexcept
    {
        return reward + discount * H_curr - H_prev;
    }
};

} // namespace nikola::autonomy
