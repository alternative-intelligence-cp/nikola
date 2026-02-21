/**
 * @file autonomy/metabolic_simulator.hpp
 * @brief Gap 5.3 — Hamiltonian-based ATP metabolic cost model.
 *
 * Metabolic Cost = α · Σ_{active nodes} |∇Ψ|² · Δt
 *
 * High-frequency waves (large Laplacian magnitude) burn more ATP than
 * standing waves, penalizing "thrashing" and noisy cognitive states.
 *
 * Normalized ATP ∈ [0, 1]:
 *   - 1.0 = fully charged
 *   - 0.15 = exhaustion threshold → enter nap
 *   - Regen rate: 0.05 / second during nap
 *
 * Energy budget at 2 kHz physics loop:
 *   Idle           ~0.001 ATP/s
 *   Active reason  ~0.05  ATP/s  (sustainable)
 *   Intense comput ~0.20  ATP/s  (burst, ~5 s capacity)
 */

#pragma once

#include <algorithm>   // std::max, std::min
#include <cmath>       // std::norm (for std::complex)
#include <span>

namespace nikola::autonomy {

// ── Gap 5.3 constants ────────────────────────────────────────────────────────

/// Cost coefficient α — scales Laplacian magnitude to ATP units.
inline constexpr float META_ALPHA             = 0.001f;

/// Passive regeneration rate (ATP/s) while in nap state.
inline constexpr float META_REGEN_RATE        = 0.05f;

/// Exhaustion threshold: below this ATP level the system must nap.
inline constexpr float META_EXHAUSTION_THRESH = 0.15f;

/// Minimum intensity threshold to count a node as "active" for costing.
inline constexpr float META_ACTIVE_THRESH     = 1e-6f;

// ── MetabolicSimulator ───────────────────────────────────────────────────────

/**
 * @class MetabolicSimulator
 * @brief Tracks normalized ATP ∈ [0, 1] using Hamiltonian kinetic cost.
 *
 * Designed to integrate with the physics tier's SoA layout.
 * All methods are noexcept; no heap allocations.
 *
 * Usage:
 * @code
 *   MetabolicSimulator meta;
 *
 *   // Per physics step (Δt ≈ 0.5 ms at 2 kHz):
 *   meta.consume_from_laplacian(psi_real, psi_imag, lap_real, lap_imag, dt);
 *
 *   // During nap:
 *   meta.recharge(dt);
 *
 *   if (meta.is_exhausted()) { nap_controller.enter(); }
 * @endcode
 */
class MetabolicSimulator {
public:
    explicit MetabolicSimulator(float initial_atp = 1.0f) noexcept
        : atp_(std::clamp(initial_atp, 0.0f, 1.0f)) {}

    // ── primary consumption methods ───────────────────────────────────────

    /**
     * @brief Consume ATP proportional to Laplacian kinetic energy.
     *
     * cost = α · Σ_{active} |∇²Ψ_i|² · dt
     *
     * @param psi_real   Re(Ψ) per node (TorusGrid SoA).
     * @param psi_imag   Im(Ψ) per node.
     * @param lap_real   Re(∇²Ψ) per node (output of Hamiltonian).
     * @param lap_imag   Im(∇²Ψ) per node.
     * @param dt         Timestep in seconds.
     */
    void consume_from_laplacian(std::span<const float> psi_real,
                                std::span<const float> psi_imag,
                                std::span<const float> lap_real,
                                std::span<const float> lap_imag,
                                float dt) noexcept {
        const std::size_t N = std::min({psi_real.size(), psi_imag.size(),
                                        lap_real.size(), lap_imag.size()});
        float cost = 0.0f;
        for (std::size_t i = 0; i < N; ++i) {
            float pr = psi_real[i], pi = psi_imag[i];
            if (pr*pr + pi*pi > META_ACTIVE_THRESH) {
                float lr = lap_real[i], li = lap_imag[i];
                cost += lr*lr + li*li;   // |∇²Ψ|²
            }
        }
        atp_ = std::max(0.0f, atp_ - META_ALPHA * cost * dt);
    }

    /**
     * @brief Consume ATP by direct energy rate (simplified interface).
     *
     * Used when you already have an energy rate (e.g., from PhysicsOracle).
     *
     * @param energy_rate  Kinetic energy per second.
     * @param dt           Timestep.
     */
    void consume_by_rate(float energy_rate, float dt) noexcept {
        float depletion = META_ALPHA * energy_rate * dt;
        atp_ = std::max(0.0f, atp_ - depletion);
    }

    // ── regeneration ──────────────────────────────────────────────────────

    /**
     * @brief Passive ATP regeneration (call during nap cycles).
     * @param dt  Elapsed seconds.
     */
    void recharge(float dt) noexcept {
        atp_ = std::min(1.0f, atp_ + META_REGEN_RATE * dt);
    }

    /**
     * @brief Instant recharge to a specific level (testing / reset).
     */
    void set_atp(float level) noexcept {
        atp_ = std::clamp(level, 0.0f, 1.0f);
    }

    // ── observers ────────────────────────────────────────────────────────────

    [[nodiscard]] float atp()          const noexcept { return atp_; }
    [[nodiscard]] bool  is_exhausted() const noexcept { return atp_ < META_EXHAUSTION_THRESH; }
    [[nodiscard]] bool  is_full()      const noexcept { return atp_ >= 1.0f; }

private:
    float atp_ = 1.0f;
};

} // namespace nikola::autonomy
