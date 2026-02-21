/**
 * @file autonomy/dream_weave.hpp
 * @brief Gap 5.5 — Dream-Weave counterfactual consolidation.
 *
 * Runs memory consolidation during NAP cycles via iterated physics steps.
 * Termination criterion: Frobenius norm of wavefunction change < 10^-4.
 *
 *   ||ΔΨ||_F = √( Σ_i [(ΔΨ_r_i)² + (ΔΨ_i_i)²] ) < DREAM_CONVERGENCE_THRESHOLD
 *
 * Biological analogy: REM sleep replays recent experiences with variations,
 * adjusting the metric tensor toward stable low-energy configurations, and
 * pruning weak connections (low-amplitude nodes).
 *
 * Designed to be called from NapController::on_nap_tick.
 * Owns no physics resources; accepts a stepper callback so it adapts
 * to both the real Propagator and lightweight test doubles.
 */

#pragma once

#include <cmath>       // std::sqrt
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace nikola::autonomy {

// ── Gap 5.5 constants ─────────────────────────────────────────────────────────

/// Frobenius norm convergence threshold.  Spec: 10^-4.
inline constexpr float DREAM_CONVERGENCE_THRESHOLD = 1e-4f;

/// Maximum dream iterations before forced exit.  Spec: 1000.
inline constexpr int   DREAM_MAX_ITERATIONS        = 1000;

// ── DreamResult ────────────────────────────────────────────────────────────────

struct DreamResult {
    bool  converged   = false;
    int   iterations  = 0;
    float final_delta = 0.0f;   ///< ||ΔΨ||_F at termination
};

// ── DreamWeaveEngine ──────────────────────────────────────────────────────────

/**
 * @class DreamWeaveEngine
 * @brief Runs counterfactual consolidation loops until wavefunction settles.
 *
 * The engine is decoupled from the physics tier via a stepper callback:
 *   using Stepper = std::function<void(std::span<float>, std::span<float>)>
 *
 * The stepper receives writable spans of psi_real / psi_imag and evolves
 * them by one dream tick (e.g., call Propagator::step() internally).
 *
 * After each step the engine computes the Frobenius norm ||ΔΨ||_F between
 * the current and previous wavefunction snapshot.  Converged when < 10^-4.
 *
 * Usage:
 * @code
 *   // Wire into NapController
 *   nap.on_nap_tick = [&](float elapsed) {
 *       auto result = dream.run(psi_real_buf, psi_imag_buf, stepper);
 *       if (result.converged)
 *           memory.consolidate(psi_real_buf, psi_imag_buf);
 *   };
 * @endcode
 */
class DreamWeaveEngine {
public:
    /// Stepper callback signature: evolves psi by one dream step in place.
    using Stepper = std::function<void(std::span<float>, std::span<float>)>;

    DreamWeaveEngine() = default;

    // ── primary interface ─────────────────────────────────────────────────

    /**
     * @brief Run consolidation until convergence or max iterations.
     *
     * @param psi_real  In/out wavefunction real part (modified in place).
     * @param psi_imag  In/out wavefunction imaginary part (modified in place).
     * @param stepper   Physics step callback — evolves psi by one tick.
     * @param threshold Frobenius convergence threshold (default DREAM_CONVERGENCE_THRESHOLD).
     * @param max_iter  Maximum iterations (default DREAM_MAX_ITERATIONS).
     * @return DreamResult — converged flag, iteration count, final ‖ΔΨ‖_F.
     */
    [[nodiscard]]
    DreamResult run(std::span<float>        psi_real,
                    std::span<float>        psi_imag,
                    Stepper                 stepper,
                    float threshold = DREAM_CONVERGENCE_THRESHOLD,
                    int   max_iter  = DREAM_MAX_ITERATIONS) {
        const std::size_t N = std::min(psi_real.size(), psi_imag.size());

        // Snapshot before first step
        prev_r_.assign(psi_real.begin(), psi_real.begin() + N);
        prev_i_.assign(psi_imag.begin(), psi_imag.begin() + N);

        DreamResult result;

        for (int iter = 0; iter < max_iter; ++iter) {
            // Advance one dream step
            stepper(psi_real, psi_imag);

            // ‖ΔΨ‖_F = √( Σ [(Ψ_r - prev_r)² + (Ψ_i - prev_i)²] )
            float sum = 0.0f;
            for (std::size_t i = 0; i < N; ++i) {
                float dr = psi_real[i] - prev_r_[i];
                float di = psi_imag[i] - prev_i_[i];
                sum += dr*dr + di*di;
            }
            float delta = std::sqrt(sum);

            result.iterations  = iter + 1;
            result.final_delta = delta;

            if (delta < threshold) {
                result.converged = true;
                ++convergence_count_;
                return result;
            }

            // Update snapshot
            std::copy(psi_real.begin(), psi_real.begin() + N, prev_r_.begin());
            std::copy(psi_imag.begin(), psi_imag.begin() + N, prev_i_.begin());
        }

        ++no_convergence_count_;
        return result;   // converged = false
    }

    /**
     * @brief Compute Frobenius norm between two wavefunction snapshots.
     *
     * Utility function — also used by tests to verify the metric directly.
     */
    [[nodiscard]] static
    float frobenius_norm(std::span<const float> a_real, std::span<const float> a_imag,
                         std::span<const float> b_real, std::span<const float> b_imag) noexcept {
        const std::size_t N = std::min({a_real.size(), a_imag.size(),
                                        b_real.size(), b_imag.size()});
        float sum = 0.0f;
        for (std::size_t i = 0; i < N; ++i) {
            float dr = a_real[i] - b_real[i];
            float di = a_imag[i] - b_imag[i];
            sum += dr*dr + di*di;
        }
        return std::sqrt(sum);
    }

    // ── observers ────────────────────────────────────────────────────────────

    [[nodiscard]] uint32_t convergence_count()    const noexcept { return convergence_count_; }
    [[nodiscard]] uint32_t no_convergence_count() const noexcept { return no_convergence_count_; }

    void reset_counters() noexcept { convergence_count_ = 0; no_convergence_count_ = 0; }

private:
    std::vector<float> prev_r_, prev_i_;   // snapshot buffers (reused across calls)
    uint32_t convergence_count_    = 0u;
    uint32_t no_convergence_count_ = 0u;
};

} // namespace nikola::autonomy
