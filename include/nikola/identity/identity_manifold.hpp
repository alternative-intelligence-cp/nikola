/**
 * @file identity/identity_manifold.hpp
 * @brief GAP-M1 (Part 2) — IdentityManifold: standing identity wave on the torus.
 *
 * Bridges the SelfConceptVector to the live physics engine by modulating
 * per-node fields on the TorusGrid:
 *
 *   state_field (s):  s_eff(i) = s_dyn(i) + α · Φ_self(i)
 *                     α ≈ 0.05 — identity refractive bias
 *                     Higher Φ → slower wave speed → resonance trapping
 *
 *   resonance (r):    r_eff(i) = clamp(r_base(i) + β · Φ_self(i), 0, 1)
 *                     β ≈ 0.10 — identity memory protection
 *                     Higher Φ → higher plasticity freeze → memory preservation
 *
 * Φ_self(i) is the standing identity wave at node i, computed by projecting
 * the SCV onto the grid via deterministic node hashing.  The wave has
 * compact support: only SCV_DIM (128) positions carry nonzero amplitude,
 * spread across the grid via modular arithmetic on the node index.
 *
 * Materialization triggers:
 *   - Startup (load identity → materialize)
 *   - Post-NAP consolidation
 *   - After significant preference learning
 *
 * Thread safety: NOT thread-safe.  Must be serialised with physics step().
 *
 * Header-only — no separate .cpp needed.
 *
 * Reference:
 *   Integration Report §21.5 (IdentityManifold)
 *   RELEASE_0.3.x.md GAP-M1
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include <nikola/identity/self_concept_vector.hpp>

namespace nikola::identity {

// ============================================================================
// Constants
// ============================================================================

/// Identity refractive bias coupling strength.
inline constexpr float MANIFOLD_ALPHA = 0.05f;

/// Identity resonance protection coupling strength.
inline constexpr float MANIFOLD_BETA  = 0.10f;

/// Maximum absolute Φ_self(i) value (prevents runaway bias).
inline constexpr float MANIFOLD_PHI_MAX = 1.0f;

// ============================================================================
// IdentityManifold
// ============================================================================

/**
 * @brief Standing identity wave on the toroidal manifold.
 *
 * The manifold holds a pre-computed per-node identity field Φ_self and
 * applies it to the grid's state_field and resonance arrays each tick.
 *
 * Usage:
 * @code
 *   IdentityManifold manifold;
 *   manifold.materialize(scv, grid.num_active_nodes());
 *   // Each physics step (or every N steps):
 *   manifold.apply_bias(grid);
 * @endcode
 */
class IdentityManifold {
public:
    IdentityManifold() = default;

    /**
     * @brief Materialize the standing identity wave from an SCV.
     *
     * Projects the 128-D SCV onto an N-node grid by distributing each SCV
     * component to grid nodes via modular hashing with cosine spread.
     *
     * @param scv         Source self-concept vector.
     * @param num_nodes   Number of active nodes in the target grid.
     * @param alpha       Refractive bias strength (default: MANIFOLD_ALPHA).
     * @param beta        Resonance protection strength (default: MANIFOLD_BETA).
     */
    void materialize(const SelfConceptVector& scv,
                     size_t num_nodes,
                     float alpha = MANIFOLD_ALPHA,
                     float beta  = MANIFOLD_BETA) {
        alpha_ = alpha;
        beta_  = beta;
        num_nodes_ = num_nodes;
        phi_.assign(num_nodes, 0.0f);
        materialized_ = true;

        if (num_nodes == 0) return;

        const auto& v = scv.vec();

        // Project each SCV dimension onto the grid
        for (int d = 0; d < SCV_DIM; ++d) {
            if (std::abs(v[d]) < 1e-12) continue;

            // Deterministic mapping: SCV dim d → grid position
            size_t center = (static_cast<size_t>(d) * 1099511628211ULL) % num_nodes;

            // Spread across nearby nodes (modular, toroidal wrapping)
            int spread = std::max(1, static_cast<int>(num_nodes / (SCV_DIM * 2)));
            for (int k = -spread; k <= spread; ++k) {
                size_t idx = (center + static_cast<size_t>((k % static_cast<int>(num_nodes))
                             + static_cast<int>(num_nodes))) % num_nodes;
                double weight = std::cos(3.14159265358979 * k / (2.0 * spread));
                phi_[idx] += static_cast<float>(v[d] * weight);
            }
        }

        // Clamp to [-PHI_MAX, +PHI_MAX]
        for (float& p : phi_) {
            p = std::clamp(p, -MANIFOLD_PHI_MAX, MANIFOLD_PHI_MAX);
        }

        materialized_ = true;
    }

    /**
     * @brief Apply identity bias to grid fields (hot path).
     *
     * Modulates:
     *   state_field[i] += α · Φ_self[i]     (refractive index bias)
     *   resonance[i]   += β · Φ_self[i]     (memory protection bias)
     *
     * Both fields are clamped to physical bounds after application:
     *   state_field: no hard bounds (physics handles via c_eff = c₀/(1+s))
     *   resonance:   [0, 1]
     *
     * @param state_field  Mutable pointer to grid state_field array.
     * @param resonance    Mutable pointer to grid resonance array.
     * @param num_nodes    Number of active nodes (must match materialization).
     */
    void apply_bias(float* state_field, float* resonance, size_t num_nodes) const noexcept {
        if (!materialized_ || num_nodes != num_nodes_) return;

        for (size_t i = 0; i < num_nodes; ++i) {
            state_field[i] += alpha_ * phi_[i];
            resonance[i]   += beta_  * phi_[i];
            resonance[i]    = std::clamp(resonance[i], 0.0f, 1.0f);
        }
    }

    /**
     * @brief Remove identity bias from grid fields (reverse of apply_bias).
     *
     * Call before re-materializing with an updated SCV to prevent drift.
     */
    void remove_bias(float* state_field, float* resonance, size_t num_nodes) const noexcept {
        if (!materialized_ || num_nodes != num_nodes_) return;

        for (size_t i = 0; i < num_nodes; ++i) {
            state_field[i] -= alpha_ * phi_[i];
            resonance[i]   -= beta_  * phi_[i];
            resonance[i]    = std::clamp(resonance[i], 0.0f, 1.0f);
        }
    }

    // ── Accessors ───────────────────────────────────────────────────────

    /// Whether the manifold has been materialized.
    [[nodiscard]] bool is_materialized() const noexcept { return materialized_; }

    /// Access the raw Φ_self field.
    [[nodiscard]] const std::vector<float>& phi() const noexcept { return phi_; }

    /// Current coupling strengths.
    [[nodiscard]] float alpha() const noexcept { return alpha_; }
    [[nodiscard]] float beta()  const noexcept { return beta_; }

    /// Number of nodes this manifold was materialized for.
    [[nodiscard]] size_t num_nodes() const noexcept { return num_nodes_; }

    /**
     * @brief Compute identity field energy: E_id = Σ Φ²_self(i).
     *
     * A measure of how strongly identity is imprinted on the manifold.
     */
    [[nodiscard]] double identity_energy() const noexcept {
        double sum = 0.0;
        for (float p : phi_) sum += static_cast<double>(p) * p;
        return sum;
    }

    /**
     * @brief Mean absolute identity field.
     *
     * Useful for checking that materialization produced nonzero bias.
     */
    [[nodiscard]] double mean_abs_phi() const noexcept {
        if (phi_.empty()) return 0.0;
        double sum = 0.0;
        for (float p : phi_) sum += std::abs(static_cast<double>(p));
        return sum / static_cast<double>(phi_.size());
    }

private:
    std::vector<float> phi_;        ///< Per-node identity field Φ_self
    float alpha_ = MANIFOLD_ALPHA;  ///< Refractive bias strength
    float beta_  = MANIFOLD_BETA;   ///< Resonance protection strength
    size_t num_nodes_ = 0;          ///< Node count at materialization
    bool materialized_ = false;     ///< Whether materialize() has been called
};

} // namespace nikola::identity
