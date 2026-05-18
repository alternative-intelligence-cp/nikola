/**
 * @file physics/covariant_transport.hpp
 * @brief GAP-M2 — Covariant State Transport via Cholesky frames.
 *
 * When the metric tensor evolves (Hebbian plasticity), hidden-state vectors
 * embedded in the manifold must be parallel-transported to remain consistent
 * with the new geometry.  Without transport, a state vector that was unit-norm
 * under the old metric may have distorted norm under the new metric — leading
 * to "identity drift" where the agent's hidden states become inconsistent
 * with the geometry they inhabit.
 *
 * **Algorithm (Cholesky Frame Transport):**
 *
 *   Given old metric g and new metric g', with Cholesky factors L, L':
 *     g = L L^T,   g' = L' L'^T
 *
 *   A vector h expressed in the old geometry is transported by:
 *     1. Project to orthonormal frame: ĥ = L^{-1} h        (forward sub)
 *     2. Re-embed in new geometry:     h' = L' ĥ            (matrix multiply)
 *
 *   This preserves the g-norm: h^T g h = ĥ^T ĥ = h'^T g' h'
 *
 * The transport is exact (not approximate) for the Cholesky frame and
 * preserves all inner products under the metric change.
 *
 * **Integration points:**
 *   - HebbianMetric::step() → transport hidden states after metric update
 *   - NAP consolidation → transport accumulated memories across geometry shift
 *   - IdentityManifold re-materialization → transport SCV projections
 *
 * Thread safety: none.  Per-node transport must be serialised with metric updates.
 *
 * Header-only — uses existing MetricTensorCache infrastructure.
 *
 * Reference:
 *   Integration Report §22 (Covariant State Transport)
 *   RELEASE_0.3.x.md GAP-M2
 */
#pragma once

#include <array>
#include <cmath>
#include <stdexcept>

#include <nikola/physics/metric_tensor.hpp>

namespace nikola::physics {

// ============================================================================
// Constants
// ============================================================================

/// Maximum norm ratio change before transport is flagged as suspicious.
inline constexpr double TRANSPORT_NORM_TOLERANCE = 1e-6;

// ============================================================================
// Core Transport Functions
// ============================================================================

/**
 * @brief Multiply upper-triangular L^T by vector: η = L^T h.
 *
 * Projects h into the Euclidean (orthonormal) frame where the metric
 * becomes identity: ||η||² = h^T g h.
 *
 * @param L  Lower Cholesky factor (packed lower-triangle, 45 doubles).
 * @param h  Input vector in curved geometry (9-D).
 * @return   η = L^T h in the orthonormal frame.
 */
[[nodiscard]] inline std::array<double, METRIC_DIM>
project_to_frame(const std::array<double, METRIC_LOWER_SIZE>& L,
                 const std::array<double, METRIC_DIM>& h) noexcept {
    // η_i = Σ_j L^T[i][j] h[j] = Σ_j L[j][i] h[j]  (j >= i for lower L)
    std::array<double, METRIC_DIM> eta{};
    for (int i = 0; i < METRIC_DIM; ++i) {
        double s = 0.0;
        for (int j = i; j < METRIC_DIM; ++j) {
            s += L[metric_lower_idx(j, i)] * h[j];
        }
        eta[i] = s;
    }
    return eta;
}

/**
 * @brief Solve L_new^T h' = η for h' (backward substitution).
 *
 * Re-embeds an orthonormal-frame vector into the new curved geometry
 * so that h'^T g_new h' = ||η||².
 *
 * @param L_new  Cholesky factor of new metric (lower triangle, 45 doubles).
 * @param eta    Vector in orthonormal frame (9-D).
 * @return       h' = L_new^{-T} η in the new geometry.
 */
[[nodiscard]] inline std::array<double, METRIC_DIM>
embed_from_frame(const std::array<double, METRIC_LOWER_SIZE>& L_new,
                 const std::array<double, METRIC_DIM>& eta) noexcept {
    return backward_sub_9(L_new, eta);
}

/**
 * @brief Transport a single 9-D hidden-state vector across a metric change.
 *
 * Implements the full Cholesky frame transport:
 *   h' = L_new^{-T} · L_old^T · h
 *
 * Preserves the g-norm: h^T g_old h = h'^T g_new h'
 *
 * @param L_old  Cholesky factor of old metric.
 * @param L_new  Cholesky factor of new metric.
 * @param h      Input hidden-state vector in old geometry.
 * @return       Transported vector in new geometry.
 */
[[nodiscard]] inline std::array<double, METRIC_DIM>
transport_vector(const std::array<double, METRIC_LOWER_SIZE>& L_old,
                 const std::array<double, METRIC_LOWER_SIZE>& L_new,
                 const std::array<double, METRIC_DIM>& h) noexcept {
    auto h_hat = project_to_frame(L_old, h);
    return embed_from_frame(L_new, h_hat);
}

// ============================================================================
// Metric Norm Computation
// ============================================================================

/**
 * @brief Compute the g-norm squared of a vector: ||h||²_g = h^T g h.
 *
 * Uses the packed lower-triangle metric representation.
 *
 * @param g  Metric tensor (lower triangle, 45 doubles).
 * @param h  Vector (9-D).
 * @return   h^T g h (non-negative for SPD g).
 */
[[nodiscard]] inline double
metric_norm_sq(const std::array<double, METRIC_LOWER_SIZE>& g,
               const std::array<double, METRIC_DIM>& h) noexcept {
    double result = 0.0;
    for (int i = 0; i < METRIC_DIM; ++i) {
        // Diagonal term
        result += g[metric_lower_idx(i, i)] * h[i] * h[i];
        // Off-diagonal terms (counted twice by symmetry)
        for (int j = 0; j < i; ++j) {
            result += 2.0 * g[metric_lower_idx(i, j)] * h[i] * h[j];
        }
    }
    return result;
}

// ============================================================================
// CovariantTransporter  —  stateful transport across sequential metric updates
// ============================================================================

/**
 * @brief Manages covariant transport of hidden-state vectors across metric evolution.
 *
 * Caches the old and new Cholesky factors to enable batch transport of
 * multiple vectors during a single metric update step.
 *
 * Usage:
 * @code
 *   CovariantTransporter transporter;
 *   // On metric update:
 *   transporter.begin_transport(old_cache, new_cache);
 *   auto h_new = transporter.transport(h_old);
 *   // Verify:
 *   assert(transporter.verify_norm(g_old, g_new, h_old, h_new));
 * @endcode
 */
class CovariantTransporter {
public:
    CovariantTransporter() = default;

    /**
     * @brief Prepare for batch transport between two metric states.
     *
     * Caches the Cholesky factors from both metric caches.  Both caches
     * must be valid (have computed Cholesky factors).
     *
     * @param old_cache  MetricTensorCache from before the metric update.
     * @param new_cache  MetricTensorCache from after the metric update.
     * @throws std::logic_error if either cache is invalid.
     */
    void begin_transport(const MetricTensorCache& old_cache,
                         const MetricTensorCache& new_cache) {
        if (!old_cache.is_valid()) {
            throw std::logic_error(
                "CovariantTransporter: old metric cache is not valid");
        }
        if (!new_cache.is_valid()) {
            throw std::logic_error(
                "CovariantTransporter: new metric cache is not valid");
        }
        L_old_ = old_cache.cholesky();
        L_new_ = new_cache.cholesky();
        ready_ = true;
    }

    /**
     * @brief Transport a single hidden-state vector.
     *
     * Must be called after begin_transport().
     *
     * @param h  Input vector in old geometry.
     * @return   Transported vector in new geometry.
     * @throws std::logic_error if begin_transport() hasn't been called.
     */
    [[nodiscard]] std::array<double, METRIC_DIM>
    transport(const std::array<double, METRIC_DIM>& h) const {
        if (!ready_) {
            throw std::logic_error(
                "CovariantTransporter: begin_transport() not called");
        }
        return transport_vector(L_old_, L_new_, h);
    }

    /**
     * @brief Verify that transport preserved the g-norm.
     *
     * Checks |h^T g_old h - h'^T g_new h'| < tolerance.
     *
     * @param g_old   Old metric (lower triangle).
     * @param g_new   New metric (lower triangle).
     * @param h_old   Original vector.
     * @param h_new   Transported vector.
     * @param tol     Tolerance (default: TRANSPORT_NORM_TOLERANCE).
     * @return        true if norm is preserved within tolerance.
     */
    [[nodiscard]] static bool verify_norm(
        const std::array<double, METRIC_LOWER_SIZE>& g_old,
        const std::array<double, METRIC_LOWER_SIZE>& g_new,
        const std::array<double, METRIC_DIM>& h_old,
        const std::array<double, METRIC_DIM>& h_new,
        double tol = TRANSPORT_NORM_TOLERANCE) noexcept
    {
        double norm_old = metric_norm_sq(g_old, h_old);
        double norm_new = metric_norm_sq(g_new, h_new);
        return std::abs(norm_old - norm_new) < tol;
    }

    /// Whether the transporter has been prepared via begin_transport().
    [[nodiscard]] bool is_ready() const noexcept { return ready_; }

    /// Reset state (must call begin_transport() again before transport()).
    void reset() noexcept { ready_ = false; }

private:
    std::array<double, METRIC_LOWER_SIZE> L_old_{};
    std::array<double, METRIC_LOWER_SIZE> L_new_{};
    bool ready_ = false;
};

} // namespace nikola::physics
