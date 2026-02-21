/**
 * @file metric_tensor.hpp
 * @brief Lazy Cholesky decomposition cache for the 9D metric tensor.
 *
 * Each node in the Nikola manifold carries a 9×9 symmetric positive-definite
 * metric tensor \f$g_{ij}\f$ that encodes the local curvature of toroidal
 * space.  The curved-space Laplacian requires computing \f$g^{ij}\f$ (the
 * inverse metric) at every node on every timestep — an O(n³) = O(729)
 * decomposition per node if performed naively.
 *
 * **Key observation:** The metric tensor evolves on the *plasticity* timescale
 * (~1000 timesteps), not the *physics* timescale.  We can therefore cache the
 * Cholesky factor \f$L\f$ (where \f$g = L L^T\f$) and reuse it, recomputing
 * only when the tensor has changed by more than a configurable threshold.
 *
 * This header provides:
 *
 *   - `MetricTensorCache`  — per-node Cholesky cache with dirty tracking
 *   - `metric_lower_idx()` — row-major lower-triangle index helper
 *   - `cholesky_9x9()`     — standalone Cholesky decomposition for a 9×9
 *                            symmetric positive-definite matrix
 *   - `forward_sub_9()`    — forward triangular substitution  ( L  y = v )
 *   - `backward_sub_9()`   — backward triangular substitution ( LT x = y )
 *
 * Usage
 * -----
 * @code
 *   MetricTensorCache cache;
 *   // On plasticity update:
 *   cache.update_if_changed(new_g_lower_triangle);
 *   // On physics step:
 *   auto grad_inv = cache.apply_inverse(gradient_vector);
 * @endcode
 *
 * Thread safety: none.  Each node owns its own cache; updates must be
 * serialised per-node (or the caller manages locking).
 *
 * Reference: nikola engineering guide §5 (Lazy Cholesky Decomposition),
 *            implementation checklist item 0.5.
 */
#pragma once

#include <array>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <cassert>

namespace nikola::physics {

// ============================================================================
// Dimension constant
// ============================================================================

inline constexpr int METRIC_DIM = 9;   ///< Spatial dimensions of the manifold

/// Number of independent components in a symmetric 9×9 matrix.
inline constexpr int METRIC_LOWER_SIZE = METRIC_DIM * (METRIC_DIM + 1) / 2;  // 45

// ============================================================================
// Index helpers for packed lower-triangle storage
// ============================================================================

/**
 * @brief Row-major lower-triangle index for element (i, j) where i >= j.
 *
 * Elements are stored as:
 *   [g00, g10, g11, g20, g21, g22, g30, … , g88]
 *
 * @param i  Row index    [0, METRIC_DIM)
 * @param j  Column index [0, i]
 * @return   Flat index into the 45-element lower-triangle array.
 */
[[nodiscard]] inline constexpr int metric_lower_idx(int i, int j) noexcept {
    assert(i >= j && i < METRIC_DIM && j >= 0);
    return i * (i + 1) / 2 + j;
}

/**
 * @brief Retrieve element (i, j) from a packed symmetric matrix.
 *
 * Handles both lower (i >= j) and upper (i < j) triangles by symmetry.
 */
[[nodiscard]] inline double metric_get(
    const std::array<double, METRIC_LOWER_SIZE>& g, int i, int j) noexcept
{
    return (i >= j) ? g[metric_lower_idx(i, j)]
                    : g[metric_lower_idx(j, i)];
}

// ============================================================================
// Standalone Cholesky decomposition  (9×9)
// ============================================================================

/**
 * @brief Compute the lower Cholesky factor of a 9×9 symmetric positive-definite
 *        matrix stored in packed lower-triangle format.
 *
 * On success, `L_out` contains the lower factor \f$L\f$ such that
 * \f$g = L L^T\f$ (also in packed lower-triangle format).
 *
 * @param g      Input symmetric positive-definite matrix (lower triangle).
 * @param L_out  Output lower Cholesky factor (same layout as g).
 * @return `true` on success, `false` if the matrix is not positive-definite
 *         (diagonal element would be negative under the square root).
 */
[[nodiscard]] inline bool cholesky_9x9(
    const std::array<double, METRIC_LOWER_SIZE>& g,
    std::array<double, METRIC_LOWER_SIZE>&       L_out) noexcept
{
    L_out.fill(0.0);

    for (int j = 0; j < METRIC_DIM; ++j) {
        // Compute L[j][j] (diagonal element)
        double diag = metric_get(g, j, j);
        for (int k = 0; k < j; ++k) {
            const double lkj = L_out[metric_lower_idx(j, k)];
            diag -= lkj * lkj;
        }
        if (diag <= 0.0) return false;   // not positive-definite
        L_out[metric_lower_idx(j, j)] = std::sqrt(diag);

        // Compute L[i][j] for i > j (sub-diagonal column entries)
        for (int i = j + 1; i < METRIC_DIM; ++i) {
            double s = metric_get(g, i, j);
            for (int k = 0; k < j; ++k)
                s -= L_out[metric_lower_idx(i, k)] * L_out[metric_lower_idx(j, k)];
            L_out[metric_lower_idx(i, j)] = s / L_out[metric_lower_idx(j, j)];
        }
    }
    return true;
}

// ============================================================================
// Forward / backward substitution  (9-D)
// ============================================================================

/**
 * @brief Forward substitution: solve  L y = v  for y.
 *
 * @param L  Lower Cholesky factor (packed lower-triangle, from cholesky_9x9).
 * @param v  Right-hand side vector [METRIC_DIM].
 * @return   Solution vector y.
 */
[[nodiscard]] inline std::array<double, METRIC_DIM> forward_sub_9(
    const std::array<double, METRIC_LOWER_SIZE>& L,
    const std::array<double, METRIC_DIM>&        v) noexcept
{
    std::array<double, METRIC_DIM> y{};
    for (int i = 0; i < METRIC_DIM; ++i) {
        double s = v[i];
        for (int j = 0; j < i; ++j)
            s -= L[metric_lower_idx(i, j)] * y[j];
        y[i] = s / L[metric_lower_idx(i, i)];
    }
    return y;
}

/**
 * @brief Backward substitution: solve  L^T x = y  for x.
 *
 * @param L  Lower Cholesky factor (packed lower-triangle).
 * @param y  Right-hand side vector [METRIC_DIM].
 * @return   Solution vector x.
 */
[[nodiscard]] inline std::array<double, METRIC_DIM> backward_sub_9(
    const std::array<double, METRIC_LOWER_SIZE>& L,
    const std::array<double, METRIC_DIM>&        y) noexcept
{
    std::array<double, METRIC_DIM> x{};
    for (int i = METRIC_DIM - 1; i >= 0; --i) {
        double s = y[i];
        for (int j = i + 1; j < METRIC_DIM; ++j)
            s -= L[metric_lower_idx(j, i)] * x[j];   // L^T[i][j] = L[j][i]
        x[i] = s / L[metric_lower_idx(i, i)];
    }
    return x;
}

// ============================================================================
// MetricTensorCache  —  lazy cached Cholesky per manifold node
// ============================================================================

/**
 * @brief Lazy Cholesky decomposition cache for a single 9D metric tensor.
 *
 * The cache stores:
 *   - The last committed metric \f$g\f$ (lower triangle, 45 doubles).
 *   - The corresponding Cholesky factor \f$L\f$ (lower triangle, 45 doubles).
 *   - A validity flag (false until first update).
 *
 * The cache is *invalid* (must recompute) when:
 *   1. It has never been initialised; or
 *   2. The new metric differs from the cached one by more than
 *      `change_threshold` (max-norm on the lower triangle).
 *
 * **Typical usage:**
 *   - Call `update_if_changed()` once per plasticity step (~1000 physics ticks)
 *     for nodes whose metrics have been modified.
 *   - Call `apply_inverse()` once per physics step (fast O(n²) substitution).
 */
class MetricTensorCache {
public:
    // ------------------------------------------------------------------ config

    /// Max-norm threshold above which the Cholesky is recomputed.
    double change_threshold = 1e-6;

    // ------------------------------------------------------------------ ctors

    MetricTensorCache() noexcept { g_.fill(0.0); L_.fill(0.0); }

    /**
     * @brief Initialise with a specific metric (forces immediate decomposition).
     *
     * @throws std::invalid_argument if the matrix is not positive-definite.
     */
    explicit MetricTensorCache(const std::array<double, METRIC_LOWER_SIZE>& g) {
        force_update(g);
    }

    /// Build a cache for the flat (Euclidean) 9D metric (g = I, diagonal = 1).
    [[nodiscard]] static MetricTensorCache flat() {
        std::array<double, METRIC_LOWER_SIZE> g{};
        for (int i = 0; i < METRIC_DIM; ++i)
            g[metric_lower_idx(i, i)] = 1.0;
        return MetricTensorCache{g};
    }

    // ------------------------------------------------------------------ state

    /// True once the cache has a valid Cholesky factor.
    [[nodiscard]] bool is_valid() const noexcept { return valid_; }

    /// Access the stored metric (lower triangle).
    [[nodiscard]] const std::array<double, METRIC_LOWER_SIZE>& metric() const noexcept {
        return g_;
    }

    /// Access the cached Cholesky factor.
    [[nodiscard]] const std::array<double, METRIC_LOWER_SIZE>& cholesky() const noexcept {
        return L_;
    }

    // ------------------------------------------------------------------ update

    /**
     * @brief Conditionally update the Cholesky factor.
     *
     * The decomposition is recomputed **only** if the new metric differs from
     * the cached one by more than `change_threshold`.
     *
     * @param new_g  New metric tensor (lower triangle, 45 doubles).
     * @return `true` if the Cholesky was recomputed; `false` if cache was reused.
     * @throws std::invalid_argument if the new metric is not positive-definite.
     */
    bool update_if_changed(const std::array<double, METRIC_LOWER_SIZE>& new_g) {
        if (valid_ && !needs_update(new_g))
            return false;
        force_update(new_g);
        return true;
    }

    /**
     * @brief Unconditionally recompute the Cholesky factor from `new_g`.
     *
     * @throws std::invalid_argument if the matrix is not positive-definite.
     */
    void force_update(const std::array<double, METRIC_LOWER_SIZE>& new_g) {
        if (!cholesky_9x9(new_g, L_)) {
            valid_ = false;
            throw std::invalid_argument(
                "MetricTensorCache: metric is not positive-definite");
        }
        g_     = new_g;
        valid_ = true;
    }

    /**
     * @brief Invalidate the cache (forces recomputation on next call to
     *        update_if_changed).
     */
    void invalidate() noexcept { valid_ = false; }

    // ------------------------------------------------------------------ apply

    /**
     * @brief Apply the inverse metric \f$g^{-1}\f$ to a vector \f$v\f$.
     *
     * Solves \f$g\,x = v\f$ in two O(n²) triangular substitution passes
     * using the cached Cholesky factor.  Must be called after a successful
     * `update_if_changed()`.
     *
     * @param v  Input covariant vector [METRIC_DIM].
     * @return   Contravariant vector x = g⁻¹ v.
     * @throws std::logic_error if the cache is not valid.
     */
    [[nodiscard]] std::array<double, METRIC_DIM>
    apply_inverse(const std::array<double, METRIC_DIM>& v) const {
        if (!valid_) {
            throw std::logic_error(
                "MetricTensorCache::apply_inverse called before update");
        }
        // g x = v → L L^T x = v → L y = v;  L^T x = y
        const auto y = forward_sub_9(L_, v);
        return backward_sub_9(L_, y);
    }

    /**
     * @brief Compute the matrix-vector product \f$g\,v\f$ (forward metric).
     *
     * @param v  Input vector [METRIC_DIM].
     * @return   g × v.
     */
    [[nodiscard]] std::array<double, METRIC_DIM>
    apply(const std::array<double, METRIC_DIM>& v) const noexcept {
        std::array<double, METRIC_DIM> out{};
        for (int i = 0; i < METRIC_DIM; ++i)
            for (int j = 0; j < METRIC_DIM; ++j)
                out[i] += metric_get(g_, i, j) * v[j];
        return out;
    }

    /**
     * @brief Return log(det(g)) = 2 × Σ log(L[k][k]).
     *
     * Useful for normalisation in probability density computations.
     * @throws std::logic_error if cache is not valid.
     */
    [[nodiscard]] double log_det() const {
        if (!valid_)
            throw std::logic_error("MetricTensorCache::log_det: cache not valid");
        double s = 0.0;
        for (int k = 0; k < METRIC_DIM; ++k)
            s += std::log(L_[metric_lower_idx(k, k)]);
        return 2.0 * s;   // det(g) = det(L)^2; log det(L) = Σ log L[k][k]
    }

private:
    // ------------------------------------------------------------------ state

    [[nodiscard]] bool needs_update(
        const std::array<double, METRIC_LOWER_SIZE>& new_g) const noexcept
    {
        double max_diff = 0.0;
        for (int i = 0; i < METRIC_LOWER_SIZE; ++i) {
            const double d = std::abs(new_g[i] - g_[i]);
            if (d > max_diff) max_diff = d;
        }
        return max_diff > change_threshold;
    }

    std::array<double, METRIC_LOWER_SIZE> g_{};    ///< Last committed metric
    std::array<double, METRIC_LOWER_SIZE> L_{};    ///< Cholesky factor
    bool valid_{false};
};

} // namespace nikola::physics
