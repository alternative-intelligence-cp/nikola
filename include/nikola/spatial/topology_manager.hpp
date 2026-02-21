/**
 * @file spatial/topology_manager.hpp
 * @brief 9D toroidal manifold geometry: metric validation, coordinate conversion,
 *        and dopamine-modulated Hebbian metric learning.
 *
 * Implements Phase 2 Manifold Geometry gaps from the engineering report:
 *
 *   Gap 2.1 — Metric tensor positive-definiteness (Gerschgorin + Tikhonov)
 *   Gap 2.2 — Hilbert curve Gray-code rotation (deferred to hilbert_scanner)
 *   Gap 2.3 — Anisotropic resolution: x,y,z=64; t=128; r,s=16; u,v,w=32
 *   Gap 2.4 — Dual integer/float coordinate system + quadratic peak interpolation
 *   Gap 2.5 — Dopamine-modulated Hebbian metric learning rate schedule
 *
 * Reference: docs/info/integration/sections/06_implementation_specifications/
 *            02_geometry_spatial_implementation.md
 */
#pragma once

#include <nikola/foundation/toroidal_grid.hpp>

#include <array>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <stdexcept>

namespace nikola::spatial {

using foundation::GridConfig;
using foundation::TORUS_DIMS;

// ============================================================================
// Gap 2.3 — Anisotropic resolution allocation
// ============================================================================

/**
 * @brief Standard anisotropic resolutions from the engineering report.
 *
 * Dimension layout (matches GridConfig::anisotropic_default()):
 *   0-2  (x,y,z)     : N = 64   — high spatial resolution for visual/audio
 *   3    (t)          : N = 128  — time window ~64 ms @ 2 kHz
 *   4-5  (r,s)        : N = 16   — coarse neurochemical modulation
 *   6-8  (u,v,w)      : N = 32   — quantum superposition bins
 */
inline constexpr std::array<int, TORUS_DIMS> ANISOTROPIC_RESOLUTION = {
    64, 64, 64, 128, 16, 16, 32, 32, 32
};

// ============================================================================
// Gap 2.4 — Dual integer / float coordinate system
// ============================================================================

/// Integer grid coordinates (0 … N_d − 1 per dimension).
struct Coord9DInt {
    std::array<uint16_t, TORUS_DIMS> c{};
};

/// Physical (normalised) coordinates in [0, 1) per dimension.
struct Coord9DFloat {
    std::array<float, TORUS_DIMS> c{};
};

/**
 * @brief Convert integer grid coordinates → physical coordinates.
 *
 * Formula:  x_float = x_int / N_d   (torus length L_d = 1 throughout)
 */
[[nodiscard]]
inline Coord9DFloat to_physical(const Coord9DInt& ic, const GridConfig& cfg) noexcept {
    Coord9DFloat out{};
    for (int d = 0; d < TORUS_DIMS; ++d) {
        out.c[d] = static_cast<float>(ic.c[d]) /
                   static_cast<float>(cfg.resolution[d]);
    }
    return out;
}

/**
 * @brief Convert physical coordinates → integer grid coordinates.
 *
 * Clamps to [0, N_d − 1] and rounds down.
 */
[[nodiscard]]
inline Coord9DInt to_integer(const Coord9DFloat& fc, const GridConfig& cfg) noexcept {
    Coord9DInt out{};
    for (int d = 0; d < TORUS_DIMS; ++d) {
        const float mapped = fc.c[d] * static_cast<float>(cfg.resolution[d]);
        const int   iq     = static_cast<int>(mapped);
        const int   Nd     = cfg.resolution[d];
        out.c[d] = static_cast<uint16_t>(std::clamp(iq, 0, Nd - 1));
    }
    return out;
}

/**
 * @brief Quadratic interpolation for sub-grid peak location (Gap 2.4).
 *
 * Fits a parabola through three amplitude samples (left, centre, right) and
 * returns the fractional offset of the vertex from centre in [−0.5, +0.5].
 *
 * Usage: true_coord = centre_index + interpolate_peak(a[i-1], a[i], a[i+1])
 *
 * @param v_left    Amplitude at (centre − 1)
 * @param v_centre  Amplitude at centre
 * @param v_right   Amplitude at (centre + 1)
 * @return          Fractional offset in [−0.5, +0.5]
 */
[[nodiscard]]
inline float interpolate_peak(float v_left, float v_centre, float v_right) noexcept {
    const float denom = 2.f * (v_left - 2.f * v_centre + v_right);
    if (std::abs(denom) < 1e-6f) return 0.f;  // flat neighbourhood → peak at centre
    return std::clamp((v_left - v_right) / denom, -0.5f, 0.5f);
}

// ============================================================================
// Gap 2.1 — Metric tensor validation (Gerschgorin + Tikhonov)
// ============================================================================

/**
 * @brief Validates and repairs a 9×9 Riemannian metric tensor g.
 *
 * The metric must be positive-definite before Cholesky decomposition (used by
 * the physics engine).  A degenerate metric crashes the propagator.
 *
 * Two-stage strategy (engineering report §Gap 2.1):
 *   1. Fast path:  Gerschgorin circle theorem — O(81) ops, no sqrt.
 *      If all diagonal entries strictly exceed the sum of off-diagonal
 *      magnitudes in their row, the matrix is strictly diagonally dominant
 *      and (since diagonals are positive) positive-definite.
 *   2. Fallback:   Tikhonov regularisation — g'_ij = g_ij + δ·I.
 *      δ = 1e-5 "stiffens" space, keeps physics tractable.
 *
 * @param g  Pointer to flat 81-element (row-major) metric tensor.
 *           Modified in-place if Tikhonov fallback is triggered.
 * @return   true  if already positive-definite (Gerschgorin passed).
 *           false if Tikhonov was applied (metric was degenerate,
 *                 now remedied — caller should log as a learning event).
 */
class MetricValidator {
public:
    static constexpr int   DIMS            = TORUS_DIMS;
    static constexpr float TIKHONOV_DELTA  = 1e-5f;

    /**
     * @brief Check strict diagonal dominance (all diagonal > row sums).
     */
    [[nodiscard]]
    static bool gerschgorin_check(const float g[81]) noexcept {
        for (int i = 0; i < DIMS; ++i) {
            const float diag    = g[i * DIMS + i];
            if (diag <= 0.f) return false;   // diagonal must be positive
            float row_sum = 0.f;
            for (int j = 0; j < DIMS; ++j) {
                if (j != i) row_sum += std::abs(g[i * DIMS + j]);
            }
            if (diag <= row_sum) return false;
        }
        return true;
    }

    /**
     * @brief Ensure g is positive-definite; apply adaptive Tikhonov if needed.
     *
     * Strategy:
     *   1. Fast Gerschgorin check → return true if already valid.
     *   2. Adaptive Tikhonov: for each row i, add exactly enough to the
     *      diagonal so that diag_ii > Σ_{j≠i}|g_ij| + ε.
     *      This is the minimum-disturbance fix guaranteed to pass Gerschgorin.
     *
     * @return true  = already valid, false = Tikhonov applied.
     */
    static bool ensure_positive_definite(float g[81]) noexcept {
        if (gerschgorin_check(g)) return true;

        // Adaptive per-row Tikhonov: add exactly enough to each diagonal.
        for (int i = 0; i < DIMS; ++i) {
            float row_sum = 0.f;
            for (int j = 0; j < DIMS; ++j) {
                if (j != i) row_sum += std::abs(g[i * DIMS + j]);
            }
            const float deficit = row_sum + TIKHONOV_DELTA - g[i * DIMS + i];
            if (deficit > 0.f) {
                g[i * DIMS + i] += deficit;
            }
        }

        return false;   // caller: log as a learning/anomaly event
    }
};

// ============================================================================
// Gap 2.5 — Dopamine-modulated metric learning rate
// ============================================================================

/**
 * @brief Hebbian metric learner with dopamine-modulated annealing (Gap 2.5).
 *
 * Learning rate schedule:
 *
 *   η(t) = η_base × D(t) × 1 / (1 + τ × age_seconds)
 *
 * where:
 *   η_base = 0.01   — base plasticity
 *   D(t)   ∈ [0,1]  — dopamine level (reward signal)
 *   age    seconds  — time since node creation
 *   τ      = 0.001  — consolidation time constant
 *
 * Biological rationale:
 *   Young nodes (recent memories) are highly plastic.
 *   Old nodes (consolidated memories) resist change unless
 *   dopamine (surprise/reward) is high.
 */
class MetricLearner {
public:
    static constexpr float ETA_BASE = 0.01f;
    static constexpr float TAU      = 0.001f;

    /**
     * @brief Compute instantaneous learning rate.
     *
     * @param dopamine      Dopamine level in [0, 1].
     * @param age_seconds   Seconds since the memory was formed.
     * @return              η in [0, η_base].
     */
    [[nodiscard]]
    float compute_learning_rate(float dopamine, float age_seconds) const noexcept {
        const float age_factor = 1.0f / (1.0f + TAU * age_seconds);
        return ETA_BASE * dopamine * age_factor;
    }

    /**
     * @brief Hebbian update of metric tensor.
     *
     * Δg_ij = η × correlation[i,j]
     *
     * After update, applies MetricValidator to keep g positive-definite.
     *
     * @param g             81-element row-major metric tensor (in/out).
     * @param correlation   81-element co-activation outer product Ψ_i × Ψ_j*.
     * @param dopamine      Reward signal [0, 1].
     * @param age_seconds   Node age in seconds.
     * @return              true if metric was already valid, false if repaired.
     */
    bool update_metric(float g[81], const float correlation[81],
                       float dopamine, float age_seconds) noexcept
    {
        const float lr = compute_learning_rate(dopamine, age_seconds);
        for (int i = 0; i < 81; ++i) {
            g[i] += lr * correlation[i];
        }
        return MetricValidator::ensure_positive_definite(g);
    }
};

// ============================================================================
// TopologyManager — high-level interface
// ============================================================================

/**
 * @brief Manages the 9D toroidal manifold geometry for a given GridConfig.
 *
 * Provides:
 *   - Coordinate conversion between integer and physical spaces (Gap 2.4)
 *   - Metric tensor validation and repair (Gap 2.1)
 *   - Dopamine-modulated Hebbian learning (Gap 2.5)
 *   - Toroidal distance computation
 *   - Sub-grid peak interpolation (Gap 2.4)
 *
 * The default global metric tensor is the identity (flat space).
 * Use update_metric() to apply Hebbian learning.
 */
class TopologyManager {
public:
    // ------------------------------------------------------------------ construction

    explicit TopologyManager(GridConfig config = GridConfig::anisotropic_default())
        : config_(std::move(config))
    {
        // Initialise metric as identity (flat Euclidean space).
        std::fill(metric_, metric_ + 81, 0.f);
        for (int k = 0; k < TORUS_DIMS; ++k) metric_[k * TORUS_DIMS + k] = 1.f;
    }

    // ------------------------------------------------------------------ coordinate conversion (Gap 2.4)

    /// Integer → physical coordinates.
    [[nodiscard]]
    Coord9DFloat to_physical(const Coord9DInt& ic) const noexcept {
        return nikola::spatial::to_physical(ic, config_);
    }

    /// Physical → integer coordinates.
    [[nodiscard]]
    Coord9DInt to_integer(const Coord9DFloat& fc) const noexcept {
        return nikola::spatial::to_integer(fc, config_);
    }

    /// Round-trip: integer → physical → integer (lossless if within bounds).
    [[nodiscard]]
    Coord9DInt round_trip(const Coord9DInt& ic) const noexcept {
        return to_integer(to_physical(ic));
    }

    // ------------------------------------------------------------------ peak interpolation (Gap 2.4)

    /**
     * @brief Sub-grid quadratic peak interpolation.
     * @return Fractional offset of the amplitude peak from the centre index.
     */
    [[nodiscard]]
    static float peak_offset(float v_left, float v_centre, float v_right) noexcept {
        return interpolate_peak(v_left, v_centre, v_right);
    }

    // ------------------------------------------------------------------ metric management (Gap 2.1)

    /**
     * @brief Validate (and repair) the current global metric tensor.
     * @return true if valid, false if Tikhonov was applied.
     */
    bool validate_metric() noexcept {
        return MetricValidator::ensure_positive_definite(metric_);
    }

    /// Read the current 9×9 metric tensor (row-major, 81 floats).
    const float* metric() const noexcept { return metric_; }

    /// Write a new metric tensor (must be 81 floats, row-major).
    void set_metric(const float g[81]) noexcept {
        std::copy(g, g + 81, metric_);
    }

    // ------------------------------------------------------------------ metric learning (Gap 2.5)

    /**
     * @brief Apply one Hebbian update to the global metric.
     *
     * @param correlation   Outer-product co-activation (81 floats, row-major).
     * @param dopamine      Dopamine level [0, 1].
     * @param age_seconds   Age of the associated memory/node in seconds.
     * @return              true if metric was already valid, false if repaired.
     */
    bool update_metric(const float correlation[81],
                       float dopamine, float age_seconds) noexcept
    {
        return learner_.update_metric(metric_, correlation, dopamine, age_seconds);
    }

    /**
     * @brief Learning rate for given dopamine / age (for inspection/testing).
     */
    [[nodiscard]]
    float learning_rate(float dopamine, float age_seconds) const noexcept {
        return learner_.compute_learning_rate(dopamine, age_seconds);
    }

    // ------------------------------------------------------------------ toroidal distance

    /**
     * @brief Flat toroidal L2 distance (Euclidean in fractional coordinates).
     *
     * For each dimension d:
     *   Δx_d = min(|a_d - b_d|, N_d - |a_d - b_d|) / N_d   (toroidal wrap)
     * Distance = sqrt(Σ Δx_d²)
     */
    [[nodiscard]]
    float toroidal_distance(const Coord9DInt& a, const Coord9DInt& b) const noexcept {
        float sum2 = 0.f;
        for (int d = 0; d < TORUS_DIMS; ++d) {
            const float Nd   = static_cast<float>(config_.resolution[d]);
            const float diff = std::abs(static_cast<float>(a.c[d]) -
                                        static_cast<float>(b.c[d]));
            const float wrap = std::min(diff, Nd - diff) / Nd;
            sum2 += wrap * wrap;
        }
        return std::sqrt(sum2);
    }

    // ------------------------------------------------------------------ config access

    const GridConfig& config() const noexcept { return config_; }

    MetricLearner&       learner()       noexcept { return learner_; }
    const MetricLearner& learner() const noexcept { return learner_; }

private:
    GridConfig    config_;
    MetricLearner learner_;
    float         metric_[81];  ///< 9×9 row-major Riemannian metric tensor
};

} // namespace nikola::spatial
