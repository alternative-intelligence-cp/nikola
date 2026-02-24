/**
 * @file coordinate_semantics.hpp
 * @brief GAP-041: Glossary of 9D Coordinate Semantics
 *
 * @spec FABRICATION-READY — docs/info/integration/sections/02_foundations/
 *       01_9d_toroidal_geometry.md §GAP-041
 *
 * ### The 9-Dimensional Toroidal Manifold T⁹
 * Nikola's memory substrate is T⁹ = S¹ × S¹ × … × S¹ (9 circles).
 * Unlike flat R^n in Transformer embeddings, T⁹ is:
 *   • Compact  — finite volume, uniform data density, no "curse of dimensionality"
 *   • Boundary-less — no edge effects distorting peripheral concepts
 *   • Homogeneous — every point has identical local topology
 *
 * The 9 dimensions are NOT interchangeable — each maps to specific physical
 * properties of the wave medium:
 *
 *   Domain I   — Systemic (dims 0,1)  : physics constants of local neighbourhood
 *   Domain II  — Temporal (dim 2)     : causal backbone
 *   Domain III — Quantum  (dims 3,4,5): information content (complex amplitudes)
 *   Domain IV  — Spatial  (dims 6,7,8): discrete lattice address
 *
 * ### Physical formulas (spec §GAP-041)
 *   Effective wave speed   : c_eff = c0 / (1 + ŝ)          (ŝ ∈ [0,2])
 *   Damping coefficient    : γ = α × (1 − r̂)               (r̂ ∈ [0,1])
 *   Spatial resolution     : 14 bits/dim → 2^14 = 16,384 nodes/axis
 *   Morton encoding width  : 9 × 14 = 126 bits (fits in __uint128_t)
 */
#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace nikola::physics {

// ---------------------------------------------------------------------------
// Fundamental constants
// ---------------------------------------------------------------------------

/// Total number of dimensions in the toroidal manifold T⁹.
inline constexpr int DIM_COUNT = 9;

/// Number of bits per spatial dimension (x, y, z) in the 128-bit Morton key.
/// Supports 2^14 = 16,384 nodes per axis — "effectively infinite" for AGI.
inline constexpr int BITS_PER_SPATIAL_DIM = 14;

/// Maximum node index per spatial axis (half-open: [0, SPATIAL_AXIS_MAX)).
inline constexpr int SPATIAL_AXIS_MAX = 1 << BITS_PER_SPATIAL_DIM;  // 16384

/// Total Morton key bits: 9 dims × 14 bits = 126 bits (fits in __uint128_t).
inline constexpr int MORTON_KEY_BITS = DIM_COUNT * BITS_PER_SPATIAL_DIM;

/// Nominal (maximum) wave speed c₀ — normalised to 1.0.
inline constexpr double WAVE_SPEED_NOMINAL = 1.0;

/// Range of the Resonance dimension r̂ ∈ [RESONANCE_MIN, RESONANCE_MAX].
inline constexpr double RESONANCE_MIN = 0.0;
inline constexpr double RESONANCE_MAX = 1.0;

/// Range of the State dimension ŝ ∈ [STATE_MIN, STATE_MAX].
inline constexpr double STATE_MIN = 0.0;
inline constexpr double STATE_MAX = 2.0;

// ---------------------------------------------------------------------------
// Domain classification
// ---------------------------------------------------------------------------

/**
 * @brief The four functional domains of T⁹ as defined in spec §GAP-041.
 *
 * SYSTEMIC  — "physics constants" controlling energy flow and wave velocity
 * TEMPORAL  — causal ordering with toroidal cyclicity
 * QUANTUM   — complex amplitude carrier (superposition & interference)
 * SPATIAL   — discrete lattice address for semantic mapping
 */
enum class CoordDomain : int {
    SYSTEMIC  = 0,   ///< Dims 0,1: Resonance (r), State (s)
    TEMPORAL  = 1,   ///< Dim  2:   Time (t) — cyclic
    QUANTUM   = 2,   ///< Dims 3,4,5: u, v, w — complex float
    SPATIAL   = 3,   ///< Dims 6,7,8: x, y, z — 14-bit integer
};

// ---------------------------------------------------------------------------
// Dimension index
// ---------------------------------------------------------------------------

/**
 * @brief Strongly-typed 0-based index for the 9 T⁹ dimensions.
 *
 * The ordering is canonical across the codebase (Morton encoding, SoA
 * layout, GGUF serialisation, Physics Oracle).
 */
enum class Dim9 : int {
    RESONANCE = 0,   ///< r — gain/Q-factor, damping, LTP
    STATE     = 1,   ///< s — refractive index, wave velocity, focus
    TIME      = 2,   ///< t — cyclic causal backbone [0, T_period)
    U         = 3,   ///< u — wavefunction component (complex)
    V         = 4,   ///< v — wavefunction component (complex)
    W         = 5,   ///< w — wavefunction component (complex)
    X         = 6,   ///< x — spatial integer address (14-bit)
    Y         = 7,   ///< y — spatial integer address (14-bit)
    Z         = 8,   ///< z — spatial integer address (14-bit)
};

// ---------------------------------------------------------------------------
// Per-dimension queries
// ---------------------------------------------------------------------------

/**
 * @brief Return the domain of a given dimension.
 *
 * | Dim9         | Domain   |
 * |--------------|----------|
 * | RESONANCE, STATE | SYSTEMIC |
 * | TIME             | TEMPORAL |
 * | U, V, W          | QUANTUM  |
 * | X, Y, Z          | SPATIAL  |
 */
[[nodiscard]] inline constexpr CoordDomain dim_domain(Dim9 d) noexcept {
    switch (d) {
        case Dim9::RESONANCE:
        case Dim9::STATE:
            return CoordDomain::SYSTEMIC;
        case Dim9::TIME:
            return CoordDomain::TEMPORAL;
        case Dim9::U:
        case Dim9::V:
        case Dim9::W:
            return CoordDomain::QUANTUM;
        case Dim9::X:
        case Dim9::Y:
        case Dim9::Z:
            return CoordDomain::SPATIAL;
    }
    // unreachable — enumerator values are exhaustive
    return CoordDomain::SYSTEMIC;
}

/**
 * @brief Human-readable lowercase name of the dimension.
 * Used in diagnostics, GGUF metadata keys, and Python bindings.
 */
[[nodiscard]] inline constexpr std::string_view dim_name(Dim9 d) noexcept {
    constexpr std::array<std::string_view, 9> names {
        "resonance", "state", "time", "u", "v", "w", "x", "y", "z"
    };
    return names[static_cast<int>(d)];
}

/**
 * @brief Single-character symbol used in formulae and visualisation labels.
 * (r, s, t, u, v, w, x, y, z)
 */
[[nodiscard]] inline constexpr char dim_symbol(Dim9 d) noexcept {
    constexpr std::array<char, 9> symbols {
        'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
    };
    return symbols[static_cast<int>(d)];
}

/**
 * @brief True for integer-valued dimensions (x, y, z).
 * False for continuous floating-point dimensions (r, s, t, u, v, w).
 */
[[nodiscard]] inline constexpr bool dim_is_integer(Dim9 d) noexcept {
    return dim_domain(d) == CoordDomain::SPATIAL;
}

/**
 * @brief True for cyclically-wrapped dimensions.
 *
 * TIME wraps at T_period (toroidal cyclicity).
 * SPATIAL dims (x, y, z) also exhibit toroidal wrapping at lattice boundaries.
 * RESONANCE, STATE, and QUANTUM dims are clamped, not wrapped.
 */
[[nodiscard]] inline constexpr bool dim_is_cyclic(Dim9 d) noexcept {
    return d == Dim9::TIME
        || d == Dim9::X
        || d == Dim9::Y
        || d == Dim9::Z;
}

/**
 * @brief True for complex-valued dimensions (u, v, w).
 */
[[nodiscard]] inline constexpr bool dim_is_complex(Dim9 d) noexcept {
    return dim_domain(d) == CoordDomain::QUANTUM;
}

// ---------------------------------------------------------------------------
// Physical formulas (spec §GAP-041)
// ---------------------------------------------------------------------------

/**
 * @brief Effective wave speed at a node with State value s_normalized.
 *
 * Formula (spec §GAP-041 §"Dimension 2: State"):
 *   c_eff = c₀ / (1 + ŝ)        ŝ ∈ [0, 2]
 *
 * @param s_normalized  State dimension value ŝ in [STATE_MIN, STATE_MAX]
 *
 * Behaviour:
 *   s = 0.0 → c_eff = 1.0   (vacuum, maximum speed, "scanning/skimming")
 *   s = 1.0 → c_eff = 0.5   (moderate focus)
 *   s = 2.0 → c_eff ≈ 0.333 (deep focus, "refractive trap")
 */
[[nodiscard]] inline constexpr double wave_speed_effective(double s_normalized) noexcept {
    return WAVE_SPEED_NOMINAL / (1.0 + s_normalized);
}

/**
 * @brief Damping coefficient γ at a node given resonance r and base rate α.
 *
 * Formula (spec §GAP-041 §"Dimension 1: Resonance"):
 *   γ = α × (1 − r̂)           r̂ ∈ [0, 1]
 *
 * @param resonance  Resonance dimension value r̂ in [RESONANCE_MIN, RESONANCE_MAX]
 * @param alpha      Base damping rate α (physics engine constant, > 0)
 *
 * Behaviour:
 *   r = 1.0 → γ = 0       (superconductor of information, LTP — no damping)
 *   r = 0.5 → γ = α/2     (half-damped working memory)
 *   r = 0.0 → γ = α       (fully dissipative, memories fade instantly)
 */
[[nodiscard]] inline constexpr double damping_coefficient(double resonance,
                                                           double alpha) noexcept {
    return alpha * (1.0 - resonance);
}

/**
 * @brief Maximum addressable node count per spatial axis.
 *
 * With BITS_PER_SPATIAL_DIM = 14:
 *   max_nodes_per_axis() == 2^14 == 16,384
 *
 * This ensures no Morton-code collision for any physically realizable grid.
 */
[[nodiscard]] inline constexpr int max_nodes_per_axis() noexcept {
    return SPATIAL_AXIS_MAX;
}

/**
 * @brief Total addressable nodes in the 3D spatial lattice.
 *
 * 16384³ ≈ 4.4 × 10¹² — far larger than any practical Nikola deployment.
 * Stored as int64 to avoid overflow in the return type.
 */
[[nodiscard]] inline constexpr std::int64_t max_spatial_nodes() noexcept {
    return static_cast<std::int64_t>(SPATIAL_AXIS_MAX)
         * static_cast<std::int64_t>(SPATIAL_AXIS_MAX)
         * static_cast<std::int64_t>(SPATIAL_AXIS_MAX);
}

// ---------------------------------------------------------------------------
// Domain name helper
// ---------------------------------------------------------------------------

/**
 * @brief Human-readable name for a CoordDomain value.
 */
[[nodiscard]] inline constexpr std::string_view domain_name(CoordDomain dom) noexcept {
    switch (dom) {
        case CoordDomain::SYSTEMIC:  return "Systemic";
        case CoordDomain::TEMPORAL:  return "Temporal";
        case CoordDomain::QUANTUM:   return "Quantum";
        case CoordDomain::SPATIAL:   return "Spatial";
    }
    return "Unknown";
}

} // namespace nikola::physics
