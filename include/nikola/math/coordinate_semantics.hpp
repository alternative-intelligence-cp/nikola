// SPDX-License-Identifier: MIT
// GAP-041: 9D Coordinate Dimension Semantics Glossary
// Phase 87 — nikola::math
//
// Authoritative names, types, ranges and predicates for each of the
// nine dimensions of the Nikola toroidal coordinate space.
//
// Source: 01_9d_toroidal_geometry.md §"9D Dimension Glossary"

#pragma once

#include <cstdint>
#include <string_view>
#include <complex>

namespace nikola::math {

// ─── Domain groupings ─────────────────────────────────────────────────────────

/// The four physical domains that the 9D space partitions into.
enum class CoordinateDomain : uint8_t {
    SYSTEMIC  = 0,   ///< dims 1–2  (r, s) — neural resonance & state
    TEMPORAL  = 1,   ///< dim  3    (t)     — causal ordering / clock
    QUANTUM   = 2,   ///< dims 4–6  (u,v,w) — superposition / interference
    SPATIAL   = 3    ///< dims 7–9  (x,y,z) — topological address
};

/// Each of the 9 named dimensions.
enum class Dimension : uint8_t {
    R = 0,  ///< Systemic — resonance / LTP weight  [0, 1]
    S = 1,  ///< Systemic — state / refractive index [0, 2]
    T = 2,  ///< Temporal — cyclic time              [0, T_period)
    U = 3,  ///< Quantum  — superposition component  complex<float>
    V = 4,  ///< Quantum  — superposition component  complex<float>
    W = 5,  ///< Quantum  — superposition component  complex<float>
    X = 6,  ///< Spatial  — semantic address dim-7   int14
    Y = 7,  ///< Spatial  — semantic address dim-8   int14
    Z = 8   ///< Spatial  — semantic address dim-9   int14
};

// ─── Systemic dimensions (r, s) ───────────────────────────────────────────────

/// dim-1 r — resonance / long-term-potentiation weight
/// Type: float, range [0, 1]
inline constexpr float RESONANCE_MIN         = 0.0f;
inline constexpr float RESONANCE_MAX         = 1.0f;
/// Threshold above which a node is considered high-Q (strongly potentiated)
inline constexpr float RESONANCE_HIGH_Q      = 0.8f;
/// Threshold below which a node is considered transient / low-Q
inline constexpr float RESONANCE_LOW_Q       = 0.2f;
/// Default resting resonance (untrained node)
inline constexpr float RESONANCE_DEFAULT     = 0.5f;

/// dim-2 s — refractive state index
/// Type: float, range [0, 2]; s = 0 → vacuum, s = 1 → baseline, s > 1 → deep focus
/// Effective signal speed: c_eff = c0 / (1 + s̃)  where s̃ = s - 1 (deviation from baseline)
inline constexpr float STATE_MIN             = 0.0f;
inline constexpr float STATE_MAX             = 2.0f;
inline constexpr float STATE_VACUUM          = 0.0f;
inline constexpr float STATE_BASELINE        = 1.0f;
inline constexpr float STATE_DEEP_FOCUS_THR  = 1.5f;  ///< s > 1.5  → deep focus

// ─── Temporal dimension (t) ───────────────────────────────────────────────────

/// dim-3 t — cyclic causal time
/// Type: float, range [0, T_period), wraps modulo T_period.
/// T_period is engine-instance specific; not encoded here.
/// CAUSAL CONSTRAINT: messages with t_recv < t_send are retrograde — reject.
inline constexpr float TEMPORAL_ORIGIN       = 0.0f;  ///< epoch reference

// ─── Quantum dimensions (u, v, w) ─────────────────────────────────────────────

/// dims 4–6 u, v, w — complex superposition amplitudes
/// Type: std::complex<float>
/// Normalisation: |u|² + |v|² + |w|² ≤ 1.0  (probabilistic amplitude)
/// A node with |amplitude|² < QUANTUM_DECOHERENCE_THR is treated as classical.
inline constexpr float QUANTUM_DECOHERENCE_THR = 1e-6f;

// ─── Spatial dimensions (x, y, z) ─────────────────────────────────────────────

/// dims 7–9 x, y, z — semantic topological address
/// Type: int (14-bit resolution, signed interpretation for offsets)
/// Range for absolute addresses: [0, SPATIAL_MAX]
inline constexpr int SPATIAL_BITS            = 14;
inline constexpr int SPATIAL_MAX             = (1 << SPATIAL_BITS) - 1;  ///< 16383
inline constexpr int SPATIAL_MIN             = 0;

// ─── Domain-membership queries ────────────────────────────────────────────────

[[nodiscard]] constexpr CoordinateDomain domain_of(Dimension d) noexcept {
    switch (d) {
        case Dimension::R:
        case Dimension::S:  return CoordinateDomain::SYSTEMIC;
        case Dimension::T:  return CoordinateDomain::TEMPORAL;
        case Dimension::U:
        case Dimension::V:
        case Dimension::W:  return CoordinateDomain::QUANTUM;
        default:            return CoordinateDomain::SPATIAL;
    }
}

[[nodiscard]] constexpr bool is_systemic(Dimension d) noexcept { return domain_of(d) == CoordinateDomain::SYSTEMIC; }
[[nodiscard]] constexpr bool is_temporal(Dimension d) noexcept { return domain_of(d) == CoordinateDomain::TEMPORAL; }
[[nodiscard]] constexpr bool is_quantum  (Dimension d) noexcept { return domain_of(d) == CoordinateDomain::QUANTUM; }
[[nodiscard]] constexpr bool is_spatial  (Dimension d) noexcept { return domain_of(d) == CoordinateDomain::SPATIAL; }

// ─── Resonance (r) predicates ─────────────────────────────────────────────────

/// True when a node exhibits high-Q long-term potentiation (r > 0.8)
[[nodiscard]] constexpr bool is_high_q_resonance(float r) noexcept { return r > RESONANCE_HIGH_Q; }
/// True when a node is transient / not yet potentiated (r < 0.2)
[[nodiscard]] constexpr bool is_transient(float r) noexcept { return r < RESONANCE_LOW_Q; }
/// True when r is within normalised bounds
[[nodiscard]] constexpr bool is_valid_resonance(float r) noexcept { return r >= RESONANCE_MIN && r <= RESONANCE_MAX; }

// ─── State (s) predicates ─────────────────────────────────────────────────────

/// True when the node is in deep-focus mode (intensive processing, s > 1.5)
[[nodiscard]] constexpr bool is_deep_focus(float s) noexcept { return s > STATE_DEEP_FOCUS_THR; }
/// True when the node is near vacuum (minimal activity, s < 0.1)
[[nodiscard]] constexpr bool is_near_vacuum(float s) noexcept { return s < 0.1f; }
/// True when s is within normalised bounds
[[nodiscard]] constexpr bool is_valid_state(float s) noexcept { return s >= STATE_MIN && s <= STATE_MAX; }

/// Effective signal propagation speed in normalised units (c0 = 1.0 by convention)
[[nodiscard]] constexpr float effective_speed(float s) noexcept { return 1.0f / (1.0f + (s - STATE_BASELINE)); }

// ─── Spatial (x, y, z) predicates ────────────────────────────────────────────

/// True when an absolute spatial coordinate is within the addressable range
[[nodiscard]] constexpr bool is_valid_spatial(int coord) noexcept { return coord >= SPATIAL_MIN && coord <= SPATIAL_MAX; }

// ─── Quantum amplitude predicates ────────────────────────────────────────────

/// True when a quantum amplitude has effectively decohered to classical
[[nodiscard]] inline bool is_decohered(std::complex<float> amp) noexcept {
    return (amp.real() * amp.real() + amp.imag() * amp.imag()) < QUANTUM_DECOHERENCE_THR;
}

// ─── Label helpers (no allocation) ───────────────────────────────────────────

[[nodiscard]] constexpr std::string_view dimension_label(Dimension d) noexcept {
    switch (d) {
        case Dimension::R: return "r (resonance)";
        case Dimension::S: return "s (state)";
        case Dimension::T: return "t (time)";
        case Dimension::U: return "u (quantum-u)";
        case Dimension::V: return "v (quantum-v)";
        case Dimension::W: return "w (quantum-w)";
        case Dimension::X: return "x (spatial-x)";
        case Dimension::Y: return "y (spatial-y)";
        case Dimension::Z: return "z (spatial-z)";
        default:           return "unknown";
    }
}

[[nodiscard]] constexpr std::string_view domain_label(CoordinateDomain d) noexcept {
    switch (d) {
        case CoordinateDomain::SYSTEMIC: return "systemic";
        case CoordinateDomain::TEMPORAL: return "temporal";
        case CoordinateDomain::QUANTUM:  return "quantum";
        case CoordinateDomain::SPATIAL:  return "spatial";
        default:                         return "unknown";
    }
}

} // namespace nikola::math
