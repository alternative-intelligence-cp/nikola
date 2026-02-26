// SPDX-License-Identifier: MIT
// GAP-016: Inner Monologue Recursion & ATP Policy Constants
// Phase 89 — nikola::cognitive
//
// Encodes the hard limits, cost model and threshold values that govern
// how deep Nikola's inner monologue may recurse and when it must rest.
//
// Key derivations:
//   • Recursion hard limit  D_hard = 12   (Mamba-9D context horizon / phase
//                                          coherence floor at λ = 0.15 per step)
//   • Recursion soft limit  D_soft =  7   (Miller's Law 7 ± 2, thermodynamic
//                                          optimum under RECURSION_PENALTY_RATE)
//   • Per-depth ATP cost    C(d)   = C_base × (1 + λ)^d   (geometric tax)
//   • Spectral entropy limit 0.85  — above this the signal is indistinguishable
//                                    from thermal noise (decoherence / scrambling)
//
// Source: 03_neuroplastic_transformer.md §"Inner Monologue Policy"

#pragma once

#include <cstdint>
#include <cmath>
#include <string_view>

namespace nikola::cognitive {

// ─── Recursion depth limits ───────────────────────────────────────────────────

/// Hard maximum recursion depth permitted by the Mamba-9D context horizon.
/// Exceeding this causes a COG-001 RunawayCognitiveLoop fault.
inline constexpr int RECURSION_HARD_LIMIT    = 12;

/// Soft (preferred) recursion depth based on Miller's Law and thermodynamic
/// optimum under the geometric ATP penalty.
inline constexpr int RECURSION_SOFT_LIMIT    = 7;

static_assert(RECURSION_SOFT_LIMIT < RECURSION_HARD_LIMIT,
    "Soft limit must be below hard limit");

// ─── ATP energy model ─────────────────────────────────────────────────────────

/// ATP reserve fraction below which the Engine forces an immediate Forced-Nap.
/// Corresponds to the survival / emergency energy floor.
inline constexpr double ATP_RESERVE_THRESHOLD  = 0.15;

/// ATP reserve fraction below which cognition is considered critically exhausted.
inline constexpr double ATP_RESERVE_CRITICAL   = 0.05;

/// Baseline ATP cost consumed per active reasoning step at depth 0.
inline constexpr double ATP_BASE_COST_PER_STEP = 0.05;

/// Compound geometric penalty rate per recursion depth increment (λ = 15 %).
/// Per-depth cost: C(d) = ATP_BASE_COST_PER_STEP × (1 + RECURSION_PENALTY_RATE)^d
inline constexpr double RECURSION_PENALTY_RATE = 0.15;

// ─── Spectral entropy thresholds ─────────────────────────────────────────────

/// Spectral entropy above this value is indistinguishable from thermal noise;
/// the cognitive step is considered incoherent and must be discarded.
inline constexpr double SPECTRAL_ENTROPY_LIMIT   = 0.85;

/// Rate-of-change of spectral entropy that indicates rapid phase decoherence
/// ("scrambling").  If ΔH/Δstep > ENTROPY_GRADIENT_LIMIT, trigger soft-scram.
inline constexpr double ENTROPY_GRADIENT_LIMIT   = 0.05;

// ─── Boredom / teleological deadlock ─────────────────────────────────────────

/// Boredom dopamine spike magnitude injected when resolving a BoredomSingularity
/// (COG-002).  Expressed as a fraction of maximum reward signal.
inline constexpr double BOREDOM_LOOP_SPIKE       = 0.20;

// ─── Trap cluster geometry (working-memory layout) ───────────────────────────

/// Number of torus nodes in one trap cluster (central node + 18-point stencil).
inline constexpr int  TRAP_NODES               = 19;

/// Approximate memory occupied by one trap node, in kilobytes.
inline constexpr double TRAP_KB_PER_NODE       = 3.4;

/// Total memory for one trap cluster (≈ 64.6 KB ≈ 65 KB).
inline constexpr double TRAP_KB_TOTAL          = TRAP_NODES * TRAP_KB_PER_NODE; // 64.6

/// Maximum simultaneously active trap clusters within the 1 ms physics frame.
inline constexpr int  MAX_ACTIVE_TRAPS         = 9;

// ─── Computed ATP cost table ─────────────────────────────────────────────────

/// ATP cost consumed at recursion depth d:
///     cost(d) = ATP_BASE_COST_PER_STEP × (1 + RECURSION_PENALTY_RATE)^d
[[nodiscard]] inline constexpr double atp_cost_at_depth(int d) noexcept {
    // constexpr-compatible manual pow for small integer d
    double factor = 1.0;
    for (int i = 0; i < d; ++i) factor *= (1.0 + RECURSION_PENALTY_RATE);
    return ATP_BASE_COST_PER_STEP * factor;
}

/// Cumulative ATP cost from depth 0 through depth d (inclusive).
[[nodiscard]] inline constexpr double atp_cumulative_cost(int d) noexcept {
    double total = 0.0;
    for (int i = 0; i <= d; ++i) total += atp_cost_at_depth(i);
    return total;
}

// ─── Policy query functions ───────────────────────────────────────────────────

/// True if the given depth is within the soft (preferred) limit.
[[nodiscard]] constexpr bool depth_within_soft_limit(int d) noexcept {
    return d <= RECURSION_SOFT_LIMIT;
}

/// True if the given depth is within the hard (absolute) limit.
[[nodiscard]] constexpr bool depth_within_hard_limit(int d) noexcept {
    return d < RECURSION_HARD_LIMIT;
}

/// True if spending 'cost' ATP would leave the reserve above the threshold.
/// @param current_reserve   Current ATP reserve fraction in [0, 1].
/// @param cost              Fractional ATP cost of the next step.
[[nodiscard]] constexpr bool atp_affordable(double current_reserve, double cost) noexcept {
    return (current_reserve - cost) >= ATP_RESERVE_THRESHOLD;
}

/// True when the ATP reserve has dropped to the critical floor (< 5 %).
[[nodiscard]] constexpr bool atp_critical(double reserve) noexcept {
    return reserve < ATP_RESERVE_CRITICAL;
}

/// True when spectral entropy indicates the cognitive step is incoherent.
[[nodiscard]] constexpr bool entropy_incoherent(double entropy) noexcept {
    return entropy >= SPECTRAL_ENTROPY_LIMIT;
}

/// True when entropy rate-of-change indicates rapid scrambling.
[[nodiscard]] constexpr bool entropy_scrambling(double delta_entropy) noexcept {
    return delta_entropy > ENTROPY_GRADIENT_LIMIT;
}

/// True when the trap cluster count would exceed the 1 ms frame budget.
[[nodiscard]] constexpr bool trap_budget_exceeded(int active_traps) noexcept {
    return active_traps > MAX_ACTIVE_TRAPS;
}

// ─── Label helpers ───────────────────────────────────────────────────────────

[[nodiscard]] constexpr std::string_view depth_policy_label(int d) noexcept {
    if (d < 0)                        return "invalid";
    if (d <= RECURSION_SOFT_LIMIT)    return "nominal";
    if (d < RECURSION_HARD_LIMIT)     return "overextended";
    return "hard-limit-breach";
}

[[nodiscard]] constexpr std::string_view atp_reserve_label(double reserve) noexcept {
    if (reserve < ATP_RESERVE_CRITICAL)  return "critical";
    if (reserve < ATP_RESERVE_THRESHOLD) return "warning";
    return "nominal";
}

} // namespace nikola::cognitive
