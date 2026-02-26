#pragma once
// =============================================================================
// nikola/system/latency_budget.hpp
// Phase 86 — GAP-025: End-to-End Latency Budget Allocation
//
// SOURCE: Gemini Deep Research Round 2, Batch 25-27 (December 15, 2025)
// SPEC:   docs/info/integration/sections/02_foundations/02_wave_interference_physics.md
//         §GAP-025 (lines ~9649–9852)
//
// Authoritative per-component latency budget for the 1000 Hz physics loop.
// Total = 1000 μs; allocatable = 900 μs (10% safety margin for OS jitter).
// =============================================================================

#include <cstdint>
#include <string_view>

namespace nikola::system {

// ---------------------------------------------------------------------------
// § Enumerations
// ---------------------------------------------------------------------------

/// Physics loop component whose time usage is tracked.
enum class LoopComponent : uint8_t {
    PHYSICS_KERNEL    = 0,  ///< Wave propagation (Strang splitting, FFT, soliton)
    COGNITIVE_SCANNER = 1,  ///< Mamba-9D causal-foliated Hilbert scan + SSM
    ENGS              = 2,  ///< Neurochemical gating (RPE, parameter broadcast)
    INFRASTRUCTURE    = 3,  ///< IPC / Seqlock / ZMQ control check
    SAFETY_MARGIN     = 4,  ///< OS jitter, interrupt handling reserve
};

/// Physics kernel sub-component.
enum class PhysicsSubComponent : uint8_t {
    METRIC_TENSOR_UPDATE = 0,  ///< Hebbian-Riemannian plasticity (memory-bound)
    POTENTIAL_STEP       = 1,  ///< V̂/2 point-wise phase rotation
    KINETIC_STEP_FFT     = 2,  ///< T̂ via dimensional operator splitting FFT
    NONLINEAR_SOLITON    = 3,  ///< β|Ψ|²Ψ cubic nonlinearity
    BOUNDARY_CONDITIONS  = 4,  ///< Toroidal periodic wrapping, Morton codes
};

/// Cognitive scanner sub-component.
enum class CognitiveSubComponent : uint8_t {
    HILBERT_SCAN   = 0,  ///< Causal-foliated 9D→1D Hilbert traversal
    SSM_RECURRENCE = 1,  ///< Mamba h_t = A h_{t-1} + B x_t
};

/// Tick health classification.
enum class TickHealth : uint8_t {
    NOMINAL   = 0,  ///< ≤ 900 μs — all components within budget
    WARNING   = 1,  ///< 900 – 950 μs — approaching limit; throttle neurogenesis
    CRITICAL  = 2,  ///< 950 – 1050 μs — Soft SCRAM threshold
    OVERRUN   = 3,  ///< > 1050 μs — hard overrun; "Time Dilation" / Goldfish Effect
};

// ---------------------------------------------------------------------------
// § Top-level budget constants (μs)
// ---------------------------------------------------------------------------

/// Total hard physics loop period in microseconds (1 / 1000 Hz).
inline constexpr double TICK_BUDGET_TOTAL_US    = 1000.0;

/// OS jitter safety margin (10%).
inline constexpr double TICK_SAFETY_MARGIN_US   = 100.0;

/// Net allocatable budget for computational components.
inline constexpr double TICK_BUDGET_NET_US      = TICK_BUDGET_TOTAL_US - TICK_SAFETY_MARGIN_US; // 900

// ---------------------------------------------------------------------------
// § Per-component budgets (μs)
// ---------------------------------------------------------------------------

/// Physics kernel allocation (66.6% of net budget).
inline constexpr double BUDGET_PHYSICS_KERNEL_US    = 600.0;

/// Mamba-9D cognitive scanner allocation (22% of net budget).
inline constexpr double BUDGET_COGNITIVE_SCANNER_US = 200.0;

/// ENGS neurochemical gating allocation (5.5% of net budget).
inline constexpr double BUDGET_ENGS_US              = 50.0;

/// Infrastructure / IPC / ZMQ allocation (5.5% of net budget).
inline constexpr double BUDGET_INFRASTRUCTURE_US    = 50.0;

// Sanity: all component budgets sum to net budget.
static_assert(
    BUDGET_PHYSICS_KERNEL_US + BUDGET_COGNITIVE_SCANNER_US +
    BUDGET_ENGS_US + BUDGET_INFRASTRUCTURE_US == TICK_BUDGET_NET_US,
    "Component budgets must sum to net allocatable budget"
);

// ---------------------------------------------------------------------------
// § Physics kernel sub-component budgets (μs)
// ---------------------------------------------------------------------------

/// Metric tensor update (Hebbian-Riemannian plasticity, memory-bound, SoA).
inline constexpr double BUDGET_METRIC_UPDATE_US     = 50.0;

/// Potential step V̂/2 — point-wise GPU phase rotation.
inline constexpr double BUDGET_POTENTIAL_STEP_US    = 100.0;

/// Kinetic step T̂ via FFT — most expensive, requires dimensional splitting.
inline constexpr double BUDGET_KINETIC_FFT_US       = 300.0;

/// Nonlinear soliton step N̂ — cubic β|Ψ|²Ψ.
inline constexpr double BUDGET_NONLINEAR_US         = 100.0;

/// Boundary conditions + toroidal wrapping + Morton code handling.
inline constexpr double BUDGET_BOUNDARY_US          = 50.0;

static_assert(
    BUDGET_METRIC_UPDATE_US + BUDGET_POTENTIAL_STEP_US +
    BUDGET_KINETIC_FFT_US + BUDGET_NONLINEAR_US + BUDGET_BOUNDARY_US ==
    BUDGET_PHYSICS_KERNEL_US,
    "Physics sub-component budgets must sum to physics kernel budget"
);

// ---------------------------------------------------------------------------
// § Cognitive scanner sub-component budgets (μs)
// ---------------------------------------------------------------------------

/// Causal-foliated Hilbert scan (9D→1D, pre-computed indices via SoA gather).
inline constexpr double BUDGET_HILBERT_SCAN_US      = 80.0;

/// SSM recurrence h_t = A h_{t-1} + B x_t (first-order Taylor approximation).
inline constexpr double BUDGET_SSM_RECURRENCE_US    = 120.0;

static_assert(
    BUDGET_HILBERT_SCAN_US + BUDGET_SSM_RECURRENCE_US ==
    BUDGET_COGNITIVE_SCANNER_US,
    "Cognitive scanner sub-budgets must sum to scanner budget"
);

// ---------------------------------------------------------------------------
// § ENGS sub-component budgets (μs)
// ---------------------------------------------------------------------------

/// Reward Prediction Error computation (energy reduction sum, already computed).
inline constexpr double BUDGET_RPE_US               = 20.0;

/// Global parameter broadcast to GPU constant memory (atomic write).
inline constexpr double BUDGET_PARAM_BROADCAST_US   = 30.0;

static_assert(
    BUDGET_RPE_US + BUDGET_PARAM_BROADCAST_US == BUDGET_ENGS_US,
    "ENGS sub-budgets must sum to ENGS budget"
);

// ---------------------------------------------------------------------------
// § Infrastructure sub-component budgets (μs)
// ---------------------------------------------------------------------------

/// Seqlock /dev/shm ring buffer write (zero-copy, wait-free for writer).
inline constexpr double BUDGET_SEQLOCK_WRITE_US     = 20.0;

/// ZeroMQ non-blocking DEALER socket poll for SCRAM/NAP commands.
inline constexpr double BUDGET_ZMQ_POLL_US          = 30.0;

static_assert(
    BUDGET_SEQLOCK_WRITE_US + BUDGET_ZMQ_POLL_US == BUDGET_INFRASTRUCTURE_US,
    "Infrastructure sub-budgets must sum to infrastructure budget"
);

// ---------------------------------------------------------------------------
// § Alerting thresholds (μs)
// ---------------------------------------------------------------------------

/// Warning threshold: throttle neurogenesis.
/// Spec: "Warning: 950 μs → Throttle neurogenesis (stop adding nodes)"
inline constexpr double TICK_WARNING_US             = 950.0;

/// Critical threshold: Soft SCRAM (apply global damping γ=0.5).
/// Spec: "Critical: 1050 μs → Soft SCRAM (suppress wave complexity)"
inline constexpr double TICK_CRITICAL_US            = 1050.0;

// ---------------------------------------------------------------------------
// § Physics Oracle telemetry constants
// ---------------------------------------------------------------------------

/// Energy drift warning threshold (fraction, 0.01%).
inline constexpr double ENERGY_DRIFT_WARNING        = 0.0001;

/// Energy drift critical threshold (fraction, 0.1%).
/// Spec: "Critical: 0.1% → Emergency Manifold Renormalization"
inline constexpr double ENERGY_DRIFT_CRITICAL_ORACLE = 0.001;

/// ATP warning threshold (fraction, 15%).
inline constexpr double ATP_RESERVE_WARNING          = 0.15;

/// ATP critical threshold (fraction, 5%) — triggers forced Nap.
inline constexpr double ATP_RESERVE_CRITICAL         = 0.05;

/// Hardware watchdog: 2 missed ticks = 2000 μs without a "pet".
inline constexpr double WATCHDOG_TIMEOUT_US          = 2000.0;

// ---------------------------------------------------------------------------
// § Budget fraction helpers
// ---------------------------------------------------------------------------

/// Fraction of the total tick budget consumed by a given duration.
[[nodiscard]] constexpr double tick_budget_fraction(double usage_us) noexcept {
    return usage_us / TICK_BUDGET_TOTAL_US;
}

/// Fraction of the net (allocatable) budget consumed.
[[nodiscard]] constexpr double net_budget_fraction(double usage_us) noexcept {
    return usage_us / TICK_BUDGET_NET_US;
}

/// Return the allocated budget (μs) for a top-level loop component.
[[nodiscard]] constexpr double component_budget_us(LoopComponent c) noexcept {
    switch (c) {
        case LoopComponent::PHYSICS_KERNEL:    return BUDGET_PHYSICS_KERNEL_US;
        case LoopComponent::COGNITIVE_SCANNER: return BUDGET_COGNITIVE_SCANNER_US;
        case LoopComponent::ENGS:              return BUDGET_ENGS_US;
        case LoopComponent::INFRASTRUCTURE:    return BUDGET_INFRASTRUCTURE_US;
        case LoopComponent::SAFETY_MARGIN:     return TICK_SAFETY_MARGIN_US;
    }
    return 0.0;
}

/// True when `usage_us` is within the allocated budget for `component`.
[[nodiscard]] constexpr bool component_within_budget(double usage_us, LoopComponent c) noexcept {
    return usage_us <= component_budget_us(c);
}

// ---------------------------------------------------------------------------
// § Tick health classification
// ---------------------------------------------------------------------------

/// Classify overall tick duration.
/// NOMINAL   : tick_us < TICK_WARNING_US   (< 950 μs)
/// WARNING   : tick_us < TICK_CRITICAL_US  (950 – 1050 μs)
/// CRITICAL  : tick_us < WATCHDOG_TIMEOUT_US (1050 – 2000 μs)
/// OVERRUN   : tick_us >= WATCHDOG_TIMEOUT_US (≥ 2000 μs)
[[nodiscard]] constexpr TickHealth classify_tick(double tick_us) noexcept {
    if (tick_us <  TICK_WARNING_US)    return TickHealth::NOMINAL;
    if (tick_us <  TICK_CRITICAL_US)   return TickHealth::WARNING;
    if (tick_us <  WATCHDOG_TIMEOUT_US) return TickHealth::CRITICAL;
    return                                    TickHealth::OVERRUN;
}

[[nodiscard]] constexpr bool tick_nominal(double tick_us) noexcept {
    return classify_tick(tick_us) == TickHealth::NOMINAL;
}

[[nodiscard]] constexpr bool tick_warning(double tick_us) noexcept {
    return classify_tick(tick_us) == TickHealth::WARNING;
}

[[nodiscard]] constexpr bool tick_critical(double tick_us) noexcept {
    auto h = classify_tick(tick_us);
    return h == TickHealth::CRITICAL || h == TickHealth::OVERRUN;
}

// ---------------------------------------------------------------------------
// § Label helpers
// ---------------------------------------------------------------------------

[[nodiscard]] constexpr std::string_view loop_component_label(LoopComponent c) noexcept {
    switch (c) {
        case LoopComponent::PHYSICS_KERNEL:    return "PHYSICS_KERNEL";
        case LoopComponent::COGNITIVE_SCANNER: return "COGNITIVE_SCANNER";
        case LoopComponent::ENGS:              return "ENGS";
        case LoopComponent::INFRASTRUCTURE:    return "INFRASTRUCTURE";
        case LoopComponent::SAFETY_MARGIN:     return "SAFETY_MARGIN";
    }
    return "UNKNOWN";
}

[[nodiscard]] constexpr std::string_view tick_health_label(TickHealth h) noexcept {
    switch (h) {
        case TickHealth::NOMINAL:  return "NOMINAL";
        case TickHealth::WARNING:  return "WARNING";
        case TickHealth::CRITICAL: return "CRITICAL";
        case TickHealth::OVERRUN:  return "OVERRUN";
    }
    return "UNKNOWN";
}

[[nodiscard]] constexpr std::string_view physics_sub_label(PhysicsSubComponent c) noexcept {
    switch (c) {
        case PhysicsSubComponent::METRIC_TENSOR_UPDATE: return "METRIC_TENSOR_UPDATE";
        case PhysicsSubComponent::POTENTIAL_STEP:       return "POTENTIAL_STEP";
        case PhysicsSubComponent::KINETIC_STEP_FFT:     return "KINETIC_STEP_FFT";
        case PhysicsSubComponent::NONLINEAR_SOLITON:    return "NONLINEAR_SOLITON";
        case PhysicsSubComponent::BOUNDARY_CONDITIONS:  return "BOUNDARY_CONDITIONS";
    }
    return "UNKNOWN";
}

} // namespace nikola::system
