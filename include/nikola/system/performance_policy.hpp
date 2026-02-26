#pragma once
// =============================================================================
// nikola/system/performance_policy.hpp
// Phase 83 — GAP-043: Performance Tuning Cookbook
//
// SOURCE: Gemini Deep Research Round 2, Batch 41-44 (December 16, 2025)
// SPEC:   docs/info/integration/sections/10_appendices/04_hardware_optimization.md
//         §GAP-043 (lines ~546–665)
//
// Spec-authoritative performance targets, tuning-knob ranges, benchmark
// baselines, and predicate helpers that encode Nikola's "Phase 0" mandate:
//
//   • Memory-bound architecture — Data Locality above all
//   • Physics loop at 1000 Hz (Δt = 1 ms)
//   • Laplacian precision ~10⁻⁷ (Kahan summation)
//   • Energy drift < 0.01 % (symplectic integrator)
//   • L1/L2 cache hit rate target ≥ 95 %
//
// All constants are pure constexpr, header-only, no external deps.
// =============================================================================

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <string_view>

namespace nikola::system {

// ---------------------------------------------------------------------------
// § Enumerations
// ---------------------------------------------------------------------------

/// Hardware deployment profile controlling physics rate and precision.
enum class HardwareProfile : uint8_t {
    CPU_ONLY          = 0,  ///< AVX-512 only, physics at 200 Hz (dt=5ms)
    SINGLE_GPU        = 1,  ///< RTX 4090 class, FP32, ~14 M active nodes
    MULTI_GPU_CLUSTER = 2,  ///< A100/H100 NVLink, FP64, >100 M nodes
};

/// Diagnostic category assigned during latency analysis flowchart.
/// §"Scenario A: System Latency is High (>100ms response)"
enum class BottleneckType : uint8_t {
    NONE             = 0,  ///< Latency within spec
    CACHE_MISS       = 1,  ///< L1/L2 miss rate > 10% — check SoA alignment
    ZMQ_BACKPRESSURE = 2,  ///< ZMQ HWM reached — throttle physics
    SHM_OVERHEAD     = 3,  ///< shm_unlink lagging — reduce segment size
};

/// Action index during energy-divergence diagnosis flowchart.
/// §"Scenario B: Energy Divergence (Hallucinations/Crashes)"
enum class EnergyDivergenceAction : uint8_t {
    NONE              = 0,  ///< Energy within limits
    REDUCE_TIMESTEP   = 1,  ///< Halve physics_dt immediately (> 0.01% drift)
    CHECK_INTEGRATOR  = 2,  ///< Verify Split-Operator symplectic method
    CHECK_KAHAN       = 3,  ///< Verify Kahan summation active for Laplacian
    CHECK_DOPAMINE    = 4,  ///< Dopamine pinned at 1 or 0 → gating failure
};

/// Benchmark pass/fail classification relative to spec baselines.
enum class BenchmarkStatus : uint8_t {
    WITHIN_TARGET = 0,  ///< Metric ≤ spec target
    ABOVE_TARGET  = 1,  ///< Metric > target but ≤ failure threshold
    CRITICAL      = 2,  ///< Metric > failure threshold → P0 failure
};

/// Recommended tuning direction for a knob.
enum class TuningDirection : uint8_t {
    HOLD     = 0,  ///< Parameter is within optimal range
    INCREASE = 1,  ///< Knob should be increased
    DECREASE = 2,  ///< Knob should be decreased
};

// ---------------------------------------------------------------------------
// § Physics loop constants
// ---------------------------------------------------------------------------

/// Default integration timestep in seconds.
/// Spec: "physics_dt = 1ms" (range 0.1ms – 5ms)
inline constexpr double PHYSICS_DT_DEFAULT_S   = 1.0e-3;

/// Default timestep in milliseconds (convenience form).
inline constexpr double PHYSICS_DT_DEFAULT_MS  = 1.0;

/// Minimum allowed timestep in milliseconds.
inline constexpr double PHYSICS_DT_MIN_MS      = 0.1;

/// Maximum allowed timestep in milliseconds.
inline constexpr double PHYSICS_DT_MAX_MS      = 5.0;

/// Target physics loop rate in Hz (at default timestep).
inline constexpr double PHYSICS_LOOP_HZ        = 1000.0;      // 1 / 1ms

/// Physics loop period budget in microseconds.
inline constexpr double PHYSICS_BUDGET_US      = 1000.0;      // 1 / 1000 Hz × 1e6

/// Reduced physics rate for CPU-only profile.
/// Spec: "Profile 1: CPU-Only → physics_dt=5ms, 200Hz"
inline constexpr double PHYSICS_LOOP_HZ_CPU    = 200.0;

// ---------------------------------------------------------------------------
// § Tuning-knob ranges
// ---------------------------------------------------------------------------

/// Hebbian learning rate: controls metric tensor warping speed.
/// Spec: "hebbian_rate: Default=0.01, Range=0.001–0.1"
inline constexpr double HEBBIAN_RATE_DEFAULT   = 0.01;
inline constexpr double HEBBIAN_RATE_MIN       = 0.001;
inline constexpr double HEBBIAN_RATE_MAX       = 0.1;

/// Metabolic cost of long-term memory plasticity.
/// Spec: "metabolic_cost_plasticity: Default=1.5, Range=1.0–5.0"
inline constexpr double METABOLIC_COST_PLASTICITY_DEFAULT = 1.5;
inline constexpr double METABOLIC_COST_PLASTICITY_MIN     = 1.0;
inline constexpr double METABOLIC_COST_PLASTICITY_MAX     = 5.0;

/// ATP fraction that triggers a Nap (consolidation).
/// Spec: "nap_interval_trigger: Default=15%, Range=5%–30%"
inline constexpr double NAP_TRIGGER_DEFAULT    = 0.15;
inline constexpr double NAP_TRIGGER_MIN        = 0.05;
inline constexpr double NAP_TRIGGER_MAX        = 0.30;

/// Dither noise amplitude injected to prevent Resonance Lock-in.
/// Spec: "dither_amplitude: Default=1e-4, Range=1e-5–1e-3"
inline constexpr double DITHER_AMPLITUDE_DEFAULT = 1.0e-4;
inline constexpr double DITHER_AMPLITUDE_MIN     = 1.0e-5;
inline constexpr double DITHER_AMPLITUDE_MAX     = 1.0e-3;

/// Signal-to-Noise Ratio threshold in dB.
/// Spec: "Decrease [dither] if SNR drops below 20dB"
inline constexpr double SNR_DB_THRESHOLD       = 20.0;

// ---------------------------------------------------------------------------
// § Cache performance targets
// ---------------------------------------------------------------------------

/// Target L1/L2 cache hit rate (fraction, not percentage).
/// Spec: "~95% hit rate"
inline constexpr double CACHE_HIT_TARGET       = 0.95;

/// Failure threshold: hit rate below this is a P0 regression.
/// Spec: "< 85% → indicates AoS regression"
inline constexpr double CACHE_HIT_FAILURE      = 0.85;

/// Cache miss rate above which SoA layout should be investigated.
/// Spec: "Miss Rate > 10% → Verify Structure-of-Arrays (SoA) alignment"
inline constexpr double CACHE_MISS_THRESHOLD   = 0.10;

// ---------------------------------------------------------------------------
// § Laplacian numerical precision
// ---------------------------------------------------------------------------

/// Target Laplacian error with Kahan summation active.
/// Spec: "Laplacian Accuracy (Kahan): Baseline Target = Error ~10⁻⁷"
inline constexpr double LAPLACIAN_PRECISION_TARGET = 1.0e-7;

/// Failure threshold: error above this indicates Kahan summation failure.
/// Spec: "Failure Threshold > 10⁻⁵"
inline constexpr double LAPLACIAN_PRECISION_LIMIT  = 1.0e-5;

// ---------------------------------------------------------------------------
// § Energy conservation limits
// ---------------------------------------------------------------------------

/// Maximum acceptable relative energy drift per normalised iteration.
/// Spec: "Energy Drift: 24-hour Stability Test Baseline < 0.01%"
inline constexpr double ENERGY_DRIFT_TARGET    = 0.0001;   // 0.01%

/// Critical failure threshold.
/// Spec: "Failure Threshold > 0.05%"  (Scenario B triggers reduce_timestep)
inline constexpr double ENERGY_DRIFT_CRITICAL  = 0.0005;   // 0.05%

// ---------------------------------------------------------------------------
// § Physics latency benchmarks (single-step time)
// ---------------------------------------------------------------------------

/// Target step latency for 81³ = 531 441 node grid (ms).
/// Spec: "BM_WavePropagation_81^3: Baseline Target = 7.8 ms/step"
inline constexpr double BM_WAVE_81_TARGET_MS   = 7.8;

/// Failure threshold for 81³ grid (ms).
/// Spec: "Failure Threshold > 12 ms"
inline constexpr double BM_WAVE_81_FAIL_MS     = 12.0;

/// Target step latency for 27³ = 19 683 node grid (ms).
/// Spec: "BM_WavePropagation_27^3: Baseline Target = 0.48 ms (Critical P0)"
inline constexpr double BM_WAVE_27_TARGET_MS   = 0.48;

/// Failure threshold for 27³ grid (ms).
/// Spec: "Failure Threshold > 1 ms (Critical P0 requirement)"
inline constexpr double BM_WAVE_27_FAIL_MS     = 1.0;

// ---------------------------------------------------------------------------
// § Grid geometry constants
// ---------------------------------------------------------------------------

/// Base of the balanced nonary grid (3^N topology).
inline constexpr int    GRID_BASE              = 3;

/// Number of spatial dimensions in the 9D Torus.
inline constexpr int    GRID_DIM               = 9;

/// Default block size: 3^9 nodes per block (matches GRID_BASE^GRID_DIM).
/// Spec: "block_size: Default=19683 (= 3^9)"
inline constexpr int    BLOCK_SIZE_DEFAULT     = 19683;     // 3^9

/// Minimum block size (3^3 = 27, one 3D face).
inline constexpr int    BLOCK_SIZE_MIN         = 27;         // 3^3

// ---------------------------------------------------------------------------
// § CUDA launch overhead constants
// ---------------------------------------------------------------------------

/// Number of sequential kernel launches per symplectic step.
/// Spec: "Split-Operator: Kinetic → Potential → Nonlinear → Damping = 5-6 kernels"
inline constexpr int    KERNEL_LAUNCHES_PER_STEP = 6;

/// Typical kernel-launch overhead in microseconds (driver + PCIe).
/// Spec: "Driver Overhead 5–20 μs; PCIe Latency 2–5 μs" → combined ≈ 15 μs
inline constexpr double KERNEL_OVERHEAD_US      = 15.0;

/// Total launch overhead per timestep with 6 kernels (μs).
/// Spec: "Total Overhead ≈ 6 × 15 μs = 90 μs"
inline constexpr double TOTAL_LAUNCH_OVERHEAD_US = 90.0;    // 6 × 15

/// Fraction of the 1000 μs budget consumed by launch overhead.
/// Spec: "consumes nearly 10% of 1000 μs budget"
inline constexpr double LAUNCH_OVERHEAD_FRACTION =
    TOTAL_LAUNCH_OVERHEAD_US / PHYSICS_BUDGET_US;             // 0.09

/// Temporal decoherence threshold: step budget can be breached above this.
/// Spec: "Temporal Decoherence threshold (500 μs)"
inline constexpr double TEMPORAL_DECOHERENCE_US  = 500.0;

/// CUDA Graph launch overhead (replaces 6-kernel overhead with 1 call).
/// Spec: "Launch overhead reduces from 6×15μs to 1×5μs"
inline constexpr double GRAPH_LAUNCH_OVERHEAD_US = 5.0;

/// Neurogenesis graph re-capture cost.
/// Spec: "Re-instantiation: ~200 μs; must occur in Plasticity Windows only"
inline constexpr double NEUROGENESIS_RECAPTURE_US = 200.0;

/// Maximum VRAM-limited node count on a single RTX 4090 (24 GB).
/// Spec: "Grid size limited to ~14M active nodes due to 24GB VRAM"
inline constexpr int64_t RTX4090_MAX_NODES       = 14'000'000LL;

/// CUDA thread-block size for single-GPU profile (Profile 2).
/// Spec: "CUDA_BLOCK_SIZE = 256"
inline constexpr int     CUDA_BLOCK_SIZE_DEFAULT = 256;

// ---------------------------------------------------------------------------
// § Tuning-knob predicates: range checks
// ---------------------------------------------------------------------------

/// True when `dt_ms` is within the documented safe range.
[[nodiscard]] constexpr bool timestep_within_range(double dt_ms) noexcept {
    return dt_ms >= PHYSICS_DT_MIN_MS && dt_ms <= PHYSICS_DT_MAX_MS;
}

/// True when the Hebbian learning rate is within its spec range.
[[nodiscard]] constexpr bool hebbian_rate_in_range(double rate) noexcept {
    return rate >= HEBBIAN_RATE_MIN && rate <= HEBBIAN_RATE_MAX;
}

/// True when metabolic cost of plasticity is within its spec range.
[[nodiscard]] constexpr bool metabolic_cost_in_range(double cost) noexcept {
    return cost >= METABOLIC_COST_PLASTICITY_MIN
        && cost <= METABOLIC_COST_PLASTICITY_MAX;
}

/// True when the Nap trigger fraction is within spec.
[[nodiscard]] constexpr bool nap_trigger_in_range(double frac) noexcept {
    return frac >= NAP_TRIGGER_MIN && frac <= NAP_TRIGGER_MAX;
}

/// True when dither amplitude is within its documented operating range.
[[nodiscard]] constexpr bool dither_amplitude_in_range(double amp) noexcept {
    return amp >= DITHER_AMPLITUDE_MIN && amp <= DITHER_AMPLITUDE_MAX;
}

/// True when SNR is acceptable (no need to reduce dithering).
[[nodiscard]] constexpr bool snr_db_acceptable(double snr_db) noexcept {
    return snr_db >= SNR_DB_THRESHOLD;
}

// ---------------------------------------------------------------------------
// § Cache performance predicates
// ---------------------------------------------------------------------------

/// True when cache hit rate meets or exceeds the 95% target.
[[nodiscard]] constexpr bool cache_hit_at_target(double hit_rate) noexcept {
    return hit_rate >= CACHE_HIT_TARGET;
}

/// True when hit rate has fallen below the P0 failure threshold.
[[nodiscard]] constexpr bool cache_hit_critical(double hit_rate) noexcept {
    return hit_rate < CACHE_HIT_FAILURE;
}

/// True when the miss rate implies a SoA alignment problem.
[[nodiscard]] constexpr bool cache_miss_bottleneck(double miss_rate) noexcept {
    return miss_rate > CACHE_MISS_THRESHOLD;
}

/// Cache miss rate derived from hit rate.
[[nodiscard]] constexpr double miss_rate(double hit_rate) noexcept {
    return 1.0 - hit_rate;
}

// ---------------------------------------------------------------------------
// § Laplacian precision predicates
// ---------------------------------------------------------------------------

/// True when Laplacian error is at or below the Kahan target.
[[nodiscard]] constexpr bool laplacian_precision_ok(double error) noexcept {
    return error <= LAPLACIAN_PRECISION_TARGET;
}

/// True when Laplacian error represents a Kahan summation failure (P0).
[[nodiscard]] constexpr bool laplacian_precision_critical(double error) noexcept {
    return error > LAPLACIAN_PRECISION_LIMIT;
}

// ---------------------------------------------------------------------------
// § Energy conservation predicates
// ---------------------------------------------------------------------------

/// True when drift is within the 24-hour stability target.
[[nodiscard]] constexpr bool energy_drift_ok(double drift_fraction) noexcept {
    return drift_fraction <= ENERGY_DRIFT_TARGET;
}

/// True when drift triggers immediate action (halve dt, Scenario B step 1).
[[nodiscard]] constexpr bool energy_drift_critical(double drift_fraction) noexcept {
    return drift_fraction > ENERGY_DRIFT_CRITICAL;
}

/// Return the recommended action for the given energy drift.
[[nodiscard]] constexpr EnergyDivergenceAction
energy_divergence_action(double drift_fraction) noexcept
{
    if (drift_fraction <= ENERGY_DRIFT_TARGET)  return EnergyDivergenceAction::NONE;
    if (drift_fraction <= ENERGY_DRIFT_CRITICAL) return EnergyDivergenceAction::REDUCE_TIMESTEP;
    // Above critical: cascading check
    return EnergyDivergenceAction::REDUCE_TIMESTEP;
}

// ---------------------------------------------------------------------------
// § Physics latency benchmarks
// ---------------------------------------------------------------------------

/// Classify a measured step latency against spec baselines.
/// grid_size must be 27 (small grid) or 81 (large grid).
[[nodiscard]] constexpr BenchmarkStatus
classify_latency(double measured_ms, int grid_size) noexcept
{
    double target  = (grid_size == 81) ? BM_WAVE_81_TARGET_MS : BM_WAVE_27_TARGET_MS;
    double failure = (grid_size == 81) ? BM_WAVE_81_FAIL_MS   : BM_WAVE_27_FAIL_MS;
    if (measured_ms <= target)  return BenchmarkStatus::WITHIN_TARGET;
    if (measured_ms <= failure) return BenchmarkStatus::ABOVE_TARGET;
    return                             BenchmarkStatus::CRITICAL;
}

/// True when step latency is at/below the spec baseline target.
[[nodiscard]] constexpr bool latency_within_target(double ms, int grid_size) noexcept {
    return classify_latency(ms, grid_size) == BenchmarkStatus::WITHIN_TARGET;
}

/// True when latency exceeds the P0 failure threshold.
[[nodiscard]] constexpr bool latency_critical(double ms, int grid_size) noexcept {
    return classify_latency(ms, grid_size) == BenchmarkStatus::CRITICAL;
}

// ---------------------------------------------------------------------------
// § CUDA overhead budget analysis
// ---------------------------------------------------------------------------

/// Fraction of the physics budget consumed by launch overhead.
[[nodiscard]] constexpr double overhead_budget_fraction(
    double overhead_us, double step_budget_us = PHYSICS_BUDGET_US) noexcept
{
    if (step_budget_us <= 0.0) return 0.0;
    return overhead_us / step_budget_us;
}

/// True when overhead is within the 10% budget limit.
[[nodiscard]] constexpr bool overhead_within_budget(
    double overhead_us, double step_budget_us = PHYSICS_BUDGET_US) noexcept
{
    return overhead_budget_fraction(overhead_us, step_budget_us) <= LAUNCH_OVERHEAD_FRACTION + 0.01;
}

/// True when a single-step duration is at risk of Temporal Decoherence.
[[nodiscard]] constexpr bool decoherence_risk(double step_actual_us) noexcept {
    return step_actual_us >= TEMPORAL_DECOHERENCE_US;
}

/// Total launch overhead for N kernel launches using standard (non-Graph) path.
[[nodiscard]] constexpr double standard_launch_overhead_us(int n_kernels) noexcept {
    return n_kernels * KERNEL_OVERHEAD_US;
}

/// Overhead saving from switching to CUDA Graph (single launch call).
[[nodiscard]] constexpr double graph_overhead_saving_us(int n_kernels) noexcept {
    return standard_launch_overhead_us(n_kernels) - GRAPH_LAUNCH_OVERHEAD_US;
}

// ---------------------------------------------------------------------------
// § Grid geometry helpers
// ---------------------------------------------------------------------------

/// Node count for a full cubic block at scale N: N^GRID_DIM.
/// Valid compact sizes: 3, 9, 27, 81.  Returns 0 for invalid sizes.
[[nodiscard]] constexpr int64_t block_node_count(int grid_size) noexcept {
    // Only support powers of 3 up to 81
    int64_t count = 1LL;
    for (int i = 0; i < GRID_DIM; ++i) count *= grid_size;
    return count;
}

/// True when the block size is a valid power-of-3 grid dimension.
[[nodiscard]] constexpr bool is_valid_block_size(int block_size) noexcept {
    if (block_size <= 0) return false;
    int n = block_size;
    while (n > 1) {
        if (n % GRID_BASE != 0) return false;
        n /= GRID_BASE;
    }
    return true;
}

/// True when node_count is within the RTX 4090 VRAM limit.
[[nodiscard]] constexpr bool fits_rtx4090(int64_t node_count) noexcept {
    return node_count <= RTX4090_MAX_NODES;
}

// ---------------------------------------------------------------------------
// § Tuning direction advisors
// ---------------------------------------------------------------------------

/// Advise whether to increase/decrease/hold the Hebbian rate.
/// Rationale: increase for stagnant learning; decrease for instability.
[[nodiscard]] constexpr TuningDirection
advise_hebbian_rate(bool is_manic_switching, bool is_stagnant) noexcept
{
    if (is_manic_switching) return TuningDirection::DECREASE;
    if (is_stagnant)        return TuningDirection::INCREASE;
    return                         TuningDirection::HOLD;
}

/// Advise NAP trigger adjustment based on stability vs task complexity.
[[nodiscard]] constexpr TuningDirection
advise_nap_trigger(bool needs_more_stability, bool needs_longer_wake) noexcept
{
    if (needs_more_stability) return TuningDirection::INCREASE;  // more frequent, shorter naps
    if (needs_longer_wake)    return TuningDirection::DECREASE;
    return                           TuningDirection::HOLD;
}

// ---------------------------------------------------------------------------
// § Label helpers
// ---------------------------------------------------------------------------

[[nodiscard]] constexpr std::string_view hardware_profile_label(
    HardwareProfile p) noexcept
{
    switch (p) {
        case HardwareProfile::CPU_ONLY:          return "CPU_ONLY";
        case HardwareProfile::SINGLE_GPU:        return "SINGLE_GPU";
        case HardwareProfile::MULTI_GPU_CLUSTER: return "MULTI_GPU_CLUSTER";
    }
    return "UNKNOWN_PROFILE";
}

[[nodiscard]] constexpr std::string_view bottleneck_label(BottleneckType b) noexcept {
    switch (b) {
        case BottleneckType::NONE:             return "NONE";
        case BottleneckType::CACHE_MISS:       return "CACHE_MISS";
        case BottleneckType::ZMQ_BACKPRESSURE: return "ZMQ_BACKPRESSURE";
        case BottleneckType::SHM_OVERHEAD:     return "SHM_OVERHEAD";
    }
    return "UNKNOWN_BOTTLENECK";
}

[[nodiscard]] constexpr std::string_view energy_divergence_action_label(
    EnergyDivergenceAction a) noexcept
{
    switch (a) {
        case EnergyDivergenceAction::NONE:             return "NONE";
        case EnergyDivergenceAction::REDUCE_TIMESTEP:  return "REDUCE_TIMESTEP";
        case EnergyDivergenceAction::CHECK_INTEGRATOR: return "CHECK_INTEGRATOR";
        case EnergyDivergenceAction::CHECK_KAHAN:      return "CHECK_KAHAN";
        case EnergyDivergenceAction::CHECK_DOPAMINE:   return "CHECK_DOPAMINE";
    }
    return "UNKNOWN_ACTION";
}

[[nodiscard]] constexpr std::string_view benchmark_status_label(
    BenchmarkStatus s) noexcept
{
    switch (s) {
        case BenchmarkStatus::WITHIN_TARGET: return "WITHIN_TARGET";
        case BenchmarkStatus::ABOVE_TARGET:  return "ABOVE_TARGET";
        case BenchmarkStatus::CRITICAL:      return "CRITICAL";
    }
    return "UNKNOWN_STATUS";
}

[[nodiscard]] constexpr std::string_view tuning_direction_label(
    TuningDirection d) noexcept
{
    switch (d) {
        case TuningDirection::HOLD:     return "HOLD";
        case TuningDirection::INCREASE: return "INCREASE";
        case TuningDirection::DECREASE: return "DECREASE";
    }
    return "UNKNOWN_DIRECTION";
}

} // namespace nikola::system
