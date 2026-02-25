#pragma once

// ============================================================
// perf_tuning.hpp — GAP-043: Performance Tuning Cookbook
//
// Documents the Nikola v0.0.4 "Phase 0" performance tuning mandates:
//   • Optimization philosophy: Memory-Bound, not compute-bound
//   • Operator knob defaults, ranges, and naming constants
//   • Diagnostic thresholds (cache miss rate, latency, SNR, drift)
//   • Benchmark baseline targets and failure thresholds
//   • Hardware deployment profiles (CPU-only, RTX 4090, A100 cluster)
//   • Grid-size constants for the 9D toroidal manifold
//
// Key insight: Nikola's bottleneck is moving 9D grid state between
// VRAM and Compute Units (bandwidth-bound), NOT Matrix Multiplication
// FLOPS. All tuning centres on Data Locality, Cache Efficiency, and
// Bandwidth Saturation.
//
// Namespace:   nikola::infrastructure
// Standard:    C++23, header-only, no external dependencies
// Source spec: GAP-043 — Performance Tuning Cookbook
//              (Gemini Deep Research Round 2, Batch 41-44)
// ============================================================

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace nikola::infrastructure {

// ============================================================
// §1  Optimization Focus Areas ("Phase 0" Mandate)
// ============================================================

/// Primary axis along which a tuning change is expected to deliver gains.
enum class OptimizationFocus : uint8_t {
    DATA_LOCALITY         = 0u,  ///< Cluster hot nodes in memory for spatial reuse.
    CACHE_EFFICIENCY      = 1u,  ///< Maximise L1/L2 hit rate via SoA + Hilbert ordering.
    BANDWIDTH_SATURATION  = 2u,  ///< Saturate available VRAM ↔ Compute bandwidth.
    MEMORY_BOUND_TUNING   = 3u,  ///< General memory-bound path (not FLOPS-bound).
};

/// Number of defined optimisation focus areas.
inline constexpr std::size_t OPTIMIZATION_FOCUS_COUNT = 4u;

// ============================================================
// §2  Hardware Deployment Profiles
// ============================================================

/// Target hardware tier for a Nikola deployment.
enum class HardwareProfile : uint8_t {
    CPU_ONLY     = 0u,  ///< Intel Core i9 / Xeon / AMD Ryzen 9 — AVX-512 MANDATORY.
    SINGLE_GPU   = 1u,  ///< Single NVIDIA RTX 4090 (24 GB VRAM) — FP32, CUDA.
    MULTI_GPU    = 2u,  ///< 4× or 8× NVIDIA A100 (80 GB) with NVLink — FP64 optional.
};

/// Number of defined hardware profiles.
inline constexpr std::size_t HARDWARE_PROFILE_COUNT = 3u;

// ============================================================
// §3  Grid Size Constants
// ============================================================

/// Exponent base for 9D Torus block sizing: $3^n$.
inline constexpr uint32_t GRID_BASE = 3u;

/// Number of spatial dimensions in the toroidal manifold.
inline constexpr uint32_t GRID_DIMENSIONS = 9u;

/// Default block size: $3^9 = 19{,}683$ nodes per block.
/// Fixed at compile time; changing requires recompilation.
/// Must align with $3^9$ for efficient Torus mapping.
inline constexpr uint32_t BLOCK_SIZE_DEFAULT = 19'683u;  // 3^9

/// Small benchmark grid edge length (nodes per dimension): 27.
/// Benchmark: BM_WavePropagation_27^3  → 27×27×27 = 19 683 nodes.
inline constexpr uint32_t SMALL_GRID_EDGE = 27u;

/// Small benchmark grid total node count: $27^3 = 19{,}683$.
inline constexpr uint32_t SMALL_GRID_NODES = 19'683u;

/// Large benchmark grid edge length (nodes per dimension): 81.
/// Benchmark: BM_WavePropagation_81^3  → 81×81×81 = 531 441 nodes.
inline constexpr uint32_t LARGE_GRID_EDGE = 81u;

/// Large benchmark grid total node count: $81^3 = 531{,}441$.
inline constexpr uint32_t LARGE_GRID_NODES = 531'441u;

/// SoA memory alignment (bytes) required for AVX-512 vectorisation.
inline constexpr uint32_t SOA_ALIGNMENT_BYTES = 64u;

// ============================================================
// §4  GPU-Specific Tuning Constants
// ============================================================

/// CUDA thread-block size for the RTX 4090 (single-GPU) profile.
inline constexpr uint32_t CUDA_BLOCK_SIZE_RTX4090 = 256u;

/// Maximum active nodes on a single RTX 4090 (24 GB VRAM).
/// Approx. 14 million nodes.
inline constexpr uint32_t MAX_ACTIVE_NODES_RTX4090 = 14'000'000u;

/// Minimum active nodes scale for a multi-GPU A100 cluster.
/// A100 cluster can exceed 100 million active nodes.
inline constexpr uint32_t MIN_ACTIVE_NODES_A100_CLUSTER = 100'000'000u;

// ============================================================
// §5  Operator Tuning Knob Names and Defaults
// ============================================================

/// Config key for the Hebbian learning rate η.
inline constexpr std::string_view KNOB_HEBBIAN_RATE = "hebbian_rate";

/// Config key for the metabolic plasticity cost per memory write.
inline constexpr std::string_view KNOB_METABOLIC_COST = "metabolic_cost_plasticity";

/// Config key for the ATP threshold (fraction) that triggers a Nap cycle.
inline constexpr std::string_view KNOB_NAP_TRIGGER = "nap_interval_trigger";

/// Config key for the physics integration timestep Δt.
inline constexpr std::string_view KNOB_PHYSICS_DT = "physics_dt";

/// Config key for the grid block size (fixed at compile time).
inline constexpr std::string_view KNOB_BLOCK_SIZE = "block_size";

/// Config key for the thermal dither noise amplitude injected into the field.
inline constexpr std::string_view KNOB_DITHER_AMPLITUDE = "dither_amplitude";

/// Total number of operator-tunable knobs.
inline constexpr std::size_t KNOB_COUNT = 6u;

// ---- Defaults -------------------------------------------------------

/// Default Hebbian learning rate η — controls metric tensor warping speed.
inline constexpr double HEBBIAN_RATE_DEFAULT = 0.01;

/// Default metabolic plasticity cost — cost to write to long-term memory.
inline constexpr double METABOLIC_COST_DEFAULT = 1.5;

/// Default ATP Nap trigger (fraction of budget): 15%.
inline constexpr double NAP_TRIGGER_DEFAULT = 0.15;

/// Default physics integration timestep: 1 ms.
inline constexpr double PHYSICS_DT_DEFAULT_MS = 1.0;

/// Default dither noise amplitude: 1×10⁻⁴.
inline constexpr double DITHER_AMPLITUDE_DEFAULT = 1.0e-4;

// ---- Range Bounds —--------------------------------------------------

/// Minimum Hebbian learning rate (conservative — prevents manic switching).
inline constexpr double HEBBIAN_RATE_MIN = 0.001;

/// Maximum Hebbian learning rate (aggressive learning).
inline constexpr double HEBBIAN_RATE_MAX = 0.1;

/// Minimum metabolic plasticity cost.
inline constexpr double METABOLIC_COST_MIN = 1.0;

/// Maximum metabolic plasticity cost (forces high-resonance-only writes).
inline constexpr double METABOLIC_COST_MAX = 5.0;

/// Minimum Nap trigger fraction: 5% (longer wake periods; complex tasks).
inline constexpr double NAP_TRIGGER_MIN = 0.05;

/// Maximum Nap trigger fraction: 30% (frequent short naps; max stability).
inline constexpr double NAP_TRIGGER_MAX = 0.30;

/// Minimum physics timestep (high resolution): 0.1 ms.
inline constexpr double PHYSICS_DT_MIN_MS = 0.1;

/// Maximum physics timestep (CPU-only profile): 5 ms → 200 Hz physics.
inline constexpr double PHYSICS_DT_MAX_MS = 5.0;

/// CPU-only physics frequency when dt = 5 ms: 200 Hz.
inline constexpr uint32_t CPU_PHYSICS_HZ = 200u;

/// GPU physics frequency when dt = 1 ms: 1 000 Hz.
inline constexpr uint32_t GPU_PHYSICS_HZ = 1'000u;

/// Minimum dither noise amplitude (low noise floor).
inline constexpr double DITHER_AMPLITUDE_MIN = 1.0e-5;

/// Maximum dither noise amplitude (prevents resonance lock-in).
inline constexpr double DITHER_AMPLITUDE_MAX = 1.0e-3;

// ============================================================
// §6  Diagnostic Thresholds
// ============================================================

/// Physics tick duration above which the system is flagged as "high latency".
/// Unit: milliseconds. Triggers cache-miss / AVX-512 diagnostics.
inline constexpr double TICK_LATENCY_HIGH_MS = 1.0;

/// L1/L2 cache miss rate above which SoA alignment / Hilbert indexing must
/// be verified (Scenario A, step 1).
/// Unit: fraction (10 % = 0.10).
inline constexpr double CACHE_MISS_RATE_THRESHOLD = 0.10;

/// Signal-to-Noise Ratio minimum before dither amplitude should be reduced.
/// Unit: decibels.
inline constexpr double SNR_MIN_DB = 20.0;

/// Energy drift fraction above which the "energy divergence" diagnostic path
/// is entered (mirrors GAP-042 PHY-002 condition).
/// Ratio: 0.01 % → 1×10⁻⁴.
inline constexpr double ENERGY_DRIFT_DIAG_THRESHOLD = 1.0e-4;

// ============================================================
// §7  Benchmark Baseline Targets and Failure Thresholds
// ============================================================

/// Large-grid (81×81×81) wave-propagation step latency target.
/// Unit: milliseconds.
inline constexpr double BM_LARGE_GRID_TARGET_MS = 7.8;

/// Large-grid step latency failure threshold.
/// Unit: milliseconds (exceed → performance regression).
inline constexpr double BM_LARGE_GRID_FAIL_MS = 12.0;

/// Small-grid (27×27×27) wave-propagation step latency target.
/// Unit: milliseconds. This is a Critical P0 requirement.
inline constexpr double BM_SMALL_GRID_TARGET_MS = 0.48;

/// Small-grid step latency failure threshold (Critical P0).
/// Unit: milliseconds.
inline constexpr double BM_SMALL_GRID_FAIL_MS = 1.0;

/// Memory bandwidth utilisation target: 100% (complete SoA efficiency).
inline constexpr double BM_BANDWIDTH_UTIL_TARGET = 1.0;

/// Memory bandwidth utilisation failure threshold: 80%.
/// Below this indicates AoS (Array-of-Structs regression).
inline constexpr double BM_BANDWIDTH_UTIL_FAIL = 0.80;

/// L1/L2 cache hit rate target: 95%.
inline constexpr double BM_CACHE_HIT_TARGET = 0.95;

/// Cache hit rate failure threshold: 85%.
inline constexpr double BM_CACHE_HIT_FAIL = 0.85;

/// Laplacian computation precision (Kahan summation) target error.
inline constexpr double BM_LAPLACIAN_PRECISION_TARGET = 1.0e-7;

/// Laplacian precision failure threshold. Exceeding this indicates
/// Kahan summation is disabled or broken.
inline constexpr double BM_LAPLACIAN_PRECISION_FAIL = 1.0e-5;

/// 24-hour stability energy drift target: <0.01 % → <1×10⁻⁴.
inline constexpr double BM_ENERGY_DRIFT_TARGET = 1.0e-4;

/// 24-hour stability energy drift failure threshold: >0.05 % → >5×10⁻⁴.
inline constexpr double BM_ENERGY_DRIFT_FAIL = 5.0e-4;

/// Total number of defined benchmark metrics.
inline constexpr std::size_t BENCHMARK_METRIC_COUNT = 6u;

// ============================================================
// §8  Query Functions
// ============================================================

/// Returns the default physics timestep (ms) for the given hardware profile.
[[nodiscard]] constexpr double default_physics_dt_ms(HardwareProfile profile) noexcept {
    switch (profile) {
        case HardwareProfile::CPU_ONLY:   return PHYSICS_DT_MAX_MS;   // 5 ms → 200 Hz
        case HardwareProfile::SINGLE_GPU: return PHYSICS_DT_DEFAULT_MS; // 1 ms → 1 kHz
        case HardwareProfile::MULTI_GPU:  return PHYSICS_DT_DEFAULT_MS; // 1 ms → 1 kHz
    }
    return PHYSICS_DT_DEFAULT_MS;
}

/// Returns the physics loop frequency (Hz) for the given hardware profile.
[[nodiscard]] constexpr uint32_t physics_frequency_hz(HardwareProfile profile) noexcept {
    switch (profile) {
        case HardwareProfile::CPU_ONLY:   return CPU_PHYSICS_HZ;
        case HardwareProfile::SINGLE_GPU: return GPU_PHYSICS_HZ;
        case HardwareProfile::MULTI_GPU:  return GPU_PHYSICS_HZ;
    }
    return GPU_PHYSICS_HZ;
}

/// Returns true when CUDA should be enabled for the given profile.
[[nodiscard]] constexpr bool requires_cuda(HardwareProfile profile) noexcept {
    return profile != HardwareProfile::CPU_ONLY;
}

/// Returns true when distributed sharding (MPI/NCCL) is required.
[[nodiscard]] constexpr bool requires_distributed_sharding(HardwareProfile profile) noexcept {
    return profile == HardwareProfile::MULTI_GPU;
}

/// Returns true when FP64 precision is natively recommended for the profile.
[[nodiscard]] constexpr bool prefers_fp64(HardwareProfile profile) noexcept {
    return profile == HardwareProfile::MULTI_GPU;
}

/// Returns the human-readable label for a HardwareProfile.
[[nodiscard]] constexpr std::string_view hardware_profile_name(HardwareProfile profile) noexcept {
    switch (profile) {
        case HardwareProfile::CPU_ONLY:   return "cpu_only";
        case HardwareProfile::SINGLE_GPU: return "single_gpu_rtx4090";
        case HardwareProfile::MULTI_GPU:  return "multi_gpu_a100";
    }
    return "";
}

/// Returns the human-readable label for an OptimizationFocus.
[[nodiscard]] constexpr std::string_view optimization_focus_name(OptimizationFocus focus) noexcept {
    switch (focus) {
        case OptimizationFocus::DATA_LOCALITY:        return "data_locality";
        case OptimizationFocus::CACHE_EFFICIENCY:     return "cache_efficiency";
        case OptimizationFocus::BANDWIDTH_SATURATION: return "bandwidth_saturation";
        case OptimizationFocus::MEMORY_BOUND_TUNING:  return "memory_bound_tuning";
    }
    return "";
}

/// Returns true when a physics timestep (ms) is within the valid range.
[[nodiscard]] constexpr bool is_valid_physics_dt(double dt_ms) noexcept {
    return dt_ms >= PHYSICS_DT_MIN_MS && dt_ms <= PHYSICS_DT_MAX_MS;
}

/// Returns true when a Hebbian learning rate is within the valid range.
[[nodiscard]] constexpr bool is_valid_hebbian_rate(double rate) noexcept {
    return rate >= HEBBIAN_RATE_MIN && rate <= HEBBIAN_RATE_MAX;
}

/// Returns true when a metabolic plasticity cost is within the valid range.
[[nodiscard]] constexpr bool is_valid_metabolic_cost(double cost) noexcept {
    return cost >= METABOLIC_COST_MIN && cost <= METABOLIC_COST_MAX;
}

/// Returns true when a Nap trigger fraction is within the valid range.
[[nodiscard]] constexpr bool is_valid_nap_trigger(double fraction) noexcept {
    return fraction >= NAP_TRIGGER_MIN && fraction <= NAP_TRIGGER_MAX;
}

/// Returns true when a dither amplitude is within the valid range.
[[nodiscard]] constexpr bool is_valid_dither_amplitude(double amplitude) noexcept {
    return amplitude >= DITHER_AMPLITUDE_MIN && amplitude <= DITHER_AMPLITUDE_MAX;
}

/// Returns true when the L1/L2 cache miss rate exceeds the diagnostic threshold.
/// Triggers SoA alignment / Hilbert curve verification (Scenario A, step 1).
[[nodiscard]] constexpr bool is_cache_miss_alarm(double miss_rate) noexcept {
    return miss_rate > CACHE_MISS_RATE_THRESHOLD;
}

/// Returns true when a physics tick duration indicates a high-latency condition.
[[nodiscard]] constexpr bool is_high_latency_tick(double tick_ms) noexcept {
    return tick_ms > TICK_LATENCY_HIGH_MS;
}

/// Returns true when the large-grid benchmark exceeds its failure threshold.
[[nodiscard]] constexpr bool is_large_grid_benchmark_fail(double latency_ms) noexcept {
    return latency_ms > BM_LARGE_GRID_FAIL_MS;
}

/// Returns true when the small-grid (P0 critical) benchmark exceeds its failure threshold.
[[nodiscard]] constexpr bool is_small_grid_benchmark_fail(double latency_ms) noexcept {
    return latency_ms > BM_SMALL_GRID_FAIL_MS;
}

/// Returns true when memory bandwidth utilisation has regressed below the failure threshold.
/// Indicates an Array-of-Structs (AoS) layout regression.
[[nodiscard]] constexpr bool is_bandwidth_regression(double utilisation) noexcept {
    return utilisation < BM_BANDWIDTH_UTIL_FAIL;
}

/// Returns true when L1/L2 cache hit rate has fallen below the failure threshold.
[[nodiscard]] constexpr bool is_cache_hit_fail(double hit_rate) noexcept {
    return hit_rate < BM_CACHE_HIT_FAIL;
}

/// Returns true when the Laplacian error exceeds the Kahan summation failure threshold.
[[nodiscard]] constexpr bool is_laplacian_precision_fail(double error) noexcept {
    return error > BM_LAPLACIAN_PRECISION_FAIL;
}

/// Returns true when 24-hour energy drift exceeds the stability failure threshold.
[[nodiscard]] constexpr bool is_energy_drift_fail(double drift_ratio) noexcept {
    return drift_ratio > BM_ENERGY_DRIFT_FAIL;
}

}  // namespace nikola::infrastructure
