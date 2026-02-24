/**
 * @file economy/metabolic_calibrator.hpp
 * @brief Phase 57 — GAP-012: Metabolic Cost Calibration via Hardware Benchmarking
 *
 * Grounds the ENGS simulated ATP budget in physical hardware performance.
 * Without calibration "1.0 ATP" is arbitrary; this module auto-derives
 * Nikola Metabolic Units (NMUs) from measured FLOPS, memory bandwidth, and
 * PCIe latency — making the metabolic system hardware-aware.
 *
 * Spec requirements:
 *
 *   Base NMU formula (anchors 1.0 NMU to cost of 1 ms identity maintenance):
 *     NMU = (peak_flops × 10⁻¹²) + (bandwidth_GBs × 10⁻³)
 *
 *   Operation cost taxonomy:
 *     propagation = base_nmu × 0.1   (wave propagation — compute-bound)
 *     plasticity  = base_nmu × 1.5   (neuroplasticity update — memory-bound)
 *     tool_usage  = base_nmu × 5.0   (external tool call — context switch + I/O)
 *
 *   Thermal coupling (GPU temperature multiplier):
 *     M(T) = 1 + max(0, ((T_gpu − T_target) / (T_crit − T_target))²)
 *     T_target ≈ 60 °C, T_crit ≈ 85 °C → forces Nap state near thermal limit
 *
 *   Neurochemical modulation:
 *     C_eff(N) = C_raw / (1 + N_t)   — NE lowers cost during stress ("sprint")
 *     C_eff(S) = C_raw × (1 + S_t)   — S raises cost for impulsive actions
 *
 *   Self-regulation: low-FLOPS device thinks slower and sleeps more—no manual
 *   tuning required.
 *
 * @see §GAP-012 in 01_computational_neurochemistry.md
 * @since Phase 57
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace nikola::economy {

// ── Spec constants ────────────────────────────────────────────────────────────

/// NMU scaling from FLOPS: 1 TFLOP/s = 1.0 NMU contribution
inline constexpr double NMU_FLOPS_SCALE   = 1.0e-12;

/// NMU scaling from bandwidth: 1 GB/s = 0.001 NMU contribution
inline constexpr double NMU_BW_SCALE      = 1.0e-3;

/// Operation cost ratios (spec §GAP-012 taxonomy)
inline constexpr float  COST_RATIO_PROPAGATION = 0.1f;   ///< Wave propagation (compute-bound)
inline constexpr float  COST_RATIO_PLASTICITY  = 1.5f;   ///< Neuroplasticity update (memory-bound)
inline constexpr float  COST_RATIO_TOOL        = 5.0f;   ///< External tool call (I/O + context switch)

/// Thermal coupling: GPU target and critical temperatures (°C)
inline constexpr float  T_TARGET_C             = 60.0f;
inline constexpr float  T_CRIT_C               = 85.0f;

/// Minimum believable NMU (guards against benchmark returning 0)
inline constexpr float  NMU_MIN               = 1e-4f;

// ── HardwareStats ─────────────────────────────────────────────────────────────

/**
 * @brief Raw benchmark measurements from hardware characterisation.
 *
 * Sources:
 *   - peak_flops:        AVX-512 nonary addition loop (10⁹ ops)
 *   - bandwidth_gbs:     Sequential 1 GB memcpy throughput
 *   - pcie_latency_us:   Host↔Device round-trip latency
 */
struct HardwareStats {
    double peak_flops       = 0.0;  ///< Peak FLOPS (floating-point ops/s)
    double bandwidth_gbs    = 0.0;  ///< Memory bandwidth (GB/s)
    double pcie_latency_us  = 0.0;  ///< PCIe / host↔device latency (µs)

    /// True iff at least FLOPS or BW is non-trivially positive
    [[nodiscard]] bool is_valid() const noexcept {
        return (peak_flops > 0.0) && (bandwidth_gbs > 0.0);
    }
};

// ── OperationCosts ────────────────────────────────────────────────────────────

/**
 * @brief Calibrated cost table for the three ENGS operation classes.
 *
 * These are the values the MetabolicController should use for each
 * neuroplastic operation type, in NMU/operation.
 */
struct OperationCosts {
    float base_nmu   = 0.0f;  ///< Fundamental unit: 1 ms identity maintenance
    float propagation = 0.0f; ///< Thinking: base × 0.1
    float plasticity  = 0.0f; ///< Learning: base × 1.5
    float tool_usage  = 0.0f; ///< Acting:   base × 5.0

    [[nodiscard]] bool is_calibrated() const noexcept { return base_nmu > NMU_MIN; }
};

// ── MetabolicCalibrator ───────────────────────────────────────────────────────

/**
 * @brief Derives NMUs from hardware benchmarks and computes operation costs.
 *
 * Usage (production):
 *   MetabolicCalibrator cal;
 *   auto stats = cal.run_bootstrap_benchmark();
 *   auto costs = cal.calibrate(stats);
 *   // costs.propagation / .plasticity / .tool_usage → set on MetabolicController
 *
 * Usage (tests — inject synthetic stats):
 *   HardwareStats hw{1e12, 100.0, 500.0};  // 1 TFLOP/s, 100 GB/s, 500 µs
 *   auto costs = MetabolicCalibrator::compute_costs(hw);
 *
 * Thermal and neurochemical modulation can be applied afterward via the
 * static helpers thermal_multiplier() and effective_cost().
 */
class MetabolicCalibrator {
public:
    // ── Benchmark ─────────────────────────────────────────────────────────────

    /**
     * @brief Run the three-component hardware characterization benchmark.
     *
     * Components:
     *   1. FLOPS  — timed addition loop over a float array (ops/s)
     *   2. BW     — timed sequential 1 MB memcpy (GB/s)
     *   3. Latency — round-trip copy latency (µs, host-only approximation)
     *
     * This is a CPU-only approximation.  In production with CUDA, replace with:
     *   1. cuBLAS SGEMM for GPU FLOPS
     *   2. cudaMemcpy 1 GB for HBM bandwidth
     *   3. cudaMemcpy small buffer for PCIe latency
     *
     * @param flops_iters Number of addition-loop iterations. Tune down for speed.
     */
    [[nodiscard]] static HardwareStats
    run_bootstrap_benchmark(std::size_t flops_iters = 1'000'000) noexcept {
        HardwareStats stats;

        // 1. FLOPS benchmark — timed float addition loop
        {
            constexpr std::size_t OPS_PER_ITER = 8;  // unrolled 8-wide
            std::vector<float> buf(OPS_PER_ITER, 1.0f);
            float acc = 0.0f;

            const auto t0 = std::chrono::steady_clock::now();
            for (std::size_t it = 0; it < flops_iters; ++it) {
                for (float v : buf) acc += v;
            }
            const auto t1 = std::chrono::steady_clock::now();

            const double elapsed_s =
                std::chrono::duration<double>(t1 - t0).count();
            const double total_ops = static_cast<double>(flops_iters) * OPS_PER_ITER;

            // Prevent optimizer removing acc
            if (acc < 0.0f) stats.peak_flops = 0.0;  // never true
            stats.peak_flops = (elapsed_s > 1e-9) ? (total_ops / elapsed_s) : 0.0;
        }

        // 2. Bandwidth benchmark — 1 MB sequential memcpy
        {
            constexpr std::size_t BUF_BYTES = 1u << 20;  // 1 MB
            std::vector<char> src(BUF_BYTES, 'A');
            std::vector<char> dst(BUF_BYTES, 0);

            const auto t0 = std::chrono::steady_clock::now();
            std::memcpy(dst.data(), src.data(), BUF_BYTES);
            const auto t1 = std::chrono::steady_clock::now();

            const double elapsed_s =
                std::chrono::duration<double>(t1 - t0).count();
            const double bytes_copied = static_cast<double>(BUF_BYTES);
            stats.bandwidth_gbs = (elapsed_s > 1e-9)
                ? (bytes_copied / elapsed_s / 1e9)
                : 0.0;
        }

        // 3. Latency — round-trip copy of single float (host approximation)
        {
            constexpr int PING_ROUNDS = 32;
            float val = 1.0f;

            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < PING_ROUNDS; ++i) {
                float tmp = val;
                std::memcpy(&val, &tmp, sizeof(float));
            }
            const auto t1 = std::chrono::steady_clock::now();

            const double elapsed_us =
                std::chrono::duration<double>(t1 - t0).count() * 1e6;
            stats.pcie_latency_us = elapsed_us / PING_ROUNDS;
        }

        return stats;
    }

    // ── Cost Derivation ───────────────────────────────────────────────────────

    /**
     * @brief Derive NMU and operation costs from hardware stats.
     *
     * Spec formula:
     *   base_nmu = (FLOPS × 1e-12) + (BW_GB/s × 1e-3)
     *
     * Then:
     *   propagation = base_nmu × 0.1
     *   plasticity  = base_nmu × 1.5
     *   tool_usage  = base_nmu × 5.0
     */
    [[nodiscard]] static OperationCosts
    compute_costs(const HardwareStats& hw) noexcept {
        if (!hw.is_valid()) {
            // Fallback: unit NMU (CPU-class baseline — better than zero)
            return OperationCosts{
                .base_nmu    = 1.0f,
                .propagation = COST_RATIO_PROPAGATION,
                .plasticity  = COST_RATIO_PLASTICITY,
                .tool_usage  = COST_RATIO_TOOL,
            };
        }

        const float nmu = static_cast<float>(
            hw.peak_flops    * NMU_FLOPS_SCALE +
            hw.bandwidth_gbs * NMU_BW_SCALE
        );
        const float safe_nmu = std::max(nmu, NMU_MIN);

        return OperationCosts{
            .base_nmu    = safe_nmu,
            .propagation = safe_nmu * COST_RATIO_PROPAGATION,
            .plasticity  = safe_nmu * COST_RATIO_PLASTICITY,
            .tool_usage  = safe_nmu * COST_RATIO_TOOL,
        };
    }

    /**
     * @brief Convenience: run benchmark then compute costs in one call.
     */
    [[nodiscard]] static OperationCosts calibrate(std::size_t flops_iters = 1'000'000) noexcept {
        return compute_costs(run_bootstrap_benchmark(flops_iters));
    }

    // ── Thermal Coupling ──────────────────────────────────────────────────────

    /**
     * @brief Thermal cost multiplier M(T) — spec §GAP-012.
     *
     * Formula:
     *   M(T) = 1 + max(0, ((T_gpu − T_target) / (T_crit − T_target))²)
     *
     * Behaviour:
     *   T ≤ T_target  → M = 1.0 (no penalty)
     *   T = T_crit    → M = 2.0 (doubled cost — forces Nap)
     *   T > T_crit    → M > 2.0 (extrapolates, encourages immediate sleep)
     *
     * @param gpu_temp_c  Current GPU temperature (°C)
     */
    [[nodiscard]] static float thermal_multiplier(float gpu_temp_c) noexcept {
        const float range = T_CRIT_C - T_TARGET_C;  // 25 °C
        if (range < 1e-3f) return 1.0f;              // degenerate guard

        const float excess   = gpu_temp_c - T_TARGET_C;
        const float ratio    = excess / range;
        // Spec: max(0, ratio) before squaring — below T_target, ratio is negative
        // → clamped to 0 → no penalty.  Above T_crit, ratio > 1 → M > 2.
        const float clamped  = std::max(0.0f, ratio);

        return 1.0f + clamped * clamped;
    }

    // ── Neurochemical Modulation ──────────────────────────────────────────────

    /**
     * @brief Effective cost with Norepinephrine sprint modulation.
     *
     * Spec: C_eff = C_raw / (1 + N_t)
     * Biological: NE allows the system to "sprint" — compute more for less ATP.
     *
     * @param raw_cost       Unmodulated operation cost
     * @param norepinephrine N_t ∈ [0, 1]
     */
    [[nodiscard]] static float effective_cost_ne(float raw_cost,
                                                   float norepinephrine) noexcept {
        const float n = std::clamp(norepinephrine, 0.0f, 1.0f);
        return raw_cost / (1.0f + n);
    }

    /**
     * @brief Effective cost with Serotonin stability surcharge.
     *
     * Spec: higher-S state raises cost for impulsive actions.
     * Formula: C_eff = C_raw × (1 + S_t)
     * Biological: serotonin promotes stable, careful operation—impulsive
     * shortcuts cost more energy (discourages mania).
     *
     * @param raw_cost   Unmodulated operation cost
     * @param serotonin  S_t ∈ [0, 1]
     */
    [[nodiscard]] static float effective_cost_serotonin(float raw_cost,
                                                          float serotonin) noexcept {
        const float s = std::clamp(serotonin, 0.0f, 1.0f);
        return raw_cost * (1.0f + s);
    }

    /**
     * @brief Combined effective cost: NE sprint + serotonin surcharge.
     *
     * Order matches spec: divide by (1+N) first, then apply S surcharge.
     * A high-NE/low-S state (adrenaline sprint) produces the lowest cost.
     * A low-NE/high-S state (calm deliberation) produces the highest cost.
     *
     * @param raw_cost        Base calibrated cost
     * @param norepinephrine  N_t ∈ [0, 1]
     * @param serotonin       S_t ∈ [0, 1]
     */
    [[nodiscard]] static float effective_cost(float raw_cost,
                                               float norepinephrine,
                                               float serotonin) noexcept {
        return effective_cost_serotonin(
            effective_cost_ne(raw_cost, norepinephrine),
            serotonin);
    }

    /**
     * @brief Apply thermal multiplier and neurochemical modulation together.
     *
     * Full modulated cost = raw × M(T) / (1+N) × (1+S)
     */
    [[nodiscard]] static float full_modulated_cost(float raw_cost,
                                                    float gpu_temp_c,
                                                    float norepinephrine,
                                                    float serotonin) noexcept {
        return effective_cost(raw_cost * thermal_multiplier(gpu_temp_c),
                              norepinephrine, serotonin);
    }
};

} // namespace nikola::economy
