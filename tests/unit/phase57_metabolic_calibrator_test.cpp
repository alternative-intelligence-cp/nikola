/**
 * @file phase57_metabolic_calibrator_test.cpp
 * @brief Phase 57 — GAP-012: Metabolic Cost Calibration via Hardware Benchmarking
 *
 * Spec: docs/info/integration/sections/05_autonomous_systems/
 *       01_computational_neurochemistry.md  §GAP-012
 *
 * Tests all spec formulas for the MetabolicCalibrator:
 *   1. NMU derivation from hardware stats
 *   2. Operation cost taxonomy (propagation / plasticity / tool_usage ratios)
 *   3. Thermal coupling M(T) formula
 *   4. Neurochemical modulation: NE sprint (÷(1+N)), Serotonin surcharge (×(1+S))
 *   5. Self-regulation property (low-FLOPS device costs less in absolute NMU)
 *   6. Bootstrap benchmark sanity (runs on real CPU, returns valid stats)
 *
 * Tests (20 cases, 72 assertions):
 *   §1  – HardwareStats validity guards
 *   §2  – Base NMU formula: exactly (FLOPS×1e-12) + (BW×1e-3)
 *   §3  – Spec example: 1 TFLOP/s + 100 GB/s → NMU = 1.1
 *   §4  – Operation cost ratios from spec taxonomy
 *   §5  – OperationCosts ratio invariants: plasticity/propagation = 15×
 *   §6  – OperationCosts ratio invariants: tool_usage/propagation = 50×
 *   §7  – Fallback for invalid HardwareStats → unit NMU calibration
 *   §8  – Thermal multiplier: T ≤ T_target → M = 1.0 (no penalty)
 *   §9  – Thermal multiplier: T = T_crit (85°C) → M = 2.0
 *   §10 – Thermal multiplier: T = midpoint (72.5°C) → M = 1.25
 *   §11 – Thermal multiplier: T > T_crit → M > 2.0 (extrapolates)
 *   §12 – Thermal monotone: M increases with temperature
 *   §13 – NE sprint: N=0 → no reduction (C_eff = C_raw)
 *   §14 – NE sprint: N=1 → half cost (C_eff = C_raw / 2)
 *   §15 – NE sprint: N clamped at [0,1] (N=2.0 → same as N=1.0)
 *   §16 – Serotonin surcharge: S=0 → no surcharge
 *   §17 – Serotonin surcharge: S=1 → doubled cost
 *   §18 – Combined: high-NE/low-S is cheapest; low-NE/high-S is costliest
 *   §19 – Self-regulation: laptop(0.5T FLOPS + 30 GB/s) < workstation(5T + 300 GB/s)
 *   §20 – Bootstrap benchmark: returns valid HardwareStats (positive FLOPS + BW)
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <nikola/economy/metabolic_calibrator.hpp>

using namespace nikola::economy;
using MC = MetabolicCalibrator;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// ── §1 ── HardwareStats validity ──────────────────────────────────────────────

TEST_CASE("§1 HardwareStats validity guards", "[phase57][gap012][stats]") {
    HardwareStats valid{1e12, 100.0, 500.0};
    CHECK(valid.is_valid());

    HardwareStats zero_flops{0.0, 100.0, 500.0};
    CHECK_FALSE(zero_flops.is_valid());  // FLOPS = 0

    HardwareStats zero_bw{1e12, 0.0, 500.0};
    CHECK_FALSE(zero_bw.is_valid());  // BW = 0

    HardwareStats default_constructed{};
    CHECK_FALSE(default_constructed.is_valid());
}

// ── §2 ── Base NMU formula ────────────────────────────────────────────────────

TEST_CASE("§2 base NMU formula (FLOPS*1e-12) + (BW*1e-3)", "[phase57][gap012][nmu]") {
    // Chosen so each term is a round number
    HardwareStats hw;
    hw.peak_flops    = 2e12;   // 2 TFLOP/s → 2.0 contribution
    hw.bandwidth_gbs = 400.0;  // 400 GB/s  → 0.4 contribution
    hw.pcie_latency_us = 500.0;

    const auto costs = MC::compute_costs(hw);
    const float expected_nmu = static_cast<float>(2e12 * 1e-12 + 400.0 * 1e-3);
    // = 2.0 + 0.4 = 2.4

    CHECK_THAT(costs.base_nmu, WithinAbs(expected_nmu, 1e-4f));
    CHECK(costs.is_calibrated());
}

// ── §3 ── Spec example: 1 TFLOP/s + 100 GB/s → NMU = 1.1 ───────────────────

TEST_CASE("§3 spec example 1TFLOP + 100GBs = NMU 1.1", "[phase57][gap012][nmu]") {
    HardwareStats hw;
    hw.peak_flops    = 1e12;   // 1 TFLOP/s → 1.0
    hw.bandwidth_gbs = 100.0;  // 100 GB/s  → 0.1
    hw.pcie_latency_us = 500.0;

    const auto costs = MC::compute_costs(hw);
    // NMU = 1.0 + 0.1 = 1.1
    CHECK_THAT(costs.base_nmu, WithinAbs(1.1f, 1e-4f));
}

// ── §4 ── Operation cost ratios from spec taxonomy ────────────────────────────

TEST_CASE("§4 operation cost ratios match spec taxonomy", "[phase57][gap012][costs]") {
    HardwareStats hw{1e12, 100.0, 500.0};  // NMU = 1.1
    const auto costs = MC::compute_costs(hw);

    // propagation = NMU × 0.1
    CHECK_THAT(costs.propagation, WithinAbs(costs.base_nmu * COST_RATIO_PROPAGATION, 1e-5f));
    // plasticity  = NMU × 1.5
    CHECK_THAT(costs.plasticity,  WithinAbs(costs.base_nmu * COST_RATIO_PLASTICITY,  1e-5f));
    // tool_usage  = NMU × 5.0
    CHECK_THAT(costs.tool_usage,  WithinAbs(costs.base_nmu * COST_RATIO_TOOL,        1e-5f));

    // Absolute spot-checks for NMU=1.1
    CHECK_THAT(costs.propagation, WithinAbs(0.11f, 1e-4f));
    CHECK_THAT(costs.plasticity,  WithinAbs(1.65f, 1e-4f));
    CHECK_THAT(costs.tool_usage,  WithinAbs(5.50f, 1e-4f));
}

// ── §5 ── plasticity / propagation = 15× ─────────────────────────────────────

TEST_CASE("§5 plasticity is 15x more expensive than propagation", "[phase57][gap012][costs]") {
    // ratio: 1.5 / 0.1 = 15
    HardwareStats hw{2e12, 200.0, 500.0};
    const auto costs = MC::compute_costs(hw);

    CHECK_THAT(costs.plasticity / costs.propagation, WithinAbs(15.0f, 1e-3f));
}

// ── §6 ── tool_usage / propagation = 50× ────────────────────────────────────

TEST_CASE("§6 tool_usage is 50x more expensive than propagation", "[phase57][gap012][costs]") {
    // ratio: 5.0 / 0.1 = 50
    HardwareStats hw{2e12, 200.0, 500.0};
    const auto costs = MC::compute_costs(hw);

    CHECK_THAT(costs.tool_usage / costs.propagation, WithinAbs(50.0f, 1e-3f));
}

// ── §7 ── Fallback for invalid HardwareStats → unit NMU ──────────────────────

TEST_CASE("§7 invalid HardwareStats falls back to unit NMU", "[phase57][gap012][fallback]") {
    HardwareStats invalid{};  // all zeros
    const auto costs = MC::compute_costs(invalid);

    // Fallback: base_nmu = 1.0, ratios preserved
    CHECK_THAT(costs.base_nmu,    WithinAbs(1.0f, 1e-5f));
    CHECK_THAT(costs.propagation, WithinAbs(COST_RATIO_PROPAGATION, 1e-5f));
    CHECK_THAT(costs.plasticity,  WithinAbs(COST_RATIO_PLASTICITY,  1e-5f));
    CHECK_THAT(costs.tool_usage,  WithinAbs(COST_RATIO_TOOL,        1e-5f));
    CHECK(costs.is_calibrated());
}

// ── §8 ── Thermal: T ≤ T_target → M = 1.0 ────────────────────────────────────

TEST_CASE("§8 thermal_multiplier cold temperatures give M=1.0", "[phase57][gap012][thermal]") {
    // At or below target temperature — no penalty
    CHECK_THAT(MC::thermal_multiplier(20.0f),  WithinAbs(1.0f, 1e-5f));
    CHECK_THAT(MC::thermal_multiplier(59.9f),  WithinAbs(1.0f, 1e-5f));
    CHECK_THAT(MC::thermal_multiplier(60.0f),  WithinAbs(1.0f, 1e-5f));  // exactly at target
}

// ── §9 ── Thermal: T = T_crit → M = 2.0 ─────────────────────────────────────

TEST_CASE("§9 thermal_multiplier at T_crit = 2.0", "[phase57][gap012][thermal]") {
    // At T_crit: ratio = 1.0, ratio² = 1.0, M = 1.0 + 1.0 = 2.0
    CHECK_THAT(MC::thermal_multiplier(T_CRIT_C), WithinAbs(2.0f, 1e-5f));
    CHECK_THAT(MC::thermal_multiplier(85.0f),    WithinAbs(2.0f, 1e-5f));
}

// ── §10 ── Thermal: T = 72.5°C (midpoint) → M = 1.25 ─────────────────────────

TEST_CASE("§10 thermal_multiplier midpoint T=72.5 gives M=1.25", "[phase57][gap012][thermal]") {
    // T_mid = 60 + (85-60)/2 = 72.5°C
    // ratio = (72.5-60)/(85-60) = 12.5/25 = 0.5
    // M = 1 + 0.5² = 1.25
    CHECK_THAT(MC::thermal_multiplier(72.5f), WithinAbs(1.25f, 1e-4f));
}

// ── §11 ── Thermal: T > T_crit → M > 2.0 ────────────────────────────────────

TEST_CASE("§11 thermal_multiplier T > T_crit extrapolates above 2.0", "[phase57][gap012][thermal]") {
    // T = 95°C: ratio = (95-60)/25 = 1.4, M = 1 + 1.96 = 2.96
    const float m95 = MC::thermal_multiplier(95.0f);
    CHECK(m95 > 2.0f);
    CHECK_THAT(m95, WithinAbs(2.96f, 1e-3f));
}

// ── §12 ── Thermal multiplier is monotone ─────────────────────────────────────

TEST_CASE("§12 thermal_multiplier monotone increasing above T_target", "[phase57][gap012][thermal]") {
    float prev = MC::thermal_multiplier(60.0f);
    for (int t = 61; t <= 100; ++t) {
        const float m = MC::thermal_multiplier(static_cast<float>(t));
        CHECK(m >= prev);
        prev = m;
    }
}

// ── §13 ── NE sprint: N=0 → no reduction ─────────────────────────────────────

TEST_CASE("§13 effective_cost_ne N=0 no reduction", "[phase57][gap012][ne]") {
    constexpr float RAW = 3.0f;
    CHECK_THAT(MC::effective_cost_ne(RAW, 0.0f), WithinAbs(RAW, 1e-5f));
}

// ── §14 ── NE sprint: N=1 → half cost ────────────────────────────────────────

TEST_CASE("§14 effective_cost_ne N=1 halves cost", "[phase57][gap012][ne]") {
    constexpr float RAW = 3.0f;
    CHECK_THAT(MC::effective_cost_ne(RAW, 1.0f), WithinAbs(RAW / 2.0f, 1e-5f));
}

// ── §15 ── NE clamped at [0,1] ────────────────────────────────────────────────

TEST_CASE("§15 effective_cost_ne clamps N to [0,1]", "[phase57][gap012][ne]") {
    constexpr float RAW = 4.0f;
    // N=2.0 must give same result as N=1.0 (clamped)
    CHECK_THAT(MC::effective_cost_ne(RAW, 2.0f),
               WithinAbs(MC::effective_cost_ne(RAW, 1.0f), 1e-5f));
    // N=-0.5 must give same result as N=0.0 (clamped)
    CHECK_THAT(MC::effective_cost_ne(RAW, -0.5f),
               WithinAbs(MC::effective_cost_ne(RAW, 0.0f), 1e-5f));
}

// ── §16 ── Serotonin surcharge: S=0 → no surcharge ───────────────────────────

TEST_CASE("§16 effective_cost_serotonin S=0 no surcharge", "[phase57][gap012][serotonin]") {
    constexpr float RAW = 2.5f;
    CHECK_THAT(MC::effective_cost_serotonin(RAW, 0.0f), WithinAbs(RAW, 1e-5f));
}

// ── §17 ── Serotonin surcharge: S=1 → doubled cost ───────────────────────────

TEST_CASE("§17 effective_cost_serotonin S=1 doubles cost", "[phase57][gap012][serotonin]") {
    constexpr float RAW = 2.5f;
    CHECK_THAT(MC::effective_cost_serotonin(RAW, 1.0f), WithinAbs(RAW * 2.0f, 1e-5f));
}

// ── §18 ── Combined: high-NE/low-S cheapest, low-NE/high-S costliest ─────────

TEST_CASE("§18 combined effective_cost high-NE/low-S cheapest", "[phase57][gap012][combined]") {
    constexpr float RAW = 5.0f;

    // Sprint: N=1, S=0 → 5/2 × 1 = 2.5
    const float sprint   = MC::effective_cost(RAW, 1.0f, 0.0f);
    // Normal: N=0.5, S=0.5 → 5/1.5 × 1.5 = 5.0
    const float normal   = MC::effective_cost(RAW, 0.5f, 0.5f);
    // Cautious: N=0, S=1 → 5/1 × 2 = 10.0
    const float cautious = MC::effective_cost(RAW, 0.0f, 1.0f);

    CHECK(sprint < normal);
    CHECK(normal < cautious);

    CHECK_THAT(sprint,   WithinAbs(2.5f, 1e-4f));
    CHECK_THAT(cautious, WithinAbs(10.0f, 1e-4f));
}

// ── §19 ── Self-regulation: laptop < workstation in absolute NMU costs ────────

TEST_CASE("§19 self-regulation low-FLOPS costs less per NMU than high-FLOPS", "[phase57][gap012][selfregulation]") {
    // Laptop-class: 0.5 TFLOP/s + 30 GB/s
    HardwareStats laptop{5e11, 30.0, 100.0};
    const float nmu_laptop = MC::compute_costs(laptop).base_nmu;

    // Workstation: 5 TFLOP/s + 300 GB/s
    HardwareStats workstation{5e12, 300.0, 500.0};
    const float nmu_ws = MC::compute_costs(workstation).base_nmu;

    // Data-center: 100 TFLOP/s + 3000 GB/s (H100-class)
    HardwareStats datacenter{1e14, 3000.0, 200.0};
    const float nmu_dc = MC::compute_costs(datacenter).base_nmu;

    // NMU scales with hardware: laptop < workstation < datacenter
    CHECK(nmu_laptop < nmu_ws);
    CHECK(nmu_ws < nmu_dc);

    // Spot check laptop NMU: 0.5×1e12 × 1e-12 + 30×1e-3 = 0.5 + 0.03 = 0.53
    CHECK_THAT(nmu_laptop, WithinAbs(0.53f, 1e-3f));
    // Workstation: 5×1e12 × 1e-12 + 300×1e-3 = 5.0 + 0.3 = 5.3
    CHECK_THAT(nmu_ws, WithinAbs(5.3f, 1e-3f));
}

// ── §20 ── Bootstrap benchmark: returns valid HardwareStats ───────────────────

TEST_CASE("§20 run_bootstrap_benchmark returns valid positive stats", "[phase57][gap012][benchmark]") {
    // Use minimal iteration count so the test is fast
    const HardwareStats stats = MC::run_bootstrap_benchmark(10'000);

    // Must return positive values (real hardware, not infinities or zeros)
    CHECK(stats.peak_flops > 0.0);
    CHECK(stats.bandwidth_gbs > 0.0);
    CHECK(stats.pcie_latency_us >= 0.0);
    CHECK(stats.is_valid());

    // Derived costs must be calibrated
    const auto costs = MC::compute_costs(stats);
    CHECK(costs.is_calibrated());
    CHECK(costs.propagation > 0.0f);
    CHECK(costs.plasticity  > 0.0f);
    CHECK(costs.tool_usage  > 0.0f);

    // Ratio invariants hold on real hardware too
    CHECK_THAT(costs.plasticity  / costs.propagation, WithinAbs(15.0f, 1e-2f));
    CHECK_THAT(costs.tool_usage  / costs.propagation, WithinAbs(50.0f, 1e-2f));
}
