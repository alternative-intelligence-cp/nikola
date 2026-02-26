// =============================================================================
// tests/unit/phase83_performance_policy_test.cpp
// Phase 83 — GAP-043: Performance Tuning Cookbook
//
// Tests for nikola::system::performance_policy.hpp
// Spec: docs/info/integration/sections/10_appendices/04_hardware_optimization.md
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "nikola/system/performance_policy.hpp"

using namespace nikola::system;
using Catch::Approx;

// ---------------------------------------------------------------------------
// § Enum: HardwareProfile
// ---------------------------------------------------------------------------

TEST_CASE("HardwareProfile enum values are distinct and ordered", "[enums][phase83]") {
    CHECK(static_cast<int>(HardwareProfile::CPU_ONLY)          == 0);
    CHECK(static_cast<int>(HardwareProfile::SINGLE_GPU)        == 1);
    CHECK(static_cast<int>(HardwareProfile::MULTI_GPU_CLUSTER) == 2);
}

TEST_CASE("hardware_profile_label returns correct strings", "[enums][phase83]") {
    CHECK(hardware_profile_label(HardwareProfile::CPU_ONLY)          == "CPU_ONLY");
    CHECK(hardware_profile_label(HardwareProfile::SINGLE_GPU)        == "SINGLE_GPU");
    CHECK(hardware_profile_label(HardwareProfile::MULTI_GPU_CLUSTER) == "MULTI_GPU_CLUSTER");
}

// ---------------------------------------------------------------------------
// § Enum: BottleneckType
// ---------------------------------------------------------------------------

TEST_CASE("BottleneckType enum values are distinct", "[enums][phase83]") {
    CHECK(static_cast<int>(BottleneckType::NONE)             == 0);
    CHECK(static_cast<int>(BottleneckType::CACHE_MISS)       == 1);
    CHECK(static_cast<int>(BottleneckType::ZMQ_BACKPRESSURE) == 2);
    CHECK(static_cast<int>(BottleneckType::SHM_OVERHEAD)     == 3);
}

TEST_CASE("bottleneck_label returns correct strings", "[enums][phase83]") {
    CHECK(bottleneck_label(BottleneckType::NONE)             == "NONE");
    CHECK(bottleneck_label(BottleneckType::CACHE_MISS)       == "CACHE_MISS");
    CHECK(bottleneck_label(BottleneckType::ZMQ_BACKPRESSURE) == "ZMQ_BACKPRESSURE");
    CHECK(bottleneck_label(BottleneckType::SHM_OVERHEAD)     == "SHM_OVERHEAD");
}

// ---------------------------------------------------------------------------
// § Enum: EnergyDivergenceAction
// ---------------------------------------------------------------------------

TEST_CASE("EnergyDivergenceAction enum values are distinct", "[enums][phase83]") {
    CHECK(static_cast<int>(EnergyDivergenceAction::NONE)             == 0);
    CHECK(static_cast<int>(EnergyDivergenceAction::REDUCE_TIMESTEP)  == 1);
    CHECK(static_cast<int>(EnergyDivergenceAction::CHECK_INTEGRATOR) == 2);
    CHECK(static_cast<int>(EnergyDivergenceAction::CHECK_KAHAN)      == 3);
    CHECK(static_cast<int>(EnergyDivergenceAction::CHECK_DOPAMINE)   == 4);
}

TEST_CASE("energy_divergence_action_label returns correct strings", "[enums][phase83]") {
    CHECK(energy_divergence_action_label(EnergyDivergenceAction::NONE)             == "NONE");
    CHECK(energy_divergence_action_label(EnergyDivergenceAction::REDUCE_TIMESTEP)  == "REDUCE_TIMESTEP");
    CHECK(energy_divergence_action_label(EnergyDivergenceAction::CHECK_INTEGRATOR) == "CHECK_INTEGRATOR");
    CHECK(energy_divergence_action_label(EnergyDivergenceAction::CHECK_KAHAN)      == "CHECK_KAHAN");
    CHECK(energy_divergence_action_label(EnergyDivergenceAction::CHECK_DOPAMINE)   == "CHECK_DOPAMINE");
}

// ---------------------------------------------------------------------------
// § Enum: BenchmarkStatus
// ---------------------------------------------------------------------------

TEST_CASE("BenchmarkStatus enum values are distinct and ordered", "[enums][phase83]") {
    CHECK(static_cast<int>(BenchmarkStatus::WITHIN_TARGET) == 0);
    CHECK(static_cast<int>(BenchmarkStatus::ABOVE_TARGET)  == 1);
    CHECK(static_cast<int>(BenchmarkStatus::CRITICAL)      == 2);
}

TEST_CASE("benchmark_status_label returns correct strings", "[enums][phase83]") {
    CHECK(benchmark_status_label(BenchmarkStatus::WITHIN_TARGET) == "WITHIN_TARGET");
    CHECK(benchmark_status_label(BenchmarkStatus::ABOVE_TARGET)  == "ABOVE_TARGET");
    CHECK(benchmark_status_label(BenchmarkStatus::CRITICAL)      == "CRITICAL");
}

// ---------------------------------------------------------------------------
// § Enum: TuningDirection
// ---------------------------------------------------------------------------

TEST_CASE("TuningDirection enum values are distinct", "[enums][phase83]") {
    CHECK(static_cast<int>(TuningDirection::HOLD)     == 0);
    CHECK(static_cast<int>(TuningDirection::INCREASE) == 1);
    CHECK(static_cast<int>(TuningDirection::DECREASE) == 2);
}

TEST_CASE("tuning_direction_label returns correct strings", "[enums][phase83]") {
    CHECK(tuning_direction_label(TuningDirection::HOLD)     == "HOLD");
    CHECK(tuning_direction_label(TuningDirection::INCREASE) == "INCREASE");
    CHECK(tuning_direction_label(TuningDirection::DECREASE) == "DECREASE");
}

// ---------------------------------------------------------------------------
// § Physics loop constants
// ---------------------------------------------------------------------------

TEST_CASE("Physics loop constants have correct values", "[constants][physics][phase83]") {
    CHECK(PHYSICS_DT_DEFAULT_MS  == Approx(1.0));
    CHECK(PHYSICS_DT_MIN_MS      == Approx(0.1));
    CHECK(PHYSICS_DT_MAX_MS      == Approx(5.0));
    CHECK(PHYSICS_LOOP_HZ        == Approx(1000.0));
    CHECK(PHYSICS_LOOP_HZ_CPU    == Approx(200.0));
    CHECK(PHYSICS_BUDGET_US      == Approx(1000.0)); // 1/1000Hz × 1e6
}

TEST_CASE("Physics DT in seconds equals DT_MS / 1000", "[constants][physics][phase83]") {
    CHECK(PHYSICS_DT_DEFAULT_S == Approx(PHYSICS_DT_DEFAULT_MS * 1.0e-3));
}

TEST_CASE("Physics loop Hz and budget are reciprocals", "[constants][physics][phase83]") {
    // 1 / Hz × 1e6 μs == BUDGET_US
    CHECK(PHYSICS_BUDGET_US == Approx(1.0e6 / PHYSICS_LOOP_HZ));
}

TEST_CASE("CPU physics Hz is 1/5th of full-rate Hz", "[constants][physics][phase83]") {
    // CPU at 200 Hz → dt = 5ms = PHYSICS_DT_MAX_MS
    CHECK(1.0 / PHYSICS_LOOP_HZ_CPU * 1000.0 == Approx(PHYSICS_DT_MAX_MS));
}

// ---------------------------------------------------------------------------
// § Tuning knob range constants
// ---------------------------------------------------------------------------

TEST_CASE("Hebbian rate constants satisfy ordering: min < default < max", "[constants][knobs][phase83]") {
    CHECK(HEBBIAN_RATE_MIN     == Approx(0.001));
    CHECK(HEBBIAN_RATE_DEFAULT == Approx(0.01));
    CHECK(HEBBIAN_RATE_MAX     == Approx(0.1));
    REQUIRE(HEBBIAN_RATE_MIN < HEBBIAN_RATE_DEFAULT);
    REQUIRE(HEBBIAN_RATE_DEFAULT < HEBBIAN_RATE_MAX);
}

TEST_CASE("Hebbian rate default is geometric centre of range", "[constants][knobs][phase83]") {
    // 0.001 × 100 = 0.1, geometric mean ≈ 0.01
    double geom_mean = HEBBIAN_RATE_MIN * HEBBIAN_RATE_MAX; // = 0.0001
    CHECK(HEBBIAN_RATE_DEFAULT * HEBBIAN_RATE_DEFAULT == Approx(geom_mean).epsilon(1e-9));
}

TEST_CASE("Metabolic cost constants satisfy ordering", "[constants][knobs][phase83]") {
    CHECK(METABOLIC_COST_PLASTICITY_MIN     == Approx(1.0));
    CHECK(METABOLIC_COST_PLASTICITY_DEFAULT == Approx(1.5));
    CHECK(METABOLIC_COST_PLASTICITY_MAX     == Approx(5.0));
    REQUIRE(METABOLIC_COST_PLASTICITY_MIN < METABOLIC_COST_PLASTICITY_DEFAULT);
    REQUIRE(METABOLIC_COST_PLASTICITY_DEFAULT < METABOLIC_COST_PLASTICITY_MAX);
}

TEST_CASE("NAP trigger constants satisfy ordering", "[constants][knobs][phase83]") {
    CHECK(NAP_TRIGGER_MIN     == Approx(0.05));
    CHECK(NAP_TRIGGER_DEFAULT == Approx(0.15));
    CHECK(NAP_TRIGGER_MAX     == Approx(0.30));
    REQUIRE(NAP_TRIGGER_MIN < NAP_TRIGGER_DEFAULT);
    REQUIRE(NAP_TRIGGER_DEFAULT < NAP_TRIGGER_MAX);
}

TEST_CASE("Dither amplitude constants satisfy ordering", "[constants][knobs][phase83]") {
    CHECK(DITHER_AMPLITUDE_MIN     == Approx(1.0e-5));
    CHECK(DITHER_AMPLITUDE_DEFAULT == Approx(1.0e-4));
    CHECK(DITHER_AMPLITUDE_MAX     == Approx(1.0e-3));
    REQUIRE(DITHER_AMPLITUDE_MIN < DITHER_AMPLITUDE_DEFAULT);
    REQUIRE(DITHER_AMPLITUDE_DEFAULT < DITHER_AMPLITUDE_MAX);
}

TEST_CASE("SNR threshold is 20 dB", "[constants][knobs][phase83]") {
    CHECK(SNR_DB_THRESHOLD == Approx(20.0));
}

// ---------------------------------------------------------------------------
// § Cache constants
// ---------------------------------------------------------------------------

TEST_CASE("Cache hit constants satisfy ordering", "[constants][cache][phase83]") {
    CHECK(CACHE_HIT_TARGET      == Approx(0.95));
    CHECK(CACHE_HIT_FAILURE     == Approx(0.85));
    CHECK(CACHE_MISS_THRESHOLD  == Approx(0.10));
    REQUIRE(CACHE_HIT_FAILURE < CACHE_HIT_TARGET);
}

TEST_CASE("Cache miss threshold plus hit failure sums to ~0.95", "[constants][cache][phase83]") {
    // miss_threshold 10% corresponds to success rate 90%, which is between 85% and 95%
    CHECK(1.0 - CACHE_MISS_THRESHOLD == Approx(0.90));
}

// ---------------------------------------------------------------------------
// § Laplacian constants
// ---------------------------------------------------------------------------

TEST_CASE("Laplacian precision constants satisfy ordering", "[constants][laplacian][phase83]") {
    CHECK(LAPLACIAN_PRECISION_TARGET == Approx(1.0e-7));
    CHECK(LAPLACIAN_PRECISION_LIMIT  == Approx(1.0e-5));
    REQUIRE(LAPLACIAN_PRECISION_TARGET < LAPLACIAN_PRECISION_LIMIT);
}

TEST_CASE("Laplacian limit is 100× coarser than target", "[constants][laplacian][phase83]") {
    CHECK(LAPLACIAN_PRECISION_LIMIT / LAPLACIAN_PRECISION_TARGET == Approx(100.0));
}

// ---------------------------------------------------------------------------
// § Energy drift constants
// ---------------------------------------------------------------------------

TEST_CASE("Energy drift constants satisfy ordering", "[constants][energy][phase83]") {
    CHECK(ENERGY_DRIFT_TARGET   == Approx(0.0001));
    CHECK(ENERGY_DRIFT_CRITICAL == Approx(0.0005));
    REQUIRE(ENERGY_DRIFT_TARGET < ENERGY_DRIFT_CRITICAL);
}

TEST_CASE("Energy drift critical is 5× the target", "[constants][energy][phase83]") {
    CHECK(ENERGY_DRIFT_CRITICAL / ENERGY_DRIFT_TARGET == Approx(5.0));
}

// ---------------------------------------------------------------------------
// § Benchmark constants
// ---------------------------------------------------------------------------

TEST_CASE("Wave-81 benchmark constants satisfy ordering", "[constants][bench][phase83]") {
    CHECK(BM_WAVE_81_TARGET_MS == Approx(7.8));
    CHECK(BM_WAVE_81_FAIL_MS   == Approx(12.0));
    REQUIRE(BM_WAVE_81_TARGET_MS < BM_WAVE_81_FAIL_MS);
}

TEST_CASE("Wave-27 benchmark constants satisfy ordering", "[constants][bench][phase83]") {
    CHECK(BM_WAVE_27_TARGET_MS == Approx(0.48));
    CHECK(BM_WAVE_27_FAIL_MS   == Approx(1.0));
    REQUIRE(BM_WAVE_27_TARGET_MS < BM_WAVE_27_FAIL_MS);
}

TEST_CASE("Wave-81 target is ~16× wave-27 target (grid volume scaling)", "[constants][bench][phase83]") {
    // 81/27 = 3 per dimension, 9D → 3^9 = 19683× nodes.
    // Latency scales sub-linearly; ratio is empirical ~7.8/0.48 ≈ 16.25
    CHECK(BM_WAVE_81_TARGET_MS / BM_WAVE_27_TARGET_MS > 10.0);
    CHECK(BM_WAVE_81_TARGET_MS / BM_WAVE_27_TARGET_MS < 20.0);
}

// ---------------------------------------------------------------------------
// § Grid geometry constants
// ---------------------------------------------------------------------------

TEST_CASE("Grid geometry constants have correct values", "[constants][grid][phase83]") {
    CHECK(GRID_BASE          == 3);
    CHECK(GRID_DIM           == 9);
    CHECK(BLOCK_SIZE_DEFAULT == 19683); // 3^9
    CHECK(BLOCK_SIZE_MIN     == 27);    // 3^3
}

TEST_CASE("BLOCK_SIZE_DEFAULT equals GRID_BASE^GRID_DIM", "[constants][grid][phase83]") {
    int expected = 1;
    for (int i = 0; i < GRID_DIM; ++i) expected *= GRID_BASE;
    CHECK(BLOCK_SIZE_DEFAULT == expected);
}

// ---------------------------------------------------------------------------
// § CUDA overhead constants
// ---------------------------------------------------------------------------

TEST_CASE("CUDA overhead constants have correct values", "[constants][cuda][phase83]") {
    CHECK(KERNEL_LAUNCHES_PER_STEP   == 6);
    CHECK(KERNEL_OVERHEAD_US         == Approx(15.0));
    CHECK(TOTAL_LAUNCH_OVERHEAD_US   == Approx(90.0));
    CHECK(TEMPORAL_DECOHERENCE_US    == Approx(500.0));
    CHECK(GRAPH_LAUNCH_OVERHEAD_US   == Approx(5.0));
    CHECK(NEUROGENESIS_RECAPTURE_US  == Approx(200.0));
    CHECK(CUDA_BLOCK_SIZE_DEFAULT    == 256);
}

TEST_CASE("TOTAL_LAUNCH_OVERHEAD_US equals 6 × KERNEL_OVERHEAD_US", "[constants][cuda][phase83]") {
    CHECK(TOTAL_LAUNCH_OVERHEAD_US ==
          Approx(KERNEL_LAUNCHES_PER_STEP * KERNEL_OVERHEAD_US));
}

TEST_CASE("Launch overhead fraction is approximately 9%", "[constants][cuda][phase83]") {
    // 90 μs / 1000 μs = 0.09 (≈ 10% per spec)
    CHECK(LAUNCH_OVERHEAD_FRACTION == Approx(0.09).epsilon(0.001));
}

TEST_CASE("RTX 4090 max nodes is 14 million", "[constants][cuda][phase83]") {
    CHECK(RTX4090_MAX_NODES == 14'000'000LL);
}

// ---------------------------------------------------------------------------
// § timestep_within_range
// ---------------------------------------------------------------------------

TEST_CASE("timestep_within_range: boundaries", "[predicates][timestep][phase83]") {
    CHECK( timestep_within_range(0.1));   // min boundary
    CHECK( timestep_within_range(1.0));   // default
    CHECK( timestep_within_range(5.0));   // max boundary
    CHECK(!timestep_within_range(0.09));  // just below min
    CHECK(!timestep_within_range(5.01));  // just above max
    CHECK(!timestep_within_range(0.0));   // zero
    CHECK(!timestep_within_range(-1.0));  // negative
}

// ---------------------------------------------------------------------------
// § hebbian_rate_in_range
// ---------------------------------------------------------------------------

TEST_CASE("hebbian_rate_in_range: boundaries and out-of-range", "[predicates][knobs][phase83]") {
    CHECK( hebbian_rate_in_range(0.001));  // min boundary
    CHECK( hebbian_rate_in_range(0.01));   // default
    CHECK( hebbian_rate_in_range(0.1));    // max boundary
    CHECK(!hebbian_rate_in_range(0.0009)); // below min
    CHECK(!hebbian_rate_in_range(0.11));   // above max
    CHECK(!hebbian_rate_in_range(0.0));
    CHECK(!hebbian_rate_in_range(-0.05));
}

// ---------------------------------------------------------------------------
// § metabolic_cost_in_range
// ---------------------------------------------------------------------------

TEST_CASE("metabolic_cost_in_range: boundaries", "[predicates][knobs][phase83]") {
    CHECK( metabolic_cost_in_range(1.0));  // min
    CHECK( metabolic_cost_in_range(1.5));  // default
    CHECK( metabolic_cost_in_range(5.0));  // max
    CHECK(!metabolic_cost_in_range(0.99));
    CHECK(!metabolic_cost_in_range(5.01));
}

// ---------------------------------------------------------------------------
// § nap_trigger_in_range
// ---------------------------------------------------------------------------

TEST_CASE("nap_trigger_in_range: boundaries", "[predicates][knobs][phase83]") {
    CHECK( nap_trigger_in_range(0.05));   // min
    CHECK( nap_trigger_in_range(0.15));   // default
    CHECK( nap_trigger_in_range(0.30));   // max
    CHECK(!nap_trigger_in_range(0.04));
    CHECK(!nap_trigger_in_range(0.31));
}

// ---------------------------------------------------------------------------
// § dither_amplitude_in_range
// ---------------------------------------------------------------------------

TEST_CASE("dither_amplitude_in_range: boundaries", "[predicates][knobs][phase83]") {
    CHECK( dither_amplitude_in_range(1.0e-5));  // min
    CHECK( dither_amplitude_in_range(1.0e-4));  // default
    CHECK( dither_amplitude_in_range(1.0e-3));  // max
    CHECK(!dither_amplitude_in_range(9.0e-6));  // below min
    CHECK(!dither_amplitude_in_range(1.1e-3));  // above max
}

// ---------------------------------------------------------------------------
// § snr_db_acceptable
// ---------------------------------------------------------------------------

TEST_CASE("snr_db_acceptable: threshold boundary", "[predicates][snr][phase83]") {
    CHECK( snr_db_acceptable(20.0));   // at threshold
    CHECK( snr_db_acceptable(25.0));   // above
    CHECK(!snr_db_acceptable(19.9));   // below
    CHECK(!snr_db_acceptable(0.0));
}

// ---------------------------------------------------------------------------
// § Cache predicates
// ---------------------------------------------------------------------------

TEST_CASE("cache_hit_at_target: 95% target boundary", "[predicates][cache][phase83]") {
    CHECK( cache_hit_at_target(0.95));
    CHECK( cache_hit_at_target(0.99));
    CHECK( cache_hit_at_target(1.0));
    CHECK(!cache_hit_at_target(0.94));
    CHECK(!cache_hit_at_target(0.0));
}

TEST_CASE("cache_hit_critical: 85% failure boundary", "[predicates][cache][phase83]") {
    CHECK( cache_hit_critical(0.84));
    CHECK( cache_hit_critical(0.0));
    CHECK(!cache_hit_critical(0.85));  // at boundary: not yet critical
    CHECK(!cache_hit_critical(0.88));
    CHECK(!cache_hit_critical(0.95));
}

TEST_CASE("cache_miss_bottleneck: 10% threshold", "[predicates][cache][phase83]") {
    CHECK( cache_miss_bottleneck(0.101));
    CHECK( cache_miss_bottleneck(0.5));
    CHECK(!cache_miss_bottleneck(0.10)); // exactly at threshold: not bottleneck yet
    CHECK(!cache_miss_bottleneck(0.05));
}

TEST_CASE("miss_rate is complementary to hit_rate", "[predicates][cache][phase83]") {
    CHECK(miss_rate(0.95) == Approx(0.05));
    CHECK(miss_rate(0.85) == Approx(0.15));
    CHECK(miss_rate(0.0)  == Approx(1.0));
    CHECK(miss_rate(1.0)  == Approx(0.0));
}

// ---------------------------------------------------------------------------
// § Laplacian predicates
// ---------------------------------------------------------------------------

TEST_CASE("laplacian_precision_ok: 1e-7 target", "[predicates][laplacian][phase83]") {
    CHECK( laplacian_precision_ok(1.0e-7));   // at target
    CHECK( laplacian_precision_ok(1.0e-8));   // better than target
    CHECK( laplacian_precision_ok(0.0));
    CHECK(!laplacian_precision_ok(1.1e-7));   // slightly above
    CHECK(!laplacian_precision_ok(1.0e-5));
}

TEST_CASE("laplacian_precision_critical: 1e-5 failure threshold", "[predicates][laplacian][phase83]") {
    CHECK( laplacian_precision_critical(1.1e-5));
    CHECK( laplacian_precision_critical(1.0e-4));
    CHECK(!laplacian_precision_critical(1.0e-5)); // at threshold: not yet critical
    CHECK(!laplacian_precision_critical(1.0e-6));
    CHECK(!laplacian_precision_critical(0.0));
}

// ---------------------------------------------------------------------------
// § Energy drift predicates
// ---------------------------------------------------------------------------

TEST_CASE("energy_drift_ok: 0.01% target boundary", "[predicates][energy][phase83]") {
    CHECK( energy_drift_ok(0.0001));     // at target
    CHECK( energy_drift_ok(0.00005));    // well below target
    CHECK( energy_drift_ok(0.0));
    CHECK(!energy_drift_ok(0.00011));    // just above target
    CHECK(!energy_drift_ok(0.001));
}

TEST_CASE("energy_drift_critical: 0.05% failure boundary", "[predicates][energy][phase83]") {
    CHECK( energy_drift_critical(0.00051));
    CHECK( energy_drift_critical(0.01));
    CHECK(!energy_drift_critical(0.0005)); // at boundary: not critical
    CHECK(!energy_drift_critical(0.0001));
    CHECK(!energy_drift_critical(0.0));
}

TEST_CASE("energy_divergence_action: correct state transitions", "[predicates][energy][phase83]") {
    // Within target → NONE
    CHECK(energy_divergence_action(0.0001) == EnergyDivergenceAction::NONE);
    CHECK(energy_divergence_action(0.0)    == EnergyDivergenceAction::NONE);

    // Between target and critical → REDUCE_TIMESTEP
    CHECK(energy_divergence_action(0.0002) == EnergyDivergenceAction::REDUCE_TIMESTEP);
    CHECK(energy_divergence_action(0.0005) == EnergyDivergenceAction::REDUCE_TIMESTEP);

    // Above critical → still REDUCE_TIMESTEP (first step in flowchart)
    CHECK(energy_divergence_action(0.001)  == EnergyDivergenceAction::REDUCE_TIMESTEP);
}

// ---------------------------------------------------------------------------
// § classify_latency / latency predicates
// ---------------------------------------------------------------------------

TEST_CASE("classify_latency: 81-grid three-region coverage", "[predicates][latency][phase83]") {
    // WITHIN_TARGET
    CHECK(classify_latency(7.8,  81) == BenchmarkStatus::WITHIN_TARGET);
    CHECK(classify_latency(1.0,  81) == BenchmarkStatus::WITHIN_TARGET);
    // ABOVE_TARGET
    CHECK(classify_latency(8.0,  81) == BenchmarkStatus::ABOVE_TARGET);
    CHECK(classify_latency(12.0, 81) == BenchmarkStatus::ABOVE_TARGET);
    // CRITICAL
    CHECK(classify_latency(12.1, 81) == BenchmarkStatus::CRITICAL);
    CHECK(classify_latency(50.0, 81) == BenchmarkStatus::CRITICAL);
}

TEST_CASE("classify_latency: 27-grid three-region coverage", "[predicates][latency][phase83]") {
    // WITHIN_TARGET
    CHECK(classify_latency(0.48, 27) == BenchmarkStatus::WITHIN_TARGET);
    CHECK(classify_latency(0.10, 27) == BenchmarkStatus::WITHIN_TARGET);
    // ABOVE_TARGET
    CHECK(classify_latency(0.50, 27) == BenchmarkStatus::ABOVE_TARGET);
    CHECK(classify_latency(1.0,  27) == BenchmarkStatus::ABOVE_TARGET);
    // CRITICAL
    CHECK(classify_latency(1.01, 27) == BenchmarkStatus::CRITICAL);
    CHECK(classify_latency(5.0,  27) == BenchmarkStatus::CRITICAL);
}

TEST_CASE("latency_within_target is consistent with classify_latency", "[predicates][latency][phase83]") {
    CHECK( latency_within_target(7.8,  81));
    CHECK(!latency_within_target(8.0,  81));
    CHECK( latency_within_target(0.48, 27));
    CHECK(!latency_within_target(0.49, 27));
}

TEST_CASE("latency_critical is consistent with classify_latency", "[predicates][latency][phase83]") {
    CHECK(!latency_critical(7.8,  81));
    CHECK(!latency_critical(11.0, 81));
    CHECK( latency_critical(12.1, 81));
    CHECK(!latency_critical(0.48, 27));
    CHECK( latency_critical(1.01, 27));
}

// ---------------------------------------------------------------------------
// § CUDA overhead helpers
// ---------------------------------------------------------------------------

TEST_CASE("overhead_budget_fraction: 90μs / 1000μs = 0.09", "[cuda][phase83]") {
    CHECK(overhead_budget_fraction(90.0, 1000.0) == Approx(0.09));
    CHECK(overhead_budget_fraction(0.0,  1000.0) == Approx(0.0));
    CHECK(overhead_budget_fraction(1000.0, 1000.0) == Approx(1.0));
}

TEST_CASE("overhead_budget_fraction: default budget uses 1000μs", "[cuda][phase83]") {
    CHECK(overhead_budget_fraction(TOTAL_LAUNCH_OVERHEAD_US) == Approx(LAUNCH_OVERHEAD_FRACTION).epsilon(1e-9));
}

TEST_CASE("overhead_budget_fraction: zero budget returns 0.0", "[cuda][phase83]") {
    CHECK(overhead_budget_fraction(90.0, 0.0) == Approx(0.0));
}

TEST_CASE("overhead_within_budget: 90μs out of 1000μs is within budget", "[cuda][phase83]") {
    CHECK(overhead_within_budget(90.0));         // exactly 9%
    CHECK(overhead_within_budget(0.0));           // trivially in budget
    CHECK(!overhead_within_budget(200.0));        // 20% exceeds 10%+1% margin
}

TEST_CASE("decoherence_risk: 500μs boundary", "[cuda][phase83]") {
    CHECK( decoherence_risk(500.0));
    CHECK( decoherence_risk(600.0));
    CHECK(!decoherence_risk(499.9));
    CHECK(!decoherence_risk(0.0));
}

TEST_CASE("standard_launch_overhead_us: linear scaling", "[cuda][phase83]") {
    CHECK(standard_launch_overhead_us(1) == Approx(15.0));
    CHECK(standard_launch_overhead_us(6) == Approx(90.0));
    CHECK(standard_launch_overhead_us(0) == Approx(0.0));
}

TEST_CASE("graph_overhead_saving_us: 6-kernel saving vs 1-graph launch", "[cuda][phase83]") {
    // 6 × 15 μs  - 5 μs = 85 μs saved
    CHECK(graph_overhead_saving_us(6) == Approx(90.0 - 5.0));
    CHECK(graph_overhead_saving_us(1) == Approx(15.0 - 5.0));
}

// ---------------------------------------------------------------------------
// § Grid geometry helpers
// ---------------------------------------------------------------------------

TEST_CASE("block_node_count: scale-3 result is 3^9 = 19683", "[grid][phase83]") {
    CHECK(block_node_count(3) == 19683LL);
}

TEST_CASE("block_node_count: scale-1 result is 1", "[grid][phase83]") {
    CHECK(block_node_count(1) == 1LL);
}

TEST_CASE("is_valid_block_size: powers of 3", "[grid][phase83]") {
    CHECK( is_valid_block_size(1));      // 3^0
    CHECK( is_valid_block_size(3));      // 3^1
    CHECK( is_valid_block_size(9));      // 3^2
    CHECK( is_valid_block_size(27));     // 3^3
    CHECK( is_valid_block_size(81));     // 3^4
    CHECK( is_valid_block_size(729));    // 3^6
    CHECK( is_valid_block_size(19683));  // 3^9
}

TEST_CASE("is_valid_block_size: non-powers-of-3 return false", "[grid][phase83]") {
    CHECK(!is_valid_block_size(0));
    CHECK(!is_valid_block_size(2));
    CHECK(!is_valid_block_size(4));
    CHECK(!is_valid_block_size(256));    // CUDA block size, not grid block size
    CHECK(!is_valid_block_size(-1));
}

TEST_CASE("fits_rtx4090: 14M node boundary", "[grid][phase83]") {
    CHECK( fits_rtx4090(14'000'000LL));
    CHECK( fits_rtx4090(1'000'000LL));
    CHECK( fits_rtx4090(0LL));
    CHECK(!fits_rtx4090(14'000'001LL));
    CHECK(!fits_rtx4090(100'000'000LL));
}

// ---------------------------------------------------------------------------
// § Tuning direction advisors
// ---------------------------------------------------------------------------

TEST_CASE("advise_hebbian_rate: correct tri-state logic", "[advisors][phase83]") {
    // Manic switching → decrease
    CHECK(advise_hebbian_rate(true,  false) == TuningDirection::DECREASE);
    CHECK(advise_hebbian_rate(true,  true)  == TuningDirection::DECREASE); // manic wins
    // Stagnant (and not manic) → increase
    CHECK(advise_hebbian_rate(false, true)  == TuningDirection::INCREASE);
    // Neither → hold
    CHECK(advise_hebbian_rate(false, false) == TuningDirection::HOLD);
}

TEST_CASE("advise_nap_trigger: correct tri-state logic", "[advisors][phase83]") {
    CHECK(advise_nap_trigger(true,  false) == TuningDirection::INCREASE);
    CHECK(advise_nap_trigger(false, true)  == TuningDirection::DECREASE);
    CHECK(advise_nap_trigger(false, false) == TuningDirection::HOLD);
}

// ---------------------------------------------------------------------------
// § Integration / scenario tests
// ---------------------------------------------------------------------------

TEST_CASE("Scenario A: high latency → bottleneck detection chain", "[scenario][phase83]") {
    // Simulated: step takes 15ms on 81-grid → CRITICAL
    double measured_ms = 15.0;
    CHECK(latency_critical(measured_ms, 81));
    CHECK(classify_latency(measured_ms, 81) == BenchmarkStatus::CRITICAL);
    CHECK(benchmark_status_label(BenchmarkStatus::CRITICAL) == "CRITICAL");

    // Cache miss also elevated → diagnose CACHE_MISS bottleneck
    double miss_r = 0.12;
    CHECK(cache_miss_bottleneck(miss_r));
    CHECK(!cache_hit_at_target(1.0 - miss_r));
}

TEST_CASE("Scenario B: energy drift → immediate action", "[scenario][phase83]") {
    double drift = 0.00015; // 0.015% — above target but below critical
    CHECK(!energy_drift_ok(drift));
    CHECK(!energy_drift_critical(drift));
    CHECK(energy_divergence_action(drift) == EnergyDivergenceAction::REDUCE_TIMESTEP);

    // After halving dt: 0.5ms → still within range
    double new_dt = 0.5;
    CHECK(timestep_within_range(new_dt));
}

TEST_CASE("Scenario C: all-green nominal state", "[scenario][phase83]") {
    // All predicates return OK for nominal operating conditions
    CHECK(timestep_within_range(PHYSICS_DT_DEFAULT_MS));
    CHECK(hebbian_rate_in_range(HEBBIAN_RATE_DEFAULT));
    CHECK(nap_trigger_in_range(NAP_TRIGGER_DEFAULT));
    CHECK(dither_amplitude_in_range(DITHER_AMPLITUDE_DEFAULT));
    CHECK(metabolic_cost_in_range(METABOLIC_COST_PLASTICITY_DEFAULT));
    CHECK(cache_hit_at_target(CACHE_HIT_TARGET));
    CHECK(laplacian_precision_ok(LAPLACIAN_PRECISION_TARGET));
    CHECK(energy_drift_ok(ENERGY_DRIFT_TARGET));
    CHECK(latency_within_target(BM_WAVE_81_TARGET_MS, 81));
    CHECK(latency_within_target(BM_WAVE_27_TARGET_MS, 27));
    CHECK(overhead_within_budget(TOTAL_LAUNCH_OVERHEAD_US));
    CHECK(!decoherence_risk(TOTAL_LAUNCH_OVERHEAD_US));
}

TEST_CASE("Scenario D: CPU-only profile uses 5ms timestep", "[scenario][phase83]") {
    // On CPU-only profile physics runs at 200Hz → dt = 5ms
    double cpu_dt = 1.0 / PHYSICS_LOOP_HZ_CPU * 1000.0; // = 5.0
    CHECK(timestep_within_range(cpu_dt));
    CHECK(cpu_dt == Approx(PHYSICS_DT_MAX_MS));

    // RTX 4090 check: 81-grid = 531441 nodes → fits easily
    CHECK(fits_rtx4090(static_cast<int64_t>(block_node_count(3)))); // 19683
}
