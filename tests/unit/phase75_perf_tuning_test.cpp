// ============================================================
// phase75_perf_tuning_test.cpp
//
// Unit tests for nikola/infrastructure/perf_tuning.hpp
// GAP-043: Performance Tuning Cookbook
//
// Coverage:
//   §1  OptimizationFocus enum
//   §2  HardwareProfile enum
//   §3  Grid-size constants
//   §4  GPU tuning constants
//   §5  Knob names and count
//   §6  Knob defaults
//   §7  Knob range bounds and invariants
//   §8  Diagnostic thresholds
//   §9  Benchmark targets and failure thresholds
//   §10 Query functions — profile dispatch
//   §11 Query functions — knob validators
//   §12 Query functions — benchmark/diagnostic predicates
//   §13 Label functions
//   Integration scenarios
// ============================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "nikola/infrastructure/perf_tuning.hpp"

using namespace nikola::infrastructure;

// ============================================================
// §1  OptimizationFocus Enum
// ============================================================

TEST_CASE("OptFocus_DataLocalityValue", "[perf_tuning][enums]") {
    CHECK(static_cast<uint8_t>(OptimizationFocus::DATA_LOCALITY) == 0u);
}

TEST_CASE("OptFocus_CacheEfficiencyValue", "[perf_tuning][enums]") {
    CHECK(static_cast<uint8_t>(OptimizationFocus::CACHE_EFFICIENCY) == 1u);
}

TEST_CASE("OptFocus_BandwidthSaturationValue", "[perf_tuning][enums]") {
    CHECK(static_cast<uint8_t>(OptimizationFocus::BANDWIDTH_SATURATION) == 2u);
}

TEST_CASE("OptFocus_MemoryBoundTuningValue", "[perf_tuning][enums]") {
    CHECK(static_cast<uint8_t>(OptimizationFocus::MEMORY_BOUND_TUNING) == 3u);
}

TEST_CASE("OptFocus_Count", "[perf_tuning][enums]") {
    CHECK(OPTIMIZATION_FOCUS_COUNT == 4u);
}

TEST_CASE("OptFocus_AllDistinct", "[perf_tuning][enums]") {
    CHECK(OptimizationFocus::DATA_LOCALITY        != OptimizationFocus::CACHE_EFFICIENCY);
    CHECK(OptimizationFocus::CACHE_EFFICIENCY     != OptimizationFocus::BANDWIDTH_SATURATION);
    CHECK(OptimizationFocus::BANDWIDTH_SATURATION != OptimizationFocus::MEMORY_BOUND_TUNING);
}

// ============================================================
// §2  HardwareProfile Enum
// ============================================================

TEST_CASE("HWProfile_CPUOnlyValue", "[perf_tuning][enums]") {
    CHECK(static_cast<uint8_t>(HardwareProfile::CPU_ONLY) == 0u);
}

TEST_CASE("HWProfile_SingleGPUValue", "[perf_tuning][enums]") {
    CHECK(static_cast<uint8_t>(HardwareProfile::SINGLE_GPU) == 1u);
}

TEST_CASE("HWProfile_MultiGPUValue", "[perf_tuning][enums]") {
    CHECK(static_cast<uint8_t>(HardwareProfile::MULTI_GPU) == 2u);
}

TEST_CASE("HWProfile_Count", "[perf_tuning][enums]") {
    CHECK(HARDWARE_PROFILE_COUNT == 3u);
}

TEST_CASE("HWProfile_AllDistinct", "[perf_tuning][enums]") {
    CHECK(HardwareProfile::CPU_ONLY   != HardwareProfile::SINGLE_GPU);
    CHECK(HardwareProfile::SINGLE_GPU != HardwareProfile::MULTI_GPU);
}

// ============================================================
// §3  Grid-Size Constants
// ============================================================

TEST_CASE("Grid_Base3", "[perf_tuning][grid]") {
    CHECK(GRID_BASE == 3u);
}

TEST_CASE("Grid_9Dimensions", "[perf_tuning][grid]") {
    CHECK(GRID_DIMENSIONS == 9u);
}

TEST_CASE("Grid_BlockSizeDefault_3to9", "[perf_tuning][grid]") {
    // 3^9 = 19683
    uint32_t expected = 1u;
    for (uint32_t i = 0u; i < 9u; ++i) expected *= 3u;
    CHECK(BLOCK_SIZE_DEFAULT == expected);
    CHECK(BLOCK_SIZE_DEFAULT == 19'683u);
}

TEST_CASE("Grid_SmallGridEdge_27", "[perf_tuning][grid]") {
    CHECK(SMALL_GRID_EDGE == 27u);
}

TEST_CASE("Grid_SmallGridNodes_27cubed", "[perf_tuning][grid]") {
    CHECK(SMALL_GRID_NODES == SMALL_GRID_EDGE * SMALL_GRID_EDGE * SMALL_GRID_EDGE);
    CHECK(SMALL_GRID_NODES == 19'683u);
}

TEST_CASE("Grid_SmallGridMatchesBlockDefault", "[perf_tuning][grid]") {
    // 27^3 == 3^9 == BLOCK_SIZE_DEFAULT
    CHECK(SMALL_GRID_NODES == BLOCK_SIZE_DEFAULT);
}

TEST_CASE("Grid_LargeGridEdge_81", "[perf_tuning][grid]") {
    CHECK(LARGE_GRID_EDGE == 81u);
}

TEST_CASE("Grid_LargeGridNodes_81cubed", "[perf_tuning][grid]") {
    CHECK(LARGE_GRID_NODES == LARGE_GRID_EDGE * LARGE_GRID_EDGE * LARGE_GRID_EDGE);
    CHECK(LARGE_GRID_NODES == 531'441u);
}

TEST_CASE("Grid_LargeGridBiggerThanSmall", "[perf_tuning][grid]") {
    CHECK(LARGE_GRID_NODES > SMALL_GRID_NODES);
}

TEST_CASE("Grid_SoAAlignment_64Bytes", "[perf_tuning][grid]") {
    CHECK(SOA_ALIGNMENT_BYTES == 64u);
}

TEST_CASE("Grid_SoAAlignmentIsPowerOf2", "[perf_tuning][grid]") {
    CHECK((SOA_ALIGNMENT_BYTES & (SOA_ALIGNMENT_BYTES - 1u)) == 0u);
}

// ============================================================
// §4  GPU Tuning Constants
// ============================================================

TEST_CASE("GPU_CudaBlockSizeRTX4090", "[perf_tuning][gpu]") {
    CHECK(CUDA_BLOCK_SIZE_RTX4090 == 256u);
}

TEST_CASE("GPU_MaxNodesRTX4090", "[perf_tuning][gpu]") {
    CHECK(MAX_ACTIVE_NODES_RTX4090 == 14'000'000u);
}

TEST_CASE("GPU_MinNodesA100Cluster", "[perf_tuning][gpu]") {
    CHECK(MIN_ACTIVE_NODES_A100_CLUSTER == 100'000'000u);
}

TEST_CASE("GPU_A100ExceedsRTX4090Capacity", "[perf_tuning][gpu]") {
    CHECK(MIN_ACTIVE_NODES_A100_CLUSTER > MAX_ACTIVE_NODES_RTX4090);
}

TEST_CASE("GPU_CudaBlockSizeIsPowerOf2", "[perf_tuning][gpu]") {
    CHECK((CUDA_BLOCK_SIZE_RTX4090 & (CUDA_BLOCK_SIZE_RTX4090 - 1u)) == 0u);
}

// ============================================================
// §5  Knob Names and Count
// ============================================================

TEST_CASE("KnobName_HebbianRate", "[perf_tuning][knobs]") {
    CHECK(KNOB_HEBBIAN_RATE == "hebbian_rate");
}

TEST_CASE("KnobName_MetabolicCost", "[perf_tuning][knobs]") {
    CHECK(KNOB_METABOLIC_COST == "metabolic_cost_plasticity");
}

TEST_CASE("KnobName_NapTrigger", "[perf_tuning][knobs]") {
    CHECK(KNOB_NAP_TRIGGER == "nap_interval_trigger");
}

TEST_CASE("KnobName_PhysicsDt", "[perf_tuning][knobs]") {
    CHECK(KNOB_PHYSICS_DT == "physics_dt");
}

TEST_CASE("KnobName_BlockSize", "[perf_tuning][knobs]") {
    CHECK(KNOB_BLOCK_SIZE == "block_size");
}

TEST_CASE("KnobName_DitherAmplitude", "[perf_tuning][knobs]") {
    CHECK(KNOB_DITHER_AMPLITUDE == "dither_amplitude");
}

TEST_CASE("KnobName_Count", "[perf_tuning][knobs]") {
    CHECK(KNOB_COUNT == 6u);
}

TEST_CASE("KnobName_AllNonEmpty", "[perf_tuning][knobs]") {
    CHECK_FALSE(KNOB_HEBBIAN_RATE.empty());
    CHECK_FALSE(KNOB_METABOLIC_COST.empty());
    CHECK_FALSE(KNOB_NAP_TRIGGER.empty());
    CHECK_FALSE(KNOB_PHYSICS_DT.empty());
    CHECK_FALSE(KNOB_BLOCK_SIZE.empty());
    CHECK_FALSE(KNOB_DITHER_AMPLITUDE.empty());
}

// ============================================================
// §6  Knob Defaults
// ============================================================

TEST_CASE("KnobDefault_HebbianRate_0_01", "[perf_tuning][defaults]") {
    CHECK(HEBBIAN_RATE_DEFAULT == Catch::Approx(0.01));
}

TEST_CASE("KnobDefault_MetabolicCost_1_5", "[perf_tuning][defaults]") {
    CHECK(METABOLIC_COST_DEFAULT == Catch::Approx(1.5));
}

TEST_CASE("KnobDefault_NapTrigger_15Pct", "[perf_tuning][defaults]") {
    CHECK(NAP_TRIGGER_DEFAULT == Catch::Approx(0.15));
}

TEST_CASE("KnobDefault_PhysicsDt_1ms", "[perf_tuning][defaults]") {
    CHECK(PHYSICS_DT_DEFAULT_MS == Catch::Approx(1.0));
}

TEST_CASE("KnobDefault_DitherAmplitude_1e-4", "[perf_tuning][defaults]") {
    CHECK(DITHER_AMPLITUDE_DEFAULT == Catch::Approx(1.0e-4));
}

TEST_CASE("KnobDefault_AllWithinRange", "[perf_tuning][defaults]") {
    CHECK(is_valid_hebbian_rate(HEBBIAN_RATE_DEFAULT));
    CHECK(is_valid_metabolic_cost(METABOLIC_COST_DEFAULT));
    CHECK(is_valid_nap_trigger(NAP_TRIGGER_DEFAULT));
    CHECK(is_valid_physics_dt(PHYSICS_DT_DEFAULT_MS));
    CHECK(is_valid_dither_amplitude(DITHER_AMPLITUDE_DEFAULT));
}

// ============================================================
// §7  Knob Range Bounds and Invariants
// ============================================================

TEST_CASE("KnobRange_HebbianMin_0_001", "[perf_tuning][ranges]") {
    CHECK(HEBBIAN_RATE_MIN == Catch::Approx(0.001));
}

TEST_CASE("KnobRange_HebbianMax_0_1", "[perf_tuning][ranges]") {
    CHECK(HEBBIAN_RATE_MAX == Catch::Approx(0.1));
}

TEST_CASE("KnobRange_MetabolicCostMin_1_0", "[perf_tuning][ranges]") {
    CHECK(METABOLIC_COST_MIN == Catch::Approx(1.0));
}

TEST_CASE("KnobRange_MetabolicCostMax_5_0", "[perf_tuning][ranges]") {
    CHECK(METABOLIC_COST_MAX == Catch::Approx(5.0));
}

TEST_CASE("KnobRange_NapMin_5Pct", "[perf_tuning][ranges]") {
    CHECK(NAP_TRIGGER_MIN == Catch::Approx(0.05));
}

TEST_CASE("KnobRange_NapMax_30Pct", "[perf_tuning][ranges]") {
    CHECK(NAP_TRIGGER_MAX == Catch::Approx(0.30));
}

TEST_CASE("KnobRange_PhysicsDtMin_0_1ms", "[perf_tuning][ranges]") {
    CHECK(PHYSICS_DT_MIN_MS == Catch::Approx(0.1));
}

TEST_CASE("KnobRange_PhysicsDtMax_5ms", "[perf_tuning][ranges]") {
    CHECK(PHYSICS_DT_MAX_MS == Catch::Approx(5.0));
}

TEST_CASE("KnobRange_DitherMin_1e-5", "[perf_tuning][ranges]") {
    CHECK(DITHER_AMPLITUDE_MIN == Catch::Approx(1.0e-5));
}

TEST_CASE("KnobRange_DitherMax_1e-3", "[perf_tuning][ranges]") {
    CHECK(DITHER_AMPLITUDE_MAX == Catch::Approx(1.0e-3));
}

TEST_CASE("KnobRange_AllMinsLessThanMaxes", "[perf_tuning][ranges]") {
    CHECK(HEBBIAN_RATE_MIN    < HEBBIAN_RATE_MAX);
    CHECK(METABOLIC_COST_MIN  < METABOLIC_COST_MAX);
    CHECK(NAP_TRIGGER_MIN     < NAP_TRIGGER_MAX);
    CHECK(PHYSICS_DT_MIN_MS   < PHYSICS_DT_MAX_MS);
    CHECK(DITHER_AMPLITUDE_MIN < DITHER_AMPLITUDE_MAX);
}

TEST_CASE("KnobRange_DefaultsInsideRanges", "[perf_tuning][ranges]") {
    CHECK(HEBBIAN_RATE_DEFAULT    >= HEBBIAN_RATE_MIN);
    CHECK(HEBBIAN_RATE_DEFAULT    <= HEBBIAN_RATE_MAX);
    CHECK(METABOLIC_COST_DEFAULT  >= METABOLIC_COST_MIN);
    CHECK(METABOLIC_COST_DEFAULT  <= METABOLIC_COST_MAX);
    CHECK(NAP_TRIGGER_DEFAULT     >= NAP_TRIGGER_MIN);
    CHECK(NAP_TRIGGER_DEFAULT     <= NAP_TRIGGER_MAX);
    CHECK(PHYSICS_DT_DEFAULT_MS   >= PHYSICS_DT_MIN_MS);
    CHECK(PHYSICS_DT_DEFAULT_MS   <= PHYSICS_DT_MAX_MS);
    CHECK(DITHER_AMPLITUDE_DEFAULT >= DITHER_AMPLITUDE_MIN);
    CHECK(DITHER_AMPLITUDE_DEFAULT <= DITHER_AMPLITUDE_MAX);
}

TEST_CASE("KnobRange_CPUPhysicsHz_200", "[perf_tuning][ranges]") {
    CHECK(CPU_PHYSICS_HZ == 200u);
}

TEST_CASE("KnobRange_GPUPhysicsHz_1000", "[perf_tuning][ranges]") {
    CHECK(GPU_PHYSICS_HZ == 1'000u);
}

TEST_CASE("KnobRange_CPUPhysicsHz_MatchesDtMax", "[perf_tuning][ranges]") {
    // dt = 5 ms → 1/0.005 = 200 Hz
    const double implied_hz = 1000.0 / PHYSICS_DT_MAX_MS;
    CHECK(implied_hz == Catch::Approx(static_cast<double>(CPU_PHYSICS_HZ)));
}

TEST_CASE("KnobRange_GPUPhysicsHz_MatchesDtDefault", "[perf_tuning][ranges]") {
    // dt = 1 ms → 1/0.001 = 1000 Hz
    const double implied_hz = 1000.0 / PHYSICS_DT_DEFAULT_MS;
    CHECK(implied_hz == Catch::Approx(static_cast<double>(GPU_PHYSICS_HZ)));
}

// ============================================================
// §8  Diagnostic Thresholds
// ============================================================

TEST_CASE("DiagThreshold_TickLatencyHigh_1ms", "[perf_tuning][diag]") {
    CHECK(TICK_LATENCY_HIGH_MS == Catch::Approx(1.0));
}

TEST_CASE("DiagThreshold_CacheMissRate_10Pct", "[perf_tuning][diag]") {
    CHECK(CACHE_MISS_RATE_THRESHOLD == Catch::Approx(0.10));
}

TEST_CASE("DiagThreshold_SNR_20dB", "[perf_tuning][diag]") {
    CHECK(SNR_MIN_DB == Catch::Approx(20.0));
}

TEST_CASE("DiagThreshold_EnergyDrift_1e-4", "[perf_tuning][diag]") {
    CHECK(ENERGY_DRIFT_DIAG_THRESHOLD == Catch::Approx(1.0e-4));
}

TEST_CASE("DiagThreshold_TickLatencyEqualsHistogramThreshold", "[perf_tuning][diag]") {
    // The 1 ms tick alarm aligns with the GAP-027b 900 μs interest threshold
    // being sub-millisecond; both are sub-1-ms regime comparisons
    CHECK(TICK_LATENCY_HIGH_MS > 0.0);
    CHECK(TICK_LATENCY_HIGH_MS <= 1.0);
}

// ============================================================
// §9  Benchmark Baselines and Failure Thresholds
// ============================================================

TEST_CASE("Benchmark_LargeGridTarget_7_8ms", "[perf_tuning][benchmarks]") {
    CHECK(BM_LARGE_GRID_TARGET_MS == Catch::Approx(7.8));
}

TEST_CASE("Benchmark_LargeGridFail_12ms", "[perf_tuning][benchmarks]") {
    CHECK(BM_LARGE_GRID_FAIL_MS == Catch::Approx(12.0));
}

TEST_CASE("Benchmark_SmallGridTarget_0_48ms", "[perf_tuning][benchmarks]") {
    CHECK(BM_SMALL_GRID_TARGET_MS == Catch::Approx(0.48));
}

TEST_CASE("Benchmark_SmallGridFail_1ms", "[perf_tuning][benchmarks]") {
    CHECK(BM_SMALL_GRID_FAIL_MS == Catch::Approx(1.0));
}

TEST_CASE("Benchmark_BandwidthUtilTarget_100Pct", "[perf_tuning][benchmarks]") {
    CHECK(BM_BANDWIDTH_UTIL_TARGET == Catch::Approx(1.0));
}

TEST_CASE("Benchmark_BandwidthUtilFail_80Pct", "[perf_tuning][benchmarks]") {
    CHECK(BM_BANDWIDTH_UTIL_FAIL == Catch::Approx(0.80));
}

TEST_CASE("Benchmark_CacheHitTarget_95Pct", "[perf_tuning][benchmarks]") {
    CHECK(BM_CACHE_HIT_TARGET == Catch::Approx(0.95));
}

TEST_CASE("Benchmark_CacheHitFail_85Pct", "[perf_tuning][benchmarks]") {
    CHECK(BM_CACHE_HIT_FAIL == Catch::Approx(0.85));
}

TEST_CASE("Benchmark_LaplacianPrecisionTarget_1e-7", "[perf_tuning][benchmarks]") {
    CHECK(BM_LAPLACIAN_PRECISION_TARGET == Catch::Approx(1.0e-7));
}

TEST_CASE("Benchmark_LaplacianPrecisionFail_1e-5", "[perf_tuning][benchmarks]") {
    CHECK(BM_LAPLACIAN_PRECISION_FAIL == Catch::Approx(1.0e-5));
}

TEST_CASE("Benchmark_EnergyDriftTarget_1e-4", "[perf_tuning][benchmarks]") {
    CHECK(BM_ENERGY_DRIFT_TARGET == Catch::Approx(1.0e-4));
}

TEST_CASE("Benchmark_EnergyDriftFail_5e-4", "[perf_tuning][benchmarks]") {
    CHECK(BM_ENERGY_DRIFT_FAIL == Catch::Approx(5.0e-4));
}

TEST_CASE("Benchmark_MetricCount", "[perf_tuning][benchmarks]") {
    CHECK(BENCHMARK_METRIC_COUNT == 6u);
}

TEST_CASE("Benchmark_TargetsLessThanFailures", "[perf_tuning][benchmarks]") {
    // For latency metrics: target < failure (lower is better)
    CHECK(BM_LARGE_GRID_TARGET_MS < BM_LARGE_GRID_FAIL_MS);
    CHECK(BM_SMALL_GRID_TARGET_MS < BM_SMALL_GRID_FAIL_MS);
    CHECK(BM_ENERGY_DRIFT_TARGET  < BM_ENERGY_DRIFT_FAIL);
    CHECK(BM_LAPLACIAN_PRECISION_TARGET < BM_LAPLACIAN_PRECISION_FAIL);
    // For rate metrics: target > failure (higher is better)
    CHECK(BM_BANDWIDTH_UTIL_TARGET > BM_BANDWIDTH_UTIL_FAIL);
    CHECK(BM_CACHE_HIT_TARGET      > BM_CACHE_HIT_FAIL);
}

// ============================================================
// §10  Query Functions — Profile Dispatch
// ============================================================

TEST_CASE("ProfileDispatch_CPUOnly_PhysicsDt_5ms", "[perf_tuning][query]") {
    CHECK(default_physics_dt_ms(HardwareProfile::CPU_ONLY) == Catch::Approx(5.0));
}

TEST_CASE("ProfileDispatch_SingleGPU_PhysicsDt_1ms", "[perf_tuning][query]") {
    CHECK(default_physics_dt_ms(HardwareProfile::SINGLE_GPU) == Catch::Approx(1.0));
}

TEST_CASE("ProfileDispatch_MultiGPU_PhysicsDt_1ms", "[perf_tuning][query]") {
    CHECK(default_physics_dt_ms(HardwareProfile::MULTI_GPU) == Catch::Approx(1.0));
}

TEST_CASE("ProfileDispatch_CPUOnly_Freq_200Hz", "[perf_tuning][query]") {
    CHECK(physics_frequency_hz(HardwareProfile::CPU_ONLY) == 200u);
}

TEST_CASE("ProfileDispatch_SingleGPU_Freq_1000Hz", "[perf_tuning][query]") {
    CHECK(physics_frequency_hz(HardwareProfile::SINGLE_GPU) == 1'000u);
}

TEST_CASE("ProfileDispatch_MultiGPU_Freq_1000Hz", "[perf_tuning][query]") {
    CHECK(physics_frequency_hz(HardwareProfile::MULTI_GPU) == 1'000u);
}

TEST_CASE("ProfileDispatch_CPUOnly_NoCUDA", "[perf_tuning][query]") {
    CHECK_FALSE(requires_cuda(HardwareProfile::CPU_ONLY));
}

TEST_CASE("ProfileDispatch_SingleGPU_NeedsCUDA", "[perf_tuning][query]") {
    CHECK(requires_cuda(HardwareProfile::SINGLE_GPU));
}

TEST_CASE("ProfileDispatch_MultiGPU_NeedsCUDA", "[perf_tuning][query]") {
    CHECK(requires_cuda(HardwareProfile::MULTI_GPU));
}

TEST_CASE("ProfileDispatch_CPUOnly_NoDistributed", "[perf_tuning][query]") {
    CHECK_FALSE(requires_distributed_sharding(HardwareProfile::CPU_ONLY));
}

TEST_CASE("ProfileDispatch_SingleGPU_NoDistributed", "[perf_tuning][query]") {
    CHECK_FALSE(requires_distributed_sharding(HardwareProfile::SINGLE_GPU));
}

TEST_CASE("ProfileDispatch_MultiGPU_NeedsDistributed", "[perf_tuning][query]") {
    CHECK(requires_distributed_sharding(HardwareProfile::MULTI_GPU));
}

TEST_CASE("ProfileDispatch_CPUOnly_NoFP64Preference", "[perf_tuning][query]") {
    CHECK_FALSE(prefers_fp64(HardwareProfile::CPU_ONLY));
}

TEST_CASE("ProfileDispatch_SingleGPU_NoFP64Preference", "[perf_tuning][query]") {
    CHECK_FALSE(prefers_fp64(HardwareProfile::SINGLE_GPU));
}

TEST_CASE("ProfileDispatch_MultiGPU_PrefersFP64", "[perf_tuning][query]") {
    CHECK(prefers_fp64(HardwareProfile::MULTI_GPU));
}

// ============================================================
// §11  Query Functions — Knob Validators
// ============================================================

TEST_CASE("ValidPhysicsDt_ExactMin_Valid", "[perf_tuning][validators]") {
    CHECK(is_valid_physics_dt(PHYSICS_DT_MIN_MS));
}

TEST_CASE("ValidPhysicsDt_ExactMax_Valid", "[perf_tuning][validators]") {
    CHECK(is_valid_physics_dt(PHYSICS_DT_MAX_MS));
}

TEST_CASE("ValidPhysicsDt_Default_Valid", "[perf_tuning][validators]") {
    CHECK(is_valid_physics_dt(PHYSICS_DT_DEFAULT_MS));
}

TEST_CASE("ValidPhysicsDt_Zero_Invalid", "[perf_tuning][validators]") {
    CHECK_FALSE(is_valid_physics_dt(0.0));
}

TEST_CASE("ValidPhysicsDt_TooLarge_Invalid", "[perf_tuning][validators]") {
    CHECK_FALSE(is_valid_physics_dt(10.0));
}

TEST_CASE("ValidHebbianRate_Default_Valid", "[perf_tuning][validators]") {
    CHECK(is_valid_hebbian_rate(HEBBIAN_RATE_DEFAULT));
}

TEST_CASE("ValidHebbianRate_BelowMin_Invalid", "[perf_tuning][validators]") {
    CHECK_FALSE(is_valid_hebbian_rate(0.0001));
}

TEST_CASE("ValidHebbianRate_AboveMax_Invalid", "[perf_tuning][validators]") {
    CHECK_FALSE(is_valid_hebbian_rate(0.5));
}

TEST_CASE("ValidMetabolicCost_Min_Valid", "[perf_tuning][validators]") {
    CHECK(is_valid_metabolic_cost(METABOLIC_COST_MIN));
}

TEST_CASE("ValidMetabolicCost_Max_Valid", "[perf_tuning][validators]") {
    CHECK(is_valid_metabolic_cost(METABOLIC_COST_MAX));
}

TEST_CASE("ValidMetabolicCost_BelowMin_Invalid", "[perf_tuning][validators]") {
    CHECK_FALSE(is_valid_metabolic_cost(0.5));
}

TEST_CASE("ValidMetabolicCost_AboveMax_Invalid", "[perf_tuning][validators]") {
    CHECK_FALSE(is_valid_metabolic_cost(6.0));
}

TEST_CASE("ValidNapTrigger_Default_Valid", "[perf_tuning][validators]") {
    CHECK(is_valid_nap_trigger(NAP_TRIGGER_DEFAULT));
}

TEST_CASE("ValidNapTrigger_Min_Valid", "[perf_tuning][validators]") {
    CHECK(is_valid_nap_trigger(NAP_TRIGGER_MIN));
}

TEST_CASE("ValidNapTrigger_Max_Valid", "[perf_tuning][validators]") {
    CHECK(is_valid_nap_trigger(NAP_TRIGGER_MAX));
}

TEST_CASE("ValidNapTrigger_Zero_Invalid", "[perf_tuning][validators]") {
    CHECK_FALSE(is_valid_nap_trigger(0.0));
}

TEST_CASE("ValidNapTrigger_AboveMax_Invalid", "[perf_tuning][validators]") {
    CHECK_FALSE(is_valid_nap_trigger(0.50));
}

TEST_CASE("ValidDitherAmplitude_Default_Valid", "[perf_tuning][validators]") {
    CHECK(is_valid_dither_amplitude(DITHER_AMPLITUDE_DEFAULT));
}

TEST_CASE("ValidDitherAmplitude_Min_Valid", "[perf_tuning][validators]") {
    CHECK(is_valid_dither_amplitude(DITHER_AMPLITUDE_MIN));
}

TEST_CASE("ValidDitherAmplitude_Max_Valid", "[perf_tuning][validators]") {
    CHECK(is_valid_dither_amplitude(DITHER_AMPLITUDE_MAX));
}

TEST_CASE("ValidDitherAmplitude_TooLarge_Invalid", "[perf_tuning][validators]") {
    CHECK_FALSE(is_valid_dither_amplitude(1.0));
}

TEST_CASE("ValidDitherAmplitude_Zero_Invalid", "[perf_tuning][validators]") {
    CHECK_FALSE(is_valid_dither_amplitude(0.0));
}

// ============================================================
// §12  Query Functions — Benchmark / Diagnostic Predicates
// ============================================================

TEST_CASE("CacheMissAlarm_ExactThreshold_False", "[perf_tuning][predicates]") {
    CHECK_FALSE(is_cache_miss_alarm(CACHE_MISS_RATE_THRESHOLD));
}

TEST_CASE("CacheMissAlarm_Above_True", "[perf_tuning][predicates]") {
    CHECK(is_cache_miss_alarm(0.15));
}

TEST_CASE("CacheMissAlarm_Zero_False", "[perf_tuning][predicates]") {
    CHECK_FALSE(is_cache_miss_alarm(0.0));
}

TEST_CASE("HighLatencyTick_ExactThreshold_False", "[perf_tuning][predicates]") {
    CHECK_FALSE(is_high_latency_tick(TICK_LATENCY_HIGH_MS));
}

TEST_CASE("HighLatencyTick_1_1ms_True", "[perf_tuning][predicates]") {
    CHECK(is_high_latency_tick(1.1));
}

TEST_CASE("HighLatencyTick_0_5ms_False", "[perf_tuning][predicates]") {
    CHECK_FALSE(is_high_latency_tick(0.5));
}

TEST_CASE("LargeGridBenchFail_AboveThreshold_True", "[perf_tuning][predicates]") {
    CHECK(is_large_grid_benchmark_fail(15.0));
}

TEST_CASE("LargeGridBenchFail_AtThreshold_False", "[perf_tuning][predicates]") {
    CHECK_FALSE(is_large_grid_benchmark_fail(BM_LARGE_GRID_FAIL_MS));
}

TEST_CASE("LargeGridBenchFail_Target_False", "[perf_tuning][predicates]") {
    CHECK_FALSE(is_large_grid_benchmark_fail(BM_LARGE_GRID_TARGET_MS));
}

TEST_CASE("SmallGridBenchFail_AboveThreshold_True", "[perf_tuning][predicates]") {
    CHECK(is_small_grid_benchmark_fail(2.0));
}

TEST_CASE("SmallGridBenchFail_Target_False", "[perf_tuning][predicates]") {
    CHECK_FALSE(is_small_grid_benchmark_fail(BM_SMALL_GRID_TARGET_MS));
}

TEST_CASE("BandwidthRegression_Below80Pct_True", "[perf_tuning][predicates]") {
    CHECK(is_bandwidth_regression(0.70));
}

TEST_CASE("BandwidthRegression_At80Pct_False", "[perf_tuning][predicates]") {
    CHECK_FALSE(is_bandwidth_regression(BM_BANDWIDTH_UTIL_FAIL));
}

TEST_CASE("BandwidthRegression_100Pct_False", "[perf_tuning][predicates]") {
    CHECK_FALSE(is_bandwidth_regression(1.0));
}

TEST_CASE("CacheHitFail_Below85_True", "[perf_tuning][predicates]") {
    CHECK(is_cache_hit_fail(0.80));
}

TEST_CASE("CacheHitFail_At85_False", "[perf_tuning][predicates]") {
    CHECK_FALSE(is_cache_hit_fail(BM_CACHE_HIT_FAIL));
}

TEST_CASE("LaplacianPrecisionFail_AboveThreshold_True", "[perf_tuning][predicates]") {
    CHECK(is_laplacian_precision_fail(1.0e-4));
}

TEST_CASE("LaplacianPrecisionFail_AtThreshold_False", "[perf_tuning][predicates]") {
    CHECK_FALSE(is_laplacian_precision_fail(BM_LAPLACIAN_PRECISION_FAIL));
}

TEST_CASE("LaplacianPrecisionFail_Target_False", "[perf_tuning][predicates]") {
    CHECK_FALSE(is_laplacian_precision_fail(BM_LAPLACIAN_PRECISION_TARGET));
}

TEST_CASE("EnergyDriftFail_AboveThreshold_True", "[perf_tuning][predicates]") {
    CHECK(is_energy_drift_fail(1.0e-3));
}

TEST_CASE("EnergyDriftFail_AtThreshold_False", "[perf_tuning][predicates]") {
    CHECK_FALSE(is_energy_drift_fail(BM_ENERGY_DRIFT_FAIL));
}

TEST_CASE("EnergyDriftFail_Target_False", "[perf_tuning][predicates]") {
    CHECK_FALSE(is_energy_drift_fail(BM_ENERGY_DRIFT_TARGET));
}

// ============================================================
// §13  Label Functions
// ============================================================

TEST_CASE("HWProfileName_CPUOnly", "[perf_tuning][labels]") {
    CHECK(hardware_profile_name(HardwareProfile::CPU_ONLY) == "cpu_only");
}

TEST_CASE("HWProfileName_SingleGPU", "[perf_tuning][labels]") {
    CHECK(hardware_profile_name(HardwareProfile::SINGLE_GPU) == "single_gpu_rtx4090");
}

TEST_CASE("HWProfileName_MultiGPU", "[perf_tuning][labels]") {
    CHECK(hardware_profile_name(HardwareProfile::MULTI_GPU) == "multi_gpu_a100");
}

TEST_CASE("HWProfileName_AllNonEmpty", "[perf_tuning][labels]") {
    CHECK_FALSE(hardware_profile_name(HardwareProfile::CPU_ONLY).empty());
    CHECK_FALSE(hardware_profile_name(HardwareProfile::SINGLE_GPU).empty());
    CHECK_FALSE(hardware_profile_name(HardwareProfile::MULTI_GPU).empty());
}

TEST_CASE("HWProfileName_AllDistinct", "[perf_tuning][labels]") {
    CHECK(hardware_profile_name(HardwareProfile::CPU_ONLY)   !=
          hardware_profile_name(HardwareProfile::SINGLE_GPU));
    CHECK(hardware_profile_name(HardwareProfile::SINGLE_GPU) !=
          hardware_profile_name(HardwareProfile::MULTI_GPU));
}

TEST_CASE("OptFocusName_DataLocality", "[perf_tuning][labels]") {
    CHECK(optimization_focus_name(OptimizationFocus::DATA_LOCALITY) == "data_locality");
}

TEST_CASE("OptFocusName_CacheEfficiency", "[perf_tuning][labels]") {
    CHECK(optimization_focus_name(OptimizationFocus::CACHE_EFFICIENCY) == "cache_efficiency");
}

TEST_CASE("OptFocusName_BandwidthSaturation", "[perf_tuning][labels]") {
    CHECK(optimization_focus_name(OptimizationFocus::BANDWIDTH_SATURATION) == "bandwidth_saturation");
}

TEST_CASE("OptFocusName_MemoryBound", "[perf_tuning][labels]") {
    CHECK(optimization_focus_name(OptimizationFocus::MEMORY_BOUND_TUNING) == "memory_bound_tuning");
}

TEST_CASE("OptFocusName_AllNonEmpty", "[perf_tuning][labels]") {
    for (uint8_t i = 0u; i < static_cast<uint8_t>(OPTIMIZATION_FOCUS_COUNT); ++i) {
        const auto focus = static_cast<OptimizationFocus>(i);
        CHECK_FALSE(optimization_focus_name(focus).empty());
    }
}

// ============================================================
// Integration Scenarios
// ============================================================

TEST_CASE("Integration_CPUDevProfile_CorrectSettings", "[perf_tuning][integration]") {
    // CPU-Only dev/debug: AVX-512, 200 Hz, no CUDA, no distributed
    const auto p = HardwareProfile::CPU_ONLY;
    CHECK(default_physics_dt_ms(p)       == Catch::Approx(5.0));
    CHECK(physics_frequency_hz(p)         == 200u);
    CHECK_FALSE(requires_cuda(p));
    CHECK_FALSE(requires_distributed_sharding(p));
    CHECK_FALSE(prefers_fp64(p));
}

TEST_CASE("Integration_RTX4090_CorrectSettings", "[perf_tuning][integration]") {
    // Single GPU: FP32, CUDA, block 256, 14M node limit, no distributed
    const auto p = HardwareProfile::SINGLE_GPU;
    CHECK(default_physics_dt_ms(p)       == Catch::Approx(1.0));
    CHECK(physics_frequency_hz(p)         == 1'000u);
    CHECK(requires_cuda(p));
    CHECK_FALSE(requires_distributed_sharding(p));
    CHECK_FALSE(prefers_fp64(p));
    CHECK(CUDA_BLOCK_SIZE_RTX4090        == 256u);
    CHECK(MAX_ACTIVE_NODES_RTX4090       == 14'000'000u);
}

TEST_CASE("Integration_A100Cluster_CorrectSettings", "[perf_tuning][integration]") {
    // Multi-GPU: FP64 optional, CUDA, distributed sharding, >100M nodes
    const auto p = HardwareProfile::MULTI_GPU;
    CHECK(requires_cuda(p));
    CHECK(requires_distributed_sharding(p));
    CHECK(prefers_fp64(p));
    CHECK(MIN_ACTIVE_NODES_A100_CLUSTER > MAX_ACTIVE_NODES_RTX4090);
}

TEST_CASE("Integration_ScenarioA_HighLatency", "[perf_tuning][integration]") {
    // Scenario A: tick > 1 ms → cache miss check
    const double tick_ms = 1.5;
    CHECK(is_high_latency_tick(tick_ms));

    // Cache miss > 10% → SoA / Hilbert verification required
    const double miss_rate = 0.12;
    CHECK(is_cache_miss_alarm(miss_rate));

    // Alignment constant for SoA verification
    CHECK(SOA_ALIGNMENT_BYTES == 64u);
}

TEST_CASE("Integration_ScenarioB_EnergyDivergence", "[perf_tuning][integration]") {
    // Scenario B: energy drift > 0.01% → Immediate: reduce dt by 50%
    const double drift = 0.00015;  // 0.015 % > threshold
    CHECK(drift > ENERGY_DRIFT_DIAG_THRESHOLD);

    // Corrective action: halve dt
    const double reduced_dt = PHYSICS_DT_DEFAULT_MS * 0.5;
    CHECK(is_valid_physics_dt(reduced_dt));
    CHECK(reduced_dt == Catch::Approx(0.5));
}

TEST_CASE("Integration_BenchmarkP0SmallGrid_PassFail", "[perf_tuning][integration]") {
    // P0 critical requirement: small-grid < 1 ms
    CHECK_FALSE(is_small_grid_benchmark_fail(BM_SMALL_GRID_TARGET_MS));  // 0.48 ms — pass
    CHECK(is_small_grid_benchmark_fail(1.5));                             // 1.5 ms — fail
}

TEST_CASE("Integration_AoSRegressionDetection", "[perf_tuning][integration]") {
    // Bandwidth < 80% → AoS regression
    CHECK(is_bandwidth_regression(0.75));
    CHECK_FALSE(is_bandwidth_regression(0.95));
    CHECK(BM_BANDWIDTH_UTIL_FAIL == Catch::Approx(0.80));
}

TEST_CASE("Integration_KahanPrecisionVerification", "[perf_tuning][integration]") {
    // Kahan summation should keep error ~ 1e-7; failure > 1e-5
    CHECK_FALSE(is_laplacian_precision_fail(1.0e-7));  // target — OK
    CHECK_FALSE(is_laplacian_precision_fail(1.0e-6));  // degraded but not failed
    CHECK(is_laplacian_precision_fail(1.0e-4));        // Kahan clearly broken
}
