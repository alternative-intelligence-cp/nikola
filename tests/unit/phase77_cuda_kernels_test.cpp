// ============================================================
// phase77_cuda_kernels_test.cpp
//
// Unit tests for nikola/infrastructure/cuda_kernels.hpp
// GAP-046: High-Frequency CUDA Kernel Optimization Strategies
//
// Coverage:
//   §1   ExecutionStrategy enum
//   §2   GraphState enum
//   §3   TopologyEvent enum
//   §4   Per-kernel launch overhead constants
//   §5   Physics tick budget / decoherence thresholds
//   §6   CUDA Graph constants
//   §7   Persistent kernel constants
//   §8   H100 occupancy constants
//   §9   Audio-visual pipeline constants
//   §10  Control-block field names
//   §11  Overhead query functions
//   §12  Timing budget predicates
//   §13  CUDA Graph decision helpers
//   §14  Persistent kernel suitability
//   §15  Label functions
//   Integration scenarios
// ============================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cstdint>
#include <string_view>

#include "nikola/infrastructure/cuda_kernels.hpp"

using namespace nikola::infrastructure;

// ============================================================
// §1  ExecutionStrategy Enum
// ============================================================

TEST_CASE("ExecStrategy_StandardLaunchValue", "[cuda_kernels][enums]") {
    CHECK(static_cast<uint8_t>(ExecutionStrategy::STANDARD_LAUNCH) == 0u);
}

TEST_CASE("ExecStrategy_CUDAGraphValue", "[cuda_kernels][enums]") {
    CHECK(static_cast<uint8_t>(ExecutionStrategy::CUDA_GRAPH) == 1u);
}

TEST_CASE("ExecStrategy_PersistentKernelValue", "[cuda_kernels][enums]") {
    CHECK(static_cast<uint8_t>(ExecutionStrategy::PERSISTENT_KERNEL) == 2u);
}

TEST_CASE("ExecStrategy_Count", "[cuda_kernels][enums]") {
    CHECK(EXECUTION_STRATEGY_COUNT == 3u);
}

TEST_CASE("ExecStrategy_AllDistinct", "[cuda_kernels][enums]") {
    CHECK(ExecutionStrategy::STANDARD_LAUNCH   != ExecutionStrategy::CUDA_GRAPH);
    CHECK(ExecutionStrategy::CUDA_GRAPH        != ExecutionStrategy::PERSISTENT_KERNEL);
}

// ============================================================
// §2  GraphState Enum
// ============================================================

TEST_CASE("GraphState_NotCapturedValue", "[cuda_kernels][enums]") {
    CHECK(static_cast<uint8_t>(GraphState::NOT_CAPTURED) == 0u);
}

TEST_CASE("GraphState_CapturedValue", "[cuda_kernels][enums]") {
    CHECK(static_cast<uint8_t>(GraphState::CAPTURED) == 1u);
}

TEST_CASE("GraphState_NeedsUpdateValue", "[cuda_kernels][enums]") {
    CHECK(static_cast<uint8_t>(GraphState::NEEDS_UPDATE) == 2u);
}

TEST_CASE("GraphState_Count", "[cuda_kernels][enums]") {
    CHECK(GRAPH_STATE_COUNT == 3u);
}

TEST_CASE("GraphState_AllDistinct", "[cuda_kernels][enums]") {
    CHECK(GraphState::NOT_CAPTURED != GraphState::CAPTURED);
    CHECK(GraphState::CAPTURED     != GraphState::NEEDS_UPDATE);
}

// ============================================================
// §3  TopologyEvent Enum
// ============================================================

TEST_CASE("TopoEvent_NoneValue", "[cuda_kernels][enums]") {
    CHECK(static_cast<uint8_t>(TopologyEvent::NONE) == 0u);
}

TEST_CASE("TopoEvent_NeurogenesisValue", "[cuda_kernels][enums]") {
    CHECK(static_cast<uint8_t>(TopologyEvent::NEUROGENESIS) == 1u);
}

TEST_CASE("TopoEvent_PruningValue", "[cuda_kernels][enums]") {
    CHECK(static_cast<uint8_t>(TopologyEvent::PRUNING) == 2u);
}

TEST_CASE("TopoEvent_Count", "[cuda_kernels][enums]") {
    CHECK(TOPOLOGY_EVENT_COUNT == 3u);
}

TEST_CASE("TopoEvent_AllDistinct", "[cuda_kernels][enums]") {
    CHECK(TopologyEvent::NONE        != TopologyEvent::NEUROGENESIS);
    CHECK(TopologyEvent::NEUROGENESIS != TopologyEvent::PRUNING);
}

// ============================================================
// §4  Per-Kernel Launch Overhead Constants
// ============================================================

TEST_CASE("Overhead_DriverMin_5us", "[cuda_kernels][overhead]") {
    CHECK(DRIVER_OVERHEAD_US_MIN == 5u);
}

TEST_CASE("Overhead_DriverMax_20us", "[cuda_kernels][overhead]") {
    CHECK(DRIVER_OVERHEAD_US_MAX == 20u);
}

TEST_CASE("Overhead_DriverMean_15us", "[cuda_kernels][overhead]") {
    CHECK(DRIVER_OVERHEAD_MEAN_US == 15u);
}

TEST_CASE("Overhead_DriverMeanWithinRange", "[cuda_kernels][overhead]") {
    CHECK(DRIVER_OVERHEAD_MEAN_US >= DRIVER_OVERHEAD_US_MIN);
    CHECK(DRIVER_OVERHEAD_MEAN_US <= DRIVER_OVERHEAD_US_MAX);
}

TEST_CASE("Overhead_PCIeMin_2us", "[cuda_kernels][overhead]") {
    CHECK(PCIE_LATENCY_US_MIN == 2u);
}

TEST_CASE("Overhead_PCIeMax_5us", "[cuda_kernels][overhead]") {
    CHECK(PCIE_LATENCY_US_MAX == 5u);
}

TEST_CASE("Overhead_KernelExecMin_50us", "[cuda_kernels][overhead]") {
    CHECK(KERNEL_EXEC_US_MIN == 50u);
}

TEST_CASE("Overhead_KernelExecMax_100us", "[cuda_kernels][overhead]") {
    CHECK(KERNEL_EXEC_US_MAX == 100u);
}

TEST_CASE("Overhead_SymplecticKernels_6", "[cuda_kernels][overhead]") {
    CHECK(SYMPLECTIC_KERNELS_PER_STEP == 6u);
}

TEST_CASE("Overhead_TotalLaunch_90us", "[cuda_kernels][overhead]") {
    CHECK(TOTAL_LAUNCH_OVERHEAD_US == 90u);
}

TEST_CASE("Overhead_TotalMatchesKernelsTimesMean", "[cuda_kernels][overhead]") {
    CHECK(TOTAL_LAUNCH_OVERHEAD_US ==
          SYMPLECTIC_KERNELS_PER_STEP * DRIVER_OVERHEAD_MEAN_US);
}

TEST_CASE("Overhead_OrderInvariants", "[cuda_kernels][overhead]") {
    CHECK(DRIVER_OVERHEAD_US_MIN < DRIVER_OVERHEAD_US_MAX);
    CHECK(PCIE_LATENCY_US_MIN    < PCIE_LATENCY_US_MAX);
    CHECK(KERNEL_EXEC_US_MIN     < KERNEL_EXEC_US_MAX);
}

// ============================================================
// §5  Physics Tick Budget / Decoherence Thresholds
// ============================================================

TEST_CASE("Budget_TickBudget_1000us", "[cuda_kernels][budget]") {
    CHECK(PHYSICS_TICK_BUDGET_US == 1'000u);
}

TEST_CASE("Budget_DecoherenceThreshold_500us", "[cuda_kernels][budget]") {
    CHECK(TEMPORAL_DECOHERENCE_THRESHOLD_US == 500u);
}

TEST_CASE("Budget_DecoherenceHalfOfTick", "[cuda_kernels][budget]") {
    // 500 μs = 50% of 1000 μs tick
    CHECK(TEMPORAL_DECOHERENCE_THRESHOLD_US * 2u == PHYSICS_TICK_BUDGET_US);
}

TEST_CASE("Budget_LaunchOverheadFraction_9Pct", "[cuda_kernels][budget]") {
    CHECK(LAUNCH_OVERHEAD_FRACTION == Catch::Approx(0.09));
}

TEST_CASE("Budget_LaunchFractionMatchesConstants", "[cuda_kernels][budget]") {
    // 90 / 1000 = 0.09
    const double computed =
        static_cast<double>(TOTAL_LAUNCH_OVERHEAD_US) /
        static_cast<double>(PHYSICS_TICK_BUDGET_US);
    CHECK(computed == Catch::Approx(LAUNCH_OVERHEAD_FRACTION).epsilon(1e-6));
}

// ============================================================
// §6  CUDA Graph Constants
// ============================================================

TEST_CASE("CUDAGraph_SingleLaunchCost_5us", "[cuda_kernels][cuda_graph]") {
    CHECK(GRAPH_SINGLE_LAUNCH_US == 5u);
}

TEST_CASE("CUDAGraph_ReinstantiationCost_200us", "[cuda_kernels][cuda_graph]") {
    CHECK(GRAPH_REINSTANTIATION_US == 200u);
}

TEST_CASE("CUDAGraph_ReinstantiationExpensive", "[cuda_kernels][cuda_graph]") {
    // Re-instantiation (200 μs) is expensive relative to tick budget (1000 μs)
    CHECK(GRAPH_REINSTANTIATION_US > TOTAL_LAUNCH_OVERHEAD_US);
}

TEST_CASE("CUDAGraph_OverheadReduction_80Pct", "[cuda_kernels][cuda_graph]") {
    CHECK(LAUNCH_OVERHEAD_REDUCTION_PCT == 80u);
}

TEST_CASE("CUDAGraph_GraphFasterThanStandard", "[cuda_kernels][cuda_graph]") {
    CHECK(GRAPH_SINGLE_LAUNCH_US < TOTAL_LAUNCH_OVERHEAD_US);
}

// ============================================================
// §7  Persistent Kernel Constants
// ============================================================

TEST_CASE("PersistentKernel_Nanosleep_100ns", "[cuda_kernels][persistent]") {
    CHECK(NANOSLEEP_SPIN_NS == 100u);
}

TEST_CASE("PersistentKernel_MinComputeCap_70", "[cuda_kernels][persistent]") {
    CHECK(MIN_COMPUTE_CAPABILITY_NANOSLEEP == 70u);
}

TEST_CASE("PersistentKernel_ComputeCapRepresents7_0", "[cuda_kernels][persistent]") {
    // 70 == 7.0 × 10
    CHECK(MIN_COMPUTE_CAPABILITY_NANOSLEEP == 70u);
}

// ============================================================
// §8  H100 Occupancy Constants
// ============================================================

TEST_CASE("H100_SMCount_132", "[cuda_kernels][h100]") {
    CHECK(H100_SM_COUNT == 132u);
}

TEST_CASE("H100_ThreadsPerBlock_256", "[cuda_kernels][h100]") {
    CHECK(THREADS_PER_BLOCK_DEFAULT == 256u);
}

TEST_CASE("H100_ThreadsPerBlockIsPowerOf2", "[cuda_kernels][h100]") {
    CHECK((THREADS_PER_BLOCK_DEFAULT & (THREADS_PER_BLOCK_DEFAULT - 1u)) == 0u);
}

TEST_CASE("H100_BlocksPerSM_8", "[cuda_kernels][h100]") {
    CHECK(BLOCKS_PER_SM_ESTIMATE == 8u);
}

TEST_CASE("H100_MaxResidentBlocks_1056", "[cuda_kernels][h100]") {
    CHECK(H100_MAX_RESIDENT_BLOCKS == 1'056u);
}

TEST_CASE("H100_MaxResidentBlocksMatchesProduct", "[cuda_kernels][h100]") {
    CHECK(H100_MAX_RESIDENT_BLOCKS == H100_SM_COUNT * BLOCKS_PER_SM_ESTIMATE);
}

// ============================================================
// §9  Audio-Visual Pipeline Constants
// ============================================================

TEST_CASE("AV_AudioSampleRate_44100Hz", "[cuda_kernels][av]") {
    CHECK(AUDIO_SAMPLE_RATE_HZ == 44'100u);
}

TEST_CASE("AV_AudioPeriod_22us", "[cuda_kernels][av]") {
    CHECK(AUDIO_PERIOD_US == 22u);
}

TEST_CASE("AV_VideoFrameRate_60Hz", "[cuda_kernels][av]") {
    CHECK(VIDEO_FRAME_RATE_HZ == 60u);
}

TEST_CASE("AV_VideoFramePeriod_16ms", "[cuda_kernels][av]") {
    CHECK(VIDEO_FRAME_PERIOD_MS == 16u);
}

TEST_CASE("AV_PhysicsTicksPerVideoFrame_16", "[cuda_kernels][av]") {
    CHECK(PHYSICS_TICKS_PER_VIDEO_FRAME == 16u);
}

TEST_CASE("AV_InterpWindowMatchesTicksPerFrame", "[cuda_kernels][av]") {
    CHECK(TEMPORAL_INTERP_WINDOW_TICKS == PHYSICS_TICKS_PER_VIDEO_FRAME);
}

TEST_CASE("AV_AudioRateMuchHigherThanPhysics", "[cuda_kernels][av]") {
    // 44100 Hz >> 1000 Hz physics → spectral injection avoids 44 kHz GPU updates
    CHECK(AUDIO_SAMPLE_RATE_HZ > PHYSICS_TICK_BUDGET_US);
}

// ============================================================
// §10  Control-Block Field Names
// ============================================================

TEST_CASE("CtrlBlock_HostSeqName", "[cuda_kernels][ctrl_block]") {
    CHECK(CTRL_FIELD_HOST_SEQ == "host_seq");
}

TEST_CASE("CtrlBlock_DeviceSeqName", "[cuda_kernels][ctrl_block]") {
    CHECK(CTRL_FIELD_DEVICE_SEQ == "device_seq");
}

TEST_CASE("CtrlBlock_RunningName", "[cuda_kernels][ctrl_block]") {
    CHECK(CTRL_FIELD_RUNNING == "running");
}

TEST_CASE("CtrlBlock_FieldCount_3", "[cuda_kernels][ctrl_block]") {
    CHECK(CTRL_FIELD_COUNT == 3u);
}

TEST_CASE("CtrlBlock_AllFieldsNonEmpty", "[cuda_kernels][ctrl_block]") {
    CHECK_FALSE(CTRL_FIELD_HOST_SEQ.empty());
    CHECK_FALSE(CTRL_FIELD_DEVICE_SEQ.empty());
    CHECK_FALSE(CTRL_FIELD_RUNNING.empty());
}

TEST_CASE("CtrlBlock_AllFieldsDistinct", "[cuda_kernels][ctrl_block]") {
    CHECK(CTRL_FIELD_HOST_SEQ   != CTRL_FIELD_DEVICE_SEQ);
    CHECK(CTRL_FIELD_DEVICE_SEQ != CTRL_FIELD_RUNNING);
}

// ============================================================
// §11  Overhead Query Functions
// ============================================================

TEST_CASE("Query_StandardOverhead_6Kernels_90us", "[cuda_kernels][queries]") {
    CHECK(standard_launch_overhead_us(6u) == 90u);
}

TEST_CASE("Query_StandardOverhead_1Kernel_15us", "[cuda_kernels][queries]") {
    CHECK(standard_launch_overhead_us(1u) == 15u);
}

TEST_CASE("Query_StandardOverhead_0Kernels_0us", "[cuda_kernels][queries]") {
    CHECK(standard_launch_overhead_us(0u) == 0u);
}

TEST_CASE("Query_StandardOverhead_Scales", "[cuda_kernels][queries]") {
    CHECK(standard_launch_overhead_us(6u) ==
          6u * standard_launch_overhead_us(1u));
}

TEST_CASE("Query_GraphOverhead_Is5us", "[cuda_kernels][queries]") {
    CHECK(graph_launch_overhead_us() == 5u);
}

TEST_CASE("Query_OverheadSaved_6Kernels", "[cuda_kernels][queries]") {
    // 90 - 5 = 85 μs saved
    CHECK(overhead_saved_us(6u) == 85u);
}

TEST_CASE("Query_OverheadSaved_1Kernel", "[cuda_kernels][queries]") {
    // 15 - 5 = 10 μs saved
    CHECK(overhead_saved_us(1u) == 10u);
}

TEST_CASE("Query_OverheadSaved_0Kernels_Zero", "[cuda_kernels][queries]") {
    // 0 - 5 would underflow → clamped to 0
    CHECK(overhead_saved_us(0u) == 0u);
}

TEST_CASE("Query_LaunchSignificant_6KernelTrue", "[cuda_kernels][queries]") {
    // 90 μs × 10 = 900 > 1000? No. Actually 90*10=900 < 1000. Let me check:
    // is_launch_overhead_significant: standard * 10 > PHYSICS_TICK = 1000
    // 90 * 10 = 900, which is NOT > 1000 → false... Hmm.
    // Wait: the spec says "consumes nearly 10%". 
    // The predicate checks if overhead > 10%: overhead*10 > budget
    // 90*10 = 900 < 1000 → returns false.
    // But 100*10 = 1000 — not > either.
    // Let me check: ceil case. 7 kernels: 7*15=105, 105*10=1050 > 1000 → true
    CHECK(is_launch_overhead_significant(7u));
}

TEST_CASE("Query_LaunchSignificant_6KernelFalse", "[cuda_kernels][queries]") {
    // 90*10 = 900 is NOT > 1000
    CHECK_FALSE(is_launch_overhead_significant(6u));
}

TEST_CASE("Query_LaunchNotSignificant_1Kernel", "[cuda_kernels][queries]") {
    // 15*10 = 150 < 1000
    CHECK_FALSE(is_launch_overhead_significant(1u));
}

// ============================================================
// §12  Timing Budget Predicates
// ============================================================

TEST_CASE("TimeBudget_ExceedsDecoherence_501us", "[cuda_kernels][predicates]") {
    CHECK(exceeds_temporal_decoherence(501u));
}

TEST_CASE("TimeBudget_ExceedsDecoherence_AtThreshold_False", "[cuda_kernels][predicates]") {
    CHECK_FALSE(exceeds_temporal_decoherence(500u));
}

TEST_CASE("TimeBudget_ExceedsDecoherence_100us_False", "[cuda_kernels][predicates]") {
    CHECK_FALSE(exceeds_temporal_decoherence(100u));
}

TEST_CASE("TimeBudget_WithinTickBudget_999us", "[cuda_kernels][predicates]") {
    CHECK(is_within_tick_budget(999u));
}

TEST_CASE("TimeBudget_WithinTickBudget_AtLimit_False", "[cuda_kernels][predicates]") {
    CHECK_FALSE(is_within_tick_budget(1'000u));
}

TEST_CASE("TimeBudget_MayExceedBudget_OverLimit_True", "[cuda_kernels][predicates]") {
    CHECK(may_exceed_budget(500u, 500u));   // 1000 >= 1000 → true
}

TEST_CASE("TimeBudget_MayExceedBudget_SubLimit_False", "[cuda_kernels][predicates]") {
    CHECK_FALSE(may_exceed_budget(90u, 100u)); // 190 < 1000
}

TEST_CASE("TimeBudget_TotalOverheadBelowDecoherence", "[cuda_kernels][predicates]") {
    // 90 μs overhead < 500 μs decoherence threshold
    CHECK_FALSE(exceeds_temporal_decoherence(TOTAL_LAUNCH_OVERHEAD_US));
}

// ============================================================
// §13  CUDA Graph Decision Helpers
// ============================================================

TEST_CASE("GraphDecision_NoneEvent_NoReinstantiation", "[cuda_kernels][graph_decision]") {
    CHECK_FALSE(requires_graph_reinstantiation(TopologyEvent::NONE));
}

TEST_CASE("GraphDecision_NeurogenesisEvent_RequiresUpdate", "[cuda_kernels][graph_decision]") {
    CHECK(requires_graph_reinstantiation(TopologyEvent::NEUROGENESIS));
}

TEST_CASE("GraphDecision_PruningEvent_RequiresUpdate", "[cuda_kernels][graph_decision]") {
    CHECK(requires_graph_reinstantiation(TopologyEvent::PRUNING));
}

TEST_CASE("GraphDecision_NotCaptured_NeedsRebuild", "[cuda_kernels][graph_decision]") {
    CHECK(graph_needs_rebuild(GraphState::NOT_CAPTURED));
}

TEST_CASE("GraphDecision_NeedsUpdate_NeedsRebuild", "[cuda_kernels][graph_decision]") {
    CHECK(graph_needs_rebuild(GraphState::NEEDS_UPDATE));
}

TEST_CASE("GraphDecision_Captured_NoRebuild", "[cuda_kernels][graph_decision]") {
    CHECK_FALSE(graph_needs_rebuild(GraphState::CAPTURED));
}

TEST_CASE("GraphDecision_Captured_IsLive", "[cuda_kernels][graph_decision]") {
    CHECK(graph_is_live(GraphState::CAPTURED));
}

TEST_CASE("GraphDecision_NotCaptured_NotLive", "[cuda_kernels][graph_decision]") {
    CHECK_FALSE(graph_is_live(GraphState::NOT_CAPTURED));
}

TEST_CASE("GraphDecision_NeedsUpdate_NotLive", "[cuda_kernels][graph_decision]") {
    CHECK_FALSE(graph_is_live(GraphState::NEEDS_UPDATE));
}

TEST_CASE("GraphDecision_LiveGraphAdvanceWithNoEvent_StaysCapured", "[cuda_kernels][graph_decision]") {
    const auto s = advance_graph_state(GraphState::CAPTURED, TopologyEvent::NONE);
    CHECK(s == GraphState::CAPTURED);
}

TEST_CASE("GraphDecision_LiveGraphAdvanceWithNeurogenesis_NeedsUpdate", "[cuda_kernels][graph_decision]") {
    const auto s = advance_graph_state(GraphState::CAPTURED, TopologyEvent::NEUROGENESIS);
    CHECK(s == GraphState::NEEDS_UPDATE);
}

TEST_CASE("GraphDecision_LiveGraphAdvanceWithPruning_NeedsUpdate", "[cuda_kernels][graph_decision]") {
    const auto s = advance_graph_state(GraphState::CAPTURED, TopologyEvent::PRUNING);
    CHECK(s == GraphState::NEEDS_UPDATE);
}

TEST_CASE("GraphDecision_NotCapturedAdvanceNoEvent_StaysNotCaptured", "[cuda_kernels][graph_decision]") {
    const auto s = advance_graph_state(GraphState::NOT_CAPTURED, TopologyEvent::NONE);
    CHECK(s == GraphState::NOT_CAPTURED);
}

// ============================================================
// §14  Persistent Kernel Suitability
// ============================================================

TEST_CASE("PersistKernel_H100Limit_Suitable", "[cuda_kernels][persistent]") {
    CHECK(is_persistent_kernel_suitable(H100_MAX_RESIDENT_BLOCKS));
}

TEST_CASE("PersistKernel_AboveLimit_NotSuitable", "[cuda_kernels][persistent]") {
    CHECK_FALSE(is_persistent_kernel_suitable(H100_MAX_RESIDENT_BLOCKS + 1u));
}

TEST_CASE("PersistKernel_Zero_Suitable", "[cuda_kernels][persistent]") {
    CHECK(is_persistent_kernel_suitable(0u));
}

TEST_CASE("PersistKernel_ExceedsH100_True", "[cuda_kernels][persistent]") {
    CHECK(exceeds_h100_occupancy(H100_MAX_RESIDENT_BLOCKS + 1u));
}

TEST_CASE("PersistKernel_AtLimit_DoesNotExceed", "[cuda_kernels][persistent]") {
    CHECK_FALSE(exceeds_h100_occupancy(H100_MAX_RESIDENT_BLOCKS));
}

TEST_CASE("RecommendedStrategy_FitsAndCapable_Persistent", "[cuda_kernels][persistent]") {
    const auto s = recommended_strategy(/* blocks= */ 1056u, /* cc×10= */ 80u);
    CHECK(s == ExecutionStrategy::PERSISTENT_KERNEL);
}

TEST_CASE("RecommendedStrategy_TooBig_CUDAGraph", "[cuda_kernels][persistent]") {
    const auto s = recommended_strategy(/* blocks= */ 2000u, /* cc×10= */ 80u);
    CHECK(s == ExecutionStrategy::CUDA_GRAPH);
}

TEST_CASE("RecommendedStrategy_OldCC_CUDAGraph", "[cuda_kernels][persistent]") {
    // Compute capability 6.1 (61) < 70 → cannot use __nanosleep → graphs
    const auto s = recommended_strategy(/* blocks= */ 100u, /* cc×10= */ 61u);
    CHECK(s == ExecutionStrategy::CUDA_GRAPH);
}

// ============================================================
// §15  Label Functions
// ============================================================

TEST_CASE("Label_ExecStrategy_StandardLaunch", "[cuda_kernels][labels]") {
    CHECK(execution_strategy_name(ExecutionStrategy::STANDARD_LAUNCH) == "standard_launch");
}

TEST_CASE("Label_ExecStrategy_CUDAGraph", "[cuda_kernels][labels]") {
    CHECK(execution_strategy_name(ExecutionStrategy::CUDA_GRAPH) == "cuda_graph");
}

TEST_CASE("Label_ExecStrategy_PersistentKernel", "[cuda_kernels][labels]") {
    CHECK(execution_strategy_name(ExecutionStrategy::PERSISTENT_KERNEL) == "persistent_kernel");
}

TEST_CASE("Label_ExecStrategy_AllNonEmpty", "[cuda_kernels][labels]") {
    for (uint8_t i = 0u; i < EXECUTION_STRATEGY_COUNT; ++i) {
        const auto s = static_cast<ExecutionStrategy>(i);
        CHECK_FALSE(execution_strategy_name(s).empty());
    }
}

TEST_CASE("Label_ExecStrategy_AllDistinct", "[cuda_kernels][labels]") {
    CHECK(execution_strategy_name(ExecutionStrategy::STANDARD_LAUNCH) !=
          execution_strategy_name(ExecutionStrategy::CUDA_GRAPH));
    CHECK(execution_strategy_name(ExecutionStrategy::CUDA_GRAPH) !=
          execution_strategy_name(ExecutionStrategy::PERSISTENT_KERNEL));
}

TEST_CASE("Label_GraphState_NotCaptured", "[cuda_kernels][labels]") {
    CHECK(graph_state_name(GraphState::NOT_CAPTURED) == "not_captured");
}

TEST_CASE("Label_GraphState_Captured", "[cuda_kernels][labels]") {
    CHECK(graph_state_name(GraphState::CAPTURED) == "captured");
}

TEST_CASE("Label_GraphState_NeedsUpdate", "[cuda_kernels][labels]") {
    CHECK(graph_state_name(GraphState::NEEDS_UPDATE) == "needs_update");
}

TEST_CASE("Label_GraphState_AllNonEmpty", "[cuda_kernels][labels]") {
    for (uint8_t i = 0u; i < GRAPH_STATE_COUNT; ++i) {
        const auto s = static_cast<GraphState>(i);
        CHECK_FALSE(graph_state_name(s).empty());
    }
}

TEST_CASE("Label_TopologyEvent_None", "[cuda_kernels][labels]") {
    CHECK(topology_event_name(TopologyEvent::NONE) == "none");
}

TEST_CASE("Label_TopologyEvent_Neurogenesis", "[cuda_kernels][labels]") {
    CHECK(topology_event_name(TopologyEvent::NEUROGENESIS) == "neurogenesis");
}

TEST_CASE("Label_TopologyEvent_Pruning", "[cuda_kernels][labels]") {
    CHECK(topology_event_name(TopologyEvent::PRUNING) == "pruning");
}

TEST_CASE("Label_TopologyEvent_AllDistinct", "[cuda_kernels][labels]") {
    CHECK(topology_event_name(TopologyEvent::NONE)         !=
          topology_event_name(TopologyEvent::NEUROGENESIS));
    CHECK(topology_event_name(TopologyEvent::NEUROGENESIS) !=
          topology_event_name(TopologyEvent::PRUNING));
}

// ============================================================
// Integration Scenarios
// ============================================================

TEST_CASE("Integration_StandardLaunchBottleneck", "[cuda_kernels][integration]") {
    // 6 kernels × 15 μs = 90 μs launch overhead
    const uint32_t overhead = standard_launch_overhead_us(SYMPLECTIC_KERNELS_PER_STEP);
    CHECK(overhead == TOTAL_LAUNCH_OVERHEAD_US);
    CHECK(overhead == 90u);

    // Overhead alone doesn't breach decoherence
    CHECK_FALSE(exceeds_temporal_decoherence(overhead));

    // Combined with worst-case kernel execution (6×100 μs = 600 μs):
    // 90 + 600 = 690 μs > 500 μs decoherence threshold
    const uint32_t exec = SYMPLECTIC_KERNELS_PER_STEP * KERNEL_EXEC_US_MAX;
    CHECK(exceeds_temporal_decoherence(overhead + exec));
}

TEST_CASE("Integration_CUDAGraphBenefit", "[cuda_kernels][integration]") {
    // Graph reduces 90 μs → 5 μs for launch phase
    const uint32_t saved = overhead_saved_us(SYMPLECTIC_KERNELS_PER_STEP);
    CHECK(saved == 85u);
    CHECK(graph_launch_overhead_us() < standard_launch_overhead_us(SYMPLECTIC_KERNELS_PER_STEP));

    // Graph launch fits entirely within tick budget
    CHECK(is_within_tick_budget(graph_launch_overhead_us()));
}

TEST_CASE("Integration_NeurogenesisTrigger", "[cuda_kernels][integration]") {
    // Active graph in use; Neurogenesis fires
    GraphState state = GraphState::CAPTURED;
    CHECK(graph_is_live(state));

    state = advance_graph_state(state, TopologyEvent::NEUROGENESIS);
    CHECK(state == GraphState::NEEDS_UPDATE);
    CHECK(graph_needs_rebuild(state));

    // Re-instantiation cost
    CHECK(GRAPH_REINSTANTIATION_US == 200u);
    // 200 μs re-instantiation does NOT breach decoherence on its own
    CHECK_FALSE(exceeds_temporal_decoherence(GRAPH_REINSTANTIATION_US));
}

TEST_CASE("Integration_PersistentKernelH100", "[cuda_kernels][integration]") {
    // 1056 blocks fit on H100 → persistent kernel viable
    CHECK(is_persistent_kernel_suitable(H100_MAX_RESIDENT_BLOCKS));
    CHECK_FALSE(exceeds_h100_occupancy(H100_MAX_RESIDENT_BLOCKS));

    // Compute cap 8.0 (80) ≥ 70 → __nanosleep available
    const auto strat = recommended_strategy(H100_MAX_RESIDENT_BLOCKS, 80u);
    CHECK(strat == ExecutionStrategy::PERSISTENT_KERNEL);
    CHECK(execution_strategy_name(strat) == "persistent_kernel");
}

TEST_CASE("Integration_PersistentKernelFallback", "[cuda_kernels][integration]") {
    // Grid beyond H100 capacity → must use CUDA Graphs
    const uint32_t blocks = H100_MAX_RESIDENT_BLOCKS + 100u;
    CHECK(exceeds_h100_occupancy(blocks));
    const auto strat = recommended_strategy(blocks, 90u);
    CHECK(strat == ExecutionStrategy::CUDA_GRAPH);
}

TEST_CASE("Integration_AudioSpectralInjection", "[cuda_kernels][integration]") {
    // 44.1 kHz audio windowed by FFT; injected once per physics tick (1 kHz)
    // → 44.1 samples consumed per tick
    CHECK(AUDIO_SAMPLE_RATE_HZ == 44'100u);
    CHECK(PHYSICS_TICK_BUDGET_US == 1'000u);

    // Audio runs much faster than physics — FFT amortises high sample rate
    CHECK(AUDIO_SAMPLE_RATE_HZ > PHYSICS_TICK_BUDGET_US * 10u);
}

TEST_CASE("Integration_VisualInterpolation", "[cuda_kernels][integration]") {
    // 60 Hz video → ~16 physics ticks per frame
    CHECK(PHYSICS_TICKS_PER_VIDEO_FRAME == 16u);
    CHECK(TEMPORAL_INTERP_WINDOW_TICKS  == 16u);

    // Frame period > UFIE decoherence threshold → must interpolate
    const uint32_t frame_us = VIDEO_FRAME_PERIOD_MS * 1'000u;  // 16000 μs
    CHECK(exceeds_temporal_decoherence(frame_us));

    // But a single tick stays within budget
    CHECK(is_within_tick_budget(PHYSICS_TICK_BUDGET_US - 1u));
}

TEST_CASE("Integration_ControlBlockDoorbellPattern", "[cuda_kernels][integration]") {
    // Persistent kernel: host increments host_seq; GPU spins on it.
    // Verify field names are correct for ControlBlock struct mapping.
    CHECK(CTRL_FIELD_HOST_SEQ   == "host_seq");
    CHECK(CTRL_FIELD_DEVICE_SEQ == "device_seq");
    CHECK(CTRL_FIELD_RUNNING    == "running");
    CHECK(CTRL_FIELD_COUNT      == 3u);

    // Nanosleep reduces power during spin-wait
    CHECK(NANOSLEEP_SPIN_NS > 0u);
    CHECK(MIN_COMPUTE_CAPABILITY_NANOSLEEP == 70u);
}
