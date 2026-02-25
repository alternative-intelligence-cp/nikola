// ============================================================
// nikola/infrastructure/cuda_kernels.hpp
//
// GAP-046: High-Frequency CUDA Kernel Optimization Strategies
//
// Encodes the CUDA optimisation specification:
//   §1   ExecutionStrategy enum  (STANDARD / CUDA_GRAPH / PERSISTENT)
//   §2   GraphState enum         (NOT_CAPTURED / CAPTURED / NEEDS_UPDATE)
//   §3   TopologyEvent enum      (NONE / NEUROGENESIS / PRUNING)
//   §4   Per-kernel launch overhead constants (driver, PCIe, execution)
//   §5   Physics tick budget and temporal-decoherence thresholds
//   §6   CUDA Graph constants (single-launch cost, re-instantiation cost,
//         reduction percentage)
//   §7   Persistent kernel constants (nanosleep, compute capability gate)
//   §8   H100 GPU occupancy constants
//   §9   Audio-visual pipeline constants
//   §10  Control-block field name constants
//   §11  Overhead query functions
//   §12  Timing budget predicates
//   §13  CUDA Graph decision helpers
//   §14  Persistent kernel suitability
//   §15  Label functions
// ============================================================

#pragma once

#include <cstdint>
#include <string_view>

namespace nikola::infrastructure {

// ============================================================
// §1  ExecutionStrategy Enum
// ============================================================

enum class ExecutionStrategy : uint8_t {
    STANDARD_LAUNCH    = 0,   ///< CPU issues each kernel call independently
    CUDA_GRAPH         = 1,   ///< Entire 5-6 step sequence via single graph launch
    PERSISTENT_KERNEL  = 2    ///< Mega-kernel spins on doorbell; no launch overhead
};

inline constexpr uint8_t EXECUTION_STRATEGY_COUNT = 3u;

// ============================================================
// §2  GraphState Enum
// ============================================================

enum class GraphState : uint8_t {
    NOT_CAPTURED  = 0,   ///< Graph has never been captured
    CAPTURED      = 1,   ///< Executable graph is live on GPU
    NEEDS_UPDATE  = 2    ///< Topology changed; re-instantiation required
};

inline constexpr uint8_t GRAPH_STATE_COUNT = 3u;

// ============================================================
// §3  TopologyEvent Enum
// ============================================================

enum class TopologyEvent : uint8_t {
    NONE        = 0,   ///< Grid stable; no re-capture needed
    NEUROGENESIS = 1,  ///< Active node count increased — must update graph
    PRUNING      = 2   ///< Active node count decreased — must update graph
};

inline constexpr uint8_t TOPOLOGY_EVENT_COUNT = 3u;

// ============================================================
// §4  Per-Kernel Launch Overhead Constants (all in microseconds)
// ============================================================

/// Minimum driver overhead per kernel launch.
inline constexpr uint32_t DRIVER_OVERHEAD_US_MIN   = 5u;

/// Maximum driver overhead per kernel launch.
inline constexpr uint32_t DRIVER_OVERHEAD_US_MAX   = 20u;

/// Typical / mean driver overhead used in overhead calculations.
inline constexpr uint32_t DRIVER_OVERHEAD_MEAN_US  = 15u;

/// Minimum PCIe command transmission latency per launch.
inline constexpr uint32_t PCIE_LATENCY_US_MIN      = 2u;

/// Maximum PCIe command transmission latency per launch.
inline constexpr uint32_t PCIE_LATENCY_US_MAX      = 5u;

/// Minimum execution time for a sparse-grid kernel.
inline constexpr uint32_t KERNEL_EXEC_US_MIN       = 50u;

/// Maximum execution time for a sparse-grid kernel.
inline constexpr uint32_t KERNEL_EXEC_US_MAX       = 100u;

/// Number of sequential kernel launches in one symplectic UFIE step
/// (Kinetic → Potential → Nonlinear → Damping + input/output kernels).
inline constexpr uint32_t SYMPLECTIC_KERNELS_PER_STEP = 6u;

/// Total per-tick overhead with standard launch model: 6 × 15 μs.
inline constexpr uint32_t TOTAL_LAUNCH_OVERHEAD_US    = 90u;

// ============================================================
// §5  Physics Tick Budget and Temporal-Decoherence Thresholds
// ============================================================

/// Full physics tick budget at 1000 Hz: 1000 μs.
inline constexpr uint32_t PHYSICS_TICK_BUDGET_US       = 1'000u;

/// Maximum tolerable end-to-end latency before "cognitive seizure".
inline constexpr uint32_t TEMPORAL_DECOHERENCE_THRESHOLD_US = 500u;

/// Fraction of tick budget consumed by standard kernel-launch overhead
/// (= TOTAL_LAUNCH_OVERHEAD_US / PHYSICS_TICK_BUDGET_US = 0.09).
inline constexpr double LAUNCH_OVERHEAD_FRACTION = 0.09;

// ============================================================
// §6  CUDA Graph Constants
// ============================================================

/// Cost of a single cudaGraphLaunch call that replaces 6 kernel launches.
inline constexpr uint32_t GRAPH_SINGLE_LAUNCH_US   = 5u;

/// Cost to re-capture/re-instantiate a graph after topology change.
inline constexpr uint32_t GRAPH_REINSTANTIATION_US = 200u;

/// Overhead reduction achieved by switching from standard to graph mode.
/// Standard = 90 μs; Graph = 5 μs → ~94% raw, spec states 80%.
inline constexpr uint32_t LAUNCH_OVERHEAD_REDUCTION_PCT = 80u;

// ============================================================
// §7  Persistent Kernel Constants
// ============================================================

/// Spin-wait sleep duration passed to __nanosleep() between doorbell polls.
inline constexpr uint32_t NANOSLEEP_SPIN_NS = 100u;

/// Minimum CUDA Compute Capability (as integer ×10) for __nanosleep().
/// "Requires Compute Capability 7.0+" → stored as 70.
inline constexpr uint32_t MIN_COMPUTE_CAPABILITY_NANOSLEEP = 70u;

// ============================================================
// §8  H100 GPU Occupancy Constants
// ============================================================

/// Number of Streaming Multiprocessors on NVIDIA H100.
inline constexpr uint32_t H100_SM_COUNT           = 132u;

/// Default threads-per-block for symplectic kernels.
inline constexpr uint32_t THREADS_PER_BLOCK_DEFAULT = 256u;

/// Estimated maximum concurrent thread-blocks per SM at 256 threads/block.
inline constexpr uint32_t BLOCKS_PER_SM_ESTIMATE  = 8u;

/// Maximum resident blocks on H100 available for a persistent kernel
/// (= H100_SM_COUNT × BLOCKS_PER_SM_ESTIMATE = 132 × 8 = 1056).
inline constexpr uint32_t H100_MAX_RESIDENT_BLOCKS =
    H100_SM_COUNT * BLOCKS_PER_SM_ESTIMATE;  // 1056

// ============================================================
// §9  Audio-Visual Pipeline Constants
// ============================================================

/// PCM audio sample rate for spectral injection into 9D Resonance dimension.
inline constexpr uint32_t AUDIO_SAMPLE_RATE_HZ    = 44'100u;

/// Approximate audio sample period in microseconds (1/44100 × 1e6 ≈ 22 μs).
inline constexpr uint32_t AUDIO_PERIOD_US         = 22u;

/// Video frame rate (Hz).
inline constexpr uint32_t VIDEO_FRAME_RATE_HZ     = 60u;

/// Approximate video frame period in milliseconds (1000/60 ≈ 16 ms).
inline constexpr uint32_t VIDEO_FRAME_PERIOD_MS   = 16u;

/// Physics ticks that elapse per video frame (1000 Hz / 60 Hz ≈ 16 ticks).
inline constexpr uint32_t PHYSICS_TICKS_PER_VIDEO_FRAME = 16u;

/// Temporal-interpolation window in physics ticks used to smooth visual
/// inputs and prevent step-function ripple in UFIE.
inline constexpr uint32_t TEMPORAL_INTERP_WINDOW_TICKS = 16u;

// ============================================================
// §10  Control-Block Field Name Constants
// ============================================================

/// CPU-side sequence counter name: incremented to trigger a GPU tick.
inline constexpr std::string_view CTRL_FIELD_HOST_SEQ    = "host_seq";

/// GPU-side acknowledgement counter name: incremented after processing.
inline constexpr std::string_view CTRL_FIELD_DEVICE_SEQ  = "device_seq";

/// Loop-termination flag name in the persistent-kernel ControlBlock.
inline constexpr std::string_view CTRL_FIELD_RUNNING     = "running";

inline constexpr uint8_t CTRL_FIELD_COUNT = 3u;

// ============================================================
// §11  Overhead Query Functions
// ============================================================

/// Estimated total launch overhead in μs for n_kernels standard launches.
[[nodiscard]] constexpr uint32_t
standard_launch_overhead_us(uint32_t n_kernels) noexcept {
    return n_kernels * DRIVER_OVERHEAD_MEAN_US;
}

/// CUDA-graph launch overhead (one cudaGraphLaunch replacing n_kernels).
[[nodiscard]] constexpr uint32_t graph_launch_overhead_us() noexcept {
    return GRAPH_SINGLE_LAUNCH_US;
}

/// μs saved per tick by switching from standard to CUDA graph mode.
[[nodiscard]] constexpr uint32_t
overhead_saved_us(uint32_t n_kernels) noexcept {
    const uint32_t standard = standard_launch_overhead_us(n_kernels);
    return standard > GRAPH_SINGLE_LAUNCH_US ? standard - GRAPH_SINGLE_LAUNCH_US : 0u;
}

/// Returns true when standard launch overhead takes > 10% of the tick budget.
[[nodiscard]] constexpr bool
is_launch_overhead_significant(uint32_t n_kernels) noexcept {
    return standard_launch_overhead_us(n_kernels) * 10u > PHYSICS_TICK_BUDGET_US;
}

// ============================================================
// §12  Timing Budget Predicates
// ============================================================

/// Returns true when latency has breached the Temporal Decoherence threshold.
[[nodiscard]] constexpr bool
exceeds_temporal_decoherence(uint32_t latency_us) noexcept {
    return latency_us > TEMPORAL_DECOHERENCE_THRESHOLD_US;
}

/// Returns true when elapsed time fits within the physics-tick budget.
[[nodiscard]] constexpr bool
is_within_tick_budget(uint32_t elapsed_us) noexcept {
    return elapsed_us < PHYSICS_TICK_BUDGET_US;
}

/// Returns true when overhead + kernel execution can exceed the tick budget.
[[nodiscard]] constexpr bool
may_exceed_budget(uint32_t overhead_us, uint32_t exec_us) noexcept {
    return (overhead_us + exec_us) >= PHYSICS_TICK_BUDGET_US;
}

// ============================================================
// §13  CUDA Graph Decision Helpers
// ============================================================

/// Returns true for any topology event that requires graph re-instantiation.
[[nodiscard]] constexpr bool
requires_graph_reinstantiation(TopologyEvent event) noexcept {
    return event != TopologyEvent::NONE;
}

/// Returns true when the graph state needs to be rebuilt before next launch.
[[nodiscard]] constexpr bool
graph_needs_rebuild(GraphState state) noexcept {
    return state == GraphState::NOT_CAPTURED || state == GraphState::NEEDS_UPDATE;
}

/// Returns true when the graph state is ready for immediate launch.
[[nodiscard]] constexpr bool graph_is_live(GraphState state) noexcept {
    return state == GraphState::CAPTURED;
}

/// Computes new GraphState after a topology event is observed.
[[nodiscard]] constexpr GraphState
advance_graph_state(GraphState current, TopologyEvent event) noexcept {
    if (event != TopologyEvent::NONE)   return GraphState::NEEDS_UPDATE;
    if (current == GraphState::NOT_CAPTURED) return GraphState::NOT_CAPTURED;
    return GraphState::CAPTURED;
}

// ============================================================
// §14  Persistent Kernel Suitability
// ============================================================

/// Returns true when the requested block count fits within H100 SM residency,
/// satisfying the cooperative-groups requirement for global synchronisation.
[[nodiscard]] constexpr bool
is_persistent_kernel_suitable(uint32_t block_count) noexcept {
    return block_count <= H100_MAX_RESIDENT_BLOCKS;
}

/// Returns true if block count exceeds H100 residency — must fall back to
/// CUDA Graphs.
[[nodiscard]] constexpr bool
exceeds_h100_occupancy(uint32_t block_count) noexcept {
    return block_count > H100_MAX_RESIDENT_BLOCKS;
}

/// Recommended strategy for a given block count and compute capability
/// (stored as capability × 10, e.g. 7.0 → 70).
[[nodiscard]] constexpr ExecutionStrategy
recommended_strategy(uint32_t block_count,
                     uint32_t compute_capability_x10) noexcept {
    if (block_count <= H100_MAX_RESIDENT_BLOCKS &&
        compute_capability_x10 >= MIN_COMPUTE_CAPABILITY_NANOSLEEP)
        return ExecutionStrategy::PERSISTENT_KERNEL;
    return ExecutionStrategy::CUDA_GRAPH;
}

// ============================================================
// §15  Label Functions
// ============================================================

[[nodiscard]] constexpr std::string_view
execution_strategy_name(ExecutionStrategy s) noexcept {
    switch (s) {
        case ExecutionStrategy::STANDARD_LAUNCH:   return "standard_launch";
        case ExecutionStrategy::CUDA_GRAPH:        return "cuda_graph";
        case ExecutionStrategy::PERSISTENT_KERNEL: return "persistent_kernel";
    }
    return "unknown";
}

[[nodiscard]] constexpr std::string_view
graph_state_name(GraphState s) noexcept {
    switch (s) {
        case GraphState::NOT_CAPTURED: return "not_captured";
        case GraphState::CAPTURED:     return "captured";
        case GraphState::NEEDS_UPDATE: return "needs_update";
    }
    return "unknown";
}

[[nodiscard]] constexpr std::string_view
topology_event_name(TopologyEvent e) noexcept {
    switch (e) {
        case TopologyEvent::NONE:         return "none";
        case TopologyEvent::NEUROGENESIS: return "neurogenesis";
        case TopologyEvent::PRUNING:      return "pruning";
    }
    return "unknown";
}

} // namespace nikola::infrastructure
