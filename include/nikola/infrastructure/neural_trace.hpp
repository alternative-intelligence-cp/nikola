#pragma once

// ============================================================
// neural_trace.hpp — GAP-027b: Observability and Tracing Integration
//
// Documents the Nikola "Neural Trace" observability architecture:
//   • Tail-based "Interest" sampling strategy
//   • Semantic span attribute names (W3C / OTel convention)
//   • Prometheus metric identifiers and histogram buckets
//   • Performance budgets (span overhead, flush latency, eval budget)
//   • Trace-context propagation via NeuralSpike Protobuf field 16
//
// Namespace:   nikola::infrastructure
// Standard:    C++23, header-only, no external dependencies
// Source spec: GAP-027 — Observability and Tracing Integration
//              (Gemini Deep Research Round 2, Batch 25-27)
// ============================================================

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace nikola::infrastructure {

// ============================================================
// §1  Trace Context Protocol
// ============================================================

/// Protobuf field number for the OTel W3C trace-context map in NeuralSpike.
/// map<string, string> trace_context = 16;
inline constexpr uint32_t TRACE_CONTEXT_PROTO_FIELD = 16u;

/// Standard W3C traceparent key injected into the NeuralSpike trace_context map.
inline constexpr std::string_view TRACE_PARENT_KEY = "traceparent";

/// W3C traceparent version byte (always "00").
inline constexpr std::string_view TRACE_PARENT_VERSION = "00";

/// Minimum length of a valid W3C traceparent value string.
/// Format: "00-<32hex>-<16hex>-<2hex>"  → 55 characters.
inline constexpr std::size_t TRACE_PARENT_MIN_LENGTH = 55u;

// ============================================================
// §2  Semantic Span Attribute Keys
// ============================================================

/// Global resonance r ∈ [0,1]. Low = confusion / weak memory recall.
inline constexpr std::string_view SPAN_ATTR_RESONANCE = "nikola.resonance";

/// Total system Hamiltonian energy. Latency spikes correlate with
/// high-energy "epileptic" states where Ψ diverges.
inline constexpr std::string_view SPAN_ATTR_ENERGY_HAMILTONIAN = "nikola.energy.hamiltonian";

/// Count of new nodes created this cognitive cycle.
/// High count indicates a "Learning Spurt" causing latency.
inline constexpr std::string_view SPAN_ATTR_NEUROGENESIS_COUNT = "nikola.neurogenesis.count";

/// Current dopamine level. Explains reward-driven path selection.
inline constexpr std::string_view SPAN_ATTR_DOPAMINE = "nikola.neurochemistry.dopamine";

/// Morton Code (hex string) of the active region in the 9D manifold.
/// Indicates physically where in 9D space the current "thought" occurs.
inline constexpr std::string_view SPAN_ATTR_COORDINATES = "nikola.coordinates";

/// Total number of semantic span attribute keys.
inline constexpr std::size_t SPAN_ATTR_COUNT = 5u;

// ============================================================
// §3  Tail-Based "Interest" Sampling Thresholds
// ============================================================

/// Physics tick time above which a cognitive cycle is classified as
/// "High Latency" and marked interesting (triggers trace flush).
/// Unit: microseconds.
inline constexpr uint32_t INTEREST_LATENCY_THRESHOLD_US = 900u;

/// Maximum fractional deviation from energy conservation before a
/// cycle is classified as "High Energy Drift" (triggers trace flush).
/// Ratio: 0.01 % → 1e-4.
inline constexpr double INTEREST_ENERGY_DRIFT_THRESHOLD = 1.0e-4;

/// Dopamine level above which a cycle is classified as a
/// "High Reward / Eureka" event (triggers trace flush).
/// Value ∈ (0, 1].
inline constexpr float INTEREST_DOPAMINE_THRESHOLD = 0.8f;

// ============================================================
// §4  Interest Trigger Classification
// ============================================================

/// Reason a completed cognitive cycle was deemed "interesting"
/// and selected for trace flush to the Jaeger collector.
enum class InterestTrigger : uint8_t {
    NONE         = 0u,  ///< Routine cycle — traces overwritten, 0 bytes stored.
    HIGH_LATENCY = 1u,  ///< Tick time > INTEREST_LATENCY_THRESHOLD_US (900 μs).
    ENERGY_DRIFT = 2u,  ///< Energy conservation violation > 0.01 %.
    HIGH_REWARD  = 3u,  ///< Dopamine spike > INTEREST_DOPAMINE_THRESHOLD (0.8).
    ERROR        = 4u,  ///< Any component crash or exception.
};

/// Number of distinct InterestTrigger values (including NONE).
inline constexpr std::size_t INTEREST_TRIGGER_COUNT = 5u;

// ============================================================
// §5  Sampling Decision
// ============================================================

/// Decision produced by the Orchestrator's tail-based Interest evaluator.
enum class SamplingDecision : uint8_t {
    DISCARD = 0u,  ///< Cycle not interesting; local ring buffer overwritten.
    FLUSH   = 1u,  ///< Cycle interesting; publish FLUSH_TRACE command on Control Plane.
};

// ============================================================
// §6  Observability Backend Types
// ============================================================

/// External observability backends integrated with the Neural Trace system.
enum class ObservabilityBackend : uint8_t {
    JAEGER     = 0u,  ///< Trace waterfall visualisation of reasoning chains.
    PROMETHEUS = 1u,  ///< Aggregate time-series metrics (gauges, histograms).
};

/// Number of supported observability backends.
inline constexpr std::size_t OBSERVABILITY_BACKEND_COUNT = 2u;

// ============================================================
// §7  Prometheus Metric Identifiers
// ============================================================

/// Gauge: current count of active neural nodes. Monitors "brain" size.
inline constexpr std::string_view METRIC_ACTIVE_NODES_TOTAL = "nikola_active_nodes_total";

/// Histogram: physics tick duration in seconds. Identifies CFL violation frequency.
inline constexpr std::string_view METRIC_TICK_LATENCY_SECONDS =
    "nikola_physics_tick_latency_seconds";

/// Gauge: current dopamine level. Tracks agent emotional state over time.
inline constexpr std::string_view METRIC_DOPAMINE_LEVEL = "nikola_dopamine_level";

/// Total number of defined Prometheus metric names.
inline constexpr std::size_t PROMETHEUS_METRIC_COUNT = 3u;

// ============================================================
// §8  Prometheus Histogram Buckets (tick latency)
// ============================================================

/// Number of histogram bucket boundaries for nikola_physics_tick_latency_seconds.
inline constexpr std::size_t TICK_LATENCY_BUCKET_COUNT = 5u;

/// Upper-bound values (in microseconds) for the tick-latency histogram buckets.
/// Maps to seconds as: bucket_us / 1 000 000 for Prometheus exposition.
inline constexpr std::array<uint32_t, TICK_LATENCY_BUCKET_COUNT>
    TICK_LATENCY_BUCKETS_US = {100u, 500u, 900u, 1'000u, 5'000u};

// ============================================================
// §9  Performance Budgets
// ============================================================

/// Maximum overhead for generating a single local span (ring-buffer write).
/// Unit: microseconds. Must be < 1 μs.
inline constexpr uint32_t SPAN_GENERATION_MAX_US = 1u;

/// Maximum time allowed for a tail-sampled trace flush to the Jaeger collector.
/// Unit: milliseconds.
inline constexpr uint32_t TRACE_FLUSH_MAX_MS = 10u;

/// Maximum time allowed for the Orchestrator to evaluate the Interest heuristic.
/// Unit: microseconds.
inline constexpr uint32_t INTEREST_EVAL_MAX_US = 100u;

// ============================================================
// §10  Storage Efficiency Characteristics
// ============================================================

/// Bytes stored per routine (non-interesting) cognitive cycle.
/// Ring buffer is overwritten; no network traffic.
inline constexpr std::size_t ROUTINE_STORAGE_BYTES = 0u;

/// Minimum estimated bytes preserved for an interesting cognitive cycle.
/// Range: 1 KB – 10 KB per cycle.
inline constexpr std::size_t INTERESTING_EVENT_MIN_BYTES = 1'024u;

/// Maximum estimated bytes preserved for an interesting cognitive cycle.
inline constexpr std::size_t INTERESTING_EVENT_MAX_BYTES = 10'240u;

// ============================================================
// §11  Query Functions
// ============================================================

/// Returns true when the given tick duration exceeds the high-latency threshold.
[[nodiscard]] constexpr bool is_high_latency(uint32_t tick_us) noexcept {
    return tick_us > INTEREST_LATENCY_THRESHOLD_US;
}

/// Returns true when the fractional energy drift exceeds the conservation threshold.
[[nodiscard]] constexpr bool is_high_energy_drift(double drift_ratio) noexcept {
    return drift_ratio > INTEREST_ENERGY_DRIFT_THRESHOLD;
}

/// Returns true when the dopamine level triggers a "Eureka / High Reward" event.
[[nodiscard]] constexpr bool is_high_reward(float dopamine) noexcept {
    return dopamine > INTEREST_DOPAMINE_THRESHOLD;
}

/// Classifies a cognitive cycle into an InterestTrigger.
/// Priority order: ERROR > HIGH_LATENCY > HIGH_REWARD > ENERGY_DRIFT > NONE.
[[nodiscard]] constexpr InterestTrigger classify_interest(
    uint32_t tick_us,
    double   drift_ratio,
    float    dopamine,
    bool     has_error) noexcept
{
    if (has_error)                        return InterestTrigger::ERROR;
    if (is_high_latency(tick_us))         return InterestTrigger::HIGH_LATENCY;
    if (is_high_reward(dopamine))         return InterestTrigger::HIGH_REWARD;
    if (is_high_energy_drift(drift_ratio)) return InterestTrigger::ENERGY_DRIFT;
    return InterestTrigger::NONE;
}

/// Converts an InterestTrigger to the corresponding SamplingDecision.
[[nodiscard]] constexpr SamplingDecision sampling_decision(InterestTrigger trigger) noexcept {
    return (trigger == InterestTrigger::NONE) ? SamplingDecision::DISCARD
                                              : SamplingDecision::FLUSH;
}

/// Returns the histogram bucket index (0-based) for the given tick duration.
/// Returns TICK_LATENCY_BUCKET_COUNT if the value exceeds all defined buckets
/// (i.e., falls into the implicit +Inf bucket).
[[nodiscard]] constexpr std::size_t tick_latency_bucket(uint32_t tick_us) noexcept {
    for (std::size_t i = 0u; i < TICK_LATENCY_BUCKET_COUNT; ++i) {
        if (tick_us <= TICK_LATENCY_BUCKETS_US[i]) {
            return i;
        }
    }
    return TICK_LATENCY_BUCKET_COUNT;  // +Inf bucket
}

/// Validates that a traceparent string meets the minimum W3C length requirement.
[[nodiscard]] constexpr bool is_valid_traceparent(std::string_view value) noexcept {
    return value.size() >= TRACE_PARENT_MIN_LENGTH;
}

/// Returns true when a resonance value is within the valid normalised range [0, 1].
[[nodiscard]] constexpr bool is_valid_resonance(float resonance) noexcept {
    return resonance >= 0.0f && resonance <= 1.0f;
}

/// Returns true when a dopamine value is within the valid normalised range [0, 1].
[[nodiscard]] constexpr bool is_valid_dopamine(float dopamine) noexcept {
    return dopamine >= 0.0f && dopamine <= 1.0f;
}

/// Returns the human-readable name of a Prometheus metric for a given backend.
/// Only returns non-empty text for PROMETHEUS backend.
[[nodiscard]] constexpr std::string_view backend_label(ObservabilityBackend backend) noexcept {
    switch (backend) {
        case ObservabilityBackend::JAEGER:     return "jaeger";
        case ObservabilityBackend::PROMETHEUS: return "prometheus";
    }
    return "";
}

/// Returns the human-readable label for an InterestTrigger.
[[nodiscard]] constexpr std::string_view trigger_label(InterestTrigger trigger) noexcept {
    switch (trigger) {
        case InterestTrigger::NONE:         return "none";
        case InterestTrigger::HIGH_LATENCY: return "high_latency";
        case InterestTrigger::ENERGY_DRIFT: return "energy_drift";
        case InterestTrigger::HIGH_REWARD:  return "high_reward";
        case InterestTrigger::ERROR:        return "error";
    }
    return "";
}

}  // namespace nikola::infrastructure
