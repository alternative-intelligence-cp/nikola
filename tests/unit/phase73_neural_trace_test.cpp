// ============================================================
// phase73_neural_trace_test.cpp
//
// Unit tests for nikola/infrastructure/neural_trace.hpp
// GAP-027b: Observability and Tracing Integration
//
// Coverage:
//   §1  Trace context protocol constants
//   §2  Semantic span attribute keys
//   §3  Interest sampling thresholds
//   §4  InterestTrigger enum values and count
//   §5  SamplingDecision enum values
//   §6  ObservabilityBackend enum values and count
//   §7  Prometheus metric identifiers
//   §8  Prometheus histogram buckets
//   §9  Performance budgets
//   §10 Storage efficiency constants
//   §11 Query functions
//   Integration scenarios
// ============================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "nikola/infrastructure/neural_trace.hpp"

using namespace nikola::infrastructure;

// ============================================================
// §1  Trace Context Protocol Constants
// ============================================================

TEST_CASE("TraceContext_ProtoFieldNumber", "[neural_trace][trace_context]") {
    CHECK(TRACE_CONTEXT_PROTO_FIELD == 16u);
}

TEST_CASE("TraceContext_TraceParentKey", "[neural_trace][trace_context]") {
    CHECK(TRACE_PARENT_KEY == "traceparent");
}

TEST_CASE("TraceContext_VersionByte", "[neural_trace][trace_context]") {
    CHECK(TRACE_PARENT_VERSION == "00");
}

TEST_CASE("TraceContext_MinLength", "[neural_trace][trace_context]") {
    // "00-<32>-<16>-<2>" = 2+1+32+1+16+1+2 = 55
    CHECK(TRACE_PARENT_MIN_LENGTH == 55u);
}

TEST_CASE("TraceContext_ProtoFieldIsNonZero", "[neural_trace][trace_context]") {
    CHECK(TRACE_CONTEXT_PROTO_FIELD > 0u);
}

// ============================================================
// §2  Semantic Span Attribute Keys
// ============================================================

TEST_CASE("SpanAttr_ResonanceKey", "[neural_trace][span_attrs]") {
    CHECK(SPAN_ATTR_RESONANCE == "nikola.resonance");
}

TEST_CASE("SpanAttr_EnergyHamiltonianKey", "[neural_trace][span_attrs]") {
    CHECK(SPAN_ATTR_ENERGY_HAMILTONIAN == "nikola.energy.hamiltonian");
}

TEST_CASE("SpanAttr_NeurogenesisCountKey", "[neural_trace][span_attrs]") {
    CHECK(SPAN_ATTR_NEUROGENESIS_COUNT == "nikola.neurogenesis.count");
}

TEST_CASE("SpanAttr_DopamineKey", "[neural_trace][span_attrs]") {
    CHECK(SPAN_ATTR_DOPAMINE == "nikola.neurochemistry.dopamine");
}

TEST_CASE("SpanAttr_CoordinatesKey", "[neural_trace][span_attrs]") {
    CHECK(SPAN_ATTR_COORDINATES == "nikola.coordinates");
}

TEST_CASE("SpanAttr_AllKeysUseNikolaNamespace", "[neural_trace][span_attrs]") {
    auto starts_with = [](std::string_view str, std::string_view prefix) {
        return str.size() >= prefix.size() && str.substr(0, prefix.size()) == prefix;
    };
    const std::string_view nikola_ns = "nikola.";
    CHECK(starts_with(SPAN_ATTR_RESONANCE,          nikola_ns));
    CHECK(starts_with(SPAN_ATTR_ENERGY_HAMILTONIAN,  nikola_ns));
    CHECK(starts_with(SPAN_ATTR_NEUROGENESIS_COUNT,  nikola_ns));
    CHECK(starts_with(SPAN_ATTR_DOPAMINE,            nikola_ns));
    CHECK(starts_with(SPAN_ATTR_COORDINATES,         nikola_ns));
}

TEST_CASE("SpanAttr_Count", "[neural_trace][span_attrs]") {
    CHECK(SPAN_ATTR_COUNT == 5u);
}

TEST_CASE("SpanAttr_AllKeysNonEmpty", "[neural_trace][span_attrs]") {
    CHECK_FALSE(SPAN_ATTR_RESONANCE.empty());
    CHECK_FALSE(SPAN_ATTR_ENERGY_HAMILTONIAN.empty());
    CHECK_FALSE(SPAN_ATTR_NEUROGENESIS_COUNT.empty());
    CHECK_FALSE(SPAN_ATTR_DOPAMINE.empty());
    CHECK_FALSE(SPAN_ATTR_COORDINATES.empty());
}

// ============================================================
// §3  Interest Sampling Thresholds
// ============================================================

TEST_CASE("InterestThreshold_LatencyUs", "[neural_trace][thresholds]") {
    CHECK(INTEREST_LATENCY_THRESHOLD_US == 900u);
}

TEST_CASE("InterestThreshold_EnergyDrift", "[neural_trace][thresholds]") {
    CHECK(INTEREST_ENERGY_DRIFT_THRESHOLD == Catch::Approx(0.0001));
}

TEST_CASE("InterestThreshold_Dopamine", "[neural_trace][thresholds]") {
    CHECK(INTEREST_DOPAMINE_THRESHOLD == Catch::Approx(0.8f));
}

TEST_CASE("InterestThreshold_LatencyIsSubMillisecond", "[neural_trace][thresholds]") {
    // 900 μs < 1000 μs (1 ms)
    CHECK(INTEREST_LATENCY_THRESHOLD_US < 1000u);
}

TEST_CASE("InterestThreshold_EnergyDriftIsSmallFraction", "[neural_trace][thresholds]") {
    CHECK(INTEREST_ENERGY_DRIFT_THRESHOLD < 0.01);
    CHECK(INTEREST_ENERGY_DRIFT_THRESHOLD > 0.0);
}

TEST_CASE("InterestThreshold_DopamineInUnitRange", "[neural_trace][thresholds]") {
    CHECK(INTEREST_DOPAMINE_THRESHOLD > 0.0f);
    CHECK(INTEREST_DOPAMINE_THRESHOLD < 1.0f);
}

// ============================================================
// §4  InterestTrigger Enum
// ============================================================

TEST_CASE("InterestTrigger_NoneValue", "[neural_trace][trigger]") {
    CHECK(static_cast<uint8_t>(InterestTrigger::NONE) == 0u);
}

TEST_CASE("InterestTrigger_HighLatencyValue", "[neural_trace][trigger]") {
    CHECK(static_cast<uint8_t>(InterestTrigger::HIGH_LATENCY) == 1u);
}

TEST_CASE("InterestTrigger_EnergyDriftValue", "[neural_trace][trigger]") {
    CHECK(static_cast<uint8_t>(InterestTrigger::ENERGY_DRIFT) == 2u);
}

TEST_CASE("InterestTrigger_HighRewardValue", "[neural_trace][trigger]") {
    CHECK(static_cast<uint8_t>(InterestTrigger::HIGH_REWARD) == 3u);
}

TEST_CASE("InterestTrigger_ErrorValue", "[neural_trace][trigger]") {
    CHECK(static_cast<uint8_t>(InterestTrigger::ERROR) == 4u);
}

TEST_CASE("InterestTrigger_Count", "[neural_trace][trigger]") {
    CHECK(INTEREST_TRIGGER_COUNT == 5u);
}

// ============================================================
// §5  SamplingDecision Enum
// ============================================================

TEST_CASE("SamplingDecision_DiscardValue", "[neural_trace][sampling]") {
    CHECK(static_cast<uint8_t>(SamplingDecision::DISCARD) == 0u);
}

TEST_CASE("SamplingDecision_FlushValue", "[neural_trace][sampling]") {
    CHECK(static_cast<uint8_t>(SamplingDecision::FLUSH) == 1u);
}

TEST_CASE("SamplingDecision_DiscardNotFlush", "[neural_trace][sampling]") {
    CHECK(SamplingDecision::DISCARD != SamplingDecision::FLUSH);
}

// ============================================================
// §6  ObservabilityBackend Enum
// ============================================================

TEST_CASE("ObservabilityBackend_JaegerValue", "[neural_trace][backend]") {
    CHECK(static_cast<uint8_t>(ObservabilityBackend::JAEGER) == 0u);
}

TEST_CASE("ObservabilityBackend_PrometheusValue", "[neural_trace][backend]") {
    CHECK(static_cast<uint8_t>(ObservabilityBackend::PROMETHEUS) == 1u);
}

TEST_CASE("ObservabilityBackend_Count", "[neural_trace][backend]") {
    CHECK(OBSERVABILITY_BACKEND_COUNT == 2u);
}

TEST_CASE("ObservabilityBackend_DistinctValues", "[neural_trace][backend]") {
    CHECK(ObservabilityBackend::JAEGER != ObservabilityBackend::PROMETHEUS);
}

// ============================================================
// §7  Prometheus Metric Identifiers
// ============================================================

TEST_CASE("PrometheusMetrics_ActiveNodesTotal", "[neural_trace][prometheus]") {
    CHECK(METRIC_ACTIVE_NODES_TOTAL == "nikola_active_nodes_total");
}

TEST_CASE("PrometheusMetrics_TickLatencySeconds", "[neural_trace][prometheus]") {
    CHECK(METRIC_TICK_LATENCY_SECONDS == "nikola_physics_tick_latency_seconds");
}

TEST_CASE("PrometheusMetrics_DopamineLevel", "[neural_trace][prometheus]") {
    CHECK(METRIC_DOPAMINE_LEVEL == "nikola_dopamine_level");
}

TEST_CASE("PrometheusMetrics_Count", "[neural_trace][prometheus]") {
    CHECK(PROMETHEUS_METRIC_COUNT == 3u);
}

TEST_CASE("PrometheusMetrics_AllUseNikolaPrefix", "[neural_trace][prometheus]") {
    auto starts_with = [](std::string_view str, std::string_view prefix) {
        return str.size() >= prefix.size() && str.substr(0, prefix.size()) == prefix;
    };
    const std::string_view nikola_pfx = "nikola_";
    CHECK(starts_with(METRIC_ACTIVE_NODES_TOTAL,    nikola_pfx));
    CHECK(starts_with(METRIC_TICK_LATENCY_SECONDS,  nikola_pfx));
    CHECK(starts_with(METRIC_DOPAMINE_LEVEL,        nikola_pfx));
}

TEST_CASE("PrometheusMetrics_AllNonEmpty", "[neural_trace][prometheus]") {
    CHECK_FALSE(METRIC_ACTIVE_NODES_TOTAL.empty());
    CHECK_FALSE(METRIC_TICK_LATENCY_SECONDS.empty());
    CHECK_FALSE(METRIC_DOPAMINE_LEVEL.empty());
}

TEST_CASE("PrometheusMetrics_TickLatencyEndsWithSeconds", "[neural_trace][prometheus]") {
    // Prometheus convention for latency histograms
    const std::string_view metric = METRIC_TICK_LATENCY_SECONDS;
    const std::string_view suffix = "_seconds";
    CHECK(metric.size() > suffix.size());
    CHECK(metric.substr(metric.size() - suffix.size()) == suffix);
}

// ============================================================
// §8  Prometheus Histogram Buckets
// ============================================================

TEST_CASE("HistogramBuckets_Count", "[neural_trace][histogram]") {
    CHECK(TICK_LATENCY_BUCKET_COUNT == 5u);
}

TEST_CASE("HistogramBuckets_ArraySize", "[neural_trace][histogram]") {
    CHECK(TICK_LATENCY_BUCKETS_US.size() == TICK_LATENCY_BUCKET_COUNT);
}

TEST_CASE("HistogramBuckets_FirstBucket_100us", "[neural_trace][histogram]") {
    CHECK(TICK_LATENCY_BUCKETS_US[0] == 100u);
}

TEST_CASE("HistogramBuckets_SecondBucket_500us", "[neural_trace][histogram]") {
    CHECK(TICK_LATENCY_BUCKETS_US[1] == 500u);
}

TEST_CASE("HistogramBuckets_ThirdBucket_900us", "[neural_trace][histogram]") {
    CHECK(TICK_LATENCY_BUCKETS_US[2] == 900u);
}

TEST_CASE("HistogramBuckets_FourthBucket_1000us", "[neural_trace][histogram]") {
    CHECK(TICK_LATENCY_BUCKETS_US[3] == 1'000u);
}

TEST_CASE("HistogramBuckets_FifthBucket_5000us", "[neural_trace][histogram]") {
    CHECK(TICK_LATENCY_BUCKETS_US[4] == 5'000u);
}

TEST_CASE("HistogramBuckets_StrictlyIncreasing", "[neural_trace][histogram]") {
    for (std::size_t i = 1u; i < TICK_LATENCY_BUCKET_COUNT; ++i) {
        CHECK(TICK_LATENCY_BUCKETS_US[i] > TICK_LATENCY_BUCKETS_US[i - 1u]);
    }
}

TEST_CASE("HistogramBuckets_ThresholdBucketPresent", "[neural_trace][histogram]") {
    // The interest threshold (900 μs) must appear as a bucket boundary
    bool found = false;
    for (std::size_t i = 0u; i < TICK_LATENCY_BUCKET_COUNT; ++i) {
        if (TICK_LATENCY_BUCKETS_US[i] == INTEREST_LATENCY_THRESHOLD_US) {
            found = true;
        }
    }
    CHECK(found);
}

// ============================================================
// §9  Performance Budgets
// ============================================================

TEST_CASE("PerfBudget_SpanGenerationMax", "[neural_trace][perf]") {
    CHECK(SPAN_GENERATION_MAX_US == 1u);
}

TEST_CASE("PerfBudget_TraceFlushMax", "[neural_trace][perf]") {
    CHECK(TRACE_FLUSH_MAX_MS == 10u);
}

TEST_CASE("PerfBudget_InterestEvalMax", "[neural_trace][perf]") {
    CHECK(INTEREST_EVAL_MAX_US == 100u);
}

TEST_CASE("PerfBudget_SpanLessThanInterestEval", "[neural_trace][perf]") {
    // Span write should be faster than the full interest evaluation
    CHECK(SPAN_GENERATION_MAX_US < INTEREST_EVAL_MAX_US);
}

TEST_CASE("PerfBudget_InterestEvalLessThanThreshold", "[neural_trace][perf]") {
    // Evaluation overhead must be smaller than the threshold it gates
    CHECK(INTEREST_EVAL_MAX_US < INTEREST_LATENCY_THRESHOLD_US);
}

// ============================================================
// §10  Storage Efficiency Constants
// ============================================================

TEST_CASE("Storage_RoutineIsZeroBytes", "[neural_trace][storage]") {
    CHECK(ROUTINE_STORAGE_BYTES == 0u);
}

TEST_CASE("Storage_InterestingEventMinBytes", "[neural_trace][storage]") {
    CHECK(INTERESTING_EVENT_MIN_BYTES == 1'024u);   // 1 KB
}

TEST_CASE("Storage_InterestingEventMaxBytes", "[neural_trace][storage]") {
    CHECK(INTERESTING_EVENT_MAX_BYTES == 10'240u);  // 10 KB
}

TEST_CASE("Storage_MinLessThanMax", "[neural_trace][storage]") {
    CHECK(INTERESTING_EVENT_MIN_BYTES < INTERESTING_EVENT_MAX_BYTES);
}

TEST_CASE("Storage_RoutineLessThanMin", "[neural_trace][storage]") {
    CHECK(ROUTINE_STORAGE_BYTES < INTERESTING_EVENT_MIN_BYTES);
}

// ============================================================
// §11  Query Functions — is_high_latency
// ============================================================

TEST_CASE("IsHighLatency_ExactThreshold_False", "[neural_trace][query]") {
    // Strictly greater than; equal does NOT trigger
    CHECK_FALSE(is_high_latency(INTEREST_LATENCY_THRESHOLD_US));
}

TEST_CASE("IsHighLatency_OneBeyondThreshold_True", "[neural_trace][query]") {
    CHECK(is_high_latency(INTEREST_LATENCY_THRESHOLD_US + 1u));
}

TEST_CASE("IsHighLatency_Zero_False", "[neural_trace][query]") {
    CHECK_FALSE(is_high_latency(0u));
}

TEST_CASE("IsHighLatency_MaxUint32_True", "[neural_trace][query]") {
    CHECK(is_high_latency(UINT32_MAX));
}

TEST_CASE("IsHighLatency_899us_False", "[neural_trace][query]") {
    CHECK_FALSE(is_high_latency(899u));
}

TEST_CASE("IsHighLatency_1000us_True", "[neural_trace][query]") {
    CHECK(is_high_latency(1'000u));
}

// ============================================================
// §11  Query Functions — is_high_energy_drift
// ============================================================

TEST_CASE("IsHighEnergyDrift_ExactThreshold_False", "[neural_trace][query]") {
    CHECK_FALSE(is_high_energy_drift(INTEREST_ENERGY_DRIFT_THRESHOLD));
}

TEST_CASE("IsHighEnergyDrift_SlightlyAbove_True", "[neural_trace][query]") {
    CHECK(is_high_energy_drift(INTEREST_ENERGY_DRIFT_THRESHOLD + 1.0e-10));
}

TEST_CASE("IsHighEnergyDrift_Zero_False", "[neural_trace][query]") {
    CHECK_FALSE(is_high_energy_drift(0.0));
}

TEST_CASE("IsHighEnergyDrift_OnePercent_True", "[neural_trace][query]") {
    CHECK(is_high_energy_drift(0.01));
}

TEST_CASE("IsHighEnergyDrift_HalfThreshold_False", "[neural_trace][query]") {
    CHECK_FALSE(is_high_energy_drift(INTEREST_ENERGY_DRIFT_THRESHOLD / 2.0));
}

// ============================================================
// §11  Query Functions — is_high_reward
// ============================================================

TEST_CASE("IsHighReward_ExactThreshold_False", "[neural_trace][query]") {
    CHECK_FALSE(is_high_reward(INTEREST_DOPAMINE_THRESHOLD));
}

TEST_CASE("IsHighReward_SlightlyAbove_True", "[neural_trace][query]") {
    CHECK(is_high_reward(INTEREST_DOPAMINE_THRESHOLD + 0.001f));
}

TEST_CASE("IsHighReward_Zero_False", "[neural_trace][query]") {
    CHECK_FALSE(is_high_reward(0.0f));
}

TEST_CASE("IsHighReward_One_True", "[neural_trace][query]") {
    CHECK(is_high_reward(1.0f));
}

TEST_CASE("IsHighReward_HalfUnit_False", "[neural_trace][query]") {
    CHECK_FALSE(is_high_reward(0.5f));
}

// ============================================================
// §11  Query Functions — classify_interest
// ============================================================

TEST_CASE("ClassifyInterest_AllNormal_None", "[neural_trace][classify]") {
    InterestTrigger t = classify_interest(100u, 0.0, 0.0f, false);
    CHECK(t == InterestTrigger::NONE);
}

TEST_CASE("ClassifyInterest_HighLatency", "[neural_trace][classify]") {
    InterestTrigger t = classify_interest(901u, 0.0, 0.0f, false);
    CHECK(t == InterestTrigger::HIGH_LATENCY);
}

TEST_CASE("ClassifyInterest_HighReward", "[neural_trace][classify]") {
    InterestTrigger t = classify_interest(100u, 0.0, 0.9f, false);
    CHECK(t == InterestTrigger::HIGH_REWARD);
}

TEST_CASE("ClassifyInterest_EnergyDrift", "[neural_trace][classify]") {
    InterestTrigger t = classify_interest(100u, 0.001, 0.0f, false);
    CHECK(t == InterestTrigger::ENERGY_DRIFT);
}

TEST_CASE("ClassifyInterest_Error_PriorityOverLatency", "[neural_trace][classify]") {
    // Error overrides high latency
    InterestTrigger t = classify_interest(5000u, 1.0, 1.0f, true);
    CHECK(t == InterestTrigger::ERROR);
}

TEST_CASE("ClassifyInterest_HighLatency_PriorityOverReward", "[neural_trace][classify]") {
    InterestTrigger t = classify_interest(5000u, 0.0, 0.9f, false);
    CHECK(t == InterestTrigger::HIGH_LATENCY);
}

TEST_CASE("ClassifyInterest_HighReward_PriorityOverDrift", "[neural_trace][classify]") {
    InterestTrigger t = classify_interest(100u, 0.001, 0.9f, false);
    CHECK(t == InterestTrigger::HIGH_REWARD);
}

TEST_CASE("ClassifyInterest_ExactThresholds_None", "[neural_trace][classify]") {
    // Exact threshold values → not triggered
    InterestTrigger t = classify_interest(
        INTEREST_LATENCY_THRESHOLD_US,
        INTEREST_ENERGY_DRIFT_THRESHOLD,
        INTEREST_DOPAMINE_THRESHOLD,
        false);
    CHECK(t == InterestTrigger::NONE);
}

// ============================================================
// §11  Query Functions — sampling_decision
// ============================================================

TEST_CASE("SamplingDecision_None_Discard", "[neural_trace][sampling]") {
    CHECK(sampling_decision(InterestTrigger::NONE) == SamplingDecision::DISCARD);
}

TEST_CASE("SamplingDecision_HighLatency_Flush", "[neural_trace][sampling]") {
    CHECK(sampling_decision(InterestTrigger::HIGH_LATENCY) == SamplingDecision::FLUSH);
}

TEST_CASE("SamplingDecision_EnergyDrift_Flush", "[neural_trace][sampling]") {
    CHECK(sampling_decision(InterestTrigger::ENERGY_DRIFT) == SamplingDecision::FLUSH);
}

TEST_CASE("SamplingDecision_HighReward_Flush", "[neural_trace][sampling]") {
    CHECK(sampling_decision(InterestTrigger::HIGH_REWARD) == SamplingDecision::FLUSH);
}

TEST_CASE("SamplingDecision_Error_Flush", "[neural_trace][sampling]") {
    CHECK(sampling_decision(InterestTrigger::ERROR) == SamplingDecision::FLUSH);
}

// ============================================================
// §11  Query Functions — tick_latency_bucket
// ============================================================

TEST_CASE("TickBucket_ValueAtOrBelowFirst_BucketZero", "[neural_trace][histogram]") {
    CHECK(tick_latency_bucket(100u) == 0u);
}

TEST_CASE("TickBucket_ValueBelowFirst_BucketZero", "[neural_trace][histogram]") {
    CHECK(tick_latency_bucket(50u) == 0u);
}

TEST_CASE("TickBucket_ValueAtSecond_BucketOne", "[neural_trace][histogram]") {
    CHECK(tick_latency_bucket(500u) == 1u);
}

TEST_CASE("TickBucket_ValueAtThreshold_BucketTwo", "[neural_trace][histogram]") {
    // 900 μs = interest threshold, at the third bucket boundary
    CHECK(tick_latency_bucket(900u) == 2u);
}

TEST_CASE("TickBucket_ValueAtFourth_BucketThree", "[neural_trace][histogram]") {
    CHECK(tick_latency_bucket(1'000u) == 3u);
}

TEST_CASE("TickBucket_ValueAtFifth_BucketFour", "[neural_trace][histogram]") {
    CHECK(tick_latency_bucket(5'000u) == 4u);
}

TEST_CASE("TickBucket_ValueExceedsAll_InfBucket", "[neural_trace][histogram]") {
    CHECK(tick_latency_bucket(10'000u) == TICK_LATENCY_BUCKET_COUNT);
}

TEST_CASE("TickBucket_ZeroValue_BucketZero", "[neural_trace][histogram]") {
    CHECK(tick_latency_bucket(0u) == 0u);
}

// ============================================================
// §11  Query Functions — is_valid_traceparent
// ============================================================

TEST_CASE("ValidTraceParent_ExactMinLength_Valid", "[neural_trace][validate]") {
    // 55-char traceparent
    const std::string_view tp = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
    CHECK(tp.size() == 55u);
    CHECK(is_valid_traceparent(tp));
}

TEST_CASE("ValidTraceParent_LongerString_Valid", "[neural_trace][validate]") {
    // Longer is allowed
    const std::string_view tp = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-extra";
    CHECK(is_valid_traceparent(tp));
}

TEST_CASE("ValidTraceParent_EmptyString_Invalid", "[neural_trace][validate]") {
    CHECK_FALSE(is_valid_traceparent(""));
}

TEST_CASE("ValidTraceParent_ShortString_Invalid", "[neural_trace][validate]") {
    CHECK_FALSE(is_valid_traceparent("00-short"));
}

TEST_CASE("ValidTraceParent_54Chars_Invalid", "[neural_trace][validate]") {
    // One character short
    const std::string_view tp = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-0";
    CHECK(tp.size() == 54u);
    CHECK_FALSE(is_valid_traceparent(tp));
}

// ============================================================
// §11  Query Functions — is_valid_resonance / is_valid_dopamine
// ============================================================

TEST_CASE("ValidResonance_Zero_Valid", "[neural_trace][validate]") {
    CHECK(is_valid_resonance(0.0f));
}

TEST_CASE("ValidResonance_One_Valid", "[neural_trace][validate]") {
    CHECK(is_valid_resonance(1.0f));
}

TEST_CASE("ValidResonance_Half_Valid", "[neural_trace][validate]") {
    CHECK(is_valid_resonance(0.5f));
}

TEST_CASE("ValidResonance_Negative_Invalid", "[neural_trace][validate]") {
    CHECK_FALSE(is_valid_resonance(-0.01f));
}

TEST_CASE("ValidResonance_AboveOne_Invalid", "[neural_trace][validate]") {
    CHECK_FALSE(is_valid_resonance(1.01f));
}

TEST_CASE("ValidDopamine_Zero_Valid", "[neural_trace][validate]") {
    CHECK(is_valid_dopamine(0.0f));
}

TEST_CASE("ValidDopamine_One_Valid", "[neural_trace][validate]") {
    CHECK(is_valid_dopamine(1.0f));
}

TEST_CASE("ValidDopamine_Negative_Invalid", "[neural_trace][validate]") {
    CHECK_FALSE(is_valid_dopamine(-0.1f));
}

TEST_CASE("ValidDopamine_AboveOne_Invalid", "[neural_trace][validate]") {
    CHECK_FALSE(is_valid_dopamine(1.001f));
}

// ============================================================
// §11  Query Functions — backend_label
// ============================================================

TEST_CASE("BackendLabel_Jaeger", "[neural_trace][label]") {
    CHECK(backend_label(ObservabilityBackend::JAEGER) == "jaeger");
}

TEST_CASE("BackendLabel_Prometheus", "[neural_trace][label]") {
    CHECK(backend_label(ObservabilityBackend::PROMETHEUS) == "prometheus");
}

TEST_CASE("BackendLabel_AllNonEmpty", "[neural_trace][label]") {
    CHECK_FALSE(backend_label(ObservabilityBackend::JAEGER).empty());
    CHECK_FALSE(backend_label(ObservabilityBackend::PROMETHEUS).empty());
}

TEST_CASE("BackendLabel_AllDistinct", "[neural_trace][label]") {
    CHECK(backend_label(ObservabilityBackend::JAEGER) !=
          backend_label(ObservabilityBackend::PROMETHEUS));
}

// ============================================================
// §11  Query Functions — trigger_label
// ============================================================

TEST_CASE("TriggerLabel_None", "[neural_trace][label]") {
    CHECK(trigger_label(InterestTrigger::NONE) == "none");
}

TEST_CASE("TriggerLabel_HighLatency", "[neural_trace][label]") {
    CHECK(trigger_label(InterestTrigger::HIGH_LATENCY) == "high_latency");
}

TEST_CASE("TriggerLabel_EnergyDrift", "[neural_trace][label]") {
    CHECK(trigger_label(InterestTrigger::ENERGY_DRIFT) == "energy_drift");
}

TEST_CASE("TriggerLabel_HighReward", "[neural_trace][label]") {
    CHECK(trigger_label(InterestTrigger::HIGH_REWARD) == "high_reward");
}

TEST_CASE("TriggerLabel_Error", "[neural_trace][label]") {
    CHECK(trigger_label(InterestTrigger::ERROR) == "error");
}

TEST_CASE("TriggerLabel_AllNonEmpty", "[neural_trace][label]") {
    CHECK_FALSE(trigger_label(InterestTrigger::NONE).empty());
    CHECK_FALSE(trigger_label(InterestTrigger::HIGH_LATENCY).empty());
    CHECK_FALSE(trigger_label(InterestTrigger::ENERGY_DRIFT).empty());
    CHECK_FALSE(trigger_label(InterestTrigger::HIGH_REWARD).empty());
    CHECK_FALSE(trigger_label(InterestTrigger::ERROR).empty());
}

TEST_CASE("TriggerLabel_AllDistinct", "[neural_trace][label]") {
    const std::string_view labels[] = {
        trigger_label(InterestTrigger::NONE),
        trigger_label(InterestTrigger::HIGH_LATENCY),
        trigger_label(InterestTrigger::ENERGY_DRIFT),
        trigger_label(InterestTrigger::HIGH_REWARD),
        trigger_label(InterestTrigger::ERROR),
    };
    for (std::size_t i = 0; i < 5; ++i) {
        for (std::size_t j = i + 1; j < 5; ++j) {
            CHECK(labels[i] != labels[j]);
        }
    }
}

// ============================================================
// Integration Scenarios
// ============================================================

TEST_CASE("Integration_QuietCycle_DiscardedNoStorage", "[neural_trace][integration]") {
    // Routine cognitive cycle: fast tick, no violations, normal dopamine, no error
    const uint32_t tick_us    = 50u;
    const double   drift      = 0.0;
    const float    dopamine   = 0.3f;
    const bool     has_error  = false;

    const InterestTrigger trigger  = classify_interest(tick_us, drift, dopamine, has_error);
    const SamplingDecision decision = sampling_decision(trigger);

    CHECK(trigger == InterestTrigger::NONE);
    CHECK(decision == SamplingDecision::DISCARD);

    // Storage cost for discarded cycle
    CHECK(ROUTINE_STORAGE_BYTES == 0u);
}

TEST_CASE("Integration_EurekaMoment_FlushedToJaeger", "[neural_trace][integration]") {
    // Dopamine-driven "Eureka" — high reward triggers flush
    const uint32_t tick_us    = 300u;
    const double   drift      = 0.0;
    const float    dopamine   = 0.95f;
    const bool     has_error  = false;

    const InterestTrigger trigger  = classify_interest(tick_us, drift, dopamine, has_error);
    const SamplingDecision decision = sampling_decision(trigger);

    CHECK(trigger == InterestTrigger::HIGH_REWARD);
    CHECK(decision == SamplingDecision::FLUSH);
}

TEST_CASE("Integration_CrashEvent_FlushedWithHighestPriority", "[neural_trace][integration]") {
    // Error present alongside all other triggers — Error wins
    const InterestTrigger trigger = classify_interest(5000u, 1.0, 1.0f, true);
    CHECK(trigger == InterestTrigger::ERROR);
    CHECK(sampling_decision(trigger) == SamplingDecision::FLUSH);
}

TEST_CASE("Integration_PhysicsTickBucketing", "[neural_trace][integration]") {
    // Verify bucket assignment covers every range in the spec
    CHECK(tick_latency_bucket(0u)     == 0u);   // ≤100 μs
    CHECK(tick_latency_bucket(100u)   == 0u);   // bucket 0 boundary
    CHECK(tick_latency_bucket(101u)   == 1u);   // just beyond bucket 0
    CHECK(tick_latency_bucket(500u)   == 1u);   // bucket 1 boundary
    CHECK(tick_latency_bucket(501u)   == 2u);   // just beyond bucket 1
    CHECK(tick_latency_bucket(900u)   == 2u);   // bucket 2 = interest threshold
    CHECK(tick_latency_bucket(901u)   == 3u);   // just beyond threshold → bucket 3
    CHECK(tick_latency_bucket(1000u)  == 3u);   // bucket 3 boundary (1 ms)
    CHECK(tick_latency_bucket(1001u)  == 4u);   // just beyond 1 ms
    CHECK(tick_latency_bucket(5000u)  == 4u);   // bucket 4 boundary (5 ms)
    CHECK(tick_latency_bucket(5001u)  == TICK_LATENCY_BUCKET_COUNT);  // +Inf
}

TEST_CASE("Integration_TraceParentPropagation", "[neural_trace][integration]") {
    // Canonical W3C traceparent round-trip validation
    const std::string_view canonical =
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
    CHECK(canonical.size() == TRACE_PARENT_MIN_LENGTH);
    CHECK(is_valid_traceparent(canonical));

    // Verify key name injected into trace_context map
    CHECK(TRACE_PARENT_KEY == "traceparent");

    // Verify proto field number
    CHECK(TRACE_CONTEXT_PROTO_FIELD == 16u);
}

TEST_CASE("Integration_StorageBounds_InterestingEvent", "[neural_trace][integration]") {
    CHECK(INTERESTING_EVENT_MIN_BYTES > 0u);
    CHECK(INTERESTING_EVENT_MAX_BYTES >= INTERESTING_EVENT_MIN_BYTES);
    CHECK(INTERESTING_EVENT_MAX_BYTES <= 65536u);   // sanity: ≤64 KB per cycle
}

TEST_CASE("Integration_PerfBudgetHierarchy", "[neural_trace][integration]") {
    // Span write < interest eval < latency threshold < flush budget (ms)
    CHECK(SPAN_GENERATION_MAX_US < INTEREST_EVAL_MAX_US);
    CHECK(INTEREST_EVAL_MAX_US   < INTEREST_LATENCY_THRESHOLD_US);
    // Convert flush to μs: 10 ms = 10 000 μs — must exceed per-cycle thresholds
    CHECK(TRACE_FLUSH_MAX_MS * 1000u > INTEREST_LATENCY_THRESHOLD_US);
}
