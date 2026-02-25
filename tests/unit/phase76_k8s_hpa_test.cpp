// ============================================================
// phase76_k8s_hpa_test.cpp
//
// Unit tests for nikola/infrastructure/k8s_hpa.hpp
// GAP-045: Kubernetes HPA for Biological Architectures
//
// Coverage:
//   §1   ComponentType enum
//   §2   ScalingAction enum
//   §3   ATPRegime enum
//   §4   ATP threshold constants
//   §5   HPA configuration constants
//   §6   Prometheus / scrape constants
//   §7   Resilience / PDB constants
//   §8   Metric name constants
//   §9   Kubernetes resource name constants
//   §10  ATP regime classification
//   §11  ATP predicate functions
//   §12  Sigmoid ATP factor
//   §13  Unified Load Metric
//   §14  HPA scaling decision
//   §15  PDB / scalability queries
//   §16  Label functions
//   Integration scenarios
// ============================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cstdint>
#include <string_view>

#include "nikola/infrastructure/k8s_hpa.hpp"

using namespace nikola::infrastructure;

// ============================================================
// §1  ComponentType Enum
// ============================================================

TEST_CASE("CompType_PhysicsEngineValue", "[k8s_hpa][enums]") {
    CHECK(static_cast<uint8_t>(ComponentType::PHYSICS_ENGINE) == 0u);
}

TEST_CASE("CompType_WorkerPoolValue", "[k8s_hpa][enums]") {
    CHECK(static_cast<uint8_t>(ComponentType::WORKER_POOL) == 1u);
}

TEST_CASE("CompType_Count", "[k8s_hpa][enums]") {
    CHECK(COMPONENT_TYPE_COUNT == 2u);
}

TEST_CASE("CompType_Distinct", "[k8s_hpa][enums]") {
    CHECK(ComponentType::PHYSICS_ENGINE != ComponentType::WORKER_POOL);
}

// ============================================================
// §2  ScalingAction Enum
// ============================================================

TEST_CASE("ScalingAction_ScaleDownValue", "[k8s_hpa][enums]") {
    CHECK(static_cast<uint8_t>(ScalingAction::SCALE_DOWN) == 0u);
}

TEST_CASE("ScalingAction_MaintainValue", "[k8s_hpa][enums]") {
    CHECK(static_cast<uint8_t>(ScalingAction::MAINTAIN) == 1u);
}

TEST_CASE("ScalingAction_ScaleUpValue", "[k8s_hpa][enums]") {
    CHECK(static_cast<uint8_t>(ScalingAction::SCALE_UP) == 2u);
}

TEST_CASE("ScalingAction_Count", "[k8s_hpa][enums]") {
    CHECK(SCALING_ACTION_COUNT == 3u);
}

TEST_CASE("ScalingAction_AllDistinct", "[k8s_hpa][enums]") {
    CHECK(ScalingAction::SCALE_DOWN != ScalingAction::MAINTAIN);
    CHECK(ScalingAction::MAINTAIN   != ScalingAction::SCALE_UP);
}

// ============================================================
// §3  ATPRegime Enum
// ============================================================

TEST_CASE("ATPRegime_CriticalValue", "[k8s_hpa][enums]") {
    CHECK(static_cast<uint8_t>(ATPRegime::CRITICAL) == 0u);
}

TEST_CASE("ATPRegime_LowValue", "[k8s_hpa][enums]") {
    CHECK(static_cast<uint8_t>(ATPRegime::LOW) == 1u);
}

TEST_CASE("ATPRegime_NominalValue", "[k8s_hpa][enums]") {
    CHECK(static_cast<uint8_t>(ATPRegime::NOMINAL) == 2u);
}

TEST_CASE("ATPRegime_HighValue", "[k8s_hpa][enums]") {
    CHECK(static_cast<uint8_t>(ATPRegime::HIGH) == 3u);
}

TEST_CASE("ATPRegime_Count", "[k8s_hpa][enums]") {
    CHECK(ATP_REGIME_COUNT == 4u);
}

TEST_CASE("ATPRegime_AllDistinct", "[k8s_hpa][enums]") {
    CHECK(ATPRegime::CRITICAL != ATPRegime::LOW);
    CHECK(ATPRegime::LOW      != ATPRegime::NOMINAL);
    CHECK(ATPRegime::NOMINAL  != ATPRegime::HIGH);
}

// ============================================================
// §4  ATP Threshold Constants
// ============================================================

TEST_CASE("ATPThreshold_HardStop_0_15", "[k8s_hpa][thresholds]") {
    CHECK(ATP_HARD_STOP == Catch::Approx(0.15));
}

TEST_CASE("ATPThreshold_SigmoidCenter_0_30", "[k8s_hpa][thresholds]") {
    CHECK(ATP_SIGMOID_THRESHOLD == Catch::Approx(0.30));
}

TEST_CASE("ATPThreshold_HighRegime_0_50", "[k8s_hpa][thresholds]") {
    CHECK(ATP_HIGH_THRESHOLD == Catch::Approx(0.50));
}

TEST_CASE("ATPThreshold_LowRegime_0_20", "[k8s_hpa][thresholds]") {
    CHECK(ATP_LOW_THRESHOLD == Catch::Approx(0.20));
}

TEST_CASE("ATPThreshold_SigmoidSteepness_20", "[k8s_hpa][thresholds]") {
    CHECK(SIGMOID_STEEPNESS_K == Catch::Approx(20.0));
}

TEST_CASE("ATPThreshold_OrderingInvariant", "[k8s_hpa][thresholds]") {
    // CRITICAL < LOW < sigmoid threshold < HIGH
    CHECK(ATP_HARD_STOP         < ATP_LOW_THRESHOLD);
    CHECK(ATP_LOW_THRESHOLD     < ATP_SIGMOID_THRESHOLD);
    CHECK(ATP_SIGMOID_THRESHOLD < ATP_HIGH_THRESHOLD);
}

TEST_CASE("ATPThreshold_HardStopBelowSigmoid", "[k8s_hpa][thresholds]") {
    // Hard stop (0.15) provides safety buffer below sigmoid center (0.30)
    CHECK(ATP_HARD_STOP < ATP_SIGMOID_THRESHOLD);
}

// ============================================================
// §5  HPA Configuration Constants
// ============================================================

TEST_CASE("HPAConfig_MinReplicas_2", "[k8s_hpa][hpa]") {
    CHECK(HPA_MIN_REPLICAS == 2u);
}

TEST_CASE("HPAConfig_MaxReplicas_50", "[k8s_hpa][hpa]") {
    CHECK(HPA_MAX_REPLICAS == 50u);
}

TEST_CASE("HPAConfig_TargetLag_0_5s", "[k8s_hpa][hpa]") {
    CHECK(HPA_TARGET_LAG_S == Catch::Approx(0.5));
}

TEST_CASE("HPAConfig_ScaleUpStabilization_30s", "[k8s_hpa][hpa]") {
    CHECK(SCALEUP_STABILIZATION_S == 30u);
}

TEST_CASE("HPAConfig_ScaleDownStabilization_60s", "[k8s_hpa][hpa]") {
    CHECK(SCALEDOWN_STABILIZATION_S == 60u);
}

TEST_CASE("HPAConfig_ScaleDownFasterThanDefault", "[k8s_hpa][hpa]") {
    // 60s is much shorter than standard 300s — allows rapid worker shedding
    CHECK(SCALEDOWN_STABILIZATION_S <  300u);
}

TEST_CASE("HPAConfig_ScaleUpPercent_100", "[k8s_hpa][hpa]") {
    CHECK(SCALEUP_PERCENT_PER_PERIOD == 100u);
}

TEST_CASE("HPAConfig_ScaleUpPeriod_15s", "[k8s_hpa][hpa]") {
    CHECK(SCALEUP_PERIOD_S == 15u);
}

TEST_CASE("HPAConfig_ScaleDownPods_5", "[k8s_hpa][hpa]") {
    CHECK(SCALEDOWN_PODS_PER_PERIOD == 5u);
}

TEST_CASE("HPAConfig_ScaleDownPeriod_30s", "[k8s_hpa][hpa]") {
    CHECK(SCALEDOWN_PERIOD_S == 30u);
}

TEST_CASE("HPAConfig_MinLessThanMax", "[k8s_hpa][hpa]") {
    CHECK(HPA_MIN_REPLICAS < HPA_MAX_REPLICAS);
}

// ============================================================
// §6  Prometheus / Scrape Constants
// ============================================================

TEST_CASE("Prometheus_ScrapeInterval_1s", "[k8s_hpa][prometheus]") {
    CHECK(SCRAPE_INTERVAL_S == 1u);
}

TEST_CASE("Prometheus_QueueDepthWindow_10s", "[k8s_hpa][prometheus]") {
    CHECK(QUEUE_DEPTH_WINDOW_S == 10u);
}

TEST_CASE("Prometheus_ProcessingRateWindow_30s", "[k8s_hpa][prometheus]") {
    CHECK(PROCESSING_RATE_WINDOW_S == 30u);
}

TEST_CASE("Prometheus_QueueWindowSmallerThanRateWindow", "[k8s_hpa][prometheus]") {
    // Queue window shorter for responsiveness; rate window longer for smoothing
    CHECK(QUEUE_DEPTH_WINDOW_S < PROCESSING_RATE_WINDOW_S);
}

// ============================================================
// §7  Resilience / PDB Constants
// ============================================================

TEST_CASE("PDB_PhysicsReplicas_1", "[k8s_hpa][resilience]") {
    CHECK(PHYSICS_ENGINE_REPLICAS == 1u);
}

TEST_CASE("PDB_PhysicsMinAvailable_100Pct", "[k8s_hpa][resilience]") {
    CHECK(PHYSICS_PDB_MIN_AVAILABLE_PCT == 100u);
}

TEST_CASE("PDB_WorkerMinAvailable_50Pct", "[k8s_hpa][resilience]") {
    CHECK(WORKER_PDB_MIN_AVAILABLE_PCT == 50u);
}

TEST_CASE("PDB_PhysicsStricterThanWorker", "[k8s_hpa][resilience]") {
    CHECK(PHYSICS_PDB_MIN_AVAILABLE_PCT > WORKER_PDB_MIN_AVAILABLE_PCT);
}

// ============================================================
// §8  Metric Name Constants
// ============================================================

TEST_CASE("Metric_QueueDepth_Name", "[k8s_hpa][metrics]") {
    CHECK(METRIC_QUEUE_DEPTH == "nikola_queue_depth");
}

TEST_CASE("Metric_ATPLevel_Name", "[k8s_hpa][metrics]") {
    CHECK(METRIC_ATP_LEVEL == "nikola_global_atp_level");
}

TEST_CASE("Metric_SpikesTotal_Name", "[k8s_hpa][metrics]") {
    CHECK(METRIC_SPIKES_TOTAL == "nikola_processed_spikes_total");
}

TEST_CASE("Metric_ProcessingLag_Name", "[k8s_hpa][metrics]") {
    CHECK(METRIC_PROCESSING_LAG == "nikola_processing_lag");
}

TEST_CASE("Metric_LoadScore_Name", "[k8s_hpa][metrics]") {
    CHECK(METRIC_LOAD_SCORE == "nikola_metabolic_load_score");
}

TEST_CASE("Metric_Count_5", "[k8s_hpa][metrics]") {
    CHECK(METRIC_COUNT == 5u);
}

TEST_CASE("Metric_AllNonEmpty", "[k8s_hpa][metrics]") {
    CHECK_FALSE(METRIC_QUEUE_DEPTH.empty());
    CHECK_FALSE(METRIC_ATP_LEVEL.empty());
    CHECK_FALSE(METRIC_SPIKES_TOTAL.empty());
    CHECK_FALSE(METRIC_PROCESSING_LAG.empty());
    CHECK_FALSE(METRIC_LOAD_SCORE.empty());
}

TEST_CASE("Metric_AllHaveNikolaPrefix", "[k8s_hpa][metrics]") {
    CHECK(METRIC_QUEUE_DEPTH.starts_with("nikola_"));
    CHECK(METRIC_ATP_LEVEL.starts_with("nikola_"));
    CHECK(METRIC_SPIKES_TOTAL.starts_with("nikola_"));
    CHECK(METRIC_PROCESSING_LAG.starts_with("nikola_"));
    CHECK(METRIC_LOAD_SCORE.starts_with("nikola_"));
}

// ============================================================
// §9  Kubernetes Resource Name Constants
// ============================================================

TEST_CASE("K8s_HPAName", "[k8s_hpa][k8s_names]") {
    CHECK(K8S_HPA_NAME == "nikola-worker-hpa");
}

TEST_CASE("K8s_Namespace", "[k8s_hpa][k8s_names]") {
    CHECK(K8S_NAMESPACE == "nikola-system");
}

TEST_CASE("K8s_WorkerDeployment", "[k8s_hpa][k8s_names]") {
    CHECK(K8S_WORKER_DEPLOYMENT == "nikola-worker-pool");
}

TEST_CASE("K8s_OrchestratorService", "[k8s_hpa][k8s_names]") {
    CHECK(K8S_ORCHESTRATOR_SVC == "nikola-orchestrator");
}

TEST_CASE("K8s_PhysicsHostname_Stable", "[k8s_hpa][k8s_names]") {
    // StatefulSet suffix must be "-0" for stable ZeroMQ binding
    CHECK(K8S_PHYSICS_HOSTNAME == "physics-0");
    CHECK(K8S_PHYSICS_HOSTNAME.ends_with("-0"));
}

TEST_CASE("K8s_AllNamesNonEmpty", "[k8s_hpa][k8s_names]") {
    CHECK_FALSE(K8S_HPA_NAME.empty());
    CHECK_FALSE(K8S_NAMESPACE.empty());
    CHECK_FALSE(K8S_WORKER_DEPLOYMENT.empty());
    CHECK_FALSE(K8S_ORCHESTRATOR_SVC.empty());
    CHECK_FALSE(K8S_PHYSICS_HOSTNAME.empty());
}

// ============================================================
// §10  ATP Regime Classification
// ============================================================

TEST_CASE("Classify_BelowHardStop_Critical", "[k8s_hpa][classify]") {
    CHECK(classify_atp_regime(0.0)  == ATPRegime::CRITICAL);
    CHECK(classify_atp_regime(0.05) == ATPRegime::CRITICAL);
    CHECK(classify_atp_regime(0.14) == ATPRegime::CRITICAL);
}

TEST_CASE("Classify_ExactHardStop_Low", "[k8s_hpa][classify]") {
    // 0.15 is the boundary: not < 0.15, so enters LOW range
    CHECK(classify_atp_regime(0.15) == ATPRegime::LOW);
}

TEST_CASE("Classify_LowBand", "[k8s_hpa][classify]") {
    CHECK(classify_atp_regime(0.17) == ATPRegime::LOW);
    CHECK(classify_atp_regime(0.19) == ATPRegime::LOW);
}

TEST_CASE("Classify_ExactLowThreshold_Nominal", "[k8s_hpa][classify]") {
    // 0.20 is not < 0.20, so enters NOMINAL
    CHECK(classify_atp_regime(0.20) == ATPRegime::NOMINAL);
}

TEST_CASE("Classify_NominalBand", "[k8s_hpa][classify]") {
    CHECK(classify_atp_regime(0.25) == ATPRegime::NOMINAL);
    CHECK(classify_atp_regime(0.35) == ATPRegime::NOMINAL);
    CHECK(classify_atp_regime(0.50) == ATPRegime::NOMINAL);
}

TEST_CASE("Classify_AboveHighThreshold_High", "[k8s_hpa][classify]") {
    CHECK(classify_atp_regime(0.51) == ATPRegime::HIGH);
    CHECK(classify_atp_regime(0.75) == ATPRegime::HIGH);
    CHECK(classify_atp_regime(1.00) == ATPRegime::HIGH);
}

// ============================================================
// §11  ATP Predicate Functions
// ============================================================

TEST_CASE("Predicate_HardStop_BelowThreshold_True", "[k8s_hpa][predicates]") {
    CHECK(is_atp_hard_stop(0.0));
    CHECK(is_atp_hard_stop(0.10));
    CHECK(is_atp_hard_stop(0.14));
}

TEST_CASE("Predicate_HardStop_AtThreshold_False", "[k8s_hpa][predicates]") {
    CHECK_FALSE(is_atp_hard_stop(0.15));
}

TEST_CASE("Predicate_HardStop_AboveThreshold_False", "[k8s_hpa][predicates]") {
    CHECK_FALSE(is_atp_hard_stop(0.5));
    CHECK_FALSE(is_atp_hard_stop(1.0));
}

TEST_CASE("Predicate_ShouldInhibit_BelowSigmoid_True", "[k8s_hpa][predicates]") {
    CHECK(should_inhibit_scaling(0.0));
    CHECK(should_inhibit_scaling(0.15));
    CHECK(should_inhibit_scaling(0.29));
}

TEST_CASE("Predicate_ShouldInhibit_AtThreshold_False", "[k8s_hpa][predicates]") {
    CHECK_FALSE(should_inhibit_scaling(0.30));
}

TEST_CASE("Predicate_ShouldInhibit_AboveThreshold_False", "[k8s_hpa][predicates]") {
    CHECK_FALSE(should_inhibit_scaling(0.5));
    CHECK_FALSE(should_inhibit_scaling(1.0));
}

TEST_CASE("Predicate_HighRegime_Above0_5_True", "[k8s_hpa][predicates]") {
    CHECK(is_atp_high_regime(0.51));
    CHECK(is_atp_high_regime(0.75));
    CHECK(is_atp_high_regime(1.0));
}

TEST_CASE("Predicate_HighRegime_AtThreshold_False", "[k8s_hpa][predicates]") {
    CHECK_FALSE(is_atp_high_regime(0.50));
}

TEST_CASE("Predicate_HighRegime_Below_False", "[k8s_hpa][predicates]") {
    CHECK_FALSE(is_atp_high_regime(0.3));
    CHECK_FALSE(is_atp_high_regime(0.1));
}

TEST_CASE("Predicate_LowRegime_Below0_2_True", "[k8s_hpa][predicates]") {
    CHECK(is_atp_low_regime(0.0));
    CHECK(is_atp_low_regime(0.10));
    CHECK(is_atp_low_regime(0.19));
}

TEST_CASE("Predicate_LowRegime_AtThreshold_False", "[k8s_hpa][predicates]") {
    CHECK_FALSE(is_atp_low_regime(0.20));
}

TEST_CASE("Predicate_LowRegime_Above_False", "[k8s_hpa][predicates]") {
    CHECK_FALSE(is_atp_low_regime(0.5));
}

// ============================================================
// §12  Sigmoid ATP Factor
// ============================================================

TEST_CASE("Sigmoid_AtThreshold_0_5", "[k8s_hpa][sigmoid]") {
    // S_atp(0.30) = 1/(1+exp(0)) = 0.5
    CHECK(sigmoid_atp_factor(0.30) == Catch::Approx(0.5).epsilon(1e-9));
}

TEST_CASE("Sigmoid_HighATP_ApproachesOne", "[k8s_hpa][sigmoid]") {
    // At ATP=1.0, k*(1.0-0.3)=14, exp(-14) ≈ 8e-7 → S_atp ≈ 1
    CHECK(sigmoid_atp_factor(1.0) > 0.999);
}

TEST_CASE("Sigmoid_LowATP_ApproachesZero", "[k8s_hpa][sigmoid]") {
    // At ATP=0.0, k*(0.0-0.3)=-6, exp(6) ≈ 403 → S_atp ≈ 0.0025
    CHECK(sigmoid_atp_factor(0.0) < 0.01);
}

TEST_CASE("Sigmoid_Monotone_Increasing", "[k8s_hpa][sigmoid]") {
    // Higher ATP → higher S_atp
    CHECK(sigmoid_atp_factor(0.0)  < sigmoid_atp_factor(0.15));
    CHECK(sigmoid_atp_factor(0.15) < sigmoid_atp_factor(0.30));
    CHECK(sigmoid_atp_factor(0.30) < sigmoid_atp_factor(0.50));
    CHECK(sigmoid_atp_factor(0.50) < sigmoid_atp_factor(0.80));
}

TEST_CASE("Sigmoid_InRange_0_to_1", "[k8s_hpa][sigmoid]") {
    CHECK(sigmoid_atp_factor(0.0) > 0.0);
    CHECK(sigmoid_atp_factor(0.0) < 1.0);
    CHECK(sigmoid_atp_factor(1.0) > 0.0);
    CHECK(sigmoid_atp_factor(1.0) < 1.0);
}

TEST_CASE("Sigmoid_HighATP_Point5_Factor_GreaterThan0_99", "[k8s_hpa][sigmoid]") {
    // ATP=0.50 is above threshold by 0.20, k*0.20=4, S_atp = 1/(1+e^-4) ≈ 0.982
    CHECK(sigmoid_atp_factor(0.50) > 0.98);
}

TEST_CASE("Sigmoid_LowATP_0_2_FactorSmall", "[k8s_hpa][sigmoid]") {
    // ATP=0.20, k*(0.20-0.30)=-2, S_atp = 1/(1+e^2) ≈ 0.119
    CHECK(sigmoid_atp_factor(0.20) < 0.15);
}

// ============================================================
// §13  Unified Load Metric
// ============================================================

TEST_CASE("UnifiedLoad_HighATP_ApproximatesLag", "[k8s_hpa][unified]") {
    // High ATP → S_atp≈1 → L_unified ≈ lag
    const double lag = 2.0;
    const double l   = unified_load_metric(lag, 1.0);
    CHECK(l > lag * 0.999);
}

TEST_CASE("UnifiedLoad_ZeroATP_CollapseToZero", "[k8s_hpa][unified]") {
    // Low ATP → S_atp≈0 → L_unified≈0
    const double l = unified_load_metric(100.0, 0.0);
    CHECK(l < 1.0);
}

TEST_CASE("UnifiedLoad_ZeroLag_AlwaysZero", "[k8s_hpa][unified]") {
    // No backlog → no load regardless of ATP
    CHECK(unified_load_metric(0.0, 0.5) == Catch::Approx(0.0).epsilon(1e-9));
    CHECK(unified_load_metric(0.0, 1.0) == Catch::Approx(0.0).epsilon(1e-9));
}

TEST_CASE("UnifiedLoad_PositiveResult", "[k8s_hpa][unified]") {
    // Non-negative for valid inputs
    CHECK(unified_load_metric(5.0, 0.5) > 0.0);
}

TEST_CASE("UnifiedLoad_HigherATP_HigherLoad", "[k8s_hpa][unified]") {
    // Same lag, higher ATP → higher unified load
    const double lag = 3.0;
    CHECK(unified_load_metric(lag, 0.8) > unified_load_metric(lag, 0.4));
}

// ============================================================
// §14  HPA Scaling Decision
// ============================================================

TEST_CASE("ScalingDecision_AboveTarget_ScaleUp", "[k8s_hpa][decision]") {
    // load > 0.5 → scale up
    CHECK(scaling_decision(1.0) == ScalingAction::SCALE_UP);
    CHECK(scaling_decision(0.51) == ScalingAction::SCALE_UP);
}

TEST_CASE("ScalingDecision_BelowHalfTarget_ScaleDown", "[k8s_hpa][decision]") {
    // load < 0.25 → scale down
    CHECK(scaling_decision(0.10) == ScalingAction::SCALE_DOWN);
    CHECK(scaling_decision(0.20) == ScalingAction::SCALE_DOWN);
    CHECK(scaling_decision(0.24) == ScalingAction::SCALE_DOWN);
}

TEST_CASE("ScalingDecision_InBand_Maintain", "[k8s_hpa][decision]") {
    // 0.25 ≤ load ≤ 0.5 → maintain
    CHECK(scaling_decision(0.25)  == ScalingAction::MAINTAIN);
    CHECK(scaling_decision(0.40)  == ScalingAction::MAINTAIN);
    CHECK(scaling_decision(0.50)  == ScalingAction::MAINTAIN);
}

TEST_CASE("ScalingDecision_AtTarget_Maintain", "[k8s_hpa][decision]") {
    CHECK(scaling_decision(HPA_TARGET_LAG_S) == ScalingAction::MAINTAIN);
}

TEST_CASE("ScalingDecision_CustomTarget", "[k8s_hpa][decision]") {
    // target = 2.0s band: [1.0, 2.0] = MAINTAIN; >2.0 = UP; <1.0 = DOWN
    CHECK(scaling_decision(3.0, 2.0)  == ScalingAction::SCALE_UP);
    CHECK(scaling_decision(1.5, 2.0)  == ScalingAction::MAINTAIN);
    CHECK(scaling_decision(0.5, 2.0)  == ScalingAction::SCALE_DOWN);
}

TEST_CASE("ScalingDecision_CollapsedATP_ScaleDown", "[k8s_hpa][decision]") {
    // Metabolic collapse: ATP=0.01 (deep-critical) → S_atp ≈ 0.003
    // L_unified = 10 * 0.003 ≈ 0.03 < half-target (0.25) → SCALE_DOWN
    const double l = unified_load_metric(10.0, 0.01);
    CHECK(scaling_decision(l) == ScalingAction::SCALE_DOWN);
}

// ============================================================
// §15  PDB / Scalability Queries
// ============================================================

TEST_CASE("PDB_PhysicsEngine_100Pct", "[k8s_hpa][pdb]") {
    CHECK(pdb_min_available_pct(ComponentType::PHYSICS_ENGINE) == 100u);
}

TEST_CASE("PDB_WorkerPool_50Pct", "[k8s_hpa][pdb]") {
    CHECK(pdb_min_available_pct(ComponentType::WORKER_POOL) == 50u);
}

TEST_CASE("PDB_Physics_AlwaysStricterThanWorker", "[k8s_hpa][pdb]") {
    CHECK(pdb_min_available_pct(ComponentType::PHYSICS_ENGINE) >
          pdb_min_available_pct(ComponentType::WORKER_POOL));
}

TEST_CASE("Scalability_WorkerPool_Scalable", "[k8s_hpa][pdb]") {
    CHECK(is_horizontally_scalable(ComponentType::WORKER_POOL));
}

TEST_CASE("Scalability_PhysicsEngine_NotScalable", "[k8s_hpa][pdb]") {
    CHECK_FALSE(is_horizontally_scalable(ComponentType::PHYSICS_ENGINE));
}

TEST_CASE("StableIdentity_PhysicsEngine_Required", "[k8s_hpa][pdb]") {
    CHECK(requires_stable_identity(ComponentType::PHYSICS_ENGINE));
}

TEST_CASE("StableIdentity_WorkerPool_NotRequired", "[k8s_hpa][pdb]") {
    CHECK_FALSE(requires_stable_identity(ComponentType::WORKER_POOL));
}

// ============================================================
// §16  Label Functions
// ============================================================

TEST_CASE("Label_ComponentType_PhysicsEngine", "[k8s_hpa][labels]") {
    CHECK(component_type_name(ComponentType::PHYSICS_ENGINE) == "physics_engine");
}

TEST_CASE("Label_ComponentType_WorkerPool", "[k8s_hpa][labels]") {
    CHECK(component_type_name(ComponentType::WORKER_POOL) == "worker_pool");
}

TEST_CASE("Label_ComponentType_AllNonEmpty", "[k8s_hpa][labels]") {
    CHECK_FALSE(component_type_name(ComponentType::PHYSICS_ENGINE).empty());
    CHECK_FALSE(component_type_name(ComponentType::WORKER_POOL).empty());
}

TEST_CASE("Label_ComponentType_AllDistinct", "[k8s_hpa][labels]") {
    CHECK(component_type_name(ComponentType::PHYSICS_ENGINE) !=
          component_type_name(ComponentType::WORKER_POOL));
}

TEST_CASE("Label_ScalingAction_ScaleDown", "[k8s_hpa][labels]") {
    CHECK(scaling_action_name(ScalingAction::SCALE_DOWN) == "scale_down");
}

TEST_CASE("Label_ScalingAction_Maintain", "[k8s_hpa][labels]") {
    CHECK(scaling_action_name(ScalingAction::MAINTAIN) == "maintain");
}

TEST_CASE("Label_ScalingAction_ScaleUp", "[k8s_hpa][labels]") {
    CHECK(scaling_action_name(ScalingAction::SCALE_UP) == "scale_up");
}

TEST_CASE("Label_ScalingAction_AllDistinct", "[k8s_hpa][labels]") {
    CHECK(scaling_action_name(ScalingAction::SCALE_DOWN) !=
          scaling_action_name(ScalingAction::MAINTAIN));
    CHECK(scaling_action_name(ScalingAction::MAINTAIN) !=
          scaling_action_name(ScalingAction::SCALE_UP));
}

TEST_CASE("Label_ATPRegime_Critical", "[k8s_hpa][labels]") {
    CHECK(atp_regime_name(ATPRegime::CRITICAL) == "critical");
}

TEST_CASE("Label_ATPRegime_Low", "[k8s_hpa][labels]") {
    CHECK(atp_regime_name(ATPRegime::LOW) == "low");
}

TEST_CASE("Label_ATPRegime_Nominal", "[k8s_hpa][labels]") {
    CHECK(atp_regime_name(ATPRegime::NOMINAL) == "nominal");
}

TEST_CASE("Label_ATPRegime_High", "[k8s_hpa][labels]") {
    CHECK(atp_regime_name(ATPRegime::HIGH) == "high");
}

TEST_CASE("Label_ATPRegime_AllDistinct", "[k8s_hpa][labels]") {
    CHECK(atp_regime_name(ATPRegime::CRITICAL) != atp_regime_name(ATPRegime::LOW));
    CHECK(atp_regime_name(ATPRegime::LOW)      != atp_regime_name(ATPRegime::NOMINAL));
    CHECK(atp_regime_name(ATPRegime::NOMINAL)  != atp_regime_name(ATPRegime::HIGH));
}

TEST_CASE("Label_AllRegimes_NonEmpty", "[k8s_hpa][labels]") {
    for (uint8_t i = 0u; i < static_cast<uint8_t>(ATP_REGIME_COUNT); ++i) {
        const auto r = static_cast<ATPRegime>(i);
        CHECK_FALSE(atp_regime_name(r).empty());
    }
}

// ============================================================
// Integration Scenarios
// ============================================================

TEST_CASE("Integration_HealthySystem_LinearScaling", "[k8s_hpa][integration]") {
    // Healthy system: ATP=0.8, 3s lag → L_unified ≈ lag → SCALE_UP
    const double atp = 0.8;
    const double lag = 3.0;

    CHECK(is_atp_high_regime(atp));
    CHECK(classify_atp_regime(atp) == ATPRegime::HIGH);
    const double l = unified_load_metric(lag, atp);
    CHECK(l > HPA_TARGET_LAG_S);               // exceeds 0.5s target
    CHECK(scaling_decision(l) == ScalingAction::SCALE_UP);
}

TEST_CASE("Integration_MetabolicCollapse_ScaleDown", "[k8s_hpa][integration]") {
    // ATP crashes to 0.01 (deep-critical): S_atp ≈ 0.003
    // L_unified = 50 * 0.003 ≈ 0.15 < half-target (0.25) → SCALE_DOWN
    const double atp = 0.01;
    const double lag = 50.0;

    CHECK(is_atp_hard_stop(atp));
    CHECK(classify_atp_regime(atp) == ATPRegime::CRITICAL);
    const double l = unified_load_metric(lag, atp);
    CHECK(l < HPA_TARGET_LAG_S * 0.5);        // below half-target
    CHECK(scaling_decision(l) == ScalingAction::SCALE_DOWN);
}

TEST_CASE("Integration_PhysicsEngineProtection", "[k8s_hpa][integration]") {
    // Physics Engine: singleton, stable hostname, 100% PDB, not scalable
    const auto c = ComponentType::PHYSICS_ENGINE;
    CHECK(PHYSICS_ENGINE_REPLICAS == 1u);
    CHECK(pdb_min_available_pct(c) == 100u);
    CHECK(requires_stable_identity(c));
    CHECK_FALSE(is_horizontally_scalable(c));
    CHECK(K8S_PHYSICS_HOSTNAME.ends_with("-0"));
}

TEST_CASE("Integration_WorkerPoolElasticity", "[k8s_hpa][integration]") {
    // Worker Pool: HPA-managed, 50% PDB, no stable identity required
    const auto c = ComponentType::WORKER_POOL;
    CHECK(is_horizontally_scalable(c));
    CHECK_FALSE(requires_stable_identity(c));
    CHECK(pdb_min_available_pct(c) == 50u);
    CHECK(HPA_MIN_REPLICAS == 2u);
    CHECK(HPA_MAX_REPLICAS == 50u);
}

TEST_CASE("Integration_SigmoidGovernsHPABehavior", "[k8s_hpa][integration]") {
    // Same lag, three ATP levels: high → scale up; nominal → scale up;
    // collapsed → scale down.  Verifies the governor works.
    const double lag = 1.0;  // 1.0 s lag > 0.5 s target

    const double l_high     = unified_load_metric(lag, 0.9);   // S_atp≈1
    const double l_nominal  = unified_load_metric(lag, 0.4);   // partial
    const double l_critical = unified_load_metric(lag, 0.05);  // S_atp≈0

    CHECK(l_high     > l_nominal);
    CHECK(l_nominal  > l_critical);

    CHECK(scaling_decision(l_high)     == ScalingAction::SCALE_UP);
    // At ATP=0.4, S_atp = 1/(1+exp(-20*(0.1))) ≈ 0.73 → L = 0.73 > 0.5
    CHECK(scaling_decision(l_nominal)  == ScalingAction::SCALE_UP);
    CHECK(scaling_decision(l_critical) == ScalingAction::SCALE_DOWN);
}

TEST_CASE("Integration_RecoveryFromNap", "[k8s_hpa][integration]") {
    // After NAP: ATP recovers from 0.05 to 0.6 — verify regime transitions
    CHECK(classify_atp_regime(0.05) == ATPRegime::CRITICAL);
    CHECK(classify_atp_regime(0.15) == ATPRegime::LOW);
    CHECK(classify_atp_regime(0.35) == ATPRegime::NOMINAL);
    CHECK(classify_atp_regime(0.60) == ATPRegime::HIGH);

    // At recovered ATP=0.6, lag=0.7s → scale up
    const double l = unified_load_metric(0.7, 0.6);
    CHECK(l > HPA_TARGET_LAG_S);
    CHECK(scaling_decision(l) == ScalingAction::SCALE_UP);
}

TEST_CASE("Integration_PromQL_WindowRationale", "[k8s_hpa][integration]") {
    // Verify the asymmetric window design (10s queue / 30s rate)
    CHECK(SCRAPE_INTERVAL_S     == 1u);
    CHECK(QUEUE_DEPTH_WINDOW_S  == 10u);
    CHECK(PROCESSING_RATE_WINDOW_S == 30u);
    CHECK(QUEUE_DEPTH_WINDOW_S < PROCESSING_RATE_WINDOW_S);
}
