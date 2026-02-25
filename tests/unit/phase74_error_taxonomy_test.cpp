// ============================================================
// phase74_error_taxonomy_test.cpp
//
// Unit tests for nikola/infrastructure/error_taxonomy.hpp
// GAP-042: Error Code Taxonomy and Handling Guide
//
// Coverage:
//   §1  Severity enum values and count
//   §2  ErrorCategory enum values and count
//   §3  RecoveryStrategy enum values and count
//   §4  ErrorCode enum values and per-category counts
//   §5  Fault-condition thresholds
//   §6  JSON structured-log field name constants
//   §7  category_of() — all 14 error codes
//   §8  severity_of() — all 14 error codes
//   §9  recovery_of() — all 14 error codes
//   §10 Predicate helpers
//   §11 Label functions (non-empty, distinct)
//   Integration scenarios
// ============================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "nikola/infrastructure/error_taxonomy.hpp"

using namespace nikola::infrastructure;

// ============================================================
// §1  Severity Enum
// ============================================================

TEST_CASE("Severity_CriticalValue", "[error_taxonomy][severity]") {
    CHECK(static_cast<uint8_t>(Severity::CRITICAL) == 0u);
}

TEST_CASE("Severity_HighValue", "[error_taxonomy][severity]") {
    CHECK(static_cast<uint8_t>(Severity::HIGH) == 1u);
}

TEST_CASE("Severity_MediumValue", "[error_taxonomy][severity]") {
    CHECK(static_cast<uint8_t>(Severity::MEDIUM) == 2u);
}

TEST_CASE("Severity_LowValue", "[error_taxonomy][severity]") {
    CHECK(static_cast<uint8_t>(Severity::LOW) == 3u);
}

TEST_CASE("Severity_Count", "[error_taxonomy][severity]") {
    CHECK(SEVERITY_COUNT == 4u);
}

TEST_CASE("Severity_AllDistinct", "[error_taxonomy][severity]") {
    CHECK(Severity::CRITICAL != Severity::HIGH);
    CHECK(Severity::HIGH     != Severity::MEDIUM);
    CHECK(Severity::MEDIUM   != Severity::LOW);
}

// ============================================================
// §2  ErrorCategory Enum
// ============================================================

TEST_CASE("ErrorCategory_INFValue", "[error_taxonomy][category]") {
    CHECK(static_cast<uint8_t>(ErrorCategory::INF) == 0u);
}

TEST_CASE("ErrorCategory_PHYValue", "[error_taxonomy][category]") {
    CHECK(static_cast<uint8_t>(ErrorCategory::PHY) == 1u);
}

TEST_CASE("ErrorCategory_COGValue", "[error_taxonomy][category]") {
    CHECK(static_cast<uint8_t>(ErrorCategory::COG) == 2u);
}

TEST_CASE("ErrorCategory_AUTOValue", "[error_taxonomy][category]") {
    CHECK(static_cast<uint8_t>(ErrorCategory::AUTO) == 3u);
}

TEST_CASE("ErrorCategory_Count", "[error_taxonomy][category]") {
    CHECK(ERROR_CATEGORY_COUNT == 4u);
}

// ============================================================
// §3  RecoveryStrategy Enum
// ============================================================

TEST_CASE("RecoveryStrategy_HardReset", "[error_taxonomy][recovery]") {
    CHECK(static_cast<uint8_t>(RecoveryStrategy::HARD_RESET) == 0u);
}

TEST_CASE("RecoveryStrategy_RePairing", "[error_taxonomy][recovery]") {
    CHECK(static_cast<uint8_t>(RecoveryStrategy::RE_PAIRING) == 1u);
}

TEST_CASE("RecoveryStrategy_Throttling", "[error_taxonomy][recovery]") {
    CHECK(static_cast<uint8_t>(RecoveryStrategy::THROTTLING) == 2u);
}

TEST_CASE("RecoveryStrategy_GarbageCollection", "[error_taxonomy][recovery]") {
    CHECK(static_cast<uint8_t>(RecoveryStrategy::GARBAGE_COLLECTION) == 3u);
}

TEST_CASE("RecoveryStrategy_SoftScram", "[error_taxonomy][recovery]") {
    CHECK(static_cast<uint8_t>(RecoveryStrategy::SOFT_SCRAM) == 4u);
}

TEST_CASE("RecoveryStrategy_StepReduction", "[error_taxonomy][recovery]") {
    CHECK(static_cast<uint8_t>(RecoveryStrategy::STEP_REDUCTION) == 5u);
}

TEST_CASE("RecoveryStrategy_Regularization", "[error_taxonomy][recovery]") {
    CHECK(static_cast<uint8_t>(RecoveryStrategy::REGULARIZATION) == 6u);
}

TEST_CASE("RecoveryStrategy_ReIgnition", "[error_taxonomy][recovery]") {
    CHECK(static_cast<uint8_t>(RecoveryStrategy::RE_IGNITION) == 7u);
}

TEST_CASE("RecoveryStrategy_AdminOverride", "[error_taxonomy][recovery]") {
    CHECK(static_cast<uint8_t>(RecoveryStrategy::ADMIN_OVERRIDE) == 8u);
}

TEST_CASE("RecoveryStrategy_StimulusInjection", "[error_taxonomy][recovery]") {
    CHECK(static_cast<uint8_t>(RecoveryStrategy::STIMULUS_INJECTION) == 9u);
}

TEST_CASE("RecoveryStrategy_ForcedNap", "[error_taxonomy][recovery]") {
    CHECK(static_cast<uint8_t>(RecoveryStrategy::FORCED_NAP) == 10u);
}

TEST_CASE("RecoveryStrategy_GoalPurge", "[error_taxonomy][recovery]") {
    CHECK(static_cast<uint8_t>(RecoveryStrategy::GOAL_PURGE) == 11u);
}

TEST_CASE("RecoveryStrategy_Masking", "[error_taxonomy][recovery]") {
    CHECK(static_cast<uint8_t>(RecoveryStrategy::MASKING) == 12u);
}

TEST_CASE("RecoveryStrategy_Count", "[error_taxonomy][recovery]") {
    CHECK(RECOVERY_STRATEGY_COUNT == 13u);
}

// ============================================================
// §4  ErrorCode Enum — Values and Subgroup Counts
// ============================================================

TEST_CASE("ErrorCode_INF001Value", "[error_taxonomy][errorcode]") {
    CHECK(static_cast<uint8_t>(ErrorCode::INF_001) == 0u);
}

TEST_CASE("ErrorCode_INF005Value", "[error_taxonomy][errorcode]") {
    CHECK(static_cast<uint8_t>(ErrorCode::INF_005) == 4u);
}

TEST_CASE("ErrorCode_PHY001Value", "[error_taxonomy][errorcode]") {
    CHECK(static_cast<uint8_t>(ErrorCode::PHY_001) == 5u);
}

TEST_CASE("ErrorCode_PHY004Value", "[error_taxonomy][errorcode]") {
    CHECK(static_cast<uint8_t>(ErrorCode::PHY_004) == 8u);
}

TEST_CASE("ErrorCode_COG001Value", "[error_taxonomy][errorcode]") {
    CHECK(static_cast<uint8_t>(ErrorCode::COG_001) == 9u);
}

TEST_CASE("ErrorCode_COG005Value", "[error_taxonomy][errorcode]") {
    CHECK(static_cast<uint8_t>(ErrorCode::COG_005) == 13u);
}

TEST_CASE("ErrorCode_TotalCount", "[error_taxonomy][errorcode]") {
    CHECK(ERROR_CODE_COUNT == 14u);
}

TEST_CASE("ErrorCode_INFSubgroupCount", "[error_taxonomy][errorcode]") {
    CHECK(INF_ERROR_COUNT == 5u);
}

TEST_CASE("ErrorCode_PHYSubgroupCount", "[error_taxonomy][errorcode]") {
    CHECK(PHY_ERROR_COUNT == 4u);
}

TEST_CASE("ErrorCode_COGSubgroupCount", "[error_taxonomy][errorcode]") {
    CHECK(COG_ERROR_COUNT == 5u);
}

TEST_CASE("ErrorCode_SubgroupsSumToTotal", "[error_taxonomy][errorcode]") {
    CHECK(INF_ERROR_COUNT + PHY_ERROR_COUNT + COG_ERROR_COUNT == ERROR_CODE_COUNT);
}

// ============================================================
// §5  Fault-Condition Thresholds
// ============================================================

TEST_CASE("Threshold_TemporalDecoherence_50ms", "[error_taxonomy][threshold]") {
    CHECK(TEMPORAL_DECOHERENCE_THRESHOLD_MS == 50u);
}

TEST_CASE("Threshold_HeartbeatFailure_500ms", "[error_taxonomy][threshold]") {
    CHECK(HEARTBEAT_FAILURE_THRESHOLD_MS == 500u);
}

TEST_CASE("Threshold_EnergyDriftRatio", "[error_taxonomy][threshold]") {
    CHECK(ENERGY_DRIFT_MAX_RATIO == Catch::Approx(1.0e-4));
}

TEST_CASE("Threshold_EnergyDriftEvalSteps", "[error_taxonomy][threshold]") {
    CHECK(ENERGY_DRIFT_EVAL_STEPS == 100u);
}

TEST_CASE("Threshold_SoftScramDamping_Half", "[error_taxonomy][threshold]") {
    CHECK(SOFT_SCRAM_DAMPING == Catch::Approx(0.5));
}

TEST_CASE("Threshold_StepReductionFactor_Half", "[error_taxonomy][threshold]") {
    CHECK(STEP_REDUCTION_FACTOR == Catch::Approx(0.5));
}

TEST_CASE("Threshold_ATPExhaustion_5Percent", "[error_taxonomy][threshold]") {
    CHECK(ATP_EXHAUSTION_THRESHOLD == Catch::Approx(0.05));
}

TEST_CASE("Threshold_HeartbeatGreaterThanDecoherence", "[error_taxonomy][threshold]") {
    // 500 ms >> 50 ms — heartbeat window is deliberately wider
    CHECK(HEARTBEAT_FAILURE_THRESHOLD_MS > TEMPORAL_DECOHERENCE_THRESHOLD_MS);
}

TEST_CASE("Threshold_EnergyDriftIsSmall", "[error_taxonomy][threshold]") {
    CHECK(ENERGY_DRIFT_MAX_RATIO < 0.01);
    CHECK(ENERGY_DRIFT_MAX_RATIO > 0.0);
}

TEST_CASE("Threshold_SoftScramAndStepReductionBothHalf", "[error_taxonomy][threshold]") {
    CHECK(SOFT_SCRAM_DAMPING == Catch::Approx(STEP_REDUCTION_FACTOR));
}

// ============================================================
// §6  JSON Log Field Names
// ============================================================

TEST_CASE("LogField_SchemaURI", "[error_taxonomy][log]") {
    CHECK(LOG_SCHEMA_URI == "http://nikola-agi.com/schemas/v0.0.4/log-entry.json");
}

TEST_CASE("LogField_Timestamp", "[error_taxonomy][log]") {
    CHECK(LOG_FIELD_TIMESTAMP == "timestamp");
}

TEST_CASE("LogField_Level", "[error_taxonomy][log]") {
    CHECK(LOG_FIELD_LEVEL == "level");
}

TEST_CASE("LogField_ComponentId", "[error_taxonomy][log]") {
    CHECK(LOG_FIELD_COMPONENT_ID == "component_id");
}

TEST_CASE("LogField_ErrorCode", "[error_taxonomy][log]") {
    CHECK(LOG_FIELD_ERROR_CODE == "error_code");
}

TEST_CASE("LogField_Message", "[error_taxonomy][log]") {
    CHECK(LOG_FIELD_MESSAGE == "message");
}

TEST_CASE("LogField_Context", "[error_taxonomy][log]") {
    CHECK(LOG_FIELD_CONTEXT == "context");
}

TEST_CASE("LogField_RecoveryAction", "[error_taxonomy][log]") {
    CHECK(LOG_FIELD_RECOVERY_ACTION == "recovery_action");
}

TEST_CASE("LogField_TraceId", "[error_taxonomy][log]") {
    CHECK(LOG_FIELD_TRACE_ID == "trace_id");
}

TEST_CASE("LogField_Count", "[error_taxonomy][log]") {
    CHECK(LOG_FIELD_COUNT == 8u);
}

TEST_CASE("LogField_AllNonEmpty", "[error_taxonomy][log]") {
    CHECK_FALSE(LOG_FIELD_TIMESTAMP.empty());
    CHECK_FALSE(LOG_FIELD_LEVEL.empty());
    CHECK_FALSE(LOG_FIELD_COMPONENT_ID.empty());
    CHECK_FALSE(LOG_FIELD_ERROR_CODE.empty());
    CHECK_FALSE(LOG_FIELD_MESSAGE.empty());
    CHECK_FALSE(LOG_FIELD_CONTEXT.empty());
    CHECK_FALSE(LOG_FIELD_RECOVERY_ACTION.empty());
    CHECK_FALSE(LOG_FIELD_TRACE_ID.empty());
}

// ============================================================
// §7  category_of() — all 14 error codes
// ============================================================

TEST_CASE("CategoryOf_INF001_INF", "[error_taxonomy][category_of]") {
    CHECK(category_of(ErrorCode::INF_001) == ErrorCategory::INF);
}

TEST_CASE("CategoryOf_INF002_INF", "[error_taxonomy][category_of]") {
    CHECK(category_of(ErrorCode::INF_002) == ErrorCategory::INF);
}

TEST_CASE("CategoryOf_INF003_INF", "[error_taxonomy][category_of]") {
    CHECK(category_of(ErrorCode::INF_003) == ErrorCategory::INF);
}

TEST_CASE("CategoryOf_INF004_INF", "[error_taxonomy][category_of]") {
    CHECK(category_of(ErrorCode::INF_004) == ErrorCategory::INF);
}

TEST_CASE("CategoryOf_INF005_INF", "[error_taxonomy][category_of]") {
    CHECK(category_of(ErrorCode::INF_005) == ErrorCategory::INF);
}

TEST_CASE("CategoryOf_PHY001_PHY", "[error_taxonomy][category_of]") {
    CHECK(category_of(ErrorCode::PHY_001) == ErrorCategory::PHY);
}

TEST_CASE("CategoryOf_PHY002_PHY", "[error_taxonomy][category_of]") {
    CHECK(category_of(ErrorCode::PHY_002) == ErrorCategory::PHY);
}

TEST_CASE("CategoryOf_PHY003_PHY", "[error_taxonomy][category_of]") {
    CHECK(category_of(ErrorCode::PHY_003) == ErrorCategory::PHY);
}

TEST_CASE("CategoryOf_PHY004_PHY", "[error_taxonomy][category_of]") {
    CHECK(category_of(ErrorCode::PHY_004) == ErrorCategory::PHY);
}

TEST_CASE("CategoryOf_COG001_COG", "[error_taxonomy][category_of]") {
    CHECK(category_of(ErrorCode::COG_001) == ErrorCategory::COG);
}

TEST_CASE("CategoryOf_COG002_COG", "[error_taxonomy][category_of]") {
    CHECK(category_of(ErrorCode::COG_002) == ErrorCategory::COG);
}

TEST_CASE("CategoryOf_COG003_COG", "[error_taxonomy][category_of]") {
    CHECK(category_of(ErrorCode::COG_003) == ErrorCategory::COG);
}

TEST_CASE("CategoryOf_COG004_COG", "[error_taxonomy][category_of]") {
    CHECK(category_of(ErrorCode::COG_004) == ErrorCategory::COG);
}

TEST_CASE("CategoryOf_COG005_COG", "[error_taxonomy][category_of]") {
    CHECK(category_of(ErrorCode::COG_005) == ErrorCategory::COG);
}

// ============================================================
// §8  severity_of() — all 14 error codes
// ============================================================

TEST_CASE("SeverityOf_INF001_Critical", "[error_taxonomy][severity_of]") {
    CHECK(severity_of(ErrorCode::INF_001) == Severity::CRITICAL);
}

TEST_CASE("SeverityOf_INF002_High", "[error_taxonomy][severity_of]") {
    CHECK(severity_of(ErrorCode::INF_002) == Severity::HIGH);
}

TEST_CASE("SeverityOf_INF003_High", "[error_taxonomy][severity_of]") {
    CHECK(severity_of(ErrorCode::INF_003) == Severity::HIGH);
}

TEST_CASE("SeverityOf_INF004_Medium", "[error_taxonomy][severity_of]") {
    CHECK(severity_of(ErrorCode::INF_004) == Severity::MEDIUM);
}

TEST_CASE("SeverityOf_INF005_Low", "[error_taxonomy][severity_of]") {
    CHECK(severity_of(ErrorCode::INF_005) == Severity::LOW);
}

TEST_CASE("SeverityOf_PHY001_Critical", "[error_taxonomy][severity_of]") {
    CHECK(severity_of(ErrorCode::PHY_001) == Severity::CRITICAL);
}

TEST_CASE("SeverityOf_PHY002_Critical", "[error_taxonomy][severity_of]") {
    CHECK(severity_of(ErrorCode::PHY_002) == Severity::CRITICAL);
}

TEST_CASE("SeverityOf_PHY003_High", "[error_taxonomy][severity_of]") {
    CHECK(severity_of(ErrorCode::PHY_003) == Severity::HIGH);
}

TEST_CASE("SeverityOf_PHY004_Medium", "[error_taxonomy][severity_of]") {
    CHECK(severity_of(ErrorCode::PHY_004) == Severity::MEDIUM);
}

TEST_CASE("SeverityOf_COG001_Critical", "[error_taxonomy][severity_of]") {
    CHECK(severity_of(ErrorCode::COG_001) == Severity::CRITICAL);
}

TEST_CASE("SeverityOf_COG002_High", "[error_taxonomy][severity_of]") {
    CHECK(severity_of(ErrorCode::COG_002) == Severity::HIGH);
}

TEST_CASE("SeverityOf_COG003_Medium", "[error_taxonomy][severity_of]") {
    CHECK(severity_of(ErrorCode::COG_003) == Severity::MEDIUM);
}

TEST_CASE("SeverityOf_COG004_High", "[error_taxonomy][severity_of]") {
    CHECK(severity_of(ErrorCode::COG_004) == Severity::HIGH);
}

TEST_CASE("SeverityOf_COG005_Low", "[error_taxonomy][severity_of]") {
    CHECK(severity_of(ErrorCode::COG_005) == Severity::LOW);
}

// ============================================================
// §9  recovery_of() — all 14 error codes
// ============================================================

TEST_CASE("RecoveryOf_INF001_HardReset", "[error_taxonomy][recovery_of]") {
    CHECK(recovery_of(ErrorCode::INF_001) == RecoveryStrategy::HARD_RESET);
}

TEST_CASE("RecoveryOf_INF002_RePairing", "[error_taxonomy][recovery_of]") {
    CHECK(recovery_of(ErrorCode::INF_002) == RecoveryStrategy::RE_PAIRING);
}

TEST_CASE("RecoveryOf_INF003_Throttling", "[error_taxonomy][recovery_of]") {
    CHECK(recovery_of(ErrorCode::INF_003) == RecoveryStrategy::THROTTLING);
}

TEST_CASE("RecoveryOf_INF004_HardReset", "[error_taxonomy][recovery_of]") {
    CHECK(recovery_of(ErrorCode::INF_004) == RecoveryStrategy::HARD_RESET);
}

TEST_CASE("RecoveryOf_INF005_GarbageCollection", "[error_taxonomy][recovery_of]") {
    CHECK(recovery_of(ErrorCode::INF_005) == RecoveryStrategy::GARBAGE_COLLECTION);
}

TEST_CASE("RecoveryOf_PHY001_SoftScram", "[error_taxonomy][recovery_of]") {
    CHECK(recovery_of(ErrorCode::PHY_001) == RecoveryStrategy::SOFT_SCRAM);
}

TEST_CASE("RecoveryOf_PHY002_StepReduction", "[error_taxonomy][recovery_of]") {
    CHECK(recovery_of(ErrorCode::PHY_002) == RecoveryStrategy::STEP_REDUCTION);
}

TEST_CASE("RecoveryOf_PHY003_Regularization", "[error_taxonomy][recovery_of]") {
    CHECK(recovery_of(ErrorCode::PHY_003) == RecoveryStrategy::REGULARIZATION);
}

TEST_CASE("RecoveryOf_PHY004_ReIgnition", "[error_taxonomy][recovery_of]") {
    CHECK(recovery_of(ErrorCode::PHY_004) == RecoveryStrategy::RE_IGNITION);
}

TEST_CASE("RecoveryOf_COG001_AdminOverride", "[error_taxonomy][recovery_of]") {
    CHECK(recovery_of(ErrorCode::COG_001) == RecoveryStrategy::ADMIN_OVERRIDE);
}

TEST_CASE("RecoveryOf_COG002_StimulusInjection", "[error_taxonomy][recovery_of]") {
    CHECK(recovery_of(ErrorCode::COG_002) == RecoveryStrategy::STIMULUS_INJECTION);
}

TEST_CASE("RecoveryOf_COG003_ForcedNap", "[error_taxonomy][recovery_of]") {
    CHECK(recovery_of(ErrorCode::COG_003) == RecoveryStrategy::FORCED_NAP);
}

TEST_CASE("RecoveryOf_COG004_GoalPurge", "[error_taxonomy][recovery_of]") {
    CHECK(recovery_of(ErrorCode::COG_004) == RecoveryStrategy::GOAL_PURGE);
}

TEST_CASE("RecoveryOf_COG005_Masking", "[error_taxonomy][recovery_of]") {
    CHECK(recovery_of(ErrorCode::COG_005) == RecoveryStrategy::MASKING);
}

// ============================================================
// §10  Predicate Helpers
// ============================================================

TEST_CASE("IsCritical_INF001_True", "[error_taxonomy][predicates]") {
    CHECK(is_critical(ErrorCode::INF_001));
}

TEST_CASE("IsCritical_PHY002_True", "[error_taxonomy][predicates]") {
    CHECK(is_critical(ErrorCode::PHY_002));
}

TEST_CASE("IsCritical_INF005_False", "[error_taxonomy][predicates]") {
    CHECK_FALSE(is_critical(ErrorCode::INF_005));
}

TEST_CASE("IsCritical_INF004_False", "[error_taxonomy][predicates]") {
    CHECK_FALSE(is_critical(ErrorCode::INF_004));
}

TEST_CASE("IsInfrastructureError_INF001_True", "[error_taxonomy][predicates]") {
    CHECK(is_infrastructure_error(ErrorCode::INF_003));
}

TEST_CASE("IsInfrastructureError_PHY001_False", "[error_taxonomy][predicates]") {
    CHECK_FALSE(is_infrastructure_error(ErrorCode::PHY_001));
}

TEST_CASE("IsPhysicsError_PHY001_True", "[error_taxonomy][predicates]") {
    CHECK(is_physics_error(ErrorCode::PHY_001));
}

TEST_CASE("IsPhysicsError_COG001_False", "[error_taxonomy][predicates]") {
    CHECK_FALSE(is_physics_error(ErrorCode::COG_001));
}

TEST_CASE("IsCognitiveError_COG003_True", "[error_taxonomy][predicates]") {
    CHECK(is_cognitive_error(ErrorCode::COG_003));
}

TEST_CASE("IsCognitiveError_INF001_False", "[error_taxonomy][predicates]") {
    CHECK_FALSE(is_cognitive_error(ErrorCode::INF_001));
}

TEST_CASE("RequiresSoftScram_PHY001_True", "[error_taxonomy][predicates]") {
    CHECK(requires_soft_scram(ErrorCode::PHY_001));
}

TEST_CASE("RequiresSoftScram_PHY002_False", "[error_taxonomy][predicates]") {
    // PHY-002 uses STEP_REDUCTION, not SOFT_SCRAM
    CHECK_FALSE(requires_soft_scram(ErrorCode::PHY_002));
}

TEST_CASE("RequiresForcedNap_COG003_True", "[error_taxonomy][predicates]") {
    CHECK(requires_forced_nap(ErrorCode::COG_003));
}

TEST_CASE("RequiresForcedNap_COG001_False", "[error_taxonomy][predicates]") {
    CHECK_FALSE(requires_forced_nap(ErrorCode::COG_001));
}

TEST_CASE("IsTemporalDecoherence_ExactThreshold_False", "[error_taxonomy][predicates]") {
    CHECK_FALSE(is_temporal_decoherence(TEMPORAL_DECOHERENCE_THRESHOLD_MS));
}

TEST_CASE("IsTemporalDecoherence_OneBeyond_True", "[error_taxonomy][predicates]") {
    CHECK(is_temporal_decoherence(TEMPORAL_DECOHERENCE_THRESHOLD_MS + 1u));
}

TEST_CASE("IsTemporalDecoherence_50ms_False", "[error_taxonomy][predicates]") {
    CHECK_FALSE(is_temporal_decoherence(50u));
}

TEST_CASE("IsTemporalDecoherence_51ms_True", "[error_taxonomy][predicates]") {
    CHECK(is_temporal_decoherence(51u));
}

TEST_CASE("IsHeartbeatFailure_ExactThreshold_False", "[error_taxonomy][predicates]") {
    CHECK_FALSE(is_heartbeat_failure(HEARTBEAT_FAILURE_THRESHOLD_MS));
}

TEST_CASE("IsHeartbeatFailure_OneBeyond_True", "[error_taxonomy][predicates]") {
    CHECK(is_heartbeat_failure(HEARTBEAT_FAILURE_THRESHOLD_MS + 1u));
}

TEST_CASE("IsHeartbeatFailure_Zero_False", "[error_taxonomy][predicates]") {
    CHECK_FALSE(is_heartbeat_failure(0u));
}

TEST_CASE("IsEnergyDriftViolation_ExactThreshold_False", "[error_taxonomy][predicates]") {
    CHECK_FALSE(is_energy_drift_violation(ENERGY_DRIFT_MAX_RATIO));
}

TEST_CASE("IsEnergyDriftViolation_SlightlyAbove_True", "[error_taxonomy][predicates]") {
    CHECK(is_energy_drift_violation(ENERGY_DRIFT_MAX_RATIO + 1.0e-10));
}

TEST_CASE("IsEnergyDriftViolation_Zero_False", "[error_taxonomy][predicates]") {
    CHECK_FALSE(is_energy_drift_violation(0.0));
}

TEST_CASE("IsATPExhausted_Below5Percent_True", "[error_taxonomy][predicates]") {
    CHECK(is_atp_exhausted(0.04));
}

TEST_CASE("IsATPExhausted_ExactThreshold_False", "[error_taxonomy][predicates]") {
    CHECK_FALSE(is_atp_exhausted(ATP_EXHAUSTION_THRESHOLD));
}

TEST_CASE("IsATPExhausted_FullBudget_False", "[error_taxonomy][predicates]") {
    CHECK_FALSE(is_atp_exhausted(1.0));
}

TEST_CASE("IsATPExhausted_Zero_True", "[error_taxonomy][predicates]") {
    CHECK(is_atp_exhausted(0.0));
}

// ============================================================
// §11  Label Functions
// ============================================================

TEST_CASE("SeverityName_Critical", "[error_taxonomy][labels]") {
    CHECK(severity_name(Severity::CRITICAL) == "critical");
}

TEST_CASE("SeverityName_High", "[error_taxonomy][labels]") {
    CHECK(severity_name(Severity::HIGH) == "high");
}

TEST_CASE("SeverityName_Medium", "[error_taxonomy][labels]") {
    CHECK(severity_name(Severity::MEDIUM) == "medium");
}

TEST_CASE("SeverityName_Low", "[error_taxonomy][labels]") {
    CHECK(severity_name(Severity::LOW) == "low");
}

TEST_CASE("SeverityName_AllNonEmpty", "[error_taxonomy][labels]") {
    CHECK_FALSE(severity_name(Severity::CRITICAL).empty());
    CHECK_FALSE(severity_name(Severity::HIGH).empty());
    CHECK_FALSE(severity_name(Severity::MEDIUM).empty());
    CHECK_FALSE(severity_name(Severity::LOW).empty());
}

TEST_CASE("CategoryName_INF", "[error_taxonomy][labels]") {
    CHECK(category_name(ErrorCategory::INF) == "INF");
}

TEST_CASE("CategoryName_PHY", "[error_taxonomy][labels]") {
    CHECK(category_name(ErrorCategory::PHY) == "PHY");
}

TEST_CASE("CategoryName_COG", "[error_taxonomy][labels]") {
    CHECK(category_name(ErrorCategory::COG) == "COG");
}

TEST_CASE("CategoryName_AUTO", "[error_taxonomy][labels]") {
    CHECK(category_name(ErrorCategory::AUTO) == "AUTO");
}

TEST_CASE("RecoveryName_SoftScram", "[error_taxonomy][labels]") {
    CHECK(recovery_name(RecoveryStrategy::SOFT_SCRAM) == "soft_scram");
}

TEST_CASE("RecoveryName_HardReset", "[error_taxonomy][labels]") {
    CHECK(recovery_name(RecoveryStrategy::HARD_RESET) == "hard_reset");
}

TEST_CASE("RecoveryName_ForcedNap", "[error_taxonomy][labels]") {
    CHECK(recovery_name(RecoveryStrategy::FORCED_NAP) == "forced_nap");
}

TEST_CASE("RecoveryName_GoalPurge", "[error_taxonomy][labels]") {
    CHECK(recovery_name(RecoveryStrategy::GOAL_PURGE) == "goal_purge");
}

TEST_CASE("RecoveryName_AllNonEmpty", "[error_taxonomy][labels]") {
    for (uint8_t i = 0u; i < static_cast<uint8_t>(RECOVERY_STRATEGY_COUNT); ++i) {
        const auto rs = static_cast<RecoveryStrategy>(i);
        CHECK_FALSE(recovery_name(rs).empty());
    }
}

TEST_CASE("ErrorCodeName_PHY002", "[error_taxonomy][labels]") {
    CHECK(error_code_name(ErrorCode::PHY_002) == "PHY-002");
}

TEST_CASE("ErrorCodeName_COG001", "[error_taxonomy][labels]") {
    CHECK(error_code_name(ErrorCode::COG_001) == "COG-001");
}

TEST_CASE("ErrorCodeName_INF001", "[error_taxonomy][labels]") {
    CHECK(error_code_name(ErrorCode::INF_001) == "INF-001");
}

TEST_CASE("ErrorCodeName_AllNonEmpty", "[error_taxonomy][labels]") {
    for (uint8_t i = 0u; i < static_cast<uint8_t>(ERROR_CODE_COUNT); ++i) {
        const auto code = static_cast<ErrorCode>(i);
        CHECK_FALSE(error_code_name(code).empty());
    }
}

TEST_CASE("ErrorCodeName_AllHaveDashSeparator", "[error_taxonomy][labels]") {
    for (uint8_t i = 0u; i < static_cast<uint8_t>(ERROR_CODE_COUNT); ++i) {
        const auto   code = static_cast<ErrorCode>(i);
        const auto   name = error_code_name(code);
        bool has_dash = false;
        for (char c : name) {
            if (c == '-') { has_dash = true; break; }
        }
        CHECK(has_dash);
    }
}

// ============================================================
// Integration Scenarios
// ============================================================

TEST_CASE("Integration_TemporalDecoherence_FullPath", "[error_taxonomy][integration]") {
    // Simulate INF-001: control plane 75 ms behind data plane
    const uint32_t delta_ms = 75u;
    CHECK(is_temporal_decoherence(delta_ms));

    // Confirm correct taxonomy entries
    CHECK(category_of(ErrorCode::INF_001)  == ErrorCategory::INF);
    CHECK(severity_of(ErrorCode::INF_001)  == Severity::CRITICAL);
    CHECK(recovery_of(ErrorCode::INF_001)  == RecoveryStrategy::HARD_RESET);
    CHECK(is_critical(ErrorCode::INF_001));
    CHECK(error_code_name(ErrorCode::INF_001) == "INF-001");
}

TEST_CASE("Integration_EpilepticResonance_SoftScram", "[error_taxonomy][integration]") {
    // PHY-001: wavefunction diverges → Soft SCRAM
    CHECK(category_of(ErrorCode::PHY_001) == ErrorCategory::PHY);
    CHECK(severity_of(ErrorCode::PHY_001) == Severity::CRITICAL);
    CHECK(requires_soft_scram(ErrorCode::PHY_001));

    // Soft SCRAM parameters from spec
    CHECK(SOFT_SCRAM_DAMPING == Catch::Approx(0.5));
    CHECK(recovery_name(RecoveryStrategy::SOFT_SCRAM) == "soft_scram");
}

TEST_CASE("Integration_EnergyDrift_StepReduction", "[error_taxonomy][integration]") {
    // PHY-002: drift at 0.015% (spec example) → violation detected
    const double drift = 0.00015;   // 0.015 %
    CHECK(is_energy_drift_violation(drift));

    CHECK(recovery_of(ErrorCode::PHY_002) == RecoveryStrategy::STEP_REDUCTION);
    CHECK(STEP_REDUCTION_FACTOR == Catch::Approx(0.5));
    CHECK(ENERGY_DRIFT_EVAL_STEPS == 100u);
}

TEST_CASE("Integration_ATPExhaustion_ForcedNap", "[error_taxonomy][integration]") {
    // COG-003: 3% ATP remaining → exhausted → forced nap
    const double budget = 0.03;
    CHECK(is_atp_exhausted(budget));
    CHECK(requires_forced_nap(ErrorCode::COG_003));
    CHECK(recovery_of(ErrorCode::COG_003) == RecoveryStrategy::FORCED_NAP);
}

TEST_CASE("Integration_LogSchemaFieldsUsable", "[error_taxonomy][integration]") {
    // Verify that all fields expected in the JSON schema are present and correct
    CHECK(LOG_FIELD_ERROR_CODE   == "error_code");
    CHECK(LOG_FIELD_TRACE_ID     == "trace_id");
    CHECK(LOG_FIELD_CONTEXT      == "context");
    CHECK(LOG_FIELD_COUNT        == 8u);
    CHECK_FALSE(LOG_SCHEMA_URI.empty());
}

TEST_CASE("Integration_AllCriticalCodesAreKnown", "[error_taxonomy][integration]") {
    // Only INF-001, PHY-001, PHY-002, COG-001 must be CRITICAL per spec
    CHECK(is_critical(ErrorCode::INF_001));
    CHECK(is_critical(ErrorCode::PHY_001));
    CHECK(is_critical(ErrorCode::PHY_002));
    CHECK(is_critical(ErrorCode::COG_001));

    // Remaining codes must NOT be critical
    CHECK_FALSE(is_critical(ErrorCode::INF_002));
    CHECK_FALSE(is_critical(ErrorCode::INF_003));
    CHECK_FALSE(is_critical(ErrorCode::INF_004));
    CHECK_FALSE(is_critical(ErrorCode::INF_005));
    CHECK_FALSE(is_critical(ErrorCode::PHY_003));
    CHECK_FALSE(is_critical(ErrorCode::PHY_004));
    CHECK_FALSE(is_critical(ErrorCode::COG_002));
    CHECK_FALSE(is_critical(ErrorCode::COG_003));
    CHECK_FALSE(is_critical(ErrorCode::COG_004));
    CHECK_FALSE(is_critical(ErrorCode::COG_005));
}

TEST_CASE("Integration_BoredomSingularity_StimulusInjection", "[error_taxonomy][integration]") {
    // COG-002: entropy gradient ≈ 0 → Stimulus Injection (Curiosity + Norepinephrine)
    CHECK(category_of(ErrorCode::COG_002)  == ErrorCategory::COG);
    CHECK(severity_of(ErrorCode::COG_002)  == Severity::HIGH);
    CHECK(recovery_of(ErrorCode::COG_002)  == RecoveryStrategy::STIMULUS_INJECTION);
    CHECK(recovery_name(RecoveryStrategy::STIMULUS_INJECTION) == "stimulus_injection");
}

TEST_CASE("Integration_TeleologicalDeadlock_GoalPurge", "[error_taxonomy][integration]") {
    // COG-004: circular dependency A→B→A in goal DAG → prune + dopamine spike
    CHECK(severity_of(ErrorCode::COG_004)  == Severity::HIGH);
    CHECK(recovery_of(ErrorCode::COG_004)  == RecoveryStrategy::GOAL_PURGE);
    CHECK(recovery_name(RecoveryStrategy::GOAL_PURGE) == "goal_purge");
}
