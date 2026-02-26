// =============================================================================
// tests/unit/phase88_error_taxonomy_test.cpp
// Phase 88 — GAP-042: System Error Taxonomy (INF / PHY / COG)
//
// Tests for nikola::system::error_taxonomy.hpp
// Spec: docs/info/integration/sections/04_infrastructure/02_orchestrator_router.md
// =============================================================================

#include <catch2/catch_test_macros.hpp>

#include "nikola/system/error_taxonomy.hpp"

using namespace nikola::system;

// ---------------------------------------------------------------------------
// § Enum smoke-tests
// ---------------------------------------------------------------------------

TEST_CASE("ErrorCategory enum values are distinct", "[enums][phase88]") {
    CHECK(static_cast<int>(ErrorCategory::INF)  == 0);
    CHECK(static_cast<int>(ErrorCategory::PHY)  == 1);
    CHECK(static_cast<int>(ErrorCategory::COG)  == 2);
    CHECK(static_cast<int>(ErrorCategory::AUTO) == 3);
}

TEST_CASE("ErrorSeverity enum is ordered CRITICAL→LOW", "[enums][phase88]") {
    CHECK(static_cast<int>(ErrorSeverity::CRITICAL) < static_cast<int>(ErrorSeverity::HIGH));
    CHECK(static_cast<int>(ErrorSeverity::HIGH)     < static_cast<int>(ErrorSeverity::MEDIUM));
    CHECK(static_cast<int>(ErrorSeverity::MEDIUM)   < static_cast<int>(ErrorSeverity::LOW));
}

// ---------------------------------------------------------------------------
// § Catalogue completeness
// ---------------------------------------------------------------------------

TEST_CASE("ERROR_CATALOGUE contains 14 entries", "[catalogue][phase88]") {
    CHECK(ERROR_CATALOGUE_SIZE == 14u);
}

TEST_CASE("All INF entries (5) are present in catalogue", "[catalogue][phase88]") {
    int count = 0;
    for (const auto& e : ERROR_CATALOGUE) {
        if (e.category == ErrorCategory::INF) ++count;
    }
    CHECK(count == 5);
}

TEST_CASE("All PHY entries (4) are present in catalogue", "[catalogue][phase88]") {
    int count = 0;
    for (const auto& e : ERROR_CATALOGUE) {
        if (e.category == ErrorCategory::PHY) ++count;
    }
    CHECK(count == 4);
}

TEST_CASE("All COG entries (5) are present in catalogue", "[catalogue][phase88]") {
    int count = 0;
    for (const auto& e : ERROR_CATALOGUE) {
        if (e.category == ErrorCategory::COG) ++count;
    }
    CHECK(count == 5);
}

// ---------------------------------------------------------------------------
// § Specific error descriptors
// ---------------------------------------------------------------------------

TEST_CASE("INF_001 is CRITICAL TemporalDecoherence with HardReset", "[descriptors][phase88]") {
    CHECK(INF_001.category  == ErrorCategory::INF);
    CHECK(INF_001.code      == 1);
    CHECK(INF_001.severity  == ErrorSeverity::CRITICAL);
    CHECK(INF_001.recovery  == RecoveryStrategy::HARD_RESET);
    CHECK(INF_001.name      == "TemporalDecoherence");
}

TEST_CASE("INF_002 is HIGH CryptographicAmnesia with RePairing", "[descriptors][phase88]") {
    CHECK(INF_002.severity == ErrorSeverity::HIGH);
    CHECK(INF_002.recovery == RecoveryStrategy::RE_PAIRING);
}

TEST_CASE("PHY_001 is CRITICAL EpilepticResonance with SoftSCRAM", "[descriptors][phase88]") {
    CHECK(PHY_001.category == ErrorCategory::PHY);
    CHECK(PHY_001.severity == ErrorSeverity::CRITICAL);
    CHECK(PHY_001.recovery == RecoveryStrategy::SOFT_SCRAM);
    CHECK(PHY_001.name     == "EpilepticResonance");
}

TEST_CASE("PHY_002 is CRITICAL EnergyNonConservation with StepReduction", "[descriptors][phase88]") {
    CHECK(PHY_002.severity == ErrorSeverity::CRITICAL);
    CHECK(PHY_002.recovery == RecoveryStrategy::STEP_REDUCTION);
}

TEST_CASE("PHY_003 is HIGH MetricSingularity with Regularization", "[descriptors][phase88]") {
    CHECK(PHY_003.severity == ErrorSeverity::HIGH);
    CHECK(PHY_003.recovery == RecoveryStrategy::REGULARIZATION);
}

TEST_CASE("COG_001 is CRITICAL RunawayCognitiveLoop with AdminOverride", "[descriptors][phase88]") {
    CHECK(COG_001.category == ErrorCategory::COG);
    CHECK(COG_001.severity == ErrorSeverity::CRITICAL);
    CHECK(COG_001.recovery == RecoveryStrategy::ADMIN_OVERRIDE);
}

TEST_CASE("COG_003 is MEDIUM ATPExhaustion with ForcedNap", "[descriptors][phase88]") {
    CHECK(COG_003.severity == ErrorSeverity::MEDIUM);
    CHECK(COG_003.recovery == RecoveryStrategy::FORCED_NAP);
}

TEST_CASE("COG_005 is LOW Hallucination with Masking", "[descriptors][phase88]") {
    CHECK(COG_005.severity == ErrorSeverity::LOW);
    CHECK(COG_005.recovery == RecoveryStrategy::MASKING);
}

// ---------------------------------------------------------------------------
// § lookup_error
// ---------------------------------------------------------------------------

TEST_CASE("lookup_error finds INF_001 by category + code", "[lookup][phase88]") {
    const auto* e = lookup_error(ErrorCategory::INF, 1);
    REQUIRE(e != nullptr);
    CHECK(e->name == "TemporalDecoherence");
}

TEST_CASE("lookup_error returns nullptr for non-existent code", "[lookup][phase88]") {
    CHECK(lookup_error(ErrorCategory::INF, 99) == nullptr);
    CHECK(lookup_error(ErrorCategory::AUTO,  1) == nullptr);
}

TEST_CASE("lookup_error finds PHY_004", "[lookup][phase88]") {
    const auto* e = lookup_error(ErrorCategory::PHY, 4);
    REQUIRE(e != nullptr);
    CHECK(e->name == "VacuumCollapse");
}

// ---------------------------------------------------------------------------
// § is_fatal
// ---------------------------------------------------------------------------

TEST_CASE("is_fatal returns true only for CRITICAL severity", "[functions][phase88]") {
    CHECK(is_fatal(INF_001) == true);   // CRITICAL
    CHECK(is_fatal(INF_002) == false);  // HIGH
    CHECK(is_fatal(PHY_001) == true);   // CRITICAL
    CHECK(is_fatal(INF_003) == false);  // HIGH
    CHECK(is_fatal(COG_005) == false);  // LOW
}

// ---------------------------------------------------------------------------
// § Label helpers
// ---------------------------------------------------------------------------

TEST_CASE("category_label returns correct strings", "[labels][phase88]") {
    CHECK(category_label(ErrorCategory::INF)  == "INF");
    CHECK(category_label(ErrorCategory::PHY)  == "PHY");
    CHECK(category_label(ErrorCategory::COG)  == "COG");
    CHECK(category_label(ErrorCategory::AUTO) == "AUTO");
}

TEST_CASE("severity_label returns correct strings", "[labels][phase88]") {
    CHECK(severity_label(ErrorSeverity::CRITICAL) == "CRITICAL");
    CHECK(severity_label(ErrorSeverity::HIGH)     == "HIGH");
    CHECK(severity_label(ErrorSeverity::MEDIUM)   == "MEDIUM");
    CHECK(severity_label(ErrorSeverity::LOW)      == "LOW");
}

TEST_CASE("recovery_label returns non-empty strings for all strategies", "[labels][phase88]") {
    for (int i = 0; i <= 13; ++i) {
        auto label = recovery_label(static_cast<RecoveryStrategy>(i));
        CHECK(!label.empty());
        CHECK(label != "Unknown");
    }
}
