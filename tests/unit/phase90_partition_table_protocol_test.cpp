// =============================================================================
// tests/unit/phase90_partition_table_protocol_test.cpp
// Phase 90 — GAP-019: Distributed Partition Table Update Protocol (2P-EBP)
//
// Tests for nikola::system::partition_table_protocol.hpp
// Spec: docs/info/integration/sections/04_infrastructure/01_zeromq_spine.md
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "nikola/system/partition_table_protocol.hpp"

using namespace nikola::system;
using Catch::Approx;

// ---------------------------------------------------------------------------
// § Enums
// ---------------------------------------------------------------------------

TEST_CASE("PartitionEpochState enum values are ordered and distinct", "[enums][phase90]") {
    CHECK(static_cast<int>(PartitionEpochState::STABLE)            == 0);
    CHECK(static_cast<int>(PartitionEpochState::PREPARE_MIGRATION) == 1);
    CHECK(static_cast<int>(PartitionEpochState::MIGRATING)         == 2);
    CHECK(static_cast<int>(PartitionEpochState::VERIFYING)         == 3);
    CHECK(static_cast<int>(PartitionEpochState::COMMITTED)         == 4);
    CHECK(static_cast<int>(PartitionEpochState::ROLLBACK)          == 5);
}

TEST_CASE("PartitionControlType enum values are ordered", "[enums][phase90]") {
    CHECK(static_cast<int>(PartitionControlType::HEARTBEAT)         == 0);
    CHECK(static_cast<int>(PartitionControlType::PREPARE_MIGRATION) == 1);
    CHECK(static_cast<int>(PartitionControlType::BEGIN_MIGRATION)   == 2);
    CHECK(static_cast<int>(PartitionControlType::COMMIT_EPOCH)      == 3);
    CHECK(static_cast<int>(PartitionControlType::ROLLBACK)          == 4);
    CHECK(static_cast<int>(PartitionControlType::ABORT)             == 5);
}

// ---------------------------------------------------------------------------
// § Numeric constants
// ---------------------------------------------------------------------------

TEST_CASE("LIF_TRIGGER_THRESHOLD is 0.20", "[constants][phase90]") {
    CHECK(LIF_TRIGGER_THRESHOLD == Approx(0.20));
}

TEST_CASE("MIGRATION_VRAM_SAFETY_LIMIT is 0.90", "[constants][phase90]") {
    CHECK(MIGRATION_VRAM_SAFETY_LIMIT == Approx(0.90));
}

TEST_CASE("ROLLBACK_TIMEOUT_MS is 5000", "[constants][phase90]") {
    CHECK(ROLLBACK_TIMEOUT_MS == 5000u);
}

TEST_CASE("STABILITY_PENALTY_COOLDOWN_MS is 3600000 (1 hour)", "[constants][phase90]") {
    CHECK(STABILITY_PENALTY_COOLDOWN_MS == 3'600'000u);
}

TEST_CASE("MORTON_KEY_BYTES is 16 (128-bit)", "[constants][phase90]") {
    CHECK(MORTON_KEY_BYTES == 16u);
}

TEST_CASE("MIGRATION_METRIC_COMPONENTS is 45", "[constants][phase90]") {
    CHECK(MIGRATION_METRIC_COMPONENTS == 45u);
}

// ---------------------------------------------------------------------------
// § rebalancing_needed
// ---------------------------------------------------------------------------

TEST_CASE("rebalancing_needed triggers above 0.20 LIF", "[functions][phase90]") {
    CHECK(rebalancing_needed(0.19) == false);
    CHECK(rebalancing_needed(0.20) == false);  // not strictly above
    CHECK(rebalancing_needed(0.21) == true);
}

// ---------------------------------------------------------------------------
// § migration_vram_safe
// ---------------------------------------------------------------------------

TEST_CASE("migration_vram_safe approves utilisation at or below 90%", "[functions][phase90]") {
    CHECK(migration_vram_safe(0.89) == true);
    CHECK(migration_vram_safe(0.90) == true);
    CHECK(migration_vram_safe(0.91) == false);
}

// ---------------------------------------------------------------------------
// § rollback_timeout_exceeded
// ---------------------------------------------------------------------------

TEST_CASE("rollback_timeout_exceeded triggers at 5000 ms", "[functions][phase90]") {
    CHECK(rollback_timeout_exceeded(4999) == false);
    CHECK(rollback_timeout_exceeded(5000) == true);
    CHECK(rollback_timeout_exceeded(9999) == true);
}

// ---------------------------------------------------------------------------
// § load_imbalance_factor
// ---------------------------------------------------------------------------

TEST_CASE("load_imbalance_factor computes (max-min)/mean", "[functions][phase90]") {
    // max=120, min=80, mean=100 → LIF = 40/100 = 0.40
    CHECK(load_imbalance_factor(120.0, 80.0, 100.0) == Approx(0.40));
}

TEST_CASE("load_imbalance_factor returns 0 for zero mean", "[functions][phase90]") {
    CHECK(load_imbalance_factor(100.0, 50.0, 0.0) == Approx(0.0));
}

TEST_CASE("load_imbalance_factor is 0 for perfectly balanced cluster", "[functions][phase90]") {
    CHECK(load_imbalance_factor(100.0, 100.0, 100.0) == Approx(0.0));
}

// ---------------------------------------------------------------------------
// § Label helpers
// ---------------------------------------------------------------------------

TEST_CASE("epoch_state_label returns correct strings", "[labels][phase90]") {
    CHECK(epoch_state_label(PartitionEpochState::STABLE)            == "STABLE");
    CHECK(epoch_state_label(PartitionEpochState::PREPARE_MIGRATION) == "PREPARE_MIGRATION");
    CHECK(epoch_state_label(PartitionEpochState::MIGRATING)         == "MIGRATING");
    CHECK(epoch_state_label(PartitionEpochState::VERIFYING)         == "VERIFYING");
    CHECK(epoch_state_label(PartitionEpochState::COMMITTED)         == "COMMITTED");
    CHECK(epoch_state_label(PartitionEpochState::ROLLBACK)          == "ROLLBACK");
}

TEST_CASE("control_type_label returns correct strings", "[labels][phase90]") {
    CHECK(control_type_label(PartitionControlType::HEARTBEAT)         == "HEARTBEAT");
    CHECK(control_type_label(PartitionControlType::PREPARE_MIGRATION) == "PREPARE_MIGRATION");
    CHECK(control_type_label(PartitionControlType::COMMIT_EPOCH)      == "COMMIT_EPOCH");
    CHECK(control_type_label(PartitionControlType::ROLLBACK)          == "ROLLBACK");
    CHECK(control_type_label(PartitionControlType::ABORT)             == "ABORT");
}
