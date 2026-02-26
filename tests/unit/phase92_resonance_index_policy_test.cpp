// =============================================================================
// tests/unit/phase92_resonance_index_policy_test.cpp
// Phase 92 — GAP-024: Ingestion Pipeline → Resonance Index Synchronisation
//
// Tests for nikola::memory::resonance_index_policy.hpp
// Spec: docs/info/integration/sections/03_cognitive_systems/04_memory_data_systems.md
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "nikola/memory/resonance_index_policy.hpp"

using namespace nikola::memory;
using Catch::Approx;

// ---------------------------------------------------------------------------
// § Enums
// ---------------------------------------------------------------------------

TEST_CASE("MergeType enum values are distinct", "[enums][phase92]") {
    CHECK(static_cast<int>(MergeType::INCREMENTAL)  == 0);
    CHECK(static_cast<int>(MergeType::FULL_REBUILD) == 1);
}

TEST_CASE("MergeTrigger enum values are distinct", "[enums][phase92]") {
    CHECK(static_cast<int>(MergeTrigger::NODE_THRESHOLD) == 0);
    CHECK(static_cast<int>(MergeTrigger::TIME_ELAPSED)   == 1);
    CHECK(static_cast<int>(MergeTrigger::NAP_STATE)      == 2);
    CHECK(static_cast<int>(MergeTrigger::FRAGMENTATION)  == 3);
}

// ---------------------------------------------------------------------------
// § Threshold constants
// ---------------------------------------------------------------------------

TEST_CASE("MEMTABLE_MERGE_TRIGGER_NODES is 10000", "[constants][phase92]") {
    CHECK(MEMTABLE_MERGE_TRIGGER_NODES == 10'000u);
}

TEST_CASE("MEMTABLE_MERGE_TRIGGER_MS is 1000", "[constants][phase92]") {
    CHECK(MEMTABLE_MERGE_TRIGGER_MS == 1'000u);
}

TEST_CASE("MERGER_POLL_INTERVAL_MS is 100", "[constants][phase92]") {
    CHECK(MERGER_POLL_INTERVAL_MS == 100u);
}

TEST_CASE("REBUILD_ATP_NAP_THRESHOLD is 0.15", "[constants][phase92]") {
    CHECK(REBUILD_ATP_NAP_THRESHOLD == Approx(0.15));
}

TEST_CASE("REBUILD_FRAGMENTATION_THRESHOLD is 0.20", "[constants][phase92]") {
    CHECK(REBUILD_FRAGMENTATION_THRESHOLD == Approx(0.20));
}

TEST_CASE("VISIBILITY_LAG_MAX_MS is 500", "[constants][phase92]") {
    CHECK(VISIBILITY_LAG_MAX_MS == 500u);
}

// ---------------------------------------------------------------------------
// § should_merge_incremental
// ---------------------------------------------------------------------------

TEST_CASE("should_merge_incremental triggers on node threshold", "[functions][phase92]") {
    CHECK(should_merge_incremental(10'000, 0)   == true);
    CHECK(should_merge_incremental(9'999, 0)    == false);
    CHECK(should_merge_incremental(9'999, 1001) == true);  // time threshold
}

TEST_CASE("should_merge_incremental triggers on time threshold", "[functions][phase92]") {
    CHECK(should_merge_incremental(0, 1'000) == true);
    CHECK(should_merge_incremental(0,   999) == false);
}

TEST_CASE("should_merge_incremental returns false when both below threshold", "[functions][phase92]") {
    CHECK(should_merge_incremental(5'000, 500) == false);
}

// ---------------------------------------------------------------------------
// § should_rebuild_full
// ---------------------------------------------------------------------------

TEST_CASE("should_rebuild_full triggers on nap state (ATP < 0.15)", "[functions][phase92]") {
    CHECK(should_rebuild_full(0.14, 0.0) == true);
    CHECK(should_rebuild_full(0.15, 0.0) == false);
}

TEST_CASE("should_rebuild_full triggers on fragmentation > 0.20", "[functions][phase92]") {
    CHECK(should_rebuild_full(1.0, 0.21) == true);
    CHECK(should_rebuild_full(1.0, 0.19) == false);
}

TEST_CASE("should_rebuild_full returns false when both below threshold", "[functions][phase92]") {
    CHECK(should_rebuild_full(0.50, 0.10) == false);
}

// ---------------------------------------------------------------------------
// § visibility_lag_acceptable
// ---------------------------------------------------------------------------

TEST_CASE("visibility_lag_acceptable allows lag <= 500 ms", "[functions][phase92]") {
    CHECK(visibility_lag_acceptable(0)   == true);
    CHECK(visibility_lag_acceptable(500) == true);
    CHECK(visibility_lag_acceptable(501) == false);
}

// ---------------------------------------------------------------------------
// § merge_type_for
// ---------------------------------------------------------------------------

TEST_CASE("merge_type_for maps nap/fragmentation to FULL_REBUILD", "[functions][phase92]") {
    CHECK(merge_type_for(MergeTrigger::NAP_STATE)     == MergeType::FULL_REBUILD);
    CHECK(merge_type_for(MergeTrigger::FRAGMENTATION) == MergeType::FULL_REBUILD);
}

TEST_CASE("merge_type_for maps node/time thresholds to INCREMENTAL", "[functions][phase92]") {
    CHECK(merge_type_for(MergeTrigger::NODE_THRESHOLD) == MergeType::INCREMENTAL);
    CHECK(merge_type_for(MergeTrigger::TIME_ELAPSED)   == MergeType::INCREMENTAL);
}

// ---------------------------------------------------------------------------
// § Label helpers
// ---------------------------------------------------------------------------

TEST_CASE("merge_type_label returns correct strings", "[labels][phase92]") {
    CHECK(merge_type_label(MergeType::INCREMENTAL)  == "incremental");
    CHECK(merge_type_label(MergeType::FULL_REBUILD) == "full_rebuild");
}

TEST_CASE("merge_trigger_label returns correct strings", "[labels][phase92]") {
    CHECK(merge_trigger_label(MergeTrigger::NODE_THRESHOLD) == "node_threshold");
    CHECK(merge_trigger_label(MergeTrigger::TIME_ELAPSED)   == "time_elapsed");
    CHECK(merge_trigger_label(MergeTrigger::NAP_STATE)      == "nap_state");
    CHECK(merge_trigger_label(MergeTrigger::FRAGMENTATION)  == "fragmentation");
}
