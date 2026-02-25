// ============================================================
// tests/unit/phase70_resonance_index_test.cpp
// Phase 70 — GAP-024  Ingestion Pipeline → Resonance Index Synchronization
// ============================================================
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <array>
#include <cstdint>
#include <vector>

#include "nikola/memory/resonance_index.hpp"

using namespace nikola::memory;
using Catch::Approx;

// ────────────────────────────────────────────────────────────────────────────
// §1  Constants
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("MemTable flush threshold is 10,000 nodes", "[constants]") {
    CHECK(MEMTABLE_FLUSH_THRESHOLD == 10'000u);
}

TEST_CASE("Flush interval is 1 second (1,000 ms)", "[constants]") {
    CHECK(FLUSH_INTERVAL_MS == 1'000u);
}

TEST_CASE("Merge poll interval is 100 ms", "[constants]") {
    CHECK(MERGE_POLL_INTERVAL_MS == 100u);
}

TEST_CASE("Maximum visibility lag is 500 ms", "[constants]") {
    CHECK(MAX_VISIBILITY_LAG_MS == 500u);
}

TEST_CASE("Minor merge budget is 10 ms", "[constants]") {
    CHECK(MINOR_MERGE_MAX_MS == 10u);
}

TEST_CASE("Fragmentation rebuild threshold is 20%", "[constants]") {
    CHECK(FRAGMENTATION_REBUILD_THRESHOLD == Approx(0.20f).epsilon(1e-6f));
}

TEST_CASE("Major rebuild ATP threshold is 15%", "[constants]") {
    CHECK(MAJOR_REBUILD_ATP_THRESHOLD == Approx(0.15f).epsilon(1e-6f));
}

TEST_CASE("Timing constants ordering invariants", "[constants]") {
    // Poll interval << visibility lag
    CHECK(MERGE_POLL_INTERVAL_MS < MAX_VISIBILITY_LAG_MS);
    // Minor merge budget << visibility lag
    CHECK(MINOR_MERGE_MAX_MS < MAX_VISIBILITY_LAG_MS);
    // Poll interval < flush interval (polls more frequently than mandatory flush)
    CHECK(MERGE_POLL_INTERVAL_MS < FLUSH_INTERVAL_MS);
}

// ────────────────────────────────────────────────────────────────────────────
// §2  Minor merge trigger: MemTable size threshold
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("is_flush_threshold_exceeded: below threshold", "[minor_trigger]") {
    CHECK(is_flush_threshold_exceeded(0u)      == false);
    CHECK(is_flush_threshold_exceeded(5'000u)  == false);
    CHECK(is_flush_threshold_exceeded(9'999u)  == false);
}

TEST_CASE("is_flush_threshold_exceeded: exactly at threshold is NOT exceeded (strictly >)", "[minor_trigger]") {
    CHECK(is_flush_threshold_exceeded(10'000u) == false);
}

TEST_CASE("is_flush_threshold_exceeded: above threshold", "[minor_trigger]") {
    CHECK(is_flush_threshold_exceeded(10'001u) == true);
    CHECK(is_flush_threshold_exceeded(50'000u) == true);
}

// ────────────────────────────────────────────────────────────────────────────
// §3  Minor merge trigger: time elapsed
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("is_flush_interval_elapsed: below interval", "[minor_trigger]") {
    CHECK(is_flush_interval_elapsed(0u)     == false);
    CHECK(is_flush_interval_elapsed(500u)   == false);
    CHECK(is_flush_interval_elapsed(999u)   == false);
}

TEST_CASE("is_flush_interval_elapsed: exactly at 1 second is NOT elapsed (strictly >)", "[minor_trigger]") {
    CHECK(is_flush_interval_elapsed(1'000u) == false);
}

TEST_CASE("is_flush_interval_elapsed: above interval", "[minor_trigger]") {
    CHECK(is_flush_interval_elapsed(1'001u) == true);
    CHECK(is_flush_interval_elapsed(5'000u) == true);
}

// ────────────────────────────────────────────────────────────────────────────
// §4  should_trigger_minor_merge (OR of size and time)
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("should_trigger_minor_merge: neither condition", "[minor_trigger]") {
    CHECK(should_trigger_minor_merge(0u,      0u)     == false);
    CHECK(should_trigger_minor_merge(9'999u,  999u)   == false);
    CHECK(should_trigger_minor_merge(10'000u, 1'000u) == false);  // both exact boundaries
}

TEST_CASE("should_trigger_minor_merge: size exceeded only", "[minor_trigger]") {
    CHECK(should_trigger_minor_merge(10'001u, 0u)    == true);
    CHECK(should_trigger_minor_merge(50'000u, 500u)  == true);
}

TEST_CASE("should_trigger_minor_merge: time elapsed only", "[minor_trigger]") {
    CHECK(should_trigger_minor_merge(0u,     1'001u) == true);
    CHECK(should_trigger_minor_merge(100u,   2'000u) == true);
}

TEST_CASE("should_trigger_minor_merge: both conditions active", "[minor_trigger]") {
    CHECK(should_trigger_minor_merge(20'000u, 2'000u) == true);
}

// ────────────────────────────────────────────────────────────────────────────
// §5  is_major_rebuild_triggered_by_atp
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("is_major_rebuild_triggered_by_atp: above 15% — no trigger", "[major_trigger]") {
    CHECK(is_major_rebuild_triggered_by_atp(0.16f) == false);
    CHECK(is_major_rebuild_triggered_by_atp(0.50f) == false);
    CHECK(is_major_rebuild_triggered_by_atp(1.00f) == false);
}

TEST_CASE("is_major_rebuild_triggered_by_atp: at 15% — no trigger (strictly <)", "[major_trigger]") {
    CHECK(is_major_rebuild_triggered_by_atp(0.15f) == false);
}

TEST_CASE("is_major_rebuild_triggered_by_atp: below 15% — triggers", "[major_trigger]") {
    CHECK(is_major_rebuild_triggered_by_atp(0.14f) == true);
    CHECK(is_major_rebuild_triggered_by_atp(0.00f) == true);
}

TEST_CASE("is_major_rebuild_triggered_by_atp: out of range throws", "[major_trigger]") {
    CHECK_THROWS_AS(is_major_rebuild_triggered_by_atp(-0.1f), std::invalid_argument);
    CHECK_THROWS_AS(is_major_rebuild_triggered_by_atp(1.1f),  std::invalid_argument);
}

// ────────────────────────────────────────────────────────────────────────────
// §6  is_major_rebuild_triggered_by_fragmentation
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("is_major_rebuild_triggered_by_fragmentation: below 20% — no trigger", "[major_trigger]") {
    CHECK(is_major_rebuild_triggered_by_fragmentation(0.00f) == false);
    CHECK(is_major_rebuild_triggered_by_fragmentation(0.10f) == false);
    CHECK(is_major_rebuild_triggered_by_fragmentation(0.19f) == false);
}

TEST_CASE("is_major_rebuild_triggered_by_fragmentation: at 20% — no trigger (strictly >)", "[major_trigger]") {
    CHECK(is_major_rebuild_triggered_by_fragmentation(0.20f) == false);
}

TEST_CASE("is_major_rebuild_triggered_by_fragmentation: above 20% — triggers", "[major_trigger]") {
    CHECK(is_major_rebuild_triggered_by_fragmentation(0.21f) == true);
    CHECK(is_major_rebuild_triggered_by_fragmentation(1.00f) == true);
}

TEST_CASE("is_major_rebuild_triggered_by_fragmentation: out of range throws", "[major_trigger]") {
    CHECK_THROWS_AS(is_major_rebuild_triggered_by_fragmentation(-0.1f), std::invalid_argument);
    CHECK_THROWS_AS(is_major_rebuild_triggered_by_fragmentation(1.1f),  std::invalid_argument);
}

// ────────────────────────────────────────────────────────────────────────────
// §7  classify_rebuild_reason
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("classify_rebuild_reason: no conditions → NONE", "[rebuild_reason]") {
    CHECK(classify_rebuild_reason(0.50f, 0.10f) == RebuildReason::NONE);
    CHECK(classify_rebuild_reason(0.15f, 0.20f) == RebuildReason::NONE);  // exact boundaries
}

TEST_CASE("classify_rebuild_reason: ATP only → NAP_ATP", "[rebuild_reason]") {
    CHECK(classify_rebuild_reason(0.10f, 0.05f) == RebuildReason::NAP_ATP);
    CHECK(classify_rebuild_reason(0.00f, 0.00f) == RebuildReason::NAP_ATP);
}

TEST_CASE("classify_rebuild_reason: fragmentation only → FRAGMENTATION", "[rebuild_reason]") {
    CHECK(classify_rebuild_reason(0.50f, 0.25f) == RebuildReason::FRAGMENTATION);
    CHECK(classify_rebuild_reason(0.20f, 1.00f) == RebuildReason::FRAGMENTATION);
}

TEST_CASE("classify_rebuild_reason: both conditions → BOTH", "[rebuild_reason]") {
    CHECK(classify_rebuild_reason(0.10f, 0.25f) == RebuildReason::BOTH);
}

TEST_CASE("should_trigger_major_rebuild matches classify_rebuild_reason != NONE", "[rebuild_reason]") {
    // False cases
    CHECK(should_trigger_major_rebuild(0.50f, 0.10f) == false);
    // True cases
    CHECK(should_trigger_major_rebuild(0.10f, 0.05f) == true);
    CHECK(should_trigger_major_rebuild(0.50f, 0.25f) == true);
    CHECK(should_trigger_major_rebuild(0.10f, 0.25f) == true);
}

// ────────────────────────────────────────────────────────────────────────────
// §8  classify_merge_level  (MAJOR has priority over MINOR)
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("classify_merge_level: no conditions → NONE", "[merge_level]") {
    CHECK(classify_merge_level(0u,       0u,     0.50f, 0.10f) == MergeLevel::NONE);
    CHECK(classify_merge_level(9'999u,   999u,   0.50f, 0.05f) == MergeLevel::NONE);
}

TEST_CASE("classify_merge_level: only minor trigger active → MINOR", "[merge_level]") {
    CHECK(classify_merge_level(10'001u, 0u,     0.50f, 0.10f) == MergeLevel::MINOR);
    CHECK(classify_merge_level(0u,      2'000u, 0.50f, 0.10f) == MergeLevel::MINOR);
}

TEST_CASE("classify_merge_level: only major trigger active → MAJOR", "[merge_level]") {
    CHECK(classify_merge_level(0u,     0u,     0.10f, 0.10f) == MergeLevel::MAJOR);
    CHECK(classify_merge_level(0u,     0u,     0.50f, 0.30f) == MergeLevel::MAJOR);
}

TEST_CASE("classify_merge_level: both minor and major active → MAJOR wins", "[merge_level]") {
    // Minor triggered by size + major triggered by ATP
    CHECK(classify_merge_level(20'000u, 2'000u, 0.05f, 0.05f) == MergeLevel::MAJOR);
}

// ────────────────────────────────────────────────────────────────────────────
// §9  Visibility lag  T_lag = T_batch + T_merge + T_swap
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("visibility_lag_ms: zero components → zero", "[visibility]") {
    CHECK(visibility_lag_ms(0.0, 0.0, 0.0) == Approx(0.0).margin(1e-12));
}

TEST_CASE("visibility_lag_ms: sum of components", "[visibility]") {
    // 200 ms batch + 10 ms merge + 0.001 ms swap ≈ 210.001 ms
    CHECK(visibility_lag_ms(200.0, 10.0, 0.001) == Approx(210.001).epsilon(1e-9));
}

TEST_CASE("visibility_lag_ms: negative component throws", "[visibility]") {
    CHECK_THROWS_AS(visibility_lag_ms(-1.0, 0.0, 0.0), std::invalid_argument);
    CHECK_THROWS_AS(visibility_lag_ms(0.0, -1.0, 0.0), std::invalid_argument);
    CHECK_THROWS_AS(visibility_lag_ms(0.0, 0.0, -1.0), std::invalid_argument);
}

TEST_CASE("is_visibility_lag_acceptable: below and exactly at 500 ms", "[visibility]") {
    CHECK(is_visibility_lag_acceptable(0.0)    == true);
    CHECK(is_visibility_lag_acceptable(499.9)  == true);
    CHECK(is_visibility_lag_acceptable(500.0)  == true);   // ≤ 500 passes
}

TEST_CASE("is_visibility_lag_acceptable: above 500 ms fails", "[visibility]") {
    CHECK(is_visibility_lag_acceptable(500.1)  == false);
    CHECK(is_visibility_lag_acceptable(1000.0) == false);
}

TEST_CASE("is_minor_merge_within_budget: below 10 ms passes", "[visibility]") {
    CHECK(is_minor_merge_within_budget(0.0)  == true);
    CHECK(is_minor_merge_within_budget(9.9)  == true);
}

TEST_CASE("is_minor_merge_within_budget: 10 ms exactly fails (strictly <)", "[visibility]") {
    CHECK(is_minor_merge_within_budget(10.0) == false);
    CHECK(is_minor_merge_within_budget(10.1) == false);
}

// ────────────────────────────────────────────────────────────────────────────
// §10  Snapshot isolation predicates
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("has_active_readers: ref_count < 2 means no live query", "[snapshot]") {
    CHECK(has_active_readers(0L)  == false);
    CHECK(has_active_readers(1L)  == false);
}

TEST_CASE("has_active_readers: ref_count >= 2 means live query", "[snapshot]") {
    CHECK(has_active_readers(2L)  == true);
    CHECK(has_active_readers(10L) == true);
}

TEST_CASE("is_snapshot_safe_to_discard: complement of has_active_readers", "[snapshot]") {
    CHECK(is_snapshot_safe_to_discard(0L)  == true);
    CHECK(is_snapshot_safe_to_discard(1L)  == true);
    CHECK(is_snapshot_safe_to_discard(2L)  == false);
    CHECK(is_snapshot_safe_to_discard(10L) == false);
}

TEST_CASE("Snapshot predicates are exclusive for 2+", "[snapshot]") {
    for (long r = 0L; r <= 5L; ++r) {
        // Exactly one of the two is true at any ref_count
        const bool active   = has_active_readers(r);
        const bool safe_del = is_snapshot_safe_to_discard(r);
        CHECK(active != safe_del);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// §11  Hilbert key ordering
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("hilbert_precedes: basic ordering", "[hilbert]") {
    CHECK(hilbert_precedes(0u,  1u)  == true);
    CHECK(hilbert_precedes(100u, 101u) == true);
    CHECK(hilbert_precedes(5u,  5u)  == false);  // equal is not "precedes"
    CHECK(hilbert_precedes(10u, 9u)  == false);
}

TEST_CASE("is_hilbert_sorted: empty and single-element arrays", "[hilbert]") {
    CHECK(is_hilbert_sorted(nullptr, 0u) == true);
    const std::array<uint64_t, 1> single = {42u};
    CHECK(is_hilbert_sorted(single.data(), 1u) == true);
}

TEST_CASE("is_hilbert_sorted: sorted array returns true", "[hilbert]") {
    const std::array<uint64_t, 5> sorted = {1u, 5u, 10u, 20u, 100u};
    CHECK(is_hilbert_sorted(sorted.data(), sorted.size()) == true);
}

TEST_CASE("is_hilbert_sorted: duplicate keys are allowed (non-strict)", "[hilbert]") {
    const std::array<uint64_t, 3> with_dupes = {1u, 5u, 5u};
    CHECK(is_hilbert_sorted(with_dupes.data(), with_dupes.size()) == true);
}

TEST_CASE("is_hilbert_sorted: unsorted array returns false", "[hilbert]") {
    const std::array<uint64_t, 4> unsorted = {1u, 5u, 3u, 10u};
    CHECK(is_hilbert_sorted(unsorted.data(), unsorted.size()) == false);
}

TEST_CASE("key_in_range: within and outside range", "[hilbert]") {
    CHECK(key_in_range(50u,  0u,  100u) == true);
    CHECK(key_in_range(0u,   0u,  100u) == true);   // lower bound inclusive
    CHECK(key_in_range(100u, 0u,  100u) == true);   // upper bound inclusive
    CHECK(key_in_range(101u, 0u,  100u) == false);
    CHECK(key_in_range(0u,   1u,  100u) == false);  // below lower bound
}

// ────────────────────────────────────────────────────────────────────────────
// §12  Merge poll scheduling
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("is_merge_poll_due: below 100 ms", "[poll]") {
    CHECK(is_merge_poll_due(0u)   == false);
    CHECK(is_merge_poll_due(50u)  == false);
    CHECK(is_merge_poll_due(99u)  == false);
}

TEST_CASE("is_merge_poll_due: at exactly 100 ms (≥ threshold)", "[poll]") {
    CHECK(is_merge_poll_due(100u)  == true);
    CHECK(is_merge_poll_due(200u)  == true);
    CHECK(is_merge_poll_due(5000u) == true);
}

TEST_CASE("poll_cycles_in_window: expected cycle counts", "[poll]") {
    CHECK(poll_cycles_in_window(0u)     == 0u);
    CHECK(poll_cycles_in_window(100u)   == 1u);
    CHECK(poll_cycles_in_window(500u)   == 5u);
    CHECK(poll_cycles_in_window(1'000u) == 10u);
    // Visibility lag window: 500ms / 100ms = 5 cycles
    CHECK(poll_cycles_in_window(MAX_VISIBILITY_LAG_MS) == 5u);
}

// ────────────────────────────────────────────────────────────────────────────
// §13  Atomic-swap safety predicates
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("is_safe_swap_point: only true between ticks", "[atomic_swap]") {
    CHECK(is_safe_swap_point(true)  == true);
    CHECK(is_safe_swap_point(false) == false);
}

TEST_CASE("shadow_ready_for_swap: only SHADOW_BUILD state is ready", "[atomic_swap]") {
    CHECK(shadow_ready_for_swap(IndexState::SHADOW_BUILD) == true);
    CHECK(shadow_ready_for_swap(IndexState::STABLE)       == false);
    CHECK(shadow_ready_for_swap(IndexState::SWAPPING)     == false);
}

// ────────────────────────────────────────────────────────────────────────────
// §14  Diagnostic names
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("merge_level_name: all enumerants", "[diagnostics]") {
    CHECK(merge_level_name(MergeLevel::NONE)  == "NONE");
    CHECK(merge_level_name(MergeLevel::MINOR) == "MINOR");
    CHECK(merge_level_name(MergeLevel::MAJOR) == "MAJOR");
}

TEST_CASE("rebuild_reason_name: all enumerants", "[diagnostics]") {
    CHECK(rebuild_reason_name(RebuildReason::NONE)          == "NONE");
    CHECK(rebuild_reason_name(RebuildReason::NAP_ATP)       == "NAP_ATP");
    CHECK(rebuild_reason_name(RebuildReason::FRAGMENTATION) == "FRAGMENTATION");
    CHECK(rebuild_reason_name(RebuildReason::BOTH)          == "BOTH");
}

TEST_CASE("index_state_name: all enumerants", "[diagnostics]") {
    CHECK(index_state_name(IndexState::STABLE)       == "STABLE");
    CHECK(index_state_name(IndexState::SHADOW_BUILD) == "SHADOW_BUILD");
    CHECK(index_state_name(IndexState::SWAPPING)     == "SWAPPING");
}

TEST_CASE("Diagnostic names are non-empty string_views", "[diagnostics]") {
    CHECK_FALSE(merge_level_name(MergeLevel::MINOR).empty());
    CHECK_FALSE(rebuild_reason_name(RebuildReason::BOTH).empty());
    CHECK_FALSE(index_state_name(IndexState::SWAPPING).empty());
}

// ────────────────────────────────────────────────────────────────────────────
// §15  Invariants
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("Invariant: flush threshold > 0", "[invariants]") {
    CHECK(MEMTABLE_FLUSH_THRESHOLD > 0u);
}

TEST_CASE("Invariant: poll cycles in visibility window equals expected", "[invariants]") {
    // 500 ms / 100 ms = exactly 5 polling cycles within the lag window
    CHECK(poll_cycles_in_window(MAX_VISIBILITY_LAG_MS) == 5u);
}

TEST_CASE("Invariant: minor merge budget < visibility lag", "[invariants]") {
    CHECK(MINOR_MERGE_MAX_MS < MAX_VISIBILITY_LAG_MS);
}

TEST_CASE("Invariant: classify_merge_level and should_trigger agree", "[invariants]") {
    // When MAJOR fires, both major and minor are considered triggered
    const MergeLevel level = classify_merge_level(0u, 0u, 0.05f, 0.30f);
    CHECK(level == MergeLevel::MAJOR);
    CHECK(should_trigger_major_rebuild(0.05f, 0.30f) == true);
}

TEST_CASE("Invariant: ATP threshold and fragmentation threshold are both < 1.0", "[invariants]") {
    CHECK(MAJOR_REBUILD_ATP_THRESHOLD    < 1.0f);
    CHECK(FRAGMENTATION_REBUILD_THRESHOLD < 1.0f);
}

TEST_CASE("Invariant: visibility_lag_ms is additive (order doesn't matter)", "[invariants]") {
    const double a = 200.0, b = 10.0, c = 0.001;
    CHECK(visibility_lag_ms(a, b, c) == Approx(visibility_lag_ms(c, a, b)).epsilon(1e-9));
    CHECK(visibility_lag_ms(a, b, c) == Approx(visibility_lag_ms(b, c, a)).epsilon(1e-9));
}

TEST_CASE("Invariant: zero-lag is always acceptable", "[invariants]") {
    CHECK(is_visibility_lag_acceptable(0.0) == true);
}

// ────────────────────────────────────────────────────────────────────────────
// §16  Integration scenarios
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("Integration: normal operation — MemTable fills then flushes", "[integration]") {
    // System starts clean; no triggers
    CHECK(classify_merge_level(0u, 0u, 0.80f, 0.05f) == MergeLevel::NONE);

    // After 100 ms poll check — still nothing if small MemTable
    CHECK(is_merge_poll_due(100u) == true);
    CHECK(classify_merge_level(500u, 100u, 0.80f, 0.05f) == MergeLevel::NONE);

    // MemTable exceeds threshold after rapid ingestion
    CHECK(classify_merge_level(10'001u, 100u, 0.80f, 0.05f) == MergeLevel::MINOR);

    // Minor merge completes in 5 ms — within budget
    CHECK(is_minor_merge_within_budget(5.0) == true);

    // Total lag = 100ms batch + 5ms merge + 0.001ms swap
    const double lag = visibility_lag_ms(100.0, 5.0, 0.001);
    CHECK(is_visibility_lag_acceptable(lag) == true);
}

TEST_CASE("Integration: Nap State triggers major rebuild", "[integration]") {
    // ATP drops to 12%; fragmentation is fine (10%)
    const float atp   = 0.12f;
    const float frag  = 0.10f;

    CHECK(classify_rebuild_reason(atp, frag) == RebuildReason::NAP_ATP);
    CHECK(classify_merge_level(0u, 0u, atp, frag) == MergeLevel::MAJOR);
    CHECK(should_trigger_major_rebuild(atp, frag) == true);
    // After rebuild, fragmentation resets; system stable again
    CHECK(classify_merge_level(0u, 0u, 0.50f, 0.05f) == MergeLevel::NONE);
}

TEST_CASE("Integration: Shadow merge swap protocol", "[integration]") {
    // Merger has built shadow — phase transitions
    CHECK(shadow_ready_for_swap(IndexState::SHADOW_BUILD)  == true);
    CHECK(shadow_ready_for_swap(IndexState::STABLE)        == false);

    // Swap only allowed between ticks
    CHECK(is_safe_swap_point(false) == false);  // mid-tick: no
    CHECK(is_safe_swap_point(true)  == true);   // tick boundary: yes

    // After swap, old snapshot ref-count drops; if no readers → safe to free
    CHECK(is_snapshot_safe_to_discard(1L) == true);
    CHECK(has_active_readers(1L)          == false);
}

TEST_CASE("Integration: Hilbert sort preserved through merge lifecycle", "[integration]") {
    // Pre-merge: unsorted MemTable drain is expected (written in arrival order)
    const std::array<uint64_t, 4> unsorted = {50u, 10u, 80u, 30u};
    CHECK(is_hilbert_sorted(unsorted.data(), unsorted.size()) == false);

    // Post-merge (sorted): Active Index sorted by Hilbert key
    std::vector<uint64_t> to_sort(unsorted.begin(), unsorted.end());
    std::sort(to_sort.begin(), to_sort.end());
    CHECK(is_hilbert_sorted(to_sort.data(), to_sort.size()) == true);

    // Keys are partition-routable after sort
    CHECK(key_in_range(to_sort[0], 0u, 50u)   == true);
    CHECK(key_in_range(to_sort[3], 50u, 100u)  == true);
}

TEST_CASE("Integration: spec scenario — 1-second mandatory flush", "[integration]") {
    // Empty MemTable for 1001 ms → time trigger fires
    const size_t   empty_memtable = 0u;
    const uint64_t elapsed        = 1'001u;

    CHECK(should_trigger_minor_merge(empty_memtable, elapsed) == true);
    CHECK(classify_merge_level(empty_memtable, elapsed, 0.80f, 0.05f) == MergeLevel::MINOR);
}

TEST_CASE("Integration: worst-case lag still within budget", "[integration]") {
    // Batch window = 1 second (flush interval = long wait)
    // Minor merge ≈ 9.9 ms (just within budget)
    // Swap ≈ 0.001 ms (nanoseconds)
    const double lag = visibility_lag_ms(
        static_cast<double>(FLUSH_INTERVAL_MS),
        9.9,
        0.001);
    // Total ≈ 1009.9 ms — exceeds 500 ms spec
    // Spec says max lag is 500 ms → batch must be bounded to ≤ 490 ms
    // (The merge+swap are negligible; batching must stay << 500 ms)
    // Verify: 100ms batch (nominal poll interval) + 9.9ms merge fits
    const double nominal_lag = visibility_lag_ms(
        static_cast<double>(MERGE_POLL_INTERVAL_MS),
        9.9,
        0.001);
    CHECK(is_visibility_lag_acceptable(nominal_lag) == true);
    // The 1-second-batch scenario exceeds the budget (confirms spec requires
    // the batch period to be poll-driven at 100 ms, not flush-at-1s)
    CHECK(is_visibility_lag_acceptable(lag) == false);
}
