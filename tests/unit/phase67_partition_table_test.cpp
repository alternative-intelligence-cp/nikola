// ============================================================
// Phase 67 — GAP-019 Distributed Partition Table Update Protocol
// tests/unit/phase67_partition_table_test.cpp
//
// Test domains:
//  §1  Constants
//  §2  load_imbalance_factor
//  §3  is_rebalance_triggered
//  §4  is_vram_safe
//  §5  node / migration payload sizing
//  §6  classify_epoch_message
//  §7  stale message handling rules
//  §8  epoch_is_valid_successor
//  §9  timeout and penalty constants
//  §10 compute_crc32c / crc32c_matches
//  §11 is_positive_definite
//  §12 state_allows_physics / state_is_suspended
//  §13 is_valid_state_transition
//  §14 diagnostic name helpers
//  §15 Invariants
//  §16 Integration scenarios
// ============================================================
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/infrastructure/partition_table.hpp>

#include <array>
#include <cstdint>
#include <vector>

using namespace nikola::infrastructure;
using Catch::Approx;

// ── §1 Constants ─────────────────────────────────────────────────────────────

TEST_CASE("§1.1 LIF trigger threshold is 0.20", "[constants][GAP-019]") {
    REQUIRE(LIF_TRIGGER_THRESHOLD == Approx(0.20f));
}

TEST_CASE("§1.2 VRAM safety fraction is 0.90", "[constants][GAP-019]") {
    REQUIRE(VRAM_SAFETY_FRACTION == Approx(0.90f));
}

TEST_CASE("§1.3 Rollback timeout is 5000 ms", "[constants][GAP-019]") {
    REQUIRE(ROLLBACK_TIMEOUT_MS == 5'000u);
}

TEST_CASE("§1.4 Stability penalty is 1 hour (3 600 000 ms)", "[constants][GAP-019]") {
    REQUIRE(STABILITY_PENALTY_MS == 3'600'000ULL);
    // Sanity: 60 s × 60 min = 3600 s
    REQUIRE(STABILITY_PENALTY_MS == 3600u * 1000u);
}

TEST_CASE("§1.5 Structural sizing constants", "[constants][GAP-019]") {
    REQUIRE(METRIC_TENSOR_COMPONENTS == 45);   // (9×10)/2
    REQUIRE(NEUROCHEMICAL_COMPONENTS == 5);    // r, s, u, v, w
    REQUIRE(MORTON_KEY_BYTES         == 16);   // 128-bit
    REQUIRE(NODE_PAYLOAD_BYTES       == 232u);
}

TEST_CASE("§1.6 Pause duration range is reasonable", "[constants][GAP-019]") {
    REQUIRE(PAUSE_DURATION_MIN_MS == 10u);
    REQUIRE(PAUSE_DURATION_MAX_MS == 50u);
    REQUIRE(PAUSE_DURATION_MIN_MS < PAUSE_DURATION_MAX_MS);
}

// ── §2 load_imbalance_factor ─────────────────────────────────────────────────

TEST_CASE("§2.1 Balanced cluster gives LIF = 0", "[lif][GAP-019]") {
    const std::array<uint64_t,4> counts{100,100,100,100};
    REQUIRE(load_imbalance_factor(counts) == Approx(0.0f).margin(1e-6f));
}

TEST_CASE("§2.2 Single rank gives LIF = 0", "[lif][GAP-019]") {
    const std::array<uint64_t,1> counts{500};
    REQUIRE(load_imbalance_factor(counts) == Approx(0.0f).margin(1e-6f));
}

TEST_CASE("§2.3 Spec example: max=240 min=160 avg=200 → LIF=0.40", "[lif][GAP-019]") {
    // (240−160)/200 = 80/200 = 0.4
    const std::array<uint64_t,2> counts{160, 240};
    // avg = 200, max−min = 80 → LIF = 0.4
    REQUIRE(load_imbalance_factor(counts) == Approx(0.4f).epsilon(0.01f));
}

TEST_CASE("§2.4 LIF exceeds trigger at 21% imbalance", "[lif][GAP-019]") {
    // max=210, min=190, avg=200 → LIF = 20/200 = 0.10 → below trigger
    const std::array<uint64_t,2> below{190, 210};
    REQUIRE(load_imbalance_factor(below) == Approx(0.10f).epsilon(0.01f));

    // max=242, min=158, avg=200 → LIF = 84/200 = 0.42 → above trigger
    const std::array<uint64_t,2> above{158, 242};
    REQUIRE(load_imbalance_factor(above) > LIF_TRIGGER_THRESHOLD);
}

TEST_CASE("§2.5 LIF is non-negative for all valid inputs", "[lif][GAP-019]") {
    const std::array<uint64_t,3> counts{50, 100, 150};
    REQUIRE(load_imbalance_factor(counts) >= 0.0f);
}

TEST_CASE("§2.6 Three-rank mixed load LIF formula", "[lif][GAP-019]") {
    // Counts: 50, 100, 150  avg = 100, max-min = 100 → LIF = 1.0
    const std::array<uint64_t,3> counts{50, 100, 150};
    REQUIRE(load_imbalance_factor(counts) == Approx(1.0f).epsilon(0.01f));
}

TEST_CASE("§2.7 Empty span throws invalid_argument", "[lif][error][GAP-019]") {
    REQUIRE_THROWS_AS(load_imbalance_factor(std::span<const uint64_t>{}),
                      std::invalid_argument);
}

// ── §3 is_rebalance_triggered ────────────────────────────────────────────────

TEST_CASE("§3.1 LIF < threshold → no rebalance", "[trigger][GAP-019]") {
    REQUIRE_FALSE(is_rebalance_triggered(0.0f));
    REQUIRE_FALSE(is_rebalance_triggered(0.10f));
    REQUIRE_FALSE(is_rebalance_triggered(0.19f));
}

TEST_CASE("§3.2 LIF exactly at threshold → no rebalance (strict >)", "[trigger][GAP-019]") {
    REQUIRE_FALSE(is_rebalance_triggered(0.20f));
}

TEST_CASE("§3.3 LIF > threshold → rebalance triggered", "[trigger][GAP-019]") {
    REQUIRE(is_rebalance_triggered(0.201f));
    REQUIRE(is_rebalance_triggered(0.50f));
    REQUIRE(is_rebalance_triggered(1.00f));
}

// ── §4 is_vram_safe ───────────────────────────────────────────────────────────

TEST_CASE("§4.1 Well below 90% ceiling is safe", "[vram][GAP-019]") {
    const uint64_t capacity = 80ULL * 1024u * 1024u * 1024u; // 80 GB
    const uint64_t used     = 40ULL * 1024u * 1024u * 1024u; // 50%
    REQUIRE(is_vram_safe(used, capacity));
}

TEST_CASE("§4.2 Exactly at 90% ceiling is unsafe (spec: abort if ≥ 90%)", "[vram][GAP-019]") {
    const uint64_t capacity = 1'000'000u;
    const uint64_t at_ceiling = 900'000u;
    // is_vram_safe returns false when estimated >= 90%
    REQUIRE_FALSE(is_vram_safe(at_ceiling, capacity));
}

TEST_CASE("§4.3 One byte below 90% ceiling is safe", "[vram][GAP-019]") {
    const uint64_t capacity = 1'000'000u;
    const uint64_t just_under = 899'999u;
    REQUIRE(is_vram_safe(just_under, capacity));
}

TEST_CASE("§4.4 Above ceiling triggers abort path", "[vram][GAP-019]") {
    const uint64_t capacity = 1'000'000u;
    REQUIRE_FALSE(is_vram_safe(950'000u, capacity));
    REQUIRE_FALSE(is_vram_safe(1'000'000u, capacity));
}

TEST_CASE("§4.5 Empty node import (0 bytes) is always safe", "[vram][GAP-019]") {
    REQUIRE(is_vram_safe(0u, 1'000'000u));
}

// ── §5 Payload Sizing ─────────────────────────────────────────────────────────

TEST_CASE("§5.1 node_payload_bytes() matches spec decomposition", "[payload][GAP-019]") {
    // 16 (Morton) + 4×4 (Ψ fields) + 45×4 (metric) + 5×4 (neurochemical) = 232
    constexpr size_t expected = 16u + 16u + 180u + 20u;
    REQUIRE(node_payload_bytes() == expected);
    REQUIRE(node_payload_bytes() == 232u);
}

TEST_CASE("§5.2 node_payload_bytes() runtime matches NODE_PAYLOAD_BYTES constant", "[payload][GAP-019]") {
    REQUIRE(node_payload_bytes() == NODE_PAYLOAD_BYTES);
}

TEST_CASE("§5.3 migration_payload_bytes scales linearly with node count", "[payload][GAP-019]") {
    REQUIRE(migration_payload_bytes(0)   == 0u);
    REQUIRE(migration_payload_bytes(1)   == 232u);
    REQUIRE(migration_payload_bytes(100) == 23'200u);
    REQUIRE(migration_payload_bytes(1000) == 232'000u);
}

TEST_CASE("§5.4 Payload divisible by 4 (float alignment)", "[payload][GAP-019]") {
    REQUIRE(node_payload_bytes() % 4u == 0u);
}

// ── §6 classify_epoch_message ────────────────────────────────────────────────

TEST_CASE("§6.1 Equal epochs → CURRENT", "[epoch][GAP-019]") {
    REQUIRE(classify_epoch_message(5, 5) == EpochMessageRelation::CURRENT);
    REQUIRE(classify_epoch_message(0, 0) == EpochMessageRelation::CURRENT);
}

TEST_CASE("§6.2 msg.epoch < local.epoch → STALE_VALID", "[epoch][GAP-019]") {
    REQUIRE(classify_epoch_message(4, 5) == EpochMessageRelation::STALE_VALID);
    REQUIRE(classify_epoch_message(0, 100) == EpochMessageRelation::STALE_VALID);
}

TEST_CASE("§6.3 msg.epoch > local.epoch → FROM_FUTURE", "[epoch][GAP-019]") {
    REQUIRE(classify_epoch_message(6, 5) == EpochMessageRelation::FROM_FUTURE);
    REQUIRE(classify_epoch_message(100, 0) == EpochMessageRelation::FROM_FUTURE);
}

TEST_CASE("§6.4 One-step transitions produce expected relations", "[epoch][GAP-019]") {
    // Sender migrated one step ahead — local hasn't committed yet
    REQUIRE(classify_epoch_message(11, 10) == EpochMessageRelation::FROM_FUTURE);
    // Sender one step behind — stale
    REQUIRE(classify_epoch_message(9,  10) == EpochMessageRelation::STALE_VALID);
}

// ── §7 Stale Message Handling ─────────────────────────────────────────────────

TEST_CASE("§7.1 Stale + still owner → process (not forward)", "[stale][GAP-019]") {
    REQUIRE(stale_should_process(true)  == true);
    REQUIRE(stale_should_forward(true)  == false);
}

TEST_CASE("§7.2 Stale + ownership moved → forward (not process)", "[stale][GAP-019]") {
    REQUIRE(stale_should_process(false) == false);
    REQUIRE(stale_should_forward(false) == true);
}

TEST_CASE("§7.3 Process and forward are mutually exclusive", "[stale][GAP-019]") {
    for (bool owner : {true, false}) {
        // Exactly one of process/forward should be true
        REQUIRE(stale_should_process(owner) != stale_should_forward(owner));
    }
}

// ── §8 Epoch Sequencing ───────────────────────────────────────────────────────

TEST_CASE("§8.1 Valid successor: next == current + 1", "[epoch_seq][GAP-019]") {
    REQUIRE(epoch_is_valid_successor(1, 0));
    REQUIRE(epoch_is_valid_successor(100, 99));
    REQUIRE(epoch_is_valid_successor(1'000'000, 999'999));
}

TEST_CASE("§8.2 Same epoch is not a valid successor", "[epoch_seq][GAP-019]") {
    REQUIRE_FALSE(epoch_is_valid_successor(5, 5));
}

TEST_CASE("§8.3 Skip by 2+ is not a valid successor", "[epoch_seq][GAP-019]") {
    REQUIRE_FALSE(epoch_is_valid_successor(7, 5));
    REQUIRE_FALSE(epoch_is_valid_successor(0, 5));
}

// ── §9 Timeout and Penalty ────────────────────────────────────────────────────

TEST_CASE("§9.1 rollback_timeout_ms() returns spec value", "[timeout][GAP-019]") {
    REQUIRE(rollback_timeout_ms() == 5'000u);
}

TEST_CASE("§9.2 stability_penalty_ms() returns 1-hour spec value", "[timeout][GAP-019]") {
    REQUIRE(stability_penalty_ms() == 3'600'000ULL);
}

TEST_CASE("§9.3 Stability penalty >> rollback timeout", "[timeout][GAP-019]") {
    REQUIRE(stability_penalty_ms() > rollback_timeout_ms());
    // Penalty is ×720 the timeout — imposes a long cooldown
    REQUIRE(stability_penalty_ms() >= 720u * rollback_timeout_ms());
}

// ── §10 CRC32C ────────────────────────────────────────────────────────────────

TEST_CASE("§10.1 CRC32C of empty span", "[crc32c][GAP-019]") {
    // CRC32C("") = 0x00000000 (final XOR of 0xFFFFFFFF ^ 0xFFFFFFFF)
    const uint32_t crc = compute_crc32c({});
    REQUIRE(crc == 0x00000000u);
}

TEST_CASE("§10.2 CRC32C well-known vector: {0x00} = 0x527D5351", "[crc32c][GAP-019]") {
    const std::array<uint8_t,1> data{0x00u};
    REQUIRE(compute_crc32c(data) == 0x527D5351u);
}

TEST_CASE("§10.3 CRC32C well-known vector: {0xFF} = 0xFF000000", "[crc32c][GAP-019]") {
    const std::array<uint8_t,1> data{0xFFu};
    REQUIRE(compute_crc32c(data) == 0xFF000000u);
}

TEST_CASE("§10.4 CRC32C of '123456789' = 0xE3069283", "[crc32c][GAP-019]") {
    // Standard CRC32C check vector (IETF RFC 3720 §B.4)
    const std::array<uint8_t,9> data{
        '1','2','3','4','5','6','7','8','9'
    };
    REQUIRE(compute_crc32c(data) == 0xE306'9283u);
}

TEST_CASE("§10.5 crc32c_matches returns true for correct checksum", "[crc32c][GAP-019]") {
    const std::array<uint8_t,4> payload{0x01u,0x02u,0x03u,0x04u};
    const uint32_t expected = compute_crc32c(payload);
    REQUIRE(crc32c_matches(payload, expected));
}

TEST_CASE("§10.6 crc32c_matches returns false for corrupted data", "[crc32c][GAP-019]") {
    const std::array<uint8_t,4> original{0x10u,0x20u,0x30u,0x40u};
    const uint32_t expected = compute_crc32c(original);

    std::array<uint8_t,4> corrupted = original;
    corrupted[2] ^= 0xFFu;  // flip byte
    REQUIRE_FALSE(crc32c_matches(corrupted, expected));
}

TEST_CASE("§10.7 CRC32C is deterministic", "[crc32c][GAP-019]") {
    const std::array<uint8_t,8> data{0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88};
    REQUIRE(compute_crc32c(data) == compute_crc32c(data));
}

TEST_CASE("§10.8 Different data produces different CRC32C (high probability)", "[crc32c][GAP-019]") {
    const std::array<uint8_t,4> a{0x01,0x02,0x03,0x04};
    const std::array<uint8_t,4> b{0x01,0x02,0x03,0x05};
    REQUIRE(compute_crc32c(a) != compute_crc32c(b));
}

// ── §11 is_positive_definite ─────────────────────────────────────────────────

TEST_CASE("§11.1 All positive eigenvalues → SPD", "[spd][GAP-019]") {
    const std::array<float,3> ev{1.0f, 2.5f, 0.001f};
    REQUIRE(is_positive_definite(ev));
}

TEST_CASE("§11.2 Any non-positive eigenvalue fails SPD", "[spd][GAP-019]") {
    const std::array<float,3> zero_ev{1.0f, 0.0f, 2.0f};
    REQUIRE_FALSE(is_positive_definite(zero_ev));

    const std::array<float,3> neg_ev{1.0f, -0.1f, 2.0f};
    REQUIRE_FALSE(is_positive_definite(neg_ev));
}

TEST_CASE("§11.3 Single positive eigenvalue → SPD", "[spd][GAP-019]") {
    const std::array<float,1> ev{0.0001f};
    REQUIRE(is_positive_definite(ev));
}

TEST_CASE("§11.4 Flat metric (all equal positive) → SPD", "[spd][GAP-019]") {
    const std::array<float,9> ev{1,1,1,1,1,1,1,1,1};
    REQUIRE(is_positive_definite(ev));
}

TEST_CASE("§11.5 Empty span throws invalid_argument", "[spd][error][GAP-019]") {
    REQUIRE_THROWS_AS(is_positive_definite(std::span<const float>{}),
                      std::invalid_argument);
}

// ── §12 Physics Gate ──────────────────────────────────────────────────────────

TEST_CASE("§12.1 Only STABLE allows physics", "[state][GAP-019]") {
    REQUIRE(state_allows_physics(MigrationState::STABLE));
    REQUIRE_FALSE(state_allows_physics(MigrationState::PAUSED));
    REQUIRE_FALSE(state_allows_physics(MigrationState::MIGRATING));
    REQUIRE_FALSE(state_allows_physics(MigrationState::COMMITTING));
}

TEST_CASE("§12.2 All non-STABLE states are suspended", "[state][GAP-019]") {
    REQUIRE_FALSE(state_is_suspended(MigrationState::STABLE));
    REQUIRE(state_is_suspended(MigrationState::PAUSED));
    REQUIRE(state_is_suspended(MigrationState::MIGRATING));
    REQUIRE(state_is_suspended(MigrationState::COMMITTING));
}

TEST_CASE("§12.3 allows_physics and is_suspended are strict complements", "[state][GAP-019]") {
    for (auto s : {MigrationState::STABLE, MigrationState::PAUSED,
                   MigrationState::MIGRATING, MigrationState::COMMITTING}) {
        REQUIRE(state_allows_physics(s) != state_is_suspended(s));
    }
}

// ── §13 State Transitions ─────────────────────────────────────────────────────

TEST_CASE("§13.1 Legal 2P-EBP forward transitions", "[transition][GAP-019]") {
    using S = MigrationState;
    REQUIRE(is_valid_state_transition(S::STABLE,     S::PAUSED));
    REQUIRE(is_valid_state_transition(S::PAUSED,     S::MIGRATING));
    REQUIRE(is_valid_state_transition(S::MIGRATING,  S::COMMITTING));
    REQUIRE(is_valid_state_transition(S::COMMITTING, S::STABLE));
}

TEST_CASE("§13.2 ROLLBACK — any state → STABLE is always legal", "[transition][GAP-019]") {
    using S = MigrationState;
    REQUIRE(is_valid_state_transition(S::STABLE,     S::STABLE));
    REQUIRE(is_valid_state_transition(S::PAUSED,     S::STABLE));
    REQUIRE(is_valid_state_transition(S::MIGRATING,  S::STABLE));
    REQUIRE(is_valid_state_transition(S::COMMITTING, S::STABLE));
}

TEST_CASE("§13.3 Skip transitions are illegal", "[transition][GAP-019]") {
    using S = MigrationState;
    // Cannot skip PAUSED
    REQUIRE_FALSE(is_valid_state_transition(S::STABLE, S::MIGRATING));
    REQUIRE_FALSE(is_valid_state_transition(S::STABLE, S::COMMITTING));
    // Cannot go backwards (except via ROLLBACK → STABLE)
    REQUIRE_FALSE(is_valid_state_transition(S::MIGRATING, S::PAUSED));
    REQUIRE_FALSE(is_valid_state_transition(S::COMMITTING, S::PAUSED));
    REQUIRE_FALSE(is_valid_state_transition(S::COMMITTING, S::MIGRATING));
}

// ── §14 Diagnostic Names ──────────────────────────────────────────────────────

TEST_CASE("§14.1 message_type_name covers all enum values", "[names][GAP-019]") {
    REQUIRE(message_type_name(PartitionMessageType::HEARTBEAT)         == "HEARTBEAT");
    REQUIRE(message_type_name(PartitionMessageType::PREPARE_MIGRATION) == "PREPARE_MIGRATION");
    REQUIRE(message_type_name(PartitionMessageType::BEGIN_MIGRATION)   == "BEGIN_MIGRATION");
    REQUIRE(message_type_name(PartitionMessageType::COMMIT_EPOCH)      == "COMMIT_EPOCH");
    REQUIRE(message_type_name(PartitionMessageType::ROLLBACK)          == "ROLLBACK");
    REQUIRE(message_type_name(PartitionMessageType::ABORT)             == "ABORT");
}

TEST_CASE("§14.2 migration_state_name covers all enum values", "[names][GAP-019]") {
    REQUIRE(migration_state_name(MigrationState::STABLE)     == "STABLE");
    REQUIRE(migration_state_name(MigrationState::PAUSED)     == "PAUSED");
    REQUIRE(migration_state_name(MigrationState::MIGRATING)  == "MIGRATING");
    REQUIRE(migration_state_name(MigrationState::COMMITTING) == "COMMITTING");
}

TEST_CASE("§14.3 epoch_relation_name covers all enum values", "[names][GAP-019]") {
    REQUIRE(epoch_relation_name(EpochMessageRelation::STALE_VALID) == "STALE_VALID");
    REQUIRE(epoch_relation_name(EpochMessageRelation::CURRENT)     == "CURRENT");
    REQUIRE(epoch_relation_name(EpochMessageRelation::FROM_FUTURE) == "FROM_FUTURE");
}

// ── §15 Invariants ────────────────────────────────────────────────────────────

TEST_CASE("§15.1 LIF ≥ 0 for all valid non-empty inputs", "[invariants][GAP-019]") {
    const std::array<uint64_t,4> counts{10, 20, 30, 40};
    REQUIRE(load_imbalance_factor(counts) >= 0.0f);

    const std::array<uint64_t,2> equal{100, 100};
    REQUIRE(load_imbalance_factor(equal) >= 0.0f);
}

TEST_CASE("§15.2 LIF = 0 iff max == min (balanced cluster)", "[invariants][GAP-019]") {
    const std::array<uint64_t,5> balanced{200, 200, 200, 200, 200};
    REQUIRE(load_imbalance_factor(balanced) == Approx(0.0f).margin(1e-6f));

    const std::array<uint64_t,2> unbalanced{100, 300};
    REQUIRE(load_imbalance_factor(unbalanced) > 0.0f);
}

TEST_CASE("§15.3 Stability penalty is exactly 720 × rollback timeout", "[invariants][GAP-019]") {
    // 1 hour / 5s = 720 windows
    REQUIRE(STABILITY_PENALTY_MS == static_cast<uint64_t>(ROLLBACK_TIMEOUT_MS) * 720u);
}

TEST_CASE("§15.4 node_payload_bytes decomposes correctly", "[invariants][GAP-019]") {
    size_t reconstructed =
        static_cast<size_t>(MORTON_KEY_BYTES)
      + 4u * sizeof(float)
      + static_cast<size_t>(METRIC_TENSOR_COMPONENTS) * sizeof(float)
      + static_cast<size_t>(NEUROCHEMICAL_COMPONENTS)  * sizeof(float);
    REQUIRE(node_payload_bytes() == reconstructed);
}

TEST_CASE("§15.5 CRC32C of arbitrary byte string is reproducible", "[invariants][GAP-019]") {
    std::vector<uint8_t> payload(NODE_PAYLOAD_BYTES, 0xABu);
    const uint32_t crc1 = compute_crc32c(payload);
    const uint32_t crc2 = compute_crc32c(payload);
    REQUIRE(crc1 == crc2);
}

TEST_CASE("§15.6 Epoch classification is exhaustive and non-overlapping", "[invariants][GAP-019]") {
    // For any pair exactly one of {STALE, CURRENT, FUTURE} is true
    auto exactly_one = [](uint64_t msg_e, uint64_t local_e) {
        auto rel = classify_epoch_message(msg_e, local_e);
        int count = 0;
        if (rel == EpochMessageRelation::STALE_VALID)  ++count;
        if (rel == EpochMessageRelation::CURRENT)       ++count;
        if (rel == EpochMessageRelation::FROM_FUTURE)   ++count;
        return count == 1;
    };
    REQUIRE(exactly_one(0, 0));
    REQUIRE(exactly_one(4, 5));
    REQUIRE(exactly_one(6, 5));
}

// ── §16 Integration Scenarios ────────────────────────────────────────────────

TEST_CASE("§16.1 Full 2P-EBP happy path state walk", "[integration][GAP-019]") {
    using S = MigrationState;

    S state = S::STABLE;

    // Phase 2: Orchestrator broadcasts PREPARE_MIGRATION
    REQUIRE(state_allows_physics(state));
    REQUIRE(is_valid_state_transition(state, S::PAUSED));
    state = S::PAUSED;
    REQUIRE(state_is_suspended(state));

    // Phase 3: Orchestrator broadcasts BEGIN_MIGRATION
    REQUIRE(is_valid_state_transition(state, S::MIGRATING));
    state = S::MIGRATING;

    // Phase 4: MIGRATION_ACK received → COMMIT_EPOCH
    REQUIRE(is_valid_state_transition(state, S::COMMITTING));
    state = S::COMMITTING;

    // Finalization complete → resume physics at T+1
    REQUIRE(is_valid_state_transition(state, S::STABLE));
    state = S::STABLE;
    REQUIRE(state_allows_physics(state));
}

TEST_CASE("§16.2 Rollback path: any state → STABLE", "[integration][GAP-019]") {
    using S = MigrationState;

    for (auto s : {S::STABLE, S::PAUSED, S::MIGRATING, S::COMMITTING}) {
        REQUIRE(is_valid_state_transition(s, S::STABLE));
    }
    // After rollback, physics resumes immediately
    REQUIRE(state_allows_physics(S::STABLE));
}

TEST_CASE("§16.3 LIF monitoring → trigger → VRAM safety check flow", "[integration][GAP-019]") {
    // Example: 4-rank cluster detects imbalance
    const std::array<uint64_t,4> counts{80, 80, 80, 160}; // last rank overloaded
    const float lif = load_imbalance_factor(counts);
    REQUIRE(lif > 0.0f);
    REQUIRE(is_rebalance_triggered(lif));

    // Overloaded rank estimates post-migration VRAM
    const uint64_t vram_cap     = 80ULL * 1024u * 1024u * 1024u; // 80 GB H100
    const uint64_t estimated    = 40ULL * 1024u * 1024u * 1024u; // moves 40 GB out → 40 GB remain
    REQUIRE(is_vram_safe(estimated, vram_cap));
    // → would send PREPARE_ACK
}

TEST_CASE("§16.4 Message causality: cross-epoch message routing", "[integration][GAP-019]") {
    const uint64_t local_epoch = 7u;

    // Stale message from rank that hasn't migrated yet
    auto stale = classify_epoch_message(6u, local_epoch);
    REQUIRE(stale == EpochMessageRelation::STALE_VALID);
    // If this rank still owns the target node, process it
    REQUIRE(stale_should_process(true));

    // Current epoch — normal processing
    auto current = classify_epoch_message(7u, local_epoch);
    REQUIRE(current == EpochMessageRelation::CURRENT);

    // Future epoch — sender has already committed epoch 8
    auto future = classify_epoch_message(8u, local_epoch);
    REQUIRE(future == EpochMessageRelation::FROM_FUTURE);
    // → buffer until local transitions to epoch 8
}

TEST_CASE("§16.5 CRC32C data integrity check on MigrationPayload", "[integration][GAP-019]") {
    // Simulate a small node payload
    const size_t payload_size = node_payload_bytes();
    std::vector<uint8_t> payload(payload_size);
    for (size_t i = 0u; i < payload_size; ++i)
        payload[i] = static_cast<uint8_t>(i & 0xFFu);

    const uint32_t checksum = compute_crc32c(payload);

    // Receiver validates
    REQUIRE(crc32c_matches(payload, checksum));

    // Simulate 1-bit corruption during transport
    payload[payload_size / 2u] ^= 0x01u;
    REQUIRE_FALSE(crc32c_matches(payload, checksum));
}

TEST_CASE("§16.6 SPD validation rejects torn metric tensor", "[integration][GAP-019]") {
    // Healthy metric tensor: all eigenvalues positive
    const std::array<float,9> healthy{1.0f,2.0f,0.5f,1.5f,3.0f,0.8f,2.2f,1.1f,0.3f};
    REQUIRE(is_positive_definite(healthy));

    // Torn metric (mid-migration memory corruption → one eigenvalue goes negative)
    std::array<float,9> torn = healthy;
    torn[4] = -0.001f;
    REQUIRE_FALSE(is_positive_definite(torn));
    // → worker would discard staging buffer and trigger ROLLBACK
}

TEST_CASE("§16.7 Epoch successor chain validates migration sequence", "[integration][GAP-019]") {
    // Simulate 5 sequential migrations
    uint64_t epoch = 0u;
    for (int i = 0; i < 5; ++i) {
        REQUIRE(epoch_is_valid_successor(epoch + 1u, epoch));
        ++epoch;
    }
    REQUIRE(epoch == 5u);
}
