#pragma once
// ============================================================
// nikola/infrastructure/partition_table.hpp
// GAP-019 — Distributed Partition Table Update Protocol
//
// Implements the Two-Phase Epoch Barrier Protocol (2P-EBP):
//   Phase 1: Monitoring & Trigger (Load Imbalance Factor)
//   Phase 2: PREPARE Barrier (physics pause, VRAM safety)
//   Phase 3: MIGRATION Transaction (SoA-to-AoS serialisation)
//   Phase 4: Verification & COMMIT (CRC32C + SPD metric check)
//
// Namespace: nikola::infrastructure
// ============================================================

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string_view>

namespace nikola::infrastructure {

// ── § Constants ─────────────────────────────────────────────────────────────

/// LIF trigger: if load imbalance factor > 0.20 → initiate rebalancing.
constexpr float    LIF_TRIGGER_THRESHOLD        = 0.20f;

/// VRAM safety ceiling: abort if post-migration footprint ≥ 90 % of capacity.
constexpr float    VRAM_SAFETY_FRACTION         = 0.90f;

/// Orchestrator rollback timeout: 5 000 ms waiting for MIGRATION_ACK.
constexpr uint32_t ROLLBACK_TIMEOUT_MS          = 5'000u;

/// Stability penalty duration: 1 hour cooldown after a rollback.
constexpr uint64_t STABILITY_PENALTY_MS         = 3'600'000ULL;

/// Components of the symmetric 9×9 Riemannian metric tensor: (9×10)/2 = 45.
constexpr int      METRIC_TENSOR_COMPONENTS     = 45;

/// Neurochemical state dimensionality: r, s, u, v, w.
constexpr int      NEUROCHEMICAL_COMPONENTS     = 5;

/// Morton key width in bytes: 128-bit Big Endian split point.
constexpr int      MORTON_KEY_BYTES             = 16;

/// Per-node serialised payload size in MigrationPayload (bytes):
///   Morton key (16) + Ψ_real/imag/dot_real/dot_imag (4 × 4 = 16)
///   + g_ij (45 × 4 = 180) + neurochemical (5 × 4 = 20) = 232.
constexpr size_t   NODE_PAYLOAD_BYTES           = 232u;

/// Estimated pause duration range (milliseconds).
constexpr uint32_t PAUSE_DURATION_MIN_MS        = 10u;
constexpr uint32_t PAUSE_DURATION_MAX_MS        = 50u;

// ── § Enumerations ───────────────────────────────────────────────────────────

/// PartitionControl message type — mirrors protobuf enum in spec.
enum class PartitionMessageType : uint8_t {
    HEARTBEAT         = 0,
    PREPARE_MIGRATION = 1,
    BEGIN_MIGRATION   = 2,
    COMMIT_EPOCH      = 3,
    ROLLBACK          = 4,
    ABORT             = 5,
};

/// Worker state in the 2P-EBP state machine.
enum class MigrationState : uint8_t {
    STABLE,        ///< Normal physics operation (physics loop running).
    PAUSED,        ///< Physics loop suspended; awaiting BEGIN_MIGRATION.
    MIGRATING,     ///< Data transfer in progress via PAIR sockets.
    COMMITTING,    ///< Atomic pointer swap; staging → main grid.
};

/// Relationship of a received ZeroMQ message's epoch to local epoch.
enum class EpochMessageRelation : uint8_t {
    STALE_VALID,   ///< msg.epoch < local.epoch → process if owner, else forward.
    CURRENT,       ///< msg.epoch == local.epoch  → normal processing.
    FROM_FUTURE,   ///< msg.epoch >  local.epoch → buffer until local transition.
};

// ── § Phase 1 — Load Imbalance Factor ───────────────────────────────────────

/// Compute Load Imbalance Factor (LIF):
///   \f$ LIF = \frac{\max(N_i) - \min(N_i)}{\bar{N}} \f$
///
/// @param node_counts  Node counts per rank (non-empty).
/// @returns  LIF ≥ 0; 0.0 for a single rank or a perfectly balanced cluster.
/// @throws   std::invalid_argument if span is empty.
[[nodiscard]] inline float load_imbalance_factor(
    std::span<const uint64_t> node_counts)
{
    if (node_counts.empty())
        throw std::invalid_argument("load_imbalance_factor: node_counts is empty");
    if (node_counts.size() == 1u)
        return 0.0f;

    const uint64_t max_n = *std::max_element(node_counts.begin(), node_counts.end());
    const uint64_t min_n = *std::min_element(node_counts.begin(), node_counts.end());
    const double   avg_n = static_cast<double>(
        std::accumulate(node_counts.begin(), node_counts.end(), uint64_t{0u}))
        / static_cast<double>(node_counts.size());

    if (avg_n <= 0.0)
        return 0.0f;

    return static_cast<float>((max_n - min_n) / avg_n);
}

/// True when LIF exceeds the 20 % trigger threshold.
[[nodiscard]] constexpr bool is_rebalance_triggered(float lif) noexcept
{
    return lif > LIF_TRIGGER_THRESHOLD;
}

// ── § Phase 2 — VRAM Safety Check ───────────────────────────────────────────

/// True when estimated post-migration footprint is within VRAM safety ceiling.
/// If estimated_bytes ≥ 90 % of vram_capacity_bytes the worker must ABORT.
[[nodiscard]] constexpr bool is_vram_safe(
    uint64_t estimated_bytes,
    uint64_t vram_capacity_bytes) noexcept
{
    // Avoid overflow: compare as doubles (node counts fit in double mantissa).
    return static_cast<double>(estimated_bytes)
           < static_cast<double>(vram_capacity_bytes) * VRAM_SAFETY_FRACTION;
}

// ── § Phase 3 — Payload Sizing ───────────────────────────────────────────────

/// Bytes in a single serialised node (spec § "Serialization"):
///   16 (Morton) + 16 (4 floats Ψ) + 180 (45 × float g_ij) + 20 (5 × float) = 232.
[[nodiscard]] constexpr size_t node_payload_bytes() noexcept
{
    return static_cast<size_t>(MORTON_KEY_BYTES)
         + 4u * sizeof(float)                              // Ψ_real, Ψ_imag, Ψ̇_real, Ψ̇_imag
         + static_cast<size_t>(METRIC_TENSOR_COMPONENTS) * sizeof(float)
         + static_cast<size_t>(NEUROCHEMICAL_COMPONENTS)  * sizeof(float);
}

/// Total MigrationPayload bytes for a batch of @p node_count nodes.
[[nodiscard]] constexpr size_t migration_payload_bytes(size_t node_count) noexcept
{
    return node_count * node_payload_bytes();
}

// ── § Epoch Classification ───────────────────────────────────────────────────

/// Classify incoming message w.r.t. local epoch (spec § "Message Causality").
[[nodiscard]] constexpr EpochMessageRelation classify_epoch_message(
    uint64_t msg_epoch,
    uint64_t local_epoch) noexcept
{
    if (msg_epoch < local_epoch) return EpochMessageRelation::STALE_VALID;
    if (msg_epoch > local_epoch) return EpochMessageRelation::FROM_FUTURE;
    return EpochMessageRelation::CURRENT;
}

/// Stale-valid processing rule: process in-place when this rank still owns target.
[[nodiscard]] constexpr bool stale_should_process(bool still_owner) noexcept
{
    return still_owner;
}

/// Stale-valid forwarding rule: forward via PT_ε when ownership has moved.
[[nodiscard]] constexpr bool stale_should_forward(bool still_owner) noexcept
{
    return !still_owner;
}

// ── § Epoch Sequencing ───────────────────────────────────────────────────────

/// Epochs are monotonically incremented by 1 per migration cycle.
[[nodiscard]] constexpr bool epoch_is_valid_successor(
    uint64_t next_epoch,
    uint64_t current_epoch) noexcept
{
    return next_epoch == current_epoch + 1u;
}

// ── § Timeout and Penalty Constants ─────────────────────────────────────────

[[nodiscard]] constexpr uint32_t rollback_timeout_ms() noexcept
{
    return ROLLBACK_TIMEOUT_MS;
}

[[nodiscard]] constexpr uint64_t stability_penalty_ms() noexcept
{
    return STABILITY_PENALTY_MS;
}

// ── § Phase 4 — CRC32C Data Integrity ───────────────────────────────────────

/// Compute CRC32C (Castagnoli, poly 0x1EDC6F41, reflected 0x82F63B78) over @p data.
///
/// Table is generated at compile time via consteval — no runtime initialisation.
[[nodiscard]] inline uint32_t compute_crc32c(
    std::span<const uint8_t> data) noexcept
{
    // Reflected CRC32C polynomial: 0x82F63B78
    static constexpr auto TABLE = []() consteval {
        std::array<uint32_t, 256> t{};
        for (uint32_t i = 0u; i < 256u; ++i) {
            uint32_t crc = i;
            for (int j = 0; j < 8; ++j)
                crc = (crc >> 1) ^ (0x82F63B78u * (crc & 1u));
            t[i] = crc;
        }
        return t;
    }();

    uint32_t crc = 0xFFFF'FFFFu;
    for (uint8_t byte : data)
        crc = TABLE[(crc ^ byte) & 0xFFu] ^ (crc >> 8);
    return crc ^ 0xFFFF'FFFFu;
}

/// True when CRC32C over @p data matches @p expected_checksum.
[[nodiscard]] inline bool crc32c_matches(
    std::span<const uint8_t> data,
    uint32_t expected_checksum) noexcept
{
    return compute_crc32c(data) == expected_checksum;
}

// ── § Phase 4 — SPD Metric Tensor Validation ────────────────────────────────

/// True when every eigenvalue in @p eigenvalues is strictly positive (SPD).
/// Used to verify the imported metric tensor preserves Riemannian structure.
///
/// @throws std::invalid_argument if span is empty.
[[nodiscard]] inline bool is_positive_definite(
    std::span<const float> eigenvalues)
{
    if (eigenvalues.empty())
        throw std::invalid_argument("is_positive_definite: eigenvalues span is empty");

    for (float ev : eigenvalues) {
        if (ev <= 0.0f)
            return false;
    }
    return true;
}

// ── § 2P-EBP State Machine ───────────────────────────────────────────────────

/// True only in STABLE state — the sole state where physics is permitted to run.
[[nodiscard]] constexpr bool state_allows_physics(MigrationState s) noexcept
{
    return s == MigrationState::STABLE;
}

/// True in any non-STABLE state (physics loop must be paused).
[[nodiscard]] constexpr bool state_is_suspended(MigrationState s) noexcept
{
    return s != MigrationState::STABLE;
}

/// Validate legal 2P-EBP transitions:
///   STABLE → PAUSED       (PREPARE_MIGRATION received)
///   PAUSED → MIGRATING    (BEGIN_MIGRATION received)
///   MIGRATING → COMMITTING (COMMIT_EPOCH received + validation passed)
///   COMMITTING → STABLE   (finalization complete)
///   Any → STABLE          (ROLLBACK — always valid)
[[nodiscard]] constexpr bool is_valid_state_transition(
    MigrationState from,
    MigrationState to) noexcept
{
    using S = MigrationState;
    if (to == S::STABLE)                                 return true;  // ROLLBACK
    if (from == S::STABLE     && to == S::PAUSED)        return true;
    if (from == S::PAUSED     && to == S::MIGRATING)     return true;
    if (from == S::MIGRATING  && to == S::COMMITTING)    return true;
    if (from == S::COMMITTING && to == S::STABLE)        return true;
    return false;
}

// ── § Diagnostic Names ────────────────────────────────────────────────────────

[[nodiscard]] constexpr std::string_view message_type_name(
    PartitionMessageType t) noexcept
{
    switch (t) {
        case PartitionMessageType::HEARTBEAT:         return "HEARTBEAT";
        case PartitionMessageType::PREPARE_MIGRATION: return "PREPARE_MIGRATION";
        case PartitionMessageType::BEGIN_MIGRATION:   return "BEGIN_MIGRATION";
        case PartitionMessageType::COMMIT_EPOCH:      return "COMMIT_EPOCH";
        case PartitionMessageType::ROLLBACK:          return "ROLLBACK";
        case PartitionMessageType::ABORT:             return "ABORT";
    }
    return "UNKNOWN";
}

[[nodiscard]] constexpr std::string_view migration_state_name(
    MigrationState s) noexcept
{
    switch (s) {
        case MigrationState::STABLE:     return "STABLE";
        case MigrationState::PAUSED:     return "PAUSED";
        case MigrationState::MIGRATING:  return "MIGRATING";
        case MigrationState::COMMITTING: return "COMMITTING";
    }
    return "UNKNOWN";
}

[[nodiscard]] constexpr std::string_view epoch_relation_name(
    EpochMessageRelation r) noexcept
{
    switch (r) {
        case EpochMessageRelation::STALE_VALID:  return "STALE_VALID";
        case EpochMessageRelation::CURRENT:      return "CURRENT";
        case EpochMessageRelation::FROM_FUTURE:  return "FROM_FUTURE";
    }
    return "UNKNOWN";
}

} // namespace nikola::infrastructure
