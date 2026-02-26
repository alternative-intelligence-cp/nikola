// SPDX-License-Identifier: MIT
// GAP-019: Distributed Partition Table Update Protocol (2P-EBP)
// Phase 90 — nikola::system
//
// Encodes the constants, state machine and policy functions for the
// Two-Phase Epoch-Based Partitioning (2P-EBP) protocol that rebalances
// the TorusGridSoA across distributed ranks without dropping the wavefunction.
//
// Source: 01_zeromq_spine.md §"Distributed Partition Table Update Protocol"

#pragma once

#include <cstdint>
#include <string_view>

namespace nikola::system {

// ─── Protocol state machine ───────────────────────────────────────────────────

/// The local state each worker rank transitions through during a 2P-EBP cycle.
enum class PartitionEpochState : uint8_t {
    STABLE             = 0,  ///< Normal operation; physics loop running
    PREPARE_MIGRATION  = 1,  ///< Physics loop paused; computing export/import sets
    MIGRATING          = 2,  ///< Data in flight between peer ranks
    VERIFYING          = 3,  ///< CRC32C + SPD metric check on received nodes
    COMMITTED          = 4,  ///< Pointer swap complete; physics resuming
    ROLLBACK           = 5   ///< Migration aborted; reverting to epoch ε
};

/// Control-plane message types carried inside PartitionControl protobuf.
enum class PartitionControlType : uint8_t {
    HEARTBEAT          = 0,
    PREPARE_MIGRATION  = 1,
    BEGIN_MIGRATION    = 2,
    COMMIT_EPOCH       = 3,
    ROLLBACK           = 4,
    ABORT              = 5
};

// ─── Rebalancing trigger ──────────────────────────────────────────────────────

/// Load Imbalance Factor threshold.
/// LIF = (max(N_i) - min(N_i)) / mean(N_i)
/// If LIF > LIF_TRIGGER_THRESHOLD the Orchestrator initiates rebalancing.
inline constexpr double LIF_TRIGGER_THRESHOLD      = 0.20;  ///< 20 % imbalance

// ─── Safety limits ────────────────────────────────────────────────────────────

/// Maximum VRAM utilisation (fraction) permitted after receiving migrated nodes.
/// If estimated post-migration memory exceeds this, send ABORT instead of ACK.
inline constexpr double MIGRATION_VRAM_SAFETY_LIMIT = 0.90;  ///< 90 %

// ─── Timing constants ─────────────────────────────────────────────────────────

/// Rollback timeout: Orchestrator waits this many milliseconds for MIGRATION_ACK
/// from all ranks before broadcasting ROLLBACK_MIGRATION.
inline constexpr uint32_t ROLLBACK_TIMEOUT_MS       = 5000;

/// Stability penalty cooldown after a rollback: Orchestrator suppresses the next
/// rebalancing attempt for this many milliseconds (1 hour).
inline constexpr uint32_t STABILITY_PENALTY_COOLDOWN_MS = 3'600'000;

// ─── Approximate pause duration (informational) ───────────────────────────────

/// Minimum observed pause duration during prepare phase (milliseconds).
inline constexpr uint32_t PAUSE_DURATION_MIN_MS     =  10;
/// Maximum observed pause duration during prepare phase (milliseconds).
inline constexpr uint32_t PAUSE_DURATION_MAX_MS     =  50;

// ─── Checksumming ─────────────────────────────────────────────────────────────

/// Algorithm used to validate migrated node batches: CRC32C.
/// (Castagnoli polynomial — hardware-accelerated on modern CPUs/GPUs.)
inline constexpr std::string_view MIGRATION_CHECKSUM_ALGORITHM = "CRC32C";

// ─── Node serialisation format ────────────────────────────────────────────────

/// Bytes in one 128-bit Morton key (Big Endian on the wire).
inline constexpr uint8_t MORTON_KEY_BYTES           = 16;

/// Number of metric tensor components packed per node in MigrationPayload.
inline constexpr uint8_t MIGRATION_METRIC_COMPONENTS = 45;

// ─── Policy functions ────────────────────────────────────────────────────────

/// True when the Load Imbalance Factor justifies triggering rebalancing.
[[nodiscard]] constexpr bool rebalancing_needed(double lif) noexcept {
    return lif > LIF_TRIGGER_THRESHOLD;
}

/// True when estimated post-migration VRAM utilisation is below the safety cap.
[[nodiscard]] constexpr bool migration_vram_safe(double vram_fraction) noexcept {
    return vram_fraction <= MIGRATION_VRAM_SAFETY_LIMIT;
}

/// True when rollback should be initiated because the ACK wait has expired.
[[nodiscard]] constexpr bool rollback_timeout_exceeded(uint32_t elapsed_ms) noexcept {
    return elapsed_ms >= ROLLBACK_TIMEOUT_MS;
}

/// Compute the Load Imbalance Factor from max, min and mean node counts.
[[nodiscard]] constexpr double load_imbalance_factor(
    double max_nodes, double min_nodes, double mean_nodes) noexcept
{
    if (mean_nodes <= 0.0) return 0.0;
    return (max_nodes - min_nodes) / mean_nodes;
}

// ─── State label helpers ─────────────────────────────────────────────────────

[[nodiscard]] constexpr std::string_view epoch_state_label(PartitionEpochState s) noexcept {
    switch (s) {
        case PartitionEpochState::STABLE:            return "STABLE";
        case PartitionEpochState::PREPARE_MIGRATION: return "PREPARE_MIGRATION";
        case PartitionEpochState::MIGRATING:         return "MIGRATING";
        case PartitionEpochState::VERIFYING:         return "VERIFYING";
        case PartitionEpochState::COMMITTED:         return "COMMITTED";
        case PartitionEpochState::ROLLBACK:          return "ROLLBACK";
        default:                                     return "UNKNOWN";
    }
}

[[nodiscard]] constexpr std::string_view control_type_label(PartitionControlType t) noexcept {
    switch (t) {
        case PartitionControlType::HEARTBEAT:         return "HEARTBEAT";
        case PartitionControlType::PREPARE_MIGRATION: return "PREPARE_MIGRATION";
        case PartitionControlType::BEGIN_MIGRATION:   return "BEGIN_MIGRATION";
        case PartitionControlType::COMMIT_EPOCH:      return "COMMIT_EPOCH";
        case PartitionControlType::ROLLBACK:          return "ROLLBACK";
        case PartitionControlType::ABORT:             return "ABORT";
        default:                                      return "UNKNOWN";
    }
}

} // namespace nikola::system
