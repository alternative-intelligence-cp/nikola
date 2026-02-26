// SPDX-License-Identifier: MIT
// GAP-042: System Error Taxonomy — INF / PHY / COG error codes
// Phase 88 — nikola::system
//
// Defines the canonical error catalogue for the Nikola runtime.
// Three subsystem prefixes: INF (infrastructure), PHY (physics engine),
// COG (cognitive layer).  Each entry carries category, severity, a
// short human-readable name, and the prescribed recovery strategy.
//
// Source: 02_orchestrator_router.md §"Error Taxonomy"

#pragma once

#include <cstdint>
#include <string_view>

namespace nikola::system {

// ─── Category ────────────────────────────────────────────────────────────────

enum class ErrorCategory : uint8_t {
    INF  = 0,   ///< Infrastructure errors (IPC, crypto, timing, SHM)
    PHY  = 1,   ///< Physics engine errors (resonance, energy, metric)
    COG  = 2,   ///< Cognitive layer errors (loop runaway, ATP, goals)
    AUTO = 3    ///< Auto-generated / synthetic (testing, simulation)
};

// ─── Severity ────────────────────────────────────────────────────────────────

enum class ErrorSeverity : uint8_t {
    CRITICAL = 0,   ///< Halt immediately — safety boundary breached
    HIGH     = 1,   ///< Degrade gracefully, alert operator
    MEDIUM   = 2,   ///< Warning — monitor and recover if persistent
    LOW      = 3    ///< Informational — note for telemetry
};

// ─── Recovery strategies ─────────────────────────────────────────────────────

enum class RecoveryStrategy : uint8_t {
    HARD_RESET         = 0,  ///< Full subsystem restart
    RE_PAIRING         = 1,  ///< Re-establish cryptographic session
    THROTTLING         = 2,  ///< Reduce message / computation rate
    SOFT_RESTART       = 3,  ///< Graceful restart (drain queue first)
    GARBAGE_COLLECT    = 4,  ///< Release leaked shared memory
    SOFT_SCRAM         = 5,  ///< Quantum Zeno freeze — damp resonance
    STEP_REDUCTION     = 6,  ///< Reduce integration step size
    REGULARIZATION     = 7,  ///< Add εI to metric to prevent singularity
    RE_IGNITION        = 8,  ///< Re-seed energy floor from vacuum baseline
    FORCED_NAP         = 9,  ///< Suspend cognitive loop, flush working memory
    GOAL_PURGE         = 10, ///< Clear teleological goal stack + dopamine spike
    MASKING            = 11, ///< Suppress hallucinogenic output channel
    STIMULUS_INJECTION = 12, ///< Inject external stimulus to break deadlock
    ADMIN_OVERRIDE     = 13  ///< Human-in-the-loop or supervisor agent takes control
};

// ─── Error descriptor ────────────────────────────────────────────────────────

struct ErrorDescriptor {
    ErrorCategory    category;
    uint8_t          code;         ///< 1-based within category (INF-001 → 1)
    ErrorSeverity    severity;
    RecoveryStrategy recovery;
    std::string_view name;         ///< Short identifier, e.g. "TemporalDecoherence"
    std::string_view description;  ///< One-sentence human description
};

// ─── INF errors ──────────────────────────────────────────────────────────────

inline constexpr ErrorDescriptor INF_001 {
    ErrorCategory::INF, 1, ErrorSeverity::CRITICAL, RecoveryStrategy::HARD_RESET,
    "TemporalDecoherence",
    "Message age exceeds decoherence limit for its class — causal ordering violated."
};
inline constexpr ErrorDescriptor INF_002 {
    ErrorCategory::INF, 2, ErrorSeverity::HIGH, RecoveryStrategy::RE_PAIRING,
    "CryptographicAmnesia",
    "Session keys are missing or corrupt — cryptographic identity lost."
};
inline constexpr ErrorDescriptor INF_003 {
    ErrorCategory::INF, 3, ErrorSeverity::HIGH, RecoveryStrategy::THROTTLING,
    "BandwidthSaturation",
    "ZeroMQ spine throughput exceeds headroom — back-pressure required."
};
inline constexpr ErrorDescriptor INF_004 {
    ErrorCategory::INF, 4, ErrorSeverity::MEDIUM, RecoveryStrategy::SOFT_RESTART,
    "HeartbeatFailure",
    "Heartbeat timeout elapsed without ACK — peer may have crashed."
};
inline constexpr ErrorDescriptor INF_005 {
    ErrorCategory::INF, 5, ErrorSeverity::LOW, RecoveryStrategy::GARBAGE_COLLECT,
    "SHMLeak",
    "Shared-memory region count exceeds expected maximum — possible leak."
};

// ─── PHY errors ──────────────────────────────────────────────────────────────

inline constexpr ErrorDescriptor PHY_001 {
    ErrorCategory::PHY, 1, ErrorSeverity::CRITICAL, RecoveryStrategy::SOFT_SCRAM,
    "EpilepticResonance",
    "Resonance amplitude entered runaway oscillation — Quantum Zeno freeze required."
};
inline constexpr ErrorDescriptor PHY_002 {
    ErrorCategory::PHY, 2, ErrorSeverity::CRITICAL, RecoveryStrategy::STEP_REDUCTION,
    "EnergyNonConservation",
    "Energy drift exceeds 0.01 % per 100 integration steps — numerical instability."
};
inline constexpr ErrorDescriptor PHY_003 {
    ErrorCategory::PHY, 3, ErrorSeverity::HIGH, RecoveryStrategy::REGULARIZATION,
    "MetricSingularity",
    "Metric tensor determinant approaching zero — add εI regularisation."
};
inline constexpr ErrorDescriptor PHY_004 {
    ErrorCategory::PHY, 4, ErrorSeverity::MEDIUM, RecoveryStrategy::RE_IGNITION,
    "VacuumCollapse",
    "Field energy dropped below minimum vacuum floor — re-ignition required."
};

// ─── COG errors ──────────────────────────────────────────────────────────────

inline constexpr ErrorDescriptor COG_001 {
    ErrorCategory::COG, 1, ErrorSeverity::CRITICAL, RecoveryStrategy::ADMIN_OVERRIDE,
    "RunawayCognitiveLoop",
    "Recursion depth exceeded hard limit — forced nap and admin override."
};
inline constexpr ErrorDescriptor COG_002 {
    ErrorCategory::COG, 2, ErrorSeverity::HIGH, RecoveryStrategy::STIMULUS_INJECTION,
    "BoredomSingularity",
    "Reward signal collapsed and no novel goal available — inject external stimulus."
};
inline constexpr ErrorDescriptor COG_003 {
    ErrorCategory::COG, 3, ErrorSeverity::MEDIUM, RecoveryStrategy::FORCED_NAP,
    "ATPExhaustion",
    "ATP reserve dropped below 5 % — suspend cognition and recover energy."
};
inline constexpr ErrorDescriptor COG_004 {
    ErrorCategory::COG, 4, ErrorSeverity::HIGH, RecoveryStrategy::GOAL_PURGE,
    "TeleologicalDeadlock",
    "Goal satisfaction loop detected with no progress — purge goals and spike dopamine."
};
inline constexpr ErrorDescriptor COG_005 {
    ErrorCategory::COG, 5, ErrorSeverity::LOW, RecoveryStrategy::MASKING,
    "Hallucination",
    "Output diverges significantly from grounded sensory input — mask channel."
};

// ─── Catalogue (ordered for linear search) ───────────────────────────────────

inline constexpr ErrorDescriptor ERROR_CATALOGUE[] = {
    INF_001, INF_002, INF_003, INF_004, INF_005,
    PHY_001, PHY_002, PHY_003, PHY_004,
    COG_001, COG_002, COG_003, COG_004, COG_005
};
inline constexpr std::size_t ERROR_CATALOGUE_SIZE =
    sizeof(ERROR_CATALOGUE) / sizeof(ERROR_CATALOGUE[0]);  // 14

// ─── Lookup helpers ──────────────────────────────────────────────────────────

/// Find the first descriptor matching category + 1-based code.
/// Returns nullptr if not found.
[[nodiscard]] constexpr const ErrorDescriptor*
lookup_error(ErrorCategory cat, uint8_t code) noexcept {
    for (const auto& e : ERROR_CATALOGUE) {
        if (e.category == cat && e.code == code) return &e;
    }
    return nullptr;
}

/// True when the descriptor mandates an immediate halt (CRITICAL severity).
[[nodiscard]] constexpr bool is_fatal(const ErrorDescriptor& e) noexcept {
    return e.severity == ErrorSeverity::CRITICAL;
}

// ─── Label helpers ───────────────────────────────────────────────────────────

[[nodiscard]] constexpr std::string_view category_label(ErrorCategory c) noexcept {
    switch (c) {
        case ErrorCategory::INF:  return "INF";
        case ErrorCategory::PHY:  return "PHY";
        case ErrorCategory::COG:  return "COG";
        case ErrorCategory::AUTO: return "AUTO";
        default:                  return "UNK";
    }
}

[[nodiscard]] constexpr std::string_view severity_label(ErrorSeverity s) noexcept {
    switch (s) {
        case ErrorSeverity::CRITICAL: return "CRITICAL";
        case ErrorSeverity::HIGH:     return "HIGH";
        case ErrorSeverity::MEDIUM:   return "MEDIUM";
        case ErrorSeverity::LOW:      return "LOW";
        default:                      return "UNKNOWN";
    }
}

[[nodiscard]] constexpr std::string_view recovery_label(RecoveryStrategy r) noexcept {
    switch (r) {
        case RecoveryStrategy::HARD_RESET:         return "HardReset";
        case RecoveryStrategy::RE_PAIRING:         return "RePairing";
        case RecoveryStrategy::THROTTLING:         return "Throttling";
        case RecoveryStrategy::SOFT_RESTART:       return "SoftRestart";
        case RecoveryStrategy::GARBAGE_COLLECT:    return "GarbageCollect";
        case RecoveryStrategy::SOFT_SCRAM:         return "SoftSCRAM";
        case RecoveryStrategy::STEP_REDUCTION:     return "StepReduction";
        case RecoveryStrategy::REGULARIZATION:     return "Regularization";
        case RecoveryStrategy::RE_IGNITION:        return "ReIgnition";
        case RecoveryStrategy::FORCED_NAP:         return "ForcedNap";
        case RecoveryStrategy::GOAL_PURGE:         return "GoalPurge";
        case RecoveryStrategy::MASKING:            return "Masking";
        case RecoveryStrategy::STIMULUS_INJECTION: return "StimulusInjection";
        case RecoveryStrategy::ADMIN_OVERRIDE:     return "AdminOverride";
        default:                                   return "Unknown";
    }
}

} // namespace nikola::system
