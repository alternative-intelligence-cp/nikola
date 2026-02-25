#pragma once

// ============================================================
// error_taxonomy.hpp — GAP-042: Error Code Taxonomy and Handling Guide
//
// Documents Nikola's "Homeostatic Regulation" error philosophy:
//   • Structured error taxonomy across 4 architectural layers
//   • Severity classification (CRITICAL / HIGH / MEDIUM / LOW)
//   • Recovery strategy catalogue (13 strategies)
//   • Per-error-code metadata: category, severity, recovery
//   • Numerical thresholds that define each fault condition
//   • JSON structured-logging field name constants
//
// Philosophy: errors are thermodynamic violations, not logic faults.
// The "Soft SCRAM" pattern dissipates excess energy without killing
// cognitive state — analogous to a biological autonomic reflex.
//
// Namespace:   nikola::infrastructure
// Standard:    C++23, header-only, no external dependencies
// Source spec: GAP-042 — Error Code Taxonomy and Handling Guide
//              (Gemini Deep Research Round 2, Batch 41-44)
// ============================================================

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace nikola::infrastructure {

// ============================================================
// §1  Severity Levels
// ============================================================

/// Operational severity of an error condition.
enum class Severity : uint8_t {
    CRITICAL = 0u,  ///< System instability; immediate intervention required.
    HIGH     = 1u,  ///< Degraded performance; prompt recovery necessary.
    MEDIUM   = 2u,  ///< Recoverable degradation; automated fix available.
    LOW      = 3u,  ///< Informational; tolerable without immediate action.
};

/// Number of distinct severity levels.
inline constexpr std::size_t SEVERITY_COUNT = 4u;

// ============================================================
// §2  Error Category (Architectural Layer)
// ============================================================

/// Top-level architectural category encoded in every error code prefix.
enum class ErrorCategory : uint8_t {
    INF  = 0u,  ///< Infrastructure & Communications (digital substrate, ZMQ, SHM).
    PHY  = 1u,  ///< Physics Engine (wave simulation, energy conservation, metric).
    COG  = 2u,  ///< Cognitive & Autonomous (reasoning, neurochemistry, goals).
    AUTO = 3u,  ///< Autonomous sub-system (future extension; reserved).
};

/// Number of defined error categories.
inline constexpr std::size_t ERROR_CATEGORY_COUNT = 4u;

// ============================================================
// §3  Recovery Strategies
// ============================================================

/// Automated recovery action triggered when an error code is detected.
enum class RecoveryStrategy : uint8_t {
    HARD_RESET         = 0u,   ///< SIGKILL + /dev/shm cleanup + process restart (INF-001, INF-004).
    RE_PAIRING         = 1u,   ///< Re-load identity keys; enter Safe Mode if failure (INF-002).
    THROTTLING         = 2u,   ///< Raise significance threshold θ to shrink packet size (INF-003).
    GARBAGE_COLLECTION = 3u,   ///< Unlink stale SHM segments based on PID liveness (INF-005).
    SOFT_SCRAM         = 4u,   ///< Apply global damping γ=0.5, clamp + renormalize (PHY-001).
    STEP_REDUCTION     = 5u,   ///< Halve integration timestep Δt; retry; then Soft SCRAM (PHY-002).
    REGULARIZATION     = 6u,   ///< Tikhonov regularization: add εI to metric diagonal (PHY-003).
    RE_IGNITION        = 7u,   ///< Inject thermal noise + pilot wave to restore energy floor (PHY-004).
    ADMIN_OVERRIDE     = 8u,   ///< ZMQ Spine prioritises STOP; force Nap to reset working mem (COG-001).
    STIMULUS_INJECTION = 9u,   ///< Inject Curiosity goal; boost Norepinephrine (COG-002).
    FORCED_NAP         = 10u,  ///< Suspend high-level tasks; enter Dream-Weave to recharge ATP (COG-003).
    GOAL_PURGE         = 11u,  ///< Prune least-prioritised goal in cycle; spike Dopamine (COG-004).
    MASKING            = 12u,  ///< Re-generate attention mask; zero vacuum nodes (COG-005).
};

/// Number of defined recovery strategies.
inline constexpr std::size_t RECOVERY_STRATEGY_COUNT = 13u;

// ============================================================
// §4  Error Codes
// ============================================================

/// Structured error code identifiers across all architectural layers.
enum class ErrorCode : uint8_t {
    // ---- Infrastructure & Communications (INF) --------------------
    INF_001 = 0u,   ///< CRITICAL — Temporal Decoherence (control/data plane desync > 50 ms).
    INF_002 = 1u,   ///< HIGH     — Cryptographic Amnesia (lost identity keys / handshake fail).
    INF_003 = 2u,   ///< HIGH     — Bandwidth Saturation (PCIe/network throughput exceeded).
    INF_004 = 3u,   ///< MEDIUM   — Heartbeat Failure (no heartbeat for > 500 ms).
    INF_005 = 4u,   ///< LOW      — Shared Memory Leak (stale segments in /dev/shm).

    // ---- Physics Engine (PHY) -------------------------------------
    PHY_001 = 5u,   ///< CRITICAL — Epileptic Resonance (wavefunction amplitude diverges).
    PHY_002 = 6u,   ///< CRITICAL — Energy Non-Conservation (Hamiltonian drift > 0.01% / 100 steps).
    PHY_003 = 7u,   ///< HIGH     — Metric Singularity (metric tensor det → 0 or negative eigenvalues).
    PHY_004 = 8u,   ///< MEDIUM   — Vacuum Collapse (total energy below thermal floor).

    // ---- Cognitive & Autonomous (COG/AUTO) ------------------------
    COG_001 = 9u,   ///< CRITICAL — Runaway Cognitive Loop (100% CPU, goal completion → 0).
    COG_002 = 10u,  ///< HIGH     — Boredom Singularity (entropy gradient ≈ 0; local minima).
    COG_003 = 11u,  ///< MEDIUM   — ATP Exhaustion (metabolic budget < 5%).
    COG_004 = 12u,  ///< HIGH     — Teleological Deadlock (circular dependency in goal DAG).
    COG_005 = 13u,  ///< LOW      — Hallucination (GGUF attention mask failure; perplexity spike).
};

/// Total number of defined error codes.
inline constexpr std::size_t ERROR_CODE_COUNT = 14u;

/// Number of Infrastructure error codes.
inline constexpr std::size_t INF_ERROR_COUNT = 5u;

/// Number of Physics Engine error codes.
inline constexpr std::size_t PHY_ERROR_COUNT = 4u;

/// Number of Cognitive/Autonomous error codes.
inline constexpr std::size_t COG_ERROR_COUNT = 5u;

// ============================================================
// §5  Fault-Condition Thresholds
// ============================================================

/// INF-001: Control/data plane timestamp desynchronisation threshold.
/// Unit: milliseconds. Desync exceeding this value triggers Hard Reset.
inline constexpr uint32_t TEMPORAL_DECOHERENCE_THRESHOLD_MS = 50u;

/// INF-004: Maximum silence from a component before watchdog fires.
/// Unit: milliseconds.
inline constexpr uint32_t HEARTBEAT_FAILURE_THRESHOLD_MS = 500u;

/// PHY-002: Maximum fractional Hamiltonian energy drift over the
/// evaluation window before step reduction is triggered.
/// Ratio: 0.01 % → 1e-4.
inline constexpr double ENERGY_DRIFT_MAX_RATIO = 1.0e-4;

/// PHY-002: Number of integration steps over which the energy drift
/// integral is evaluated.
inline constexpr uint32_t ENERGY_DRIFT_EVAL_STEPS = 100u;

/// PHY-001: Normalised damping coefficient γ applied during a Soft SCRAM.
/// Wavefunction amplitudes are multiplied by (1 − SOFT_SCRAM_DAMPING).
inline constexpr double SOFT_SCRAM_DAMPING = 0.5;

/// PHY-002: Fractional factor by which the integration timestep Δt is
/// reduced when energy drift is detected (halve).
inline constexpr double STEP_REDUCTION_FACTOR = 0.5;

/// COG-003: Metabolic budget fraction below which ATP Exhaustion fires.
/// 5 % → 0.05.
inline constexpr double ATP_EXHAUSTION_THRESHOLD = 0.05;

// ============================================================
// §6  Structured Log Field Names (JSON Schema)
// ============================================================

/// URI of the Nikola structured-log JSON Schema (v0.0.4).
inline constexpr std::string_view LOG_SCHEMA_URI =
    "http://nikola-agi.com/schemas/v0.0.4/log-entry.json";

/// ISO-8601 event timestamp field.
inline constexpr std::string_view LOG_FIELD_TIMESTAMP       = "timestamp";
/// Log level (DEBUG / INFO / WARNING / ERROR / CRITICAL).
inline constexpr std::string_view LOG_FIELD_LEVEL           = "level";
/// Emitting component identifier (e.g., "PHYSICS_ENGINE").
inline constexpr std::string_view LOG_FIELD_COMPONENT_ID    = "component_id";
/// Structured error code string (e.g., "PHY-002").
inline constexpr std::string_view LOG_FIELD_ERROR_CODE      = "error_code";
/// Human-readable description of the event.
inline constexpr std::string_view LOG_FIELD_MESSAGE         = "message";
/// Domain-specific measurements captured at the time of the event.
inline constexpr std::string_view LOG_FIELD_CONTEXT         = "context";
/// Automated recovery action taken in response to the error.
inline constexpr std::string_view LOG_FIELD_RECOVERY_ACTION = "recovery_action";
/// W3C / OTel trace ID linking to the Neural Trace flush (GAP-027b).
inline constexpr std::string_view LOG_FIELD_TRACE_ID        = "trace_id";

/// Total number of top-level JSON log field names.
inline constexpr std::size_t LOG_FIELD_COUNT = 8u;

// ============================================================
// §7  Metadata Query Functions
// ============================================================

/// Returns the architectural category for a given error code.
[[nodiscard]] constexpr ErrorCategory category_of(ErrorCode code) noexcept {
    switch (code) {
        case ErrorCode::INF_001:
        case ErrorCode::INF_002:
        case ErrorCode::INF_003:
        case ErrorCode::INF_004:
        case ErrorCode::INF_005:
            return ErrorCategory::INF;

        case ErrorCode::PHY_001:
        case ErrorCode::PHY_002:
        case ErrorCode::PHY_003:
        case ErrorCode::PHY_004:
            return ErrorCategory::PHY;

        case ErrorCode::COG_001:
        case ErrorCode::COG_002:
        case ErrorCode::COG_003:
        case ErrorCode::COG_004:
        case ErrorCode::COG_005:
            return ErrorCategory::COG;
    }
    return ErrorCategory::INF;  // unreachable
}

/// Returns the severity level for a given error code.
[[nodiscard]] constexpr Severity severity_of(ErrorCode code) noexcept {
    switch (code) {
        case ErrorCode::INF_001: return Severity::CRITICAL;
        case ErrorCode::INF_002: return Severity::HIGH;
        case ErrorCode::INF_003: return Severity::HIGH;
        case ErrorCode::INF_004: return Severity::MEDIUM;
        case ErrorCode::INF_005: return Severity::LOW;

        case ErrorCode::PHY_001: return Severity::CRITICAL;
        case ErrorCode::PHY_002: return Severity::CRITICAL;
        case ErrorCode::PHY_003: return Severity::HIGH;
        case ErrorCode::PHY_004: return Severity::MEDIUM;

        case ErrorCode::COG_001: return Severity::CRITICAL;
        case ErrorCode::COG_002: return Severity::HIGH;
        case ErrorCode::COG_003: return Severity::MEDIUM;
        case ErrorCode::COG_004: return Severity::HIGH;
        case ErrorCode::COG_005: return Severity::LOW;
    }
    return Severity::LOW;  // unreachable
}

/// Returns the primary automated recovery strategy for a given error code.
[[nodiscard]] constexpr RecoveryStrategy recovery_of(ErrorCode code) noexcept {
    switch (code) {
        case ErrorCode::INF_001: return RecoveryStrategy::HARD_RESET;
        case ErrorCode::INF_002: return RecoveryStrategy::RE_PAIRING;
        case ErrorCode::INF_003: return RecoveryStrategy::THROTTLING;
        case ErrorCode::INF_004: return RecoveryStrategy::HARD_RESET;
        case ErrorCode::INF_005: return RecoveryStrategy::GARBAGE_COLLECTION;

        case ErrorCode::PHY_001: return RecoveryStrategy::SOFT_SCRAM;
        case ErrorCode::PHY_002: return RecoveryStrategy::STEP_REDUCTION;
        case ErrorCode::PHY_003: return RecoveryStrategy::REGULARIZATION;
        case ErrorCode::PHY_004: return RecoveryStrategy::RE_IGNITION;

        case ErrorCode::COG_001: return RecoveryStrategy::ADMIN_OVERRIDE;
        case ErrorCode::COG_002: return RecoveryStrategy::STIMULUS_INJECTION;
        case ErrorCode::COG_003: return RecoveryStrategy::FORCED_NAP;
        case ErrorCode::COG_004: return RecoveryStrategy::GOAL_PURGE;
        case ErrorCode::COG_005: return RecoveryStrategy::MASKING;
    }
    return RecoveryStrategy::HARD_RESET;  // unreachable
}

/// Returns the canonical string representation of an error code (e.g., "PHY-002").
[[nodiscard]] constexpr std::string_view error_code_name(ErrorCode code) noexcept {
    switch (code) {
        case ErrorCode::INF_001: return "INF-001";
        case ErrorCode::INF_002: return "INF-002";
        case ErrorCode::INF_003: return "INF-003";
        case ErrorCode::INF_004: return "INF-004";
        case ErrorCode::INF_005: return "INF-005";
        case ErrorCode::PHY_001: return "PHY-001";
        case ErrorCode::PHY_002: return "PHY-002";
        case ErrorCode::PHY_003: return "PHY-003";
        case ErrorCode::PHY_004: return "PHY-004";
        case ErrorCode::COG_001: return "COG-001";
        case ErrorCode::COG_002: return "COG-002";
        case ErrorCode::COG_003: return "COG-003";
        case ErrorCode::COG_004: return "COG-004";
        case ErrorCode::COG_005: return "COG-005";
    }
    return "";  // unreachable
}

/// Returns the human-readable name for a Severity level.
[[nodiscard]] constexpr std::string_view severity_name(Severity sev) noexcept {
    switch (sev) {
        case Severity::CRITICAL: return "critical";
        case Severity::HIGH:     return "high";
        case Severity::MEDIUM:   return "medium";
        case Severity::LOW:      return "low";
    }
    return "";
}

/// Returns the human-readable prefix for an ErrorCategory.
[[nodiscard]] constexpr std::string_view category_name(ErrorCategory cat) noexcept {
    switch (cat) {
        case ErrorCategory::INF:  return "INF";
        case ErrorCategory::PHY:  return "PHY";
        case ErrorCategory::COG:  return "COG";
        case ErrorCategory::AUTO: return "AUTO";
    }
    return "";
}

/// Returns the snake_case identifier for a RecoveryStrategy.
[[nodiscard]] constexpr std::string_view recovery_name(RecoveryStrategy rs) noexcept {
    switch (rs) {
        case RecoveryStrategy::HARD_RESET:         return "hard_reset";
        case RecoveryStrategy::RE_PAIRING:         return "re_pairing";
        case RecoveryStrategy::THROTTLING:         return "throttling";
        case RecoveryStrategy::GARBAGE_COLLECTION: return "garbage_collection";
        case RecoveryStrategy::SOFT_SCRAM:         return "soft_scram";
        case RecoveryStrategy::STEP_REDUCTION:     return "step_reduction";
        case RecoveryStrategy::REGULARIZATION:     return "regularization";
        case RecoveryStrategy::RE_IGNITION:        return "re_ignition";
        case RecoveryStrategy::ADMIN_OVERRIDE:     return "admin_override";
        case RecoveryStrategy::STIMULUS_INJECTION: return "stimulus_injection";
        case RecoveryStrategy::FORCED_NAP:         return "forced_nap";
        case RecoveryStrategy::GOAL_PURGE:         return "goal_purge";
        case RecoveryStrategy::MASKING:            return "masking";
    }
    return "";
}

// ============================================================
// §8  Predicate Helpers
// ============================================================

/// Returns true when the error code represents a CRITICAL severity condition.
[[nodiscard]] constexpr bool is_critical(ErrorCode code) noexcept {
    return severity_of(code) == Severity::CRITICAL;
}

/// Returns true when the error code belongs to the Infrastructure layer.
[[nodiscard]] constexpr bool is_infrastructure_error(ErrorCode code) noexcept {
    return category_of(code) == ErrorCategory::INF;
}

/// Returns true when the error code belongs to the Physics Engine layer.
[[nodiscard]] constexpr bool is_physics_error(ErrorCode code) noexcept {
    return category_of(code) == ErrorCategory::PHY;
}

/// Returns true when the error code belongs to the Cognitive/Autonomous layer.
[[nodiscard]] constexpr bool is_cognitive_error(ErrorCode code) noexcept {
    return category_of(code) == ErrorCategory::COG;
}

/// Returns true when the primary recovery for this code is a Soft SCRAM.
[[nodiscard]] constexpr bool requires_soft_scram(ErrorCode code) noexcept {
    return recovery_of(code) == RecoveryStrategy::SOFT_SCRAM;
}

/// Returns true when a forced Nap cycle is the primary recovery action.
[[nodiscard]] constexpr bool requires_forced_nap(ErrorCode code) noexcept {
    return recovery_of(code) == RecoveryStrategy::FORCED_NAP;
}

/// Returns true when the control/data plane timestamp delta exceeds the
/// Temporal Decoherence threshold (INF-001 condition).
[[nodiscard]] constexpr bool is_temporal_decoherence(uint32_t delta_ms) noexcept {
    return delta_ms > TEMPORAL_DECOHERENCE_THRESHOLD_MS;
}

/// Returns true when a component's silence exceeds the heartbeat threshold
/// (INF-004 condition).
[[nodiscard]] constexpr bool is_heartbeat_failure(uint32_t elapsed_ms) noexcept {
    return elapsed_ms > HEARTBEAT_FAILURE_THRESHOLD_MS;
}

/// Returns true when the fractional Hamiltonian energy drift exceeds the
/// conservation threshold (PHY-002 condition).
[[nodiscard]] constexpr bool is_energy_drift_violation(double drift_ratio) noexcept {
    return drift_ratio > ENERGY_DRIFT_MAX_RATIO;
}

/// Returns true when the metabolic ATP budget fraction is below the
/// exhaustion threshold (COG-003 condition).
[[nodiscard]] constexpr bool is_atp_exhausted(double budget_fraction) noexcept {
    return budget_fraction < ATP_EXHAUSTION_THRESHOLD;
}

}  // namespace nikola::infrastructure
