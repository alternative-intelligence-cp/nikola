#pragma once
// =============================================================================
// nikola/system/temporal_coherence.hpp
// Phase 84 — GAP-020: Temporal Decoherence Detection Thresholds
//
// SOURCE: Gemini Deep Research Round 2, Batch 19-21 (December 15, 2025)
// SPEC:   docs/info/integration/sections/04_infrastructure/01_zeromq_spine.md
//         §GAP-020 (lines ~1007–1120)
//
// Adaptive thresholding strategy: high-freq physics signals require 113μs
// latency tolerance (derived from f_max=441Hz Emitter Array, ε_φ=π/10).
// PTP clock synchronisation states and Seqlock IPC policy.
// =============================================================================

#include <cstdint>
#include <cstddef>
#include <string_view>

namespace nikola::system {

// ---------------------------------------------------------------------------
// § Enumerations
// ---------------------------------------------------------------------------

/// Message class determining which adaptive latency tier is applied.
enum class MessageClass : uint8_t {
    HIGH_FREQ_PHYSICS = 0,  ///< Wave propagation updates at f_max=441 Hz
    VISUAL_INPUT      = 1,  ///< Camera/display frames at 60 Hz
    COGNITIVE_STATE   = 2,  ///< Mamba-9D state at Theta/Alpha ~13.3 Hz
    CONTROL_ADMIN     = 3,  ///< DC-level commands / config changes
    SENSORY_AUDIO     = 4,  ///< PCM audio at 44.1 kHz (buffered)
};

/// Physics Oracle synchronisation lock states.
/// §"Physics Oracle Timekeeper State Machine"
enum class SyncState : uint8_t {
    SYNC_LOCKED   = 0,  ///< |θ| < 50 μs — physics simulation permitted
    SYNC_WARNING  = 1,  ///< 50 μs ≤ |θ| < 100 μs — Oracle applies Virtual Time Dilation
    SYNC_SCRAM    = 2,  ///< |θ| > 150 μs — Soft SCRAM; detach from cluster
};

/// Action applied when a message exceeds its decoherence threshold.
enum class DecoherenceAction : uint8_t {
    ACCEPT          = 0,  ///< Within threshold — process normally
    HARD_DROP       = 1,  ///< High-freq: phase-corrupt signal → discard immediately
    INTERPOLATE     = 2,  ///< Visual: sample-and-hold or optical flow
    PREDICTIVE_CODE = 3,  ///< Cognitive: Kalman filter project to T_now
    PROCESS         = 4,  ///< Admin: atemporal state change — process anyway
    JITTER_BUFFER   = 5,  ///< Audio: re-clock to physics time
};

// ---------------------------------------------------------------------------
// § Emitter Array Frequency Constants (Threshold Derivation Basis)
// ---------------------------------------------------------------------------

/// Base emitter frequency: π × φ¹ ≈ 5.083 Hz (Golden Ratio fundamental).
inline constexpr double EMITTER_F1_HZ          = 5.083;

/// Maximum driven frequency: π × φ⁸ ≈ 146.6 Hz (8th harmonic).
inline constexpr double EMITTER_F8_HZ          = 146.6;

/// Internal harmonic limit (Nyquist compliance, nonlinear N̂ operator): 441 Hz.
/// Spec: "Internal Harmonic Limit ≈ 441 Hz"
inline constexpr double EMITTER_F_MAX_HZ       = 441.0;

/// Phase integrity constraint ε_φ = π/10 (18°).
/// Guarantees >95% theoretical amplitude (cos(18°) ≈ 0.951).
inline constexpr double PHASE_EPSILON_RAD      = 0.31415926535897932; // π/10

/// cos(ε_φ) — interference amplitude retention at the phase limit.
inline constexpr double PHASE_AMPLITUDE_RETENTION = 0.951;  // cos(π/10)

// ---------------------------------------------------------------------------
// § Primary Temporal Decoherence Threshold (Derived)
// ---------------------------------------------------------------------------

/// Maximum allowable message age for high-frequency physics signals (μs).
/// Derivation: τ_max = ε_φ / (2π × f_max) = (π/10) / (2π × 441) ≈ 113 μs
/// Spec: "Standard TCP/IP loopback (500-1500 μs) is order of magnitude too slow"
inline constexpr double DECOHERENCE_LIMIT_PHYSICS_US  = 113.0;

/// Maximum allowable age for visual input (μs): 1 / (2 × 60 Hz).
inline constexpr double DECOHERENCE_LIMIT_VISUAL_US   = 8300.0;   // 8.3 ms

/// Maximum allowable age for cognitive state messages (μs): 10 ms.
inline constexpr double DECOHERENCE_LIMIT_COGNITIVE_US = 10000.0; // 10 ms

/// Maximum allowable age for control/admin messages (μs): 100 ms.
inline constexpr double DECOHERENCE_LIMIT_CONTROL_US  = 100000.0; // 100 ms

/// Presentation delay for sensory (audio) jitter buffer (μs): 50 ms.
inline constexpr double DECOHERENCE_LIMIT_AUDIO_US    = 50000.0;  // 50 ms

// ---------------------------------------------------------------------------
// § PTP Clock Synchronisation Thresholds
// ---------------------------------------------------------------------------

/// Clock offset below which physics simulation is permitted (μs).
/// Spec: "|θ| < 50 μs → SYNC_LOCKED"
inline constexpr double PTP_LOCK_THRESHOLD_US    = 50.0;

/// Clock offset entering warning/dilation zone (μs).
/// Spec: "50 μs < |θ| < 100 μs → SYNC_WARNING (Virtual Time Dilation)"
inline constexpr double PTP_WARNING_THRESHOLD_US = 100.0;

/// Clock offset triggering Soft SCRAM (μs).
/// Spec: "|θ| > 150 μs → Decoherence SCRAM (113 μs limit + margin)"
inline constexpr double PTP_SCRAM_THRESHOLD_US   = 150.0;

/// NTP accuracy (μs) — insufficient for Nikola; PTP mandatory.
/// Spec: "Standard NTP (1-10ms accuracy) is insufficient"
inline constexpr double NTP_ACCURACY_US          = 1000.0; // 1 ms typical

/// PTP hardware-timestamp accuracy target (μs).
inline constexpr double PTP_ACCURACY_TARGET_US   = 1.0;  // sub-microsecond

// ---------------------------------------------------------------------------
// § Future-message jitter tolerance
// ---------------------------------------------------------------------------

/// Negative age below which a message is flagged as "from the future" (μs).
/// Spec: "Allow small skew for clock jitter: -50 μs tolerance"
inline constexpr double FUTURE_SKEW_TOLERANCE_US = -50.0;

// ---------------------------------------------------------------------------
// § Watchdog constants
// ---------------------------------------------------------------------------

/// Hardware watchdog timer deadline — physics thread must "pet" within this.
/// Spec: "If watchdog not reset within 2000 μs (2 ticks) → assumes deadlock"
inline constexpr double WATCHDOG_DEADLINE_US     = 2000.0;

/// Number of missed ticks before watchdog fires.
inline constexpr int    WATCHDOG_MISSED_TICKS    = 2;

/// Heartbeat timeout: time before component declared dead (ms).
/// Spec: "Component failed to emit heartbeat for 500ms → INF-004"
inline constexpr double HEARTBEAT_TIMEOUT_MS     = 500.0;

// ---------------------------------------------------------------------------
// § Per–message-class threshold accessors
// ---------------------------------------------------------------------------

/// Return the latency limit (μs) for a given message class.
[[nodiscard]] constexpr double decoherence_limit_us(MessageClass cls) noexcept {
    switch (cls) {
        case MessageClass::HIGH_FREQ_PHYSICS: return DECOHERENCE_LIMIT_PHYSICS_US;
        case MessageClass::VISUAL_INPUT:      return DECOHERENCE_LIMIT_VISUAL_US;
        case MessageClass::COGNITIVE_STATE:   return DECOHERENCE_LIMIT_COGNITIVE_US;
        case MessageClass::CONTROL_ADMIN:     return DECOHERENCE_LIMIT_CONTROL_US;
        case MessageClass::SENSORY_AUDIO:     return DECOHERENCE_LIMIT_AUDIO_US;
    }
    return DECOHERENCE_LIMIT_CONTROL_US;
}

/// Return the decoherence action for a given message class.
[[nodiscard]] constexpr DecoherenceAction decoherence_action(MessageClass cls) noexcept {
    switch (cls) {
        case MessageClass::HIGH_FREQ_PHYSICS: return DecoherenceAction::HARD_DROP;
        case MessageClass::VISUAL_INPUT:      return DecoherenceAction::INTERPOLATE;
        case MessageClass::COGNITIVE_STATE:   return DecoherenceAction::PREDICTIVE_CODE;
        case MessageClass::CONTROL_ADMIN:     return DecoherenceAction::PROCESS;
        case MessageClass::SENSORY_AUDIO:     return DecoherenceAction::JITTER_BUFFER;
    }
    return DecoherenceAction::PROCESS;
}

// ---------------------------------------------------------------------------
// § Temporal coherence predicates
// ---------------------------------------------------------------------------

/// True when `age_us` is within the limit for the given message class.
[[nodiscard]] constexpr bool is_temporally_coherent(double age_us, MessageClass cls) noexcept {
    if (age_us < FUTURE_SKEW_TOLERANCE_US) return false; // from the future
    return age_us <= decoherence_limit_us(cls);
}

/// True when a physics message is still coherent (< 113 μs old).
[[nodiscard]] constexpr bool physics_coherent(double age_us) noexcept {
    return is_temporally_coherent(age_us, MessageClass::HIGH_FREQ_PHYSICS);
}

/// True when the clock offset is within the SYNC_LOCKED band.
[[nodiscard]] constexpr bool ptp_sync_locked(double offset_us) noexcept {
    double abs_offset = offset_us < 0.0 ? -offset_us : offset_us;
    return abs_offset < PTP_LOCK_THRESHOLD_US;
}

/// Classify the current PTP synchronisation state.
[[nodiscard]] constexpr SyncState ptp_sync_state(double offset_us) noexcept {
    double abs_offset = offset_us < 0.0 ? -offset_us : offset_us;
    if (abs_offset < PTP_LOCK_THRESHOLD_US)    return SyncState::SYNC_LOCKED;
    if (abs_offset < PTP_SCRAM_THRESHOLD_US)   return SyncState::SYNC_WARNING;
    return                                            SyncState::SYNC_SCRAM;
}

/// True when the watchdog has been missed for too long (μs elapsed).
[[nodiscard]] constexpr bool watchdog_triggered(double elapsed_us) noexcept {
    return elapsed_us >= WATCHDOG_DEADLINE_US;
}

/// True when a component has missed its heartbeat window (ms elapsed).
[[nodiscard]] constexpr bool heartbeat_missed(double elapsed_ms) noexcept {
    return elapsed_ms >= HEARTBEAT_TIMEOUT_MS;
}

// ---------------------------------------------------------------------------
// § Label helpers
// ---------------------------------------------------------------------------

[[nodiscard]] constexpr std::string_view message_class_label(MessageClass c) noexcept {
    switch (c) {
        case MessageClass::HIGH_FREQ_PHYSICS: return "HIGH_FREQ_PHYSICS";
        case MessageClass::VISUAL_INPUT:      return "VISUAL_INPUT";
        case MessageClass::COGNITIVE_STATE:   return "COGNITIVE_STATE";
        case MessageClass::CONTROL_ADMIN:     return "CONTROL_ADMIN";
        case MessageClass::SENSORY_AUDIO:     return "SENSORY_AUDIO";
    }
    return "UNKNOWN";
}

[[nodiscard]] constexpr std::string_view sync_state_label(SyncState s) noexcept {
    switch (s) {
        case SyncState::SYNC_LOCKED:  return "SYNC_LOCKED";
        case SyncState::SYNC_WARNING: return "SYNC_WARNING";
        case SyncState::SYNC_SCRAM:   return "SYNC_SCRAM";
    }
    return "UNKNOWN";
}

[[nodiscard]] constexpr std::string_view decoherence_action_label(DecoherenceAction a) noexcept {
    switch (a) {
        case DecoherenceAction::ACCEPT:           return "ACCEPT";
        case DecoherenceAction::HARD_DROP:        return "HARD_DROP";
        case DecoherenceAction::INTERPOLATE:      return "INTERPOLATE";
        case DecoherenceAction::PREDICTIVE_CODE:  return "PREDICTIVE_CODE";
        case DecoherenceAction::PROCESS:          return "PROCESS";
        case DecoherenceAction::JITTER_BUFFER:    return "JITTER_BUFFER";
    }
    return "UNKNOWN";
}

} // namespace nikola::system
