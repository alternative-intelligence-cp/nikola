/**
 * @file temporal_decoherence.hpp
 * @brief GAP-020: Temporal Decoherence Detection Thresholds
 *
 * @spec FABRICATION-READY — docs/info/integration/sections/04_infrastructure/
 *       01_zeromq_spine.md §GAP-020
 *
 * ### Background: The Physics of Synchronization
 * Every NeuralSpike message carries a 64-bit nanosecond timestamp.  When a
 * message arrives at a downstream node the integration epoch has already
 * advanced by Δt = T_now − T_source.  The resulting phase error is:
 *
 *   Δφ = ω × Δt       [radians]
 *
 * Once Δφ > π/2 (Rayleigh criterion, λ/4) constructive interference flips to
 * destructive — the delayed signal actively erases the memory it should
 * reinforce, dissipating spectral energy into noise ("mind decoheres").
 *
 * ### Threshold Derivation (spec §GAP-020)
 * Base harmonic  : f₁ = π × φ¹        ≈   5.083 Hz  (φ = golden ratio)
 * Driven maximum : f₈ = π × φ⁸        ≈ 146.6  Hz
 * Internal limit : f_max = 441 Hz      (Nyquist compliance, nonlinear N̂)
 * Phase budget   : ε_φ = π/10 (18°)   → retains cos(18°) ≈ 95% amplitude
 *
 *   τ_max = ε_φ / (2π f_max)
 *         = (π/10) / (2π × 441)
 *         = 1 / (20 × 441)
 *         ≈ 113.379 μs  →  113 μs rounded
 *
 * ### Tiered Latency Table (spec §GAP-020 "Adaptive Threshold")
 * | Message Class   | Carrier Hz | τ_max       | Action        |
 * |-----------------|------------|-------------|---------------|
 * | Physics Update  | 441 Hz     | 113 μs      | Hard Drop     |
 * | Visual Input    | 60 Hz      | 8.333 ms    | Interpolate   |
 * | Cognitive State | 13.3 Hz    | 10 ms       | Kalman Pred.  |
 * | Control / Admin | DC (0 Hz)  | 100 ms      | Process       |
 * | Sensory (Audio) | 44.1 kHz   | 50 ms buf.  | Jitter Buffer |
 *
 * ### Clock Sync State Machine (PTP / IEEE 1588)
 *   SYNC_LOCKED  : |θ| < 50 μs   → physics simulation permitted
 *   SYNC_WARNING : 50 μs ≤ |θ| < 100 μs → Oracle applies Virtual Time Dilation
 *   SYNC_SCRAM   : |θ| ≥ 150 μs  → soft SCRAM, detach from cluster
 *
 * Standard TCP/IP loopback latency (500–1500 μs) is an order of magnitude too
 * slow for physics messages. This spec validates the requirement for Shared
 * Memory (Seqlock) IPC on the Data Plane.
 */
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace nikola::infrastructure {

// ---------------------------------------------------------------------------
// Fundamental constants
// ---------------------------------------------------------------------------

/// Golden ratio φ = (1 + √5) / 2.
inline constexpr double GOLDEN_RATIO = 1.6180339887498948482;

/// Base harmonic frequency: f₁ = π × φ¹ ≈ 5.083 Hz.
inline constexpr double BASE_HARMONIC_FREQ_HZ = M_PI * GOLDEN_RATIO;

/// Maximum driven harmonic order used in the emitter array.
inline constexpr int HARMONIC_ORDER_MAX = 8;

/// Internal harmonic limit for Nyquist compliance / nonlinear N̂ operator.
/// All internal wavepackets are ≤ 441 Hz.
inline constexpr double INTERNAL_HARMONIC_LIMIT_HZ = 441.0;

/// Phase integrity budget ε_φ = π/10 (18°).
/// Guarantees cos(ε_φ) ≈ 0.951 → >95% constructive amplitude.
inline constexpr double PHASE_INTEGRITY_EPSILON_RAD = M_PI / 10.0;

/// Amplitude retention at the phase budget limit: cos(π/10) ≈ 0.951.
inline constexpr double PHASE_INTEGRITY_AMPLITUDE_RATIO = 0.9510565162951535;

/// Exact physics latency ceiling:
///   τ_max = ε_φ / (2π × f_max) = (π/10) / (2π × 441) = 1/8820 s
///         ≈ 113,379 ns
inline constexpr int64_t DECOHERENCE_TAU_MAX_NS = 113'379;  // exact: 1e9/8820

/// Rounded physics latency (as used in reference implementation):  113 μs.
inline constexpr int64_t DECOHERENCE_TAU_MAX_NS_ROUNDED = 113'000;

/// Future-message tolerance: allow up to −50 μs skew for PTP clock jitter.
/// A message with age_ns < −50,000 is considered "from the future" and dropped.
inline constexpr int64_t JITTER_FUTURE_TOLERANCE_NS = -50'000;

// ---------------------------------------------------------------------------
// Clock synchronisation thresholds (PTP / IEEE 1588)
// ---------------------------------------------------------------------------

/// SYNC_LOCKED: |θ| < 50 μs — physics simulation is permitted.
inline constexpr int64_t CLOCK_SYNC_LOCK_THRESHOLD_NS = 50'000;

/// SYNC_WARNING threshold: |θ| < 100 μs — Oracle compensates via Virtual
/// Time Dilation; simulation continues with reduced confidence.
inline constexpr int64_t CLOCK_SYNC_WARNING_THRESHOLD_NS = 100'000;

/// SYNC_SCRAM threshold: |θ| ≥ 150 μs — exceeds τ_max + margin.
/// Triggers soft SCRAM; node detaches from cluster to prevent coherence
/// pollution.
inline constexpr int64_t CLOCK_SYNC_SCRAM_THRESHOLD_NS = 150'000;

/// Isochronous sensory buffer presentation delay: 50 ms.
/// Audio/video are re-clocked to physics time within this window.
inline constexpr int64_t SENSORY_BUFFER_DELAY_NS = 50'000'000;

// ---------------------------------------------------------------------------
// Message classification
// ---------------------------------------------------------------------------

/**
 * @brief Message class used to select the appropriate latency threshold.
 *
 * Spec §GAP-020 "Adaptive Threshold" tiered table.
 */
enum class MessageType : int {
    PHYSICS_UPDATE  = 0,  ///< 441 Hz harmonic limit → 113 μs hard drop
    VISUAL_INPUT    = 1,  ///< 60 Hz frame rate → 8.333 ms interpolate
    COGNITIVE_STATE = 2,  ///< 13.3 Hz theta/alpha → 10 ms Kalman predict
    CONTROL_ADMIN   = 3,  ///< DC (atemporal) → 100 ms process
    SENSORY_AUDIO   = 4,  ///< 44.1 kHz PCM → 50 ms jitter buffer
};

/**
 * @brief Prescribed action when the latency threshold is exceeded.
 */
enum class CoherenceAction : int {
    HARD_DROP      = 0,  ///< Discard — phase-corrupt signal adds entropy
    INTERPOLATE    = 1,  ///< Sample-and-hold or optical-flow fill
    KALMAN_PREDICT = 2,  ///< Project state to T_now via Kalman filter
    PROCESS        = 3,  ///< Atemporal — apply regardless
    JITTER_BUFFER  = 4,  ///< Re-clock to physics time within 50 ms window
};

// ---------------------------------------------------------------------------
// Clock synchronisation state (PTP state machine)
// ---------------------------------------------------------------------------

/**
 * @brief Physics Oracle Timekeeper state (spec §GAP-020 §"State Machine").
 */
enum class ClockSyncState : int {
    SYNC_LOCKED  = 0,  ///< |θ| < 50 μs  — physics simulation permitted
    SYNC_WARNING = 1,  ///< 50 μs ≤ |θ| < 100 μs — Oracle compensates
    SYNC_SCRAM   = 2,  ///< |θ| ≥ 150 μs — soft SCRAM, detach from cluster
};

// ---------------------------------------------------------------------------
// Threshold accessors
// ---------------------------------------------------------------------------

/**
 * @brief Return the maximum allowable message age (ns) for a given class.
 *
 * Values:
 *   PHYSICS_UPDATE  → 113,379 ns  (exact τ_max from formula)
 *   VISUAL_INPUT    → 8,333,333 ns  (1/(2×60) s)
 *   COGNITIVE_STATE → 10,000,000 ns (10 ms)
 *   CONTROL_ADMIN   → 100,000,000 ns (100 ms)
 *   SENSORY_AUDIO   → 50,000,000 ns  (50 ms jitter buffer)
 */
[[nodiscard]] inline constexpr int64_t latency_limit_ns(MessageType t) noexcept {
    switch (t) {
        case MessageType::PHYSICS_UPDATE:  return DECOHERENCE_TAU_MAX_NS;
        case MessageType::VISUAL_INPUT:    return  8'333'333;
        case MessageType::COGNITIVE_STATE: return 10'000'000;
        case MessageType::CONTROL_ADMIN:   return 100'000'000;
        case MessageType::SENSORY_AUDIO:   return 50'000'000;
    }
    return 100'000'000;  // safe fallback
}

/**
 * @brief Return the prescribed action when the threshold for @p t is violated.
 */
[[nodiscard]] inline constexpr CoherenceAction coherence_action(MessageType t) noexcept {
    switch (t) {
        case MessageType::PHYSICS_UPDATE:  return CoherenceAction::HARD_DROP;
        case MessageType::VISUAL_INPUT:    return CoherenceAction::INTERPOLATE;
        case MessageType::COGNITIVE_STATE: return CoherenceAction::KALMAN_PREDICT;
        case MessageType::CONTROL_ADMIN:   return CoherenceAction::PROCESS;
        case MessageType::SENSORY_AUDIO:   return CoherenceAction::JITTER_BUFFER;
    }
    return CoherenceAction::PROCESS;
}

// ---------------------------------------------------------------------------
// Harmonic frequency helper
// ---------------------------------------------------------------------------

/**
 * @brief Compute the n-th order golden-ratio harmonic: f_n = π × φ^n.
 *
 * Spec §GAP-020: "Base frequency f₁ = π·φ¹; maximum driven f₈ = π·φ⁸".
 *
 * @param order  Harmonic index n ≥ 1
 */
[[nodiscard]] inline double harmonic_freq_hz(int order) {
    if (order < 1)
        throw std::invalid_argument("harmonic_freq_hz: order must be >= 1");
    return M_PI * std::pow(GOLDEN_RATIO, static_cast<double>(order));
}

// ---------------------------------------------------------------------------
// Phase-error calculations
// ---------------------------------------------------------------------------

/**
 * @brief Phase error accumulated over a delay.
 *
 * Formula: Δφ = ω × Δt  = 2π × f × delay_s
 *
 * @param freq_hz   Carrier frequency of the wavepacket
 * @param delay_ns  Message age (T_now − T_source) in nanoseconds
 * @return Phase error in radians
 */
[[nodiscard]] inline double phase_error_rad(double freq_hz, int64_t delay_ns) noexcept {
    const double delay_s = static_cast<double>(delay_ns) * 1e-9;
    return 2.0 * M_PI * freq_hz * delay_s;
}

/**
 * @brief Interference amplitude ratio given a phase error.
 *
 * Returns cos(Δφ): +1.0 = full constructive, 0.0 = quadrature, −1.0 = full
 * destructive (complete memory erasure).
 *
 * @param phi_rad  Phase error in radians
 */
[[nodiscard]] inline double phase_amplitude_ratio(double phi_rad) noexcept {
    return std::cos(phi_rad);
}

// ---------------------------------------------------------------------------
// TemporalCoherenceChecker
// ---------------------------------------------------------------------------

/**
 * @brief Stateless validator for temporal coherence of incoming messages.
 *
 * Spec §GAP-020 "Implementation: Timestamp Enforcement":
 *   verify_temporal_coherence(msg_timestamp_ns, MessageType) →
 *     true  = coherent (pass through to physics engine)
 *     false = decoherent (drop / route to remediation path)
 *
 * Two failure modes:
 *   1. "From the future": age_ns < JITTER_FUTURE_TOLERANCE_NS  (clock skew)
 *   2. "Too old":         age_ns > latency_limit_ns(type)
 */
class TemporalCoherenceChecker {
public:
    TemporalCoherenceChecker() = delete;

    /**
     * @brief Verify temporal coherence given explicit now and message timestamps.
     *
     * @param now_ns            Current physics-clock time in nanoseconds
     * @param msg_timestamp_ns  Message creation timestamp (PTP source)
     * @param type              Message classification
     * @return true if coherent, false if decoherent (should be dropped)
     */
    [[nodiscard]] static bool verify(int64_t now_ns,
                                     int64_t msg_timestamp_ns,
                                     MessageType type) noexcept {
        const int64_t age_ns = now_ns - msg_timestamp_ns;

        // Future check: reject if age < –50 μs (PTP skew tolerance exceeded)
        if (age_ns < JITTER_FUTURE_TOLERANCE_NS)
            return false;

        // Staleness check: reject if age > class threshold
        if (age_ns > latency_limit_ns(type))
            return false;

        return true;
    }

    /**
     * @brief Determine PTP clock sync state from measured offset magnitude.
     *
     * @param abs_offset_ns  |θ| — absolute clock offset in nanoseconds
     */
    [[nodiscard]] static ClockSyncState classify_clock_state(int64_t abs_offset_ns) noexcept {
        if (abs_offset_ns < CLOCK_SYNC_LOCK_THRESHOLD_NS)    return ClockSyncState::SYNC_LOCKED;
        if (abs_offset_ns < CLOCK_SYNC_SCRAM_THRESHOLD_NS)   return ClockSyncState::SYNC_WARNING;
        return ClockSyncState::SYNC_SCRAM;
    }

    /**
     * @brief True if this message class requires a Hard Drop on violation.
     */
    [[nodiscard]] static constexpr bool requires_hard_drop(MessageType t) noexcept {
        return coherence_action(t) == CoherenceAction::HARD_DROP;
    }

    /**
     * @brief True if the clock is in a state that allows physics simulation.
     * Physics simulation is ONLY safe in SYNC_LOCKED.
     */
    [[nodiscard]] static constexpr bool physics_permitted(ClockSyncState s) noexcept {
        return s == ClockSyncState::SYNC_LOCKED;
    }
};

} // namespace nikola::infrastructure
