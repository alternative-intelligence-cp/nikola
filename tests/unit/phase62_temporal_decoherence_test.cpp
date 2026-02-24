/**
 * @file phase62_temporal_decoherence_test.cpp
 * @brief Phase 62 — GAP-020: Temporal Decoherence Detection Thresholds
 *
 * Coverage map
 * ════════════
 * §1  Constants: GOLDEN_RATIO, BASE_HARMONIC_FREQ_HZ, HARMONIC_ORDER_MAX,
 *     INTERNAL_HARMONIC_LIMIT_HZ — numeric values from spec derivation
 * §2  Phase budget constants: PHASE_INTEGRITY_EPSILON_RAD, amplitude ratio
 * §3  τ_max derivation: DECOHERENCE_TAU_MAX_NS = 1e9/8820 ≈ 113,379 ns
 * §4  Future-tolerance and sensory-buffer delay constants
 * §5  Clock sync thresholds: LOCK, WARNING, SCRAM
 * §6  latency_limit_ns: all five message types
 * §7  coherence_action: all five message types
 * §8  latency_limit_ns ordering invariant (physics < audio < visual < cognitive < control)
 * §9  harmonic_freq_hz: f1, f8 vs spec values; strictly increasing
 * §10 harmonic_freq_hz: throws on order < 1
 * §11 phase_error_rad: zero delay → zero error; proportional scaling
 * §12 phase_error_rad at τ_max → equals PHASE_INTEGRITY_EPSILON_RAD
 * §13 phase_amplitude_ratio: Δφ=0→1, Δφ=π/2→0, Δφ=π→-1
 * §14 amplitude at epsilon_phi > 0.95  (spec "retains >95% amplitude")
 * §15 classify_clock_state: boundary values for LOCKED / WARNING / SCRAM
 * §16 classify_clock_state: all three regions
 * §17 TemporalCoherenceChecker::verify: coherent messages pass
 * §18 verify: message exactly at limit is coherent (boundary inclusion)
 * §19 verify: message just over limit is incoherent
 * §20 verify: future message beyond jitter tolerance is rejected
 * §21 verify: future message within jitter tolerance is accepted
 * §22 requires_hard_drop: only PHYSICS_UPDATE
 * §23 physics_permitted: only SYNC_LOCKED
 * §24 Integration: full physics message lifecycle (send → receive → verify)
 * §25 Integration: clock drift progression through states
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "nikola/infrastructure/temporal_decoherence.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>

using namespace nikola::infrastructure;
using Catch::Approx;

// ============================================================================
// §1 — Golden ratio and harmonic frequency constants
// ============================================================================

TEST_CASE("GAP-020 §1: GOLDEN_RATIO is φ = (1+√5)/2", "[gap020][constants]") {
    const double phi = (1.0 + std::sqrt(5.0)) / 2.0;
    REQUIRE(GOLDEN_RATIO == Approx(phi).epsilon(1e-12));
    // Defining property: φ² = φ + 1
    REQUIRE(GOLDEN_RATIO * GOLDEN_RATIO == Approx(GOLDEN_RATIO + 1.0).epsilon(1e-12));
}

TEST_CASE("GAP-020 §2: BASE_HARMONIC_FREQ_HZ = π × φ¹ ≈ 5.083 Hz", "[gap020][constants]") {
    REQUIRE(BASE_HARMONIC_FREQ_HZ == Approx(M_PI * GOLDEN_RATIO).epsilon(1e-9));
    // Spec: "f₁ = π·φ¹ ≈ 5.083 Hz"
    REQUIRE(BASE_HARMONIC_FREQ_HZ == Approx(5.083).epsilon(0.001));
}

TEST_CASE("GAP-020 §3: HARMONIC_ORDER_MAX == 8", "[gap020][constants]") {
    REQUIRE(HARMONIC_ORDER_MAX == 8);
}

TEST_CASE("GAP-020 §4: INTERNAL_HARMONIC_LIMIT_HZ == 441.0 Hz", "[gap020][constants]") {
    REQUIRE(INTERNAL_HARMONIC_LIMIT_HZ == Approx(441.0));
}

// ============================================================================
// §2 — Phase budget constants
// ============================================================================

TEST_CASE("GAP-020 §5: PHASE_INTEGRITY_EPSILON_RAD = π/10 = 18°", "[gap020][phase]") {
    REQUIRE(PHASE_INTEGRITY_EPSILON_RAD == Approx(M_PI / 10.0).epsilon(1e-12));
    // In degrees
    const double deg = PHASE_INTEGRITY_EPSILON_RAD * (180.0 / M_PI);
    REQUIRE(deg == Approx(18.0).epsilon(1e-9));
}

TEST_CASE("GAP-020 §6: PHASE_INTEGRITY_AMPLITUDE_RATIO = cos(π/10) ≈ 0.951",
          "[gap020][phase]") {
    REQUIRE(PHASE_INTEGRITY_AMPLITUDE_RATIO == Approx(std::cos(M_PI / 10.0)).epsilon(1e-12));
    // Spec: ">95% theoretical amplitude"
    REQUIRE(PHASE_INTEGRITY_AMPLITUDE_RATIO > 0.95);
    REQUIRE(PHASE_INTEGRITY_AMPLITUDE_RATIO < 1.0);
}

// ============================================================================
// §3 — τ_max derivation
// ============================================================================

TEST_CASE("GAP-020 §7: DECOHERENCE_TAU_MAX_NS derives from the formula",
          "[gap020][tau]") {
    // τ_max = ε_φ / (2π f_max) = (π/10) / (2π × 441) = 1/(20 × 441) = 1/8820 s
    const double tau_exact_ns = 1e9 / (20.0 * INTERNAL_HARMONIC_LIMIT_HZ);
    REQUIRE(static_cast<double>(DECOHERENCE_TAU_MAX_NS) == Approx(tau_exact_ns).epsilon(1.0));
    // Must be ≈ 113 μs (spec value) — within 500 ns of rounded value
    REQUIRE(DECOHERENCE_TAU_MAX_NS == Approx(113'000.0).margin(500.0));
}

TEST_CASE("GAP-020 §8: DECOHERENCE_TAU_MAX_NS_ROUNDED == 113,000 ns (113 μs)",
          "[gap020][tau]") {
    REQUIRE(DECOHERENCE_TAU_MAX_NS_ROUNDED == 113'000);
    // Exact value must be close to rounded value
    REQUIRE(std::abs(DECOHERENCE_TAU_MAX_NS - DECOHERENCE_TAU_MAX_NS_ROUNDED) < 500);
}

// ============================================================================
// §4 — Future tolerance and sensory buffer
// ============================================================================

TEST_CASE("GAP-020 §9: Future tolerance and sensory buffer constants",
          "[gap020][constants]") {
    REQUIRE(JITTER_FUTURE_TOLERANCE_NS == -50'000);         // −50 μs
    REQUIRE(SENSORY_BUFFER_DELAY_NS    == 50'000'000);      // 50 ms
}

// ============================================================================
// §5 — Clock sync thresholds
// ============================================================================

TEST_CASE("GAP-020 §10: Clock sync threshold values (LOCK < WARNING < SCRAM)",
          "[gap020][clock]") {
    REQUIRE(CLOCK_SYNC_LOCK_THRESHOLD_NS    ==  50'000);   //  50 μs
    REQUIRE(CLOCK_SYNC_WARNING_THRESHOLD_NS == 100'000);   // 100 μs
    REQUIRE(CLOCK_SYNC_SCRAM_THRESHOLD_NS   == 150'000);   // 150 μs

    // Ordering invariant
    REQUIRE(CLOCK_SYNC_LOCK_THRESHOLD_NS < CLOCK_SYNC_WARNING_THRESHOLD_NS);
    REQUIRE(CLOCK_SYNC_WARNING_THRESHOLD_NS < CLOCK_SYNC_SCRAM_THRESHOLD_NS);

    // Physics τ_max is between LOCK and SCRAM (validates design margin)
    REQUIRE(DECOHERENCE_TAU_MAX_NS > CLOCK_SYNC_LOCK_THRESHOLD_NS);
    REQUIRE(DECOHERENCE_TAU_MAX_NS < CLOCK_SYNC_SCRAM_THRESHOLD_NS);
}

// ============================================================================
// §6 — latency_limit_ns
// ============================================================================

TEST_CASE("GAP-020 §11: latency_limit_ns values for all five message types",
          "[gap020][threshold]") {
    REQUIRE(latency_limit_ns(MessageType::PHYSICS_UPDATE)  == DECOHERENCE_TAU_MAX_NS);
    REQUIRE(latency_limit_ns(MessageType::VISUAL_INPUT)    ==   8'333'333);
    REQUIRE(latency_limit_ns(MessageType::COGNITIVE_STATE) ==  10'000'000);
    REQUIRE(latency_limit_ns(MessageType::CONTROL_ADMIN)   == 100'000'000);
    REQUIRE(latency_limit_ns(MessageType::SENSORY_AUDIO)   ==  50'000'000);
}

// ============================================================================
// §7 — coherence_action
// ============================================================================

TEST_CASE("GAP-020 §12: coherence_action for all five message types",
          "[gap020][action]") {
    REQUIRE(coherence_action(MessageType::PHYSICS_UPDATE)  == CoherenceAction::HARD_DROP);
    REQUIRE(coherence_action(MessageType::VISUAL_INPUT)    == CoherenceAction::INTERPOLATE);
    REQUIRE(coherence_action(MessageType::COGNITIVE_STATE) == CoherenceAction::KALMAN_PREDICT);
    REQUIRE(coherence_action(MessageType::CONTROL_ADMIN)   == CoherenceAction::PROCESS);
    REQUIRE(coherence_action(MessageType::SENSORY_AUDIO)   == CoherenceAction::JITTER_BUFFER);
}

// ============================================================================
// §8 — Ordering invariants
// ============================================================================

TEST_CASE("GAP-020 §13: Latency limits are strictly ordered physics < visual < cognitive",
          "[gap020][ordering]") {
    // Physics is most restrictive, admin is most permissive
    REQUIRE(latency_limit_ns(MessageType::PHYSICS_UPDATE)
          < latency_limit_ns(MessageType::VISUAL_INPUT));
    REQUIRE(latency_limit_ns(MessageType::VISUAL_INPUT)
          < latency_limit_ns(MessageType::COGNITIVE_STATE));
    REQUIRE(latency_limit_ns(MessageType::COGNITIVE_STATE)
          < latency_limit_ns(MessageType::SENSORY_AUDIO));
    REQUIRE(latency_limit_ns(MessageType::SENSORY_AUDIO)
          < latency_limit_ns(MessageType::CONTROL_ADMIN));
}

// ============================================================================
// §9 — harmonic_freq_hz
// ============================================================================

TEST_CASE("GAP-020 §14: harmonic_freq_hz(1) ≈ 5.083 Hz and (8) ≈ 146.6 Hz",
          "[gap020][harmonic]") {
    REQUIRE(harmonic_freq_hz(1) == Approx(M_PI * GOLDEN_RATIO).epsilon(1e-9));
    REQUIRE(harmonic_freq_hz(1) == Approx(5.083).epsilon(0.001));

    const double f8_expected = M_PI * std::pow(GOLDEN_RATIO, 8.0);
    REQUIRE(harmonic_freq_hz(8) == Approx(f8_expected).epsilon(1e-6));
    REQUIRE(harmonic_freq_hz(8) == Approx(146.6).epsilon(0.1));  // spec "≈ 146.6 Hz"
}

TEST_CASE("GAP-020 §15: harmonic_freq_hz is strictly increasing in order",
          "[gap020][harmonic]") {
    for (int n = 1; n < 10; ++n) {
        REQUIRE(harmonic_freq_hz(n + 1) > harmonic_freq_hz(n));
    }
}

TEST_CASE("GAP-020 §16: harmonic_freq_hz(8) < INTERNAL_HARMONIC_LIMIT_HZ",
          "[gap020][harmonic]") {
    // The driven maximum f₈ ≈ 146.6 Hz must be below the 441 Hz internal limit
    REQUIRE(harmonic_freq_hz(HARMONIC_ORDER_MAX) < INTERNAL_HARMONIC_LIMIT_HZ);
}

// ============================================================================
// §10 — harmonic_freq_hz error handling
// ============================================================================

TEST_CASE("GAP-020 §17: harmonic_freq_hz throws invalid_argument for order < 1",
          "[gap020][harmonic]") {
    REQUIRE_THROWS_AS(harmonic_freq_hz(0),  std::invalid_argument);
    REQUIRE_THROWS_AS(harmonic_freq_hz(-1), std::invalid_argument);
    REQUIRE_NOTHROW(harmonic_freq_hz(1));
}

// ============================================================================
// §11 — phase_error_rad
// ============================================================================

TEST_CASE("GAP-020 §18: phase_error_rad: zero delay → zero error", "[gap020][phase_err]") {
    REQUIRE(phase_error_rad(441.0, 0) == Approx(0.0));
    REQUIRE(phase_error_rad(13.3, 0)  == Approx(0.0));
}

TEST_CASE("GAP-020 §19: phase_error_rad grows linearly with frequency and delay",
          "[gap020][phase_err]") {
    // Doubling frequency doubles the phase error
    const double phi_100 = phase_error_rad(100.0, 1000);
    const double phi_200 = phase_error_rad(200.0, 1000);
    REQUIRE(phi_200 == Approx(2.0 * phi_100).epsilon(1e-9));

    // Doubling delay doubles the phase error
    const double phi_d1 = phase_error_rad(441.0, 10'000);
    const double phi_d2 = phase_error_rad(441.0, 20'000);
    REQUIRE(phi_d2 == Approx(2.0 * phi_d1).epsilon(1e-9));
}

// ============================================================================
// §12 — phase_error_rad at τ_max
// ============================================================================

TEST_CASE("GAP-020 §20: phase_error at τ_max equals PHASE_INTEGRITY_EPSILON_RAD",
          "[gap020][phase_err]") {
    // By construction: τ_max = ε_φ / (2π × f_max)
    // → 2π × f_max × τ_max = ε_φ
    const double phi = phase_error_rad(INTERNAL_HARMONIC_LIMIT_HZ, DECOHERENCE_TAU_MAX_NS);
    REQUIRE(phi == Approx(PHASE_INTEGRITY_EPSILON_RAD).epsilon(1e-4));
}

// ============================================================================
// §13 — phase_amplitude_ratio
// ============================================================================

TEST_CASE("GAP-020 §21: phase_amplitude_ratio: key values on unit circle",
          "[gap020][amplitude]") {
    REQUIRE(phase_amplitude_ratio(0.0)          == Approx(1.0).margin(1e-12));
    REQUIRE(phase_amplitude_ratio(M_PI / 2.0)   == Approx(0.0).margin(1e-12));
    REQUIRE(phase_amplitude_ratio(M_PI)         == Approx(-1.0).margin(1e-12));
    REQUIRE(phase_amplitude_ratio(2.0 * M_PI)   == Approx(1.0).margin(1e-12));
}

// ============================================================================
// §14 — Amplitude at ε_φ
// ============================================================================

TEST_CASE("GAP-020 §22: Amplitude at epsilon_phi > 95% (spec guarantee)",
          "[gap020][amplitude]") {
    const double ratio = phase_amplitude_ratio(PHASE_INTEGRITY_EPSILON_RAD);
    REQUIRE(ratio == Approx(PHASE_INTEGRITY_AMPLITUDE_RATIO).epsilon(1e-9));
    REQUIRE(ratio > 0.95);
    // Rayleigh criterion (π/2) → amplitude = 0; we're well within it
    REQUIRE(ratio > phase_amplitude_ratio(M_PI / 2.0));
}

// ============================================================================
// §15-16 — classify_clock_state
// ============================================================================

TEST_CASE("GAP-020 §23: classify_clock_state — SYNC_LOCKED below 50 μs",
          "[gap020][clock_state]") {
    REQUIRE(TemporalCoherenceChecker::classify_clock_state(0)      == ClockSyncState::SYNC_LOCKED);
    REQUIRE(TemporalCoherenceChecker::classify_clock_state(25'000) == ClockSyncState::SYNC_LOCKED);
    REQUIRE(TemporalCoherenceChecker::classify_clock_state(49'999) == ClockSyncState::SYNC_LOCKED);
}

TEST_CASE("GAP-020 §24: classify_clock_state — SYNC_WARNING between 50 μs and 150 μs",
          "[gap020][clock_state]") {
    REQUIRE(TemporalCoherenceChecker::classify_clock_state(50'000)  == ClockSyncState::SYNC_WARNING);
    REQUIRE(TemporalCoherenceChecker::classify_clock_state(100'000) == ClockSyncState::SYNC_WARNING);
    REQUIRE(TemporalCoherenceChecker::classify_clock_state(149'999) == ClockSyncState::SYNC_WARNING);
}

TEST_CASE("GAP-020 §25: classify_clock_state — SYNC_SCRAM at or above 150 μs",
          "[gap020][clock_state]") {
    REQUIRE(TemporalCoherenceChecker::classify_clock_state(150'000) == ClockSyncState::SYNC_SCRAM);
    REQUIRE(TemporalCoherenceChecker::classify_clock_state(200'000) == ClockSyncState::SYNC_SCRAM);
    REQUIRE(TemporalCoherenceChecker::classify_clock_state(1'000'000) == ClockSyncState::SYNC_SCRAM);
}

// ============================================================================
// §17 — verify: coherent messages pass
// ============================================================================

TEST_CASE("GAP-020 §26: verify accepts messages well within their limits",
          "[gap020][verify]") {
    const int64_t now = 1'000'000'000LL;  // arbitrary reference

    // Physics: well within 113 μs
    REQUIRE(TemporalCoherenceChecker::verify(now, now - 50'000, MessageType::PHYSICS_UPDATE));
    // Visual: within 8.3 ms
    REQUIRE(TemporalCoherenceChecker::verify(now, now - 5'000'000, MessageType::VISUAL_INPUT));
    // Cognitive: within 10 ms
    REQUIRE(TemporalCoherenceChecker::verify(now, now - 8'000'000, MessageType::COGNITIVE_STATE));
    // Control: within 100 ms
    REQUIRE(TemporalCoherenceChecker::verify(now, now - 50'000'000, MessageType::CONTROL_ADMIN));
    // Audio: within 50 ms
    REQUIRE(TemporalCoherenceChecker::verify(now, now - 30'000'000, MessageType::SENSORY_AUDIO));
}

// ============================================================================
// §18 — verify: message exactly at limit
// ============================================================================

TEST_CASE("GAP-020 §27: verify: message exactly at latency limit is coherent",
          "[gap020][verify]") {
    const int64_t now = 1'000'000'000LL;

    // Physics: age == 113,379 ns (exactly at limit — spec uses >, so this passes)
    REQUIRE(TemporalCoherenceChecker::verify(
        now, now - DECOHERENCE_TAU_MAX_NS, MessageType::PHYSICS_UPDATE));

    // Control: exactly at 100 ms
    REQUIRE(TemporalCoherenceChecker::verify(
        now, now - 100'000'000, MessageType::CONTROL_ADMIN));
}

// ============================================================================
// §19 — verify: messages over limit are rejected
// ============================================================================

TEST_CASE("GAP-020 §28: verify rejects messages older than their class limit",
          "[gap020][verify]") {
    const int64_t now = 1'000'000'000LL;

    // Physics: 114 μs — just past 113,379 ns
    REQUIRE_FALSE(TemporalCoherenceChecker::verify(
        now, now - 114'000, MessageType::PHYSICS_UPDATE));

    // Visual: 10 ms (> 8.333 ms)
    REQUIRE_FALSE(TemporalCoherenceChecker::verify(
        now, now - 10'000'000, MessageType::VISUAL_INPUT));

    // Cognitive: 11 ms (> 10 ms)
    REQUIRE_FALSE(TemporalCoherenceChecker::verify(
        now, now - 11'000'000, MessageType::COGNITIVE_STATE));

    // Control: 101 ms (> 100 ms)
    REQUIRE_FALSE(TemporalCoherenceChecker::verify(
        now, now - 101'000'000, MessageType::CONTROL_ADMIN));
}

// ============================================================================
// §20 — verify: far-future messages rejected
// ============================================================================

TEST_CASE("GAP-020 §29: verify rejects messages from the future beyond jitter tolerance",
          "[gap020][verify]") {
    const int64_t now = 1'000'000'000LL;

    // 60 μs in the future (beyond −50 μs tolerance) → rejected
    REQUIRE_FALSE(TemporalCoherenceChecker::verify(
        now, now + 60'000, MessageType::PHYSICS_UPDATE));
    REQUIRE_FALSE(TemporalCoherenceChecker::verify(
        now, now + 60'000, MessageType::COGNITIVE_STATE));
}

// ============================================================================
// §21 — verify: slight future within jitter tolerance accepted
// ============================================================================

TEST_CASE("GAP-020 §30: verify accepts messages slightly in the future (within jitter)",
          "[gap020][verify]") {
    const int64_t now = 1'000'000'000LL;

    // 30 μs in the future — within −50 μs tolerance → coherent
    REQUIRE(TemporalCoherenceChecker::verify(now, now + 30'000, MessageType::PHYSICS_UPDATE));
    REQUIRE(TemporalCoherenceChecker::verify(now, now + 49'999, MessageType::COGNITIVE_STATE));
}

// ============================================================================
// §22 — requires_hard_drop
// ============================================================================

TEST_CASE("GAP-020 §31: requires_hard_drop only true for PHYSICS_UPDATE",
          "[gap020][action]") {
    REQUIRE(TemporalCoherenceChecker::requires_hard_drop(MessageType::PHYSICS_UPDATE));
    REQUIRE_FALSE(TemporalCoherenceChecker::requires_hard_drop(MessageType::VISUAL_INPUT));
    REQUIRE_FALSE(TemporalCoherenceChecker::requires_hard_drop(MessageType::COGNITIVE_STATE));
    REQUIRE_FALSE(TemporalCoherenceChecker::requires_hard_drop(MessageType::CONTROL_ADMIN));
    REQUIRE_FALSE(TemporalCoherenceChecker::requires_hard_drop(MessageType::SENSORY_AUDIO));
}

// ============================================================================
// §23 — physics_permitted
// ============================================================================

TEST_CASE("GAP-020 §32: physics_permitted only in SYNC_LOCKED state",
          "[gap020][clock_state]") {
    REQUIRE(TemporalCoherenceChecker::physics_permitted(ClockSyncState::SYNC_LOCKED));
    REQUIRE_FALSE(TemporalCoherenceChecker::physics_permitted(ClockSyncState::SYNC_WARNING));
    REQUIRE_FALSE(TemporalCoherenceChecker::physics_permitted(ClockSyncState::SYNC_SCRAM));
}

// ============================================================================
// §24 — Integration: physics message lifecycle
// ============================================================================

TEST_CASE("GAP-020 §33: Integration — physics message send/receive lifecycle",
          "[gap020][integration]") {
    // Simulate a message sent at t=0 received at varying ages
    const int64_t t_send = 500'000'000'000LL;  // arbitrary epoch

    // Message received after 50 μs — coherent
    {
        const int64_t t_recv = t_send + 50'000;
        REQUIRE(TemporalCoherenceChecker::verify(t_recv, t_send, MessageType::PHYSICS_UPDATE));
        const double phi = phase_error_rad(INTERNAL_HARMONIC_LIMIT_HZ, 50'000);
        REQUIRE(phase_amplitude_ratio(phi) > PHASE_INTEGRITY_AMPLITUDE_RATIO);
    }

    // Message received after 113,379 ns — exactly at threshold (still coherent)
    {
        const int64_t t_recv = t_send + DECOHERENCE_TAU_MAX_NS;
        REQUIRE(TemporalCoherenceChecker::verify(t_recv, t_send, MessageType::PHYSICS_UPDATE));
        const double phi = phase_error_rad(INTERNAL_HARMONIC_LIMIT_HZ, DECOHERENCE_TAU_MAX_NS);
        REQUIRE(phi == Approx(PHASE_INTEGRITY_EPSILON_RAD).epsilon(1e-4));
    }

    // Message received after 200 μs — decoherent (TCP loopback regime)
    {
        const int64_t t_recv = t_send + 200'000;
        REQUIRE_FALSE(TemporalCoherenceChecker::verify(t_recv, t_send, MessageType::PHYSICS_UPDATE));
        const double phi = phase_error_rad(INTERNAL_HARMONIC_LIMIT_HZ, 200'000);
        // Phase error exceeds ε_φ → interference is degraded
        // 2π × 441 × 200e-6 ≈ 0.554 rad > π/10 ≈ 0.314 rad
        REQUIRE(phi > PHASE_INTEGRITY_EPSILON_RAD);
        // Amplitude has fallen below the 95% spec guarantee
        REQUIRE(phase_amplitude_ratio(phi) < PHASE_INTEGRITY_AMPLITUDE_RATIO);
    }
}

// ============================================================================
// §25 — Integration: clock drift state progression
// ============================================================================

TEST_CASE("GAP-020 §34: Integration — PTP clock drift state machine progression",
          "[gap020][integration]") {
    // Simulate worsening clock drift
    const std::array<int64_t, 6> drifts{
        0, 30'000, 50'000, 100'000, 149'999, 150'000
    };
    const std::array<ClockSyncState, 6> expected{
        ClockSyncState::SYNC_LOCKED,
        ClockSyncState::SYNC_LOCKED,
        ClockSyncState::SYNC_WARNING,
        ClockSyncState::SYNC_WARNING,
        ClockSyncState::SYNC_WARNING,
        ClockSyncState::SYNC_SCRAM,
    };

    for (std::size_t i = 0; i < drifts.size(); ++i) {
        const auto state = TemporalCoherenceChecker::classify_clock_state(drifts[i]);
        REQUIRE(state == expected[i]);
        // Physics is only safe in LOCKED
        if (expected[i] == ClockSyncState::SYNC_LOCKED) {
            REQUIRE(TemporalCoherenceChecker::physics_permitted(state));
        } else {
            REQUIRE_FALSE(TemporalCoherenceChecker::physics_permitted(state));
        }
    }
}
