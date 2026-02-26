// =============================================================================
// tests/unit/phase84_temporal_coherence_test.cpp
// Phase 84 — GAP-020: Temporal Decoherence Detection Thresholds
//
// Tests for nikola::system::temporal_coherence.hpp
// Spec: docs/info/integration/sections/04_infrastructure/01_zeromq_spine.md
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "nikola/system/temporal_coherence.hpp"

using namespace nikola::system;
using Catch::Approx;

// ---------------------------------------------------------------------------
// § Enums
// ---------------------------------------------------------------------------

TEST_CASE("MessageClass enum values are distinct", "[enums][phase84]") {
    CHECK(static_cast<int>(MessageClass::HIGH_FREQ_PHYSICS) == 0);
    CHECK(static_cast<int>(MessageClass::VISUAL_INPUT)      == 1);
    CHECK(static_cast<int>(MessageClass::COGNITIVE_STATE)   == 2);
    CHECK(static_cast<int>(MessageClass::CONTROL_ADMIN)     == 3);
    CHECK(static_cast<int>(MessageClass::SENSORY_AUDIO)     == 4);
}

TEST_CASE("SyncState enum values are distinct", "[enums][phase84]") {
    CHECK(static_cast<int>(SyncState::SYNC_LOCKED)  == 0);
    CHECK(static_cast<int>(SyncState::SYNC_WARNING) == 1);
    CHECK(static_cast<int>(SyncState::SYNC_SCRAM)   == 2);
}

TEST_CASE("DecoherenceAction enum values are distinct", "[enums][phase84]") {
    CHECK(static_cast<int>(DecoherenceAction::ACCEPT)            == 0);
    CHECK(static_cast<int>(DecoherenceAction::HARD_DROP)         == 1);
    CHECK(static_cast<int>(DecoherenceAction::INTERPOLATE)       == 2);
    CHECK(static_cast<int>(DecoherenceAction::PREDICTIVE_CODE)   == 3);
    CHECK(static_cast<int>(DecoherenceAction::PROCESS)           == 4);
    CHECK(static_cast<int>(DecoherenceAction::JITTER_BUFFER)     == 5);
}

// ---------------------------------------------------------------------------
// § Physical constants
// ---------------------------------------------------------------------------

TEST_CASE("EMITTER_F_MAX_HZ is 441.0", "[constants][phase84]") {
    CHECK(EMITTER_F_MAX_HZ == Approx(441.0));
}

TEST_CASE("PHASE_EPSILON_RAD equals pi/10", "[constants][phase84]") {
    CHECK(PHASE_EPSILON_RAD == Approx(3.14159265358979323846 / 10.0).epsilon(1e-9));
}

// ---------------------------------------------------------------------------
// § Decoherence limits
// ---------------------------------------------------------------------------

TEST_CASE("Decoherence limits match spec values", "[constants][phase84]") {
    CHECK(DECOHERENCE_LIMIT_PHYSICS_US  == Approx(113.0));
    CHECK(DECOHERENCE_LIMIT_VISUAL_US   == Approx(8300.0));
    CHECK(DECOHERENCE_LIMIT_COGNITIVE_US == Approx(10000.0));
    CHECK(DECOHERENCE_LIMIT_CONTROL_US  == Approx(100000.0));
    CHECK(DECOHERENCE_LIMIT_AUDIO_US    == Approx(50000.0));
}

TEST_CASE("decoherence_limit_us returns correct values per class", "[functions][phase84]") {
    CHECK(decoherence_limit_us(MessageClass::HIGH_FREQ_PHYSICS) == Approx(113.0));
    CHECK(decoherence_limit_us(MessageClass::VISUAL_INPUT)      == Approx(8300.0));
    CHECK(decoherence_limit_us(MessageClass::COGNITIVE_STATE)   == Approx(10000.0));
    CHECK(decoherence_limit_us(MessageClass::CONTROL_ADMIN)     == Approx(100000.0));
    CHECK(decoherence_limit_us(MessageClass::SENSORY_AUDIO)     == Approx(50000.0));
}

// ---------------------------------------------------------------------------
// § is_temporally_coherent
// ---------------------------------------------------------------------------

TEST_CASE("is_temporally_coherent accepts messages within limit", "[functions][phase84]") {
    CHECK(is_temporally_coherent(0.0,    MessageClass::HIGH_FREQ_PHYSICS) == true);
    CHECK(is_temporally_coherent(100.0,  MessageClass::HIGH_FREQ_PHYSICS) == true);
    CHECK(is_temporally_coherent(112.9,  MessageClass::HIGH_FREQ_PHYSICS) == true);
    CHECK(is_temporally_coherent(8000.0, MessageClass::VISUAL_INPUT)      == true);
}

TEST_CASE("is_temporally_coherent rejects messages above limit", "[functions][phase84]") {
    CHECK(is_temporally_coherent(113.1,   MessageClass::HIGH_FREQ_PHYSICS) == false);
    CHECK(is_temporally_coherent(8300.1,  MessageClass::VISUAL_INPUT)      == false);
    CHECK(is_temporally_coherent(10001.0, MessageClass::COGNITIVE_STATE)   == false);
}

TEST_CASE("physics_coherent is a convenience alias for HIGH_FREQ_PHYSICS", "[functions][phase84]") {
    CHECK(physics_coherent(0.0)   == true);
    CHECK(physics_coherent(112.9) == true);
    CHECK(physics_coherent(113.1) == false);
}

// ---------------------------------------------------------------------------
// § PTP sync state machine
// ---------------------------------------------------------------------------

TEST_CASE("PTP thresholds match spec values", "[constants][phase84]") {
    CHECK(PTP_LOCK_THRESHOLD_US    == Approx(50.0));
    CHECK(PTP_WARNING_THRESHOLD_US == Approx(100.0));
    CHECK(PTP_SCRAM_THRESHOLD_US   == Approx(150.0));
}

TEST_CASE("ptp_sync_state classifies offsets correctly", "[functions][phase84]") {
    CHECK(ptp_sync_state(0.0)   == SyncState::SYNC_LOCKED);
    CHECK(ptp_sync_state(49.9)  == SyncState::SYNC_LOCKED);
    CHECK(ptp_sync_state(50.1)  == SyncState::SYNC_WARNING);
    CHECK(ptp_sync_state(99.9)  == SyncState::SYNC_WARNING);
    CHECK(ptp_sync_state(150.1) == SyncState::SYNC_SCRAM);
}

TEST_CASE("ptp_sync_locked is true only below LOCK threshold", "[functions][phase84]") {
    CHECK(ptp_sync_locked(49.9)  == true);
    CHECK(ptp_sync_locked(50.1)  == false);
}

// ---------------------------------------------------------------------------
// § Watchdog and heartbeat
// ---------------------------------------------------------------------------

TEST_CASE("WATCHDOG_DEADLINE_US is 2000.0", "[constants][phase84]") {
    CHECK(WATCHDOG_DEADLINE_US == Approx(2000.0));
}

TEST_CASE("HEARTBEAT_TIMEOUT_MS is 500.0", "[constants][phase84]") {
    CHECK(HEARTBEAT_TIMEOUT_MS == Approx(500.0));
}

TEST_CASE("watchdog_triggered fires above 2ms", "[functions][phase84]") {
    CHECK(watchdog_triggered(1999.9) == false);
    CHECK(watchdog_triggered(2000.1) == true);
}

TEST_CASE("heartbeat_missed fires above 500ms", "[functions][phase84]") {
    CHECK(heartbeat_missed(499.9) == false);
    CHECK(heartbeat_missed(500.1) == true);
}

// ---------------------------------------------------------------------------
// § Future-skew tolerance
// ---------------------------------------------------------------------------

TEST_CASE("FUTURE_SKEW_TOLERANCE_US is -50.0", "[constants][phase84]") {
    CHECK(FUTURE_SKEW_TOLERANCE_US == Approx(-50.0));
}
