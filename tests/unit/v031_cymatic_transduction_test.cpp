/**
 * @file tests/unit/v031_cymatic_transduction_test.cpp
 * @brief v0.3.1 — CymaticTransduction constants, Goertzel, emitter frequencies
 *
 * Tests:
 *   §1  Golden Ratio constant accuracy
 *   §2  Emitter frequencies match π·φⁿ formula
 *   §3  Emitter frequency monotonically increasing
 *   §4  E8 3rd harmonic within Nyquist
 *   §5  Sampling constants consistency
 *   §6  FIR filter specification
 *   §7  Latency budget met (<10ms)
 *   §8  Goertzel detects target frequency
 *   §9  Goertzel silent input → zero
 *   §10 Goertzel rejects off-frequency
 *   §11 Cognitive band labels
 *   §12 Prime phase offsets (8 primes, descending)
 *   §13 Synchronizer e₉ frequency
 *   §14 Nyquist helpers
 *   §15 EMITTER_FREQS array matches emitter_freq_hz()
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <nikola/multimodal/cymatic_transduction.hpp>

#include <cmath>
#include <vector>

using namespace nikola::multimodal;

// ── Helper ──────────────────────────────────────────────────────────────────

/// Generate a sine wave at a given frequency.
static std::vector<float> gen_sine(double freq_hz, double sample_rate,
                                    size_t n_samples, double amplitude = 1.0)
{
    std::vector<float> pcm(n_samples);
    for (size_t i = 0; i < n_samples; ++i) {
        double t = static_cast<double>(i) / sample_rate;
        pcm[i] = static_cast<float>(amplitude * std::sin(2.0 * M_PI * freq_hz * t));
    }
    return pcm;
}

// ============================================================================
// §1 Golden Ratio
// ============================================================================

TEST_CASE("§1 Golden Ratio constant", "[v031][cymatic]") {
    REQUIRE_THAT(GOLDEN_RATIO, Catch::Matchers::WithinRel(1.6180339887498948, 1e-12));
}

// ============================================================================
// §2 Emitter frequencies match formula
// ============================================================================

TEST_CASE("§2 Emitter frequencies match π·φⁿ", "[v031][cymatic]") {
    for (int n = 1; n <= EMITTER_COUNT; ++n) {
        double expected = M_PI * std::pow(GOLDEN_RATIO, static_cast<double>(n));
        REQUIRE_THAT(emitter_freq_hz(n), Catch::Matchers::WithinRel(expected, 1e-12));
    }
}

// ============================================================================
// §3 Frequencies monotonically increasing
// ============================================================================

TEST_CASE("§3 Emitter frequencies ascending", "[v031][cymatic]") {
    for (int n = 2; n <= EMITTER_COUNT; ++n) {
        REQUIRE(emitter_freq_hz(n) > emitter_freq_hz(n - 1));
    }
}

// ============================================================================
// §4 E8 3rd harmonic within Nyquist
// ============================================================================

TEST_CASE("§4 E8 3rd harmonic < Grid Nyquist", "[v031][cymatic]") {
    double e8_3rd = emitter_harmonic_hz(8, 3);
    REQUIRE(e8_3rd < GRID_NYQUIST_HZ);
}

// ============================================================================
// §5 Sampling constants
// ============================================================================

TEST_CASE("§5 Sampling constants consistency", "[v031][cymatic]") {
    REQUIRE(CAPTURE_RATE_HZ == 48000);
    REQUIRE(INJECTION_RATE_HZ == 1000);
    REQUIRE(DECIMATION_FACTOR == 48);
    REQUIRE(DECIMATION_FACTOR == CAPTURE_RATE_HZ / INJECTION_RATE_HZ);
    REQUIRE(GRID_NYQUIST_HZ == 500.0);
}

// ============================================================================
// §6 FIR specification
// ============================================================================

TEST_CASE("§6 FIR filter specification", "[v031][cymatic]") {
    REQUIRE(FIR_TAPS == 300);
    REQUIRE(FIR_GROUP_DELAY_SAMPLES == 150);
    REQUIRE(FIR_PASSBAND_HZ == 150.0);
    REQUIRE(FIR_TRANSITION_END_HZ == 450.0);
    REQUIRE(FIR_ATTENUATION_DB == -60.0);
}

// ============================================================================
// §7 Latency budget
// ============================================================================

TEST_CASE("§7 Latency budget met", "[v031][cymatic]") {
    REQUIRE(latency_budget_met());
    REQUIRE(TOTAL_LATENCY_MS < LATENCY_REQUIREMENT_MS);
    REQUIRE(TOTAL_LATENCY_MS < 10.0);
}

// ============================================================================
// §8 Goertzel detects target frequency
// ============================================================================

TEST_CASE("§8 Goertzel detects target frequency", "[v031][cymatic]") {
    constexpr double sr = 48000.0;
    constexpr size_t N = 48000;  // 1 second of audio

    // Generate sine at E4 frequency (~21.53 Hz)
    double target = EMITTER_FREQS[3];  // E4
    auto pcm = gen_sine(target, sr, N);

    double amp = goertzel_amplitude(pcm, target, sr);
    // Should detect amplitude close to 1.0 for a unit-amplitude sine
    REQUIRE(amp > 0.8);
}

// ============================================================================
// §9 Goertzel silent input
// ============================================================================

TEST_CASE("§9 Goertzel silent input → zero", "[v031][cymatic]") {
    std::vector<float> silence(4800, 0.0f);
    double amp = goertzel_amplitude(silence, 100.0, 48000.0);
    REQUIRE_THAT(amp, Catch::Matchers::WithinAbs(0.0, 1e-10));
}

// ============================================================================
// §10 Goertzel rejects off-frequency
// ============================================================================

TEST_CASE("§10 Goertzel rejects off-frequency", "[v031][cymatic]") {
    constexpr double sr = 48000.0;
    constexpr size_t N = 48000;

    // Generate sine at 100 Hz, probe at 200 Hz
    auto pcm = gen_sine(100.0, sr, N);
    double amp = goertzel_amplitude(pcm, 200.0, sr);
    REQUIRE(amp < 0.1);  // Should be near zero
}

// ============================================================================
// §11 Cognitive band labels
// ============================================================================

TEST_CASE("§11 Cognitive band labels populated", "[v031][cymatic]") {
    REQUIRE(EMITTER_LABELS[1] == "Delta (Metacognitive Timing)");
    REQUIRE(EMITTER_LABELS[8] == "Fast Ripple (Error Correction)");
    REQUIRE(EMITTER_LABELS[0].empty());  // index 0 unused
}

// ============================================================================
// §12 Prime phase offsets
// ============================================================================

TEST_CASE("§12 Prime phase offsets descending", "[v031][cymatic]") {
    // Values should be: 23, 19, 17, 13, 11, 7, 5, 3 (descending primes)
    REQUIRE(PRIME_PHASE_OFFSETS_DEG[0] == 23.0);
    REQUIRE(PRIME_PHASE_OFFSETS_DEG[7] == 3.0);
    for (int i = 1; i < EMITTER_COUNT; ++i) {
        REQUIRE(PRIME_PHASE_OFFSETS_DEG[i] < PRIME_PHASE_OFFSETS_DEG[i - 1]);
    }
}

// ============================================================================
// §13 Synchronizer e₉
// ============================================================================

TEST_CASE("§13 Synchronizer e₉ frequency", "[v031][cymatic]") {
    // e₉ = π × (1/φ) × √2 × (32/27) ≈ 3.254 Hz
    double expected = M_PI * (1.0 / GOLDEN_RATIO) * std::sqrt(2.0) * (32.0 / 27.0);
    REQUIRE_THAT(SYNCHRONIZER_FREQ_HZ, Catch::Matchers::WithinRel(expected, 1e-10));
    REQUIRE(SYNCHRONIZER_FREQ_HZ > 3.0);
    REQUIRE(SYNCHRONIZER_FREQ_HZ < 4.0);
}

// ============================================================================
// §14 Nyquist helpers
// ============================================================================

TEST_CASE("§14 Nyquist helpers", "[v031][cymatic]") {
    // Nyquist min for E1 should be ~10.17 Hz
    REQUIRE(nyquist_min_for_emitter_hz(1) > 10.0);
    REQUIRE(nyquist_min_for_emitter_hz(1) < 11.0);

    // All emitter fundamentals should be covered by injection rate
    for (int n = 1; n <= EMITTER_COUNT; ++n) {
        REQUIRE(injection_covers_freq(emitter_freq_hz(n)));
    }
}

// ============================================================================
// §15 EMITTER_FREQS array consistency
// ============================================================================

TEST_CASE("§15 EMITTER_FREQS matches emitter_freq_hz()", "[v031][cymatic]") {
    for (int i = 0; i < EMITTER_COUNT; ++i) {
        REQUIRE_THAT(EMITTER_FREQS[i],
            Catch::Matchers::WithinRel(emitter_freq_hz(i + 1), 1e-12));
    }
}
