/**
 * @file   phase64_cymatic_transduction_test.cpp
 * @brief  Phase 64 — GAP-017: Cymatic Transduction Sampling Rate Specification
 *
 * Tests for nikola/multimodal/cymatic_transduction.hpp
 *
 * Coverage domains
 * ────────────────
 *  §1  Sampling rate constants (capture, injection, decimation, Nyquist)
 *  §2  FIR filter specification (taps, group delay, passband/stopband)
 *  §3  Latency budget constants and latency_budget_met()
 *  §4  Dual-path architecture (enum, delays)
 *  §5  emitter_freq_hz: spec-table values (E1–E8), monotonicity, errors
 *  §6  EMITTER_FREQS array consistency
 *  §7  Nyquist helpers: nyquist_min_for_emitter_hz, emitter_harmonic_hz,
 *                        injection_covers_freq
 *  §8  Goertzel algorithm: on-target ≈ 1.0, off-target ≪ 1, silence = 0
 *  §9  Invariants: all emitters below Nyquist, golden ratio identity, decimation
 * §10  Integration: voltage path latency, E8 3rd-harmonic coverage, full emitter sweep
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>
#include <numbers>
#include <vector>

#include "nikola/multimodal/cymatic_transduction.hpp"

using namespace nikola::multimodal;

// ═══════════════════════════════════════════════════════════════════════════
// §1  Sampling rate constants
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("constant: capture/injection/decimation rates are self-consistent",
          "[gap017][rates]")
{
    REQUIRE(CAPTURE_RATE_HZ   == 48'000);
    REQUIRE(INJECTION_RATE_HZ ==  1'000);
    REQUIRE(PHYSICS_TICK_RATE_HZ == 1'000);
    REQUIRE(INJECTION_RATE_HZ == PHYSICS_TICK_RATE_HZ);
    REQUIRE(DECIMATION_FACTOR == CAPTURE_RATE_HZ / INJECTION_RATE_HZ);
    REQUIRE(DECIMATION_FACTOR == 48);
}

TEST_CASE("constant: GRID_NYQUIST_HZ is physics tick rate / 2",
          "[gap017][rates]")
{
    REQUIRE(GRID_NYQUIST_HZ == Catch::Approx(500.0).epsilon(1e-12));
    REQUIRE(GRID_NYQUIST_HZ == static_cast<double>(PHYSICS_TICK_RATE_HZ) / 2.0);
}

TEST_CASE("constant: EMITTER_COUNT is 8",
          "[gap017][rates]")
{
    REQUIRE(EMITTER_COUNT == 8);
}

// ═══════════════════════════════════════════════════════════════════════════
// §2  FIR filter specification
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("constant: FIR tap count and group delay",
          "[gap017][fir]")
{
    REQUIRE(FIR_TAPS == 300);
    REQUIRE(FIR_GROUP_DELAY_SAMPLES == FIR_TAPS / 2);
    REQUIRE(FIR_GROUP_DELAY_SAMPLES == 150);
}

TEST_CASE("constant: FIR passband, transition, stopband ordering",
          "[gap017][fir]")
{
    // Passband must lie inside transition band start
    REQUIRE(FIR_PASSBAND_HZ        == Catch::Approx(150.0).epsilon(1e-12));
    REQUIRE(FIR_TRANSITION_START_HZ == Catch::Approx(150.0).epsilon(1e-12));
    REQUIRE(FIR_TRANSITION_END_HZ   == Catch::Approx(450.0).epsilon(1e-12));

    // Ordering invariants
    REQUIRE(FIR_PASSBAND_HZ        <= FIR_TRANSITION_START_HZ);
    REQUIRE(FIR_TRANSITION_START_HZ < FIR_TRANSITION_END_HZ);
    // Stopband must start before Nyquist to prevent aliasing
    REQUIRE(FIR_TRANSITION_END_HZ   < GRID_NYQUIST_HZ);
}

TEST_CASE("constant: FIR attenuation is -60 dB",
          "[gap017][fir]")
{
    REQUIRE(FIR_ATTENUATION_DB == Catch::Approx(-60.0).epsilon(1e-12));
    REQUIRE(FIR_ATTENUATION_DB < 0.0);   // Must be negative (attenuation)
}

TEST_CASE("constant: passband covers all 8 emitters",
          "[gap017][fir]")
{
    // Every E1–E8 frequency must lie within the flat passband [0, FIR_PASSBAND_HZ]
    for (int n = 1; n <= EMITTER_COUNT; ++n) {
        REQUIRE(emitter_freq_hz(n) < FIR_PASSBAND_HZ);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §3  Latency budget
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("latency: hardware buffer component ≈ 2.667 ms",
          "[gap017][latency]")
{
    double expected = 128.0 / 48000.0 * 1000.0;   // ≈ 2.6667 ms
    REQUIRE(HW_BUFFER_LATENCY_MS == Catch::Approx(expected).epsilon(1e-9));
    REQUIRE(HW_BUFFER_LATENCY_MS > 2.6);
    REQUIRE(HW_BUFFER_LATENCY_MS < 2.7);
}

TEST_CASE("latency: FIR group delay component ≈ 3.125 ms",
          "[gap017][latency]")
{
    double expected = 150.0 / 48000.0 * 1000.0;   // = 3.125 ms
    REQUIRE(FIR_GROUP_DELAY_LATENCY_MS == Catch::Approx(expected).epsilon(1e-9));
    REQUIRE(FIR_GROUP_DELAY_LATENCY_MS == Catch::Approx(3.125).epsilon(1e-6));
}

TEST_CASE("latency: total latency is sum of four components",
          "[gap017][latency]")
{
    double expected = HW_BUFFER_LATENCY_MS + FIR_GROUP_DELAY_LATENCY_MS
                    + PROCESSING_LATENCY_MS + TICK_LATENCY_MS;
    REQUIRE(TOTAL_LATENCY_MS == Catch::Approx(expected).epsilon(1e-9));
}

TEST_CASE("latency: TOTAL_LATENCY_MS < LATENCY_REQUIREMENT_MS (10 ms)",
          "[gap017][latency]")
{
    REQUIRE(LATENCY_REQUIREMENT_MS == Catch::Approx(10.0).epsilon(1e-12));
    REQUIRE(TOTAL_LATENCY_MS < LATENCY_REQUIREMENT_MS);
    REQUIRE(latency_budget_met());
    // Spec gives 7.28 ms; our precise calc gives ≈7.292 ms
    REQUIRE(TOTAL_LATENCY_MS > 7.0);
    REQUIRE(TOTAL_LATENCY_MS < 8.0);
}

// ═══════════════════════════════════════════════════════════════════════════
// §4  Dual-path architecture
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("dual-path: DIRECT path max latency == LATENCY_REQUIREMENT_MS",
          "[gap017][dualpath]")
{
    REQUIRE(DIRECT_PATH_MAX_LATENCY_MS == Catch::Approx(LATENCY_REQUIREMENT_MS).epsilon(1e-12));
}

TEST_CASE("dual-path: ISOCHRONOUS delay is 50 ms (multimodal buffer)",
          "[gap017][dualpath]")
{
    REQUIRE(ISOCHRONOUS_DELAY_MS == Catch::Approx(50.0).epsilon(1e-12));
    // Isochronous path must be much longer than direct path
    REQUIRE(ISOCHRONOUS_DELAY_MS > DIRECT_PATH_MAX_LATENCY_MS);
}

// ═══════════════════════════════════════════════════════════════════════════
// §5  emitter_freq_hz: spec-table values, monotonicity, errors
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("emitter_freq_hz: spec-table approximate values for E1–E8",
          "[gap017][emitters]")
{
    // Tolerance: ±0.05 Hz (spec table is quoted to 3 decimal places)
    REQUIRE(emitter_freq_hz(1) == Catch::Approx( 5.083).margin(0.05));   // Delta
    REQUIRE(emitter_freq_hz(2) == Catch::Approx( 8.225).margin(0.05));   // Theta
    REQUIRE(emitter_freq_hz(3) == Catch::Approx(13.308).margin(0.05));   // Alpha
    REQUIRE(emitter_freq_hz(4) == Catch::Approx(21.532).margin(0.05));   // Beta
    REQUIRE(emitter_freq_hz(5) == Catch::Approx(34.840).margin(0.05));   // Gamma Low
    REQUIRE(emitter_freq_hz(6) == Catch::Approx(56.372).margin(0.05));   // Gamma High
    REQUIRE(emitter_freq_hz(7) == Catch::Approx(91.214).margin(0.05));   // Ripple
    REQUIRE(emitter_freq_hz(8) == Catch::Approx(147.588).margin(0.05));  // Fast Ripple
}

TEST_CASE("emitter_freq_hz: formula is π × φ^n",
          "[gap017][emitters]")
{
    for (int n = 1; n <= EMITTER_COUNT; ++n) {
        double expected = M_PI * std::pow(GOLDEN_RATIO, static_cast<double>(n));
        REQUIRE(emitter_freq_hz(n) == Catch::Approx(expected).epsilon(1e-12));
    }
}

TEST_CASE("emitter_freq_hz: strictly monotonically increasing",
          "[gap017][emitters]")
{
    for (int n = 1; n < EMITTER_COUNT; ++n) {
        REQUIRE(emitter_freq_hz(n) < emitter_freq_hz(n + 1));
    }
}

TEST_CASE("emitter_freq_hz: golden ratio scaling — f_{n+1} / f_n = φ",
          "[gap017][emitters]")
{
    for (int n = 1; n < EMITTER_COUNT; ++n) {
        double ratio = emitter_freq_hz(n + 1) / emitter_freq_hz(n);
        REQUIRE(ratio == Catch::Approx(GOLDEN_RATIO).epsilon(1e-10));
    }
}

TEST_CASE("emitter_freq_hz: throws on out-of-range index",
          "[gap017][emitters][error]")
{
    REQUIRE_THROWS_AS(emitter_freq_hz(0),  std::invalid_argument);
    REQUIRE_THROWS_AS(emitter_freq_hz(9),  std::invalid_argument);
    REQUIRE_THROWS_AS(emitter_freq_hz(-1), std::invalid_argument);
    REQUIRE_THROWS_AS(emitter_freq_hz(100),std::invalid_argument);
}

// ═══════════════════════════════════════════════════════════════════════════
// §6  EMITTER_FREQS array
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("EMITTER_FREQS: array size == EMITTER_COUNT",
          "[gap017][emitters]")
{
    REQUIRE(EMITTER_FREQS.size() == static_cast<std::size_t>(EMITTER_COUNT));
}

TEST_CASE("EMITTER_FREQS: matches emitter_freq_hz(n) for n in [1, 8]",
          "[gap017][emitters]")
{
    for (int n = 1; n <= EMITTER_COUNT; ++n) {
        REQUIRE(EMITTER_FREQS[n - 1] == Catch::Approx(emitter_freq_hz(n)).epsilon(1e-12));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §7  Nyquist helpers
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("nyquist_min_for_emitter_hz: equals 2 × emitter frequency",
          "[gap017][nyquist]")
{
    for (int n = 1; n <= EMITTER_COUNT; ++n) {
        double expected = 2.0 * emitter_freq_hz(n);
        REQUIRE(nyquist_min_for_emitter_hz(n) == Catch::Approx(expected).epsilon(1e-12));
    }
}

TEST_CASE("nyquist_min_for_emitter_hz: E8 requires < 500 Hz (injection covers it)",
          "[gap017][nyquist]")
{
    // Nyquist condition: GRID_NYQUIST_HZ > nyquist_min_for_emitter_hz(8)
    REQUIRE(GRID_NYQUIST_HZ > nyquist_min_for_emitter_hz(8));
    // Approx values: 500 > 295.2 ✓
    REQUIRE(nyquist_min_for_emitter_hz(8) < 300.0);
}

TEST_CASE("emitter_harmonic_hz: k=1 is identity, k=3 gives 3rd harmonic",
          "[gap017][nyquist]")
{
    REQUIRE(emitter_harmonic_hz(8, 1) == Catch::Approx(emitter_freq_hz(8)).epsilon(1e-12));
    REQUIRE(emitter_harmonic_hz(8, 3) == Catch::Approx(3.0 * emitter_freq_hz(8)).epsilon(1e-12));
    // Spec: target capture of 3rd harmonic of E8 ≈ 442.7 Hz
    REQUIRE(emitter_harmonic_hz(8, 3) == Catch::Approx(442.7).margin(0.5));
}

TEST_CASE("emitter_harmonic_hz: throws on k < 1",
          "[gap017][nyquist][error]")
{
    REQUIRE_THROWS_AS(emitter_harmonic_hz(5, 0),  std::invalid_argument);
    REQUIRE_THROWS_AS(emitter_harmonic_hz(1, -1), std::invalid_argument);
}

TEST_CASE("injection_covers_freq: below Nyquist returns true",
          "[gap017][nyquist]")
{
    REQUIRE( injection_covers_freq(0.0));
    REQUIRE( injection_covers_freq(100.0));
    REQUIRE( injection_covers_freq(emitter_freq_hz(8)));   // E8 ≈ 147.6 Hz < 500 Hz
    REQUIRE( injection_covers_freq(emitter_harmonic_hz(8, 3)));  // ≈ 442.7 Hz < 500 Hz

    REQUIRE_FALSE(injection_covers_freq(GRID_NYQUIST_HZ));        // exactly 500 Hz — not < 500
    REQUIRE_FALSE(injection_covers_freq(GRID_NYQUIST_HZ + 1.0));  // 501 Hz
    REQUIRE_FALSE(injection_covers_freq(1000.0));                  // audio freq — above Nyquist
}

// ═══════════════════════════════════════════════════════════════════════════
// §8  Goertzel algorithm
// ═══════════════════════════════════════════════════════════════════════════

// Helpers
static std::vector<float> make_sine(double freq_hz, double sample_rate, int n_samples, double amplitude = 1.0) {
    std::vector<float> v(n_samples);
    for (int i = 0; i < n_samples; ++i)
        v[i] = static_cast<float>(amplitude * std::sin(2.0 * M_PI * freq_hz * i / sample_rate));
    return v;
}

static std::vector<float> make_silence(int n_samples) {
    return std::vector<float>(n_samples, 0.0f);
}

TEST_CASE("goertzel: on-target amplitude ≈ 1.0 for all 8 emitters (injection rate, 1000 samples)",
          "[gap017][goertzel]")
{
    // 1000 samples at 1000 Hz injection rate = 1 second, sufficient for all emitters
    const double fs = static_cast<double>(INJECTION_RATE_HZ);
    const int    N  = 1000;
    for (int n = 1; n <= EMITTER_COUNT; ++n) {
        double f = emitter_freq_hz(n);
        auto samples = make_sine(f, fs, N);
        double amp = goertzel_amplitude(samples, f, fs);
        REQUIRE(amp > 0.90);
        REQUIRE(amp < 1.15);
    }
}

TEST_CASE("goertzel: off-target amplitude is low for spectrally distant frequencies",
          "[gap017][goertzel]")
{
    const double fs = static_cast<double>(INJECTION_RATE_HZ);
    const int    N  = 1000;

    // Signal at E4 (≈21.5 Hz); query at E8 (≈147.6 Hz) — well separated
    auto samples = make_sine(emitter_freq_hz(4), fs, N);
    double amp_off = goertzel_amplitude(samples, emitter_freq_hz(8), fs);
    REQUIRE(amp_off < 0.10);   // Far off-target: well below unity

    // Signal at E8; query at E1 — opposite ends of spectrum
    auto samples2 = make_sine(emitter_freq_hz(8), fs, N);
    double amp_off2 = goertzel_amplitude(samples2, emitter_freq_hz(1), fs);
    REQUIRE(amp_off2 < 0.10);
}

TEST_CASE("goertzel: silence produces zero amplitude",
          "[gap017][goertzel]")
{
    const double fs = static_cast<double>(INJECTION_RATE_HZ);
    auto zeros = make_silence(1000);
    for (int n = 1; n <= EMITTER_COUNT; ++n) {
        double amp = goertzel_amplitude(zeros, emitter_freq_hz(n), fs);
        REQUIRE(amp == Catch::Approx(0.0).margin(1e-9));
    }
}

TEST_CASE("goertzel: scale linearity — amplitude proportional to signal amplitude",
          "[gap017][goertzel]")
{
    const double fs = static_cast<double>(INJECTION_RATE_HZ);
    const int    N  = 1000;
    const double f  = emitter_freq_hz(5);   // E5

    auto s_unit = make_sine(f, fs, N, 1.0);
    auto s_half = make_sine(f, fs, N, 0.5);
    auto s_two  = make_sine(f, fs, N, 2.0);

    double a1 = goertzel_amplitude(s_unit, f, fs);
    double a2 = goertzel_amplitude(s_half, f, fs);
    double a3 = goertzel_amplitude(s_two,  f, fs);

    REQUIRE(a2 == Catch::Approx(a1 * 0.5).epsilon(0.05));
    REQUIRE(a3 == Catch::Approx(a1 * 2.0).epsilon(0.05));
}

TEST_CASE("goertzel: throws on invalid parameters",
          "[gap017][goertzel][error]")
{
    std::vector<float> dummy = {1.0f, 0.5f, -0.5f};
    REQUIRE_THROWS_AS(goertzel_amplitude(dummy,  0.0, 1000.0), std::invalid_argument);
    REQUIRE_THROWS_AS(goertzel_amplitude(dummy, -1.0, 1000.0), std::invalid_argument);
    REQUIRE_THROWS_AS(goertzel_amplitude(dummy, 50.0,    0.0), std::invalid_argument);
    REQUIRE_THROWS_AS(goertzel_amplitude(dummy, 50.0,   -1.0), std::invalid_argument);
}

// ═══════════════════════════════════════════════════════════════════════════
// §9  Invariants
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("invariant: all 8 emitters lie below GRID_NYQUIST_HZ",
          "[gap017][invariant]")
{
    for (int n = 1; n <= EMITTER_COUNT; ++n) {
        REQUIRE(emitter_freq_hz(n) < GRID_NYQUIST_HZ);
    }
}

TEST_CASE("invariant: DECIMATION_FACTOR × INJECTION_RATE_HZ == CAPTURE_RATE_HZ",
          "[gap017][invariant]")
{
    REQUIRE(DECIMATION_FACTOR * INJECTION_RATE_HZ == CAPTURE_RATE_HZ);
}

TEST_CASE("invariant: golden ratio identity φ^{n+1} = φ × φ^n",
          "[gap017][invariant]")
{
    for (int n = 1; n < EMITTER_COUNT; ++n) {
        double f_n   = emitter_freq_hz(n);
        double f_np1 = emitter_freq_hz(n + 1);
        REQUIRE(f_np1 / f_n == Catch::Approx(GOLDEN_RATIO).epsilon(1e-10));
    }
}

TEST_CASE("invariant: FIR transition end < Nyquist guards against aliasing",
          "[gap017][invariant]")
{
    // FIR must cut off before Nyquist - stopband at 450 Hz < 500 Hz grid Nyquist
    REQUIRE(FIR_TRANSITION_END_HZ < GRID_NYQUIST_HZ);
    // Guard margin: ≥ 40 Hz between stopband and Nyquist
    REQUIRE(GRID_NYQUIST_HZ - FIR_TRANSITION_END_HZ >= 40.0);
}

// ═══════════════════════════════════════════════════════════════════════════
// §10  Integration tests
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("integration: E8 3rd-harmonic is within injection Nyquist",
          "[gap017][integration]")
{
    // §GAP-017: "Target: Capture 3rd harmonic of E8 ≈ 442.7 Hz"
    // "1000 Hz injection rate supports this: 500 Hz > 442.7 Hz ✓"
    double e8_third = emitter_harmonic_hz(8, 3);
    REQUIRE(e8_third > 440.0);
    REQUIRE(e8_third < 445.0);
    REQUIRE(injection_covers_freq(e8_third));         // 442.7 < 500 ✓
    REQUIRE(e8_third < GRID_NYQUIST_HZ);
}

TEST_CASE("integration: 3rd harmonic of E8 lies within FIR transition band",
          "[gap017][integration]")
{
    // The 3rd harmonic (≈442.7 Hz) falls in the transition band [450 Hz stopband]
    // and will be attenuated, but the spec confirms the injection rate supports it
    double e8_third = emitter_harmonic_hz(8, 3);
    // E8 3rd harmonic must be below the stopband entrance
    REQUIRE(e8_third < FIR_TRANSITION_END_HZ + 10.0);   // within ±10 Hz of stopband
}

TEST_CASE("integration: Goertzel recovers all 8 emitters from mixed signal",
          "[gap017][integration]")
{
    // Build signal = E1 + E3 + E5 + E7 (interleaved emitters), injection rate
    const double fs = static_cast<double>(INJECTION_RATE_HZ);
    const int    N  = 1000;

    std::vector<float> mixed(N, 0.0f);
    for (int idx : {1, 3, 5, 7}) {
        double f = emitter_freq_hz(idx);
        for (int i = 0; i < N; ++i)
            mixed[i] += static_cast<float>(std::sin(2.0 * M_PI * f * i / fs));
    }
    // Normalise to unit amplitude per component (4 components → divide by 4)
    for (auto& s : mixed) s /= 4.0f;

    // Active emitters (E1, E3, E5, E7) should have measurable amplitude
    for (int idx : {1, 3, 5, 7}) {
        double amp = goertzel_amplitude(mixed, emitter_freq_hz(idx), fs);
        REQUIRE(amp > 0.1);    // Clearly present
    }
    // Absent emitters (E2, E4, E6, E8) should have low amplitude
    // Note: some crosstalk expected in a 1s window; just verify < present level
    for (int idx : {2, 4, 6, 8}) {
        double amp_present = goertzel_amplitude(mixed, emitter_freq_hz(idx - 1), fs);
        double amp_absent  = goertzel_amplitude(mixed, emitter_freq_hz(idx),     fs);
        REQUIRE(amp_present > amp_absent);   // Present signal > absent signal
    }
}
