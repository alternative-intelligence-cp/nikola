/**
 * @file tests/unit/audio_input_test.cpp
 * @brief Unit tests for AudioInput: PCM → Goertzel → Nit[128] embedding.
 *
 * Covers:
 *   §A — Band extraction: Goertzel accuracy on pure sine tones
 *   §B — Nit embedding: dimension, range, phase coding, silence
 *   §C — Full pipeline: process() end-to-end
 *   §D — Edge cases: empty input, DC offset, high frequency
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/multimodal/audio_input.hpp>
#include <nikola/multimodal/cymatic_transduction.hpp>
#include <nikola/foundation/nit.hpp>

#include <array>
#include <cmath>
#include <numbers>
#include <vector>

using namespace nikola::multimodal;
using namespace nikola::foundation;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a pure sine wave at the given frequency.
static std::vector<float> generate_sine(double freq_hz, double sample_rate,
                                         int num_samples, float amplitude = 1.0f) {
    std::vector<float> pcm(static_cast<size_t>(num_samples));
    for (int i = 0; i < num_samples; ++i) {
        double t = static_cast<double>(i) / sample_rate;
        pcm[static_cast<size_t>(i)] =
            amplitude * static_cast<float>(std::sin(2.0 * std::numbers::pi * freq_hz * t));
    }
    return pcm;
}

/// Generate silence (all zeros).
static std::vector<float> generate_silence(int num_samples) {
    return std::vector<float>(static_cast<size_t>(num_samples), 0.0f);
}

// ─────────────────────────────────────────────────────────────────────────────
// §A — Band extraction
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("§A-1 AudioInput — extract_bands returns 8 amplitudes", "[audio_input][bandA]") {
    auto pcm = generate_sine(EMITTER_FREQS[0], CAPTURE_RATE_HZ, 4800);
    auto bands = AudioInput::extract_bands(pcm);
    REQUIRE(bands.size() == static_cast<size_t>(EMITTER_COUNT));
    for (auto a : bands) {
        CHECK(std::isfinite(a));
        CHECK(a >= 0.0);
    }
}

TEST_CASE("§A-2 AudioInput — E1 sine detected in band 0", "[audio_input][bandA]") {
    // Generate sine at E1 frequency (~5.08 Hz), long enough for detection
    double freq = EMITTER_FREQS[0];  // ~5.08 Hz
    int samples = static_cast<int>(CAPTURE_RATE_HZ * 2);  // 2 seconds
    auto pcm = generate_sine(freq, CAPTURE_RATE_HZ, samples);
    auto bands = AudioInput::extract_bands(pcm);

    // Band 0 (E1) should have the largest amplitude
    double max_amp = *std::max_element(bands.begin(), bands.end());
    CHECK(bands[0] > 0.0);
    CHECK(bands[0] == max_amp);
}

TEST_CASE("§A-3 AudioInput — E4 sine peaks in band 3", "[audio_input][bandA]") {
    // E4 = Beta ~21.53 Hz
    double freq = EMITTER_FREQS[3];
    int samples = static_cast<int>(CAPTURE_RATE_HZ * 2);
    auto pcm = generate_sine(freq, CAPTURE_RATE_HZ, samples);
    auto bands = AudioInput::extract_bands(pcm);

    // Band 3 should have the highest amplitude
    size_t max_idx = 0;
    for (size_t i = 1; i < bands.size(); ++i) {
        if (bands[i] > bands[max_idx]) max_idx = i;
    }
    CHECK(max_idx == 3);
}

TEST_CASE("§A-4 AudioInput — silence yields all-zero bands", "[audio_input][bandA]") {
    auto pcm = generate_silence(4800);
    auto bands = AudioInput::extract_bands(pcm);
    for (auto a : bands) {
        CHECK(a == Catch::Approx(0.0).margin(1e-12));
    }
}

TEST_CASE("§A-5 AudioInput — E8 sine detected in band 7", "[audio_input][bandA]") {
    // E8 = FastRipple ~147.58 Hz
    double freq = EMITTER_FREQS[7];
    int samples = static_cast<int>(CAPTURE_RATE_HZ * 2);
    auto pcm = generate_sine(freq, CAPTURE_RATE_HZ, samples);
    auto bands = AudioInput::extract_bands(pcm);

    size_t max_idx = 0;
    for (size_t i = 1; i < bands.size(); ++i) {
        if (bands[i] > bands[max_idx]) max_idx = i;
    }
    CHECK(max_idx == 7);
}

// ─────────────────────────────────────────────────────────────────────────────
// §B — Nit embedding
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("§B-1 AudioInput — embed produces 128 nits", "[audio_input][bandB]") {
    AudioInput::BandAmplitudes bands{};
    bands.fill(0.5);
    auto nits = AudioInput::embed(bands);
    REQUIRE(nits.size() == 128);
}

TEST_CASE("§B-2 AudioInput — all nits within [-4, +4]", "[audio_input][bandB]") {
    // Max amplitude in all bands
    AudioInput::BandAmplitudes bands{};
    bands.fill(1.0);
    auto nits = AudioInput::embed(bands);
    for (auto n : nits) {
        CHECK(n >= NIT_MIN);
        CHECK(n <= NIT_MAX);
    }
}

TEST_CASE("§B-3 AudioInput — silence embeds to all zeros", "[audio_input][bandB]") {
    AudioInput::BandAmplitudes bands{};
    bands.fill(0.0);
    auto nits = AudioInput::embed(bands);
    for (auto n : nits) {
        CHECK(n == NIT_ZERO);
    }
}

TEST_CASE("§B-4 AudioInput — single band non-zero populates correct subspace",
          "[audio_input][bandB]") {
    // Only band 2 (Alpha) active
    AudioInput::BandAmplitudes bands{};
    bands.fill(0.0);
    bands[2] = 0.8;
    auto nits = AudioInput::embed(bands);

    // Bands 0, 1, 3-7 should be all zeros
    for (int k = 0; k < EMITTER_COUNT; ++k) {
        if (k == 2) continue;
        bool all_zero = true;
        for (int j = 0; j < NITS_PER_BAND; ++j) {
            if (nits[static_cast<size_t>(k * NITS_PER_BAND + j)] != 0) {
                all_zero = false;
                break;
            }
        }
        CHECK(all_zero);
    }

    // Band 2 should have some non-zero nits
    bool has_nonzero = false;
    for (int j = 0; j < NITS_PER_BAND; ++j) {
        if (nits[static_cast<size_t>(2 * NITS_PER_BAND + j)] != 0)
            has_nonzero = true;
    }
    CHECK(has_nonzero);
}

TEST_CASE("§B-5 AudioInput — amplitude > 1.0 clamped safely", "[audio_input][bandB]") {
    AudioInput::BandAmplitudes bands{};
    bands.fill(5.0);  // Way over 1.0
    auto nits = AudioInput::embed(bands);
    for (auto n : nits) {
        CHECK(n >= NIT_MIN);
        CHECK(n <= NIT_MAX);
    }
}

TEST_CASE("§B-6 AudioInput — different bands produce different embeddings",
          "[audio_input][bandB]") {
    AudioInput::BandAmplitudes b1{}, b2{};
    b1.fill(0.0); b1[0] = 1.0;
    b2.fill(0.0); b2[4] = 1.0;
    auto n1 = AudioInput::embed(b1);
    auto n2 = AudioInput::embed(b2);
    // The two embeddings should differ (different subspaces)
    CHECK(n1 != n2);
}

// ─────────────────────────────────────────────────────────────────────────────
// §C — Full pipeline
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("§C-1 AudioInput — process() end-to-end sine → non-trivial embedding",
          "[audio_input][bandC]") {
    double freq = EMITTER_FREQS[3];  // E4 Beta
    int samples = static_cast<int>(CAPTURE_RATE_HZ * 2);
    auto pcm = generate_sine(freq, CAPTURE_RATE_HZ, samples);
    auto nits = AudioInput::process(pcm);

    REQUIRE(nits.size() == 128);
    // Should have at least some non-zero nits (sine was detected)
    int nonzero = 0;
    for (auto n : nits) nonzero += (n != 0) ? 1 : 0;
    CHECK(nonzero > 0);
}

TEST_CASE("§C-2 AudioInput — process() silence → all zeros", "[audio_input][bandC]") {
    auto pcm = generate_silence(4800);
    auto nits = AudioInput::process(pcm);
    REQUIRE(nits.size() == 128);
    for (auto n : nits) {
        CHECK(n == NIT_ZERO);
    }
}

TEST_CASE("§C-3 AudioInput — process() custom sample rate", "[audio_input][bandC]") {
    // 1 kHz sample rate — only covers up to 500 Hz Nyquist, should still work
    double freq = EMITTER_FREQS[3];  // ~21.53 Hz, within 500 Hz Nyquist
    auto pcm = generate_sine(freq, 1000.0, 2000);
    auto nits = AudioInput::process(pcm, 1000.0);
    REQUIRE(nits.size() == 128);
    for (auto n : nits) {
        CHECK(n >= NIT_MIN);
        CHECK(n <= NIT_MAX);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §D — Edge cases
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("§D-1 AudioInput — empty PCM → 128 zeros", "[audio_input][bandD]") {
    std::span<const float> empty{};
    auto nits = AudioInput::process(empty);
    REQUIRE(nits.size() == 128);
    for (auto n : nits) {
        CHECK(n == NIT_ZERO);
    }
}

TEST_CASE("§D-2 AudioInput — DC offset (no oscillation) → near-zero embedding",
          "[audio_input][bandD]") {
    // Constant 0.5 — no oscillation at any cognitive band
    std::vector<float> pcm(4800, 0.5f);
    auto nits = AudioInput::process(pcm);
    REQUIRE(nits.size() == 128);
    // DC shouldn't excite any cognitive band (all > 5 Hz)
    // Some small leakage possible, but most nits should be zero
    int nonzero = 0;
    for (auto n : nits) nonzero += (n != 0) ? 1 : 0;
    CHECK(nonzero < 48);  // At most ~37% leakage from spectral sidelobes
}

TEST_CASE("§D-3 AudioInput — very short buffer (10 samples) doesn't crash",
          "[audio_input][bandD]") {
    auto pcm = generate_sine(100.0, CAPTURE_RATE_HZ, 10);
    auto nits = AudioInput::process(pcm);
    REQUIRE(nits.size() == 128);
    for (auto n : nits) {
        CHECK(n >= NIT_MIN);
        CHECK(n <= NIT_MAX);
    }
}
