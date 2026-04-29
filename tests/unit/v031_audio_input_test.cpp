/**
 * @file tests/unit/v031_audio_input_test.cpp
 * @brief v0.3.1 — AudioInput PCM → Nit[128] embedding tests
 *
 * Tests:
 *   §1  Empty PCM → all zeros
 *   §2  Silent PCM → all zeros
 *   §3  extract_bands with single emitter frequency
 *   §4  embed() output range [-4, +4]
 *   §5  embed() output dimension = 128
 *   §6  embed() all-zero bands → all-zero nits
 *   §7  embed() high amplitude → nits reach extremes
 *   §8  process() end-to-end with known sine
 *   §9  Band isolation — different freqs excite different bands
 *   §10 NITS_PER_BAND and AUDIO_EMBEDDING_DIM constants
 *   §11 Amplitude clamping — values > 1.0 clamped
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <nikola/multimodal/audio_input.hpp>
#include <nikola/foundation/nit.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

using namespace nikola::multimodal;
using namespace nikola::foundation;

// ── Helper ──────────────────────────────────────────────────────────────────

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
// §1 Empty PCM → all zeros
// ============================================================================

TEST_CASE("§1 Empty PCM → zero nits", "[v031][audio_input]") {
    auto nits = AudioInput::process({});
    REQUIRE(nits.size() == AUDIO_EMBEDDING_DIM);
    for (auto n : nits) {
        REQUIRE(n == NIT_ZERO);
    }
}

// ============================================================================
// §2 Silent PCM → all zeros
// ============================================================================

TEST_CASE("§2 Silent PCM → zero nits", "[v031][audio_input]") {
    std::vector<float> silence(4800, 0.0f);
    auto nits = AudioInput::process(silence);
    REQUIRE(nits.size() == AUDIO_EMBEDDING_DIM);
    for (auto n : nits) {
        REQUIRE(n == NIT_ZERO);
    }
}

// ============================================================================
// §3 extract_bands with known frequency
// ============================================================================

TEST_CASE("§3 extract_bands detects emitter frequency", "[v031][audio_input]") {
    constexpr double sr = 48000.0;
    constexpr size_t N = 48000;  // 1 second

    // Generate sine at E4 frequency (~21.53 Hz)
    double target = EMITTER_FREQS[3];
    auto pcm = gen_sine(target, sr, N);

    auto bands = AudioInput::extract_bands(pcm, sr);

    // Band 3 (E4) should have significant amplitude
    REQUIRE(bands[3] > 0.5);

    // Other bands should be much lower
    for (int i = 0; i < EMITTER_COUNT; ++i) {
        if (i != 3) {
            REQUIRE(bands[i] < bands[3]);
        }
    }
}

// ============================================================================
// §4 embed() output range
// ============================================================================

TEST_CASE("§4 embed() output within [-4, +4]", "[v031][audio_input]") {
    AudioInput::BandAmplitudes bands = {0.5, 0.8, 0.2, 1.0, 0.1, 0.9, 0.3, 0.7};
    auto nits = AudioInput::embed(bands);

    for (auto n : nits) {
        REQUIRE(n >= NIT_MIN);
        REQUIRE(n <= NIT_MAX);
    }
}

// ============================================================================
// §5 embed() dimension = 128
// ============================================================================

TEST_CASE("§5 embed() returns 128 nits", "[v031][audio_input]") {
    AudioInput::BandAmplitudes bands = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
    auto nits = AudioInput::embed(bands);
    REQUIRE(nits.size() == 128);
}

// ============================================================================
// §6 Zero bands → zero nits
// ============================================================================

TEST_CASE("§6 embed() zero bands → zero nits", "[v031][audio_input]") {
    AudioInput::BandAmplitudes bands = {0, 0, 0, 0, 0, 0, 0, 0};
    auto nits = AudioInput::embed(bands);
    for (auto n : nits) {
        REQUIRE(n == NIT_ZERO);
    }
}

// ============================================================================
// §7 High amplitude → nits reach extremes
// ============================================================================

TEST_CASE("§7 embed() high amplitude → nits reach ±4", "[v031][audio_input]") {
    AudioInput::BandAmplitudes bands = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    auto nits = AudioInput::embed(bands);

    bool has_max = false, has_min = false;
    for (auto n : nits) {
        if (n == NIT_MAX) has_max = true;
        if (n == NIT_MIN) has_min = true;
    }
    REQUIRE(has_max);
    REQUIRE(has_min);
}

// ============================================================================
// §8 process() end-to-end
// ============================================================================

TEST_CASE("§8 process() end-to-end with sine", "[v031][audio_input]") {
    constexpr double sr = 48000.0;
    constexpr size_t N = 48000;

    // Generate sine at E1 frequency (~5.08 Hz)
    double target = EMITTER_FREQS[0];
    auto pcm = gen_sine(target, sr, N);

    auto nits = AudioInput::process(pcm, sr);
    REQUIRE(nits.size() == AUDIO_EMBEDDING_DIM);

    // First 16 nits (band 0) should have some non-zero values
    bool band0_active = false;
    for (int j = 0; j < NITS_PER_BAND; ++j) {
        if (nits[j] != NIT_ZERO) band0_active = true;
    }
    REQUIRE(band0_active);
}

// ============================================================================
// §9 Band isolation
// ============================================================================

TEST_CASE("§9 Different frequencies excite different bands", "[v031][audio_input]") {
    constexpr double sr = 48000.0;
    constexpr size_t N = 48000;

    // E1 sine → band 0 active
    auto pcm1 = gen_sine(EMITTER_FREQS[0], sr, N);
    auto bands1 = AudioInput::extract_bands(pcm1, sr);

    // E8 sine → band 7 active
    auto pcm8 = gen_sine(EMITTER_FREQS[7], sr, N);
    auto bands8 = AudioInput::extract_bands(pcm8, sr);

    // Each should peak at its own band
    REQUIRE(bands1[0] > bands1[7]);
    REQUIRE(bands8[7] > bands8[0]);
}

// ============================================================================
// §10 Constants
// ============================================================================

TEST_CASE("§10 NITS_PER_BAND and AUDIO_EMBEDDING_DIM", "[v031][audio_input]") {
    REQUIRE(NITS_PER_BAND == 16);
    REQUIRE(AUDIO_EMBEDDING_DIM == 128);
    REQUIRE(AUDIO_EMBEDDING_DIM == EMITTER_COUNT * NITS_PER_BAND);
}

// ============================================================================
// §11 Amplitude clamping > 1.0
// ============================================================================

TEST_CASE("§11 embed() clamps amplitudes > 1.0", "[v031][audio_input]") {
    AudioInput::BandAmplitudes normal = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    AudioInput::BandAmplitudes clipped = {5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0};

    auto nits_normal = AudioInput::embed(normal);
    auto nits_clipped = AudioInput::embed(clipped);

    // Both should produce identical output since 5.0 clamps to 1.0
    REQUIRE(nits_normal == nits_clipped);
}
