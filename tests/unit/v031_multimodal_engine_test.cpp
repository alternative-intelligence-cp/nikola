/**
 * @file tests/unit/v031_multimodal_engine_test.cpp
 * @brief v0.3.1 — MultimodalEngine facade integration tests
 *
 * Tests:
 *   §1  Construction with default config
 *   §2  tick_audio returns 8 emitter positions
 *   §3  tick_audio_nits returns 128 nits
 *   §4  tick_audio_nits with empty PCM → zero nits
 *   §5  tick_visual returns injection coords
 *   §6  snapshot counters increment
 *   §7  Audio disabled → skip tick_audio
 *   §8  Visual disabled → skip tick_visual
 *   §9  Move semantics
 *   §10 tick_audio_nits with known sine → non-zero
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <nikola/multimodal/multimodal_engine.hpp>
#include <nikola/foundation/nit.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <vector>

using namespace nikola::multimodal;
using namespace nikola::foundation;

// ── Helper ──────────────────────────────────────────────────────────────────

static MultimodalConfig test_config() {
    MultimodalConfig cfg;
    cfg.checkpoint_dir = "/tmp/nikola_v031_test_checkpoints";
    cfg.enable_checkpoints = false;  // avoid filesystem side effects
    return cfg;
}

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
// §1 Construction
// ============================================================================

TEST_CASE("§1 MultimodalEngine constructs with default config", "[v031][multimodal_engine]") {
    MultimodalEngine engine(test_config());
    auto snap = engine.snapshot();
    REQUIRE(snap.audio_ticks == 0);
    REQUIRE(snap.visual_ticks == 0);
    REQUIRE(snap.checkpoint_count == 0);
}

// ============================================================================
// §2 tick_audio returns 8 positions
// ============================================================================

TEST_CASE("§2 tick_audio returns 8 emitter positions", "[v031][multimodal_engine]") {
    MultimodalEngine engine(test_config());
    std::vector<float> dummy_pcm(4800, 0.0f);

    auto positions = engine.tick_audio(dummy_pcm, 0);
    REQUIRE(positions.size() == NUM_EMITTERS);

    for (int i = 0; i < NUM_EMITTERS; ++i) {
        REQUIRE(positions[i].emitter_id == i);
        REQUIRE(positions[i].frequency_hz > 0.0);
    }
}

// ============================================================================
// §3 tick_audio_nits returns 128 nits
// ============================================================================

TEST_CASE("§3 tick_audio_nits returns 128 nits", "[v031][multimodal_engine]") {
    MultimodalEngine engine(test_config());
    constexpr double sr = 48000.0;
    constexpr size_t N = 48000;

    auto pcm = gen_sine(EMITTER_FREQS[0], sr, N);
    auto nits = engine.tick_audio_nits(pcm, sr);

    REQUIRE(nits.size() == AUDIO_EMBEDDING_DIM);
    for (auto n : nits) {
        REQUIRE(n >= NIT_MIN);
        REQUIRE(n <= NIT_MAX);
    }
}

// ============================================================================
// §4 tick_audio_nits empty PCM → zero nits
// ============================================================================

TEST_CASE("§4 tick_audio_nits empty PCM → zero nits", "[v031][multimodal_engine]") {
    MultimodalEngine engine(test_config());
    auto nits = engine.tick_audio_nits({});

    REQUIRE(nits.size() == AUDIO_EMBEDDING_DIM);
    for (auto n : nits) {
        REQUIRE(n == NIT_ZERO);
    }
}

// ============================================================================
// §5 tick_visual returns injection coords
// ============================================================================

TEST_CASE("§5 tick_visual returns injection coords", "[v031][multimodal_engine]") {
    MultimodalEngine engine(test_config());

    // Create a simple 8x8 grayscale image with a gradient
    constexpr int W = 8, H = 8;
    std::vector<float> image(W * H);
    for (int i = 0; i < W * H; ++i) {
        image[i] = static_cast<float>(i) / static_cast<float>(W * H);
    }

    auto injections = engine.tick_visual(image, W, H, 0);
    // Should produce some injection coordinates
    REQUIRE(!injections.empty());
}

// ============================================================================
// §6 Snapshot counters increment
// ============================================================================

TEST_CASE("§6 Snapshot counters increment correctly", "[v031][multimodal_engine]") {
    MultimodalEngine engine(test_config());
    std::vector<float> pcm(4800, 0.0f);

    engine.tick_audio(pcm, 0);
    engine.tick_audio(pcm, 1);
    auto snap = engine.snapshot();
    REQUIRE(snap.audio_ticks == 2);

    // Visual tick
    constexpr int W = 8, H = 8;
    std::vector<float> image(W * H, 0.5f);
    engine.tick_visual(image, W, H, 0);
    snap = engine.snapshot();
    REQUIRE(snap.visual_ticks == 1);
}

// ============================================================================
// §7 Audio disabled → skip
// ============================================================================

TEST_CASE("§7 Audio disabled → tick_audio returns empty", "[v031][multimodal_engine]") {
    auto cfg = test_config();
    cfg.enable_audio = false;
    MultimodalEngine engine(cfg);

    std::vector<float> pcm(4800, 0.0f);
    engine.tick_audio(pcm, 0);

    // When audio is disabled, tick counter should not increment
    auto snap = engine.snapshot();
    REQUIRE(snap.audio_ticks == 0);
}

// ============================================================================
// §8 Visual disabled → skip
// ============================================================================

TEST_CASE("§8 Visual disabled → tick_visual returns empty", "[v031][multimodal_engine]") {
    auto cfg = test_config();
    cfg.enable_visual = false;
    MultimodalEngine engine(cfg);

    constexpr int W = 8, H = 8;
    std::vector<float> image(W * H, 0.5f);
    auto injections = engine.tick_visual(image, W, H, 0);

    REQUIRE(injections.empty());
    auto snap = engine.snapshot();
    REQUIRE(snap.visual_ticks == 0);
}

// ============================================================================
// §9 Move semantics
// ============================================================================

TEST_CASE("§9 Move semantics work", "[v031][multimodal_engine]") {
    MultimodalEngine engine1(test_config());
    std::vector<float> pcm(4800, 0.0f);
    engine1.tick_audio(pcm, 0);

    // Move construct
    MultimodalEngine engine2(std::move(engine1));
    auto snap = engine2.snapshot();
    REQUIRE(snap.audio_ticks == 1);

    // Move assign
    MultimodalEngine engine3(test_config());
    engine3 = std::move(engine2);
    snap = engine3.snapshot();
    REQUIRE(snap.audio_ticks == 1);
}

// ============================================================================
// §10 tick_audio_nits with known sine → non-zero
// ============================================================================

TEST_CASE("§10 tick_audio_nits with sine → non-zero output", "[v031][multimodal_engine]") {
    MultimodalEngine engine(test_config());
    constexpr double sr = 48000.0;
    constexpr size_t N = 48000;

    // Generate sine at E4 frequency
    auto pcm = gen_sine(EMITTER_FREQS[3], sr, N);
    auto nits = engine.tick_audio_nits(pcm, sr);

    bool has_nonzero = false;
    for (auto n : nits) {
        if (n != NIT_ZERO) has_nonzero = true;
    }
    REQUIRE(has_nonzero);

    // Audio tick counter should have incremented
    auto snap = engine.snapshot();
    REQUIRE(snap.audio_ticks == 1);
}
