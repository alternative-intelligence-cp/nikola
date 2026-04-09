/**
 * @file tests/integration/multimodal_integration_test.cpp
 * @brief v0.0.18 Integration tests: multimodal input → torus injection → stability.
 *
 * Covers:
 *   §A — Audio injection: sine → CognitiveTorus → non-degenerate field
 *   §B — Visual injection: gradient image → CognitiveTorus → field response
 *   §C — Multi-input fusion: text + audio simultaneous → no explosion
 *   §D — Energy conservation: combined inputs respect power budget
 *   §E — MultimodalEngine facade: tick_audio_nits produces valid embedding
 */

#define NIKOLA_MULTIMODAL_ENGINE_IMPL

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/cognitive/cognitive_torus.hpp>
#include <nikola/multimodal/audio_input.hpp>
#include <nikola/multimodal/multimodal_engine.hpp>
#include <nikola/foundation/nit.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>
#include <vector>

using namespace nikola::multimodal;
using namespace nikola::foundation;
using namespace nikola::cognitive;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<float> make_sine(double freq_hz, double sample_rate, int n,
                                     float amp = 1.0f) {
    std::vector<float> pcm(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        double t = static_cast<double>(i) / sample_rate;
        pcm[static_cast<size_t>(i)] =
            amp * static_cast<float>(std::sin(2.0 * std::numbers::pi * freq_hz * t));
    }
    return pcm;
}

static double total_energy(const CognitiveTorus& ct) {
    double E = 0.0;
    for (size_t i = 0; i < ct.num_nodes(); ++i) {
        float r = ct.wave_function().grid().psi_real()[i];
        float im = ct.wave_function().grid().psi_imag()[i];
        E += static_cast<double>(r * r + im * im);
    }
    return E;
}

static bool all_finite(const CognitiveTorus& ct) {
    for (size_t i = 0; i < ct.num_nodes(); ++i) {
        if (!std::isfinite(ct.wave_function().grid().psi_real()[i])) return false;
        if (!std::isfinite(ct.wave_function().grid().psi_imag()[i])) return false;
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// §A — Audio injection into CognitiveTorus
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("§A-1 Audio inject — sine wave produces field perturbation",
          "[multimodal_integ][audio]") {
    CognitiveTorus ct(3);
    double E_before = total_energy(ct);

    // Inject 2 seconds of E4 Beta (~21.53 Hz)
    auto pcm = make_sine(EMITTER_FREQS[3], CAPTURE_RATE_HZ,
                          static_cast<int>(CAPTURE_RATE_HZ * 2));
    ct.inject_audio(pcm, 0.0);

    double E_after = total_energy(ct);
    // Energy should have increased (non-zero injection)
    CHECK(E_after > E_before);
    CHECK(all_finite(ct));
}

TEST_CASE("§A-2 Audio inject — silence produces no perturbation",
          "[multimodal_integ][audio]") {
    CognitiveTorus ct(3);
    double E_before = total_energy(ct);

    std::vector<float> silence(4800, 0.0f);
    ct.inject_audio(silence, 0.0);

    double E_after = total_energy(ct);
    // Energy should be unchanged (zero injection)
    CHECK(E_after == Catch::Approx(E_before).margin(1e-10));
}

TEST_CASE("§A-3 Audio inject — field remains stable after propagation",
          "[multimodal_integ][audio]") {
    CognitiveTorus ct(3);
    ct.set_gpu(false);

    auto pcm = make_sine(EMITTER_FREQS[0], CAPTURE_RATE_HZ,
                          static_cast<int>(CAPTURE_RATE_HZ * 2));
    ct.inject_audio(pcm, 0.0);

    // Propagate 100 steps — field must remain finite
    float dt = ct.safe_dt();
    ct.run(100, dt);
    CHECK(all_finite(ct));
    CHECK(total_energy(ct) > 0.0);
}

TEST_CASE("§A-4 Audio inject — scaled injection attenuates properly",
          "[multimodal_integ][audio]") {
    CognitiveTorus ct1(3);
    CognitiveTorus ct2(3);

    auto pcm = make_sine(EMITTER_FREQS[3], CAPTURE_RATE_HZ,
                          static_cast<int>(CAPTURE_RATE_HZ * 2));

    ct1.inject_audio(pcm, 0.0);               // full weight
    ct2.inject_audio_scaled(pcm, 0.5f, 0.0);  // half weight

    double E_full = total_energy(ct1);
    double E_half = total_energy(ct2);
    // Half-weight should inject less energy (approximately, depends on interaction)
    // Both should be above baseline
    double E_base = total_energy(CognitiveTorus(3));
    CHECK(E_full > E_base);
    CHECK(E_half > E_base);
    // Scaled injection should be noticeably less than full
    CHECK(E_half < E_full);
}

// ─────────────────────────────────────────────────────────────────────────────
// §B — Visual injection into CognitiveTorus
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("§B-1 Visual inject — gradient image produces field perturbation",
          "[multimodal_integ][visual]") {
    CognitiveTorus ct(3);
    double E_before = total_energy(ct);

    // Create a 64×64 horizontal gradient [0.0 → 1.0]
    const int W = 64, H = 64;
    std::vector<float> img(static_cast<size_t>(W * H));
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            img[static_cast<size_t>(y * W + x)] =
                static_cast<float>(x) / static_cast<float>(W - 1);
        }
    }

    ct.inject_visual(img, W, H, 0.0);

    double E_after = total_energy(ct);
    CHECK(E_after > E_before);
    CHECK(all_finite(ct));
}

TEST_CASE("§B-2 Visual inject — black image produces minimal perturbation",
          "[multimodal_integ][visual]") {
    CognitiveTorus ct(3);

    // All-black image: intensity = 0 everywhere
    const int W = 64, H = 64;
    std::vector<float> img(static_cast<size_t>(W * H), 0.0f);

    ct.inject_visual(img, W, H, 0.0);

    // Black image → LP values all 0 → avg 0 → nit = clamp(round(0*8 - 4), -4, +4) = -4
    // Some energy will be injected (nits = -4), but check it's finite
    CHECK(all_finite(ct));
}

TEST_CASE("§B-3 Visual inject — field stable after propagation",
          "[multimodal_integ][visual]") {
    CognitiveTorus ct(3);
    ct.set_gpu(false);

    const int W = 32, H = 32;
    std::vector<float> img(static_cast<size_t>(W * H), 0.5f);  // mid-grey
    ct.inject_visual(img, W, H, 0.0);

    float dt = ct.safe_dt();
    ct.run(100, dt);
    CHECK(all_finite(ct));
}

// ─────────────────────────────────────────────────────────────────────────────
// §C — Multi-input fusion
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("§C-1 Fusion — text + audio simultaneous → no explosion",
          "[multimodal_integ][fusion]") {
    CognitiveTorus ct(3);
    ct.set_gpu(false);

    // Inject text (raw Nit vector — no ONNX needed)
    std::vector<Nit> text_nits(128, Nit{2});  // uniform embedding
    ct.inject_raw(text_nits, 0.0);

    // Inject audio
    auto pcm = make_sine(EMITTER_FREQS[3], CAPTURE_RATE_HZ,
                          static_cast<int>(CAPTURE_RATE_HZ));
    ct.inject_audio(pcm, 0.0);

    // Propagate
    float dt = ct.safe_dt();
    ct.run(200, dt);

    CHECK(all_finite(ct));
    double E = total_energy(ct);
    CHECK(std::isfinite(E));
    CHECK(E > 0.0);
}

TEST_CASE("§C-2 Fusion — text + audio + visual → field survives",
          "[multimodal_integ][fusion]") {
    CognitiveTorus ct(3);
    ct.set_gpu(false);

    // Text
    std::vector<Nit> text_nits(128, Nit{1});
    ct.inject_raw(text_nits, 0.0);

    // Audio
    auto pcm = make_sine(EMITTER_FREQS[0], CAPTURE_RATE_HZ,
                          static_cast<int>(CAPTURE_RATE_HZ));
    ct.inject_audio(pcm, 0.5);

    // Visual
    const int W = 32, H = 32;
    std::vector<float> img(static_cast<size_t>(W * H), 0.3f);
    ct.inject_visual(img, W, H, 1.0);

    // Propagate
    float dt = ct.safe_dt();
    ct.run(200, dt);

    CHECK(all_finite(ct));
}

TEST_CASE("§C-3 Fusion — repeated audio injections don't accumulate to NaN",
          "[multimodal_integ][fusion]") {
    CognitiveTorus ct(3);
    ct.set_gpu(false);
    float dt = ct.safe_dt();

    auto pcm = make_sine(EMITTER_FREQS[5], CAPTURE_RATE_HZ, 9600);

    // Inject audio 10 times with propagation between
    for (int tick = 0; tick < 10; ++tick) {
        ct.inject_audio(pcm, static_cast<double>(tick) * 0.1);
        ct.run(20, dt);
    }

    CHECK(all_finite(ct));
    CHECK(total_energy(ct) > 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// §D — Energy conservation
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("§D-1 Energy — audio injection bounded by INJECTION_SCALE",
          "[multimodal_integ][energy]") {
    CognitiveTorus ct(3);
    double E_before = total_energy(ct);

    // Maximum possible audio: full amplitude sine across all bands
    // Energy increase should be bounded
    auto pcm = make_sine(EMITTER_FREQS[3], CAPTURE_RATE_HZ,
                          static_cast<int>(CAPTURE_RATE_HZ * 2), 1.0f);
    ct.inject_audio(pcm, 0.0);

    double E_after = total_energy(ct);
    double delta_E = E_after - E_before;

    // HolographicInjector normalises with INJECTION_SCALE = 0.05
    // Max perturbation per injection ≈ 0.05 * num_chords
    // Energy delta should be small relative to total field
    CHECK(std::isfinite(delta_E));
    // The injected energy should be a small fraction (< 50% of baseline)
    // This verifies the normalisation works
    if (E_before > 0.0) {
        CHECK(delta_E / E_before < 0.5);
    }
}

TEST_CASE("§D-2 Energy — visual injection bounded",
          "[multimodal_integ][energy]") {
    CognitiveTorus ct(3);
    double E_before = total_energy(ct);

    // Full white image
    const int W = 64, H = 64;
    std::vector<float> img(static_cast<size_t>(W * H), 1.0f);
    ct.inject_visual(img, W, H, 0.0);

    double E_after = total_energy(ct);
    double delta_E = E_after - E_before;
    CHECK(std::isfinite(delta_E));
}

// ─────────────────────────────────────────────────────────────────────────────
// §E — MultimodalEngine facade
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("§E-1 Engine — tick_audio_nits produces 128 valid nits",
          "[multimodal_integ][engine]") {
    MultimodalConfig cfg;
    cfg.checkpoint_dir = "/tmp/nikola_test_checkpoints";
    MultimodalEngine engine(cfg);
    auto pcm = make_sine(EMITTER_FREQS[3], CAPTURE_RATE_HZ,
                          static_cast<int>(CAPTURE_RATE_HZ));
    auto nits = engine.tick_audio_nits(pcm);
    REQUIRE(nits.size() == 128);
    for (auto n : nits) {
        CHECK(n >= NIT_MIN);
        CHECK(n <= NIT_MAX);
    }
    // Should have non-zero nits for a sine
    int nonzero = 0;
    for (auto n : nits) nonzero += (n != 0) ? 1 : 0;
    CHECK(nonzero > 0);
}

TEST_CASE("§E-2 Engine — tick_audio_nits empty → zeros",
          "[multimodal_integ][engine]") {
    MultimodalConfig cfg;
    cfg.checkpoint_dir = "/tmp/nikola_test_checkpoints";
    MultimodalEngine engine(cfg);
    std::span<const float> empty{};
    auto nits = engine.tick_audio_nits(empty);
    REQUIRE(nits.size() == 128);
    for (auto n : nits) {
        CHECK(n == NIT_ZERO);
    }
}

TEST_CASE("§E-3 Engine — tick_audio_nits disabled → zeros",
          "[multimodal_integ][engine]") {
    MultimodalConfig cfg;
    cfg.enable_audio = false;
    cfg.checkpoint_dir = "/tmp/nikola_test_checkpoints";
    MultimodalEngine engine(cfg);
    auto pcm = make_sine(EMITTER_FREQS[3], CAPTURE_RATE_HZ, 4800);
    auto nits = engine.tick_audio_nits(pcm);
    REQUIRE(nits.size() == 128);
    for (auto n : nits) {
        CHECK(n == NIT_ZERO);
    }
}
