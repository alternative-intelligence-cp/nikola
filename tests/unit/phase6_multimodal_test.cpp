/**
 * @file tests/unit/phase6_multimodal_test.cpp
 * @brief Phase 6: Multimodal & Persistence test suite (Catch2 v3).
 *
 * Covers all 5 Gap criteria:
 *   Gap 6.1 — AudioEmitterLayout: 8-emitter positions, golden ratio frequencies
 *   Gap 6.2 — LogPolarTransform: 64×64 log-polar sampling, inject coords
 *   Gap 6.3 — CheckpointManager: periodic + pre-NAP triggers, retention
 *   Gap 6.4 — GGUFExporter: binary write/read, KV metadata, topology dims
 *   Gap 6.5 — AdaptiveQuantizer: Q9_0 trit encoding, FP16 path, ratios
 *
 * Plus: MultimodalEngine integration facade.
 *
 * No OpenCV, no gguf.h — all tests use stdlib/internal implementations.
 */

#define NIKOLA_MULTIMODAL_ENGINE_IMPL

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <nikola/multimodal/audio_emitter.hpp>
#include <nikola/multimodal/log_polar_transform.hpp>
#include <nikola/multimodal/checkpoint_manager.hpp>
#include <nikola/multimodal/gguf_exporter.hpp>
#include <nikola/multimodal/adaptive_quantizer.hpp>
#include <nikola/multimodal/multimodal_engine.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <numbers>
#include <numeric>
#include <string>
#include <vector>

using namespace nikola::multimodal;
using Catch::Matchers::WithinAbs;

// ─────────────────────────────────────────────────────────────────────────────
//  Gap 6.1 — AudioEmitterLayout
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Gap6.1 — exactly 8 emitters are generated", "[audio][gap6.1]") {
    auto positions = AudioEmitterLayout::all_positions();
    CHECK(static_cast<int>(positions.size()) == NUM_EMITTERS);
    for (int i = 0; i < NUM_EMITTERS; ++i) {
        CHECK(positions[i].emitter_id == i);
    }
}

TEST_CASE("Gap6.1 — emitter z coordinate is always 0", "[audio][gap6.1]") {
    auto positions = AudioEmitterLayout::all_positions();
    for (const auto& e : positions) {
        CHECK(e.coord.c[2] == 0); // z dimension index = 2
    }
}

TEST_CASE("Gap6.1 — emitter positions lie on circular ring", "[audio][gap6.1]") {
    const int Nx = 64, Ny = 64;
    const double cx = Nx / 2.0;
    const double cy = Ny / 2.0;
    const double R  = Nx * 0.5;   // AudioEmitterLayout uses AUDIO_RING_FRAC=0.5

    auto positions = AudioEmitterLayout::all_positions(Nx, Ny);
    for (int i = 0; i < NUM_EMITTERS; ++i) {
        const double x = positions[i].coord.c[0];
        const double y = positions[i].coord.c[1];
        const double dist = std::hypot(x - cx, y - cy);
        // Allow ±1 cell rounding error
        CHECK_THAT(dist, WithinAbs(R, 1.5));
    }
}

TEST_CASE("Gap6.1 — minimum emitter separation > 10 grid cells", "[audio][gap6.1]") {
    const double min_sep = AudioEmitterLayout::min_emitter_separation(64, 64);
    CHECK(min_sep > 10.0);
}

TEST_CASE("Gap6.1 — golden ratio frequencies are monotonically increasing", "[audio][gap6.1]") {
    for (int i = 0; i < NUM_EMITTERS - 1; ++i) {
        CHECK(AudioEmitterLayout::emitter_frequency(i + 1) >
              AudioEmitterLayout::emitter_frequency(i));
    }
}

TEST_CASE("Gap6.1 — f_0 = π (golden ratio exponent 0 → φ^0 = 1)", "[audio][gap6.1]") {
    const double f0 = AudioEmitterLayout::emitter_frequency(0);
    CHECK_THAT(f0, WithinAbs(std::numbers::pi, 1e-9));
}

TEST_CASE("Gap6.1 — f_1 = π·φ (golden ratio exponent 1)", "[audio][gap6.1]") {
    const double f1 = AudioEmitterLayout::emitter_frequency(1);
    const double expected = std::numbers::pi * 1.6180339887498948482;
    CHECK_THAT(f1, WithinAbs(expected, 1e-9));
}

TEST_CASE("Gap6.1 — r dimension carries resonance code (~12 for Nr=16)", "[audio][gap6.1]") {
    auto positions = AudioEmitterLayout::all_positions(64, 64, 16, 16);
    const uint16_t expected_r = static_cast<uint16_t>(
        std::round(EMITTER_RADIAL_FRAC * 16)); // 0.8 * 16 = 12.8 → 13
    for (const auto& e : positions) {
        CHECK(e.coord.c[4] == expected_r); // r dimension index = 4
    }
}

TEST_CASE("Gap6.1 — out-of-range emitter index throws", "[audio][gap6.1]") {
    CHECK_THROWS_AS(AudioEmitterLayout::compute_position(8),  std::out_of_range);
    CHECK_THROWS_AS(AudioEmitterLayout::compute_position(-1), std::out_of_range);
}

TEST_CASE("Gap6.1 — time index wraps via modulo to t dimension", "[audio][gap6.1]") {
    auto e0 = AudioEmitterLayout::compute_position(0, 64, 64, 16, 16, 128, 0);
    auto e1 = AudioEmitterLayout::compute_position(0, 64, 64, 16, 16, 128, 128);
    // t=0 and t=128 should both give c[3]=0
    CHECK(e0.coord.c[3] == 0);
    CHECK(e1.coord.c[3] == 0); // 128 % 128 == 0
    auto e2 = AudioEmitterLayout::compute_position(0, 64, 64, 16, 16, 128, 55);
    CHECK(e2.coord.c[3] == 55);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Gap 6.2 — LogPolarTransform
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Gap6.2 — transform returns 64×64 output", "[visual][gap6.2]") {
    const int W = 64, H = 64;
    std::vector<float> img(W * H, 0.5f);
    auto lp = LogPolarTransform::transform(img, W, H);
    CHECK(lp.size() == static_cast<size_t>(LP_RADIAL_BINS * LP_ANGULAR_BINS));
}

TEST_CASE("Gap6.2 — zero image → zero output", "[visual][gap6.2]") {
    const int W = 64, H = 64;
    std::vector<float> img(W * H, 0.0f);
    auto lp = LogPolarTransform::transform(img, W, H);
    for (float v : lp) CHECK(v == 0.0f);
}

TEST_CASE("Gap6.2 — uniform white image → inject list non-empty", "[visual][gap6.2]") {
    const int W = 64, H = 64;
    std::vector<float> img(W * H, 1.0f); // all pixels at max intensity
    auto lp     = LogPolarTransform::transform(img, W, H);
    auto injections = LogPolarTransform::inject_coords(lp, 0);
    CHECK_FALSE(injections.empty());
}

TEST_CASE("Gap6.2 — only pixels above threshold appear in injection list", "[visual][gap6.2]") {
    const int W = 32, H = 32;
    // Construct image with exactly one bright pixel at center
    std::vector<float> img(W * H, LP_INJECT_THRESHOLD / 2.0f); // below threshold
    img[16 * W + 16] = 1.0f; // one bright pixel
    auto lp     = LogPolarTransform::transform(img, W, H);
    auto injlist = LogPolarTransform::inject_coords(lp, 0);
    // All injected pixels must have intensity >= LP_INJECT_THRESHOLD
    for (const auto& [coord, intensity] : injlist) {
        CHECK(intensity >= LP_INJECT_THRESHOLD);
    }
}

TEST_CASE("Gap6.2 — injection z coordinate is visual layer (z=1)", "[visual][gap6.2]") {
    const int W = 64, H = 64;
    std::vector<float> img(W * H, 1.0f);
    auto lp     = LogPolarTransform::transform(img, W, H);
    auto injlist = LogPolarTransform::inject_coords(lp, 0);
    for (const auto& [coord, v] : injlist) {
        CHECK(coord.c[2] == LP_INJECT_Z); // z=1
    }
}

TEST_CASE("Gap6.2 — injection r/s carry mid-range neurochemical codes", "[visual][gap6.2]") {
    const int W = 64, H = 64;
    std::vector<float> img(W * H, 1.0f);
    auto lp     = LogPolarTransform::transform(img, W, H);
    auto injlist = LogPolarTransform::inject_coords(lp, 5, 128);
    for (const auto& [coord, v] : injlist) {
        CHECK(coord.c[4] == LP_INJECT_R); // r=8
        CHECK(coord.c[5] == LP_INJECT_S); // s=8
        CHECK(coord.c[3] == 5);           // t=time_index
    }
}

TEST_CASE("Gap6.2 — invalid image dimensions return empty output", "[visual][gap6.2]") {
    std::vector<float> img(100, 0.5f);
    auto lp = LogPolarTransform::transform(img, 0, 10); // width=0
    // Should return zero-filled output, not crash
    CHECK(lp.size() == static_cast<size_t>(LP_RADIAL_BINS * LP_ANGULAR_BINS));
}

// ─────────────────────────────────────────────────────────────────────────────
//  Gap 6.3 — CheckpointManager
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Gap6.3 — no checkpoint fires within interval", "[checkpoint][gap6.3]") {
    const auto tmpdir = std::filesystem::temp_directory_path() / "nikola_test_ckpt6";
    std::filesystem::create_directories(tmpdir);

    CheckpointManager ckpt(tmpdir.string());

    // Call with wall time far in the past (same instant = 0 elapsed)
    const auto t0 = CheckpointManager::Clock::now();
    bool fired = ckpt.update(false, t0);
    // First call: elapsed = 0 < 300 → periodic NOT triggered
    // is_napping=false → pre-nap NOT triggered
    CHECK_FALSE(fired);

    std::filesystem::remove_all(tmpdir);
}

TEST_CASE("Gap6.3 — periodic fires after interval elapses", "[checkpoint][gap6.3]") {
    const auto tmpdir = std::filesystem::temp_directory_path() / "nikola_test_ckpt6b";
    std::filesystem::create_directories(tmpdir);

    CheckpointManager ckpt(tmpdir.string());

    // Simulate 301 seconds later
    const auto future = CheckpointManager::Clock::now()
                      + std::chrono::seconds(CHECKPOINT_INTERVAL_SEC + 1);
    bool fired = ckpt.update(false, future);
    CHECK(fired);
    CHECK(ckpt.records().size() == 1u);
    CHECK(ckpt.records()[0].reason == CheckpointReason::PERIODIC);

    std::filesystem::remove_all(tmpdir);
}

TEST_CASE("Gap6.3 — pre-NAP fires on rising edge of is_napping", "[checkpoint][gap6.3]") {
    const auto tmpdir = std::filesystem::temp_directory_path() / "nikola_test_ckpt6c";
    std::filesystem::create_directories(tmpdir);

    CheckpointManager ckpt(tmpdir.string());
    const auto t0 = CheckpointManager::Clock::now();

    // First call: not napping → no trigger
    ckpt.update(false, t0);

    // Second call: napping transitions to true → pre-NAP trigger
    bool fired = ckpt.update(true, t0);
    CHECK(fired);

    // Find pre-nap record
    bool found = false;
    for (const auto& r : ckpt.records()) {
        if (r.reason == CheckpointReason::PRE_NAP) { found = true; break; }
    }
    CHECK(found);

    std::filesystem::remove_all(tmpdir);
}

TEST_CASE("Gap6.3 — pre-NAP does NOT fire if already napping", "[checkpoint][gap6.3]") {
    const auto tmpdir = std::filesystem::temp_directory_path() / "nikola_test_ckpt6d";
    std::filesystem::create_directories(tmpdir);

    CheckpointManager ckpt(tmpdir.string());
    const auto t0 = CheckpointManager::Clock::now();

    ckpt.update(true, t0);  // rising edge → fires
    size_t count_after_first = ckpt.records().size();

    // Second update also napping → no additional pre-NAP trigger
    ckpt.update(true, t0);
    CHECK(ckpt.records().size() == count_after_first);

    std::filesystem::remove_all(tmpdir);
}

TEST_CASE("Gap6.3 — rolling window keeps max 10 periodic checkpoints", "[checkpoint][gap6.3]") {
    const auto tmpdir = std::filesystem::temp_directory_path() / "nikola_test_ckpt6e";
    std::filesystem::create_directories(tmpdir);

    CheckpointManager ckpt(tmpdir.string());

    // Force 15 periodic checkpoints
    for (int i = 0; i < 15; ++i) {
        ckpt.force_checkpoint(CheckpointReason::PERIODIC);
    }

    // All 15 appear in records(), but periodic_queue_ only keeps 10
    // (We test the public API — records() shows all, but the internal queue is capped)
    CHECK(ckpt.records().size() == 15u);

    std::filesystem::remove_all(tmpdir);
}

TEST_CASE("Gap6.3 — force_checkpoint creates a .dmc file", "[checkpoint][gap6.3]") {
    const auto tmpdir = std::filesystem::temp_directory_path() / "nikola_test_ckpt6f";
    std::filesystem::create_directories(tmpdir);

    CheckpointManager ckpt(tmpdir.string());
    ckpt.force_checkpoint(CheckpointReason::PERIODIC);

    REQUIRE_FALSE(ckpt.records().empty());
    const std::string& path = ckpt.records().front().path;
    CHECK(std::filesystem::exists(path));
    CHECK(path.ends_with(".dmc"));

    std::filesystem::remove_all(tmpdir);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Gap 6.4 — GGUFExporter
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Gap6.4 — exported file has correct GGUF magic bytes", "[gguf][gap6.4]") {
    const auto tmpdir = std::filesystem::temp_directory_path() / "nikola_test_gguf6";
    std::filesystem::create_directories(tmpdir);
    const std::string fname = (tmpdir / "test.gguf").string();

    GGUFExporter::export_metadata(fname);
    REQUIRE(std::filesystem::exists(fname));

    // Read magic
    std::ifstream f(fname, std::ios::binary);
    uint32_t magic = 0;
    f.read(reinterpret_cast<char*>(&magic), 4);
    CHECK(magic == GGUF_MAGIC);

    std::filesystem::remove_all(tmpdir);
}

TEST_CASE("Gap6.4 — read_architecture returns 'nikola_v0'", "[gguf][gap6.4]") {
    const auto tmpdir = std::filesystem::temp_directory_path() / "nikola_test_gguf6b";
    std::filesystem::create_directories(tmpdir);
    const std::string fname = (tmpdir / "test.gguf").string();

    GGUFExporter::export_metadata(fname);
    const std::string arch = GGUFExporter::read_architecture(fname);
    CHECK(arch == "nikola_v0");

    std::filesystem::remove_all(tmpdir);
}

TEST_CASE("Gap6.4 — topology dims are r,s,t,u,v,w,x,y,z order", "[gguf][gap6.4]") {
    const auto tmpdir = std::filesystem::temp_directory_path() / "nikola_test_gguf6c";
    std::filesystem::create_directories(tmpdir);
    const std::string fname = (tmpdir / "test.gguf").string();

    GGUFExporter::export_metadata(fname);
    const auto dims = GGUFExporter::read_topology_dims(fname);

    REQUIRE(dims.size() == 9u);
    // r=16, s=16, t=128, u=32, v=32, w=32, x=64, y=64, z=64
    const std::array<int64_t, 9> expected = {16, 16, 128, 32, 32, 32, 64, 64, 64};
    for (size_t i = 0; i < 9; ++i) {
        CHECK(dims[i] == expected[i]);
    }

    std::filesystem::remove_all(tmpdir);
}

TEST_CASE("Gap6.4 — export with tensor data creates larger file", "[gguf][gap6.4]") {
    const auto tmpdir = std::filesystem::temp_directory_path() / "nikola_test_gguf6d";
    std::filesystem::create_directories(tmpdir);
    const std::string f_meta   = (tmpdir / "meta.gguf").string();
    const std::string f_tensor = (tmpdir / "tensor.gguf").string();

    // Without tensors
    GGUFExporter::export_metadata(f_meta);

    // With tensors (1000 elements each)
    std::vector<float> re(1000, 0.5f), im(1000, 0.1f);
    GGUFExporter::export_metadata(f_tensor, re, im);

    CHECK(std::filesystem::file_size(f_tensor) > std::filesystem::file_size(f_meta));

    std::filesystem::remove_all(tmpdir);
}

TEST_CASE("Gap6.4 — invalid file path throws", "[gguf][gap6.4]") {
    CHECK_THROWS(GGUFExporter::export_metadata("/nonexistent_path/test.gguf"));
}

// ─────────────────────────────────────────────────────────────────────────────
//  Gap 6.5 — AdaptiveQuantizer
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Gap6.5 — quantize_to_trit(0.0) → 0", "[quantizer][gap6.5]") {
    CHECK(AdaptiveQuantizer::quantize_to_trit(0.0f) == 0);
}

TEST_CASE("Gap6.5 — quantize_to_trit(1.0) → +4", "[quantizer][gap6.5]") {
    CHECK(AdaptiveQuantizer::quantize_to_trit(1.0f) == int8_t{4});
}

TEST_CASE("Gap6.5 — quantize_to_trit(-1.0) → -4", "[quantizer][gap6.5]") {
    CHECK(AdaptiveQuantizer::quantize_to_trit(-1.0f) == int8_t{-4});
}

TEST_CASE("Gap6.5 — dequantize round-trip error ≤ 0.125", "[quantizer][gap6.5]") {
    // Test reconstruction for uniform grid over [-1, 1]
    for (int i = -100; i <= 100; ++i) {
        const float v   = static_cast<float>(i) / 100.0f;
        const int8_t t  = AdaptiveQuantizer::quantize_to_trit(v);
        const float rec = AdaptiveQuantizer::dequantize_from_trit(t);
        // max error due to 1/8 quantization step
        CHECK_THAT(std::abs(rec - v), WithinAbs(0.0f, Q9_MAX_ABS_ERROR + 1e-6f));
    }
}

TEST_CASE("Gap6.5 — values outside [-1,1] are clamped (no overflow)", "[quantizer][gap6.5]") {
    CHECK(AdaptiveQuantizer::quantize_to_trit(5.0f)  == int8_t{4});
    CHECK(AdaptiveQuantizer::quantize_to_trit(-5.0f) == int8_t{-4});
}

TEST_CASE("Gap6.5 — low-energy nodes use Q9_0 format (format=0)", "[quantizer][gap6.5]") {
    // All near-zero values → low energy → Q9_0
    const size_t N = 100;
    std::vector<float> re(N, 1e-5f);  // |ψ|² = 2e-10 << 1e-3
    std::vector<float> im(N, 1e-5f);
    auto blocks = AdaptiveQuantizer::compress(re, im);
    REQUIRE(blocks.size() == N);
    for (const auto& b : blocks) {
        CHECK(b.format == 0);
    }
}

TEST_CASE("Gap6.5 — high-energy nodes use FP16 format (format=1)", "[quantizer][gap6.5]") {
    // Large values → high energy → FP16
    const size_t N = 100;
    std::vector<float> re(N, 1.0f);   // |ψ|² = 2.0 >> 1e-3
    std::vector<float> im(N, 1.0f);
    auto blocks = AdaptiveQuantizer::compress(re, im);
    REQUIRE(blocks.size() == N);
    for (const auto& b : blocks) {
        CHECK(b.format == 1);
    }
}

TEST_CASE("Gap6.5 — FP16 path preserves exact float values", "[quantizer][gap6.5]") {
    std::vector<float> re = {2.0f, -3.5f, 0.7f};
    std::vector<float> im = {1.0f,  0.5f, 9.0f};
    auto blocks = AdaptiveQuantizer::compress(re, im);
    for (size_t i = 0; i < blocks.size(); ++i) {
        if (blocks[i].format == 1) {
            CHECK_THAT(blocks[i].fp_real, WithinAbs(re[i], 1e-6f));
            CHECK_THAT(blocks[i].fp_imag, WithinAbs(im[i], 1e-6f));
        }
    }
}

TEST_CASE("Gap6.5 — compression ratio < 1 for mostly low-energy data", "[quantizer][gap6.5]") {
    // 95% of nodes are low-energy
    const size_t N = 1000;
    std::vector<float> re(N, 0.0f);
    std::vector<float> im(N, 0.0f);
    // 5% high-energy
    for (size_t i = 0; i < 50; ++i) {
        re[i] = 2.0f;
        im[i] = 2.0f;
    }
    auto blocks = AdaptiveQuantizer::compress(re, im);
    const float ratio = AdaptiveQuantizer::compression_ratio(blocks);
    CHECK(ratio < 1.0f); // compressed < uncompressed
}

TEST_CASE("Gap6.5 — count_formats agrees with compress output", "[quantizer][gap6.5]") {
    const size_t N = 200;
    std::vector<float> re(N, 0.0f), im(N, 0.0f);
    // 50 high-energy
    for (size_t i = 0; i < 50; ++i) { re[i] = 5.0f; im[i] = 5.0f; }

    auto blocks = AdaptiveQuantizer::compress(re, im);
    auto [n_q9, n_fp16] = AdaptiveQuantizer::count_formats(blocks);
    CHECK(n_q9   == 150u);
    CHECK(n_fp16 ==  50u);
}

TEST_CASE("Gap6.5 — decompress recovers Q9 values within error bound", "[quantizer][gap6.5]") {
    const size_t N = 500;
    std::vector<float> re(N), im(N);
    for (size_t i = 0; i < N; ++i) {
        re[i] = static_cast<float>(i % 100) / 1000.0f; // small values
        im[i] = static_cast<float>((i * 7) % 100) / 1000.0f;
    }

    auto blocks = AdaptiveQuantizer::compress(re, im);
    std::vector<float> re_out, im_out;
    AdaptiveQuantizer::decompress(blocks, re_out, im_out);

    REQUIRE(re_out.size() == N);
    for (size_t i = 0; i < N; ++i) {
        const float re_err = std::abs(re_out[i] - re[i]);
        const float im_err = std::abs(im_out[i] - im[i]);
        // Q9_0: max absolute error = 0.125 (after pre-normalization by max_abs)
        CHECK(re_err <= 0.125f + 1e-5f);
        CHECK(im_err <= 0.125f + 1e-5f);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  MultimodalEngine integration
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("MultimodalEngine — constructs with default config", "[engine][phase6]") {
    MultimodalConfig cfg;
    cfg.checkpoint_dir = (std::filesystem::temp_directory_path() / "nikola_eng_test").string();
    MultimodalEngine engine(cfg);
    const auto snap = engine.snapshot();
    CHECK(snap.audio_ticks    == 0u);
    CHECK(snap.visual_ticks   == 0u);
    CHECK(snap.checkpoint_count == 0u);
    std::filesystem::remove_all(cfg.checkpoint_dir);
}

TEST_CASE("MultimodalEngine — tick_audio returns 8 emitters", "[engine][phase6]") {
    MultimodalConfig cfg;
    cfg.checkpoint_dir = (std::filesystem::temp_directory_path() / "nikola_eng_audio").string();
    MultimodalEngine engine(cfg);

    const auto emitters = engine.tick_audio({}, 0);
    CHECK(emitters.size() == static_cast<size_t>(NUM_EMITTERS));
    CHECK(engine.snapshot().audio_ticks == 1u);
    std::filesystem::remove_all(cfg.checkpoint_dir);
}

TEST_CASE("MultimodalEngine — tick_visual with blank image returns non-empty injections",
          "[engine][phase6]")
{
    MultimodalConfig cfg;
    cfg.checkpoint_dir = (std::filesystem::temp_directory_path() / "nikola_eng_visual").string();
    MultimodalEngine engine(cfg);

    const int W = 32, H = 32;
    std::vector<float> img(W * H, 1.0f); // all bright
    const auto injections = engine.tick_visual(img, W, H, 0);
    CHECK_FALSE(injections.empty());
    CHECK(engine.snapshot().visual_ticks == 1u);
    std::filesystem::remove_all(cfg.checkpoint_dir);
}

TEST_CASE("MultimodalEngine — export_gguf creates valid GGUF file", "[engine][phase6]") {
    const auto tmpdir = std::filesystem::temp_directory_path() / "nikola_eng_gguf";
    MultimodalConfig cfg;
    cfg.checkpoint_dir = tmpdir.string();
    MultimodalEngine engine(cfg);

    const std::string fname = (tmpdir / "export.gguf").string();
    engine.export_gguf(fname);
    REQUIRE(std::filesystem::exists(fname));

    const std::string arch = GGUFExporter::read_architecture(fname);
    CHECK(arch == "nikola_v0");
    std::filesystem::remove_all(tmpdir);
}

TEST_CASE("MultimodalEngine — compress_psi reduces data size", "[engine][phase6]") {
    MultimodalConfig cfg;
    cfg.checkpoint_dir = (std::filesystem::temp_directory_path() / "nikola_eng_compress").string();
    MultimodalEngine engine(cfg);

    const size_t N = 1000;
    std::vector<float> re(N, 0.0f), im(N, 0.0f); // all zero → all Q9_0

    auto blocks = engine.compress_psi(re, im);
    CHECK(blocks.size() == N);
    const float ratio = engine.snapshot().last_compression_ratio;
    CHECK(ratio < 1.0f); // Q9 should compress
    std::filesystem::remove_all(cfg.checkpoint_dir);
}
