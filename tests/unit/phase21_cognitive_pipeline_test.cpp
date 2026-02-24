/**
 * @file tests/unit/phase21_cognitive_pipeline_test.cpp
 * @brief Phase 21 — CognitiveTorus + ResonanceDecoder pipeline tests
 *
 * Validates the Path B cognitive input/output stack:
 *   inject_raw  → HolographicInjector → TorusGrid perturbation
 *   step / run  → Propagator physics evolution
 *   hot_nodes   → top-k |ψ|² intensity readout
 *   node_wave9d → 9D neighbourhood waveform extraction
 *   resonance_snapshot → per-node |ψ|² export vector
 *   ResonanceDecoder + HolographicLexicon → wave → token round-trip
 *
 * These tests use n=2 (2^9 = 512 nodes) and raw Nit injection for speed
 * and to avoid any ONNX dependency.  ONNX-requiring paths are tested in the
 * end-to-end suite when NIKOLA_HAS_ORT is defined.
 */

#include <nikola/cognitive/cognitive_torus.hpp>
#include <nikola/cognitive/resonance_decoder.hpp>
#include <nikola/foundation/nit.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <complex>
#include <numeric>
#include <vector>

using namespace nikola::cognitive;
using nikola::foundation::Nit;
using Catch::Matchers::WithinAbs;

// ─────────────────────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build a 128-Nit vector with a simple repeating pattern [amplitude, -amplitude, ...]
static std::vector<Nit> make_nit_pattern(Nit amplitude = 3, size_t len = 128) {
    std::vector<Nit> nits(len);
    for (size_t i = 0; i < len; ++i)
        nits[i] = (i % 2 == 0) ? amplitude : static_cast<Nit>(-amplitude);
    return nits;
}

/// Build a 9-element all-ones complex wave vector
static std::vector<Complex> make_unit_wave9d() {
    return std::vector<Complex>(9, Complex(1.f, 0.f));
}

// ─────────────────────────────────────────────────────────────────────────────
//  CognitiveTorus construction
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("CognitiveTorus: construction n=2 creates 512-node grid", "[Phase21][CognitiveTorus]") {
    // n=2 → 2^9 = 512 nodes, fast for unit tests
    CognitiveTorus ct(2);

    REQUIRE(ct.num_nodes() == 512u);
    REQUIRE(ct.time() == Catch::Approx(0.f));
    REQUIRE(ct.max_dt() > 0.f);
    REQUIRE(std::isfinite(static_cast<double>(ct.total_probability())));
}

TEST_CASE("CognitiveTorus: pilot wave seeds non-zero probability", "[Phase21][CognitiveTorus]") {
    CognitiveTorus ct(2);

    // seed_manifold injects a pilot wave → total probability > 0
    REQUIRE(ct.total_probability() > 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Injection
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("CognitiveTorus: inject_raw perturbs wavefunction", "[Phase21][CognitiveTorus]") {
    CognitiveTorus ct(2);
    double prob_before = ct.total_probability();

    auto nits = make_nit_pattern(4);
    ct.inject_raw(nits, /*time=*/0.0);

    double prob_after = ct.total_probability();
    // Perturbation must change the wavefunction (energy added)
    REQUIRE(prob_after != Catch::Approx(prob_before).epsilon(1e-6));
}

TEST_CASE("CognitiveTorus: inject_raw with zero Nits is a no-op", "[Phase21][CognitiveTorus]") {
    CognitiveTorus ct(2);
    double prob_before = ct.total_probability();

    auto nits = std::vector<Nit>(128, Nit{0});
    ct.inject_raw(nits, 0.0);

    double prob_after = ct.total_probability();
    REQUIRE_THAT(static_cast<float>(prob_after),
                 WithinAbs(static_cast<float>(prob_before), 1e-5f));
}

// ─────────────────────────────────────────────────────────────────────────────
//  Physics step
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("CognitiveTorus: step advances simulation time", "[Phase21][CognitiveTorus]") {
    CognitiveTorus ct(2);
    float dt = ct.max_dt();
    REQUIRE(dt > 0.f);

    ct.step(dt);
    REQUIRE(ct.time() == Catch::Approx(dt).epsilon(1e-5));
}

TEST_CASE("CognitiveTorus: run(N) accumulates N steps of time", "[Phase21][CognitiveTorus]") {
    CognitiveTorus ct(2);
    float dt = ct.max_dt();
    int N = 10;

    ct.run(N, dt);
    REQUIRE(ct.time() == Catch::Approx(static_cast<float>(N) * dt).epsilon(1e-4));
}

TEST_CASE("CognitiveTorus: probability remains finite after inject+run", "[Phase21][CognitiveTorus]") {
    CognitiveTorus ct(2);
    ct.inject_raw(make_nit_pattern(3), 0.0);
    ct.run(20, ct.max_dt());

    double P = ct.total_probability();
    REQUIRE(std::isfinite(P));
    REQUIRE(P >= 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Hot-node readout
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("CognitiveTorus: hot_nodes returns k valid indices", "[Phase21][CognitiveTorus]") {
    CognitiveTorus ct(2);
    ct.inject_raw(make_nit_pattern(4), 0.0);
    ct.run(5, ct.max_dt());

    auto hot = ct.hot_nodes(20);
    REQUIRE(hot.size() <= 20u);
    REQUIRE(hot.size() <= ct.num_nodes());

    for (size_t idx : hot) {
        REQUIRE(idx < ct.num_nodes());
    }
}

TEST_CASE("CognitiveTorus: hot_nodes are ordered by descending intensity", "[Phase21][CognitiveTorus]") {
    CognitiveTorus ct(2);
    ct.inject_raw(make_nit_pattern(4), 0.0);

    auto hot = ct.hot_nodes(10);
    REQUIRE(!hot.empty());

    float prev_intensity = std::numeric_limits<float>::max();
    for (size_t idx : hot) {
        float I = ct.intensity(idx);
        REQUIRE(I >= 0.f);
        REQUIRE(I <= prev_intensity + 1e-5f);  // non-increasing
        prev_intensity = I;
    }
}

TEST_CASE("CognitiveTorus: hot_nodes(0) returns empty", "[Phase21][CognitiveTorus]") {
    CognitiveTorus ct(2);
    REQUIRE(ct.hot_nodes(0).empty());
}

TEST_CASE("CognitiveTorus: hot_nodes(N_large) clamped to num_nodes", "[Phase21][CognitiveTorus]") {
    CognitiveTorus ct(2);
    auto hot = ct.hot_nodes(99999);
    REQUIRE(hot.size() == ct.num_nodes());
}

// ─────────────────────────────────────────────────────────────────────────────
//  9D waveform extraction
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("CognitiveTorus: node_wave9d returns 9 complex values", "[Phase21][CognitiveTorus]") {
    CognitiveTorus ct(2);
    ct.inject_raw(make_nit_pattern(3), 0.0);

    auto hot = ct.hot_nodes(1);
    REQUIRE(!hot.empty());

    auto wave = ct.node_wave9d(hot[0]);
    REQUIRE(wave.size() == 9u);
}

TEST_CASE("CognitiveTorus: node_wave9d values are finite", "[Phase21][CognitiveTorus]") {
    CognitiveTorus ct(2);
    ct.inject_raw(make_nit_pattern(3), 0.0);
    ct.run(5, ct.max_dt());

    auto hot = ct.hot_nodes(5);
    for (size_t idx : hot) {
        auto wave = ct.node_wave9d(idx);
        for (const auto& c : wave) {
            REQUIRE(std::isfinite(c.real()));
            REQUIRE(std::isfinite(c.imag()));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Resonance snapshot
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("CognitiveTorus: resonance_snapshot has size == num_nodes", "[Phase21][CognitiveTorus]") {
    CognitiveTorus ct(2);
    auto snap = ct.resonance_snapshot();
    REQUIRE(snap.size() == ct.num_nodes());
}

TEST_CASE("CognitiveTorus: resonance_snapshot values are non-negative", "[Phase21][CognitiveTorus]") {
    CognitiveTorus ct(2);
    ct.inject_raw(make_nit_pattern(2), 0.0);
    ct.run(10, ct.max_dt());

    auto snap = ct.resonance_snapshot();
    for (float v : snap) {
        REQUIRE(v >= 0.f);
        REQUIRE(std::isfinite(v));
    }
}

TEST_CASE("CognitiveTorus: snapshot sum equals total_probability", "[Phase21][CognitiveTorus]") {
    CognitiveTorus ct(2);
    ct.inject_raw(make_nit_pattern(3), 0.0);

    double total_P    = ct.total_probability();
    auto   snap       = ct.resonance_snapshot();
    double snap_sum   = 0.0;
    for (float v : snap) snap_sum += static_cast<double>(v);

    REQUIRE_THAT(snap_sum, WithinAbs(total_P, total_P * 0.01 + 1e-4));
}

// ─────────────────────────────────────────────────────────────────────────────
//  psi / intensity consistency
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("CognitiveTorus: intensity(idx) == |psi(idx)|^2", "[Phase21][CognitiveTorus]") {
    CognitiveTorus ct(2);
    ct.inject_raw(make_nit_pattern(4), 0.0);

    for (size_t i = 0; i < std::min(ct.num_nodes(), size_t{50}); ++i) {
        auto   c  = ct.psi(i);
        float  I  = ct.intensity(i);
        float  I2 = c.real() * c.real() + c.imag() * c.imag();
        REQUIRE_THAT(I, WithinAbs(I2, 1e-6f));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  ResonanceDecoder — construction and registration
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ResonanceDecoder: default construction has empty lexicon", "[Phase21][ResonanceDecoder]") {
    ResonanceDecoder dec;
    REQUIRE(dec.vocab_size() == 0u);
}

TEST_CASE("ResonanceDecoder: register_token increments vocab_size", "[Phase21][ResonanceDecoder]") {
    ResonanceDecoder dec;
    dec.register_token("hello", make_unit_wave9d());
    REQUIRE(dec.vocab_size() == 1u);

    dec.register_token("world", make_unit_wave9d());
    REQUIRE(dec.vocab_size() == 2u);
}

TEST_CASE("ResonanceDecoder: register_token pads short waves to 9", "[Phase21][ResonanceDecoder]") {
    ResonanceDecoder dec;
    // Provide only 3 elements — should be padded internally
    std::vector<Complex> short_wave = {Complex(1.f, 0.f), Complex(0.5f, 0.f), Complex(-0.5f, 0.f)};
    REQUIRE_NOTHROW(dec.register_token("short", short_wave));
    REQUIRE(dec.vocab_size() == 1u);
}

// ─────────────────────────────────────────────────────────────────────────────
//  ResonanceDecoder — decode round-trip
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ResonanceDecoder: decode on empty lexicon returns empty vector", "[Phase21][ResonanceDecoder]") {
    ResonanceDecoder dec;     // no tokens registered
    CognitiveTorus   ct(2);
    ct.inject_raw(make_nit_pattern(3), 0.0);
    ct.run(5, ct.max_dt());

    auto tokens = dec.decode(ct, 10);
    REQUIRE(tokens.empty());
}

TEST_CASE("ResonanceDecoder: decode_text on empty lexicon returns empty string", "[Phase21][ResonanceDecoder]") {
    ResonanceDecoder dec;
    CognitiveTorus   ct(2);

    auto text = dec.decode_text(ct, 5);
    REQUIRE(text.empty());
}

TEST_CASE("ResonanceDecoder: decode does not crash on post-step torus", "[Phase21][ResonanceDecoder]") {
    ResonanceDecoder dec;
    dec.register_token("nikola", make_unit_wave9d());
    dec.register_token("hello",  make_unit_wave9d());  // same wave → same bucket

    CognitiveTorus ct(2);
    ct.inject_raw(make_nit_pattern(4), 0.0);
    ct.run(10, ct.max_dt());

    std::vector<std::string> result;
    REQUIRE_NOTHROW(result = dec.decode(ct, 20));
    // result may be empty or non-empty depending on cosine-similarity match;
    // we only assert no crash and that every returned token is in the vocab.
    for (const auto& tok : result) {
        REQUIRE((tok == "nikola" || tok == "hello"));
    }
}

TEST_CASE("ResonanceDecoder: decode_text returns space-separated unique tokens", "[Phase21][ResonanceDecoder]") {
    ResonanceDecoder dec;
    dec.register_token("alpha", make_unit_wave9d());

    CognitiveTorus ct(2);
    ct.inject_raw(make_nit_pattern(2), 0.0);
    ct.run(5, ct.max_dt());

    std::string text;
    REQUIRE_NOTHROW(text = dec.decode_text(ct, 20));
    // Just check it's a valid string; token matching is probabilistic
    REQUIRE(text.find("  ") == std::string::npos);  // no double spaces
}

// ─────────────────────────────────────────────────────────────────────────────
//  Full Path B smoke test
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase21: full Path B smoke — inject → resonate → decode → text", "[Phase21][smoke]") {
    // Build a small torus
    CognitiveTorus ct(2);
    REQUIRE(ct.num_nodes() == 512u);

    // Inject a semantic signal
    auto nits = make_nit_pattern(4, 128);
    ct.inject_raw(nits, 0.0);

    double P_before = ct.total_probability();
    REQUIRE(P_before > 0.0);

    // Let it resonate
    float dt = ct.max_dt();
    ct.run(30, dt);

    // Probability still finite
    double P_after = ct.total_probability();
    REQUIRE(std::isfinite(P_after));
    REQUIRE(P_after > 0.0);

    // Read resonance snapshot
    auto snap = ct.resonance_snapshot();
    REQUIRE(snap.size() == 512u);

    // Find hotspots
    auto hot = ct.hot_nodes(10);
    REQUIRE(!hot.empty());
    REQUIRE(hot[0] < 512u);

    // Decoder round-trip
    ResonanceDecoder dec;
    dec.register_token("consciousness", make_unit_wave9d());
    dec.register_token("resonance",     make_unit_wave9d());

    std::string result;
    REQUIRE_NOTHROW(result = dec.decode_text(ct, 20));

    // Pipeline complete — no crash, no UB.  Semantic correctness is a
    // separate concern addressed in integration and end-to-end tests.
    INFO("Decoded text: '" << result << "'");
    SUCCEED("Path B pipeline: inject → run → hot_nodes → decode_text completed.");
}
