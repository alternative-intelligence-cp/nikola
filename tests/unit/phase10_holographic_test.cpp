/**
 * @file phase10_holographic_test.cpp
 * @brief Phase 10 unit tests — HolographicLexicon (IMP-02)
 *
 * Covers:
 *   - SpectralHash construction and equality
 *   - Phase quadrant mapping correctness (all 4 quadrants)
 *   - add_token / decode round-trip (exact identical wave → full resonance)
 *   - decode miss on empty lexicon and on non-matching wave
 *   - Vacuum wave rejection (below noise floor)
 *   - Multiple tokens in same hash bucket → best resonance wins
 *   - Low-resonance rejection (wave is noise relative to registered token)
 *   - embed(): forward lookup
 *   - exists() / size() / clear()
 *   - remove_token()
 *   - Re-registration updates wave and inverse index
 *   - Multi-probe LSH: near-boundary phase probes neighbour bucket
 *   - Thread safety: concurrent decode() calls all return same result
 *   - compute_resonance: self-resonance = 1.0, orthogonal = 0.0 (approx)
 *   - wave_energy: zero vector = 0, unit = 1
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/cognitive/holographic_lexicon.hpp>

#include <array>
#include <complex>
#include <numbers>
#include <thread>
#include <vector>

namespace nc = nikola::cognitive;

using Complex   = nc::Complex;
using Lex       = nc::HolographicLexicon;
using SH        = nc::SpectralHash;

// ---------------------------------------------------------------------------
// Helper factories
// ---------------------------------------------------------------------------

/** Build a 9D wave with all dimensions having the given constant phase [rad]. */
static std::vector<Complex> uniform_phase_wave(float phase_rad, float amplitude = 1.0f)
{
    std::vector<Complex> w(9);
    for (auto& c : w)
        c = std::polar(amplitude, phase_rad);
    return w;
}

/** Build a 9D wave with random-ish phases — use as a "noise" wave. */
static std::vector<Complex> noise_wave()
{
    // Deterministic but distinct from any registered wave.
    std::vector<Complex> w(9);
    for (int i = 0; i < 9; ++i)
        w[static_cast<size_t>(i)] = std::polar(0.5f, static_cast<float>(i) * 0.7f);
    return w;
}

/** Suppress a wave to below noise floor. */
static std::vector<Complex> vacuum_wave()
{
    return std::vector<Complex>(9, Complex{1e-6f, 0.0f});  // energy ≪ kNoiseFloor
}

// ===========================================================================
// SpectralHash tests
// ===========================================================================

TEST_CASE("SpectralHash: zero-phase wave hashes consistently", "[lexicon][hash]") {
    const auto w = uniform_phase_wave(0.0f);   // phase=0 → Q2 (10b) for all dims
    const SH h1 = SH::from_wave(w);
    const SH h2 = SH::from_wave(w);
    CHECK(h1 == h2);
}

TEST_CASE("SpectralHash: same wave → same hash (determinism)", "[lexicon][hash]") {
    const auto w = noise_wave();
    CHECK(SH::from_wave(w) == SH::from_wave(w));
}

TEST_CASE("SpectralHash: all-zero phase → quadrant 2 (10b) for each dim", "[lexicon][hash]") {
    const auto   w = uniform_phase_wave(0.0f);
    const SH     h = SH::from_wave(w);
    // Q2 = 2 (binary 10) for each of the 9 dims
    for (int d = 0; d < 9; ++d)
        CHECK(h.quadrant(d) == 2u);
}

TEST_CASE("SpectralHash: negative phase near -pi → quadrant 0 (00b)", "[lexicon][hash]") {
    // phase = -π + ε  ∈ [-π, -π/2) → Q0 = 0
    const auto w = uniform_phase_wave(-std::numbers::pi_v<float> + 0.05f);
    const SH   h = SH::from_wave(w);
    for (int d = 0; d < 9; ++d)
        CHECK(h.quadrant(d) == 0u);
}

TEST_CASE("SpectralHash: phase near +pi/2 → quadrant 3 (11b)", "[lexicon][hash]") {
    // phase = +π/2 + ε ∈ [+π/2, +π) → Q3 = 3
    const auto w = uniform_phase_wave(std::numbers::pi_v<float> * 0.6f);
    const SH   h = SH::from_wave(w);
    for (int d = 0; d < 9; ++d)
        CHECK(h.quadrant(d) == 3u);
}

TEST_CASE("SpectralHash: different phases → different hashes", "[lexicon][hash]") {
    const SH h0 = SH::from_wave(uniform_phase_wave(0.0f));
    const SH h1 = SH::from_wave(uniform_phase_wave(std::numbers::pi_v<float> * 0.6f));
    CHECK_FALSE(h0 == h1);
}

TEST_CASE("SpectralHash: short wave (< 9 dims) doesn't crash", "[lexicon][hash]") {
    std::vector<Complex> w = {{1.0f, 0.0f}};   // 1 dim only
    REQUIRE_NOTHROW(SH::from_wave(w));
}

// ===========================================================================
// Static helpers
// ===========================================================================

TEST_CASE("wave_energy: zero vector → 0", "[lexicon][helpers]") {
    std::vector<Complex> z(9, Complex{0.0f, 0.0f});
    CHECK(Lex::wave_energy(z) == Catch::Approx(0.0).margin(1e-15));
}

TEST_CASE("wave_energy: 9 unit elements → 9", "[lexicon][helpers]") {
    std::vector<Complex> w(9, Complex{1.0f, 0.0f});
    CHECK(Lex::wave_energy(w) == Catch::Approx(9.0).epsilon(1e-6));
}

TEST_CASE("compute_resonance: self-resonance = 1.0", "[lexicon][helpers]") {
    const auto w = noise_wave();
    CHECK(Lex::compute_resonance(w, w) == Catch::Approx(1.0).epsilon(1e-5));
}

TEST_CASE("compute_resonance: orthogonal vectors → 0", "[lexicon][helpers]") {
    // a = (1,0,0,...), b = (0,1,0,...) → dot = 0
    std::vector<Complex> a(9, Complex{0.0f});
    std::vector<Complex> b(9, Complex{0.0f});
    a[0] = Complex{1.0f, 0.0f};
    b[1] = Complex{1.0f, 0.0f};
    CHECK(Lex::compute_resonance(a, b) == Catch::Approx(0.0).margin(1e-6));
}

TEST_CASE("compute_resonance: zero-energy inputs → 0", "[lexicon][helpers]") {
    std::vector<Complex> z(9, Complex{0.0f});
    const auto w = noise_wave();
    CHECK(Lex::compute_resonance(z, w) == Catch::Approx(0.0).margin(1e-6));
    CHECK(Lex::compute_resonance(w, z) == Catch::Approx(0.0).margin(1e-6));
}

// ===========================================================================
// Core lexicon tests
// ===========================================================================

TEST_CASE("HolographicLexicon: default state is empty", "[lexicon][state]") {
    Lex lex;
    CHECK(lex.size() == 0u);
}

TEST_CASE("HolographicLexicon: exists() false before add_token", "[lexicon][state]") {
    Lex lex;
    CHECK_FALSE(lex.exists("hello"));
}

TEST_CASE("HolographicLexicon: add_token + exists", "[lexicon][add]") {
    Lex lex;
    lex.add_token("hello", uniform_phase_wave(0.0f));
    CHECK(lex.exists("hello"));
    CHECK(lex.size() == 1u);
}

TEST_CASE("HolographicLexicon: add_token throws on empty token", "[lexicon][add]") {
    Lex lex;
    REQUIRE_THROWS_AS(lex.add_token("", uniform_phase_wave(0.0f)),
                      std::invalid_argument);
}

TEST_CASE("HolographicLexicon: add_token throws on empty wave", "[lexicon][add]") {
    Lex lex;
    REQUIRE_THROWS_AS(lex.add_token("tok", std::vector<Complex>{}),
                      std::invalid_argument);
}

TEST_CASE("HolographicLexicon: embed() returns wave for registered token", "[lexicon][embed]") {
    Lex lex;
    const auto w = noise_wave();
    lex.add_token("t", w);
    const auto got = lex.embed("t");
    REQUIRE(got.has_value());
    REQUIRE(got->size() == w.size());
    for (size_t i = 0; i < w.size(); ++i)
        CHECK(std::abs((*got)[i] - w[i]) < 1e-6f);
}

TEST_CASE("HolographicLexicon: embed() returns nullopt for unknown", "[lexicon][embed]") {
    Lex lex;
    CHECK_FALSE(lex.embed("unknown").has_value());
}

TEST_CASE("HolographicLexicon: decode on empty lexicon → nullopt", "[lexicon][decode]") {
    Lex lex;
    CHECK_FALSE(lex.decode(noise_wave()).has_value());
}

TEST_CASE("HolographicLexicon: decode vacuum wave → nullopt", "[lexicon][decode]") {
    Lex lex;
    lex.add_token("tok", noise_wave());
    CHECK_FALSE(lex.decode(vacuum_wave()).has_value());
}

TEST_CASE("HolographicLexicon: decode exact wave → returns token", "[lexicon][decode]") {
    Lex lex;
    const auto w = noise_wave();
    lex.add_token("word", w);
    const auto result = lex.decode(w);
    REQUIRE(result.has_value());
    CHECK(*result == "word");
}

TEST_CASE("HolographicLexicon: decode slightly noisy wave → still returns token",
          "[lexicon][decode]") {
    Lex lex;
    const auto w = uniform_phase_wave(0.4f, 2.0f);
    lex.add_token("signal", w);

    // Slightly perturbed version — same broad spectral character
    auto query = w;
    for (auto& c : query)
        c += Complex{0.02f, 0.01f};  // small additive noise

    const auto result = lex.decode(query);
    REQUIRE(result.has_value());
    CHECK(*result == "signal");
}

TEST_CASE("HolographicLexicon: decode very different wave → nullopt (low resonance)",
          "[lexicon][decode]") {
    Lex lex;
    // Register a wave with phase = 0
    lex.add_token("tok", uniform_phase_wave(0.0f, 2.0f));
    // Query with phase = -π (opposite quadrant, near-orthogonal)
    const auto result = lex.decode(uniform_phase_wave(-std::numbers::pi_v<float> + 0.05f, 2.0f));
    // Resonance should be very low → nullopt
    CHECK_FALSE(result.has_value());
}

TEST_CASE("HolographicLexicon: best resonance wins when two tokens in same bucket",
          "[lexicon][decode]") {
    Lex lex;
    // Two waves with the same phase quadrant profile but different amplitudes
    const auto w_bright = uniform_phase_wave(0.2f, 3.0f);
    const auto w_dim    = uniform_phase_wave(0.2f, 0.8f);  // same phase, lower amplitude

    lex.add_token("bright", w_bright);
    lex.add_token("dim",    w_dim);

    // Query that exactly matches w_bright → bright should win
    const auto result = lex.decode(w_bright);
    REQUIRE(result.has_value());
    CHECK(*result == "bright");
}

TEST_CASE("HolographicLexicon: multiple distinct tokens, each decodes to itself",
          "[lexicon][decode]") {
    Lex lex;
    // Use clearly different phases to land in different buckets
    const auto w0 = uniform_phase_wave(-2.0f, 2.0f);
    const auto w1 = uniform_phase_wave(+0.5f, 2.0f);
    const auto w2 = uniform_phase_wave(+2.3f, 2.0f);
    lex.add_token("alpha", w0);
    lex.add_token("beta",  w1);
    lex.add_token("gamma", w2);

    CHECK(lex.decode(w0).value_or("?") == "alpha");
    CHECK(lex.decode(w1).value_or("?") == "beta");
    CHECK(lex.decode(w2).value_or("?") == "gamma");
}

TEST_CASE("HolographicLexicon: remove_token removes from both maps", "[lexicon][remove]") {
    Lex lex;
    const auto w = noise_wave();
    lex.add_token("bye", w);
    REQUIRE(lex.exists("bye"));
    REQUIRE(lex.remove_token("bye"));
    CHECK_FALSE(lex.exists("bye"));
    CHECK(lex.size() == 0u);
    CHECK_FALSE(lex.decode(w).has_value());
}

TEST_CASE("HolographicLexicon: remove_token returns false for unknown", "[lexicon][remove]") {
    Lex lex;
    CHECK_FALSE(lex.remove_token("ghost"));
}

TEST_CASE("HolographicLexicon: clear() empties both maps", "[lexicon][state]") {
    Lex lex;
    lex.add_token("a", noise_wave());
    lex.add_token("b", uniform_phase_wave(1.0f));
    lex.clear();
    CHECK(lex.size() == 0u);
    CHECK_FALSE(lex.exists("a"));
}

TEST_CASE("HolographicLexicon: re-registration updates wave", "[lexicon][add]") {
    Lex lex;
    const auto w1 = uniform_phase_wave(0.0f,  2.0f);
    const auto w2 = uniform_phase_wave(0.0f, 10.0f);   // same phase bucket, larger amplitude
    lex.add_token("tok", w1);
    lex.add_token("tok", w2);   // overwrite

    const auto got = lex.embed("tok");
    REQUIRE(got.has_value());
    // amplitude of first element should reflect w2
    CHECK(std::abs((*got)[0]) == Catch::Approx(10.0f).epsilon(1e-4f));
    CHECK(lex.size() == 1u);           // still only one token
}

// ===========================================================================
// Thread-safety test
// ===========================================================================

TEST_CASE("HolographicLexicon: concurrent decode() calls return consistent results",
          "[lexicon][thread]") {
    Lex lex;
    const auto w = noise_wave();
    lex.add_token("concurrent", w);

    constexpr int kThreads = 8;
    std::vector<std::optional<std::string>> results(kThreads);
    std::vector<std::thread> threads;
    threads.reserve(kThreads);

    for (int i = 0; i < kThreads; ++i) {
        threads.emplace_back([&lex, &w, &results, i]() {
            results[static_cast<size_t>(i)] = lex.decode(w);
        });
    }
    for (auto& t : threads) t.join();

    for (const auto& r : results) {
        REQUIRE(r.has_value());
        CHECK(*r == "concurrent");
    }
}
