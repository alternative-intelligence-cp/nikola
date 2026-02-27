/**
 * @file phase127_spectral_filter_test.cpp
 * @brief Phase 127 — SpectralFilter unit tests
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/cognitive/spectral_filter.hpp>

#include <array>
#include <cmath>
#include <complex>
#include <numbers>

using namespace nikola::cognitive;
using Catch::Approx;
using cx = std::complex<double>;

// Helper: build a spectrum where each index i has value (i+1, 0)
static std::array<cx, 9> make_index_spectrum() {
    std::array<cx, 9> s{};
    for (int i = 0; i < 9; ++i) {
        s[static_cast<size_t>(i)] = cx(static_cast<double>(i + 1), 0.0);
    }
    return s;
}

// Helper: zero spectrum
static std::array<cx, 9> zero9() {
    return {};
}

static std::array<cx, 3> zero3() {
    return {};
}

// ---------------------------------------------------------------------------
// band_start / band_width (constexpr)
// ---------------------------------------------------------------------------

TEST_CASE("SpectralFilter::band_start — correct indices", "[Phase127][static]") {
    REQUIRE(SpectralFilter::band_start(SpectralBand::CONTEXT) == 0);
    REQUIRE(SpectralFilter::band_start(SpectralBand::BRIDGE)  == 3);
    REQUIRE(SpectralFilter::band_start(SpectralBand::DETAIL)  == 4);
    REQUIRE(SpectralFilter::band_start(SpectralBand::SOCIAL)  == 7);
    REQUIRE(SpectralFilter::band_start(SpectralBand::SYNC)    == 8);
}

TEST_CASE("SpectralFilter::band_width — correct widths", "[Phase127][static]") {
    REQUIRE(SpectralFilter::band_width(SpectralBand::CONTEXT) == 3);
    REQUIRE(SpectralFilter::band_width(SpectralBand::BRIDGE)  == 1);
    REQUIRE(SpectralFilter::band_width(SpectralBand::DETAIL)  == 3);
    REQUIRE(SpectralFilter::band_width(SpectralBand::SOCIAL)  == 1);
    REQUIRE(SpectralFilter::band_width(SpectralBand::SYNC)    == 1);
}

// ---------------------------------------------------------------------------
// extract_band
// ---------------------------------------------------------------------------

TEST_CASE("SpectralFilter::extract_band — CONTEXT extracts [0,1,2]",
          "[Phase127][extract]") {
    SpectralFilter sf;
    const auto spectrum = make_index_spectrum();
    const auto band = sf.extract_band(spectrum, SpectralBand::CONTEXT);

    REQUIRE(band[0] == cx(1.0, 0.0));
    REQUIRE(band[1] == cx(2.0, 0.0));
    REQUIRE(band[2] == cx(3.0, 0.0));
}

TEST_CASE("SpectralFilter::extract_band — BRIDGE extracts [3] padded",
          "[Phase127][extract]") {
    SpectralFilter sf;
    const auto spectrum = make_index_spectrum();
    const auto band = sf.extract_band(spectrum, SpectralBand::BRIDGE);

    REQUIRE(band[0] == cx(4.0, 0.0));        // emitter index 3 → value 4
    REQUIRE(band[1] == cx(0.0, 0.0));        // zero-padded
    REQUIRE(band[2] == cx(0.0, 0.0));
}

TEST_CASE("SpectralFilter::extract_band — DETAIL extracts [4,5,6]",
          "[Phase127][extract]") {
    SpectralFilter sf;
    const auto spectrum = make_index_spectrum();
    const auto band = sf.extract_band(spectrum, SpectralBand::DETAIL);

    REQUIRE(band[0] == cx(5.0, 0.0));
    REQUIRE(band[1] == cx(6.0, 0.0));
    REQUIRE(band[2] == cx(7.0, 0.0));
}

TEST_CASE("SpectralFilter::extract_band — SOCIAL extracts [7] padded",
          "[Phase127][extract]") {
    SpectralFilter sf;
    const auto spectrum = make_index_spectrum();
    const auto band = sf.extract_band(spectrum, SpectralBand::SOCIAL);

    REQUIRE(band[0] == cx(8.0, 0.0));
    REQUIRE(band[1] == cx(0.0, 0.0));
    REQUIRE(band[2] == cx(0.0, 0.0));
}

TEST_CASE("SpectralFilter::extract_band — SYNC extracts [8] padded",
          "[Phase127][extract]") {
    SpectralFilter sf;
    const auto spectrum = make_index_spectrum();
    const auto band = sf.extract_band(spectrum, SpectralBand::SYNC);

    REQUIRE(band[0] == cx(9.0, 0.0));
    REQUIRE(band[1] == cx(0.0, 0.0));
    REQUIRE(band[2] == cx(0.0, 0.0));
}

// ---------------------------------------------------------------------------
// reconstruct
// ---------------------------------------------------------------------------

TEST_CASE("SpectralFilter::reconstruct — context + detail placed correctly",
          "[Phase127][reconstruct]") {
    SpectralFilter sf;

    std::array<cx, 3> ctx  = {cx(1.0,0), cx(2.0,0), cx(3.0,0)};
    std::array<cx, 3> det  = {cx(5.0,0), cx(6.0,0), cx(7.0,0)};

    const auto out = sf.reconstruct(ctx, det);

    REQUIRE(out[0] == cx(1.0, 0));
    REQUIRE(out[1] == cx(2.0, 0));
    REQUIRE(out[2] == cx(3.0, 0));
    REQUIRE(out[3] == cx(0.0, 0));   // BRIDGE zeroed
    REQUIRE(out[4] == cx(5.0, 0));
    REQUIRE(out[5] == cx(6.0, 0));
    REQUIRE(out[6] == cx(7.0, 0));
    REQUIRE(out[7] == cx(0.0, 0));   // SOCIAL zeroed
    REQUIRE(out[8] == cx(0.0, 0));   // SYNC zeroed
}

TEST_CASE("SpectralFilter::reconstruct — roundtrip extract + reconstruct",
          "[Phase127][reconstruct]") {
    SpectralFilter sf;
    const auto spectrum = make_index_spectrum();

    const auto ctx_band = sf.extract_band(spectrum, SpectralBand::CONTEXT);
    const auto det_band = sf.extract_band(spectrum, SpectralBand::DETAIL);
    const auto rebuilt  = sf.reconstruct(ctx_band, det_band);

    // Only CONTEXT + DETAIL slots should match; others are zeroed
    REQUIRE(rebuilt[0] == spectrum[0]);
    REQUIRE(rebuilt[1] == spectrum[1]);
    REQUIRE(rebuilt[2] == spectrum[2]);
    REQUIRE(rebuilt[4] == spectrum[4]);
    REQUIRE(rebuilt[5] == spectrum[5]);
    REQUIRE(rebuilt[6] == spectrum[6]);
}

// ---------------------------------------------------------------------------
// full_reconstruct
// ---------------------------------------------------------------------------

TEST_CASE("SpectralFilter::full_reconstruct — all 9 slots populated",
          "[Phase127][reconstruct]") {
    SpectralFilter sf;

    std::array<cx,3> ctx = {cx(1,0),cx(2,0),cx(3,0)};
    std::array<cx,3> bri = {cx(4,0),cx(0,0),cx(0,0)};
    std::array<cx,3> det = {cx(5,0),cx(6,0),cx(7,0)};
    std::array<cx,3> soc = {cx(8,0),cx(0,0),cx(0,0)};
    std::array<cx,3> syn = {cx(9,0),cx(0,0),cx(0,0)};

    const auto out = sf.full_reconstruct(ctx, bri, det, soc, syn);

    for (int i = 0; i < 9; ++i) {
        REQUIRE(out[static_cast<size_t>(i)].real() == Approx(i + 1));
        REQUIRE(out[static_cast<size_t>(i)].imag() == Approx(0.0));
    }
}

TEST_CASE("SpectralFilter::full_reconstruct — full roundtrip",
          "[Phase127][reconstruct]") {
    SpectralFilter sf;
    const auto spectrum = make_index_spectrum();

    const auto ctx = sf.extract_band(spectrum, SpectralBand::CONTEXT);
    const auto bri = sf.extract_band(spectrum, SpectralBand::BRIDGE);
    const auto det = sf.extract_band(spectrum, SpectralBand::DETAIL);
    const auto soc = sf.extract_band(spectrum, SpectralBand::SOCIAL);
    const auto syn = sf.extract_band(spectrum, SpectralBand::SYNC);

    const auto rebuilt = sf.full_reconstruct(ctx, bri, det, soc, syn);

    // Every original emitter value should be preserved
    for (size_t i = 0; i < 9; ++i) {
        REQUIRE(rebuilt[i].real() == Approx(spectrum[i].real()));
        REQUIRE(rebuilt[i].imag() == Approx(spectrum[i].imag()));
    }
}

// ---------------------------------------------------------------------------
// apply_gain
// ---------------------------------------------------------------------------

TEST_CASE("SpectralFilter::apply_gain — scales target band only",
          "[Phase127][gain]") {
    SpectralFilter sf;
    const auto spectrum = make_index_spectrum();

    const auto out = sf.apply_gain(spectrum, SpectralBand::DETAIL, 2.0);

    // DETAIL positions [4,5,6] doubled
    REQUIRE(out[4].real() == Approx(5.0 * 2.0));
    REQUIRE(out[5].real() == Approx(6.0 * 2.0));
    REQUIRE(out[6].real() == Approx(7.0 * 2.0));

    // Other positions untouched
    REQUIRE(out[0].real() == Approx(1.0));
    REQUIRE(out[3].real() == Approx(4.0));
    REQUIRE(out[7].real() == Approx(8.0));
}

TEST_CASE("SpectralFilter::apply_gain — zero gain silences band",
          "[Phase127][gain]") {
    SpectralFilter sf;
    const auto spectrum = make_index_spectrum();
    const auto out = sf.apply_gain(spectrum, SpectralBand::CONTEXT, 0.0);

    REQUIRE(out[0] == cx(0.0, 0.0));
    REQUIRE(out[1] == cx(0.0, 0.0));
    REQUIRE(out[2] == cx(0.0, 0.0));
    REQUIRE(out[3].real() == Approx(4.0));   // BRIDGE untouched
}

// ---------------------------------------------------------------------------
// bandpass
// ---------------------------------------------------------------------------

TEST_CASE("SpectralFilter::bandpass — keeps CONTEXT..DETAIL range",
          "[Phase127][bandpass]") {
    SpectralFilter sf;
    const auto spectrum = make_index_spectrum();

    const auto out = sf.bandpass(spectrum,
                                  SpectralBand::CONTEXT,
                                  SpectralBand::DETAIL);

    // [0..6] kept, [7,8] zeroed
    for (int i = 0; i <= 6; ++i) {
        REQUIRE(out[static_cast<size_t>(i)] == spectrum[static_cast<size_t>(i)]);
    }
    REQUIRE(out[7] == cx(0.0, 0.0));
    REQUIRE(out[8] == cx(0.0, 0.0));
}

TEST_CASE("SpectralFilter::bandpass — single band passed",
          "[Phase127][bandpass]") {
    SpectralFilter sf;
    const auto spectrum = make_index_spectrum();

    const auto out = sf.bandpass(spectrum,
                                  SpectralBand::BRIDGE,
                                  SpectralBand::BRIDGE);

    REQUIRE(out[3] == cx(4.0, 0.0));   // BRIDGE kept
    REQUIRE(out[0] == cx(0.0, 0.0));   // others zeroed
    REQUIRE(out[4] == cx(0.0, 0.0));
}

// ---------------------------------------------------------------------------
// normalise
// ---------------------------------------------------------------------------

TEST_CASE("SpectralFilter::normalise — max magnitude becomes 1.0",
          "[Phase127][normalise]") {
    SpectralFilter sf;
    const auto spectrum = make_index_spectrum();  // max real value = 9

    const auto out = sf.normalise(spectrum);

    REQUIRE(out[8].real() == Approx(1.0));   // 9/9 = 1.0
    REQUIRE(out[0].real() == Approx(1.0 / 9.0));
    REQUIRE(out[4].real() == Approx(5.0 / 9.0));
}

TEST_CASE("SpectralFilter::normalise — zero spectrum unchanged",
          "[Phase127][normalise]") {
    SpectralFilter sf;
    const auto zero = zero9();
    const auto out  = sf.normalise(zero);

    for (const auto& z : out) {
        REQUIRE(std::abs(z) == Approx(0.0));
    }
}

// ---------------------------------------------------------------------------
// band_energy / band_magnitude / band_phase_mean
// ---------------------------------------------------------------------------

TEST_CASE("SpectralFilter::band_energy — correct squared sum",
          "[Phase127][energy]") {
    // band = {3+4i, 0, 0} → |3+4i|² = 25
    std::array<cx,3> band = {cx(3.0, 4.0), cx(0,0), cx(0,0)};
    REQUIRE(SpectralFilter::band_energy(band) == Approx(25.0));
}

TEST_CASE("SpectralFilter::band_magnitude — sqrt of energy",
          "[Phase127][energy]") {
    std::array<cx,3> band = {cx(3.0, 4.0), cx(0,0), cx(0,0)};
    // energy=25, magnitude = sqrt(25) = 5
    REQUIRE(SpectralFilter::band_magnitude(band) == Approx(5.0));
}

TEST_CASE("SpectralFilter::band_energy — all elements contribute",
          "[Phase127][energy]") {
    // {1+0i, 2+0i, 3+0i} → 1 + 4 + 9 = 14
    std::array<cx,3> band = {cx(1,0), cx(2,0), cx(3,0)};
    REQUIRE(SpectralFilter::band_energy(band) == Approx(14.0));
}

TEST_CASE("SpectralFilter::band_energy — zero band",
          "[Phase127][energy]") {
    REQUIRE(SpectralFilter::band_energy(zero3()) == Approx(0.0));
    REQUIRE(SpectralFilter::band_magnitude(zero3()) == Approx(0.0));
}

TEST_CASE("SpectralFilter::band_phase_mean — real-only input is 0",
          "[Phase127][phase]") {
    std::array<cx,3> band = {cx(1,0), cx(2,0), cx(3,0)};
    REQUIRE(SpectralFilter::band_phase_mean(band) == Approx(0.0));
}

TEST_CASE("SpectralFilter::band_phase_mean — mean of known angles",
          "[Phase127][phase]") {
    [[maybe_unused]] const double pi = std::numbers::pi;
    // Two elements at π/2 and -π/2, one at 0
    std::array<cx,3> band = {
        cx(0.0,  1.0),    // arg = +π/2
        cx(0.0, -1.0),    // arg = -π/2
        cx(1.0,  0.0),    // arg = 0
    };
    // mean = (π/2 + (-π/2) + 0) / 3 = 0
    const double mean = SpectralFilter::band_phase_mean(band);
    REQUIRE(mean == Approx(0.0).margin(1e-9));
}

TEST_CASE("SpectralFilter::band_phase_mean — zero band returns 0",
          "[Phase127][phase]") {
    REQUIRE(SpectralFilter::band_phase_mean(zero3()) == Approx(0.0));
}

// ---------------------------------------------------------------------------
// dominant_band
// ---------------------------------------------------------------------------

TEST_CASE("SpectralFilter::dominant_band — CONTEXT has most energy",
          "[Phase127][dominant]") {
    SpectralFilter sf;
    std::array<cx,9> s{};
    s[0] = cx(10, 0);
    s[1] = cx(10, 0);
    s[2] = cx(10, 0);
    // CONTEXT energy = 300, everyone else much lower

    REQUIRE(sf.dominant_band(s) == SpectralBand::CONTEXT);
}

TEST_CASE("SpectralFilter::dominant_band — SYNC wins when strongest",
          "[Phase127][dominant]") {
    SpectralFilter sf;
    std::array<cx,9> s{};
    s[8] = cx(100, 0);   // SYNC index

    REQUIRE(sf.dominant_band(s) == SpectralBand::SYNC);
}

TEST_CASE("SpectralFilter::dominant_band — DETAIL wins",
          "[Phase127][dominant]") {
    SpectralFilter sf;
    std::array<cx,9> s{};
    s[4] = cx(5, 0);
    s[5] = cx(5, 0);
    s[6] = cx(5, 0);    // DETAIL energy = 75; others tiny

    REQUIRE(sf.dominant_band(s) == SpectralBand::DETAIL);
}

// ---------------------------------------------------------------------------
// compute_stats
// ---------------------------------------------------------------------------

TEST_CASE("SpectralFilter::compute_stats — total_energy sums all bands",
          "[Phase127][stats]") {
    SpectralFilter sf;
    // All-real spectrum: 1,2,...,9
    const auto spectrum = make_index_spectrum();
    const auto s = sf.compute_stats(spectrum);

    // total = sum i^2 for i=1..9 = 285
    REQUIRE(s.total_energy == Approx(285.0));
}

TEST_CASE("SpectralFilter::compute_stats — CONTEXT energy correct",
          "[Phase127][stats]") {
    SpectralFilter sf;
    // spectrum: [1,2,3,0,0,...] only CONTEXT populated
    std::array<cx,9> spec{};
    spec[0] = cx(1,0); spec[1] = cx(2,0); spec[2] = cx(3,0);
    // energy = 1+4+9 = 14

    const auto s = sf.compute_stats(spec);
    REQUIRE(s.context.energy    == Approx(14.0));
    REQUIRE(s.context.magnitude == Approx(std::sqrt(14.0)));
    REQUIRE(s.bridge.energy     == Approx(0.0));
    REQUIRE(s.dominant          == SpectralBand::CONTEXT);
}

TEST_CASE("SpectralFilter::compute_stats — zero spectrum",
          "[Phase127][stats]") {
    SpectralFilter sf;
    const auto s = sf.compute_stats(zero9());

    REQUIRE(s.total_energy == Approx(0.0));
    REQUIRE(s.context.energy  == Approx(0.0));
    REQUIRE(s.detail.energy   == Approx(0.0));
}
