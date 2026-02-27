/**
 * @file spectral_filter.cpp
 * @brief Phase 127 — SpectralFilter implementation
 */

#include <nikola/cognitive/spectral_filter.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace nikola::cognitive {

// ---------------------------------------------------------------------------
// extract_band
// ---------------------------------------------------------------------------

std::array<std::complex<double>, 3> SpectralFilter::extract_band(
    const std::array<std::complex<double>, 9>& spectrum,
    SpectralBand band) const {

    const int start = band_start(band);
    const int width = band_width(band);

    std::array<std::complex<double>, 3> out{};  // zero-initialised

    for (int i = 0; i < width; ++i) {
        out[static_cast<size_t>(i)] = spectrum[static_cast<size_t>(start + i)];
    }
    return out;
}

// ---------------------------------------------------------------------------
// reconstruct  (context + detail only)
// ---------------------------------------------------------------------------

std::array<std::complex<double>, 9> SpectralFilter::reconstruct(
    const std::array<std::complex<double>, 3>& context_band,
    const std::array<std::complex<double>, 3>& detail_band) const {

    std::array<std::complex<double>, 9> out{};

    // CONTEXT → [0,1,2]
    out[0] = context_band[0];
    out[1] = context_band[1];
    out[2] = context_band[2];

    // BRIDGE  → [3]  (zero)
    // DETAIL  → [4,5,6]
    out[4] = detail_band[0];
    out[5] = detail_band[1];
    out[6] = detail_band[2];

    // SOCIAL → [7]  (zero)
    // SYNC   → [8]  (zero)

    return out;
}

// ---------------------------------------------------------------------------
// full_reconstruct  (all five bands)
// ---------------------------------------------------------------------------

std::array<std::complex<double>, 9> SpectralFilter::full_reconstruct(
    const std::array<std::complex<double>, 3>& context_band,
    const std::array<std::complex<double>, 3>& bridge_band,
    const std::array<std::complex<double>, 3>& detail_band,
    const std::array<std::complex<double>, 3>& social_band,
    const std::array<std::complex<double>, 3>& sync_band) const {

    std::array<std::complex<double>, 9> out{};

    // CONTEXT [0,1,2]
    out[0] = context_band[0];
    out[1] = context_band[1];
    out[2] = context_band[2];

    // BRIDGE [3] — single emitter; only [0] used
    out[3] = bridge_band[0];

    // DETAIL [4,5,6]
    out[4] = detail_band[0];
    out[5] = detail_band[1];
    out[6] = detail_band[2];

    // SOCIAL [7] — single emitter
    out[7] = social_band[0];

    // SYNC [8] — single emitter
    out[8] = sync_band[0];

    return out;
}

// ---------------------------------------------------------------------------
// apply_gain
// ---------------------------------------------------------------------------

std::array<std::complex<double>, 9> SpectralFilter::apply_gain(
    const std::array<std::complex<double>, 9>& spectrum,
    SpectralBand band,
    double gain) const {

    auto out = spectrum;
    const int start = band_start(band);
    const int width = band_width(band);

    for (int i = 0; i < width; ++i) {
        out[static_cast<size_t>(start + i)] *= gain;
    }
    return out;
}

// ---------------------------------------------------------------------------
// bandpass
// ---------------------------------------------------------------------------

std::array<std::complex<double>, 9> SpectralFilter::bandpass(
    const std::array<std::complex<double>, 9>& spectrum,
    SpectralBand lower,
    SpectralBand upper) const {

    const int lb = static_cast<int>(lower);
    const int ub = static_cast<int>(upper);

    std::array<std::complex<double>, 9> out{};

    // Enumerate all bands; copy those within [lower, upper]
    constexpr SpectralBand all_bands[] = {
        SpectralBand::CONTEXT,
        SpectralBand::BRIDGE,
        SpectralBand::DETAIL,
        SpectralBand::SOCIAL,
        SpectralBand::SYNC,
    };

    for (const auto b : all_bands) {
        const int bi = static_cast<int>(b);
        if (bi < lb || bi > ub) continue;

        const int start = band_start(b);
        const int width = band_width(b);
        for (int i = 0; i < width; ++i) {
            out[static_cast<size_t>(start + i)] =
                spectrum[static_cast<size_t>(start + i)];
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// normalise
// ---------------------------------------------------------------------------

std::array<std::complex<double>, 9> SpectralFilter::normalise(
    const std::array<std::complex<double>, 9>& spectrum) const {

    double max_mag = 0.0;
    for (const auto& z : spectrum) {
        const double m = std::abs(z);
        if (m > max_mag) max_mag = m;
    }

    if (max_mag == 0.0) return spectrum;  // all-zero: return unchanged

    auto out = spectrum;
    for (auto& z : out) {
        z /= max_mag;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Static energy / analysis helpers
// ---------------------------------------------------------------------------

double SpectralFilter::band_energy(
    const std::array<std::complex<double>, 3>& band) noexcept {

    double e = 0.0;
    for (const auto& z : band) {
        const double m = std::abs(z);
        e += m * m;
    }
    return e;
}

double SpectralFilter::band_magnitude(
    const std::array<std::complex<double>, 3>& band) noexcept {
    return std::sqrt(band_energy(band));
}

double SpectralFilter::band_phase_mean(
    const std::array<std::complex<double>, 3>& band) noexcept {

    double phase_sum = 0.0;
    int    count     = 0;

    for (const auto& z : band) {
        if (std::abs(z) > 1e-15) {
            phase_sum += std::arg(z);
            ++count;
        }
    }
    return count > 0 ? phase_sum / count : 0.0;
}

SpectralBand SpectralFilter::dominant_band(
    const std::array<std::complex<double>, 9>& spectrum) const {

    constexpr SpectralBand all_bands[] = {
        SpectralBand::CONTEXT,
        SpectralBand::BRIDGE,
        SpectralBand::DETAIL,
        SpectralBand::SOCIAL,
        SpectralBand::SYNC,
    };

    SpectralBand best  = SpectralBand::CONTEXT;
    double best_energy = 0.0;

    for (const auto b : all_bands) {
        const auto band = extract_band(spectrum, b);
        const double e  = band_energy(band);
        if (e > best_energy) {
            best_energy = e;
            best        = b;
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

SpectralFilter::Stats SpectralFilter::compute_stats(
    const std::array<std::complex<double>, 9>& spectrum) const {

    Stats s;

    auto make_band_stats = [&](SpectralBand b) -> BandStats {
        const auto band = extract_band(spectrum, b);
        return BandStats{
            band_energy(band),
            band_magnitude(band),
            band_phase_mean(band)
        };
    };

    s.context = make_band_stats(SpectralBand::CONTEXT);
    s.bridge  = make_band_stats(SpectralBand::BRIDGE);
    s.detail  = make_band_stats(SpectralBand::DETAIL);
    s.social  = make_band_stats(SpectralBand::SOCIAL);
    s.sync    = make_band_stats(SpectralBand::SYNC);

    s.total_energy = s.context.energy + s.bridge.energy +
                     s.detail.energy  + s.social.energy + s.sync.energy;
    s.dominant     = dominant_band(spectrum);

    return s;
}

} // namespace nikola::cognitive
