#pragma once
/**
 * @file spectral_filter.hpp
 * @brief Phase 127 — SpectralFilter: frequency-domain band separator
 *
 * Separates the 9 irrational-ratio (π·φⁿ) emitters into five functional
 * bands and provides reconstruction, energy analysis, and bandpass ops.
 *
 * Band layout (zero-indexed in the 9-element spectrum):
 *
 *   CONTEXT  [0,1,2]  ← Emitters 1–3  ( ~5–13 Hz)   Mamba domain
 *   BRIDGE   [3]      ← Emitter  4    (~21 Hz)        Coupling
 *   DETAIL   [4,5,6]  ← Emitters 5–7  (~34–91 Hz)    Transformer domain
 *   SOCIAL   [7]      ← Emitter  8                    IRSP communication
 *   SYNC     [8]      ← Emitter  9                    Synchronisation / timing
 *
 * For bands with fewer than 3 active emitters (BRIDGE, SOCIAL, SYNC) the
 * 3-element output slots [1] and [2] are zero-padded on extract and ignored
 * on reconstruct.
 *
 * No external dependencies — purely std::{array, complex, cmath}.
 */

#include <array>
#include <complex>
#include <cmath>

namespace nikola::cognitive {

// ---------------------------------------------------------------------------
// SpectralBand
// ---------------------------------------------------------------------------

enum class SpectralBand {
    CONTEXT = 0,   ///< Emitters 1-3  (~5–13 Hz)  — Mamba domain
    BRIDGE  = 1,   ///< Emitter  4    (~21 Hz)     — Coupling
    DETAIL  = 2,   ///< Emitters 5-7  (~34–91 Hz)  — Transformer domain
    SOCIAL  = 3,   ///< Emitter  8                 — IRSP communication
    SYNC    = 4,   ///< Emitter  9                 — Synchronisation
};

// ---------------------------------------------------------------------------
// SpectralFilter
// ---------------------------------------------------------------------------

class SpectralFilter {
public:

    // --- Band index mapping (static, constexpr-accessible) ------------------

    /**
     * @brief Returns the starting index in a 9-element spectrum for `band`.
     *   CONTEXT→0  BRIDGE→3  DETAIL→4  SOCIAL→7  SYNC→8
     */
    static constexpr int band_start(SpectralBand band) noexcept {
        switch (band) {
        case SpectralBand::CONTEXT: return 0;
        case SpectralBand::BRIDGE:  return 3;
        case SpectralBand::DETAIL:  return 4;
        case SpectralBand::SOCIAL:  return 7;
        case SpectralBand::SYNC:    return 8;
        }
        return 0;
    }

    /**
     * @brief How many real emitters the band owns (1 or 3).
     */
    static constexpr int band_width(SpectralBand band) noexcept {
        switch (band) {
        case SpectralBand::CONTEXT: return 3;
        case SpectralBand::BRIDGE:  return 1;
        case SpectralBand::DETAIL:  return 3;
        case SpectralBand::SOCIAL:  return 1;
        case SpectralBand::SYNC:    return 1;
        }
        return 1;
    }

    // --- Primary API --------------------------------------------------------

    /**
     * @brief Extract a 3-slot view of one spectral band.
     *
     * Slots beyond `band_width(band)` are zero-padded.
     */
    std::array<std::complex<double>, 3> extract_band(
        const std::array<std::complex<double>, 9>& spectrum,
        SpectralBand band) const;

    /**
     * @brief Reconstruct full 9-element spectrum from CONTEXT + DETAIL bands.
     *
     * BRIDGE, SOCIAL, SYNC positions are set to zero; use full_reconstruct()
     * if those bands need to be preserved.
     */
    std::array<std::complex<double>, 9> reconstruct(
        const std::array<std::complex<double>, 3>& context_band,
        const std::array<std::complex<double>, 3>& detail_band) const;

    /**
     * @brief Full reconstruction from all five bands.
     *
     * For single-emitter bands (BRIDGE/SOCIAL/SYNC) only slot [0] is used.
     */
    std::array<std::complex<double>, 9> full_reconstruct(
        const std::array<std::complex<double>, 3>& context_band,
        const std::array<std::complex<double>, 3>& bridge_band,
        const std::array<std::complex<double>, 3>& detail_band,
        const std::array<std::complex<double>, 3>& social_band,
        const std::array<std::complex<double>, 3>& sync_band)  const;

    /**
     * @brief Apply a gain factor to one band and return the modified spectrum.
     *
     * All other bands are unchanged.
     */
    std::array<std::complex<double>, 9> apply_gain(
        const std::array<std::complex<double>, 9>& spectrum,
        SpectralBand band,
        double gain) const;

    /**
     * @brief Zero out all bands outside [lower, upper] (inclusive).
     *
     * Band ordering: CONTEXT < BRIDGE < DETAIL < SOCIAL < SYNC.
     */
    std::array<std::complex<double>, 9> bandpass(
        const std::array<std::complex<double>, 9>& spectrum,
        SpectralBand lower,
        SpectralBand upper) const;

    /**
     * @brief Scale spectrum so the maximum magnitude is 1.0.
     * Returns the input unchanged if all values are zero.
     */
    std::array<std::complex<double>, 9> normalise(
        const std::array<std::complex<double>, 9>& spectrum) const;

    // --- Energy / analysis --------------------------------------------------

    /**
     * @brief Sum of squared magnitudes: Σ |z_i|².
     */
    static double band_energy(
        const std::array<std::complex<double>, 3>& band) noexcept;

    /**
     * @brief Square root of band_energy — RMS magnitude.
     */
    static double band_magnitude(
        const std::array<std::complex<double>, 3>& band) noexcept;

    /**
     * @brief Mean phase angle (argument) of non-zero elements, in radians.
     * Returns 0.0 if all elements are zero.
     */
    static double band_phase_mean(
        const std::array<std::complex<double>, 3>& band) noexcept;

    /**
     * @brief Returns the SpectralBand with the highest energy in `spectrum`.
     */
    SpectralBand dominant_band(
        const std::array<std::complex<double>, 9>& spectrum) const;

    // --- Stats --------------------------------------------------------------

    struct BandStats {
        double energy;    ///< sum |z|²
        double magnitude; ///< sqrt(energy) = RMS magnitude
        double phase;     ///< mean phase angle (radians)
    };

    struct Stats {
        BandStats context;
        BandStats bridge;
        BandStats detail;
        BandStats social;
        BandStats sync;
        double    total_energy = 0.0;
        SpectralBand dominant  = SpectralBand::CONTEXT;
    };

    Stats compute_stats(
        const std::array<std::complex<double>, 9>& spectrum) const;
};

} // namespace nikola::cognitive
