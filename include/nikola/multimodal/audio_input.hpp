/**
 * @file audio_input.hpp
 * @brief Audio PCM → Nit[128] embedding for HolographicInjector injection.
 *
 * Implements the audio sensory transduction path:
 *
 *   PCM[N] → Goertzel(8 cognitive bands) → amplitudes[8] → Nit[128]
 *
 * Each of the 8 cognitive-band amplitudes maps to a 16-nit subspace via
 * phase-coded cosine expansion:
 *
 *   nit[k*16 + j] = clamp(round(A_k × 4.0 × cos(2π·j/16 + k·π·φ)), -4, +4)
 *
 * This produces orthogonal-ish subspaces per band while mapping amplitude
 * into the balanced-nonary range [-4, +4].  The golden-ratio phase offsets
 * minimise inter-band correlation in the embedding space.
 *
 * Header-only — no separate .cpp needed.
 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <span>
#include <vector>

#include "nikola/foundation/nit.hpp"
#include "nikola/multimodal/cymatic_transduction.hpp"

namespace nikola::multimodal {

// ============================================================================
// Constants
// ============================================================================

/// Number of nit dimensions per cognitive band (128 / 8 = 16).
inline constexpr int NITS_PER_BAND = 16;

/// Total embedding size (must match HolographicInjector's expected 128).
inline constexpr int AUDIO_EMBEDDING_DIM = EMITTER_COUNT * NITS_PER_BAND;

static_assert(AUDIO_EMBEDDING_DIM == 128,
              "Audio embedding dimension must be 128 to match injector");

// ============================================================================
// AudioInput
// ============================================================================

/**
 * @brief Converts PCM audio samples to a 128-dim Nit embedding for torus injection.
 *
 * Static-only utility class.  All methods are pure (no internal state).
 */
class AudioInput {
public:
    /// Normalised power in each of the 8 cognitive bands.
    using BandAmplitudes = std::array<double, EMITTER_COUNT>;

    // ── Band extraction ──────────────────────────────────────────────────

    /**
     * @brief Extract amplitude of each cognitive band via Goertzel.
     *
     * Runs goertzel_amplitude() on all 8 emitter frequencies (E1–E8).
     * If the PCM buffer is empty, returns all zeros.
     *
     * @param pcm          Flat PCM float samples (any sample rate)
     * @param sample_rate  Sample rate in Hz (default: 48 kHz capture rate)
     * @return             8-element array of normalised amplitudes ≥ 0
     */
    [[nodiscard]] static BandAmplitudes extract_bands(
            std::span<const float> pcm,
            double sample_rate = static_cast<double>(CAPTURE_RATE_HZ))
    {
        BandAmplitudes amps{};
        if (pcm.empty()) return amps;

        for (int i = 0; i < EMITTER_COUNT; ++i) {
            amps[static_cast<size_t>(i)] =
                goertzel_amplitude(pcm, EMITTER_FREQS[static_cast<size_t>(i)], sample_rate);
        }
        return amps;
    }

    // ── Nit embedding ────────────────────────────────────────────────────

    /**
     * @brief Map 8 band amplitudes to a 128-dim Nit embedding.
     *
     * Each band k ∈ [0, 7] with amplitude A_k occupies nits [k*16 .. k*16+15]:
     *
     *   nit[k*16 + j] = clamp(round(A_k × 4.0 × cos(2π·j/16 + k·π·φ)), -4, +4)
     *
     * This phase-coded expansion creates a unique "fingerprint" per band.
     * Golden-ratio phase offsets (k × π × φ) minimise inter-band correlation.
     *
     * Amplitudes are clamped to [0, 1] before encoding.  Amplitudes > 1 from
     * very loud signals are truncated to keep the embedding within Nit range.
     *
     * @param bands  8 amplitudes from extract_bands() (≥ 0)
     * @return       128-element balanced-nonary vector in [-4, +4]
     */
    [[nodiscard]] static std::vector<foundation::Nit> embed(const BandAmplitudes& bands)
    {
        std::vector<foundation::Nit> nits(AUDIO_EMBEDDING_DIM, foundation::NIT_ZERO);

        for (int k = 0; k < EMITTER_COUNT; ++k) {
            // Clamp amplitude to [0, 1] for safe Nit encoding
            const double amp = std::clamp(bands[static_cast<size_t>(k)], 0.0, 1.0);
            const double phase_offset =
                static_cast<double>(k) * std::numbers::pi * GOLDEN_RATIO;

            for (int j = 0; j < NITS_PER_BAND; ++j) {
                const double angle =
                    2.0 * std::numbers::pi * static_cast<double>(j) / NITS_PER_BAND
                    + phase_offset;
                const double val = amp * 4.0 * std::cos(angle);
                const int clamped = std::clamp(
                    static_cast<int>(std::round(val)),
                    static_cast<int>(foundation::NIT_MIN),
                    static_cast<int>(foundation::NIT_MAX));
                nits[static_cast<size_t>(k * NITS_PER_BAND + j)] =
                    static_cast<foundation::Nit>(clamped);
            }
        }
        return nits;
    }

    // ── Full pipeline ────────────────────────────────────────────────────

    /**
     * @brief End-to-end PCM → Nit[128] pipeline.
     *
     *   PCM → Goertzel(8 bands) → phase-coded Nit embedding
     *
     * @param pcm          Flat PCM float samples
     * @param sample_rate  Sample rate in Hz (default: 48 kHz)
     * @return             128-element balanced-nonary vector
     */
    [[nodiscard]] static std::vector<foundation::Nit> process(
            std::span<const float> pcm,
            double sample_rate = static_cast<double>(CAPTURE_RATE_HZ))
    {
        return embed(extract_bands(pcm, sample_rate));
    }
};

} // namespace nikola::multimodal
