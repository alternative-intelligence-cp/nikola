#pragma once
/**
 * @file   visual_frame_rate.hpp
 * @brief  GAP-018: Visual Cymatics Frame Rate Adaptation
 *
 * Bridges the temporal mismatch between the 1,000 Hz physics engine and
 * 60/120 Hz display hardware.
 *
 * Core insight: do NOT average wave amplitude across physics ticks.
 * Average energy instead: B_acc += |H_t|² at every tick, then
 *
 *   I_out[x,y] = ToneMap( sqrt( B_acc[x,y] / N_ticks ) )
 *
 * where  ToneMap(x) = x / (1 + x)  (Reinhard sigmoid for HDR compression).
 *
 * Averaging amplitude would let fast oscillations cancel to zero ("phase
 * cancellation"), making 200 Hz solitons invisible.  Energy accumulation turns
 * them into coherent motion-blur streaks — the physically correct display.
 *
 * Three subsystems
 * ────────────────
 *  §A  Rate constants & derived parameters
 *  §B  Pure-math functions (tone mapping, normalization, energy accumulation)
 *  §C  Nyquist / aliasing — Chromatic Aberration threshold & shift formula
 *  §D  Stroboscopic mode — phase-locked frame capture
 *  §E  Buffer sizing helpers
 *
 * Namespace : nikola::multimodal
 * C++ std   : C++23
 *
 * §07_multimodal/03_visual_cymatics.md §GAP-018
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace nikola::multimodal {

// ═══════════════════════════════════════════════════════════════════════════
// §A  Rate constants and derived parameters
// ═══════════════════════════════════════════════════════════════════════════

/// Physics integration rate — 1,000 Hz, 1 tick per millisecond.
inline constexpr int VISUAL_PHYSICS_RATE_HZ = 1'000;

/// Standard consumer display refresh rate.
inline constexpr double DISPLAY_RATE_60HZ  =  60.0;

/// High-refresh display rate.
inline constexpr double DISPLAY_RATE_120HZ = 120.0;

/// Target frames per second for V-Sync @ 60 Hz.
inline constexpr double DISPLAY_FRAME_PERIOD_60HZ_MS  = 1000.0 / DISPLAY_RATE_60HZ;   // ≈16.667 ms

/// Target frames per second for V-Sync @ 120 Hz.
inline constexpr double DISPLAY_FRAME_PERIOD_120HZ_MS = 1000.0 / DISPLAY_RATE_120HZ;  // ≈ 8.333 ms

/// Nominal physics ticks accumulated per 60 Hz display frame (floor).
/// Real accumulation window is 16 or 17 ticks to stay phase-locked with V-Sync.
inline constexpr int TICKS_PER_FRAME_60HZ  = VISUAL_PHYSICS_RATE_HZ / static_cast<int>(DISPLAY_RATE_60HZ);   // 16

/// Nominal physics ticks accumulated per 120 Hz display frame (floor).
inline constexpr int TICKS_PER_FRAME_120HZ = VISUAL_PHYSICS_RATE_HZ / static_cast<int>(DISPLAY_RATE_120HZ);  // 8

/// Display Nyquist limit at 60 Hz — highest frequency representable without aliasing.
/// Frequencies above this cannot be shown as distinct flickers; they produce Moiré.
inline constexpr double DISPLAY_NYQUIST_60HZ  = DISPLAY_RATE_60HZ  / 2.0;  // 30.0 Hz

/// Display Nyquist limit at 120 Hz.
inline constexpr double DISPLAY_NYQUIST_120HZ = DISPLAY_RATE_120HZ / 2.0;  // 60.0 Hz

// ═══════════════════════════════════════════════════════════════════════════
// §B  Buffer-count constants (triple-buffer Seqlock architecture)
// ═══════════════════════════════════════════════════════════════════════════

/// Number of framebuffers in the triple-buffer Seqlock system.
///  [0] Accumulation (physics thread writes)
///  [1] Back         (shared, swap target)
///  [2] Front        (GPU scanout)
inline constexpr int TRIPLE_BUFFER_COUNT = 3;

/// Maximum time budget for an atomic Seqlock swap (μs).
inline constexpr int64_t SEQLOCK_SWAP_BUDGET_US = 10;

/// Maximum computation budget per physics tick for energy accumulation (ms).
inline constexpr double ENERGY_ACCUM_BUDGET_MS = 0.1;

// ═══════════════════════════════════════════════════════════════════════════
// §C  Chromatic aberration / aliasing detection
// ═══════════════════════════════════════════════════════════════════════════

/// Maximum chromatic shift in pixels for the highest detectable super-Nyquist activity.
inline constexpr int CHROMATIC_SHIFT_MAX_PIXELS = 50;

// ═══════════════════════════════════════════════════════════════════════════
// §D  Stroboscopic mode
// ═══════════════════════════════════════════════════════════════════════════

/// Emitter index used as strobe trigger — E1, the fundamental (5.083 Hz).
/// Frame is captured when this emitter's global phase crosses zero.
inline constexpr int STROBOSCOPIC_TRIGGER_EMITTER = 1;

/// Phase tolerance (radians) for zero-crossing detection in stroboscopic mode.
inline constexpr double STROBE_PHASE_ZERO_TOLERANCE = 0.1;   // ≈ 5.7°

// ═══════════════════════════════════════════════════════════════════════════
// §E  Derived-parameter functions
// ═══════════════════════════════════════════════════════════════════════════

/**
 * @brief Display Nyquist limit for an arbitrary refresh rate.
 *
 *   F_Nyquist = refresh_rate / 2
 *
 * @param refresh_rate_hz  Display refresh rate in Hz (must be positive)
 * @throws std::invalid_argument if refresh_rate_hz ≤ 0
 */
[[nodiscard]] inline double display_nyquist_hz(double refresh_rate_hz)
{
    if (refresh_rate_hz <= 0.0)
        throw std::invalid_argument("refresh_rate_hz must be positive");
    return refresh_rate_hz / 2.0;
}

/**
 * @brief Number of physics ticks accumulated per display frame (real-valued).
 *
 *   ticks = VISUAL_PHYSICS_RATE_HZ / refresh_rate_hz
 *
 * For 60 Hz: 1000/60 ≈ 16.667 ticks (actual implementation alternates 16/17).
 */
[[nodiscard]] inline double ticks_per_frame(double refresh_rate_hz)
{
    if (refresh_rate_hz <= 0.0)
        throw std::invalid_argument("refresh_rate_hz must be positive");
    return static_cast<double>(VISUAL_PHYSICS_RATE_HZ) / refresh_rate_hz;
}

/**
 * @brief Display frame period in milliseconds.
 *
 *   T_frame = 1000 / refresh_rate_hz
 */
[[nodiscard]] inline constexpr double frame_period_ms(double refresh_rate_hz) noexcept
{
    // Note: no runtime throw in constexpr — caller must pass a positive value.
    return 1000.0 / refresh_rate_hz;
}

// ═══════════════════════════════════════════════════════════════════════════
// §F  Energy accumulation (physics → accumulation buffer)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * @brief Accumulate energy of one holographic frame into the accumulation buffer.
 *
 * Per §GAP-018 Algorithm Step 2:
 *   B_acc[i] += |H[i]|²
 *
 * Accumulating intensity (|H|²) rather than amplitude (H) prevents destructive
 * phase cancellation of fast oscillations — they appear as motion blur, not void.
 *
 * @param acc_buffer   Running accumulation buffer (must be size n)
 * @param hologram     Instantaneous holographic projection sample (size n)
 * @throws std::invalid_argument if sizes differ
 */
inline void accumulate_energy(
        std::span<float>       acc_buffer,
        std::span<const float> hologram)
{
    if (acc_buffer.size() != hologram.size())
        throw std::invalid_argument("acc_buffer and hologram must have equal size");
    for (std::size_t i = 0; i < acc_buffer.size(); ++i)
        acc_buffer[i] += hologram[i] * hologram[i];
}

/**
 * @brief Normalize accumulation buffer → perceptual display amplitude.
 *
 * Per §GAP-018 Render Loop step and normalization formula:
 *   I_out[i] = sqrt( B_acc[i] / N_ticks )
 *
 * Square root restores perceptual amplitude from accumulated energy.
 * Caller should then pass result through tone_map() before display.
 *
 * @param acc_buffer  Accumulated energy (B_acc) — not modified
 * @param n_ticks     Number of physics ticks that were accumulated (must be > 0)
 * @param out         Output buffer for normalized values (size == acc_buffer.size())
 * @throws std::invalid_argument on size mismatch or n_ticks == 0
 */
inline void normalize_accumulation(
        std::span<const float> acc_buffer,
        int                    n_ticks,
        std::span<float>       out)
{
    if (n_ticks <= 0)
        throw std::invalid_argument("n_ticks must be positive");
    if (acc_buffer.size() != out.size())
        throw std::invalid_argument("acc_buffer and out must have equal size");

    const float inv_n = 1.0f / static_cast<float>(n_ticks);
    for (std::size_t i = 0; i < acc_buffer.size(); ++i)
        out[i] = std::sqrt(acc_buffer[i] * inv_n);
}

// ═══════════════════════════════════════════════════════════════════════════
// §G  Tone mapping
// ═══════════════════════════════════════════════════════════════════════════

/**
 * @brief Reinhard sigmoid tone mapping for HDR compression.
 *
 * Compresses the high dynamic range of resonance peaks to displayable [0, 1):
 *
 *   ToneMap(x) = x / (1 + x)
 *
 * Properties:
 *   ToneMap(0) = 0          ToneMap(1) = 0.5
 *   ToneMap(∞) → 1.0        always < 1, always ≥ 0 for x ≥ 0
 *
 * @param x  Input amplitude (≥ 0)
 */
[[nodiscard]] inline constexpr float tone_map(float x) noexcept
{
    return x / (1.0f + x);
}

/**
 * @brief Apply tone_map() to every element of a buffer in-place.
 */
inline void apply_tone_map(std::span<float> buf) noexcept
{
    for (auto& v : buf)
        v = tone_map(v);
}

/**
 * @brief Inverse of tone_map — recover linear amplitude from tone-mapped value.
 *
 *   ToneMap⁻¹(y) = y / (1 - y)    for y ∈ [0, 1)
 *
 * @throws std::invalid_argument if y ≥ 1 or y < 0
 */
[[nodiscard]] inline float tone_map_inverse(float y)
{
    if (y < 0.0f || y >= 1.0f)
        throw std::invalid_argument("tone_map_inverse requires y in [0, 1)");
    return y / (1.0f - y);
}

// ═══════════════════════════════════════════════════════════════════════════
// §H  Aliasing detection and Chromatic Aberration
// ═══════════════════════════════════════════════════════════════════════════

/**
 * @brief True if local_freq exceeds the display Nyquist — region will alias.
 *
 * Per §GAP-018: "If frequency exceeds display Nyquist (dΨ/dt > F_display/2),
 * introduce Chromatic Aberration shader shift."
 */
[[nodiscard]] inline constexpr bool is_super_nyquist(
        double local_freq_hz,
        double display_rate_hz) noexcept
{
    return local_freq_hz > display_rate_hz / 2.0;
}

/**
 * @brief Chromatic aberration shift in pixels for a super-Nyquist region.
 *
 * Per §GAP-018:  shift = k × (ω_local − ω_nyquist)
 *
 * Linear scale factor k is chosen so that the highest realistic frequency
 * (== display rate; one octave above Nyquist) maps to CHROMATIC_SHIFT_MAX_PIXELS.
 *
 *   shift_px = CHROMATIC_SHIFT_MAX_PIXELS × (excess / omega_nyquist)
 *
 * Returns 0 for sub-Nyquist frequencies.
 *
 * @param local_freq_hz   Pixel-local oscillation frequency (Hz)
 * @param display_rate_hz Display refresh rate (Hz)
 * @throws std::invalid_argument if display_rate_hz ≤ 0
 */
[[nodiscard]] inline double chromatic_shift_pixels(
        double local_freq_hz,
        double display_rate_hz)
{
    if (display_rate_hz <= 0.0)
        throw std::invalid_argument("display_rate_hz must be positive");

    const double nyquist = display_rate_hz / 2.0;
    if (local_freq_hz <= nyquist) return 0.0;

    const double excess = local_freq_hz - nyquist;
    const double shift  = static_cast<double>(CHROMATIC_SHIFT_MAX_PIXELS) * (excess / nyquist);
    // Clamp at maximum
    return (shift < static_cast<double>(CHROMATIC_SHIFT_MAX_PIXELS))
                ? shift
                : static_cast<double>(CHROMATIC_SHIFT_MAX_PIXELS);
}

// ═══════════════════════════════════════════════════════════════════════════
// §I  Stroboscopic mode helpers
// ═══════════════════════════════════════════════════════════════════════════

/**
 * @brief True when the E1 emitter phase is within tolerance of a zero-crossing.
 *
 * Per §GAP-018: "Visualizer captures frame only when Global Phase φ of Emitter 1
 * crosses Zero."  This phase-locks the display to the 5.083 Hz fundamental,
 * freezing standing wave patterns for debuggability.
 *
 * @param e1_phase_rad   Current global phase of E1 emitter (radians, any range)
 * @param tolerance_rad  Acceptance window around zero (default: STROBE_PHASE_ZERO_TOLERANCE)
 */
[[nodiscard]] inline bool stroboscopic_trigger(
        double e1_phase_rad,
        double tolerance_rad = STROBE_PHASE_ZERO_TOLERANCE) noexcept
{
    // Normalise to [−π, +π]
    double phi = std::fmod(e1_phase_rad, 2.0 * M_PI);
    if (phi >  M_PI) phi -= 2.0 * M_PI;
    if (phi < -M_PI) phi += 2.0 * M_PI;
    return std::abs(phi) <= tolerance_rad;
}

}  // namespace nikola::multimodal
