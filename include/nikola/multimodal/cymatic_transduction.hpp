#pragma once
/**
 * @file   cymatic_transduction.hpp
 * @brief  GAP-017: Cymatic Transduction Sampling Rate Specification
 *
 * Defines all constants, invariant functions, and lightweight algorithms that
 * govern Nikola's audio→wave-physics bridge:
 *
 *   External hardware  →  Anti-alias FIR  →  Decimation  →  Physics injection
 *   48,000 Hz capture      300 taps            ÷48             1,000 Hz tick
 *
 * 8 Golden Ratio harmonic emitters (E1–E8) encode cognitive bands:
 *   f_n = π × φ^n   for n ∈ {1 … 8}
 *
 * Sampling rate derivation
 * ────────────────────────
 *   Physics tick rate     : 1,000 Hz
 *   Grid Nyquist limit    : 500 Hz    (F_physics / 2)
 *   E8 (highest emitter)  : ≈ 147.576 Hz
 *   E8 3rd-harmonic       : ≈ 442.7 Hz  <  500 Hz Nyquist  ✓
 *   Minimum Nyquist for E8: 2 × 147.576 ≈ 295.2 Hz  <  500 Hz  ✓
 *   Capture rate          : 48,000 Hz  (48:1 decimation to injection rate)
 *
 * Latency budget (target: <10 ms)
 * ────────────────────────────────
 *   Hardware buffer (128 samples @ 48 kHz) ≈ 2.667 ms
 *   FIR group delay (150 samples @ 48 kHz) ≈ 3.125 ms
 *   Processing + FFT                       ≈ 0.500 ms
 *   Physics tick window                    ≈ 1.000 ms
 *   Total                                  ≈ 7.292 ms  (<10 ms ✓)
 *
 * Namespace : nikola::multimodal
 * C++ std   : C++23
 *
 * §07_multimodal/01_cymatic_transduction.md §GAP-017
 */

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string_view>

namespace nikola::multimodal {

// ═══════════════════════════════════════════════════════════════════════════
// §1  Golden Ratio foundation
// ═══════════════════════════════════════════════════════════════════════════

/// φ (phi) — golden ratio, (1 + √5) / 2.
inline constexpr double GOLDEN_RATIO = 1.6180339887498948482;

/// Number of Golden Ratio harmonic emitters (E1 … E8).
inline constexpr int EMITTER_COUNT = 8;

/// Synchronizer e₉: π × (1/φ) × √2 × (32/27) ≈ 3.254 Hz @ 0° Δϕ (reference clock)
inline constexpr double SYNCHRONIZER_FREQ_HZ =
    M_PI * (1.0 / GOLDEN_RATIO) * M_SQRT2 * (32.0 / 27.0);

/// Prime phase offsets per emitter in degrees (spec: descending as φⁿ ascends)
inline constexpr std::array<double, EMITTER_COUNT> PRIME_PHASE_OFFSETS_DEG = {
    23.0, 19.0, 17.0, 13.0, 11.0, 7.0, 5.0, 3.0
};

/// Prime phase offsets in radians
inline constexpr std::array<double, EMITTER_COUNT> PRIME_PHASE_OFFSETS_RAD = {
    23.0 * M_PI / 180.0, 19.0 * M_PI / 180.0, 17.0 * M_PI / 180.0, 13.0 * M_PI / 180.0,
    11.0 * M_PI / 180.0,  7.0 * M_PI / 180.0,  5.0 * M_PI / 180.0,  3.0 * M_PI / 180.0
};

/// 179° phase asymmetry (Zenodo/ATPM): prevents total destructive interference,
/// leaving ~1.745% residual energy for fuzzy associative recall.
inline constexpr double PHASE_ASYMMETRY_RAD = 179.0 * M_PI / 180.0;

// ═══════════════════════════════════════════════════════════════════════════
// §2  Sampling rate constants
// ═══════════════════════════════════════════════════════════════════════════

/// Hardware audio capture rate — hardware-native at CD-quality+.
inline constexpr int CAPTURE_RATE_HZ = 48'000;

/// Physics injection rate — locked to physics tick.
inline constexpr int INJECTION_RATE_HZ = 1'000;

/// Physics engine tick rate (same as injection rate).
inline constexpr int PHYSICS_TICK_RATE_HZ = 1'000;

/// Decimation factor: CAPTURE_RATE_HZ / INJECTION_RATE_HZ.
inline constexpr int DECIMATION_FACTOR = 48;   // 48,000 / 1,000

/// Grid Nyquist limit — highest representable frequency in the physics grid.
inline constexpr double GRID_NYQUIST_HZ = static_cast<double>(PHYSICS_TICK_RATE_HZ) / 2.0;   // 500.0 Hz

// ═══════════════════════════════════════════════════════════════════════════
// §3  Anti-alias FIR filter specification
// ═══════════════════════════════════════════════════════════════════════════

/// FIR filter type: Equiripple linear-phase (preserves relative phase between E1 and E8).
/// Non-linear-phase IIR (e.g., Butterworth) is prohibited — destroys cognitive semantics.

/// Filter tap count.
inline constexpr int FIR_TAPS = 300;

/// Group delay = N/2 samples (linear phase FIR property).
inline constexpr int FIR_GROUP_DELAY_SAMPLES = FIR_TAPS / 2;   // 150

/// Passband edge — flat response guaranteed for all E1…E8 emitters.
inline constexpr double FIR_PASSBAND_HZ = 150.0;

/// Transition band start (same as passband edge).
inline constexpr double FIR_TRANSITION_START_HZ = 150.0;

/// Transition band end — stopband begins before Nyquist.
inline constexpr double FIR_TRANSITION_END_HZ = 450.0;

/// Stopband attenuation (dB, negative value).
/// Prevents high-amplitude noise from aliasing into Balanced Nonary range [−4, +4].
inline constexpr double FIR_ATTENUATION_DB = -60.0;

// ═══════════════════════════════════════════════════════════════════════════
// §4  Latency budget components (milliseconds)
// ═══════════════════════════════════════════════════════════════════════════

/// Hardware buffer size (samples @ CAPTURE_RATE_HZ).
inline constexpr int HW_BUFFER_SAMPLES = 128;

/// Hardware buffer latency (ms):  128 / 48000 × 1000.
inline constexpr double HW_BUFFER_LATENCY_MS =
    static_cast<double>(HW_BUFFER_SAMPLES) / static_cast<double>(CAPTURE_RATE_HZ) * 1000.0;
    // = 2.6667 ms

/// FIR group delay latency (ms):  150 / 48000 × 1000.
inline constexpr double FIR_GROUP_DELAY_LATENCY_MS =
    static_cast<double>(FIR_GROUP_DELAY_SAMPLES) / static_cast<double>(CAPTURE_RATE_HZ) * 1000.0;
    // = 3.125 ms

/// Processing + FFT overhead (ms) — conservative estimate.
inline constexpr double PROCESSING_LATENCY_MS = 0.5;

/// Physics tick window (ms).
inline constexpr double TICK_LATENCY_MS = 1.0;

/// Total end-to-end latency budget (ms):  HW + filter + processing + tick.
inline constexpr double TOTAL_LATENCY_MS =
    HW_BUFFER_LATENCY_MS + FIR_GROUP_DELAY_LATENCY_MS + PROCESSING_LATENCY_MS + TICK_LATENCY_MS;
    // = 7.292 ms

/// System latency requirement — must not be exceeded.
inline constexpr double LATENCY_REQUIREMENT_MS = 10.0;

// ═══════════════════════════════════════════════════════════════════════════
// §5  Dual-path architecture constants
// ═══════════════════════════════════════════════════════════════════════════

/// Enum distinguishing Direct (reflexive) and Isochronous (multimodal binding) paths.
enum class TransductionPath : uint8_t {
    DIRECT      = 0,   ///< <10ms — cymatic transduction, reflexive attention
    ISOCHRONOUS = 1,   ///< 50ms  — multimodal binding, AV sync
};

/// Isochronous buffer delay for multimodal binding (ms).
inline constexpr double ISOCHRONOUS_DELAY_MS = 50.0;

/// Direct path maximum acceptable latency (ms) — same as LATENCY_REQUIREMENT_MS.
inline constexpr double DIRECT_PATH_MAX_LATENCY_MS = 10.0;

// ═══════════════════════════════════════════════════════════════════════════
// §6  S-DFT / Goertzel parameters
// ═══════════════════════════════════════════════════════════════════════════

/// Sliding-DFT window size — 48 samples = 1 ms at capture rate (instantaneous for physics).
inline constexpr int SDFT_WINDOW_SAMPLES = 48;

/// Crosstalk isolation requirement: adjacent-emitter energy must be < 1% (−40 dB).
inline constexpr double CROSSTALK_THRESHOLD = 0.01;

// ═══════════════════════════════════════════════════════════════════════════
// §7  Cognitive-band labels (parallel to emitter indices 1–8)
// ═══════════════════════════════════════════════════════════════════════════

enum class CognitiveBand : uint8_t {
    DELTA       = 1,   ///< E1  — Metacognitive Timing (5.08 Hz)
    THETA       = 2,   ///< E2  — Working Memory (8.22 Hz)
    ALPHA       = 3,   ///< E3  — Idle / Relaxed Focus (13.31 Hz)
    BETA        = 4,   ///< E4  — Active Processing (21.53 Hz)
    GAMMA_LOW   = 5,   ///< E5  — Feature Binding (34.84 Hz)
    GAMMA_HIGH  = 6,   ///< E6  — Memory Retrieval (56.37 Hz)
    RIPPLE      = 7,   ///< E7  — Sharp Wave Ripples (91.21 Hz)
    FAST_RIPPLE = 8,   ///< E8  — Error Correction / Precision (147.58 Hz)
};

/// Human-readable label for each emitter index n ∈ [1, 8].
inline constexpr std::array<std::string_view, 9> EMITTER_LABELS = {
    /*[0] unused*/ "",
    /*[1] E1*/ "Delta (Metacognitive Timing)",
    /*[2] E2*/ "Theta (Working Memory)",
    /*[3] E3*/ "Alpha (Idle/Relaxed Focus)",
    /*[4] E4*/ "Beta (Active Processing)",
    /*[5] E5*/ "Gamma-Low (Feature Binding)",
    /*[6] E6*/ "Gamma-High (Memory Retrieval)",
    /*[7] E7*/ "Ripple (Sharp Wave Ripples)",
    /*[8] E8*/ "Fast Ripple (Error Correction)",
};

// ═══════════════════════════════════════════════════════════════════════════
// §8  Emitter frequency functions
// ═══════════════════════════════════════════════════════════════════════════

/**
 * @brief Compute emitter frequency: f_n = π × φ^n.
 *
 * @param n  Emitter index ∈ [1, 8]
 * @return   Frequency in Hz
 * @throws   std::invalid_argument if n < 1 or n > EMITTER_COUNT
 */
[[nodiscard]] inline double emitter_freq_hz(int n)
{
    if (n < 1 || n > EMITTER_COUNT)
        throw std::invalid_argument("emitter index must be in [1, 8]");
    return M_PI * std::pow(GOLDEN_RATIO, static_cast<double>(n));
}

/**
 * @brief Pre-computed array of all 8 emitter frequencies (index 0 = E1).
 *
 * Populated at namespace scope so tests can use it without calling pow() per test.
 * emitter_freqs[0] = E1 ≈ 5.083 Hz, emitter_freqs[7] = E8 ≈ 147.576 Hz.
 */
inline const std::array<double, EMITTER_COUNT> EMITTER_FREQS = []() {
    std::array<double, EMITTER_COUNT> a{};
    for (int i = 0; i < EMITTER_COUNT; ++i)
        a[i] = M_PI * std::pow(GOLDEN_RATIO, static_cast<double>(i + 1));
    return a;
}();

// ═══════════════════════════════════════════════════════════════════════════
// §9  Nyquist validation helpers
// ═══════════════════════════════════════════════════════════════════════════

/**
 * @brief Minimum sampling rate required to represent emitter n (Nyquist criterion).
 *
 *   F_s_min = 2 × f_n
 */
[[nodiscard]] inline double nyquist_min_for_emitter_hz(int n)
{
    return 2.0 * emitter_freq_hz(n);
}

/**
 * @brief Frequency of the k-th harmonic of emitter n.
 *
 *   f_{n,k} = k × f_n
 */
[[nodiscard]] inline double emitter_harmonic_hz(int n, int k)
{
    if (k < 1)
        throw std::invalid_argument("harmonic order must be >= 1");
    return static_cast<double>(k) * emitter_freq_hz(n);
}

/**
 * @brief True if the injection rate covers the given frequency (i.e., frequency < Nyquist).
 */
[[nodiscard]] inline constexpr bool injection_covers_freq(double freq_hz) noexcept
{
    return freq_hz < GRID_NYQUIST_HZ;
}

// ═══════════════════════════════════════════════════════════════════════════
// §10  Goertzel single-frequency energy estimator
// ═══════════════════════════════════════════════════════════════════════════

/**
 * @brief Goertzel algorithm — extract amplitude of a single target frequency.
 *
 * Implements §GAP-017 "SIMD-accelerated Goertzel algorithms for 8 specific
 * frequencies" (this scalar reference version).
 *
 * For a window of W samples at sample rate F_s, computes:
 *
 *   power = s[W-1]² + s[W-2]² - coeff × s[W-1] × s[W-2]
 *   amplitude = √(power / W)   (normalised)
 *
 * where the recurrence is:
 *   s[n] = x[n] + coeff × s[n-1] - s[n-2];   s[-1] = s[-2] = 0
 *   coeff = 2 × cos(2π × target_hz / sample_rate_hz)
 *
 * @param samples          Input sample buffer (any length ≥ 1)
 * @param target_hz        Target frequency (Hz)
 * @param sample_rate_hz   Sample rate of the input buffer (Hz)
 * @return                 Normalised amplitude ∈ [0, ∞), ideally ≈ 1.0 for a
 *                         unit-amplitude sinusoid at target_hz, ≈ 0 for silence.
 * @throws std::invalid_argument if target_hz ≤ 0 or sample_rate_hz ≤ 0
 */
[[nodiscard]] inline double goertzel_amplitude(
        std::span<const float> samples,
        double target_hz,
        double sample_rate_hz)
{
    if (target_hz     <= 0.0) throw std::invalid_argument("target_hz must be positive");
    if (sample_rate_hz <= 0.0) throw std::invalid_argument("sample_rate_hz must be positive");

    const double omega = 2.0 * M_PI * target_hz / sample_rate_hz;
    const double coeff = 2.0 * std::cos(omega);

    double s_prev2 = 0.0;
    double s_prev1 = 0.0;
    for (float x : samples) {
        double s = static_cast<double>(x) + coeff * s_prev1 - s_prev2;
        s_prev2 = s_prev1;
        s_prev1 = s;
    }

    const double power = s_prev1 * s_prev1 + s_prev2 * s_prev2 - coeff * s_prev1 * s_prev2;
    const double n = static_cast<double>(samples.size());
    // Normalised: 2 * sqrt(|power|) / N → ≈ 1.0 for a unit-amplitude sinusoid
    // at the target frequency (standard Goertzel normalisation).
    return (n > 0.0) ? 2.0 * std::sqrt(std::abs(power)) / n : 0.0;
}

// ═══════════════════════════════════════════════════════════════════════════
// §11  Latency budget query (runtime, not constexpr because it uses floating-point)
// ═══════════════════════════════════════════════════════════════════════════

/// Returns true if total system latency meets the <10ms requirement.
[[nodiscard]] inline constexpr bool latency_budget_met() noexcept
{
    return TOTAL_LATENCY_MS < LATENCY_REQUIREMENT_MS;
}

}  // namespace nikola::multimodal
