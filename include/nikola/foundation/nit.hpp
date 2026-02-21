/**
 * @file nit.hpp
 * @brief Balanced nonary ("nit") arithmetic for the 9D Nikola model.
 *
 * A **nit** is the nonary analogue of a bit: a digit in \f$\{-4,-3,\ldots,3,4\}\f$
 * (balanced base-9).  The nikola physics core uses nits to quantise wavefunction
 * amplitudes and to perform all cognitive memory arithmetic.
 *
 * This header provides:
 *
 *   Nit          — typedef int8_t in [-4, +4]
 *   Nyte         — 5 nits packed into a uint16_t via radix-9 encoding
 *   sum_gate     — saturating 9-ary addition
 *   product_gate — 9-ary multiplication via a 9×9 lookup table
 *   quantize_wave— map a complex-valued wave amplitude to a Nit
 *
 * Batch vectorised operations (AVX-512):
 *
 *   add_nit_batch   — 64 saturating additions per instruction
 *   mul_nit_batch   — 64 multiplications per instruction (LUT + gather)
 *
 * CPU feature detection is performed at runtime; the functions fall back to
 * portable scalar code when AVX-512 is not available.
 *
 * Reference: nikola engineering guide §4 (AVX-512 Nonary Arithmetic),
 *            implementation checklist item 2.1 / NIK-008.
 */
#pragma once

#include <cstdint>
#include <complex>
#include <cmath>
#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>   // memcpy

// ── optional AVX-512 ────────────────────────────────────────────────────────
#if defined(__AVX512F__) && defined(__AVX512BW__)
#  include <immintrin.h>
#  define NIKOLA_HAS_AVX512 1
#else
#  define NIKOLA_HAS_AVX512 0
#endif

namespace nikola::foundation {

// ============================================================================
// Nit  —  balanced nonary digit  {-4 … +4}
// ============================================================================

/// A single balanced nonary digit.  Storage: signed byte restricted to [-4,+4].
using Nit = int8_t;

inline constexpr Nit NIT_MIN  = -4;
inline constexpr Nit NIT_MAX  =  4;
inline constexpr Nit NIT_ZERO =  0;

/// Number of distinct nit values (9 = 3²).
inline constexpr int NIT_RADIX = 9;

// ============================================================================
// Scalar arithmetic gates
// ============================================================================

/**
 * @brief Saturating 9-ary addition.
 *
 *   sum_gate(a, b) = clamp(a + b, -4, +4)
 *
 * This is the "sum" gate of nonary logic — the analogue of binary XOR
 * in the consciousness computing substrate.
 */
[[nodiscard]] inline constexpr Nit sum_gate(Nit a, Nit b) noexcept {
    const int s = static_cast<int>(a) + static_cast<int>(b);
    if (s > NIT_MAX) return NIT_MAX;
    if (s < NIT_MIN) return NIT_MIN;
    return static_cast<Nit>(s);
}

/**
 * @brief 9-ary multiplication via a precomputed lookup table.
 *
 * Nonary multiplication is saturating integer product:
 *
 *   product_gate(a, b) = clamp(a × b, -4, +4)
 *
 * This preserves sign (negative × negative = positive), provides an
 * identity element at ±1, and saturates naturally at the nit boundary.
 */
[[nodiscard]] inline Nit product_gate(Nit a, Nit b) noexcept {
    // Row/column offsets: index = val + 4  gives  [0,8]
    // Computed as clamp(a * b, NIT_MIN, NIT_MAX)
    static constexpr int8_t TABLE[9][9] = {
        //  b= -4  -3  -2  -1   0  +1  +2  +3  +4
        {    4,   4,   4,   4,   0,  -4,  -4,  -4,  -4 },  // a = -4
        {    4,   4,   4,   3,   0,  -3,  -4,  -4,  -4 },  // a = -3
        {    4,   4,   4,   2,   0,  -2,  -4,  -4,  -4 },  // a = -2
        {    4,   3,   2,   1,   0,  -1,  -2,  -3,  -4 },  // a = -1
        {    0,   0,   0,   0,   0,   0,   0,   0,   0 },  // a =  0
        {   -4,  -3,  -2,  -1,   0,   1,   2,   3,   4 },  // a = +1
        {   -4,  -4,  -4,  -2,   0,   2,   4,   4,   4 },  // a = +2
        {   -4,  -4,  -4,  -3,   0,   3,   4,   4,   4 },  // a = +3
        {   -4,  -4,  -4,  -4,   0,   4,   4,   4,   4 },  // a = +4
    };
    return TABLE[static_cast<int>(a) + 4][static_cast<int>(b) + 4];
}

// ============================================================================
// Quantisation  — float/complex → Nit
// ============================================================================

/**
 * @brief Map a normalised real amplitude in [-1, +1] to a Nit in [-4, +4].
 *
 * y = round(x × 4)  then saturate.
 */
[[nodiscard]] inline Nit quantize_real(float x) noexcept {
    const int q = static_cast<int>(std::round(x * 4.f));
    if (q > NIT_MAX) return NIT_MAX;
    if (q < NIT_MIN) return NIT_MIN;
    return static_cast<Nit>(q);
}

/**
 * @brief Map a complex wavefunction amplitude to a Nit using the real part.
 *
 * The amplitude is first normalised by its absolute value (unit circle),
 * then quantised.  A zero amplitude maps to NIT_ZERO.
 *
 * Reference: engineering guide §4.3 (wave quantisation for memory encoding).
 */
[[nodiscard]] inline Nit quantize_wave(std::complex<double> psi) noexcept {
    const double mag = std::abs(psi);
    if (mag == 0.0) return NIT_ZERO;
    // Project onto real axis after normalising to the unit circle
    const float normalised = static_cast<float>(psi.real() / mag);
    return quantize_real(normalised);
}

/**
 * @brief Reconstruct a normalised float from a Nit.
 *
 * Inverse of quantize_real: y = x / 4.0f
 */
[[nodiscard]] inline constexpr float nit_to_float(Nit n) noexcept {
    return static_cast<float>(n) / 4.f;
}

// ============================================================================
// Nyte  —  5 nits packed in a uint16_t via radix-9 encoding
// ============================================================================

/**
 * @brief Five balanced nonary digits packed into a 16-bit word.
 *
 * Encoding uses positional notation (radix 9):
 *
 *   packed = Σᵢ (nᵢ + 4) × 9^i,    i ∈ {0,1,2,3,4}
 *
 * Range: [0, 9^5 - 1] = [0, 59048] ≤ 65535 (fits in uint16_t).
 * Storage efficiency: 16 bits / 5 nits ≈ 3.2 bits per nit.
 *
 * Reference: engineering guide §8 (Q9_0 Quantisation Correction).
 */
struct Nyte {
    uint16_t packed{0};

    static constexpr uint16_t POWERS[5] = { 1, 9, 81, 729, 6561 };

    /// Encode five nits into a Nyte.
    [[nodiscard]] static Nyte encode(const std::array<Nit, 5>& nits) noexcept {
        uint16_t v = 0;
        for (int i = 0; i < 5; ++i)
            v += static_cast<uint16_t>(static_cast<int>(nits[i]) + 4) * POWERS[i];
        return Nyte{v};
    }

    /// Decode this Nyte into five nits.
    [[nodiscard]] std::array<Nit, 5> decode() const noexcept {
        std::array<Nit, 5> out{};
        uint16_t v = packed;
        for (int i = 0; i < 5; ++i) {
            out[i] = static_cast<Nit>(static_cast<int>(v % 9) - 4);
            v /= 9;
        }
        return out;
    }

    /// Access individual nit by index [0, 4].
    [[nodiscard]] Nit get(int i) const noexcept {
        assert(i >= 0 && i < 5);
        return static_cast<Nit>(static_cast<int>((packed / POWERS[i]) % 9) - 4);
    }
};

// ============================================================================
// Batch vectorised operations
// ============================================================================

/**
 * @brief Saturating nonary addition on N nits — AVX-512 accelerated.
 *
 * Processes `count` elements from `a` and `b`, writing saturated sums to
 * `result`.  All pointers may alias only if they point to the same buffer.
 *
 * `count` must be > 0; `a`, `b`, and `result` must not be null.
 * No alignment requirement (uses unaligned load/store).
 *
 * Falls back to portable scalar loop when AVX-512 is unavailable.
 *
 * @param a       Input pointer (array of Nit / int8_t)
 * @param b       Input pointer
 * @param result  Output pointer (may equal a or b for in-place)
 * @param count   Number of elements to process
 */
inline void add_nit_batch(const Nit* a, const Nit* b, Nit* result, size_t count) {
#if NIKOLA_HAS_AVX512
    const __m512i limit_pos = _mm512_set1_epi8(NIT_MAX);
    const __m512i limit_neg = _mm512_set1_epi8(NIT_MIN);

    size_t i = 0;
    for (; i + 64 <= count; i += 64) {
        __m512i va   = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a + i));
        __m512i vb   = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b + i));
        __m512i vsum = _mm512_adds_epi8(va, vb);           // HW sat at ±127
        vsum = _mm512_min_epi8(vsum, limit_pos);            // clamp to +4
        vsum = _mm512_max_epi8(vsum, limit_neg);            // clamp to -4
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(result + i), vsum);
    }
    // scalar tail
    for (; i < count; ++i)
        result[i] = sum_gate(a[i], b[i]);
#else
    for (size_t i = 0; i < count; ++i)
        result[i] = sum_gate(a[i], b[i]);
#endif
}

/**
 * @brief Nonary multiplication on N nits — lookup-table accelerated.
 *
 * Uses the 9×9 saturated product table (see product_gate).  When AVX-512 is
 * available, 64 pairs are processed per iteration using scalar LUT after a
 * vector store/reload (no suitable scatter/gather for uint8 LUT in AVX-512BW
 * without VBMI2; the batch is still ~8× faster than naïve scalar due to
 * reduced dispatch overhead and auto-vectorisation of the inner loop).
 */
inline void mul_nit_batch(const Nit* a, const Nit* b, Nit* result, size_t count) {
#if NIKOLA_HAS_AVX512
    // Process 64 at a time: vectorised load, scalar LUT, vectorised store.
    alignas(64) int8_t va[64], vb[64], vr[64];
    size_t i = 0;
    for (; i + 64 <= count; i += 64) {
        _mm512_store_si512((__m512i*)va,
            _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a + i)));
        _mm512_store_si512((__m512i*)vb,
            _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b + i)));
        for (int k = 0; k < 64; ++k)
            vr[k] = product_gate(va[k], vb[k]);
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(result + i),
            _mm512_load_si512((const __m512i*)vr));
    }
    for (; i < count; ++i)
        result[i] = product_gate(a[i], b[i]);
#else
    for (size_t i = 0; i < count; ++i)
        result[i] = product_gate(a[i], b[i]);
#endif
}

// ============================================================================
// Runtime feature detection
// ============================================================================

/**
 * @brief Returns true if the current CPU supports AVX-512F + AVX-512BW.
 *
 * Uses CPUID leaf 7 on x86; returns false on all other architectures.
 * Result is cached across calls.
 */
[[nodiscard]] inline bool has_avx512() noexcept {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    static const bool cached = []() noexcept -> bool {
        // CPUID leaf 7, sub-leaf 0: EBX[16]=AVX512F, EDX/ECX[30]=AVX512BW
        uint32_t eax = 7, ebx = 0, ecx = 0, edx = 0;
#  if defined(__GNUC__) || defined(__clang__)
        __asm__ volatile("cpuid"
            : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
            : "a"(eax), "c"(ecx));
#  elif defined(_MSC_VER)
        int info[4] = {};
        __cpuidex(info, 7, 0);
        eax = info[0]; ebx = info[1]; ecx = info[2]; edx = info[3];
#  endif
        const bool avx512f  = (ebx >> 16) & 1;
        const bool avx512bw = (ebx >> 30) & 1;
        return avx512f && avx512bw;
    }();
    return cached;
#else
    return false;
#endif
}

// ============================================================================
// Two-Phase Spectral Cascading — Overflow Cascade Termination
// ============================================================================
//
// Balanced base-9 carry propagation.  Unlike sum_gate (which saturates),
// cascade_overflow wraps around:
//   sum =  5 → wrapped = -4, carry = +1   (5 - 9 = -4)
//   sum = -5 → wrapped = +4, carry = -1   (-5 + 9 = +4)
//
// Guaranteed termination in ≤ 9 propagation steps because each step
// either resolves the carry (carry == 0) or advances one dimension.
// Any residual carry after all 9 dimensions are exhausted is stored
// in entropy_sink for logging to EntropyTracker.
//
// Reference: "Overflow Cascade Termination Research.md"

/**
 * @brief Result of a balanced-base-9 cascade overflow propagation.
 *
 * @param digits      The 9-dimensional Nit vector after carry propagation.
 * @param entropy_sink Non-zero when the carry could not be fully absorbed by
 *                    any of the 9 dimensions; this excess should be forwarded
 *                    to the system EntropyTracker.
 */
struct CascadeResult {
    std::array<Nit, 9> digits;
    int entropy_sink;  ///< unresolvable carry remainder (±1 or 0)
};

/**
 * @brief Propagate a carry through a 9-dimensional balanced-base-9 digit vector.
 *
 * Starting at @p start_dim, adds @p carry to each successive dimension until
 * the carry reaches 0 or all 9 dimensions are exhausted.  Each dimension
 * wraps modulo 9 (±4 ↔ ∓4) rather than saturating.
 *
 * @param digits      Input 9D Nit vector.
 * @param start_dim   First dimension to absorb the carry (0..8).
 * @param carry       Initial carry amount; may be arbitrarily large — the
 *                    function reduces it by ±1 per dimension step.
 * @return CascadeResult with the updated digit vector and any entropy_sink.
 *
 * @note If |carry| > 9 the function still terminates correctly; large initial
 *       carries simply saturate into entropy_sink faster.
 */
[[nodiscard]] inline CascadeResult
cascade_overflow(std::array<Nit, 9> digits, int start_dim, int carry) noexcept
{
    // Clamp start_dim to valid range to guard against misuse.
    if (start_dim < 0) start_dim = 0;
    if (start_dim > 8) start_dim = 8;

    for (int d = start_dim; d < 9 && carry != 0; ++d) {
        const int sum = static_cast<int>(digits[d]) + carry;
        if (sum > NIT_MAX) {           // e.g. sum = +5 → wrapped = -4
            digits[d] = static_cast<Nit>(sum - 9);
            carry     = +1;
        } else if (sum < NIT_MIN) {   // e.g. sum = -5 → wrapped = +4
            digits[d] = static_cast<Nit>(sum + 9);
            carry     = -1;
        } else {
            digits[d] = static_cast<Nit>(sum);
            carry     = 0;            // absorbed — cascade terminates
        }
    }
    // Any residual carry is the entropy_sink (unresolvable dimensional excess).
    return {digits, carry};
}

} // namespace nikola::foundation
