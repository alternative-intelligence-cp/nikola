/**
 * @file torus_block_simd.hpp
 * @brief Phase 116 — GAP-021 final: AVX-512 SIMD arithmetic kernels for TorusBlock.
 *
 * @spec docs/info/integration/sections/02_foundations/01_9d_toroidal_geometry.md
 *       §GAP-021 "AVX-512 vectorized inner loop"
 *
 * Problem
 * -------
 * The TorusBlock SoA layout (soa_layout.hpp) guarantees 64-byte alignment of all
 * psi_real / psi_imag / metric_tensor fields, enabling AVX-512 aligned loads.
 * What was missing (GAP-021 partial) was the actual `_mm512_*` arithmetic path.
 *
 * This header provides five inline SIMD kernels operating on TorusBlock fields:
 *
 *   psi_zero(b)                 — zero both wavefunction channels
 *   psi_scale(b, α)             — multiply all psi by scalar  Ψ ← α·Ψ
 *   psi_add_scaled(dst, src, α) — SAXPY          dst.Ψ ← dst.Ψ + α·src.Ψ
 *   psi_norm_sq(b)              — Born norm       Σᵢ(|re|² + |im|²)
 *   psi_renormalize(b)          — unit-norm       Ψ ← Ψ / ||Ψ||
 *   metric_scale(b, α)          — scale all 45 metric components by α
 *
 * AVX-512 path (compiled when -mavx512f is passed):
 *   - 16 float32 lanes per ZMM register
 *   - Aligned loads (`vmovaps`) / stores (`vmovaps`) — safe because alignas(64)
 *   - FMA via `vfmadd213ps` for SAXPY and norm accumulation
 *   - Horizontal reduction via `_mm512_reduce_add_ps` for norm_sq
 *   - `FULL_LOOPS = BLOCK_SIZE / 16 = 1230` vectorised iterations
 *   - `EPILOGUE = BLOCK_SIZE % 16 = 3` scalar tail elements
 *
 * Scalar fallback (no -mavx512f):
 *   - Identical arithmetic in plain C++ loops; compiler auto-vectorises to
 *     whatever SIMD ISA is available (SSE4, AVX2, etc.).
 *
 * Usage
 * -----
 * @code
 *   auto blk = std::make_unique<TorusBlock>();
 *   simd::psi_scale(*blk, 0.5f);          // halve amplitudes
 *   simd::psi_add_scaled(*blk, src, 1.f); // blk += src
 *   float n2 = simd::psi_norm_sq(*blk);
 *   simd::psi_renormalize(*blk);          // ||Ψ|| = 1
 * @endcode
 */
#pragma once

#include <nikola/physics/soa_layout.hpp>      // TorusBlock, AVX512_ALIGNMENT, METRIC_TENSOR_COMPONENTS

#include <cmath>    // std::sqrt
#include <cstring>  // std::memset

#ifdef __AVX512F__
#  include <immintrin.h>
#endif

namespace nikola::physics::simd {

// ---------------------------------------------------------------------------
// Compile-time geometry constants for the SIMD loop
// ---------------------------------------------------------------------------

/// Float32 lanes per AVX-512 ZMM register (64 bytes / 4 bytes-per-float = 16).
inline constexpr std::size_t K_ZMM_F = AVX512_ALIGNMENT / sizeof(float);

static_assert(K_ZMM_F == 16u, "Expected 16 float32 lanes per ZMM register");

/// Number of full 16-wide iterations over one TorusBlock channel.
/// FULL_LOOPS = 19683 / 16 = 1230
inline constexpr std::size_t FULL_LOOPS =
    static_cast<std::size_t>(TorusBlock::BLOCK_SIZE) / K_ZMM_F;

/// First element index not covered by the SIMD loop.
/// EPILOGUE_START = 1230 × 16 = 19680
inline constexpr std::size_t EPILOGUE_START = FULL_LOOPS * K_ZMM_F;

/// Number of scalar tail elements after the vectorised loop.
/// EPILOGUE = 19683 − 19680 = 3
inline constexpr std::size_t EPILOGUE_COUNT =
    static_cast<std::size_t>(TorusBlock::BLOCK_SIZE) - EPILOGUE_START;

// Sanity checks
static_assert(EPILOGUE_START + EPILOGUE_COUNT ==
              static_cast<std::size_t>(TorusBlock::BLOCK_SIZE),
              "EPILOGUE arithmetic mismatch");
static_assert(EPILOGUE_COUNT == 3u,
              "Expected 3 tail elements for BLOCK_SIZE=19683 with ZMM width 16");

// ---------------------------------------------------------------------------
// ISA detection helper
// ---------------------------------------------------------------------------

/**
 * @brief True when the translation unit was compiled with AVX-512F support.
 *
 * If true, all kernels below use `_mm512_*` intrinsics.
 * If false, plain C++ loops are used (compiler may auto-vectorise).
 */
[[nodiscard]] inline constexpr bool is_avx512_active() noexcept {
#ifdef __AVX512F__
    return true;
#else
    return false;
#endif
}

// ---------------------------------------------------------------------------
// Kernel: psi_zero
// ---------------------------------------------------------------------------

/**
 * @brief Zero both wavefunction channels of a TorusBlock.
 *
 * Equivalent to:  Ψ ← 0  (for every node in the block)
 *
 * AVX-512 path: 2 × 1230 aligned stores of _mm512_setzero_ps() then 3-element epilogue.
 */
inline void psi_zero(TorusBlock& b) noexcept {
    float* pr = b.psi_real.data();
    float* pi = b.psi_imag.data();

#ifdef __AVX512F__
    const __m512 zero = _mm512_setzero_ps();
    for (std::size_t k = 0; k < FULL_LOOPS; ++k) {
        _mm512_store_ps(pr + k * K_ZMM_F, zero);
        _mm512_store_ps(pi + k * K_ZMM_F, zero);
    }
    for (std::size_t j = EPILOGUE_START; j < static_cast<std::size_t>(TorusBlock::BLOCK_SIZE); ++j) {
        pr[j] = 0.f;
        pi[j] = 0.f;
    }
#else
    for (std::size_t j = 0; j < static_cast<std::size_t>(TorusBlock::BLOCK_SIZE); ++j) {
        pr[j] = 0.f;
        pi[j] = 0.f;
    }
#endif
}

// ---------------------------------------------------------------------------
// Kernel: psi_scale
// ---------------------------------------------------------------------------

/**
 * @brief Scale the wavefunction in-place:  Ψ ← α · Ψ
 *
 * Used for per-step global damping and Born-rule normalisation correction.
 *
 * @param b     TorusBlock to modify.
 * @param alpha Scalar multiplier.
 */
inline void psi_scale(TorusBlock& b, float alpha) noexcept {
    float* pr = b.psi_real.data();
    float* pi = b.psi_imag.data();

#ifdef __AVX512F__
    const __m512 a = _mm512_set1_ps(alpha);
    for (std::size_t k = 0; k < FULL_LOOPS; ++k) {
        _mm512_store_ps(pr + k * K_ZMM_F,
                        _mm512_mul_ps(_mm512_load_ps(pr + k * K_ZMM_F), a));
        _mm512_store_ps(pi + k * K_ZMM_F,
                        _mm512_mul_ps(_mm512_load_ps(pi + k * K_ZMM_F), a));
    }
    for (std::size_t j = EPILOGUE_START; j < static_cast<std::size_t>(TorusBlock::BLOCK_SIZE); ++j) {
        pr[j] *= alpha;
        pi[j] *= alpha;
    }
#else
    for (std::size_t j = 0; j < static_cast<std::size_t>(TorusBlock::BLOCK_SIZE); ++j) {
        pr[j] *= alpha;
        pi[j] *= alpha;
    }
#endif
}

// ---------------------------------------------------------------------------
// Kernel: psi_add_scaled  (complex SAXPY)
// ---------------------------------------------------------------------------

/**
 * @brief SAXPY on the wavefunction channels:  dst.Ψ ← dst.Ψ + α · src.Ψ
 *
 * Corresponds to the propagator drift step:
 *   Ψ += vel · τ   (psi_real and psi_imag independently)
 *
 * AVX-512 path: `vfmadd213ps`  (FMA: a·b + c  →  here: src·α + dst)
 *
 * @param dst   Destination block (updated in-place).
 * @param src   Source block (read-only).
 * @param alpha Scalar weight.
 */
inline void psi_add_scaled(TorusBlock& dst,
                            const TorusBlock& src,
                            float alpha) noexcept {
    float*       dr = dst.psi_real.data();
    float*       di = dst.psi_imag.data();
    const float* sr = src.psi_real.data();
    const float* si = src.psi_imag.data();

#ifdef __AVX512F__
    const __m512 a = _mm512_set1_ps(alpha);
    for (std::size_t k = 0; k < FULL_LOOPS; ++k) {
        const std::size_t off = k * K_ZMM_F;
        // dr[off..] += alpha * sr[off..]
        __m512 dv = _mm512_load_ps(dr + off);
        __m512 sv = _mm512_load_ps(sr + off);
        _mm512_store_ps(dr + off, _mm512_fmadd_ps(a, sv, dv));

        // di[off..] += alpha * si[off..]
        __m512 dvi = _mm512_load_ps(di + off);
        __m512 svi = _mm512_load_ps(si + off);
        _mm512_store_ps(di + off, _mm512_fmadd_ps(a, svi, dvi));
    }
    for (std::size_t j = EPILOGUE_START; j < static_cast<std::size_t>(TorusBlock::BLOCK_SIZE); ++j) {
        dr[j] += alpha * sr[j];
        di[j] += alpha * si[j];
    }
#else
    for (std::size_t j = 0; j < static_cast<std::size_t>(TorusBlock::BLOCK_SIZE); ++j) {
        dr[j] += alpha * sr[j];
        di[j] += alpha * si[j];
    }
#endif
}

// ---------------------------------------------------------------------------
// Kernel: psi_norm_sq
// ---------------------------------------------------------------------------

/**
 * @brief Compute the Born norm² of the wavefunction:  Σᵢ (re_i² + im_i²)
 *
 * Used before renormalization and in diagnostics.
 *
 * AVX-512 path: FMA accumulation into two ZMM accumulators, then
 * `_mm512_reduce_add_ps` horizontal sum, plus 3-element scalar epilogue.
 *
 * @param b  Block to measure.
 * @return   Sum of squared amplitudes (non-negative).
 */
[[nodiscard]] inline float psi_norm_sq(const TorusBlock& b) noexcept {
    const float* pr = b.psi_real.data();
    const float* pi = b.psi_imag.data();

#ifdef __AVX512F__
    __m512 acc_r = _mm512_setzero_ps();
    __m512 acc_i = _mm512_setzero_ps();

    for (std::size_t k = 0; k < FULL_LOOPS; ++k) {
        const std::size_t off = k * K_ZMM_F;
        __m512 rv = _mm512_load_ps(pr + off);
        __m512 iv = _mm512_load_ps(pi + off);
        acc_r = _mm512_fmadd_ps(rv, rv, acc_r);   // acc_r += re²
        acc_i = _mm512_fmadd_ps(iv, iv, acc_i);   // acc_i += im²
    }

    float total = _mm512_reduce_add_ps(acc_r) + _mm512_reduce_add_ps(acc_i);

    for (std::size_t j = EPILOGUE_START; j < static_cast<std::size_t>(TorusBlock::BLOCK_SIZE); ++j)
        total += pr[j]*pr[j] + pi[j]*pi[j];

    return total;
#else
    float total = 0.f;
    for (std::size_t j = 0; j < static_cast<std::size_t>(TorusBlock::BLOCK_SIZE); ++j)
        total += pr[j]*pr[j] + pi[j]*pi[j];
    return total;
#endif
}

// ---------------------------------------------------------------------------
// Kernel: psi_renormalize
// ---------------------------------------------------------------------------

/**
 * @brief Normalize the wavefunction to unit Born norm: Ψ ← Ψ / ||Ψ||
 *
 * No-op if ||Ψ|| < 1e-30 (handles zero / near-zero state gracefully).
 *
 * Internally calls psi_norm_sq + psi_scale (both fully SIMD-accelerated).
 */
inline void psi_renormalize(TorusBlock& b) noexcept {
    const float n2 = psi_norm_sq(b);
    if (n2 > 1e-30f)
        psi_scale(b, 1.f / std::sqrt(n2));
}

// ---------------------------------------------------------------------------
// Kernel: metric_scale
// ---------------------------------------------------------------------------

/**
 * @brief Scale all 45 metric tensor component arrays by α.
 *
 * Used for metric damping / diffusion:  g_{ij} ← α · g_{ij}
 *
 * Iterates over the 45 independent components; each component array is
 * BLOCK_SIZE floats. AVX-512 path matches psi_scale geometry.
 *
 * @param b     Block to modify.
 * @param alpha Damping factor (typically close to 1).
 */
inline void metric_scale(TorusBlock& b, float alpha) noexcept {
#ifdef __AVX512F__
    const __m512 a = _mm512_set1_ps(alpha);
    for (int c = 0; c < METRIC_TENSOR_COMPONENTS; ++c) {
        float* mp = b.metric_tensor[static_cast<std::size_t>(c)].data();
        for (std::size_t k = 0; k < FULL_LOOPS; ++k) {
            // Use unaligned load/store: metric_tensor[c] for c>0 is not
            // 64-byte aligned because sizeof(array<float,19683>)=78732 % 64=12.
            _mm512_storeu_ps(mp + k * K_ZMM_F,
                             _mm512_mul_ps(_mm512_loadu_ps(mp + k * K_ZMM_F), a));
        }
        for (std::size_t j = EPILOGUE_START; j < static_cast<std::size_t>(TorusBlock::BLOCK_SIZE); ++j)
            mp[j] *= alpha;
    }
#else
    for (int c = 0; c < METRIC_TENSOR_COMPONENTS; ++c) {
        float* mp = b.metric_tensor[static_cast<std::size_t>(c)].data();
        for (std::size_t j = 0; j < static_cast<std::size_t>(TorusBlock::BLOCK_SIZE); ++j)
            mp[j] *= alpha;
    }
#endif
}

} // namespace nikola::physics::simd
