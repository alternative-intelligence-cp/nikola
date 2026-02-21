/**
 * @file adaptive_quantizer.hpp
 * @brief Gap 6.5 — AdaptiveQuantizer (Q9_0 / FP16 mixed precision)
 *
 * Compression strategy:
 *   Low energy  (|Ψ|² < HIGH_ENERGY_THRESHOLD):  Q9_0  — 5-trit balanced nonary
 *   High energy (peaks):                          FP16  — uncompressed
 *
 * Q9_0 encoding:
 *   quantize_to_trit(v ∈ [-1,1]) → int8_t ∈ [-4,+4]  (scale × 4)
 *   dequantize_from_trit(t)      → float = t / 4.0f
 *
 * Storage cost per complex node:
 *   FP32  : 8 bytes   (2 × float32)
 *   FP16  : 4 bytes   (2 × float16)
 *   Q9_0  : 2 bytes   (2 × int8_t) — approximate ~2.5 bytes when packing overhead added
 *
 * For 1 M nodes: FP32=8 MB; Adaptive(95% Q9_0, 5% FP16) ≈ 2.8 MB (~65% reduction)
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

namespace nikola::multimodal {

// ============================================================================
// Constants
// ============================================================================

inline constexpr float   Q9_ENERGY_THRESHOLD     = 1e-3f; // |Ψ|² above this → FP16
inline constexpr int8_t  Q9_TRIT_MIN             = -4;
inline constexpr int8_t  Q9_TRIT_MAX             =  4;
inline constexpr float   Q9_TRIT_SCALE           =  4.0f;
inline constexpr float   Q9_MAX_ABS_ERROR        =  0.125f; // 1/(2*SCALE) = 1/8

// ============================================================================
// Q9Block — compressed representation of one complex node
// ============================================================================

/**
 * Compressed representation of one wavefunction element (ψ_real + ψ_imag).
 */
struct Q9Block {
    uint8_t format{0};   //< 0 = Q9_0 (trit), 1 = FP16 (half approximated as FP32)
    int8_t  q_real{0};   //< Q9_0: trit [-4,+4]; FP16: ignored (fp_real used)
    int8_t  q_imag{0};   //< Q9_0: trit [-4,+4]; FP16: ignored (fp_imag used)
    float   fp_real{0};  //< FP16 path: original float value
    float   fp_imag{0};  //< FP16 path: original float value
};

// ============================================================================
// Gap 6.5 — AdaptiveQuantizer
// ============================================================================

class AdaptiveQuantizer {
public:
    /**
     * Map a float in [-1, 1] to a 5-trit balanced nonary integer in [-4, +4].
     *
     * Values outside [-1, 1] are clamped before scaling.
     */
    static int8_t quantize_to_trit(float v) noexcept
    {
        const float clamped = std::clamp(v, -1.0f, 1.0f);
        const float scaled  = clamped * Q9_TRIT_SCALE;
        const int   rounded = static_cast<int>(std::round(scaled));
        return static_cast<int8_t>(
            std::clamp(rounded,
                       static_cast<int>(Q9_TRIT_MIN),
                       static_cast<int>(Q9_TRIT_MAX)));
    }

    /**
     * Reconstruct a float from a 5-trit balanced nonary integer.
     */
    static float dequantize_from_trit(int8_t t) noexcept
    {
        return static_cast<float>(t) / Q9_TRIT_SCALE;
    }

    /**
     * Compress paired wavefunction arrays into Q9Block vector.
     *
     * @param psi_real  Real part of wavefunction (N elements)
     * @param psi_imag  Imaginary part, same length
     * @return Vector of N Q9Block entries
     */
    static std::vector<Q9Block> compress(std::span<const float> psi_real,
                                          std::span<const float> psi_imag)
    {
        const size_t N = std::min(psi_real.size(), psi_imag.size());
        std::vector<Q9Block> out;
        out.reserve(N);

        for (size_t i = 0; i < N; ++i) {
            const float re = psi_real[i];
            const float im = psi_imag[i];
            const float energy = re * re + im * im;

            Q9Block blk{};
            if (energy >= Q9_ENERGY_THRESHOLD) {
                // High-energy: FP16 path (store as FP32 approximation)
                blk.format  = 1;
                blk.fp_real = re;
                blk.fp_imag = im;
            } else {
                // Low-energy: Q9_0 trit quantization
                // Values must fit in [-1, 1]; scale by 1/max_abs if needed
                const float max_abs = std::max(std::abs(re), std::abs(im));
                float norm_re = re;
                float norm_im = im;
                if (max_abs > 1.0f) {
                    norm_re /= max_abs;
                    norm_im /= max_abs;
                }
                blk.format = 0;
                blk.q_real = quantize_to_trit(norm_re);
                blk.q_imag = quantize_to_trit(norm_im);
            }
            out.push_back(blk);
        }
        return out;
    }

    /**
     * Decompress Q9Block array back into wavefunction arrays.
     */
    static void decompress(const std::vector<Q9Block>& blocks,
                            std::vector<float>& psi_real_out,
                            std::vector<float>& psi_imag_out)
    {
        psi_real_out.resize(blocks.size());
        psi_imag_out.resize(blocks.size());

        for (size_t i = 0; i < blocks.size(); ++i) {
            const Q9Block& blk = blocks[i];
            if (blk.format == 1) {
                psi_real_out[i] = blk.fp_real;
                psi_imag_out[i] = blk.fp_imag;
            } else {
                psi_real_out[i] = dequantize_from_trit(blk.q_real);
                psi_imag_out[i] = dequantize_from_trit(blk.q_imag);
            }
        }
    }

    /**
     * Estimate the compressed size in bytes for the given blocks.
     *
     * Q9_0 block:  2 bytes (q_real + q_imag as int8)
     * FP16 block:  4 bytes (2 × float16 approximated)
     */
    static size_t compressed_bytes(const std::vector<Q9Block>& blocks) noexcept
    {
        size_t total = 0;
        for (const auto& blk : blocks) {
            total += (blk.format == 0) ? 2u : 4u;
        }
        return total;
    }

    /**
     * Compute compression ratio vs uncompressed FP32 (8 bytes / complex node).
     */
    static float compression_ratio(const std::vector<Q9Block>& blocks) noexcept
    {
        if (blocks.empty()) return 1.0f;
        const float uncompressed = static_cast<float>(blocks.size()) * 8.0f; // 2×FP32
        const float compressed   = static_cast<float>(compressed_bytes(blocks));
        return (compressed > 0.0f) ? (compressed / uncompressed) : 1.0f;
    }

    /**
     * Count Q9_0 and FP16 blocks.
     */
    static std::pair<size_t, size_t> count_formats(const std::vector<Q9Block>& blocks)
    {
        size_t n_q9 = 0, n_fp16 = 0;
        for (const auto& blk : blocks) {
            if (blk.format == 0) ++n_q9; else ++n_fp16;
        }
        return {n_q9, n_fp16};
    }
};

} // namespace nikola::multimodal
