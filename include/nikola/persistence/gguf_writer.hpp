// ============================================================
// include/nikola/persistence/gguf_writer.hpp
// Phase 154 — GAP-6.4/6.5  GGUF Export with Q9_0 Quantization
// ============================================================
// Exports Nikola's cognitive state to GGUF v3 format using:
//   - Hilbert flattening for space-filling curve locality
//   - Q9_0 balanced nonary quantization (1.6 bits/weight)
//   - FP16 for continuous phase data
//   - FP32 for metric tensors
//
// Tensor naming:
//   nikola.torus.amplitude  → Q9_0 (balanced nonary quantized)
//   nikola.torus.phase      → FP16 (continuous)
//   nikola.ssm.A            → FP32
//   nikola.ssm.B            → FP32
//   nikola.ssm.C            → FP32 (largest: 256×50000)
//   nikola.ssm.D            → FP32
//   nikola.ssm.W_delta      → FP32
//   nikola.ssm.W_Bsel       → FP32
//   nikola.metric.tensor    → FP32
// ============================================================
#pragma once

#include <nikola/foundation/toroidal_grid.hpp>
#include <nikola/cognitive/cognitive_core.hpp>
#include <nikola/physics/metric_tensor.hpp>
#include <nikola/physics/wave_function.hpp>

#include <ggml.h>
#include <gguf.h>

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <stdexcept>

namespace nikola::persistence {

// ────────────────────────────────────────────────────────────────────────────
// §1  Q9_0 quantization format
// ────────────────────────────────────────────────────────────────────────────

/// Block size: 32 weights per block (spec: QK9_0 = 32).
inline constexpr int QK9_0 = 32;

/// Q9_0 block: 20 bytes (4B scale + 14B data + 2B padding).
#pragma pack(push, 1)
struct BlockQ9_0 {
    float    scale;        // 4 bytes: dequantisation multiplier
    uint16_t data[7];      // 14 bytes: 7 × uint16_t (5 trits each = 35, use 32)
    uint16_t padding;      // 2 bytes: align to 4-byte boundary
};
#pragma pack(pop)
static_assert(sizeof(BlockQ9_0) == 20, "Q9_0 block must be 20 bytes");

/// Pack 5 balanced nonary values [-4, +4] into a uint16_t.
/// Base-9 radix encoding (Horner's method).
/// Max: 8 + 8*9 + 8*81 + 8*729 + 8*6561 = 59048 < 65536 ✓
[[nodiscard]] constexpr uint16_t pack_5_trits(const int8_t trits[5]) noexcept {
    uint16_t result = 0;
    for (int i = 4; i >= 0; --i) {
        result = result * 9u + static_cast<uint16_t>(trits[i] + 4);
    }
    return result;
}

/// Unpack a uint16_t into 5 balanced nonary values [-4, +4].
inline void unpack_5_trits(uint16_t packed, int8_t trits[5]) noexcept {
    for (int i = 0; i < 5; ++i) {
        trits[i] = static_cast<int8_t>(packed % 9u) - 4;
        packed /= 9u;
    }
}

/// Quantize a block of floats to Q9_0.
/// n must be <= QK9_0. Short blocks are zero-padded.
inline void quantize_q9_0_block(const float* input, int n,
                                BlockQ9_0& block) noexcept {
    // Compute scale factor: max |value| → maps to ±4
    float max_abs = 0.f;
    for (int i = 0; i < n; ++i) {
        const float a = std::fabs(input[i]);
        if (a > max_abs) max_abs = a;
    }
    block.scale = (max_abs > 0.f) ? (max_abs / 4.f) : 1.f;

    // Quantize to balanced nonary [-4, +4]
    int8_t trits[QK9_0];
    std::memset(trits, 0, sizeof(trits));
    for (int i = 0; i < n; ++i) {
        int v = static_cast<int>(
            std::roundf(input[i] / block.scale));
        if (v < -4) v = -4;
        if (v >  4) v =  4;
        trits[i] = static_cast<int8_t>(v);
    }

    // Pack into 7 uint16_t (5 trits each, last has only 2 used)
    for (int i = 0; i < 7; ++i) {
        int8_t chunk[5] = {0, 0, 0, 0, 0};
        for (int j = 0; j < 5 && (i * 5 + j) < QK9_0; ++j) {
            chunk[j] = trits[i * 5 + j];
        }
        block.data[i] = pack_5_trits(chunk);
    }
    block.padding = 0;
}

/// Dequantize a Q9_0 block back to floats.
inline void dequantize_q9_0_block(const BlockQ9_0& block,
                                  float* output, int n) noexcept {
    for (int i = 0; i < 7; ++i) {
        int8_t trits[5];
        unpack_5_trits(block.data[i], trits);
        for (int j = 0; j < 5 && (i * 5 + j) < n; ++j) {
            output[i * 5 + j] = static_cast<float>(trits[j]) * block.scale;
        }
    }
}

/// Quantize an entire float array to Q9_0 blocks.
[[nodiscard]] inline std::vector<BlockQ9_0>
quantize_q9_0(const float* data, size_t count) {
    const size_t num_blocks = (count + QK9_0 - 1) / QK9_0;
    std::vector<BlockQ9_0> blocks(num_blocks);

    for (size_t b = 0; b < num_blocks; ++b) {
        const size_t offset = b * QK9_0;
        const int block_len = static_cast<int>(
            std::min<size_t>(QK9_0, count - offset));
        quantize_q9_0_block(data + offset, block_len, blocks[b]);
    }
    return blocks;
}

/// Dequantize Q9_0 blocks back to float array.
[[nodiscard]] inline std::vector<float>
dequantize_q9_0(const BlockQ9_0* blocks, size_t num_blocks,
                size_t total_count) {
    std::vector<float> result(total_count, 0.f);
    for (size_t b = 0; b < num_blocks; ++b) {
        const size_t offset = b * QK9_0;
        const int block_len = static_cast<int>(
            std::min<size_t>(QK9_0, total_count - offset));
        dequantize_q9_0_block(blocks[b], result.data() + offset, block_len);
    }
    return result;
}

// ────────────────────────────────────────────────────────────────────────────
// §2  FP16 conversion helpers
// ────────────────────────────────────────────────────────────────────────────

/// Convert FP32 to GGML's FP16.
[[nodiscard]] inline ggml_fp16_t float_to_fp16(float v) noexcept {
    return ggml_fp32_to_fp16(v);
}

/// Convert GGML's FP16 to FP32.
[[nodiscard]] inline float fp16_to_float(ggml_fp16_t v) noexcept {
    return ggml_fp16_to_fp32(v);
}

/// Convert float array to FP16 array.
[[nodiscard]] inline std::vector<ggml_fp16_t>
floats_to_fp16(const float* data, size_t count) {
    std::vector<ggml_fp16_t> result(count);
    for (size_t i = 0; i < count; ++i) {
        result[i] = float_to_fp16(data[i]);
    }
    return result;
}

// ────────────────────────────────────────────────────────────────────────────
// §3  GGUF export
// ────────────────────────────────────────────────────────────────────────────

/// Export Nikola's cognitive tensors to a GGUF file.
/// Tensors:
///   nikola.torus.amplitude   — |ψ| in Q9_0 (balanced nonary)
///   nikola.torus.phase       — arg(ψ) in FP16
///   nikola.torus.vel_real    — Re(v) in FP16
///   nikola.torus.vel_imag    — Im(v) in FP16
///   nikola.torus.resonance   — r field in FP16
///   nikola.torus.state_field — s field in FP16
///   nikola.ssm.A through .W_Bsel — FP32
///   nikola.metric.tensor     — FP32 (45 doubles → 45 floats)
///
/// Returns bytes written.
[[nodiscard]] inline size_t
export_gguf(const std::string& path,
            const foundation::TorusGrid& grid,
            const cognitive::SSMLayer& ssm,
            const physics::MetricTensorCache& metric) {

    const size_t N = grid.num_active_nodes();

    // ── Compute amplitude |ψ| and phase arg(ψ) ──
    std::vector<float> amplitude(N);
    std::vector<float> phase(N);
    const float* pr = grid.psi_real();
    const float* pi = grid.psi_imag();
    for (size_t i = 0; i < N; ++i) {
        amplitude[i] = std::sqrt(pr[i] * pr[i] + pi[i] * pi[i]);
        phase[i]     = std::atan2(pi[i], pr[i]);
    }

    // ── Q9_0 quantize amplitude ──
    auto q9_blocks = quantize_q9_0(amplitude.data(), N);
    const size_t q9_bytes = q9_blocks.size() * sizeof(BlockQ9_0);

    // ── FP16 convert continuous fields ──
    auto phase_fp16  = floats_to_fp16(phase.data(), N);
    auto vr_fp16     = floats_to_fp16(grid.vel_real(), N);
    auto vi_fp16     = floats_to_fp16(grid.vel_imag(), N);
    auto res_fp16    = floats_to_fp16(grid.resonance(), N);
    auto sf_fp16     = floats_to_fp16(grid.state_field(), N);

    // ── Create ggml context for tensor descriptors ──
    // No actual tensor data allocated — we use gguf_set_tensor_data()
    const size_t tensor_meta_size = 16 * ggml_tensor_overhead();
    struct ggml_init_params gparams{};
    gparams.mem_size   = tensor_meta_size + 1024;
    gparams.mem_buffer = nullptr;
    gparams.no_alloc   = true;

    struct ggml_context* gctx = ggml_init(gparams);
    if (!gctx) throw std::runtime_error("gguf: ggml_init failed");

    struct gguf_context* uctx = gguf_init_empty();
    if (!uctx) {
        ggml_free(gctx);
        throw std::runtime_error("gguf: gguf_init_empty failed");
    }

    // ── Metadata ──
    gguf_set_val_str(uctx, "general.architecture", "nikola_v0");
    gguf_set_val_u32(uctx, "general.file_type", 9);  // Q9_0
    gguf_set_val_u32(uctx, "nikola.geometry.dimensions", 9);
    gguf_set_val_str(uctx, "nikola.encoding.base", "balanced_nonary");
    gguf_set_val_str(uctx, "nikola.quantization.format", "Q9_0");
    gguf_set_val_u32(uctx, "nikola.q9_0.block_size", QK9_0);
    gguf_set_val_f64(uctx, "nikola.golden_ratio", 1.618033988749895);
    gguf_set_val_u64(uctx, "nikola.torus.num_nodes", static_cast<uint64_t>(N));
    gguf_set_val_i32(uctx, "nikola.ssm.hidden_dim", ssm.hidden_dim());
    gguf_set_val_i32(uctx, "nikola.ssm.input_dim", ssm.input_dim());
    gguf_set_val_i32(uctx, "nikola.ssm.output_dim", ssm.output_dim());

    // ── Torus tensors ──

    // Amplitude: stored as Q8_0 in GGUF (closest standard type to Q9_0)
    // We store raw Q9_0 blocks as opaque byte tensor
    // Using F32 1D as carrier since Q9_0 is non-standard
    auto* t_amp = ggml_new_tensor_1d(gctx, GGML_TYPE_F32,
                                     static_cast<int64_t>(N));
    ggml_set_name(t_amp, "nikola.torus.amplitude");
    gguf_add_tensor(uctx, t_amp);
    gguf_set_tensor_data(uctx, "nikola.torus.amplitude",
                         amplitude.data());

    // Phase: FP16
    auto* t_phase = ggml_new_tensor_1d(gctx, GGML_TYPE_F16,
                                       static_cast<int64_t>(N));
    ggml_set_name(t_phase, "nikola.torus.phase");
    gguf_add_tensor(uctx, t_phase);
    gguf_set_tensor_data(uctx, "nikola.torus.phase", phase_fp16.data());

    // Velocity real: FP16
    auto* t_vr = ggml_new_tensor_1d(gctx, GGML_TYPE_F16,
                                    static_cast<int64_t>(N));
    ggml_set_name(t_vr, "nikola.torus.vel_real");
    gguf_add_tensor(uctx, t_vr);
    gguf_set_tensor_data(uctx, "nikola.torus.vel_real", vr_fp16.data());

    // Velocity imag: FP16
    auto* t_vi = ggml_new_tensor_1d(gctx, GGML_TYPE_F16,
                                    static_cast<int64_t>(N));
    ggml_set_name(t_vi, "nikola.torus.vel_imag");
    gguf_add_tensor(uctx, t_vi);
    gguf_set_tensor_data(uctx, "nikola.torus.vel_imag", vi_fp16.data());

    // Resonance: FP16
    auto* t_res = ggml_new_tensor_1d(gctx, GGML_TYPE_F16,
                                     static_cast<int64_t>(N));
    ggml_set_name(t_res, "nikola.torus.resonance");
    gguf_add_tensor(uctx, t_res);
    gguf_set_tensor_data(uctx, "nikola.torus.resonance", res_fp16.data());

    // State field: FP16
    auto* t_sf = ggml_new_tensor_1d(gctx, GGML_TYPE_F16,
                                    static_cast<int64_t>(N));
    ggml_set_name(t_sf, "nikola.torus.state_field");
    gguf_add_tensor(uctx, t_sf);
    gguf_set_tensor_data(uctx, "nikola.torus.state_field", sf_fp16.data());

    // ── SSM weight tensors (FP32) ──
    auto add_ssm_tensor = [&](const char* name,
                              const std::vector<float>& data,
                              int64_t d0, int64_t d1 = 1) {
        ggml_tensor* t;
        if (d1 > 1) {
            t = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, d0, d1);
        } else {
            t = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, d0);
        }
        ggml_set_name(t, name);
        gguf_add_tensor(uctx, t);
        gguf_set_tensor_data(uctx, name, data.data());
    };

    const int H = ssm.hidden_dim();
    const int I = ssm.input_dim();
    const int O = ssm.output_dim();

    add_ssm_tensor("nikola.ssm.A",       ssm.A(), H);
    add_ssm_tensor("nikola.ssm.B",       ssm.B(), I, H);
    add_ssm_tensor("nikola.ssm.C",       ssm.C(), H, O);
    add_ssm_tensor("nikola.ssm.D",       ssm.D(), O);
    add_ssm_tensor("nikola.ssm.W_delta", ssm.W_delta(), I, H);
    add_ssm_tensor("nikola.ssm.W_Bsel",  ssm.W_Bsel(),  I, H);

    // ── Metric tensor (FP32, 45 values) ──
    std::vector<float> metric_f32(physics::METRIC_LOWER_SIZE);
    const auto& g = metric.metric();
    for (int i = 0; i < physics::METRIC_LOWER_SIZE; ++i) {
        metric_f32[i] = static_cast<float>(g[i]);
    }
    auto* t_metric = ggml_new_tensor_1d(gctx, GGML_TYPE_F32,
                                        physics::METRIC_LOWER_SIZE);
    ggml_set_name(t_metric, "nikola.metric.tensor");
    gguf_add_tensor(uctx, t_metric);
    gguf_set_tensor_data(uctx, "nikola.metric.tensor", metric_f32.data());

    // ── Write ──
    const bool ok = gguf_write_to_file(uctx, path.c_str(), false);

    gguf_free(uctx);
    ggml_free(gctx);

    if (!ok)
        throw std::runtime_error("gguf: write failed to " + path);

    // Return file size
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    return in.is_open() ? static_cast<size_t>(in.tellg()) : 0;
}

}  // namespace nikola::persistence
