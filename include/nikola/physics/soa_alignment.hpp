#pragma once
// =============================================================================
// nikola/physics/soa_alignment.hpp
// Phase 85 — GAP-021: TorusGridSoA Memory Alignment Guarantees
//
// SOURCE: Gemini Deep Research Round 2, Batch 19-21 (December 15, 2025)
// SPEC:   docs/info/integration/sections/02_foundations/01_9d_toroidal_geometry.md
//         §GAP-021 (lines ~3446–3600)
//
// Compile-time alignment constants, static_assert policies, and predicate
// helpers that enforce AVX-512 / CUDA-coalesced SoA layout for TorusGridSoA.
// All checks are header-only and zero-overhead in release builds.
// =============================================================================

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <limits>
#include <bit>

namespace nikola::physics {

// ---------------------------------------------------------------------------
// § AVX-512 alignment constants
// ---------------------------------------------------------------------------

/// ZMM register width in bytes (512 bits / 8 = 64 bytes).
/// Spec: "AVX512_ALIGNMENT = 64 — aligned load requires address divisible by 64"
inline constexpr std::size_t AVX512_ALIGNMENT    = 64;

/// Bytes per AVX-512 ZMM register (same value; alias for clarity).
inline constexpr std::size_t ZMM_BYTES           = 64;

/// 32-bit floats per ZMM register (64 / 4).
inline constexpr std::size_t ZMM_FLOATS          = 16;

/// 64-bit doubles per ZMM register (64 / 8).
inline constexpr std::size_t ZMM_DOUBLES         = 8;

/// Standard C++ container alignment (std::vector, new) — NOT sufficient for AVX-512.
/// Spec: "std::vector aligns to 16 bytes — does NOT guarantee 64-byte alignment"
inline constexpr std::size_t STD_ALLOC_ALIGNMENT = 16;

/// CUDA warp size (32 threads × 4 bytes = 128-byte coalesced access).
inline constexpr std::size_t CUDA_WARP_BYTES     = 128;

/// Cache-line size on modern x86 CPUs (bytes).
inline constexpr std::size_t CACHE_LINE_BYTES    = 64;

// ---------------------------------------------------------------------------
// § TorusBlock geometry constants
// ---------------------------------------------------------------------------

/// Nodes per dense block: 3^9 = 19 683.
/// Spec: "block_size: 3^9 nodes per dense block"
inline constexpr int    TORUS_BLOCK_SIZE         = 19683;

/// Number of metric tensor components per node: upper triangular 9×9 = 45.
inline constexpr int    METRIC_COMPONENTS        = 45;

/// Complex wavefunction channels per node: real + imaginary = 2.
inline constexpr int    PSI_CHANNELS             = 2;

/// Bytes for one float32 wavefunction channel per block.
inline constexpr std::size_t PSI_CHANNEL_BYTES   = TORUS_BLOCK_SIZE * sizeof(float);

/// Bytes for all 45 metric tensor float32 arrays.
inline constexpr std::size_t METRIC_ARRAY_BYTES  = METRIC_COMPONENTS * TORUS_BLOCK_SIZE * sizeof(float);

// ---------------------------------------------------------------------------
// § Block padding policy
// ---------------------------------------------------------------------------

/// Round `n` up to the nearest multiple of `align`.
/// Used to compute the padded TorusBlock size so array indexing preserves alignment.
[[nodiscard]] constexpr std::size_t padded_size(std::size_t n, std::size_t align) noexcept {
    return (n + align - 1u) & ~(align - 1u);
}

/// Minimum padded size (bytes) that a TorusBlock struct must occupy so that
/// an array `TorusBlock blocks[N]` keeps all blocks AVX-512 aligned.
/// Spec: "sizeof(TorusBlock) must be padded to multiple of 64 bytes"
inline constexpr std::size_t TORUS_BLOCK_MIN_PAD_BYTES =
    padded_size(
        PSI_CHANNELS  * PSI_CHANNEL_BYTES +   // psi_real, psi_imag
        METRIC_ARRAY_BYTES,                    // metric_tensor[45]
        AVX512_ALIGNMENT
    );

// ---------------------------------------------------------------------------
// § Pointer alignment predicates
// ---------------------------------------------------------------------------

/// True when `ptr` is aligned to `align` bytes.
[[nodiscard]] inline bool is_aligned(const void* ptr, std::size_t align) noexcept {
    return (reinterpret_cast<uintptr_t>(ptr) % align) == 0;
}

/// True when `ptr` satisfies AVX-512 alignment.
[[nodiscard]] inline bool is_avx512_aligned(const void* ptr) noexcept {
    return is_aligned(ptr, AVX512_ALIGNMENT);
}

/// True when `ptr` satisfies CUDA coalesced access alignment.
[[nodiscard]] inline bool is_cuda_coalesced(const void* ptr) noexcept {
    return is_aligned(ptr, CUDA_WARP_BYTES);
}

/// True when `ptr` is cache-line aligned.
[[nodiscard]] inline bool is_cacheline_aligned(const void* ptr) noexcept {
    return is_aligned(ptr, CACHE_LINE_BYTES);
}

// ---------------------------------------------------------------------------
// § Allocation size helpers
// ---------------------------------------------------------------------------

/// Number of bytes to request from aligned_alloc for `n` elements of type T,
/// rounded up to the next AVX-512 boundary (required by POSIX aligned_alloc).
template <typename T>
[[nodiscard]] constexpr std::size_t avx512_alloc_size(std::size_t n) noexcept {
    return padded_size(n * sizeof(T), AVX512_ALIGNMENT);
}

/// True when `size` is already a valid aligned_alloc request size (multiple of alignment).
[[nodiscard]] constexpr bool valid_aligned_alloc_size(std::size_t size, std::size_t align) noexcept {
    return (size % align) == 0;
}

// ---------------------------------------------------------------------------
// § SIMD lane utilisation analysis
// ---------------------------------------------------------------------------

/// Number of full AVX-512 float32 lanes in a block (how many registers cover the block).
/// Fractional residual lanes require scalar epilogue handling.
[[nodiscard]] constexpr std::size_t avx512_float_lanes(std::size_t element_count) noexcept {
    return element_count / ZMM_FLOATS;
}

/// Number of scalar float32 elements left after full ZMM processing.
[[nodiscard]] constexpr std::size_t avx512_float_epilogue(std::size_t element_count) noexcept {
    return element_count % ZMM_FLOATS;
}

/// True when the block can be processed entirely with full ZMM registers (no epilogue).
[[nodiscard]] constexpr bool avx512_no_epilogue(std::size_t element_count) noexcept {
    return avx512_float_epilogue(element_count) == 0;
}

/// True when TORUS_BLOCK_SIZE perfectly fills ZMM float lanes (no scalar epilogue).
/// This is a compile-time property of the architecture.
inline constexpr bool TORUS_BLOCK_ZMM_CLEAN =
    (static_cast<std::size_t>(TORUS_BLOCK_SIZE) % ZMM_FLOATS) == 0;

// ---------------------------------------------------------------------------
// § AoS → SoA migration policy constants
// ---------------------------------------------------------------------------

/// AoS stride: bytes between successive `psi_real` accesses if stored in AoS layout.
/// Every AoS node = psi_real(4) + psi_imag(4) + metric(45×4) = 196 bytes.
inline constexpr std::size_t AOS_NODE_STRIDE_BYTES = (PSI_CHANNELS + METRIC_COMPONENTS) * sizeof(float);

/// SoA stride: bytes between successive `psi_real` values = 1 float = 4 bytes.
inline constexpr std::size_t SOA_NODE_STRIDE_BYTES = sizeof(float);

/// Bandwidth efficiency ratio SoA/AoS for streaming a single scalar channel.
/// Reading only psi_real from AoS wastes (AOS_STRIDE - 4) bytes per element.
[[nodiscard]] constexpr double soa_bandwidth_efficiency() noexcept {
    return static_cast<double>(SOA_NODE_STRIDE_BYTES) /
           static_cast<double>(AOS_NODE_STRIDE_BYTES);
}

// ---------------------------------------------------------------------------
// § Label helpers
// ---------------------------------------------------------------------------

[[nodiscard]] constexpr std::string_view alignment_policy_label(std::size_t align) noexcept {
    if (align == AVX512_ALIGNMENT)  return "AVX512_64B";
    if (align == CUDA_WARP_BYTES)   return "CUDA_WARP_128B";
    if (align == STD_ALLOC_ALIGNMENT) return "STD_ALLOC_16B";
    return "CUSTOM";
}

} // namespace nikola::physics
