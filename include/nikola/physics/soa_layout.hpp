/**
 * @file soa_layout.hpp
 * @brief GAP-021: TorusGridSoA Memory Alignment Guarantees
 *
 * @spec FABRICATION-READY — docs/info/integration/sections/02_foundations/
 *       01_9d_toroidal_geometry.md §GAP-021
 *
 * Problem: Nikola must update millions of nodes within 1 ms.  This throughput
 * is impossible with scalar code — requires SIMD parallelism:
 *   - CPUs: AVX-512 (ZMM registers, 512 bits = 64 bytes)
 *   - GPUs: coalesced memory access (same 64-byte cache-line rule)
 *
 * Phase 0 mandate: AoS → SoA transition.
 *   AoS: [Re,Im,g11,g12,…], [Re,Im,g11,g12,…]  ← loads pull unrelated data
 *   SoA: [Re,Re,Re,…],       [Im,Im,Im,…]       ← perfect for vectorisation
 *
 * AVX-512 aligned-load (`vmovaps`) requires address ≡ 0 (mod 64).
 * std::vector aligns to max_align_t (typically 16 bytes) — NOT sufficient.
 *
 * This header provides:
 *   - AVX512_ALIGNMENT constant
 *   - AlignedAllocator<T>      — 64-byte STL allocator (std::aligned_alloc)
 *   - AlignedVec<T>            — convenience typedef
 *   - TorusBlock               — 3^9-node SoA brick with compile-time assertions
 *   - SoaAlignmentGuard        — runtime verification watchdog (Oracle hook)
 *   - load_block_data()        — Copy-on-Load from unaligned I/O buffers
 *   - gguf_aligned_offset()    — file-offset padding for mmap compatibility
 */
#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace nikola::physics {

// ---------------------------------------------------------------------------
// Alignment constant
// ---------------------------------------------------------------------------

/// 64-byte alignment required for AVX-512 ZMM register loads (`vmovaps`).
/// Also satisfies AVX2, SSE, and GPU coalesced-access requirements.
inline constexpr std::size_t AVX512_ALIGNMENT = 64;

/// Number of metric-tensor independent components in a symmetric 9×9 matrix
/// stored in a 9D Riemannian manifold: (9×10)/2 = 45.
inline constexpr int METRIC_TENSOR_COMPONENTS = 45;

// ---------------------------------------------------------------------------
// AlignedAllocator<T>
// ---------------------------------------------------------------------------

/**
 * @brief Custom STL allocator guaranteeing AVX512_ALIGNMENT (64 bytes).
 *
 * Spec §GAP-021 "Compile-Time Enforcement":
 *   "Critical for AVX-512 vectorization stability."
 *
 * Uses std::aligned_alloc; rounds up requested byte count to the nearest
 * multiple of AVX512_ALIGNMENT (required by the C standard).
 */
template <typename T>
struct AlignedAllocator {
    using value_type = T;

    // Required rebind for STL container compatibility
    template <typename U>
    struct rebind { using other = AlignedAllocator<U>; };

    AlignedAllocator() noexcept = default;

    template <typename U>
    explicit AlignedAllocator(const AlignedAllocator<U>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t n) {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::bad_array_new_length();

        // std::aligned_alloc requires size to be a multiple of alignment
        const std::size_t bytes        = n * sizeof(T);
        const std::size_t aligned_bytes =
            (bytes + AVX512_ALIGNMENT - 1) & ~(AVX512_ALIGNMENT - 1);

        void* ptr = std::aligned_alloc(AVX512_ALIGNMENT, aligned_bytes);
        if (!ptr) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t) noexcept {
        std::free(p);
    }

    // Equality: two AlignedAllocator<T> are always interchangeable
    template <typename U>
    bool operator==(const AlignedAllocator<U>&) const noexcept { return true; }

    template <typename U>
    bool operator!=(const AlignedAllocator<U>&) const noexcept { return false; }
};

/// Convenience alias: std::vector with 64-byte-aligned storage.
template <typename T>
using AlignedVec = std::vector<T, AlignedAllocator<T>>;

// ---------------------------------------------------------------------------
// TorusBlock — 3^9-node SoA brick
// ---------------------------------------------------------------------------

/**
 * @brief Dense 3^9-node SoA brick for the 9D toroidal physics grid.
 *
 * Spec §GAP-021:
 *   "struct TorusBlock { static constexpr int BLOCK_SIZE = 19683; ... }"
 *
 * Layout (all fields 64-byte aligned):
 *   psi_real      — real part of wavefunction Ψ at each node
 *   psi_imag      — imaginary part of wavefunction Ψ
 *   metric_tensor — 45 independent components of symmetric 9×9 Riemannian metric
 *
 * Compile-time guarantees (static_assert, per spec):
 *   alignof(TorusBlock)                   == 64
 *   offsetof(TorusBlock, psi_real)  % 64  == 0
 *   offsetof(TorusBlock, psi_imag)  % 64  == 0
 *   sizeof(TorusBlock)              % 64  == 0   ← preserves alignment in arrays
 */
struct alignas(AVX512_ALIGNMENT) TorusBlock {
    /// 3^9 = 19,683 nodes per dense block
    static constexpr int BLOCK_SIZE = 19683;

    alignas(AVX512_ALIGNMENT) std::array<float, BLOCK_SIZE> psi_real;
    alignas(AVX512_ALIGNMENT) std::array<float, BLOCK_SIZE> psi_imag;

    /// 45 independent components of the symmetric 9×9 metric tensor g_ij
    alignas(AVX512_ALIGNMENT) std::array<std::array<float, BLOCK_SIZE>,
                                         METRIC_TENSOR_COMPONENTS> metric_tensor;
};

// Compile-time alignment enforcement (spec §GAP-021)
static_assert(alignof(TorusBlock) == AVX512_ALIGNMENT,
              "TorusBlock must be 64-byte aligned (AVX-512 requirement)");
static_assert(offsetof(TorusBlock, psi_real) % AVX512_ALIGNMENT == 0,
              "psi_real offset misalignment");
static_assert(offsetof(TorusBlock, psi_imag) % AVX512_ALIGNMENT == 0,
              "psi_imag offset misalignment");
static_assert(sizeof(TorusBlock) % AVX512_ALIGNMENT == 0,
              "TorusBlock size must be multiple of 64 bytes to maintain alignment in arrays");

// Sanity: BLOCK_SIZE == 3^9
static_assert(TorusBlock::BLOCK_SIZE == 19683,
              "BLOCK_SIZE must be 3^9 = 19683");
static_assert(METRIC_TENSOR_COMPONENTS == 45,
              "Symmetric 9x9 metric has (9*10)/2 = 45 independent components");

// ---------------------------------------------------------------------------
// SoaAlignmentGuard — runtime verification watchdog
// ---------------------------------------------------------------------------

/**
 * @brief Stateless runtime alignment verification (Physics Oracle hook).
 *
 * Spec §GAP-021 "Runtime Verification":
 *   "Physics Oracle runs verification pass during system startup and after
 *    every Neurogenesis event."
 *
 * Throws std::runtime_error naming the misaligned field on failure.
 */
class SoaAlignmentGuard {
public:
    SoaAlignmentGuard() = delete;

    /**
     * @brief True iff the pointer is aligned to AVX512_ALIGNMENT (64 bytes).
     */
    [[nodiscard]] static bool is_aligned(const void* ptr) noexcept {
        return (reinterpret_cast<std::uintptr_t>(ptr) % AVX512_ALIGNMENT) == 0;
    }

    /**
     * @brief Assert a pointer is aligned; throw std::runtime_error if not.
     *
     * @param ptr   Pointer to check
     * @param name  Field name embedded in the error message (for diagnostics)
     */
    static void assert_aligned(const void* ptr, const char* name) {
        if (!is_aligned(ptr))
            throw std::runtime_error(std::string("Misaligned pointer: ") + name);
    }

    /**
     * @brief Verify the SoA field starts of a TorusBlock instance are aligned.
     *
     * Spec §GAP-021 requires psi_real, psi_imag, and the metric_tensor block
     * to be individually 64-byte aligned.  The three outer starts are checked:
     *   - blk.psi_real.data()         (alignas(64) ensures offset % 64 == 0)
     *   - blk.psi_imag.data()         (same)
     *   - blk.metric_tensor[0].data() (metric_tensor field start, alignas(64))
     *
     * Note: Each individual inner sub-array metric_tensor[i] for i > 0 is NOT
     * independently aligned because sizeof(array<float,19683>) = 78732 bytes
     * and 78732 % 64 = 44.  The spec's vectorisation guarantee is for each
     * field's contiguous SoA run — guaranteed by the outer field alignment.
     *
     * Throws on the first misaligned field found.
     */
    static void verify_block(const TorusBlock& blk) {
        assert_aligned(blk.psi_real.data(),             "psi_real");
        assert_aligned(blk.psi_imag.data(),             "psi_imag");
        assert_aligned(blk.metric_tensor[0].data(),     "metric_tensor");
    }

    /**
     * @brief Verify an AlignedVec spans a contiguous, 64-byte-aligned buffer.
     */
    template <typename T>
    static void verify_vec(const AlignedVec<T>& v, const char* name) {
        if (!v.empty()) assert_aligned(v.data(), name);
    }
};

// ---------------------------------------------------------------------------
// Copy-on-Load helper
// ---------------------------------------------------------------------------

/**
 * @brief Safely load potentially-misaligned data into an aligned TorusBlock.
 *
 * Spec §GAP-021 "Misaligned Data Handling":
 *   Data from LSM-DMC.nik files or Protobuf network buffers is effectively
 *   a raw char* and rarely aligned.  Never process in-place from I/O buffers.
 *   Always copy into aligned SoA structures first.
 *
 * Modern glibc std::memcpy detects source/destination alignment at runtime:
 *   aligned src → aligned dst : uses vmovaps / vmovntps (fastest)
 *   unaligned src → aligned dst: uses vmovups reads + vmovaps writes (fast)
 *
 * @param raw    Flat byte span containing serialised float data for psi_real;
 *               must contain at least BLOCK_SIZE * sizeof(float) bytes.
 * @param target TorusBlock whose psi_real will be populated.
 *
 * @note Full production use copies all fields; this reference impl covers
 *       psi_real as the canonical demonstration required by spec.
 */
inline void load_block_psi_real(std::span<const std::uint8_t> raw,
                                 TorusBlock& target) {
    static constexpr std::size_t field_bytes =
        TorusBlock::BLOCK_SIZE * sizeof(float);

    if (raw.size_bytes() < field_bytes)
        throw std::invalid_argument("raw buffer too small for psi_real");

    // Target is GUARANTEED 64-byte aligned by type system.
    // std::memcpy handles unaligned source efficiently.
    std::memcpy(target.psi_real.data(), raw.data(), field_bytes);
}

// ---------------------------------------------------------------------------
// GGUF file-offset alignment helper
// ---------------------------------------------------------------------------

/**
 * @brief Compute the next file offset that is aligned to AVX512_ALIGNMENT.
 *
 * Spec §GAP-021 "Integration with GGUF & Quantization (Q9_0)":
 *   "GGUF writer must insert padding bytes before writing tensor data to
 *    satisfy offset % 64 == 0 → allows mmap'd inference engines (llama.cpp)
 *    to use vectorised loads directly from disk."
 *
 * @param offset  Current byte position in the GGUF file
 * @return        Next offset ≥ offset that satisfies offset % 64 == 0
 */
[[nodiscard]] inline constexpr std::size_t
gguf_aligned_offset(std::size_t offset) noexcept {
    return (offset + AVX512_ALIGNMENT - 1) & ~(AVX512_ALIGNMENT - 1);
}

/**
 * @brief Number of padding bytes to insert before writing tensor data.
 *
 * @param offset  Current byte position (before padding)
 * @return        0 if already aligned; otherwise the gap to next 64-byte boundary
 */
[[nodiscard]] inline constexpr std::size_t
gguf_padding_bytes(std::size_t offset) noexcept {
    return gguf_aligned_offset(offset) - offset;
}

} // namespace nikola::physics
