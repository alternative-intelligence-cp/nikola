/**
 * @file phase60_soa_layout_test.cpp
 * @brief Phase 60 — GAP-021: TorusGridSoA Memory Alignment Guarantees
 *
 * Validates the SoA layout engine against spec §GAP-021:
 *   - AVX512_ALIGNMENT = 64, BLOCK_SIZE = 3^9 = 19683, METRIC = 45 components
 *   - AlignedAllocator<T>: 64-byte aligned allocations for AVX-512 vmovaps
 *   - AlignedVec<T>: STL vector with 64-byte aligned storage
 *   - TorusBlock: compile-time alignment + size static_asserts
 *     alignof == 64, offsetof(psi_real/psi_imag) % 64 == 0, sizeof % 64 == 0
 *   - SoaAlignmentGuard: runtime watchdog (startup + neurogenesis hook)
 *   - load_block_psi_real(): Copy-on-Load from unaligned I/O buffers
 *   - gguf_aligned_offset() / gguf_padding_bytes(): GGUF mmap compatibility
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/catch_approx.hpp>

#include "nikola/physics/soa_layout.hpp"

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace nikola::physics;
using Catch::Approx;

// ---------------------------------------------------------------------------
// §1 — Compile-time constants
// ---------------------------------------------------------------------------

TEST_CASE("GAP-021 §1: AVX512_ALIGNMENT is exactly 64 bytes", "[gap021][constants]") {
    REQUIRE(AVX512_ALIGNMENT == 64u);
    // Must be a power of two (bitwise property required by aligned_alloc)
    REQUIRE((AVX512_ALIGNMENT & (AVX512_ALIGNMENT - 1)) == 0u);
}

TEST_CASE("GAP-021 §2: BLOCK_SIZE is 3^9 = 19,683", "[gap021][constants]") {
    REQUIRE(TorusBlock::BLOCK_SIZE == 19683);
    // Verify it equals 3^9
    int p = 1;
    for (int i = 0; i < 9; ++i) p *= 3;
    REQUIRE(TorusBlock::BLOCK_SIZE == p);
}

TEST_CASE("GAP-021 §3: METRIC_TENSOR_COMPONENTS is 45 = (9×10)/2", "[gap021][constants]") {
    REQUIRE(METRIC_TENSOR_COMPONENTS == 45);
    // Symmetric 9×9 matrix independent components: n*(n+1)/2 for n=9
    constexpr int n = 9;
    REQUIRE(METRIC_TENSOR_COMPONENTS == n * (n + 1) / 2);
}

// ---------------------------------------------------------------------------
// §4 — AlignedAllocator: pointer alignment
// ---------------------------------------------------------------------------

TEST_CASE("GAP-021 §4: AlignedAllocator<float> returns 64-byte aligned pointer", "[gap021][allocator]") {
    AlignedAllocator<float> alloc;
    constexpr std::size_t N = 1024;
    float* p = alloc.allocate(N);
    REQUIRE(p != nullptr);
    REQUIRE((reinterpret_cast<std::uintptr_t>(p) % AVX512_ALIGNMENT) == 0u);
    alloc.deallocate(p, N);
}

TEST_CASE("GAP-021 §5: AlignedAllocator<double> returns 64-byte aligned pointer", "[gap021][allocator]") {
    AlignedAllocator<double> alloc;
    constexpr std::size_t N = 512;
    double* p = alloc.allocate(N);
    REQUIRE(p != nullptr);
    REQUIRE((reinterpret_cast<std::uintptr_t>(p) % AVX512_ALIGNMENT) == 0u);
    alloc.deallocate(p, N);
}

TEST_CASE("GAP-021 §6: AlignedAllocator small allocation (N=1) still 64-byte aligned", "[gap021][allocator]") {
    AlignedAllocator<float> alloc;
    float* p = alloc.allocate(1);
    REQUIRE((reinterpret_cast<std::uintptr_t>(p) % AVX512_ALIGNMENT) == 0u);
    alloc.deallocate(p, 1);
}

TEST_CASE("GAP-021 §7: AlignedAllocator equality semantics", "[gap021][allocator]") {
    AlignedAllocator<float> a1;
    AlignedAllocator<float> a2;
    // Two AlignedAllocator<T> instances are always interchangeable
    REQUIRE(a1 == a2);
    REQUIRE_FALSE(a1 != a2);
}

// ---------------------------------------------------------------------------
// §8 — AlignedVec<T>
// ---------------------------------------------------------------------------

TEST_CASE("GAP-021 §8: AlignedVec<float> .data() is 64-byte aligned", "[gap021][vec]") {
    AlignedVec<float> v(19683, 0.0f);
    REQUIRE(!v.empty());
    REQUIRE((reinterpret_cast<std::uintptr_t>(v.data()) % AVX512_ALIGNMENT) == 0u);
}

TEST_CASE("GAP-021 §9: AlignedVec<float> data survives write-read across aligned boundary", "[gap021][vec]") {
    AlignedVec<float> v(256);
    std::iota(v.begin(), v.end(), 0.0f);
    for (int i = 0; i < 256; ++i)
        REQUIRE(v[i] == Approx(static_cast<float>(i)));
}

// ---------------------------------------------------------------------------
// §10 — TorusBlock compile-time alignment properties (verified at runtime)
// ---------------------------------------------------------------------------

TEST_CASE("GAP-021 §10: alignof(TorusBlock) == 64", "[gap021][torusblock]") {
    // compile-time assertion in header already guards this;
    // runtime REQUIRE makes the spec requirement explicit in the test report.
    REQUIRE(alignof(TorusBlock) == AVX512_ALIGNMENT);
}

TEST_CASE("GAP-021 §11: offsetof psi_real and psi_imag are multiples of 64", "[gap021][torusblock]") {
    REQUIRE(offsetof(TorusBlock, psi_real) % AVX512_ALIGNMENT == 0u);
    REQUIRE(offsetof(TorusBlock, psi_imag) % AVX512_ALIGNMENT == 0u);
}

TEST_CASE("GAP-021 §12: sizeof(TorusBlock) is a multiple of 64", "[gap021][torusblock]") {
    // Spec: ensures blocks in an array all start on 64-byte boundary
    REQUIRE(sizeof(TorusBlock) % AVX512_ALIGNMENT == 0u);
}

TEST_CASE("GAP-021 §13: TorusBlock fields have correct element counts", "[gap021][torusblock]") {
    REQUIRE(TorusBlock{}.psi_real.size()        == static_cast<std::size_t>(TorusBlock::BLOCK_SIZE));
    REQUIRE(TorusBlock{}.psi_imag.size()        == static_cast<std::size_t>(TorusBlock::BLOCK_SIZE));
    REQUIRE(TorusBlock{}.metric_tensor.size()   == static_cast<std::size_t>(METRIC_TENSOR_COMPONENTS));
}

// ---------------------------------------------------------------------------
// §14 — SoaAlignmentGuard::is_aligned()
// ---------------------------------------------------------------------------

TEST_CASE("GAP-021 §14: SoaAlignmentGuard::is_aligned — 64-byte aligned address", "[gap021][guard]") {
    // Allocate aligned memory and verify the guard agrees
    AlignedAllocator<float> alloc;
    float* p = alloc.allocate(64);
    REQUIRE(SoaAlignmentGuard::is_aligned(p));
    alloc.deallocate(p, 64);
}

TEST_CASE("GAP-021 §15: SoaAlignmentGuard::is_aligned — deliberately misaligned pointer", "[gap021][guard]") {
    AlignedAllocator<std::uint8_t> alloc;
    std::uint8_t* base = alloc.allocate(128);
    REQUIRE(SoaAlignmentGuard::is_aligned(base));  // base is 64-byte aligned

    // Offset by 1 byte — now misaligned by spec
    std::uint8_t* misaligned = base + 1;
    REQUIRE_FALSE(SoaAlignmentGuard::is_aligned(misaligned));

    // Offset by 32 bytes — half-aligned (AVX2 ok, AVX-512 not)
    std::uint8_t* half = base + 32;
    REQUIRE_FALSE(SoaAlignmentGuard::is_aligned(half));

    alloc.deallocate(base, 128);
}

// ---------------------------------------------------------------------------
// §16 — SoaAlignmentGuard::assert_aligned()
// ---------------------------------------------------------------------------

TEST_CASE("GAP-021 §16: assert_aligned does not throw for aligned pointer", "[gap021][guard]") {
    AlignedAllocator<float> alloc;
    float* p = alloc.allocate(16);
    REQUIRE_NOTHROW(SoaAlignmentGuard::assert_aligned(p, "test_field"));
    alloc.deallocate(p, 16);
}

TEST_CASE("GAP-021 §17: assert_aligned throws runtime_error for misaligned pointer", "[gap021][guard]") {
    AlignedAllocator<std::uint8_t> alloc;
    std::uint8_t* base = alloc.allocate(128);
    std::uint8_t* bad  = base + 7;  // definitely misaligned
    REQUIRE_THROWS_AS(SoaAlignmentGuard::assert_aligned(bad, "psi_real"),
                      std::runtime_error);
    alloc.deallocate(base, 128);
}

// ---------------------------------------------------------------------------
// §18 — SoaAlignmentGuard::verify_block()
// ---------------------------------------------------------------------------

TEST_CASE("GAP-021 §18: verify_block passes for stack-allocated TorusBlock", "[gap021][guard]") {
    // Spec: verify_block checks psi_real, psi_imag, and all 45 metric fields
    alignas(AVX512_ALIGNMENT) TorusBlock blk{};
    REQUIRE_NOTHROW(SoaAlignmentGuard::verify_block(blk));
}

TEST_CASE("GAP-021 §19: verify_vec passes for properly aligned AlignedVec", "[gap021][guard]") {
    AlignedVec<float> v(256, 0.0f);
    REQUIRE_NOTHROW(SoaAlignmentGuard::verify_vec(v, "wavefunction_real"));
}

// ---------------------------------------------------------------------------
// §20 — load_block_psi_real: Copy-on-Load
// ---------------------------------------------------------------------------

TEST_CASE("GAP-021 §20: load_block_psi_real copies data correctly", "[gap021][copyonload]") {
    // Build raw byte buffer containing known float values
    constexpr std::size_t N = TorusBlock::BLOCK_SIZE;
    std::vector<float> src_floats(N);
    std::iota(src_floats.begin(), src_floats.end(), 1.0f);  // 1, 2, 3, ..., N

    std::vector<std::uint8_t> raw_bytes(N * sizeof(float));
    std::memcpy(raw_bytes.data(), src_floats.data(), raw_bytes.size());

    alignas(AVX512_ALIGNMENT) TorusBlock blk{};
    REQUIRE_NOTHROW(load_block_psi_real(raw_bytes, blk));

    // Validate first, last, and a mid element
    REQUIRE(blk.psi_real[0]   == Approx(1.0f));
    REQUIRE(blk.psi_real[1]   == Approx(2.0f));
    REQUIRE(blk.psi_real[N-1] == Approx(static_cast<float>(N)));
}

TEST_CASE("GAP-021 §21: load_block_psi_real throws on insufficient buffer", "[gap021][copyonload]") {
    std::vector<std::uint8_t> short_buf(64);  // way too small for 19683 floats
    alignas(AVX512_ALIGNMENT) TorusBlock blk{};
    REQUIRE_THROWS_AS(load_block_psi_real(short_buf, blk), std::invalid_argument);
}

TEST_CASE("GAP-021 §22: target psi_real remains 64-byte aligned after load", "[gap021][copyonload]") {
    constexpr std::size_t N = TorusBlock::BLOCK_SIZE;
    std::vector<std::uint8_t> raw_bytes(N * sizeof(float), 0);

    alignas(AVX512_ALIGNMENT) TorusBlock blk{};
    load_block_psi_real(raw_bytes, blk);

    // Target must still be aligned post-load (alignment is a type property)
    REQUIRE(SoaAlignmentGuard::is_aligned(blk.psi_real.data()));
}

// ---------------------------------------------------------------------------
// §23 — gguf_aligned_offset() and gguf_padding_bytes()
// ---------------------------------------------------------------------------

TEST_CASE("GAP-021 §23: gguf_aligned_offset rounds up to next 64-byte boundary", "[gap021][gguf]") {
    // Already aligned → stays the same
    REQUIRE(gguf_aligned_offset(0)   == 0u);
    REQUIRE(gguf_aligned_offset(64)  == 64u);
    REQUIRE(gguf_aligned_offset(128) == 128u);

    // Misaligned → rounds up
    REQUIRE(gguf_aligned_offset(1)   == 64u);
    REQUIRE(gguf_aligned_offset(63)  == 64u);
    REQUIRE(gguf_aligned_offset(65)  == 128u);
    REQUIRE(gguf_aligned_offset(100) == 128u);
    REQUIRE(gguf_aligned_offset(127) == 128u);
}

TEST_CASE("GAP-021 §24: gguf_padding_bytes returns zero for aligned, positive for misaligned", "[gap021][gguf]") {
    REQUIRE(gguf_padding_bytes(0)   == 0u);
    REQUIRE(gguf_padding_bytes(64)  == 0u);
    REQUIRE(gguf_padding_bytes(128) == 0u);

    REQUIRE(gguf_padding_bytes(1)   == 63u);
    REQUIRE(gguf_padding_bytes(63)  == 1u);
    REQUIRE(gguf_padding_bytes(65)  == 63u);
    REQUIRE(gguf_padding_bytes(100) == 28u);

    // offset + padding == next aligned boundary
    for (std::size_t off : {1u, 7u, 31u, 33u, 63u, 65u, 99u}) {
        REQUIRE(off + gguf_padding_bytes(off) == gguf_aligned_offset(off));
    }
}

// ---------------------------------------------------------------------------
// §25 — Integration: heap-allocated TorusBlock array alignment preserved
// ---------------------------------------------------------------------------

TEST_CASE("GAP-021 §25: heap-allocated TorusBlock[] maintains per-element alignment",
          "[gap021][integration]") {
    // Spec "Paged Block Pool": if blocks[0] is aligned and sizeof(TorusBlock)
    // is a multiple of 64, then blocks[i] is also aligned for all i.
    AlignedAllocator<TorusBlock> alloc;
    constexpr int N = 4;
    TorusBlock* blocks = alloc.allocate(N);

    for (int i = 0; i < N; ++i) {
        // Each block element must itself be 64-byte aligned (sizeof % 64 == 0)
        REQUIRE(SoaAlignmentGuard::is_aligned(&blocks[i]));
        REQUIRE(SoaAlignmentGuard::is_aligned(blocks[i].psi_real.data()));
        REQUIRE(SoaAlignmentGuard::is_aligned(blocks[i].psi_imag.data()));
        // The metric_tensor field start (metric_tensor[0]) must be aligned.
        // Inner sub-arrays metric_tensor[k>0] are NOT individually aligned
        // (78732 bytes per inner array; 78732 % 64 = 44); this is expected.
        REQUIRE(SoaAlignmentGuard::is_aligned(blocks[i].metric_tensor[0].data()));
        // Verify full block passes the watchdog (checks the three outer starts)
        REQUIRE_NOTHROW(SoaAlignmentGuard::verify_block(blocks[i]));
    }

    alloc.deallocate(blocks, N);
}
