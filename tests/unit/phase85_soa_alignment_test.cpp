// =============================================================================
// tests/unit/phase85_soa_alignment_test.cpp
// Phase 85 — GAP-021: TorusGridSoA Memory Alignment Guarantees
//
// Tests for nikola::physics::soa_alignment.hpp
// Spec: docs/info/integration/sections/02_foundations/01_9d_toroidal_geometry.md
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cstdlib>

#include "nikola/physics/soa_alignment.hpp"

using namespace nikola::physics;
using Catch::Approx;

// ---------------------------------------------------------------------------
// § Alignment constants
// ---------------------------------------------------------------------------

TEST_CASE("AVX512_ALIGNMENT is 64 bytes", "[constants][phase85]") {
    CHECK(AVX512_ALIGNMENT == 64u);
}

TEST_CASE("ZMM_BYTES is 64", "[constants][phase85]") {
    CHECK(ZMM_BYTES == 64u);
}

TEST_CASE("ZMM_FLOATS is 16", "[constants][phase85]") {
    CHECK(ZMM_FLOATS == 16u);
}

TEST_CASE("ZMM_DOUBLES is 8", "[constants][phase85]") {
    CHECK(ZMM_DOUBLES == 8u);
}

TEST_CASE("CACHE_LINE_BYTES is 64", "[constants][phase85]") {
    CHECK(CACHE_LINE_BYTES == 64u);
}

TEST_CASE("CUDA_WARP_BYTES is 128", "[constants][phase85]") {
    CHECK(CUDA_WARP_BYTES == 128u);
}

TEST_CASE("STD_ALLOC_ALIGNMENT is 16", "[constants][phase85]") {
    CHECK(STD_ALLOC_ALIGNMENT == 16u);
}

// ---------------------------------------------------------------------------
// § Grid geometry constants
// ---------------------------------------------------------------------------

TEST_CASE("TORUS_BLOCK_SIZE is 3^9 = 19683", "[constants][phase85]") {
    CHECK(TORUS_BLOCK_SIZE == 19683u);
}

TEST_CASE("METRIC_COMPONENTS is 45", "[constants][phase85]") {
    CHECK(METRIC_COMPONENTS == 45u);
}

TEST_CASE("PSI_CHANNELS is 2", "[constants][phase85]") {
    CHECK(PSI_CHANNELS == 2u);
}

// ---------------------------------------------------------------------------
// § Stride constants
// ---------------------------------------------------------------------------

TEST_CASE("AOS_NODE_STRIDE_BYTES is (PSI_CHANNELS + METRIC_COMPONENTS)*4", "[constants][phase85]") {
    // (2 + 45) * 4 = 188 bytes
    CHECK(AOS_NODE_STRIDE_BYTES == (PSI_CHANNELS + METRIC_COMPONENTS) * sizeof(float));
    CHECK(AOS_NODE_STRIDE_BYTES == 188u);
}

TEST_CASE("SOA_NODE_STRIDE_BYTES is sizeof(float) = 4", "[constants][phase85]") {
    CHECK(SOA_NODE_STRIDE_BYTES == sizeof(float));
    CHECK(SOA_NODE_STRIDE_BYTES == 4u);
}

// ---------------------------------------------------------------------------
// § padded_size
// ---------------------------------------------------------------------------

TEST_CASE("padded_size rounds up to alignment boundary", "[functions][phase85]") {
    CHECK(padded_size(0,  64) == 0u);
    CHECK(padded_size(1,  64) == 64u);
    CHECK(padded_size(63, 64) == 64u);
    CHECK(padded_size(64, 64) == 64u);
    CHECK(padded_size(65, 64) == 128u);
    CHECK(padded_size(128, 64) == 128u);
}

// ---------------------------------------------------------------------------
// § is_aligned / is_avx512_aligned
// ---------------------------------------------------------------------------

TEST_CASE("is_aligned correctly identifies alignment", "[functions][phase85]") {
    alignas(64) char buf[128];
    CHECK(is_aligned(buf, 64) == true);
    CHECK(is_aligned(buf + 1, 64) == false);
    CHECK(is_aligned(buf, 16) == true);
}

TEST_CASE("is_avx512_aligned requires 64-byte alignment", "[functions][phase85]") {
    alignas(64) char buf[64];
    CHECK(is_avx512_aligned(buf) == true);
    CHECK(is_avx512_aligned(buf + 1) == false);
}

TEST_CASE("is_cacheline_aligned matches 64-byte alignment", "[functions][phase85]") {
    alignas(64) char buf[64];
    CHECK(is_cacheline_aligned(buf) == true);
}

// ---------------------------------------------------------------------------
// § avx512_alloc_size
// ---------------------------------------------------------------------------

TEST_CASE("avx512_alloc_size<float> returns padded byte count", "[functions][phase85]") {
    CHECK(avx512_alloc_size<float>(16)  == 64u);   // exactly one ZMM
    CHECK(avx512_alloc_size<float>(17)  == 128u);  // spills into second ZMM
    CHECK(avx512_alloc_size<float>(1)   == 64u);   // minimum one ZMM
}

// ---------------------------------------------------------------------------
// § AVX-512 lane helpers
// ---------------------------------------------------------------------------

TEST_CASE("avx512_float_lanes computes full ZMM lanes", "[functions][phase85]") {
    CHECK(avx512_float_lanes(0)  == 0u);
    CHECK(avx512_float_lanes(16) == 1u);
    CHECK(avx512_float_lanes(32) == 2u);
    CHECK(avx512_float_lanes(17) == 1u);  // 17/16 = 1 full lane
}

TEST_CASE("avx512_float_epilogue computes remainder floats", "[functions][phase85]") {
    CHECK(avx512_float_epilogue(16) == 0u);
    CHECK(avx512_float_epilogue(17) == 1u);
    CHECK(avx512_float_epilogue(0)  == 0u);
}

TEST_CASE("avx512_no_epilogue is true when n divisible by ZMM_FLOATS", "[functions][phase85]") {
    CHECK(avx512_no_epilogue(0)  == true);
    CHECK(avx512_no_epilogue(16) == true);
    CHECK(avx512_no_epilogue(32) == true);
    CHECK(avx512_no_epilogue(17) == false);
}

// ---------------------------------------------------------------------------
// § SoA bandwidth efficiency
// ---------------------------------------------------------------------------

TEST_CASE("soa_bandwidth_efficiency returns value in (0, 1]", "[functions][phase85]") {
    double eff = soa_bandwidth_efficiency();
    CHECK(eff > 0.0);
    CHECK(eff <= 1.0);
}
