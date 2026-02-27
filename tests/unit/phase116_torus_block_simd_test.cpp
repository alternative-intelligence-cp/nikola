/**
 * @file phase116_torus_block_simd_test.cpp
 * @brief Phase 116 — GAP-021 final: AVX-512 SIMD kernels for TorusBlock.
 *
 * Tests the SIMD arithmetic kernels in torus_block_simd.hpp.
 * Compiled with -mavx512f so the intrinsic path is exercised on this machine.
 *
 * All tests allocate TorusBlock on the heap (std::make_unique) because the
 * struct is ~3.7 MB (19683 nodes × 47 float channels).
 *
 * Tests:
 *   1.  FULL_LOOPS == 1230  (BLOCK_SIZE / 16)
 *   2.  EPILOGUE_START == 19680 and EPILOGUE_COUNT == 3
 *   3.  is_avx512_active() reflects compile-time flag
 *   4.  psi_zero: all psi_real and psi_imag elements become 0
 *   5.  psi_scale by 1.0 is identity
 *   6.  psi_scale by 0.0 zeros all psi fields
 *   7.  psi_scale by 2.0: first element, last element, epilogue elements doubled
 *   8.  psi_scale linearity: scale(α·β) == scale(α) then scale(β), same result
 *   9.  psi_add_scaled SAXPY: dst += α·src correct at first, last, epilogue
 *  10.  psi_norm_sq on known uniform state: value matches analytical formula
 *  11.  psi_norm_sq epilogue test: only set last 3 elements; norm is correct
 *  12.  psi_norm_sq on zero block is 0
 *  13.  psi_renormalize: norm_sq ≈ 1.0 after call
 *  14.  psi_renormalize on zero state: no crash, state stays zero
 *  15.  psi_renormalize idempotent: second call leaves norm ~1.0
 *  16.  metric_scale: components at index 0 and epilogue scaled correctly
 *  17.  metric_scale by 0: all metric tensor values zeroed
 *  18.  psi_add_scaled accumulation: three identical adds == scale by 3
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <nikola/physics/torus_block_simd.hpp>

#include <cmath>
#include <memory>

using namespace nikola::physics;
using namespace nikola::physics::simd;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Allocate a heap TorusBlock, value-initialized to zero.
static std::unique_ptr<TorusBlock> make_block() {
    return std::make_unique<TorusBlock>();
}

/// Fill psi_real and psi_imag of a block with a constant value.
static void fill_psi(TorusBlock& b, float r_val, float i_val) {
    b.psi_real.fill(r_val);
    b.psi_imag.fill(i_val);
}

/// Fill all 45 metric tensor components with a constant value.
static void fill_metric(TorusBlock& b, float val) {
    for (auto& comp : b.metric_tensor)
        comp.fill(val);
}

// Shortcut indices
static constexpr std::size_t IDX_FIRST    = 0;
static constexpr std::size_t IDX_LAST_ZMM = FULL_LOOPS * K_ZMM_F - 1u; // 19679
static constexpr std::size_t IDX_EPI_0    = EPILOGUE_START;        // 19680
static constexpr std::size_t IDX_EPI_1    = EPILOGUE_START + 1u;   // 19681
static constexpr std::size_t IDX_EPI_2    = EPILOGUE_START + 2u;   // 19682

// ---------------------------------------------------------------------------
// Test 1: FULL_LOOPS constant
// ---------------------------------------------------------------------------

TEST_CASE("S6-SIMD constants: FULL_LOOPS == 1230", "[phase116][simd][gap021]") {
    REQUIRE(FULL_LOOPS == 1230u);
}

// ---------------------------------------------------------------------------
// Test 2: Epilogue constants
// ---------------------------------------------------------------------------

TEST_CASE("S6-SIMD constants: EPILOGUE_START == 19680, EPILOGUE_COUNT == 3",
          "[phase116][simd][gap021]") {
    REQUIRE(EPILOGUE_START   == 19680u);
    REQUIRE(EPILOGUE_COUNT   ==     3u);
    REQUIRE(EPILOGUE_START + EPILOGUE_COUNT == static_cast<std::size_t>(TorusBlock::BLOCK_SIZE));
}

// ---------------------------------------------------------------------------
// Test 3: AVX-512 compile-time flag
// ---------------------------------------------------------------------------

TEST_CASE("S6-SIMD: is_avx512_active() reflects compile flag",
          "[phase116][simd]") {
#ifdef __AVX512F__
    REQUIRE(is_avx512_active() == true);
#else
    REQUIRE(is_avx512_active() == false);
#endif
}

// ---------------------------------------------------------------------------
// Test 4: psi_zero
// ---------------------------------------------------------------------------

TEST_CASE("S6-SIMD: psi_zero zeros all psi channels", "[phase116][simd]") {
    auto b = make_block();
    fill_psi(*b, 7.f, -3.f);

    psi_zero(*b);

    for (std::size_t i = 0; i < static_cast<std::size_t>(TorusBlock::BLOCK_SIZE); ++i) {
        REQUIRE(b->psi_real[i] == 0.f);
        REQUIRE(b->psi_imag[i] == 0.f);
    }
}

// ---------------------------------------------------------------------------
// Test 5: psi_scale by 1.0 is identity
// ---------------------------------------------------------------------------

TEST_CASE("S6-SIMD: psi_scale by 1.0 is identity", "[phase116][simd]") {
    auto b = make_block();
    fill_psi(*b, 5.f, -2.f);

    psi_scale(*b, 1.f);

    CHECK(b->psi_real[IDX_FIRST]    == 5.f);
    CHECK(b->psi_real[IDX_LAST_ZMM] == 5.f);
    CHECK(b->psi_real[IDX_EPI_2]    == 5.f);
    CHECK(b->psi_imag[IDX_EPI_0]    == -2.f);
}

// ---------------------------------------------------------------------------
// Test 6: psi_scale by 0.0 zeros psi
// ---------------------------------------------------------------------------

TEST_CASE("S6-SIMD: psi_scale by 0.0 zeros all psi", "[phase116][simd]") {
    auto b = make_block();
    fill_psi(*b, 99.f, 99.f);

    psi_scale(*b, 0.f);

    CHECK(b->psi_real[IDX_FIRST]    == 0.f);
    CHECK(b->psi_real[IDX_LAST_ZMM] == 0.f);
    CHECK(b->psi_real[IDX_EPI_2]    == 0.f);
    CHECK(b->psi_imag[IDX_EPI_0]    == 0.f);
}

// ---------------------------------------------------------------------------
// Test 7: psi_scale by 2.0 doubles values — first, last-ZMM, epilogue
// ---------------------------------------------------------------------------

TEST_CASE("S6-SIMD: psi_scale by 2.0 doubles values at key indices",
          "[phase116][simd]") {
    auto b = make_block();
    fill_psi(*b, 3.f, -1.f);

    psi_scale(*b, 2.f);

    // Beginning of SIMD loop
    CHECK(b->psi_real[IDX_FIRST]    == 6.f);
    // Last element of last full ZMM load (index 19679)
    CHECK(b->psi_real[IDX_LAST_ZMM] == 6.f);
    // Epilogue elements
    CHECK(b->psi_real[IDX_EPI_0]    == 6.f);
    CHECK(b->psi_real[IDX_EPI_1]    == 6.f);
    CHECK(b->psi_real[IDX_EPI_2]    == 6.f);
    // Imaginary channel
    CHECK(b->psi_imag[IDX_FIRST]    == -2.f);
    CHECK(b->psi_imag[IDX_EPI_2]    == -2.f);
}

// ---------------------------------------------------------------------------
// Test 8: psi_scale linearity — scale(α·β) == scale(α) then scale(β)
// ---------------------------------------------------------------------------

TEST_CASE("S6-SIMD: psi_scale linearity scale(a*b) == scale(a) then scale(b)",
          "[phase116][simd]") {
    const float alpha = 1.5f, beta = 0.4f;

    auto b1 = make_block();
    auto b2 = make_block();
    fill_psi(*b1, 2.f, 1.f);
    fill_psi(*b2, 2.f, 1.f);

    psi_scale(*b1, alpha * beta);

    psi_scale(*b2, alpha);
    psi_scale(*b2, beta);

    // Result should match to floating-point precision
    REQUIRE_THAT(b1->psi_real[IDX_FIRST],
                 Catch::Matchers::WithinULP(b2->psi_real[IDX_FIRST], 4));
    REQUIRE_THAT(b1->psi_real[IDX_EPI_2],
                 Catch::Matchers::WithinULP(b2->psi_real[IDX_EPI_2], 4));
}

// ---------------------------------------------------------------------------
// Test 9: psi_add_scaled SAXPY correctness
// ---------------------------------------------------------------------------

TEST_CASE("S6-SIMD: psi_add_scaled dst += alpha*src at first, last, epilogue",
          "[phase116][simd]") {
    auto dst = make_block();
    auto src = make_block();

    fill_psi(*dst, 1.f, 2.f);
    fill_psi(*src, 3.f, 4.f);

    psi_add_scaled(*dst, *src, 2.f);   // dst.re = 1 + 2*3 = 7; dst.im = 2 + 2*4 = 10

    CHECK(dst->psi_real[IDX_FIRST]    == 7.f);
    CHECK(dst->psi_real[IDX_LAST_ZMM] == 7.f);
    CHECK(dst->psi_real[IDX_EPI_0]    == 7.f);
    CHECK(dst->psi_real[IDX_EPI_1]    == 7.f);
    CHECK(dst->psi_real[IDX_EPI_2]    == 7.f);
    CHECK(dst->psi_imag[IDX_FIRST]    == 10.f);
    CHECK(dst->psi_imag[IDX_EPI_2]    == 10.f);
}

// ---------------------------------------------------------------------------
// Test 10: psi_norm_sq on uniform state
// ---------------------------------------------------------------------------

TEST_CASE("S6-SIMD: psi_norm_sq on uniform state matches analytical value",
          "[phase116][simd]") {
    // All psi_real = r, psi_imag = im → norm_sq = N*(r²+im²)
    const float r = 3.f, im = 4.f;
    const float expected = static_cast<float>(TorusBlock::BLOCK_SIZE) * (r*r + im*im);
    // = 19683 * (9 + 16) = 19683 * 25 = 492075

    auto b = make_block();
    fill_psi(*b, r, im);

    const float n2 = psi_norm_sq(*b);

    REQUIRE_THAT(n2, Catch::Matchers::WithinRel(expected, 1e-4f));
}

// ---------------------------------------------------------------------------
// Test 11: psi_norm_sq epilogue — only last 3 elements set
// ---------------------------------------------------------------------------

TEST_CASE("S6-SIMD: psi_norm_sq correctly sums epilogue elements",
          "[phase116][simd]") {
    // Set only the 3 epilogue psi_real elements; imag = 0
    // Expected = 3² + 4² + 5² = 9 + 16 + 25 = 50
    auto b = make_block();   // all zeros
    b->psi_real[IDX_EPI_0] = 3.f;
    b->psi_real[IDX_EPI_1] = 4.f;
    b->psi_real[IDX_EPI_2] = 5.f;

    const float n2 = psi_norm_sq(*b);

    REQUIRE_THAT(n2, Catch::Matchers::WithinAbs(50.f, 1e-4f));
}

// ---------------------------------------------------------------------------
// Test 12: psi_norm_sq on zero block
// ---------------------------------------------------------------------------

TEST_CASE("S6-SIMD: psi_norm_sq on zero block returns 0", "[phase116][simd]") {
    auto b = make_block();   // all zeros
    REQUIRE(psi_norm_sq(*b) == 0.f);
}

// ---------------------------------------------------------------------------
// Test 13: psi_renormalize produces unit norm
// ---------------------------------------------------------------------------

TEST_CASE("S6-SIMD: psi_renormalize yields norm_sq ≈ 1",
          "[phase116][simd]") {
    auto b = make_block();
    fill_psi(*b, 3.f, 4.f);   // known non-unit state

    psi_renormalize(*b);

    const float n2 = psi_norm_sq(*b);
    REQUIRE_THAT(n2, Catch::Matchers::WithinAbs(1.f, 1e-4f));
}

// ---------------------------------------------------------------------------
// Test 14: psi_renormalize on zero state — no crash, stays zero
// ---------------------------------------------------------------------------

TEST_CASE("S6-SIMD: psi_renormalize on zero state is no-op",
          "[phase116][simd]") {
    auto b = make_block();   // all zeros

    REQUIRE_NOTHROW(psi_renormalize(*b));

    CHECK(b->psi_real[IDX_FIRST] == 0.f);
    CHECK(b->psi_imag[IDX_EPI_2] == 0.f);
}

// ---------------------------------------------------------------------------
// Test 15: psi_renormalize idempotent
// ---------------------------------------------------------------------------

TEST_CASE("S6-SIMD: psi_renormalize is idempotent (second call yields norm ≈ 1)",
          "[phase116][simd]") {
    auto b = make_block();
    fill_psi(*b, 1.f, 1.f);

    psi_renormalize(*b);   // first call
    psi_renormalize(*b);   // second call

    const float n2 = psi_norm_sq(*b);
    REQUIRE_THAT(n2, Catch::Matchers::WithinAbs(1.f, 1e-4f));
}

// ---------------------------------------------------------------------------
// Test 16: metric_scale spot-check
// ---------------------------------------------------------------------------

TEST_CASE("S6-SIMD: metric_scale scales all 45 components at key indices",
          "[phase116][simd]") {
    auto b = make_block();
    fill_metric(*b, 2.f);

    metric_scale(*b, 3.f);

    // Check component 0, indices first, last-ZMM, and all 3 epilogue
    CHECK(b->metric_tensor[0][IDX_FIRST]    == 6.f);
    CHECK(b->metric_tensor[0][IDX_LAST_ZMM] == 6.f);
    CHECK(b->metric_tensor[0][IDX_EPI_0]    == 6.f);
    CHECK(b->metric_tensor[0][IDX_EPI_2]    == 6.f);

    // Check last component (44)
    CHECK(b->metric_tensor[44][IDX_FIRST]    == 6.f);
    CHECK(b->metric_tensor[44][IDX_EPI_2]    == 6.f);
}

// ---------------------------------------------------------------------------
// Test 17: metric_scale by 0 zeroes all metric components
// ---------------------------------------------------------------------------

TEST_CASE("S6-SIMD: metric_scale by 0 zeros all metric tensor data",
          "[phase116][simd]") {
    auto b = make_block();
    fill_metric(*b, 99.f);

    metric_scale(*b, 0.f);

    for (int c = 0; c < METRIC_TENSOR_COMPONENTS; ++c) {
        CHECK(b->metric_tensor[static_cast<std::size_t>(c)][IDX_FIRST]  == 0.f);
        CHECK(b->metric_tensor[static_cast<std::size_t>(c)][IDX_EPI_2]  == 0.f);
    }
}

// ---------------------------------------------------------------------------
// Test 18: psi_add_scaled accumulation == psi_scale * 3
// ---------------------------------------------------------------------------

TEST_CASE("S6-SIMD: three psi_add_scaled(dst, src, 1) == psi_scale(dst_scaled, 3)",
          "[phase116][simd]") {
    // Start: both blocks psi_real=1, psi_imag=2
    auto accum  = make_block();
    auto scaled = make_block();
    auto src    = make_block();

    fill_psi(*accum,  1.f, 2.f);
    fill_psi(*scaled, 1.f, 2.f);
    fill_psi(*src,    1.f, 2.f);

    // accum = 1 + 1 + 1 + 1 = 4 (start + 3 adds of src=1)
    psi_add_scaled(*accum, *src, 1.f);
    psi_add_scaled(*accum, *src, 1.f);
    psi_add_scaled(*accum, *src, 1.f);

    // scaled = 1 * 4 = 4
    psi_scale(*scaled, 4.f);

    // Results should match exactly (same floating-point operations)
    REQUIRE_THAT(accum->psi_real[IDX_FIRST],
                 Catch::Matchers::WithinULP(scaled->psi_real[IDX_FIRST], 4));
    REQUIRE_THAT(accum->psi_real[IDX_EPI_2],
                 Catch::Matchers::WithinULP(scaled->psi_real[IDX_EPI_2], 4));
    REQUIRE_THAT(accum->psi_imag[IDX_FIRST],
                 Catch::Matchers::WithinULP(scaled->psi_imag[IDX_FIRST], 4));
    REQUIRE_THAT(accum->psi_imag[IDX_EPI_2],
                 Catch::Matchers::WithinULP(scaled->psi_imag[IDX_EPI_2], 4));
}
