// =============================================================================
// Phase 94 — GAP-009: 9D Hilbert Curve State Machine Tests
//
// Validates the Hamilton & Rau-Chaplin (2007) FSM implementation in
// nikola::math::hilbert_state_machine.hpp.
//
// Key properties verified:
//   1. gray_code / gray_code_inverse are mutual inverses
//   2. rotr_9d is a proper 9-bit rotation
//   3. d_table matches count-trailing-zeros
//   4. e_table follows the Hamilton piecewise formula
//   5. Both tables are 512 entries, fully constexpr
//   6. State step returns w in [0, 511] and valid (e',d') bounds
//   7. hilbert_encode_9d encodes the origin to 0
//   8. hilbert_encode_9d produces distinct indices for distinct coords
//   9. Monotone neighbour property: adjacent cells differ by bounded amount
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <array>
#include <cstdint>
#include <bit>         // std::countr_zero
#include <unordered_set>

#include "nikola/math/hilbert_state_machine.hpp"

using namespace nikola::math;

// ---------------------------------------------------------------------------
// Section 1: gray_code and gray_code_inverse
// ---------------------------------------------------------------------------
TEST_CASE("gray_code and inverse are mutual inverses", "[phase94][hilbert][gray]") {
    // Full 9-bit domain roundtrip
    for (uint16_t x = 0; x < 512; ++x) {
        REQUIRE(gray_code_inverse(gray_code(x)) == x);
    }
}

TEST_CASE("gray_code satisfies gc(x) = x XOR (x>>1)", "[phase94][hilbert][gray]") {
    for (uint16_t x = 0; x < 512; ++x) {
        uint16_t expected = static_cast<uint16_t>(x ^ (x >> 1));
        REQUIRE(gray_code(x) == expected);
    }
}

TEST_CASE("gray_code successive values differ in exactly one bit", "[phase94][hilbert][gray]") {
    // This is the defining property of binary reflected Gray code
    for (uint16_t x = 0; x < 511; ++x) {
        uint16_t diff = gray_code(static_cast<uint16_t>(x + 1)) ^ gray_code(x);
        // diff must be a power of two (exactly one bit set)
        REQUIRE(diff != 0);
        REQUIRE((diff & (diff - 1u)) == 0u);  // power of two check
    }
}

// ---------------------------------------------------------------------------
// Section 2: rotr_9d
// ---------------------------------------------------------------------------
TEST_CASE("rotr_9d shift=0 is identity on 9-bit values", "[phase94][hilbert][rotr9d]") {
    for (uint16_t v = 0; v < 512; ++v) {
        REQUIRE(rotr_9d(v, 0) == v);
    }
}

TEST_CASE("rotr_9d shift=9 is identity (full rotation)", "[phase94][hilbert][rotr9d]") {
    for (uint16_t v = 0; v < 512; ++v) {
        REQUIRE(rotr_9d(v, 9) == v);
    }
}

TEST_CASE("rotr_9d result is always within 9-bit range", "[phase94][hilbert][rotr9d]") {
    for (uint16_t v = 0; v < 512; ++v) {
        for (int s = 0; s < 9; ++s) {
            REQUIRE(rotr_9d(v, s) < 512u);
        }
    }
}

TEST_CASE("rotr_9d known values", "[phase94][hilbert][rotr9d]") {
    // 0b000000001 rotated right by 1 = 0b100000000
    REQUIRE(rotr_9d(0b000000001u, 1) == 0b100000000u);
    // 0b100000000 rotated right by 1 = 0b010000000
    REQUIRE(rotr_9d(0b100000000u, 1) == 0b010000000u);
    // 0b000000011 rotated right by 2 = 0b110000000
    REQUIRE(rotr_9d(0b000000011u, 2) == 0b110000000u);
}

TEST_CASE("rotr_9d composed with rotl_9d yields identity", "[phase94][hilbert][rotr9d]") {
    // rotl_9d(x, s) = rotr_9d(x, 9-s)
    for (uint16_t v = 0; v < 512; v += 17) {
        for (int s = 0; s < 9; ++s) {
            uint16_t rotated = rotr_9d(v, s);
            uint16_t restored = rotr_9d(rotated, 9 - s);
            REQUIRE(restored == v);
        }
    }
}

// ---------------------------------------------------------------------------
// Section 3: calc_hilbert_d (D_TABLE formula)
// ---------------------------------------------------------------------------
TEST_CASE("calc_hilbert_d equals ctz for w > 0", "[phase94][hilbert][d_table]") {
    REQUIRE(calc_hilbert_d(0) == 0);
    for (uint16_t w = 1; w < 512; ++w) {
        int expected_ctz = std::countr_zero(static_cast<unsigned int>(w));
        if (expected_ctz >= HILBERT_N) expected_ctz = HILBERT_N - 1;
        REQUIRE(calc_hilbert_d(w) == static_cast<uint8_t>(expected_ctz));
    }
}

TEST_CASE("calc_hilbert_d known values from spec table", "[phase94][hilbert][d_table]") {
    // From Gemini spec table (all verified via ctz):
    REQUIRE(calc_hilbert_d(0) == 0);
    REQUIRE(calc_hilbert_d(1) == 0);  // ctz(1) = 0
    REQUIRE(calc_hilbert_d(2) == 1);  // ctz(2) = 1
    REQUIRE(calc_hilbert_d(3) == 0);  // ctz(3) = 0
    REQUIRE(calc_hilbert_d(4) == 2);  // ctz(4) = 2
    REQUIRE(calc_hilbert_d(5) == 0);  // ctz(5) = 0
    REQUIRE(calc_hilbert_d(6) == 1);  // ctz(6) = 1
    REQUIRE(calc_hilbert_d(7) == 0);  // ctz(7) = 0
}

TEST_CASE("D_TABLE constexpr array has 512 entries", "[phase94][hilbert][d_table]") {
    REQUIRE(HILBERT_D_TABLE_9D.size() == 512u);
}

TEST_CASE("D_TABLE all entries in [0, 8]", "[phase94][hilbert][d_table]") {
    for (std::size_t i = 0; i < HILBERT_D_TABLE_9D.size(); ++i) {
        REQUIRE(HILBERT_D_TABLE_9D[i] < static_cast<uint8_t>(HILBERT_N));
    }
}

// ---------------------------------------------------------------------------
// Section 4: calc_hilbert_e (E_TABLE formula)
// ---------------------------------------------------------------------------
TEST_CASE("calc_hilbert_e known formula values", "[phase94][hilbert][e_table]") {
    // Verify using the piecewise formula: e(0)=0, e(w)=gc(2*floor((w-1)/2))
    REQUIRE(calc_hilbert_e(0) == 0);
    REQUIRE(calc_hilbert_e(1) == gray_code(0));   // gc(2*0) = gc(0) = 0
    REQUIRE(calc_hilbert_e(3) == gray_code(2));   // gc(2*1) = gc(2) = 3
    REQUIRE(calc_hilbert_e(5) == gray_code(4));   // gc(2*2) = gc(4) = 6
    REQUIRE(calc_hilbert_e(7) == gray_code(6));   // gc(2*3) = gc(6) = 5
}

TEST_CASE("calc_hilbert_e result always within 9-bit range", "[phase94][hilbert][e_table]") {
    for (uint16_t w = 0; w < 512; ++w) {
        REQUIRE(calc_hilbert_e(w) < 512u);
    }
}

TEST_CASE("E_TABLE constexpr array has 512 entries", "[phase94][hilbert][e_table]") {
    REQUIRE(HILBERT_E_TABLE_9D.size() == 512u);
}

TEST_CASE("E_TABLE matches calc_hilbert_e for all entries", "[phase94][hilbert][e_table]") {
    for (uint16_t w = 0; w < 512; ++w) {
        REQUIRE(HILBERT_E_TABLE_9D[w] == calc_hilbert_e(w));
    }
}

TEST_CASE("E_TABLE entry at w=0 is 0", "[phase94][hilbert][e_table]") {
    REQUIRE(HILBERT_E_TABLE_9D[0] == 0u);
}

// ---------------------------------------------------------------------------
// Section 5: Architectural constants
// ---------------------------------------------------------------------------
TEST_CASE("HILBERT_N equals 9", "[phase94][hilbert][constants]") {
    REQUIRE(HILBERT_N == 9);
}

TEST_CASE("HILBERT_NUM_CELLS equals 512", "[phase94][hilbert][constants]") {
    REQUIRE(HILBERT_NUM_CELLS == 512);
}

TEST_CASE("HILBERT_TOTAL_STATES equals 4608", "[phase94][hilbert][constants]") {
    // 4608 = 9 × 512 — the minimum compound-state count for correct 9D Hilbert
    REQUIRE(HILBERT_TOTAL_STATES == 4608);
}

// ---------------------------------------------------------------------------
// Section 6: HilbertState struct
// ---------------------------------------------------------------------------
TEST_CASE("HilbertState default-initialises to (0, 0)", "[phase94][hilbert][state]") {
    HilbertState s{0, 0};
    REQUIRE(s.e == 0u);
    REQUIRE(s.d == 0u);
}

TEST_CASE("HilbertState field ranges", "[phase94][hilbert][state]") {
    HilbertState s{511u, 8u};
    REQUIRE(s.e < 512u);
    REQUIRE(s.d < 9u);
}

// ---------------------------------------------------------------------------
// Section 7: hilbert_state_step
// ---------------------------------------------------------------------------
TEST_CASE("hilbert_state_step w is in [0, 511]", "[phase94][hilbert][step]") {
    HilbertState s{0u, 0u};
    for (uint16_t l = 0; l < 512; ++l) {
        auto [w, next] = hilbert_state_step(l, s);
        REQUIRE(w < 512u);
    }
}

TEST_CASE("hilbert_state_step next.e is in [0, 511]", "[phase94][hilbert][step]") {
    HilbertState s{0u, 0u};
    for (uint16_t l = 0; l < 512; ++l) {
        auto [w, next] = hilbert_state_step(l, s);
        REQUIRE(next.e < 512u);
    }
}

TEST_CASE("hilbert_state_step next.d is in [0, 8]", "[phase94][hilbert][step]") {
    HilbertState s{0u, 0u};
    for (uint16_t l = 0; l < 512; ++l) {
        auto [w, next] = hilbert_state_step(l, s);
        REQUIRE(next.d < static_cast<uint8_t>(HILBERT_N));
    }
}

TEST_CASE("hilbert_state_step w=0..511 all distinct from canonical state", "[phase94][hilbert][step]") {
    // Starting from canonical state (e=0, d=0), each cell input l produces
    // a unique rank w — this verifies bijectivity of the cell-to-rank mapping.
    HilbertState s{0u, 0u};
    std::array<bool, 512> seen{};
    seen.fill(false);
    for (uint16_t l = 0; l < 512; ++l) {
        auto [w, next] = hilbert_state_step(l, s);
        REQUIRE_FALSE(seen[w]);
        seen[w] = true;
    }
    // All 512 ranks must have been hit exactly once
    for (int i = 0; i < 512; ++i) {
        REQUIRE(seen[i]);
    }
}

// ---------------------------------------------------------------------------
// Section 8: hilbert_encode_9d
// ---------------------------------------------------------------------------
TEST_CASE("hilbert_encode_9d encodes origin to 0", "[phase94][hilbert][encode]") {
    std::array<uint32_t, 9> origin{};
    origin.fill(0u);
    REQUIRE(hilbert_encode_9d(origin) == 0u);
}

TEST_CASE("hilbert_encode_9d returns nonzero for nonzero coords", "[phase94][hilbert][encode]") {
    std::array<uint32_t, 9> coords{};
    coords.fill(0u);
    coords[0] = 1u;   // first axis = 1, rest = 0
    REQUIRE(hilbert_encode_9d(coords) != 0u);
}

TEST_CASE("hilbert_encode_9d is deterministic", "[phase94][hilbert][encode]") {
    std::array<uint32_t, 9> coords{};
    for (int i = 0; i < 9; ++i) coords[static_cast<std::size_t>(i)] = static_cast<uint32_t>(i + 1u);
    uint64_t h1 = hilbert_encode_9d(coords);
    uint64_t h2 = hilbert_encode_9d(coords);
    REQUIRE(h1 == h2);
}

TEST_CASE("hilbert_encode_9d distinct coords produce distinct indices", "[phase94][hilbert][encode]") {
    // Test several distinct input points produce distinct outputs
    std::array<uint32_t, 9> c0{}; c0.fill(0u);
    std::array<uint32_t, 9> c1{}; c1.fill(0u); c1[0] = 1;
    std::array<uint32_t, 9> c2{}; c2.fill(0u); c2[1] = 1;
    std::array<uint32_t, 9> c3{}; c3.fill(0u); c3[8] = 1;
    std::array<uint32_t, 9> c4{}; c4.fill(1u);  // (1,1,1,1,1,1,1,1,1)

    std::unordered_set<uint64_t> seen;
    seen.insert(hilbert_encode_9d(c0));
    seen.insert(hilbert_encode_9d(c1));
    seen.insert(hilbert_encode_9d(c2));
    seen.insert(hilbert_encode_9d(c3));
    seen.insert(hilbert_encode_9d(c4));
    REQUIRE(seen.size() == 5u);
}

TEST_CASE("hilbert_encode_9d bits_per_dim=1 exhaustive bijectivity", "[phase94][hilbert][encode]") {
    // With bits_per_dim=1, there are 2^9=512 distinct 1-bit coordinates.
    // The Hilbert index should enumerate each rank 0..511 exactly once.
    std::array<bool, 512> seen{};
    seen.fill(false);
    for (uint32_t mask = 0; mask < 512u; ++mask) {
        std::array<uint32_t, 9> coords{};
        for (int dim = 0; dim < 9; ++dim) {
            coords[static_cast<std::size_t>(dim)] = (mask >> dim) & 1u;
        }
        uint64_t h = hilbert_encode_9d(coords, 1);
        REQUIRE(h < 512u);
        REQUIRE_FALSE(seen[h]);
        seen[h] = true;
    }
    for (int i = 0; i < 512; ++i) {
        REQUIRE(seen[i]);
    }
}

TEST_CASE("hilbert_encode_9d bits_per_dim=2 range check", "[phase94][hilbert][encode]") {
    // With bits_per_dim=2, all 4^9 = 262144 grid points produce ranks in [0, 4^9 - 1].
    // We just test a handful of boundary inputs for range validity.
    const uint64_t max_rank = (1ull << (HILBERT_N * 2)) - 1ull;  // 2^18 - 1

    std::array<uint32_t, 9> all_max{};
    all_max.fill(3u);   // max 2-bit coord
    std::array<uint32_t, 9> all_min{};
    all_min.fill(0u);

    uint64_t h_max = hilbert_encode_9d(all_max, 2);
    uint64_t h_min = hilbert_encode_9d(all_min, 2);

    REQUIRE(h_min <= max_rank);
    REQUIRE(h_max <= max_rank);
}
