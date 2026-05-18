/**
 * @file tests/unit/v031_audio_emitter_test.cpp
 * @brief v0.3.1 — AudioEmitterLayout spatial placement & frequency tests
 *
 * Tests:
 *   §1  compute_position bounds (all coords within grid)
 *   §2  emitter 0 position at θ=0 → rightmost x
 *   §3  all_positions returns 8 distinct positions
 *   §4  emitter_frequency matches cymatic_transduction
 *   §5  min_emitter_separation > 10 grid cells
 *   §6  out-of-range index throws
 *   §7  z=0 plane constraint
 *   §8  time wrapping
 *   §9  r and s coordinate mapping
 *   §10 default grid dimensions
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <nikola/multimodal/audio_emitter.hpp>

#include <cmath>
#include <set>

using namespace nikola::multimodal;

// ============================================================================
// §1 Position bounds
// ============================================================================

TEST_CASE("§1 All emitter coords within grid bounds", "[v031][audio_emitter]") {
    constexpr int NX = 64, NY = 64, NR = 16, NS = 16, NT = 128;
    for (int n = 0; n < NUM_EMITTERS; ++n) {
        auto ep = AudioEmitterLayout::compute_position(n, NX, NY, NR, NS, NT, 0);
        REQUIRE(ep.coord.c[0] < NX);  // x
        REQUIRE(ep.coord.c[1] < NY);  // y
        REQUIRE(ep.coord.c[2] == 0);  // z=0
        REQUIRE(ep.coord.c[3] < NT);  // t
        REQUIRE(ep.coord.c[4] < NR);  // r
        REQUIRE(ep.coord.c[5] < NS);  // s
    }
}

// ============================================================================
// §2 Emitter 0 at θ=0 → rightmost x
// ============================================================================

TEST_CASE("§2 Emitter 0 at rightmost x", "[v031][audio_emitter]") {
    auto ep = AudioEmitterLayout::compute_position(0);
    // θ=0 → x = cx + R*cos(0) = 32 + 32 = 64 → clamped to 63
    REQUIRE(ep.coord.c[0] == 63);
    // y = cy + R*sin(0) = 32 + 0 = 32
    REQUIRE(ep.coord.c[1] == 32);
}

// ============================================================================
// §3 All 8 positions are distinct
// ============================================================================

TEST_CASE("§3 All 8 emitter positions distinct", "[v031][audio_emitter]") {
    auto positions = AudioEmitterLayout::all_positions();
    // Check that x,y pairs are all distinct
    std::set<std::pair<uint16_t, uint16_t>> xy_pairs;
    for (int i = 0; i < NUM_EMITTERS; ++i) {
        xy_pairs.insert({positions[i].coord.c[0], positions[i].coord.c[1]});
    }
    REQUIRE(xy_pairs.size() == NUM_EMITTERS);
}

// ============================================================================
// §4 Frequency matches cymatic_transduction
// ============================================================================

TEST_CASE("§4 emitter_frequency matches cymatic π·φⁿ", "[v031][audio_emitter]") {
    for (int n = 1; n <= NUM_EMITTERS; ++n) {
        REQUIRE_THAT(AudioEmitterLayout::emitter_frequency(n),
                     Catch::Matchers::WithinRel(emitter_freq_hz(n), 1e-12));
    }
}

// ============================================================================
// §5 Min separation > 10
// ============================================================================

TEST_CASE("§5 Min emitter separation > 10 grid cells", "[v031][audio_emitter]") {
    double sep = AudioEmitterLayout::min_emitter_separation();
    REQUIRE(sep > 10.0);
}

// ============================================================================
// §6 Out-of-range throws
// ============================================================================

TEST_CASE("§6 Out-of-range emitter index throws", "[v031][audio_emitter]") {
    REQUIRE_THROWS_AS(AudioEmitterLayout::compute_position(-1), std::out_of_range);
    REQUIRE_THROWS_AS(AudioEmitterLayout::compute_position(8), std::out_of_range);
    REQUIRE_THROWS_AS(AudioEmitterLayout::compute_position(100), std::out_of_range);
}

// ============================================================================
// §7 z=0 plane constraint
// ============================================================================

TEST_CASE("§7 All emitters on z=0 plane", "[v031][audio_emitter]") {
    auto positions = AudioEmitterLayout::all_positions();
    for (int i = 0; i < NUM_EMITTERS; ++i) {
        REQUIRE(positions[i].coord.c[2] == 0);
    }
}

// ============================================================================
// §8 Time wrapping
// ============================================================================

TEST_CASE("§8 Time index wraps at grid_nt", "[v031][audio_emitter]") {
    constexpr int NT = 128;
    auto ep0 = AudioEmitterLayout::compute_position(0, 64, 64, 16, 16, NT, 0);
    auto ep_wrapped = AudioEmitterLayout::compute_position(0, 64, 64, 16, 16, NT, NT);
    REQUIRE(ep0.coord.c[3] == ep_wrapped.coord.c[3]);  // both t=0

    auto ep5 = AudioEmitterLayout::compute_position(0, 64, 64, 16, 16, NT, 5);
    auto ep133 = AudioEmitterLayout::compute_position(0, 64, 64, 16, 16, NT, NT + 5);
    REQUIRE(ep5.coord.c[3] == ep133.coord.c[3]);  // both t=5
}

// ============================================================================
// §9 r and s coordinate mapping
// ============================================================================

TEST_CASE("§9 r and s coordinate mapping", "[v031][audio_emitter]") {
    constexpr int NR = 16, NS = 16;
    auto ep = AudioEmitterLayout::compute_position(0, 64, 64, NR, NS, 128, 0);
    // r = round(0.8 * 16) = round(12.8) = 13
    REQUIRE(ep.coord.c[4] == 13);
    // s = NS - 1 = 15
    REQUIRE(ep.coord.c[5] == NS - 1);
}

// ============================================================================
// §10 Default grid dimensions
// ============================================================================

TEST_CASE("§10 Default parameters produce valid layout", "[v031][audio_emitter]") {
    auto positions = AudioEmitterLayout::all_positions();
    REQUIRE(positions.size() == NUM_EMITTERS);
    for (int i = 0; i < NUM_EMITTERS; ++i) {
        REQUIRE(positions[i].emitter_id == i);
        REQUIRE(positions[i].frequency_hz > 0.0);
    }
}
