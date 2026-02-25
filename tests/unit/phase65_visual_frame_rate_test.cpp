/**
 * @file   phase65_visual_frame_rate_test.cpp
 * @brief  Phase 65 — GAP-018: Visual Cymatics Frame Rate Adaptation
 *
 * Tests for nikola/multimodal/visual_frame_rate.hpp
 *
 * Coverage domains
 * ────────────────
 *  §1  Rate constants   (physics, 60/120 Hz display, frame periods, tick counts, Nyquist)
 *  §2  Buffer/cost constants (triple buffer, seqlock budget, energy budget, chroma max)
 *  §3  display_nyquist_hz — formula and error handling
 *  §4  ticks_per_frame   — formula, 60/120 Hz cases, error
 *  §5  frame_period_ms   — formula
 *  §6  accumulate_energy — energy = |H|², multi-tick cumulation, size mismatch
 *  §7  normalize_accumulation — sqrt(B/N) formula, edge cases, errors
 *  §8  tone_map / apply_tone_map / tone_map_inverse — values, range, roundtrip, errors
 *  §9  is_super_nyquist   — threshold predicate
 * §10  chromatic_shift_pixels — zero, linear, clamped, errors
 * §11  stroboscopic_trigger — near-zero, off-zero, 2π wrap-around
 * §12  Invariants — monotone tone map, energy positivity, Nyquist consistency
 * §13  Integration — full accumulate → normalize → tone_map pipeline
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>
#include <vector>

#include "nikola/multimodal/visual_frame_rate.hpp"

using namespace nikola::multimodal;

// ═══════════════════════════════════════════════════════════════════════════
// §1  Rate constants
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("constant: VISUAL_PHYSICS_RATE_HZ is 1000",
          "[gap018][constants]")
{
    REQUIRE(VISUAL_PHYSICS_RATE_HZ == 1000);
}

TEST_CASE("constant: display rates are 60 and 120 Hz",
          "[gap018][constants]")
{
    REQUIRE(DISPLAY_RATE_60HZ  == Catch::Approx(60.0).epsilon(1e-15));
    REQUIRE(DISPLAY_RATE_120HZ == Catch::Approx(120.0).epsilon(1e-15));
    REQUIRE(DISPLAY_RATE_120HZ == 2.0 * DISPLAY_RATE_60HZ);
}

TEST_CASE("constant: frame periods are reciprocals of display rates",
          "[gap018][constants]")
{
    // 1000 / 60 ≈ 16.667 ms
    REQUIRE(DISPLAY_FRAME_PERIOD_60HZ_MS  == Catch::Approx(1000.0 / 60.0).epsilon(1e-10));
    // 1000 / 120 ≈ 8.333 ms
    REQUIRE(DISPLAY_FRAME_PERIOD_120HZ_MS == Catch::Approx(1000.0 / 120.0).epsilon(1e-10));
    REQUIRE(DISPLAY_FRAME_PERIOD_60HZ_MS  > 16.0);
    REQUIRE(DISPLAY_FRAME_PERIOD_60HZ_MS  < 17.0);
    REQUIRE(DISPLAY_FRAME_PERIOD_120HZ_MS >  8.0);
    REQUIRE(DISPLAY_FRAME_PERIOD_120HZ_MS <  9.0);
    // 120Hz period is exactly half the 60Hz period
    REQUIRE(DISPLAY_FRAME_PERIOD_120HZ_MS == Catch::Approx(DISPLAY_FRAME_PERIOD_60HZ_MS / 2.0).epsilon(1e-10));
}

TEST_CASE("constant: TICKS_PER_FRAME values are floor-correct",
          "[gap018][constants]")
{
    REQUIRE(TICKS_PER_FRAME_60HZ  == 16);   // floor(1000/60) = floor(16.667)
    REQUIRE(TICKS_PER_FRAME_120HZ ==  8);   // floor(1000/120) = floor(8.333)
    REQUIRE(TICKS_PER_FRAME_60HZ  == 2 * TICKS_PER_FRAME_120HZ);
}

TEST_CASE("constant: display Nyquist limits are half display rates",
          "[gap018][constants]")
{
    REQUIRE(DISPLAY_NYQUIST_60HZ  == Catch::Approx(30.0).epsilon(1e-15));
    REQUIRE(DISPLAY_NYQUIST_120HZ == Catch::Approx(60.0).epsilon(1e-15));
    REQUIRE(DISPLAY_NYQUIST_60HZ  == DISPLAY_RATE_60HZ  / 2.0);
    REQUIRE(DISPLAY_NYQUIST_120HZ == DISPLAY_RATE_120HZ / 2.0);
    REQUIRE(DISPLAY_NYQUIST_120HZ == 2.0 * DISPLAY_NYQUIST_60HZ);
}

// ═══════════════════════════════════════════════════════════════════════════
// §2  Buffer / cost constants
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("constant: TRIPLE_BUFFER_COUNT is 3",
          "[gap018][constants]")
{
    REQUIRE(TRIPLE_BUFFER_COUNT == 3);
}

TEST_CASE("constant: seqlock swap budget is 10 μs",
          "[gap018][constants]")
{
    REQUIRE(SEQLOCK_SWAP_BUDGET_US == 10);
}

TEST_CASE("constant: energy accumulation budget is 0.1 ms per tick",
          "[gap018][constants]")
{
    REQUIRE(ENERGY_ACCUM_BUDGET_MS == Catch::Approx(0.1).epsilon(1e-15));
    // More stringent: must be well under 1ms tick budget
    REQUIRE(ENERGY_ACCUM_BUDGET_MS < 1.0);
}

TEST_CASE("constant: CHROMATIC_SHIFT_MAX_PIXELS is 50",
          "[gap018][constants]")
{
    REQUIRE(CHROMATIC_SHIFT_MAX_PIXELS == 50);
}

TEST_CASE("constant: stroboscopic trigger is emitter 1 (E1 fundamental)",
          "[gap018][constants]")
{
    REQUIRE(STROBOSCOPIC_TRIGGER_EMITTER == 1);
}

// ═══════════════════════════════════════════════════════════════════════════
// §3  display_nyquist_hz
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("display_nyquist_hz: equals refresh_rate / 2",
          "[gap018][derived]")
{
    REQUIRE(display_nyquist_hz(60.0)  == Catch::Approx(30.0).epsilon(1e-12));
    REQUIRE(display_nyquist_hz(120.0) == Catch::Approx(60.0).epsilon(1e-12));
    REQUIRE(display_nyquist_hz(144.0) == Catch::Approx(72.0).epsilon(1e-12));
    REQUIRE(display_nyquist_hz(240.0) == Catch::Approx(120.0).epsilon(1e-12));
}

TEST_CASE("display_nyquist_hz: throws on non-positive rate",
          "[gap018][derived][error]")
{
    REQUIRE_THROWS_AS(display_nyquist_hz(  0.0), std::invalid_argument);
    REQUIRE_THROWS_AS(display_nyquist_hz( -1.0), std::invalid_argument);
    REQUIRE_THROWS_AS(display_nyquist_hz(-60.0), std::invalid_argument);
}

// ═══════════════════════════════════════════════════════════════════════════
// §4  ticks_per_frame
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("ticks_per_frame: 60 Hz gives ~16.667 ticks per frame",
          "[gap018][derived]")
{
    double tpf = ticks_per_frame(60.0);
    REQUIRE(tpf == Catch::Approx(1000.0 / 60.0).epsilon(1e-10));
    REQUIRE(tpf > 16.0);
    REQUIRE(tpf < 17.0);
    // Floor matches constant
    REQUIRE(static_cast<int>(tpf) == TICKS_PER_FRAME_60HZ);
}

TEST_CASE("ticks_per_frame: 120 Hz gives ~8.333 ticks per frame",
          "[gap018][derived]")
{
    double tpf = ticks_per_frame(120.0);
    REQUIRE(tpf == Catch::Approx(1000.0 / 120.0).epsilon(1e-10));
    // Exactly half the 60 Hz value
    REQUIRE(tpf == Catch::Approx(ticks_per_frame(60.0) / 2.0).epsilon(1e-10));
    REQUIRE(static_cast<int>(tpf) == TICKS_PER_FRAME_120HZ);
}

TEST_CASE("ticks_per_frame: higher refresh rate → fewer ticks per frame",
          "[gap018][derived]")
{
    REQUIRE(ticks_per_frame(60.0) > ticks_per_frame(120.0));
    REQUIRE(ticks_per_frame(120.0) > ticks_per_frame(240.0));
}

TEST_CASE("ticks_per_frame: throws on non-positive rate",
          "[gap018][derived][error]")
{
    REQUIRE_THROWS_AS(ticks_per_frame( 0.0), std::invalid_argument);
    REQUIRE_THROWS_AS(ticks_per_frame(-1.0), std::invalid_argument);
}

// ═══════════════════════════════════════════════════════════════════════════
// §5  frame_period_ms
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("frame_period_ms: matches spec values for 60 and 120 Hz",
          "[gap018][derived]")
{
    REQUIRE(frame_period_ms(60.0)  == Catch::Approx(DISPLAY_FRAME_PERIOD_60HZ_MS).epsilon(1e-10));
    REQUIRE(frame_period_ms(120.0) == Catch::Approx(DISPLAY_FRAME_PERIOD_120HZ_MS).epsilon(1e-10));
    // 60 fps period is double 120 fps period
    REQUIRE(frame_period_ms(60.0) == Catch::Approx(2.0 * frame_period_ms(120.0)).epsilon(1e-10));
}

// ═══════════════════════════════════════════════════════════════════════════
// §6  accumulate_energy
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("accumulate_energy: adds |H|² to accumulation buffer",
          "[gap018][energy]")
{
    std::vector<float> acc(4, 0.0f);
    std::vector<float> h = {1.0f, -2.0f, 3.0f, 0.0f};

    accumulate_energy(acc, h);

    REQUIRE(acc[0] == Catch::Approx(1.0f).epsilon(1e-6f));   // 1²
    REQUIRE(acc[1] == Catch::Approx(4.0f).epsilon(1e-6f));   // (-2)²
    REQUIRE(acc[2] == Catch::Approx(9.0f).epsilon(1e-6f));   // 3²
    REQUIRE(acc[3] == Catch::Approx(0.0f).margin(1e-9f));    // 0²
}

TEST_CASE("accumulate_energy: negative amplitudes accumulate same as positive (|H|²)",
          "[gap018][energy]")
{
    std::vector<float> acc_pos(3, 0.0f);
    std::vector<float> acc_neg(3, 0.0f);
    std::vector<float> h_pos = { 1.0f,  2.0f,  3.0f};
    std::vector<float> h_neg = {-1.0f, -2.0f, -3.0f};

    accumulate_energy(acc_pos, h_pos);
    accumulate_energy(acc_neg, h_neg);

    for (int i = 0; i < 3; ++i)
        REQUIRE(acc_pos[i] == Catch::Approx(acc_neg[i]).epsilon(1e-6f));
}

TEST_CASE("accumulate_energy: multiple calls cumulate correctly",
          "[gap018][energy]")
{
    // 16 frames each with H[0]=1.0 → acc[0] should equal 16
    std::vector<float> acc(2, 0.0f);
    std::vector<float> h = {1.0f, 0.5f};

    for (int tick = 0; tick < 16; ++tick)
        accumulate_energy(acc, h);

    REQUIRE(acc[0] == Catch::Approx(16.0f).epsilon(1e-5f));   // 16 × 1² = 16
    REQUIRE(acc[1] == Catch::Approx( 4.0f).epsilon(1e-5f));   // 16 × 0.5² = 4
}

TEST_CASE("accumulate_energy: throws on size mismatch",
          "[gap018][energy][error]")
{
    std::vector<float> acc(4, 0.0f);
    std::vector<float> h(5, 1.0f);
    REQUIRE_THROWS_AS(accumulate_energy(acc, h), std::invalid_argument);
}

// ═══════════════════════════════════════════════════════════════════════════
// §7  normalize_accumulation
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("normalize_accumulation: sqrt(B_acc / N_ticks) formula",
          "[gap018][normalize]")
{
    // B_acc = 16 (16 ticks of H=1.0), N=16 → sqrt(16/16) = 1.0
    std::vector<float> acc = {16.0f, 4.0f, 0.0f};
    std::vector<float> out(3, -1.0f);

    normalize_accumulation(acc, 16, out);

    REQUIRE(out[0] == Catch::Approx(1.0f).epsilon(1e-5f));   // sqrt(16/16) = 1.0
    REQUIRE(out[1] == Catch::Approx(0.5f).epsilon(1e-5f));   // sqrt(4/16)  = 0.5
    REQUIRE(out[2] == Catch::Approx(0.0f).margin(1e-9f));    // sqrt(0/16)  = 0
}

TEST_CASE("normalize_accumulation: different N_ticks changes brightness",
          "[gap018][normalize]")
{
    // Same accumulation (B=9), but N=9 vs N=1
    std::vector<float> acc = {9.0f};
    std::vector<float> out_n9(1), out_n1(1);

    normalize_accumulation(acc, 9, out_n9);   // sqrt(9/9)  = 1.0
    normalize_accumulation(acc, 1, out_n1);   // sqrt(9/1)  = 3.0

    REQUIRE(out_n9[0] == Catch::Approx(1.0f).epsilon(1e-5f));
    REQUIRE(out_n1[0] == Catch::Approx(3.0f).epsilon(1e-5f));
    REQUIRE(out_n9[0] < out_n1[0]);   // more ticks → dimmer per-tick
}

TEST_CASE("normalize_accumulation: always non-negative output",
          "[gap018][normalize]")
{
    std::vector<float> acc = {0.0f, 1.0f, 100.0f, 0.01f};
    std::vector<float> out(4);
    normalize_accumulation(acc, 16, out);
    for (float v : out)
        REQUIRE(v >= 0.0f);
}

TEST_CASE("normalize_accumulation: throws on n_ticks == 0",
          "[gap018][normalize][error]")
{
    std::vector<float> acc = {1.0f};
    std::vector<float> out(1);
    REQUIRE_THROWS_AS(normalize_accumulation(acc,  0, out), std::invalid_argument);
    REQUIRE_THROWS_AS(normalize_accumulation(acc, -1, out), std::invalid_argument);
}

TEST_CASE("normalize_accumulation: throws on size mismatch",
          "[gap018][normalize][error]")
{
    std::vector<float> acc(4, 1.0f);
    std::vector<float> out(5, 0.0f);
    REQUIRE_THROWS_AS(normalize_accumulation(acc, 1, out), std::invalid_argument);
}

// ═══════════════════════════════════════════════════════════════════════════
// §8  tone_map / apply_tone_map / tone_map_inverse
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("tone_map: spec reference values x/(1+x)",
          "[gap018][tonemap]")
{
    REQUIRE(tone_map(0.0f)   == Catch::Approx(0.0f).margin(1e-7f));    // 0/(1+0) = 0
    REQUIRE(tone_map(1.0f)   == Catch::Approx(0.5f).epsilon(1e-6f));   // 1/(1+1) = 0.5
    REQUIRE(tone_map(3.0f)   == Catch::Approx(0.75f).epsilon(1e-6f));  // 3/(1+3) = 0.75
    REQUIRE(tone_map(9.0f)   == Catch::Approx(0.9f).epsilon(1e-5f));   // 9/10 = 0.9
    REQUIRE(tone_map(99.0f)  == Catch::Approx(0.99f).epsilon(1e-4f));  // 99/100
    REQUIRE(tone_map(999.0f) == Catch::Approx(0.999f).epsilon(1e-3f)); // 999/1000
}

TEST_CASE("tone_map: output is always in [0, 1)",
          "[gap018][tonemap]")
{
    for (float x : {0.0f, 0.001f, 0.1f, 0.5f, 1.0f, 2.0f, 10.0f, 1000.0f}) {
        float y = tone_map(x);
        REQUIRE(y >= 0.0f);
        REQUIRE(y < 1.0f);
    }
}

TEST_CASE("tone_map: strictly monotonically increasing",
          "[gap018][tonemap]")
{
    float prev = tone_map(0.0f);
    for (float x : {0.01f, 0.1f, 0.5f, 1.0f, 2.0f, 5.0f, 10.0f, 100.0f}) {
        float cur = tone_map(x);
        REQUIRE(cur > prev);
        prev = cur;
    }
}

TEST_CASE("apply_tone_map: applies tone_map element-wise in place",
          "[gap018][tonemap]")
{
    std::vector<float> buf = {0.0f, 1.0f, 3.0f};
    apply_tone_map(buf);
    REQUIRE(buf[0] == Catch::Approx(0.0f).margin(1e-7f));
    REQUIRE(buf[1] == Catch::Approx(0.5f).epsilon(1e-6f));
    REQUIRE(buf[2] == Catch::Approx(0.75f).epsilon(1e-6f));
}

TEST_CASE("tone_map_inverse: roundtrip tone_map_inverse(tone_map(x)) ≈ x",
          "[gap018][tonemap]")
{
    for (float x : {0.001f, 0.1f, 0.5f, 1.0f, 2.0f, 5.0f, 10.0f}) {
        float y  = tone_map(x);
        float x2 = tone_map_inverse(y);
        REQUIRE(x2 == Catch::Approx(x).epsilon(1e-5f));
    }
}

TEST_CASE("tone_map_inverse: throws on y >= 1 or y < 0",
          "[gap018][tonemap][error]")
{
    REQUIRE_THROWS_AS(tone_map_inverse( 1.0f),  std::invalid_argument);
    REQUIRE_THROWS_AS(tone_map_inverse( 1.5f),  std::invalid_argument);
    REQUIRE_THROWS_AS(tone_map_inverse(-0.01f), std::invalid_argument);
}

// ═══════════════════════════════════════════════════════════════════════════
// §9  is_super_nyquist
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("is_super_nyquist: below or at Nyquist is false",
          "[gap018][alias]")
{
    REQUIRE_FALSE(is_super_nyquist( 0.0,  60.0));
    REQUIRE_FALSE(is_super_nyquist(29.9,  60.0));
    REQUIRE_FALSE(is_super_nyquist(30.0,  60.0));   // exactly Nyquist — not strictly above
    REQUIRE_FALSE(is_super_nyquist(59.9, 120.0));
}

TEST_CASE("is_super_nyquist: strictly above Nyquist is true",
          "[gap018][alias]")
{
    REQUIRE(is_super_nyquist(30.001,  60.0));
    REQUIRE(is_super_nyquist(50.0,    60.0));
    REQUIRE(is_super_nyquist(500.0,   60.0));
    REQUIRE(is_super_nyquist(60.001, 120.0));
}

TEST_CASE("is_super_nyquist: physics emitters all below 60 Hz display Nyquist",
          "[gap018][alias]")
{
    // All 8 emitters (E1 ≈ 5.08 Hz … E8 ≈ 147.6 Hz)
    // E7 and E8 exceed 60 Hz Nyquist (30 Hz) — they will alias
    REQUIRE_FALSE(is_super_nyquist( 5.083, 60.0));   // E1 — safe
    REQUIRE_FALSE(is_super_nyquist(13.308, 60.0));   // E3 — safe
    REQUIRE(      is_super_nyquist(56.374, 60.0));   // E6 ≈ 56 > 30 — aliases
    REQUIRE(      is_super_nyquist(147.588, 60.0));  // E8 — aliases
    // But with 120 Hz screen, E1–E5 are safe
    REQUIRE_FALSE(is_super_nyquist(34.841, 120.0));  // E5 ≈ 35 < 60 — safe
    REQUIRE(      is_super_nyquist(91.214, 120.0));  // E7 ≈ 91 > 60 — aliases
}

// ═══════════════════════════════════════════════════════════════════════════
// §10  chromatic_shift_pixels
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("chromatic_shift_pixels: sub-Nyquist returns 0",
          "[gap018][chromatic]")
{
    REQUIRE(chromatic_shift_pixels( 0.0, 60.0) == Catch::Approx(0.0).margin(1e-9));
    REQUIRE(chromatic_shift_pixels(29.9, 60.0) == Catch::Approx(0.0).margin(1e-9));
    REQUIRE(chromatic_shift_pixels(30.0, 60.0) == Catch::Approx(0.0).margin(1e-9));
}

TEST_CASE("chromatic_shift_pixels: at 2× Nyquist (one octave above) gives max shift",
          "[gap018][chromatic]")
{
    // excess = 2×Nyquist − Nyquist = Nyquist; shift = MAX × (Nyquist/Nyquist) = MAX
    double shift = chromatic_shift_pixels(60.0, 60.0);   // 2× Nyquist=30
    REQUIRE(shift == Catch::Approx(static_cast<double>(CHROMATIC_SHIFT_MAX_PIXELS)).epsilon(1e-9));
}

TEST_CASE("chromatic_shift_pixels: linear in the middle of the range",
          "[gap018][chromatic]")
{
    // At 1.5× Nyquist: excess = 0.5×Nyquist; shift = MAX × 0.5 = 25
    double shift = chromatic_shift_pixels(45.0, 60.0);   // 45 = 1.5×30
    REQUIRE(shift == Catch::Approx(25.0).epsilon(1e-9));
}

TEST_CASE("chromatic_shift_pixels: caps at CHROMATIC_SHIFT_MAX_PIXELS for very high frequencies",
          "[gap018][chromatic]")
{
    double shift_high = chromatic_shift_pixels(10000.0, 60.0);
    REQUIRE(shift_high == Catch::Approx(static_cast<double>(CHROMATIC_SHIFT_MAX_PIXELS)).epsilon(1e-9));
}

TEST_CASE("chromatic_shift_pixels: monotonically increasing with frequency (above Nyquist)",
          "[gap018][chromatic]")
{
    for (double f : {31.0, 35.0, 40.0, 50.0, 59.0}) {
        REQUIRE(chromatic_shift_pixels(f, 60.0) < chromatic_shift_pixels(f + 5.0, 60.0));
    }
}

TEST_CASE("chromatic_shift_pixels: throws on non-positive display rate",
          "[gap018][chromatic][error]")
{
    REQUIRE_THROWS_AS(chromatic_shift_pixels(100.0,  0.0), std::invalid_argument);
    REQUIRE_THROWS_AS(chromatic_shift_pixels(100.0, -1.0), std::invalid_argument);
}

// ═══════════════════════════════════════════════════════════════════════════
// §11  stroboscopic_trigger
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("stroboscopic_trigger: near phase=0 triggers",
          "[gap018][strobe]")
{
    REQUIRE( stroboscopic_trigger(0.0));
    REQUIRE( stroboscopic_trigger(0.05));       // within default tolerance 0.1
    REQUIRE( stroboscopic_trigger(-0.05));
    REQUIRE( stroboscopic_trigger(0.1));        // exactly at tolerance
    REQUIRE( stroboscopic_trigger(-0.1));
}

TEST_CASE("stroboscopic_trigger: outside tolerance does not trigger",
          "[gap018][strobe]")
{
    REQUIRE_FALSE(stroboscopic_trigger(0.11));
    REQUIRE_FALSE(stroboscopic_trigger(-0.11));
    REQUIRE_FALSE(stroboscopic_trigger(M_PI));    // opposite side of circle
    REQUIRE_FALSE(stroboscopic_trigger(M_PI / 2.0));
}

TEST_CASE("stroboscopic_trigger: 2π and −2π wrap back to trigger",
          "[gap018][strobe]")
{
    // 2π radians = full cycle, equivalent to 0
    REQUIRE(stroboscopic_trigger(2.0 * M_PI));
    REQUIRE(stroboscopic_trigger(-2.0 * M_PI));
    REQUIRE(stroboscopic_trigger(4.0 * M_PI));   // 2 full cycles
}

TEST_CASE("stroboscopic_trigger: custom tolerance works",
          "[gap018][strobe]")
{
    // Narrow tolerance
    REQUIRE( stroboscopic_trigger(0.01, 0.05));
    REQUIRE_FALSE(stroboscopic_trigger(0.06, 0.05));

    // Wide tolerance
    REQUIRE( stroboscopic_trigger(M_PI / 4.0, M_PI / 3.0));  // ~0.785 within π/3≈1.047
}

// ═══════════════════════════════════════════════════════════════════════════
// §12  Invariants
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("invariant: tone_map output is always < 1 for any positive input",
          "[gap018][invariant]")
{
    for (float x : {1e-6f, 0.1f, 1.0f, 10.0f, 1000.0f, 1e6f}) {
        REQUIRE(tone_map(x) < 1.0f);
        REQUIRE(tone_map(x) >= 0.0f);
    }
}

TEST_CASE("invariant: energy accumulation never decreases buffer",
          "[gap018][invariant]")
{
    std::vector<float> acc(5, 0.0f);
    std::vector<float> h = {1.0f, -0.5f, 2.0f, -3.0f, 0.1f};

    for (int i = 0; i < 100; ++i) {
        std::vector<float> prev = acc;
        accumulate_energy(acc, h);
        for (std::size_t k = 0; k < acc.size(); ++k)
            REQUIRE(acc[k] >= prev[k]);
    }
}

TEST_CASE("invariant: ticks_per_frame equals frame_period_ms (both = 1000/rate)",
          "[gap018][invariant]")
{
    // The physics clock runs at 1000 Hz (1 tick per ms), so the number of
    // physics ticks that fit in one display frame equals the frame period in ms.
    // i.e.  ticks_per_frame(r) = 1000 / r  ==  frame_period_ms(r)
    for (double rate : {60.0, 120.0, 144.0, 240.0}) {
        REQUIRE(ticks_per_frame(rate) == Catch::Approx(frame_period_ms(rate)).epsilon(1e-9));
    }
}

TEST_CASE("invariant: display_nyquist_hz is consistent with constants at 60 and 120 Hz",
          "[gap018][invariant]")
{
    REQUIRE(display_nyquist_hz(DISPLAY_RATE_60HZ)  == Catch::Approx(DISPLAY_NYQUIST_60HZ).epsilon(1e-12));
    REQUIRE(display_nyquist_hz(DISPLAY_RATE_120HZ) == Catch::Approx(DISPLAY_NYQUIST_120HZ).epsilon(1e-12));
}

// ═══════════════════════════════════════════════════════════════════════════
// §13  Integration: full accumulate → normalize → tone_map pipeline
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("integration: 16 unit-amplitude ticks normalize to display 0.5 after tone_map",
          "[gap018][integration]")
{
    // Spec example: 16 ticks of H=1.0 → B_acc=16 → normalize: sqrt(16/16)=1.0
    //              → tone_map(1.0) = 0.5
    const int N   = 16;
    const int SZ  = 4;
    std::vector<float> acc(SZ, 0.0f);
    std::vector<float> h(SZ, 1.0f);   // Unit amplitude hologram

    for (int tick = 0; tick < N; ++tick)
        accumulate_energy(acc, h);

    std::vector<float> out(SZ);
    normalize_accumulation(acc, N, out);

    // Normalized should be 1.0
    for (float v : out)
        REQUIRE(v == Catch::Approx(1.0f).epsilon(1e-5f));

    apply_tone_map(out);

    // Tone-mapped should be 0.5
    for (float v : out)
        REQUIRE(v == Catch::Approx(0.5f).epsilon(1e-5f));
}

TEST_CASE("integration: double-amplitude hologram gives brighter display pixel",
          "[gap018][integration]")
{
    const int N = 16;
    std::vector<float> acc1(1, 0.0f), acc2(1, 0.0f);
    std::vector<float> h1 = {1.0f}, h2 = {2.0f};

    for (int tick = 0; tick < N; ++tick) {
        accumulate_energy(acc1, h1);
        accumulate_energy(acc2, h2);
    }

    std::vector<float> out1(1), out2(1);
    normalize_accumulation(acc1, N, out1);
    normalize_accumulation(acc2, N, out2);
    apply_tone_map(out1);
    apply_tone_map(out2);

    // Double amplitude → 4× energy → normalize gives 2.0 → tone_map(2.0) = 2/3 ≈ 0.667
    REQUIRE(out2[0] > out1[0]);
    REQUIRE(out2[0] == Catch::Approx(2.0f / 3.0f).epsilon(1e-4f));
}

TEST_CASE("integration: silence (H=0 every tick) produces zero display output",
          "[gap018][integration]")
{
    std::vector<float> acc(5, 0.0f);
    std::vector<float> h_zero(5, 0.0f);

    for (int tick = 0; tick < 16; ++tick)
        accumulate_energy(acc, h_zero);

    std::vector<float> out(5);
    normalize_accumulation(acc, 16, out);
    apply_tone_map(out);

    for (float v : out)
        REQUIRE(v == Catch::Approx(0.0f).margin(1e-9f));
}
