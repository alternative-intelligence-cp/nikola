/**
 * @file   phase63_nonary_overflow_test.cpp
 * @brief  Phase 63 — GAP-044: Nonary Overflow Probability Distribution
 *
 * Tests for nikola/foundation/nonary_overflow.hpp
 *
 * Coverage domains
 * ────────────────
 *  §1  Constants  (base, bits-per-trit, clip boundary, pair counts, probabilities)
 *  §2  Arithmetic range limits
 *  §3  add_saturated        — hard clipping semantics
 *  §4  add_with_carry       — Spectral Cascading; energy conservation identity
 *  §5  multiply_saturated   — low-pass hard clip
 *  §6  carry_decompose      — §GAP-044 spec example (A=13) + edge cases
 *  §7  information_loss_gaussian — tails of Gaussian beyond ±4.5
 *  §8  overflow_prob_gaussian    — P(|a+b| > 4) under Gaussian inputs
 *  §9  is_saturation_alert  — 1 % threshold predicate
 * §10  SaturationMonitor    — accumulator lifecycle
 * §11  Invariants           — overflow identity, monotonicity, error-handling
 * §12  Integration          — monitor + arithmetic workflow
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>
#include "nikola/foundation/nonary_overflow.hpp"

using namespace nikola::foundation;

// ═══════════════════════════════════════════════════════════════════════════
// §1  Constants
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("constant: NONARY_BASE and NONARY_DIGIT_COUNT are 9",
          "[gap044][constants]")
{
    REQUIRE(NONARY_BASE        == 9);
    REQUIRE(NONARY_DIGIT_COUNT == 9);
    REQUIRE(NONARY_BASE == NONARY_DIGIT_COUNT);
}

TEST_CASE("constant: NONARY_INFO_BITS_PER_TRIT ≈ log₂(9) ≈ 3.170",
          "[gap044][constants]")
{
    // log₂(9) = ln(9)/ln(2)
    double expected = std::log(9.0) / std::log(2.0);
    REQUIRE(NONARY_INFO_BITS_PER_TRIT == Catch::Approx(expected).epsilon(1e-12));
    // Verify it exceeds log₂(8)=3 and is below log₂(16)=4
    REQUIRE(NONARY_INFO_BITS_PER_TRIT > 3.0);
    REQUIRE(NONARY_INFO_BITS_PER_TRIT < 4.0);
    // Matches spec quote: ≈ 3.17 bits
    REQUIRE(NONARY_INFO_BITS_PER_TRIT == Catch::Approx(3.170).epsilon(0.001));
}

TEST_CASE("constant: NONARY_CLIP_BOUNDARY is 4.5",
          "[gap044][constants]")
{
    REQUIRE(NONARY_CLIP_BOUNDARY == Catch::Approx(4.5).epsilon(1e-15));
    // Must lie strictly between NIT_MAX (4) and the next integer (5)
    REQUIRE(NONARY_CLIP_BOUNDARY > static_cast<double>(NIT_MAX));
    REQUIRE(NONARY_CLIP_BOUNDARY < 5.0);
}

TEST_CASE("constant: overflow pair counts and analytical probability",
          "[gap044][constants]")
{
    REQUIRE(NONARY_TOTAL_PAIR_COUNT == 81);   // 9×9
    REQUIRE(NONARY_OVERFLOW_PAIR_COUNT == 20);

    double expected_prob = 20.0 / 81.0;
    REQUIRE(NONARY_OVERFLOW_PROB_ADD_UNIFORM == Catch::Approx(expected_prob).epsilon(1e-12));
    // Quoted spec range: "approx 22%" (analytical 24.7% is consistent approximation)
    REQUIRE(NONARY_OVERFLOW_PROB_ADD_UNIFORM > 0.20);
    REQUIRE(NONARY_OVERFLOW_PROB_ADD_UNIFORM < 0.30);
}

TEST_CASE("constant: Gaussian operational overflow probability < 5%",
          "[gap044][constants]")
{
    REQUIRE(NONARY_OVERFLOW_PROB_ADD_GAUSSIAN == Catch::Approx(0.05).epsilon(1e-12));
    REQUIRE(NONARY_OVERFLOW_PROB_ADD_GAUSSIAN < NONARY_OVERFLOW_PROB_ADD_UNIFORM);
}

TEST_CASE("constant: saturation alert threshold is 1%",
          "[gap044][constants]")
{
    REQUIRE(SATURATION_RATE_ALERT_THRESHOLD == Catch::Approx(0.01).epsilon(1e-15));
}

TEST_CASE("constant: dither amplitude max is 0.5 (Voronoi cell radius)",
          "[gap044][constants]")
{
    REQUIRE(NONARY_DITHER_AMPLITUDE_MAX == Catch::Approx(0.5).epsilon(1e-15));
    // Must be strictly less than 1 (would cross into adjacent Voronoi cell)
    REQUIRE(NONARY_DITHER_AMPLITUDE_MAX < 1.0);
}

// ═══════════════════════════════════════════════════════════════════════════
// §2  Arithmetic range limits
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("constant: arithmetic range limits are consistent",
          "[gap044][constants]")
{
    // Addition bounds
    REQUIRE(NONARY_ADD_RESULT_MAX == NIT_MAX + NIT_MAX);   // +4 + +4 = +8
    REQUIRE(NONARY_ADD_RESULT_MIN == NIT_MIN + NIT_MIN);   // −4 + −4 = −8

    // Multiplication bounds: (−4)×(−4) = +16; (+4)×(−4) = −16
    REQUIRE(NONARY_MUL_RESULT_MAX == NIT_MAX * NIT_MAX);   // 4*4 = 16
    REQUIRE(NONARY_MUL_RESULT_MIN == NIT_MIN * NIT_MAX);   // -4*4 = -16

    // Multiplication range is wider than addition range
    REQUIRE(NONARY_MUL_RESULT_MAX > NONARY_ADD_RESULT_MAX);
}

// ═══════════════════════════════════════════════════════════════════════════
// §3  add_saturated
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("add_saturated: values within range pass through unchanged",
          "[gap044][add_saturated]")
{
    REQUIRE(add_saturated( 2,  2) ==  4);
    REQUIRE(add_saturated( 1, -1) ==  0);
    REQUIRE(add_saturated(-3,  3) ==  0);
    REQUIRE(add_saturated(-2, -1) == -3);
    REQUIRE(add_saturated( 0,  0) ==  0);
}

TEST_CASE("add_saturated: positive overflow clamps to NIT_MAX",
          "[gap044][add_saturated]")
{
    REQUIRE(add_saturated( 4,  4) == NIT_MAX);   // +8 → +4
    REQUIRE(add_saturated( 4,  1) == NIT_MAX);   // +5 → +4
    REQUIRE(add_saturated( 3,  3) == NIT_MAX);   // +6 → +4
    REQUIRE(add_saturated( 4,  3) == NIT_MAX);   // +7 → +4
}

TEST_CASE("add_saturated: negative overflow clamps to NIT_MIN",
          "[gap044][add_saturated]")
{
    REQUIRE(add_saturated(-4, -4) == NIT_MIN);   // −8 → −4
    REQUIRE(add_saturated(-4, -1) == NIT_MIN);   // −5 → −4
    REQUIRE(add_saturated(-3, -3) == NIT_MIN);   // −6 → −4
}

TEST_CASE("add_saturated: boundary cases at exactly ±4",
          "[gap044][add_saturated]")
{
    REQUIRE(add_saturated( 4,  0) ==  4);   // exactly NIT_MAX, no clip
    REQUIRE(add_saturated(-4,  0) == -4);   // exactly NIT_MIN, no clip
    REQUIRE(add_saturated( 3,  1) ==  4);   // exactly NIT_MAX
    REQUIRE(add_saturated(-3, -1) == -4);   // exactly NIT_MIN
}

// ═══════════════════════════════════════════════════════════════════════════
// §4  add_with_carry  (Spectral Cascading)
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("add_with_carry: no overflow produces carry=0",
          "[gap044][carry]")
{
    int carry = -99;

    REQUIRE(add_with_carry(2, 2, carry) ==  4);   REQUIRE(carry == 0);
    REQUIRE(add_with_carry(1, -1, carry) == 0);   REQUIRE(carry == 0);
    REQUIRE(add_with_carry(4, 0, carry) ==  4);   REQUIRE(carry == 0);
    REQUIRE(add_with_carry(-4, 0, carry) == -4);  REQUIRE(carry == 0);
}

TEST_CASE("add_with_carry: positive overflow emits carry=+1",
          "[gap044][carry]")
{
    int carry = 0;
    int8_t rem;

    // a=4, b=4: raw=8 → carry=1, rem=−1  (1×9 + (−1) = 8)
    rem = add_with_carry(4, 4, carry);
    REQUIRE(carry == 1);
    REQUIRE(rem   == -1);
    REQUIRE(carry * NONARY_CARRY_DIVISOR + rem == 8);

    // a=4, b=1: raw=5 → carry=1, rem=−4  (1×9 + (−4) = 5)
    rem = add_with_carry(4, 1, carry);
    REQUIRE(carry == 1);
    REQUIRE(rem   == -4);
    REQUIRE(carry * NONARY_CARRY_DIVISOR + rem == 5);

    // a=4, b=3: raw=7 → carry=1, rem=−2  (1×9 + (−2) = 7)
    rem = add_with_carry(4, 3, carry);
    REQUIRE(carry == 1);
    REQUIRE(rem   == -2);
    REQUIRE(carry * NONARY_CARRY_DIVISOR + rem == 7);
}

TEST_CASE("add_with_carry: negative overflow emits carry=−1",
          "[gap044][carry]")
{
    int carry = 0;
    int8_t rem;

    // a=−4, b=−4: raw=−8 → carry=−1, rem=+1  (−1×9 + 1 = −8)
    rem = add_with_carry(-4, -4, carry);
    REQUIRE(carry == -1);
    REQUIRE(rem   ==  1);
    REQUIRE(carry * NONARY_CARRY_DIVISOR + rem == -8);

    // a=−4, b=−1: raw=−5 → carry=−1, rem=+4  (−1×9 + 4 = −5)
    rem = add_with_carry(-4, -1, carry);
    REQUIRE(carry == -1);
    REQUIRE(rem   ==  4);
    REQUIRE(carry * NONARY_CARRY_DIVISOR + rem == -5);
}

TEST_CASE("add_with_carry: energy conservation identity holds for all 81 pairs",
          "[gap044][carry][invariant]")
{
    for (int a = NIT_MIN; a <= NIT_MAX; ++a) {
        for (int b = NIT_MIN; b <= NIT_MAX; ++b) {
            int carry = 0;
            int rem = static_cast<int>(
                add_with_carry(static_cast<Nit>(a), static_cast<Nit>(b), carry));
            // Energy conservation: carry×9 + remainder == a + b
            REQUIRE(carry * NONARY_CARRY_DIVISOR + rem == a + b);
            // Remainder always in valid Nit range
            REQUIRE(rem >= NIT_MIN);
            REQUIRE(rem <= NIT_MAX);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §5  multiply_saturated
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("multiply_saturated: small products pass through",
          "[gap044][multiply]")
{
    REQUIRE(multiply_saturated( 2,  2) ==  4);
    REQUIRE(multiply_saturated( 2, -2) == -4);
    REQUIRE(multiply_saturated(-2, -2) ==  4);
    REQUIRE(multiply_saturated( 1,  3) ==  3);
    REQUIRE(multiply_saturated( 0,  4) ==  0);
    REQUIRE(multiply_saturated(-1,  0) ==  0);
}

TEST_CASE("multiply_saturated: positive saturation clamps to NIT_MAX",
          "[gap044][multiply]")
{
    // Spec example: +3 × +2 = +6 → saturates to +4
    REQUIRE(multiply_saturated( 3,  2) == NIT_MAX);   // +6 → +4
    REQUIRE(multiply_saturated( 4,  4) == NIT_MAX);   // +16 → +4  (max case)
    REQUIRE(multiply_saturated(-4, -4) == NIT_MAX);   // +16 → +4
    REQUIRE(multiply_saturated( 4,  2) == NIT_MAX);   // +8 → +4
}

TEST_CASE("multiply_saturated: negative saturation clamps to NIT_MIN",
          "[gap044][multiply]")
{
    REQUIRE(multiply_saturated( 4, -4) == NIT_MIN);   // −16 → −4
    REQUIRE(multiply_saturated(-4,  4) == NIT_MIN);   // −16 → −4
    REQUIRE(multiply_saturated( 3, -2) == NIT_MIN);   // −6 → −4
}

TEST_CASE("multiply_saturated: sign (phase) is always preserved",
          "[gap044][multiply]")
{
    // Negative × positive → negative result (clamped to NIT_MIN, not inverted)
    REQUIRE(multiply_saturated(-4,  1) == -4);
    REQUIRE(multiply_saturated( 4, -1) == -4);
    // Positive × positive → positive
    REQUIRE(multiply_saturated( 4,  1) ==  4);
}

// ═══════════════════════════════════════════════════════════════════════════
// §6  carry_decompose
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("carry_decompose: GAP-044 spec example A=13 gives carry=1, remainder=4",
          "[gap044][carry_decompose]")
{
    int carry = 0, rem = 0;
    carry_decompose(13, carry, rem);
    REQUIRE(carry == 1);
    REQUIRE(rem   == 4);
    REQUIRE(carry * NONARY_CARRY_DIVISOR + rem == 13);
}

TEST_CASE("carry_decompose: negative amplitude −13",
          "[gap044][carry_decompose]")
{
    int carry = 0, rem = 0;
    carry_decompose(-13, carry, rem);
    REQUIRE(carry == -1);
    REQUIRE(rem   == -4);
    REQUIRE(carry * NONARY_CARRY_DIVISOR + rem == -13);
}

TEST_CASE("carry_decompose: amplitudes within [-4, +4] give zero carry",
          "[gap044][carry_decompose]")
{
    for (int a = NIT_MIN; a <= NIT_MAX; ++a) {
        int carry = 99, rem = 99;
        carry_decompose(a, carry, rem);
        REQUIRE(carry == 0);
        REQUIRE(rem   == a);
    }
}

TEST_CASE("carry_decompose: edge cases A=5 and A=−5",
          "[gap044][carry_decompose]")
{
    // 5 = 1×9 + (−4)  [adjustment: 5 > 4 → rem=5−9=−4, carry=1]
    int carry = 0, rem = 0;
    carry_decompose(5, carry, rem);
    REQUIRE(carry == 1);
    REQUIRE(rem   == -4);
    REQUIRE(carry * NONARY_CARRY_DIVISOR + rem == 5);

    // −5 = −1×9 + 4
    carry_decompose(-5, carry, rem);
    REQUIRE(carry == -1);
    REQUIRE(rem   ==  4);
    REQUIRE(carry * NONARY_CARRY_DIVISOR + rem == -5);
}

TEST_CASE("carry_decompose: maximum multiplication result A=16",
          "[gap044][carry_decompose]")
{
    // 16 = 2×9 + (−2)  [16/9=1 → rem=7 > 4 → rem=−2, carry=2]
    int carry = 0, rem = 0;
    carry_decompose(16, carry, rem);
    REQUIRE(carry == 2);
    REQUIRE(rem   == -2);
    REQUIRE(carry * NONARY_CARRY_DIVISOR + rem == 16);

    carry_decompose(-16, carry, rem);
    REQUIRE(carry == -2);
    REQUIRE(rem   ==  2);
    REQUIRE(carry * NONARY_CARRY_DIVISOR + rem == -16);
}

TEST_CASE("carry_decompose: identity holds for integer range [-16, +16]",
          "[gap044][carry_decompose][invariant]")
{
    for (int a = NONARY_MUL_RESULT_MIN; a <= NONARY_MUL_RESULT_MAX; ++a) {
        int carry = 0, rem = 0;
        carry_decompose(a, carry, rem);
        REQUIRE(carry * NONARY_CARRY_DIVISOR + rem == a);
        REQUIRE(rem >= NIT_MIN);
        REQUIRE(rem <= NIT_MAX);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §7  information_loss_gaussian
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("information_loss_gaussian: tight distribution (small σ) → loss near 0",
          "[gap044][info_loss]")
{
    // σ = 0.5: erfc(4.5/(0.5×√2)) = erfc(6.364) → essentially 0
    double L = information_loss_gaussian(0.5);
    REQUIRE(L >= 0.0);
    REQUIRE(L < 1e-8);
}

TEST_CASE("information_loss_gaussian: normal cognitive operation (σ=1) → L ≪ 0.01",
          "[gap044][info_loss]")
{
    // erfc(4.5/√2) = erfc(3.182) ≈ 1.47e-5
    double L = information_loss_gaussian(1.0);
    REQUIRE(L > 0.0);
    REQUIRE(L < 0.01);   // spec: normal operation L ≪ 0.01
}

TEST_CASE("information_loss_gaussian: wide distribution (large σ) → loss near 1",
          "[gap044][info_loss]")
{
    // σ = 50: erfc(4.5/(50×√2)) = erfc(0.0636) ≈ 0.928
    double L = information_loss_gaussian(50.0);
    REQUIRE(L > 0.9);
    REQUIRE(L <= 1.0);
}

TEST_CASE("information_loss_gaussian: is monotonically increasing with σ",
          "[gap044][info_loss]")
{
    REQUIRE(information_loss_gaussian(0.5) < information_loss_gaussian(1.0));
    REQUIRE(information_loss_gaussian(1.0) < information_loss_gaussian(2.0));
    REQUIRE(information_loss_gaussian(2.0) < information_loss_gaussian(5.0));
}

TEST_CASE("information_loss_gaussian: throws on σ ≤ 0",
          "[gap044][info_loss][error]")
{
    REQUIRE_THROWS_AS(information_loss_gaussian( 0.0), std::invalid_argument);
    REQUIRE_THROWS_AS(information_loss_gaussian(-1.0), std::invalid_argument);
}

// ═══════════════════════════════════════════════════════════════════════════
// §8  overflow_prob_gaussian
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("overflow_prob_gaussian: σ=1 is consistent with spec '< 5%'",
          "[gap044][overflow_prob]")
{
    // erfc(2/1) = erfc(2) ≈ 0.00468
    double p = overflow_prob_gaussian(1.0);
    REQUIRE(p > 0.0);
    REQUIRE(p < NONARY_OVERFLOW_PROB_ADD_GAUSSIAN);  // < 5%
    REQUIRE(p == Catch::Approx(std::erfc(2.0)).epsilon(1e-12));
}

TEST_CASE("overflow_prob_gaussian: tight distribution (small σ) → probability near 0",
          "[gap044][overflow_prob]")
{
    // σ = 0.5: erfc(2/0.5) = erfc(4) ≈ 1.54e-8
    double p = overflow_prob_gaussian(0.5);
    REQUIRE(p >= 0.0);
    REQUIRE(p < 1e-6);
}

TEST_CASE("overflow_prob_gaussian: is monotonically increasing with σ",
          "[gap044][overflow_prob]")
{
    REQUIRE(overflow_prob_gaussian(0.5)  < overflow_prob_gaussian(1.0));
    REQUIRE(overflow_prob_gaussian(1.0)  < overflow_prob_gaussian(2.0));
    REQUIRE(overflow_prob_gaussian(2.0)  < overflow_prob_gaussian(5.0));
}

TEST_CASE("overflow_prob_gaussian: throws on σ ≤ 0",
          "[gap044][overflow_prob][error]")
{
    REQUIRE_THROWS_AS(overflow_prob_gaussian( 0.0), std::invalid_argument);
    REQUIRE_THROWS_AS(overflow_prob_gaussian(-0.5), std::invalid_argument);
}

// ═══════════════════════════════════════════════════════════════════════════
// §9  is_saturation_alert
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("is_saturation_alert: zero total_ops → always false",
          "[gap044][alert]")
{
    REQUIRE_FALSE(is_saturation_alert(0,   0));
    REQUIRE_FALSE(is_saturation_alert(100, 0));
}

TEST_CASE("is_saturation_alert: exactly 1% is NOT an alert (strictly > 1%)",
          "[gap044][alert]")
{
    // 1/100 = exactly 1% → not an alert
    REQUIRE_FALSE(is_saturation_alert(  1,  100));
    REQUIRE_FALSE(is_saturation_alert( 10, 1000));
    REQUIRE_FALSE(is_saturation_alert(  0,  100));
}

TEST_CASE("is_saturation_alert: above 1% threshold triggers alert",
          "[gap044][alert]")
{
    // 2/100 = 2% → alert
    REQUIRE(is_saturation_alert( 2, 100));
    REQUIRE(is_saturation_alert(11, 1000));   // 1.1%
    REQUIRE(is_saturation_alert(50, 100));    // 50%
    REQUIRE(is_saturation_alert(100, 100));   // 100%
}

// ═══════════════════════════════════════════════════════════════════════════
// §10  SaturationMonitor
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("SaturationMonitor: newly constructed monitor is clean",
          "[gap044][monitor]")
{
    SaturationMonitor mon;
    REQUIRE(mon.saturated_count() == 0);
    REQUIRE(mon.total_count()     == 0);
    REQUIRE(mon.saturation_rate() == Catch::Approx(0.0).margin(1e-15));
    REQUIRE_FALSE(mon.alert());
}

TEST_CASE("SaturationMonitor: record_clean increments only total",
          "[gap044][monitor]")
{
    SaturationMonitor mon;
    mon.record_clean();
    mon.record_clean();
    mon.record_clean();
    REQUIRE(mon.total_count()     == 3);
    REQUIRE(mon.saturated_count() == 0);
    REQUIRE_FALSE(mon.alert());
}

TEST_CASE("SaturationMonitor: record_saturated increments both counters",
          "[gap044][monitor]")
{
    SaturationMonitor mon;
    mon.record_saturated();
    REQUIRE(mon.saturated_count() == 1);
    REQUIRE(mon.total_count()     == 1);
    REQUIRE(mon.saturation_rate() == Catch::Approx(1.0).epsilon(1e-12));
    REQUIRE(mon.alert());  // 100% > 1%
}

TEST_CASE("SaturationMonitor: rate and alert track 1% boundary",
          "[gap044][monitor]")
{
    SaturationMonitor mon;
    for (int i = 0; i < 99; ++i) mon.record_clean();
    mon.record_saturated();   // exactly 1% (1/100)

    REQUIRE(mon.total_count()     == 100);
    REQUIRE(mon.saturated_count() == 1);
    REQUIRE(mon.saturation_rate() == Catch::Approx(0.01).epsilon(1e-12));
    REQUIRE_FALSE(mon.alert());   // exactly 1% is NOT an alert

    mon.record_saturated();   // now 2/101 > 1%
    REQUIRE(mon.alert());
}

TEST_CASE("SaturationMonitor: reset clears all state",
          "[gap044][monitor]")
{
    SaturationMonitor mon;
    for (int i = 0; i < 50; ++i) mon.record_clean();
    for (int i = 0; i < 10; ++i) mon.record_saturated();
    REQUIRE(mon.total_count() == 60);
    REQUIRE(mon.alert());

    mon.reset();
    REQUIRE(mon.total_count()     == 0);
    REQUIRE(mon.saturated_count() == 0);
    REQUIRE_FALSE(mon.alert());
}

// ═══════════════════════════════════════════════════════════════════════════
// §11  Invariant cross-checks
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("invariant: add_saturated matches add_with_carry when carry=0",
          "[gap044][invariant]")
{
    // When no carry is emitted, both functions must return identical results
    int carry;
    for (int a = NIT_MIN; a <= NIT_MAX; ++a) {
        for (int b = NIT_MIN; b <= NIT_MAX; ++b) {
            int raw = a + b;
            if (raw >= NIT_MIN && raw <= NIT_MAX) {
                Nit sat = add_saturated(static_cast<Nit>(a), static_cast<Nit>(b));
                Nit wc  = add_with_carry(static_cast<Nit>(a), static_cast<Nit>(b), carry);
                REQUIRE(carry == 0);
                REQUIRE(sat   == wc);
            }
        }
    }
}

TEST_CASE("invariant: overflow_prob_gaussian > information_loss_gaussian at same σ",
          "[gap044][invariant]")
{
    // overflow_prob_gaussian(σ) = erfc(2/σ)       — threshold 4 on sum N(0,2σ²)
    // information_loss_gaussian(σ) = erfc(3.182/σ) — threshold 4.5 on N(0,σ²)
    //
    // Since 2 < 3.182, erfc(2/σ) > erfc(3.182/σ) for all σ > 0.
    // The addition sum overflows more readily than a single operand clips:
    // its variance is doubled (√2 wider) while the threshold is lower (4 vs 4.5).
    for (double sigma : {0.5, 1.0, 1.5, 2.0, 5.0}) {
        double p_overflow = overflow_prob_gaussian(sigma);
        double p_clip     = information_loss_gaussian(sigma);
        REQUIRE(p_overflow > p_clip);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §12  Integration test: mixing arithmetic + monitor
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("integration: SaturationMonitor tracks carry events across a workload",
          "[gap044][integration]")
{
    SaturationMonitor mon;

    // Simulate 100 additions: 91 clean pair (+2,+2) and 9 saturating pair (+4,+4)
    int carry;
    for (int i = 0; i < 91; ++i) {
        [[maybe_unused]] auto r1 = add_with_carry(static_cast<Nit>(2), static_cast<Nit>(2), carry);
        if (carry != 0) mon.record_saturated(); else mon.record_clean();
    }
    for (int i = 0; i < 9; ++i) {
        [[maybe_unused]] auto r2 = add_with_carry(static_cast<Nit>(4), static_cast<Nit>(4), carry);
        if (carry != 0) mon.record_saturated(); else mon.record_clean();
    }

    REQUIRE(mon.total_count()     == 100);
    REQUIRE(mon.saturated_count() == 9);
    REQUIRE(mon.saturation_rate() == Catch::Approx(0.09).epsilon(1e-12));
    REQUIRE(mon.alert());   // 9% >> 1%
}

TEST_CASE("integration: carry decomposition of realistic multiplication burst",
          "[gap044][integration]")
{
    // Verify that multiplying two maximum-amplitude Nits and decomposing the
    // carry preserves energy for all sign combinations
    const Nit extremes[] = {NIT_MIN, NIT_MAX};
    for (Nit a : extremes) {
        for (Nit b : extremes) {
            int product = static_cast<int>(a) * static_cast<int>(b);
            int carry = 0, rem = 0;
            carry_decompose(product, carry, rem);
            REQUIRE(carry * NONARY_CARRY_DIVISOR + rem == product);
            REQUIRE(rem >= NIT_MIN);
            REQUIRE(rem <= NIT_MAX);
        }
    }
}
