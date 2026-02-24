/**
 * @file phase49_boredom_singularity_fix_test.cpp
 * @brief Phase 49 — AUTO-04 Boredom Singularity Fix: sigmoidal accumulation.
 *
 * Validates that BoredomRegulator::update() now uses the spec §6.2 formula:
 *
 *   ΔB(t) = α_acc · (1 − tanh(k · H(Ψ))) · dt   ← bounded accumulation
 *         − decay_rate · dt                         ← passive drain
 *
 * Key properties this upgrade provides:
 *   • H = 0 → ΔB = α_acc      (finite maximum — Boredom Singularity FIXED)
 *   • H → ∞ → ΔB = 0          (no accumulation under rich stimulation)
 *   • θ_explore raised 0.7 → 0.8 per spec §6.3
 *   • Passive decay enables boredom to return to 0 after exploration
 *
 * §1   last_delta_b() accessor present; initial value = 0.0
 * §2   H=0 → last_delta_b() == α_acc (peak finite accumulation, no singularity)
 * §3   H=20 → last_delta_b() ≈ 0 (tanh saturation)
 * §4   ΔB monotonically non-increasing with H across [0, 2, 4, 6, 10]
 * §5   Boredom accumulates when H=0 (net positive per step)
 * §6   Boredom decays when H=20 (accumulation < passive drain)
 * §7   BOREDOM_EXPLORE_THRESH constant == 0.8 (spec §6.3 upgrade)
 * §8   should_explore() false when boredom < 0.8
 * §9   should_explore() true after driving boredom above 0.8 with H=0
 * §10  k_param() accessor returns constructor value
 * §11  higher k → smaller last_delta_b at same entropy (faster saturation)
 * §12  alpha_acc() accessor; scales peak ΔB linearly
 * §13  last_delta_b() matches α_acc·(1−tanh(k·H)) analytically
 * §14  H=0 × 1000 iterations: boredom clamped to 1.0, no NaN / overflow
 * §15  reset() clears boredom, last_entropy, last_delta_b
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <nikola/autonomy/entropy_estimator.hpp>

using namespace nikola::autonomy;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// ── §1  last_delta_b() accessor present; initial = 0.0 ───────────────────────

TEST_CASE("[P49-§1] last_delta_b() accessor present; initial value = 0.0",
          "[phase49]") {
    BoredomRegulator b;
    REQUIRE_THAT(b.last_delta_b(), WithinAbs(0.0f, 1e-7f));
}

// ── §2  H=0 → ΔB = α_acc (finite maximum, Boredom Singularity fixed) ─────────

TEST_CASE("[P49-§2] H=0 → last_delta_b == alpha_acc (peak finite rate)",
          "[phase49]") {
    BoredomRegulator b;
    b.update(0.0f, 1.0f);
    // tanh(0) = 0 → δb = α_acc × (1 − 0) = α_acc
    REQUIRE_THAT(b.last_delta_b(), WithinAbs(BOREDOM_ALPHA_ACC, 1e-5f));
    // Must be finite — this verifies the singularity is eliminated
    REQUIRE(std::isfinite(b.last_delta_b()));
    REQUIRE(std::isfinite(b.level()));
}

// ── §3  H=20 → ΔB ≈ 0 (tanh saturation) ─────────────────────────────────────

TEST_CASE("[P49-§3] H=20 → last_delta_b ≈ 0 (tanh saturated)",
          "[phase49]") {
    BoredomRegulator b;
    b.update(20.0f, 1.0f);
    // tanh(0.5 × 20) = tanh(10) ≈ 1 - 4.5e-9  →  δb ≈ 4.5e-10
    REQUIRE_THAT(b.last_delta_b(), WithinAbs(0.0f, 1e-4f));
}

// ── §4  ΔB monotonically non-increasing with H ───────────────────────────────

TEST_CASE("[P49-§4] last_delta_b() monotonically non-increasing with entropy",
          "[phase49]") {
    const float h_values[] = {0.0f, 2.0f, 4.0f, 6.0f, 10.0f};
    BoredomRegulator b;
    float prev_db = std::numeric_limits<float>::max();
    for (float h : h_values) {
        b.reset();
        b.update(h, 1.0f);
        const float db = b.last_delta_b();
        // Non-increasing: each higher entropy → less or equal ΔB
        REQUIRE(db <= prev_db + 1e-5f);
        prev_db = db;
    }
}

// ── §5  Boredom accumulates at H=0 ───────────────────────────────────────────

TEST_CASE("[P49-§5] boredom accumulates with H=0 (net positive per step)",
          "[phase49]") {
    BoredomRegulator b;
    // At H=0: net/s = α_acc - decay_rate = 0.1 - 0.01 = 0.09 > 0
    b.update(0.0f, 1.0f);
    REQUIRE(b.level() > 0.0f);

    float prev = b.level();
    for (int i = 0; i < 5; ++i) {
        b.update(0.0f, 1.0f);
        REQUIRE(b.level() >= prev - 1e-5f);  // non-decreasing
        prev = b.level();
    }
}

// ── §6  Boredom decays when H=20 ─────────────────────────────────────────────

TEST_CASE("[P49-§6] boredom decays with H=20 (accumulation < passive drain)",
          "[phase49]") {
    BoredomRegulator b;
    // Drive boredom up first
    for (int i = 0; i < 8; ++i) b.update(0.0f, 1.0f);  // ≈0.72
    const float hi = b.level();
    REQUIRE(hi > 0.5f);  // ensure we have some boredom to drain

    // High entropy: ΔB ≈ 0 but decay = 0.01/s → net negative
    b.update(20.0f, 1.0f);
    REQUIRE(b.level() < hi);
}

// ── §7  BOREDOM_EXPLORE_THRESH == 0.8 (spec §6.3 upgrade) ───────────────────

TEST_CASE("[P49-§7] BOREDOM_EXPLORE_THRESH constant == 0.8",
          "[phase49]") {
    REQUIRE_THAT(BOREDOM_EXPLORE_THRESH, WithinAbs(0.8f, 1e-6f));
}

// ── §8  should_explore() false when boredom < threshold ──────────────────────

TEST_CASE("[P49-§8] should_explore() false when boredom < 0.8",
          "[phase49]") {
    BoredomRegulator b;
    // Fresh regulator: boredom = 0
    REQUIRE_FALSE(b.should_explore());

    // A few updates — still well below 0.8
    for (int i = 0; i < 3; ++i) b.update(0.0f, 1.0f);  // ≈0.27
    REQUIRE_FALSE(b.should_explore());
    REQUIRE(b.level() < BOREDOM_EXPLORE_THRESH);
}

// ── §9  should_explore() true after > 0.8 ────────────────────────────────────

TEST_CASE("[P49-§9] should_explore() true after boredom driven above 0.8",
          "[phase49]") {
    BoredomRegulator b;
    // Net 0.09/s × 10 s = 0.90 > 0.80 threshold
    for (int i = 0; i < 10; ++i) b.update(0.0f, 1.0f);
    REQUIRE(b.level() >= BOREDOM_EXPLORE_THRESH);
    REQUIRE(b.should_explore());
}

// ── §10 k_param() accessor returns constructor value ─────────────────────────

TEST_CASE("[P49-§10] k_param() accessor returns the constructor-specified value",
          "[phase49]") {
    BoredomRegulator slow(BOREDOM_ALPHA_ACC, 0.25f, BOREDOM_DECAY_RATE);
    REQUIRE_THAT(slow.k_param(), WithinAbs(0.25f, 1e-6f));

    BoredomRegulator fast(BOREDOM_ALPHA_ACC, 2.0f, BOREDOM_DECAY_RATE);
    REQUIRE_THAT(fast.k_param(), WithinAbs(2.0f, 1e-6f));
}

// ── §11 higher k → smaller last_delta_b at H>0 (faster saturation) ──────────

TEST_CASE("[P49-§11] higher k gives smaller ΔB at same H (faster saturation)",
          "[phase49]") {
    // At H=2: tanh(k*2) grows with k → (1-tanh) shrinks with k
    const float test_h = 2.0f;

    BoredomRegulator slow(BOREDOM_ALPHA_ACC, 0.5f, BOREDOM_DECAY_RATE);  // k=0.5
    BoredomRegulator fast(BOREDOM_ALPHA_ACC, 2.0f, BOREDOM_DECAY_RATE);  // k=2.0

    slow.update(test_h, 1.0f);
    fast.update(test_h, 1.0f);

    // k=2.0: δb = 0.1*(1-tanh(4)) ≈ 0.1*0.0007 ≈ 0.00073
    // k=0.5: δb = 0.1*(1-tanh(1)) ≈ 0.1*0.238 ≈ 0.0238
    REQUIRE(fast.last_delta_b() < slow.last_delta_b());
}

// ── §12 alpha_acc() accessor; scales peak ΔB linearly ────────────────────────

TEST_CASE("[P49-§12] alpha_acc() accessor; doubles alpha → doubles peak ΔB",
          "[phase49]") {
    BoredomRegulator b1(0.1f, BOREDOM_K, BOREDOM_DECAY_RATE);
    BoredomRegulator b2(0.2f, BOREDOM_K, BOREDOM_DECAY_RATE);

    b1.update(0.0f, 0.001f);  // tiny dt to avoid clamp masking the ratio
    b2.update(0.0f, 0.001f);

    REQUIRE_THAT(b1.alpha_acc(), WithinAbs(0.1f, 1e-6f));
    REQUIRE_THAT(b2.alpha_acc(), WithinAbs(0.2f, 1e-6f));
    // At H=0 both are just reading back alpha_acc (tanh(0)=0)
    REQUIRE_THAT(b2.last_delta_b(),
                 WithinRel(b1.last_delta_b() * 2.0f, 1e-4f));
}

// ── §13 last_delta_b() matches formula analytically ──────────────────────────

TEST_CASE("[P49-§13] last_delta_b() == α_acc·(1−tanh(k·H)) analytically",
          "[phase49]") {
    const float alpha = 0.1f;
    const float k     = 0.5f;
    const float H     = 3.0f;

    BoredomRegulator b(alpha, k, BOREDOM_DECAY_RATE);
    b.update(H, 1.0f);

    const float expected = alpha * (1.0f - std::tanh(k * H));
    REQUIRE_THAT(b.last_delta_b(), WithinAbs(expected, 1e-5f));
}

// ── §14 H=0 × 1000 iterations: clamped to 1.0, no NaN / overflow ─────────────

TEST_CASE("[P49-§14] H=0 × 1000 iterations: boredom clamped ≤ 1.0, no NaN",
          "[phase49]") {
    BoredomRegulator b;
    for (int i = 0; i < 1000; ++i) b.update(0.0f, 0.1f);  // 100 simulated seconds

    REQUIRE(std::isfinite(b.level()));
    REQUIRE(b.level() <= 1.0f);
    REQUIRE(b.level() >= 0.0f);
    REQUIRE(std::isfinite(b.last_delta_b()));
}

// ── §15 reset() clears boredom, last_entropy, last_delta_b ───────────────────

TEST_CASE("[P49-§15] reset() clears all state to zero",
          "[phase49]") {
    BoredomRegulator b;
    for (int i = 0; i < 5; ++i) b.update(2.0f, 1.0f);

    // Verify non-zero state before reset
    REQUIRE(b.level() > 0.0f);
    REQUIRE(b.last_entropy() > 0.0f);
    REQUIRE(b.last_delta_b() > 0.0f);

    b.reset();
    REQUIRE_THAT(b.level(),        WithinAbs(0.0f, 1e-7f));
    REQUIRE_THAT(b.last_entropy(), WithinAbs(0.0f, 1e-7f));
    REQUIRE_THAT(b.last_delta_b(), WithinAbs(0.0f, 1e-7f));
    REQUIRE_FALSE(b.should_explore());
}
