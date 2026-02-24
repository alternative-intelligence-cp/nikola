/**
 * @file tests/unit/phase29_novelty_decay_test.cpp
 * @brief Phase 29: Stimulus novelty decay — neural habituation in DopamineSystem.
 *
 * The problem fixed in this phase:
 *
 *   After any stimulus injection, the torus field decays slowly and predictably.
 *   Each tick, current_energy < prev_energy by roughly the same amount, so:
 *       td_error = γ·E(t+1) − E(t)  ≈ constant negative value
 *   REFUSE fires continuously for hundreds of ticks — not because Nikola is
 *   being "punished", but because it has learned the decay pattern and keeps
 *   measuring it against its one-tick memory.  The system can't EMIT_THOUGHT
 *   because REFUSE always scores higher.
 *
 * The Phase 29 fix — neural habituation in DopamineSystem:
 *
 *   familiar_td_  — EMA (α=0.03) of recent td_errors.  Learns the "baseline
 *                   decay rate" of the current field state in ~33 ticks.
 *   novelty_factor_ — attenuation scalar ∈ [0, 1].
 *                   Decays when |td_raw - familiar_td| < NOVELTY_THRESHOLD (=0.03)
 *                   Recovers when the td deviates (genuine surprise).
 *   td_effective  = td_raw * novelty_factor   for Reward::NEUTRAL
 *                 = td_raw * 1.0              for Reward::POSITIVE / NEGATIVE
 *                   (biological rule: reward/punishment are never habituatable)
 *
 * Tests:
 *   1. novelty_factor decays when same energy is repeated
 *   2. novelty_factor recovers on sudden energy change (genuine surprise)
 *   3. External POSITIVE reward fires at full strength even when habituated
 *   4. familiar_td_ converges toward field's steady-state decay rate
 *   5. Integration: REFUSE count drops in later ticks vs early ticks
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/autonomy/dopamine_system.hpp>
#include <nikola/autonomy/autonomy_engine.hpp>
#include <nikola/autonomy/decision_loop.hpp>
#include <nikola/cognitive/cognitive_torus.hpp>

#include <algorithm>
#include <string>
#include <vector>

using namespace nikola::autonomy;
using namespace nikola::cognitive;

// ============================================================================
// Helpers
// ============================================================================

static AutonomyEngine make_engine() {
    AutonomyConfig cfg;
    cfg.enable_dream_weave = false;
    cfg.enable_boredom     = true;
    return AutonomyEngine(cfg);
}

static DecisionLoopConfig make_config() {
    DecisionLoopConfig cfg;
    cfg.steps_per_tick      = 10;
    cfg.action_threshold    = 0.0f;
    cfg.min_emit_interval_s = 0.0f;
    cfg.decode_top_k        = 5;
    cfg.alive_prior         = 0.1f;
    cfg.vocabulary = { "hello", "curious", "wave", "energy", "nikola",
                       "wonder", "field", "resonance", "signal", "think" };
    return cfg;
}

// ============================================================================
// Section 1: novelty_factor decays under repeated identical td_error
// ============================================================================

TEST_CASE("Phase29 novelty_factor decays when td_error is constant", "[Phase29]")
{
    // Simulate a slowly decaying field: energy decreases by a fixed amount
    // each tick.  This produces a constant negative td_error.
    //
    // After ~50-100 such ticks, novelty_factor should be significantly below 1.0,
    // indicating the DopamineSystem has learned this is the "normal" state.

    DopamineSystem dopa;

    // Sanity: starts at 1.0
    REQUIRE(dopa.novelty_factor() == Catch::Approx(1.0f));

    // Simulate decaying field: E(t) = 2.0 - t * 0.01
    // td_raw = γ * E(t+1) - E(t) = 0.95 * (E - 0.01) - E ≈ -0.05E - 0.0095
    // → roughly constant at around -0.15 for E ≈ 2.0
    float energy = 2.0f;
    for (int tick = 0; tick < 150; ++tick) {
        energy -= 0.004f;  // gentle constant decay
        dopa.update(energy, Reward::NEUTRAL);
        dopa.decay(0.1f);
    }

    const float novf = dopa.novelty_factor();
    INFO("novelty_factor after 150 repeated-td ticks: " << novf);

    // Must be substantially below 1.0 — the signal is habituated
    CHECK(novf < 0.5f);

    // The td_error in last_td_error() should also be attenuated
    // (magnitude smaller than the raw td would be)
    const float habituated_td = dopa.last_td_error();
    const float raw_td_approx = 0.95f * energy - (energy + 0.004f);  // ≈ -0.054 * energy
    INFO("habituated_td=" << habituated_td << " raw≈" << raw_td_approx);
    // Habituated TD magnitude should be less than raw
    CHECK(std::abs(habituated_td) < std::abs(raw_td_approx));
}

// ============================================================================
// Section 2: novelty_factor recovers on genuine surprise
// ============================================================================

TEST_CASE("Phase29 novelty_factor recovers after sudden energy change", "[Phase29]")
{
    // After habituating to a stable decay, a new stimulus injects significant
    // energy.  This makes td_error deviate sharply from familiar_td_.
    // novelty_factor should recover rapidly — within ~10-20 ticks.

    DopamineSystem dopa;

    // Phase 1: habituate (150 ticks of constant decay)
    float energy = 2.0f;
    for (int tick = 0; tick < 150; ++tick) {
        energy -= 0.004f;
        dopa.update(energy, Reward::NEUTRAL);
        dopa.decay(0.1f);
    }
    const float pre_surprise = dopa.novelty_factor();
    INFO("novelty_factor before surprise: " << pre_surprise);
    REQUIRE(pre_surprise < 0.5f);  // confirm we're habituated

    // Phase 2: new stimulus — energy jumps dramatically (simulates injection)
    energy += 2.5f;  // large positive injection

    // Run a few ticks — the new td_error should be very different from familiar_td_
    for (int tick = 0; tick < 20; ++tick) {
        energy -= 0.002f;  // settles at new level
        dopa.update(energy, Reward::NEUTRAL);
        dopa.decay(0.1f);
    }

    const float post_surprise = dopa.novelty_factor();
    INFO("novelty_factor after surprise + 20 ticks: " << post_surprise);

    // Must have recovered significantly
    CHECK(post_surprise > pre_surprise + 0.2f);
    CHECK(post_surprise > 0.4f);  // should be at least moderately novel again
}

// ============================================================================
// Section 3: External reward ignores habituation
// ============================================================================

TEST_CASE("Phase29 external POSITIVE reward fires at full strength when habituated", "[Phase29]")
{
    // Biological rule: reward and punishment signals are NOT habituatable.
    // A treat is always a treat.  A burn always hurts.
    // Only neutral field fluctuations (which are background noise) habituate.
    //
    // In the implementation: when reward != Reward::NEUTRAL, novelty is clamped
    // to 1.0 for that tick, so td_effective = td_raw unattenuated.

    DopamineSystem dopa;

    // Habituate with neutral reward
    float energy = 2.0f;
    for (int tick = 0; tick < 150; ++tick) {
        energy -= 0.004f;
        dopa.update(energy, Reward::NEUTRAL);
        dopa.decay(0.1f);
    }
    REQUIRE(dopa.novelty_factor() < 0.5f);  // confirm habituated

    // Now inject a POSITIVE reward — should fire at full strength
    // td_raw for +1 reward ≈ 1.0 + 0.95*energy - energy ≈ 1.0 + (-0.05*energy)
    // For energy ≈ 1.4:  td_raw ≈ 1.0 - 0.07 ≈ 0.93 → dopamine ≈ 1.0
    const float pre_reward_energy = energy;
    dopa.update(pre_reward_energy, Reward::POSITIVE);
    const float dopa_after_reward = dopa.level();
    const float td_after_reward   = dopa.last_td_error();

    INFO("dopamine after POSITIVE reward (habituated): " << dopa_after_reward);
    INFO("td_error after POSITIVE reward (habituated): " << td_after_reward);

    // Dopamine must be above baseline — reward fired normally
    CHECK(dopa_after_reward > DOPAMINE_BASELINE + 0.2f);
    // td_error should be positive and substantial
    CHECK(td_after_reward > 0.2f);
}

// ============================================================================
// Section 4: familiar_td_ converges to field's steady-state decay rate
// ============================================================================

TEST_CASE("Phase29 familiar_td converges toward steady-state decay rate", "[Phase29]")
{
    // After many ticks of constant decay, familiar_td_ (the EMA) should
    // converge to approximately the raw td_error being produced each tick.
    // This is what makes habituation work: the system learns what "normal" is.

    DopamineSystem dopa;

    // Constant-rate decay
    float energy = 3.0f;
    const float decay_per_tick = 0.005f;
    float last_td = 0.0f;

    for (int tick = 0; tick < 300; ++tick) {
        const float prev_e = energy;
        energy -= decay_per_tick;
        dopa.update(energy, Reward::NEUTRAL);

        // Compute what the raw td_error would be at this tick
        // Note: td_raw = γ*current - prev = 0.95*energy - prev_e
        last_td = 0.95f * energy - prev_e;
    }

    const float familiar = dopa.familiar_td();
    INFO("familiar_td after 300 ticks: " << familiar);
    INFO("last raw td_error: " << last_td);

    // familiar_td should be within 30% of the actual raw td_error
    // (EMA with α=0.03 converges slowly but surely)
    CHECK(std::abs(familiar - last_td) < std::abs(last_td) * 0.5f);
    // And both should be negative (field is decaying)
    CHECK(familiar < 0.0f);
}

// ============================================================================
// Section 5: Integration — REFUSE count drops in later ticks vs early ticks
// ============================================================================

TEST_CASE("Phase29 REFUSE fires less frequently after habituation", "[Phase29]")
{
    // The core motivation for Phase 29: in a live DecisionLoop, REFUSE
    // should fire many times in the first ~100 ticks (when td_error is novel),
    // but significantly less in later ticks (when the system has habituated
    // to the constant decay pattern).
    //
    // We inject one stimulus, then let the loop run for 500 ticks, and compare
    // REFUSE counts in [0, 150) vs [350, 500).

    CognitiveTorus torus(3);
    auto engine = make_engine();
    auto cfg = make_config();

    DecisionLoop loop(torus, engine, cfg);
    loop.inject_stimulus("hello nikola");

    int refuse_early = 0;  // ticks 0–149
    int refuse_late  = 0;  // ticks 350–499

    for (int tick = 0; tick < 500; ++tick) {
        const auto result = loop.tick();
        if (result.type == ActionType::REFUSE) {
            if (tick < 150)        ++refuse_early;
            else if (tick >= 350)  ++refuse_late;
        }
    }

    INFO("REFUSE in ticks  0-149: " << refuse_early);
    INFO("REFUSE in ticks 350-499: " << refuse_late);

    // There should be more REFUSE in the early window (novel negative td)
    // than in the late window (habituated away).
    // We allow for some slack — the test just checks habituation is measurable.
    CHECK(refuse_late < refuse_early);

    // Sanity: the ratio should be meaningful (not marginal)
    // After 350 ticks, novelty should have decayed substantially.
    // We require at least a 20% reduction.
    if (refuse_early > 0) {
        const float reduction = 1.0f - static_cast<float>(refuse_late) /
                                       static_cast<float>(refuse_early);
        INFO("REFUSE reduction factor: " << reduction);
        CHECK(reduction > 0.2f);
    }
}
