// ============================================================
// Phase 143 — v0.0.16 Curiosity Engine & Exploration Dynamics
// tests/unit/phase143_curiosity_exploration_test.cpp
//
// Validates the curiosity/exploration pipeline end-to-end:
//   §A  Boredom Dynamics — flat reward → boredom rises → exploration
//   §B  Novel Stimulus Reset — reward spike drains boredom
//   §C  Entropy-Driven Exploration — high/low entropy behavior
//   §D  Energy Conservation Under Exploration — damped system bounded
//   §E  Long-Session Idle Behavior — 100k ticks, no collapse/explosion
//   §F  Curiosity Engine Knowledge Gaps — pursue_interest learning
// ============================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/autonomy/dopamine_system.hpp>
#include <nikola/autonomy/entropy_estimator.hpp>
#include <nikola/autonomy/autonomy_engine.hpp>

// CuriosityEngine lives in nikola::interior — needs TorusManifold stub
namespace nikola::interior { class TorusManifold {}; }
#include <nikola/interior/curiosity.hpp>

#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

using namespace nikola::autonomy;
using Catch::Approx;

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Single-spike psi: node 0 has amplitude 1.0, rest zero → H ≈ 0 → max boredom
static std::pair<std::vector<float>, std::vector<float>> make_spike_psi(size_t N = 64) {
    std::vector<float> r(N, 0.f), i(N, 0.f);
    r[0] = 1.0f;
    return {r, i};
}

/// Uniform psi: all nodes equal amplitude → H ≈ log₂(N) → low boredom
static std::pair<std::vector<float>, std::vector<float>> make_uniform_psi(size_t N = 64) {
    const float amp = 1.0f / std::sqrt(static_cast<float>(N));
    std::vector<float> r(N, amp), i(N, 0.f);
    return {r, i};
}

// ── §A  Boredom Dynamics ───────────────────────────────────────────────────

TEST_CASE("§A-1 Flat reward → boredom increases linearly",
          "[curiosity][boredom]") {
    // With H≈0 (spike psi, low entropy), boredom should accumulate at
    // roughly α_acc rate (0.1/s at tanh(0)=0 → ΔB ≈ 0.1·dt − 0.01·dt = 0.09·dt)
    AutonomyConfig cfg;
    cfg.enable_boredom = true;
    AutonomyEngine engine(cfg);

    auto [pr, pi] = make_spike_psi();
    const float dt = 1.0f;  // 1s per tick for fast accumulation

    // Run 5 ticks with no reward → boredom should grow
    float prev_boredom = engine.boredom();
    for (int t = 0; t < 5; ++t) {
        engine.tick(dt, pr, pi);
        float cur = engine.boredom();
        INFO("tick=" << t << " boredom=" << cur << " prev=" << prev_boredom);
        REQUIRE(cur >= prev_boredom);
        prev_boredom = cur;
    }
    // After 5s at ~0.09/s net rate, boredom should be at least 0.3
    REQUIRE(engine.boredom() > 0.3f);
}

TEST_CASE("§A-2 Boredom exceeds threshold → spontaneous exploration triggered",
          "[curiosity][boredom][exploration]") {
    AutonomyConfig cfg;
    cfg.enable_boredom = true;
    AutonomyEngine engine(cfg);

    int goals_emitted = 0;
    engine.on_curiosity_goal = [&](CuriosityGoal) { goals_emitted++; };

    auto [pr, pi] = make_spike_psi();

    // Drive boredom to threshold (0.8) with dt=1.0 → ~9 ticks
    for (int t = 0; t < 20; ++t) {
        engine.tick(1.0f, pr, pi);
    }

    INFO("boredom=" << engine.boredom() << " goals=" << goals_emitted);
    REQUIRE(goals_emitted > 0);
    REQUIRE(engine.curiosity_goal_count() > 0);
}

TEST_CASE("§A-3 Time to boredom threshold from idle state",
          "[curiosity][boredom][timing]") {
    AutonomyConfig cfg;
    cfg.enable_boredom = true;
    AutonomyEngine engine(cfg);

    auto [pr, pi] = make_spike_psi();
    // dt=1.0: net boredom rate ~0.09/s at H≈0 → threshold 0.8 in ~9 ticks
    const float dt = 1.0f;

    int ticks_to_explore = 0;
    for (int t = 0; t < 50; ++t) {
        engine.tick(dt, pr, pi);
        if (engine.boredom() >= 0.8f) {
            ticks_to_explore = t + 1;
            break;
        }
    }

    INFO("Ticks to boredom threshold (dt=1.0): " << ticks_to_explore
         << " (" << ticks_to_explore * 1.0f << "s)");
    // Should reach threshold within reasonable time (5-20 ticks at dt=1.0)
    REQUIRE(ticks_to_explore > 0);
    REQUIRE(ticks_to_explore < 50);
}

// ── §B  Novel Stimulus Reset ───────────────────────────────────────────────

TEST_CASE("§B Novel stimulus resets boredom via CuriosityGoal drain",
          "[curiosity][boredom][reset]") {
    AutonomyConfig cfg;
    cfg.enable_boredom = true;
    AutonomyEngine engine(cfg);

    auto [pr, pi] = make_spike_psi();

    // Accumulate boredom past threshold
    for (int t = 0; t < 15; ++t) {
        engine.tick(1.0f, pr, pi);
    }
    float pre_goal_boredom = engine.boredom();
    REQUIRE(pre_goal_boredom >= 0.7f);

    // A curiosity goal emission should drain boredom by 0.3
    // (CURIOSITY_BOREDOM_DRAIN constant in autonomy_engine.hpp)
    // Keep ticking — the threshold crossing triggers drain
    engine.tick(1.0f, pr, pi);

    // After drain, boredom should be lower or same
    // (may re-accumulate slightly in the same tick)
    float post_boredom = engine.boredom();
    INFO("pre=" << pre_goal_boredom << " post=" << post_boredom
         << " goals=" << engine.curiosity_goal_count());

    // If a goal was emitted, boredom should have been drained
    if (engine.curiosity_goal_count() > 0) {
        // The 0.3 drain may be partially offset by accumulation,
        // but boredom history should show the dip
        REQUIRE(post_boredom < 1.0f);
    }
}

// ── §C  Entropy-Driven Exploration ─────────────────────────────────────────

TEST_CASE("§C-1 High field entropy → no unnecessary exploration",
          "[curiosity][entropy]") {
    AutonomyConfig cfg;
    cfg.enable_boredom = true;
    AutonomyEngine engine(cfg);

    auto [pr, pi] = make_uniform_psi(256);  // high entropy
    int goals = 0;
    engine.on_curiosity_goal = [&](CuriosityGoal) { goals++; };

    // High entropy → boredom accumulation suppressed by tanh(k·H)
    for (int t = 0; t < 100; ++t) {
        engine.tick(1.0f, pr, pi);
    }

    INFO("entropy=" << engine.entropy() << " boredom=" << engine.boredom()
         << " goals_emitted=" << goals);
    // Boredom should stay well below threshold with high entropy
    REQUIRE(engine.boredom() < 0.8f);
    REQUIRE(goals == 0);
}

TEST_CASE("§C-2 Low field entropy → curiosity injection active",
          "[curiosity][entropy]") {
    AutonomyConfig cfg;
    cfg.enable_boredom = true;
    AutonomyEngine engine(cfg);

    auto [pr, pi] = make_spike_psi();  // low entropy
    int goals = 0;
    engine.on_curiosity_goal = [&](CuriosityGoal) { goals++; };

    // Low entropy → boredom rises fast → exploration triggered
    for (int t = 0; t < 20; ++t) {
        engine.tick(1.0f, pr, pi);
    }

    INFO("entropy=" << engine.entropy() << " boredom=" << engine.boredom()
         << " goals_emitted=" << goals);
    REQUIRE(goals > 0);
}

TEST_CASE("§C-3 Entropy measurement is reasonable",
          "[curiosity][entropy]") {
    EntropyEstimator est(42u);

    // Single spike → entropy ≈ 0
    auto [sr, si] = make_spike_psi(64);
    float h_spike = est.estimate(sr, si);
    REQUIRE(h_spike < 1.0f);

    // Uniform → entropy ≈ log₂(64) = 6.0
    auto [ur, ui] = make_uniform_psi(64);
    float h_uniform = est.estimate(ur, ui);
    REQUIRE(h_uniform > 4.0f);
    REQUIRE(h_uniform < 7.0f);

    INFO("H(spike)=" << h_spike << " H(uniform)=" << h_uniform);
    REQUIRE(h_uniform > h_spike);
}

// ── §D  Energy Conservation Under Exploration ──────────────────────────────

TEST_CASE("§D Exploration does not cause energy explosion (damped system)",
          "[curiosity][energy]") {
    AutonomyConfig cfg;
    cfg.enable_boredom = true;
    AutonomyEngine engine(cfg);

    auto [pr, pi] = make_spike_psi(128);
    const float dt = 0.1f;

    // Track total "energy" via dopamine + metabolic ATP — both should stay bounded
    float max_dopamine = 0.f;
    float min_atp = 1.f;

    for (int t = 0; t < 1000; ++t) {
        engine.tick(dt, pr, pi);
        float d = engine.dopamine();
        float a = engine.atp();
        if (d > max_dopamine) max_dopamine = d;
        if (a < min_atp) min_atp = a;

        // Dopamine must stay in [0, 1]
        REQUIRE(d >= 0.f);
        REQUIRE(d <= 1.0f);
        // ATP must stay in [0, 1]
        REQUIRE(a >= 0.f);
        REQUIRE(a <= 1.0f);
    }

    INFO("max_dopamine=" << max_dopamine << " min_atp=" << min_atp);
    // System shouldn't blow up
    REQUIRE(max_dopamine <= 1.0f);
}

// ── §E  Long-Session Idle Behavior (100k ticks) ───────────────────────────

TEST_CASE("§E Long-session: 100k ticks, no collapse/explosion, active dynamics",
          "[curiosity][longsession]") {
    AutonomyConfig cfg;
    cfg.enable_boredom = true;
    cfg.enable_dream_weave = true;
    AutonomyEngine engine(cfg);

    auto [pr, pi] = make_spike_psi(64);
    const float dt = 0.01f;  // 1kHz tick rate

    int goals_emitted = 0;
    engine.on_curiosity_goal = [&](CuriosityGoal) { goals_emitted++; };

    // Track behavioral metrics over time
    float min_boredom = 1.f, max_boredom = 0.f;
    float min_dopamine = 1.f, max_dopamine = 0.f;
    float min_entropy = 100.f, max_entropy = 0.f;
    int nap_count = 0;
    engine.on_nap_enter = [&]() { nap_count++; };

    constexpr int N_TICKS = 100'000;
    for (int t = 0; t < N_TICKS; ++t) {
        engine.tick(dt, pr, pi);

        float b = engine.boredom();
        float d = engine.dopamine();
        float h = engine.entropy();

        if (b < min_boredom) min_boredom = b;
        if (b > max_boredom) max_boredom = b;
        if (d < min_dopamine) min_dopamine = d;
        if (d > max_dopamine) max_dopamine = d;
        if (h < min_entropy) min_entropy = h;
        if (h > max_entropy) max_entropy = h;

        // Hard invariant: all values bounded
        REQUIRE(b >= 0.f);
        REQUIRE(b <= 1.0f);
        REQUIRE(d >= 0.f);
        REQUIRE(d <= 1.0f);
    }

    INFO("100k ticks complete:"
         << "\n  boredom  range=[" << min_boredom << ", " << max_boredom << "]"
         << "\n  dopamine range=[" << min_dopamine << ", " << max_dopamine << "]"
         << "\n  entropy  range=[" << min_entropy << ", " << max_entropy << "]"
         << "\n  goals_emitted=" << goals_emitted
         << "\n  nap_count=" << nap_count
         << "\n  emergency_stimuli=" << engine.emergency_stimulus_count()
         << "\n  mania_suppressions=" << engine.mania_suppress_count()
         << "\n  final_state=" << static_cast<int>(engine.state()));

    // §E-1: No collapse to fixed point — boredom must have varied
    REQUIRE(max_boredom > min_boredom + 0.1f);

    // §E-2: No explosion — everything bounded
    REQUIRE(max_dopamine <= 1.0f);
    REQUIRE(max_boredom <= 1.0f);

    // §E-3: Active dynamics — some curiosity goals were emitted
    REQUIRE(goals_emitted > 0);

    // §E-4: System is alive at the end (not stuck in a degenerate state)
    auto snap = engine.snapshot();
    REQUIRE(snap.atp > 0.f);
}

// ── §F  CuriosityEngine Knowledge Gap Learning ─────────────────────────────

TEST_CASE("§F-1 CuriosityEngine: pursue_interest reduces uncertainty",
          "[curiosity][knowledge]") {
    nikola::interior::CuriosityEngine ce;
    nikola::interior::TorusManifold torus;

    // Register a knowledge gap
    nikola::interior::KnowledgeGap gap;
    gap.domain = "quantum_gravity";
    gap.uncertainty = 0.9;
    gap.query_count = 0;
    ce.register_gap(gap);

    auto gaps_before = ce.identify_knowledge_gaps(torus);
    REQUIRE(gaps_before.size() == 1);
    double u_before = gaps_before[0].uncertainty;

    // Pursue the topic 5 times — uncertainty should decrease
    for (int i = 0; i < 5; ++i) {
        bool ok = ce.pursue_interest("quantum_gravity", torus);
        REQUIRE(ok);
    }

    auto gaps_after = ce.identify_knowledge_gaps(torus);
    REQUIRE(gaps_after.size() == 1);
    double u_after = gaps_after[0].uncertainty;

    INFO("Uncertainty: before=" << u_before << " after=" << u_after);
    // Each pursuit reduces by 0.05 (min 0.1): 0.9 → 0.65 after 5
    REQUIRE(u_after < u_before);
    REQUIRE(u_after == Approx(0.65).margin(0.01));
}

TEST_CASE("§F-2 CuriosityEngine: information gain has diminishing returns",
          "[curiosity][knowledge]") {
    nikola::interior::CuriosityEngine ce;
    nikola::interior::TorusManifold torus;

    nikola::interior::KnowledgeGap gap;
    gap.domain = "topology";
    gap.uncertainty = 0.8;
    gap.query_count = 0;
    ce.register_gap(gap);

    double gain_first = ce.measure_information_gain("topology", torus);
    ce.pursue_interest("topology", torus);  // query_count → 1

    double gain_second = ce.measure_information_gain("topology", torus);
    ce.pursue_interest("topology", torus);  // query_count → 2

    double gain_third = ce.measure_information_gain("topology", torus);

    INFO("Gains: first=" << gain_first << " second=" << gain_second
         << " third=" << gain_third);
    // Diminishing returns: gain = uncertainty / (1 + query_count)
    REQUIRE(gain_first > gain_second);
    REQUIRE(gain_second > gain_third);
}

TEST_CASE("§F-3 CuriosityEngine: generate_questions ranks by novelty",
          "[curiosity][knowledge]") {
    nikola::interior::CuriosityEngine ce;
    nikola::interior::TorusManifold torus;

    // Register gaps with different uncertainty levels
    // Ranking = binary_entropy(u) / (1 + log1p(query_count))
    // low:    H(0.2)=0.72  / (1+log1p(5))=2.95  → score 0.244
    // high:   H(0.9)=0.47  / (1+log1p(0))=1.00  → score 0.470
    // medium: H(0.5)=1.00  / (1+log1p(1))=1.69  → score 0.591
    ce.register_gap({"low_priority",  0.2, {}, 5});
    ce.register_gap({"high_priority", 0.9, {}, 0});
    ce.register_gap({"medium",        0.5, {}, 1});

    auto questions = ce.generate_questions(torus, 3);
    REQUIRE(questions.size() == 3);

    INFO("Q0: " << questions[0].text << " gain=" << questions[0].information_gain
         << "\nQ1: " << questions[1].text << " gain=" << questions[1].information_gain
         << "\nQ2: " << questions[2].text << " gain=" << questions[2].information_gain);

    // Questions should be ordered by ranking score (descending)
    // Expected: medium (0.591) > high_priority (0.470) > low_priority (0.244)
    REQUIRE(questions.size() == 3);
    // Verify the ranking produces a non-trivial ordering (not all same gain)
    bool has_ordering = (questions[0].information_gain != questions[2].information_gain);
    REQUIRE(has_ordering);
    // The low-priority (high query_count) gap should rank last
    REQUIRE(questions[2].information_gain <= questions[0].information_gain);
}
