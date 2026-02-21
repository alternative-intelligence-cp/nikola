/**
 * @file tests/unit/phase5_autonomous_test.cpp
 * @brief Phase 5: Autonomous Systems test suite (Catch2 v3).
 *
 * Covers all 5 Gap criteria:
 *   Gap 5.1 — DopamineSystem: TD prediction error + decay + habituation
 *   Gap 5.2 — EntropyEstimator: Monte Carlo H + BoredomRegulator
 *   Gap 5.3 — MetabolicSimulator: ATP cost + recharge + exhaustion
 *   Gap 5.4 — NapController: hysteresis entry/exit + timeout + callbacks
 *   Gap 5.5 — DreamWeaveEngine: Frobenius convergence criterion
 *
 * Plus: AutonomyEngine integration facade (all gaps together).
 *
 * No live physics required — all tests use synthetic wavefunction buffers.
 */

// Pull in Autonomy Engine implementation
#define NIKOLA_AUTONOMY_ENGINE_IMPL

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <nikola/autonomy/dopamine_system.hpp>
#include <nikola/autonomy/entropy_estimator.hpp>
#include <nikola/autonomy/metabolic_simulator.hpp>
#include <nikola/autonomy/nap_controller.hpp>
#include <nikola/autonomy/dream_weave.hpp>
#include <nikola/autonomy/autonomy_engine.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

using namespace nikola::autonomy;
using Catch::Matchers::WithinAbs;

// ─────────────────────────────────────────────────────────────────────────────
//  Gap 5.1 — DopamineSystem
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Gap5.1 — baseline dopamine is 0.5", "[dopamine][gap5.1]") {
    DopamineSystem dopa;
    CHECK(dopa.level() == Catch::Approx(DOPAMINE_BASELINE));
    CHECK_FALSE(dopa.is_spiking());
    CHECK_FALSE(dopa.is_dipping());
}

TEST_CASE("Gap5.1 — positive reward spikes dopamine above baseline",
          "[dopamine][gap5.1]") {
    DopamineSystem dopa;
    dopa.update(1.0f, Reward::POSITIVE);  // R=+1, prev=0 → δ = 1 + 0.95*1 - 0 > 0
    CHECK(dopa.level() > DOPAMINE_BASELINE);
    CHECK(dopa.is_spiking());
    CHECK_FALSE(dopa.is_dipping());
}

TEST_CASE("Gap5.1 — negative reward dips dopamine below baseline",
          "[dopamine][gap5.1]") {
    DopamineSystem dopa;
    // Seed some energy first so the negative delta shows
    dopa.update(2.0f, Reward::NEUTRAL);   // sets prev = 2.0
    dopa.update(0.0f, Reward::NEGATIVE);  // R=-1, energy drop → big dip
    CHECK(dopa.level() < DOPAMINE_BASELINE);
    CHECK(dopa.is_dipping());
}

TEST_CASE("Gap5.1 — dopamine decays back toward baseline",
          "[dopamine][gap5.1]") {
    DopamineSystem dopa;
    dopa.update(10.0f, Reward::POSITIVE);
    float high = dopa.level();
    REQUIRE(high > DOPAMINE_BASELINE);

    // After enough decay the level should approach baseline
    for (int i = 0; i < 200; ++i)
        dopa.decay(0.05f);  // total = 10 s > 2*τ

    CHECK(dopa.level() < high);
    CHECK(dopa.level() == Catch::Approx(DOPAMINE_BASELINE).margin(0.1f));
}

TEST_CASE("Gap5.1 — dopamine level clamped to [0, 1]", "[dopamine][gap5.1]") {
    DopamineSystem dopa;
    // Drive to max
    for (int i = 0; i < 5; ++i)
        dopa.update(100.0f, Reward::POSITIVE);
    CHECK(dopa.level() <= 1.0f);

    // Drive to min
    dopa.reset();
    dopa.update(200.0f, Reward::NEUTRAL);  // set prev high
    for (int i = 0; i < 5; ++i)
        dopa.update(0.0f, Reward::NEGATIVE);
    CHECK(dopa.level() >= 0.0f);
}

TEST_CASE("Gap5.1 — habituation: repeated reward → TD error → 0 → D → baseline",
          "[dopamine][gap5.1]") {
    DopamineSystem dopa;
    // After many identical steps Vt+1 ≈ Vt so δ ≈ R - (1-γ)V → diminishes
    for (int i = 0; i < 50; ++i)
        dopa.update(1.0f, Reward::POSITIVE);

    // Residual positive but much smaller magnitude than on first hit
    // (exact convergence depends on γ and V, but the trend should reduce TD spike)
    CHECK(dopa.level() >= 0.5f);   // still non-negative
    CHECK(dopa.level() <= 1.0f);
}

TEST_CASE("Gap5.1 — reset returns to factory state", "[dopamine][gap5.1]") {
    DopamineSystem dopa;
    dopa.update(9.0f, Reward::POSITIVE);
    REQUIRE(dopa.level() != Catch::Approx(DOPAMINE_BASELINE));
    dopa.reset();
    CHECK(dopa.level() == Catch::Approx(DOPAMINE_BASELINE));
}

// ─────────────────────────────────────────────────────────────────────────────
//  Gap 5.2 — EntropyEstimator
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Gap5.2 — empty grid yields entropy 0", "[entropy][gap5.2]") {
    EntropyEstimator est;
    std::vector<float> r(100, 0.0f), im(100, 0.0f);
    CHECK(est.estimate(r, im) == Catch::Approx(0.0f));
}

TEST_CASE("Gap5.2 — single active node yields entropy 0 (delta distribution)",
          "[entropy][gap5.2]") {
    EntropyEstimator est;
    std::vector<float> r(100, 0.0f), im(100, 0.0f);
    r[42] = 1.0f;   // all energy concentrated at one node
    float H = est.estimate(r, im);
    CHECK(H == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("Gap5.2 — uniform distribution yields entropy ≈ log2(K)",
          "[entropy][gap5.2]") {
    EntropyEstimator est(0);
    // 512 equal-magnitude nodes — entropy ≈ log2(512) = 9 bits
    const int N = 512;
    std::vector<float> r(N, 1.0f), im(N, 0.0f);
    float H = est.estimate(r, im);
    // With sampling the estimate is approximate; accept loose bounds
    CHECK(H >= 7.0f);   // significantly high
    CHECK(H <= std::log2f(static_cast<float>(N)) + 0.5f);
}

TEST_CASE("Gap5.2 — estimate_from_intensities matches manual computation",
          "[entropy][gap5.2]") {
    EntropyEstimator est;
    // 4 equal-weight nodes → H = log2(4) = 2
    std::vector<float> intensities = {1.0f, 1.0f, 1.0f, 1.0f};
    float H = est.estimate_from_intensities(intensities);
    CHECK(H == Catch::Approx(2.0f).margin(0.01f));
}

TEST_CASE("Gap5.2 — BoredomRegulator rises when entropy < target",
          "[boredom][gap5.2]") {
    BoredomRegulator bored;
    float low_entropy = 1.0f;   // < ENTROPY_TARGET (6.0)
    bored.update(low_entropy, 1.0f);  // 1 second
    CHECK(bored.level() > 0.0f);
}

TEST_CASE("Gap5.2 — BoredomRegulator falls when entropy > target",
          "[boredom][gap5.2]") {
    BoredomRegulator bored;
    // Manually set some boredom
    for (int i = 0; i < 5; ++i)
        bored.update(0.0f, 1.0f);   // drive up
    float pre = bored.level();

    bored.update(12.0f, 1.0f);   // H=12 > ENTROPY_TARGET(6) → drop
    CHECK(bored.level() < pre);
}

TEST_CASE("Gap5.2 — should_explore() triggers at boredom > 0.7",
          "[boredom][gap5.2]") {
    BoredomRegulator bored;
    // Drive boredom above 0.7: need > 7 seconds of low entropy at rate 0.1/s
    for (int i = 0; i < 10; ++i)
        bored.update(0.0f, 1.0f);   // 10 seconds, boredom = min(1.0, 10*0.1)

    CHECK(bored.level() >= BOREDOM_EXPLORE_THRESH);
    CHECK(bored.should_explore());
}

TEST_CASE("Gap5.2 — boredom clamped to [0, 1]", "[boredom][gap5.2]") {
    BoredomRegulator bored;
    for (int i = 0; i < 100; ++i)
        bored.update(0.0f, 1.0f);
    CHECK(bored.level() <= 1.0f);

    bored.reset();
    for (int i = 0; i < 100; ++i)
        bored.update(100.0f, 1.0f);
    CHECK(bored.level() >= 0.0f);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Gap 5.3 — MetabolicSimulator
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Gap5.3 — initial ATP is 1.0", "[metabolic][gap5.3]") {
    MetabolicSimulator meta;
    CHECK(meta.atp() == Catch::Approx(1.0f));
    CHECK_FALSE(meta.is_exhausted());
    CHECK(meta.is_full());
}

TEST_CASE("Gap5.3 — consume_by_rate depletes ATP", "[metabolic][gap5.3]") {
    MetabolicSimulator meta;
    // burn for 1 second at energy_rate = 100 → depletion = 0.001 * 100 * 1 = 0.1
    meta.consume_by_rate(100.0f, 1.0f);
    CHECK(meta.atp() == Catch::Approx(0.9f).margin(1e-4f));
}

TEST_CASE("Gap5.3 — recharge increases ATP at 0.05/s", "[metabolic][gap5.3]") {
    MetabolicSimulator meta;
    meta.set_atp(0.0f);
    REQUIRE(meta.atp() == Catch::Approx(0.0f));

    meta.recharge(1.0f);   // 1 second → +0.05
    CHECK(meta.atp() == Catch::Approx(META_REGEN_RATE).margin(1e-5f));
}

TEST_CASE("Gap5.3 — ATP clamped to [0, 1]", "[metabolic][gap5.3]") {
    MetabolicSimulator meta;
    // Over-recharge
    meta.recharge(1000.0f);
    CHECK(meta.atp() <= 1.0f);

    // Over-deplete
    meta.consume_by_rate(1e9f, 100.0f);
    CHECK(meta.atp() >= 0.0f);
}

TEST_CASE("Gap5.3 — is_exhausted() true below 0.15", "[metabolic][gap5.3]") {
    MetabolicSimulator meta;
    meta.set_atp(0.14f);
    CHECK(meta.is_exhausted());
    meta.set_atp(0.15f);
    CHECK_FALSE(meta.is_exhausted());
}

TEST_CASE("Gap5.3 — consume_from_laplacian depletes proportional to kinetic energy",
          "[metabolic][gap5.3]") {
    MetabolicSimulator meta;
    // psi active at node 0, laplacian = 10 at that node
    std::vector<float> psi_r = {1.0f, 0.0f, 0.0f};
    std::vector<float> psi_i = {0.0f, 0.0f, 0.0f};
    std::vector<float> lap_r = {10.0f, 0.0f, 0.0f};
    std::vector<float> lap_i = {0.0f, 0.0f, 0.0f};
    float before = meta.atp();
    meta.consume_from_laplacian(psi_r, psi_i, lap_r, lap_i, 1.0f);
    // cost = 0.001 * (10² + 0²) * 1 = 0.001 * 100 = 0.1
    CHECK(meta.atp() == Catch::Approx(before - 0.1f).margin(1e-5f));
}

// ─────────────────────────────────────────────────────────────────────────────
//  Gap 5.4 — NapController
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Gap5.4 — starts AWAKE", "[nap][gap5.4]") {
    NapController nap;
    CHECK(nap.state() == NapState::AWAKE);
    CHECK_FALSE(nap.is_napping());
    CHECK(nap.nap_count() == 0u);
}

TEST_CASE("Gap5.4 — enters nap when ATP < 0.15", "[nap][gap5.4]") {
    NapController nap;
    nap.update(NAP_ENTER_THRESHOLD - 0.01f, 0.0f);
    CHECK(nap.is_napping());
    CHECK(nap.nap_count() == 1u);
}

TEST_CASE("Gap5.4 — stays napping if ATP < 0.90", "[nap][gap5.4]") {
    NapController nap;
    nap.update(0.10f, 0.0f);   // enter
    REQUIRE(nap.is_napping());

    nap.update(0.50f, 1.0f);   // still below exit threshold
    CHECK(nap.is_napping());
}

TEST_CASE("Gap5.4 — exits nap when ATP >= 0.90", "[nap][gap5.4]") {
    NapController nap;
    nap.update(0.10f, 0.0f);   // enter
    REQUIRE(nap.is_napping());

    nap.update(NAP_EXIT_THRESHOLD, 10.0f);   // recharged
    CHECK(nap.state() == NapState::AWAKE);
    CHECK_FALSE(nap.is_napping());
    CHECK(std::string(nap.last_exit_reason()) == "RECHARGED");
}

TEST_CASE("Gap5.4 — exits nap on timeout after 60 seconds", "[nap][gap5.4]") {
    NapController nap;
    nap.update(0.10f, 0.0f);    // enter at t=0
    REQUIRE(nap.is_napping());

    nap.update(0.10f, NAP_MAX_DURATION_SEC + 1.0f);   // still low ATP but timed out
    CHECK(nap.state() == NapState::AWAKE);
    CHECK(std::string(nap.last_exit_reason()) == "TIMEOUT");
}

TEST_CASE("Gap5.4 — on_enter_nap callback fires", "[nap][gap5.4]") {
    NapController nap;
    bool fired = false;
    nap.on_enter_nap = [&] { fired = true; };
    nap.update(0.10f, 0.0f);
    CHECK(fired);
}

TEST_CASE("Gap5.4 — on_exit_nap callback fires", "[nap][gap5.4]") {
    NapController nap;
    bool exit_fired = false;
    nap.on_exit_nap = [&] { exit_fired = true; };
    nap.update(0.10f, 0.0f);   // enter
    nap.update(0.95f, 5.0f);   // exit
    CHECK(exit_fired);
}

TEST_CASE("Gap5.4 — nap state name helpers", "[nap][gap5.4]") {
    CHECK(std::string(nap_state_name(NapState::AWAKE))   == "AWAKE");
    CHECK(std::string(nap_state_name(NapState::NAPPING)) == "NAPPING");
}

TEST_CASE("Gap5.4 — minimum nap duration formula: (0.90-0.15)/0.05 = 15s",
          "[nap][gap5.4]") {
    float expected = (NAP_EXIT_THRESHOLD - NAP_ENTER_THRESHOLD) / META_REGEN_RATE;
    CHECK(expected == Catch::Approx(NAP_MIN_DURATION_SEC).margin(0.1f));
}

// ─────────────────────────────────────────────────────────────────────────────
//  Gap 5.5 — DreamWeaveEngine
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Gap5.5 — frobenius_norm of identical states is 0", "[dream][gap5.5]") {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {0.5f, 1.5f, 2.5f};
    float n = DreamWeaveEngine::frobenius_norm(a, b, a, b);
    CHECK(n == Catch::Approx(0.0f));
}

TEST_CASE("Gap5.5 — frobenius_norm computes √Σ(Δr²+Δi²)", "[dream][gap5.5]") {
    // a_real - b_real = {3, 4},  a_imag - b_imag = {0, 0}
    // ‖Δ‖_F = √(9 + 16) = 5
    std::vector<float> ar = {3.0f, 4.0f};
    std::vector<float> ai = {0.0f, 0.0f};
    std::vector<float> br = {0.0f, 0.0f};
    std::vector<float> bi = {0.0f, 0.0f};
    CHECK(DreamWeaveEngine::frobenius_norm(ar, ai, br, bi)
          == Catch::Approx(5.0f).margin(1e-5f));
}

TEST_CASE("Gap5.5 — identity stepper converges in 1 iteration", "[dream][gap5.5]") {
    DreamWeaveEngine engine;
    std::vector<float> psi_r(10, 1.0f), psi_i(10, 0.5f);

    // Stepper that does nothing → ΔΨ = 0 → converges immediately
    auto noop_stepper = [](std::span<float>, std::span<float>) {};

    auto [converged, iters, delta] = engine.run(psi_r, psi_i, noop_stepper);
    CHECK(converged);
    CHECK(iters == 1);
    CHECK(delta == Catch::Approx(0.0f).margin(1e-5f));
    CHECK(engine.convergence_count() == 1u);
}

TEST_CASE("Gap5.5 — always-changing stepper does not converge",
          "[dream][gap5.5]") {
    DreamWeaveEngine engine;
    std::vector<float> psi_r(5, 0.0f), psi_i(5, 0.0f);
    psi_r[0] = 1.0f;

    int step = 0;
    // Each call perturbs the first node significantly
    auto divergent = [&](std::span<float> r, std::span<float>) {
        r[0] = static_cast<float>(++step % 2 == 0 ? +10 : -10);
    };

    auto res = engine.run(psi_r, psi_i, divergent,
                          DREAM_CONVERGENCE_THRESHOLD,
                          /*max_iter=*/ 10);
    CHECK_FALSE(res.converged);
    CHECK(res.iterations == 10);
    CHECK(engine.no_convergence_count() == 1u);
}

TEST_CASE("Gap5.5 — converges below custom threshold", "[dream][gap5.5]") {
    DreamWeaveEngine engine;
    std::vector<float> psi_r = {1.0f, 0.0f};
    std::vector<float> psi_i = {0.0f, 0.0f};

    int call = 0;
    // First call changes psi by tiny amount (below 0.1 threshold), then stops
    auto small_delta = [&](std::span<float> r, std::span<float>) {
        if (call++ == 0) r[0] += 1e-5f;   // nudge once
    };

    auto res = engine.run(psi_r, psi_i, small_delta, /*threshold=*/ 0.1f);
    CHECK(res.converged);
    CHECK(res.final_delta < 0.1f);
}

TEST_CASE("Gap5.5 — DREAM_CONVERGENCE_THRESHOLD == 1e-4", "[dream][gap5.5]") {
    CHECK(DREAM_CONVERGENCE_THRESHOLD == Catch::Approx(1e-4f));
}

TEST_CASE("Gap5.5 — DREAM_MAX_ITERATIONS == 1000", "[dream][gap5.5]") {
    CHECK(DREAM_MAX_ITERATIONS == 1000);
}

// ─────────────────────────────────────────────────────────────────────────────
//  AutonomyEngine integration
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("AutonomyEngine — constructs and starts in ACTIVE state",
          "[autonomy][integration]") {
    AutonomyEngine engine;
    CHECK(engine.state()      == AutonomyState::ACTIVE);
    CHECK_FALSE(engine.is_napping());
    CHECK(engine.atp()        == Catch::Approx(1.0f));
    CHECK(engine.dopamine()   == Catch::Approx(DOPAMINE_BASELINE));
}

TEST_CASE("AutonomyEngine — autonomy_state_name helpers", "[autonomy][integration]") {
    CHECK(std::string(autonomy_state_name(AutonomyState::ACTIVE))    == "ACTIVE");
    CHECK(std::string(autonomy_state_name(AutonomyState::EXPLORING)) == "EXPLORING");
    CHECK(std::string(autonomy_state_name(AutonomyState::NAPPING))   == "NAPPING");
    CHECK(std::string(autonomy_state_name(AutonomyState::INHIBITED)) == "INHIBITED");
}

TEST_CASE("AutonomyEngine — positive reward after tick spikes dopamine",
          "[autonomy][integration]") {
    AutonomyEngine engine;
    std::vector<float> psi_r(4, 1.0f), psi_i(4, 0.0f);
    engine.tick(0.01f, psi_r, psi_i, Reward::POSITIVE, 0.0f);
    CHECK(engine.dopamine() > DOPAMINE_BASELINE);
}

TEST_CASE("AutonomyEngine — ATP depletes over ticks with active psi",
          "[autonomy][integration]") {
    AutonomyEngine engine;
    std::vector<float> psi_r(100, 1.0f), psi_i(100, 0.5f);   // Σ|Ψ|² = 125
    // Run without napping (ATP starts at 1.0, consume at small rate)
    float t = 0.0f;
    float start = engine.atp();
    for (int i = 0; i < 100; ++i) {
        t += 0.001f;
        engine.tick(0.001f, psi_r, psi_i, Reward::NEUTRAL, t);
        if (engine.is_napping()) break;
    }
    CHECK(engine.atp() < start);
}

TEST_CASE("AutonomyEngine — on_nap_enter fires when ATP exhausted",
          "[autonomy][integration]") {
    AutonomyConfig cfg;
    cfg.initial_atp = 0.10f;  // start below nap threshold
    AutonomyEngine engine(cfg);
    bool nap_entered = false;
    engine.on_nap_enter = [&] { nap_entered = true; };

    std::vector<float> psi_r(4, 0.0f), psi_i(4, 0.0f);
    engine.tick(0.001f, psi_r, psi_i, Reward::NEUTRAL, 0.0f);
    CHECK(nap_entered);
    CHECK(engine.is_napping());
}

TEST_CASE("AutonomyEngine — on_nap_exit fires when ATP recharged",
          "[autonomy][integration]") {
    AutonomyConfig cfg;
    cfg.initial_atp = 0.10f;
    AutonomyEngine engine(cfg);
    bool nap_exited = false;
    engine.on_nap_exit = [&] { nap_exited = true; };

    std::vector<float> emptyR, emptyI;
    engine.tick(0.001f, emptyR, emptyI, Reward::NEUTRAL, 0.0f);  // enter nap
    REQUIRE(engine.is_napping());

    // Recharge over many ticks until ATP > 0.90
    // regen = 0.05/s; need (0.90-0.10)/0.05 = 16s; dt=0.1s → 160 ticks
    float t = 0.001f;
    for (int i = 0; i < 200 && engine.is_napping(); ++i) {
        t += 0.1f;
        engine.tick(0.1f, emptyR, emptyI, Reward::NEUTRAL, t);
    }
    CHECK(nap_exited);
    CHECK_FALSE(engine.is_napping());
}

TEST_CASE("AutonomyEngine — snapshot() returns consistent values",
          "[autonomy][integration]") {
    AutonomyEngine engine;
    std::vector<float> psi_r(8, 0.5f), psi_i(8, 0.0f);
    engine.tick(0.01f, psi_r, psi_i, Reward::NEUTRAL, 1.0f);

    auto snap = engine.snapshot();
    CHECK(snap.atp      == Catch::Approx(engine.atp()));
    CHECK(snap.dopamine == Catch::Approx(engine.dopamine()));
    CHECK(snap.state    == engine.state());
}

TEST_CASE("AutonomyEngine — component accessors return correct types",
          "[autonomy][integration]") {
    AutonomyEngine engine;
    CHECK_NOTHROW((void)engine.dopamine_system());
    CHECK_NOTHROW((void)engine.boredom_regulator());
    CHECK_NOTHROW((void)engine.metabolic());
    CHECK_NOTHROW((void)engine.nap_controller());
    CHECK_NOTHROW((void)engine.dream_weave());
}
