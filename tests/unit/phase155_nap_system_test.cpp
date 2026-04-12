/**
 * @file phase155_nap_system_test.cpp
 * @brief v0.1.18 — NAP System & Memory Consolidation test suite.
 *
 * §1  Constants valid
 * §2  Default orchestrator state
 * §3  Full NAP lifecycle (entry → recharge → exit)
 * §4  NAP timeout at 60 s
 * §5  Consolidation callback invoked during NAP
 * §6  Dream-weave convergence during NAP
 * §7  Dream engine cycle during NAP
 * §8  Checkpoint on entry
 * §9  Dream isolation (waking psi unchanged)
 * §10 Z-score normalization
 * §11 NapCycleReport populated
 * §12 Multiple NAP cycles
 * §13 IdentityProfile defaults
 * §14 Preference learning
 * §15 Memory recording with FIFO cap
 * §16 JSON save/load round-trip
 * §17 Identity stable across simulated NAP
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <nikola/autonomy/nap_orchestrator.hpp>
#include <nikola/interior/identity_manager.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using namespace nikola::autonomy;
using namespace nikola::interior;
using Catch::Matchers::WithinAbs;

// ============================================================================
// RAII temp directory helper
// ============================================================================

struct TmpDir {
    std::string path;
    TmpDir() {
        char tpl[] = "/tmp/nikola_test_155_XXXXXX";
        char* p = mkdtemp(tpl);
        REQUIRE(p != nullptr);
        path = p;
    }
    ~TmpDir() { std::filesystem::remove_all(path); }
    TmpDir(const TmpDir&) = delete;
    TmpDir& operator=(const TmpDir&) = delete;
};

// ============================================================================
// Helper: run a full NAP cycle with dt=1s steps
//
// Returns the number of ticks it took for the nap to end.
// ============================================================================

static int run_nap_cycle(NapOrchestrator& orch, float& atp, float& time) {
    // First update triggers nap entry
    float dt = 1.0f;
    orch.update(atp, time, dt);
    REQUIRE(orch.is_napping());

    int ticks = 0;
    while (orch.is_napping() && ticks < 200) {
        time += dt;
        orch.update(atp, time, dt);
        ++ticks;
    }
    return ticks;
}

// ============================================================================
// Converging stepper for DreamWeaveEngine: decays amplitude by 5 % per step,
// guaranteeing Frobenius convergence within a few hundred iterations.
// ============================================================================

static const auto CONVERGING_STEPPER = [](std::span<float> r, std::span<float> i) {
    for (auto& v : r) v *= 0.95f;
    for (auto& v : i) v *= 0.95f;
};

// ============================================================================
// §1 — Constants
// ============================================================================

TEST_CASE("§1  v0.1.18 constants valid", "[phase155][nap]") {
    // NapController thresholds
    REQUIRE(NAP_ENTER_THRESHOLD == 0.15f);
    REQUIRE(NAP_EXIT_THRESHOLD  == 0.90f);
    REQUIRE(NAP_MAX_DURATION_SEC == 60.0f);

    // NapOrchestratorConfig defaults
    NapOrchestratorConfig cfg;
    REQUIRE(cfg.atp_recharge_rate    == 0.05f);
    REQUIRE(cfg.norepinephrine_level == 0.3f);
    REQUIRE(cfg.checkpoint_on_entry  == true);
    REQUIRE(cfg.run_dream_weave      == true);
    REQUIRE(cfg.run_dream_engine     == true);
    REQUIRE(cfg.run_consolidation    == true);

    // IdentityManager constants
    REQUIRE(IDENTITY_MAX_MEMORIES    == 1000);
    REQUIRE(IDENTITY_PREFERENCE_LEARN == 0.1);
}

// ============================================================================
// §2 — Default orchestrator state
// ============================================================================

TEST_CASE("§2  Default orchestrator state", "[phase155][nap]") {
    NapOrchestrator orch;
    REQUIRE(orch.state()      == NapState::AWAKE);
    REQUIRE(orch.is_napping() == false);
    REQUIRE(orch.nap_count()  == 0);
    REQUIRE(orch.last_report().duration_s == 0.f);
}

// ============================================================================
// §3 — Full NAP lifecycle (entry → recharge → exit)
// ============================================================================

TEST_CASE("§3  Full NAP lifecycle", "[phase155][nap]") {
    NapOrchestrator orch;
    float atp  = 0.10f;   // below NAP_ENTER_THRESHOLD
    float time = 0.0f;
    float dt   = 1.0f;

    // First update should trigger NAP entry
    orch.update(atp, time, dt);
    REQUIRE(orch.is_napping());
    REQUIRE(orch.nap_count() == 1);

    // Tick until exit (recharge 0.05/s → need 16 ticks to reach 0.90)
    int ticks = 0;
    while (orch.is_napping() && ticks < 200) {
        time += dt;
        orch.update(atp, time, dt);
        ++ticks;
    }

    REQUIRE_FALSE(orch.is_napping());
    REQUIRE(atp >= NAP_EXIT_THRESHOLD);

    // Report
    const auto& report = orch.last_report();
    REQUIRE_THAT(report.atp_at_entry, WithinAbs(0.10, 0.01));
    REQUIRE(report.atp_at_exit >= NAP_EXIT_THRESHOLD);
    REQUIRE(report.duration_s > 0.f);
    REQUIRE(std::string(report.exit_reason) == "RECHARGED");
}

// ============================================================================
// §4 — NAP timeout
// ============================================================================

TEST_CASE("§4  NAP timeout at 60 s", "[phase155][nap]") {
    // Tiny recharge rate so ATP never reaches exit threshold before 60 s
    NapOrchestratorConfig cfg;
    cfg.atp_recharge_rate = 0.001f;   // 0.001/s → after 60 s: 0.10 + 0.06 = 0.16
    NapOrchestrator orch(cfg);

    float atp  = 0.10f;
    float time = 0.0f;
    float dt   = 1.0f;

    orch.update(atp, time, dt);   // enter nap
    REQUIRE(orch.is_napping());

    int ticks = 0;
    while (orch.is_napping() && ticks < 200) {
        time += dt;
        orch.update(atp, time, dt);
        ++ticks;
    }

    REQUIRE_FALSE(orch.is_napping());
    REQUIRE(std::string(orch.last_report().exit_reason) == "TIMEOUT");
    REQUIRE_THAT(orch.last_report().duration_s,
                 WithinAbs(NAP_MAX_DURATION_SEC, 1.5));
}

// ============================================================================
// §5 — Consolidation callback invoked
// ============================================================================

TEST_CASE("§5  Consolidation during NAP", "[phase155][nap]") {
    int consolidation_calls = 0;
    NapOrchestrator orch;
    orch.set_consolidation_fn([&]() -> ConsolidationResult {
        ++consolidation_calls;
        return {3, 2};   // 3 pruned, 2 replayed
    });

    float atp  = 0.10f;
    float time = 0.0f;

    int ticks = run_nap_cycle(orch, atp, time);
    (void)ticks;

    // Consolidation should have fired exactly once
    REQUIRE(consolidation_calls == 1);
    REQUIRE(orch.last_report().memories_pruned  == 3);
    REQUIRE(orch.last_report().memories_replayed == 2);
}

// ============================================================================
// §6 — Dream-weave convergence
// ============================================================================

TEST_CASE("§6  Dream-weave during NAP", "[phase155][nap]") {
    DreamWeaveEngine dw;
    NapOrchestrator orch;
    orch.set_dream_weave(&dw);
    orch.set_dream_stepper(CONVERGING_STEPPER);

    // Non-trivial waking psi
    std::vector<float> psi_r = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> psi_i = {0.1f, 0.2f, 0.3f, 0.4f};
    orch.init_dream_buffers(psi_r, psi_i);

    float atp  = 0.10f;
    float time = 0.0f;
    run_nap_cycle(orch, atp, time);

    const auto& report = orch.last_report();
    REQUIRE(report.dream_converged);
    REQUIRE(report.dream_iterations > 0);
    REQUIRE(report.dream_final_delta < DREAM_CONVERGENCE_THRESHOLD);
}

// ============================================================================
// §7 — Dream engine cycle
// ============================================================================

TEST_CASE("§7  Dream engine cycle during NAP", "[phase155][nap]") {
    DreamEngine de;

    // Record a few waking experiences before nap
    NikolaState s1;
    s1.dopamine = 0.8f;  s1.atp = 0.5f;  s1.entropy = 0.3f;
    s1.boredom  = 0.9f;  // above DREAM_IDLE_THRESHOLD (0.60)
    de.record_experience("high_reward", s1, 1.0f);

    NikolaState s2;
    s2.dopamine = 0.7f;  s2.atp = 0.4f;  s2.entropy = 0.3f;
    s2.boredom  = 0.9f;
    de.record_experience("moderate_reward", s2, 0.5f);

    NapOrchestrator orch;
    orch.set_dream_engine(&de);

    float atp  = 0.10f;
    float time = 0.0f;
    run_nap_cycle(orch, atp, time);

    // Dream engine should have run a cycle
    // With 2 similar experiences, fragments may or may not form depending
    // on similarity threshold, but the cycle should have executed.
    REQUIRE(de.dream_log().size() >= 1);
}

// ============================================================================
// §8 — Checkpoint on entry
// ============================================================================

TEST_CASE("§8  Checkpoint on entry", "[phase155][nap]") {
    bool checkpoint_called = false;
    NapOrchestrator orch;
    orch.set_checkpoint_fn([&]() -> bool {
        checkpoint_called = true;
        return true;
    });

    float atp  = 0.10f;
    float time = 0.0f;
    float dt   = 1.0f;

    orch.update(atp, time, dt);   // triggers entry
    REQUIRE(checkpoint_called);
    REQUIRE(orch.is_napping());

    // Let it finish
    while (orch.is_napping()) {
        time += dt;
        orch.update(atp, time, dt);
    }
    REQUIRE(orch.last_report().checkpointed);
}

// ============================================================================
// §9 — Dream isolation: waking psi unchanged
// ============================================================================

TEST_CASE("§9  Dream isolation", "[phase155][nap]") {
    DreamWeaveEngine dw;
    NapOrchestrator orch;
    orch.set_dream_weave(&dw);
    orch.set_dream_stepper(CONVERGING_STEPPER);

    std::vector<float> waking_r = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> waking_i = {0.1f, 0.2f, 0.3f, 0.4f};
    auto original_r = waking_r;
    auto original_i = waking_i;

    orch.init_dream_buffers(waking_r, waking_i);

    float atp  = 0.10f;
    float time = 0.0f;
    run_nap_cycle(orch, atp, time);

    // Waking buffers must be unchanged (dream-weave operated on copies)
    REQUIRE(waking_r == original_r);
    REQUIRE(waking_i == original_i);

    // Dream buffers should have been modified (z-normalized then stepped)
    REQUIRE(orch.dream_psi_real() != original_r);
}

// ============================================================================
// §10 — Z-score normalization
// ============================================================================

TEST_CASE("§10 Z-score normalization", "[phase155][nap]") {
    std::vector<float> data = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f};
    NapOrchestrator::z_normalize(data);

    // Mean should be 0 ± tolerance
    float sum = 0.f;
    for (float v : data) sum += v;
    REQUIRE_THAT(sum / 5.f, WithinAbs(0.0, 1e-5));

    // Variance should be 1 ± tolerance
    float sq_sum = 0.f;
    for (float v : data) sq_sum += v * v;
    float var = sq_sum / 5.f;
    REQUIRE_THAT(var, WithinAbs(1.0, 1e-4));
}

// ============================================================================
// §11 — NapCycleReport
// ============================================================================

TEST_CASE("§11 NapCycleReport populated", "[phase155][nap]") {
    DreamWeaveEngine dw;
    NapOrchestrator orch;
    orch.set_dream_weave(&dw);
    orch.set_dream_stepper(CONVERGING_STEPPER);
    orch.set_consolidation_fn([]() -> ConsolidationResult { return {5, 3}; });
    orch.set_checkpoint_fn([]() -> bool { return true; });

    std::vector<float> psi_r = {1.f, 2.f, 3.f};
    std::vector<float> psi_i = {0.1f, 0.2f, 0.3f};
    orch.init_dream_buffers(psi_r, psi_i);

    float atp  = 0.10f;
    float time = 0.0f;
    run_nap_cycle(orch, atp, time);

    const auto& r = orch.last_report();
    REQUIRE(r.duration_s > 0.f);
    REQUIRE_THAT(r.atp_at_entry, WithinAbs(0.10, 0.01));
    REQUIRE(r.atp_at_exit >= NAP_EXIT_THRESHOLD);
    REQUIRE(r.checkpointed);
    REQUIRE(r.memories_pruned  == 5);
    REQUIRE(r.memories_replayed == 3);
    REQUIRE(r.dream_converged);
    REQUIRE(r.dream_iterations > 0);
    REQUIRE(r.dream_final_delta < DREAM_CONVERGENCE_THRESHOLD);
    REQUIRE(std::string(r.exit_reason) == "RECHARGED");
}

// ============================================================================
// §12 — Multiple NAP cycles
// ============================================================================

TEST_CASE("§12 Multiple NAP cycles", "[phase155][nap]") {
    NapOrchestrator orch;

    for (int cycle = 1; cycle <= 3; ++cycle) {
        float atp  = 0.10f;
        float time = static_cast<float>(cycle * 100);
        run_nap_cycle(orch, atp, time);
        REQUIRE(orch.nap_count() == static_cast<uint32_t>(cycle));
    }

    REQUIRE(orch.nap_count() == 3);
}

// ============================================================================
// §13 — IdentityProfile defaults
// ============================================================================

TEST_CASE("§13 IdentityProfile defaults", "[phase155][identity]") {
    IdentityProfile p;
    REQUIRE(p.name == "Nikola");
    REQUIRE(p.preferences.empty());
    REQUIRE(p.memories.empty());
    REQUIRE(p.topic_counts.empty());
}

// ============================================================================
// §14 — Preference learning
// ============================================================================

TEST_CASE("§14 Preference learning", "[phase155][identity]") {
    TmpDir tmp;
    IdentityManager mgr(tmp.path);

    mgr.update_preference("physics", +1.0);   // +0.1
    mgr.update_preference("physics", +1.0);   // +0.1 → 0.2
    mgr.update_preference("art",     -1.0);   // -0.1

    REQUIRE_THAT(mgr.profile().preferences.at("physics"),
                 WithinAbs(0.2, 1e-9));
    REQUIRE_THAT(mgr.profile().preferences.at("art"),
                 WithinAbs(-0.1, 1e-9));
}

// ============================================================================
// §15 — Memory recording with FIFO cap
// ============================================================================

TEST_CASE("§15 Memory recording FIFO cap", "[phase155][identity]") {
    TmpDir tmp;
    IdentityManager mgr(tmp.path);

    // Fill to max
    for (size_t i = 0; i < IDENTITY_MAX_MEMORIES; ++i)
        mgr.record_memory("event_" + std::to_string(i));

    REQUIRE(mgr.profile().memories.size() == IDENTITY_MAX_MEMORIES);

    // Adding one more evicts the oldest
    mgr.record_memory("overflow");
    REQUIRE(mgr.profile().memories.size() == IDENTITY_MAX_MEMORIES);
    REQUIRE(mgr.profile().memories.front() == "event_1");
    REQUIRE(mgr.profile().memories.back()  == "overflow");
}

// ============================================================================
// §16 — JSON save/load round-trip
// ============================================================================

TEST_CASE("§16 JSON round-trip", "[phase155][identity]") {
    TmpDir tmp;

    // Create and populate a profile
    {
        IdentityManager mgr(tmp.path);
        mgr.profile().name = "TestAgent";
        mgr.update_preference("math",    +2.0);   // 0.2
        mgr.update_preference("cooking", -1.0);   // -0.1
        mgr.record_memory("learned calculus");
        mgr.record_memory("baked a cake");
        mgr.increment_topic("science");
        mgr.increment_topic("science");
        mgr.increment_topic("music");
        REQUIRE(mgr.save());
    }

    // Load in a fresh manager
    {
        IdentityManager mgr(tmp.path);
        REQUIRE(mgr.load());

        const auto& p = mgr.profile();
        REQUIRE(p.name == "TestAgent");
        REQUIRE_THAT(p.preferences.at("math"),    WithinAbs(0.2,  1e-6));
        REQUIRE_THAT(p.preferences.at("cooking"), WithinAbs(-0.1, 1e-6));
        REQUIRE(p.memories.size() == 2);
        REQUIRE(p.memories[0] == "learned calculus");
        REQUIRE(p.memories[1] == "baked a cake");
        REQUIRE(p.topic_counts.at("science") == 2);
        REQUIRE(p.topic_counts.at("music")   == 1);
    }
}

// ============================================================================
// §17 — Identity stable across simulated NAP
// ============================================================================

TEST_CASE("§17 Identity stable across NAP", "[phase155][identity]") {
    TmpDir tmp;
    IdentityManager mgr(tmp.path);

    mgr.profile().name = "Nikola";
    mgr.update_preference("physics", +3.0);      // 0.3
    mgr.record_memory("discovered resonance");
    mgr.increment_topic("quantum");
    REQUIRE(mgr.save());

    // Simulate a NAP cycle (orchestrator doesn't touch identity)
    NapOrchestrator orch;
    float atp  = 0.10f;
    float time = 0.0f;
    run_nap_cycle(orch, atp, time);

    // Reload identity
    IdentityManager mgr2(tmp.path);
    REQUIRE(mgr2.load());

    const auto& p = mgr2.profile();
    REQUIRE(p.name == "Nikola");
    REQUIRE_THAT(p.preferences.at("physics"), WithinAbs(0.3, 1e-6));
    REQUIRE(p.memories.size() == 1);
    REQUIRE(p.memories[0] == "discovered resonance");
    REQUIRE(p.topic_counts.at("quantum") == 1);
}
