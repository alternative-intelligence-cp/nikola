/**
 * @file tests/unit/phase113_evolutionary_orchestrator_test.cpp
 * @brief Phase 113 — EvolutionaryOrchestrator SIE loop controller test suite.
 *
 * Exercises the full SIE validation pipeline:
 *   Gate 1 — CodePatternBlacklist (security scan)
 *   Gate 2 — PhysicsOracle (energy conservation + reversibility)
 *   Gate 3 — ModuleSwapper (dlopen hot-swap)
 *
 * Reuses the Phase 112 test plugin (phase112_test_plugin.so) as the candidate
 * shared library.  The plugin directory is injected at compile time via
 * PHASE113_PLUGIN_DIR — set to the same output dir as Phase 112.
 */

#include <nikola/autonomy/evolutionary_orchestrator.hpp>
#include <nikola/autonomy/metabolic_controller.hpp>
#include <nikola/security/code_blacklist.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <filesystem>
#include <optional>
#include <string>
#include <thread>
#include <vector>

// ── Plugin path ───────────────────────────────────────────────────────────────
#ifndef PHASE113_PLUGIN_DIR
#  define PHASE113_PLUGIN_DIR "."
#endif

static const std::string k_plugin =
    std::string(PHASE113_PLUGIN_DIR) + "/phase112_test_plugin.so";

static const std::string k_bad_path =
    "/nonexistent/phase113/missing.so";

// ── Convenience aliases ───────────────────────────────────────────────────────
using nikola::autonomy::EvolutionaryOrchestrator;
using nikola::autonomy::CycleStatus;
using nikola::autonomy::CycleReport;
using nikola::autonomy::PhysicsMeasurement;
using nikola::autonomy::MetabolicController;
using nikola::autonomy::MetabolicExhaustionException;
using nikola::security::CodePatternBlacklist;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Sensible nap threshold for tests — well below working ATP so tests don't
/// accidentally trigger nap-cycle logic.
static constexpr float k_test_nap_threshold = 5.0f;

/// Physics provider that returns perfect measurements (energy conserved,
/// fully reversible).  Always passes Gate 2.
static auto good_physics_provider() {
    return [](void*) -> std::optional<PhysicsMeasurement> {
        return PhysicsMeasurement{
            .H_initial        = 100.0,
            .H_final          = 100.0,   // zero drift
            .reversibility_l2 = 0.0,     // perfect reversibility
            .skip_oracle      = false,
        };
    };
}

/// Physics provider that returns measurements indicating energy drift failure.
static auto bad_energy_provider() {
    return [](void*) -> std::optional<PhysicsMeasurement> {
        return PhysicsMeasurement{
            .H_initial        = 100.0,
            .H_final          = 200.0,   // 100% drift — well above 1e-4 limit
            .reversibility_l2 = 0.0,
            .skip_oracle      = false,
        };
    };
}

/// Physics provider that returns measurements indicating reversibility failure.
static auto bad_reversibility_provider() {
    return [](void*) -> std::optional<PhysicsMeasurement> {
        return PhysicsMeasurement{
            .H_initial        = 100.0,
            .H_final          = 100.0,
            .reversibility_l2 = 1.0,     // way above 1e-6 limit
            .skip_oracle      = false,
        };
    };
}

/// Physics provider that returns nullopt (measurement failed).
static auto null_physics_provider() {
    return [](void*) -> std::optional<PhysicsMeasurement> {
        return std::nullopt;
    };
}

/// Physics provider that requests oracle skip.
static auto skip_oracle_provider() {
    return [](void*) -> std::optional<PhysicsMeasurement> {
        return PhysicsMeasurement{.skip_oracle = true};
    };
}

/// Confirm plugin exists before any test that needs it.
static void require_plugin() {
    REQUIRE(std::filesystem::exists(k_plugin));
}

// =============================================================================
// Section 1 — Construction & initial state
// =============================================================================

TEST_CASE("EO — default-constructed state", "[phase113][eo]") {
    MetabolicController  ctrl{10'000.0f, k_test_nap_threshold};
    CodePatternBlacklist bl;
    EvolutionaryOrchestrator eo{ctrl, bl};

    CHECK_FALSE(eo.has_active());
    CHECK_FALSE(eo.has_previous());
    CHECK(eo.active_factory() == nullptr);
    CHECK(eo.active_path().empty());

    const auto s = eo.stats();
    CHECK(s.total     == 0);
    CHECK(s.succeeded == 0);
}

// =============================================================================
// Section 2 — ATP gating (Gate 0 / MetabolicLock)
// =============================================================================

TEST_CASE("EO — ATP_DENIED when controller has insufficient energy", "[phase113][eo]") {
    MetabolicController  ctrl{1.0f, k_test_nap_threshold};   // 1 ATP — far below 750 needed
    CodePatternBlacklist bl;
    EvolutionaryOrchestrator eo{ctrl, bl};

    CycleReport rep = eo.run_cycle(k_plugin);
    CHECK(rep.status == CycleStatus::ATP_DENIED);
    CHECK_FALSE(eo.has_active());
}

TEST_CASE("EO — ATP stats incremented on ATP_DENIED", "[phase113][eo]") {
    MetabolicController  ctrl{1.0f, k_test_nap_threshold};
    CodePatternBlacklist bl;
    EvolutionaryOrchestrator eo{ctrl, bl};

    (void)eo.run_cycle(k_plugin);
    (void)eo.run_cycle(k_plugin);

    const auto s = eo.stats();
    CHECK(s.total      == 2);
    CHECK(s.atp_denied == 2);
    CHECK(s.succeeded  == 0);
}

// =============================================================================
// Section 3 — Gate 1 (Security)
// =============================================================================

TEST_CASE("EO — Gate 1 skipped when no source code supplied", "[phase113][eo]") {
    require_plugin();
    MetabolicController  ctrl{10'000.0f, k_test_nap_threshold};
    CodePatternBlacklist bl;
    EvolutionaryOrchestrator eo{ctrl, bl};

    CycleReport rep = eo.run_cycle(k_plugin, "" /* empty source */);
    CHECK(rep.gate1_security_passed);
    // Gate 1 skip still allows further gates.
}

TEST_CASE("EO — SECURITY_REJECTED on blacklisted source code", "[phase113][eo]") {
    MetabolicController  ctrl{10'000.0f, k_test_nap_threshold};
    CodePatternBlacklist bl;
    EvolutionaryOrchestrator eo{ctrl, bl};

    // CodePatternBlacklist flags "system(" as a forbidden pattern.
    const std::string dangerous_source =
        "void evil() { system(\"rm -rf /\"); }";

    CycleReport rep = eo.run_cycle(k_plugin, dangerous_source);
    CHECK(rep.status == CycleStatus::SECURITY_REJECTED);
    CHECK_FALSE(rep.gate1_security_passed);
    CHECK_FALSE(eo.has_active());   // must not have loaded anything
}

TEST_CASE("EO — SECURITY_REJECTED increments stats correctly", "[phase113][eo]") {
    MetabolicController  ctrl{10'000.0f, k_test_nap_threshold};
    CodePatternBlacklist bl;
    EvolutionaryOrchestrator eo{ctrl, bl};

    (void)eo.run_cycle(k_plugin, "void safe(){}");        // clean
    (void)eo.run_cycle(k_plugin, "system(\"hack\");");    // dirty
    (void)eo.run_cycle(k_plugin, "system(\"hack\");");    // dirty

    const auto s = eo.stats();
    CHECK(s.total              == 3);
    CHECK(s.security_rejected  == 2);
}

// =============================================================================
// Section 4 — Gate 2 (Physics Oracle)
// =============================================================================

TEST_CASE("EO — Gate 2 skipped when no physics provider supplied", "[phase113][eo]") {
    require_plugin();
    MetabolicController  ctrl{10'000.0f, k_test_nap_threshold};
    CodePatternBlacklist bl;
    EvolutionaryOrchestrator eo{ctrl, bl};

    CycleReport rep = eo.run_cycle(k_plugin);
    CHECK(rep.gate2_physics_passed);
    // Skipping gate 2 means SUCCESS is still reachable.
    CHECK(rep.status == CycleStatus::SUCCESS);
}

TEST_CASE("EO — PHYSICS_REJECTED on energy drift failure", "[phase113][eo]") {
    require_plugin();
    MetabolicController  ctrl{10'000.0f, k_test_nap_threshold};
    CodePatternBlacklist bl;
    EvolutionaryOrchestrator eo{ctrl, bl};

    CycleReport rep = eo.run_cycle(k_plugin, {}, bad_energy_provider());
    CHECK(rep.status == CycleStatus::PHYSICS_REJECTED);
    CHECK_FALSE(rep.gate2_physics_passed);
    CHECK(rep.energy_drift_ratio > 0.0);
    CHECK_FALSE(eo.has_active());
}

TEST_CASE("EO — PHYSICS_REJECTED on reversibility failure", "[phase113][eo]") {
    require_plugin();
    MetabolicController  ctrl{10'000.0f, k_test_nap_threshold};
    CodePatternBlacklist bl;
    EvolutionaryOrchestrator eo{ctrl, bl};

    CycleReport rep = eo.run_cycle(k_plugin, {}, bad_reversibility_provider());
    CHECK(rep.status == CycleStatus::PHYSICS_REJECTED);
    CHECK_FALSE(rep.gate2_physics_passed);
}

TEST_CASE("EO — PHYSICS_REJECTED when provider returns nullopt", "[phase113][eo]") {
    require_plugin();
    MetabolicController  ctrl{10'000.0f, k_test_nap_threshold};
    CodePatternBlacklist bl;
    EvolutionaryOrchestrator eo{ctrl, bl};

    CycleReport rep = eo.run_cycle(k_plugin, {}, null_physics_provider());
    CHECK(rep.status == CycleStatus::PHYSICS_REJECTED);
    CHECK_FALSE(eo.has_active());
}

TEST_CASE("EO — physics gate passes with skip_oracle flag", "[phase113][eo]") {
    require_plugin();
    MetabolicController  ctrl{10'000.0f, k_test_nap_threshold};
    CodePatternBlacklist bl;
    EvolutionaryOrchestrator eo{ctrl, bl};

    CycleReport rep = eo.run_cycle(k_plugin, {}, skip_oracle_provider());
    CHECK(rep.gate2_physics_passed);
    CHECK(rep.status == CycleStatus::SUCCESS);
}

TEST_CASE("EO — physics stats increment on rejection", "[phase113][eo]") {
    require_plugin();
    MetabolicController  ctrl{10'000.0f, k_test_nap_threshold};
    CodePatternBlacklist bl;
    EvolutionaryOrchestrator eo{ctrl, bl};

    (void)eo.run_cycle(k_plugin, {}, bad_energy_provider());

    const auto s = eo.stats();
    CHECK(s.physics_rejected == 1);
}

// =============================================================================
// Section 5 — Gate 3 (ModuleSwapper) + full success path
// =============================================================================

TEST_CASE("EO — SUCCESS on full happy path (no gates skipped)", "[phase113][eo]") {
    require_plugin();
    MetabolicController  ctrl{10'000.0f, k_test_nap_threshold};
    CodePatternBlacklist bl;
    EvolutionaryOrchestrator eo{ctrl, bl};

    const std::string safe_source = "void* nikola_module_factory() { return nullptr; }";
    CycleReport rep = eo.run_cycle(k_plugin, safe_source, good_physics_provider());

    REQUIRE(rep.status == CycleStatus::SUCCESS);
    CHECK(rep.gate1_security_passed);
    CHECK(rep.gate2_physics_passed);
    CHECK(rep.gate3_swap_passed);
    CHECK(rep.atp_consumed == EvolutionaryOrchestrator::TOTAL_COST);
    CHECK(rep.elapsed_ms > 0.0);
    CHECK(eo.has_active());
    CHECK(eo.active_factory() != nullptr);
    CHECK(eo.active_path() == k_plugin);
}

TEST_CASE("EO — LOAD_FAILED on bad .so path", "[phase113][eo]") {
    MetabolicController  ctrl{10'000.0f, k_test_nap_threshold};
    CodePatternBlacklist bl;
    EvolutionaryOrchestrator eo{ctrl, bl};

    CycleReport rep = eo.run_cycle(k_bad_path);
    CHECK(rep.status == CycleStatus::LOAD_FAILED);
    CHECK_FALSE(eo.has_active());
}

TEST_CASE("EO — SAME_MODULE on duplicate path", "[phase113][eo]") {
    require_plugin();
    MetabolicController  ctrl{10'000.0f, k_test_nap_threshold};
    CodePatternBlacklist bl;
    EvolutionaryOrchestrator eo{ctrl, bl};

    REQUIRE(eo.run_cycle(k_plugin).status == CycleStatus::SUCCESS);
    CycleReport rep = eo.run_cycle(k_plugin);
    CHECK(rep.status == CycleStatus::SAME_MODULE);
}

TEST_CASE("EO — ATP is deducted on SUCCESS", "[phase113][eo]") {
    require_plugin();
    MetabolicController  ctrl{10'000.0f, k_test_nap_threshold};
    CodePatternBlacklist bl;
    EvolutionaryOrchestrator eo{ctrl, bl};

    const float before = ctrl.get_current_atp();
    REQUIRE(eo.run_cycle(k_plugin).status == CycleStatus::SUCCESS);
    const float after  = ctrl.get_current_atp();

    CHECK(before - after == Catch::Approx(EvolutionaryOrchestrator::TOTAL_COST).epsilon(0.01));
}

TEST_CASE("EO — ATP is NOT deducted on failure (refunded)", "[phase113][eo]") {
    MetabolicController  ctrl{10'000.0f, k_test_nap_threshold};
    CodePatternBlacklist bl;
    EvolutionaryOrchestrator eo{ctrl, bl};

    const float before = ctrl.get_current_atp();
    (void)eo.run_cycle(k_bad_path);   // LOAD_FAILED — lock should refund
    const float after  = ctrl.get_current_atp();

    CHECK(before == Catch::Approx(after).epsilon(0.01));
}

// =============================================================================
// Section 6 — Rollback
// =============================================================================

TEST_CASE("EO — rollback with no previous returns false", "[phase113][eo]") {
    MetabolicController  ctrl{10'000.0f, k_test_nap_threshold};
    CodePatternBlacklist bl;
    EvolutionaryOrchestrator eo{ctrl, bl};

    CHECK_FALSE(eo.rollback());
}

TEST_CASE("EO — rollback after two successes restores first module", "[phase113][eo]") {
    require_plugin();

    const std::string copy_path =
        std::string(PHASE113_PLUGIN_DIR) + "/phase113_rollback_copy.so";

    std::filesystem::copy_file(k_plugin, copy_path,
        std::filesystem::copy_options::overwrite_existing);

    MetabolicController  ctrl{10'000.0f, k_test_nap_threshold};
    CodePatternBlacklist bl;
    EvolutionaryOrchestrator eo{ctrl, bl};

    REQUIRE(eo.run_cycle(k_plugin).status    == CycleStatus::SUCCESS);
    REQUIRE(eo.run_cycle(copy_path).status   == CycleStatus::SUCCESS);

    CHECK(eo.active_path() == copy_path);
    REQUIRE(eo.rollback());
    CHECK(eo.active_path() == k_plugin);
    CHECK_FALSE(eo.has_previous());

    std::filesystem::remove(copy_path);
}

// =============================================================================
// Section 7 — Stats
// =============================================================================

TEST_CASE("EO — stats reflect mixed cycle outcomes", "[phase113][eo]") {
    require_plugin();
    MetabolicController  ctrl{10'000.0f, k_test_nap_threshold};
    CodePatternBlacklist bl;
    EvolutionaryOrchestrator eo{ctrl, bl};

    (void)eo.run_cycle(k_bad_path);                              // LOAD_FAILED
    (void)eo.run_cycle(k_plugin, "system(\"rm -rf /\");");       // SECURITY_REJECTED
    (void)eo.run_cycle(k_plugin, {}, bad_energy_provider());     // PHYSICS_REJECTED
    REQUIRE(eo.run_cycle(k_plugin).status == CycleStatus::SUCCESS);  // SUCCESS

    const auto s = eo.stats();
    CHECK(s.total              == 4);
    CHECK(s.succeeded          == 1);
    CHECK(s.security_rejected  == 1);
    CHECK(s.physics_rejected   == 1);
    CHECK(s.load_failed        == 1);
    CHECK(s.atp_denied         == 0);
}

// =============================================================================
// Section 8 — CycleReport boolean operator
// =============================================================================

TEST_CASE("EO — CycleReport bool operator true on SUCCESS", "[phase113][eo]") {
    require_plugin();
    MetabolicController  ctrl{10'000.0f, k_test_nap_threshold};
    CodePatternBlacklist bl;
    EvolutionaryOrchestrator eo{ctrl, bl};

    CycleReport rep = eo.run_cycle(k_plugin);
    CHECK(static_cast<bool>(rep));
}

TEST_CASE("EO — CycleReport bool operator false on failure", "[phase113][eo]") {
    MetabolicController  ctrl{10'000.0f, k_test_nap_threshold};
    CodePatternBlacklist bl;
    EvolutionaryOrchestrator eo{ctrl, bl};

    CycleReport rep = eo.run_cycle(k_bad_path);
    CHECK_FALSE(static_cast<bool>(rep));
}

// =============================================================================
// Section 9 — cycle_status_str helper
// =============================================================================

TEST_CASE("EO — cycle_status_str covers all enumerators", "[phase113][eo]") {
    using nikola::autonomy::cycle_status_str;
    CHECK(cycle_status_str(CycleStatus::SUCCESS)           == "SUCCESS");
    CHECK(cycle_status_str(CycleStatus::ATP_DENIED)        == "ATP_DENIED");
    CHECK(cycle_status_str(CycleStatus::SECURITY_REJECTED) == "SECURITY_REJECTED");
    CHECK(cycle_status_str(CycleStatus::PHYSICS_REJECTED)  == "PHYSICS_REJECTED");
    CHECK(cycle_status_str(CycleStatus::LOAD_FAILED)       == "LOAD_FAILED");
    CHECK(cycle_status_str(CycleStatus::SYMBOL_MISSING)    == "SYMBOL_MISSING");
    CHECK(cycle_status_str(CycleStatus::SAME_MODULE)       == "SAME_MODULE");
}

// =============================================================================
// Section 10 — Thread-safety smoke test
// =============================================================================

TEST_CASE("EO — concurrent run_cycle calls do not crash", "[phase113][eo][thread]") {
    require_plugin();

    // Use separate controllers to avoid ATP contention between threads.
    // Each thread gets its own orchestrator with ample ATP.
    constexpr int N = 6;
    std::vector<CycleStatus> results(N, CycleStatus::ATP_DENIED);

    {
        std::vector<std::thread> threads;
        threads.reserve(N);
        for (int i = 0; i < N; ++i) {
            threads.emplace_back([&results, i]() {
                MetabolicController  ctrl{10'000.0f, k_test_nap_threshold};
                CodePatternBlacklist bl;
                EvolutionaryOrchestrator eo{ctrl, bl};
                results[i] = eo.run_cycle(k_plugin).status;
            });
        }
        for (auto& t : threads) t.join();
    }

    // Each thread should independently succeed.
    for (int i = 0; i < N; ++i)
        CHECK(results[i] == CycleStatus::SUCCESS);
}
