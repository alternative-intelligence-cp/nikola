// =============================================================================
// tests/unit/phase89_inner_monologue_policy_test.cpp
// Phase 89 — GAP-016: Inner Monologue Recursion & ATP Policy
//
// Tests for nikola::cognitive::inner_monologue_policy.hpp
// Spec: docs/info/integration/sections/03_cognitive_systems/03_neuroplastic_transformer.md
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "nikola/cognitive/inner_monologue_policy.hpp"

using namespace nikola::cognitive;
using Catch::Approx;

// ---------------------------------------------------------------------------
// § Recursion depth limits
// ---------------------------------------------------------------------------

TEST_CASE("RECURSION_HARD_LIMIT is 12", "[constants][phase89]") {
    CHECK(RECURSION_HARD_LIMIT == 12);
}

TEST_CASE("RECURSION_SOFT_LIMIT is 7", "[constants][phase89]") {
    CHECK(RECURSION_SOFT_LIMIT == 7);
}

TEST_CASE("Soft limit is strictly less than hard limit", "[constants][phase89]") {
    CHECK(RECURSION_SOFT_LIMIT < RECURSION_HARD_LIMIT);
}

// ---------------------------------------------------------------------------
// § ATP energy model constants
// ---------------------------------------------------------------------------

TEST_CASE("ATP_RESERVE_THRESHOLD is 0.15", "[constants][phase89]") {
    CHECK(ATP_RESERVE_THRESHOLD == Approx(0.15));
}

TEST_CASE("ATP_RESERVE_CRITICAL is 0.05", "[constants][phase89]") {
    CHECK(ATP_RESERVE_CRITICAL == Approx(0.05));
}

TEST_CASE("ATP_BASE_COST_PER_STEP is 0.05", "[constants][phase89]") {
    CHECK(ATP_BASE_COST_PER_STEP == Approx(0.05));
}

TEST_CASE("RECURSION_PENALTY_RATE is 0.15", "[constants][phase89]") {
    CHECK(RECURSION_PENALTY_RATE == Approx(0.15));
}

// ---------------------------------------------------------------------------
// § Spectral entropy constants
// ---------------------------------------------------------------------------

TEST_CASE("SPECTRAL_ENTROPY_LIMIT is 0.85", "[constants][phase89]") {
    CHECK(SPECTRAL_ENTROPY_LIMIT == Approx(0.85));
}

TEST_CASE("ENTROPY_GRADIENT_LIMIT is 0.05", "[constants][phase89]") {
    CHECK(ENTROPY_GRADIENT_LIMIT == Approx(0.05));
}

// ---------------------------------------------------------------------------
// § Boredom / trap cluster constants
// ---------------------------------------------------------------------------

TEST_CASE("BOREDOM_LOOP_SPIKE is 0.20", "[constants][phase89]") {
    CHECK(BOREDOM_LOOP_SPIKE == Approx(0.20));
}

TEST_CASE("TRAP_NODES is 19 (central + 18-point stencil)", "[constants][phase89]") {
    CHECK(TRAP_NODES == 19);
}

TEST_CASE("TRAP_KB_TOTAL equals TRAP_NODES * TRAP_KB_PER_NODE", "[constants][phase89]") {
    CHECK(TRAP_KB_TOTAL == Approx(TRAP_NODES * TRAP_KB_PER_NODE).epsilon(1e-9));
}

TEST_CASE("MAX_ACTIVE_TRAPS is 9", "[constants][phase89]") {
    CHECK(MAX_ACTIVE_TRAPS == 9);
}

// ---------------------------------------------------------------------------
// § atp_cost_at_depth
// ---------------------------------------------------------------------------

TEST_CASE("atp_cost_at_depth(0) equals ATP_BASE_COST_PER_STEP", "[functions][phase89]") {
    CHECK(atp_cost_at_depth(0) == Approx(ATP_BASE_COST_PER_STEP));
}

TEST_CASE("atp_cost_at_depth(1) equals base × (1 + lambda)", "[functions][phase89]") {
    double expected = ATP_BASE_COST_PER_STEP * (1.0 + RECURSION_PENALTY_RATE);
    CHECK(atp_cost_at_depth(1) == Approx(expected).epsilon(1e-9));
}

TEST_CASE("atp_cost_at_depth increases monotonically with depth", "[functions][phase89]") {
    for (int d = 0; d < RECURSION_HARD_LIMIT - 1; ++d) {
        CHECK(atp_cost_at_depth(d + 1) > atp_cost_at_depth(d));
    }
}

// ---------------------------------------------------------------------------
// § atp_cumulative_cost
// ---------------------------------------------------------------------------

TEST_CASE("atp_cumulative_cost(0) equals atp_cost_at_depth(0)", "[functions][phase89]") {
    CHECK(atp_cumulative_cost(0) == Approx(atp_cost_at_depth(0)));
}

TEST_CASE("atp_cumulative_cost increases with depth", "[functions][phase89]") {
    CHECK(atp_cumulative_cost(5) > atp_cumulative_cost(4));
}

// ---------------------------------------------------------------------------
// § Depth policy queries
// ---------------------------------------------------------------------------

TEST_CASE("depth_within_soft_limit accepts <= 7", "[functions][phase89]") {
    CHECK(depth_within_soft_limit(0)  == true);
    CHECK(depth_within_soft_limit(7)  == true);
    CHECK(depth_within_soft_limit(8)  == false);
}

TEST_CASE("depth_within_hard_limit accepts < 12", "[functions][phase89]") {
    CHECK(depth_within_hard_limit(11) == true);
    CHECK(depth_within_hard_limit(12) == false);
}

// ---------------------------------------------------------------------------
// § ATP affordability and critical checks
// ---------------------------------------------------------------------------

TEST_CASE("atp_affordable returns false when spending would go below threshold", "[functions][phase89]") {
    double reserve = 0.20;
    double cost    = 0.10;
    // 0.20 - 0.10 = 0.10 < 0.15 → not affordable
    CHECK(atp_affordable(reserve, cost) == false);
}

TEST_CASE("atp_affordable returns true when reserve stays above threshold", "[functions][phase89]") {
    double reserve = 0.30;
    double cost    = 0.05;
    // 0.30 - 0.05 = 0.25 >= 0.15 → affordable
    CHECK(atp_affordable(reserve, cost) == true);
}

TEST_CASE("atp_critical triggers below 0.05", "[functions][phase89]") {
    CHECK(atp_critical(0.04) == true);
    CHECK(atp_critical(0.05) == false);
}

// ---------------------------------------------------------------------------
// § Entropy predicates
// ---------------------------------------------------------------------------

TEST_CASE("entropy_incoherent triggers at or above 0.85", "[functions][phase89]") {
    CHECK(entropy_incoherent(0.85) == true);
    CHECK(entropy_incoherent(0.84) == false);
}

TEST_CASE("entropy_scrambling triggers above 0.05", "[functions][phase89]") {
    CHECK(entropy_scrambling(0.06) == true);
    CHECK(entropy_scrambling(0.05) == false);
}

// ---------------------------------------------------------------------------
// § Trap budget
// ---------------------------------------------------------------------------

TEST_CASE("trap_budget_exceeded triggers when active_traps > MAX_ACTIVE_TRAPS", "[functions][phase89]") {
    CHECK(trap_budget_exceeded(9)  == false);
    CHECK(trap_budget_exceeded(10) == true);
}

// ---------------------------------------------------------------------------
// § Label helpers
// ---------------------------------------------------------------------------

TEST_CASE("depth_policy_label classifies correctly", "[labels][phase89]") {
    CHECK(depth_policy_label(5)  == "nominal");
    CHECK(depth_policy_label(7)  == "nominal");
    CHECK(depth_policy_label(8)  == "overextended");
    CHECK(depth_policy_label(11) == "overextended");
    CHECK(depth_policy_label(12) == "hard-limit-breach");
}

TEST_CASE("atp_reserve_label classifies correctly", "[labels][phase89]") {
    CHECK(atp_reserve_label(0.04)  == "critical");
    CHECK(atp_reserve_label(0.10)  == "warning");
    CHECK(atp_reserve_label(0.50)  == "nominal");
}
