// =============================================================================
// tests/unit/phase86_latency_budget_test.cpp
// Phase 86 — GAP-025: End-to-End Latency Budget Allocation
//
// Tests for nikola::system::latency_budget.hpp
// Spec: docs/info/integration/sections/02_wave_interference_physics.md
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "nikola/system/latency_budget.hpp"

using namespace nikola::system;
using Catch::Approx;

// ---------------------------------------------------------------------------
// § Enums
// ---------------------------------------------------------------------------

TEST_CASE("LoopComponent enum values are distinct", "[enums][phase86]") {
    CHECK(static_cast<int>(LoopComponent::PHYSICS_KERNEL)       == 0);
    CHECK(static_cast<int>(LoopComponent::COGNITIVE_SCANNER)    == 1);
    CHECK(static_cast<int>(LoopComponent::ENGS)                 == 2);
    CHECK(static_cast<int>(LoopComponent::INFRASTRUCTURE)       == 3);
    CHECK(static_cast<int>(LoopComponent::SAFETY_MARGIN)        == 4);
}

TEST_CASE("TickHealth enum values are ordered", "[enums][phase86]") {
    CHECK(static_cast<int>(TickHealth::NOMINAL)   == 0);
    CHECK(static_cast<int>(TickHealth::WARNING)   == 1);
    CHECK(static_cast<int>(TickHealth::CRITICAL)  == 2);
    CHECK(static_cast<int>(TickHealth::OVERRUN)   == 3);
}

// ---------------------------------------------------------------------------
// § Master budget constants
// ---------------------------------------------------------------------------

TEST_CASE("TICK_BUDGET_TOTAL_US is 1000.0", "[constants][phase86]") {
    CHECK(TICK_BUDGET_TOTAL_US == Approx(1000.0));
}

TEST_CASE("TICK_SAFETY_MARGIN_US is 100.0", "[constants][phase86]") {
    CHECK(TICK_SAFETY_MARGIN_US == Approx(100.0));
}

TEST_CASE("TICK_BUDGET_NET_US is 900.0", "[constants][phase86]") {
    CHECK(TICK_BUDGET_NET_US == Approx(900.0));
}

// ---------------------------------------------------------------------------
// § Component allocations
// ---------------------------------------------------------------------------

TEST_CASE("Component budgets sum to TICK_BUDGET_NET_US", "[constants][phase86]") {
    double sum = BUDGET_PHYSICS_KERNEL_US
               + BUDGET_COGNITIVE_SCANNER_US
               + BUDGET_ENGS_US
               + BUDGET_INFRASTRUCTURE_US;
    CHECK(sum == Approx(TICK_BUDGET_NET_US));
}

TEST_CASE("Physics kernel budget is 600 μs", "[constants][phase86]") {
    CHECK(BUDGET_PHYSICS_KERNEL_US == Approx(600.0));
}

TEST_CASE("Cognitive scanner budget is 200 μs", "[constants][phase86]") {
    CHECK(BUDGET_COGNITIVE_SCANNER_US == Approx(200.0));
}

TEST_CASE("ENGS budget is 50 μs", "[constants][phase86]") {
    CHECK(BUDGET_ENGS_US == Approx(50.0));
}

TEST_CASE("Infrastructure budget is 50 μs", "[constants][phase86]") {
    CHECK(BUDGET_INFRASTRUCTURE_US == Approx(50.0));
}

// ---------------------------------------------------------------------------
// § Threshold constants
// ---------------------------------------------------------------------------

TEST_CASE("TICK thresholds are correctly ordered", "[constants][phase86]") {
    CHECK(TICK_WARNING_US  < TICK_CRITICAL_US);
    CHECK(TICK_CRITICAL_US < WATCHDOG_TIMEOUT_US);
    CHECK(TICK_WARNING_US  == Approx(950.0));
    CHECK(TICK_CRITICAL_US == Approx(1050.0));
    CHECK(WATCHDOG_TIMEOUT_US == Approx(2000.0));
}

// ---------------------------------------------------------------------------
// § classify_tick
// ---------------------------------------------------------------------------

TEST_CASE("classify_tick returns NOMINAL for fast ticks", "[functions][phase86]") {
    CHECK(classify_tick(0.0)   == TickHealth::NOMINAL);
    CHECK(classify_tick(900.0) == TickHealth::NOMINAL);
    CHECK(classify_tick(949.9) == TickHealth::NOMINAL);
}

TEST_CASE("classify_tick returns WARNING in [950, 1050)", "[functions][phase86]") {
    CHECK(classify_tick(950.0)  == TickHealth::WARNING);
    CHECK(classify_tick(1049.9) == TickHealth::WARNING);
}

TEST_CASE("classify_tick returns CRITICAL in [1050, 2000)", "[functions][phase86]") {
    CHECK(classify_tick(1050.0) == TickHealth::CRITICAL);
    CHECK(classify_tick(1999.9) == TickHealth::CRITICAL);
}

TEST_CASE("classify_tick returns OVERRUN at or above watchdog timeout", "[functions][phase86]") {
    CHECK(classify_tick(2000.0) == TickHealth::OVERRUN);
    CHECK(classify_tick(5000.0) == TickHealth::OVERRUN);
}

// ---------------------------------------------------------------------------
// § tick_nominal / tick_warning / tick_critical
// ---------------------------------------------------------------------------

TEST_CASE("tick_nominal / warning / critical boolean helpers agree with classify_tick", "[functions][phase86]") {
    CHECK(tick_nominal(500.0)  == true);
    CHECK(tick_warning(960.0)  == true);
    CHECK(tick_critical(1100.0) == true);
    CHECK(tick_nominal(1100.0) == false);
}

// ---------------------------------------------------------------------------
// § component_budget_us
// ---------------------------------------------------------------------------

TEST_CASE("component_budget_us returns spec values", "[functions][phase86]") {
    CHECK(component_budget_us(LoopComponent::PHYSICS_KERNEL)    == Approx(600.0));
    CHECK(component_budget_us(LoopComponent::COGNITIVE_SCANNER) == Approx(200.0));
    CHECK(component_budget_us(LoopComponent::ENGS)              == Approx(50.0));
    CHECK(component_budget_us(LoopComponent::INFRASTRUCTURE)    == Approx(50.0));
    CHECK(component_budget_us(LoopComponent::SAFETY_MARGIN)     == Approx(100.0));
}

// ---------------------------------------------------------------------------
// § component_within_budget
// ---------------------------------------------------------------------------

TEST_CASE("component_within_budget accepts time at or below allocation", "[functions][phase86]") {
    CHECK(component_within_budget(599.9, LoopComponent::PHYSICS_KERNEL)    == true);
    CHECK(component_within_budget(600.0, LoopComponent::PHYSICS_KERNEL)    == true);
    CHECK(component_within_budget(600.1, LoopComponent::PHYSICS_KERNEL)    == false);
}

// ---------------------------------------------------------------------------
// § tick_budget_fraction
// ---------------------------------------------------------------------------

TEST_CASE("tick_budget_fraction of 1000 μs equals 1.0", "[functions][phase86]") {
    CHECK(tick_budget_fraction(1000.0) == Approx(1.0));
}

TEST_CASE("tick_budget_fraction of 500 μs equals 0.5", "[functions][phase86]") {
    CHECK(tick_budget_fraction(500.0) == Approx(0.5));
}

// ---------------------------------------------------------------------------
// § net_budget_fraction
// ---------------------------------------------------------------------------

TEST_CASE("net_budget_fraction of TICK_BUDGET_NET_US equals 1.0", "[functions][phase86]") {
    CHECK(net_budget_fraction(TICK_BUDGET_NET_US) == Approx(1.0));
}
