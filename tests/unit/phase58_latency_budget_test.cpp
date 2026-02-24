/**
 * @file phase58_latency_budget_test.cpp
 * @brief Phase 58 — GAP-025: End-to-End Latency Budget Allocation
 *
 * Validates the LatencyBudget engine against spec §GAP-025:
 *   - 1000 Hz tick budget, safety margin, allocatable ceiling
 *   - Component budget allocation (Physics=600, Scanner=200, ENGS=50, IPC=50)
 *   - Alert thresholds: tick (950 / 1050 μs), energy drift (0.01% / 0.1%),
 *     ATP reserve (15% / 5%), amplitude (4.0 / 5.0)
 *   - "Drop or Degrade" policy: NO_DEGRADE / SKIP_NONLINEAR / DROP_FRAME
 *   - Hardware watchdog timeout (2000 μs)
 *   - Composite TelemetrySnapshot assessment
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/catch_approx.hpp>

#include "nikola/physics/latency_budget.hpp"

using namespace nikola::physics;
using Catch::Approx;
using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

// ---------------------------------------------------------------------------
// §1 — Budget constants
// ---------------------------------------------------------------------------

TEST_CASE("GAP-025 §1: tick budget constants are exact", "[gap025][constants]") {
    REQUIRE(TICK_BUDGET_US        == Approx(1000.0));
    REQUIRE(TICK_SAFETY_MARGIN_US == Approx(100.0));
    REQUIRE(TICK_ALLOCATABLE_US   == Approx(900.0));
    // safety margin + allocatable = total
    REQUIRE(TICK_SAFETY_MARGIN_US + TICK_ALLOCATABLE_US == Approx(TICK_BUDGET_US));
}

TEST_CASE("GAP-025 §2: component budget values match spec", "[gap025][constants]") {
    REQUIRE(BUDGET_PHYSICS_KERNEL_US == Approx(600.0));
    REQUIRE(BUDGET_SCANNER_US        == Approx(200.0));
    REQUIRE(BUDGET_ENGS_US           == Approx(50.0));
    REQUIRE(BUDGET_INFRASTRUCTURE_US == Approx(50.0));
}

TEST_CASE("GAP-025 §3: component budgets sum exactly to allocatable budget", "[gap025][constants]") {
    double sum = BUDGET_PHYSICS_KERNEL_US
               + BUDGET_SCANNER_US
               + BUDGET_ENGS_US
               + BUDGET_INFRASTRUCTURE_US;
    REQUIRE(sum == Approx(TICK_ALLOCATABLE_US));
    REQUIRE(LatencyBudget::total_component_budget_us() == Approx(TICK_ALLOCATABLE_US));
}

// ---------------------------------------------------------------------------
// §4 — component_budget_us()
// ---------------------------------------------------------------------------

TEST_CASE("GAP-025 §4: component_budget_us returns correct per-component values", "[gap025][budget]") {
    REQUIRE(LatencyBudget::component_budget_us(Component::PHYSICS_KERNEL) == Approx(600.0));
    REQUIRE(LatencyBudget::component_budget_us(Component::SCANNER)        == Approx(200.0));
    REQUIRE(LatencyBudget::component_budget_us(Component::ENGS)           == Approx(50.0));
    REQUIRE(LatencyBudget::component_budget_us(Component::INFRASTRUCTURE) == Approx(50.0));
}

// ---------------------------------------------------------------------------
// §5 — budget_fraction()
// ---------------------------------------------------------------------------

TEST_CASE("GAP-025 §5: budget fractions are correct ratios of allocatable budget", "[gap025][budget]") {
    // Physics = 600/900 = 0.6667
    REQUIRE(LatencyBudget::budget_fraction(Component::PHYSICS_KERNEL) == Approx(600.0 / 900.0));
    // Scanner = 200/900 = 0.2222
    REQUIRE(LatencyBudget::budget_fraction(Component::SCANNER) == Approx(200.0 / 900.0));
    // ENGS = 50/900
    REQUIRE(LatencyBudget::budget_fraction(Component::ENGS) == Approx(50.0 / 900.0));
    // Infrastructure = 50/900
    REQUIRE(LatencyBudget::budget_fraction(Component::INFRASTRUCTURE) == Approx(50.0 / 900.0));
    // All fractions sum to 1.0
    double total_frac
        = LatencyBudget::budget_fraction(Component::PHYSICS_KERNEL)
        + LatencyBudget::budget_fraction(Component::SCANNER)
        + LatencyBudget::budget_fraction(Component::ENGS)
        + LatencyBudget::budget_fraction(Component::INFRASTRUCTURE);
    REQUIRE(total_frac == Approx(1.0));
}

// ---------------------------------------------------------------------------
// §6 — watchdog_timeout_us()
// ---------------------------------------------------------------------------

TEST_CASE("GAP-025 §6: watchdog timeout is exactly 2000 μs (2 ticks)", "[gap025][watchdog]") {
    REQUIRE(LatencyBudget::watchdog_timeout_us() == Approx(2000.0));
    // Must be exactly 2× the tick budget
    REQUIRE(LatencyBudget::watchdog_timeout_us() == Approx(2.0 * TICK_BUDGET_US));
}

// ---------------------------------------------------------------------------
// §7 — tick_alert()
// ---------------------------------------------------------------------------

TEST_CASE("GAP-025 §7: tick_alert — well below warning threshold is OK", "[gap025][tick]") {
    // 100 μs — tiny tick
    REQUIRE(LatencyBudget::tick_alert(100'000LL) == AlertLevel::OK);
    // 600 μs — physics kernel exact
    REQUIRE(LatencyBudget::tick_alert(600'000LL) == AlertLevel::OK);
    // 900 μs — full allocatable budget used
    REQUIRE(LatencyBudget::tick_alert(900'000LL) == AlertLevel::OK);
    // 950 μs — exactly at warning boundary → still OK
    REQUIRE(LatencyBudget::tick_alert(950'000LL) == AlertLevel::OK);
}

TEST_CASE("GAP-025 §8: tick_alert — between thresholds is WARNING", "[gap025][tick]") {
    // 950,001 ns = just past the 950 μs OK boundary
    REQUIRE(LatencyBudget::tick_alert(950'001LL) == AlertLevel::WARNING);
    // 1,000 μs — full budget exhausted
    REQUIRE(LatencyBudget::tick_alert(1'000'000LL) == AlertLevel::WARNING);
    // 1,050 μs — exactly at critical boundary → still WARNING
    REQUIRE(LatencyBudget::tick_alert(1'050'000LL) == AlertLevel::WARNING);
}

TEST_CASE("GAP-025 §9: tick_alert — over critical threshold is CRITICAL", "[gap025][tick]") {
    // 1,050,001 ns — just past critical boundary
    REQUIRE(LatencyBudget::tick_alert(1'050'001LL) == AlertLevel::CRITICAL);
    // 2,000 μs — watchdog fires at this point
    REQUIRE(LatencyBudget::tick_alert(2'000'000LL) == AlertLevel::CRITICAL);
}

// ---------------------------------------------------------------------------
// §10 — energy_drift_alert()
// ---------------------------------------------------------------------------

TEST_CASE("GAP-025 §10: energy_drift_alert thresholds", "[gap025][energy]") {
    // Well below WARNING (< 0.01%)
    REQUIRE(LatencyBudget::energy_drift_alert(0.0)     == AlertLevel::OK);
    REQUIRE(LatencyBudget::energy_drift_alert(1e-6)    == AlertLevel::OK);
    REQUIRE(LatencyBudget::energy_drift_alert(9.99e-5) == AlertLevel::OK);
    // Exactly at WARNING threshold (1e-4 = 0.01%)
    REQUIRE(LatencyBudget::energy_drift_alert(1e-4)    == AlertLevel::WARNING);
    // Between WARNING and CRITICAL
    REQUIRE(LatencyBudget::energy_drift_alert(5e-4)    == AlertLevel::WARNING);
    REQUIRE(LatencyBudget::energy_drift_alert(9.99e-4) == AlertLevel::WARNING);
    // Exactly at CRITICAL threshold (1e-3 = 0.1%)
    REQUIRE(LatencyBudget::energy_drift_alert(1e-3)    == AlertLevel::CRITICAL);
    REQUIRE(LatencyBudget::energy_drift_alert(1e-2)    == AlertLevel::CRITICAL);
}

// ---------------------------------------------------------------------------
// §11 — atp_alert()
// ---------------------------------------------------------------------------

TEST_CASE("GAP-025 §11: atp_alert thresholds", "[gap025][atp]") {
    // Well above WARNING (> 15%)
    REQUIRE(LatencyBudget::atp_alert(100.0) == AlertLevel::OK);
    REQUIRE(LatencyBudget::atp_alert(50.0)  == AlertLevel::OK);
    REQUIRE(LatencyBudget::atp_alert(15.1)  == AlertLevel::OK);
    // Exactly at WARNING threshold (≤ 15%)
    REQUIRE(LatencyBudget::atp_alert(15.0)  == AlertLevel::WARNING);
    // Between WARNING and CRITICAL ([5%, 15%])
    REQUIRE(LatencyBudget::atp_alert(10.0)  == AlertLevel::WARNING);
    REQUIRE(LatencyBudget::atp_alert(5.0)   == AlertLevel::WARNING);
    // Below CRITICAL threshold (< 5%)
    REQUIRE(LatencyBudget::atp_alert(4.99)  == AlertLevel::CRITICAL);
    REQUIRE(LatencyBudget::atp_alert(0.0)   == AlertLevel::CRITICAL);
}

// ---------------------------------------------------------------------------
// §12 — amplitude_alert()
// ---------------------------------------------------------------------------

TEST_CASE("GAP-025 §12: amplitude_alert thresholds", "[gap025][amplitude]") {
    // Normal regime (< 4.0)
    REQUIRE(LatencyBudget::amplitude_alert(0.0) == AlertLevel::OK);
    REQUIRE(LatencyBudget::amplitude_alert(3.5) == AlertLevel::OK);
    REQUIRE(LatencyBudget::amplitude_alert(3.99) == AlertLevel::OK);
    // WARNING zone [4.0, 5.0)
    REQUIRE(LatencyBudget::amplitude_alert(4.0) == AlertLevel::WARNING);
    REQUIRE(LatencyBudget::amplitude_alert(4.5) == AlertLevel::WARNING);
    REQUIRE(LatencyBudget::amplitude_alert(4.99) == AlertLevel::WARNING);
    // CRITICAL: hard limit (≥ 5.0) — triggers SCRAM
    REQUIRE(LatencyBudget::amplitude_alert(5.0) == AlertLevel::CRITICAL);
    REQUIRE(LatencyBudget::amplitude_alert(6.0) == AlertLevel::CRITICAL);
}

// ---------------------------------------------------------------------------
// §13 — degrade_policy() interior cases
// ---------------------------------------------------------------------------

TEST_CASE("GAP-025 §13: degrade_policy — within allocatable budget", "[gap025][policy]") {
    // 0 ns (instantaneous)
    REQUIRE(LatencyBudget::degrade_policy(0LL) == DegradePolicy::NO_DEGRADE);
    // 500 μs — typical fast tick
    REQUIRE(LatencyBudget::degrade_policy(500'000LL) == DegradePolicy::NO_DEGRADE);
    // 900 μs exactly — at the allocatable ceiling
    REQUIRE(LatencyBudget::degrade_policy(900'000LL) == DegradePolicy::NO_DEGRADE);
}

TEST_CASE("GAP-025 §14: degrade_policy — between allocatable and hard limit", "[gap025][policy]") {
    // 900,001 ns — just over allocatable budget → skip nonlinear step
    REQUIRE(LatencyBudget::degrade_policy(900'001LL) == DegradePolicy::SKIP_NONLINEAR);
    // 950 μs
    REQUIRE(LatencyBudget::degrade_policy(950'000LL) == DegradePolicy::SKIP_NONLINEAR);
    // 1,000 μs exactly — at the hard limit boundary
    REQUIRE(LatencyBudget::degrade_policy(1'000'000LL) == DegradePolicy::SKIP_NONLINEAR);
}

TEST_CASE("GAP-025 §15: degrade_policy — over hard limit drops frame", "[gap025][policy]") {
    // 1,000,001 ns — just over hard limit
    REQUIRE(LatencyBudget::degrade_policy(1'000'001LL) == DegradePolicy::DROP_FRAME);
    // 2,000 μs — watchdog deadline
    REQUIRE(LatencyBudget::degrade_policy(2'000'000LL) == DegradePolicy::DROP_FRAME);
    // Very long stall
    REQUIRE(LatencyBudget::degrade_policy(10'000'000LL) == DegradePolicy::DROP_FRAME);
}

// ---------------------------------------------------------------------------
// §16 — assess_overall — all OK
// ---------------------------------------------------------------------------

TEST_CASE("GAP-025 §16: assess_overall — nominal snapshot is OK and healthy", "[gap025][composite]") {
    TelemetrySnapshot snap;
    snap.tick_duration_ns    = 700'000;   // 700 μs — within budget
    snap.energy_drift_ratio  = 1e-6;      // well under 0.01%
    snap.lock_contention_cnt = 0;
    snap.atp_reserve_pct     = 80.0;      // 80% — healthy
    snap.amplitude_max       = 2.0;       // well under 4.0

    REQUIRE(LatencyBudget::assess_overall(snap) == AlertLevel::OK);
    REQUIRE(LatencyBudget::is_healthy(snap));
    REQUIRE_FALSE(LatencyBudget::requires_scram(snap));
}

// ---------------------------------------------------------------------------
// §17 — assess_overall — WARNING propagation
// ---------------------------------------------------------------------------

TEST_CASE("GAP-025 §17: assess_overall — single WARNING metric lifts result to WARNING", "[gap025][composite]") {
    TelemetrySnapshot snap;
    snap.tick_duration_ns    = 960'000;  // 960 μs — WARNING
    snap.energy_drift_ratio  = 1e-6;
    snap.lock_contention_cnt = 0;
    snap.atp_reserve_pct     = 80.0;
    snap.amplitude_max       = 1.0;

    REQUIRE(LatencyBudget::assess_overall(snap) == AlertLevel::WARNING);
    REQUIRE_FALSE(LatencyBudget::is_healthy(snap));
    REQUIRE_FALSE(LatencyBudget::requires_scram(snap));

    // ATP entering warning zone also lifts result
    TelemetrySnapshot snap2 = snap;
    snap2.tick_duration_ns = 500'000;     // back to OK
    snap2.atp_reserve_pct  = 10.0;        // WARNING
    REQUIRE(LatencyBudget::assess_overall(snap2) == AlertLevel::WARNING);
}

// ---------------------------------------------------------------------------
// §18 — assess_overall — CRITICAL propagation
// ---------------------------------------------------------------------------

TEST_CASE("GAP-025 §18: assess_overall — any CRITICAL metric triggers SCRAM", "[gap025][composite]") {
    TelemetrySnapshot snap;
    snap.tick_duration_ns    = 700'000;
    snap.energy_drift_ratio  = 1e-6;
    snap.lock_contention_cnt = 2;
    snap.atp_reserve_pct     = 80.0;
    snap.amplitude_max       = 5.5;      // CRITICAL amplitude

    REQUIRE(LatencyBudget::assess_overall(snap) == AlertLevel::CRITICAL);
    REQUIRE(LatencyBudget::requires_scram(snap));

    // CRITICAL energy drift
    TelemetrySnapshot snap2;
    snap2.tick_duration_ns    = 500'000;
    snap2.energy_drift_ratio  = 2e-3;    // CRITICAL
    snap2.atp_reserve_pct     = 80.0;
    snap2.amplitude_max       = 1.0;
    REQUIRE(LatencyBudget::assess_overall(snap2) == AlertLevel::CRITICAL);
    REQUIRE(LatencyBudget::requires_scram(snap2));
}

// ---------------------------------------------------------------------------
// §19 — TelemetrySnapshot is_sane()
// ---------------------------------------------------------------------------

TEST_CASE("GAP-025 §19: TelemetrySnapshot is_sane validates field invariants", "[gap025][snapshot]") {
    TelemetrySnapshot good;
    good.tick_duration_ns    = 500'000;
    good.energy_drift_ratio  = 1e-5;
    good.lock_contention_cnt = 0;
    good.atp_reserve_pct     = 50.0;
    good.amplitude_max       = 2.5;
    REQUIRE(good.is_sane());

    // Default-constructed is also sane (all zeros / 100%)
    TelemetrySnapshot def;
    REQUIRE(def.is_sane());

    // Negative tick duration breaks sanity
    TelemetrySnapshot bad1 = good;
    bad1.tick_duration_ns = -1;
    REQUIRE_FALSE(bad1.is_sane());

    // Negative energy drift breaks sanity
    TelemetrySnapshot bad2 = good;
    bad2.energy_drift_ratio = -0.01;
    REQUIRE_FALSE(bad2.is_sane());

    // Negative ATP breaks sanity
    TelemetrySnapshot bad3 = good;
    bad3.atp_reserve_pct = -1.0;
    REQUIRE_FALSE(bad3.is_sane());
}

// ---------------------------------------------------------------------------
// §20 — Integration: Drop-or-Degrade + multi-metric cascade
// ---------------------------------------------------------------------------

TEST_CASE("GAP-025 §20: integration — cascade from healthy to degraded to SCRAM", "[gap025][integration]") {
    // Healthy tick
    {
        TelemetrySnapshot snap;
        snap.tick_duration_ns    = 800'000;
        snap.energy_drift_ratio  = 5e-6;
        snap.atp_reserve_pct     = 90.0;
        snap.amplitude_max       = 1.5;
        REQUIRE(LatencyBudget::degrade_policy(snap.tick_duration_ns) == DegradePolicy::NO_DEGRADE);
        REQUIRE(LatencyBudget::assess_overall(snap) == AlertLevel::OK);
    }
    // Degraded tick (skip nonlinear step — Hamiltonial conservation preserved)
    {
        TelemetrySnapshot snap;
        snap.tick_duration_ns    = 960'000;   // 960 μs > 950 μs → WARNING tick
        snap.energy_drift_ratio  = 2e-4;       // 0.02% > 0.01% → WARNING energy
        snap.atp_reserve_pct     = 20.0;       // 20% > 15% → OK
        snap.amplitude_max       = 3.0;        // < 4.0 → OK
        REQUIRE(LatencyBudget::degrade_policy(snap.tick_duration_ns) == DegradePolicy::SKIP_NONLINEAR);
        REQUIRE(LatencyBudget::assess_overall(snap) == AlertLevel::WARNING);
    }
    // SCRAM scenario: tick over limit + energy exploding
    {
        TelemetrySnapshot snap;
        snap.tick_duration_ns    = 1'200'000;  // DROP_FRAME
        snap.energy_drift_ratio  = 5e-3;        // CRITICAL drift
        snap.atp_reserve_pct     = 2.0;          // CRITICAL ATP
        snap.amplitude_max       = 5.1;          // CRITICAL amplitude
        REQUIRE(LatencyBudget::degrade_policy(snap.tick_duration_ns) == DegradePolicy::DROP_FRAME);
        REQUIRE(LatencyBudget::assess_overall(snap) == AlertLevel::CRITICAL);
        REQUIRE(LatencyBudget::requires_scram(snap));
    }
}
