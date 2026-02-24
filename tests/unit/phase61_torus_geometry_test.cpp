/**
 * @file phase61_torus_geometry_test.cpp
 * @brief Phase 61 — GAP-024 + GAP-041: Metric Tensor Consolidation Scheduler
 *        and 9D Coordinate Semantics
 *
 * GAP-024 (ConsolidationScheduler) tests
 * ────────────────────────────────────────
 * §1  Constant values (MAX_INTERVAL, PERTURBATION_LIMIT, METABOLIC_FLOOR,
 *     DEFERRAL_FACTOR, FAST_PATH_REDUCTION, AVAILABILITY_TARGET)
 * §2  should_consolidate: perturbation > limit → triggers immediately
 * §3  should_consolidate: is_napping → triggers regardless of ATP
 * §4  should_consolidate: time > 5 min, ATP ok → triggers
 * §5  should_consolidate: time > 5 min, ATP low → defers (not yet at cap)
 * §6  should_consolidate: time at hard deadline (2×) even with low ATP → forces
 * §7  should_consolidate: steady-state (low time, ok ATP, low perturbation) → false
 * §8  should_consolidate: perturbation priority overrides nap=false and low ATP
 * §9  on_consolidated resets timer and perturbation norm to zero
 * §10 advance(dt) accumulates time accurately, multiple calls
 * §11 advance rejects negative dt
 * §12 is_overdue / is_past_deadline helpers
 * §13 update_perturbation tracks max, not clobbered by smaller values
 * §14 Custom config (override max_interval_sec) roundtrip
 *
 * GAP-041 (CoordinateSemantics) tests
 * ──────────────────────────────────────
 * §15 DIM_COUNT == 9, BITS_PER_SPATIAL_DIM == 14, SPATIAL_AXIS_MAX == 16384
 * §16 MORTON_KEY_BITS == 126 (9 × 14)
 * §17 Domain classification for all 9 Dim9 enumerators
 * §18 dim_name: all 9 names correct
 * §19 dim_symbol: all 9 symbols correct
 * §20 dim_is_integer: only X, Y, Z are integer
 * §21 dim_is_cyclic: TIME, X, Y, Z are cyclic; others are not
 * §22 dim_is_complex: only U, V, W are complex
 * §23 wave_speed_effective: s=0→c0, s=1→0.5, s=2→1/3
 * §24 damping_coefficient: r=1→0, r=0→α, r=0.5→α/2
 * §25 max_nodes_per_axis == 16384, max_spatial_nodes == 16384³
 * §26 domain_name: all four human-readable names correct
 * §27 Integration: full ConsolidationScheduler tick loop simulation
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "nikola/physics/consolidation_scheduler.hpp"
#include "nikola/physics/coordinate_semantics.hpp"

#include <cstdint>
#include <stdexcept>

using namespace nikola::physics;
using Catch::Approx;

// ============================================================================
// §1 — Constants (GAP-024)
// ============================================================================

TEST_CASE("GAP-024 §1: Consolidation constants match spec", "[gap024][constants]") {
    REQUIRE(CONSOLIDATION_MAX_INTERVAL_SEC     == Approx(300.0));   // 5 minutes
    REQUIRE(CONSOLIDATION_PERTURBATION_LIMIT   == Approx(0.1));     // 10% deviation
    REQUIRE(CONSOLIDATION_METABOLIC_FLOOR      == Approx(0.2));     // 20% ATP minimum
    REQUIRE(CONSOLIDATION_DEFERRAL_FACTOR      == Approx(2.0));     // cap at 10 min
    // Performance targets (not tested for exact runtime, but values must be documented)
    REQUIRE(CONSOLIDATION_FAST_PATH_FLOPS_REDUCTION == Approx(0.90));
    REQUIRE(CONSOLIDATION_AVAILABILITY_TARGET       == Approx(0.999));
}

// ============================================================================
// §2 — Perturbation trigger (highest priority)
// ============================================================================

TEST_CASE("GAP-024 §2: Perturbation above limit triggers immediately", "[gap024][trigger]") {
    ConsolidationScheduler sched;
    // Time is zero, ATP is fine, not napping — only perturbation is high
    REQUIRE(sched.should_consolidate(0.11, /*nap=*/false, /*atp=*/0.8));
    REQUIRE(sched.should_consolidate(0.50, /*nap=*/false, /*atp=*/0.0));  // even at zero ATP
    REQUIRE(sched.should_consolidate(1.00, /*nap=*/false, /*atp=*/0.0));
}

TEST_CASE("GAP-024 §3: Perturbation exactly at limit does NOT trigger", "[gap024][trigger]") {
    ConsolidationScheduler sched;
    // spec: GREATER THAN, not greater-than-or-equal
    REQUIRE_FALSE(sched.should_consolidate(0.10, false, 0.8));
    REQUIRE_FALSE(sched.should_consolidate(0.05, false, 0.8));
    REQUIRE_FALSE(sched.should_consolidate(0.00, false, 0.8));
}

// ============================================================================
// §3 — Nap trigger
// ============================================================================

TEST_CASE("GAP-024 §4: Nap state triggers consolidation regardless of ATP", "[gap024][trigger]") {
    ConsolidationScheduler sched;
    // No time elapsed, perturbation fine, but is_napping = true
    REQUIRE(sched.should_consolidate(0.0, /*nap=*/true, /*atp=*/0.0));
    REQUIRE(sched.should_consolidate(0.0, /*nap=*/true, /*atp=*/0.05));
    REQUIRE(sched.should_consolidate(0.0, /*nap=*/true, /*atp=*/0.8));
}

// ============================================================================
// §4 — Time-based trigger, sufficient ATP
// ============================================================================

TEST_CASE("GAP-024 §5: Time > 5 min with adequate ATP triggers consolidation", "[gap024][trigger]") {
    ConsolidationScheduler sched;
    sched.time_since_last_update = 301.0;   // just over 5 min
    REQUIRE(sched.should_consolidate(0.0, false, /*atp=*/0.5));
    REQUIRE(sched.should_consolidate(0.0, false, /*atp=*/1.0));
    REQUIRE(sched.should_consolidate(0.0, false, /*atp=*/CONSOLIDATION_METABOLIC_FLOOR));
}

// ============================================================================
// §5 — Deferral: time > 5 min but ATP below metabolic floor (before hard deadline)
// ============================================================================

TEST_CASE("GAP-024 §6: Time > 5 min + low ATP defers below hard deadline", "[gap024][deferral]") {
    ConsolidationScheduler sched;
    // Set time at 6 minutes (360 s) — overdue but well below 2× = 600 s cap
    sched.time_since_last_update = 360.0;
    REQUIRE_FALSE(sched.should_consolidate(0.0, false, /*atp=*/0.05));
    REQUIRE_FALSE(sched.should_consolidate(0.0, false, /*atp=*/0.19));
    REQUIRE_FALSE(sched.should_consolidate(0.0, false, /*atp=*/0.0));
}

// ============================================================================
// §6 — Hard deadline: forces consolidation past 2× regardless of ATP
// ============================================================================

TEST_CASE("GAP-024 §7: Hard deadline 2×MAX_INTERVAL forces consolidation even at low ATP",
          "[gap024][deferral]") {
    ConsolidationScheduler sched;
    // Exactly at 10 minutes = 2 × 300 s
    sched.time_since_last_update = 600.0;
    REQUIRE(sched.should_consolidate(0.0, false, /*atp=*/0.0));
    REQUIRE(sched.should_consolidate(0.0, false, /*atp=*/0.05));
    REQUIRE(sched.should_consolidate(0.0, false, /*atp=*/0.19));
}

TEST_CASE("GAP-024 §8: Past hard deadline also forces consolidation", "[gap024][deferral]") {
    ConsolidationScheduler sched;
    sched.time_since_last_update = 700.0;   // way past 10 min cap
    REQUIRE(sched.should_consolidate(0.0, false, 0.0));
}

// ============================================================================
// §7 — Steady-state: no trigger
// ============================================================================

TEST_CASE("GAP-024 §9: No trigger in steady-state (low time, ok ATP, low perturbation)",
          "[gap024][steady]") {
    ConsolidationScheduler sched;
    // Brand-new scheduler: time=0, norm=0, not napping, ATP fine
    REQUIRE_FALSE(sched.should_consolidate(0.0, false, 0.8));
    REQUIRE_FALSE(sched.should_consolidate(0.09, false, 0.8));  // just under limit

    // 4 minutes elapsed — not yet at 5 min threshold
    sched.time_since_last_update = 240.0;
    REQUIRE_FALSE(sched.should_consolidate(0.05, false, 0.8));
}

// ============================================================================
// §8 — Perturbation priority overrides everything
// ============================================================================

TEST_CASE("GAP-024 §10: Perturbation trigger overrides low ATP and nap=false", "[gap024][trigger]") {
    ConsolidationScheduler sched;
    // Worst case: low ATP, no nap, no time elapsed — but perturbation is critical
    REQUIRE(sched.should_consolidate(0.99, false, 0.0));
}

// ============================================================================
// §9 — on_consolidated resets state
// ============================================================================

TEST_CASE("GAP-024 §11: on_consolidated resets timer and perturbation norm", "[gap024][reset]") {
    ConsolidationScheduler sched;
    sched.time_since_last_update = 500.0;
    sched.max_perturbation_norm  = 0.5;

    sched.on_consolidated();

    REQUIRE(sched.time_since_last_update == Approx(0.0));
    REQUIRE(sched.max_perturbation_norm  == Approx(0.0));
    // After reset, must not trigger (time is zero)
    REQUIRE_FALSE(sched.should_consolidate(0.0, false, 0.8));
}

// ============================================================================
// §10 — advance() accumulates time
// ============================================================================

TEST_CASE("GAP-024 §12: advance(dt) accumulates time correctly", "[gap024][advance]") {
    ConsolidationScheduler sched;
    sched.advance(0.001);   // 1 ms tick
    REQUIRE(sched.time_since_last_update == Approx(0.001));

    for (int i = 0; i < 999; ++i) sched.advance(0.001);   // 1000 ticks = 1 s total
    REQUIRE(sched.time_since_last_update == Approx(1.0).epsilon(1e-6));

    // Advance to just over 5 minutes
    sched.advance(299.001);
    REQUIRE(sched.time_since_last_update > 300.0);
    REQUIRE(sched.should_consolidate(0.0, false, 0.5));
}

// ============================================================================
// §11 — advance rejects negative dt
// ============================================================================

TEST_CASE("GAP-024 §13: advance() throws on negative dt", "[gap024][advance]") {
    ConsolidationScheduler sched;
    REQUIRE_THROWS_AS(sched.advance(-0.001), std::invalid_argument);
    REQUIRE_NOTHROW(sched.advance(0.0));    // zero is allowed (no-op)
}

// ============================================================================
// §12 — Helper predicates is_overdue / is_past_deadline
// ============================================================================

TEST_CASE("GAP-024 §14: is_overdue and is_past_deadline predicates", "[gap024][helpers]") {
    ConsolidationScheduler sched;
    REQUIRE_FALSE(sched.is_overdue());
    REQUIRE_FALSE(sched.is_past_deadline());

    sched.time_since_last_update = 301.0;   // just over 5 min
    REQUIRE(sched.is_overdue());
    REQUIRE_FALSE(sched.is_past_deadline());

    sched.time_since_last_update = 600.0;   // at hard deadline
    REQUIRE(sched.is_overdue());
    REQUIRE(sched.is_past_deadline());
}

// ============================================================================
// §13 — update_perturbation tracks running maximum
// ============================================================================

TEST_CASE("GAP-024 §15: update_perturbation tracks max, ignores smaller values",
          "[gap024][perturbation]") {
    ConsolidationScheduler sched;
    sched.update_perturbation(0.05);
    REQUIRE(sched.max_perturbation_norm == Approx(0.05));

    sched.update_perturbation(0.08);   // larger → updates
    REQUIRE(sched.max_perturbation_norm == Approx(0.08));

    sched.update_perturbation(0.03);   // smaller → does not overwrite
    REQUIRE(sched.max_perturbation_norm == Approx(0.08));

    // Reset clears it
    sched.on_consolidated();
    REQUIRE(sched.max_perturbation_norm == Approx(0.0));
}

// ============================================================================
// §14 — Custom configuration
// ============================================================================

TEST_CASE("GAP-024 §16: Custom max_interval_sec is respected", "[gap024][config]") {
    ConsolidationScheduler sched;
    sched.max_interval_sec = 10.0;   // 10-second interval for fast unit tests

    sched.time_since_last_update = 9.0;
    REQUIRE_FALSE(sched.should_consolidate(0.0, false, 0.8));

    sched.time_since_last_update = 11.0;
    REQUIRE(sched.should_consolidate(0.0, false, 0.8));

    // Deferral cap is also scaled: 2×10 = 20 s
    sched.time_since_last_update = 15.0;
    REQUIRE_FALSE(sched.should_consolidate(0.0, false, 0.0));   // below 20 s → defers

    sched.time_since_last_update = 20.0;
    REQUIRE(sched.should_consolidate(0.0, false, 0.0));         // at cap → force
}

// ============================================================================
// §15 — Constants (GAP-041)
// ============================================================================

TEST_CASE("GAP-041 §1: Fundamental geometry constants", "[gap041][constants]") {
    REQUIRE(DIM_COUNT            == 9);
    REQUIRE(BITS_PER_SPATIAL_DIM == 14);
    REQUIRE(SPATIAL_AXIS_MAX     == 16384);  // 2^14
    REQUIRE(MORTON_KEY_BITS      == 126);    // 9 × 14
    REQUIRE(WAVE_SPEED_NOMINAL   == Approx(1.0));
    REQUIRE(RESONANCE_MIN        == Approx(0.0));
    REQUIRE(RESONANCE_MAX        == Approx(1.0));
    REQUIRE(STATE_MIN            == Approx(0.0));
    REQUIRE(STATE_MAX            == Approx(2.0));
}

TEST_CASE("GAP-041 §2: SPATIAL_AXIS_MAX is exactly 2^14", "[gap041][constants]") {
    REQUIRE(SPATIAL_AXIS_MAX == (1 << BITS_PER_SPATIAL_DIM));
}

// ============================================================================
// §17 — Domain classification
// ============================================================================

TEST_CASE("GAP-041 §3: Domain classification for all 9 dimensions", "[gap041][domain]") {
    // Systemic
    REQUIRE(dim_domain(Dim9::RESONANCE) == CoordDomain::SYSTEMIC);
    REQUIRE(dim_domain(Dim9::STATE)     == CoordDomain::SYSTEMIC);

    // Temporal
    REQUIRE(dim_domain(Dim9::TIME)      == CoordDomain::TEMPORAL);

    // Quantum
    REQUIRE(dim_domain(Dim9::U)         == CoordDomain::QUANTUM);
    REQUIRE(dim_domain(Dim9::V)         == CoordDomain::QUANTUM);
    REQUIRE(dim_domain(Dim9::W)         == CoordDomain::QUANTUM);

    // Spatial
    REQUIRE(dim_domain(Dim9::X)         == CoordDomain::SPATIAL);
    REQUIRE(dim_domain(Dim9::Y)         == CoordDomain::SPATIAL);
    REQUIRE(dim_domain(Dim9::Z)         == CoordDomain::SPATIAL);
}

// ============================================================================
// §18 — dim_name
// ============================================================================

TEST_CASE("GAP-041 §4: dim_name returns correct lowercase names", "[gap041][meta]") {
    REQUIRE(dim_name(Dim9::RESONANCE) == "resonance");
    REQUIRE(dim_name(Dim9::STATE)     == "state");
    REQUIRE(dim_name(Dim9::TIME)      == "time");
    REQUIRE(dim_name(Dim9::U)         == "u");
    REQUIRE(dim_name(Dim9::V)         == "v");
    REQUIRE(dim_name(Dim9::W)         == "w");
    REQUIRE(dim_name(Dim9::X)         == "x");
    REQUIRE(dim_name(Dim9::Y)         == "y");
    REQUIRE(dim_name(Dim9::Z)         == "z");
}

// ============================================================================
// §19 — dim_symbol
// ============================================================================

TEST_CASE("GAP-041 §5: dim_symbol returns correct single characters", "[gap041][meta]") {
    REQUIRE(dim_symbol(Dim9::RESONANCE) == 'r');
    REQUIRE(dim_symbol(Dim9::STATE)     == 's');
    REQUIRE(dim_symbol(Dim9::TIME)      == 't');
    REQUIRE(dim_symbol(Dim9::U)         == 'u');
    REQUIRE(dim_symbol(Dim9::V)         == 'v');
    REQUIRE(dim_symbol(Dim9::W)         == 'w');
    REQUIRE(dim_symbol(Dim9::X)         == 'x');
    REQUIRE(dim_symbol(Dim9::Y)         == 'y');
    REQUIRE(dim_symbol(Dim9::Z)         == 'z');
}

// ============================================================================
// §20 — dim_is_integer
// ============================================================================

TEST_CASE("GAP-041 §6: dim_is_integer: only spatial dims X/Y/Z are integer-valued",
          "[gap041][type]") {
    REQUIRE_FALSE(dim_is_integer(Dim9::RESONANCE));
    REQUIRE_FALSE(dim_is_integer(Dim9::STATE));
    REQUIRE_FALSE(dim_is_integer(Dim9::TIME));
    REQUIRE_FALSE(dim_is_integer(Dim9::U));
    REQUIRE_FALSE(dim_is_integer(Dim9::V));
    REQUIRE_FALSE(dim_is_integer(Dim9::W));
    REQUIRE(dim_is_integer(Dim9::X));
    REQUIRE(dim_is_integer(Dim9::Y));
    REQUIRE(dim_is_integer(Dim9::Z));
}

// ============================================================================
// §21 — dim_is_cyclic
// ============================================================================

TEST_CASE("GAP-041 §7: dim_is_cyclic: TIME and spatial dims are toroidally wrapped",
          "[gap041][type]") {
    REQUIRE_FALSE(dim_is_cyclic(Dim9::RESONANCE));
    REQUIRE_FALSE(dim_is_cyclic(Dim9::STATE));
    REQUIRE(dim_is_cyclic(Dim9::TIME));
    REQUIRE_FALSE(dim_is_cyclic(Dim9::U));
    REQUIRE_FALSE(dim_is_cyclic(Dim9::V));
    REQUIRE_FALSE(dim_is_cyclic(Dim9::W));
    REQUIRE(dim_is_cyclic(Dim9::X));
    REQUIRE(dim_is_cyclic(Dim9::Y));
    REQUIRE(dim_is_cyclic(Dim9::Z));
}

// ============================================================================
// §22 — dim_is_complex
// ============================================================================

TEST_CASE("GAP-041 §8: dim_is_complex: only quantum dims U/V/W are complex-valued",
          "[gap041][type]") {
    REQUIRE_FALSE(dim_is_complex(Dim9::RESONANCE));
    REQUIRE_FALSE(dim_is_complex(Dim9::STATE));
    REQUIRE_FALSE(dim_is_complex(Dim9::TIME));
    REQUIRE(dim_is_complex(Dim9::U));
    REQUIRE(dim_is_complex(Dim9::V));
    REQUIRE(dim_is_complex(Dim9::W));
    REQUIRE_FALSE(dim_is_complex(Dim9::X));
    REQUIRE_FALSE(dim_is_complex(Dim9::Y));
    REQUIRE_FALSE(dim_is_complex(Dim9::Z));
}

// ============================================================================
// §23 — wave_speed_effective
// ============================================================================

TEST_CASE("GAP-041 §9: wave_speed_effective: c_eff = c0 / (1 + s)", "[gap041][physics]") {
    // spec: c_eff = c0 / (1 + ŝ),  c0 = 1.0
    REQUIRE(wave_speed_effective(0.0) == Approx(1.0));          // vacuum, max speed
    REQUIRE(wave_speed_effective(1.0) == Approx(0.5));          // moderate focus
    REQUIRE(wave_speed_effective(2.0) == Approx(1.0 / 3.0));    // deep focus, STATE_MAX

    // Strictly decreasing: higher s → slower wave
    REQUIRE(wave_speed_effective(0.5)  > wave_speed_effective(1.0));
    REQUIRE(wave_speed_effective(1.0)  > wave_speed_effective(1.5));
    REQUIRE(wave_speed_effective(1.5)  > wave_speed_effective(2.0));
}

// ============================================================================
// §24 — damping_coefficient
// ============================================================================

TEST_CASE("GAP-041 §10: damping_coefficient: gamma = alpha * (1 - r)", "[gap041][physics]") {
    constexpr double alpha = 0.5;

    // spec: γ = α × (1 − r̂)
    REQUIRE(damping_coefficient(1.0, alpha) == Approx(0.0));        // LTP — no damping
    REQUIRE(damping_coefficient(0.5, alpha) == Approx(alpha / 2.0)); // half-damped
    REQUIRE(damping_coefficient(0.0, alpha) == Approx(alpha));       // fully dissipative

    // Zero alpha → always zero damping (undamped physics, α=0 case)
    REQUIRE(damping_coefficient(0.0, 0.0) == Approx(0.0));
    REQUIRE(damping_coefficient(0.5, 0.0) == Approx(0.0));
}

// ============================================================================
// §25 — Spatial address bounds
// ============================================================================

TEST_CASE("GAP-041 §11: max_nodes_per_axis and max_spatial_nodes", "[gap041][bounds]") {
    REQUIRE(max_nodes_per_axis() == 16384);

    const std::int64_t expected = 16384LL * 16384LL * 16384LL;
    REQUIRE(max_spatial_nodes() == expected);
    // Sanity: ≈ 4.4 × 10^12 — no overflow in int64
    REQUIRE(max_spatial_nodes() > 4'000'000'000'000LL);
    REQUIRE(max_spatial_nodes() < 5'000'000'000'000LL);
}

// ============================================================================
// §26 — domain_name helper
// ============================================================================

TEST_CASE("GAP-041 §12: domain_name returns correct strings", "[gap041][meta]") {
    REQUIRE(domain_name(CoordDomain::SYSTEMIC) == "Systemic");
    REQUIRE(domain_name(CoordDomain::TEMPORAL) == "Temporal");
    REQUIRE(domain_name(CoordDomain::QUANTUM)  == "Quantum");
    REQUIRE(domain_name(CoordDomain::SPATIAL)  == "Spatial");
}

// ============================================================================
// §27 — Integration: ConsolidationScheduler 1 kHz tick simulation
// ============================================================================

TEST_CASE("GAP-024 §17: Integration — 1 kHz tick loop: consolidation fires at ~5 min",
          "[gap024][integration]") {
    ConsolidationScheduler sched;
    constexpr double DT  = 0.001;          // 1 ms per tick
    constexpr int    N   = 300'000;        // 5 minutes at 1 kHz
    int consolidations   = 0;

    for (int tick = 0; tick < N + 10; ++tick) {
        sched.advance(DT);
        if (sched.should_consolidate(0.0, /*nap=*/false, /*atp=*/0.5)) {
            ++consolidations;
            sched.on_consolidated();
        }
    }

    // In a 5-minute + 10-tick run with no perturbation, exactly one
    // consolidation should have fired (the time-based trigger at ~300 s).
    REQUIRE(consolidations == 1);
    // After reset, timer should be near zero (last 10 ticks = 0.01 s)
    REQUIRE(sched.time_since_last_update < 0.02);
}

TEST_CASE("GAP-024 §18: Integration — epiphany burst triggers early consolidation",
          "[gap024][integration]") {
    ConsolidationScheduler sched;
    sched.max_interval_sec = 300.0;

    // Simulate 1 minute of ticking with perturbation building
    for (int tick = 0; tick < 60'000; ++tick) {
        sched.advance(0.001);
        sched.update_perturbation(0.001 * (tick / 60000.0));  // slowly growing
    }
    // At 60 s: perturbation ≈ 0.001 * (60000/60000) = 0.001 → below 0.1
    REQUIRE_FALSE(sched.should_consolidate(sched.max_perturbation_norm, false, 0.5));

    // Sudden epiphany: perturbation spikes above 10%
    sched.update_perturbation(0.15);
    REQUIRE(sched.should_consolidate(0.15, false, 0.5));
    // Also correct via stored norm
    REQUIRE(sched.max_perturbation_norm == Approx(0.15));
}
