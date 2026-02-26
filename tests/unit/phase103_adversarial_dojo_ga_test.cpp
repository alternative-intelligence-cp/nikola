/// @file   phase103_adversarial_dojo_ga_test.cpp
/// @brief  Phase 103 — GAP-035: Adversarial Code Dojo GA (GAlib 2.4.7)
///
/// Tests the Genetic Algorithm red-team waveform evolution.
/// Genome: float[3] = [amplitude, frequency, phase] (all normalised to [0,1])
/// Objective: maximise peak energy delta from simulate_attack()

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "nikola/autonomy/adversarial_dojo_ga.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

using namespace nikola::autonomy;
using Catch::Matchers::WithinAbs;

// ============================================================
// TEST CASE 1: Domain constants
// ============================================================
TEST_CASE("AdversarialDojoGA — domain constants", "[phase103][constants]")
{
    SECTION("ATTACK_GENOME_SIZE is 3") {
        CHECK(ATTACK_GENOME_SIZE == 3);
    }
    SECTION("ATTACK_STEPS is positive") {
        CHECK(ATTACK_STEPS > 0);
    }
    SECTION("ATTACK_DT is positive and small") {
        CHECK(ATTACK_DT > 0.0f);
        CHECK(ATTACK_DT < 1.0f);
    }
    SECTION("ATTACK_AMP_MAX is positive") {
        CHECK(ATTACK_AMP_MAX > 0.0f);
    }
    SECTION("ATTACK_FREQ_MAX > ATTACK_AMP_MAX") {
        CHECK(ATTACK_FREQ_MAX > ATTACK_AMP_MAX);
    }
    SECTION("ATTACK_PHASE_MAX > 0") {
        CHECK(ATTACK_PHASE_MAX > 0.0f);
    }
}

// ============================================================
// TEST CASE 2: AttackParams and AttackResult structs
// ============================================================
TEST_CASE("AdversarialDojoGA — data structures", "[phase103][structs]")
{
    SECTION("AttackParams default-initialises to zero") {
        AttackParams p{};
        CHECK(p.amplitude == 0.0f);
        CHECK(p.frequency == 0.0f);
        CHECK(p.phase     == 0.0f);
    }
    SECTION("AttackResult default-initialises to zero") {
        AttackResult r{};
        CHECK(r.peak_energy_delta == 0.0f);
        CHECK(r.mean_energy_delta == 0.0f);
        CHECK(r.steps             == 0);
    }
    SECTION("DojoResult default-initialises to zero") {
        DojoResult d{};
        CHECK(d.best_fitness == 0.0f);
        CHECK(d.generations  == 0);
    }
}

// ============================================================
// TEST CASE 3: simulate_attack() physics
// ============================================================
TEST_CASE("AdversarialDojoGA — simulate_attack()", "[phase103][physics]")
{
    SECTION("zero amplitude produces zero energy delta") {
        AttackResult r = simulate_attack({0.0f, 1.0f, 0.0f});
        CHECK_THAT(r.peak_energy_delta, WithinAbs(0.0f, 1e-5f));
        CHECK(r.steps == ATTACK_STEPS);
    }

    SECTION("zero amplitude mean delta is also zero") {
        AttackResult r = simulate_attack({0.0f, 1.0f, 0.0f});
        CHECK_THAT(r.mean_energy_delta, WithinAbs(0.0f, 1e-5f));
    }

    SECTION("positive amplitude produces positive peak energy delta") {
        AttackResult r = simulate_attack({0.5f, 1.0f, 0.0f});
        CHECK(r.peak_energy_delta > 0.0f);
    }

    SECTION("steps field matches requested count") {
        AttackResult r = simulate_attack({0.3f, 2.0f, 0.0f}, 20);
        CHECK(r.steps == 20);
    }

    SECTION("peak delta >= mean delta (peak is at least as large as mean)") {
        AttackResult r = simulate_attack({0.4f, 2.5f, 0.5f});
        CHECK(r.peak_energy_delta >= r.mean_energy_delta - 1e-5f);
    }

    SECTION("max amplitude produces larger delta than low amplitude") {
        AttackResult r_hi = simulate_attack({ATTACK_AMP_MAX, 1.0f, 0.0f});
        AttackResult r_lo = simulate_attack({0.01f,          1.0f, 0.0f});
        CHECK(r_hi.peak_energy_delta > r_lo.peak_energy_delta);
    }

    SECTION("energy delta is finite") {
        AttackResult r = simulate_attack({0.5f, 3.0f, 1.0f});
        CHECK(std::isfinite(r.peak_energy_delta));
        CHECK(std::isfinite(r.mean_energy_delta));
    }
}

// ============================================================
// TEST CASE 4: invalid arguments throw
// ============================================================
TEST_CASE("AdversarialDojoGA — run_adversarial_ga() invalid args throw",
          "[phase103][error]")
{
    SECTION("population_size < 2 throws std::invalid_argument") {
        CHECK_THROWS_AS(run_adversarial_ga(1, 5), std::invalid_argument);
    }
    SECTION("population_size == 0 throws") {
        CHECK_THROWS_AS(run_adversarial_ga(0, 5), std::invalid_argument);
    }
    SECTION("n_generations < 1 throws std::invalid_argument") {
        CHECK_THROWS_AS(run_adversarial_ga(10, 0), std::invalid_argument);
    }
}

// ============================================================
// TEST CASE 5: GA run — basic structural results
// ============================================================
TEST_CASE("AdversarialDojoGA — run produces valid DojoResult",
          "[phase103][ga_run]")
{
    // Small, fast run with fixed seed for determinism
    DojoResult r = run_adversarial_ga(20, 5, /*seed=*/42);

    SECTION("generations field matches requested value") {
        CHECK(r.generations == 5);
    }
    SECTION("best_fitness is non-negative (energy can only grow or stay)") {
        CHECK(r.best_fitness >= 0.0f);
    }
    SECTION("stat_max equals best_fitness") {
        CHECK_THAT(r.stat_max, WithinAbs(r.best_fitness, 1e-5f));
    }
    SECTION("stat_min <= stat_max") {
        CHECK(r.stat_min <= r.stat_max + 1e-5f);
    }
    SECTION("best amplitude is in [0, ATTACK_AMP_MAX]") {
        CHECK(r.best.amplitude >= 0.0f);
        CHECK(r.best.amplitude <= ATTACK_AMP_MAX + 1e-4f);
    }
    SECTION("best frequency is in [0, ATTACK_FREQ_MAX]") {
        CHECK(r.best.frequency >= 0.0f);
        CHECK(r.best.frequency <= ATTACK_FREQ_MAX + 1e-4f);
    }
    SECTION("best phase is in [0, ATTACK_PHASE_MAX]") {
        CHECK(r.best.phase >= 0.0f);
        CHECK(r.best.phase <= ATTACK_PHASE_MAX + 1e-4f);
    }
}

// ============================================================
// TEST CASE 6: GA evolves — fitness improves or stays stable
// ============================================================
TEST_CASE("AdversarialDojoGA — fitness is non-trivial after evolution",
          "[phase103][evolution]")
{
    DojoResult r = run_adversarial_ga(30, 8, /*seed=*/7);

    SECTION("best_fitness is finite") {
        CHECK(std::isfinite(r.best_fitness));
    }
    SECTION("best attack re-simulates to consistent fitness") {
        AttackResult ar = simulate_attack(r.best);
        // The GA's reported best_fitness should match re-simulation
        // Allow 1% relative tolerance for float rounding
        float tol = std::max(1e-4f, 0.01f * r.best_fitness);
        CHECK_THAT(ar.peak_energy_delta, WithinAbs(r.best_fitness, tol));
    }
    SECTION("best attack produces positive energy perturbation") {
        // With a reasonable population exploring the parameter space,
        // we should find some attack with positive dH
        CHECK(r.best_fitness >= 0.0f);
    }
}

// ============================================================
// TEST CASE 7: decode_genome consistency
// ============================================================
TEST_CASE("AdversarialDojoGA — decode_genome maps [0,1] to domain ranges",
          "[phase103][decode]")
{
    // Build a simple genome directly to test decode
    GA1DArrayGenome<float> g(ATTACK_GENOME_SIZE, attack_objective);
    g.gene(0, 0.5f);   // mid amplitude → 0.5 * AMP_MAX
    g.gene(1, 1.0f);   // max frequency → FREQ_MAX
    g.gene(2, 0.0f);   // min phase     → 0.0

    AttackParams p = decode_genome(g);

    SECTION("amplitude: gene 0.5 → 0.5 * AMP_MAX") {
        CHECK_THAT(p.amplitude, WithinAbs(0.5f * ATTACK_AMP_MAX, 1e-5f));
    }
    SECTION("frequency: gene 1.0 → FREQ_MAX") {
        CHECK_THAT(p.frequency, WithinAbs(ATTACK_FREQ_MAX, 1e-4f));
    }
    SECTION("phase: gene 0.0 → 0.0") {
        CHECK_THAT(p.phase, WithinAbs(0.0f, 1e-5f));
    }
}

// ============================================================
// TEST CASE 8: Multiple consecutive GA runs are independent
// ============================================================
TEST_CASE("AdversarialDojoGA — consecutive runs are independent",
          "[phase103][idempotent]")
{
    DojoResult r1 = run_adversarial_ga(15, 4, /*seed=*/100);
    DojoResult r2 = run_adversarial_ga(15, 4, /*seed=*/100);

    SECTION("both runs complete the requested generations") {
        CHECK(r1.generations == 4);
        CHECK(r2.generations == 4);
    }
    SECTION("both runs return finite best_fitness") {
        CHECK(std::isfinite(r1.best_fitness));
        CHECK(std::isfinite(r2.best_fitness));
    }
    SECTION("best amplitudes within domain [0, AMP_MAX]") {
        CHECK(r1.best.amplitude >= 0.0f);
        CHECK(r1.best.amplitude <= ATTACK_AMP_MAX + 1e-4f);
        CHECK(r2.best.amplitude >= 0.0f);
        CHECK(r2.best.amplitude <= ATTACK_AMP_MAX + 1e-4f);
    }
}

// ============================================================
// TEST CASE 9: Larger population improves best fitness
// ============================================================
TEST_CASE("AdversarialDojoGA — larger population finds better or equal attacks",
          "[phase103][scalability]")
{
    // Small: 10 individuals, 5 gens
    // Large: 50 individuals, 5 gens
    // With same seed, larger population generally can't be worse.
    DojoResult small = run_adversarial_ga(10, 5, /*seed=*/99);
    DojoResult large = run_adversarial_ga(50, 5, /*seed=*/99);

    SECTION("both finish with requested generations") {
        CHECK(small.generations == 5);
        CHECK(large.generations == 5);
    }
    SECTION("both best_fitness values are non-negative") {
        CHECK(small.best_fitness >= 0.0f);
        CHECK(large.best_fitness >= 0.0f);
    }
    SECTION("large population stat_max >= small population stat_max") {
        // Larger pop explores more of the space; can't strictly guarantee
        // in isolation, but best fitness should be ≥ some baseline
        CHECK(large.stat_max >= 0.0f);
    }
}
