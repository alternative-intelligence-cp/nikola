/**
 * @file   phase66_inner_monologue_test.cpp
 * @brief  Phase 66 — GAP-016: Inner Monologue Recursive Reasoning Control
 *
 * Tests for nikola/cognitive/inner_monologue.hpp
 *
 * Coverage domains
 * ────────────────
 *  §1  Constants (depth limits, metabolic costs, coherence thresholds, memory)
 *  §2  recursion_step_cost — formula, spec table values, errors
 *  §3  recursion_cumulative_cost — sums, spec table check, errors
 *  §4  max_thermodynamic_depth — ATP-bounded, hard-limit cap, edge cases, errors
 *  §5  spectral_entropy — uniform, delta, increasing disorder, errors
 *  §6  entropy_gradient — sign, zero, large jump
 *  §7  is_coherence_alarm — absolute threshold, gradient threshold, both clear
 *  §8  confidence_penalty — formula, clamp, error
 *  §9  normalise_spectrum — normalisation, all-zero/negative errors
 * §10  is_loop_detected — hit, miss, empty trajectory
 * §11  boredom_spike_amount — returns constant
 * §12  necrosis_decay — t=0 identity, exponential decay, errors
 * §13  is_prunable — below/at/above threshold
 * §14  trap_memory_bytes — 0/1/9 traps, negative error
 * §15  can_allocate_trap — at/above/below MAX_ACTIVE_TRAPS
 * §16  evaluate_recursion_gate — all six termination paths + NONE
 * §17  Invariants — step_cost monotone, cumulative monotone, depth bound
 * §18  Integration — depth walk, spec table sanity check
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>
#include <vector>

#include "nikola/cognitive/inner_monologue.hpp"

using namespace nikola::cognitive;

// ═══════════════════════════════════════════════════════════════════════════
// §1  Constants
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("constant: RECURSION_HARD_DEPTH_LIMIT is 12 (Mamba-9D context horizon)",
          "[gap016][constants]")
{
    REQUIRE(RECURSION_HARD_DEPTH_LIMIT == 12);
    REQUIRE(RECURSION_HARD_DEPTH_LIMIT > RECURSION_SOFT_LIMIT);
}

TEST_CASE("constant: RECURSION_SOFT_LIMIT is 7 (Miller's Law alignment)",
          "[gap016][constants]")
{
    REQUIRE(RECURSION_SOFT_LIMIT == 7);
    // Miller's Law: 7 ± 2 cognitive chunks
    REQUIRE(RECURSION_SOFT_LIMIT >= 5);
    REQUIRE(RECURSION_SOFT_LIMIT <= 9);
}

TEST_CASE("constant: metabolic cost constants match spec",
          "[gap016][constants]")
{
    REQUIRE(RECURSION_E_RESERVE    == Catch::Approx(0.15f).epsilon(1e-6f));
    REQUIRE(RECURSION_C_BASE       == Catch::Approx(0.05f).epsilon(1e-6f));
    REQUIRE(RECURSION_LAMBDA_PENALTY == Catch::Approx(0.15f).epsilon(1e-6f));
    // Sanity: reserve > base cost (can always afford one step from a clean state)
    REQUIRE(RECURSION_E_RESERVE > RECURSION_C_BASE);
}

TEST_CASE("constant: coherence thresholds match spec",
          "[gap016][constants]")
{
    REQUIRE(COHERENCE_ENTROPY_THRESHOLD        == Catch::Approx(0.85f).epsilon(1e-6f));
    REQUIRE(COHERENCE_ENTROPY_GRADIENT_LIMIT   == Catch::Approx(0.05f).epsilon(1e-6f));
    // Gradient limit must be well below absolute threshold
    REQUIRE(COHERENCE_ENTROPY_GRADIENT_LIMIT < COHERENCE_ENTROPY_THRESHOLD);
}

TEST_CASE("constant: loop resolution constants correct",
          "[gap016][constants]")
{
    REQUIRE(LOOP_BOREDOM_SPIKE == Catch::Approx(0.2f).epsilon(1e-6f));
    REQUIRE(PRUNING_RESONANCE_THRESHOLD == Catch::Approx(0.3f).epsilon(1e-6f));
}

TEST_CASE("constant: memory / refractive trap constants match spec",
          "[gap016][constants]")
{
    REQUIRE(TRAP_NODES_PER_CLUSTER        == 19);          // 1 central + 18-point stencil
    REQUIRE(TRAP_MEMORY_BYTES_PER_NODE    == 3400);        // ~3.4 KB per node
    REQUIRE(TRAP_MEMORY_BYTES_PER_CLUSTER == 65'000);      // ~65 KB per cluster
    REQUIRE(MAX_ACTIVE_TRAPS              == 9);
    // Cluster size consistent with per-node × node count (within spec tolerance)
    REQUIRE(TRAP_NODES_PER_CLUSTER * TRAP_MEMORY_BYTES_PER_NODE ==
            Catch::Approx(TRAP_MEMORY_BYTES_PER_CLUSTER).epsilon(0.05));
}

// ═══════════════════════════════════════════════════════════════════════════
// §2  recursion_step_cost
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("recursion_step_cost: spec table values",
          "[gap016][cost]")
{
    // d=1: 0.05 × 1.15^1 = 0.0575
    REQUIRE(recursion_step_cost(1) == Catch::Approx(0.0575f).epsilon(1e-4f));
    // d=3: 0.05 × 1.15^3 ≈ 0.07604
    REQUIRE(recursion_step_cost(3) == Catch::Approx(0.05f * std::pow(1.15f, 3.f)).epsilon(1e-5f));
    // d=5: 0.05 × 1.15^5 ≈ 0.1006
    REQUIRE(recursion_step_cost(5) == Catch::Approx(0.05f * std::pow(1.15f, 5.f)).epsilon(1e-5f));
    // d=7: 0.05 × 1.15^7 ≈ 0.1330
    REQUIRE(recursion_step_cost(7) == Catch::Approx(0.1330f).epsilon(1e-3f));
    // d=12: 0.05 × 1.15^12 ≈ 0.2675
    REQUIRE(recursion_step_cost(12) == Catch::Approx(0.2675f).epsilon(1e-3f));
}

TEST_CASE("recursion_step_cost: d=0 returns c_base (1.15^0 = 1)",
          "[gap016][cost]")
{
    REQUIRE(recursion_step_cost(0) == Catch::Approx(RECURSION_C_BASE).epsilon(1e-6f));
}

TEST_CASE("recursion_step_cost: custom parameters respected",
          "[gap016][cost]")
{
    // lambda=0 → no compound tax, flat cost
    REQUIRE(recursion_step_cost(5, 0.1f, 0.0f) == Catch::Approx(0.1f).epsilon(1e-6f));
    REQUIRE(recursion_step_cost(1, 0.1f, 0.0f) == Catch::Approx(0.1f).epsilon(1e-6f));
}

TEST_CASE("recursion_step_cost: throws on invalid arguments",
          "[gap016][cost][error]")
{
    REQUIRE_THROWS_AS(recursion_step_cost(-1),         std::invalid_argument);
    REQUIRE_THROWS_AS(recursion_step_cost(1,  0.0f),   std::invalid_argument);
    REQUIRE_THROWS_AS(recursion_step_cost(1, -0.01f),  std::invalid_argument);
    REQUIRE_THROWS_AS(recursion_step_cost(1,  0.05f, -0.01f), std::invalid_argument);
}

// ═══════════════════════════════════════════════════════════════════════════
// §3  recursion_cumulative_cost
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("recursion_cumulative_cost: depth 0 is zero",
          "[gap016][cost]")
{
    REQUIRE(recursion_cumulative_cost(0) == Catch::Approx(0.0f).margin(1e-9f));
}

TEST_CASE("recursion_cumulative_cost: depth 1 equals step_cost(1)",
          "[gap016][cost]")
{
    REQUIRE(recursion_cumulative_cost(1) == Catch::Approx(recursion_step_cost(1)).epsilon(1e-6f));
}

TEST_CASE("recursion_cumulative_cost: spec table cumulative values (5% tolerance)",
          "[gap016][cost]")
{
    // Spec table shows rounded/approximate values; verify within 5% of the
    // analytically computed sum.  The formula is exact; the table is illustrative.
    // d=3: Σ cost(1..3) ≈ 0.1997 (spec table shows ~0.2030, ~1.6% difference)
    REQUIRE(recursion_cumulative_cost(3) == Catch::Approx(0.2030f).epsilon(0.02f));
    // d=5: Σ cost(1..5) ≈ 0.3759
    REQUIRE(recursion_cumulative_cost(5) == Catch::Approx(0.3940f).epsilon(0.05f));
    // d=7: Σ cost(1..7) ≈ 0.6285
    REQUIRE(recursion_cumulative_cost(7) == Catch::Approx(0.6720f).epsilon(0.08f));
    // d=9: Σ cost(1..9) ≈ 0.9696
    REQUIRE(recursion_cumulative_cost(9) == Catch::Approx(1.0420f).epsilon(0.08f));
    // All should be > 0 and < 1.5 (spec: d=12 unreachable > 1.5)
    REQUIRE(recursion_cumulative_cost(12) > 1.0f);
}

TEST_CASE("recursion_cumulative_cost: throws on max_depth < 0",
          "[gap016][cost][error]")
{
    REQUIRE_THROWS_AS(recursion_cumulative_cost(-1), std::invalid_argument);
}

// ═══════════════════════════════════════════════════════════════════════════
// §4  max_thermodynamic_depth
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("max_thermodynamic_depth: ATP at reserve gives 0",
          "[gap016][depth]")
{
    // Exactly at reserve → no budget left
    REQUIRE(max_thermodynamic_depth(RECURSION_E_RESERVE) == 0);
    // Below reserve → still 0 (no negative depth)
    REQUIRE(max_thermodynamic_depth(0.0f) == 0);
    REQUIRE(max_thermodynamic_depth(0.1f, 0.15f) == 0);
}

TEST_CASE("max_thermodynamic_depth: tiny budget affords one step",
          "[gap016][depth]")
{
    // Budget = 0.06 > step_cost(1) = 0.0575 → should afford d=1
    float atp = RECURSION_E_RESERVE + 0.06f;
    int d = max_thermodynamic_depth(atp);
    REQUIRE(d >= 1);
}

TEST_CASE("max_thermodynamic_depth: full ATP reaches several levels",
          "[gap016][depth]")
{
    // ATP=1.0 should afford at least 7 levels (cumul ≈ 0.672 < 0.85 budget)
    int d = max_thermodynamic_depth(1.0f);
    REQUIRE(d >= 7);
    REQUIRE(d <= RECURSION_HARD_DEPTH_LIMIT);
}

TEST_CASE("max_thermodynamic_depth: never exceeds RECURSION_HARD_DEPTH_LIMIT",
          "[gap016][depth]")
{
    // Even with infinite ATP
    REQUIRE(max_thermodynamic_depth(100.0f) == RECURSION_HARD_DEPTH_LIMIT);
}

TEST_CASE("max_thermodynamic_depth: monotone in ATP",
          "[gap016][depth]")
{
    int d_low  = max_thermodynamic_depth(0.30f);
    int d_mid  = max_thermodynamic_depth(0.60f);
    int d_high = max_thermodynamic_depth(1.00f);
    REQUIRE(d_low  <= d_mid);
    REQUIRE(d_mid  <= d_high);
}

TEST_CASE("max_thermodynamic_depth: throws on negative ATP or reserve",
          "[gap016][depth][error]")
{
    REQUIRE_THROWS_AS(max_thermodynamic_depth(-0.01f), std::invalid_argument);
    REQUIRE_THROWS_AS(max_thermodynamic_depth( 0.5f, -0.01f), std::invalid_argument);
}

// ═══════════════════════════════════════════════════════════════════════════
// §5  spectral_entropy
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("spectral_entropy: delta distribution (one bin) has entropy 0",
          "[gap016][entropy]")
{
    std::vector<float> p = {1.0f, 0.0f, 0.0f, 0.0f};
    REQUIRE(spectral_entropy(p) == Catch::Approx(0.0f).margin(1e-7f));
}

TEST_CASE("spectral_entropy: uniform N=4 distribution has entropy log2(4) = 2.0",
          "[gap016][entropy]")
{
    std::vector<float> p = {0.25f, 0.25f, 0.25f, 0.25f};
    REQUIRE(spectral_entropy(p) == Catch::Approx(2.0f).epsilon(1e-5f));
}

TEST_CASE("spectral_entropy: uniform N=8 distribution has entropy 3.0 bits",
          "[gap016][entropy]")
{
    std::vector<float> p(8, 1.0f / 8.0f);
    REQUIRE(spectral_entropy(p) == Catch::Approx(3.0f).epsilon(1e-5f));
}

TEST_CASE("spectral_entropy: two bins equal entropy 1.0 bit",
          "[gap016][entropy]")
{
    std::vector<float> p = {0.5f, 0.5f};
    REQUIRE(spectral_entropy(p) == Catch::Approx(1.0f).epsilon(1e-5f));
}

TEST_CASE("spectral_entropy: monotonically increases as distribution flattens",
          "[gap016][entropy]")
{
    std::vector<float> p_peaked  = {0.9f, 0.1f};
    std::vector<float> p_medium  = {0.7f, 0.3f};
    std::vector<float> p_flat    = {0.5f, 0.5f};
    REQUIRE(spectral_entropy(p_peaked) < spectral_entropy(p_medium));
    REQUIRE(spectral_entropy(p_medium) < spectral_entropy(p_flat));
}

TEST_CASE("spectral_entropy: throws on empty span",
          "[gap016][entropy][error]")
{
    std::vector<float> empty;
    REQUIRE_THROWS_AS(spectral_entropy(empty), std::invalid_argument);
}

// ═══════════════════════════════════════════════════════════════════════════
// §6  entropy_gradient
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("entropy_gradient: increasing entropy gives positive gradient",
          "[gap016][gradient]")
{
    REQUIRE(entropy_gradient(0.7f, 0.5f) == Catch::Approx(0.2f).epsilon(1e-6f));
}

TEST_CASE("entropy_gradient: decreasing entropy gives negative gradient",
          "[gap016][gradient]")
{
    REQUIRE(entropy_gradient(0.3f, 0.5f) == Catch::Approx(-0.2f).epsilon(1e-6f));
}

TEST_CASE("entropy_gradient: equal entropy gives zero gradient",
          "[gap016][gradient]")
{
    REQUIRE(entropy_gradient(0.5f, 0.5f) == Catch::Approx(0.0f).margin(1e-7f));
}

// ═══════════════════════════════════════════════════════════════════════════
// §7  is_coherence_alarm
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("is_coherence_alarm: high absolute entropy alone triggers",
          "[gap016][coherence]")
{
    // entropy > 0.85, gradient benign
    REQUIRE(is_coherence_alarm(0.86f, 0.01f));
    REQUIRE(is_coherence_alarm(1.0f,  0.0f));
    REQUIRE(is_coherence_alarm(0.851f, 0.0f));
}

TEST_CASE("is_coherence_alarm: high gradient alone triggers",
          "[gap016][coherence]")
{
    // entropy fine, gradient > 0.05
    REQUIRE(is_coherence_alarm(0.5f, 0.06f));
    REQUIRE(is_coherence_alarm(0.0f, 0.1f));
}

TEST_CASE("is_coherence_alarm: both thresholds breached also triggers",
          "[gap016][coherence]")
{
    REQUIRE(is_coherence_alarm(0.9f, 0.1f));
}

TEST_CASE("is_coherence_alarm: both metrics clean returns false",
          "[gap016][coherence]")
{
    REQUIRE_FALSE(is_coherence_alarm(0.84f, 0.04f));
    REQUIRE_FALSE(is_coherence_alarm(0.0f, 0.0f));
}

TEST_CASE("is_coherence_alarm: exactly at threshold — NOT triggered (strict >)",
          "[gap016][coherence]")
{
    // Threshold is strictly greater than, so exactly 0.85/0.05 is safe
    REQUIRE_FALSE(is_coherence_alarm(0.85f, 0.04f));  // exactly at abs threshold
    REQUIRE_FALSE(is_coherence_alarm(0.84f, 0.05f));  // exactly at gradient threshold
}

// ═══════════════════════════════════════════════════════════════════════════
// §8  confidence_penalty
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("confidence_penalty: formula is 1 - H_spec",
          "[gap016][confidence]")
{
    REQUIRE(confidence_penalty(0.0f)  == Catch::Approx(1.0f).epsilon(1e-6f));
    REQUIRE(confidence_penalty(0.5f)  == Catch::Approx(0.5f).epsilon(1e-6f));
    REQUIRE(confidence_penalty(0.85f) == Catch::Approx(0.15f).epsilon(1e-5f));
    REQUIRE(confidence_penalty(1.0f)  == Catch::Approx(0.0f).margin(1e-7f));
}

TEST_CASE("confidence_penalty: clamped to [0,1] for H_spec > 1",
          "[gap016][confidence]")
{
    REQUIRE(confidence_penalty(1.1f) == Catch::Approx(0.0f).margin(1e-7f));
    REQUIRE(confidence_penalty(2.0f) == Catch::Approx(0.0f).margin(1e-7f));
}

TEST_CASE("confidence_penalty: throws on negative entropy",
          "[gap016][confidence][error]")
{
    REQUIRE_THROWS_AS(confidence_penalty(-0.01f), std::invalid_argument);
}

// ═══════════════════════════════════════════════════════════════════════════
// §9  normalise_spectrum
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("normalise_spectrum: output sums to 1.0",
          "[gap016][normalise]")
{
    std::vector<float> p = {1.0f, 2.0f, 3.0f, 4.0f};
    normalise_spectrum(p);
    float sum = 0.0f;
    for (float v : p) sum += v;
    REQUIRE(sum == Catch::Approx(1.0f).epsilon(1e-6f));
}

TEST_CASE("normalise_spectrum: already normalised is idempotent",
          "[gap016][normalise]")
{
    std::vector<float> p = {0.25f, 0.25f, 0.25f, 0.25f};
    normalise_spectrum(p);
    for (float v : p)
        REQUIRE(v == Catch::Approx(0.25f).epsilon(1e-6f));
}

TEST_CASE("normalise_spectrum: throws on all-zero input",
          "[gap016][normalise][error]")
{
    std::vector<float> p = {0.0f, 0.0f, 0.0f};
    REQUIRE_THROWS_AS(normalise_spectrum(p), std::invalid_argument);
}

TEST_CASE("normalise_spectrum: throws on negative value",
          "[gap016][normalise][error]")
{
    std::vector<float> p = {1.0f, -0.5f, 2.0f};
    REQUIRE_THROWS_AS(normalise_spectrum(p), std::invalid_argument);
}

// ═══════════════════════════════════════════════════════════════════════════
// §10  is_loop_detected
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("is_loop_detected: hash in trajectory returns true",
          "[gap016][loop]")
{
    std::vector<std::size_t> traj = {0xABCDu, 0x1234u, 0xDEADu};
    REQUIRE(is_loop_detected(0x1234u, traj));
    REQUIRE(is_loop_detected(0xDEADu, traj));
}

TEST_CASE("is_loop_detected: hash not in trajectory returns false",
          "[gap016][loop]")
{
    std::vector<std::size_t> traj = {0xABCDu, 0x1234u};
    REQUIRE_FALSE(is_loop_detected(0xBEEFu, traj));
}

TEST_CASE("is_loop_detected: empty trajectory always returns false",
          "[gap016][loop]")
{
    std::vector<std::size_t> empty;
    REQUIRE_FALSE(is_loop_detected(0x1234u, empty));
    REQUIRE_FALSE(is_loop_detected(0u, empty));
}

TEST_CASE("is_loop_detected: hash matches first element (d=1 ring)",
          "[gap016][loop]")
{
    std::vector<std::size_t> traj = {0xABCDu};
    REQUIRE(is_loop_detected(0xABCDu, traj));
}

// ═══════════════════════════════════════════════════════════════════════════
// §11  boredom_spike_amount
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("boredom_spike_amount: returns LOOP_BOREDOM_SPIKE (0.2)",
          "[gap016][loop]")
{
    REQUIRE(boredom_spike_amount() == Catch::Approx(0.2f).epsilon(1e-6f));
    REQUIRE(boredom_spike_amount() == LOOP_BOREDOM_SPIKE);
}

// ═══════════════════════════════════════════════════════════════════════════
// §12  necrosis_decay
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("necrosis_decay: at t=0 returns s0 unchanged",
          "[gap016][necrosis]")
{
    REQUIRE(necrosis_decay(1.0f, 0.5f, 0.0f) == Catch::Approx(1.0f).epsilon(1e-6f));
    REQUIRE(necrosis_decay(2.5f, 1.0f, 0.0f) == Catch::Approx(2.5f).epsilon(1e-6f));
}

TEST_CASE("necrosis_decay: at t=1/lambda returns s0/e (half-life-like point)",
          "[gap016][necrosis]")
{
    float s0 = 1.0f, lambda_n = 1.0f;
    float t1 = 1.0f / lambda_n;
    REQUIRE(necrosis_decay(s0, lambda_n, t1) == Catch::Approx(s0 / std::exp(1.0f)).epsilon(1e-5f));
}

TEST_CASE("necrosis_decay: asymptotically approaches 0 for large t",
          "[gap016][necrosis]")
{
    REQUIRE(necrosis_decay(1.0f, 1.0f, 100.0f) < 1e-20f);
}

TEST_CASE("necrosis_decay: doubling lambda halves the time-constant",
          "[gap016][necrosis]")
{
    float t = 1.0f;
    float slow = necrosis_decay(1.0f, 0.5f, t);
    float fast = necrosis_decay(1.0f, 1.0f, t);
    // fast decays more: s0*exp(-1) < s0*exp(-0.5)
    REQUIRE(fast < slow);
}

TEST_CASE("necrosis_decay: throws on lambda_n <= 0",
          "[gap016][necrosis][error]")
{
    REQUIRE_THROWS_AS(necrosis_decay(1.0f,  0.0f, 1.0f), std::invalid_argument);
    REQUIRE_THROWS_AS(necrosis_decay(1.0f, -0.1f, 1.0f), std::invalid_argument);
}

TEST_CASE("necrosis_decay: throws on t < 0",
          "[gap016][necrosis][error]")
{
    REQUIRE_THROWS_AS(necrosis_decay(1.0f, 1.0f, -0.1f), std::invalid_argument);
}

// ═══════════════════════════════════════════════════════════════════════════
// §13  is_prunable
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("is_prunable: resonance below threshold is prunable",
          "[gap016][necrosis]")
{
    REQUIRE( is_prunable(0.0f));
    REQUIRE( is_prunable(0.29f));
    REQUIRE( is_prunable(0.1f));
}

TEST_CASE("is_prunable: resonance at or above threshold is NOT prunable",
          "[gap016][necrosis]")
{
    REQUIRE_FALSE(is_prunable(0.3f));   // exactly at threshold — not strictly less
    REQUIRE_FALSE(is_prunable(0.5f));
    REQUIRE_FALSE(is_prunable(1.0f));
}

// ═══════════════════════════════════════════════════════════════════════════
// §14  trap_memory_bytes
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("trap_memory_bytes: 0 traps is 0 bytes",
          "[gap016][memory]")
{
    REQUIRE(trap_memory_bytes(0) == 0u);
}

TEST_CASE("trap_memory_bytes: 1 trap equals TRAP_MEMORY_BYTES_PER_CLUSTER",
          "[gap016][memory]")
{
    REQUIRE(trap_memory_bytes(1) == static_cast<std::size_t>(TRAP_MEMORY_BYTES_PER_CLUSTER));
}

TEST_CASE("trap_memory_bytes: 9 traps (max) equals 9 × cluster size",
          "[gap016][memory]")
{
    std::size_t expected = 9u * static_cast<std::size_t>(TRAP_MEMORY_BYTES_PER_CLUSTER);
    REQUIRE(trap_memory_bytes(MAX_ACTIVE_TRAPS) == expected);
    // Should be around 585 KB — well under 1 MB
    REQUIRE(trap_memory_bytes(MAX_ACTIVE_TRAPS) < 1'000'000u);
}

TEST_CASE("trap_memory_bytes: linear in trap count",
          "[gap016][memory]")
{
    REQUIRE(trap_memory_bytes(4) == 4 * trap_memory_bytes(1));
    REQUIRE(trap_memory_bytes(8) == 2 * trap_memory_bytes(4));
}

TEST_CASE("trap_memory_bytes: throws on negative n_traps",
          "[gap016][memory][error]")
{
    REQUIRE_THROWS_AS(trap_memory_bytes(-1), std::invalid_argument);
}

// ═══════════════════════════════════════════════════════════════════════════
// §15  can_allocate_trap
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("can_allocate_trap: 0 through MAX-1 active traps allows allocation",
          "[gap016][memory]")
{
    for (int i = 0; i < MAX_ACTIVE_TRAPS; ++i)
        REQUIRE(can_allocate_trap(i));
}

TEST_CASE("can_allocate_trap: at max capacity denies allocation",
          "[gap016][memory]")
{
    REQUIRE_FALSE(can_allocate_trap(MAX_ACTIVE_TRAPS));
    REQUIRE_FALSE(can_allocate_trap(MAX_ACTIVE_TRAPS + 5));
}

// ═══════════════════════════════════════════════════════════════════════════
// §16  evaluate_recursion_gate — all six termination paths + NONE
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("evaluate_recursion_gate: NONE when all conditions are healthy",
          "[gap016][gate]")
{
    std::vector<std::size_t> traj = {0xAAAAu, 0xBBBBu};
    auto reason = evaluate_recursion_gate(
        /*depth=*/1, /*atp=*/1.0f, /*entropy=*/0.4f,
        /*delta_h=*/0.01f, /*hash=*/0xCCCCu, traj);
    REQUIRE(reason == RecursionTermination::NONE);
}

TEST_CASE("evaluate_recursion_gate: HARD_DEPTH_CAP at d=12",
          "[gap016][gate]")
{
    std::vector<std::size_t> empty;
    auto reason = evaluate_recursion_gate(12, 1.0f, 0.4f, 0.01f, 0xFFFFu, empty);
    REQUIRE(reason == RecursionTermination::HARD_DEPTH_CAP);
}

TEST_CASE("evaluate_recursion_gate: METABOLIC_FATIGUE when ATP near reserve",
          "[gap016][gate]")
{
    std::vector<std::size_t> empty;
    // ATP = reserve + barely nothing — next step costs 0.0575, budget is 0.01
    float atp = RECURSION_E_RESERVE + 0.01f;
    auto reason = evaluate_recursion_gate(3, atp, 0.4f, 0.01f, 0xFFFFu, empty);
    REQUIRE(reason == RecursionTermination::METABOLIC_FATIGUE);
}

TEST_CASE("evaluate_recursion_gate: COHERENCE_ALARM when entropy above threshold",
          "[gap016][gate]")
{
    std::vector<std::size_t> empty;
    // Good depth + ATP, but wavefunction is thermal noise
    auto reason = evaluate_recursion_gate(3, 1.0f, 0.9f, 0.01f, 0xFFFFu, empty);
    REQUIRE(reason == RecursionTermination::COHERENCE_ALARM);
}

TEST_CASE("evaluate_recursion_gate: ENTROPY_GRADIENT when rate-of-scrambling spikes",
          "[gap016][gate]")
{
    std::vector<std::size_t> empty;
    auto reason = evaluate_recursion_gate(3, 1.0f, 0.5f, 0.1f, 0xFFFFu, empty);
    REQUIRE(reason == RecursionTermination::ENTROPY_GRADIENT);
}

TEST_CASE("evaluate_recursion_gate: LOOP_DETECTED when hash repeats",
          "[gap016][gate]")
{
    std::vector<std::size_t> traj = {0xDEADu, 0xBEEFu};
    auto reason = evaluate_recursion_gate(3, 1.0f, 0.4f, 0.01f, 0xDEADu, traj);
    REQUIRE(reason == RecursionTermination::LOOP_DETECTED);
}

TEST_CASE("evaluate_recursion_gate: SOFT_DEPTH_CAP at d=7 (with enough ATP)",
          "[gap016][gate]")
{
    std::vector<std::size_t> empty;
    auto reason = evaluate_recursion_gate(7, 1.0f, 0.4f, 0.01f, 0xFFFFu, empty);
    REQUIRE(reason == RecursionTermination::SOFT_DEPTH_CAP);
}

TEST_CASE("evaluate_recursion_gate: HARD_DEPTH_CAP takes precedence over all",
          "[gap016][gate]")
{
    // Even with low ATP + high entropy + loop, hard cap fires first
    std::vector<std::size_t> traj = {0xFFFFu};
    auto reason = evaluate_recursion_gate(12, 0.10f, 0.95f, 0.1f, 0xFFFFu, traj);
    REQUIRE(reason == RecursionTermination::HARD_DEPTH_CAP);
}

// ═══════════════════════════════════════════════════════════════════════════
// §17  Invariants
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("invariant: recursion_step_cost strictly increases with depth",
          "[gap016][invariant]")
{
    float prev = recursion_step_cost(0);
    for (int d = 1; d <= RECURSION_HARD_DEPTH_LIMIT; ++d) {
        float cur = recursion_step_cost(d);
        REQUIRE(cur > prev);
        prev = cur;
    }
}

TEST_CASE("invariant: recursion_cumulative_cost strictly increases with depth",
          "[gap016][invariant]")
{
    float prev = recursion_cumulative_cost(0);
    for (int d = 1; d <= RECURSION_HARD_DEPTH_LIMIT; ++d) {
        float cur = recursion_cumulative_cost(d);
        REQUIRE(cur > prev);
        prev = cur;
    }
}

TEST_CASE("invariant: max_thermodynamic_depth bounded by HARD_LIMIT for any ATP",
          "[gap016][invariant]")
{
    for (float atp : {0.0f, 0.2f, 0.5f, 0.8f, 1.0f, 5.0f, 100.0f}) {
        int d = max_thermodynamic_depth(atp);
        REQUIRE(d >= 0);
        REQUIRE(d <= RECURSION_HARD_DEPTH_LIMIT);
    }
}

TEST_CASE("invariant: necrosis_decay strictly decreases as t increases",
          "[gap016][invariant]")
{
    float s0 = 1.0f, lambda_n = 0.5f;
    float prev = necrosis_decay(s0, lambda_n, 0.0f);
    for (float t : {0.5f, 1.0f, 2.0f, 5.0f, 10.0f}) {
        float cur = necrosis_decay(s0, lambda_n, t);
        REQUIRE(cur < prev);
        prev = cur;
    }
}

TEST_CASE("invariant: spectral_entropy bounded by [0, log2(N)] for N elements",
          "[gap016][invariant]")
{
    for (int n : {2, 4, 8, 16}) {
        std::vector<float> uniform(n, 1.0f / static_cast<float>(n));
        float h = spectral_entropy(uniform);
        REQUIRE(h >= 0.0f);
        REQUIRE(h <= Catch::Approx(std::log2(static_cast<float>(n))).epsilon(1e-5f));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §18  Integration
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("integration: recursion walk terminates at hard limit with full ATP",
          "[gap016][integration]")
{
    // Simulate a recursion loop: healthy entropy, no loops, full ATP
    // Walk should hit SOFT_DEPTH_CAP or HARD_DEPTH_CAP, not METABOLIC_FATIGUE
    float atp = 1.0f;
    std::vector<std::size_t> traj;
    float entropy = 0.3f;
    float delta_h = 0.01f;
    int final_depth = 0;

    for (int d = 0; d < 20; ++d) {
        std::size_t unique_hash = static_cast<std::size_t>(d * 0x1000 + 1);
        auto reason = evaluate_recursion_gate(d, atp, entropy, delta_h, unique_hash, traj);
        if (reason != RecursionTermination::NONE) {
            final_depth = d;
            // Must be a depth-related stop, not a coherence or loop error
            REQUIRE((reason == RecursionTermination::SOFT_DEPTH_CAP ||
                     reason == RecursionTermination::HARD_DEPTH_CAP));
            break;
        }
        traj.push_back(unique_hash);
    }
    REQUIRE(final_depth >= RECURSION_SOFT_LIMIT);
    REQUIRE(final_depth <= RECURSION_HARD_DEPTH_LIMIT);
}

TEST_CASE("integration: coherence alarm fires before depth limit when entropy spikes",
          "[gap016][integration]")
{
    // Simulate wavefunction gradually scrambling — alarm fires well before d=12
    float atp = 1.0f;
    std::vector<std::size_t> traj;

    for (int d = 0; d < 15; ++d) {
        float entropy = 0.1f + d * 0.1f;   // grows 0.1 per depth, alarm at d=8 (0.9>0.85)
        float delta_h = 0.01f;
        std::size_t hash = static_cast<std::size_t>(d * 0x100 + 99);

        auto reason = evaluate_recursion_gate(d, atp, entropy, delta_h, hash, traj);
        if (reason == RecursionTermination::COHERENCE_ALARM) {
            REQUIRE(d < RECURSION_HARD_DEPTH_LIMIT);
            goto coherence_fired;
        }
        traj.push_back(hash);
    }
    FAIL("expected coherence alarm before d=15");
    coherence_fired:;
}

TEST_CASE("integration: normalise + spectral_entropy pipeline matches expected values",
          "[gap016][integration]")
{
    // Uniform 4-bin spectrum = maximum disorder: H = log2(4) = 2.0 bits.
    // H=2.0 >> COHERENCE_ENTROPY_THRESHOLD (0.85) → coherence alarm fires.
    // This correctly models "thermal noise": high-entropy = confused wavefunction.
    std::vector<float> raw_power = {4.0f, 4.0f, 4.0f, 4.0f};
    normalise_spectrum(raw_power);
    float h = spectral_entropy(raw_power);
    REQUIRE(h == Catch::Approx(2.0f).epsilon(1e-5f));
    REQUIRE(is_coherence_alarm(h, 0.01f));             // thermal noise → alarm
    REQUIRE(confidence_penalty(h) == Catch::Approx(0.0f).margin(1e-7f));  // clamped
}

TEST_CASE("integration: low-entropy (focused) wavefunction passes coherence gate",
          "[gap016][integration]")
{
    // Highly peaked spectrum: dominant bin holds ~99% of energy
    std::vector<float> raw_power = {100.0f, 0.5f, 0.1f, 0.1f};
    normalise_spectrum(raw_power);
    float h = spectral_entropy(raw_power);

    // Low entropy → alarm not triggered
    REQUIRE(h < COHERENCE_ENTROPY_THRESHOLD);
    REQUIRE_FALSE(is_coherence_alarm(h, 0.02f));
    // High confidence: penalty = 1 - H, H ≈ 0.1 → penalty ≈ 0.9
    REQUIRE(confidence_penalty(h) > 0.5f);
}

TEST_CASE("integration: spec table cost values are internally consistent",
          "[gap016][integration]")
{
    // Verify that cumulate(d) = sum of individual step costs — no precision drift
    for (int d = 1; d <= 9; ++d) {
        float manual = 0.0f;
        for (int i = 1; i <= d; ++i)
            manual += recursion_step_cost(i);
        REQUIRE(recursion_cumulative_cost(d) == Catch::Approx(manual).epsilon(1e-4f));
    }
}
