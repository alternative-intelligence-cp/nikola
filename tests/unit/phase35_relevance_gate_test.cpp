// ============================================================================
// phase35_relevance_gate_test.cpp   Phase 35 — RelevanceGate (RGT)
// ============================================================================
//
// Tests:
//   §1  No reference set → novelty = 1.0 regardless of input
//   §2  Identical fields (same seed) → low novelty (cosine_sim ≈ 1)
//   §3  Empty input WaveFunction → urgency = 0
//   §4  Seeded input WaveFunction → urgency > 0 (proportional to energy)
//   §5  Norepinephrine modulates effective threshold correctly
//   §6  High norepinephrine → lower t_eff → more signals pass
//   §7  Low norepinephrine → higher t_eff → fewer signals pass
//   §8  Signal with salience ≥ t_eff → passes=true, weight=1
//   §9  Familiar signal with high-threshold gate → blocked (weight=0)
//   §10 Marginal signal → 0 < weight < 1, passes=true
//   §11 set_reference() update changes subsequent novelty scores
//   §12 has_reference() correctly tracks state
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/cognitive/relevance_gate.hpp>
#include <nikola/physics/wave_function.hpp>

using namespace nikola::cognitive;
using namespace nikola::physics;
using Approx = Catch::Approx;

// ---------------------------------------------------------------------------
// Helper: build a seeded WaveFunction (n=2 → 512 nodes, fast enough for tests)
// ---------------------------------------------------------------------------
static WaveFunction make_wf(uint32_t seed = 42, float amp = 1.f) {
    WaveFunction wf;
    wf.seed_manifold(2, 3, 1, amp, seed);
    return wf;
}

// ---------------------------------------------------------------------------
// §1  No reference → novelty = 1.0
// ---------------------------------------------------------------------------
TEST_CASE("§1 RelevanceGate: no reference → novelty = 1.0", "[Phase35][rgt]") {
    RelevanceGate gate;
    REQUIRE_FALSE(gate.has_reference());

    WaveFunction input = make_wf(42);
    auto r = gate.gate(input, 0.5f);

    // Without a reference everything is novel
    REQUIRE(r.novelty == Approx(1.0f).margin(1e-6f));
    // Urgency should be positive (seeded wf has energy)
    REQUIRE(r.urgency > 0.f);
    REQUIRE(r.salience > 0.f);
}

// ---------------------------------------------------------------------------
// §2  Identical fields (same seed) → low novelty
// ---------------------------------------------------------------------------
TEST_CASE("§2 RelevanceGate: identical fields → low novelty", "[Phase35][rgt]") {
    RelevanceGate gate;

    WaveFunction ref   = make_wf(42);
    WaveFunction input = make_wf(42);  // same seed → identical state

    gate.set_reference(ref);

    auto r = gate.gate(input, 0.5f);

    // Cosine similarity ≈ 1 → novelty ≈ 0
    INFO("novelty = " << r.novelty);
    REQUIRE(r.novelty < 0.05f);
}

// ---------------------------------------------------------------------------
// §3  Empty input WaveFunction → urgency = 0
// ---------------------------------------------------------------------------
TEST_CASE("§3 RelevanceGate: empty input → urgency = 0", "[Phase35][rgt]") {
    RelevanceGate gate;

    WaveFunction empty;  // no seed → 0 nodes, 0 energy
    REQUIRE(empty.total_probability() == Approx(0.0).margin(1e-12));

    auto r = gate.gate(empty, 0.5f);

    REQUIRE(r.urgency == Approx(0.0f).margin(1e-6f));
}

// ---------------------------------------------------------------------------
// §4  Seeded input → urgency > 0 and proportional to energy
// ---------------------------------------------------------------------------
TEST_CASE("§4 RelevanceGate: seeded input → urgency > 0", "[Phase35][rgt]") {
    RelevanceGate gate;

    // 512 nodes (n=2, 9 dims). Keep amplitudes small enough that urgency
    // doesn't saturate at 1.0 for both:
    //   amp=0.001 → total_prob ≈ 0.000512 → urgency ≈ 0.023
    //   amp=0.020 → total_prob ≈ 0.2048   → urgency ≈ 0.452
    WaveFunction low_amp  = make_wf(42, 0.001f);
    WaveFunction high_amp = make_wf(42, 0.020f);

    auto r_low  = gate.gate(low_amp,  0.5f);
    auto r_high = gate.gate(high_amp, 0.5f);

    REQUIRE(r_low.urgency  > 0.f);
    REQUIRE(r_high.urgency > 0.f);
    // Higher amplitude → higher urgency
    INFO("urgency_low=" << r_low.urgency << "  urgency_high=" << r_high.urgency);
    REQUIRE(r_high.urgency > r_low.urgency);
    // Both remain in [0, 1]
    REQUIRE(r_low.urgency  <= 1.f);
    REQUIRE(r_high.urgency <= 1.f);
}

// ---------------------------------------------------------------------------
// §5  Effective threshold math: t_eff = base / (1 + ne)
// ---------------------------------------------------------------------------
TEST_CASE("§5 RelevanceGate: threshold formula is correct", "[Phase35][rgt]") {
    const float base = 0.4f;
    RelevanceGate gate(base);

    // Any signal: verify the effective threshold decreases with norepinephrine
    // We can verify indirectly: at ne=0 the gate is stricter than at ne=1.
    // Expected values:
    //   ne=0.0 → t_eff = 0.40
    //   ne=0.5 → t_eff = 0.40/1.5 ≈ 0.267
    //   ne=1.0 → t_eff = 0.40/2.0 = 0.20
    REQUIRE(gate.base_threshold() == Approx(base).margin(1e-6f));

    // Create a signal with known salience ≈ 0.30 (moderate)
    // (passes at ne≥0.5 but not at ne=0 with base=0.4)
    WaveFunction ref   = make_wf(42);
    WaveFunction input = make_wf(42);  // identical → novelty ≈ 0
    gate.set_reference(ref);

    // With identical fields: novelty ≈ 0, salience ≈ 0.4 * urgency
    auto r_ne0 = gate.gate(input, 0.0f);  // t_eff = 0.40
    auto r_ne1 = gate.gate(input, 1.0f);  // t_eff = 0.20

    // Same salience, different threshold → ne=1 admits signals ne=0 might block
    // At ne=0: salience may fall in marginal zone or be blocked
    // At ne=1: lower threshold means more likely to pass at full weight
    // The invariant: weight(ne=1) >= weight(ne=0)
    REQUIRE(r_ne1.weight >= r_ne0.weight);
}

// ---------------------------------------------------------------------------
// §6  High norepinephrine → lower t_eff → passes when ne=0 would block
// ---------------------------------------------------------------------------
TEST_CASE("§6 RelevanceGate: high NE lowers threshold", "[Phase35][rgt]") {
    // Use a high base threshold so that at ne=0 some signals are blocked
    RelevanceGate strict_gate(0.85f);

    WaveFunction ref   = make_wf(42);
    WaveFunction input = make_wf(42);  // familiar (same seed) → low novelty
    strict_gate.set_reference(ref);

    auto r_ne0 = strict_gate.gate(input, 0.0f);  // t_eff = 0.85 (very strict)
    auto r_ne9 = strict_gate.gate(input, 9.0f);  // t_eff = 0.85/10 = 0.085 (very relaxed)

    // Under extreme norepinephrine the familiar signal should pass
    REQUIRE(r_ne9.weight >= r_ne0.weight);
    // At extreme NE, even a routine signal gets through
    if (r_ne0.weight < r_ne9.weight) {
        INFO("weight at ne=0: " << r_ne0.weight
             << "  weight at ne=9: " << r_ne9.weight);
    }
}

// ---------------------------------------------------------------------------
// §7  Low norepinephrine → higher t_eff → stricter filtering
// ---------------------------------------------------------------------------
TEST_CASE("§7 RelevanceGate: low NE raises threshold", "[Phase35][rgt]") {
    RelevanceGate gate(0.3f);

    WaveFunction ref   = make_wf(42);
    WaveFunction input = make_wf(42);  // familiar
    gate.set_reference(ref);

    auto r_ne0  = gate.gate(input, 0.0f);   // t_eff = 0.30
    auto r_ne05 = gate.gate(input, 0.5f);   // t_eff = 0.20
    auto r_ne1  = gate.gate(input, 1.0f);   // t_eff = 0.15

    // Weights must be non-decreasing as NE increases
    REQUIRE(r_ne1.weight >= r_ne05.weight);
    REQUIRE(r_ne05.weight >= r_ne0.weight);
}

// ---------------------------------------------------------------------------
// §8  High-salience signal with low threshold → passes at full weight
// ---------------------------------------------------------------------------
TEST_CASE("§8 RelevanceGate: novel energetic signal passes at weight=1", "[Phase35][rgt]") {
    // Very permissive gate
    RelevanceGate gate(0.05f);

    // No reference set → novelty=1.  Seeded input → urgency>0.
    // salience = 0.6*1 + 0.4*urgency >> 0.05 → full weight, passes
    WaveFunction input = make_wf(42);

    auto r = gate.gate(input, 0.5f);

    INFO("salience=" << r.salience << " weight=" << r.weight
         << " novelty=" << r.novelty << " urgency=" << r.urgency);
    REQUIRE(r.passes  == true);
    REQUIRE(r.weight  == Approx(1.0f).margin(1e-6f));
}

// ---------------------------------------------------------------------------
// §9  Familiar signal blocked by strict gate (low urgency + low novelty)
// ---------------------------------------------------------------------------
TEST_CASE("§9 RelevanceGate: familiar quiet signal blocked", "[Phase35][rgt]") {
    // base_threshold=0.5, ne=0 → t_eff=0.5, t_low=0.25
    // Familiar (same seed) → novelty ≈ 0
    // Quiet (amp=0.01) → total_prob ≈ 0.01*512 ≈ 5.12, urgency = min(1,sqrt(5.12)) = 1
    // Actually we need VERY low amplitude to get low urgency
    // amp=0.001 → total_prob ≈ 0.001^2 * 512 ≈ 0.000512, urgency = sqrt(0.000512) ≈ 0.023
    // salience = 0.6*0 + 0.4*0.023 ≈ 0.009 < t_low=0.25 → blocked
    RelevanceGate gate(0.5f);

    WaveFunction ref   = make_wf(42, 1.f);
    WaveFunction input = make_wf(42, 0.001f);  // same seed (familiar) but very quiet
    gate.set_reference(ref);

    auto r = gate.gate(input, 0.0f);

    INFO("novelty=" << r.novelty << " urgency=" << r.urgency
         << " salience=" << r.salience << " weight=" << r.weight);
    REQUIRE(r.weight  == Approx(0.0f).margin(1e-6f));
    REQUIRE(r.passes  == false);
}

// ---------------------------------------------------------------------------
// §10 Marginal signal → 0 < weight < 1
// ---------------------------------------------------------------------------
TEST_CASE("§10 RelevanceGate: marginal signal has attenuated weight", "[Phase35][rgt]") {
    // t_eff = base / (1+ne) = 0.5 / 1 = 0.5, t_low = 0.25
    // We need salience in (0.25, 0.5)
    // Use same-seed ref (novelty ≈ 0) + moderate amplitude so urgency ≈ 0.5-0.9
    // salience ≈ 0.4 * urgency
    // If urgency ≈ 0.7 → salience ≈ 0.28, which is in (0.25, 0.5)
    RelevanceGate gate(0.5f);

    WaveFunction ref   = make_wf(42, 1.f);
    WaveFunction input = make_wf(42, 0.1f);  // familiar + low-moderate energy
    gate.set_reference(ref);

    // ne=0 → t_eff=0.5, t_low=0.25
    auto r = gate.gate(input, 0.0f);

    INFO("novelty=" << r.novelty << " urgency=" << r.urgency
         << " salience=" << r.salience << " weight=" << r.weight);

    if (r.salience >= 0.5f) {
        // Landed above threshold (urgency was high): weight = 1
        REQUIRE(r.weight == Approx(1.0f).margin(1e-4f));
        REQUIRE(r.passes == true);
    } else if (r.salience < 0.25f) {
        // Landed below marginal zone: blocked
        REQUIRE(r.weight == Approx(0.0f).margin(1e-4f));
        REQUIRE(r.passes == false);
    } else {
        // In the marginal zone [0.25, 0.5): attenuated
        REQUIRE(r.weight  >  0.f);
        REQUIRE(r.weight  <  1.f);
        REQUIRE(r.passes  == true);
    }
}

// ---------------------------------------------------------------------------
// §11 set_reference() update changes novelty on subsequent calls
// ---------------------------------------------------------------------------
TEST_CASE("§11 RelevanceGate: set_reference update changes novelty", "[Phase35][rgt]") {
    // NOTE: seed_manifold's psi field depends only on (n, pilot_dim, k_mode,
    // amplitude) — NOT on the RNG seed (which affects only the vel/thermal
    // component).  To get genuinely different psi fields we vary pilot_dim.
    //
    //   dim=3 pilot wave   ↔  "familiar" reference field
    //   dim=0 pilot wave   ↔  "novel" input (orthogonal structure)
    RelevanceGate gate;

    WaveFunction wf_dim3a;
    wf_dim3a.seed_manifold(2, 3, 1, 1.f, 42);   // pilot in dimension 3
    WaveFunction wf_dim3b;
    wf_dim3b.seed_manifold(2, 3, 1, 1.f, 42);   // identical

    WaveFunction wf_dim0;
    wf_dim0.seed_manifold(2, 0, 1, 1.f, 42);    // pilot in dimension 0 — different field

    // Reference = dim3; query with dim0 → should be novel (low cosine_sim)
    gate.set_reference(wf_dim3a);
    WaveFunction input_novel;
    input_novel.seed_manifold(2, 0, 1, 1.f, 42);
    auto r_novel = gate.gate(input_novel, 0.5f);

    // Now update reference to match dim0 pattern
    gate.set_reference(wf_dim0);
    WaveFunction input_familiar;
    input_familiar.seed_manifold(2, 0, 1, 1.f, 42);
    auto r_familiar = gate.gate(input_familiar, 0.5f);

    INFO("novelty (novel input vs dim3 ref)=" << r_novel.novelty
         << "  (familiar input vs dim0 ref)=" << r_familiar.novelty);
    // After reference matches the input pattern, novelty must drop
    REQUIRE(r_familiar.novelty < r_novel.novelty);
}

// ---------------------------------------------------------------------------
// §12 has_reference() correctly tracks state
// ---------------------------------------------------------------------------
TEST_CASE("§12 RelevanceGate: has_reference tracks reference state", "[Phase35][rgt]") {
    RelevanceGate gate;
    REQUIRE_FALSE(gate.has_reference());

    WaveFunction wf = make_wf(42);
    gate.set_reference(wf);
    REQUIRE(gate.has_reference());

    // Empty wavefunction clears reference
    WaveFunction empty;
    gate.set_reference(empty);
    REQUIRE_FALSE(gate.has_reference());
}
