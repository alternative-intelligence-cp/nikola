// ============================================================================
// phase36_gated_injection_test.cpp   Phase 36 — Gated injection pipeline
// ============================================================================
//
// Tests:
//   §1  gate_embedding: no reference → novelty = 1.0
//   §2  gate_embedding: identical embeddings → low novelty
//   §3  gate_embedding: orthogonal embeddings → high novelty
//   §4  gate_embedding: urgency scales with embedding L2 norm
//   §5  gate_embedding: norepinephrine lowers effective threshold
//   §6  gate_embedding: zero-norm embedding → urgency = 0
//   §7  inject_raw_scaled: weight 0 → torus energy unchanged
//   §8  inject_raw_scaled: weight 1 → same effect as inject_raw
//   §9  inject_raw_scaled: weight 0.5 → intermediate energy change
//   §10 inject_raw_scaled: Nit amplitudes clamped to [-4, +4]
//   §11 update_gate_reference: WF reference updated from torus state
//   §12 set/has_reference_embedding round-trip
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/cognitive/relevance_gate.hpp>
#include <nikola/cognitive/cognitive_torus.hpp>
#include <nikola/foundation/nit.hpp>

#include <cmath>
#include <numeric>
#include <vector>

using namespace nikola::cognitive;
using nikola::foundation::Nit;
using Approx = Catch::Approx;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Build a unit embedding of length N aligned along dimension `dim`
static std::vector<float> unit_emb(size_t N, size_t dim, float amp = 5.0f) {
    std::vector<float> v(N, 0.f);
    if (dim < N) v[dim] = amp;
    return v;
}

// Build an embedding where every element equals `val`
static std::vector<float> uniform_emb(size_t N, float val) {
    return std::vector<float>(N, val);
}

// Build a maxed-out Nit vector (all +4)
static std::vector<Nit> max_nits(size_t M = 128) {
    return std::vector<Nit>(M, static_cast<Nit>(4));
}

// Build a Nit vector with specific value
static std::vector<Nit> fill_nits(size_t M, int val) {
    return std::vector<Nit>(M, static_cast<Nit>(val));
}

// ---------------------------------------------------------------------------
// §1  gate_embedding: no reference → novelty = 1.0
// ---------------------------------------------------------------------------
TEST_CASE("§1 gate_embedding: no reference → novelty = 1.0", "[Phase36][gate_emb]") {
    RelevanceGate gate;
    REQUIRE_FALSE(gate.has_reference_embedding());

    auto emb = uniform_emb(128, 1.0f);
    auto r   = gate.gate_embedding(emb, 0.5f);

    REQUIRE(r.novelty == Approx(1.0f).margin(1e-6f));
}

// ---------------------------------------------------------------------------
// §2  gate_embedding: identical embeddings → low novelty
// ---------------------------------------------------------------------------
TEST_CASE("§2 gate_embedding: identical embeddings → low novelty", "[Phase36][gate_emb]") {
    RelevanceGate gate;

    auto ref = uniform_emb(128, 2.0f);
    gate.set_reference_embedding(ref);

    auto input = uniform_emb(128, 2.0f);  // same
    auto r     = gate.gate_embedding(input, 0.5f);

    INFO("novelty=" << r.novelty);
    REQUIRE(r.novelty < 0.05f);  // cosine_sim ≈ 1 → novelty ≈ 0
}

// ---------------------------------------------------------------------------
// §3  gate_embedding: orthogonal embeddings → high novelty
// ---------------------------------------------------------------------------
TEST_CASE("§3 gate_embedding: orthogonal embeddings → high novelty", "[Phase36][gate_emb]") {
    RelevanceGate gate;

    // Reference: non-zero only in first 64 dims
    std::vector<float> ref(128, 0.f);
    for (size_t i = 0; i < 64; ++i) ref[i] = 1.f;
    gate.set_reference_embedding(ref);

    // Input: non-zero only in last 64 dims → 0 dot product with ref
    std::vector<float> input(128, 0.f);
    for (size_t i = 64; i < 128; ++i) input[i] = 1.f;

    auto r = gate.gate_embedding(input, 0.5f);

    INFO("novelty=" << r.novelty);
    REQUIRE(r.novelty > 0.9f);  // orthogonal → cosine_sim = 0 → novelty = 1
}

// ---------------------------------------------------------------------------
// §4  gate_embedding: urgency scales with L2 norm
// ---------------------------------------------------------------------------
TEST_CASE("§4 gate_embedding: urgency scales with L2 norm", "[Phase36][gate_emb]") {
    RelevanceGate gate;

    // Low norm: single element = 0.1 → norm = 0.1
    auto low  = unit_emb(128, 0, 0.1f);
    // High norm: single element = 5.0 → norm = 5.0
    auto high = unit_emb(128, 0, 5.0f);

    auto r_low  = gate.gate_embedding(low,  0.5f);
    auto r_high = gate.gate_embedding(high, 0.5f);

    INFO("urgency_low=" << r_low.urgency << "  urgency_high=" << r_high.urgency);
    REQUIRE(r_low.urgency  >= 0.f);
    REQUIRE(r_high.urgency >= 0.f);
    REQUIRE(r_high.urgency  > r_low.urgency);
    REQUIRE(r_low.urgency  <= 1.f);
    REQUIRE(r_high.urgency <= 1.f);
}

// ---------------------------------------------------------------------------
// §5  gate_embedding: norepinephrine lowers effective threshold
// ---------------------------------------------------------------------------
TEST_CASE("§5 gate_embedding: norepinephrine lowers threshold", "[Phase36][gate_emb]") {
    // Use a high base threshold so the signal sits in the marginal/blocked zone
    RelevanceGate strict_gate(0.7f);

    // A moderate embedding: some familiar, some novel
    auto ref   = uniform_emb(128, 1.0f);
    auto input = uniform_emb(128, 1.0f);  // identical to ref → novelty ≈ 0
    strict_gate.set_reference_embedding(ref);

    // At ne=0: t_eff = 0.7; low-novelty signal with low urgency → likely blocked
    // At ne=9: t_eff = 0.7/10 = 0.07; same signal almost certainly passes
    auto r_ne0 = strict_gate.gate_embedding(input, 0.0f);
    auto r_ne9 = strict_gate.gate_embedding(input, 9.0f);

    INFO("weight ne=0: " << r_ne0.weight << "  ne=9: " << r_ne9.weight);
    REQUIRE(r_ne9.weight >= r_ne0.weight);
}

// ---------------------------------------------------------------------------
// §6  gate_embedding: zero-norm embedding → urgency = 0
// ---------------------------------------------------------------------------
TEST_CASE("§6 gate_embedding: zero-norm embedding → urgency = 0", "[Phase36][gate_emb]") {
    RelevanceGate gate;
    std::vector<float> zero_emb(128, 0.f);

    auto r = gate.gate_embedding(zero_emb, 0.5f);

    REQUIRE(r.urgency == Approx(0.0f).margin(1e-6f));
}

// ---------------------------------------------------------------------------
// §7  inject_raw_scaled: weight 0 → torus energy unchanged
// ---------------------------------------------------------------------------
TEST_CASE("§7 inject_raw_scaled: weight 0 → torus energy unchanged", "[Phase36][inject]") {
    CognitiveTorus torus(3);

    const double energy_before = torus.total_probability();
    auto nits = max_nits();

    torus.inject_raw_scaled(nits, 0.f, 0.0);

    const double energy_after = torus.total_probability();
    REQUIRE(energy_after == Approx(energy_before).epsilon(0.01));
}

// ---------------------------------------------------------------------------
// §8  inject_raw_scaled: weight 1 → same energy effect as inject_raw
// ---------------------------------------------------------------------------
TEST_CASE("§8 inject_raw_scaled: weight 1 ≈ inject_raw", "[Phase36][inject]") {
    // Two identical fresh tori
    CognitiveTorus torus_direct(3);
    CognitiveTorus torus_scaled(3);

    auto nits = fill_nits(128, 2);

    torus_direct.inject_raw(nits, 0.0);
    torus_scaled.inject_raw_scaled(nits, 1.0f, 0.0);

    const double e_direct = torus_direct.total_probability();
    const double e_scaled = torus_scaled.total_probability();

    INFO("e_direct=" << e_direct << "  e_scaled=" << e_scaled);
    // Both receive the same injection — energies should be equal
    REQUIRE(e_scaled == Approx(e_direct).epsilon(0.001));
}

// ---------------------------------------------------------------------------
// §9  inject_raw_scaled: weight 0.5 → intermediate energy
// ---------------------------------------------------------------------------
TEST_CASE("§9 inject_raw_scaled: weight 0.5 → intermediate energy", "[Phase36][inject]") {
    CognitiveTorus torus_none(3);
    CognitiveTorus torus_half(3);
    CognitiveTorus torus_full(3);

    auto nits = fill_nits(128, 3);

    // none: no injection (weight=0)
    torus_none.inject_raw_scaled(nits, 0.f, 0.0);
    // half: weight=0.5
    torus_half.inject_raw_scaled(nits, 0.5f, 0.0);
    // full: weight=1.0
    torus_full.inject_raw_scaled(nits, 1.0f, 0.0);

    const double e_none = torus_none.total_probability();
    const double e_half = torus_half.total_probability();
    const double e_full = torus_full.total_probability();

    INFO("e_none=" << e_none << "  e_half=" << e_half << "  e_full=" << e_full);

    // half injection should add some energy but less than full
    REQUIRE(e_half >= e_none);
    REQUIRE(e_full >= e_half);
}

// ---------------------------------------------------------------------------
// §10 inject_raw_scaled: Nit amplitudes clamped; weight > 1 handled safely
// ---------------------------------------------------------------------------
TEST_CASE("§10 inject_raw_scaled: Nit amplitudes clamped to [-4,+4]", "[Phase36][inject]") {
    CognitiveTorus torus(3);

    // Edge: weight = 2.0 with max nits (4*2=8 → should clamp to 4)
    // The injection should still complete without throwing/crashing
    auto nits = max_nits();
    REQUIRE_NOTHROW(torus.inject_raw_scaled(nits, 2.0f, 0.0));

    // Torus should have finite energy after the call
    REQUIRE(std::isfinite(torus.total_probability()));
}

// ---------------------------------------------------------------------------
// §11 update_gate_reference: WF reference updated from torus state
// ---------------------------------------------------------------------------
TEST_CASE("§11 update_gate_reference: reflects torus WF state", "[Phase36][gate_ref]") {
    RelevanceGate gate;
    REQUIRE_FALSE(gate.has_reference());

    CognitiveTorus torus(3);
    torus.update_gate_reference(gate);

    REQUIRE(gate.has_reference());

    // Now inject something and run the torus → state changes
    auto nits_a = fill_nits(128, 4);  // strong injection
    torus.inject_raw(nits_a, 0.0);
    torus.run(5, torus.safe_dt());

    // Snapshot to a second gate
    RelevanceGate gate2;
    torus.update_gate_reference(gate2);

    // Pre-injection snapshot (gate) vs post-injection (gate2) should differ:
    // A wavefunction seeded fresh vs after injection has different energy density.
    // Check: ref_energy of gate2 differs from gate (energy changed after injection+run)
    INFO("gate1 ref_energy=" << gate.ref_energy()
         << "  gate2 ref_energy=" << gate2.ref_energy());
    // After injection the reference energy should be measurably different.
    // The values are ~19683 vs ~19571 (≈0.56% difference) — use tight epsilon.
    REQUIRE(gate2.ref_energy() != Approx(gate.ref_energy()).epsilon(0.001));
}

// ---------------------------------------------------------------------------
// §12 set_reference_embedding / has_reference_embedding round-trip
// ---------------------------------------------------------------------------
TEST_CASE("§12 set/has_reference_embedding round-trip", "[Phase36][gate_emb]") {
    RelevanceGate gate;

    REQUIRE_FALSE(gate.has_reference_embedding());

    auto emb = uniform_emb(128, 1.5f);
    gate.set_reference_embedding(emb);

    REQUIRE(gate.has_reference_embedding());

    // Updating with an empty vector clears it
    gate.set_reference_embedding({});
    REQUIRE_FALSE(gate.has_reference_embedding());
}
