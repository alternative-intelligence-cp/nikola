// ============================================================================
// phase40_heterodyne_test.cpp   Phase 40 — Heterodyne Value Aggregation
// ============================================================================
//
// Tests:
//   §1   clone(): cloned WF has same node count as original
//   §2   clone(): cloned psi matches original psi (total_probability equal)
//   §3   clone(): cloned WF is independent (modifying clone doesn't affect src)
//   §4   scale_by(): scales total_probability by s²
//   §5   scale_by(0): zeros the field (total_probability == 0)
//   §6   scale_by(1): is a no-op
//   §7   add_scaled(): size mismatch is silently ignored
//   §8   add_scaled(): result total_probability consistent with superposition
//   §9   add_scaled(): in-phase fields — energy of sum = (1+s)² · |Q|²
//   §10  forward(): has_output == true
//   §11  forward(): output WF node count matches torus
//   §12  forward(): output total_probability ≈ torus total_probability
//          (convex combination with uniform V → output == torus)
//   §13  forward(): output is_finite()
//   §14  forward(): output non-zero when torus is non-zero
//   §15  forward(): scores still sum to 1.0 alongside valid output
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/cognitive/neuroplastic_transformer.hpp>
#include <nikola/physics/wave_function.hpp>

#include <cmath>

using namespace nikola::cognitive;
using namespace nikola::physics;
namespace physics = nikola::physics;
using Approx = Catch::Approx;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static float sum8(const std::array<float, NPT_NUM_HEADS>& a) {
    float s = 0.f;
    for (float v : a) s += v;
    return s;
}

// ---------------------------------------------------------------------------
// §1   clone(): cloned WF has same node count as original
// ---------------------------------------------------------------------------
TEST_CASE("§1 clone(): node count preserved", "[Phase40][clone]") {
    WaveFunction src;
    src.seed_manifold(3, 3, 1, 1.f, 42);
    auto copy = src.clone();
    REQUIRE(copy.num_nodes() == src.num_nodes());
}

// ---------------------------------------------------------------------------
// §2   clone(): cloned psi matches original (same total_probability)
// ---------------------------------------------------------------------------
TEST_CASE("§2 clone(): psi content identical to source", "[Phase40][clone]") {
    WaveFunction src;
    src.seed_manifold(3, 3, 1, 1.f, 7);
    auto copy = src.clone();

    // total_probability = Σ|ψᵢ|²; identical psi → identical probability
    REQUIRE(copy.total_probability() == Approx(src.total_probability()).epsilon(1e-5));
}

// ---------------------------------------------------------------------------
// §3   clone(): independence — modifying clone doesn't affect source
// ---------------------------------------------------------------------------
TEST_CASE("§3 clone(): independence from source", "[Phase40][clone]") {
    WaveFunction src;
    src.seed_manifold(3, 3, 1, 1.f, 13);
    const double orig_prob = src.total_probability();

    auto copy = src.clone();
    copy.scale_by(0.f);   // zero out the clone's psi

    REQUIRE(src.total_probability() == Approx(orig_prob).epsilon(1e-6));
    REQUIRE(copy.total_probability() == Approx(0.0).margin(1e-10));
}

// ---------------------------------------------------------------------------
// §4   scale_by(): total_probability scales by s²
// ---------------------------------------------------------------------------
TEST_CASE("§4 scale_by(): probability scales as s²", "[Phase40][scale]") {
    WaveFunction wf;
    wf.seed_manifold(3, 3, 1, 1.f, 42);
    const double p0 = wf.total_probability();

    wf.scale_by(0.5f);
    REQUIRE(wf.total_probability() == Approx(p0 * 0.25).epsilon(1e-5));
}

// ---------------------------------------------------------------------------
// §5   scale_by(0): zeros the field
// ---------------------------------------------------------------------------
TEST_CASE("§5 scale_by(0): field goes to vacuum", "[Phase40][scale]") {
    WaveFunction wf;
    wf.seed_manifold(3, 3, 1, 1.f, 42);
    wf.scale_by(0.f);
    REQUIRE(wf.total_probability() == Approx(0.0).margin(1e-10));
}

// ---------------------------------------------------------------------------
// §6   scale_by(1): is a no-op
// ---------------------------------------------------------------------------
TEST_CASE("§6 scale_by(1.0): no-op", "[Phase40][scale]") {
    WaveFunction wf;
    wf.seed_manifold(3, 3, 1, 1.f, 42);
    const double p0 = wf.total_probability();
    wf.scale_by(1.0f);
    REQUIRE(wf.total_probability() == Approx(p0).epsilon(1e-7));
}

// ---------------------------------------------------------------------------
// §7   add_scaled(): size mismatch is silently ignored
// ---------------------------------------------------------------------------
TEST_CASE("§7 add_scaled(): size mismatch — silent no-op", "[Phase40][add_scaled]") {
    WaveFunction dst;
    dst.seed_manifold(3, 3, 1, 1.f, 42);  // 3^9 nodes
    const double p_before = dst.total_probability();

    WaveFunction src;
    src.seed_manifold(2, 3, 1, 1.f, 42);  // 2^9 nodes — different size
    dst.add_scaled(src, 1.0f);            // must be a no-op

    REQUIRE(dst.total_probability() == Approx(p_before).epsilon(1e-7));
}

// ---------------------------------------------------------------------------
// §8   add_scaled(): result probability consistent with superposition
// ---------------------------------------------------------------------------
TEST_CASE("§8 add_scaled(): superposition energy formula", "[Phase40][add_scaled]") {
    // dst += 1.0 * src  (identical fields, same seed)
    // |dst + src|² = 4|src|²  (constructive: each psi doubles → prob × 4)
    WaveFunction dst;
    dst.seed_manifold(3, 3, 1, 1.f, 5);
    WaveFunction src;
    src.seed_manifold(3, 3, 1, 1.f, 5);  // identical to dst

    const double p_src = src.total_probability();
    dst.add_scaled(src, 1.0f);           // dst.psi = 2 * original_psi

    // |2ψ|² = 4|ψ|²
    REQUIRE(dst.total_probability() == Approx(4.0 * p_src).epsilon(1e-4));
}

// ---------------------------------------------------------------------------
// §9   add_scaled(): in-phase + weight s
// ---------------------------------------------------------------------------
TEST_CASE("§9 add_scaled(): energy of (ψ + s·ψ) = (1+s)² · |ψ|²", "[Phase40][add_scaled]") {
    // dst.psi = psi, src.psi = psi (same seed), weight = 0.5
    // After: dst.psi = psi + 0.5*psi = 1.5*psi
    // prob  = |1.5ψ|² = 2.25 * |ψ|²
    WaveFunction dst;
    dst.seed_manifold(3, 3, 1, 1.f, 9);
    WaveFunction src;
    src.seed_manifold(3, 3, 1, 1.f, 9);

    const double p0 = src.total_probability();
    dst.add_scaled(src, 0.5f);

    const double expected = (1.0 + 0.5) * (1.0 + 0.5) * p0;  // 2.25 * p0
    INFO("actual prob = " << dst.total_probability() << "  expected = " << expected);
    REQUIRE(dst.total_probability() == Approx(expected).epsilon(1e-4));
}

// ---------------------------------------------------------------------------
// §10  forward(): has_output == true
// ---------------------------------------------------------------------------
TEST_CASE("§10 NPT forward(): has_output == true", "[Phase40][forward]") {
    NeuroplasticTransformer npt(3);
    WaveFunction torus_wf;
    torus_wf.seed_manifold(3, 3, 1, 1.f, 42);

    auto result = npt.forward(torus_wf);
    REQUIRE(result.has_output == true);
}

// ---------------------------------------------------------------------------
// §11  forward(): output node count matches torus
// ---------------------------------------------------------------------------
TEST_CASE("§11 NPT forward(): output node count matches torus", "[Phase40][forward]") {
    NeuroplasticTransformer npt(3);
    WaveFunction torus_wf;
    torus_wf.seed_manifold(3, 3, 1, 1.f, 42);

    auto result = npt.forward(torus_wf);
    REQUIRE(result.output.num_nodes() == torus_wf.num_nodes());
}

// ---------------------------------------------------------------------------
// §12  forward(): output total_probability ≈ torus total_probability
//       (uniform V: convex combination of identical fields = same field)
// ---------------------------------------------------------------------------
TEST_CASE("§12 NPT forward(): output prob ≈ torus prob (uniform V)", "[Phase40][forward]") {
    // With all V_i = torus_wf and Σ scores = 1:
    //   output_ψᵢ = Σ score[j] · torus_ψᵢ = torus_ψᵢ · Σ score[j] = torus_ψᵢ
    // So output total_probability should equal torus total_probability.
    NeuroplasticTransformer npt(3, 1.0f, 0.0f);  // alpha=0 for determinism
    WaveFunction torus_wf;
    torus_wf.seed_manifold(3, 3, 1, 1.f, 42);

    auto result = npt.forward(torus_wf);

    const double p_torus  = torus_wf.total_probability();
    const double p_output = result.output.total_probability();

    INFO("torus prob = " << p_torus << "  output prob = " << p_output);
    // Allow small floating-point accumulation error over 8 iterations + scale
    REQUIRE(p_output == Approx(p_torus).epsilon(1e-4));
}

// ---------------------------------------------------------------------------
// §13  forward(): output is_finite()
// ---------------------------------------------------------------------------
TEST_CASE("§13 NPT forward(): output WF is numerically finite", "[Phase40][forward]") {
    NeuroplasticTransformer npt(3);
    WaveFunction torus_wf;
    torus_wf.seed_manifold(3, 3, 1, 1.f, 42);

    auto result = npt.forward(torus_wf);
    REQUIRE(result.output.is_finite());
}

// ---------------------------------------------------------------------------
// §14  forward(): output non-zero when torus is non-zero
// ---------------------------------------------------------------------------
TEST_CASE("§14 NPT forward(): output carries energy when torus is non-zero",
          "[Phase40][forward]") {
    NeuroplasticTransformer npt(3);
    WaveFunction torus_wf;
    torus_wf.seed_manifold(3, 3, 1, 1.f, 42);  // amplitude=1 → non-zero energy

    auto result = npt.forward(torus_wf);
    REQUIRE(result.output.total_probability() > 1e-10);
}

// ---------------------------------------------------------------------------
// §15  forward(): scores still sum to 1.0 alongside valid output
// ---------------------------------------------------------------------------
TEST_CASE("§15 NPT forward(): scores sum to 1.0 and output is valid together",
          "[Phase40][forward]") {
    NeuroplasticTransformer npt(3);
    WaveFunction torus_wf;
    torus_wf.seed_manifold(3, 3, 1, 1.f, 42);

    auto result = npt.forward(torus_wf);

    REQUIRE(sum8(result.head_scores) == Approx(1.0f).epsilon(1e-5f));
    REQUIRE(result.has_output == true);
    REQUIRE(result.output.is_finite());
    REQUIRE(result.output.num_nodes() > 0);
}
