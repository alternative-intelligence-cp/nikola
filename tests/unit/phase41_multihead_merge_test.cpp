// ============================================================================
// phase41_multihead_merge_test.cpp   Phase 41 — Multi-Head Merge
// ============================================================================
//
// Tests:
//   §1   phase_rotate_psi(0): no-op — total_probability unchanged
//   §2   phase_rotate_psi(φ): probability invariant (|e^{iφ}|=1)
//   §3   phase_rotate_psi(π): purely-real field negates (≡ scale_by(-1))
//   §4   phase_rotate_psi(π/2): real→imag, imag→-real (quarter turn)
//   §5   project_heads(): V node count matches torus
//   §6   project_heads(): V total_probability == torus total_probability
//          (phase rotation is unitary — preserves energy)
//   §7   project_heads() at t=0: all V_i identical to torus (phase = 0)
//   §8   project_heads() at t>0: head 0 and head 7 V fields differ
//   §9   project_heads() at t>0: Re(⟨V_i|torus⟩) = cos(φ_i)·|torus|²
//          (phase rotation reduces inner product by cos of rotation angle)
//   §10  forward() at t=0: output prob ≈ torus prob (degenerate uniform)
//   §11  forward() at t>0: output is_finite and has energy
//   §12  forward(): has_output == true
//   §13  forward(): scores sum to 1.0
//   §14  forward() at t>0: output inner product with torus < torus prob
//          (proves phase differentiation is live)
//   §15  forward(): output node count matches torus
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

static constexpr float PI_F = static_cast<float>(NPT_PI);

static float sum8(const std::array<float, NPT_NUM_HEADS>& a) {
    float s = 0.f; for (float v : a) s += v; return s;
}

// ---------------------------------------------------------------------------
// §1   phase_rotate_psi(0): no-op
// ---------------------------------------------------------------------------
TEST_CASE("§1 phase_rotate_psi(0): total_probability unchanged", "[Phase41][rotate]") {
    WaveFunction wf;
    wf.seed_manifold(3, 3, 1, 1.f, 42);
    const double p0 = wf.total_probability();
    wf.phase_rotate_psi(0.f);
    REQUIRE(wf.total_probability() == Approx(p0).epsilon(1e-7));
}

// ---------------------------------------------------------------------------
// §2   phase_rotate_psi(φ): probability invariant for arbitrary φ
// ---------------------------------------------------------------------------
TEST_CASE("§2 phase_rotate_psi: probability invariant for any angle", "[Phase41][rotate]") {
    WaveFunction wf;
    wf.seed_manifold(3, 3, 1, 1.f, 7);
    const double p0 = wf.total_probability();

    for (float phi : {0.3f, 1.0f, PI_F / 3.f, PI_F, 2.f * PI_F - 0.01f}) {
        WaveFunction tmp;
        tmp.seed_manifold(3, 3, 1, 1.f, 7);
        tmp.phase_rotate_psi(phi);
        INFO("phi = " << phi << "  prob = " << tmp.total_probability());
        REQUIRE(tmp.total_probability() == Approx(p0).epsilon(1e-5));
    }
}

// ---------------------------------------------------------------------------
// §3   phase_rotate_psi(π): purely-real field → psi negated
// ---------------------------------------------------------------------------
TEST_CASE("§3 phase_rotate_psi(π): real field negated", "[Phase41][rotate]") {
    // Single-node grid; inject a purely real amplitude (0.3 + 0i).
    // After rotation by π: ψ' = ψ * (cos π + i·sin π) = ψ * (-1) = -0.3
    // Re(⟨ψ'|ψ⟩) = (-0.3)·(0.3) = -0.09 = -|ψ|²  → inner_product = -prob
    WaveFunction wf;
    wf.seed_manifold(1, 0, 1, 0.f, 0);   // vacuum 1-node
    wf.inject(0, {0.3f, 0.0f}, 1.f);
    const double prob    = wf.total_probability();  // 0.09

    WaveFunction ref;
    ref.seed_manifold(1, 0, 1, 0.f, 0);
    ref.inject(0, {0.3f, 0.0f}, 1.f);

    wf.phase_rotate_psi(PI_F);   // → -0.3 + 0i

    // Re(⟨rotated|original⟩) = (-0.3)*0.3 + 0*0 = -0.09 = -prob
    REQUIRE(wf.inner_product_re(ref) == Approx(-prob).epsilon(1e-5));
}

// ---------------------------------------------------------------------------
// §4   phase_rotate_psi(π/2): right-angle rotation
// ---------------------------------------------------------------------------
TEST_CASE("§4 phase_rotate_psi(π/2): orthogonal rotation", "[Phase41][rotate]") {
    // Single-node; inject (0.4 + 0i).
    // After +π/2: ψ' = 0.4 * (cos(π/2) + i·sin(π/2)) = 0.4 * (0 + i) = 0+0.4i
    // Re(⟨ψ'|ψ_orig⟩) = 0*0.4 + 0.4*0 = 0  → inner_product = 0
    WaveFunction wf;
    wf.seed_manifold(1, 0, 1, 0.f, 0);
    wf.inject(0, {0.4f, 0.0f}, 1.f);

    WaveFunction ref;
    ref.seed_manifold(1, 0, 1, 0.f, 0);
    ref.inject(0, {0.4f, 0.0f}, 1.f);

    wf.phase_rotate_psi(PI_F / 2.f);   // → 0 + 0.4i

    REQUIRE(wf.inner_product_re(ref) == Approx(0.0).margin(1e-5));
}

// ---------------------------------------------------------------------------
// §5   project_heads(): V node count matches torus
// ---------------------------------------------------------------------------
TEST_CASE("§5 project_heads(): V node count matches torus", "[Phase41][project]") {
    NeuroplasticTransformer npt(3);
    WaveFunction torus_wf;
    torus_wf.seed_manifold(3, 3, 1, 1.f, 42);

    npt.project_heads(torus_wf);

    for (size_t i = 0; i < NPT_NUM_HEADS; ++i) {
        INFO("head " << i << " V.num_nodes = " << npt.head(i).V.num_nodes());
        REQUIRE(npt.head(i).V.num_nodes() == torus_wf.num_nodes());
    }
}

// ---------------------------------------------------------------------------
// §6   project_heads(): V total_probability == torus total_probability
// ---------------------------------------------------------------------------
TEST_CASE("§6 project_heads(): V energy == torus energy (phase is unitary)", "[Phase41][project]") {
    NeuroplasticTransformer npt(3);
    WaveFunction torus_wf;
    torus_wf.seed_manifold(3, 3, 1, 1.f, 42);
    const double p_torus = torus_wf.total_probability();

    npt.project_heads(torus_wf);

    for (size_t i = 0; i < NPT_NUM_HEADS; ++i) {
        const double pv = npt.head(i).V.total_probability();
        INFO("head " << i << " V.prob = " << pv << "  torus.prob = " << p_torus);
        REQUIRE(pv == Approx(p_torus).epsilon(1e-5));
    }
}

// ---------------------------------------------------------------------------
// §7   project_heads() at t=0: all V_i identical to torus
// ---------------------------------------------------------------------------
TEST_CASE("§7 project_heads() at t=0: all V_i have zero phase shift", "[Phase41][project]") {
    // torus_wf.time() == 0.f by default after seed_manifold.
    // phase_i = 2π·f_i·0 = 0  →  V_i = torus_wf exactly.
    // Re(⟨V_i|torus⟩) = |torus|² for all i.
    NeuroplasticTransformer npt(3);
    WaveFunction torus_wf;
    torus_wf.seed_manifold(3, 3, 1, 1.f, 42);   // time() == 0.f

    npt.project_heads(torus_wf);
    const double p_torus = torus_wf.total_probability();

    for (size_t i = 0; i < NPT_NUM_HEADS; ++i) {
        const double ip = npt.head(i).V.inner_product_re(torus_wf);
        INFO("head " << i << " Re(⟨V|torus⟩) = " << ip << "  expected = " << p_torus);
        REQUIRE(ip == Approx(p_torus).epsilon(1e-5));
    }
}

// ---------------------------------------------------------------------------
// §8   project_heads() at t>0: head 0 and head 7 V fields differ
// ---------------------------------------------------------------------------
TEST_CASE("§8 project_heads() at t>0: head 0 and head 7 V fields differ", "[Phase41][project]") {
    // f_0 = π·φ⁰ ≈ 3.14,  f_7 = π·φ⁷ ≈ 91.21.
    // At t=0.5s: φ_0 ≈ π·0.5, φ_7 ≈ 91.21·π  (very different).
    // The inner products Re(⟨V_0|V_7⟩) should be much less than |torus|².
    NeuroplasticTransformer npt(3);
    WaveFunction torus_wf;
    torus_wf.seed_manifold(3, 3, 1, 1.f, 42);
    torus_wf.advance_time(0.5f);   // t = 0.5s

    npt.project_heads(torus_wf);

    const double ip_07 = npt.head(0).V.inner_product_re(npt.head(7).V);
    const double p     = npt.head(0).V.total_probability();

    INFO("Re(⟨V_0|V_7⟩) = " << ip_07 << "  |V_0|² = " << p);
    // If V_0 and V_7 were identical, ip would equal p.
    // Different phases → Re(⟨V_0|V_7⟩) = cos(φ_7-φ_0)·p < p.
    REQUIRE(std::abs(ip_07) < p * 0.99);  // measurably different
}

// ---------------------------------------------------------------------------
// §9   project_heads(): Re(⟨V_i|torus⟩) = cos(2π·f_i·t)·|torus|²
// ---------------------------------------------------------------------------
TEST_CASE("§9 project_heads(): inner-product follows cos(phase) law", "[Phase41][project]") {
    // Use the 1-node grid (amplitude=0.3) for analytical tractability.
    // inject (0.3 + 0i) at node 0.  At time t:
    //   V_i = (0.3·cos φ_i + i·0.3·sin φ_i)  where φ_i = 2π·f_i·t
    //   Re(⟨V_i|original⟩) = 0.3·cos(φ_i)·0.3 = cos(φ_i)·(0.3)²
    NeuroplasticTransformer npt(1);  // 1-node grid
    WaveFunction torus_wf;
    torus_wf.seed_manifold(1, 0, 1, 0.f, 0);
    torus_wf.inject(0, {0.3f, 0.0f}, 1.f);

    const float t_val = 0.1f;
    torus_wf.advance_time(t_val);   // t = 0.1s
    npt.project_heads(torus_wf);

    const double p_torus = torus_wf.total_probability();   // ≈ 0.09

    for (size_t i = 0; i < NPT_NUM_HEADS; ++i) {
        const double phi_i    = 2.0 * NPT_PI * npt.head_frequency(i) * t_val;
        const double expected = std::cos(phi_i) * p_torus;
        const double ip       = npt.head(i).V.inner_product_re(torus_wf);
        INFO("head " << i << " phi=" << phi_i
             << "  expected=" << expected << "  actual=" << ip);
        REQUIRE(ip == Approx(expected).epsilon(1e-4));
    }
}

// ---------------------------------------------------------------------------
// §10  forward() at t=0: output prob ≈ torus prob
// ---------------------------------------------------------------------------
TEST_CASE("§10 NPT forward() at t=0: output prob == torus prob", "[Phase41][forward]") {
    // At t=0 all phase rotations are 0 → V_i = torus_wf → convex combination
    // of identical fields → output = torus_wf → same total_probability.
    NeuroplasticTransformer npt(3, 1.0f, 0.0f);  // alpha=0 for determinism
    WaveFunction torus_wf;
    torus_wf.seed_manifold(3, 3, 1, 1.f, 42);   // time() == 0

    auto result = npt.forward(torus_wf);

    REQUIRE(result.output.total_probability()
            == Approx(torus_wf.total_probability()).epsilon(1e-4));
}

// ---------------------------------------------------------------------------
// §11  forward() at t>0: output is_finite and has energy
// ---------------------------------------------------------------------------
TEST_CASE("§11 NPT forward() at t>0: output finite and non-zero", "[Phase41][forward]") {
    NeuroplasticTransformer npt(3);
    WaveFunction torus_wf;
    torus_wf.seed_manifold(3, 3, 1, 1.f, 42);
    torus_wf.advance_time(1.0f);

    auto result = npt.forward(torus_wf);
    REQUIRE(result.output.is_finite());
    REQUIRE(result.output.total_probability() > 1e-10);
}

// ---------------------------------------------------------------------------
// §12  forward(): has_output == true
// ---------------------------------------------------------------------------
TEST_CASE("§12 NPT forward(): has_output == true", "[Phase41][forward]") {
    NeuroplasticTransformer npt(3);
    WaveFunction torus_wf;
    torus_wf.seed_manifold(3, 3, 1, 1.f, 42);
    REQUIRE(npt.forward(torus_wf).has_output == true);
}

// ---------------------------------------------------------------------------
// §13  forward(): scores sum to 1.0
// ---------------------------------------------------------------------------
TEST_CASE("§13 NPT forward(): scores sum to 1.0", "[Phase41][forward]") {
    NeuroplasticTransformer npt(3);
    WaveFunction torus_wf;
    torus_wf.seed_manifold(3, 3, 1, 1.f, 42);
    torus_wf.advance_time(0.7f);
    auto result = npt.forward(torus_wf);
    REQUIRE(sum8(result.head_scores) == Approx(1.0f).epsilon(1e-5f));
}

// ---------------------------------------------------------------------------
// §14  forward() at t>0: output inner product with torus < torus prob
//       (frequency differentiation is live — output is not a plain torus copy)
// ---------------------------------------------------------------------------
TEST_CASE("§14 NPT forward() at t>0: output != torus (phase differentiation live)",
          "[Phase41][forward]") {
    // At t=1s, the dominant head (head 7, highest weight via curvature bias)
    // has phase φ_7 = 2π·f_7·1  ≈  2π·91.21  ≈ 573 rad.
    // cos(573 mod 2π) is some value ≠ 1  →  Re(⟨output|torus⟩) < |torus|²
    NeuroplasticTransformer npt(3, 1.0f, 0.5f);   // default alpha
    WaveFunction torus_wf;
    torus_wf.seed_manifold(3, 3, 1, 1.f, 42);
    torus_wf.advance_time(1.0f);

    auto result = npt.forward(torus_wf);
    const double ip  = result.output.inner_product_re(torus_wf);
    const double p_t = torus_wf.total_probability();

    INFO("Re(⟨output|torus⟩) = " << ip << "  torus.prob = " << p_t);
    // Cannot equal p_t exactly because scores are non-uniform and V_i differ
    // A strict less-than holds unless frequencies accidentally align at t=1s
    REQUIRE(ip < p_t * 0.9999);
}

// ---------------------------------------------------------------------------
// §15  forward(): output node count matches torus
// ---------------------------------------------------------------------------
TEST_CASE("§15 NPT forward(): output node count matches torus", "[Phase41][forward]") {
    NeuroplasticTransformer npt(3);
    WaveFunction torus_wf;
    torus_wf.seed_manifold(3, 3, 1, 1.f, 42);
    torus_wf.advance_time(0.3f);

    auto result = npt.forward(torus_wf);
    REQUIRE(result.output.num_nodes() == torus_wf.num_nodes());
}
