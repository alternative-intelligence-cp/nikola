// ============================================================================
// phase37_npt_structure_test.cpp   Phase 37 — NPT Data Structures & Skeleton
// ============================================================================
//
// Tests:
//   §1  NeuroplasticTransformer constructs without throwing
//   §2  NPT has exactly 8 heads after construction
//   §3  Each head frequency matches π·φⁿ from HolographicInjector
//   §4  npt_all_frequencies() helper is consistent with per-head values
//   §5  Each head has the correct CognitiveBand enum value
//   §6  band_name() returns non-empty strings for all 8 bands
//   §7  NPT grid_n() accessor matches construction argument
//   §8  NPT temperature() accessor matches construction argument
//   §9  forward() stub returns AttentionResult with 8 head_scores
//   §10 forward() stub: all head_scores are zero, has_output is false
//   §11 forward() stub: output WaveFunction is finite (no NaN/inf)
//   §12 AttentionResult output grid_n matches NPT grid_n
//   §13 NPT is non-copyable (compile-time trait check)
//   §14 WaveCorrelationHead frequencies are unique (no two heads share a band)
//   §15 NPT default-constructed (grid_n=3) matches torus default construction
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/cognitive/neuroplastic_transformer.hpp>
#include <nikola/physics/wave_function.hpp>

#include <cmath>
#include <set>
#include <type_traits>

using namespace nikola::cognitive;
using namespace nikola::physics;
namespace physics = nikola::physics;  // allow physics::WaveFunction shorthand
using Approx = Catch::Approx;

// ---------------------------------------------------------------------------
// Expected frequencies: π·φⁿ — computed independently to cross-validate
// ---------------------------------------------------------------------------
static double expected_freq(size_t n) {
    static constexpr double PHI = 1.6180339887498948482;
    static constexpr double PI  = 3.1415926535897932385;
    double phi_n = 1.0;
    for (size_t i = 0; i < n; ++i) phi_n *= PHI;
    return PI * phi_n;
}

// ---------------------------------------------------------------------------
// §1  NeuroplasticTransformer constructs without throwing
// ---------------------------------------------------------------------------
TEST_CASE("§1 NPT: constructs without throwing", "[Phase37][npt]") {
    REQUIRE_NOTHROW(NeuroplasticTransformer(3));
    REQUIRE_NOTHROW(NeuroplasticTransformer(2));
    REQUIRE_NOTHROW(NeuroplasticTransformer(3, 0.5f));
}

// ---------------------------------------------------------------------------
// §2  NPT has exactly 8 heads after construction
// ---------------------------------------------------------------------------
TEST_CASE("§2 NPT: exactly 8 heads", "[Phase37][npt]") {
    NeuroplasticTransformer npt(3);
    REQUIRE(npt.num_heads() == 8);
    REQUIRE(npt.num_heads() == NPT_NUM_HEADS);
}

// ---------------------------------------------------------------------------
// §3  Each head frequency matches π·φⁿ
// ---------------------------------------------------------------------------
TEST_CASE("§3 NPT: head frequencies match π·φⁿ", "[Phase37][npt]") {
    NeuroplasticTransformer npt(3);

    for (size_t i = 0; i < NPT_NUM_HEADS; ++i) {
        const double expected = expected_freq(i);
        const double actual   = npt.head_frequency(i);
        INFO("head " << i << ": expected=" << expected << " actual=" << actual);
        REQUIRE(actual == Approx(expected).epsilon(1e-9));
    }
}

// ---------------------------------------------------------------------------
// §4  npt_all_frequencies() is consistent with per-head values
// ---------------------------------------------------------------------------
TEST_CASE("§4 NPT: npt_all_frequencies() consistent", "[Phase37][npt]") {
    const auto freqs = npt_all_frequencies();
    NeuroplasticTransformer npt(3);

    REQUIRE(freqs.size() == NPT_NUM_HEADS);
    for (size_t i = 0; i < NPT_NUM_HEADS; ++i) {
        INFO("i=" << i);
        REQUIRE(freqs[i] == Approx(npt.head_frequency(i)).epsilon(1e-9));
    }

    // Frequencies must be strictly ascending
    for (size_t i = 1; i < NPT_NUM_HEADS; ++i) {
        REQUIRE(freqs[i] > freqs[i - 1]);
    }
}

// ---------------------------------------------------------------------------
// §5  Each head has the correct CognitiveBand enum value
// ---------------------------------------------------------------------------
TEST_CASE("§5 NPT: head bands are correct", "[Phase37][npt]") {
    NeuroplasticTransformer npt(3);

    REQUIRE(npt.head_band(0) == CognitiveBand::GLOBAL_CONTEXT);
    REQUIRE(npt.head_band(1) == CognitiveBand::LONG_TERM_MEMORY);
    REQUIRE(npt.head_band(2) == CognitiveBand::WORKING_MEMORY);
    REQUIRE(npt.head_band(3) == CognitiveBand::LOGIC_REASONING_LO);
    REQUIRE(npt.head_band(4) == CognitiveBand::LOGIC_REASONING_HI);
    REQUIRE(npt.head_band(5) == CognitiveBand::SENSORY_INTEGRATION);
    REQUIRE(npt.head_band(6) == CognitiveBand::FINE_DETAIL);
    REQUIRE(npt.head_band(7) == CognitiveBand::ERROR_CORRECTION);
}

// ---------------------------------------------------------------------------
// §6  band_name() returns correct non-empty strings for all 8 bands
// ---------------------------------------------------------------------------
TEST_CASE("§6 NPT: band_name() returns correct strings", "[Phase37][npt]") {
    using B = CognitiveBand;

    REQUIRE(std::string(band_name(B::GLOBAL_CONTEXT))      == "GLOBAL_CONTEXT");
    REQUIRE(std::string(band_name(B::LONG_TERM_MEMORY))    == "LONG_TERM_MEMORY");
    REQUIRE(std::string(band_name(B::WORKING_MEMORY))      == "WORKING_MEMORY");
    REQUIRE(std::string(band_name(B::LOGIC_REASONING_LO))  == "LOGIC_REASONING_LO");
    REQUIRE(std::string(band_name(B::LOGIC_REASONING_HI))  == "LOGIC_REASONING_HI");
    REQUIRE(std::string(band_name(B::SENSORY_INTEGRATION)) == "SENSORY_INTEGRATION");
    REQUIRE(std::string(band_name(B::FINE_DETAIL))         == "FINE_DETAIL");
    REQUIRE(std::string(band_name(B::ERROR_CORRECTION))    == "ERROR_CORRECTION");
}

// ---------------------------------------------------------------------------
// §7  NPT grid_n() accessor matches construction argument
// ---------------------------------------------------------------------------
TEST_CASE("§7 NPT: grid_n() accessor", "[Phase37][npt]") {
    REQUIRE(NeuroplasticTransformer(3).grid_n() == 3);
    REQUIRE(NeuroplasticTransformer(2).grid_n() == 2);
}

// ---------------------------------------------------------------------------
// §8  NPT temperature() accessor matches construction argument
// ---------------------------------------------------------------------------
TEST_CASE("§8 NPT: temperature() accessor", "[Phase37][npt]") {
    REQUIRE(NeuroplasticTransformer(3, 1.0f).temperature() == Approx(1.0f));
    REQUIRE(NeuroplasticTransformer(3, 0.5f).temperature() == Approx(0.5f));
    REQUIRE(NeuroplasticTransformer(3, 2.0f).temperature() == Approx(2.0f));
}

// ---------------------------------------------------------------------------
// §9  forward() stub returns AttentionResult with head_scores of correct size
// ---------------------------------------------------------------------------
TEST_CASE("§9 NPT: forward() returns AttentionResult with 8 scores", "[Phase37][npt]") {
    NeuroplasticTransformer npt(3);

    physics::WaveFunction torus_wf;
    torus_wf.seed_manifold(3, 3, 1, 1.f, 42);

    auto result = npt.forward(torus_wf);

    REQUIRE(result.head_scores.size() == NPT_NUM_HEADS);
}

// ---------------------------------------------------------------------------
// §10 forward(): vacuum Q + no bias → uniform 1/8 softmax, has_output true
// ---------------------------------------------------------------------------
TEST_CASE("§10 NPT: vacuum Q + alpha=0 → uniform softmax", "[Phase37][npt]") {
    // Disable curvature bias (alpha=0) so we isolate the correlation kernel.
    // All head Q fields are vacuum → wave_correlation = 0 for all heads.
    // softmax([0,0,...,0]) = [1/8, ..., 1/8].
    NeuroplasticTransformer npt(3, 1.0f, 0.0f);   // alpha=0

    physics::WaveFunction torus_wf;
    torus_wf.seed_manifold(3, 3, 1, 1.f, 42);

    auto result = npt.forward(torus_wf);

    const float expected = 1.f / static_cast<float>(NPT_NUM_HEADS);
    for (size_t i = 0; i < NPT_NUM_HEADS; ++i) {
        INFO("head " << i << " score = " << result.head_scores[i]);
        REQUIRE(result.head_scores[i] == Approx(expected).epsilon(1e-5f));
    }
    REQUIRE(result.has_output == true);  // Phase 40: aggregation now active
}

// ---------------------------------------------------------------------------
// §11 forward() stub: output WaveFunction is finite (no NaN / inf)
// ---------------------------------------------------------------------------
TEST_CASE("§11 NPT: forward() output WaveFunction is finite", "[Phase37][npt]") {
    NeuroplasticTransformer npt(3);

    physics::WaveFunction torus_wf;
    torus_wf.seed_manifold(3, 3, 1, 1.f, 42);

    auto result = npt.forward(torus_wf);

    REQUIRE(result.output.is_finite());
    REQUIRE(result.output.num_nodes() > 0);
}

// ---------------------------------------------------------------------------
// §12 AttentionResult output WaveFunction has correct node count
// ---------------------------------------------------------------------------
TEST_CASE("§12 NPT: AttentionResult output node count matches grid", "[Phase37][npt]") {
    // grid_n=3 → 3^9 = 19683 nodes
    NeuroplasticTransformer npt3(3);
    physics::WaveFunction wf3;
    wf3.seed_manifold(3, 3, 1, 1.f, 42);
    auto r3 = npt3.forward(wf3);

    // grid_n=2 → 2^9 = 512 nodes
    NeuroplasticTransformer npt2(2);
    physics::WaveFunction wf2;
    wf2.seed_manifold(2, 3, 1, 1.f, 42);
    auto r2 = npt2.forward(wf2);

    // Output WaveFunctions should have node counts matching their grid_n
    const size_t expected3 = wf3.num_nodes();   // 19683
    const size_t expected2 = wf2.num_nodes();   // 512
    REQUIRE(r3.output.num_nodes() == expected3);
    REQUIRE(r2.output.num_nodes() == expected2);
}

// ---------------------------------------------------------------------------
// §13 NPT is non-copyable (compile-time type trait check)
// ---------------------------------------------------------------------------
TEST_CASE("§13 NPT: non-copyable type trait", "[Phase37][npt]") {
    REQUIRE_FALSE(std::is_copy_constructible_v<NeuroplasticTransformer>);
    REQUIRE_FALSE(std::is_copy_assignable_v<NeuroplasticTransformer>);
    // But must be movable
    REQUIRE(std::is_move_constructible_v<NeuroplasticTransformer>);
}

// ---------------------------------------------------------------------------
// §14 Head frequencies are all unique
// ---------------------------------------------------------------------------
TEST_CASE("§14 NPT: head frequencies are all unique", "[Phase37][npt]") {
    NeuroplasticTransformer npt(3);

    // Insert into a set; if any two are equal, set size < 8
    std::set<double> freq_set;
    for (size_t i = 0; i < NPT_NUM_HEADS; ++i)
        freq_set.insert(npt.head_frequency(i));

    REQUIRE(freq_set.size() == NPT_NUM_HEADS);
}

// ---------------------------------------------------------------------------
// §15 NPT default grid_n=3 matches CognitiveTorus default
// ---------------------------------------------------------------------------
TEST_CASE("§15 NPT: default grid_n=3 matches CognitiveTorus default", "[Phase37][npt]") {
    // The CognitiveTorus default is n=3 (3^9 = 19683 nodes).
    // An NPT constructed with the default must produce AttentionResult output
    // of the same node count.
    NeuroplasticTransformer npt;   // default grid_n=3

    REQUIRE(npt.grid_n() == 3);

    physics::WaveFunction torus_wf;
    torus_wf.seed_manifold(3, 3, 1, 1.f, 42);

    auto result = npt.forward(torus_wf);

    REQUIRE(result.output.num_nodes() == torus_wf.num_nodes());
}
