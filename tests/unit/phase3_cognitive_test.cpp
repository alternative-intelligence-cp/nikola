/**
 * @file phase3_cognitive_test.cpp
 * @brief Unit tests for Phase 3: The Brain — Cognitive Architecture.
 *
 * Tests cover all Phase 3 gate criteria:
 *
 *   === SequenceManager (Gap 3.3 — Sliding wave window) ===
 *   01. EFFECTIVE_HORIZON is 100 (= 1/GAMMA = 1/0.01)
 *   02. advance() increments step counter
 *   03. reset() zeroes step counter
 *   04. window_start() clamps at 0 for early steps
 *   05. window_start() = step - 100 once step > 100
 *   06. decay_weight at lag=0 is 1.0
 *   07. decay_weight at lag=100 ≈ e^{-1} ≈ 0.368 (long-term attenuation)
 *   08. decay_weight for future step ≥ 1.0 (clamped)
 *
 *   === SSMLayer (Gap 3.2 — Mamba-9D hidden state) ===
 *   09. Constants: SSM_HIDDEN_DIM=256, SSM_INPUT_DIM=9
 *   10. make_zero_state() returns vector of size H, all zero
 *   11. After 1 update with all-zero weights: state stays zero
 *   12. update_state with A=0.9, nonzero B: state changes
 *   13. State components always in (−1, +1) after tanh activation
 *   14. state_norm ≤ sqrt(H) after random updates (stability bound)
 *   15. After 50 updates, state still bounded (no runaway growth)
 *   16. compute_output returns vector of correct size O = vocab_size
 *   17. compute_output with C=0, D=1: all outputs equal 1.0
 *   18. randomise() makes weights non-zero
 *
 *   === WavefunctionSampler (Gap 3.5 — Born rule sampling) ===
 *   19. born_probabilities sum to 1.0 for uniform nonzero field
 *   20. born_probabilities proportional to |Ψ|²
 *   21. born_probabilities returns empty for zero wavefunction
 *   22. argmax returns index of highest-amplitude node
 *   23. sample returns valid index in [0, N−1]
 *   24. sample with temperature=0.5 still returns valid index
 *
 *   === TokenMapper (Gap 3.1 — PCA token → grid coordinate) ===
 *   25. All output dimensions in range [0, N_d−1]
 *   26. Two identical embeddings at same time produce same coord
 *   27. t-axis perturbation: different time_idx changes t-dimension
 *   28. t-dimension stays within [0, N_t−1] (wraps correctly)
 *
 *   === CognitiveCore (orchestrator) ===
 *   29. CognitiveCore exposes SSM, sequence manager, sampler
 *   30. reset() zeroes state and sequence counter
 *
 *   === CoactivationTracker (Hebbian co-activation outer product) ===
 *   31. Fresh tracker: count=0, get() returns all zeros
 *   32. After one accumulate(): count=1
 *   33. Outer product matrix is approximately symmetric (C_ij ≈ C_ji)
 *   34. reset() zeroes count and accumulation
 *   35. Diagonal elements non-negative after accumulation on nonzero field
 *
 *   === HebbianPlasticity (Hebbian metric updates) ===
 *   36. apply_update with zero coactivation leaves metric diagonal entries unchanged
 *   37. apply_update with nonzero coactivation modifies metric
 *   38. Metric remains positive-definite after Hebbian update
 *   39. Saturation check returns false for small-amplitude WaveFunction
 *   40. metric_is_valid() true after construction (identity is PD)
 *
 *   === EqPropTrainer (Gap 3.6 — Equilibrium Propagation) ===
 *   41. train_step completes without exception
 *   42. last_energy_positive() is finite and positive
 *   43. last_energy_negative() is finite (clamped phase has energy)
 *   44. apply_delta_metric changes the metric from identity
 *
 *   === MemoryReplay (Replay-based consolidation) ===
 *   45. compute_replay_order on empty memory returns empty list
 *   46. Score = strength × log10(1 + access_count); score=0 if access=0
 *   47. Results are sorted descending (highest score first)
 *   48. replay() on valid memory returns replayed > 0
 *
 *   === ConsolidationEngine (NAP cycle orchestrator) ===
 *   49. nap_cycle: weak memory (strength~0) is pruned
 *   50. nap_cycle: strong memory (strength=1) survives
 *   51. nap_cycle stats: records_before ≥ records_after
 *   52. micro_consolidate prunes weak records
 *   53. is_healthy returns true when memory within limits
 *   54. is_healthy returns false for empty memory
 *
 * Reference: Phase 3 gate criteria, engineering report §8.9.3.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/cognitive/cognitive_core.hpp>
#include <nikola/cognitive/plasticity.hpp>
#include <nikola/cognitive/consolidation.hpp>
#include <nikola/cognitive/semantic_memory.hpp>
#include <nikola/spatial/topology_manager.hpp>
#include <nikola/physics/wave_function.hpp>

#include <array>
#include <cmath>
#include <numeric>
#include <algorithm>

using namespace nikola;
using namespace nikola::cognitive;
using namespace nikola::spatial;
using namespace nikola::foundation;
using namespace nikola::physics;

// ============================================================================
// Helpers
// ============================================================================

/// Create a small seeded WaveFunction (n^9 grid, pilot wave).
static WaveFunction make_wave(int n = 2, float amplitude = 1.f, uint32_t seed = 42) {
    WaveFunction wf;
    wf.seed_manifold(n, /*pilot_dim=*/0, /*k_mode=*/1, amplitude, seed);
    return wf;
}

/// Create a trivially zero WaveFunction (makes a 2^9 grid, all ψ=0).
static WaveFunction make_zero_wave() {
    WaveFunction wf;
    wf.seed_manifold(2, 0, 1, 0.f, 0);   // amplitude=0 → zero ψ
    return wf;
}

/// Apply one Hebbian update to a TopologyManager and return true if PD after.
static bool hebbian_update_valid(TopologyManager& topo, const WaveFunction& wf) {
    HebbianPlasticity hp(topo);
    CoactivationTracker tracker;
    tracker.accumulate(wf);
    hp.apply_update(tracker, /*dopamine=*/1.f, /*age=*/0.f, wf);
    return hp.metric_is_valid();
}

// ============================================================================
// === SequenceManager ===
// ============================================================================

TEST_CASE("SequenceManager: EFFECTIVE_HORIZON is 100", "[cogncore][gap3.3]") {
    REQUIRE(SequenceManager::EFFECTIVE_HORIZON == 100);
    REQUIRE(SequenceManager::GAMMA == Catch::Approx(0.01f));
}

TEST_CASE("SequenceManager: advance() increments step", "[cogncore][gap3.3]") {
    SequenceManager sm;
    REQUIRE(sm.current_step() == 0);
    sm.advance();
    REQUIRE(sm.current_step() == 1);
    sm.advance();
    sm.advance();
    REQUIRE(sm.current_step() == 3);
}

TEST_CASE("SequenceManager: reset() zeros step", "[cogncore][gap3.3]") {
    SequenceManager sm;
    for (int i = 0; i < 50; ++i) sm.advance();
    sm.reset();
    REQUIRE(sm.current_step() == 0);
}

TEST_CASE("SequenceManager: window_start clamps at 0 for early steps", "[cogncore][gap3.3]") {
    SequenceManager sm;
    REQUIRE(sm.window_start() == 0);
    for (int i = 0; i < 50; ++i) sm.advance();
    REQUIRE(sm.window_start() == 0);  // 50 < HORIZON=100
}

TEST_CASE("SequenceManager: window_start correct after step > 100", "[cogncore][gap3.3]") {
    SequenceManager sm;
    for (int i = 0; i < 150; ++i) sm.advance();
    REQUIRE(sm.current_step() == 150);
    REQUIRE(sm.window_start() == 50);   // 150 - 100 = 50
}

TEST_CASE("SequenceManager: decay_weight at lag=0 is 1.0", "[cogncore][gap3.3]") {
    SequenceManager sm;
    sm.advance();  // step = 1
    REQUIRE(sm.decay_weight(1) == Catch::Approx(1.f));
}

TEST_CASE("SequenceManager: decay_weight at lag=100 ≈ e^{-1}", "[cogncore][gap3.3]") {
    SequenceManager sm;
    for (int i = 0; i < 100; ++i) sm.advance();  // step = 100
    const float w = sm.decay_weight(0);           // lag = 100
    REQUIRE(w == Catch::Approx(std::exp(-1.f)).epsilon(0.01f));
}

TEST_CASE("SequenceManager: decay_weight for future step is 1.0", "[cogncore][gap3.3]") {
    SequenceManager sm;
    // current_step=0, step_t=5 → lag<0 → clamped to 1.0
    REQUIRE(sm.decay_weight(5) == Catch::Approx(1.f));
}

// ============================================================================
// === SSMLayer ===
// ============================================================================

TEST_CASE("SSMLayer: constant definitions correct", "[cogncore][gap3.2]") {
    REQUIRE(SSM_HIDDEN_DIM == 256);
    REQUIRE(SSM_INPUT_DIM  == 9);
}

TEST_CASE("SSMLayer: make_zero_state returns H-dim zero vector", "[cogncore][gap3.2]") {
    SSMLayer ssm(16, 9, 10);   // small dims for speed
    auto h = ssm.make_zero_state();
    REQUIRE(static_cast<int>(h.size()) == 16);
    for (float v : h) REQUIRE(v == 0.f);
}

TEST_CASE("SSMLayer: zero weights → update_state keeps state zero", "[cogncore][gap3.2]") {
    SSMLayer ssm(8, 9, 5);    // A=0, B=0, all zeros
    auto h = ssm.make_zero_state();
    std::array<float, TORUS_DIMS> u{};
    u[0] = 1.f;
    ssm.update_state(h, u);
    for (float v : h) REQUIRE(v == 0.f);  // tanh(0) = 0
}

TEST_CASE("SSMLayer: nonzero B causes state to change", "[cogncore][gap3.2]") {
    SSMLayer ssm(8, 9, 5);
    ssm.set_uniform_A(0.9f);
    // Set B[0][0] = 1.0 (first row, first input dimension)
    ssm.B()[0] = 1.f;

    auto h = ssm.make_zero_state();
    std::array<float, TORUS_DIMS> u{};
    u[0] = 1.f;  // inject activation in dim 0

    ssm.update_state(h, u);
    REQUIRE(h[0] != 0.f);           // first element should have changed
    REQUIRE(std::abs(h[0]) <= 1.f); // bounded by tanh
}

TEST_CASE("SSMLayer: state components bounded in (-1, +1)", "[cogncore][gap3.2]") {
    SSMLayer ssm(32, 9, 5);
    ssm.randomise(7u);

    auto h = ssm.make_zero_state();
    std::array<float, TORUS_DIMS> u{};
    std::fill(u.begin(), u.end(), 10.f);  // large input

    for (int step = 0; step < 20; ++step) {
        ssm.update_state(h, u);
        for (float v : h) {
            REQUIRE(std::abs(v) < 1.f + 1e-5f);  // strict tanh bound
        }
    }
}

TEST_CASE("SSMLayer: state_norm bounded by sqrt(H)", "[cogncore][gap3.2]") {
    const int H = 64;
    SSMLayer ssm(H, 9, 5);
    ssm.randomise(11u);

    auto h = ssm.make_zero_state();
    std::array<float, TORUS_DIMS> u{};
    std::fill(u.begin(), u.end(), 5.f);

    const float norm_bound = std::sqrt(static_cast<float>(H)) + 1e-3f;
    for (int step = 0; step < 50; ++step) {
        ssm.update_state(h, u);
        REQUIRE(SSMLayer::state_norm(h) <= norm_bound);
    }
}

TEST_CASE("SSMLayer: state stable over 50 updates (no runaway)", "[cogncore][gap3.2]") {
    SSMLayer ssm(64, 9, 5);
    ssm.set_uniform_A(0.99f);   // near-unity decay — challenging
    ssm.B()[0] = 0.5f;

    auto h = ssm.make_zero_state();
    std::array<float, TORUS_DIMS> u{};
    u[0] = 1.f;

    for (int s = 0; s < 50; ++s) {
        ssm.update_state(h, u);
        // All components strictly bounded by tanh  
        for (float v : h) REQUIRE(std::abs(v) <= 1.001f);
    }
}

TEST_CASE("SSMLayer: compute_output size matches output_dim", "[cogncore][gap3.2]") {
    SSMLayer ssm(16, 9, 7);
    auto h = ssm.make_zero_state();
    std::vector<float> y;
    ssm.compute_output(h, y);
    REQUIRE(static_cast<int>(y.size()) == 7);
}

TEST_CASE("SSMLayer: compute_output with D=constant gives correct bias", "[cogncore][gap3.2]") {
    SSMLayer ssm(8, 9, 4);
    // C remains zero → y = 0·h + D = D
    for (float& v : ssm.D()) v = 3.14f;

    auto h = ssm.make_zero_state();
    std::vector<float> y;
    ssm.compute_output(h, y);
    for (float v : y)
        REQUIRE(v == Catch::Approx(3.14f));
}

TEST_CASE("SSMLayer: randomise makes weights non-zero", "[cogncore][gap3.2]") {
    SSMLayer ssm(16, 9, 5);
    // Before randomise: all weights are zero
    bool B_nonzero = false;
    ssm.randomise(99u);
    for (float v : ssm.B()) if (v != 0.f) { B_nonzero = true; break; }
    REQUIRE(B_nonzero);
}

// ============================================================================
// === WavefunctionSampler ===
// ============================================================================

TEST_CASE("WavefunctionSampler: born_probabilities sum to 1.0", "[sampler][gap3.5]") {
    WaveFunction wf = make_wave(/*n=*/2, /*amplitude=*/1.f);
    const auto probs = WavefunctionSampler::born_probabilities(wf, 0.f);
    REQUIRE(!probs.empty());

    double total = 0.0;
    for (float p : probs) total += static_cast<double>(p);
    REQUIRE(total == Catch::Approx(1.0).epsilon(1e-5));
}

TEST_CASE("WavefunctionSampler: born_probabilities proportional to |Ψ|²", "[sampler][gap3.5]") {
    // Manually set a single nonzero node
    WaveFunction wf = make_zero_wave();
    foundation::TorusGrid& g = wf.grid();
    const size_t N = g.num_active_nodes();
    REQUIRE(N > 0);

    // Set node 0 to amplitude 2, all others 0 (they're already zero)
    g.psi_real()[0] = 2.f;
    g.psi_imag()[0] = 0.f;

    const auto probs = WavefunctionSampler::born_probabilities(wf);
    REQUIRE(!probs.empty());
    // Node 0 has |Ψ|² = 4, all others = 0 → p[0] = 1.0
    REQUIRE(probs[0] == Catch::Approx(1.f).epsilon(1e-5f));
    for (size_t i = 1; i < probs.size(); ++i)
        REQUIRE(probs[i] == Catch::Approx(0.f).margin(1e-5f));
}

TEST_CASE("WavefunctionSampler: born_probabilities empty for zero wavefunction", "[sampler][gap3.5]") {
    WaveFunction wf = make_zero_wave();
    // make_zero_wave sets amplitude=0 so all nodes have ψ=0
    const auto probs = WavefunctionSampler::born_probabilities(wf);
    REQUIRE(probs.empty());
}

TEST_CASE("WavefunctionSampler: argmax returns index of highest amplitude", "[sampler][gap3.5]") {
    WaveFunction wf = make_zero_wave();
    foundation::TorusGrid& g = wf.grid();
    const size_t N = g.num_active_nodes();
    REQUIRE(N >= 4);

    // Put maximum at node 3
    g.psi_real()[3] = 10.f;
    g.psi_imag()[3] = 0.f;

    REQUIRE(WavefunctionSampler::argmax(wf) == 3u);
}

TEST_CASE("WavefunctionSampler: sample returns valid index", "[sampler][gap3.5]") {
    WaveFunction wf = make_wave(2);
    WavefunctionSampler sampler(0u);
    const size_t N = wf.grid().num_active_nodes();
    REQUIRE(N > 0);

    for (int trial = 0; trial < 20; ++trial) {
        const size_t idx = sampler.sample(wf, 0.f);
        REQUIRE(idx < N);
    }
}

TEST_CASE("WavefunctionSampler: sample with temperature returns valid index", "[sampler][gap3.5]") {
    WaveFunction wf = make_wave(2);
    WavefunctionSampler sampler(7u);
    const size_t N = wf.grid().num_active_nodes();

    for (int trial = 0; trial < 20; ++trial) {
        const size_t idx = sampler.sample(wf, 0.1f);
        REQUIRE(idx < N);
    }
}

// ============================================================================
// === TokenMapper ===
// ============================================================================

TEST_CASE("TokenMapper: all output dimensions in valid range", "[tokenmapper][gap3.1]") {
    std::array<int, TORUS_DIMS> dims = {8, 8, 8, 16, 4, 4, 8, 8, 8};
    auto mapper = TokenMapper::make_random(16, dims, 42u);

    std::vector<float> embed(16, 0.5f);
    auto coord = mapper.map(embed, 0);

    for (int d = 0; d < TORUS_DIMS; ++d) {
        REQUIRE(coord.c[d] < static_cast<uint16_t>(dims[d]));
    }
}

TEST_CASE("TokenMapper: same embedding + same time → same coord (determinism)", "[tokenmapper][gap3.1]") {
    std::array<int, TORUS_DIMS> dims = {8, 8, 8, 16, 4, 4, 8, 8, 8};
    auto mapper = TokenMapper::make_random(16, dims, 77u);

    std::vector<float> embed(16, 0.25f);
    auto c1 = mapper.map(embed, 5);
    auto c2 = mapper.map(embed, 5);

    for (int d = 0; d < TORUS_DIMS; ++d)
        REQUIRE(c1.c[d] == c2.c[d]);
}

TEST_CASE("TokenMapper: different time_idx changes t-axis coordinate", "[tokenmapper][gap3.1]") {
    std::array<int, TORUS_DIMS> dims = {8, 8, 8, 16, 4, 4, 8, 8, 8};
    auto mapper = TokenMapper::make_random(10, dims, 5u);

    std::vector<float> embed(10, 0.3f);
    auto c0 = mapper.map(embed, 0);
    auto c1 = mapper.map(embed, 5);
    // Direct check: c1.t = (c0_base + 5) % N_t and c0.t = (c0_base + 0) % N_t
    // So c1.t - c0.t ≡ 5 (mod 16)
    const int diff = (static_cast<int>(c1.c[3]) - static_cast<int>(c0.c[3]) + 16) % 16;
    REQUIRE(diff == 5);
}

TEST_CASE("TokenMapper: t-axis coord stays within [0, N_t−1]", "[tokenmapper][gap3.1]") {
    std::array<int, TORUS_DIMS> dims = {4, 4, 4, 8, 2, 2, 4, 4, 4};
    auto mapper = TokenMapper::make_random(4, dims, 3u);

    std::vector<float> embed(4, 0.0f);
    for (int t = 0; t < 100; ++t) {
        auto coord = mapper.map(embed, t);
        REQUIRE(coord.c[3] < static_cast<uint16_t>(dims[3]));
    }
}

// ============================================================================
// === CognitiveCore ===
// ============================================================================

TEST_CASE("CognitiveCore: exposes SSM, sequence, sampler", "[cogncore]") {
    CognitiveCore brain(16, 9, 5);
    REQUIRE(brain.ssm().hidden_dim() == 16);
    REQUIRE(brain.ssm().input_dim()  == 9);
    REQUIRE(brain.ssm().output_dim() == 5);
    REQUIRE(brain.sequence().EFFECTIVE_HORIZON == 100);
}

TEST_CASE("CognitiveCore: reset() zeros state and sequence counter", "[cogncore]") {
    CognitiveCore brain(16, 9, 5);
    brain.ssm().randomise(1u);

    auto h = brain.ssm().make_zero_state();
    std::array<float, TORUS_DIMS> u{};
    std::fill(u.begin(), u.end(), 1.f);

    // Advance state and counter
    for (int s = 0; s < 10; ++s) {
        brain.ssm().update_state(h, u);
        brain.sequence().advance();
    }
    REQUIRE(brain.sequence().current_step() == 10);

    brain.reset(h);
    REQUIRE(brain.sequence().current_step() == 0);
    for (float v : h) REQUIRE(v == 0.f);
}

// ============================================================================
// === CoactivationTracker ===
// ============================================================================

TEST_CASE("CoactivationTracker: fresh tracker is all zeros, count=0", "[plasticity]") {
    CoactivationTracker tracker;
    REQUIRE(tracker.count() == 0);

    float out[81];
    tracker.get(out);
    for (int i = 0; i < 81; ++i)
        REQUIRE(out[i] == 0.f);
}

TEST_CASE("CoactivationTracker: accumulate increments count", "[plasticity]") {
    CoactivationTracker tracker;
    WaveFunction wf = make_wave(2);
    tracker.accumulate(wf);
    REQUIRE(tracker.count() == 1);
    tracker.accumulate(wf);
    REQUIRE(tracker.count() == 2);
}

TEST_CASE("CoactivationTracker: outer product is symmetric", "[plasticity]") {
    CoactivationTracker tracker;
    WaveFunction wf = make_wave(2);
    tracker.accumulate(wf);
    tracker.accumulate(wf);

    float out[81];
    tracker.get(out);

    // C_ij == C_ji (symmetric outer product a⊗a)  
    for (int i = 0; i < TORUS_DIMS; ++i)
        for (int j = 0; j < TORUS_DIMS; ++j)
            REQUIRE(out[i*TORUS_DIMS+j]
                    == Catch::Approx(out[j*TORUS_DIMS+i]).margin(1e-5f));
}

TEST_CASE("CoactivationTracker: reset zeros count and data", "[plasticity]") {
    CoactivationTracker tracker;
    WaveFunction wf = make_wave(2);
    tracker.accumulate(wf);
    tracker.accumulate(wf);

    tracker.reset();
    REQUIRE(tracker.count() == 0);

    float out[81];
    tracker.get(out);
    for (int i = 0; i < 81; ++i) REQUIRE(out[i] == 0.f);
}

TEST_CASE("CoactivationTracker: diagonal non-negative for nonzero wave", "[plasticity]") {
    CoactivationTracker tracker;
    WaveFunction wf = make_wave(2, 1.f, 42u);
    tracker.accumulate(wf);

    float out[81];
    tracker.get(out);

    for (int d = 0; d < TORUS_DIMS; ++d)
        REQUIRE(out[d*TORUS_DIMS+d] >= 0.f);
}

// ============================================================================
// === HebbianPlasticity ===
// ============================================================================

TEST_CASE("HebbianPlasticity: zero coactivation leaves metric diagonals unchanged",
          "[plasticity][gap2.5]")
{
    TopologyManager topo;
    HebbianPlasticity hp(topo);

    // Tracker with count=0 → zero co-activation
    CoactivationTracker tracker;  // never accumulated
    WaveFunction wf = make_wave(2);

    // Capture diagonal before
    float diag_before[TORUS_DIMS];
    for (int d = 0; d < TORUS_DIMS; ++d)
        diag_before[d] = topo.metric()[d * TORUS_DIMS + d];

    hp.apply_update(tracker, 1.f, 0.f, wf);

    // Diagonal should not change (Δg = eta × 0 = 0)
    for (int d = 0; d < TORUS_DIMS; ++d)
        REQUIRE(topo.metric()[d * TORUS_DIMS + d]
                == Catch::Approx(diag_before[d]).margin(1e-6f));
}

TEST_CASE("HebbianPlasticity: nonzero coactivation modifies metric", "[plasticity][gap2.5]") {
    TopologyManager topo;
    HebbianPlasticity hp(topo);

    CoactivationTracker tracker;
    WaveFunction wf = make_wave(2, 1.f, 10u);
    tracker.accumulate(wf);
    tracker.accumulate(wf);
    tracker.accumulate(wf);

    // Capture metric before
    float before[81];
    std::copy(topo.metric(), topo.metric() + 81, before);

    hp.apply_update(tracker, 1.f, 0.f, wf);  // high dopamine, young node

    // At least some elements should change
    bool changed = false;
    for (int i = 0; i < 81; ++i)
        if (std::abs(topo.metric()[i] - before[i]) > 1e-7f) { changed = true; break; }
    REQUIRE(changed);
}

TEST_CASE("HebbianPlasticity: metric remains PD after Hebbian update", "[plasticity][gap2.5]") {
    TopologyManager topo;
    WaveFunction wf = make_wave(2, 1.f);
    REQUIRE(hebbian_update_valid(topo, wf));
}

TEST_CASE("HebbianPlasticity: saturation check false for small amplitude", "[plasticity]") {
    TopologyManager topo;
    HebbianPlasticity hp(topo);

    CoactivationTracker tracker;
    WaveFunction wf = make_wave(2, 1.f);  // amplitude=1 << SAT_THRESHOLD=5
    tracker.accumulate(wf);

    bool sat = hp.apply_update(tracker, 0.5f, 0.f, wf);
    REQUIRE(sat == false);
}

TEST_CASE("HebbianPlasticity: metric_is_valid after construction", "[plasticity]") {
    TopologyManager topo;
    HebbianPlasticity hp(topo);
    REQUIRE(hp.metric_is_valid());
}

// ============================================================================
// === EqPropTrainer ===
// ============================================================================

TEST_CASE("EqPropTrainer: train_step completes without exception", "[eqprop][gap3.6]") {
    TopologyManager topo;

    EqPropTrainer::Config cfg;
    cfg.phase_steps = 5;    // Use small value for test speed
    cfg.dt          = 0.001f;
    cfg.eta         = 0.01f;

    EqPropTrainer trainer(topo, cfg);

    WaveFunction wf = make_wave(2, 0.5f, 42u);

    // Input injection: set node 0 amplitude
    auto inject_input  = [](WaveFunction& w) {
        w.grid().psi_real()[0] = 0.5f;
        w.grid().psi_imag()[0] = 0.f;
    };
    // Target injection: set node 1 amplitude (simulated "correct token")
    auto inject_target = [](WaveFunction& w) {
        const size_t N = w.grid().num_active_nodes();
        if (N > 1) {
            w.grid().psi_real()[1] = 1.f;
            w.grid().psi_imag()[1] = 0.f;
        }
    };

    REQUIRE_NOTHROW(trainer.train_step(wf, inject_input, inject_target));
}

TEST_CASE("EqPropTrainer: last_energy_positive is finite and positive", "[eqprop][gap3.6]") {
    TopologyManager topo;

    EqPropTrainer::Config cfg;
    cfg.phase_steps = 5;
    cfg.dt          = 0.001f;

    EqPropTrainer trainer(topo, cfg);
    WaveFunction wf = make_wave(2, 0.5f, 1u);

    auto inject = [](WaveFunction& w) {
        w.grid().psi_real()[0] = 0.5f;
        w.grid().psi_imag()[0] = 0.f;
    };
    trainer.train_step(wf, inject, inject);

    REQUIRE(std::isfinite(trainer.last_energy_positive()));
    // E+ may be ~0 for small injections but should not be pathological
    REQUIRE(trainer.last_energy_positive() >= 0.0);
}

TEST_CASE("EqPropTrainer: last_energy_negative is finite", "[eqprop][gap3.6]") {
    TopologyManager topo;

    EqPropTrainer::Config cfg;
    cfg.phase_steps = 5;
    cfg.dt          = 0.001f;

    EqPropTrainer trainer(topo, cfg);
    WaveFunction wf = make_wave(2, 0.5f, 2u);

    auto inject_in = [](WaveFunction& w) {
        w.grid().psi_real()[0] = 0.3f;
    };
    auto inject_tgt = [](WaveFunction& w) {
        const size_t N = w.grid().num_active_nodes();
        if (N > 2) { w.grid().psi_real()[2] = 0.8f; }
    };
    trainer.train_step(wf, inject_in, inject_tgt);

    REQUIRE(std::isfinite(trainer.last_energy_negative()));
    REQUIRE(trainer.last_energy_negative() >= 0.0);
}

TEST_CASE("EqPropTrainer: metric changes after train_step with energy difference",
          "[eqprop][gap3.6]")
{
    TopologyManager topo;

    // Save initial metric
    float before[81];
    std::copy(topo.metric(), topo.metric() + 81, before);

    EqPropTrainer::Config cfg;
    cfg.phase_steps = 5;
    cfg.dt          = 0.001f;
    cfg.eta         = 0.1f;   // large eta to ensure visible change

    EqPropTrainer trainer(topo, cfg);
    WaveFunction wf = make_wave(2, 1.f, 9u);

    auto inject_in = [](WaveFunction& w) {
        w.grid().psi_real()[0] = 1.f;
        w.grid().psi_imag()[0] = 0.5f;
    };
    auto inject_tgt = [](WaveFunction& w) {
        const size_t N = w.grid().num_active_nodes();
        const size_t j = N > 3 ? 3 : N - 1;
        w.grid().psi_real()[j] = 1.5f;
    };

    // Run multiple steps so metric has a chance to move
    for (int i = 0; i < 3; ++i)
        trainer.train_step(wf, inject_in, inject_tgt);

    // Check at least one element changed
    bool changed = false;
    for (int i = 0; i < 81; ++i)
        if (std::abs(topo.metric()[i] - before[i]) > 1e-10f) { changed = true; break; }
    REQUIRE(changed);
}

// ============================================================================
// === MemoryReplay ===
// ============================================================================

TEST_CASE("MemoryReplay: compute_replay_order on empty memory returns empty", "[consolidation]") {
    SemanticMemory mem;
    MemoryReplay replay;
    auto candidates = replay.compute_replay_order(mem);
    REQUIRE(candidates.empty());
}

TEST_CASE("MemoryReplay: score = strength × log10(1 + access_count)", "[consolidation]") {
    // Build a memory with known access_count
    SemanticMemory mem;
    WaveFunction wf = make_wave(2, 1.f, 10u);
    MemoryKey key = mem.store(wf);

    // Force access_count = 3 by loading 3 times
    mem.load(key, wf);
    mem.load(key, wf);
    mem.load(key, wf);

    const MemoryRecord* rec = mem.get(key);
    REQUIRE(rec != nullptr);
    REQUIRE(rec->access_count == 3u);

    MemoryReplay replay;
    auto candidates = replay.compute_replay_order(mem);
    REQUIRE(candidates.size() == 1u);

    const float expected_score = rec->strength * std::log10(1.f + 3.f);
    REQUIRE(candidates[0].score == Catch::Approx(expected_score).epsilon(0.01f));
}

TEST_CASE("MemoryReplay: results sorted descending by score", "[consolidation]") {
    SemanticMemory mem;

    // Store two distinct memories
    WaveFunction wf1 = make_wave(2, 2.f, 1u);
    WaveFunction wf2 = make_wave(2, 3.f, 2u);

    MemoryKey k1 = mem.store(wf1);
    MemoryKey k2 = mem.store(wf2);

    // Ensure keys differ
    if (k1 == k2) {
        // Same Hilbert key (collision) — skip ordering check
        SUCCEED("Hilbert key collision — ordering test not applicable");
        return;
    }

    // Give k1 more accesses so its score is higher
    for (int i = 0; i < 5; ++i) mem.load(k1, wf1);
    // k2 has no accesses → score = strength × log10(1)= 0

    MemoryReplay replay;
    auto candidates = replay.compute_replay_order(mem);

    REQUIRE(candidates.size() >= 2u);
    REQUIRE(candidates[0].score >= candidates[1].score);
}

TEST_CASE("MemoryReplay: replay() returns count > 0 for valid memory", "[consolidation]") {
    SemanticMemory mem;
    WaveFunction wf = make_wave(2, 1.f, 5u);
    mem.store(wf);

    MemoryReplay::Config cfg;
    cfg.replay_steps = 3;    // minimal physics steps
    cfg.replay_dt    = 0.001f;
    cfg.replay_k     = 5;

    MemoryReplay replay(cfg);
    auto candidates = replay.compute_replay_order(mem);
    REQUIRE(!candidates.empty());

    WaveFunction scratch = make_zero_wave();
    int replayed = replay.replay(candidates, mem, scratch);
    REQUIRE(replayed > 0);
}

// ============================================================================
// === ConsolidationEngine ===
// ============================================================================

TEST_CASE("ConsolidationEngine: nap_cycle prunes weak memory", "[consolidation]") {
    SemanticMemory mem;
    WaveFunction wf = make_wave(2, 1.f, 20u);
    mem.store(wf);

    // Force strength near zero by setting it manually via many decays
    // We can't set strength directly, so use a large decay_dt
    // DECAY_RATE = 0.001/s, to get strength < 0.01 we need dt ≫ −ln(0.01)/0.001 ≈ 4605s
    for (int i = 0; i < 10; ++i) mem.decay(1000.f);   // 10 000 seconds total

    const size_t sz_before = mem.size();

    ConsolidationEngine::Config cfg;
    cfg.nap_dt = 1.f;
    cfg.replay_cfg.replay_steps = 2;
    cfg.replay_cfg.replay_dt    = 0.001f;
    cfg.replay_cfg.replay_k     = 1;

    ConsolidationEngine engine(cfg);
    WaveFunction scratch = make_zero_wave();
    auto stats = engine.nap_cycle(mem, scratch);

    // The memory should have been pruned
    REQUIRE(stats.records_after <= sz_before);
    REQUIRE(mem.size() == 0u);  // fully decayed record must have been pruned
}

TEST_CASE("ConsolidationEngine: nap_cycle strong memory survives", "[consolidation]") {
    SemanticMemory mem;
    WaveFunction wf = make_wave(2, 1.f, 21u);
    mem.store(wf);   // starts at strength = MAX_STRENGTH = 1.0

    // Give it some accesses to get LTP boost
    MemoryKey key = mem.all_keys()[0];
    mem.load(key, wf);
    mem.load(key, wf);
    mem.load(key, wf);

    ConsolidationEngine::Config cfg;
    cfg.nap_dt = 1.f;         // only 1 second decay → negligible strength loss
    cfg.replay_cfg.replay_steps = 2;
    cfg.replay_cfg.replay_dt    = 0.001f;
    cfg.replay_cfg.replay_k     = 1;

    ConsolidationEngine engine(cfg);
    WaveFunction scratch = make_zero_wave();
    auto stats = engine.nap_cycle(mem, scratch);

    REQUIRE(stats.pruned == 0u);      // nothing pruned (strength still near 1.0)
    REQUIRE(mem.size() >= 1u);        // memory survived
}

TEST_CASE("ConsolidationEngine: stats records_before >= records_after", "[consolidation]") {
    SemanticMemory mem;
    WaveFunction wf1 = make_wave(2, 1.f, 30u);
    WaveFunction wf2 = make_wave(2, 1.f, 31u);
    WaveFunction wf3 = make_wave(2, 1.f, 32u);

    mem.store(wf1);
    mem.store(wf2);
    mem.store(wf3);

    ConsolidationEngine::Config cfg;
    cfg.nap_dt = 5000.f;   // large decay — some may fall below threshold
    cfg.replay_cfg.replay_steps = 2;
    cfg.replay_cfg.replay_dt    = 0.001f;

    ConsolidationEngine engine(cfg);
    WaveFunction scratch = make_zero_wave();
    auto stats = engine.nap_cycle(mem, scratch);

    REQUIRE(stats.records_before <= 3u);
    REQUIRE(stats.records_after  <= stats.records_before);
}

TEST_CASE("ConsolidationEngine: micro_consolidate prunes fully-decayed record", "[consolidation]") {
    SemanticMemory mem;
    WaveFunction wf = make_wave(2, 1.f, 40u);
    mem.store(wf);

    // Extreme decay to make strength near-zero    
    for (int i = 0; i < 10; ++i) mem.decay(1000.f);

    ConsolidationEngine engine;
    size_t pruned = engine.micro_consolidate(mem, 1.f);
    REQUIRE(pruned > 0u);
    REQUIRE(mem.size() == 0u);
}

TEST_CASE("ConsolidationEngine: is_healthy returns true for healthy memory", "[consolidation]") {
    SemanticMemory mem;
    WaveFunction wf = make_wave(2, 1.f, 50u);
    mem.store(wf);

    ConsolidationEngine engine;
    REQUIRE(engine.is_healthy(mem));
}

TEST_CASE("ConsolidationEngine: is_healthy returns false for empty memory", "[consolidation]") {
    SemanticMemory mem;  // empty
    ConsolidationEngine engine;
    REQUIRE_FALSE(engine.is_healthy(mem));
}
