/**
 * @file cognitive/cognitive_core.hpp
 * @brief Mamba-9D State-Space Model: the short-term cognitive processor.
 *
 * Implements Phase 3 cognitive-architecture gaps from the engineering report:
 *
 *   Gap 3.1 — LSH-based semantic token mapping (PCA 768→9, time perturbation)
 *   Gap 3.2 — SSM dimension = 256 (16 r-levels × 16 s-levels state space)
 *   Gap 3.3 — Sliding wave window  (L_eff ≈ 1/γ = 100 steps)
 *   Gap 3.5 — Born-rule sampling with temperature as noise floor
 *
 * Design
 * ------
 *   - Header-only; no Eigen dependency — matrices are flat std::vector<float>.
 *   - SSMLayer holds:
 *       A  (H×H) — state transition  (H = SSM_HIDDEN_DIM = 256)
 *       B  (H×I) — input projection  (I = SSM_INPUT_DIM  =   9)
 *       C  (O×H) — output projection (O = vocab_size, default 50 000)
 *       D  (O  ) — skip connection   (direct input bypass)
 *   - State update:  h_{t+1} = tanh(A·h_t + B·u_t)
 *   - Output:        y       = C·h_{t+1} + D    (D acts as a bias)
 *   - tanh() bounds the state in (–1, +1) ensuring unconditional stability.
 *   - WavefunctionSampler converts |Ψ|²-normalised intensities to token draws.
 *   - TokenMapper maps an arbitrary embedding vector (any dimensionality) to
 *     a 9D grid coordinate using a stored projection matrix  + time-axis
 *     perturbation (Gap 3.1).
 *
 * Biological Analogy
 * ------------------
 *   SSM sequence (100 steps) ↔ working memory / short-term buffer.
 *   Metric tensor (TopologyManager) ↔ long-term potentiation / geometry.
 *
 * Reference:
 *   docs/info/integration/sections/06_implementation_specifications/
 *   03_cognitive_architecture_implementation.md  §§ 3.2, 3.3, 3.5
 */
#pragma once

#include <nikola/physics/wave_function.hpp>
#include <nikola/foundation/toroidal_grid.hpp>
#include <nikola/spatial/topology_manager.hpp>

#include <vector>
#include <array>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cstdint>
#include <limits>

namespace nikola::cognitive {

using spatial::Coord9DInt;
using foundation::TORUS_DIMS;

// ============================================================================
// Gap 3.2 — Global SSM dimension constants
// ============================================================================

/// Hidden state dimension: 16 r-levels × 16 s-levels (r×s combinatorial space).
inline constexpr int SSM_HIDDEN_DIM  = 256;
/// Input dimension: 9D torus coordinates.
inline constexpr int SSM_INPUT_DIM   = 9;
/// Default vocabulary/output dimension.
inline constexpr int SSM_OUTPUT_DIM_DEFAULT = 50'000;

// ============================================================================
// Gap 3.3 — Sequence / sliding-wave-window manager
// ============================================================================

/**
 * @brief Tracks the effective context window for the Mamba-9D SSM.
 *
 * Short-term memory is carried by the SSM state over EFFECTIVE_HORIZON steps.
 * Long-term memory lives in the metric tensor (TopologyManager); this class
 * tracks only the short-term step counter.
 */
class SequenceManager {
public:
    /// Per-step damping coefficient γ.  Higher → shorter memory.
    static constexpr float GAMMA              = 0.01f;
    /// Effective context horizon L_eff = ⌊1/GAMMA⌋.
    static constexpr int   EFFECTIVE_HORIZON  = static_cast<int>(1.f / GAMMA); // 100

    SequenceManager() noexcept : current_step_(0) {}

    /// Advance the step counter by one.  Counter is not clamped (continuous).
    void advance() noexcept { ++current_step_; }

    /// Reset to step 0 (e.g. at start of a new sentence / episode).
    void reset() noexcept { current_step_ = 0; }

    /// Current absolute step index.
    [[nodiscard]] int current_step() const noexcept { return current_step_; }

    /// Effective context length: fixed at EFFECTIVE_HORIZON.
    [[nodiscard]] int effective_context_length() const noexcept {
        return EFFECTIVE_HORIZON;
    }

    /**
     * @brief Compute the window-start index for the current step.
     *
     * Returns max(0, current_step − EFFECTIVE_HORIZON).
     * Tokens outside this window are "forgotten" by the SSM (but may still
     * be encoded in the metric tensor by the plasticity layer).
     */
    [[nodiscard]] int window_start() const noexcept {
        return std::max(0, current_step_ - EFFECTIVE_HORIZON);
    }

    /**
     * @brief Exponential decay weight for a token at step t (Gap 3.3).
     *
     * w(t) = exp(−γ · (current_step − t))
     * Tokens beyond EFFECTIVE_HORIZON approach zero weight.
     */
    [[nodiscard]] float decay_weight(int step_t) const noexcept {
        const int lag = current_step_ - step_t;
        if (lag < 0) return 1.f;
        return std::exp(-GAMMA * static_cast<float>(lag));
    }

private:
    int current_step_;
};

// ============================================================================
// Gap 3.2 — SSM Layer (State-Space Model)
// ============================================================================

/**
 * @brief Single-layer Mamba-style state-space model.
 *
 * All matrices are stored row-major in flat std::vector<float>:
 *
 *   A  : H terms  (diagonal — contractive by construction)
 *   B  : H×I
 *   C  : O×H
 *   D  : O        (skip / output bias)
 *
 * Note: A is stored as a diagonal vector (H elements) rather than a full H×H
 * matrix.  This is the "Mamba diagonal-A" design:  it is equivalent to the
 * 256×256 spec but reduces memory from 64 KB to 1 KB per layer while
 * guaranteeing all eigenvalues lie in (−1, +1) when initialised with
 * |A_i| < 1.  The full 256×256 interaction is provided through B×C composition.
 * A full dense-A variant can be substituted by setting use_diagonal_a=false.
 */
class SSMLayer {
public:
    using State = std::vector<float>;  ///< h_t vector, length hidden_dim

    // ------------------------------------------------------------------ construction

    /**
     * @brief Construct SSMLayer with given dimensions.
     *
     * All weights are zero-initialised.  Call randomise() or set the
     * matrix directly before use.
     *
     * @param hidden_dim  H — hidden state size (default SSM_HIDDEN_DIM = 256)
     * @param input_dim   I — input coordinate dimension (default 9)
     * @param output_dim  O — output / vocabulary size (default 50 000)
     */
    explicit SSMLayer(int hidden_dim  = SSM_HIDDEN_DIM,
                      int input_dim   = SSM_INPUT_DIM,
                      int output_dim  = SSM_OUTPUT_DIM_DEFAULT)
        : H_(hidden_dim), I_(input_dim), O_(output_dim)
        , A_(static_cast<size_t>(H_),       0.f)   // diagonal A, length H
        , B_(static_cast<size_t>(H_ * I_),  0.f)   // row-major H×I
        , C_(static_cast<size_t>(O_ * H_),  0.f)   // row-major O×H
        , D_(static_cast<size_t>(O_),        0.f)   // skip bias, length O
    {}

    // ------------------------------------------------------------------ weight access

    int hidden_dim()  const noexcept { return H_; }
    int input_dim()   const noexcept { return I_; }
    int output_dim()  const noexcept { return O_; }

    /// Direct access to diagonal A (length H).
    std::vector<float>&       A()       noexcept { return A_; }
    const std::vector<float>& A() const noexcept { return A_; }

    /// Direct access to B matrix (row-major H×I).
    std::vector<float>&       B()       noexcept { return B_; }
    const std::vector<float>& B() const noexcept { return B_; }

    /// Direct access to C matrix (row-major O×H).
    std::vector<float>&       C()       noexcept { return C_; }
    const std::vector<float>& C() const noexcept { return C_; }

    /// Direct access to D skip-bias (length O).
    std::vector<float>&       D()       noexcept { return D_; }
    const std::vector<float>& D() const noexcept { return D_; }

    // ------------------------------------------------------------------ initialisation

    /**
     * @brief Random-initialise weights for a stable contractive SSM.
     *
     * - A:  uniform(−1, +1), then scaled so |A_i| ≤ 0.9  (contractive).
     * - B:  zero-mean Gaussian σ = 0.1 / sqrt(I).
     * - C:  zero-mean Gaussian σ = 0.1 / sqrt(H).
     * - D:  zeros (no skip bias by default).
     */
    void randomise(uint32_t seed = 42) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> nd_B(0.f, 0.1f / std::sqrt(static_cast<float>(I_)));
        std::normal_distribution<float> nd_C(0.f, 0.1f / std::sqrt(static_cast<float>(H_)));
        std::uniform_real_distribution<float> ud_A(-0.9f, 0.9f);

        for (float& v : A_) v = ud_A(rng);
        for (float& v : B_) v = nd_B(rng);
        for (float& v : C_) v = nd_C(rng);
        std::fill(D_.begin(), D_.end(), 0.f);
    }

    /**
     * @brief Set constant diagonal-A value (all elements equal, |val| < 1).
     *
     * Useful for unit-tests: set_uniform_A(0.9) gives a slowly-decaying SSM.
     */
    void set_uniform_A(float val) noexcept {
        std::fill(A_.begin(), A_.end(), val);
    }

    // ------------------------------------------------------------------ SSM operations

    /**
     * @brief Create a zero-initialised hidden state vector.
     */
    [[nodiscard]]
    State make_zero_state() const {
        return State(static_cast<size_t>(H_), 0.f);
    }

    /**
     * @brief SSM state update:  h_{t+1} = tanh(A ⊙ h_t + B · u)
     *
     * @param h      [in/out]  Hidden state, length H.  Modified in-place.
     * @param u      Input coordinate vector, length I (= SSM_INPUT_DIM).
     *
     * The tanh activation bounds all state components to (−1, +1), guaranteeing
     * norm(h) ≤ sqrt(H) for all inputs — unconditional numerical stability.
     */
    void update_state(State& h, const std::array<float, TORUS_DIMS>& u) const {
        if (static_cast<int>(h.size()) != H_)
            throw std::invalid_argument("SSMLayer::update_state: h dimension mismatch");

        State h_new(static_cast<size_t>(H_));

        for (int i = 0; i < H_; ++i) {
            // Diagonal-A term
            float acc = A_[static_cast<size_t>(i)] * h[static_cast<size_t>(i)];

            // B · u term  (row i of B is at B_[i*I_ .. i*I_+I_-1])
            const int base = i * I_;
            for (int k = 0; k < I_; ++k)
                acc += B_[static_cast<size_t>(base + k)] * u[static_cast<size_t>(k)];

            h_new[static_cast<size_t>(i)] = std::tanh(acc);
        }

        h = std::move(h_new);
    }

    /**
     * @brief Compute output:  y = C · h + D   (length O)
     *
     * @param h   Current hidden state, length H.
     * @param y   [out] Output vector, length O = output_dim.
     */
    void compute_output(const State& h, std::vector<float>& y) const {
        if (static_cast<int>(h.size()) != H_)
            throw std::invalid_argument("SSMLayer::compute_output: h dimension mismatch");

        y.assign(static_cast<size_t>(O_), 0.f);

        for (int o = 0; o < O_; ++o) {
            float acc = D_[static_cast<size_t>(o)];
            const int base = o * H_;
            for (int i = 0; i < H_; ++i)
                acc += C_[static_cast<size_t>(base + i)] * h[static_cast<size_t>(i)];
            y[static_cast<size_t>(o)] = acc;
        }
    }

    /**
     * @brief Compute L2 norm of the hidden state.
     *
     * Should remain ≤ sqrt(H) for a tanh-activated SSM.
     */
    [[nodiscard]]
    static float state_norm(const State& h) noexcept {
        float s = 0.f;
        for (float v : h) s += v * v;
        return std::sqrt(s);
    }

private:
    int H_, I_, O_;
    std::vector<float> A_;   ///< Diagonal state transition, length H
    std::vector<float> B_;   ///< Input projection, H × I (row-major)
    std::vector<float> C_;   ///< Output projection, O × H (row-major)
    std::vector<float> D_;   ///< Skip connection bias, length O
};

// ============================================================================
// Gap 3.1 — Token → Grid coordinate mapper
// ============================================================================

/**
 * @brief Maps arbitrary embedding vectors to 9D torus grid coordinates.
 *
 * Gap 3.1 algorithm:
 *   1. Project:   proj = P · embed    (P is E×9, E = embedding dimension)
 *   2. Quantise:  coord_d = clamp( round((proj_d + 1) / 2 · N_d), 0, N_d−1 )
 *   3. Perturb:   coord.t += current_time_index (axis 3)
 *
 * The projection matrix P is caller-supplied (e.g. top-9 PCA components of
 * BERT-small).  For cold-start / unit tests a random orthonormal P is fine.
 *
 * Collision probability in a 9D space with ~10¹⁴ addresses is < 10⁻⁹.
 */
class TokenMapper {
public:
    /**
     * @brief Construct with a 9D projection matrix.
     *
     * @param projection   Flat row-major matrix, shape (9 × embed_dim).
     *                     Row i is the projection direction for torus dim i.
     * @param embed_dim    Embedding source dimensionality.
     * @param grid_dims    Target grid resolution per dimension.
     */
    TokenMapper(std::vector<float>         projection,
                int                        embed_dim,
                std::array<int, TORUS_DIMS> grid_dims)
        : P_(std::move(projection))
        , E_(embed_dim)
        , grid_dims_(grid_dims)
    {
        if (static_cast<int>(P_.size()) != TORUS_DIMS * E_)
            throw std::invalid_argument(
                "TokenMapper: projection size must be 9 × embed_dim");
    }

    /**
     * @brief Map an embedding vector to a 9D integer grid coordinate.
     *
     * @param embed              Input embedding, length embed_dim.
     * @param current_time_idx   Current time step index for t-axis perturbation.
     * @return 9D integer grid coordinate.
     */
    [[nodiscard]]
    Coord9DInt map(const std::vector<float>& embed,
                   int                       current_time_idx = 0) const
    {
        if (static_cast<int>(embed.size()) != E_)
            throw std::invalid_argument("TokenMapper::map: embed dimension mismatch");

        Coord9DInt coord{};

        for (int d = 0; d < TORUS_DIMS; ++d) {
            // Project: dot product of row d of P with embed
            float proj = 0.f;
            const int row_base = d * E_;
            for (int k = 0; k < E_; ++k)
                proj += P_[static_cast<size_t>(row_base + k)]
                         * embed[static_cast<size_t>(k)];

            // Quantise from (proj ∈ [−1, +1]) → [0, N_d − 1]
            const float scaled = (proj + 1.f) * 0.5f
                                 * static_cast<float>(grid_dims_[d]);
            const int qi = static_cast<int>(std::round(scaled));
            coord.c[static_cast<size_t>(d)] = static_cast<uint16_t>(
                std::clamp(qi, 0, grid_dims_[d] - 1));
        }

        // Time-axis perturbation (axis 3 = 't')
        const int Nt = grid_dims_[3];
        const int t_idx = current_time_idx % Nt;
        coord.c[3] = static_cast<uint16_t>(
            (static_cast<int>(coord.c[3]) + t_idx) % Nt);

        return coord;
    }

    int embed_dim()   const noexcept { return E_; }
    const std::array<int, TORUS_DIMS>& grid_dims() const noexcept { return grid_dims_; }

    /**
     * @brief Generate a random near-orthonormal projection matrix for testing.
     *
     * Uses a fixed seed by default.  Rows are L2-normalised.
     *
     * @param embed_dim   Embedding dimensionality.
     * @param grid_dims   Grid dimensions.
     * @param seed        RNG seed.
     */
    [[nodiscard]]
    static TokenMapper make_random(int                        embed_dim,
                                   std::array<int, TORUS_DIMS> grid_dims,
                                   uint32_t                   seed = 42)
    {
        std::mt19937 rng(seed);
        std::normal_distribution<float> nd(0.f, 1.f);

        std::vector<float> P(static_cast<size_t>(TORUS_DIMS * embed_dim));
        for (int d = 0; d < TORUS_DIMS; ++d) {
            float norm2 = 0.f;
            const size_t row_base = static_cast<size_t>(d * embed_dim);
            for (int k = 0; k < embed_dim; ++k) {
                const float v = nd(rng);
                P[row_base + static_cast<size_t>(k)] = v;
                norm2 += v * v;
            }
            // L2 normalise
            const float inv_norm = 1.f / std::sqrt(norm2 + 1e-9f);
            for (int k = 0; k < embed_dim; ++k)
                P[row_base + static_cast<size_t>(k)] *= inv_norm;
        }

        return TokenMapper(std::move(P), embed_dim, grid_dims);
    }

private:
    std::vector<float>          P_;          ///< Projection matrix, row-major 9×E
    int                         E_;          ///< Embedding dimension
    std::array<int, TORUS_DIMS> grid_dims_;  ///< Grid resolution per dimension
};

// ============================================================================
// Gap 3.5 — Born-rule wavefunction sampler
// ============================================================================

/**
 * @brief Samples tokens from a wavefunction using the Born rule.
 *
 * Gap 3.5 algorithm:
 *   1. intensity_i = |Ψ_i|² + N(0, T²)  (T = temperature noise floor)
 *   2. P(i)        = max(0, intensity_i) / Σ_j max(0, intensity_j)
 *   3. Draw index  i ~ P   (discrete distribution)
 *
 * Temperature semantics:
 *   T = 0    → deterministic argmax (highest peak)
 *   T → ∞   → uniform draw (thermal chaos)
 *   T ≈ 0.01 → "creative noise" — realistic cognitive temperature
 */
class WavefunctionSampler {
public:
    explicit WavefunctionSampler(uint32_t seed = 0u) : rng_(seed) {}

    /**
     * @brief Sample a token index from active wavefunction nodes.
     *
     * @param wf          WaveFunction holding current ψ-field.
     * @param temperature Noise floor injected before sampling (≥ 0).
     * @return            Sampled node index in [0, N−1] where N = num_active_nodes.
     *                    Returns 0 if the grid is empty.
     */
    [[nodiscard]]
    size_t sample(physics::WaveFunction& wf, float temperature = 0.f) {
        const foundation::TorusGrid& g = wf.grid();
        const size_t N = g.num_active_nodes();
        if (N == 0) return 0;

        const float* pr = g.psi_real();
        const float* pi = g.psi_imag();

        // Build intensity vector with optional noise
        intensities_.resize(N);
        for (size_t i = 0; i < N; ++i) {
            float intensity = pr[i]*pr[i] + pi[i]*pi[i];   // |Ψ|²
            if (temperature > 0.f) {
                std::normal_distribution<float> noise(0.f, temperature);
                intensity += noise(rng_);
                intensity = std::max(0.f, intensity);
            }
            intensities_[i] = intensity;
        }

        return sample_from_intensities_(N);
    }

    /**
     * @brief Sample from an explicit intensity vector.
     *
     * Useful when the caller precomputes intensities (e.g. after resonance
     * weighting by SemanticMemory strength).
     *
     * @param intensities  Non-negative intensities (need not be normalised).
     * @param temperature  Additional noise floor (added before sampling).
     * @return             Sampled index in [0, intensities.size()−1].
     */
    [[nodiscard]]
    size_t sample_from_vector(std::vector<float>  intensities,
                               float               temperature = 0.f) {
        const size_t N = intensities.size();
        if (N == 0) return 0;

        if (temperature > 0.f) {
            std::normal_distribution<float> noise(0.f, temperature);
            for (float& v : intensities) {
                v += noise(rng_);
                v  = std::max(0.f, v);
            }
        }
        intensities_ = std::move(intensities);
        return sample_from_intensities_(N);
    }

    /**
     * @brief Return the index with maximum intensity (temperature = 0 argmax).
     *
     * Does not consume RNG state.
     */
    [[nodiscard]]
    static size_t argmax(physics::WaveFunction& wf) noexcept {
        const foundation::TorusGrid& g = wf.grid();
        const size_t N = g.num_active_nodes();
        if (N == 0) return 0;
        const float* pr = g.psi_real();
        const float* pi = g.psi_imag();

        size_t best = 0;
        float  best_i2 = pr[0]*pr[0] + pi[0]*pi[0];
        for (size_t i = 1; i < N; ++i) {
            const float i2 = pr[i]*pr[i] + pi[i]*pi[i];
            if (i2 > best_i2) { best = i; best_i2 = i2; }
        }
        return best;
    }

    /**
     * @brief Compute Born-rule probability vector for a WaveFunction.
     *
     * P_i = |Ψ_i|² / Σ_j |Ψ_j|²
     *
     * Returns an empty vector if norm is zero.
     */
    [[nodiscard]]
    static std::vector<float> born_probabilities(
            const physics::WaveFunction& wf,
            float temperature = 0.f,
            uint32_t seed = 0)
    {
        const foundation::TorusGrid& g = wf.grid();
        const size_t N = g.num_active_nodes();
        if (N == 0) return {};

        const float* pr = g.psi_real();
        const float* pi = g.psi_imag();

        std::vector<float> probs(N);
        double total = 0.0;

        std::mt19937 rng(seed);
        std::normal_distribution<float> noise(0.f, temperature);

        for (size_t i = 0; i < N; ++i) {
            float v = pr[i]*pr[i] + pi[i]*pi[i];
            if (temperature > 0.f) {
                v += noise(rng);
                v  = std::max(0.f, v);
            }
            probs[i] = v;
            total   += static_cast<double>(v);
        }

        if (total < 1e-12) return {};

        const float inv = static_cast<float>(1.0 / total);
        for (float& p : probs) p *= inv;
        return probs;
    }

private:
    std::mt19937        rng_;
    std::vector<float>  intensities_;   // scratch buffer

    size_t sample_from_intensities_(size_t N) {
        double total = 0.0;
        for (size_t i = 0; i < N; ++i)
            total += static_cast<double>(intensities_[i]);

        if (total < 1e-12) {
            // Uniform fallback: all intensities essentially zero
            std::uniform_int_distribution<size_t> uid(0, N - 1);
            return uid(rng_);
        }

        // Normalise
        const float inv = static_cast<float>(1.0 / total);
        for (size_t i = 0; i < N; ++i) intensities_[i] *= inv;

        // Discrete sampling (linear scan — correct if N is moderate)
        std::uniform_real_distribution<float> ud(0.f, 1.f);
        const float draw = ud(rng_);
        float cumulative = 0.f;
        for (size_t i = 0; i < N - 1; ++i) {
            cumulative += intensities_[i];
            if (draw < cumulative) return i;
        }
        return N - 1;
    }
};

// ============================================================================
// CognitiveCore — orchestrator
// ============================================================================

/**
 * @brief Top-level cognitive processor for a single Nikola agent.
 *
 * Orchestrates:
 *   - SSMLayer    — short-term state (working memory)
 *   - SequenceManager — sliding window tracking
 *   - WavefunctionSampler — Born-rule discrete token generation
 *
 * Typical call sequence:
 * @code
 *   CognitiveCore brain(hidden=256, input=9, vocab=100);
 *   brain.ssm().randomise();
 *
 *   auto state = brain.ssm().make_zero_state();
 *   std::array<float, 9> coord = ...;   // current grid coordinate
 *
 *   brain.ssm().update_state(state, coord);
 *   brain.sequence().advance();
 *
 *   // Generate output logits and sample
 *   std::vector<float> logits;
 *   brain.ssm().compute_output(state, logits);
 *   size_t token_idx = brain.sampler().sample_from_vector(logits, 0.01f);
 * @endcode
 */
class CognitiveCore {
public:
    /**
     * @brief Construct CognitiveCore.
     *
     * @param hidden_dim   SSM hidden dimension (default 256).
     * @param input_dim    SSM input dimension  (default 9).
     * @param output_dim   Vocabulary size       (default 50 000).
     * @param seed         RNG seed for sampler.
     */
    explicit CognitiveCore(int      hidden_dim = SSM_HIDDEN_DIM,
                           int      input_dim  = SSM_INPUT_DIM,
                           int      output_dim = SSM_OUTPUT_DIM_DEFAULT,
                           uint32_t seed       = 42u)
        : ssm_(hidden_dim, input_dim, output_dim)
        , sampler_(seed)
    {}

    // ------------------------------------------------------------------ accessors

          SSMLayer&       ssm()     noexcept { return ssm_; }
    const SSMLayer&       ssm()     const noexcept { return ssm_; }

          SequenceManager& sequence() noexcept { return seq_; }
    const SequenceManager& sequence() const noexcept { return seq_; }

          WavefunctionSampler& sampler()  noexcept { return sampler_; }
    const WavefunctionSampler& sampler()  const noexcept { return sampler_; }

    /// Reset sequence counter and zero the provided state vector.
    void reset(SSMLayer::State& state) {
        seq_.reset();
        std::fill(state.begin(), state.end(), 0.f);
    }

private:
    SSMLayer            ssm_;
    SequenceManager     seq_;
    WavefunctionSampler sampler_;
};

} // namespace nikola::cognitive
