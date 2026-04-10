#pragma once
/**
 * @file cognitive/mamba9d.hpp
 * @brief Mamba-9D State-Space Model — physics-derived SSM for the T⁹ manifold.
 *
 * v0.1.6 Phase 1: Wraps SSMLayer with physics-derived parameter extraction,
 * spectral stability clamping, and Hilbert-linearized sequence processing.
 *
 * Architecture:
 *   CognitiveTorus    → PhysicsParams (intensity, phase, resonance, ρ(G))
 *   extract_ssm_params()  → SSMParams (A diagonal, clamped Δ)
 *   Mamba9D::step()       → selective_step with live parameter adaptation
 *
 * Physical parameter mapping (§3.2.3 of Engineering Report):
 *   - Intensity |ψ|² → A diagonal (higher intensity → slower decay)
 *   - Phase arg(ψ)   → B coupling modulation (phase-aligned injection)
 *   - Curvature ρ(G) → Δ clamping via SpectralStabilizer
 *   - Resonance r    → effective stiffness (1−r)·ρ(G)
 *
 * The SSM hidden dim stays at 256 (SSM_HIDDEN_DIM); each hidden unit i
 * derives its A_i from the torus dimension (i % 9), creating a structured
 * band pattern where every 9 hidden units share physics from one dimension.
 *
 * Design:
 *   - Header-only, no new .cpp required
 *   - Uses existing SSMLayer S6 selective scan (ZOH discretization)
 *   - SpectralStabilizer provides constexpr Δ clamping
 *   - SequenceManager tracks sliding window (L_eff = 100 steps)
 */

#include <nikola/cognitive/cognitive_core.hpp>
#include <nikola/cognitive/spectral_stabilizer.hpp>
#include <nikola/foundation/toroidal_grid.hpp>
#include <nikola/spatial/hilbert_scanner.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace nikola::cognitive {

// ============================================================================
// PhysicsParams — manifold observables for one time step
// ============================================================================

/**
 * @brief Physical observables extracted from the T⁹ manifold at a given
 *        time step, used to derive SSM parameters.
 *
 * Populated from CognitiveTorus by reading hot-node neighborhoods.
 */
struct PhysicsParams {
    /// Per-dimension intensity: |ψ|² at dimension-stride neighbors of anchor.
    std::array<float, 9> intensity{};

    /// Per-dimension phase: arg(ψ) at dimension-stride neighbors of anchor.
    std::array<float, 9> phase{};

    /// Global resonance estimate in [0, 1].  Higher → less stiff → larger safe Δ.
    float resonance = 0.5f;

    /// Spectral radius of the metric tensor ρ(G).  Drives Δ clamping.
    float rho_G = 1.0f;
};

// ============================================================================
// SSMParams — derived SSM configuration for one time step
// ============================================================================

/**
 * @brief SSM parameters computed from PhysicsParams via extract_ssm_params().
 */
struct SSMParams {
    /// Diagonal A values for all H hidden units.
    /// A_i = base_decay * (1 − α · normalized_intensity[i % 9])
    /// where higher intensity → A_i closer to 1 → slower decay → more memory.
    std::array<float, SSM_HIDDEN_DIM> A_diag{};

    /// Clamped time step Δ, respecting spectral stability bound.
    float delta = 0.1f;

    /// Stability classification from SpectralStabilizer.
    StabilityCondition stability = StabilityCondition::STABLE;
};

// ============================================================================
// extract_ssm_params — map physics to SSM parameters
// ============================================================================

/**
 * @brief Map manifold physics to SSM parameters.
 *
 * Parameter mapping:
 *
 *   A (diagonal, H=256):
 *     For hidden unit i, dimension d = i % 9:
 *       norm_I = clamp(intensity[d] / max_intensity, 0, 1)
 *       A_i = -(BASE_A_MAG + INTENSITY_RANGE * norm_I)
 *     Result: A_i ∈ [-(BASE_A_MAG), -(BASE_A_MAG + INTENSITY_RANGE)]
 *     Default: A_i ∈ [-0.1, -0.9] (negative = continuous-time decay)
 *     High-intensity dims get A closer to -0.9 (slower discrete decay
 *     after ZOH: Ā = exp(Δ·A), so more negative A → faster decay, but
 *     we want high intensity → MORE memory, so we use |A| large for low
 *     intensity and |A| small for high intensity).
 *
 *     CORRECTION: For Mamba ZOH where Ā = exp(Δ·A) and A < 0:
 *       - Small |A| (e.g. -0.1) → Ā ≈ exp(-0.01) ≈ 0.99 → SLOW decay
 *       - Large |A| (e.g. -0.9) → Ā ≈ exp(-0.09) ≈ 0.91 → FAST decay
 *     So high-intensity nodes should get SMALL |A| → slow decay → memory.
 *       A_i = -(BASE_A_MAG + INTENSITY_RANGE * (1 - norm_I))
 *
 *   Δ (time step):
 *     Requested from the dominant phase velocity; clamped to safe bound.
 *     Δ_requested = BASE_DELTA + PHASE_DELTA_SCALE * mean(|phase|)
 *     Δ_clamped   = clamp_delta(Δ_requested, ρ(G), resonance)
 *
 * @param physics  Physical observables from the current torus state.
 * @return         SSM parameter set ready for SSMLayer configuration.
 */
[[nodiscard]]
inline SSMParams extract_ssm_params(const PhysicsParams& physics) {
    // --- Constants ---
    // A-parameter range: continuous-time diagonal [-(0.1), -(0.9)]
    constexpr float BASE_A_MAG       = 0.1f;   // minimum |A| (high-intensity)
    constexpr float INTENSITY_RANGE  = 0.8f;    // dynamic range
    // Δ-parameter: base + phase-driven component
    constexpr float BASE_DELTA       = 0.05f;
    constexpr float PHASE_DELTA_SCALE = 0.15f;

    SSMParams params{};

    // --- A diagonal: intensity-driven per-dimension decay ---
    // Find max intensity for normalization (avoid div-by-zero)
    float max_I = 0.f;
    for (float I : physics.intensity)
        max_I = std::max(max_I, I);
    if (max_I < 1e-12f) max_I = 1.f;  // flat field fallback

    for (int i = 0; i < SSM_HIDDEN_DIM; ++i) {
        const int d = i % 9;
        const float norm_I = std::clamp(physics.intensity[d] / max_I, 0.f, 1.f);
        // High intensity → small |A| → slow decay → memory retention
        // Low intensity  → large |A| → fast decay → forget
        params.A_diag[i] = -(BASE_A_MAG + INTENSITY_RANGE * (1.f - norm_I));
    }

    // --- Δ: phase-velocity-driven, spectrally clamped ---
    float mean_abs_phase = 0.f;
    for (float p : physics.phase)
        mean_abs_phase += std::abs(p);
    mean_abs_phase /= 9.f;
    // Normalize phase to [0, 1] range (phase ∈ [-π, π])
    const float norm_phase = std::clamp(mean_abs_phase / 3.14159f, 0.f, 1.f);

    const float delta_requested = BASE_DELTA + PHASE_DELTA_SCALE * norm_phase;
    params.delta = clamp_delta(delta_requested, physics.rho_G, physics.resonance);

    // --- Stability classification ---
    // Use the matrix norm of A diagonal as the gain metric
    float a_norm = 0.f;
    for (int i = 0; i < 9; ++i) {
        // Representative: first hidden unit per dimension
        a_norm = std::max(a_norm, std::abs(params.A_diag[i]));
    }
    params.stability = classify_stability(
        params.delta, physics.rho_G, physics.resonance, a_norm);

    return params;
}

// ============================================================================
// Mamba9D — physics-aware SSM processor
// ============================================================================

/**
 * @brief Mamba-9D: wraps SSMLayer with physics-derived parameter adaptation.
 *
 * Unlike the raw SSMLayer (which uses fixed random weights), Mamba9D
 * dynamically adapts its A diagonal and Δ at every step based on
 * manifold physics.  The B/C/D matrices remain learned (or random-initialized)
 * and are not physics-derived.
 *
 * Usage:
 * @code
 *   Mamba9D mamba;
 *   auto h = mamba.ssm().make_zero_state();
 *   mamba.ssm().randomise(42);
 *   mamba.ssm().randomise_selective(42);
 *
 *   // Per-tick: extract physics, step the SSM
 *   PhysicsParams phys = extract_physics_from_torus(torus, anchor_idx);
 *   std::array<float, 9> input = get_hilbert_input(scanner, idx);
 *   mamba.step(h, input, phys);
 *
 *   // Read output
 *   std::vector<float> logits;
 *   mamba.ssm().compute_output(h, logits);
 * @endcode
 */
class Mamba9D {
public:
    /**
     * @brief Construct Mamba9D with given dimensions.
     *
     * @param hidden_dim  SSM hidden state size (default 256).
     * @param input_dim   Input dimension (default 9).
     * @param output_dim  Output / vocabulary size (default 50000).
     * @param seed        RNG seed for sampler.
     */
    explicit Mamba9D(int      hidden_dim = SSM_HIDDEN_DIM,
                     int      input_dim  = SSM_INPUT_DIM,
                     int      output_dim = SSM_OUTPUT_DIM_DEFAULT,
                     uint32_t seed       = 42u)
        : ssm_(hidden_dim, input_dim, output_dim)
        , sampler_(seed)
    {}

    // ------------------------------------------------------------------ accessors

    SSMLayer&             ssm()      noexcept { return ssm_; }
    const SSMLayer&       ssm()      const noexcept { return ssm_; }

    SequenceManager&       sequence() noexcept { return seq_; }
    const SequenceManager& sequence() const noexcept { return seq_; }

    WavefunctionSampler&       sampler()  noexcept { return sampler_; }
    const WavefunctionSampler& sampler()  const noexcept { return sampler_; }

    /// Last extracted SSM parameters (for diagnostics / testing).
    const SSMParams& last_params() const noexcept { return last_params_; }

    // ------------------------------------------------------------------ operations

    /**
     * @brief Reset the sequence counter and zero a hidden state.
     */
    void reset(SSMLayer::State& state) {
        seq_.reset();
        std::fill(state.begin(), state.end(), 0.f);
    }

    /**
     * @brief Process one input step with physics-derived parameters.
     *
     * 1. Extract SSM params from physics (A diagonal, clamped Δ)
     * 2. Apply A diagonal to the SSMLayer
     * 3. Run selective_step (S6 with ZOH discretization)
     * 4. Advance sequence counter
     *
     * @param h        [in/out] Hidden state, length hidden_dim.
     * @param input    9D input coordinate (from Hilbert scan or torus).
     * @param physics  Physical observables for this time step.
     */
    void step(SSMLayer::State& h,
              const std::array<float, foundation::TORUS_DIMS>& input,
              const PhysicsParams& physics) {
        // 1. Extract physics-derived SSM parameters
        last_params_ = extract_ssm_params(physics);

        // 2. Apply A diagonal to the SSMLayer
        auto& A = ssm_.A();
        const int H = ssm_.hidden_dim();
        for (int i = 0; i < H; ++i)
            A[i] = last_params_.A_diag[i];

        // 3. Selective step (S6 ZOH discretization, input-dependent Δ/B)
        ssm_.selective_step(h, input);

        // 4. Advance sequence manager
        seq_.advance();
    }

    /**
     * @brief Process a sequence of inputs with per-step physics.
     *
     * Processes each (input, physics) pair through step() sequentially.
     * The physics vector must be the same length as inputs.
     *
     * @param h        [in/out] Hidden state.
     * @param inputs   Sequence of 9D inputs (e.g. from Hilbert scanner).
     * @param params   Per-step physics observations.
     */
    void process_sequence(
            SSMLayer::State& h,
            const std::vector<std::array<float, foundation::TORUS_DIMS>>& inputs,
            const std::vector<PhysicsParams>& params) {
        const size_t len = std::min(inputs.size(), params.size());
        for (size_t t = 0; t < len; ++t) {
            step(h, inputs[t], params[t]);
        }
    }

    /**
     * @brief Extract PhysicsParams from a CognitiveTorus at a given anchor node.
     *
     * Reads the intensity and phase from the anchor's 9-dimensional
     * stride neighbors, plus global resonance and curvature estimates.
     *
     * @param intensity_snap  Pre-computed |ψ|² snapshot (from resonance_snapshot()).
     * @param psi_real        Raw psi_real array pointer.
     * @param psi_imag        Raw psi_imag array pointer.
     * @param anchor_idx      Hot-node index to probe neighbors of.
     * @param grid_n          Nodes per dimension.
     * @param num_nodes       Total node count.
     * @param resonance       Current resonance estimate [0,1].
     * @param rho_G           Metric tensor spectral radius.
     * @return                PhysicsParams ready for step() or extract_ssm_params().
     */
    [[nodiscard]]
    static PhysicsParams extract_physics(
            const float* psi_real,
            const float* psi_imag,
            size_t anchor_idx,
            size_t grid_n,
            size_t num_nodes,
            float resonance = 0.5f,
            float rho_G = 1.0f)
    {
        PhysicsParams phys;
        phys.resonance = resonance;
        phys.rho_G = rho_G;

        // Probe the 9 dimension-stride neighbors
        size_t stride = 1;
        for (int d = 0; d < 9; ++d) {
            const size_t neighbor = (anchor_idx + stride) % num_nodes;
            const float pr = psi_real[neighbor];
            const float pi = psi_imag[neighbor];
            phys.intensity[d] = pr * pr + pi * pi;
            phys.phase[d] = std::atan2(pi, pr);
            stride *= grid_n;
        }

        return phys;
    }

private:
    SSMLayer            ssm_;
    SequenceManager     seq_;
    WavefunctionSampler sampler_;
    SSMParams           last_params_{};  ///< Most recent extracted parameters
};

// ============================================================================
// Grid coordinate → float conversion (replaces DecisionLoop::grid_coord_to_float)
// ============================================================================

/**
 * @brief Convert a flat torus node index to a normalized 9D float coordinate
 *        in [−1, +1]⁹.
 *
 * Algorithm: modular decomposition for grid resolution n.
 *   For dim d:  digit_d = (flat_idx / n^d) % n
 *               coord_d = 2 · digit_d / (n − 1) − 1
 *   Special case n ≤ 1: all coordinates = 0.
 *
 * @param flat_idx  Flat SoA index of the node.
 * @param n         Grid resolution per dimension.
 * @return          Normalized 9D coordinate in [−1, +1]⁹.
 */
[[nodiscard]]
inline std::array<float, foundation::TORUS_DIMS>
grid_coord_to_float(size_t flat_idx, int n) noexcept {
    std::array<float, foundation::TORUS_DIMS> coord{};
    if (n <= 1) return coord;

    const float inv = 2.f / static_cast<float>(n - 1);
    size_t remaining = flat_idx;
    for (int d = 0; d < static_cast<int>(foundation::TORUS_DIMS); ++d) {
        const int digit = static_cast<int>(remaining % static_cast<size_t>(n));
        remaining /= static_cast<size_t>(n);
        coord[d] = static_cast<float>(digit) * inv - 1.f;
    }
    return coord;
}

/**
 * @brief Convert a flat torus node index to 9D integer grid coordinates.
 *
 * @param flat_idx  Flat SoA index.
 * @param n         Grid resolution per dimension.
 * @return          9D grid coordinate, each element in [0, n-1].
 */
[[nodiscard]]
inline std::array<uint32_t, 9>
flat_to_grid_coords(size_t flat_idx, int n) noexcept {
    std::array<uint32_t, 9> coord{};
    size_t remaining = flat_idx;
    for (int d = 0; d < 9; ++d) {
        coord[d] = static_cast<uint32_t>(remaining % static_cast<size_t>(n));
        remaining /= static_cast<size_t>(n);
    }
    return coord;
}

// ============================================================================
// HilbertMambaBridge — Hilbert-linearized torus → Mamba-9D pipeline
// ============================================================================

/**
 * @brief Bridges the T⁹ manifold to Mamba-9D via Hilbert-linearized sequences.
 *
 * Architecture (v0.1.6 Phase 2):
 *   1. Pre-compute Hilbert index for every grid node (O(N) at construction)
 *   2. Each tick: receive hot-node indices from CognitiveTorus
 *   3. Sort hot nodes by Hilbert index (locality-preserving order)
 *   4. For each sorted node: extract physics → step Mamba9D
 *   5. Hidden state carries forward across ticks (sliding window)
 *
 * The Hilbert ordering ensures that spatially close nodes on the torus
 * are processed sequentially, giving the SSM spatially coherent inputs.
 * The SSM's exponential decay (Ā < 1) provides the "sliding wave window"
 * with effective horizon ≈ 100 steps.
 *
 * Causal ordering: within each tick, nodes are processed in Hilbert order
 * (spatial locality within a time slice).  Across ticks, time advances
 * monotonically.  This satisfies the causal-foliation requirement.
 */
class HilbertMambaBridge {
public:
    /// Result of one tick's processing.
    struct TickResult {
        int   nodes_processed;     ///< How many hot nodes were fed to the SSM
        float state_norm;          ///< L2 norm of hidden state after processing
        StabilityCondition stability;  ///< Worst stability encountered this tick
    };

    /**
     * @brief Construct the bridge for a grid of n^9 nodes.
     *
     * Pre-computes a Hilbert index for every grid node using a HilbertScanner
     * of minimum sufficient order (smallest order where 2^order ≥ n).
     *
     * @param grid_n      Nodes per dimension (e.g. 3 for 3^9 = 19,683 nodes).
     * @param hidden_dim  SSM hidden dimension (default 256).
     * @param output_dim  SSM output dimension (default 50000).
     * @param seed        RNG seed for Mamba9D sampler.
     */
    explicit HilbertMambaBridge(int      grid_n,
                                int      hidden_dim = SSM_HIDDEN_DIM,
                                int      output_dim = SSM_OUTPUT_DIM_DEFAULT,
                                uint32_t seed       = 42u)
        : grid_n_(grid_n)
        , mamba_(hidden_dim, SSM_INPUT_DIM, output_dim, seed)
        , h_(mamba_.ssm().make_zero_state())
        , scanner_(min_order_for(grid_n))
    {
        // Initialise SSM weights so selective scan is functional
        mamba_.ssm().randomise(seed);
        mamba_.ssm().randomise_selective(seed);

        // Pre-compute Hilbert index for every grid node
        const size_t N = total_nodes();
        node_hilbert_indices_.resize(N);
        for (size_t i = 0; i < N; ++i) {
            auto gc = flat_to_grid_coords(i, grid_n_);
            node_hilbert_indices_[i] = scanner_.coords_to_index(gc);
        }
    }

    // ----------------------------------------------------------------- accessors

    Mamba9D&             mamba()     noexcept { return mamba_; }
    const Mamba9D&       mamba()     const noexcept { return mamba_; }

    SSMLayer::State&       state()     noexcept { return h_; }
    const SSMLayer::State& state()     const noexcept { return h_; }

    int grid_n() const noexcept { return grid_n_; }
    size_t total_nodes() const noexcept {
        size_t n = 1;
        for (int d = 0; d < 9; ++d) n *= static_cast<size_t>(grid_n_);
        return n;
    }

    /// Hilbert index for a grid node (pre-computed).
    uint64_t node_hilbert_index(size_t flat_idx) const noexcept {
        return node_hilbert_indices_[flat_idx];
    }

    /// The Hilbert scanner order used.
    uint32_t hilbert_order() const noexcept { return scanner_.get_order(); }

    // ----------------------------------------------------------------- operations

    /**
     * @brief Process one physics tick: sort hot nodes by Hilbert index,
     *        extract physics, and feed to Mamba9D.
     *
     * @param psi_real          Wavefunction real part (SoA, length num_nodes).
     * @param psi_imag          Wavefunction imag part (SoA, length num_nodes).
     * @param num_nodes         Total node count.
     * @param hot_node_indices  Indices of top-K highest-intensity nodes.
     * @param resonance         Current global resonance [0, 1].
     * @param rho_G             Metric tensor spectral radius.
     * @return                  TickResult with processing summary.
     */
    TickResult tick(const float* psi_real,
                    const float* psi_imag,
                    size_t num_nodes,
                    const std::vector<size_t>& hot_node_indices,
                    float resonance = 0.5f,
                    float rho_G     = 1.0f)
    {
        TickResult result{};
        result.stability = StabilityCondition::STABLE;

        if (hot_node_indices.empty()) {
            result.state_norm = SSMLayer::state_norm(h_);
            return result;
        }

        // 1. Sort hot nodes by pre-computed Hilbert index (locality-preserving)
        std::vector<size_t> sorted_nodes = hot_node_indices;
        std::sort(sorted_nodes.begin(), sorted_nodes.end(),
            [this](size_t a, size_t b) {
                return node_hilbert_indices_[a] < node_hilbert_indices_[b];
            });

        // 2. Process each node in Hilbert order
        for (size_t node_idx : sorted_nodes) {
            // Convert to normalized [-1, +1] input
            auto input = grid_coord_to_float(node_idx, grid_n_);

            // Extract physics from this node
            auto physics = Mamba9D::extract_physics(
                psi_real, psi_imag, node_idx,
                static_cast<size_t>(grid_n_), num_nodes,
                resonance, rho_G);

            // Step the Mamba9D
            mamba_.step(h_, input, physics);

            // Track worst stability
            if (mamba_.last_params().stability != StabilityCondition::STABLE)
                result.stability = mamba_.last_params().stability;

            ++result.nodes_processed;
        }

        result.state_norm = SSMLayer::state_norm(h_);
        return result;
    }

    /**
     * @brief Reset hidden state and sequence counter.
     */
    void reset() {
        mamba_.reset(h_);
    }

private:
    /// Minimum Hilbert order such that 2^order ≥ n.
    static uint32_t min_order_for(int n) noexcept {
        if (n <= 1) return 1;
        uint32_t order = 0;
        int power = 1;
        while (power < n) { power <<= 1; ++order; }
        return order;
    }

    int                    grid_n_;
    Mamba9D                mamba_;
    SSMLayer::State        h_;
    spatial::HilbertScanner scanner_;
    std::vector<uint64_t>  node_hilbert_indices_;
};

} // namespace nikola::cognitive
