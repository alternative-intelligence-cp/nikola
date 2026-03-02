#pragma once
/**
 * @file cognitive/cognitive_torus.hpp
 * @brief CognitiveTorus — bridges text injection with live torus physics.
 *
 * Architecture (Path B):
 *   inject_text(text, t)  → NonaryEmbedder  → HolographicInjector → TorusGrid
 *   step(dt)              → Propagator::step(WaveFunction, dt)
 *   hot_nodes(k)          → top-k |ψ|² indices (for resonance readout)
 *   resonance_snapshot()  → per-node |ψ|² vector (for state export)
 *
 * The WaveFunction owns the TorusGrid internally; HolographicInjector holds a
 * reference to wf_.grid() and modifies it in-place.  Propagator.step() then
 * propagates the modified field forward.  Text injection is therefore live:
 * every injected waveform immediately becomes part of the evolving physics.
 *
 * Thread safety: NOT thread-safe.  External synchronisation required if
 * inject_text() and step() are called concurrently.
 *
 * Path B spec: docs/info/engineering/03_cognitive_systems.txt §3.4
 */

#include <nikola/physics/wave_function.hpp>
#include <nikola/physics/propagator.hpp>
#ifdef NIKOLA_HAS_CUDA_KERNELS
#  include <nikola/physics/cuda_propagator.hpp>
#endif
#include <nikola/cognitive/holographic_injector.hpp>
#include <nikola/cognitive/relevance_gate.hpp>
#include <nikola/foundation/toroidal_grid.hpp>
#include <nikola/foundation/nit.hpp>

#include <algorithm>
#include <complex>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#ifdef NIKOLA_HAS_ORT
#  include <nikola/cognitive/nonary_embedder.hpp>
#endif

namespace nikola::cognitive {

// ============================================================================
// CognitiveTorus
// ============================================================================

/**
 * @brief Wraps WaveFunction + Propagator + HolographicInjector for Path B.
 *
 * Typical usage:
 * @code
 *   CognitiveTorus ct(3);          // 3^9 = 19,683 nodes
 *   ct.inject_text("Hello Nikola"); // embed + inject waveform
 *   ct.run(200, ct.max_dt());       // let it resonate for 200 steps
 *   auto hot = ct.hot_nodes(20);    // find top-20 activated nodes
 * @endcode
 */
class CognitiveTorus {
public:
    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

#ifdef NIKOLA_HAS_ORT
    /**
     * @brief Construct with grid size n and ONNX paths.
     *
     * Creates an n^9 dense toroidal grid (e.g. n=3 → 19,683 nodes) seeded
     * with a pilot wave.  NonaryEmbedder is initialised for text injection.
     *
     * @param n            Nodes per dimension (side length of the 9D cube).
     * @param tok_path     Directory containing HuggingFace tokenizer.json.
     * @param model_path   Path to the exported ONNX model (model.onnx).
     * @param pilot_dim    Dimension carrying the pilot wave (default: 3 = time).
     * @param amplitude    Pilot wave amplitude (default: 1.0).
     */
    explicit CognitiveTorus(int n = 3,
                            const std::string& tok_path   = NIKOLA_ORT_TOKENIZER_PATH,
                            const std::string& model_path = NIKOLA_ORT_MODEL_PATH,
                            int pilot_dim = 3,
                            float amplitude = 1.f)
        : wf_()
        , propagator_()
        , embedder_(tok_path, model_path)
    {
        wf_.seed_manifold(n, pilot_dim, /*k_mode=*/1, amplitude, /*seed=*/42);
        wf_.grid().precompute_adjacency();
        injector_ = std::make_unique<HolographicInjector<foundation::TorusGrid>>(wf_.grid());
    }
#else
    /**
     * @brief Construct without ONNX (manual Nit injection only).
     *
     * @param n            Nodes per dimension.
     * @param pilot_dim    Dimension carrying the pilot wave.
     * @param amplitude    Pilot wave amplitude.
     */
    explicit CognitiveTorus(int n = 3,
                            int pilot_dim  = 3,
                            float amplitude = 1.f)
        : wf_()
        , propagator_()
    {
        wf_.seed_manifold(n, pilot_dim, /*k_mode=*/1, amplitude, /*seed=*/42);
        wf_.grid().precompute_adjacency();
        injector_ = std::make_unique<HolographicInjector<foundation::TorusGrid>>(wf_.grid());
    }
#endif

    // Non-copyable — WaveFunction is non-copyable.
    CognitiveTorus(const CognitiveTorus&)            = delete;
    CognitiveTorus& operator=(const CognitiveTorus&) = delete;
    CognitiveTorus(CognitiveTorus&&)                 = default;
    CognitiveTorus& operator=(CognitiveTorus&&)      = default;

    // ------------------------------------------------------------------
    // Injection API
    // ------------------------------------------------------------------

#ifdef NIKOLA_HAS_ORT
    /**
     * @brief Embed text via BERT-Tiny and inject the resulting waveform.
     *
     * Pipeline: text → BPETokenizer → TinyTransformer → tanh→Nit quantise →
     *           HolographicInjector → perturb TorusGrid wavefunction.
     *
     * @param text  Input string (any length; truncated to MAX_SEQ_LEN tokens).
     * @param time  Current physics time (sets emitter phase, default 0.0).
     */
    void inject_text(const std::string& text, double time = 0.0) {
        injector_->inject_text(embedder_, text, time);
    }

    /**
     * @brief Gated text injection — uses a RelevanceGate to filter/attenuate.
     *
     * Pipeline:
     *   1. Embed text → 128 floats (cheap, always done for gate evaluation)
     *   2. gate_embedding() → SalienceResult with weight in [0, 1]
     *   3a. weight == 0: drop; nothing enters the torus (logs drop event)
     *   3b. weight in (0, 1): attenuated injection via inject_scaled()
     *   3c. weight == 1: full injection (equivalent to inject_text())
     *
     * The gate evaluates salience at embedding level (fast, no WF construction).
     * After a successful injection the caller should periodically call
     * update_gate_reference(gate) to keep the gate's notion of "familiar"
     * current.
     *
     * @param text           Input string.
     * @param gate           RelevanceGate to query.  Not modified.
     * @param norepinephrine Alertness level in [0, 1] (modulates threshold).
     *                       0.5f is a calm-but-awake baseline.
     * @param time           Physics time for phase calculation.
     * @return               SalienceResult so caller can observe the gate decision.
     */
    SalienceResult inject_text_gated(const std::string& text,
                                     RelevanceGate& gate,
                                     float norepinephrine = 0.5f,
                                     double time = 0.0) {
        // Step 1: embed to float vector (used for both gate eval and injection)
        auto float_vec = embedder_.embed_float(text);

        // Step 2: evaluate salience at embedding level (cheap)
        auto result = gate.gate_embedding(float_vec, norepinephrine);

        // Step 3: act on the result
        if (!result.passes) {
            std::cout << "[RelevanceGate] DROPPED \"" << text.substr(0, 30)
                      << "\" salience=" << result.salience
                      << " (ne=" << norepinephrine << ")\n";
            return result;
        }

        // Re-embed as Nit vector for injection (reuses float work)
        auto nit_vec = embedder_.embed(text);

        if (result.weight >= 1.f) {
            injector_->inject(nit_vec, time);
            std::cout << "[RelevanceGate] PASS \"" << text.substr(0, 30)
                      << "\" salience=" << result.salience << " weight=1.0\n";
        } else {
            injector_->inject_scaled(nit_vec, result.weight, time);
            std::cout << "[RelevanceGate] ATTENUATED \"" << text.substr(0, 30)
                      << "\" salience=" << result.salience
                      << " weight=" << result.weight << "\n";
        }

        return result;
    }

    /**
     * @brief Snapshot the current torus WaveFunction into the gate's references.
     *
     * Updates both paths:
     *   - WF reference (for gate() calls against raw wave data)
     *   - Embedding reference (not updated here — caller sets it separately
     *     via gate.set_reference_embedding() after each processed injection)
     *
     * Call this whenever the torus has absorbed significant new information —
     * typically every N ticks rather than every tick.
     *
     * @param gate  Gate to update.
     */
    /**
     * @brief Embed text via BERT-Tiny and return the raw Nit vector.
     *
     * Returns the 128-Nit balanced-nonary quantisation of the BERT-Tiny
     * embedding without touching the grid.  Useful for computing analytic
     * injection signatures or building custom pulses.
     *
     * @param text  Input string.
     * @return      Nit vector, length == NonaryEmbedder::EMBED_DIM (128).
     */
    std::vector<foundation::Nit> embed_nits(const std::string& text) const {
        return embedder_.embed(text);
    }
#endif

    /**
     * @brief Inject a pre-computed Nit vector directly (no ONNX required).
     *
     * Useful for deterministic tests or when the caller has already quantised
     * an embedding externally.
     *
     * @param nits  128-element balanced-nonary vector (range –4 … +4 per Nit).
     * @param time  Physics time for phase calculation.
     */
    void inject_raw(const std::vector<foundation::Nit>& nits, double time = 0.0) {
        injector_->inject(nits, time);
    }

    /**
     * @brief Zero out the entire wavefunction (Ψ → 0 at all nodes).
     *
     * Used after vocabulary calibration to start inference from a clean field.
     * The calibration records Nit pulses and lexicon entries in data structures
     * outside the torus; resetting removes the accumulated calibration energy
     * so the live tick loop starts from a neutral baseline.
     */
    void reset_field() noexcept {
        const std::size_t N = wf_.num_nodes();
        float* pr = wf_.grid().psi_real();
        float* pi = wf_.grid().psi_imag();
        std::fill(pr, pr + N, 0.0f);
        std::fill(pi, pi + N, 0.0f);
    }

    /**
     * @brief Inject a pre-computed Nit vector with salience attenuation.
     *
     * Equivalent to inject_raw() but each Nit amplitude is first multiplied
     * by @p weight (0 = no-op, 1 = full injection).  Used by the gated path
     * and available for non-ORT callers that supply their own embeddings.
     *
     * @param nits   128-element balanced-nonary vector.
     * @param weight Salience weight in [0, 1].
     * @param time   Physics time for phase.
     */
    void inject_raw_scaled(const std::vector<foundation::Nit>& nits,
                           float weight,
                           double time = 0.0) {
        injector_->inject_scaled(nits, weight, time);
    }

    /**
     * @brief Snapshot the current torus WaveFunction into the gate's WF reference.
     *
     * Call this whenever the torus has absorbed significant new information —
     * typically every N ticks rather than every tick.  Keeps the gate's notion
     * of "familiar" in sync with the live torus state.
     *
     * @param gate  Gate to update (non-const — modifies gate's stored reference).
     */
    void update_gate_reference(RelevanceGate& gate) const noexcept {
        gate.set_reference(wf_);
    }

    // ------------------------------------------------------------------
    // Physics step API
    // ------------------------------------------------------------------

    /**
     * @brief Advance the wavefunction by one Strang-split step.
     *
     * @param dt  Timestep.  Should satisfy CFL: dt ≤ max_dt().
     */
    void step(float dt) {
#ifdef NIKOLA_HAS_CUDA_KERNELS
        gpu_prop_.step_synced(wf_, dt);
        wf_.advance_time(dt);            // CUDA path doesn't call advance_time internally
#else
        propagator_.step(wf_, dt);
#endif
    }

    /**
     * @brief Run N consecutive physics steps.
     *
     * Uses the GPU propagator when compiled with CUDA (NIKOLA_HAS_CUDA_KERNELS).
     * The host WaveFunction is uploaded once, all N steps run on-device, then
     * the result is downloaded back — minimising PCIe traffic (~15 µs total
     * for 19,683 nodes regardless of step count).
     *
     * @param steps  Number of steps.
     * @param dt     Timestep per step.
     */
    void run(int steps, float dt) {
#ifdef NIKOLA_HAS_CUDA_KERNELS
        gpu_prop_.upload(wf_);                  // H→D  (~0.47 MB, ~15 µs)
        gpu_prop_.run(steps, dt);               // pure-GPU Strang-Verlet
        gpu_prop_.download(wf_);                // D→H  (~15 µs)
        wf_.advance_time(static_cast<float>(steps) * dt);  // CUDA path doesn't call advance_time internally
#else
        for (int i = 0; i < steps; ++i) step(dt);
#endif
    }

    /**
     * @brief CFL-safe maximum timestep for the current grid.
     *
     * Uses the Courant condition for a 9D wave equation:
     *   dt_max = 0.5 · min_h / (c₀ · √9)
     *
     * @note The linear CFL limit is ~0.17 for h=1, c₀=1.  The nonlinear
     *       Gross-Pitaevskii terms (β|ψ|²) require a much more conservative dt
     *       in practice.  Use safe_dt() for reliable propagation.
     */
    float max_dt() const {
        return propagator_.max_stable_dt(wf_.grid());
    }

    /**
     * @brief Nonlinearity-safe timestep — use this for actual time-stepping.
     *
     * The β·|ψ|² nonlinear self-interaction term demands dt ≪ 1/(β·|ψ|_max²).
     * With β=1.0 and pilot amplitude 1.0, empirical stability is dt ≤ 0.01.
     * This matches the hardcoded DT used in main.cpp and nikola_state_dump.cpp.
     *
     * Returns:  min(max_dt() * 0.06, 0.01f)
     */
    float safe_dt() const {
        return std::min(max_dt() * 0.06f, 0.01f);
    }

    // ------------------------------------------------------------------
    // Observables
    // ------------------------------------------------------------------

    /// Current simulation time.
    float time() const noexcept { return wf_.time(); }

    /// Number of active nodes in the grid.
    size_t num_nodes() const noexcept { return wf_.num_nodes(); }

    /// Total probability P = Σᵢ |Ψᵢ|².
    double total_probability() const { return wf_.total_probability(); }

    /**
     * @brief Complex wavefunction amplitude at node idx.
     * @param idx  Node index in the flat SoA array.
     */
    std::complex<float> psi(size_t idx) const noexcept {
        return {wf_.grid().psi_real()[idx], wf_.grid().psi_imag()[idx]};
    }

    /**
     * @brief Wavefunction intensity |ψ|² at node idx.
     */
    float intensity(size_t idx) const noexcept {
        const float r = wf_.grid().psi_real()[idx];
        const float i = wf_.grid().psi_imag()[idx];
        return r * r + i * i;
    }

    /**
     * @brief Return indices of the top-k highest-intensity nodes (|ψ|²).
     *
     * Uses partial_sort → O(N log k).  Results are in descending intensity
     * order.
     *
     * @param k  Maximum number of hot nodes to return.
     */
    [[nodiscard]]
    std::vector<size_t> hot_nodes(size_t k) const {
        const size_t N = wf_.num_nodes();
        std::vector<size_t> idx(N);
        std::iota(idx.begin(), idx.end(), size_t{0});

        k = std::min(k, N);
        if (k == 0) return {};

        std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(k), idx.end(),
            [this](size_t a, size_t b) {
                return intensity(a) > intensity(b);  // descending
            });
        idx.resize(k);
        return idx;
    }

    /**
     * @brief Extract 9-element complex waveform from a node and its
     *        dimension-stride neighbours.
     *
     * Produces a 9-element wave vector suitable for HolographicLexicon::decode().
     * Element d = psi(hot_idx + stride_d) where stride_d is the stride along
     * torus dimension d.
     *
     * Note: stride computation assumes a uniform n-per-dimension cube.
     *
     * @param hot_idx  Anchor node index (typically from hot_nodes()).
     */
    [[nodiscard]]
    std::vector<std::complex<float>> node_wave9d(size_t hot_idx) const {
        const size_t N   = wf_.num_nodes();
        const size_t gn  = wf_.grid().grid_n();  // nodes per dimension

        std::vector<std::complex<float>> wave(9);
        size_t stride = 1;
        for (int d = 0; d < 9; ++d) {
            size_t neighbor = (hot_idx + stride) % N;
            wave[static_cast<size_t>(d)] = psi(neighbor);
            stride *= gn;
        }
        return wave;
    }

    /**
     * @brief Per-node |ψ|² intensity vector (for JSON state export / Path A/B
     *        hybrid).
     *
     * Length == num_nodes().  Suitable for feeding into nikola_state_dump or
     * Aria specialist conditioning.
     */
    [[nodiscard]]
    std::vector<float> resonance_snapshot() const {
        const size_t N   = wf_.num_nodes();
        const float* pr  = wf_.grid().psi_real();
        const float* pi  = wf_.grid().psi_imag();
        std::vector<float> snap(N);
        for (size_t i = 0; i < N; ++i)
            snap[i] = pr[i] * pr[i] + pi[i] * pi[i];
        return snap;
    }

    /// Read-only access to the underlying wave function.
    const physics::WaveFunction& wave_function() const noexcept { return wf_; }

    /// Mutable access to the underlying wave function (for memory superposition).
    physics::WaveFunction& wave_function() noexcept { return wf_; }

    /// Read-only access to the underlying grid.
    const foundation::TorusGrid& grid() const noexcept { return wf_.grid(); }

private:
    physics::WaveFunction wf_;
    physics::Propagator   propagator_;   ///< CPU fallback (always constructed)
    std::unique_ptr<HolographicInjector<foundation::TorusGrid>> injector_;

#ifdef NIKOLA_HAS_CUDA_KERNELS
    physics::CudaPropagator gpu_prop_;   ///< GPU propagator (lazy-inited on first upload)
#endif

#ifdef NIKOLA_HAS_ORT
    NonaryEmbedder embedder_;
#endif
};

} // namespace nikola::cognitive
