/**
 * @file include/nikola/inference/nikola_inference.hpp
 * @brief Lightweight inference-only pipeline — no autonomy, no scoring.
 *
 * NikolaInference wraps the minimal CognitiveTorus → HilbertMambaBridge →
 * NeuroplasticTransformer → ResonanceDecoder → ThoughtComposer pipeline
 * needed to produce thoughts from stimuli.  It strips the 11-action
 * scoring, personality drift, preference learning, SIE, goals, and all
 * autonomy subsystems.
 *
 * Neuromodulators are fixed at baseline (0.5) — no dopamine/serotonin/NE
 * dynamics.  Every tick produces an output (no SILENT threshold).
 *
 * v0.2.5 — Phase 1
 */
#pragma once

#include <nikola/cognitive/cognitive_torus.hpp>
#include <nikola/cognitive/mamba9d.hpp>
#include <nikola/cognitive/neuroplastic_transformer.hpp>
#include <nikola/cognitive/resonance_decoder.hpp>
#include <nikola/cognitive/thought_composer.hpp>
#include <nikola/persistence/dmc_checkpoint.hpp>

#include <functional>
#include <string>
#include <vector>

namespace nikola::inference {

// ============================================================================
// Configuration
// ============================================================================

struct InferenceConfig {
    /// Torus grid dimension (n^9 nodes). Default 3 → 19,683 nodes.
    int grid_n = 3;

    /// Torus physics steps per inference tick.
    int steps_per_tick = 50;

    /// Number of hot nodes to decode per tick.
    size_t decode_top_k = 20;

    /// Maximum tokens to accumulate before composing a thought.
    int max_accumulated_tokens = 8;

    /// Vocabulary words for ResonanceDecoder registration.
    std::vector<std::string> vocabulary;

    /// Path to ORT tokenizer.json (optional — enables ORT injection + composition).
    std::string tokenizer_json_path;

    /// Path to ORT model.onnx (optional — enables ORT injection + composition).
    std::string model_path;

    /// Enable NPT reasoning pass each tick (default: true).
    bool enable_npt = true;

    /// Enable GPU propagator if available (default: true).
    bool enable_gpu = true;

    /// Mamba9D hidden dimension.
    int mamba_hidden_dim = 256;

    /// Mamba9D output dimension.
    int mamba_output_dim = 50000;

    /// Number of hot nodes fed to Mamba per tick.
    size_t mamba_top_k = 8;

    /// SSM temperature for token sampling.
    float ssm_temperature = 0.01f;

    /// NPT temperature.
    float npt_temperature = 1.0f;

    /// NPT curvature alpha.
    float npt_curvature_alpha = 0.5f;

    /// Fixed neuromodulator levels (no autonomy dynamics).
    float dopamine       = 0.5f;
    float serotonin      = 0.5f;
    float norepinephrine = 0.5f;
};

// ============================================================================
// InferenceResult — output of one inference tick
// ============================================================================

struct InferenceResult {
    /// Decoded tokens from current torus state.
    std::vector<std::string> tokens;

    /// Composed thought string (from ThoughtComposer).
    std::string thought;

    /// SSM-generated token (from Mamba9D vocabulary sampling).
    std::string ssm_token;

    /// Torus field energy (total |ψ|²).
    float energy = 0.f;

    /// Shannon entropy of torus field.
    float entropy = 0.f;

    /// Tick number that produced this result.
    uint64_t tick = 0;
};

// ============================================================================
// NikolaInference — the lightweight inference engine
// ============================================================================

/**
 * @class NikolaInference
 * @brief Runs the Nikola inference pipeline without autonomy.
 *
 * Usage:
 * @code
 *   InferenceConfig cfg;
 *   cfg.vocabulary = { "hello", "nikola", "curious", ... };
 *   NikolaInference engine(cfg);
 *
 *   engine.inject("What is consciousness?");
 *   auto results = engine.generate(100);  // run 100 ticks
 *   for (const auto& r : results)
 *       std::cout << r.thought << "\n";
 * @endcode
 */
class NikolaInference {
public:
    /// Construct the inference engine.
    explicit NikolaInference(InferenceConfig cfg);

    // Non-copyable, non-movable (ResonanceDecoder has shared_mutex).
    NikolaInference(const NikolaInference&) = delete;
    NikolaInference& operator=(const NikolaInference&) = delete;
    NikolaInference(NikolaInference&&) = delete;
    NikolaInference& operator=(NikolaInference&&) = delete;

    ~NikolaInference() = default;

    // ------------------------------------------------------------------ main API

    /**
     * @brief Inject a text stimulus into the torus field.
     * @param text  Human-readable input text.
     */
    void inject(const std::string& text);

    /**
     * @brief Run one inference tick and return the result.
     *
     * Pipeline: propagate torus → Mamba9D → (optional NPT) → decode → compose.
     */
    InferenceResult tick();

    /**
     * @brief Run multiple ticks and collect results.
     *
     * Returns all ticks that produced non-empty thoughts.
     * If no tick produces a thought within max_ticks, returns all results.
     *
     * @param max_ticks  Maximum number of ticks to run.
     * @return           Vector of results (non-empty thoughts only).
     */
    std::vector<InferenceResult> generate(int max_ticks);

    /**
     * @brief Convenience: inject + generate, return the first thought.
     * @param prompt     Input text.
     * @param max_ticks  Maximum ticks (default 200).
     * @return           First composed thought, or empty string.
     */
    std::string infer(const std::string& prompt, int max_ticks = 200);

    // ------------------------------------------------------------------ checkpoint

    /**
     * @brief Load SSM weights and NPT heads from a .nik checkpoint file.
     * @param path  Path to the .nik checkpoint.
     * @return      true if loaded successfully.
     */
    bool load_checkpoint(const std::string& path);

    // ------------------------------------------------------------------ callbacks

    /// Called after every tick with the result.
    std::function<void(const InferenceResult&)> on_tick;

    // ------------------------------------------------------------------ accessors

    const InferenceConfig& config() const noexcept { return cfg_; }
    uint64_t tick_count() const noexcept { return tick_count_; }

    cognitive::CognitiveTorus& torus() noexcept { return torus_; }
    const cognitive::CognitiveTorus& torus() const noexcept { return torus_; }

    /// Warm up the pipeline: run one tick to populate caches.
    void warmup();

private:
    InferenceConfig cfg_;

    /// Core cognitive components (no autonomy).
    cognitive::CognitiveTorus           torus_;
    cognitive::HilbertMambaBridge       mamba_bridge_;
    cognitive::NeuroplasticTransformer  npt_;
    cognitive::ResonanceDecoder         decoder_;
    cognitive::ThoughtComposer          composer_;

    /// Tick state.
    uint64_t tick_count_ = 0;

    /// Token accumulation across ticks (multi-word thoughts).
    std::vector<std::string>           accumulated_tokens_;
    std::unordered_set<std::string>    accumulated_unique_;

    /// Run NPT reasoning and feed output back through Mamba → torus.
    void execute_reason();

    /// Compute field energy (total |ψ|²).
    float compute_energy() const;

    /// Compute Shannon entropy of torus field.
    float compute_entropy() const;

    /// Reseed the torus field if it has collapsed to near-zero.
    bool maybe_reseed_field();
};

}  // namespace nikola::inference
