/**
 * @file src/inference/nikola_inference.cpp
 * @brief NikolaInference — lightweight inference-only pipeline.
 *
 * Implements the inference path stripped of all autonomy subsystems:
 *   inject → propagate torus → Mamba9D → NPT (optional) → decode → compose
 *
 * v0.2.5 — Phase 1
 */

#include <nikola/inference/nikola_inference.hpp>
#include <nikola/diag/scope_profiler.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <numeric>
#include <random>
#include <unordered_set>

namespace nikola::inference {

// ============================================================================
// Helpers (local)
// ============================================================================

/// Generate a crude-but-unique 9D wave from a token string (no ORT required).
/// Identical to decision_loop.cpp's version — gives each token a distinct
/// waveform for matching even without BERT embeddings.
static std::vector<std::complex<float>>
synthetic_wave9d(const std::string& token)
{
    using C = std::complex<float>;
    std::vector<C> w(9, C{0.f, 0.f});
    for (size_t i = 0; i < token.size(); ++i) {
        const float v = static_cast<float>(static_cast<unsigned char>(token[i])) / 128.f;
        const size_t d = i % 9;
        w[d] = C{w[d].real() + v, 0.f};
    }
    return w;
}

// ============================================================================
// Constructor
// ============================================================================

NikolaInference::NikolaInference(InferenceConfig cfg)
    : cfg_(std::move(cfg))
#ifdef NIKOLA_HAS_ORT
    , torus_(cfg_.grid_n, cfg_.tokenizer_json_path, cfg_.model_path)
#else
    , torus_(cfg_.grid_n)
#endif
    , mamba_bridge_(cfg_.grid_n,
                    cfg_.mamba_hidden_dim,
                    std::max(1, static_cast<int>(cfg_.vocabulary.size())),
                    42u)
    , npt_(cfg_.grid_n, cfg_.npt_temperature, cfg_.npt_curvature_alpha)
{
    torus_.set_gpu(cfg_.enable_gpu);

    // Register vocabulary tokens with the decoder.
#ifdef NIKOLA_HAS_ORT
    if (!cfg_.tokenizer_json_path.empty() && !cfg_.model_path.empty()) {
        try {
            cognitive::NonaryEmbedder embedder(
                cfg_.tokenizer_json_path, cfg_.model_path);
            for (const auto& token : cfg_.vocabulary) {
                try {
                    decoder_.register_from_embedder(embedder, token);
                } catch (...) {}
            }
        } catch (const std::exception& e) {
            std::cerr << "[NikolaInference] ORT vocab failed: " << e.what()
                      << " — using synthetic waves\n";
            for (const auto& token : cfg_.vocabulary) {
                try { decoder_.register_token(token, synthetic_wave9d(token)); } catch (...) {}
            }
        }
    } else {
        for (const auto& token : cfg_.vocabulary) {
            try { decoder_.register_token(token, synthetic_wave9d(token)); } catch (...) {}
        }
    }
#else
    for (const auto& token : cfg_.vocabulary) {
        try { decoder_.register_token(token, synthetic_wave9d(token)); } catch (...) {}
    }
#endif

    // Initialise ThoughtComposer.
#ifdef NIKOLA_HAS_ORT
    if (!cfg_.tokenizer_json_path.empty() && !cfg_.model_path.empty()) {
        composer_ = cognitive::ThoughtComposer(
            cfg_.tokenizer_json_path, cfg_.model_path);
    }
#endif

    // Calibrate vocabulary to torus space if we have tokens.
    // Uses the same inject→decode cycle as DecisionLoop, but simplified:
    // inject each token, capture the torus-space signature for matching.
    if (cfg_.vocabulary.size() > 20) {
        torus_.reset_field();
    }
}

// ============================================================================
// inject
// ============================================================================

void NikolaInference::inject(const std::string& text)
{
    if (text.empty()) return;

    // Clear accumulated tokens for new stimulus.
    accumulated_tokens_.clear();
    accumulated_unique_.clear();

#ifdef NIKOLA_HAS_ORT
    torus_.inject_text(text, static_cast<double>(torus_.time()));
#else
    // Non-ORT fallback: inject a uniform low-amplitude Nit excitation.
    using foundation::Nit;
    std::vector<Nit> pulse(128, Nit{1});
    torus_.inject_raw(pulse, static_cast<double>(torus_.time()));
#endif
}

// ============================================================================
// tick
// ============================================================================

InferenceResult NikolaInference::tick()
{
    NIKOLA_PROFILE("NikolaInference::tick");

    // ── 0. Reseed if field has collapsed ─────────────────────────────────────
    maybe_reseed_field();

    // ── 1. Advance torus physics ─────────────────────────────────────────────
    const float dt = torus_.safe_dt();
    {
        NIKOLA_PROFILE("infer::torus_run");
        torus_.run(cfg_.steps_per_tick, dt);
    }

    // ── 2. Mamba9D tick — feed hot nodes through physics-aware SSM ───────────
    std::string ssm_token;
    if (!cfg_.vocabulary.empty()) {
        NIKOLA_PROFILE("infer::mamba9d_tick");
        const auto& g = torus_.grid();
        const size_t N = g.num_active_nodes();
        const auto hot = torus_.hot_nodes(cfg_.mamba_top_k);
        if (!hot.empty()) {
            mamba_bridge_.tick(
                g.psi_real(), g.psi_imag(), N, hot,
                /* resonance */ 0.5f, /* rho_G */ 1.0f);

            std::vector<float> logits;
            mamba_bridge_.mamba().ssm().compute_output(mamba_bridge_.state(), logits);
            const size_t token_idx = mamba_bridge_.mamba().sampler().sample_from_vector(
                logits, cfg_.ssm_temperature);

            if (token_idx < cfg_.vocabulary.size()) {
                ssm_token = cfg_.vocabulary[token_idx];
            }
        }
    }

    // ── 3. NPT reasoning pass (optional) ────────────────────────────────────
    if (cfg_.enable_npt) {
        execute_reason();
    }

    // ── 4. Decode resonance tokens ──────────────────────────────────────────
    std::vector<std::string> tokens;
    try {
        tokens = decoder_.decode(torus_, cfg_.decode_top_k);
    } catch (...) {}

    // ── 5. Accumulate tokens across ticks ───────────────────────────────────
    {
        const auto max_acc = static_cast<size_t>(cfg_.max_accumulated_tokens);
        auto try_add = [&](const std::string& tok) {
            if (tok.empty() || accumulated_tokens_.size() >= max_acc) return;
            if (accumulated_unique_.insert(tok).second)
                accumulated_tokens_.push_back(tok);
        };
        for (const auto& tok : tokens)   try_add(tok);
        if (!ssm_token.empty())          try_add(ssm_token);
    }

    // ── 6. Compose thought ──────────────────────────────────────────────────
    cognitive::ThoughtContext ctx;
    ctx.tokens   = accumulated_tokens_.empty() ? tokens : accumulated_tokens_;
    ctx.dopamine = cfg_.dopamine;
    ctx.boredom  = 0.3f;    // mild curiosity baseline
    ctx.atp      = 0.8f;    // healthy energy baseline
    ctx.td_error = 0.0f;    // neutral
    ctx.entropy  = compute_entropy();

    std::string thought;
    if (!ctx.tokens.empty()) {
        thought = composer_.compose(ctx);
        // Consume accumulated tokens after composing.
        accumulated_tokens_.clear();
        accumulated_unique_.clear();
    }

    // ── 7. Build result ─────────────────────────────────────────────────────
    InferenceResult result;
    result.tokens    = std::move(tokens);
    result.thought   = std::move(thought);
    result.ssm_token = std::move(ssm_token);
    result.energy    = compute_energy();
    result.entropy   = ctx.entropy;
    result.tick      = tick_count_;

    ++tick_count_;

    if (on_tick) on_tick(result);

    return result;
}

// ============================================================================
// generate
// ============================================================================

std::vector<InferenceResult> NikolaInference::generate(int max_ticks)
{
    std::vector<InferenceResult> results;
    results.reserve(static_cast<size_t>(max_ticks));

    for (int i = 0; i < max_ticks; ++i) {
        auto r = tick();
        if (!r.thought.empty()) {
            results.push_back(std::move(r));
        }
    }

    return results;
}

// ============================================================================
// infer
// ============================================================================

std::string NikolaInference::infer(const std::string& prompt, int max_ticks)
{
    inject(prompt);

    for (int i = 0; i < max_ticks; ++i) {
        auto r = tick();
        if (!r.thought.empty()) {
            return r.thought;
        }
    }
    return {};
}

// ============================================================================
// load_checkpoint
// ============================================================================

bool NikolaInference::load_checkpoint(const std::string& path)
{
    try {
        // Build a CognitiveSnapshot targeting our components.
        persistence::CognitiveSnapshot snap;
        snap.ssm  = &mamba_bridge_.mamba().ssm();
        snap.npt  = &npt_;
        snap.grid = &torus_.wave_function().grid();

        persistence::load_checkpoint(path, snap);

        std::cout << "[NikolaInference] Loaded checkpoint: " << path << "\n";
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[NikolaInference] Checkpoint load failed: " << e.what() << "\n";
        return false;
    }
}

// ============================================================================
// warmup
// ============================================================================

void NikolaInference::warmup()
{
    // Run one tick to populate all caches (Mamba state, decoder, etc.)
    tick();
}

// ============================================================================
// execute_reason — NPT forward, feed back through Mamba → torus
// ============================================================================

void NikolaInference::execute_reason()
{
    NIKOLA_PROFILE("infer::reason");

    // Phase B2: NPT reads Mamba's perception, not raw torus.
    auto mamba_wf = mamba_bridge_.state_to_wave_function();

    // Fixed neuromodulators — no autonomy dynamics.
    auto result = npt_.forward(mamba_wf,
                               cfg_.dopamine,
                               cfg_.serotonin,
                               cfg_.norepinephrine);

    if (result.has_output) {
        // Phase B3: NPT → Mamba → torus (NPT never writes torus directly).
        auto npt_input = cognitive::HilbertMambaBridge::wave_function_to_input(
            result.output);

        cognitive::PhysicsParams feedback_physics;
        feedback_physics.resonance = 0.5f;
        feedback_physics.rho_G     = 1.0f;
        feedback_physics.intensity.fill(0.5f);
        feedback_physics.phase.fill(0.f);

        mamba_bridge_.mamba().step(mamba_bridge_.state(), npt_input, feedback_physics);

        // Mamba writes torus: α = top head score × 0.3
        const float top_score = *std::max_element(
            result.head_scores.begin(), result.head_scores.end());
        auto mamba_output_wf = mamba_bridge_.state_to_wave_function();
        torus_.wave_function().add_scaled(mamba_output_wf, top_score * 0.3f);
    }
}

// ============================================================================
// compute_energy
// ============================================================================

float NikolaInference::compute_energy() const
{
    return static_cast<float>(torus_.total_probability());
}

// ============================================================================
// compute_entropy
// ============================================================================

float NikolaInference::compute_entropy() const
{
    const auto& g = torus_.grid();
    const size_t N = g.num_active_nodes();
    const float* pr = g.psi_real();
    const float* pi = g.psi_imag();

    // Total energy
    float total = 0.f;
    for (size_t i = 0; i < N; ++i) {
        total += pr[i] * pr[i] + pi[i] * pi[i];
    }
    if (total < 1e-10f) return 0.f;

    // Shannon entropy over |ψ|² distribution
    float entropy = 0.f;
    for (size_t i = 0; i < N; ++i) {
        const float p = (pr[i] * pr[i] + pi[i] * pi[i]) / total;
        if (p > 1e-10f) {
            entropy -= p * std::log2(p);
        }
    }
    return entropy;
}

// ============================================================================
// maybe_reseed_field
// ============================================================================

bool NikolaInference::maybe_reseed_field()
{
    if (torus_.total_probability() >= 1e-3) return false;

    // Broad-spectrum curiosity heartbeat (same pattern as DecisionLoop).
    using foundation::Nit;
    static const std::array<int, 4> PATTERN = {2, -1, 1, -2};
    std::vector<Nit> pulse(128);
    for (size_t i = 0; i < 128; ++i) {
        pulse[i] = Nit(PATTERN[i % 4]);
    }

    torus_.inject_raw(pulse, static_cast<double>(torus_.time()));
    return true;
}

}  // namespace nikola::inference
