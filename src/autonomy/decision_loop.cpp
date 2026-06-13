/**
 * @file src/autonomy/decision_loop.cpp
 * @brief DecisionLoop implementation — action scoring + tick cycle.
 *
 * Scoring model (all scores in [0, ∞), SILENT baseline = 0.3):
 *
 *   NAP            (1 - ATP)²  × 2.5    peaks at 2.5 when fully exhausted
 *   REFUSE         max(0, -TD × 3.0)    proportional to punishment signal
 *   EXPLORE        boredom × ATP × 0.9  needs energy AND curiosity
 *   EMIT_THOUGHT   dopa × boredom × ATP × 1.5  needs all three
 *   STORE_MEMORY   0.8 flat when dopamine spiking AND cooldown elapsed
 *   REQUEST_LOOKUP 0.5 when entropy < 3 (field is too ordered → needs input)
 *   SILENT         0.3 baseline — wins when nothing else is compelling
 *
 * The winning action must beat SILENT by at least cfg.action_threshold to
 * avoid action on marginal differences.
 */

#include <nikola/autonomy/decision_loop.hpp>
#include <nikola/cognitive/lmdb_memory_store.hpp>    // Phase 136 — LMDB persistence
#include <nikola/persistence/lmdb_state_store.hpp>   // Phase 137 — full state persistence
#include <nikola/interior/autobiography.hpp>         // Phase 137 — autobiography member
#include <nikola/interior/preference_engine.hpp>     // v0.2.3 — PreferenceEngine
#include <nikola/interior/personality_drift.hpp>     // v0.2.3 — PersonalityDrift
#include <nikola/interior/narrative_growth.hpp>      // v0.2.3 — NarrativeGrowth
#include <nikola/interior/affective_state.hpp>       // v0.2.3 — AffectiveState

#include <algorithm>
#include <cctype>
#include <complex>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <utility>

#include <nikola/foundation/nit.hpp>
#include <nikola/diag/scope_profiler.hpp>

namespace nikola::autonomy {

// ============================================================================
// Helpers
// ============================================================================

/// Generate a crude-but-unique 9D wave from a token string (no ORT required).
/// Used to register vocabulary tokens in non-ORT builds, giving each token a
/// distinct — if semantically shallow — waveform for matching.
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

// ── Phase 16.1: Grid-to-SSM bridge ──────────────────────────────────────────
//
// Converts a flat torus node index into a normalised 9D float coordinate
// in [−1, +1] for feeding into SSMLayer::selective_step().
//
// Algorithm: modular decomposition for grid resolution n.
//   For each of the 9 dimensions, extract digit d_i = (idx / n^i) % n,
//   then normalise: coord_i = 2 * d_i / (n − 1) − 1   (maps [0, n-1] → [−1, +1]).
//   Special case n=1: all coordinates = 0.

std::array<float, 9>
DecisionLoop::grid_coord_to_float(size_t flat_idx, int n)
{
    std::array<float, 9> coord{};
    if (n <= 1) return coord;  // all zeros for trivial grid

    const float inv = 2.f / static_cast<float>(n - 1);
    size_t remaining = flat_idx;
    for (int d = 0; d < 9; ++d) {
        const int digit = static_cast<int>(remaining % static_cast<size_t>(n));
        remaining /= static_cast<size_t>(n);
        coord[static_cast<size_t>(d)] = static_cast<float>(digit) * inv - 1.f;
    }
    return coord;
}


// ============================================================================
// Constructor
// ============================================================================

DecisionLoop::DecisionLoop(nikola::cognitive::CognitiveTorus& torus,
                           AutonomyEngine&                    engine,
                           DecisionLoopConfig                 cfg)
    : torus_(torus)
    , engine_(engine)
    , cfg_(std::move(cfg))
    , npt_(static_cast<int>(torus.grid().grid_n()), 0.5f, 0.3f)  // v0.0.9: τ=0.5 (sharper), α=0.3
    , npt_last_result_(static_cast<int>(torus.grid().grid_n()))
    , mamba_bridge_(static_cast<int>(torus.grid().grid_n()),
                    nikola::cognitive::SSM_HIDDEN_DIM,
                    std::max(1, static_cast<int>(cfg_.vocabulary.size())),
                    42u)
    , autobiography_(std::make_unique<interior::AutobiographicalMemory>())
{
    const auto now = std::chrono::steady_clock::now();
    start_time_     = now;
    last_emit_time_   = now - std::chrono::seconds(60);  // allow immediate first emit
    last_store_time_  = now - std::chrono::seconds(60);
    last_reason_time_ = now - std::chrono::seconds(60);
    last_generate_time_ = now - std::chrono::seconds(60);  // v0.0.19: allow immediate

    // v0.0.19 — check if Nitpick specialist integration is configured
    aria_specialist_enabled_ = !cfg_.specialist_server_path.empty()
                             || !cfg_.ariac_path.empty();

    // Phase B1 — HilbertMambaBridge's Mamba9D was initialised in member init list.
    // SSM weights are already randomised by the bridge constructor; no extra init needed.

    // Register decoder vocabulary.
    // ORT mode: use NonaryEmbedder for semantically accurate 9D waves.
    // No-ORT mode: fall back to synthetic hash-based waves.
#ifdef NIKOLA_HAS_ORT
    if (!cfg_.tokenizer_json_path.empty() && !cfg_.transformer_model_path.empty()) {
        try {
            nikola::cognitive::NonaryEmbedder embedder(
                cfg_.tokenizer_json_path, cfg_.transformer_model_path);
            for (const auto& token : cfg_.vocabulary) {
                try {
                    decoder_.register_from_embedder(embedder, token);
                } catch (...) {}
            }
            std::cout << "[DecisionLoop] Vocabulary registered via ORT embeddings ("
                      << cfg_.vocabulary.size() << " tokens)\n";
        } catch (const std::exception& e) {
            std::cerr << "[DecisionLoop] ORT vocab registration failed: " << e.what()
                      << " — falling back to synthetic waves\n";
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
        try {
            decoder_.register_token(token, synthetic_wave9d(token));
        } catch (...) { /* vocabulary registration is best-effort */ }
    }
#endif

    // Initialise ThoughtComposer — use ORT paths from config if provided,
    // otherwise the default constructor gives the heuristic (no-ORT) mode.
#ifdef NIKOLA_HAS_ORT
    if (!cfg_.tokenizer_json_path.empty() && !cfg_.transformer_model_path.empty()) {
        thought_composer_ = nikola::cognitive::ThoughtComposer(
            cfg_.tokenizer_json_path, cfg_.transformer_model_path);
    }
#endif

    // Phase 27: Cross-injection calibration.
    // Before calibration: save original waveforms so execute_explore() can
    // still build semantically meaningful pulses from them.  The calibration
    // overwrites the lexicon entries with torus-space delta waveforms.
    if (!cfg_.vocabulary.empty()) {
        for (const auto& token : cfg_.vocabulary) {
            auto maybe = decoder_.lexicon().embed(token);
            if (maybe.has_value()) {
                original_vocab_waves_[token] = *maybe;
            }
        }
        calibrate_vocabulary_to_torus_space();
        // Reset the torus field only for large vocabularies (> 20 tokens).
        // With many tokens, the successive calibration injections accumulate
        // massive field energy that crashes the dopamine habituation system.
        // Small vocabularies (≤ 20 tokens, typical in tests) retain the
        // calibration energy which their explore/emit dynamics rely on.
        // Production nikola-run typically has 200+ words: reset is required.
        if (cfg_.vocabulary.size() > 20) {
            torus_.reset_field();
        }
    }

    // Phase 33 / 136 — load persisted SemanticMemory.
    // LMDB (Phase 136) takes precedence over the legacy binary file.
    if (!cfg_.lmdb_memory_path.empty()) {
        try {
            const size_t loaded = nikola::cognitive::load_lmdb(memory_, cfg_.lmdb_memory_path);
            if (loaded > 0)
                std::cout << "[DecisionLoop] Loaded " << loaded
                          << " memory records from LMDB " << cfg_.lmdb_memory_path << "\n";
        } catch (const std::exception& e) {
            std::cerr << "[DecisionLoop] LMDB load (first run?): " << e.what() << "\n";
        }
    } else if (!cfg_.memory_path.empty()) {
        const size_t loaded = memory_.load(cfg_.memory_path);
        if (loaded > 0)
            std::cout << "[DecisionLoop] Loaded " << loaded
                      << " memory records from " << cfg_.memory_path << "\n";
    }

    // Phase 137 — open state store and load prior session state.
    if (!cfg_.state_db_path.empty()) {
        try {
            state_store_ = std::make_unique<nikola::persistence::LmdbStateStore>(
                cfg_.state_db_path);

            // Restore NikolaState
            {
                NikolaState restored_state;
                uint64_t restored_tick = 0;
                if (state_store_->load_latest_state(restored_state, restored_tick)) {
                    last_state_ = restored_state;
                    tick_count_ = restored_tick;
                    std::cout << "[DecisionLoop] Restored NikolaState from "
                              << cfg_.state_db_path << " (tick "
                              << restored_tick << ")\n";
                }
            }

            // Restore Ψ checkpoint
            {
                physics::WaveFunction wf;
                persistence::detail::CheckpointHeader chdr{};
                if (state_store_->load_latest_checkpoint(wf, chdr)) {
                    auto& grid = torus_.wave_function().grid();
                    const size_t N = grid.num_active_nodes();
                    if (wf.num_nodes() == N) {
                        std::memcpy(grid.psi_real(), wf.grid().psi_real(), N * sizeof(float));
                        std::memcpy(grid.psi_imag(), wf.grid().psi_imag(), N * sizeof(float));
                        std::memcpy(grid.vel_real(), wf.grid().vel_real(), N * sizeof(float));
                        std::memcpy(grid.vel_imag(), wf.grid().vel_imag(), N * sizeof(float));
                        std::cout << "[DecisionLoop] Restored Ψ checkpoint ("
                                  << N << " nodes)\n";
                    }
                }
            }

            // Restore autobiography
            (void)state_store_->load_autobiography(*autobiography_);
            if (!autobiography_->events().empty()) {
                std::cout << "[DecisionLoop] Restored "
                          << autobiography_->events().size()
                          << " autobiographical events\n";
            }
        } catch (const std::exception& e) {
            std::cerr << "[DecisionLoop] State store open (first run?): "
                      << e.what() << "\n";
        }
    }

    // v0.2.3 — Initialise interior personality/preference modules.
    preferences_      = std::make_unique<interior::PreferenceEngine>();
    personality_      = std::make_unique<interior::PersonalityDrift>();
    narrative_growth_ = std::make_unique<interior::NarrativeGrowth>();
    affective_state_  = std::make_unique<interior::AffectiveState>();
}

// ============================================================================
// Destructor — flush final state on shutdown
// ============================================================================

DecisionLoop::~DecisionLoop()
{
    if (state_store_) {
        try {
            state_store_->save_state(last_state_, tick_count_);
            state_store_->save_checkpoint(torus_.wave_function(), tick_count_);
            state_store_->save_autobiography(*autobiography_);
        } catch (const std::exception& e) {
            std::cerr << "[DecisionLoop] ~DecisionLoop state flush failed: "
                      << e.what() << "\n";
        }
    }
}

// ============================================================================
// Accessor — autobiography
// ============================================================================

const interior::AutobiographicalMemory& DecisionLoop::autobiography() const noexcept
{
    return *autobiography_;
}

interior::AutobiographicalMemory& DecisionLoop::autobiography() noexcept
{
    return *autobiography_;
}

const interior::PreferenceEngine& DecisionLoop::preferences() const noexcept
{
    return *preferences_;
}

const interior::PersonalityDrift& DecisionLoop::personality() const noexcept
{
    return *personality_;
}

const interior::AffectiveState& DecisionLoop::affect() const noexcept
{
    return *affective_state_;
}

// ============================================================================
// inject_stimulus
// ============================================================================

void DecisionLoop::inject_stimulus(const std::string& text)
{
    if (text.empty()) return;
    last_stimulus_ = text;   // Track for ESCALATE evidence payload

#ifdef NIKOLA_HAS_ORT
    torus_.inject_text(text, static_cast<double>(torus_.time()));

    // ── Stimulus analytic decode ──────────────────────────────────────────
    // Embed the stimulus text to Nit vectors and find the closest vocabulary
    // token using the same analytic-signature approach as execute_explore().
    // Stored as last_stimulus_seed_ so EXPLORE can bias its first few
    // iterations toward the semantic neighbourhood of the user's prompt.
    if (!original_vocab_nits_.empty()) {
        using Injector = nikola::cognitive::HolographicInjector<nikola::foundation::TorusGrid>;
        const double t = static_cast<double>(torus_.time());
        const auto stim_nits = torus_.embed_nits(text);
        if (!stim_nits.empty()) {
            const auto actual_sig = Injector::analytic_signature(stim_nits, t);
            constexpr float kMinCos = 0.03f;
            float best_cos = kMinCos;
            std::string best_tok;
            for (const auto& [tok, tok_nits] : original_vocab_nits_) {
                if (tok_nits.empty()) continue;
                const auto expected_sig = Injector::analytic_signature(tok_nits, t);
                const float cos = Injector::signature_cosine(actual_sig, expected_sig);
                if (cos > best_cos) { best_cos = cos; best_tok = tok; }
            }
            last_stimulus_seed_ = best_tok;
        }
    }

    // ── v0.0.9: Extract vocabulary words directly from prompt text ────────
    // BERT-Tiny's cosine similarity can be imprecise for philosophical/abstract
    // prompts.  Supplement the embedding seed with literal vocabulary matches
    // in the prompt text — these are guaranteed to be prompt-relevant.
    stimulus_seeds_.clear();
    {
        // Build lowercase copy of prompt for case-insensitive matching
        std::string prompt_lower = text;
        for (char& c : prompt_lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        for (const auto& word : cfg_.vocabulary) {
            std::string word_lower = word;
            for (char& c : word_lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            if (prompt_lower.find(word_lower) != std::string::npos) {
                stimulus_seeds_.push_back(word);
            }
        }
        // Add BERT nearest-word as fallback if not already present
        if (!last_stimulus_seed_.empty()) {
            bool found = false;
            for (const auto& s : stimulus_seeds_)
                if (s == last_stimulus_seed_) { found = true; break; }
            if (!found)
                stimulus_seeds_.push_back(last_stimulus_seed_);
        }
    }

    stimulus_explore_count_ = 0;   // reset so the new seed gets first use
    // v0.0.9: clear accumulated tokens so the new prompt starts fresh
    accumulated_tokens_.clear();
    accumulated_unique_.clear();
#else
    // Without ORT we cannot embed text — the stimulus still perturbs the
    // torus by injecting a uniform low-amplitude Nit excitation.
    // This lets the architecture work in no-ORT test/daemon builds while
    // signal quality is limited (Phase 23 scope).
    using nikola::foundation::Nit;
    std::vector<Nit> pulse(128, Nit{1});   // weak uniform excitation
    torus_.inject_raw(pulse, static_cast<double>(torus_.time()));
#endif
}

void DecisionLoop::inject_stimulus(const std::string& text, float credibility)
{
    if (text.empty()) return;
    last_stimulus_ = text;   // Track for ESCALATE evidence payload

    const float cred = std::clamp(credibility, 0.f, 1.f);
    if (cred < 1e-3f) return;   // Completely distrusted — skip injection

#ifdef NIKOLA_HAS_ORT
    // ORT path: inject normally (amplitude modulation via NonaryEmbedder
    // is deferred to Phase 32 when the embedder API is extended).
    torus_.inject_text(text, static_cast<double>(torus_.time()));
    stimulus_explore_count_ = 0;  // reset so new stimulus seed gets first use
#else
    // Scale Nit amplitude by credibility:
    //   cred 0.25 → Nit{1}  (minimum non-zero)
    //   cred 0.50 → Nit{2}
    //   cred 0.75 → Nit{3}
    //   cred 1.00 → Nit{4}  (full strength)
    using nikola::foundation::Nit;
    const int8_t nit_val = static_cast<int8_t>(
        std::max(1, static_cast<int>(std::round(cred * 4.f)))
    );
    std::vector<Nit> pulse(128, Nit{nit_val});
    torus_.inject_raw(pulse, static_cast<double>(torus_.time()));
#endif
}

// ============================================================================
// read_state
// ============================================================================

NikolaState DecisionLoop::read_state() const
{
    NikolaState s;
    s.time          = torus_.time();
    s.torus_energy  = static_cast<float>(torus_.total_probability());
    s.dopamine      = engine_.dopamine();
    s.td_error      = engine_.dopamine_system().last_td_error();
    s.atp           = engine_.atp();
    s.boredom       = engine_.boredom();
    s.entropy       = engine_.entropy();
    s.last_action   = last_state_.last_action;

    // Decode current resonance tokens (best-effort — empty vocab → empty vec)
    try {
        s.tokens = decoder_.decode(torus_, cfg_.decode_top_k);
    } catch (...) {}

    return s;
}

// ============================================================================
// Scoring functions
// ============================================================================

float DecisionLoop::score_nap(const NikolaState& s) const noexcept
{
    const float deficit = 1.f - s.atp;
    return deficit * deficit * 2.5f;
}

float DecisionLoop::score_refuse(const NikolaState& s) const noexcept
{
    // Punishment signal: td_error is negative when outcome was worse than
    // expected.  The deeper the dip, the stronger the refusal impulse.
    //
    // Phase 27 — alive prior: "being alive is slightly positive."
    // Shift td_error upward by cfg_.alive_prior before scoring, so that
    // neutral stimuli (td_error ≈ 0) produce ~0 REFUSE score.  Only stimuli
    // that push td_error below −alive_prior trigger actual REFUSE.
    // Default alive_prior = 0.1 means a td_error of −0.375 → adjusted −0.275,
    // REFUSE score 0.825 instead of 1.125 — still fires, but decays faster.
    const float adjusted = s.td_error + cfg_.alive_prior;
    return std::max(0.f, -adjusted * 3.0f);
}

float DecisionLoop::score_escalate(const NikolaState& s) const noexcept
{
    // ESCALATE activates in the severe harm tier: the adjusted td_error must
    // exceed REFUSE territory by an additional 0.30 before escalation fires.
    //
    //   adjusted = td_error + alive_prior
    //   REFUSE fires when adjusted < 0                  (threshold 0.0)
    //   ESCALATE fires when adjusted < -0.30            (threshold -0.30)
    //
    // Scoring rises steeply past the escalation threshold (×5.0 multiplier)
    // so that at deep harm depth ESCALATE cleanly outscores REFUSE:
    //
    //   td_error = -0.40 (alive_prior=0.1) → adjusted=-0.30 → escalate=0.0
    //   td_error = -0.50 → adjusted=-0.40 → escalate=0.50
    //   td_error = -0.80 → adjusted=-0.70 → escalate=2.0
    const float adjusted = s.td_error + cfg_.alive_prior;
    const float over     = adjusted + 0.30f;  // Positive when below threshold
    return std::max(0.f, -over * 5.0f);
}

float DecisionLoop::score_explore(const NikolaState& s) const noexcept
{
    // BoredomRegulator fires should_explore() above 0.7.  We score it
    // continuously so the transition is smooth.  ATP gates it — exploring
    // while exhausted is wasteful.
    return s.boredom * s.atp * 0.9f;
}

float DecisionLoop::score_emit_thought(const NikolaState& s) const noexcept
{
    // v0.0.9: Speaking requires dopamine (something happened), boredom (seeking
    // expression), and energy (costs ATP).  Cooldown enforces a minimum gap.
    // Multiplier raised from 1.5 → 3.0 to beat RECALL/SILENT more reliably.
    // Token-count bonus rewards accumulation: more context → richer thought.
    if (seconds_since(last_emit_time_) < cfg_.min_emit_interval_s) return 0.f;
    // Allow EMIT_THOUGHT when *any* token source is available:
    //   1. Accumulated tokens from prior ticks (v0.0.9 multi-word path)
    //   2. Cold decoded tokens from the evolved field
    //   3. Warm decoded tokens from the last EXPLORE inject
    //   4. Most-recent seed token (belt-and-suspenders)
    if (accumulated_tokens_.empty() && s.tokens.empty()
        && last_ex_tokens_.empty() && last_seed_token_.empty()) return 0.f;
    const float base = s.dopamine * s.boredom * s.atp * 2.0f;
    // Bonus for having accumulated multi-word context (up to +0.5)
    const float token_bonus = std::min(1.0f,
        static_cast<float>(accumulated_tokens_.size()) / 4.0f) * 0.5f;
    return base + token_bonus;
}

float DecisionLoop::score_store_memory(const NikolaState& s) const noexcept
{
    // Consolidate when there was a dopamine spike (worth remembering) AND
    // enough time has passed since last consolidation.
    if (seconds_since(last_store_time_) < cfg_.min_store_interval_s) return 0.f;
    if (!s.is_spiking()) return 0.f;
    return (s.dopamine - 0.5f) * 2.0f * 0.8f;  // ∈ (0, 0.8] when spiking
}

float DecisionLoop::score_request_lookup(const NikolaState& s) const noexcept
{
    // Low Shannon entropy means the field has collapsed into a narrow pattern —
    // the system is "stuck" and could benefit from new information.
    if (s.entropy >= 3.0f) return 0.f;
    if (s.tokens.empty()) return 0.f;
    const float urgency = std::max(0.f, (3.0f - s.entropy) / 3.0f);
    return urgency * 0.5f;
}

float DecisionLoop::score_recall_memory(const NikolaState& s) const
{
    // v0.0.9: Recall only fires if there is something in memory to draw on.
    // Added cooldown to prevent RECALL from dominating the action budget.
    if (memory_.empty()) return 0.f;
    if (seconds_since(last_recall_time_) < cfg_.min_recall_interval_s) return 0.f;
    // Don't recall when exhausted — retrieval has a metabolic cost.
    if (s.is_exhausted()) return 0.f;
    // When dopamine is spiking, STORE_MEMORY should win; don't compete.
    if (s.is_spiking()) return 0.f;

    // Probe resonance: find the best-matching stored record for the current field.
    // If nothing resonates (cosine near zero), score is 0 — no spurious recall.
    const auto hits = memory_.recall(torus_.wave_function(), 1);
    if (hits.empty()) return 0.f;

    // v0.0.9: Multiplier reduced from 1.5 → 0.6 so RECALL doesn't outcompete
    // EMIT_THOUGHT.  RECALL still fires when resonance is strong but gives
    // EMIT more tick budget for producing actual output.
    return hits[0].score * s.atp * 0.6f;
}

// ============================================================================
// score_reason
// ============================================================================

float DecisionLoop::score_reason(const NikolaState& s) const noexcept
{
    // REASON fires when the field has measurable entropy (disorganised) and the
    // system has enough metabolic energy to sustain the attention computation.
    //
    // Entropy note: the AutonomyEngine uses a Monte Carlo estimator that samples
    // K=1000 nodes.  For large grids (n=3, N=19683) this gives under-counted
    // values (typically 0.3–0.7 bits), so the gate and multiplier are calibrated
    // to the estimator's actual output range:
    //
    //   Formula:  entropy × ATP × 1.0
    //   typical (n=3): entropy=0.5, ATP=0.9 → 0.45  (beats SILENT=0.3)
    //   peak    (n=3): entropy=0.7, ATP=1.0 → 0.70
    //
    // Guard rails:
    //   • entropy < 0.05:   field is in a near-vacuum — no structure to reason about
    //   • ATP < 0.25:        too tired to run the NPT attention pass
    //   • cooldown < 3s:     prevent back-to-back REASON ticks consuming all cycles
    if (s.entropy < 0.05f) return 0.f;
    if (s.atp    < 0.25f) return 0.f;
    // v0.0.9: Cooldown reduced from 3.0s to 0.5s.  The NPT structures the
    // field so decoded tokens carry multi-band spectral coherence.  Letting
    // REASON fire more often (up to 4× per prompt) dramatically improves the
    // quality of subsequent EMIT_THOUGHT output.
    if (seconds_since(last_reason_time_) < cfg_.min_reason_interval_s) return 0.f;
    return s.entropy * s.atp * 1.0f;
}

// ============================================================================
// score_generate_code
// ============================================================================

float DecisionLoop::score_generate_code(const NikolaState& s) const noexcept
{
    // GENERATE_CODE fires when:
    //  - Nitpick specialist integration is enabled (specialist_server_path set)
    //  - ATP ≥ 0.30 (code generation + compile costs ~550 ATP)
    //  - Curiosity (boredom) is moderate (> 0.4) — Nikola wants to create
    //  - Cooldown satisfied (default 30s between generates)
    if (!aria_specialist_enabled_) return 0.f;
    if (s.atp < 0.30f) return 0.f;
    if (s.boredom < 0.4f) return 0.f;
    if (seconds_since(last_generate_time_) < cfg_.min_generate_interval_s) return 0.f;
    // Score proportional to boredom × ATP × 0.5
    // Typically: boredom=0.6, ATP=0.7 → 0.21 (needs to beat SILENT=0.3+threshold)
    // At high boredom: boredom=0.9, ATP=0.8 → 0.36 (wins over SILENT)
    return s.boredom * s.atp * 0.5f;
}

// ============================================================================
// execute_generate_code
// ============================================================================

std::string DecisionLoop::execute_generate_code(const NikolaState& s)
{
    if (!sie_) {
        return "generate_code: no SIE attached (boredom=" +
               std::to_string(s.boredom).substr(0, 4) +
               " atp=" + std::to_string(s.atp).substr(0, 4) + ")";
    }

    auto result = sie_->run_cycle(s);
    last_sie_result_ = result;

    std::string payload;
    if (result) {
        payload = "SIE_SUCCESS: module deployed (proposal=" +
                  std::to_string(result.proposal_id) +
                  " elapsed=" + std::to_string(static_cast<int>(result.elapsed_ms)) + "ms)";
    } else {
        payload = "SIE_" + std::string(sie_outcome_str(result.outcome)) +
                  " (elapsed=" + std::to_string(static_cast<int>(result.elapsed_ms)) + "ms)";
    }

    // Fire callback if registered
    if (on_sie_cycle) on_sie_cycle(result);

    return payload;
}

// ============================================================================
// score_pursue_goal
// ============================================================================

float DecisionLoop::score_pursue_goal(const NikolaState& s) const noexcept
{
    // PURSUE_GOAL fires when:
    //  - GoalSystem has an actionable goal (unblocked, active)
    //  - ATP ≥ 0.25 (goal pursuit needs energy)
    //  - Cooldown: at least 2s between pursue actions
    if (s.atp < 0.25f) return 0.f;
    if (seconds_since(last_pursue_goal_time_) < cfg_.min_pursue_goal_interval_s) return 0.f;

    const auto* goal = engine_.goal_system().active_goal();
    if (!goal) return 0.f;

    // Score = goal priority × ATP × 0.8
    // High-priority goals score higher.  With priority=0.7, ATP=0.8 → 0.448
    return goal->priority * s.atp * 0.8f;
}

// ============================================================================
// execute_pursue_goal
// ============================================================================

std::string DecisionLoop::execute_pursue_goal(const NikolaState& s)
{
    auto& gs = engine_.goal_system();
    const auto* goal = gs.active_goal();
    if (!goal) {
        return "pursue_goal: no active goal";
    }

    // Advance progress by a small increment proportional to ATP
    float progress_step = 0.05f * s.atp;
    uint64_t goal_id = goal->id;
    std::string desc = goal->description;
    std::string tier = goal_tier_str(goal->tier);

    (void)gs.update_progress(goal_id, goal->progress + progress_step);

    return "pursuing: " + desc + " [" + tier + "] (progress=" +
           std::to_string(goal->progress).substr(0, 4) + ")";
}

// ============================================================================
// build_payload
// ============================================================================

std::string DecisionLoop::build_payload(ActionType type, const NikolaState& s) const
{
    switch (type) {
        case ActionType::EMIT_THOUGHT: {
            // Route through ThoughtComposer — selects the template that best
            // matches current state drives and fills it with decoded content.
            nikola::cognitive::ThoughtContext ctx;
            // v0.0.9 — Priority order for token content:
            //   1. Accumulated tokens across prior ticks (multi-word buffer)
            //   2. Cold-decoded tokens from evolved field
            //   3. Warm-decoded tokens from most recent EXPLORE
            //   4. Seed token (guaranteed fallback)
            if (!accumulated_tokens_.empty()) {
                ctx.tokens = accumulated_tokens_;
            } else if (!s.tokens.empty()) {
                ctx.tokens = s.tokens;
            } else if (!last_ex_tokens_.empty()) {
                ctx.tokens = last_ex_tokens_;
            } else if (!last_seed_token_.empty()) {
                ctx.tokens = { last_seed_token_ };
            }
            ctx.dopamine = s.dopamine;
            ctx.boredom  = s.boredom;
            ctx.atp      = s.atp;
            ctx.td_error = s.td_error;
            ctx.entropy  = s.entropy;
            return thought_composer_.compose(ctx);
        }

        case ActionType::STORE_MEMORY:
            return "consolidating: " +
                   (s.tokens.empty() ? "(resonance)" : s.tokens.front());

        case ActionType::REQUEST_LOOKUP:
            // Use the first decoded token as the query seed
            return s.tokens.empty() ? "?" : s.tokens.front();

        case ActionType::EXPLORE:
            return "exploring (boredom=" + std::to_string(s.boredom).substr(0, 4) + ")";

        case ActionType::NAP:
            return "resting (atp=" + std::to_string(s.atp).substr(0, 4) + ")";

        case ActionType::REFUSE:
            return "refused (td=" + std::to_string(s.td_error).substr(0, 6) + ")";

        case ActionType::ESCALATE:
            // Include the triggering stimulus so the evidence record is
            // self-contained: any downstream agent can see what was asked.
            return "escalated: stimulus=[" + last_stimulus_ +
                   "] td=" + std::to_string(s.td_error).substr(0, 6);

        case ActionType::RECALL_MEMORY:
            // Content surfaces in the next EMIT_THOUGHT as the superposed field
            // evolves — the payload here contextualises the event for logging.
            return "recalling (memories=" + std::to_string(memory_.size()) + ")";

        case ActionType::REASON: {
            // Report entropy (what triggered it) and the top head score
            // (how strongly the NPT structured the output).
            float top = 0.f;
            for (float sc : npt_last_result_.head_scores)
                if (sc > top) top = sc;
            return "reasoning (entropy=" + std::to_string(s.entropy).substr(0, 4) +
                   " top_head=" + std::to_string(top).substr(0, 4) + ")";
        }

        case ActionType::GENERATE_CODE:
            return "generating aria code (boredom=" +
                   std::to_string(s.boredom).substr(0, 4) +
                   " atp=" + std::to_string(s.atp).substr(0, 4) + ")";

        case ActionType::PURSUE_GOAL: {
            const auto* goal = engine_.goal_system().active_goal();
            if (goal) {
                return "pursuing: " + goal->description + " [" +
                       std::string(goal_tier_str(goal->tier)) + "]";
            }
            return "pursue_goal: no active goal";
        }

        default:
            return "";
    }
}

// ============================================================================
// tick
// ============================================================================

DecisionResult DecisionLoop::tick()
{
    NIKOLA_PROFILE("DecisionLoop::tick");
    // ── 0. Reset per-tick warm state ─────────────────────────────────────────
    last_ex_tokens_.clear();

    // ── 0b. Field liveness check ─────────────────────────────────────────────
    // Reseed before scoring so the autonomy engine has a live field to read.
    const bool reseeded = [&]{
        NIKOLA_PROFILE("torus::reseed_check");
        return maybe_reseed_field();
    }();
    (void)reseeded;  // logged via torus energy shift; not surfaced as action

    // ── 1. Advance torus physics ─────────────────────────────────────────────
    const float dt = torus_.safe_dt();
    {
        NIKOLA_PROFILE("torus::run");
        torus_.run(cfg_.steps_per_tick, dt);
    }

    // ── 2. Update AutonomyEngine with current field state ───────────────────
    const auto& g = torus_.grid();
    const size_t N = g.num_active_nodes();
    const float elapsed_dt = dt * static_cast<float>(cfg_.steps_per_tick);

    {        NIKOLA_PROFILE("autonomy::tick");
        // Consume pending reward (set to POSITIVE after a successful EXPLORE
        // that found a seed token — primes dopamine spike for next EMIT_THOUGHT).
        const Reward tick_reward = pending_reward_;
        pending_reward_ = Reward::NEUTRAL;
        engine_.tick(elapsed_dt,
                     std::span<const float>(g.psi_real(), N),
                     std::span<const float>(g.psi_imag(), N),
                     tick_reward);
    }

    // ── 2b. Phase B1 — Mamba9D via HilbertMambaBridge ──────────────────────
    // Feed top-k hot nodes through the physics-aware Mamba9D pipeline.
    // HilbertMambaBridge sorts nodes by Hilbert index (locality-preserving),
    // extracts physics params from the torus, and steps Mamba9D with adaptive
    // A diagonal and spectrally clamped Δ.  This replaces the bare SSMLayer.
    std::string ssm_token;
    if (!cfg_.vocabulary.empty()) {
        NIKOLA_PROFILE("mamba9d::tick");
        constexpr size_t MAMBA_TOP_K = 8;  // one per emitter frequency
        const auto hot = torus_.hot_nodes(MAMBA_TOP_K);
        if (!hot.empty()) {
            // Run Mamba9D over hot nodes with physics-derived parameters
            mamba_bridge_.tick(
                g.psi_real(), g.psi_imag(), N, hot,
                /* resonance */ 0.5f, /* rho_G */ 1.0f);

            // Compute output logits and sample a token
            std::vector<float> logits;
            mamba_bridge_.mamba().ssm().compute_output(mamba_bridge_.state(), logits);
            const size_t token_idx = mamba_bridge_.mamba().sampler().sample_from_vector(
                logits, 0.01f);  // low temperature — favour high-probability tokens

            // Map to vocabulary word
            if (token_idx < cfg_.vocabulary.size()) {
                ssm_token = cfg_.vocabulary[token_idx];
            }
        }
    }

    // ── 3. Snapshot internal state ──────────────────────────────────────────
    NikolaState s = [&]{
        NIKOLA_PROFILE("autonomy::read_state");
        return read_state();
    }();

    // ── 3b. v0.0.9 — Accumulate decoded tokens across ticks ─────────────────
    // Collect unique tokens from cold decode (s.tokens) and warm decode
    // (last_ex_tokens_) into a persistent buffer.  When EMIT_THOUGHT fires,
    // it consumes this buffer to produce multi-word thoughts instead of
    // single-word template fills.  Capped at max_accumulated_tokens to
    // prevent unbounded growth.
    {
        const auto max_acc = static_cast<size_t>(cfg_.max_accumulated_tokens);
        auto try_add = [&](const std::string& tok) {
            if (tok.empty() || accumulated_tokens_.size() >= max_acc) return;
            if (accumulated_unique_.insert(tok).second)
                accumulated_tokens_.push_back(tok);
        };
        for (const auto& tok : s.tokens)    try_add(tok);
        for (const auto& tok : last_ex_tokens_) try_add(tok);
        if (!last_seed_token_.empty())       try_add(last_seed_token_);
        // Phase 16.1: SSM-generated token feeds into the same accumulator
        if (!ssm_token.empty())              try_add(ssm_token);
    }

    // ── 3c. v0.2.3 — Update AffectiveState from current NikolaState ────────
    affective_state_->update(s);

    // ── 4. Score all candidates ─────────────────────────────────────────────
    static constexpr float SILENT_SCORE = 0.3f;

    struct Candidate { ActionType type; float score; };
    Candidate candidates[11];
    {
        NIKOLA_PROFILE("autonomy::score_candidates");
        candidates[0]  = { ActionType::NAP,            score_nap(s)            };
        candidates[1]  = { ActionType::REFUSE,         score_refuse(s)         };
        candidates[2]  = { ActionType::ESCALATE,       score_escalate(s)       };
        candidates[3]  = { ActionType::EXPLORE,        score_explore(s)        };
        candidates[4]  = { ActionType::EMIT_THOUGHT,   score_emit_thought(s)   };
        candidates[5]  = { ActionType::STORE_MEMORY,   score_store_memory(s)   };
        candidates[6]  = { ActionType::REQUEST_LOOKUP, score_request_lookup(s) };
        candidates[7]  = { ActionType::RECALL_MEMORY,  score_recall_memory(s)  };
        candidates[8]  = { ActionType::REASON,         score_reason(s)         };
        candidates[9]  = { ActionType::GENERATE_CODE,  score_generate_code(s)  };
        candidates[10] = { ActionType::PURSUE_GOAL,    score_pursue_goal(s)    };
    }

    // ── 4b. v0.2.3 — Apply personality multiplier, preference bias, and
    //        affective modulation AFTER base scoring, BEFORE winner selection.
    //
    //   final_score = base_score × personality_multiplier + preference_bias
    //                 + affective_nudge
    //
    //   personality_multiplier: [0.7, 1.3] from PersonalityDrift
    //   preference_bias:       [-0.15, +0.15] from PreferenceEngine
    //   affective_nudge:        valence-driven modulation:
    //       positive valence → boost EXPLORE, PURSUE_GOAL (+0.05 × valence)
    //       negative valence → boost NAP, STORE_MEMORY (+0.05 × |valence|)
    //
    //   Gated behind PERSONALITY_WARMUP_TICKS to let the base scoring
    //   establish a stable baseline before personality influence kicks in.
    //   After warmup, influence ramps up gradually over PERSONALITY_RAMP_TICKS
    //   so personality grows organically rather than snapping on suddenly.
    static constexpr uint64_t PERSONALITY_WARMUP_TICKS = 200;
    static constexpr float    PERSONALITY_RAMP_TICKS   = 5000.0f;
    static constexpr float    PERSONALITY_MAX_STRENGTH = 1.0f;

    if (cfg_.enable_personality && tick_count_ >= PERSONALITY_WARMUP_TICKS) {
        const float valence = static_cast<float>(affective_state_->valence());

        // Gradual ramp: influence grows from 0 → MAX_STRENGTH over RAMP_TICKS
        const float age = static_cast<float>(tick_count_ - PERSONALITY_WARMUP_TICKS);
        const float ramp = std::min(1.0f, age / PERSONALITY_RAMP_TICKS);
        const float strength = PERSONALITY_MAX_STRENGTH * ramp;

        for (auto& c : candidates) {
            const int action_int = static_cast<int>(c.type);

            // Personality multiplier (temperament shapes action selection)
            const float raw_mult = personality_->action_multiplier(action_int);
            const float scaled_mult = 1.0f + (raw_mult - 1.0f) * strength;
            c.score *= scaled_mult;

            // Preference bias (learned likes/dislikes for action types)
            c.score += preferences_->action_bias(action_int) * strength;

            // Affective nudge (emotional state gently biases choices)
            const float affect_weight = 0.02f * ramp;
            if (valence > 0.0f) {
                // Positive valence → more ambitious (explore, pursue goals)
                if (c.type == ActionType::EXPLORE ||
                    c.type == ActionType::PURSUE_GOAL) {
                    c.score += affect_weight * valence;
                }
            } else if (valence < 0.0f) {
                // Negative valence → more conservative (rest, consolidate)
                if (c.type == ActionType::NAP ||
                    c.type == ActionType::STORE_MEMORY) {
                    c.score += affect_weight * (-valence);
                }
            }

            // Floor at zero — scores should never go negative
            if (c.score < 0.0f) c.score = 0.0f;
        }
    }

    // Find best non-silent candidate
    const Candidate* best = nullptr;
    for (const auto& c : candidates) {
        if (!best || c.score > best->score) best = &c;
    }

    // ── 5. Select winner ────────────────────────────────────────────────────
    ActionType winner  = ActionType::SILENT;
    float      wscore  = SILENT_SCORE;

    if (best && best->score > SILENT_SCORE + cfg_.action_threshold) {
        winner = best->type;
        wscore = best->score;
    }

    // ── 6. Update cooldowns ─────────────────────────────────────────────────
    const auto now = std::chrono::steady_clock::now();
    if (winner == ActionType::EMIT_THOUGHT)  last_emit_time_   = now;
    if (winner == ActionType::STORE_MEMORY)  last_store_time_  = now;
    if (winner == ActionType::REASON)        last_reason_time_ = now;
    if (winner == ActionType::RECALL_MEMORY) last_recall_time_ = now;  // v0.0.9
    if (winner == ActionType::GENERATE_CODE) last_generate_time_ = now;  // v0.0.19

    // ── 6a. Execute side-effects for actions that modify the torus ───────────
    // EXPLORE: inject stochastic novelty into the field NOW (after scoring,
    // so this tick's scores are unaffected, but next tick benefits).
    // Also performs warm delta-decode immediately after inject, while the
    // semantic signal is strongest.  Result stored in last_ex_tokens_.
    std::string explore_payload;
    if (winner == ActionType::EXPLORE) {
        explore_payload = execute_explore(s);
        // Thread warm tokens into s.tokens so last_state_ carries them.
        if (s.tokens.empty() && !last_ex_tokens_.empty()) {
            s.tokens = last_ex_tokens_;
        }
        // Signal POSITIVE reward next tick if exploration found a seed token.
        // This causes a dopamine spike that enables EMIT_THOUGHT to win
        // over a subsequent EXPLORE cycle (dopamine > 0.6 required).
        if (!last_seed_token_.empty()) {
            pending_reward_ = Reward::POSITIVE;
        }
    }

    // ── 6b. Memory side-effects ───────────────────────────────────────────────
    // STORE_MEMORY: snapshot the current wave-field into SemanticMemory and
    // flush to disk if a memory_path is configured.  This is the primary
    // write path — each dopamine-spike moment becomes a durable record.
    if (winner == ActionType::STORE_MEMORY) {
        memory_.store(torus_.wave_function());
        save_memory();
    }
    // NAP: homeostatic maintenance — decay all record strengths by elapsed
    // physics time and prune anything that has faded below MIN_STRENGTH.
    if (winner == ActionType::NAP) {
        memory_.decay(elapsed_dt);
        memory_.consolidate();
    }
    // RECALL_MEMORY: find the most resonant stored record and superpose it
    // onto the live field, coloring current cognition with echoes of past
    // experience.  Blend weight α = resonance_score × 0.4 — a faint echo
    // stays faint; a strongly resonant memory imprints meaningfully.
    // On the next tick the shifted field may resonate with a different
    // record, producing the associative cascade described in the design notes.
    if (winner == ActionType::RECALL_MEMORY) {
        const auto hits = memory_.recall(torus_.wave_function(), 1);
        if (!hits.empty()) {
            const float alpha = hits[0].score * 0.4f;
            memory_.superpose(hits[0].key, alpha, torus_.wave_function());
        }
    }
    // REASON: run NPT forward pass and blend the frequency-structured output
    // back into the live torus field, priming the next EMIT_THOUGHT.
    if (winner == ActionType::REASON) {
        execute_reason();
    }
    // GENERATE_CODE: run the full self-improvement cycle through the SIE.
    // Specialist query → extract → compile → package → sign → deploy → store.
    std::string generate_payload;
    if (winner == ActionType::GENERATE_CODE) {
        generate_payload = execute_generate_code(s);
    }
    // PURSUE_GOAL: advance the active goal from GoalSystem.
    std::string pursue_goal_payload;
    if (winner == ActionType::PURSUE_GOAL) {
        pursue_goal_payload = execute_pursue_goal(s);
        last_pursue_goal_time_ = std::chrono::steady_clock::now();
    }

    // ── 7. Build result ──────────────────────────────────────────────────────
    s.last_action = winner;
    last_state_   = s;
    ++tick_count_;

    DecisionResult result;
    result.type    = winner;
    result.score   = wscore;

    // Select payload: side-effect actions produce their own payload,
    // otherwise build_payload() generates a descriptive string.
    if (winner == ActionType::EXPLORE && !explore_payload.empty()) {
        result.payload = explore_payload;
    } else if (winner == ActionType::GENERATE_CODE && !generate_payload.empty()) {
        result.payload = generate_payload;
    } else if (winner == ActionType::PURSUE_GOAL && !pursue_goal_payload.empty()) {
        result.payload = pursue_goal_payload;
    } else {
        result.payload = build_payload(winner, s);
    }

    // ── 7a. v0.0.9 — consume accumulated tokens AFTER build_payload reads them
    if (winner == ActionType::EMIT_THOUGHT) {
        accumulated_tokens_.clear();
        accumulated_unique_.clear();
    }

    result.state   = s;

    // ── 8. Fire callbacks ────────────────────────────────────────────────────
    if (on_tick)   on_tick(s);
    if (on_action && winner != ActionType::SILENT) on_action(result);

    // ── 8b. v0.2.3 — Feed outcome back to personality/preference modules ────
    // Gated behind the same warmup period as score modulation so preferences
    // and personality don't accumulate during the base-scoring warmup phase.
    if (cfg_.enable_personality && winner != ActionType::SILENT && tick_count_ >= PERSONALITY_WARMUP_TICKS) {
        const int action_int = static_cast<int>(winner);

        // Implicit preference learning: reinforce the chosen action
        preferences_->learn_from_action(action_int, tick_count_);

        // Personality drift: construct outcome from action result
        // Success approximation: positive TD error = likely good outcome
        float success = std::clamp(static_cast<float>(s.td_error), -1.0f, 1.0f);
        float risk    = (winner == ActionType::EXPLORE ||
                         winner == ActionType::PURSUE_GOAL ||
                         winner == ActionType::GENERATE_CODE) ? 0.6f : 0.2f;
        float complexity = 0.3f;
        if (winner == ActionType::REASON || winner == ActionType::GENERATE_CODE)
            complexity = 0.7f;

        interior::ExperienceOutcome outcome{success, action_int, risk, complexity};
        personality_->apply_outcome(outcome);

        // Milestone detection: first-action milestones
        auto milestone_desc = narrative_growth_->check_first_action(
            action_int, tick_count_);
        if (!milestone_desc.empty()) {
            narrative_growth_->record_milestone(
                *autobiography_,
                interior::MilestoneType::FIRST_ACTION,
                milestone_desc, s,
                affective_state_->current_affect(),
                tick_count_);
        }

        // Personality shift milestones (check every 100 ticks)
        if (tick_count_ % 100 == 0) {
            auto snap = personality_->snapshot();
            auto shifts = narrative_growth_->check_personality_shifts(
                snap, tick_count_);
            for (const auto& desc : shifts) {
                narrative_growth_->record_milestone(
                    *autobiography_,
                    interior::MilestoneType::PERSONALITY_SHIFT,
                    desc, s,
                    affective_state_->current_affect(),
                    tick_count_);
            }
        }
    }

    // NAP: trigger self-reflection and homeostatic decay
    if (cfg_.enable_personality && winner == ActionType::NAP) {
        // Self-reflection during NAP
        narrative_growth_->generate_reflection(
            *autobiography_,
            personality_->snapshot(),
            1000,  // look back ~1000 ticks
            tick_count_);

        // Homeostatic decay of personality (gentle regression to balanced)
        personality_->decay(elapsed_dt);

        // Preference decay
        preferences_->decay(elapsed_dt);

        // Autobiography compression if needed
        if (narrative_growth_->needs_compression(*autobiography_)) {
            narrative_growth_->compress(*autobiography_, s, tick_count_);
        }

        // Personality epoch reset during NAP (allow new drift)
        personality_->reset_epoch();
    }

    // ── 9. Phase 137 — persist state to LMDB ────────────────────────────────
    if (state_store_) {
        try {
            state_store_->save_state(s, tick_count_);
            // Checkpoint Ψ every cfg_.checkpoint_interval ticks
            if (cfg_.checkpoint_interval > 0 &&
                (tick_count_ % static_cast<uint64_t>(cfg_.checkpoint_interval)) == 0) {
                state_store_->save_checkpoint(torus_.wave_function(), tick_count_);
            }
        } catch (const std::exception& e) {
            std::cerr << "[DecisionLoop] state persist failed: " << e.what() << "\n";
        }
    }

    return result;
}

// ============================================================================
// execute_explore
// ============================================================================

std::string DecisionLoop::execute_explore(const NikolaState& s)
{
    using nikola::foundation::Nit;
    using nikola::foundation::quantize_wave;

    // ── 1. Choose a seed token ────────────────────────────────────────────────
    // Preference order:
    //   a) Decode the hottest active node — continue the thread that's already
    //      resonating (if any).
    //   b) Pick pseudorandomly from cfg_.vocabulary seeded by tick_count_
    //      (deterministic variety without std::random overhead).

    std::string seed_token;

    // v0.0.9: Stimulus-seeded explore comes FIRST — keep early thoughts
    // orbiting the prompt's semantic neighbourhood instead of whatever the
    // field happened to collapse to.  Rotate through stimulus_seeds_ (literal
    // vocab words found in the prompt text) for diversity.
    if (!stimulus_seeds_.empty()
            && stimulus_explore_count_ < static_cast<uint64_t>(cfg_.max_stimulus_explores)) {
        const size_t idx = stimulus_explore_count_ % stimulus_seeds_.size();
        seed_token = stimulus_seeds_[idx];
    } else if (!last_stimulus_seed_.empty()
            && stimulus_explore_count_ < static_cast<uint64_t>(cfg_.max_stimulus_explores)) {
        seed_token = last_stimulus_seed_;
    }

    // Try to get a token from whatever the field is already resonating on
    if (seed_token.empty()) {
        const auto hot = torus_.hot_nodes(3);
        for (size_t idx : hot) {
            const auto wave9d = torus_.node_wave9d(idx);
            if (wave9d.empty()) continue;
            auto maybe = decoder_.lexicon().decode(wave9d);
            if (maybe.has_value()) { seed_token = *maybe; break; }
        }
    }
    ++stimulus_explore_count_;

    // Fall back to pseudorandom vocabulary pick
    if (seed_token.empty() && !cfg_.vocabulary.empty()) {
        const size_t pick = tick_count_ % cfg_.vocabulary.size();
        seed_token = cfg_.vocabulary[pick];
    }

    // ── 2. Build semantic pulse from seed token's stored Nit vector ─────────
    // Phase 28: use original_vocab_nits_[seed] directly — this is the same
    // Nit pulse stored during calibration (BERT in ORT mode; tiled spectral
    // wave in non-ORT mode).  Using the stored vector ensures the inject and
    // the analytic match are self-consistent: analytic_signature(nit, t) for
    // the seed token will exactly equal analytic_signature(final_pulse, t)
    // when noise_ratio==0, giving cosine==1.0 for a clean identification.

    std::vector<Nit> semantic_pulse(128, Nit{0});

    if (!seed_token.empty()) {
        const auto nit_it = original_vocab_nits_.find(seed_token);
        if (nit_it != original_vocab_nits_.end() && !nit_it->second.empty()) {
            semantic_pulse = nit_it->second;
            semantic_pulse.resize(128, Nit{0});
        } else {
            // original_vocab_nits_ not yet populated (very early startup) —
            // fall back to tiled spectral wave from original_vocab_waves_.
            const auto orig_it = original_vocab_waves_.find(seed_token);
            if (orig_it != original_vocab_waves_.end()) {
                const auto& w9 = orig_it->second;
                for (size_t i = 0; i < 128; ++i) {
                    const std::complex<double> v{
                        static_cast<double>(w9[i % 9].real()),
                        static_cast<double>(w9[i % 9].imag())
                    };
                    semantic_pulse[i] = quantize_wave(v);
                }
            }
        }
    }

    // ── 3. Build noise pulse (LCG, same as before) ───────────────────────────
    const int amp_tier = 1 + static_cast<int>(s.boredom * 2.99f);  // 1..3
    uint64_t rng = tick_count_ * 6364136223846793005ULL + 1442695040888963407ULL;
    auto lcg_next = [&]() -> int {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<int>((rng >> 33) & 0x7) - 3;
    };

    std::vector<Nit> noise_pulse(128);
    for (auto& n : noise_pulse)
        n = Nit(std::clamp(lcg_next(), -amp_tier, amp_tier));

    // ── 4. Blend: noise_ratio scales with boredom but caps at 40% ────────────
    // Even at maximum boredom, 60% of the pulse is anchored to a known concept.
    // This keeps the decoder able to find tokens while still providing novelty.
    //
    // Think of it as: "I'm bored, I want something new — but I'm still thinking
    // in words, still somewhere in the neighbourhood of meaning."
    const float noise_ratio = s.boredom * 0.4f;  // 0.0 .. 0.4

    std::vector<Nit> final_pulse(128);
    for (size_t i = 0; i < 128; ++i) {
        const float sem   = static_cast<float>(static_cast<int>(semantic_pulse[i]));
        const float noise = static_cast<float>(static_cast<int>(noise_pulse[i]));
        const float blended = sem * (1.f - noise_ratio) + noise * noise_ratio;
        final_pulse[i] = Nit(std::clamp(static_cast<int>(std::round(blended)),
                                         -amp_tier, amp_tier));
    }

    // ── Phase 28: Analytic warm decode ──────────────────────────────────────
    // Record the injection time BEFORE calling inject_raw so that both the
    // actual call and the analytic evaluation use exactly the same t.
    // (inject_raw does not advance torus_.time() — only torus_.step() does.)
    //
    // The closed-form for chord c injected at time t:
    //   chord_c(t) = Σ_n A_{n,c} · e^{i·f_n·(t + c·Δt_c)}   where f_n = π·φⁿ
    //
    // Because f_n are Weyl-equidistributed (incommensurate irrational ratios),
    // evaluating both the actual pulse and each candidate at the SAME t gives
    // an exact cosine comparison regardless of how long the loop has run.
    // No pre-inject snapshot, no probe node, no save_state needed.
    {
        using Injector = nikola::cognitive::HolographicInjector<nikola::foundation::TorusGrid>;
        const double inject_t = static_cast<double>(torus_.time());

        torus_.inject_raw(final_pulse, inject_t);

        // Compute the analytic chord-amplitude signature for the actual injection.
        const auto actual_sig = Injector::analytic_signature(final_pulse, inject_t);

        // Compare against every vocabulary token's expected signature at the same t.
        // Token whose Nit pulse best explains the injection wins.
        constexpr float kMinAnalyticCos = 0.05f;  // gentle floor; analytic match is exact
        float best_cos = kMinAnalyticCos;
        std::string best_tok;
        for (const auto& [tok, tok_nits] : original_vocab_nits_) {
            const auto expected_sig = Injector::analytic_signature(tok_nits, inject_t);
            const float cos = Injector::signature_cosine(actual_sig, expected_sig);
            if (cos > best_cos) { best_cos = cos; best_tok = tok; }
        }

        if (!best_tok.empty()) {
            last_ex_tokens_ = { best_tok };
        } else if (!seed_token.empty()) {
            // Guaranteed fallback: if all tokens have zero Nit energy or
            // original_vocab_nits_ is empty, use the seed chosen explicitly --
            // the injection was shaped around its semantic content.
            last_ex_tokens_ = { seed_token };
        }
    }

    // Record the seed so EMIT_THOUGHT can reference the resonating concept.
    last_seed_token_ = seed_token;

    // ── 5. Payload ────────────────────────────────────────────────────────────
    const std::string tier_name = amp_tier == 1 ? "subtle" :
                                   amp_tier == 2 ? "moderate" : "strong";
    return "excitation:" + tier_name +
           (seed_token.empty() ? "" : " seed=" + seed_token) +
           " boredom=" + std::to_string(s.boredom).substr(0, 4);
}

// ============================================================================
// execute_reason
// ============================================================================

void DecisionLoop::execute_reason()
{
    // Phase B2: NPT reads Mamba's perception, not the raw torus.
    // Convert Mamba hidden state → WaveFunction via reverse HilbertMambaBridge.
    // The compression bottleneck (256 → WaveFunction) is intentional:
    // Mamba abstracts the manifold, NPT reasons over the abstraction.
    auto mamba_wf = mamba_bridge_.state_to_wave_function();

    // Phase 45: Pass live dopamine level so NPT scales Q/K learning rates
    // by η(D) = 1 + tanh(D − 0.5).  Reward spikes → faster encoding;
    // punishment dips → plasticity lock.
    // Phase 46: Pass live serotonin so NPT applies elastic restoring force.
    // Phase 47: Pass live norepinephrine so NPT modulates attention temperature.
    auto result = npt_.forward(mamba_wf,
                               engine_.dopamine(),
                               engine_.serotonin(),
                               engine_.norepinephrine());

    if (result.has_output) {
        // Phase B3: NPT output feeds back to Mamba, not directly to torus.
        // Mamba is the SOLE writer to the torus — NPT never writes directly.
        // 1. Extract dominant 9D coordinate from NPT output WaveFunction
        auto npt_input = nikola::cognitive::HilbertMambaBridge::wave_function_to_input(
            result.output);

        // 2. Construct mild feedback physics (no specific torus physics here —
        //    this is reasoning feedback, not sensory perception)
        nikola::cognitive::PhysicsParams feedback_physics;
        feedback_physics.resonance = 0.5f;
        feedback_physics.rho_G     = 1.0f;
        // Uniform intensity — reasoning doesn't bias decay rates
        feedback_physics.intensity.fill(0.5f);
        feedback_physics.phase.fill(0.f);

        // 3. Step Mamba with NPT's reasoning output (bidirectional link)
        mamba_bridge_.mamba().step(mamba_bridge_.state(), npt_input, feedback_physics);

        // 4. Mamba writes torus: project updated hidden state to wave modification
        //    Weight α = top head score × 0.3 — same blending semantics as before,
        //    but now Mamba mediates the write.
        const float top_score = *std::max_element(
            result.head_scores.begin(), result.head_scores.end());
        auto mamba_output_wf = mamba_bridge_.state_to_wave_function();
        torus_.wave_function().add_scaled(mamba_output_wf, top_score * 0.3f);
    }

    // Cache for external inspection via last_npt_result().
    npt_last_result_ = std::move(result);
}

// ============================================================================
// maybe_reseed_field
// ============================================================================

bool DecisionLoop::maybe_reseed_field()
{
    using nikola::foundation::Nit;

    // Only intervene when the field has truly died.
    if (torus_.total_probability() >= 1e-3) return false;

    // Broad-spectrum curiosity heartbeat:
    // 128 Nits in a repeating +2 / −1 / +1 / −2 pattern.
    // This creates balanced energy across all 8 emitter frequencies without
    // biasing toward any particular semantic content.
    static const std::array<int, 4> PATTERN = {2, -1, 1, -2};
    std::vector<Nit> pulse(128);
    for (size_t i = 0; i < 128; ++i) {
        pulse[i] = Nit(PATTERN[i % 4]);
    }

    torus_.inject_raw(pulse, static_cast<double>(torus_.time()));
    return true;
}

// ============================================================================
// calibrate_vocabulary_to_torus_space  (Phase 28 — analytic Nit-pulse store)
// ============================================================================

/**
 * Analytic-injection calibration  (Phase 28 replacement for delta approach).
 *
 * The Phase 27 delta approach was correct in spirit but failed in practice
 * because calibration at t≈0 and EXPLORE at t≫0 live at completely different
 * positions on the Weyl-equidistributed emitter orbit.  The same Nit vector
 * produces completely different chord amplitudes at different times — by
 * design, so that the orbit is dense in the torus space.
 *
 * Phase 28 solution: instead of storing a t₀ snapshot and comparing it to a
 * tₙ snapshot (which are in orthogonal regions of phase space), we store the
 * RAW NIT VECTORS for each vocabulary token and evaluate the closed-form
 * injection function analytically at the actual injection time:
 *
 *   chord_c(t) = Σ_n A_{n,c} · e^{i·f_n·(t + c·Δt_c)}
 *
 * Evaluating this for both the actual injected pulse and each candidate
 * token's pulse at the SAME t gives a mathematically exact cosine comparison,
 * immune to time-drift.  No snapshots, no save/restore, no probe nodes.
 *
 * Per token this method:
 *   1. Builds a 128-Nit pulse (BERT in ORT mode; tiled spectral wave otherwise)
 *   2. Stores it in original_vocab_nits_[token]
 *   3. Injects it into the torus for cold decode calibration (lexicon update)
 */
void DecisionLoop::calibrate_vocabulary_to_torus_space()
{
    using nikola::foundation::Nit;
    using nikola::foundation::quantize_wave;

    size_t calibrated = 0;

    for (const auto& token : cfg_.vocabulary) {
        // ── 1. Build 128-Nit pulse ────────────────────────────────────────────
        std::vector<Nit> pulse(128, Nit{0});

#ifdef NIKOLA_HAS_ORT
        // ORT path: use actual BERT quantisation for maximum semantic fidelity.
        try {
            pulse = torus_.embed_nits(token);
            pulse.resize(128, Nit{0});  // ensure exactly 128
        } catch (...) { pulse.assign(128, Nit{0}); }
#endif

        // Fall through (or non-ORT): tile the spectral wave from the lexicon.
        bool has_energy = std::any_of(pulse.begin(), pulse.end(),
            [](const Nit& n) { return static_cast<int>(n) != 0; });
        if (!has_energy) {
            auto it = original_vocab_waves_.find(token);
            if (it != original_vocab_waves_.end()) {
                const auto& w9 = it->second;
                for (size_t i = 0; i < 128; ++i)
                    pulse[i] = quantize_wave(std::complex<double>(
                        static_cast<double>(w9[i % 9].real()),
                        static_cast<double>(w9[i % 9].imag())));
            }
        }
        // Last resort: synthetic character-hash wave.
        if (!std::any_of(pulse.begin(), pulse.end(),
                [](const Nit& n) { return static_cast<int>(n) != 0; })) {
            const auto sw = synthetic_wave9d(token);
            for (size_t i = 0; i < 128; ++i)
                pulse[i] = quantize_wave(std::complex<double>(
                    static_cast<double>(sw[i % 9].real()),
                    static_cast<double>(sw[i % 9].imag())));
        }

        // ── 2. Phase 28 key step: store Nit pulse for analytic decode ─────────
        // HolographicInjector::analytic_signature(pulse, t) can evaluate the
        // expected injection chord amplitudes at ANY future time t without
        // needing grid access or saved snapshots.
        original_vocab_nits_[token] = pulse;

        // ── 3. Inject into torus for cold decode calibration ─────────────────
        // The absolute post-inject waveform at the hot node is registered in
        // the lexicon.  This improves cold decode at times near t=0 (startup).
        // At later times warm decode (analytic path) takes over.
        torus_.inject_raw(pulse, static_cast<double>(torus_.time()));
        const auto hot1 = torus_.hot_nodes(1);
        if (!hot1.empty()) {
            const auto post_wave = torus_.node_wave9d(hot1[0]);
            std::vector<nikola::cognitive::Complex> abs_wave(
                post_wave.cbegin(), post_wave.cend());
            decoder_.register_token(token, abs_wave);
        }

        ++calibrated;
    }

    std::cout << "[DecisionLoop] Phase 28 calibration: " << calibrated
              << "/" << cfg_.vocabulary.size()
              << " tokens stored (Nit pulses for analytic decode + torus abs waves)\n";
}

// ============================================================================
// helpers
// ============================================================================

float DecisionLoop::seconds_since(std::chrono::steady_clock::time_point t) const noexcept
{
    const auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<float>(now - t).count();
}

// ============================================================================
// save_memory
// ============================================================================

void DecisionLoop::save_memory() const
{
    if (!cfg_.lmdb_memory_path.empty()) {
        try {
            nikola::cognitive::save_lmdb(memory_, cfg_.lmdb_memory_path);
        } catch (const std::exception& e) {
            std::cerr << "[DecisionLoop] save_memory (LMDB) failed: " << e.what() << "\n";
        }
        return;
    }
    if (cfg_.memory_path.empty()) return;
    try {
        memory_.save(cfg_.memory_path);
    } catch (const std::exception& e) {
        std::cerr << "[DecisionLoop] save_memory failed: " << e.what() << "\n";
    }
}

// ============================================================================
// force_store_wavefield (training-mode bypass)
// ============================================================================

void DecisionLoop::force_store_wavefield()
{
    memory_.store(torus_.wave_function());
    save_memory();
}

// ============================================================================
// Phase 137 — full state persistence helpers
// ============================================================================

void DecisionLoop::save_full_state(bool force_checkpoint)
{
    if (!state_store_) return;
    try {
        state_store_->save_state(last_state_, tick_count_);
        if (force_checkpoint)
            state_store_->save_checkpoint(torus_.wave_function(), tick_count_);
        state_store_->save_autobiography(*autobiography_);
    } catch (const std::exception& e) {
        std::cerr << "[DecisionLoop] save_full_state failed: " << e.what() << "\n";
    }
}

void DecisionLoop::load_full_state()
{
    if (!state_store_) return;
    try {
        {
            NikolaState restored;
            uint64_t restored_tick = 0;
            if (state_store_->load_latest_state(restored, restored_tick)) {
                last_state_ = restored;
                tick_count_ = restored_tick;
            }
        }

        {
            physics::WaveFunction wf;
            persistence::detail::CheckpointHeader chdr{};
            if (state_store_->load_latest_checkpoint(wf, chdr)) {
                auto& grid = torus_.wave_function().grid();
                const size_t N = grid.num_active_nodes();
                if (wf.num_nodes() == N) {
                    std::memcpy(grid.psi_real(), wf.grid().psi_real(), N * sizeof(float));
                    std::memcpy(grid.psi_imag(), wf.grid().psi_imag(), N * sizeof(float));
                    std::memcpy(grid.vel_real(), wf.grid().vel_real(), N * sizeof(float));
                    std::memcpy(grid.vel_imag(), wf.grid().vel_imag(), N * sizeof(float));
                }
            }
        }

        (void)state_store_->load_autobiography(*autobiography_);
    } catch (const std::exception& e) {
        std::cerr << "[DecisionLoop] load_full_state failed: " << e.what() << "\n";
    }
}

} // namespace nikola::autonomy
