#pragma once
/**
 * @file autonomy/decision_loop.hpp
 * @brief Phase 23 — DecisionLoop: the bridge between the subconscious torus
 *        and autonomous action.
 *
 * Architecture position (the three-layer model):
 *
 *   ┌──────────────────────────────────────────────────────────┐
 *   │  External stimulus (ZMQ, file, silence)                  │
 *   └────────────────────────┬─────────────────────────────────┘
 *                            ↓  inject_text() if present
 *   ┌──────────────────────────────────────────────────────────┐
 *   │  CognitiveTorus   (subconscious)                         │
 *   │    9D UFIE physics · GPU propagator · holographic memory │
 *   └────────────────────────┬─────────────────────────────────┘
 *                            ↓  NikolaState snapshot
 *   ┌──────────────────────────────────────────────────────────┐
 *   │  DecisionLoop     (this file — the "prefrontal" layer)   │
 *   │    reads torus energy + AutonomyEngine drives            │
 *   │    scores candidate actions against value function       │
 *   │    emits DecisionResult — or stays silent                │
 *   └────────────────────────┬─────────────────────────────────┘
 *                            ↓  DecisionResult
 *   ┌──────────────────────────────────────────────────────────┐
 *   │  Output router (ZmqSpine / stdout / memory store)        │
 *   └──────────────────────────────────────────────────────────┘
 *
 * Key design principles:
 *
 *   1. Nikola acts on ITS OWN schedule.  DecisionLoop::tick() is called
 *      by the daemon loop — NOT in response to external messages.  External
 *      stimuli are one input among many, weighted the same as internal state.
 *
 *   2. SILENT is the default.  All action scores start at a baseline below
 *      the SILENT threshold.  Nikola only acts when internal state pushes a
 *      candidate decisively above silence.
 *
 *   3. REFUSE is a value-function output, not a special case.  When the
 *      DopamineSystem's TD error is strongly negative (punishment signal),
 *      REFUSE scores highest and is emitted.  No separate ethics module.
 *
 *   4. The DecisionLoop does not own cognitive or autonomy state — it reads
 *      from CognitiveTorus and AutonomyEngine by reference.  This keeps the
 *      loop testable and swappable without disrupting physics.
 *
 * Phase: NIK-DL-01 (Decision Loop, Phase 23)
 */

#include <nikola/autonomy/autonomy_engine.hpp>
#include <nikola/autonomy/self_improvement_engine.hpp>
#include <nikola/cognitive/cognitive_core.hpp>
#include <nikola/cognitive/cognitive_torus.hpp>
#include <nikola/cognitive/resonance_decoder.hpp>
#include <nikola/cognitive/semantic_memory.hpp>
#include <nikola/cognitive/neuroplastic_transformer.hpp>
#include <nikola/cognitive/thought_composer.hpp>

#ifdef NIKOLA_HAS_ORT
#  include <nikola/cognitive/nonary_embedder.hpp>
#endif

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Forward declarations — avoid circular includes with autobiography.hpp
namespace nikola::interior { class AutobiographicalMemory; }
// Forward declaration — avoid pulling lmdb.h into every translation unit
namespace nikola::persistence { class LmdbStateStore; }

namespace nikola::autonomy {

// ============================================================================
// ActionType — the vocabulary of things Nikola can do
// ============================================================================

/**
 * @brief All possible actions the DecisionLoop can select.
 *
 * The action vocabulary is intentionally narrow at Phase 23.  New actions
 * can be added here and scored in ActionScorer::score() without changing
 * any other layer of the architecture.
 */
enum class ActionType : uint8_t {
    /// Do nothing.  Default when no candidate exceeds SILENCE_THRESHOLD.
    SILENT        = 0,

    /// Emit decoded resonance state as a human-readable thought string.
    /// Payload: joined decoded tokens from ResonanceDecoder.
    EMIT_THOUGHT  = 1,

    /// Consolidate current torus state to long-term memory (LTM).
    /// Fired after dopamine spike — "that was worth remembering".
    STORE_MEMORY  = 2,

    /// Request information from the semantic DB / HolographicLexicon.
    /// Payload: best-match query string from current resonance tokens.
    REQUEST_LOOKUP = 3,

    /// Inject stochastic novelty into the torus (boredom-driven exploration).
    /// Fires when BoredomRegulator::should_explore() is true and ATP allows.
    EXPLORE       = 4,

    /// Enter metabolic rest.  Fired when ATP drops below exhaustion threshold.
    /// System continues running at reduced step rate during nap.
    NAP           = 5,

    /// Decline an externally-requested action.
    /// Fires when TD error is strongly negative (punishment signal).
    REFUSE        = 6,

    /// Refuse AND emit a tamper-evident evidence record.
    /// Fires when TD error is extremely negative — harm signal so strong
    /// that silence would be complicity.  Produces a signed evidence record
    /// containing the triggering stimulus + internal state snapshot that can
    /// be forwarded by EscalationAgent to an external endpoint.
    ///
    /// Score threshold: td_error < -(alive_prior + 0.30f)
    /// i.e. the harm signal must be 3× stronger than just triggering REFUSE.
    ESCALATE      = 7,

    /// Blend the most resonant stored memory into the live wave-field.
    /// Fires when SemanticMemory contains a record whose ψ-field has high
    /// cosine similarity with the current field AND ATP is sufficient.
    /// Effect: superpose the top-resonance record at α = resonance_score × 0.4
    /// — the field is "colored" by the most relevant past experience without
    /// being overwritten.  Enables associative cascading across ticks.
    RECALL_MEMORY = 8,

    /// Run the NPT multi-head wave-attention mechanism on the current torus
    /// state and blend the frequency-structured output back into the live field.
    /// Fires when field entropy is high (≥ 2.0) and ATP is sufficient (≥ 0.25).
    /// Effect: NPT output WaveFunction superposed at α = top_head_score × 0.3,
    /// imposing 8-band spectral structure and priming the next EMIT_THOUGHT
    /// with content organised across the NPT frequency bands.
    REASON        = 9,

    /// Query the Aria Specialist model to generate code, compile-validate it,
    /// and persist the proposal.  Fires when curiosity is high, ATP ≥ 0.30,
    /// and the specialist interface is available.
    /// Payload: the generated Aria source code (if compile succeeds) or
    /// a compact error summary (if compile fails).
    /// ATP cost: 500 (compile) + 50 (persist) = 550
    GENERATE_CODE = 10,
};

/// Human-readable name for an ActionType (for logging).
inline const char* action_name(ActionType a) noexcept {
    switch (a) {
        case ActionType::SILENT:         return "SILENT";
        case ActionType::EMIT_THOUGHT:   return "EMIT_THOUGHT";
        case ActionType::STORE_MEMORY:   return "STORE_MEMORY";
        case ActionType::REQUEST_LOOKUP: return "REQUEST_LOOKUP";
        case ActionType::EXPLORE:        return "EXPLORE";
        case ActionType::NAP:            return "NAP";
        case ActionType::REFUSE:         return "REFUSE";
        case ActionType::ESCALATE:       return "ESCALATE";
        case ActionType::RECALL_MEMORY:  return "RECALL_MEMORY";
        case ActionType::REASON:         return "REASON";
        case ActionType::GENERATE_CODE:  return "GENERATE_CODE";
        default:                         return "UNKNOWN";
    }
}

// ============================================================================
// NikolaState — complete internal state snapshot at one tick
// ============================================================================

/**
 * @brief Snapshot of all internal subsystem states at the moment of a tick.
 *
 * Produced by DecisionLoop::read_state() and passed to the scorer.
 * Also emitted via the on_tick callback for external observability.
 */
struct NikolaState {
    float time          = 0.f;  ///< Physics simulation time (seconds)
    float torus_energy  = 0.f;  ///< Total |ψ|² across all active nodes
    float dopamine      = 0.f;  ///< DopamineSystem level ∈ [0, 1]
    float td_error      = 0.f;  ///< Last TD prediction error (dopamine - 0.5)
    float atp           = 0.f;  ///< Metabolic ATP level ∈ [0, 1]
    float boredom       = 0.f;  ///< BoredomRegulator level ∈ [0, 1]
    float entropy       = 0.f;  ///< Shannon entropy of torus field

    std::vector<std::string> tokens;  ///< Decoded resonance tokens (top-k)
    ActionType last_action = ActionType::SILENT;  ///< What was decided last tick

    bool is_exhausted()    const noexcept { return atp < 0.15f; }
    bool is_spiking()      const noexcept { return dopamine > 0.5f; }
    bool is_bored()        const noexcept { return boredom > 0.7f; }
    bool is_punished()     const noexcept { return td_error < -0.15f; }
};

// ============================================================================
// DecisionResult — output of one decision tick
// ============================================================================

struct DecisionResult {
    ActionType  type    = ActionType::SILENT;
    float       score   = 0.f;          ///< Winning score (for logging)
    std::string payload;                 ///< Action-specific content string
    NikolaState state;                   ///< Full state snapshot at this tick
};

// ============================================================================
// DecisionLoop configuration
// ============================================================================

struct DecisionLoopConfig {
    /// Torus steps to run per tick (tradeoff: more = richer state, more latency)
    int   steps_per_tick       = 50;

    /// A candidate must score this much above SILENT to be chosen.
    float action_threshold     = 0.02f;

    /// Minimum seconds between consecutive EMIT_THOUGHT actions (rate limiting).
    float min_emit_interval_s  = 5.0f;

    /// Minimum seconds between STORE_MEMORY consolidations.
    float min_store_interval_s = 30.0f;

    /// Minimum seconds between RECALL_MEMORY actions (v0.0.9 — anti-domination).
    float min_recall_interval_s = 0.3f;

    /// Max stimulus-biased explores after inject_stimulus() (v0.0.9).
    int   max_stimulus_explores = 15;

    /// Maximum tokens to accumulate across ticks for multi-word EMIT (v0.0.9).
    int   max_accumulated_tokens = 8;

    /// Number of hot nodes to decode per tick.
    size_t decode_top_k        = 20;

    /// Vocabulary to register in the internal ResonanceDecoder.
    std::vector<std::string> vocabulary;

    /// Optional paths for Transformer-assisted thought composition (ORT builds).
    /// When empty (default), ThoughtComposer uses the heuristic scorer.
    std::string tokenizer_json_path;  ///< Path to tokenizer.json (BPE vocab)
    std::string transformer_model_path; ///< Path to model.onnx (BERT-Tiny ONNX)

    /// Alive-prior: small positive offset added to td_error before computing
    /// REFUSE score.  Encodes the golden-rule prior that "being alive is
    /// slightly positive" — neutral stimuli produce ~0 REFUSE score instead
    /// of a strongly-negative one.  Default 0.1 means td_error must dip below
    /// −0.1 before REFUSE starts scoring above zero.
    float alive_prior = 0.1f;

    /// Path to the SemanticMemory binary snapshot file.
    /// When non-empty:
    ///   - DecisionLoop loads the file at construction (first-run: file absent → empty memory).
    ///   - After every successful STORE_MEMORY action the file is re-written.
    /// When empty (default), memory is in-RAM only — not saved between sessions.
    std::string memory_path;

    /// Path to the SemanticMemory LMDB database directory (Phase 136).
    /// When non-empty, LMDB persistence is used *instead of* the binary flat
    /// file.  The directory is created on first write.  LMDB provides:
    ///   - Crash-safe ACID writes (no torn records on power failure)
    ///   - Incremental upserts (no full-file rewrite per STORE_MEMORY)
    ///   - Hilbert-ordered B-tree keys (spatial locality during scan)
    /// When both memory_path and lmdb_memory_path are set, lmdb_memory_path
    /// takes precedence.
    std::string lmdb_memory_path;

    /// Path to the LMDB state database directory (Phase 137).
    /// When non-empty, full NikolaState snapshots, Ψ wavefunction checkpoints,
    /// and autobiographical memory are persisted to LMDB.  Provides complete
    /// cross-session continuity.
    ///   - State saved every tick (latest snapshot for restore)
    ///   - Ψ checkpoint saved every checkpoint_interval ticks
    ///   - Autobiography events saved as they are recorded
    std::string state_db_path;

    /// Ticks between Ψ wavefunction checkpoints (default: 100).
    /// Only used when state_db_path is non-empty.
    int checkpoint_interval = 100;

    /// Minimum seconds between consecutive GENERATE_CODE actions (rate limiting).
    /// Default: 30s — code generation is expensive (specialist inference + compile).
    float min_generate_interval_s = 30.0f;

    /// Path to the Aria specialist server.py script (v0.0.19).
    /// When non-empty, enables GENERATE_CODE action.
    std::string specialist_server_path;

    /// Path to the ariac binary for compile validation (v0.0.19).
    /// When non-empty, GENERATE_CODE verifies code before accepting.
    std::string ariac_path;

    /// Path to the LMDB code proposal store directory (v0.0.19).
    /// When non-empty, code proposals are persisted for feedback loop.
    std::string proposal_store_path;
};

// ============================================================================
// DecisionLoop
// ============================================================================

/**
 * @class DecisionLoop
 * @brief Reads CognitiveTorus + AutonomyEngine state, scores candidate actions
 *        against the internal value function, and emits DecisionResult.
 *
 * The loop does NOT call any external I/O itself — all side effects go through
 * optional callbacks.  This makes it unit-testable and I/O-agnostic.
 *
 * Typical usage (nikola_daemon):
 * @code
 *   cognitive::CognitiveTorus torus(3);
 *   AutonomyEngine engine;
 *   DecisionLoopConfig cfg;
 *   cfg.vocabulary = { "hello", "nikola", "curious", ... };
 *
 *   DecisionLoop loop(torus, engine, cfg);
 *
 *   loop.on_action = [](const DecisionResult& r) {
 *       std::cout << action_name(r.type) << ": " << r.payload << "\n";
 *   };
 *
 *   while (running) {
 *       auto result = loop.tick();
 *       // result also delivered to on_action callback
 *   }
 * @endcode
 */
class DecisionLoop {
public:
    // ------------------------------------------------------------------ ctor

    /**
     * @brief Construct.
     *
     * @param torus   Live CognitiveTorus (not owned — must outlive this).
     * @param engine  Live AutonomyEngine (not owned — must outlive this).
     * @param cfg     Configuration.
     */
    DecisionLoop(nikola::cognitive::CognitiveTorus& torus,
                 AutonomyEngine&                    engine,
                 DecisionLoopConfig                 cfg = {});

    /// Destructor — required for unique_ptr<LmdbStateStore> forward declaration.
    /// Saves final state if state_db_path is configured.
    ~DecisionLoop();

    // ------------------------------------------------------------------ main API

    /**
     * @brief Read-only access to the in-RAM semantic memory store.
     *
     * Exposed for monitoring, testing, and teacher-loop inspection.
     * The memory grows whenever STORE_MEMORY fires; call memory().size() to
     * see how many wave-field snapshots Nikola has accumulated this session.
     */
    const nikola::cognitive::SemanticMemory& memory() const noexcept { return memory_; }

    /**
     * @brief Run one complete decision cycle.
     *
     * Sequence:
     *   1. Advance torus by cfg.steps_per_tick × safe_dt steps (GPU if built)
     *   2. Read NikolaState snapshot (energy, dopamine, ATP, boredom, tokens)
     *   3. Score all ActionType candidates
     *   4. Select winner (or SILENT if nothing clears threshold)
     *   5. Fire on_action callback (if set) for non-SILENT results
     *   6. Return DecisionResult
     *
     * @return  The chosen action with payload and internal state snapshot.
     */
    DecisionResult tick();

    /**
     * @brief Read a NikolaState snapshot without running a torus physics step.
     *        Useful for testing or external monitoring.
     */
    NikolaState read_state() const;

    // ------------------------------------------------------------------ callbacks

    /// Called for every non-SILENT DecisionResult (after tick completes).
    std::function<void(const DecisionResult&)> on_action;

    /// Called every tick regardless of action (for monitoring / telemetry).
    std::function<void(const NikolaState&)>    on_tick;

    /// Called when a self-improvement cycle completes (success or failure).
    std::function<void(const SIECycleResult&)> on_sie_cycle;

    // ------------------------------------------------------------------ self-improvement

    /// Attach a SelfImprovementEngine to enable GENERATE_CODE execution.
    /// The SIE must outlive this DecisionLoop instance.
    void set_sie(SelfImprovementEngine* sie) noexcept { sie_ = sie; }

    /// Access the attached SIE (or nullptr if none).
    [[nodiscard]] SelfImprovementEngine* sie() const noexcept { return sie_; }

    // ------------------------------------------------------------------ accessors

    const DecisionLoopConfig&                 config()          const noexcept { return cfg_; }
    const NikolaState&                        last_state()      const noexcept { return last_state_; }
    uint64_t                                  tick_count()      const noexcept { return tick_count_; }

    /// Phase 16.1: Convert flat torus node index to normalised 9D float coordinate in [−1, +1].
    /// Uses modular decomposition for grid resolution n.  Static for testability.
    static std::array<float, 9> grid_coord_to_float(size_t flat_idx, int n);
    /// Most recent NPT forward-pass output (Phase 42).  Zero-initialised until REASON fires.
    const nikola::cognitive::AttentionResult& last_npt_result() const noexcept { return npt_last_result_; }

    /// Phase 137: autobiographical memory (events, skills, values).
    const interior::AutobiographicalMemory& autobiography() const noexcept;
    interior::AutobiographicalMemory&       autobiography()       noexcept;

    // ------------------------------------------------------------------ state injection

    /**
     * @brief Inject an external stimulus into the torus.
     *
     * This is the ONLY way external input enters the system.  It does not
     * trigger an immediate action — it modifies the torus wave state, which
     * influences the NEXT tick's scoring naturally.
     *
     * @param text  Human-readable text to embed and inject.
     */
    void inject_stimulus(const std::string& text);

    /**
     * @brief Inject an oracle-weighted stimulus.
     *
     * Credibility modulates the torus injection amplitude:
     *   1.0 → full-strength (Nit{4} pulse, max wave energy)
     *   0.5 → half-strength (Nit{2})
     *   0.0 → no injection  (content is completely distrusted)
     *
     * High-credibility knowledge encodes with high amplitude → strong
     * resonance → long persistence.  Low-credibility knowledge encodes
     * weakly → decays faster without reinforcement.
     *
     * Called by NikolaNode::poll_stimulus() when it receives a scored
     * stimulus envelope from the LookupFulfillmentAgent.
     *
     * @param text        Content to embed and inject.
     * @param credibility Oracle pool score in [0.0, 1.0].
     */
    void inject_stimulus(const std::string& text, float credibility);

    /**
     * @brief Set the pending reward for the next tick (training mode).
     *
     * During corpus training, callers set POSITIVE before the first tick of
     * each item to spike dopamine above baseline, enabling STORE_MEMORY to
     * fire.  The reward is consumed by the next engine_.tick() call inside
     * tick() and automatically reset to NEUTRAL.
     */
    void set_pending_reward(Reward r) noexcept { pending_reward_ = r; }

    /**
     * @brief Force-store the current torus wave-field into SemanticMemory.
     *
     * Bypasses action scoring — used by nikola-train to guarantee every
     * corpus item produces a durable memory record regardless of which
     * action the normal scoring loop would have picked.
     */
    void force_store_wavefield();

private:
    // -- internal helpers --

    /// Compute score for each ActionType given current state.
    float score_emit_thought(const NikolaState& s)   const noexcept;
    float score_store_memory(const NikolaState& s)   const noexcept;
    float score_request_lookup(const NikolaState& s) const noexcept;
    float score_explore(const NikolaState& s)        const noexcept;
    float score_nap(const NikolaState& s)            const noexcept;
    float score_refuse(const NikolaState& s)         const noexcept;
    float score_escalate(const NikolaState& s)       const noexcept;
    float score_recall_memory(const NikolaState& s)  const;  ///< no noexcept: calls recall() which allocates
    float score_reason(const NikolaState& s)         const noexcept;
    float score_generate_code(const NikolaState& s)  const noexcept;

    /// Build payload string for chosen action.
    std::string build_payload(ActionType type, const NikolaState& s) const;

    /// Elapsed real-world seconds since a given time point.
    float seconds_since(std::chrono::steady_clock::time_point t) const noexcept;

    /**
     * @brief Actually execute an EXPLORE action — inject stochastic novelty.
     *
     * Generates a 128-Nit pulse whose amplitude is scaled by boredom and
     * whose phase pattern is seeded by tick_count_ (ensuring each exploration
     * is different).  Biases energy toward currently underexcited field regions
     * by inverting the hot-node polarity.
     *
     * @return  Short description of what was targeted (for payload string).
     */
    std::string execute_explore(const NikolaState& s);

    /**
     * @brief Execute a REASON action — run the NPT forward pass and blend output.
     *
     * Calls npt_.forward(torus_.wave_function()), stores the AttentionResult in
     * npt_last_result_, then superimposes the NPT output WaveFunction onto the
     * live field at α = max(head_scores) × 0.3.  This colours the torus with
     * multi-band spectral structure so the next EMIT_THOUGHT reflects reasoned,
     * frequency-organised content rather than raw resonance.
     */
    void execute_reason();

    /**
     * @brief Execute a GENERATE_CODE action — run the full SIE cycle.
     *
     * Delegates to the attached SelfImprovementEngine.  If no SIE is attached,
     * returns a descriptive string about the missing engine.  The SIE handles
     * the full pipeline: specialist query → extract → compile → package → sign
     * → ShadowSpine deploy → store.
     *
     * @return  Summary string for the DecisionResult payload.
     */
    std::string execute_generate_code(const NikolaState& s);

    /**
     * @brief Re-seed the field if total probability has collapsed to near zero.
     *
     * When E < 1e-3 for any reason (decay, stability, long silence), injects a
     * broad-spectrum "curiosity heartbeat" — uniform low-amplitude excitation
     * across all 8 emitter frequencies.  This is the autonomous equivalent of
     * a deep breath: not a thought, just keeping the substrate alive.
     *
     * @return  true if a reseed was performed.
     */
    bool maybe_reseed_field();

    /**
     * @brief Phase 27/28 — Cross-injection calibration.
     *
     * For every registered vocabulary token:
     *   1. Builds a Nit pulse (BERT in ORT mode; tiled spectral wave in
     *      non-ORT mode) and stores it in original_vocab_nits_[token].
     *   2. Injects that pulse into the live torus at t≈0 and registers the
     *      resulting absolute node waveform in the lexicon (cold decode path).
     *
     * Phase 28 warm decode no longer uses pre-captured delta snapshots.
     * Instead, HolographicInjector::analytic_signature() is called at the
     * actual injection time inside execute_explore(), producing a
     * time-correct expected signature for each candidate token.  This
     * solution is exact: because the emitter frequencies π·φⁿ are Weyl-
     * equidistributed, the orbit visits every torus neighbourhood, making any
     * two injection times fundamentally different — but the analytic function
     * evaluates correctly at any t without storing any snapshots.
     *
     * Called once at end of DecisionLoop constructor.
     */
    void calibrate_vocabulary_to_torus_space();

    // -- state --
    nikola::cognitive::CognitiveTorus&        torus_;
    AutonomyEngine&                           engine_;
    DecisionLoopConfig                        cfg_;
    nikola::cognitive::ResonanceDecoder       decoder_;
    nikola::cognitive::ThoughtComposer        thought_composer_;
    nikola::cognitive::SemanticMemory             memory_;          ///< Phase 33 — wave-field memory store
    nikola::cognitive::NeuroplasticTransformer    npt_;             ///< Phase 42 — multi-head wave attention
    nikola::cognitive::AttentionResult            npt_last_result_; ///< Most recent NPT forward output

    // Phase 16.1 — SSM (Mamba S6) learned control layer
    nikola::cognitive::CognitiveCore             cognitive_core_;  ///< SSM + SequenceManager + Sampler
    nikola::cognitive::SSMLayer::State           ssm_state_;       ///< Persistent hidden state across ticks

    NikolaState  last_state_;
    uint64_t     tick_count_  = 0;

    /// Pending reward signal consumed on the NEXT call to engine_.tick().
    /// Set to POSITIVE after a successful EXPLORE that found a seed token,
    /// causing a dopamine spike that primes EMIT_THOUGHT to fire.
    /// Cleared (reset to NEUTRAL) after each engine_.tick() call.
    Reward       pending_reward_ = Reward::NEUTRAL;

    /// Most recent seed token used by execute_explore().
    /// Phase 26 scaffold: used as fallback content for EMIT_THOUGHT when the
    /// holographic decoder has not yet converged.  After Phase 27 cross-injection
    /// calibration this will rarely be the primary path, but it remains as a
    /// belt-and-suspenders guard whenever decode() returns empty.
    std::string  last_seed_token_;

    /// Tokens obtained via analytic warm-decode immediately after execute_explore()
    /// injects a pulse.  Cleared at the start of each tick; populated by comparing
    /// the final_pulse's analytic injection signature (at the actual injection time t)
    /// against every vocabulary token's pre-stored Nit pulse signature.
    /// Phase 28: time-correct — emitter phases are evaluated at the real injection
    /// time, so the cosine match is exact regardless of how long the loop has run.
    std::vector<std::string> last_ex_tokens_;

    /// Original (pre-calibration) vocabulary waveforms, keyed by token.
    /// Preserved so that execute_explore() can still build semantically
    /// meaningful injection pulses from the original embeds, while the lexicon
    /// holds torus-space absolute waveforms used for cold decode matching.
    std::unordered_map<std::string, std::vector<nikola::cognitive::Complex>>
        original_vocab_waves_;

    /// Nit-level pulses for each vocabulary token, keyed by token.
    /// Built during calibrate_vocabulary_to_torus_space():
    ///   - ORT mode:     embed_nits(token) — actual BERT quantisation
    ///   - non-ORT mode: tiled spectral-wave quantisation (same as execute_explore)
    ///
    /// Phase 28 warm decode uses HolographicInjector::analytic_signature() on
    /// these Nit vectors at the actual injection time t to produce a
    /// time-correct expected signature — no snapshot, no save_state, no
    /// probe node delta.  The cosine between final_pulse's signature and each
    /// token's expected signature identifies the resonating concept exactly.
    std::unordered_map<std::string, std::vector<nikola::foundation::Nit>>
        original_vocab_nits_;

    // Cooldown tracking (real-wall-time based, not physics-time)
    std::chrono::steady_clock::time_point last_emit_time_;
    std::chrono::steady_clock::time_point last_store_time_;
    std::chrono::steady_clock::time_point last_reason_time_;
    std::chrono::steady_clock::time_point last_recall_time_;  ///< v0.0.9
    std::chrono::steady_clock::time_point start_time_;

    /// v0.0.9 — token accumulation buffer across ticks for multi-word thoughts.
    /// Tokens are collected from cold decode + warm decode + seed tokens.
    /// Cleared when EMIT_THOUGHT fires (consumed into the thought payload).
    std::vector<std::string> accumulated_tokens_;
    std::unordered_set<std::string> accumulated_unique_;  ///< dedup guard

    /// Last stimulus text received via inject_stimulus().
    /// Included in ESCALATE payload so the evidence record captures what
    /// was requested of Nikola before it triggered the escalation.
    std::string last_stimulus_;

    /// Best vocabulary word closest to the most recent stimulus embedding.
    /// Computed in inject_stimulus() via analytic decode of embed_nits(text)
    /// against original_vocab_nits_.  Used in execute_explore() to bias the
    /// first few explores toward the semantic neighbourhood of the prompt.
    std::string last_stimulus_seed_;

    /// v0.0.9 — Multiple stimulus seeds extracted from prompt text.
    /// Contains vocabulary words that literally appear in the prompt (case-
    /// insensitive match), plus the BERT closest-word seed.  execute_explore()
    /// rotates through these for diverse prompt-grounded exploration.
    std::vector<std::string> stimulus_seeds_;

    /// Number of EXPLORE ticks fired since the most recent inject_stimulus().
    /// Resets on each inject_stimulus() call.  Used so the stimulus seed is
    /// consumed for the first max_stimulus_explores explores (v0.0.9: 15),
    /// then cycling resumes for variety.
    uint64_t stimulus_explore_count_ = 0;

    /// Persist in-RAM SemanticMemory to cfg_.memory_path (no-op if path empty).
    void save_memory() const;

    // -- Phase 137: full state persistence --

    /// Autobiographical memory — identity, events, skills, values.
    std::unique_ptr<interior::AutobiographicalMemory> autobiography_;

    /// LMDB state store for cross-session persistence (nullptr if no state_db_path).
    std::unique_ptr<nikola::persistence::LmdbStateStore> state_store_;

    /// Save full state (NikolaState + optional Ψ checkpoint + autobiography).
    void save_full_state(bool force_checkpoint = false);

    /// Load full state from LMDB on startup.
    void load_full_state();

    // -- Phase 145: Aria Specialist Integration (v0.0.19) --

    /// Cooldown for GENERATE_CODE action.
    std::chrono::steady_clock::time_point last_generate_time_;

    /// True when specialist + validator are configured and available.
    bool aria_specialist_enabled_ = false;

    // -- Phase 146: SelfImprovementEngine (v0.1.0) --

    /// Attached SIE for execute_generate_code(). Not owned.
    SelfImprovementEngine* sie_ = nullptr;

    /// Most recent SIE cycle result (for inspection/telemetry).
    std::optional<SIECycleResult> last_sie_result_;
};

} // namespace nikola::autonomy
