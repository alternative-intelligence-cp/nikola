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
#include <string>
#include <unordered_map>
#include <vector>

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
    float action_threshold     = 0.05f;

    /// Minimum seconds between consecutive EMIT_THOUGHT actions (rate limiting).
    float min_emit_interval_s  = 5.0f;

    /// Minimum seconds between STORE_MEMORY consolidations.
    float min_store_interval_s = 30.0f;

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

    // ------------------------------------------------------------------ accessors

    const DecisionLoopConfig&                 config()          const noexcept { return cfg_; }
    const NikolaState&                        last_state()      const noexcept { return last_state_; }
    uint64_t                                  tick_count()      const noexcept { return tick_count_; }
    /// Most recent NPT forward-pass output (Phase 42).  Zero-initialised until REASON fires.
    const nikola::cognitive::AttentionResult& last_npt_result() const noexcept { return npt_last_result_; }

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

    NikolaState  last_state_;
    uint64_t     tick_count_  = 0;

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
    std::chrono::steady_clock::time_point start_time_;

    /// Last stimulus text received via inject_stimulus().
    /// Included in ESCALATE payload so the evidence record captures what
    /// was requested of Nikola before it triggered the escalation.
    std::string last_stimulus_;

    /// Persist in-RAM SemanticMemory to cfg_.memory_path (no-op if path empty).
    void save_memory() const;
};

} // namespace nikola::autonomy
