#pragma once
/**
 * @file cognitive/thought_composer.hpp
 * @brief ThoughtComposer — the "prefrontal voice" layer.
 *
 * Architecture position:
 *
 *   ResonanceDecoder (what is resonating)
 *         ↓  tokens + NikolaState drives
 *   ThoughtComposer  (how to express it)
 *         ↓  coherent thought string
 *   EMIT_THOUGHT payload → output router
 *
 * The ThoughtComposer answers a simple question at each EMIT_THOUGHT event:
 *   "Given what the field is resonating on AND how I'm feeling right now,
 *    which phrasing best expresses this thought?"
 *
 * It does NOT generate novel language.  It selects and fills one of 8 built-in
 * templates according to which one is most consistent with the current internal
 * state.  The torus decides WHAT to think about; the Transformer (or the
 * scoring heuristic in no-ORT mode) decides HOW to frame it.
 *
 * ───────────────────────────────────────────────────────────────────────────
 * Mode A — no-ORT (always available):
 *
 *   Each template is assigned a score derived from state drives:
 *     "drawn to {content}"         → dopamine × 1.0
 *     "wondering about {content}"  → boredom  × 1.0
 *     "feels important"            → (2 - entropy) / 2   when entropy < 2
 *     "feels off about {content}"  → -td_error × 2       when punished
 *     "{content} hard to hold"     → (entropy - 4) × 0.5 when entropy > 4
 *     "want to understand {content}"→ (1 - atp) × 0.8
 *     "resonating with {content}"  → neutral baseline 0.3
 *     "there is something about"   → fallback 0.4
 *
 *   Highest-scoring template wins.  Deterministic, O(1), no allocations beyond
 *   the output string.
 *
 * Mode B — ORT (NIKOLA_HAS_ORT, opt-in via model paths in constructor):
 *
 *   1. Build a "state descriptor" sentence from dominant drives
 *      (e.g. "excited curious" for high dopamine/boredom).
 *   2. Encode via BPETokenizer → run TinyTransformer → 128-dim state_emb.
 *   3. For each of the 8 templates filled with {content}:
 *        encode → run TinyTransformer → templ_emb
 *   4. Select template with max cosine_similarity(state_emb, templ_emb).
 *   5. Capitalise first letter.  Return.
 *
 *   This grounds template selection in the actual semantic neighbourhood of
 *   the current field state, not just scalar heuristics.
 *
 * ───────────────────────────────────────────────────────────────────────────
 * Template placeholders:
 *   {content} is replaced by the top-3 decoded tokens joined naturally:
 *     1 token  → "hello"
 *     2 tokens → "hello and world"
 *     3 tokens → "hello, world and curious"
 *     0 tokens → "something"
 *
 * Thread safety: compose() is const and re-entrant; register-at-ctor only.
 *
 * Phase: NIK-TC-01 (ThoughtComposer, Phase 24)
 */

#include <array>
#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#ifdef NIKOLA_HAS_ORT
#  include <nikola/cognitive/bpe_tokenizer.hpp>
#  include <nikola/cognitive/tiny_transformer.hpp>
#endif

namespace nikola::cognitive {

// ============================================================================
// ThoughtContext — what the DecisionLoop hands to the composer
// ============================================================================

/**
 * @brief All the state information ThoughtComposer needs to pick a phrasing.
 *
 * Intentionally a plain struct — no dependency on AutonomyEngine internals.
 * Populated by DecisionLoop::build_payload() from the current NikolaState.
 */
struct ThoughtContext {
    std::vector<std::string> tokens;  ///< Decoded resonance tokens (top-k)
    float dopamine = 0.f;             ///< DopamineSystem level ∈ [0, 1]
    float boredom  = 0.f;             ///< BoredomRegulator level ∈ [0, 1]
    float atp      = 0.f;             ///< Metabolic ATP level ∈ [0, 1]
    float td_error = 0.f;             ///< Last TD error (negative = punishment)
    float entropy  = 0.f;             ///< Shannon entropy of torus field
};

// ============================================================================
// ThoughtComposer
// ============================================================================

/**
 * @brief Selects and fills a thought-expression template given field content
 *        and internal state drives.
 *
 * The composer is lightweight by design — one instance lives in DecisionLoop
 * and is called only on EMIT_THOUGHT events (min 5-second cooldown).
 */
class ThoughtComposer {
public:
    // ------------------------------------------------------------------
    // Built-in templates — the 8 candidates
    //
    // Each contains exactly one {content} placeholder (or none for pure-affect
    // expressions that reference the content implicitly).
    // ------------------------------------------------------------------
    static constexpr size_t TEMPLATE_COUNT = 8;

    // Template indices (for testing / introspection)
    enum class Template : uint8_t {
        NEUTRAL      = 0,  ///< "resonating with {content}"
        DRAWN        = 1,  ///< "drawn to {content}"
        WONDERING    = 2,  ///< "wondering about {content}"
        IMPORTANT    = 3,  ///< "{content} feels important"
        FEELS_OFF    = 4,  ///< "something feels off about {content}"
        HARD_TO_HOLD = 5,  ///< "{content} is hard to hold"
        UNDERSTAND   = 6,  ///< "I want to understand {content} better"
        SOMETHING    = 7,  ///< "there is something about {content}"
    };

    // ------------------------------------------------------------------ ctor

    /**
     * @brief Construct with no-ORT template scoring only.
     *
     * Always available regardless of build flags.
     */
    ThoughtComposer() = default;

#ifdef NIKOLA_HAS_ORT
    /**
     * @brief Construct with Transformer-assisted template selection (ORT mode).
     *
     * If either path is empty or files are missing, falls back to no-ORT mode
     * silently.
     *
     * @param tokenizer_json_path  Path to tokenizer.json (HuggingFace BPE vocab).
     * @param model_path           Path to model.onnx (BERT-Tiny ONNX export).
     */
    ThoughtComposer(const std::string& tokenizer_json_path,
                    const std::string& model_path);
#endif

    // ------------------------------------------------------------------ API

    /**
     * @brief Select and fill the best-matching template for the given context.
     *
     * @param ctx  ThoughtContext with decoded tokens and state drives.
     * @return     A non-empty human-readable thought string.
     */
    std::string compose(const ThoughtContext& ctx) const;

    /**
     * @brief Select the best template index for this context (no filling).
     *
     * Useful for testing and introspection — see which template would win
     * without also needing to decode the template string.
     */
    Template select_template(const ThoughtContext& ctx) const;

    /// Returns true if operating in Transformer-assisted mode (ORT build + loaded).
    bool has_transformer() const noexcept;

    // ------------------------------------------------------------------ static helpers

    /**
     * @brief Build the {content} substitution string from decoded tokens.
     *
     * Takes the first `max_tokens` tokens (default 3) and joins them as:
     *   1 → "word"
     *   2 → "word and word"
     *   3 → "word, word and word"
     *   0 → "something"
     */
    static std::string build_content(const std::vector<std::string>& tokens,
                                     int max_tokens = 3);

    /**
     * @brief Fill {content} in a template string.
     */
    static std::string fill_template(const char* tmpl,
                                     const std::string& content);

    /**
     * @brief Score all templates against state drives (no-ORT heuristic).
     *
     * Returns an array of 8 non-negative scores.  Highest index returned by
     * select_template().  Exposed for unit testing.
     */
    static std::array<float, TEMPLATE_COUNT>
    score_templates(const ThoughtContext& ctx) noexcept;

private:
    // Template strings (substitution marker: {content})
    static const std::array<const char*, TEMPLATE_COUNT> TEMPLATE_STRINGS;

#ifdef NIKOLA_HAS_ORT
    // Transformer-mode members (null when ORT not available or paths not provided)
    std::unique_ptr<BPETokenizer>     tokenizer_;
    std::unique_ptr<TinyTransformer>  transformer_;

    // Build a state-descriptor sentence from dominant drives
    static std::string state_descriptor(const ThoughtContext& ctx);

    // Cosine similarity of two same-length float vectors
    static float cosine_similarity(const std::vector<float>& a,
                                   const std::vector<float>& b) noexcept;

    // Transformer-assisted template selection
    Template select_template_ort(const ThoughtContext& ctx) const;
#endif

    // Capitalise first character of a string (in-place helper)
    static void capitalise(std::string& s) noexcept;

    // v0.0.9 — template diversity: penalise repeating the same template
    mutable Template last_template_ = Template::SOMETHING;
};

} // namespace nikola::cognitive
