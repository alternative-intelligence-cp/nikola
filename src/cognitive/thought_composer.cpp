/**
 * @file src/cognitive/thought_composer.cpp
 * @brief ThoughtComposer implementation.
 *
 * See include/nikola/cognitive/thought_composer.hpp for full design rationale.
 *
 * ── No-ORT path ──────────────────────────────────────────────────────────────
 *   score_templates() maps state drives → template scores.
 *   compose() picks the winner, builds content, fills, capitalises, returns.
 *
 * ── ORT path ─────────────────────────────────────────────────────────────────
 *   select_template_ort() runs TinyTransformer on a state descriptor AND on
 *   each candidate template string, then picks by max cosine similarity.
 *   Falls back to no-ORT if transformer_ is null (bad paths, missing files).
 *
 * Phase: NIK-TC-01 (ThoughtComposer, Phase 24)
 */

#include <nikola/cognitive/thought_composer.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace nikola::cognitive {

// ============================================================================
// Static data
// ============================================================================

// The 8 candidate template strings.  Indexed by ThoughtComposer::Template.
const std::array<const char*, ThoughtComposer::TEMPLATE_COUNT>
ThoughtComposer::TEMPLATE_STRINGS = {{
    "resonating with {content}",            // NEUTRAL
    "drawn to {content}",                   // DRAWN
    "wondering about {content}",            // WONDERING
    "{content} feels important",            // IMPORTANT
    "something feels off about {content}",  // FEELS_OFF
    "{content} is hard to hold",            // HARD_TO_HOLD
    "I want to understand {content} better",// UNDERSTAND
    "there is something about {content}",   // SOMETHING (fallback)
}};

// ============================================================================
// Constructors
// ============================================================================

#ifdef NIKOLA_HAS_ORT

ThoughtComposer::ThoughtComposer(const std::string& tokenizer_json_path,
                                 const std::string& model_path)
{
    if (tokenizer_json_path.empty() || model_path.empty()) return;
    try {
        tokenizer_   = std::make_unique<BPETokenizer>(tokenizer_json_path);
        transformer_ = std::make_unique<TinyTransformer>(model_path);
        std::cout << "[ThoughtComposer] ORT mode: transformer-assisted template selection\n";
    } catch (const std::exception& e) {
        tokenizer_.reset();
        transformer_.reset();
        std::cerr << "[ThoughtComposer] ORT init failed (" << e.what()
                  << ") — falling back to heuristic mode\n";
    }
}

#endif // NIKOLA_HAS_ORT

// ============================================================================
// has_transformer
// ============================================================================

bool ThoughtComposer::has_transformer() const noexcept
{
#ifdef NIKOLA_HAS_ORT
    return transformer_ != nullptr;
#else
    return false;
#endif
}

// ============================================================================
// Static helpers
// ============================================================================

std::string ThoughtComposer::build_content(const std::vector<std::string>& tokens,
                                           int max_tokens)
{
    if (tokens.empty()) return "something";

    const int n = std::min(static_cast<int>(tokens.size()), max_tokens);

    if (n == 1) return tokens[0];

    if (n == 2) return tokens[0] + " and " + tokens[1];

    // 3+: "a, b and c"
    std::string result;
    for (int i = 0; i < n - 1; ++i) {
        result += tokens[i];
        if (i < n - 2) result += ", ";
    }
    result += " and ";
    result += tokens[n - 1];
    return result;
}

std::string ThoughtComposer::fill_template(const char* tmpl,
                                           const std::string& content)
{
    std::string result = tmpl;
    const auto pos = result.find("{content}");
    if (pos != std::string::npos) {
        result.replace(pos, 9 /* len("{content}") */, content);
    }
    return result;
}

void ThoughtComposer::capitalise(std::string& s) noexcept
{
    if (!s.empty() && s[0] >= 'a' && s[0] <= 'z') {
        s[0] = static_cast<char>(s[0] - 'a' + 'A');
    }
}

// ============================================================================
// score_templates — no-ORT heuristic
// ============================================================================

std::array<float, ThoughtComposer::TEMPLATE_COUNT>
ThoughtComposer::score_templates(const ThoughtContext& ctx) noexcept
{
    std::array<float, TEMPLATE_COUNT> scores{};

    // NEUTRAL: resonating with {content} — baseline
    scores[static_cast<size_t>(Template::NEUTRAL)] = 0.30f;

    // DRAWN: drawn to {content} — dopamine-driven excitement
    scores[static_cast<size_t>(Template::DRAWN)] =
        std::max(0.f, ctx.dopamine);

    // WONDERING: wondering about {content} — boredom/curiosity seeking
    scores[static_cast<size_t>(Template::WONDERING)] =
        std::max(0.f, ctx.boredom);

    // IMPORTANT: {content} feels important — high focus (low entropy)
    scores[static_cast<size_t>(Template::IMPORTANT)] =
        ctx.entropy < 2.0f ? (2.0f - ctx.entropy) / 2.0f : 0.f;

    // FEELS_OFF: something feels off about {content} — punishment signal
    // v0.0.9: Raised threshold from 0 to -0.2 because natural field energy
    // decay produces td_error of -0.05 to -0.15 continuously.  Only genuine
    // punishment signals (td < -0.2) should trigger FEELS_OFF.
    scores[static_cast<size_t>(Template::FEELS_OFF)] =
        ctx.td_error < -0.2f ? std::min(1.f, (-ctx.td_error - 0.2f) * 2.0f) : 0.f;

    // HARD_TO_HOLD: {content} is hard to hold — high entropy / scattered field
    scores[static_cast<size_t>(Template::HARD_TO_HOLD)] =
        ctx.entropy > 4.0f ? std::min(1.f, (ctx.entropy - 4.0f) * 0.5f) : 0.f;

    // UNDERSTAND: I want to understand {content} better — low energy / seeking
    scores[static_cast<size_t>(Template::UNDERSTAND)] =
        std::max(0.f, (1.0f - ctx.atp) * 0.8f);

    // SOMETHING: there is something about {content} — general fallback
    scores[static_cast<size_t>(Template::SOMETHING)] = 0.40f;

    return scores;
}

// ============================================================================
// select_template — picks winner from heuristic scores (or ORT)
// ============================================================================

ThoughtComposer::Template
ThoughtComposer::select_template(const ThoughtContext& ctx) const
{
#ifdef NIKOLA_HAS_ORT
    if (transformer_ && tokenizer_) {
        try {
            return select_template_ort(ctx);
        } catch (...) {
            // Fall through to heuristic on any ORT error
        }
    }
#endif
    // No-ORT: pick template with highest score, penalise repeats (v0.0.9)
    auto scores = score_templates(ctx);
    scores[static_cast<size_t>(last_template_)] *= 0.5f; // diversity penalty
    const auto max_it  = std::max_element(scores.begin(), scores.end());
    return static_cast<Template>(std::distance(scores.begin(), max_it));
}

// ============================================================================
// compose
// ============================================================================

std::string ThoughtComposer::compose(const ThoughtContext& ctx) const
{
    const Template tmpl     = select_template(ctx);
    last_template_ = tmpl;  // v0.0.9: track for diversity penalty
    const std::string content = build_content(ctx.tokens);
    const size_t idx          = static_cast<size_t>(tmpl);

    std::string result = fill_template(TEMPLATE_STRINGS[idx], content);
    capitalise(result);
    return result;
}

// ============================================================================
// ORT-specific helpers
// ============================================================================

#ifdef NIKOLA_HAS_ORT

std::string ThoughtComposer::state_descriptor(const ThoughtContext& ctx)
{
    // Build a short phrase describing dominant drives.
    // The TinyTransformer will embed this to get the "mood vector".

    // v0.0.9: raised threshold from -0.2 to -0.3 so normal field decay
    // (-0.05 to -0.15) doesn't produce the FEELS_OFF descriptor.
    if (ctx.td_error < -0.3f)
        return "concerned unsettled something wrong";
    if (ctx.entropy > 4.5f)
        return "scattered confused many directions";
    if (ctx.dopamine > 0.7f && ctx.boredom > 0.6f)
        return "excited curious interested wondering";
    if (ctx.dopamine > 0.7f)
        return "excited engaged drawn interested";
    if (ctx.boredom > 0.7f)
        return "curious wondering searching exploring";
    if (ctx.atp < 0.3f)
        return "tired uncertain seeking rest";
    if (ctx.entropy < 1.5f)
        return "focused clear important certain";
    return "thinking considering quiet";
}

float ThoughtComposer::cosine_similarity(const std::vector<float>& a,
                                         const std::vector<float>& b) noexcept
{
    if (a.size() != b.size() || a.empty()) return 0.f;

    float dot = 0.f, na = 0.f, nb = 0.f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    const float denom = std::sqrt(na) * std::sqrt(nb);
    return denom > 1e-8f ? dot / denom : 0.f;
}

ThoughtComposer::Template
ThoughtComposer::select_template_ort(const ThoughtContext& ctx) const
{
    // 1. Build state descriptor → encode → embed
    const std::string desc = state_descriptor(ctx);
    const auto desc_ids    = tokenizer_->encode(desc);
    const auto state_emb   = transformer_->forward(desc_ids);

    // 2. Build content string (for filling templates)
    const std::string content = build_content(ctx.tokens);

    // 3. For each template: fill, encode, embed, compute cosine similarity
    float best_sim = -2.f;
    size_t best_idx = static_cast<size_t>(Template::SOMETHING);

    for (size_t i = 0; i < TEMPLATE_COUNT; ++i) {
        const std::string candidate = fill_template(TEMPLATE_STRINGS[i], content);
        const auto cand_ids = tokenizer_->encode(candidate);
        if (cand_ids.empty()) continue;

        const auto cand_emb = transformer_->forward(cand_ids);
        float sim = cosine_similarity(state_emb, cand_emb);

        // v0.0.9: diversity penalty — discourage repeating the same template
        if (i == static_cast<size_t>(last_template_)) sim -= 0.15f;

        if (sim > best_sim) {
            best_sim = sim;
            best_idx = i;
        }
    }

    return static_cast<Template>(best_idx);
}

#endif // NIKOLA_HAS_ORT

} // namespace nikola::cognitive
