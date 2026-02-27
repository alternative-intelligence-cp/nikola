/**
 * @file internal_dialogue.cpp
 * @brief Phase 122 -- InternalDialogue implementation
 */

#include <nikola/interior/internal_dialogue.hpp>

#include <algorithm>
#include <cctype>
#include <numeric>
#include <set>
#include <sstream>
#include <iomanip>

namespace nikola::interior {

// ============================================================================
// ReasoningChain helpers
// ============================================================================

double ReasoningChain::mean_confidence() const noexcept {
    if (thoughts.empty()) return 0.0;
    double sum = 0.0;
    for (const auto& t : thoughts) sum += t.confidence;
    return sum / static_cast<double>(thoughts.size());
}

double ReasoningChain::peak_confidence() const noexcept {
    if (thoughts.empty()) return 0.0;
    double peak = 0.0;
    for (const auto& t : thoughts)
        if (t.confidence > peak) peak = t.confidence;
    return peak;
}

// ============================================================================
// Internal helpers
// ============================================================================

namespace {

// Tokenise a string: lowercase, split on non-alpha chars.
std::set<std::string> tokenise(const std::string& s) noexcept {
    std::set<std::string> tokens;
    std::string tok;
    for (char c : s) {
        if (std::isalpha(static_cast<unsigned char>(c))) {
            tok += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        } else {
            if (!tok.empty()) { tokens.insert(tok); tok.clear(); }
        }
    }
    if (!tok.empty()) tokens.insert(tok);
    return tokens;
}

// Negation markers (must be followed by a space in the haystack).
static const std::vector<std::string> NEGATION_MARKERS = {
    "not ", "no ", "never ", "cannot ", "can't ", "isn't ",
    "don't ", "doesn't ", "won't ", "isn't ", "aren't ",
    "wasn't ", "weren't ", "hasn't ", "haven't "
};

// lower-case a string in-place
std::string lower(std::string s) {
    for (char& c : s)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

} // anonymous namespace

// ============================================================================
// Pure static helpers
// ============================================================================

double InternalDialogue::word_overlap(const std::string& a,
                                      const std::string& b) noexcept {
    if (a.empty() && b.empty()) return 1.0;
    if (a.empty() || b.empty()) return 0.0;
    auto ta = tokenise(a);
    auto tb = tokenise(b);
    if (ta.empty() && tb.empty()) return 1.0;
    if (ta.empty() || tb.empty()) return 0.0;

    size_t intersection = 0;
    for (const auto& w : ta)
        if (tb.count(w)) ++intersection;

    size_t union_size = ta.size() + tb.size() - intersection;
    if (union_size == 0) return 0.0;
    return static_cast<double>(intersection) / static_cast<double>(union_size);
}

bool InternalDialogue::contains_negation(const std::string& a,
                                         const std::string& b) noexcept {
    // Check whether b negates something from a, or a negates something from b.
    // Strategy: if the tokens of a have significant overlap with b, and b
    // contains a negation marker, treat it as a contradiction candidate.
    auto check = [](const std::string& src, const std::string& tgt) -> bool {
        std::string lt = lower(tgt) + " ";  // append space so marker "not " matches at end
        for (const auto& marker : NEGATION_MARKERS) {
            if (lt.find(marker) != std::string::npos) {
                // tgt contains a negation -- check if src has overlapping content
                // beyond stop words
                auto ts = tokenise(src);
                auto tt = tokenise(tgt);
                // Remove negation words themselves
                static const std::set<std::string> stop_neg = {
                    "not", "no", "never", "cannot", "cant",
                    "isnt", "dont", "doesnt", "wont", "arent",
                    "wasnt", "werent", "hasnt", "havent"
                };
                size_t overlap = 0;
                for (const auto& w : ts)
                    if (!stop_neg.count(w) && tt.count(w)) ++overlap;
                if (overlap >= 1) return true;
            }
        }
        return false;
    };
    return check(a, b) || check(b, a);
}

std::vector<std::string> InternalDialogue::generate_socratic_questions(
    const std::string& assumption)
{
    std::vector<std::string> qs;
    if (assumption.empty()) return qs;
    qs.reserve(5);
    qs.push_back("What is the evidence that \"" + assumption + "\" is true?");
    qs.push_back("What would have to be true for \"" + assumption + "\" to be false?");
    qs.push_back("Why do I believe \"" + assumption + "\" rather than its opposite?");
    qs.push_back("How would I test whether \"" + assumption + "\" holds in this case?");
    qs.push_back("Who or what could challenge \"" + assumption + "\" and how would I respond?");
    return qs;
}

// ============================================================================
// Constructor
// ============================================================================

InternalDialogue::InternalDialogue() noexcept
    : has_active_(false)
    , chain_id_counter_(0)
    , tick_counter_(0)
    , circular_detections_(0)
{}

// ============================================================================
// Chain lifecycle
// ============================================================================

uint64_t InternalDialogue::start_chain(const std::string& problem) {
    if (has_active_) {
        // Auto-conclude the current chain silently
        conclude_chain("(auto-concluded)", current_.mean_confidence());
    }
    current_ = ReasoningChain{};
    current_.chain_id     = next_chain();
    current_.problem      = problem;
    current_.started_tick = next_tick();
    has_active_           = true;
    return current_.chain_id;
}

void InternalDialogue::think(const std::string& text,
                             double             confidence,
                             const std::string& reasoning_type,
                             const NikolaState* state)
{
    if (!has_active_) {
        start_chain("<unnamed>");
    }
    ThoughtEntry e;
    e.text           = text;
    e.tick           = next_tick();
    e.confidence     = confidence < 0.0 ? 0.0 : (confidence > 1.0 ? 1.0 : confidence);
    e.reasoning_type = reasoning_type;
    if (state) {
        e.dopamine_context = static_cast<double>(state->dopamine);
        e.entropy_context  = static_cast<double>(state->entropy);
        e.atp_context      = static_cast<double>(state->atp);
    }
    current_.thoughts.push_back(std::move(e));
}

void InternalDialogue::conclude_chain(const std::string& conclusion,
                                      double             confidence) {
    if (!has_active_) return;

    current_.conclusion            = conclusion;
    current_.conclusion_confidence =
        confidence > 0.0 ? confidence : current_.mean_confidence();
    current_.concluded_tick = next_tick();

    past_.push_back(std::move(current_));
    current_    = ReasoningChain{};
    has_active_ = false;
}

// ============================================================================
// Current chain accessors
// ============================================================================

bool InternalDialogue::has_active_chain() const noexcept {
    return has_active_;
}

const ReasoningChain& InternalDialogue::current_chain() const noexcept {
    return current_;
}

double InternalDialogue::chain_confidence() const noexcept {
    return has_active_ ? current_.mean_confidence() : 0.0;
}

size_t InternalDialogue::current_length() const noexcept {
    return has_active_ ? current_.thoughts.size() : 0;
}

// ============================================================================
// Introspective analysis
// ============================================================================

bool InternalDialogue::detect_circular_reasoning() const {
    const auto& thoughts = current_.thoughts;
    for (size_t i = 0; i < thoughts.size(); ++i) {
        for (size_t j = i + 1; j < thoughts.size(); ++j) {
            if (word_overlap(thoughts[i].text, thoughts[j].text)
                    >= DIALOGUE_CIRCULAR_THRESHOLD) {
                return true;
            }
        }
    }
    return false;
}

std::vector<std::pair<size_t, size_t>>
InternalDialogue::detect_contradictions() const {
    std::vector<std::pair<size_t, size_t>> pairs;
    const auto& thoughts = current_.thoughts;
    for (size_t i = 0; i < thoughts.size(); ++i) {
        for (size_t j = i + 1; j < thoughts.size(); ++j) {
            double ov = word_overlap(thoughts[i].text, thoughts[j].text);
            if (ov >= DIALOGUE_CONTRADICTION_OVERLAP &&
                contains_negation(thoughts[i].text, thoughts[j].text))
            {
                pairs.emplace_back(i, j);
            }
        }
    }
    return pairs;
}

std::string InternalDialogue::synthesize_conclusion() const {
    if (!has_active_ || current_.thoughts.empty()) return "";

    // Find the thought with highest confidence
    size_t best = 0;
    for (size_t i = 1; i < current_.thoughts.size(); ++i) {
        if (current_.thoughts[i].confidence >
            current_.thoughts[best].confidence)
            best = i;
    }
    return "Synthesis: " + current_.thoughts[best].text;
}

std::vector<std::string> InternalDialogue::question_assumption(
    const std::string& assumption) const
{
    return generate_socratic_questions(assumption);
}

std::string InternalDialogue::explain_reasoning() const {
    if (!has_active_ || current_.thoughts.empty())
        return "(no active reasoning chain)";

    std::ostringstream oss;
    oss << "Problem: " << current_.problem << "\n";
    for (size_t i = 0; i < current_.thoughts.size(); ++i) {
        const auto& t = current_.thoughts[i];
        oss << "  " << (i + 1) << ". [" << t.reasoning_type << ", "
            << std::fixed << std::setprecision(2) << t.confidence << "] "
            << t.text << "\n";
    }
    if (!current_.thoughts.empty()) {
        oss << "Mean confidence: "
            << std::fixed << std::setprecision(2)
            << current_.mean_confidence();
    }
    return oss.str();
}

// ============================================================================
// Recall
// ============================================================================

std::vector<const ReasoningChain*>
InternalDialogue::recall_similar(const std::string& query,
                                  size_t             max_results) const
{
    struct Scored { const ReasoningChain* chain; double score; };
    std::vector<Scored> scored;
    scored.reserve(past_.size());

    for (const auto& c : past_) {
        double score = word_overlap(query, c.problem);
        // Also check overlap with individual thoughts
        for (const auto& t : c.thoughts) {
            double s = word_overlap(query, t.text);
            if (s > score) score = s;
        }
        scored.push_back({&c, score});
    }

    std::stable_sort(scored.begin(), scored.end(),
                     [](const Scored& a, const Scored& b){
                         return a.score > b.score;
                     });

    size_t n = std::min(max_results, scored.size());
    std::vector<const ReasoningChain*> result;
    result.reserve(n);
    for (size_t i = 0; i < n; ++i)
        result.push_back(scored[i].chain);
    return result;
}

const std::vector<ReasoningChain>& InternalDialogue::all_chains() const noexcept {
    return past_;
}

// ============================================================================
// Stats
// ============================================================================

InternalDialogue::Stats InternalDialogue::stats() const noexcept {
    Stats s;
    s.total_chains     = static_cast<uint64_t>(past_.size())
                         + (has_active_ ? 1u : 0u);
    s.completed_chains = static_cast<uint64_t>(past_.size());
    s.circular_detections = circular_detections_;

    uint64_t total_thoughts = 0;
    double   conf_sum       = 0.0;
    for (const auto& c : past_) {
        total_thoughts += c.thoughts.size();
        if (c.is_concluded())
            conf_sum += c.conclusion_confidence;
    }
    if (has_active_)
        total_thoughts += current_.thoughts.size();

    s.total_thoughts = total_thoughts;
    s.mean_chain_confidence = past_.empty()
        ? 0.0
        : conf_sum / static_cast<double>(past_.size());
    return s;
}

} // namespace nikola::interior
