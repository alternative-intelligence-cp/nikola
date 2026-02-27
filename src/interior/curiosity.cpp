/**
 * @file src/interior/curiosity.cpp
 * @brief Phase 118 — CuriosityEngine implementation.
 *
 * All methods that accept `const TorusManifold&` or `TorusManifold&` treat the
 * parameter as a future integration hook only: the actual state is maintained
 * inside CuriosityEngine itself (gaps_, interest_history_, exploration_rate_).
 * TorusManifold is never dereferenced; its full definition is deferred.
 */
#include "nikola/interior/curiosity.hpp"

#include <algorithm>
#include <cmath>

namespace nikola::interior {

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Binary Shannon entropy H(u) = -u·log₂u − (1−u)·log₂(1−u), clamped to [0,1].
static double binary_entropy(double u) {
    if (u <= 0.0 || u >= 1.0) return 0.0;
    return -(u * std::log2(u) + (1.0 - u) * std::log2(1.0 - u));
}

// ─────────────────────────────────────────────────────────────────────────────
// register_gap
// ─────────────────────────────────────────────────────────────────────────────

void CuriosityEngine::register_gap(KnowledgeGap gap) {
    if (gap.domain.empty()) return;
    auto it = gaps_.find(gap.domain);
    if (it == gaps_.end()) {
        gaps_.emplace(gap.domain, std::move(gap));
    } else {
        // Merge: average uncertainty, keep max query_count, union memories
        KnowledgeGap& existing = it->second;
        existing.uncertainty = (existing.uncertainty + gap.uncertainty) * 0.5;
        existing.query_count = std::max(existing.query_count, gap.query_count);
        for (const auto& m : gap.related_memories) {
            if (std::find(existing.related_memories.begin(),
                          existing.related_memories.end(), m)
                    == existing.related_memories.end()) {
                existing.related_memories.push_back(m);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// measure_information_gain
// ─────────────────────────────────────────────────────────────────────────────

double CuriosityEngine::measure_information_gain(const std::string& query,
                                                  const TorusManifold& /*torus*/) {
    auto it = gaps_.find(query);
    if (it != gaps_.end()) {
        // Gain ∝ gap uncertainty, discounted by how many times we've already
        // queried it — diminishing returns after repeated attempts.
        const KnowledgeGap& g = it->second;
        double saturation = 1.0 / (1.0 + static_cast<double>(g.query_count));
        return std::clamp(g.uncertainty * saturation, 0.0, 1.0);
    }
    // Unknown domain: exploration_rate_ acts as a neutral prior.
    return exploration_rate_;
}

// ─────────────────────────────────────────────────────────────────────────────
// generate_questions
// ─────────────────────────────────────────────────────────────────────────────

std::vector<Question> CuriosityEngine::generate_questions(
        const TorusManifold& /*torus*/, int count) {

    if (count <= 0) return {};

    // Rank every known gap by "value" = entropy(uncertainty) / (1 + log(query_count+1))
    std::vector<std::pair<double, const KnowledgeGap*>> ranked;
    ranked.reserve(gaps_.size());
    for (const auto& [domain, gap] : gaps_) {
        double value = binary_entropy(gap.uncertainty)
                     / (1.0 + std::log1p(static_cast<double>(gap.query_count)));
        ranked.emplace_back(value, &gap);
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<Question> questions;
    questions.reserve(static_cast<std::size_t>(count));

    int generated = 0;
    for (const auto& [value, gap] : ranked) {
        if (generated >= count) break;
        Question q;
        q.text             = "What can be learned about '" + gap->domain + "'?";
        q.information_gain = std::clamp(gap->uncertainty, 0.0, 1.0);
        q.interestingness  = std::clamp(value, 0.0, 1.0);
        q.tags             = gap->related_memories;
        questions.push_back(std::move(q));
        ++generated;
    }

    // If we still need more, pad with a generic exploration question.
    if (generated < count) {
        Question q;
        q.text             = "What novel knowledge would most expand current understanding?";
        q.information_gain = exploration_rate_;
        q.interestingness  = exploration_rate_;
        q.tags             = {"exploration"};
        questions.push_back(std::move(q));
    }

    questions_generated_ += static_cast<uint64_t>(questions.size());

    if (curiosity_callback_) {
        for (const auto& q : questions)
            curiosity_callback_(q);
    }
    return questions;
}

// ─────────────────────────────────────────────────────────────────────────────
// pursue_interest
// ─────────────────────────────────────────────────────────────────────────────

bool CuriosityEngine::pursue_interest(const std::string& topic, TorusManifold& /*torus*/) {
    if (topic.empty()) return false;

    interest_history_.push_back(topic);
    ++topics_pursued_;

    // Register / update gap for this topic.
    auto& gap = gaps_[topic];
    gap.domain = topic;
    ++gap.query_count;
    // Each pursuit reduces uncertainty slightly — learning reduces ignorance.
    gap.uncertainty = std::clamp(gap.uncertainty - 0.05, 0.1, 1.0);

    if (curiosity_callback_) {
        Question q;
        q.text             = "Pursuing interest: " + topic;
        q.information_gain = gap.uncertainty;
        q.interestingness  = std::clamp(exploration_rate_ + (1.0 - gap.uncertainty), 0.0, 1.0);
        q.tags             = {topic};
        curiosity_callback_(q);
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// exploration rate
// ─────────────────────────────────────────────────────────────────────────────

double CuriosityEngine::get_exploration_rate() const {
    return exploration_rate_;
}

void CuriosityEngine::set_exploration_rate(double rate) {
    exploration_rate_ = std::clamp(rate, 0.0, 1.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// identify_knowledge_gaps
// ─────────────────────────────────────────────────────────────────────────────

std::vector<KnowledgeGap> CuriosityEngine::identify_knowledge_gaps(
        const TorusManifold& /*torus*/) {

    std::vector<KnowledgeGap> result;
    result.reserve(gaps_.size());
    for (const auto& [domain, gap] : gaps_)
        result.push_back(gap);

    // Sort by descending uncertainty — biggest gap first.
    std::sort(result.begin(), result.end(),
              [](const KnowledgeGap& a, const KnowledgeGap& b) {
                  return a.uncertainty > b.uncertainty;
              });
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// measure_interestingness
// ─────────────────────────────────────────────────────────────────────────────

double CuriosityEngine::measure_interestingness(const std::string& topic,
                                                 const TorusManifold& /*torus*/) {
    if (topic.empty()) return 0.0;

    auto it = gaps_.find(topic);
    if (it == gaps_.end()) {
        // Completely unknown → inherently interesting to an explorer.
        return std::clamp(exploration_rate_ + 0.2, 0.0, 1.0);
    }
    // Interest decays as the topic becomes familiar.
    const KnowledgeGap& g = it->second;
    double familiarity = 1.0 / (1.0 + static_cast<double>(g.query_count));
    return std::clamp(g.uncertainty * familiarity, 0.0, 1.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// autonomous learning
// ─────────────────────────────────────────────────────────────────────────────

void CuriosityEngine::start_autonomous_learning(TorusManifold& /*torus*/,
                                                 uint64_t /*interval_ms*/) {
    learning_active_ = true;
}

void CuriosityEngine::stop_autonomous_learning() {
    learning_active_ = false;
}

// ─────────────────────────────────────────────────────────────────────────────
// callback & stats
// ─────────────────────────────────────────────────────────────────────────────

void CuriosityEngine::set_curiosity_callback(
        std::function<void(const Question&)> callback) {
    curiosity_callback_ = std::move(callback);
}

std::map<std::string, uint64_t> CuriosityEngine::get_stats() const {
    return {
        {"questions_generated", questions_generated_},
        {"topics_pursued",      topics_pursued_},
        {"gaps_tracked",        static_cast<uint64_t>(gaps_.size())},
        {"interest_history",    static_cast<uint64_t>(interest_history_.size())}
    };
}

} // namespace nikola::interior
