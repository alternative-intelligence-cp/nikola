/**
 * @file attention_primer.cpp
 * @brief Phase 126 — AttentionPrimer implementation
 */

#include <nikola/cognitive/attention_primer.hpp>

#include <algorithm>
#include <cctype>
#include <numeric>
#include <sstream>
#include <unordered_set>

namespace nikola::cognitive {

// ---------------------------------------------------------------------------
// Static helpers
// ---------------------------------------------------------------------------

std::string AttentionPrimer::normalise_tag(const std::string& tag) {
    std::string out;
    out.reserve(tag.size());
    for (unsigned char c : tag) {
        out += static_cast<char>(std::tolower(c));
    }
    return out;
}

size_t AttentionPrimer::find_index(const std::string& normalised) const {
    for (size_t i = 0; i < topics_.size(); ++i) {
        if (normalise_tag(topics_[i].tag) == normalised) return i;
    }
    return std::string::npos;
}

double AttentionPrimer::topic_overlap(const std::string& a, const std::string& b) {
    auto tokenise = [](const std::string& s) {
        std::unordered_set<std::string> out;
        std::istringstream iss(s);
        std::string tok;
        while (iss >> tok) {
            std::string lo;
            for (unsigned char c : tok) lo += static_cast<char>(std::tolower(c));
            out.insert(lo);
        }
        return out;
    };

    auto sa = tokenise(a);
    auto sb = tokenise(b);

    if (sa.empty() && sb.empty()) return 1.0;
    if (sa.empty() || sb.empty()) return 0.0;

    size_t inter = 0;
    for (const auto& w : sa) {
        if (sb.count(w)) ++inter;
    }
    const size_t uni = sa.size() + sb.size() - inter;
    return uni == 0 ? 0.0 : static_cast<double>(inter) / static_cast<double>(uni);
}

double AttentionPrimer::state_bonus(const std::string& tag,
                                     const NikolaState& state) {
    const std::string lo = normalise_tag(tag);
    double bonus = 0.0;

    // High dopamine → reward / goal / success topics get a boost
    if (state.dopamine > 0.6f) {
        const bool reward_topic =
            lo.find("reward") != std::string::npos ||
            lo.find("goal")   != std::string::npos ||
            lo.find("achiev") != std::string::npos ||
            lo.find("succes") != std::string::npos;
        if (reward_topic) bonus += 0.15;
    }

    // High boredom → novel / explore / question topics get a boost
    if (state.boredom > 0.5f) {
        const bool explore_topic =
            lo.find("novel")   != std::string::npos ||
            lo.find("explor")  != std::string::npos ||
            lo.find("new")     != std::string::npos ||
            lo.find("curious") != std::string::npos ||
            lo.find("questio") != std::string::npos;
        if (explore_topic) bonus += 0.15;
    }

    // High entropy → uncertainty / conflict / resolve topics get a boost
    if (state.entropy > 0.6f) {
        const bool uncertainty_topic =
            lo.find("uncert")  != std::string::npos ||
            lo.find("conflic") != std::string::npos ||
            lo.find("resolv")  != std::string::npos ||
            lo.find("ambig")   != std::string::npos;
        if (uncertainty_topic) bonus += 0.10;
    }

    // Cap total bonus
    return std::min(0.25, bonus);
}

// ---------------------------------------------------------------------------
// Priming
// ---------------------------------------------------------------------------

void AttentionPrimer::prime(const std::string& tag,
                             double activation,
                             double decay_rate,
                             uint64_t tick,
                             const NikolaState* state) {
    activation = std::clamp(activation, 0.0, 1.0);
    decay_rate = std::clamp(decay_rate, 0.0, 1.0);

    const std::string norm = normalise_tag(tag);
    const size_t idx = find_index(norm);

    if (idx != std::string::npos) {
        // Boost existing entry
        topics_[idx].activation = merged_activation(topics_[idx].activation,
                                                     activation);
        topics_[idx].prime_tick  = tick;
        topics_[idx].decay_rate  = decay_rate;
        if (state) {
            topics_[idx].dopamine_ctx = state->dopamine;
            topics_[idx].entropy_ctx  = state->entropy;
        }
        if (prime_cb_) prime_cb_(topics_[idx]);
        return;
    }

    // Evict lowest-weight entry if at cap
    if (topics_.size() >= ATTENTION_MAX_TOPICS) {
        auto min_it = std::min_element(
            topics_.begin(), topics_.end(),
            [](const PrimedFocus& a, const PrimedFocus& b) {
                return a.activation < b.activation;
            });
        topics_.erase(min_it);
    }

    PrimedFocus f;
    f.tag        = tag;   // store original casing
    f.activation = activation;
    f.decay_rate = decay_rate;
    f.prime_tick = tick;
    if (state) {
        f.dopamine_ctx = state->dopamine;
        f.entropy_ctx  = state->entropy;
    }

    topics_.push_back(f);
    if (prime_cb_) prime_cb_(topics_.back());
}

void AttentionPrimer::decay_all() {
    for (auto& t : topics_) {
        t.activation *= t.decay_rate;
    }
    // Prune below minimum weight
    topics_.erase(
        std::remove_if(topics_.begin(), topics_.end(),
                       [](const PrimedFocus& t) {
                           return t.activation < ATTENTION_MIN_WEIGHT;
                       }),
        topics_.end());
}

void AttentionPrimer::remove(const std::string& tag) {
    const std::string norm = normalise_tag(tag);
    topics_.erase(
        std::remove_if(topics_.begin(), topics_.end(),
                       [&](const PrimedFocus& t) {
                           return normalise_tag(t.tag) == norm;
                       }),
        topics_.end());
}

void AttentionPrimer::clear() {
    topics_.clear();
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

double AttentionPrimer::weight_of(const std::string& tag) const {
    const size_t idx = find_index(normalise_tag(tag));
    if (idx == std::string::npos) return 0.0;
    return topics_[idx].activation;
}

bool AttentionPrimer::is_primed(const std::string& tag, double threshold) const {
    return weight_of(tag) >= threshold;
}

std::optional<PrimedFocus> AttentionPrimer::most_primed() const {
    if (topics_.empty()) return std::nullopt;

    auto it = std::max_element(
        topics_.begin(), topics_.end(),
        [](const PrimedFocus& a, const PrimedFocus& b) {
            return a.activation < b.activation;
        });
    return *it;
}

std::vector<PrimedFocus> AttentionPrimer::all_primed() const {
    auto sorted = topics_;
    std::sort(sorted.begin(), sorted.end(),
              [](const PrimedFocus& a, const PrimedFocus& b) {
                  return a.activation > b.activation;  // descending
              });
    return sorted;
}

// ---------------------------------------------------------------------------
// State-aware prediction
// ---------------------------------------------------------------------------

std::optional<PrimedFocus> AttentionPrimer::predict_focus(
    const NikolaState& state) const {
    if (topics_.empty()) return std::nullopt;

    double      best_score = -1.0;
    const PrimedFocus* best = nullptr;

    for (const auto& t : topics_) {
        const double score = t.activation + state_bonus(t.tag, state);
        if (score > best_score) {
            best_score = score;
            best       = &t;
        }
    }

    if (!best) return std::nullopt;

    // Return a copy with the combined score reflected in activation
    PrimedFocus result = *best;
    result.activation  = std::min(1.0, best_score);
    return result;
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

AttentionPrimer::Stats AttentionPrimer::stats() const {
    Stats s;
    s.topic_count = topics_.size();

    if (topics_.empty()) return s;

    double sum = 0.0;
    s.max_activation = 0.0;
    s.min_activation = 1.0;

    for (const auto& t : topics_) {
        sum += t.activation;
        if (t.activation > s.max_activation) s.max_activation = t.activation;
        if (t.activation < s.min_activation) s.min_activation = t.activation;
    }
    s.mean_activation = sum / static_cast<double>(topics_.size());
    return s;
}

} // namespace nikola::cognitive
