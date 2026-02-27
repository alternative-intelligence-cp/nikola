/**
 * @file dream_engine.cpp
 * @brief Phase 123 — DreamEngine implementation
 */

#include <nikola/interior/dream_engine.hpp>

#include <algorithm>
#include <numeric>
#include <sstream>
#include <unordered_set>
#include <cctype>
#include <cmath>

namespace nikola::interior {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace {

/// Tokenise a string into lowercase words (split on non-alpha).
std::unordered_set<std::string> tokenise_set(const std::string& text) {
    std::unordered_set<std::string> tokens;
    std::string cur;
    for (char c : text) {
        if (std::isalpha(static_cast<unsigned char>(c))) {
            cur += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        } else {
            if (!cur.empty()) { tokens.insert(cur); cur.clear(); }
        }
    }
    if (!cur.empty()) tokens.insert(cur);
    return tokens;
}

/// Return Jaccard overlap of two token sets.
double jaccard(const std::string& a, const std::string& b) {
    auto ta = tokenise_set(a);
    auto tb = tokenise_set(b);
    if (ta.empty() && tb.empty()) return 1.0;
    if (ta.empty() || tb.empty()) return 0.0;
    size_t inter = 0;
    for (const auto& w : ta) if (tb.count(w)) ++inter;
    size_t uni = ta.size() + tb.size() - inter;
    return static_cast<double>(inter) / static_cast<double>(uni);
}

/// Describe a neurochemical state in a short phrase.
std::string state_descriptor(const NikolaState& s) {
    std::string dop = (s.dopamine > 0.65f) ? "high-dopamine"
                    : (s.dopamine < 0.35f) ? "low-dopamine"
                    : "mid-dopamine";
    std::string atp = (s.atp > 0.65f) ? "high-energy"
                    : (s.atp < 0.35f) ? "depleted"
                    : "mid-energy";
    std::string ent = (s.entropy > 1.5f) ? "chaotic"
                    : (s.entropy < 0.5f) ? "ordered"
                    : "moderate-entropy";
    return dop + "/" + atp + "/" + ent;
}

const float MAX_L2 = 2.0f; // sqrt(4) components each clamped [0,~2]

} // anonymous namespace

// ---------------------------------------------------------------------------
// Pure-static implementations
// ---------------------------------------------------------------------------

double DreamEngine::state_similarity(const NikolaState& a, const NikolaState& b) {
    float dd = a.dopamine    - b.dopamine;
    float da = a.atp         - b.atp;
    float de = a.entropy     - b.entropy;
    float dt = a.torus_energy - b.torus_energy;
    double dist = std::sqrt(static_cast<double>(dd*dd + da*da + de*de + dt*dt));
    double sim  = 1.0 - dist / static_cast<double>(MAX_L2);
    return std::max(0.0, std::min(1.0, sim));
}

bool DreamEngine::is_nightmare_state(const NikolaState& s) {
    return s.entropy  > DREAM_NIGHTMARE_ENTROPY
        && s.dopamine < DREAM_NIGHTMARE_DOPAMINE;
}

bool DreamEngine::is_idle_enough(const NikolaState& s) {
    return static_cast<double>(s.boredom) >= DREAM_IDLE_THRESHOLD;
}

std::string DreamEngine::generate_insight(const Experience& a,
                                          const Experience& b,
                                          double similarity) {
    std::ostringstream ss;
    ss << "Connection [sim=" << static_cast<int>(similarity * 100) << "%]: "
       << "\"" << a.tag << "\" (" << state_descriptor(a.state) << ")"
       << " resonates with "
       << "\"" << b.tag << "\" (" << state_descriptor(b.state) << ")";
    return ss.str();
}

double DreamEngine::compute_novelty(double similarity, double mean_similarity) {
    // Surprise = pair is not super-similar, AND overall buffer is diverse.
    double novelty = (1.0 - similarity) * (1.0 - mean_similarity);
    return std::max(0.0, std::min(1.0, novelty));
}

double DreamEngine::tag_overlap(const std::string& a, const std::string& b) {
    return jaccard(a, b);
}

// ---------------------------------------------------------------------------
// record_experience
// ---------------------------------------------------------------------------

void DreamEngine::record_experience(const std::string& tag,
                                    const NikolaState& state,
                                    float reward) {
    // Evict oldest when full
    if (experiences_.size() >= DREAM_BUFFER_SIZE) {
        experiences_.erase(experiences_.begin());
    }

    Experience e;
    e.tick         = (experiences_.empty() ? 0 : experiences_.back().tick + 1);
    e.state        = state;
    e.tag          = tag;
    e.reward_signal = reward;
    e.is_nightmare  = is_nightmare_state(state);

    experiences_.push_back(std::move(e));
}

// ---------------------------------------------------------------------------
// dream
// ---------------------------------------------------------------------------

DreamCycle DreamEngine::dream(uint64_t tick) {
    DreamCycle cycle;
    cycle.start_tick = tick;

    if (experiences_.size() < 2) {
        cycle.end_tick = tick;
        dream_log_.push_back(cycle);
        if (dream_cb_) dream_cb_(cycle);
        return cycle;
    }

    const size_t n = experiences_.size();

    // Compute mean pairwise similarity for novelty baseline
    // (sample up to 64 random consecutive pairs for efficiency)
    double sum_sim = 0.0;
    size_t pair_count = 0;
    size_t step = std::max(size_t{1}, n / 8);
    for (size_t i = 0; i + step < n; i += step) {
        sum_sim += state_similarity(experiences_[i].state,
                                    experiences_[i + step].state);
        ++pair_count;
    }
    double mean_sim = (pair_count > 0) ? sum_sim / static_cast<double>(pair_count) : 0.5;

    // Collect qualifying fragments
    std::vector<DreamFragment> fragments;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            double sim = state_similarity(experiences_[i].state,
                                          experiences_[j].state);
            if (sim >= DREAM_SIMILARITY_THRESHOLD) {
                DreamFragment frag;
                frag.exp_index_a   = i;
                frag.exp_index_b   = j;
                frag.similarity    = sim;
                frag.novelty_score = compute_novelty(sim, mean_sim);
                frag.insight       = generate_insight(experiences_[i],
                                                       experiences_[j], sim);
                fragments.push_back(std::move(frag));
            }
        }
    }

    cycle.fragments_found = fragments.size();
    total_fragments_     += fragments.size();

    // Consolidate high-novelty fragments into memories
    double novelty_sum = 0.0;
    for (const auto& frag : fragments) {
        novelty_sum += frag.novelty_score;
        if (frag.novelty_score >= DREAM_CONSOLIDATION_MIN) {
            ConsolidatedMemory mem;
            mem.formation_tick      = tick;
            mem.key_insight         = frag.insight;
            mem.confidence          = frag.novelty_score;
            mem.source_exp_indices  = { frag.exp_index_a, frag.exp_index_b };
            mem.from_nightmare      = experiences_[frag.exp_index_a].is_nightmare
                                   || experiences_[frag.exp_index_b].is_nightmare;
            memories_.push_back(std::move(mem));
            ++cycle.memories_formed;
        }
    }

    cycle.mean_novelty = fragments.empty()
        ? 0.0
        : novelty_sum / static_cast<double>(fragments.size());

    // Count nightmares processed
    size_t nm = 0;
    for (const auto& e : experiences_) if (e.is_nightmare) ++nm;
    cycle.nightmares_processed = nm;

    cycle.end_tick = tick;
    dream_log_.push_back(cycle);

    if (dream_cb_) dream_cb_(cycle);
    return cycle;
}

// ---------------------------------------------------------------------------
// recall
// ---------------------------------------------------------------------------

std::vector<const ConsolidatedMemory*>
DreamEngine::recall(const std::string& query, size_t max) const {
    if (memories_.empty() || query.empty()) return {};

    std::vector<std::pair<double, const ConsolidatedMemory*>> scored;
    scored.reserve(memories_.size());

    for (const auto& mem : memories_) {
        double score = tag_overlap(query, mem.key_insight);
        // Boost by confidence
        score = score * 0.7 + mem.confidence * 0.3;
        scored.emplace_back(score, &mem);
    }

    std::sort(scored.begin(), scored.end(),
              [](const auto& x, const auto& y){ return x.first > y.first; });

    std::vector<const ConsolidatedMemory*> result;
    size_t limit = std::min(max, scored.size());
    result.reserve(limit);
    for (size_t i = 0; i < limit; ++i) result.push_back(scored[i].second);
    return result;
}

// ---------------------------------------------------------------------------
// process_nightmares
// ---------------------------------------------------------------------------

std::vector<std::string> DreamEngine::process_nightmares() const {
    std::vector<std::string> patterns;

    // Group nightmares by tag-overlap (simple clustering)
    std::vector<const Experience*> nightmares;
    for (const auto& e : experiences_)
        if (e.is_nightmare) nightmares.push_back(&e);

    if (nightmares.empty()) return patterns;

    // Find tag clusters: greedy — pick first unclustered, group with overlap>0.3
    std::vector<bool> used(nightmares.size(), false);
    for (size_t i = 0; i < nightmares.size(); ++i) {
        if (used[i]) continue;
        std::vector<const Experience*> cluster;
        cluster.push_back(nightmares[i]);
        used[i] = true;
        for (size_t j = i + 1; j < nightmares.size(); ++j) {
            if (!used[j] &&
                tag_overlap(nightmares[i]->tag, nightmares[j]->tag) > 0.3) {
                cluster.push_back(nightmares[j]);
                used[j] = true;
            }
        }
        // Summarise cluster
        std::ostringstream ss;
        ss << "Failure pattern [" << cluster.size() << " events]: "
           << "\"" << cluster[0]->tag << "\" — avg entropy=";
        double sum_e = 0.0;
        for (const auto* e : cluster) sum_e += e->state.entropy;
        ss << static_cast<int>((sum_e / cluster.size()) * 100) / 100.0
           << ", avg dopamine=";
        double sum_d = 0.0;
        for (const auto* e : cluster) sum_d += e->state.dopamine;
        ss << static_cast<int>((sum_d / cluster.size()) * 100) / 100.0;
        patterns.push_back(ss.str());
    }

    return patterns;
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

size_t DreamEngine::nightmare_count() const {
    size_t n = 0;
    for (const auto& e : experiences_) if (e.is_nightmare) ++n;
    return n;
}

DreamEngine::Stats DreamEngine::stats() const {
    Stats s;
    s.total_experiences   = experiences_.size();
    s.total_nightmares    = nightmare_count();
    s.total_fragments     = total_fragments_;
    s.total_memories      = memories_.size();
    s.total_dream_cycles  = dream_log_.size();

    if (!memories_.empty()) {
        double sum = 0.0;
        for (const auto& m : memories_) sum += m.confidence;
        s.mean_memory_confidence = sum / static_cast<double>(memories_.size());
    }
    return s;
}

} // namespace nikola::interior
