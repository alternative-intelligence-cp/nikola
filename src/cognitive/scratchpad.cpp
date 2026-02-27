/**
 * @file scratchpad.cpp
 * @brief Phase 125 — Scratchpad implementation
 */

#include <nikola/cognitive/scratchpad.hpp>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <sstream>
#include <unordered_set>
#include <numeric>

namespace nikola::cognitive {

// ---------------------------------------------------------------------------
// Static helper implementations
// ---------------------------------------------------------------------------

double Scratchpad::word_overlap(const std::string& a, const std::string& b) {
    auto tokenise = [](const std::string& s) {
        std::unordered_set<std::string> out;
        std::istringstream iss(s);
        std::string tok;
        while (iss >> tok) {
            // lower-case normalisation
            std::string lo;
            lo.reserve(tok.size());
            for (unsigned char c : tok) {
                lo += static_cast<char>(std::tolower(c));
            }
            out.insert(lo);
        }
        return out;
    };

    auto sa = tokenise(a);
    auto sb = tokenise(b);

    if (sa.empty() && sb.empty()) return 1.0;
    if (sa.empty() || sb.empty()) return 0.0;

    size_t intersection = 0;
    for (const auto& w : sa) {
        if (sb.count(w)) ++intersection;
    }

    const size_t union_sz = sa.size() + sb.size() - intersection;
    return union_sz == 0 ? 0.0
                         : static_cast<double>(intersection) /
                               static_cast<double>(union_sz);
}

double Scratchpad::score_against_pool(const std::string& hyp_text,
                                       const std::vector<CommittedEntry>& pool) {
    if (pool.empty()) return 0.0;

    double best = 0.0;
    for (const auto& entry : pool) {
        const double raw   = word_overlap(hyp_text, entry.text);
        const double score = raw * std::clamp(entry.confidence, 0.0, 1.0);
        if (score > best) best = score;
    }
    return best;
}

// ---------------------------------------------------------------------------
// Committed pool
// ---------------------------------------------------------------------------

void Scratchpad::commit(const std::string& text, double confidence) {
    // FIFO eviction if at capacity
    if (committed_.size() >= SCRATCHPAD_MAX_COMMITTED) {
        committed_.erase(committed_.begin());
    }

    CommittedEntry e;
    e.id         = next_committed_id_++;
    e.text       = text;
    e.confidence = std::clamp(confidence, 0.0, 1.0);
    committed_.push_back(std::move(e));
}

// ---------------------------------------------------------------------------
// Hypothesis lifecycle
// ---------------------------------------------------------------------------

uint64_t Scratchpad::inject(const std::string& text,
                             double confidence,
                             const NikolaState* state) {
    // FIFO eviction: drop oldest PENDING if at cap
    if (hypotheses_.size() >= SCRATCHPAD_MAX_HYPOTHESES) {
        auto it = std::find_if(hypotheses_.begin(), hypotheses_.end(),
                               [](const HypothesisEntry& e) {
                                   return e.status == HypothesisStatus::PENDING;
                               });
        if (it != hypotheses_.end()) {
            hypotheses_.erase(it);
        } else {
            // all are settled — evict oldest overall
            hypotheses_.erase(hypotheses_.begin());
        }
    }

    HypothesisEntry e;
    e.id         = next_hyp_id_++;
    e.text       = text;
    e.confidence = std::clamp(confidence, 0.0, 1.0);
    e.status     = HypothesisStatus::PENDING;
    e.resonance  = 0.0;

    if (state) {
        e.dopamine_ctx = state->dopamine;
        e.entropy_ctx  = state->entropy;
    }

    const uint64_t id = e.id;
    hypotheses_.push_back(std::move(e));
    return id;
}

double Scratchpad::measure_resonance(uint64_t id) {
    HypothesisEntry* e = find_mutable(id);
    if (!e) return 0.0;

    const double score = score_against_pool(e->text, committed_);
    e->resonance = score;
    return score;
}

bool Scratchpad::collapse_if_resonant(uint64_t id, double threshold) {
    HypothesisEntry* e = find_mutable(id);
    if (!e) return false;

    // Refresh resonance
    e->resonance = score_against_pool(e->text, committed_);

    if (e->resonance >= threshold) {
        e->status = HypothesisStatus::COLLAPSED;
        if (collapse_cb_) collapse_cb_(*e);
        return true;
    }

    e->status = HypothesisStatus::DISCARDED;
    return false;
}

void Scratchpad::discard(uint64_t id) {
    HypothesisEntry* e = find_mutable(id);
    if (e) e->status = HypothesisStatus::DISCARDED;
}

void Scratchpad::clear_pending() {
    hypotheses_.erase(
        std::remove_if(hypotheses_.begin(), hypotheses_.end(),
                       [](const HypothesisEntry& e) {
                           return e.status == HypothesisStatus::PENDING;
                       }),
        hypotheses_.end());
}

void Scratchpad::clear_all() {
    hypotheses_.clear();
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

std::vector<const HypothesisEntry*> Scratchpad::pending() const {
    std::vector<const HypothesisEntry*> out;
    for (const auto& e : hypotheses_) {
        if (e.status == HypothesisStatus::PENDING) out.push_back(&e);
    }
    return out;
}

std::vector<const HypothesisEntry*> Scratchpad::collapsed() const {
    std::vector<const HypothesisEntry*> out;
    for (const auto& e : hypotheses_) {
        if (e.status == HypothesisStatus::COLLAPSED) out.push_back(&e);
    }
    return out;
}

std::vector<const HypothesisEntry*> Scratchpad::discarded() const {
    std::vector<const HypothesisEntry*> out;
    for (const auto& e : hypotheses_) {
        if (e.status == HypothesisStatus::DISCARDED) out.push_back(&e);
    }
    return out;
}

const HypothesisEntry* Scratchpad::find(uint64_t id) const {
    for (const auto& e : hypotheses_) {
        if (e.id == id) return &e;
    }
    return nullptr;
}

HypothesisEntry* Scratchpad::find_mutable(uint64_t id) {
    for (auto& e : hypotheses_) {
        if (e.id == id) return &e;
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

Scratchpad::Stats Scratchpad::stats() const {
    Stats s;
    s.total_committed = committed_.size();

    double res_sum   = 0.0;
    size_t res_count = 0;

    for (const auto& e : hypotheses_) {
        ++s.total_injected;
        switch (e.status) {
        case HypothesisStatus::PENDING:
            ++s.total_pending;
            break;
        case HypothesisStatus::COLLAPSED:
            ++s.total_collapsed;
            res_sum += e.resonance;
            ++res_count;
            break;
        case HypothesisStatus::DISCARDED:
            ++s.total_discarded;
            res_sum += e.resonance;
            ++res_count;
            break;
        }
    }

    s.mean_resonance = res_count > 0
                           ? res_sum / static_cast<double>(res_count)
                           : 0.0;
    return s;
}

} // namespace nikola::cognitive
