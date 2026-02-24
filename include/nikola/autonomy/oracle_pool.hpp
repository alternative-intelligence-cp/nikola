#pragma once
/**
 * @file include/nikola/autonomy/oracle_pool.hpp
 * @brief Phase 31 — OraclePool: multi-oracle knowledge credibility scoring.
 *
 * An Oracle assesses a (query, content) pair and returns a confidence score
 * in [0, 1].  The OraclePool runs all registered oracles and averages their
 * verdicts to produce a single credibility weight.
 *
 * This weight is used to modulate the amplitude of torus stimulus injection:
 *   credibility 1.0 → full amplitude (strongly believed)
 *   credibility 0.5 → half amplitude (uncertain)
 *   credibility 0.0 → no injection     (completely distrusted)
 *
 * Built-in oracles:
 *   StubOracle      — always returns a fixed score; used for testing and
 *                     as a placeholder until real API oracles are wired in.
 *   CoherenceOracle — purely local heuristic; scores content by length and
 *                     the presence of obvious self-contradictions.  No network
 *                     call required.
 *
 * Extending with real oracles (Tavily, Gemini) is Phase 32+:
 *   struct TavilyOracle : public Oracle { ... };
 *   pool.add_oracle(std::make_shared<TavilyOracle>(api_key));
 *
 * Phase: NIK-ORP-01 (Oracle Pool, Phase 31)
 */

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace nikola::autonomy {

// ============================================================================
// OracleVerdict
// ============================================================================

/**
 * @brief The output of one Oracle's assessment.
 *
 *   confidence   — assessed credibility in [0.0, 1.0].
 *   rationale    — human-readable explanation (for logs/debugging).
 */
struct OracleVerdict {
    float       confidence = 0.5f;  ///< Credibility weight [0.0–1.0]
    std::string rationale;          ///< Why this score was assigned
};

// ============================================================================
// Oracle — abstract base
// ============================================================================

/**
 * @brief Abstract oracle.  Implementations assess (query, content) pairs.
 *
 * Each oracle answers the question: "Given that I asked <query>, how much
 * do I trust <content> as a reliable answer?"
 *
 * Oracles do not need to be independent — the OraclePool averages them, so
 * a biased oracle just shifts the mean rather than dominating it.
 */
class Oracle {
public:
    virtual ~Oracle() = default;

    /**
     * @brief Assess the credibility of content for a given query.
     *
     * @param query    The question that triggered the lookup.
     * @param content  The retrieved text to assess.
     * @return OracleVerdict with confidence in [0.0, 1.0].
     */
    virtual OracleVerdict assess(const std::string& query,
                                 const std::string& content) = 0;

    /// Short identifier used in logs.
    virtual std::string name() const = 0;
};

// ============================================================================
// OraclePool
// ============================================================================

/**
 * @class OraclePool
 * @brief Runs all registered oracles and returns the mean confidence.
 *
 * Empty pool → 0.5 (neutral uncertainty, neither injected strongly nor
 * rejected outright).
 */
class OraclePool {
public:
    /// Add an oracle.  Not thread-safe — call before run().
    void add_oracle(std::shared_ptr<Oracle> oracle) {
        oracles_.push_back(std::move(oracle));
    }

    /**
     * @brief Evaluate query+content against all oracles; return mean score.
     *
     * Each oracle's confidence is clamped to [0, 1] before averaging.
     * Result is in [0.0, 1.0].
     *
     * @param query    The original lookup query.
     * @param content  The retrieved content to score.
     * @return Mean oracle confidence; 0.5f if the pool is empty.
     */
    float evaluate(const std::string& query, const std::string& content) const {
        if (oracles_.empty()) return 0.5f;

        float total = 0.f;
        for (const auto& o : oracles_) {
            const auto v = o->assess(query, content);
            total += std::clamp(v.confidence, 0.f, 1.f);
        }
        return total / static_cast<float>(oracles_.size());
    }

    size_t size()  const noexcept { return oracles_.size(); }
    bool   empty() const noexcept { return oracles_.empty(); }

    const std::vector<std::shared_ptr<Oracle>>& oracles() const noexcept {
        return oracles_;
    }

private:
    std::vector<std::shared_ptr<Oracle>> oracles_;
};

// ============================================================================
// StubOracle — fixed score; deterministic; used for testing
// ============================================================================

/**
 * @brief Always returns a fixed, configurable confidence score.
 *
 * Usage:
 *   pool.add_oracle(std::make_shared<StubOracle>("fact-check", 0.9f));
 *   pool.add_oracle(std::make_shared<StubOracle>("safety",     0.6f));
 *   float mean = pool.evaluate("query", "content");  // → 0.75f
 *
 * Replace with a real oracle once the external API is integrated.
 */
class StubOracle final : public Oracle {
public:
    explicit StubOracle(std::string oracle_name, float fixed_score = 0.7f)
        : name_(std::move(oracle_name))
        , score_(std::clamp(fixed_score, 0.f, 1.f))
    {}

    OracleVerdict assess(const std::string& /*query*/,
                         const std::string& /*content*/) override {
        return { score_, "stub fixed score " + name_ };
    }

    std::string name() const override { return name_; }

private:
    std::string name_;
    float       score_;
};

// ============================================================================
// CoherenceOracle — local heuristic; no network call required
// ============================================================================

/**
 * @class CoherenceOracle
 * @brief Scores content by structural coherence.
 *
 * Scoring rules (applied in order, result clamped to [0, 1]):
 *
 *   empty content       → 0.00  (nothing to inject)
 *   len < 20 chars      → 0.20  (too short to be informative)
 *   20 ≤ len < 100      → 0.55  (minimal factual snippet)
 *   100 ≤ len < 500     → 0.75  (solid paragraph)
 *   len ≥ 500           → 0.85  (substantive content)
 *
 *   " is not " pattern  → −0.20 penalty (possible self-contradiction)
 *
 * This oracle does not understand semantics — it is a weak signal.
 * For real deployments it should be paired with a Gemini/LLM oracle.
 */
class CoherenceOracle final : public Oracle {
public:
    OracleVerdict assess(const std::string& /*query*/,
                         const std::string& content) override {
        if (content.empty()) {
            return { 0.0f, "empty content" };
        }

        const float len = static_cast<float>(content.size());

        float score;
        if      (len < 20.f)  score = 0.20f;
        else if (len < 100.f) score = 0.55f;
        else if (len < 500.f) score = 0.75f;
        else                  score = 0.85f;

        // Contradiction signal: " is not " often appears in factually
        // inconsistent content (especially LLM hallucinations).
        // Apply a modest penalty without fully rejecting the content.
        if (content.find(" is not ") != std::string::npos) {
            score -= 0.20f;
        }

        score = std::clamp(score, 0.f, 1.f);
        return { score, "coherence heuristic" };
    }

    std::string name() const override { return "coherence"; }
};

} // namespace nikola::autonomy
