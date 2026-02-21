/**
 * @file cognitive/query_engine.hpp
 * @brief Resonance-based semantic retrieval from SemanticMemory.
 *
 * The QueryEngine retrieves stored memories that best match a query
 * WaveFunction state using two complementary methods:
 *
 *   1. resonance_score()  — Born-rule inner product: |⟨a|b⟩|² / (|a|²·|b|²)
 *      Full wavefunction similarity, used by query().
 *
 *   2. query_by_coords()  — Hilbert curve proximity: |key_a − key_b|
 *      Spatial neighbourhood search; fast O(N) scan sorted by curve distance.
 *
 * Both methods return a ranked list of QueryResult (key, score) sorted
 * descending by score.
 *
 * Reference:
 *   PROJECT_STATUS.md Phase 2 — "Inner product retrieval"
 *   engineering report §3.5  "Born rule sampling with temperature as noise"
 *   engineering report §Gap 3.1 "Semantic Clustering" validation
 */
#pragma once

#include <nikola/cognitive/semantic_memory.hpp>

#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace nikola::cognitive {

// ============================================================================
// QueryResult
// ============================================================================

/**
 * @brief A single retrieval result from the memory store.
 */
struct QueryResult {
    MemoryKey key{0};    ///< Hilbert key of the matching record
    float     score{0.f}; ///< Similarity score (higher = more similar)
};

// ============================================================================
// QueryEngine
// ============================================================================

/**
 * @brief Inner-product and proximity-based semantic retrieval.
 *
 * Holds a const reference to a SemanticMemory.  The memory must outlive this
 * engine (observer pattern — does not take ownership).
 *
 * Thread safety: read-only use of SemanticMemory → safe if memory is not
 * modified concurrently.
 */
class QueryEngine {
public:
    // ------------------------------------------------------------------ construction

    /**
     * @brief Attach to an existing SemanticMemory store.
     * @param memory  Must outlive this QueryEngine.
     */
    explicit QueryEngine(const SemanticMemory& memory) noexcept
        : memory_(memory)
    {}

    // ------------------------------------------------------------------ resonance score (Born rule)

    /**
     * @brief Compute normalised resonance between two ψ-fields.
     *
     * Resonance = |⟨a|b⟩|² / (‖a‖²·‖b‖²)  — Born-rule overlap probability.
     *
     * If either ψ is the zero vector, returns 0.
     * The lengths are matched to the shorter of the two (safe prefix comparison).
     *
     * @param ar, ai  Real/imag parts of wavefunction a  (length na).
     * @param na      Number of samples in a.
     * @param br, bi  Real/imag parts of wavefunction b  (length nb).
     * @param nb      Number of samples in b.
     * @return        Resonance in [0, 1].
     */
    [[nodiscard]]
    static float resonance_score(const float* ar, const float* ai, size_t na,
                                 const float* br, const float* bi, size_t nb) noexcept
    {
        if (na == 0 || nb == 0) return 0.f;
        const size_t N = std::min(na, nb);

        // Inner product ⟨a|b⟩ = Σ a_i* · b_i  (complex, so Re(a)Re(b)+Im(a)Im(b) + j·…)
        double re_ab = 0.0, im_ab = 0.0;
        double norm_a2 = 0.0, norm_b2 = 0.0;

        for (size_t i = 0; i < N; ++i) {
            // a* · b  = (ar - j·ai)(br + j·bi)
            re_ab   += static_cast<double>(ar[i]) * br[i]
                     + static_cast<double>(ai[i]) * bi[i];
            im_ab   += static_cast<double>(ar[i]) * bi[i]
                     - static_cast<double>(ai[i]) * br[i];
            norm_a2 += static_cast<double>(ar[i]*ar[i] + ai[i]*ai[i]);
            norm_b2 += static_cast<double>(br[i]*br[i] + bi[i]*bi[i]);
        }

        if (norm_a2 < 1e-30 || norm_b2 < 1e-30) return 0.f;

        const double overlap2 = re_ab*re_ab + im_ab*im_ab;
        return static_cast<float>(overlap2 / (norm_a2 * norm_b2));
    }

    // ------------------------------------------------------------------ query by WaveFunction

    /**
     * @brief Retrieve the top-k memories most resonant with a query wavefunction.
     *
     * Computes resonance_score(query, record) for every stored record and
     * returns the k highest-scoring results, sorted descending.
     *
     * Scores are additionally weighted by record strength so faded memories
     * rank lower than fresh ones even with equal overlap.
     *
     * Complexity: O(N × M) where N = grid nodes, M = stored records.
     *
     * @param wf   Query WaveFunction.
     * @param k    Maximum results to return (0 = return all).
     * @return     Top-k QueryResult sorted by score descending.
     */
    [[nodiscard]]
    std::vector<QueryResult> query(const physics::WaveFunction& wf,
                                   size_t k = 5) const
    {
        const foundation::TorusGrid& grid = wf.grid();
        const size_t N_q  = grid.num_active_nodes();
        const float* qr   = grid.psi_real();
        const float* qi   = grid.psi_imag();

        std::vector<QueryResult> results;
        results.reserve(memory_.size());

        for (const MemoryKey key : memory_.all_keys()) {
            const MemoryRecord* rec = memory_.get(key);
            if (!rec || rec->psi_real.empty()) continue;

            const float base_score = resonance_score(
                qr, qi, N_q,
                rec->psi_real.data(), rec->psi_imag.data(), rec->psi_real.size());

            // Weight by memory strength (faded memories rank lower).
            const float score = base_score * rec->strength;
            results.push_back({key, score});
        }

        // Sort descending by score.
        std::sort(results.begin(), results.end(),
                  [](const QueryResult& a, const QueryResult& b) {
                      return a.score > b.score;
                  });

        // Trim to top-k.
        if (k > 0 && results.size() > k) {
            results.resize(k);
        }
        return results;
    }

    // ------------------------------------------------------------------ query by Hilbert proximity

    /**
     * @brief Retrieve the top-k memories closest on the Hilbert curve.
     *
     * Converts @p coords to a Hilbert key, then ranks all stored memories by
     * |stored_key − query_key| (unsigned distance on the 1D Hilbert index).
     * Memories that are spatially close in 9D space will have small distance.
     *
     * The score returned is 1 / (1 + distance) so closer = higher score.
     *
     * Complexity: O(M log M) where M = stored records.
     *
     * @param coords  9D grid coordinates (must be within scanner resolution).
     * @param k       Maximum results to return (0 = return all).
     * @return        Top-k QueryResult sorted by proximity descending.
     */
    [[nodiscard]]
    std::vector<QueryResult> query_by_coords(
            const spatial::HilbertScanner::Coord9D& coords,
            size_t k = 5) const
    {
        const MemoryKey query_key = memory_.scanner().coords_to_index(coords);

        std::vector<QueryResult> results;
        results.reserve(memory_.size());

        for (const MemoryKey key : memory_.all_keys()) {
            const MemoryRecord* rec = memory_.get(key);
            if (!rec) continue;

            // Unsigned distance on Hilbert 1D index.
            const uint64_t dist = (query_key >= key) ? (query_key - key)
                                                      : (key - query_key);
            // Proximity score: higher = closer, weighted by strength.
            const float score = rec->strength / (1.f + static_cast<float>(dist));
            results.push_back({key, score});
        }

        std::sort(results.begin(), results.end(),
                  [](const QueryResult& a, const QueryResult& b) {
                      return a.score > b.score;
                  });

        if (k > 0 && results.size() > k) {
            results.resize(k);
        }
        return results;
    }

    // ------------------------------------------------------------------ accessors

    /// Reference to the backing memory store.
    const SemanticMemory& memory() const noexcept { return memory_; }

private:
    const SemanticMemory& memory_;
};

} // namespace nikola::cognitive
