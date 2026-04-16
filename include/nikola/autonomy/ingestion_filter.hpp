#pragma once
/**
 * @file include/nikola/autonomy/ingestion_filter.hpp
 * @brief v0.2.2 Phase 3 — IngestionFilter: quality filtering for training data.
 *
 * Provides deduplication (SimHash), relevance scoring, content safety checks,
 * and size budgets.  Used by AutoIngestor before embedding each chunk.
 *
 * Design:
 *   · SimHash: 64-bit hash for near-duplicate detection (Hamming distance).
 *   · Relevance: optional callback scores chunk against active goals.
 *   · Safety: keyword blocklist for obvious harmful content.
 *   · Budget: daily ingestion volume cap.
 *   · Thread-safe: all state guarded by mutex.
 *
 * Phase: NIK-INGEST-03 (IngestionFilter, v0.2.2)
 */

#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

namespace nikola::autonomy {

// ============================================================================
// FilterVerdict — result of filtering a chunk
// ============================================================================

enum class FilterVerdict : uint8_t {
    ACCEPT,        ///< Chunk should be ingested
    REJECT_DUPLICATE,  ///< Near-duplicate of existing content
    REJECT_IRRELEVANT, ///< Below relevance threshold
    REJECT_UNSAFE,     ///< Failed safety check
    REJECT_BUDGET,     ///< Daily ingestion budget exhausted
    REJECT_EMPTY       ///< Empty or whitespace-only
};

[[nodiscard]] inline const char* verdict_name(FilterVerdict v) noexcept {
    switch (v) {
        case FilterVerdict::ACCEPT:            return "ACCEPT";
        case FilterVerdict::REJECT_DUPLICATE:  return "REJECT_DUPLICATE";
        case FilterVerdict::REJECT_IRRELEVANT: return "REJECT_IRRELEVANT";
        case FilterVerdict::REJECT_UNSAFE:     return "REJECT_UNSAFE";
        case FilterVerdict::REJECT_BUDGET:     return "REJECT_BUDGET";
        case FilterVerdict::REJECT_EMPTY:      return "REJECT_EMPTY";
        default:                               return "UNKNOWN";
    }
}

// ============================================================================
// IngestionFilterConfig
// ============================================================================

struct IngestionFilterConfig {
    /// Maximum Hamming distance for SimHash near-duplicate detection.
    /// Two chunks with SimHash Hamming distance ≤ this are considered duplicates.
    int max_hamming_distance = 3;

    /// Minimum relevance score [0, 1] for a chunk to be accepted.
    /// Only used if a relevance callback is set.  0.0 = accept everything.
    float min_relevance = 0.0f;

    /// Maximum total bytes ingested per day before budget rejection.
    size_t daily_byte_budget = 100 * 1024 * 1024;  // 100 MB

    /// Whether to run safety checks.
    bool enable_safety_check = true;

    /// Maximum number of SimHash entries to retain (memory bound).
    size_t max_hash_entries = 100000;
};

// ============================================================================
// IngestionFilterStats
// ============================================================================

struct IngestionFilterStats {
    size_t total_checked   = 0;
    size_t accepted        = 0;
    size_t rej_duplicate   = 0;
    size_t rej_irrelevant  = 0;
    size_t rej_unsafe      = 0;
    size_t rej_budget      = 0;
    size_t rej_empty       = 0;
};

// ============================================================================
// IngestionFilter
// ============================================================================

class IngestionFilter {
public:
    /// Relevance scoring callback: returns [0, 1] relevance for a chunk.
    using RelevanceFn = std::function<float(const std::string& chunk)>;

    explicit IngestionFilter(const IngestionFilterConfig& cfg = {});

    /// Set relevance scoring callback (optional).
    void set_relevance_fn(RelevanceFn fn) { relevance_fn_ = std::move(fn); }

    /// Check whether a chunk should be ingested.  Thread-safe.
    [[nodiscard]] FilterVerdict check(const std::string& chunk);

    /// Record that a chunk was ingested (adds its SimHash to the seen set).
    /// Call after successful ingestion.  Thread-safe.
    void record_ingested(const std::string& chunk);

    /// Compute 64-bit SimHash of text.
    [[nodiscard]] static uint64_t simhash(const std::string& text) noexcept;

    /// Hamming distance between two 64-bit hashes.
    [[nodiscard]] static int hamming_distance(uint64_t a, uint64_t b) noexcept;

    /// Check if text contains unsafe content patterns.
    [[nodiscard]] static bool is_unsafe(const std::string& text) noexcept;

    /// Current filter statistics.
    [[nodiscard]] IngestionFilterStats stats() const noexcept;

    /// Reset daily budget counter (call at midnight or on demand).
    void reset_daily_budget() noexcept;

    /// Reset all state (hashes + stats + budget).
    void reset() noexcept;

    /// Number of unique SimHash entries stored.
    [[nodiscard]] size_t hash_count() const noexcept;

private:
    IngestionFilterConfig cfg_;
    mutable std::mutex    mutex_;

    std::unordered_set<uint64_t> seen_hashes_;
    size_t                       daily_bytes_used_ = 0;
    IngestionFilterStats         stats_;

    RelevanceFn relevance_fn_;

    bool is_near_duplicate_(uint64_t hash) const noexcept;
};

} // namespace nikola::autonomy
