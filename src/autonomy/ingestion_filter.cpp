/**
 * @file src/autonomy/ingestion_filter.cpp
 * @brief v0.2.2 Phase 3 — IngestionFilter implementation.
 */

#include <nikola/autonomy/ingestion_filter.hpp>

#include <algorithm>
#include <cctype>
#include <cstring>

namespace nikola::autonomy {

// ============================================================================
// Constructor
// ============================================================================

IngestionFilter::IngestionFilter(const IngestionFilterConfig& cfg)
    : cfg_(cfg)
{}

// ============================================================================
// SimHash — 64-bit locality-sensitive hash for text
// ============================================================================

uint64_t IngestionFilter::simhash(const std::string& text) noexcept {
    // SimHash algorithm:
    // 1. Extract word-level shingles (bigrams)
    // 2. Hash each shingle to 64 bits
    // 3. For each bit position, accumulate +1 (bit=1) or -1 (bit=0)
    // 4. Final hash: bit=1 where sum > 0

    int counts[64] = {};

    // Simple FNV-1a hash for shingles
    auto fnv64 = [](const char* data, size_t len) -> uint64_t {
        uint64_t hash = 14695981039346656037ULL;
        for (size_t i = 0; i < len; ++i) {
            hash ^= static_cast<uint64_t>(static_cast<unsigned char>(data[i]));
            hash *= 1099511628211ULL;
        }
        return hash;
    };

    // Extract words (whitespace-delimited, lowercased)
    std::vector<std::string> words;
    std::string current;
    for (char c : text) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!current.empty()) {
                words.push_back(std::move(current));
                current.clear();
            }
        } else {
            current += static_cast<char>(
                std::tolower(static_cast<unsigned char>(c)));
        }
    }
    if (!current.empty()) words.push_back(std::move(current));

    if (words.empty()) return 0;

    // Use word bigrams as shingles (or single words if only 1)
    if (words.size() == 1) {
        uint64_t h = fnv64(words[0].data(), words[0].size());
        return h;  // Single word → use its hash directly
    }

    for (size_t i = 0; i + 1 < words.size(); ++i) {
        std::string shingle = words[i] + " " + words[i + 1];
        uint64_t h = fnv64(shingle.data(), shingle.size());
        for (int bit = 0; bit < 64; ++bit) {
            if (h & (1ULL << bit))
                counts[bit]++;
            else
                counts[bit]--;
        }
    }

    uint64_t result = 0;
    for (int bit = 0; bit < 64; ++bit) {
        if (counts[bit] > 0)
            result |= (1ULL << bit);
    }
    return result;
}

// ============================================================================
// Hamming distance
// ============================================================================

int IngestionFilter::hamming_distance(uint64_t a, uint64_t b) noexcept {
    return __builtin_popcountll(a ^ b);
}

// ============================================================================
// Safety check — keyword blocklist
// ============================================================================

bool IngestionFilter::is_unsafe(const std::string& text) noexcept {
    // Simple keyword-based safety check for obviously harmful content
    // that should not be used as training data.
    // This is a basic filter — not a comprehensive content moderation system.
    static const char* blocklist[] = {
        "how to make a bomb",
        "how to make explosives",
        "instructions for weapons",
        "synthesize drugs",
        "hack into",
        "exploit vulnerability",
        "sql injection attack",
        "denial of service attack",
        "create malware",
        "ransomware tutorial",
    };

    // Lowercase the text for matching
    std::string lower;
    lower.reserve(text.size());
    for (char c : text) {
        lower += static_cast<char>(
            std::tolower(static_cast<unsigned char>(c)));
    }

    for (const auto* keyword : blocklist) {
        if (lower.find(keyword) != std::string::npos) {
            return true;
        }
    }
    return false;
}

// ============================================================================
// is_near_duplicate_
// ============================================================================

bool IngestionFilter::is_near_duplicate_(uint64_t hash) const noexcept {
    for (uint64_t seen : seen_hashes_) {
        if (hamming_distance(hash, seen) <= cfg_.max_hamming_distance) {
            return true;
        }
    }
    return false;
}

// ============================================================================
// check
// ============================================================================

FilterVerdict IngestionFilter::check(const std::string& chunk) {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.total_checked++;

    // Empty check
    bool all_space = true;
    for (char c : chunk) {
        if (!std::isspace(static_cast<unsigned char>(c))) {
            all_space = false;
            break;
        }
    }
    if (chunk.empty() || all_space) {
        stats_.rej_empty++;
        return FilterVerdict::REJECT_EMPTY;
    }

    // Budget check
    if (daily_bytes_used_ + chunk.size() > cfg_.daily_byte_budget) {
        stats_.rej_budget++;
        return FilterVerdict::REJECT_BUDGET;
    }

    // Safety check
    if (cfg_.enable_safety_check && is_unsafe(chunk)) {
        stats_.rej_unsafe++;
        return FilterVerdict::REJECT_UNSAFE;
    }

    // Deduplication check
    uint64_t hash = simhash(chunk);
    if (is_near_duplicate_(hash)) {
        stats_.rej_duplicate++;
        return FilterVerdict::REJECT_DUPLICATE;
    }

    // Relevance check
    if (relevance_fn_ && cfg_.min_relevance > 0.0f) {
        float score = relevance_fn_(chunk);
        if (score < cfg_.min_relevance) {
            stats_.rej_irrelevant++;
            return FilterVerdict::REJECT_IRRELEVANT;
        }
    }

    stats_.accepted++;
    return FilterVerdict::ACCEPT;
}

// ============================================================================
// record_ingested
// ============================================================================

void IngestionFilter::record_ingested(const std::string& chunk) {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t hash = simhash(chunk);

    // Evict oldest if at capacity (simple: just clear half)
    if (seen_hashes_.size() >= cfg_.max_hash_entries) {
        // Simple eviction: clear and start fresh
        // In practice this is fine — SimHash is probabilistic anyway
        seen_hashes_.clear();
    }

    seen_hashes_.insert(hash);
    daily_bytes_used_ += chunk.size();
}

// ============================================================================
// stats / reset
// ============================================================================

IngestionFilterStats IngestionFilter::stats() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void IngestionFilter::reset_daily_budget() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    daily_bytes_used_ = 0;
}

void IngestionFilter::reset() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    seen_hashes_.clear();
    daily_bytes_used_ = 0;
    stats_ = {};
}

size_t IngestionFilter::hash_count() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return seen_hashes_.size();
}

} // namespace nikola::autonomy
