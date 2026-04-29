#pragma once
/**
 * @file resonance_inverted_index.hpp
 * @brief v0.3.4 — GAP-M3 Resonance Inverted Index (RII).
 *
 * Content-addressable memory index mapping resonance signatures to spatial
 * locations (Hilbert keys / node ids). The index is intentionally header-only
 * and lightweight for deterministic unit testing and low-latency lookup paths.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace nikola::memory {

inline constexpr std::size_t RII_SIGNATURE_DIM = 9;
inline constexpr float       RII_MIN_NORM      = 1.0e-8f;

using ResonanceSignature = std::array<float, RII_SIGNATURE_DIM>;

struct ResonanceRecord {
    uint64_t           location{0};
    ResonanceSignature signature{};
    float              resonance{0.0f};
    uint64_t           tick{0};
};

struct ResonanceHit {
    uint64_t location{0};
    float    score{0.0f};
    float    cosine{0.0f};
    float    resonance{0.0f};
    uint64_t tick{0};
};

class ResonanceInvertedIndex {
public:
    explicit ResonanceInvertedIndex(std::size_t probe_hamming_radius = 1)
        : probe_hamming_radius_(probe_hamming_radius) {}

    [[nodiscard]] std::size_t size() const noexcept { return records_.size(); }
    [[nodiscard]] bool empty() const noexcept { return records_.empty(); }

    void clear() {
        records_.clear();
        buckets_.clear();
    }

    void upsert(uint64_t location,
                const ResonanceSignature& signature,
                float resonance,
                uint64_t tick)
    {
        if (!std::isfinite(resonance) || resonance < 0.0f) {
            throw std::invalid_argument("ResonanceInvertedIndex::upsert: resonance must be finite and >= 0");
        }

        ResonanceSignature norm_sig = normalize(signature);
        const uint16_t new_bucket = bucket_key(norm_sig);

        auto it = records_.find(location);
        if (it != records_.end()) {
            const uint16_t old_bucket = bucket_key(it->second.signature);
            if (old_bucket != new_bucket) {
                erase_from_bucket(old_bucket, location);
            }
        }

        records_[location] = ResonanceRecord{location, norm_sig, resonance, tick};
        buckets_[new_bucket].push_back(location);
    }

    [[nodiscard]] bool contains(uint64_t location) const noexcept {
        return records_.find(location) != records_.end();
    }

    [[nodiscard]] bool try_get(uint64_t location, ResonanceRecord& out) const {
        auto it = records_.find(location);
        if (it == records_.end()) return false;
        out = it->second;
        return true;
    }

    [[nodiscard]] std::vector<ResonanceHit> query(
        const ResonanceSignature& signature,
        std::size_t top_k = 8,
        float min_cosine = 0.0f) const
    {
        if (!std::isfinite(min_cosine) || min_cosine < -1.0f || min_cosine > 1.0f) {
            throw std::invalid_argument("ResonanceInvertedIndex::query: min_cosine must be finite and in [-1, 1]");
        }
        if (top_k == 0 || records_.empty()) return {};

        const ResonanceSignature q = normalize(signature);
        const uint16_t qkey = bucket_key(q);

        std::unordered_set<uint64_t> candidate_ids;
        candidate_ids.reserve(64);

        for (uint16_t key : probed_keys(qkey)) {
            auto bit = buckets_.find(key);
            if (bit == buckets_.end()) continue;
            for (uint64_t id : bit->second) {
                if (records_.find(id) != records_.end()) {
                    candidate_ids.insert(id);
                }
            }
        }

        std::vector<ResonanceHit> hits;
        hits.reserve(candidate_ids.size());

        for (uint64_t id : candidate_ids) {
            const auto rit = records_.find(id);
            if (rit == records_.end()) continue;
            const auto& rec = rit->second;

            const float c = cosine_similarity(q, rec.signature);
            if (c < min_cosine) continue;

            // Blend angular similarity + resonance strength.
            const float score = 0.80f * c + 0.20f * rec.resonance;
            hits.push_back(ResonanceHit{rec.location, score, c, rec.resonance, rec.tick});
        }

        std::sort(hits.begin(), hits.end(), [](const ResonanceHit& a, const ResonanceHit& b) {
            if (a.score != b.score) return a.score > b.score;
            if (a.cosine != b.cosine) return a.cosine > b.cosine;
            return a.tick > b.tick;
        });

        if (hits.size() > top_k) hits.resize(top_k);
        return hits;
    }

    [[nodiscard]] std::size_t bucket_population(uint16_t key) const {
        auto it = buckets_.find(key);
        if (it == buckets_.end()) return 0;

        std::size_t live = 0;
        for (uint64_t id : it->second) {
            if (records_.find(id) != records_.end()) ++live;
        }
        return live;
    }

    [[nodiscard]] static ResonanceSignature normalize(const ResonanceSignature& sig) {
        double n2 = 0.0;
        for (float x : sig) n2 += static_cast<double>(x) * static_cast<double>(x);

        if (n2 <= static_cast<double>(RII_MIN_NORM)) {
            ResonanceSignature out{};
            out[0] = 1.0f;
            return out;
        }

        const float invn = static_cast<float>(1.0 / std::sqrt(n2));
        ResonanceSignature out{};
        for (std::size_t i = 0; i < RII_SIGNATURE_DIM; ++i) out[i] = sig[i] * invn;
        return out;
    }

    [[nodiscard]] static float cosine_similarity(const ResonanceSignature& a,
                                                 const ResonanceSignature& b) noexcept
    {
        float d = 0.0f;
        for (std::size_t i = 0; i < RII_SIGNATURE_DIM; ++i) d += a[i] * b[i];
        return std::clamp(d, -1.0f, 1.0f);
    }

    [[nodiscard]] static uint16_t bucket_key(const ResonanceSignature& s) noexcept {
        uint16_t key = 0;
        for (std::size_t i = 0; i < RII_SIGNATURE_DIM; ++i) {
            if (s[i] >= 0.0f) key |= static_cast<uint16_t>(1u << i);
        }
        return key;
    }

private:
    void erase_from_bucket(uint16_t key, uint64_t location) {
        auto bit = buckets_.find(key);
        if (bit == buckets_.end()) return;

        auto& vec = bit->second;
        vec.erase(std::remove(vec.begin(), vec.end(), location), vec.end());
        if (vec.empty()) buckets_.erase(bit);
    }

    [[nodiscard]] std::vector<uint16_t> probed_keys(uint16_t key) const {
        std::vector<uint16_t> keys;
        keys.push_back(key);

        if (probe_hamming_radius_ == 0) return keys;

        // Radius-1 neighbors by single-bit flips.
        for (std::size_t i = 0; i < RII_SIGNATURE_DIM; ++i) {
            keys.push_back(static_cast<uint16_t>(key ^ (1u << i)));
        }

        if (probe_hamming_radius_ > 1) {
            for (std::size_t i = 0; i < RII_SIGNATURE_DIM; ++i) {
                for (std::size_t j = i + 1; j < RII_SIGNATURE_DIM; ++j) {
                    keys.push_back(static_cast<uint16_t>(key ^ (1u << i) ^ (1u << j)));
                }
            }
        }

        return keys;
    }

private:
    std::size_t probe_hamming_radius_{1};
    std::unordered_map<uint64_t, ResonanceRecord> records_;
    std::unordered_map<uint16_t, std::vector<uint64_t>> buckets_;
};

} // namespace nikola::memory
