/**
 * @file holographic_lexicon.hpp
 * @brief Holographic Lexicon — O(1) Wave-to-Text and Text-to-Wave decoder.
 *
 * Implements the dual-index spectral LSH system from specification IMP-02:
 *   "Wave to Text Decoding Algorithm Research.md"
 *
 * Architecture
 * ============
 * - Forward map: token → canonical 9D waveform (for injection/embedding),
 *   backed by std::unordered_map<string, vector<Complex>>.
 * - Inverse index: SpectralHash → vector<string> (LSH bucket lookup),
 *   backed by std::unordered_map<SpectralHash, vector<string>>.
 * - Resonance verification: cosine similarity in complex space after bucket
 *   lookup to resolve collisions and filter weak matches.
 * - Multi-probe LSH: queries neighbour buckets for phases near quadrant
 *   boundaries, dramatically reducing false-negative rate.
 * - Thread-safe: shared_mutex allows parallel decode() calls to coexist
 *   with serialised add_token() writes.
 *
 * Complexity
 * ==========
 *   add_token  — O(1) average
 *   decode     — O(1) average (O(k) where k = mean bucket size ≈ 1)
 *   lookup     — O(1) average
 *
 * Phase 10, Nikola v0.0.4
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <mutex>
#include <numbers>
#include <optional>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace nikola::cognitive {

// ---------------------------------------------------------------------------
// Aliases
// ---------------------------------------------------------------------------

using Complex = std::complex<float>;

// --------------------------------------------------------------------------
// SpectralHash — 18-bit LSH key (9 dims × 2 bits)
// --------------------------------------------------------------------------

/**
 * @brief 18-bit Locality-Sensitive Hash derived from the phase quadrant of
 *        each of the 9 wavefunction dimensions.
 *
 * Phase quantisation schema (Table 2 of the spec):
 *   Q0 = [-π, -π/2)   → bits 00   (negative / inverted)
 *   Q1 = [-π/2, 0)    → bits 01   (transitioning)
 *   Q2 = [0,  +π/2)   → bits 10   (positive / aligned)
 *   Q3 = [+π/2, +π)   → bits 11   (transitioning)
 *
 * Bit layout:  hash bits [2d+1 : 2d] = quadrant of dimension d
 */
struct SpectralHash {
    uint64_t hash{0};  ///< Only bits 0..17 are meaningful (18 bits).

    bool operator==(const SpectralHash& o) const noexcept { return hash == o.hash; }

    /**
     * @brief Compute SpectralHash from a 9-element complex waveform.
     *
     * Dimensions beyond 9 (if any) are ignored. Missing dimensions are
     * treated as zero (→ phase 0 → Q2 = 10b).
     */
    [[nodiscard]] static SpectralHash from_wave(const std::vector<Complex>& w) noexcept
    {
        const int dims = static_cast<int>(std::min(w.size(), size_t{9}));
        uint64_t h = 0;
        for (int d = 0; d < dims; ++d) {
            const float phase = std::arg(w[static_cast<size_t>(d)]);  // ∈ [-π, π]
            // Normalise to [0, 1): (phase + π) / (2π)
            const float norm = (phase + std::numbers::pi_v<float>)
                             / (2.0f * std::numbers::pi_v<float>);
            // Quadrant ∈ {0,1,2,3}
            const auto q = static_cast<uint64_t>(norm * 4.0f) & 0x3u;
            h |= (q << (d * 2));
        }
        return SpectralHash{h};
    }

    /**
     * @brief Return the 2-bit quadrant for dimension @p d.
     */
    [[nodiscard]] uint32_t quadrant(int d) const noexcept
    {
        return static_cast<uint32_t>((hash >> (d * 2)) & 0x3u);
    }
};

// ---------------------------------------------------------------------------
// SpectralHash hasher for std::unordered_map
// ---------------------------------------------------------------------------

struct SpectralHashHasher {
    std::size_t operator()(SpectralHash k) const noexcept
    {
        // hash is already a good discriminator; mix it to avoid clustering
        return static_cast<std::size_t>(k.hash * 2654435761ULL);
    }
};

// ---------------------------------------------------------------------------
// HolographicLexicon
// ---------------------------------------------------------------------------

/**
 * @brief Thread-safe dual-index lexicon providing O(1) wave↔token lookup.
 *
 * Usage example:
 * @code
 *   HolographicLexicon lex;
 *   lex.add_token("hello", wave_hello);
 *   auto token = lex.decode(query_wave);   // returns optional<string>
 *   auto wave  = lex.embed("hello");       // returns optional<vector<Complex>>
 * @endcode
 */
class HolographicLexicon {
public:
    // ------------------------------------------------------------------
    // Configuration
    // ------------------------------------------------------------------

    /** Minimum cosine resonance to accept a match (prevents hallucination). */
    static constexpr double kMinResonance = 0.3;

    /** Noise-floor energy threshold; waves below this are treated as vacuum. */
    static constexpr double kNoiseFloor = 1e-9;

    /**
     * @brief Phase proximity threshold for multi-probe boundary detection.
     *
     * If a dimension's normalised phase is within this fraction of a quadrant
     * boundary (0.25, 0.5, 0.75), neighbouring hashes are also probed.
     */
    static constexpr float kBoundaryEpsilon = 0.05f;   // 5% of quadrant width

    // ------------------------------------------------------------------
    // Vocabulary management
    // ------------------------------------------------------------------

    /**
     * @brief Register a token with its canonical 9D waveform.
     *
     * Thread-safe (exclusive write lock).  Re-registering an existing token
     * updates both the waveform and the inverse-index bucket.
     *
     * @throws std::invalid_argument if @p token is empty or @p wave has fewer
     *         than 1 element.
     */
    void add_token(const std::string& token, const std::vector<Complex>& wave)
    {
        if (token.empty()) throw std::invalid_argument("HolographicLexicon: empty token");
        if (wave.empty()) throw std::invalid_argument("HolographicLexicon: empty waveform");

        const SpectralHash h = SpectralHash::from_wave(wave);

        std::unique_lock lock(mutex_);

        // If token was previously registered under a different hash, remove
        // the stale inverse-index entry.
        if (auto it = forward_map_.find(token); it != forward_map_.end()) {
            const SpectralHash old_h = SpectralHash::from_wave(it->second);
            if (old_h.hash != h.hash) remove_from_bucket(old_h, token);
        }

        forward_map_[token] = wave;
        add_to_bucket(h, token);
    }

    /**
     * @brief Remove a token from the lexicon entirely.
     * @return true if the token was found and removed.
     */
    bool remove_token(const std::string& token)
    {
        std::unique_lock lock(mutex_);
        auto it = forward_map_.find(token);
        if (it == forward_map_.end()) return false;
        const SpectralHash h = SpectralHash::from_wave(it->second);
        remove_from_bucket(h, token);
        forward_map_.erase(it);
        return true;
    }

    /** @brief Return true if @p token is currently registered. */
    [[nodiscard]] bool exists(const std::string& token) const
    {
        std::shared_lock lock(mutex_);
        return forward_map_.count(token) > 0;
    }

    /** @brief Number of tokens currently registered. */
    [[nodiscard]] std::size_t size() const
    {
        std::shared_lock lock(mutex_);
        return forward_map_.size();
    }

    /** @brief Remove all tokens. */
    void clear()
    {
        std::unique_lock lock(mutex_);
        forward_map_.clear();
        inverse_index_.clear();
    }

    // ------------------------------------------------------------------
    // Forward lookup (text → wave)
    // ------------------------------------------------------------------

    /**
     * @brief Retrieve the canonical waveform for @p token.
     * @return The registered wave, or nullopt if not found.
     */
    [[nodiscard]] std::optional<std::vector<Complex>> embed(const std::string& token) const
    {
        std::shared_lock lock(mutex_);
        if (auto it = forward_map_.find(token); it != forward_map_.end())
            return it->second;
        return std::nullopt;
    }

    // ------------------------------------------------------------------
    // Inverse decode (wave → text) — the main decoding algorithm
    // ------------------------------------------------------------------

    /**
     * @brief Decode a 9D query waveform to its closest registered token.
     *
     * Algorithm (O(1) amortised):
     *  1. Vacuum check — reject near-zero energy waves immediately.
     *  2. Compute primary SpectralHash.
     *  3. Multi-probe — add neighbour hashes for boundary-adjacent dims.
     *  4. Union of candidate buckets.
     *  5. Resonance verification — cosine similarity in complex space.
     *  6. Confidence gate — reject if best resonance < kMinResonance.
     *
     * @param query_wave  9D (or fewer) complex observation vector.
     * @return Best-matching token, or nullopt on vacuum/miss/low-confidence.
     */
    [[nodiscard]] std::optional<std::string> decode(const std::vector<Complex>& query_wave) const
    {
        // 1. Vacuum check
        if (wave_energy(query_wave) < kNoiseFloor) return std::nullopt;

        std::shared_lock lock(mutex_);
        if (forward_map_.empty()) return std::nullopt;

        // 2. Primary hash
        const SpectralHash primary = SpectralHash::from_wave(query_wave);

        // 3. Multi-probe: collect candidate hashes
        const auto probe_hashes = multi_probe_hashes(query_wave, primary);

        // 4. Gather candidate tokens (deduplicated via seen set)
        std::vector<const std::string*> candidates;
        candidates.reserve(8);
        for (SpectralHash ph : probe_hashes) {
            if (auto it = inverse_index_.find(ph); it != inverse_index_.end()) {
                for (const std::string& t : it->second) {
                    // Simple dedup: check pointer identity vs existing entries
                    bool already = false;
                    for (const std::string* p : candidates)
                        if (p == &t) { already = true; break; }
                    if (!already) candidates.push_back(&t);
                }
            }
        }
        if (candidates.empty()) return std::nullopt;

        // 5. Resonance verification
        double best_r   = -1.0;
        std::string best_token;
        for (const std::string* tp : candidates) {
            if (auto it = forward_map_.find(*tp); it != forward_map_.end()) {
                const double r = compute_resonance(query_wave, it->second);
                if (r > best_r) { best_r = r; best_token = *tp; }
            }
        }

        // 6. Confidence gate
        if (best_r < kMinResonance) return std::nullopt;
        return best_token;
    }

    // ------------------------------------------------------------------
    // Exposed helpers (useful for tests and diagnostics)
    // ------------------------------------------------------------------

    /**
     * @brief Compute cosine similarity between two complex waveforms.
     *
     * R = |Ψ_a · Ψ_b*| / (‖Ψ_a‖ · ‖Ψ_b‖)  ∈ [0, 1]
     *
     * Returns 0.0 on zero-energy inputs.
     */
    [[nodiscard]] static double compute_resonance(
        const std::vector<Complex>& a,
        const std::vector<Complex>& b) noexcept
    {
        const size_t n = std::min(a.size(), b.size());
        Complex dot{0.0f, 0.0f};
        double norm_a = 0.0;
        double norm_b = 0.0;
        for (size_t i = 0; i < n; ++i) {
            dot    += a[i] * std::conj(b[i]);
            norm_a += std::norm(a[i]);
            norm_b += std::norm(b[i]);
        }
        if (norm_a < 1e-9 || norm_b < 1e-9) return 0.0;
        return static_cast<double>(std::abs(dot))
             / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }

    /**
     * @brief Compute the total energy of a waveform vector.
     */
    [[nodiscard]] static double wave_energy(const std::vector<Complex>& w) noexcept
    {
        double e = 0.0;
        for (const auto& c : w) e += std::norm(c);
        return e;
    }

private:
    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    void add_to_bucket(SpectralHash h, const std::string& token)
    {
        auto& bucket = inverse_index_[h];
        if (std::find(bucket.begin(), bucket.end(), token) == bucket.end())
            bucket.push_back(token);
    }

    void remove_from_bucket(SpectralHash h, const std::string& token)
    {
        if (auto it = inverse_index_.find(h); it != inverse_index_.end()) {
            auto& bkt = it->second;
            bkt.erase(std::remove(bkt.begin(), bkt.end(), token), bkt.end());
            if (bkt.empty()) inverse_index_.erase(it);
        }
    }

    /**
     * @brief Generate primary hash plus up to one neighbour hash per
     *        boundary-adjacent dimension (Multi-Probe LSH).
     *
     * A dimension is "boundary adjacent" when its normalised phase is within
     * kBoundaryEpsilon of a quadrant boundary (0.25, 0.5, 0.75 of [0,1]).
     */
    [[nodiscard]] static std::vector<SpectralHash>
    multi_probe_hashes(const std::vector<Complex>& w, SpectralHash primary) noexcept
    {
        std::vector<SpectralHash> result;
        result.reserve(4);
        result.push_back(primary);

        const int dims = static_cast<int>(std::min(w.size(), size_t{9}));
        for (int d = 0; d < dims; ++d) {
            const float phase = std::arg(w[static_cast<size_t>(d)]);
            const float norm = (phase + std::numbers::pi_v<float>)
                             / (2.0f * std::numbers::pi_v<float>);
            // Fractional part within the current quadrant cell [0,1)
            const float frac = std::fmod(norm * 4.0f, 1.0f);

            // Near upper boundary of this quadrant (frac ≈ 1)?
            if (frac > (1.0f - kBoundaryEpsilon)) {
                // Probe with quadrant incremented by 1 (wrap at 3)
                const uint64_t cur_q = (primary.hash >> (d * 2)) & 0x3u;
                const uint64_t next_q = (cur_q + 1) & 0x3u;
                uint64_t alt = primary.hash & ~(uint64_t{0x3} << (d * 2));
                alt |= (next_q << (d * 2));
                result.push_back(SpectralHash{alt});
            }
            // Near lower boundary (frac ≈ 0)?
            else if (frac < kBoundaryEpsilon) {
                const uint64_t cur_q = (primary.hash >> (d * 2)) & 0x3u;
                const uint64_t prev_q = (cur_q + 3) & 0x3u;  // -1 mod 4
                uint64_t alt = primary.hash & ~(uint64_t{0x3} << (d * 2));
                alt |= (prev_q << (d * 2));
                result.push_back(SpectralHash{alt});
            }
        }
        return result;
    }

    // ------------------------------------------------------------------
    // Data members
    // ------------------------------------------------------------------

    mutable std::shared_mutex mutex_;

    /// Forward mapping: token → canonical 9D waveform.
    std::unordered_map<std::string, std::vector<Complex>> forward_map_;

    /// Inverse index: SpectralHash → candidate tokens (LSH bucket).
    std::unordered_map<SpectralHash, std::vector<std::string>, SpectralHashHasher>
        inverse_index_;
};

} // namespace nikola::cognitive
