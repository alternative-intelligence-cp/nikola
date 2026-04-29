/**
 * @file identity/self_concept_vector.hpp
 * @brief GAP-M1 (Part 1) — SelfConceptVector: physics-ready identity embedding.
 *
 * Bridges the JSON-based IdentityProfile to a dense 128-D real-valued vector
 * that can be projected onto the toroidal manifold as a refractive bias field.
 *
 * Encoding scheme (deterministic, no external model):
 *   1. Each preference (topic → affinity) is hashed to a position in [0, 128)
 *   2. Its affinity value is accumulated at that position with cosine spread
 *   3. The result is L2-normalized to unit norm
 *
 * The 128-D dimension matches the Nit[128] embedding used by HolographicInjector,
 * enabling direct comparison between identity alignment and injected content.
 *
 * Update rule:
 *   SCV_{t+1} = normalize(SCV_t + lr × δ)
 *
 * where δ is a sparse preference delta vector and lr is the learning rate.
 *
 * Reference:
 *   Integration Report §21.4 (SelfConceptVector)
 *   RELEASE_0.3.x.md GAP-M1
 *
 * Header-only — no separate .cpp needed.
 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <numbers>
#include <string>
#include <vector>

#include <nikola/interior/identity_manager.hpp>

namespace nikola::identity {

// ============================================================================
// Constants
// ============================================================================

/// SCV dimensionality — matches HolographicInjector Nit[128] embedding.
inline constexpr int SCV_DIM = 128;

/// Default learning rate for SCV evolution.
inline constexpr double SCV_LEARN_RATE = 0.05;

/// Cosine spread half-width (positions affected per preference hash).
inline constexpr int SCV_SPREAD = 4;

/// Minimum L2 norm before normalization (prevents division by zero).
inline constexpr double SCV_MIN_NORM = 1e-8;

// ============================================================================
// SelfConceptVector
// ============================================================================

/**
 * @brief Dense 128-D identity embedding derived from IdentityProfile preferences.
 *
 * The SCV encodes "who Nikola is" as a unit-norm vector in the same space
 * as injected content embeddings.  This allows the physics engine to compute
 * alignment between incoming stimuli and core identity via dot product.
 *
 * Static-only utility class plus a lightweight data container.
 */
class SelfConceptVector {
public:
    using Vec = std::array<double, SCV_DIM>;

    // ── Construction ────────────────────────────────────────────────────

    SelfConceptVector() { vec_.fill(0.0); }

    /// Access the raw vector.
    [[nodiscard]] const Vec& vec() const noexcept { return vec_; }
    [[nodiscard]] Vec&       vec()       noexcept { return vec_; }

    /// L2 norm of the vector.
    [[nodiscard]] double norm() const noexcept {
        double sum = 0.0;
        for (double v : vec_) sum += v * v;
        return std::sqrt(sum);
    }

    /// Normalize to unit length. No-op if near-zero.
    void normalize() noexcept {
        double n = norm();
        if (n < SCV_MIN_NORM) return;
        for (double& v : vec_) v /= n;
    }

    // ── Construction from IdentityProfile ───────────────────────────────

    /**
     * @brief Build SCV from an IdentityProfile's preferences.
     *
     * Each preference (topic → affinity) is projected into the 128-D space
     * using a deterministic hash + cosine spread.  The result is normalized.
     *
     * @param profile  The identity profile to encode.
     * @return         Unit-norm SCV (or zero if no preferences).
     */
    [[nodiscard]] static SelfConceptVector from_profile(
        const interior::IdentityProfile& profile)
    {
        SelfConceptVector scv;
        for (const auto& [topic, affinity] : profile.preferences) {
            accumulate_preference(scv.vec_, topic, affinity);
        }
        scv.normalize();
        return scv;
    }

    // ── Evolution ───────────────────────────────────────────────────────

    /**
     * @brief Evolve SCV toward a new experience.
     *
     * SCV_{t+1} = normalize(SCV_t + lr × δ)
     *
     * @param delta  Sparse preference delta (topic → affinity change)
     * @param lr     Learning rate (default: SCV_LEARN_RATE)
     */
    void evolve(const std::map<std::string, double>& delta,
                double lr = SCV_LEARN_RATE) {
        for (const auto& [topic, affinity] : delta) {
            accumulate_preference(vec_, topic, affinity * lr);
        }
        normalize();
    }

    // ── Alignment measurement ───────────────────────────────────────────

    /**
     * @brief Cosine similarity between this SCV and a Nit[128] embedding.
     *
     * Maps Nit values [-4, +4] to [-1, +1] for comparison.
     * Returns 0.0 if either vector is zero.
     *
     * @param nits  128-element balanced nonary vector
     * @return      Cosine similarity in [-1, +1]
     */
    [[nodiscard]] double alignment(const std::vector<int8_t>& nits) const noexcept {
        if (nits.size() != SCV_DIM) return 0.0;

        double dot = 0.0, norm_nit = 0.0, norm_scv = 0.0;
        for (int i = 0; i < SCV_DIM; ++i) {
            double ni = static_cast<double>(nits[i]) / 4.0;  // [-4,+4] → [-1,+1]
            dot      += vec_[i] * ni;
            norm_nit += ni * ni;
            norm_scv += vec_[i] * vec_[i];
        }
        double denom = std::sqrt(norm_scv * norm_nit);
        if (denom < SCV_MIN_NORM) return 0.0;
        return dot / denom;
    }

    /**
     * @brief Dot product between two SCVs.
     *
     * For unit-norm SCVs this equals cosine similarity.
     */
    [[nodiscard]] double dot(const SelfConceptVector& other) const noexcept {
        double d = 0.0;
        for (int i = 0; i < SCV_DIM; ++i)
            d += vec_[i] * other.vec_[i];
        return d;
    }

private:
    Vec vec_;

    // ── Hashing ─────────────────────────────────────────────────────────

    /**
     * @brief Deterministic string hash to a position in [0, SCV_DIM).
     *
     * Uses FNV-1a for uniformity and speed.
     */
    [[nodiscard]] static int hash_topic(const std::string& topic) noexcept {
        uint64_t h = 14695981039346656037ULL;  // FNV offset basis
        for (char c : topic) {
            h ^= static_cast<uint64_t>(static_cast<unsigned char>(c));
            h *= 1099511628211ULL;  // FNV prime
        }
        return static_cast<int>(h % SCV_DIM);
    }

    /**
     * @brief Accumulate a preference into a vector with cosine spread.
     *
     * The affinity is distributed across SCV_SPREAD * 2 + 1 positions
     * centered at the hash position, weighted by cos(π·d/(2·SPREAD)).
     */
    static void accumulate_preference(Vec& v,
                                       const std::string& topic,
                                       double affinity) noexcept {
        int center = hash_topic(topic);
        for (int d = -SCV_SPREAD; d <= SCV_SPREAD; ++d) {
            int idx = ((center + d) % SCV_DIM + SCV_DIM) % SCV_DIM;
            double weight = std::cos(std::numbers::pi * d / (2.0 * SCV_SPREAD));
            v[idx] += affinity * weight;
        }
    }
};

} // namespace nikola::identity
