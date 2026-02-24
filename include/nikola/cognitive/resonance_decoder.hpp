#pragma once
/**
 * @file cognitive/resonance_decoder.hpp
 * @brief ResonanceDecoder — maps CognitiveTorus resonance state → token sequence.
 *
 * Architecture (Path B output side):
 *
 *   CognitiveTorus::hot_nodes(k)       → k peak-intensity node indices
 *   CognitiveTorus::node_wave9d(idx)   → 9D complex waveform at each hot node
 *   HolographicLexicon::decode(wave9d) → optional<string> (LSH + cosine verify)
 *   Accumulate unique tokens           → response text
 *
 * Pre-population:
 *   The decoder owns a HolographicLexicon.  Before decoding, register tokens:
 *
 *   @code
 *   ResonanceDecoder dec;
 *
 *   // Manual: register a wave directly
 *   dec.register_token("hello", my_wave9d);
 *
 *   // ONNX-assisted: embed and register from embedder
 *   dec.register_from_embedder(embedder, "hello");
 *
 *   // Batch: register a whole vocabulary
 *   dec.register_vocabulary(embedder, {"hello", "world", "nikola", ...});
 *   @endcode
 *
 *   After ≥ 1 registration, call decode() / decode_text() on a post-step torus.
 *
 * Wave encoding convention:
 *   For each vocab token T, the canonical 9D wave is derived from the first 9
 *   floats of T's NonaryEmbedder embedding, converted to Complex via
 *   Complex(f[d], 0).  This encodes the semantic centroid of T in the 9 primary
 *   torus dimensions.
 *
 * Spec reference: docs/info/engineering/03_cognitive_systems.txt §3.4.2
 */

#include <nikola/cognitive/cognitive_torus.hpp>
#include <nikola/cognitive/holographic_lexicon.hpp>

#include <algorithm>
#include <complex>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#ifdef NIKOLA_HAS_ORT
#  include <nikola/cognitive/nonary_embedder.hpp>
#endif

namespace nikola::cognitive {

// ============================================================================
// ResonanceDecoder
// ============================================================================

/**
 * @brief Reads torus resonance state and maps it to a token sequence.
 *
 * The decoder is stateless with respect to the torus — it does not modify the
 * CognitiveTorus passed to decode().  The internal HolographicLexicon is the
 * only mutable state and is only changed via register_* methods.
 */
class ResonanceDecoder {
public:
    ResonanceDecoder() = default;

    // ------------------------------------------------------------------
    // Vocabulary registration
    // ------------------------------------------------------------------

    /**
     * @brief Register a token with its canonical 9D wave directly.
     *
     * @param token   Vocabulary entry (e.g. "hello", "world").
     * @param wave9d  9-element complex waveform (must have size ≥ 1; padded to
     *                9 if shorter).
     *
     * Thread safety: NOT safe to call concurrently with decode().
     */
    void register_token(const std::string& token,
                        const std::vector<Complex>& wave9d) {
        // Ensure wave has exactly 9 elements for consistent LSH hashing.
        std::vector<Complex> w = wave9d;
        while (w.size() < 9) w.emplace_back(0.f, 0.f);
        w.resize(9);
        lexicon_.add_token(token, w);
    }

#ifdef NIKOLA_HAS_ORT
    /**
     * @brief Embed a token string and register its waveform automatically.
     *
     * Extracts the first 9 float values of the BERT-Tiny embedding, converts
     * them to pure-real Complex values, and stores them in the lexicon.
     *
     * @param embedder  NonaryEmbedder (already initialised with ONNX paths).
     * @param token     Vocabulary entry to embed and register.
     */
    void register_from_embedder(const NonaryEmbedder& embedder,
                                const std::string& token) {
        auto floats = embedder.embed_float(token);
        std::vector<Complex> wave9d;
        wave9d.reserve(9);
        for (int d = 0; d < 9; ++d) {
            float val = (d < static_cast<int>(floats.size())) ? floats[static_cast<size_t>(d)] : 0.f;
            wave9d.emplace_back(val, 0.f);
        }
        lexicon_.add_token(token, wave9d);
    }

    /**
     * @brief Register all tokens in a vocabulary list via the embedder.
     *
     * Iterates the vocab and calls register_from_embedder() for each entry.
     *
     * @param embedder  NonaryEmbedder instance.
     * @param vocab     List of tokens to register.
     */
    void register_vocabulary(const NonaryEmbedder& embedder,
                             const std::vector<std::string>& vocab) {
        for (const auto& tok : vocab)
            register_from_embedder(embedder, tok);
    }
#endif

    // ------------------------------------------------------------------
    // Decoding
    // ------------------------------------------------------------------

    /**
     * @brief Decode the CognitiveTorus resonance state into a token list.
     *
     * Algorithm:
     *   1. Extract top_k hot nodes by |ψ|².
     *   2. For each hot node, extract its 9D neighbourhood waveform.
     *   3. Query the HolographicLexicon via LSH + cosine verification.
     *   4. Collect unique matched tokens (first match per hot node).
     *
     * @param torus   Torus to read from (post-run state).
     * @param top_k   Number of hot nodes to probe (default 20).
     * @return        Vector of matched token strings (may be empty).
     */
    [[nodiscard]]
    std::vector<std::string> decode(const CognitiveTorus& torus,
                                    size_t top_k = 20) const {
        auto hot = torus.hot_nodes(top_k);
        std::vector<std::string> results;
        std::unordered_set<std::string> seen;

        for (size_t idx : hot) {
            // node_wave9d returns std::complex<float>, same type as Complex.
            auto raw_wave = torus.node_wave9d(idx);
            std::vector<Complex> wave(raw_wave.cbegin(), raw_wave.cend());

            auto match = lexicon_.decode(wave);
            if (match && !seen.count(*match)) {
                results.push_back(*match);
                seen.insert(*match);
            }
        }
        return results;
    }

    /**
     * @brief Decode and join matched tokens into a single string.
     *
     * Tokens are space-separated.  Returns empty string if no match found.
     *
     * @param torus   Post-run CognitiveTorus to decode.
     * @param top_k   Hot nodes to probe (default 20).
     */
    [[nodiscard]]
    std::string decode_text(const CognitiveTorus& torus,
                            size_t top_k = 20) const {
        auto tokens = decode(torus, top_k);
        std::string out;
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (i) out += ' ';
            out += tokens[i];
        }
        return out;
    }

    // ------------------------------------------------------------------
    // Introspection
    // ------------------------------------------------------------------

    /// Number of tokens currently registered in the lexicon.
    size_t vocab_size() const noexcept { return lexicon_.size(); }

    /// Read-only access to the underlying lexicon.
    const HolographicLexicon& lexicon() const noexcept { return lexicon_; }

private:
    HolographicLexicon lexicon_;
};

} // namespace nikola::cognitive
