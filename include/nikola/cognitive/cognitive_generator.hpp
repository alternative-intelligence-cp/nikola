/**
 * @file cognitive_generator.hpp
 * @brief CognitiveGenerator — peak-soliton scanner and thought-to-token emitter.
 *
 * Implements COG-05 from the Nikola v0.0.4 design:
 *   "Wave to Text Decoding Algorithm Research.md" §4.2
 *
 * Lifecycle per scan() call
 * =========================
 *  1. Scan all active grid nodes for the node with highest cognitive energy:
 *       E_cog(i) = |Ψ_i|² × r_i
 *  2. If the peak energy is below the configured threshold, do nothing.
 *  3. Extract the 9-dimensional spectral signature of the peak node.
 *  4. Decode via HolographicLexicon::decode():
 *     a. Hit  → push token to output queue; apply Inhibition of Return.
 *     b. Miss → if minting is enabled and persistence count reached,
 *                create a new NEO_CONCEPT_XXXX token, register it in the
 *                lexicon, push it to the output queue.
 *  5. Inhibition of Return: inject -Ψ_peak at the peak node to clear that
 *     soliton from working memory and let the next thought emerge.
 *
 * Spectral Signature Encoding
 * ===========================
 * The 9D signature for node i is built by applying a deterministic per-
 * dimension phase rotation to Ψ_i:
 *
 *   sig[d] = Ψ_i × exp(i × d × π/9)    d ∈ {0, … , 8}
 *
 * This gives each registered token a unique spectral fingerprint derivable
 * from its wavefunction amplitude and phase, while ensuring that a token
 * registered with the same node psi value will always decode to itself.
 *
 * Thread Safety
 * =============
 * Not thread-safe in itself — scan() modifies the WaveFunction.
 * pop_token() is safe to call from a different thread if scan() has finished.
 *
 * Phase 11, Nikola v0.0.4
 */

#pragma once

#include <nikola/cognitive/holographic_lexicon.hpp>
#include <nikola/physics/wave_function.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <mutex>
#include <numbers>
#include <optional>
#include <string>
#include <vector>

// For hex formatting NEO_CONCEPT ids without <format> (C++20 portable)
#include <cstdio>
#include <cstring>

namespace nikola::cognitive {

// ---------------------------------------------------------------------------
// ConceptMinter — generates novel vocabulary tokens for orphan solitons
// ---------------------------------------------------------------------------

/**
 * @brief Mints unique "NEO_CONCEPT_XXXX" identifiers for waves that have no
 *        match in the HolographicLexicon, and registers them so future
 *        encounters return the same token.
 */
class ConceptMinter {
public:
    explicit ConceptMinter(HolographicLexicon& lex) : lex_(lex) {}

    /**
     * @brief Mint a new concept token for @p wave and register it in the lexicon.
     * @return The newly minted token string, e.g. "NEO_CONCEPT_0A3F".
     */
    std::string mint(const std::vector<Complex>& wave)
    {
        const uint32_t id = next_id_.fetch_add(1, std::memory_order_relaxed);
        char buf[32];
        std::snprintf(buf, sizeof(buf), "NEO_CONCEPT_%04X", id & 0xFFFFu);
        const std::string token(buf);
        lex_.add_token(token, wave);
        return token;
    }

    /** @brief Number of concepts minted so far. */
    [[nodiscard]] uint32_t count() const noexcept
    {
        return next_id_.load(std::memory_order_relaxed);
    }

private:
    HolographicLexicon&      lex_;
    std::atomic<uint32_t>    next_id_{0};
};

// ---------------------------------------------------------------------------
// CognitiveGenerator
// ---------------------------------------------------------------------------

class CognitiveGenerator {
public:
    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    /**
     * @brief Construct a CognitiveGenerator backed by @p lexicon.
     *
     * The lexicon must outlive this generator.
     */
    explicit CognitiveGenerator(HolographicLexicon& lexicon)
        : lexicon_(lexicon)
        , minter_(lexicon)
    {}

    // ------------------------------------------------------------------
    // Configuration
    // ------------------------------------------------------------------

    /**
     * @brief Minimum cognitive energy for a node to be considered a peak.
     *        Nodes below this threshold are treated as noise.
     * Default: 1e-12 (very sensitive; raise for noisy grids).
     */
    void set_energy_threshold(float thr) noexcept { energy_threshold_ = thr; }
    [[nodiscard]] float energy_threshold() const noexcept { return energy_threshold_; }

    /**
     * @brief Enable/disable automatic concept minting on decode misses.
     * Default: true.
     */
    void set_minting_enabled(bool e) noexcept { minting_enabled_ = e; }
    [[nodiscard]] bool minting_enabled() const noexcept { return minting_enabled_; }

    /**
     * @brief Number of consecutive scans a missed peak must persist before
     *        the ConceptMinter fires.  1 = mint immediately on first miss.
     * Default: 1.
     */
    void set_persistence_count(int n) noexcept { persistence_count_ = std::max(1, n); }
    [[nodiscard]] int persistence_count() const noexcept { return persistence_count_; }

    /**
     * @brief If true, apply Inhibition of Return (inject -Ψ) after each
     *        decoded token.  Disable for replay/diagnostic use.
     * Default: true.
     */
    void set_inhibition_enabled(bool e) noexcept { inhibition_enabled_ = e; }

    // ------------------------------------------------------------------
    // Statistics
    // ------------------------------------------------------------------

    [[nodiscard]] size_t tokens_emitted()   const noexcept { return tokens_emitted_; }
    [[nodiscard]] size_t concepts_minted()  const noexcept { return minter_.count(); }
    [[nodiscard]] size_t queue_size()       const noexcept {
        std::lock_guard<std::mutex> g(queue_mutex_);
        return output_queue_.size();
    }

    // ------------------------------------------------------------------
    // Main interface
    // ------------------------------------------------------------------

    /**
     * @brief Scan the wavefunction for the strongest peak soliton and emit a
     *        token to the output queue.
     *
     * This is the hot path — called at every physics tick (1 kHz target).
     * Does nothing if no node exceeds energy_threshold.
     *
     * @param wf  The active WaveFunction (will be modified by inhibition).
     */
    void scan(physics::WaveFunction& wf)
    {
        const physics::TorusGrid& grid = wf.grid();
        const size_t N = grid.num_active_nodes();
        if (N == 0) return;

        // ---- 1. Find peak node ----------------------------------------
        const float* pr = grid.psi_real();
        const float* pi = grid.psi_imag();
        const float* rs = grid.resonance();

        size_t  peak_idx    = 0;
        float   peak_energy = -1.0f;

        for (size_t i = 0; i < N; ++i) {
            const float e_cog = (pr[i]*pr[i] + pi[i]*pi[i]) * rs[i];
            if (e_cog > peak_energy) {
                peak_energy = e_cog;
                peak_idx    = i;
            }
        }

        if (peak_energy < energy_threshold_) return;

        // ---- 2. Build spectral signature --------------------------------
        const Complex psi_peak{pr[peak_idx], pi[peak_idx]};
        const auto sig = build_signature(psi_peak);

        // ---- 3. Decode via HolographicLexicon ---------------------------
        const auto decoded = lexicon_.decode(sig);

        if (decoded.has_value()) {
            // ---- 4a. Hit: emit token + inhibit --------------------------
            enqueue(*decoded);
            orphan_streak_ = 0;

            if (inhibition_enabled_) {
                // Inject phase-inverted wave at peak node (Ψ_suppress = -Ψ_peak)
                wf.inject(peak_idx, -psi_peak);
            }
        } else {
            // ---- 4b. Miss: track persistence & maybe mint ---------------
            ++orphan_streak_;

            if (minting_enabled_ && orphan_streak_ >= persistence_count_) {
                const std::string neo = minter_.mint(sig);
                enqueue(neo);
                orphan_streak_ = 0;

                if (inhibition_enabled_) {
                    wf.inject(peak_idx, -psi_peak);
                }
            }
        }
    }

    /**
     * @brief Pop the oldest token from the output queue.
     * @return Next token string, or nullopt if the queue is empty.
     */
    [[nodiscard]] std::optional<std::string> pop_token()
    {
        std::lock_guard<std::mutex> g(queue_mutex_);
        if (output_queue_.empty()) return std::nullopt;
        std::string tok = std::move(output_queue_.front());
        output_queue_.pop_front();
        return tok;
    }

    /**
     * @brief Drain all queued tokens into a vector (for batch processing).
     */
    [[nodiscard]] std::vector<std::string> drain()
    {
        std::lock_guard<std::mutex> g(queue_mutex_);
        std::vector<std::string> out(
            std::make_move_iterator(output_queue_.begin()),
            std::make_move_iterator(output_queue_.end()));
        output_queue_.clear();
        return out;
    }

    // ------------------------------------------------------------------
    // Signature construction (public for testing and lexicon seeding)
    // ------------------------------------------------------------------

    /**
     * @brief Build the 9D spectral signature for a single complex psi value.
     *
     *   sig[d] = Ψ × exp(i × d × π/9)    d ∈ {0, … 8}
     *
     * This is the canonical encoding used by scan() and must also be used
     * when registering tokens via add_token() to ensure decode() matches.
     */
    [[nodiscard]] static std::vector<Complex>
    build_signature(Complex psi) noexcept
    {
        std::vector<Complex> sig(9);
        constexpr float kStep = std::numbers::pi_v<float> / 9.0f;
        for (int d = 0; d < 9; ++d)
            sig[static_cast<size_t>(d)] = psi * std::polar(1.0f, d * kStep);
        return sig;
    }

private:
    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    void enqueue(const std::string& tok)
    {
        std::lock_guard<std::mutex> g(queue_mutex_);
        output_queue_.push_back(tok);
        ++tokens_emitted_;
    }

    // ------------------------------------------------------------------
    // Data members
    // ------------------------------------------------------------------

    HolographicLexicon&          lexicon_;
    ConceptMinter                minter_;

    float                        energy_threshold_{1e-12f};
    bool                         minting_enabled_{true};
    bool                         inhibition_enabled_{true};
    int                          persistence_count_{1};

    int                          orphan_streak_{0};
    size_t                       tokens_emitted_{0};

    mutable std::mutex           queue_mutex_;
    std::deque<std::string>      output_queue_;
};

} // namespace nikola::cognitive
