#pragma once
/**
 * @file polymorphic_defense.hpp
 * @brief Phase 128 — PolymorphicDefense: ASLR-style behavioral token mutation
 *
 * Implements the ASLR concept at the abstract-identity level: protected
 * behavioral entries own a random token that is periodically re-randomized,
 * so any attacker who observed a token at a previous tick can no longer use
 * it to replicate or exploit the same behavioral pattern.
 *
 * Workflow:
 *  1. register_entry(name)         → returns initial random token (uint64_t)
 *  2. validate_token(name, token)  → true only if token matches current
 *  3. randomize(mutation_rate)     → re-randomize fraction of entries
 *  4. current_token(name)          → get current token for an entry
 *
 * Continuous polymorphism:
 *  enable_continuous(interval_ms)  → spawns background thread that calls
 *                                    randomize(default_rate) periodically
 *  disable_continuous()            → stops the thread
 *
 * No TorusManifold / Coord9D / Eigen dependencies.
 *
 * Key constants:
 *  POLY_DEFAULT_MUTATION_RATE   0.10   fraction remutated per cycle
 *  POLY_MAX_ENTRIES             256    soft cap (evict oldest on overflow)
 */

#include <cstdint>
#include <string>
#include <vector>
#include <random>
#include <functional>
#include <atomic>
#include <thread>

namespace nikola::security {

inline constexpr double POLY_DEFAULT_MUTATION_RATE = 0.10;
inline constexpr size_t POLY_MAX_ENTRIES           = 256;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

struct ProtectedEntry {
    uint64_t    id             = 0;
    std::string name;
    uint64_t    token          = 0;   ///< current active token
    uint64_t    mutation_count = 0;   ///< times this entry has been remapped
    uint64_t    registered_at  = 0;   ///< tick at registration
    uint64_t    last_mutated   = 0;   ///< tick of last mutation
};

// ---------------------------------------------------------------------------
// PolymorphicDefense
// ---------------------------------------------------------------------------

class PolymorphicDefense {
public:
    PolymorphicDefense();
    ~PolymorphicDefense();

    // --- Entry management ---------------------------------------------------

    /**
     * @brief Register a named entry and receive its initial random token.
     *
     * If an entry with the same name already exists its current token is
     * returned (no duplicate).  Pool is capped at POLY_MAX_ENTRIES; oldest
     * entry is evicted on overflow.
     */
    uint64_t register_entry(const std::string& name, uint64_t tick = 0);

    /**
     * @brief Remove an entry by id.
     */
    void remove_entry(uint64_t id);

    /**
     * @brief Remove an entry by name.
     */
    void remove_named(const std::string& name);

    /**
     * @brief Clear all entries.
     */
    void clear();

    // --- Core operations ----------------------------------------------------

    /**
     * @brief Re-randomize a random fraction of entries.
     *
     * @param mutation_rate  Fraction of entries to mutate [0, 1].
     * @param tick           Current tick (stored in entry).
     */
    void randomize(double mutation_rate = POLY_DEFAULT_MUTATION_RATE,
                   uint64_t tick = 0);

    /**
     * @brief Force-remutate a specific entry.
     */
    void remutate(const std::string& name, uint64_t tick = 0);

    /**
     * @brief Returns current token for `name`; 0 if not found.
     */
    uint64_t current_token(const std::string& name) const;

    /**
     * @brief Returns true iff `token` matches the current token for `name`.
     */
    bool validate_token(const std::string& name, uint64_t token) const;

    // --- Continuous polymorphism --------------------------------------------

    /**
     * @brief Start a background thread that calls randomize() every
     *        `interval_ms` milliseconds.
     */
    void enable_continuous(uint64_t interval_ms = 60'000,
                           double mutation_rate = POLY_DEFAULT_MUTATION_RATE);

    /**
     * @brief Stop the continuous mutation thread.
     */
    void disable_continuous();

    bool is_continuous() const { return continuous_active_.load(); }

    // --- Queries ------------------------------------------------------------

    const ProtectedEntry* find(const std::string& name) const;
    size_t entry_count() const { return entries_.size(); }

    // --- Stats --------------------------------------------------------------

    struct Stats {
        size_t   total_entries    = 0;
        uint64_t total_mutations  = 0;
        double   mean_mutations_per_entry = 0.0;
        bool     continuous_active = false;
    };

    Stats stats() const;

    // --- Callback -----------------------------------------------------------

    using MutationCallback = std::function<void(const ProtectedEntry&)>;
    void on_mutation(MutationCallback cb) { mutation_cb_ = std::move(cb); }

    // --- Static helpers -----------------------------------------------------

    /**
     * @brief Generate a non-zero random 64-bit token.
     */
    static uint64_t generate_token(std::mt19937_64& rng);

private:
    std::vector<ProtectedEntry> entries_;
    uint64_t                    next_id_     = 1;
    uint64_t                    total_muts_  = 0;
    mutable std::mt19937_64     rng_;

    std::atomic<bool>           continuous_active_{false};
    std::thread                 continuous_thread_;
    MutationCallback            mutation_cb_;

    ProtectedEntry* find_mutable(const std::string& name);
    void apply_mutation(ProtectedEntry& e, uint64_t tick);
};

} // namespace nikola::security
