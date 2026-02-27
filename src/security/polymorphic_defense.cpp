/**
 * @file polymorphic_defense.cpp
 * @brief Phase 128 — PolymorphicDefense implementation
 */

#include <nikola/security/polymorphic_defense.hpp>

#include <algorithm>
#include <chrono>
#include <iterator>
#include <stdexcept>

namespace nikola::security {

// ---------------------------------------------------------------------------
// Static helpers
// ---------------------------------------------------------------------------

uint64_t PolymorphicDefense::generate_token(std::mt19937_64& rng) {
    uint64_t t = 0;
    while (t == 0) {
        t = rng();   // discard 0 — token 0 is the sentinel for "not found"
    }
    return t;
}

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

PolymorphicDefense::PolymorphicDefense()
    : rng_(std::random_device{}()) {
}

PolymorphicDefense::~PolymorphicDefense() {
    disable_continuous();
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

ProtectedEntry* PolymorphicDefense::find_mutable(const std::string& name) {
    for (auto& e : entries_) {
        if (e.name == name) return &e;
    }
    return nullptr;
}

void PolymorphicDefense::apply_mutation(ProtectedEntry& e, uint64_t tick) {
    e.token = generate_token(rng_);
    ++e.mutation_count;
    e.last_mutated = tick;
    ++total_muts_;
    if (mutation_cb_) mutation_cb_(e);
}

// ---------------------------------------------------------------------------
// Entry management
// ---------------------------------------------------------------------------

uint64_t PolymorphicDefense::register_entry(const std::string& name,
                                              uint64_t tick) {
    // Dedup
    ProtectedEntry* existing = find_mutable(name);
    if (existing) return existing->token;

    // Evict oldest if at cap
    if (entries_.size() >= POLY_MAX_ENTRIES) {
        entries_.erase(entries_.begin());
    }

    ProtectedEntry e;
    e.id            = next_id_++;
    e.name          = name;
    e.token         = generate_token(rng_);
    e.mutation_count = 0;
    e.registered_at = tick;
    e.last_mutated  = tick;

    entries_.push_back(e);
    return e.token;
}

void PolymorphicDefense::remove_entry(uint64_t id) {
    entries_.erase(
        std::remove_if(entries_.begin(), entries_.end(),
                       [id](const ProtectedEntry& e) { return e.id == id; }),
        entries_.end());
}

void PolymorphicDefense::remove_named(const std::string& name) {
    entries_.erase(
        std::remove_if(entries_.begin(), entries_.end(),
                       [&](const ProtectedEntry& e) { return e.name == name; }),
        entries_.end());
}

void PolymorphicDefense::clear() {
    entries_.clear();
}

// ---------------------------------------------------------------------------
// Core operations
// ---------------------------------------------------------------------------

void PolymorphicDefense::randomize(double mutation_rate, uint64_t tick) {
    if (entries_.empty()) return;

    mutation_rate = std::clamp(mutation_rate, 0.0, 1.0);
    const size_t n = static_cast<size_t>(
        std::max(1.0, mutation_rate * static_cast<double>(entries_.size())));

    // Shuffle indices and pick first n
    std::vector<size_t> indices(entries_.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng_);

    for (size_t i = 0; i < n && i < entries_.size(); ++i) {
        apply_mutation(entries_[indices[i]], tick);
    }
}

void PolymorphicDefense::remutate(const std::string& name, uint64_t tick) {
    ProtectedEntry* e = find_mutable(name);
    if (e) apply_mutation(*e, tick);
}

uint64_t PolymorphicDefense::current_token(const std::string& name) const {
    for (const auto& e : entries_) {
        if (e.name == name) return e.token;
    }
    return 0;
}

bool PolymorphicDefense::validate_token(const std::string& name,
                                          uint64_t token) const {
    const uint64_t current = current_token(name);
    return (current != 0) && (current == token);
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

const ProtectedEntry* PolymorphicDefense::find(const std::string& name) const {
    for (const auto& e : entries_) {
        if (e.name == name) return &e;
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

PolymorphicDefense::Stats PolymorphicDefense::stats() const {
    Stats s;
    s.total_entries   = entries_.size();
    s.total_mutations = total_muts_;
    s.continuous_active = continuous_active_.load();
    s.mean_mutations_per_entry =
        s.total_entries > 0
            ? static_cast<double>(s.total_mutations) /
                  static_cast<double>(s.total_entries)
            : 0.0;
    return s;
}

// ---------------------------------------------------------------------------
// Continuous polymorphism
// ---------------------------------------------------------------------------

void PolymorphicDefense::enable_continuous(uint64_t interval_ms,
                                             double mutation_rate) {
    if (continuous_active_.load()) return;

    continuous_active_.store(true);
    continuous_thread_ = std::thread([this, interval_ms, mutation_rate]() {
        while (continuous_active_.load()) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(interval_ms));
            if (continuous_active_.load()) {
                randomize(mutation_rate);
            }
        }
    });
}

void PolymorphicDefense::disable_continuous() {
    continuous_active_.store(false);
    if (continuous_thread_.joinable()) {
        continuous_thread_.join();
    }
}

} // namespace nikola::security
