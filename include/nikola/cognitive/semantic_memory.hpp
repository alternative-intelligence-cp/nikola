/**
 * @file cognitive/semantic_memory.hpp
 * @brief Wave-basis semantic memory using Hilbert spatial indexing.
 *
 * SemanticMemory stores snapshots of WaveFunction ψ-fields as addressable
 * "memory records", keyed by the Hilbert 1D index of the dominant (highest
 * amplitude) node.  Retrieval is resonance-based: memories whose stored
 * patterns have high inner product with the query wavefunction surface first.
 *
 * Memory lifecycle:
 *   store()        → capture current ψ-field, assign Hilbert key.
 *   load()         → replay stored ψ-field into a WaveFunction.
 *   decay(dt)      → reduce strength proportional to time (aging).
 *   consolidate()  → prune weak, promote strong memories (homeostasis).
 *
 * Integration:
 *   Depends on MEM-04 HilbertScanner (Phase 0) for spatial key derivation.
 *   Works with Phase 1 WaveFunction / TorusGrid SoA arrays.
 *
 * Reference:
 *   PROJECT_STATUS.md Phase 2 — "Wave basis storage (uses MEM-04)"
 *   engineering report §8.9.2 (Logic and Memory milestone)
 */
#pragma once

#include <nikola/spatial/hilbert_scanner.hpp>
#include <nikola/physics/wave_function.hpp>

#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace nikola::cognitive {

using MemoryKey = uint64_t;  ///< Hilbert 1D index of the dominant node

// ============================================================================
// MemoryRecord
// ============================================================================

/**
 * @brief A single stored memory — a ψ-field snapshot with metadata.
 *
 * The psi arrays have length == number of nodes in the grid at storage time.
 * If a query WaveFunction has a different node count, load() will truncate or
 * zero-pad as needed (best-effort replay).
 */
struct MemoryRecord {
    MemoryKey key{0};              ///< Hilbert index of dominant node at storage time

    std::vector<float> psi_real;  ///< Stored ψ (real part), 1 entry per node
    std::vector<float> psi_imag;  ///< Stored ψ (imaginary part), 1 entry per node

    float strength{1.f};          ///< Memory strength [0, 1]; decays over time
    float age_seconds{0.f};       ///< Seconds since creation (monotonically increasing)
    uint32_t access_count{0};     ///< Total number of recall events = boosts strength
};

// ============================================================================
// SemanticMemory
// ============================================================================

/**
 * @brief Hilbert-indexed wave-basis memory store.
 *
 * Storage model:
 *   - Each record is keyed by the Hilbert 1D index of the dominant grid node.
 *   - If a new store() maps to an existing key, the record is updated
 *     (new ψ-snapshot replaces old) and strength is reset to 1.
 *
 * Decay model:
 *   strength(t + dt) = strength(t) × exp(−DECAY_RATE × dt)
 *   Records whose strength drops below MIN_STRENGTH are pruned by consolidate().
 *
 * Consolidation:
 *   During consolidate(), frequently accessed records (access_count > threshold)
 *   receive a strength bonus (Long-Term Potentiation analogy).
 */
class SemanticMemory {
public:
    // ------------------------------------------------------------------ tuning constants

    /// Per-second exponential strength-decay constant.
    static constexpr float DECAY_RATE         = 0.001f;
    /// Records below this strength are pruned in consolidate().
    static constexpr float MIN_STRENGTH       = 0.01f;
    /// Access count above which LTP bonus is applied during consolidate().
    static constexpr uint32_t LTP_THRESHOLD   = 3;
    /// Long-Term Potentiation strength multiplier (applied once per consolidate pass).
    static constexpr float LTP_BOOST          = 1.25f;
    /// Maximum strength (cap after LTP boost).
    static constexpr float MAX_STRENGTH       = 1.f;

    // ------------------------------------------------------------------ construction

    /**
     * @brief Construct semantic memory with a given Hilbert scanner order.
     *
     * @param hilbert_order  Bits per dimension (grid size = 2^order per dim).
     *                       order=5 gives 32^9 ≈ 35 trillion addressable points.
     */
    explicit SemanticMemory(uint32_t hilbert_order = 5)
        : scanner_(hilbert_order)
    {}

    // ------------------------------------------------------------------ store

    /**
     * @brief Store the current WaveFunction ψ-field as a memory record.
     *
     * The dominant node (highest |ψ|²) determines the Hilbert key.
     * If a record with the same key already exists it is overwritten and
     * strength is reset to 1.0.
     *
     * Complexity: O(N) where N = number of active nodes.
     *
     * @param wf  WaveFunction to snapshot (state is not modified).
     * @return    Assigned Hilbert key (0 if grid is empty).
     */
    MemoryKey store(const physics::WaveFunction& wf) {
        const foundation::TorusGrid& grid = wf.grid();
        const size_t N = grid.num_active_nodes();
        if (N == 0) return 0;

        // Find dominant node (max amplitude).
        const float* pr = grid.psi_real();
        const float* pi = grid.psi_imag();

        size_t peak_idx = 0;
        float  peak_a2  = pr[0]*pr[0] + pi[0]*pi[0];
        for (size_t i = 1; i < N; ++i) {
            const float a2 = pr[i]*pr[i] + pi[i]*pi[i];
            if (a2 > peak_a2) { peak_a2 = a2; peak_idx = i; }
        }

        // Convert grid coords to Hilbert coords and compute 1D key.
        const MemoryKey key = coords_to_hilbert_key(grid, peak_idx);

        // Build / overwrite record.
        MemoryRecord& rec = records_[key];
        rec.key          = key;
        rec.psi_real.assign(pr, pr + N);
        rec.psi_imag.assign(pi, pi + N);
        rec.strength     = MAX_STRENGTH;
        rec.access_count = 0;
        // age_seconds reset only if this is a brand-new record
        if (rec.psi_real.size() == static_cast<size_t>(N) &&
            rec.age_seconds > 0.f) {
            // Overwrite of existing — keep age (consolidated memory rewired)
        } else {
            rec.age_seconds = 0.f;
        }

        return key;
    }

    // ------------------------------------------------------------------ load

    /**
     * @brief Replay a stored ψ-field into a WaveFunction.
     *
     * The stored psi arrays are written back to the grid's psi fields.
     * If the stored node count differs from the current grid size, only the
     * overlapping prefix is restored (safe partial replay).
     *
     * Also increments access_count and applies a small strength boost.
     *
     * @param key  Hilbert key to look up.
     * @param wf   Target WaveFunction (must already have nodes allocated).
     * @return     true if key found and replayed, false otherwise.
     */
    bool load(MemoryKey key, physics::WaveFunction& wf) {
        auto it = records_.find(key);
        if (it == records_.end()) return false;

        MemoryRecord& rec = it->second;
        foundation::TorusGrid& grid  = wf.grid();
        const size_t N_wf  = grid.num_active_nodes();
        const size_t N_rec = rec.psi_real.size();
        const size_t N_min = std::min(N_wf, N_rec);

        float* pr = grid.psi_real();
        float* pi = grid.psi_imag();

        for (size_t i = 0; i < N_min; ++i) {
            pr[i] = rec.psi_real[i];
            pi[i] = rec.psi_imag[i];
        }
        // Zero-pad any remaining nodes (grid has more nodes than record).
        for (size_t i = N_min; i < N_wf; ++i) {
            pr[i] = 0.f;
            pi[i] = 0.f;
        }

        ++rec.access_count;
        // Small strength boost on recall (short-term recall potentiation).
        rec.strength = std::min(MAX_STRENGTH, rec.strength + 0.05f);

        return true;
    }

    // ------------------------------------------------------------------ decay

    /**
     * @brief Age all records by @p dt seconds and decay their strength.
     *
     * strength' = strength × exp(−DECAY_RATE × dt)
     * age' = age + dt
     *
     * @param dt  Simulation time step in seconds.
     */
    void decay(float dt) noexcept {
        const float decay_factor = std::exp(-DECAY_RATE * dt);
        for (auto& [key, rec] : records_) {
            rec.strength    *= decay_factor;
            rec.age_seconds += dt;
        }
    }

    // ------------------------------------------------------------------ consolidate

    /**
     * @brief Home-ostatic consolidation pass.
     *
     * Actions performed (in order):
     *   1. Frequently accessed records (access_count ≥ LTP_THRESHOLD) receive
     *      a strength boost (Long-Term Potentiation).
     *   2. Records whose strength < MIN_STRENGTH are erased (forgetting).
     *
     * @return  Number of records pruned.
     */
    size_t consolidate() {
        // LTP boost phase.
        for (auto& [key, rec] : records_) {
            if (rec.access_count >= LTP_THRESHOLD) {
                rec.strength = std::min(MAX_STRENGTH, rec.strength * LTP_BOOST);
                rec.access_count = 0;  // reset counter after potentiation
            }
        }

        // Pruning phase.
        size_t pruned = 0;
        for (auto it = records_.begin(); it != records_.end(); ) {
            if (it->second.strength < MIN_STRENGTH) {
                it = records_.erase(it);
                ++pruned;
            } else {
                ++it;
            }
        }
        return pruned;
    }

    // ------------------------------------------------------------------ accessors

    /// Number of stored records.
    [[nodiscard]] size_t size() const noexcept { return records_.size(); }

    /// Whether a record exists for the given key.
    [[nodiscard]] bool contains(MemoryKey key) const noexcept {
        return records_.count(key) > 0;
    }

    /// Read-only access to a record (returns nullptr if not found).
    [[nodiscard]] const MemoryRecord* get(MemoryKey key) const noexcept {
        auto it = records_.find(key);
        return (it != records_.end()) ? &it->second : nullptr;
    }

    /// All stored keys (unordered).
    [[nodiscard]]
    std::vector<MemoryKey> all_keys() const {
        std::vector<MemoryKey> keys;
        keys.reserve(records_.size());
        for (const auto& [k, _] : records_) keys.push_back(k);
        return keys;
    }

    /// Underlying Hilbert scanner (for external use by QueryEngine).
    const spatial::HilbertScanner& scanner() const noexcept { return scanner_; }

private:
    // ------------------------------------------------------------------ helpers

    /**
     * @brief Convert a grid node's integer coordinates → Hilbert 1D key.
     *
     * Scales each dimension from [0, N_d) to [0, 2^hilbert_order) by
     * integer division (floor-rounding).
     */
    [[nodiscard]]
    MemoryKey coords_to_hilbert_key(const foundation::TorusGrid& grid,
                                    size_t node_idx) const noexcept
    {
        const auto& raw = grid.coords_of(node_idx);
        const uint32_t order      = scanner_.get_order();
        const uint32_t hilbert_N  = 1u << order;

        spatial::HilbertScanner::Coord9D hc{};
        for (int d = 0; d < foundation::TORUS_DIMS; ++d) {
            const int Nd = grid.config().resolution[d];
            // Scale coord to [0, hilbert_N)
            const uint32_t scaled = static_cast<uint32_t>(
                static_cast<uint64_t>(raw[d]) * hilbert_N / static_cast<uint64_t>(Nd));
            hc[d] = std::min(scaled, hilbert_N - 1u);
        }

        return scanner_.coords_to_index(hc);
    }

    // ------------------------------------------------------------------ data

    spatial::HilbertScanner scanner_;
    std::unordered_map<MemoryKey, MemoryRecord> records_;
};

} // namespace nikola::cognitive
