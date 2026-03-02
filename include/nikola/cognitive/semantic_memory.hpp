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
#include <fstream>
#include <stdexcept>
#include <string>

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

    // ------------------------------------------------------------------ RecallHit

    /**
     * @brief A single resonance search result returned by recall().
     */
    struct RecallHit {
        MemoryKey           key{0};       ///< Hilbert key of the matching record.
        float               score{0.f};   ///< Cosine similarity ∈ [0, 1] (only positive retained).
        const MemoryRecord* record{nullptr}; ///< Pointer into records_ map (valid until next store/consolidate).
    };

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

    /// True when no records are stored.
    [[nodiscard]] bool empty() const noexcept { return records_.empty(); }

    // ------------------------------------------------------------------ recall

    /**
     * @brief Find the top-k records most resonant with a query WaveFunction.
     *
     * Resonance is measured as cosine similarity between the stored ψ-field
     * vector and the current query ψ-field vector (real and imaginary parts
     * concatenated into one flat vector).  Only positive similarities are
     * returned; records with zero or negative dot product are discarded.
     *
     * Complexity: O(N_records × N_nodes).
     *
     * @param wf  Query WaveFunction (read-only).
     * @param k   Maximum number of hits to return (default: 1).
     * @return    Up to k RecallHit structs sorted by score descending.
     */
    [[nodiscard]]
    std::vector<RecallHit> recall(const physics::WaveFunction& wf, size_t k = 1) const
    {
        const foundation::TorusGrid& grid = wf.grid();
        const size_t N_wf = grid.num_active_nodes();
        const float* pr   = grid.psi_real();
        const float* pi   = grid.psi_imag();

        // Compute L2 norm of query.
        double qnorm2 = 0.0;
        for (size_t i = 0; i < N_wf; ++i)
            qnorm2 += static_cast<double>(pr[i]*pr[i] + pi[i]*pi[i]);
        const float qnorm = static_cast<float>(std::sqrt(qnorm2));
        if (qnorm < 1e-10f) return {};

        std::vector<RecallHit> results;
        results.reserve(records_.size());

        for (const auto& [key, rec] : records_) {
            const size_t N_rec = rec.psi_real.size();
            const size_t N_min = std::min(N_wf, N_rec);

            // Dot product and record norm.
            double dot   = 0.0;
            double rnorm2 = 0.0;
            for (size_t i = 0; i < N_min; ++i) {
                dot    += static_cast<double>(pr[i] * rec.psi_real[i] +
                                              pi[i] * rec.psi_imag[i]);
                rnorm2 += static_cast<double>(rec.psi_real[i]*rec.psi_real[i] +
                                              rec.psi_imag[i]*rec.psi_imag[i]);
            }
            const float rnorm = static_cast<float>(std::sqrt(rnorm2));
            if (rnorm < 1e-10f) continue;

            const float cosine = static_cast<float>(dot) / (qnorm * rnorm);
            if (cosine > 0.f) {
                // Weight by stored memory strength.
                results.push_back({key, cosine * rec.strength, &rec});
            }
        }

        // Sort descending by score.
        std::sort(results.begin(), results.end(),
                  [](const RecallHit& a, const RecallHit& b){ return a.score > b.score; });
        if (results.size() > k) results.resize(k);
        return results;
    }

    // ------------------------------------------------------------------ superpose

    /**
     * @brief Blend a stored ψ-field additively into a live WaveFunction.
     *
     * Computes:  wf.psi[i] += alpha * rec.psi[i]  for all overlapping nodes.
     * Triggers a strength boost on the recalled record (same as load()).
     *
     * @param key    Hilbert key of the stored record.
     * @param alpha  Blend weight.
     * @param wf     Target WaveFunction (modified in place).
     * @return       true if key exists and superposition applied.
     */
    bool superpose(MemoryKey key, float alpha, physics::WaveFunction& wf)
    {
        auto it = records_.find(key);
        if (it == records_.end()) return false;

        MemoryRecord& rec = it->second;
        foundation::TorusGrid& grid = wf.grid();
        const size_t N_wf  = grid.num_active_nodes();
        const size_t N_rec = rec.psi_real.size();
        const size_t N_min = std::min(N_wf, N_rec);

        float* pr = grid.psi_real();
        float* pi = grid.psi_imag();
        for (size_t i = 0; i < N_min; ++i) {
            pr[i] += alpha * rec.psi_real[i];
            pi[i] += alpha * rec.psi_imag[i];
        }

        ++rec.access_count;
        rec.strength = std::min(MAX_STRENGTH, rec.strength + 0.03f);
        return true;
    }

    // ------------------------------------------------------------------ persistence (Phase 33)

    /**
     * @brief Save all records to a binary file.
     *
     * Format (little-endian):
     *   [uint64_t n_records]
     *   for each record:
     *     [uint64_t key]
     *     [uint32_t n_nodes]
     *     [float × n_nodes  psi_real]
     *     [float × n_nodes  psi_imag]
     *     [float strength]
     *     [float age_seconds]
     *     [uint32_t access_count]
     *
     * @throws std::runtime_error on I/O failure.
     */
    void save(const std::string& path) const
    {
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f) throw std::runtime_error("SemanticMemory::save: cannot open " + path);

        const uint64_t n = static_cast<uint64_t>(records_.size());
        f.write(reinterpret_cast<const char*>(&n), sizeof(n));

        for (const auto& [key, rec] : records_) {
            f.write(reinterpret_cast<const char*>(&key), sizeof(key));
            const uint32_t nn = static_cast<uint32_t>(rec.psi_real.size());
            f.write(reinterpret_cast<const char*>(&nn), sizeof(nn));
            f.write(reinterpret_cast<const char*>(rec.psi_real.data()), nn * sizeof(float));
            f.write(reinterpret_cast<const char*>(rec.psi_imag.data()), nn * sizeof(float));
            f.write(reinterpret_cast<const char*>(&rec.strength),     sizeof(float));
            f.write(reinterpret_cast<const char*>(&rec.age_seconds),  sizeof(float));
            f.write(reinterpret_cast<const char*>(&rec.access_count), sizeof(uint32_t));
        }
        if (!f) throw std::runtime_error("SemanticMemory::save: write error on " + path);
    }

    /**
     * @brief Load records from a binary file previously written by save().
     *
     * Existing records are retained; stored records are merged (overwrite on
     * key collision). Returns the number of records successfully loaded.
     *
     * @throws std::runtime_error on I/O or format error.
     */
    size_t load(const std::string& path)
    {
        std::ifstream f(path, std::ios::binary);
        if (!f) return 0;  // file does not exist or is unreadable — best-effort; not an error

        uint64_t n = 0;
        f.read(reinterpret_cast<char*>(&n), sizeof(n));
        if (!f) return 0;  // truncated or empty file — silently ignore

        size_t loaded = 0;
        for (uint64_t i = 0; i < n; ++i) {
            MemoryKey key = 0;
            f.read(reinterpret_cast<char*>(&key), sizeof(key));
            uint32_t nn = 0;
            f.read(reinterpret_cast<char*>(&nn), sizeof(nn));
            if (!f || nn > (1u << 20)) break;  // sanity: max 1M nodes

            MemoryRecord rec{};
            rec.key = key;
            rec.psi_real.resize(nn);
            rec.psi_imag.resize(nn);
            f.read(reinterpret_cast<char*>(rec.psi_real.data()), nn * sizeof(float));
            f.read(reinterpret_cast<char*>(rec.psi_imag.data()), nn * sizeof(float));
            f.read(reinterpret_cast<char*>(&rec.strength),     sizeof(float));
            f.read(reinterpret_cast<char*>(&rec.age_seconds),  sizeof(float));
            f.read(reinterpret_cast<char*>(&rec.access_count), sizeof(uint32_t));
            if (!f) break;

            records_[key] = std::move(rec);
            ++loaded;
        }
        return loaded;
    }

    /// Underlying Hilbert scanner (for external use by QueryEngine).
    const spatial::HilbertScanner& scanner() const noexcept { return scanner_; }

    // ------------------------------------------------------------------ Phase 136 LMDB persistence helpers

    /**
     * @brief Read-only access to all stored records (for LMDB serialisation).
     * Valid until the next store() / consolidate() that modifies the map.
     */
    [[nodiscard]]
    const std::unordered_map<MemoryKey, MemoryRecord>& records() const noexcept
    {
        return records_;
    }

    /**
     * @brief Insert or replace a MemoryRecord directly (used by load_lmdb).
     * Bypasses the WaveFunction-based store() path; rec.key is the map key.
     */
    void insert_record(MemoryRecord rec)
    {
        const MemoryKey k = rec.key;
        records_[k] = std::move(rec);
    }

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
