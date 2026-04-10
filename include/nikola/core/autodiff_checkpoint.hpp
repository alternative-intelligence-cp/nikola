/**
 * @file include/nikola/core/autodiff_checkpoint.hpp
 * @brief Gradient checkpointing for memory-efficient training (CF-01)
 *
 * Problem: Full tape for 19,683 nodes × 1,000 steps ≈ 503 GB.
 * Solution: Save checkpoints every N steps (~100), recompute forward pass
 *           segments during backward. Reduces memory to ~19 MB at 10× cost.
 *
 * @see docs/info/gemini/compilation/part_4_of_9.txt lines 1900-2100
 */
#pragma once

#include "nikola/core/autodiff.hpp"
#include <algorithm>
#include <complex>
#include <cstdint>
#include <functional>
#include <vector>

namespace nikola::autodiff {

/// Snapshot of tape state at a given timestep
struct Checkpoint {
    size_t timestep        = 0;
    size_t tape_position   = 0;
    std::vector<std::complex<double>> node_values;
};

/**
 * @brief Wraps NikolaAutodiff with checkpoint/recompute for bounded memory
 *
 * Usage:
 *   CheckpointedAutodiff ad(100);  // checkpoint every 100 steps
 *   ad.set_recompute_function([&](size_t from, size_t to) {
 *       for (size_t t = from; t < to; ++t) model.forward_step(data[t]);
 *   });
 *   // forward pass — call save_checkpoint() every interval
 *   ad.backward_with_checkpointing(last_step);
 */
class CheckpointedAutodiff {
public:
    explicit CheckpointedAutodiff(size_t interval = 100)
        : checkpoint_interval_(interval) {}

    /// Set the function that rebuilds the tape between two timesteps
    void set_recompute_function(std::function<void(size_t, size_t)> fn) {
        recompute_fn_ = std::move(fn);
    }

    /// Save current tape state as a checkpoint
    void save_checkpoint(size_t timestep) {
        Checkpoint cp;
        cp.timestep      = timestep;
        cp.tape_position = tape_.get_tape_size();
        cp.node_values.reserve(cp.tape_position);

        for (size_t i = 0; i < cp.tape_position; ++i)
            cp.node_values.push_back(tape_.get_value(i));

        checkpoints_.push_back(std::move(cp));

        // Free intermediate tape memory (keep only checkpoint values)
        if (checkpoints_.size() > 1)
            tape_.clear_before(checkpoints_[checkpoints_.size() - 2].tape_position);
    }

    /// Run backward pass, recomputing forward segments from checkpoints
    void backward_with_checkpointing(size_t target_timestep) {
        // Find nearest checkpoint ≤ target
        auto it = std::lower_bound(
            checkpoints_.begin(), checkpoints_.end(), target_timestep,
            [](const Checkpoint& cp, size_t t) { return cp.timestep < t; });

        if (it != checkpoints_.begin()) --it;
        if (it == checkpoints_.end() && !checkpoints_.empty())
            it = checkpoints_.begin();

        const Checkpoint& cp = *it;

        // Restore tape values from checkpoint
        tape_.restore_values(cp.node_values, cp.tape_position);

        // Recompute forward from checkpoint to target
        if (recompute_fn_ && cp.timestep < target_timestep)
            recompute_fn_(cp.timestep, target_timestep);

        // Standard backward on the (now re-populated) tape
        tape_.backward(tape_.get_tape_size() - 1);
    }

    /// Get gradient for a specific node
    std::complex<double> get_gradient(size_t id) const {
        return tape_.get_gradient(id);
    }

    /// Access the underlying tape (for building the forward graph)
    NikolaAutodiff& get_tape() { return tape_; }
    const NikolaAutodiff& get_tape() const { return tape_; }

    /// Checkpoint interval
    size_t interval() const { return checkpoint_interval_; }

    /// Number of stored checkpoints
    size_t checkpoint_count() const { return checkpoints_.size(); }

    /// Total bytes used by checkpoint storage
    size_t checkpoint_memory_bytes() const {
        size_t total = 0;
        for (const auto& cp : checkpoints_)
            total += cp.node_values.size() * sizeof(std::complex<double>);
        return total;
    }

    /// Full reset
    void reset() {
        checkpoints_.clear();
        tape_.clear();
    }

private:
    NikolaAutodiff tape_;
    std::vector<Checkpoint> checkpoints_;
    size_t checkpoint_interval_;
    std::function<void(size_t, size_t)> recompute_fn_;
};

} // namespace nikola::autodiff
