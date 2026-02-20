/**
 * @file include/nikola/autonomy/metabolic_controller.hpp
 * @brief Metabolic energy budget system (simulated ATP).
 *
 * Regulates cognitive load to prevent "epileptic" runaway plasticity.
 * Every operation has a metabolic cost; when ATP < threshold, system
 * enters "Nap" state to recharge (simulating biological sleep).
 *
 * Thread-safe via atomic operations.
 */

#pragma once

#include <atomic>
#include <cstdint>

namespace nikola::autonomy {

/**
 * @class MetabolicController
 * @brief Manages system-wide energy budget (ATP reserve).
 *
 * Operations consume ATP:
 * - Wave propagation: ~0.1 ATP
 * - Neuroplasticity update: ~1.5 ATP  
 * - External tool call: ~5.0 ATP
 *
 * When ATP falls below threshold, triggers Nap cycle for recharge.
 * Uses MetabolicLock for safe transactional energy consumption.
 */
class MetabolicController {
public:
    /**
     * @brief Initialize with starting ATP reserve.
     * @param initial_atp Starting energy budget (default: 100.0)
     * @param nap_threshold Trigger nap when ATP < threshold (default: 10.0)
     */
    explicit MetabolicController(
        float initial_atp = 100.0f,
        float nap_threshold = 10.0f
    );

    /**
     * @brief Get current ATP level (thread-safe read).
     */
    float get_current_atp() const noexcept;

    /**
     * @brief Check if system needs to enter Nap state.
     */
    bool needs_nap() const noexcept;

    /**
     * @brief Recharge ATP (called during Nap cycles).
     * @param amount ATP to add (capped at max_atp_)
     */
    void recharge(float amount) noexcept;

    /**
     * @brief Get reference to ATP reserve for MetabolicLock.
     * 
     * WARNING: Direct access only for MetabolicLock implementation.
     * All other code must use MetabolicLock for safe consumption.
     */
    std::atomic<float>& get_atp_reserve() noexcept { return atp_reserve_; }

private:
    std::atomic<float> atp_reserve_;
    float nap_threshold_;
    float max_atp_;
};

} // namespace nikola::autonomy
