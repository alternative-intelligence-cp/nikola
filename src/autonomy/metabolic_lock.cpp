/**
 * @file src/autonomy/metabolic_lock.cpp
 * @brief Implementation of transactional ATP reservation system.
 *
 * Resolves Finding CF-04: Thermodynamic Race Condition
 * See: Section 8.2, Nikola Engineering Report v0.0.4
 */

#include "nikola/autonomy/metabolic_lock.hpp"
#include "nikola/autonomy/metabolic_controller.hpp"
#include <stdexcept>

namespace nikola::autonomy {

MetabolicLock::MetabolicLock(MetabolicController& controller, float cost)
    : controller_(controller), cost_(cost), committed_(false) {
    
    if (cost <= 0.0f) {
        throw std::invalid_argument("ATP cost must be positive");
    }

    // Atomic reservation via Compare-And-Swap loop
    // This ensures no other thread can intervene between check and deduction
    std::atomic<float>& atp_reserve = controller_.get_atp_reserve();
    
    float expected, desired;
    do {
        expected = atp_reserve.load(std::memory_order_acquire);
        
        // Insufficient ATP - fail immediately without consuming
        if (expected < cost_) {
            throw MetabolicExhaustionException(cost_, expected);
        }
        
        desired = expected - cost_;
        
        // Attempt atomic swap: only succeeds if no other thread modified reserve
        // If CAS fails, loop retries with updated 'expected' value
    } while (!atp_reserve.compare_exchange_weak(
        expected, desired,
        std::memory_order_release,
        std::memory_order_acquire
    ));
    
    // Reservation successful - energy deducted atomically
    // Operation can now proceed, guaranteed to have the energy it needs
}

MetabolicLock::~MetabolicLock() {
    // Rollback: refund energy if commit() wasn't called
    if (!committed_) {
        std::atomic<float>& atp_reserve = controller_.get_atp_reserve();
        
        // Atomic refund via fetch_add
        atp_reserve.fetch_add(cost_, std::memory_order_release);
        
        // Note: This is safe even if operation partially completed
        // Failed operations should not have produced lasting effects
        // (that's the responsibility of the caller to ensure)
    }
}

void MetabolicLock::commit() noexcept {
    // Mark transaction as successful
    // Destructor will now skip refund
    committed_ = true;
}

} // namespace nikola::autonomy
