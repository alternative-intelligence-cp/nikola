/**
 * @file src/autonomy/metabolic_controller.cpp
 * @brief Implementation of metabolic energy budget system.
 */

#include "nikola/autonomy/metabolic_controller.hpp"
#include <algorithm>

namespace nikola::autonomy {

MetabolicController::MetabolicController(float initial_atp, float nap_threshold)
    : atp_reserve_(initial_atp),
      nap_threshold_(nap_threshold),
      max_atp_(initial_atp) {
}

float MetabolicController::get_current_atp() const noexcept {
    return atp_reserve_.load(std::memory_order_acquire);
}

bool MetabolicController::needs_nap() const noexcept {
    return get_current_atp() < nap_threshold_;
}

void MetabolicController::recharge(float amount) noexcept {
    if (amount <= 0.0f) return;
    
    // Atomic recharge with cap at max_atp_
    float expected, desired;
    do {
        expected = atp_reserve_.load(std::memory_order_acquire);
        desired = std::min(expected + amount, max_atp_);
    } while (!atp_reserve_.compare_exchange_weak(
        expected, desired,
        std::memory_order_release,
        std::memory_order_acquire
    ));
}

} // namespace nikola::autonomy
