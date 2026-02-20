/**
 * @file include/nikola/autonomy/metabolic_lock.hpp
 * @brief Transactional RAII Guard for Metabolic Energy (ATP).
 *
 * Resolves Finding CF-04: Prevents thermodynamic race conditions where
 * multiple components consume energy simultaneously, driving the system
 * into illegal negative energy states.
 *
 * Implementation based on: Section 8.2, Nikola Engineering Report v0.0.4
 * 
 * CRITICAL: This is a Phase 0 blocking dependency. All other phases depend
 * on this fix for thermodynamic safety.
 *
 * Dependencies: nikola/autonomy/metabolic_controller.hpp
 */

#pragma once

#include <exception>
#include <string>
#include <atomic>
#include <memory>

namespace nikola::autonomy {

// Forward declaration
class MetabolicController;

/**
 * @class MetabolicExhaustionException
 * @brief Thrown when a transaction fails to reserve sufficient ATP.
 * 
 * Caught by the Orchestrator to trigger emergency Nap cycles.
 */
class MetabolicExhaustionException : public std::exception {
public:
    explicit MetabolicExhaustionException(float requested, float available)
        : requested_(requested), available_(available) {
        message_ = "Metabolic exhaustion: requested " + 
                   std::to_string(requested) + " ATP, only " + 
                   std::to_string(available) + " available";
    }

    const char* what() const noexcept override {
        return message_.c_str();
    }

    float requested() const noexcept { return requested_; }
    float available() const noexcept { return available_; }

private:
    float requested_;
    float available_;
    std::string message_;
};

/**
 * @class MetabolicLock
 * @brief RAII guard for atomic ATP reservation and consumption.
 *
 * Usage Pattern:
 * ```cpp
 * try {
 *     MetabolicLock lock(controller, 5.0f);  // Reserve 5.0 ATP
 *     perform_expensive_operation();         // Guaranteed safe
 *     lock.commit();                         // Mark success
 * } catch (MetabolicExhaustionException& e) {
 *     // ATP refunded automatically by destructor
 *     trigger_emergency_nap();
 * }
 * ```
 *
 * Guarantees:
 * - Energy is reserved atomically before any work begins
 * - Failed operations automatically refund reserved energy
 * - System can never spend energy it doesn't have
 * - Thread-safe via atomic CAS operations
 */
class MetabolicLock {
public:
    /**
     * @brief Attempt to reserve ATP for an operation.
     * 
     * @param controller Reference to the system's metabolic controller
     * @param cost Amount of ATP to reserve (must be > 0)
     * @throws MetabolicExhaustionException if insufficient ATP available
     * 
     * Atomically verifies and deducts ATP via Compare-And-Swap loop.
     * If reservation fails, throws immediately - no work is performed.
     */
    MetabolicLock(MetabolicController& controller, float cost);

    /**
     * @brief Destructor - refunds ATP if commit() not called.
     * 
     * Implements rollback semantics: if operation failed (exception thrown),
     * energy is automatically returned to the pool.
     */
    ~MetabolicLock();

    // Non-copyable, non-movable (RAII semantics)
    MetabolicLock(const MetabolicLock&) = delete;
    MetabolicLock& operator=(const MetabolicLock&) = delete;
    MetabolicLock(MetabolicLock&&) = delete;
    MetabolicLock& operator=(MetabolicLock&&) = delete;

    /**
     * @brief Mark transaction as successful - energy remains consumed.
     * 
     * Must be called after operation completes successfully.
     * Prevents destructor from refunding the energy.
     */
    void commit() noexcept;

    /**
     * @brief Get the reserved energy amount.
     */
    float reserved() const noexcept { return cost_; }

private:
    MetabolicController& controller_;
    float cost_;
    bool committed_;
};

} // namespace nikola::autonomy
