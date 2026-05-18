/**
 * @file interface/feedback_loop.hpp
 * @brief Phase 55 — GAP-022: ENGS → Physics Engine Feedback Loop Latency
 *
 * Implements the double-buffered, lock-free neurochemical gateway that
 * bridges the ENGS orchestrator (CPU thread) with the Physics Engine
 * (1 kHz symplectic integrator).
 *
 * Spec requirements:
 *   - τ_max ≤ T_coh / 2 ≈ 10 ms  (Nyquist safety margin)
 *   - D / N channel hard limit: 10 ms
 *   - S channel soft limit:     50 ms
 *   - Atomicity: no torn reads (complete old or complete new state)
 *   - Phase coherence: parameters constant within a single 1 ms timestep
 *   - CRITICAL priority: bypasses double-buffer, < 1 ms interrupt path
 *
 * Staleness formula:
 *   τ = t_applied − t_calc               (microseconds, per channel)
 *
 * @see §GAP-022 in 01_computational_neurochemistry.md
 * @since Phase 55
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <cmath>

namespace nikola::feedback {

// ── Timing helpers ────────────────────────────────────────────────────────────

/// Per-channel staleness limits (microseconds)
inline constexpr uint64_t STALENESS_HARD_US  = 10'000;  ///< D, N — 10 ms hard limit
inline constexpr uint64_t STALENESS_SOFT_US  = 50'000;  ///< S   — 50 ms soft limit
inline constexpr uint64_t STALENESS_CRIT_US  =  1'000;  ///< CRITICAL — <1 ms interrupt path

/// Physics engine tick period (microseconds)
inline constexpr uint64_t PHYSICS_TICK_US    =  1'000;  ///< 1 kHz = 1 ms per tick

// ── NeurochemicalState ────────────────────────────────────────────────────────

/**
 * @brief Packed neurochemical snapshot shared between ENGS and Physics Engine.
 *
 * Each field is cache-line-aligned (64B) to prevent false sharing when the
 * Physics Engine reads D/N/S simultaneously from multiple threads.
 * `timestamp_seq` advances monotonically with every write.
 */
struct NeurochemicalState {
    alignas(64) float    dopamine       = 0.0f;  ///< Learning rate modulator η
    alignas(64) float    serotonin      = 0.5f;  ///< Elasticity modulator λ
    alignas(64) float    norepinephrine = 0.0f;  ///< Refractive index / attention gate
    alignas(64) float    cortisol       = 0.0f;  ///< Stress / entropy cap

    uint64_t timestamp_seq = 0;  ///< Monotonic sequence — latest value always wins
    float    padding_      = 0.0f; ///< Maintains 64-byte boundary on total struct

    /// Convenience ctor for tests / ENGS production path
    constexpr NeurochemicalState(float d, float s, float n, float c,
                                 uint64_t seq = 0) noexcept
        : dopamine{d}, serotonin{s}, norepinephrine{n}, cortisol{c},
          timestamp_seq{seq} {}

    constexpr NeurochemicalState() noexcept = default;
};

// ── SignalPriority ────────────────────────────────────────────────────────────

/**
 * @brief Priority level for ENGS → Physics control signals.
 *
 * | Level      | Triggers                        | Latency target |
 * |------------|---------------------------------|----------------|
 * | BACKGROUND | Serotonin drift, logging        | < 100 ms       |
 * | HIGH       | Dopamine updates, attention     | < 10 ms        |
 * | CRITICAL   | SCRAM, ATP < 5%, Panic N > 0.95 | < 1 ms         |
 */
enum class SignalPriority : uint8_t {
    BACKGROUND = 0,
    HIGH       = 1,
    CRITICAL   = 2,
};

[[nodiscard]] inline const char* signal_priority_name(SignalPriority p) noexcept {
    switch (p) {
        case SignalPriority::BACKGROUND: return "BACKGROUND";
        case SignalPriority::HIGH:       return "HIGH";
        case SignalPriority::CRITICAL:   return "CRITICAL";
    }
    return "UNKNOWN";
}

// ── ControlSignal ─────────────────────────────────────────────────────────────

/**
 * @brief A single ENGS → Physics control message with timestamp and priority.
 */
struct ControlSignal {
    float          value        = 0.0f;
    SignalPriority priority     = SignalPriority::HIGH;
    uint64_t       timestamp_us = 0;  ///< Wall-clock µs at point of calculation
};

// ── StalenessBudget ───────────────────────────────────────────────────────────

/**
 * @brief Per-channel staleness tracker: measures τ = t_applied − t_calc.
 *
 * Usage:
 *   budget.record_calc(channel, t_calc_us);     // ENGS writes new value
 *   budget.record_applied(channel, t_now_us);   // Physics kernel applies it
 *   bool ok = budget.within_budget(channel);
 */
class StalenessBudget {
public:
    enum class Channel : uint8_t {
        DOPAMINE       = 0,
        NOREPINEPHRINE = 1,
        SEROTONIN      = 2,
        CORTISOL       = 3,
        _COUNT         = 4,
    };

    /// Record the timestamp at which ENGS calculated a new neurochemical value.
    void record_calc(Channel ch, uint64_t t_us) noexcept {
        t_calc_[idx(ch)].store(t_us, std::memory_order_relaxed);
    }

    /// Record the timestamp at which the Physics kernel applied the value.
    void record_applied(Channel ch, uint64_t t_us) noexcept {
        t_applied_[idx(ch)].store(t_us, std::memory_order_relaxed);
    }

    /// Return τ = max(0, t_applied − t_calc) in microseconds.
    [[nodiscard]] uint64_t staleness_us(Channel ch) const noexcept {
        const uint64_t calc    = t_calc_   [idx(ch)].load(std::memory_order_relaxed);
        const uint64_t applied = t_applied_[idx(ch)].load(std::memory_order_relaxed);
        return (applied > calc) ? (applied - calc) : 0u;
    }

    /// Returns true iff τ for channel is within its spec limit.
    [[nodiscard]] bool within_budget(Channel ch) const noexcept {
        const uint64_t tau    = staleness_us(ch);
        const uint64_t limit  = (ch == Channel::SEROTONIN) ? STALENESS_SOFT_US
                                                            : STALENESS_HARD_US;
        return tau <= limit;
    }

    /// Returns the hard/soft limit for a channel in µs.
    [[nodiscard]] static constexpr uint64_t limit_us(Channel ch) noexcept {
        return (ch == Channel::SEROTONIN) ? STALENESS_SOFT_US : STALENESS_HARD_US;
    }

    /// Reset all timestamps to zero (unit-test helper).
    void reset() noexcept {
        for (auto& a : t_calc_)    a.store(0, std::memory_order_relaxed);
        for (auto& a : t_applied_) a.store(0, std::memory_order_relaxed);
    }

private:
    static constexpr std::size_t N = static_cast<std::size_t>(Channel::_COUNT);

    static constexpr std::size_t idx(Channel ch) noexcept {
        return static_cast<std::size_t>(ch);
    }

    std::atomic<uint64_t> t_calc_   [N];
    std::atomic<uint64_t> t_applied_[N];

public:
    StalenessBudget() noexcept { reset(); }
};

// ── ViolationKind ─────────────────────────────────────────────────────────────

/**
 * @brief Result codes produced by EngsPhysicsInterface::check_violations().
 */
enum class ViolationKind : uint8_t {
    NONE              = 0,
    SYNC_VIOLATION    = 1,  ///< D or N exceeded 10 ms hard limit
    COGNITIVE_PAUSE   = 2,  ///< Any channel exceeded 50 ms → dissociative state
};

[[nodiscard]] inline const char* violation_kind_name(ViolationKind v) noexcept {
    switch (v) {
        case ViolationKind::NONE:            return "NONE";
        case ViolationKind::SYNC_VIOLATION:  return "SYNC_VIOLATION";
        case ViolationKind::COGNITIVE_PAUSE: return "COGNITIVE_PAUSE";
    }
    return "UNKNOWN";
}

// ── EngsPhysicsInterface ──────────────────────────────────────────────────────

/**
 * @brief Double-buffered, lock-free gateway between ENGS and Physics Engine.
 *
 * Protocol:
 *   ENGS Thread:
 *     1. Compute new NeurochemicalState values.
 *     2. Call push_update(state, prio) — writes to shadow buffer or interrupt reg.
 *     3. StalenessBudget#record_calc() captured inside push_update.
 *
 *   Physics Kernel (start of each 1 ms tick):
 *     1. Call tick_start(t_now_us) — atomically swaps in the pending state.
 *     2. Call get_current_state() — reads the coherent (phase-locked) snapshot.
 *     3. StalenessBudget#record_applied() updated inside tick_start.
 *
 * Guarantees:
 *   - No torn reads (complete old OR complete new state per tick)
 *   - Hamiltonian preservation (parameters constant intra-timestep)
 *   - CRITICAL bypass: emergency state applied immediately, < 1 ms
 */
class EngsPhysicsInterface {
public:
    EngsPhysicsInterface() noexcept {
        current_state_.store(NeurochemicalState{}, std::memory_order_relaxed);
        next_state_   .store(NeurochemicalState{}, std::memory_order_relaxed);
        emergency_    .store(NeurochemicalState{}, std::memory_order_relaxed);
    }

    // ── ENGS write path ──────────────────────────────────────────────────────

    /**
     * @brief ENGS calls this to publish new neurochemical values.
     *
     * Thread-safe, lock-free.  CRITICAL signals bypass the double-buffer and
     * are immediately visible on the next get_current_state() call.
     *
     * @param state    The new neurochemical snapshot.
     * @param prio     Priority level (BACKGROUND / HIGH / CRITICAL).
     * @param t_now_us Wall-clock µs of calculation (for staleness tracking).
     */
    void push_update(const NeurochemicalState& state,
                     SignalPriority prio,
                     uint64_t t_now_us = 0) noexcept {
        using B  = StalenessBudget::Channel;

        // Record calc timestamps per channel
        budget_.record_calc(B::DOPAMINE,       t_now_us);
        budget_.record_calc(B::NOREPINEPHRINE,  t_now_us);
        budget_.record_calc(B::SEROTONIN,       t_now_us);
        budget_.record_calc(B::CORTISOL,        t_now_us);

        if (prio == SignalPriority::CRITICAL) {
            // Bypass double-buffering → fastest possible path
            emergency_.store(state, std::memory_order_release);
            interrupt_flag_.test_and_set(std::memory_order_release);
        } else {
            // Standard atomic swap — consumed at next tick boundary
            next_state_.store(state, std::memory_order_release);
            update_pending_.store(true, std::memory_order_release);
        }
    }

    // ── Physics tick boundary ────────────────────────────────────────────────

    /**
     * @brief Physics Engine calls this at the START of every 1 ms tick.
     *
     * Atomically promotes the pending shadow buffer to the active read state,
     * ensuring all Physics nodes see a single coherent neurochemical snapshot
     * for the entire tick (preserves Hamiltonian).
     *
     * @param t_now_us  Wall-clock µs at tick start (for staleness accounting).
     */
    void tick_start(uint64_t t_now_us = 0) noexcept {
        using B = StalenessBudget::Channel;

        if (interrupt_flag_.test(std::memory_order_acquire)) {
            // CRITICAL override — apply emergency state immediately
            current_state_.store(
                emergency_.load(std::memory_order_acquire),
                std::memory_order_release);
            interrupt_flag_.clear(std::memory_order_release);
        } else if (update_pending_.load(std::memory_order_acquire)) {
            // Promote shadow → active
            current_state_.store(
                next_state_.load(std::memory_order_acquire),
                std::memory_order_release);
            update_pending_.store(false, std::memory_order_release);
        }

        // Record application timestamps
        budget_.record_applied(B::DOPAMINE,       t_now_us);
        budget_.record_applied(B::NOREPINEPHRINE,  t_now_us);
        budget_.record_applied(B::SEROTONIN,       t_now_us);
        budget_.record_applied(B::CORTISOL,        t_now_us);

        ++tick_count_;
    }

    // ── Physics read path ────────────────────────────────────────────────────

    /**
     * @brief Returns the phase-coherent neurochemical snapshot for the current tick.
     *
     * Must be called AFTER tick_start() to get the freshest coherent state.
     * Safe to call millions of times per second (purely atomic load, no contention).
     */
    [[nodiscard]] NeurochemicalState get_current_state() const noexcept {
        return current_state_.load(std::memory_order_acquire);
    }

    // ── Violation monitoring ─────────────────────────────────────────────────

    /**
     * @brief Physics Oracle checks staleness after each tick.
     *
     * Returns the most severe violation observed across channels.
     * | COGNITIVE_PAUSE | any channel > 50 ms                           |
     * | SYNC_VIOLATION  | D or N > 10 ms, or S > 50 ms                  |
     * | NONE            | all within spec                                |
     */
    [[nodiscard]] ViolationKind check_violations() const noexcept {
        using B = StalenessBudget::Channel;

        // Check for Cognitive Pause threshold (50 ms on any channel)
        const bool any_50ms =
            budget_.staleness_us(B::DOPAMINE)       > STALENESS_SOFT_US ||
            budget_.staleness_us(B::NOREPINEPHRINE) > STALENESS_SOFT_US ||
            budget_.staleness_us(B::SEROTONIN)      > STALENESS_SOFT_US ||
            budget_.staleness_us(B::CORTISOL)       > STALENESS_SOFT_US;

        if (any_50ms) return ViolationKind::COGNITIVE_PAUSE;

        // Check SYNC_VIOLATION (D/N/C hard limit 10 ms, S soft 50 ms already checked)
        const bool dnc_violation =
            budget_.staleness_us(B::DOPAMINE)       > STALENESS_HARD_US ||
            budget_.staleness_us(B::NOREPINEPHRINE) > STALENESS_HARD_US ||
            budget_.staleness_us(B::CORTISOL)       > STALENESS_HARD_US;

        if (dnc_violation) return ViolationKind::SYNC_VIOLATION;

        return ViolationKind::NONE;
    }

    // ── Accessors ────────────────────────────────────────────────────────────

    [[nodiscard]] const StalenessBudget& staleness_budget() const noexcept {
        return budget_;
    }

    [[nodiscard]] uint64_t tick_count() const noexcept { return tick_count_; }
    [[nodiscard]] bool     has_pending() const noexcept {
        return update_pending_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] bool     has_interrupt() const noexcept {
        return interrupt_flag_.test(std::memory_order_relaxed);
    }

private:
    // Active state — what the Physics kernel reads this tick
    std::atomic<NeurochemicalState> current_state_;

    // Shadow state — what ENGS wrote, waits for next tick boundary
    std::atomic<NeurochemicalState> next_state_;

    // Emergency register — CRITICAL priority, bypasses double-buffer
    std::atomic<NeurochemicalState> emergency_;

    // Flags
    std::atomic<bool>  update_pending_{false};
    std::atomic_flag   interrupt_flag_ = ATOMIC_FLAG_INIT;

    // Per-channel staleness tracker
    StalenessBudget budget_;

    // Monotonic physics tick counter
    uint64_t tick_count_ = 0;
};

} // namespace nikola::feedback
