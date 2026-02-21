/**
 * @file task_dispatcher.hpp
 * @brief Priority-aware task dispatcher for the Nikola Orchestrator.
 *
 * Implements a thread-safe work queue whose tasks are ordered by priority
 * (CRITICAL → HIGH → NORMAL → LOW) rather than insertion order.  Within
 * the same priority level, tasks are dispatched FIFO via a monotonic
 * sequence counter.
 *
 * Motivation
 * ==========
 * The original orchestrator dispatched all internal work via a plain FIFO
 * queue (implicit ZMQ send order).  This caused physics-tick callbacks and
 * TTS "speak" commands to compete unfairly with watchdog restarts.  The
 * TaskDispatcher replaces that implicit ordering with explicit priorities:
 *
 *   CRITICAL (0) — watchdog restarts, component fault recovery
 *   HIGH     (1) — physics cycle callbacks, sensor ingestion
 *   NORMAL   (2) — cognitive generation, TTS scheduling
 *   LOW      (3) — persistence checkpoints, background analytics
 *
 * Thread Safety
 * =============
 * All public methods are thread-safe.  Multiple producers may call
 * enqueue() concurrently; multiple consumers may call process_one() /
 * process_all() concurrently (each task is dispatched exactly once).
 *
 * Integration with Orchestrator
 * =============================
 * The orchestrator Impl now holds a TaskDispatcher instance.  The watchdog
 * thread periodically calls process_all() to drain the queue.  Upstream
 * callers inject tasks via Orchestrator::enqueue_task().
 *
 * Phase 13, Nikola v0.0.4
 */

#pragma once

#include <atomic>
#include <functional>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

namespace nikola::infrastructure {

// ---------------------------------------------------------------------------
// TaskPriority
// ---------------------------------------------------------------------------

/**
 * @brief Numeric task priority.  Lower value = higher urgency.
 */
enum class TaskPriority : int {
    CRITICAL = 0,   ///< Fault recovery, watchdog restarts
    HIGH     = 1,   ///< Physics ticks, sensor ingestion
    NORMAL   = 2,   ///< Cognitive generation, TTS scheduling
    LOW      = 3,   ///< Checkpoints, background analytics
};

[[nodiscard]] inline const char* task_priority_name(TaskPriority p) noexcept
{
    switch (p) {
        case TaskPriority::CRITICAL: return "CRITICAL";
        case TaskPriority::HIGH:     return "HIGH";
        case TaskPriority::NORMAL:   return "NORMAL";
        case TaskPriority::LOW:      return "LOW";
    }
    return "UNKNOWN";
}

// ---------------------------------------------------------------------------
// Task
// ---------------------------------------------------------------------------

/**
 * @brief A dispatchable unit of work with identity, priority, and payload.
 */
struct Task {
    TaskPriority              priority   = TaskPriority::NORMAL;
    std::string               name;         ///< Human-readable label
    std::function<void()>     fn;           ///< Work to execute
    uint64_t                  seq_id = 0;   ///< Monotonic insertion counter

    // Priority queue ordering: lower priority value first; FIFO within tier.
    bool operator>(const Task& o) const noexcept
    {
        if (static_cast<int>(priority) != static_cast<int>(o.priority))
            return static_cast<int>(priority) > static_cast<int>(o.priority);
        return seq_id > o.seq_id;  // higher seq_id = enqueued later → lower urgency
    }
};

// ---------------------------------------------------------------------------
// TaskStats
// ---------------------------------------------------------------------------

struct TaskStats {
    uint64_t enqueued_critical = 0;
    uint64_t enqueued_high     = 0;
    uint64_t enqueued_normal   = 0;
    uint64_t enqueued_low      = 0;
    uint64_t dispatched        = 0;
    uint64_t errors            = 0;
};

// ---------------------------------------------------------------------------
// TaskDispatcher
// ---------------------------------------------------------------------------

/**
 * @brief Thread-safe priority task queue.
 *
 * @code
 *   TaskDispatcher td;
 *   td.enqueue(TaskPriority::HIGH, "physics_tick", []{ tick(); });
 *   td.enqueue(TaskPriority::LOW,  "checkpoint",   []{ save(); });
 *
 *   // In scheduler loop:
 *   td.process_all();
 * @endcode
 */
class TaskDispatcher {
public:
    TaskDispatcher() = default;

    // ------------------------------------------------------------------
    // Enqueueing
    // ------------------------------------------------------------------

    /**
     * @brief Add a task to the dispatcher.
     *
     * @param priority  Task priority tier.
     * @param name      Human-readable label (for diagnostics).
     * @param fn        Callable to invoke when dispatched.
     * @throws std::invalid_argument if @p fn is null.
     */
    void enqueue(TaskPriority priority, std::string name, std::function<void()> fn)
    {
        if (!fn) throw std::invalid_argument("TaskDispatcher: null task fn");

        const uint64_t seq = next_seq_.fetch_add(1, std::memory_order_relaxed);

        Task t;
        t.priority = priority;
        t.name     = std::move(name);
        t.fn       = std::move(fn);
        t.seq_id   = seq;

        {
            std::lock_guard<std::mutex> g(mutex_);
            queue_.push(std::move(t));
        }

        // Update stats
        switch (priority) {
            case TaskPriority::CRITICAL: stats_.enqueued_critical++; break;
            case TaskPriority::HIGH:     stats_.enqueued_high++;     break;
            case TaskPriority::NORMAL:   stats_.enqueued_normal++;   break;
            case TaskPriority::LOW:      stats_.enqueued_low++;      break;
        }
    }

    // ------------------------------------------------------------------
    // Processing
    // ------------------------------------------------------------------

    /**
     * @brief Pop and execute the highest-priority pending task.
     * @return true if a task was dispatched; false if the queue was empty.
     */
    bool process_one()
    {
        Task t;
        {
            std::lock_guard<std::mutex> g(mutex_);
            if (queue_.empty()) return false;
            t = queue_.top();
            queue_.pop();
        }
        try {
            t.fn();
        } catch (...) {
            ++stats_.errors;
        }
        ++stats_.dispatched;
        return true;
    }

    /**
     * @brief Drain up to @p max_tasks tasks in priority order.
     *
     * @param max_tasks  Maximum tasks to run (0 = unlimited).
     * @return Number of tasks dispatched.
     */
    std::size_t process_all(std::size_t max_tasks = 0)
    {
        std::size_t count = 0;
        while (process_one()) {
            ++count;
            if (max_tasks > 0 && count >= max_tasks) break;
        }
        return count;
    }

    /**
     * @brief Drains all tasks of priority ≤ @p max_priority.
     *
     * Useful for servicing only CRITICAL + HIGH tasks during overload.
     *
     * @return Number of tasks dispatched.
     */
    std::size_t process_up_to(TaskPriority max_priority)
    {
        std::size_t count = 0;
        while (true) {
            std::optional<Task> t;
            {
                std::lock_guard<std::mutex> g(mutex_);
                if (queue_.empty()) break;
                if (queue_.top().priority > max_priority) break;
                t = queue_.top();
                queue_.pop();
            }
            if (!t.has_value()) break;
            try {
                t->fn();
            } catch (...) {
                ++stats_.errors;
            }
            ++stats_.dispatched;
            ++count;
        }
        return count;
    }

    // ------------------------------------------------------------------
    // Observation
    // ------------------------------------------------------------------

    /** @brief Number of pending tasks. */
    [[nodiscard]] std::size_t size() const
    {
        std::lock_guard<std::mutex> g(mutex_);
        return queue_.size();
    }

    [[nodiscard]] bool empty() const
    {
        std::lock_guard<std::mutex> g(mutex_);
        return queue_.empty();
    }

    /**
     * @brief Peek at the priority of the next task without removing it.
     * @return TaskPriority of the front task, or nullopt if empty.
     */
    [[nodiscard]] std::optional<TaskPriority> peek_priority() const
    {
        std::lock_guard<std::mutex> g(mutex_);
        if (queue_.empty()) return std::nullopt;
        return queue_.top().priority;
    }

    /**
     * @brief Peek at the name of the next task without removing it.
     */
    [[nodiscard]] std::optional<std::string> peek_name() const
    {
        std::lock_guard<std::mutex> g(mutex_);
        if (queue_.empty()) return std::nullopt;
        return queue_.top().name;
    }

    /** @brief Cumulative statistics since construction. */
    [[nodiscard]] TaskStats stats() const noexcept { return stats_; }

    /** @brief Clear all pending (un-dispatched) tasks. */
    void clear()
    {
        std::lock_guard<std::mutex> g(mutex_);
        while (!queue_.empty()) queue_.pop();
    }

private:
    using PQ = std::priority_queue<Task,
                                   std::vector<Task>,
                                   std::greater<Task>>;

    mutable std::mutex    mutex_;
    PQ                    queue_;
    std::atomic<uint64_t> next_seq_{0};
    TaskStats             stats_;
};

} // namespace nikola::infrastructure
