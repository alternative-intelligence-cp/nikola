/**
 * @file include/nikola/autonomy/goal_system.hpp
 * @brief Phase 33 — GoalSystem: hierarchical goal management with DAG
 *        dependency tracking, dopamine integration, and persistence.
 *
 * Architecture position:
 *
 *   ┌────────────────────────────────────────────────────────────────────┐
 *   │  GoalSystem                                                        │
 *   │    GoalDAG (hierarchical goals, dependency edges, cycle detection) │
 *   │    Priority scoring: urgency × importance × feasibility            │
 *   │    Active goal selection → DecisionLoop PURSUE_GOAL action         │
 *   │    Completion → DopamineSystem reward signal                       │
 *   │    Persistence → LmdbStateStore (goals DBI)                        │
 *   └────────────────────────────────────────────────────────────────────┘
 *
 * Goal tiers map to time horizons:
 *   SHORT  — achievable within a single session (minutes to hours)
 *   MID    — multi-session objectives (hours to days)
 *   LONG   — strategic direction (days to weeks+)
 *
 * The GoalDAG enforces:
 *   1. Acyclicity — insertions that create cycles are rejected
 *   2. Parent-child hierarchy — sub-goals block parent completion
 *   3. Dependency edges — goal A depends on goal B finishing first
 *
 * Thread safety: GoalSystem operations are serialised by an internal mutex.
 *
 * Phase: NIK-GOAL-01 (GoalSystem, Phase 33)
 */
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nikola::autonomy {

// ============================================================================
// Goal Tier
// ============================================================================

/// Time-horizon tier for a goal.
enum class GoalTier : uint8_t {
    SHORT = 0,  ///< Minutes to hours
    MID   = 1,  ///< Hours to days
    LONG  = 2   ///< Days to weeks+
};

/// Human-readable label for a GoalTier.
[[nodiscard]] constexpr const char* goal_tier_str(GoalTier t) noexcept {
    switch (t) {
        case GoalTier::SHORT: return "SHORT";
        case GoalTier::MID:   return "MID";
        case GoalTier::LONG:  return "LONG";
    }
    return "UNKNOWN";
}

// ============================================================================
// Goal Status
// ============================================================================

/// Lifecycle state of a goal.
enum class GoalStatus : uint8_t {
    ACTIVE      = 0,  ///< Actively being pursued
    BLOCKED     = 1,  ///< Waiting on dependency / sub-goal
    COMPLETED   = 2,  ///< Successfully achieved
    ABANDONED   = 3,  ///< Given up (cost registered)
    PAUSED      = 4   ///< Temporarily set aside
};

/// Human-readable label for a GoalStatus.
[[nodiscard]] constexpr const char* goal_status_str(GoalStatus s) noexcept {
    switch (s) {
        case GoalStatus::ACTIVE:    return "ACTIVE";
        case GoalStatus::BLOCKED:   return "BLOCKED";
        case GoalStatus::COMPLETED: return "COMPLETED";
        case GoalStatus::ABANDONED: return "ABANDONED";
        case GoalStatus::PAUSED:    return "PAUSED";
    }
    return "UNKNOWN";
}

// ============================================================================
// Goal
// ============================================================================

/// A single goal in the hierarchical system.
struct Goal {
    uint64_t    id{0};              ///< Unique identifier (monotonic)
    std::string description;        ///< Human-readable goal description
    GoalTier    tier{GoalTier::SHORT};
    GoalStatus  status{GoalStatus::ACTIVE};
    float       priority{0.5f};     ///< Computed priority ∈ [0, 1]
    float       urgency{0.5f};      ///< Time pressure ∈ [0, 1]
    float       importance{0.5f};   ///< Strategic value ∈ [0, 1]
    float       feasibility{1.0f};  ///< How achievable ∈ [0, 1]
    float       progress{0.0f};     ///< Completion fraction ∈ [0, 1]

    /// Parent goal ID (0 = top-level, no parent).
    uint64_t parent_id{0};

    /// Wall-clock timestamps.
    double created_at{0.0};         ///< Seconds since epoch
    double completed_at{0.0};       ///< Seconds since epoch (0 if not completed)

    /// True if this goal can be worked on (not blocked/completed/abandoned).
    [[nodiscard]] bool is_actionable() const noexcept {
        return status == GoalStatus::ACTIVE;
    }

    /// True if this goal is terminal (completed or abandoned).
    [[nodiscard]] bool is_terminal() const noexcept {
        return status == GoalStatus::COMPLETED ||
               status == GoalStatus::ABANDONED;
    }

    /// Compute priority from urgency × importance × feasibility.
    void recompute_priority() noexcept {
        priority = urgency * importance * feasibility;
    }
};

// ============================================================================
// GoalDAG — directed acyclic graph of goals
// ============================================================================

/**
 * @class GoalDAG
 * @brief Manages goals as nodes and dependency edges in a DAG.
 *
 * Supports:
 *   - Adding goals with optional parent (sub-goal relationship)
 *   - Adding dependency edges (goal A depends on goal B)
 *   - Cycle detection on every edge insertion
 *   - Topological ordering for scheduling
 *   - Sub-goal queries
 */
class GoalDAG {
public:
    GoalDAG() = default;

    // ── Goal management ───────────────────────────────────────────────────────

    /// Add a goal to the DAG. Returns the assigned goal ID.
    /// If parent_id != 0 and the parent doesn't exist, returns 0 (failure).
    uint64_t add_goal(Goal goal);

    /// Remove a goal and all its edges. Returns false if goal doesn't exist.
    /// Does NOT remove sub-goals — they become top-level.
    bool remove_goal(uint64_t id);

    /// Get a goal by ID. Returns nullptr if not found.
    [[nodiscard]] Goal* get(uint64_t id);
    [[nodiscard]] const Goal* get(uint64_t id) const;

    /// Get all goals.
    [[nodiscard]] std::vector<const Goal*> all_goals() const;

    /// Get all goals with the given status.
    [[nodiscard]] std::vector<const Goal*> goals_with_status(GoalStatus status) const;

    /// Number of goals in the DAG.
    [[nodiscard]] std::size_t size() const noexcept { return goals_.size(); }

    /// True if DAG is empty.
    [[nodiscard]] bool empty() const noexcept { return goals_.empty(); }

    // ── Dependency edges ──────────────────────────────────────────────────────

    /// Add a dependency: `dependent` cannot start until `dependency` is completed.
    /// Returns false if it would create a cycle, or if either goal doesn't exist.
    bool add_dependency(uint64_t dependent, uint64_t dependency);

    /// Remove a dependency edge. Returns false if edge doesn't exist.
    bool remove_dependency(uint64_t dependent, uint64_t dependency);

    /// Get all dependencies of a goal (goals that must complete first).
    [[nodiscard]] std::vector<uint64_t> dependencies_of(uint64_t id) const;

    /// Get all dependents of a goal (goals waiting on this one).
    [[nodiscard]] std::vector<uint64_t> dependents_of(uint64_t id) const;

    /// Check if a goal is blocked (has uncompleted dependencies or sub-goals).
    [[nodiscard]] bool is_blocked(uint64_t id) const;

    // ── Sub-goal queries ──────────────────────────────────────────────────────

    /// Get direct sub-goals of a goal.
    [[nodiscard]] std::vector<uint64_t> sub_goals(uint64_t parent_id) const;

    /// Check if all sub-goals of a goal are completed.
    [[nodiscard]] bool all_sub_goals_completed(uint64_t parent_id) const;

    // ── Cycle detection ───────────────────────────────────────────────────────

    /// Check if adding an edge from `from` to `to` would create a cycle.
    /// Uses DFS reachability: would `to` → ... → `from` be reachable?
    [[nodiscard]] bool would_create_cycle(uint64_t from, uint64_t to) const;

    // ── Serialization ─────────────────────────────────────────────────────────

    /// Serialize the entire DAG to a binary buffer.
    [[nodiscard]] std::vector<uint8_t> serialize() const;

    /// Deserialize a DAG from a binary buffer. Replaces current contents.
    /// Returns false on malformed data.
    bool deserialize(const uint8_t* data, std::size_t len);
    bool deserialize(const std::vector<uint8_t>& buf) {
        return deserialize(buf.data(), buf.size());
    }

    /// Clear all goals and edges.
    void clear() noexcept;

private:
    /// DFS helper for cycle detection.
    bool can_reach_(uint64_t from, uint64_t target,
                    std::unordered_set<uint64_t>& visited) const;

    /// Next goal ID counter.
    uint64_t next_id_{1};

    /// Goals indexed by ID.
    std::unordered_map<uint64_t, Goal> goals_;

    /// Forward edges: dependent → {dependencies}.
    std::unordered_map<uint64_t, std::unordered_set<uint64_t>> deps_;

    /// Reverse edges: dependency → {dependents}.
    std::unordered_map<uint64_t, std::unordered_set<uint64_t>> rdeps_;
};

// ============================================================================
// GoalSystem — high-level goal management
// ============================================================================

/// Configuration for the GoalSystem.
struct GoalSystemConfig {
    /// Dopamine reward scaling for goal completion by tier.
    float short_reward  = 0.3f;
    float mid_reward    = 0.6f;
    float long_reward   = 1.0f;

    /// Dopamine penalty for goal abandonment.
    float abandon_penalty = -0.15f;

    /// Progress increment that triggers proportional dopamine.
    float progress_dopamine_scale = 0.1f;

    /// Auto-generate training goals when boredom exceeds this.
    float boredom_training_threshold = 0.8f;

    /// Auto-generate exploration goals when dopamine is below this for N cycles.
    float low_dopamine_threshold = 0.2f;
    int   low_dopamine_cycles    = 50;
};

/**
 * @class GoalSystem
 * @brief High-level goal manager with dopamine integration and
 *        autonomous motivation.
 *
 * Wraps GoalDAG with:
 *   - Dopamine reward/penalty signals on goal events
 *   - Priority scoring and active goal selection
 *   - Autonomous goal generation from cognitive state
 *   - Thread-safe operations via internal mutex
 *
 * Lifecycle:
 *   1. Construct with GoalSystemConfig
 *   2. Set dopamine callback via set_reward_fn()
 *   3. Create goals, update progress, complete/abandon
 *   4. Call active_goal() to get the highest-priority actionable goal
 */
class GoalSystem {
public:
    /// Reward signal callback — called on goal events.
    /// Parameters: (dopamine_delta, event_description)
    using RewardFn = std::function<void(float delta, const std::string& description)>;

    explicit GoalSystem(GoalSystemConfig config = {});

    // ── Configuration ─────────────────────────────────────────────────────────

    /// Set the reward callback. Called on goal completion, progress, abandonment.
    void set_reward_fn(RewardFn fn);

    /// Access the configuration.
    [[nodiscard]] const GoalSystemConfig& config() const noexcept { return config_; }

    // ── Goal creation ─────────────────────────────────────────────────────────

    /// Create a new goal. Returns the goal ID.
    uint64_t create_goal(const std::string& description,
                         GoalTier tier,
                         float urgency = 0.5f,
                         float importance = 0.5f,
                         uint64_t parent_id = 0);

    /// Add a dependency between goals. Returns false on cycle detection.
    bool add_dependency(uint64_t dependent, uint64_t dependency);

    // ── Goal lifecycle ────────────────────────────────────────────────────────

    /// Mark a goal as completed. Triggers dopamine reward.
    /// If the goal has uncompleted sub-goals, returns false.
    bool complete_goal(uint64_t id);

    /// Abandon a goal. Triggers small negative dopamine signal.
    bool abandon_goal(uint64_t id);

    /// Pause a goal (temporarily deprioritise).
    bool pause_goal(uint64_t id);

    /// Resume a paused goal.
    bool resume_goal(uint64_t id);

    /// Update progress on a goal. Triggers proportional dopamine.
    /// progress must be ∈ [0, 1]. Returns false if goal not found.
    bool update_progress(uint64_t id, float progress);

    // ── Goal queries ──────────────────────────────────────────────────────────

    /// Get the highest-priority actionable (unblocked, active) goal.
    /// Returns nullptr if no actionable goals exist.
    [[nodiscard]] const Goal* active_goal() const;

    /// Get a goal by ID.
    [[nodiscard]] const Goal* get_goal(uint64_t id) const;

    /// Get all goals with the given status.
    [[nodiscard]] std::vector<const Goal*> goals_with_status(GoalStatus status) const;

    /// Get all goals.
    [[nodiscard]] std::vector<const Goal*> all_goals() const;

    /// Number of goals.
    [[nodiscard]] std::size_t goal_count() const noexcept;

    /// Number of completed goals.
    [[nodiscard]] std::size_t completed_count() const noexcept { return completed_count_; }

    /// Number of abandoned goals.
    [[nodiscard]] std::size_t abandoned_count() const noexcept { return abandoned_count_; }

    // ── Autonomous motivation ─────────────────────────────────────────────────

    /// Check cognitive state and auto-generate goals if warranted.
    /// Called from the decision loop each tick.
    /// @param boredom    Current boredom level ∈ [0, 1]
    /// @param dopamine   Current dopamine level ∈ [0, 1]
    /// @param tick       Current tick count (for rate limiting)
    void check_motivation(float boredom, float dopamine, uint64_t tick);

    // ── Persistence ───────────────────────────────────────────────────────────

    /// Serialize the full goal state to a binary buffer.
    [[nodiscard]] std::vector<uint8_t> serialize() const;

    /// Deserialize goal state from a binary buffer. Replaces current state.
    bool deserialize(const uint8_t* data, std::size_t len);
    bool deserialize(const std::vector<uint8_t>& buf) {
        return deserialize(buf.data(), buf.size());
    }

    /// Direct access to the DAG (for advanced queries / testing).
    [[nodiscard]] const GoalDAG& dag() const noexcept { return dag_; }

private:
    /// Emit a reward signal through the callback.
    void emit_reward_(float delta, const std::string& description);

    /// Reward scale for a goal tier.
    [[nodiscard]] float tier_reward_(GoalTier tier) const noexcept;

    /// Refresh blocked status for all goals (after completion/dependency change).
    void refresh_blocked_status_();

    GoalSystemConfig config_;
    GoalDAG          dag_;
    RewardFn         reward_fn_;

    std::size_t completed_count_{0};
    std::size_t abandoned_count_{0};

    /// Rate-limiting for auto-generated goals.
    uint64_t last_training_goal_tick_{0};
    uint64_t last_exploration_goal_tick_{0};
    int      low_dopamine_streak_{0};

    mutable std::mutex mutex_;
};

} // namespace nikola::autonomy
