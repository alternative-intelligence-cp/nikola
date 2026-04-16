/**
 * @file src/autonomy/goal_system.cpp
 * @brief Phase 33 — GoalSystem implementation.
 *
 * Implements:
 *   - GoalDAG: add/remove goals, dependency edges, cycle detection,
 *     sub-goal queries, binary serialization
 *   - GoalSystem: high-level wrapper with dopamine integration,
 *     priority scoring, autonomous motivation, persistence
 */

#include <nikola/autonomy/goal_system.hpp>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <queue>
#include <stdexcept>

namespace nikola::autonomy {

// ============================================================================
// Serialization helpers (hand-rolled binary, no external libs)
// ============================================================================

namespace {

inline void write_u8(std::vector<uint8_t>& buf, uint8_t v) {
    buf.push_back(v);
}

inline void write_u64(std::vector<uint8_t>& buf, uint64_t v) {
    for (int i = 7; i >= 0; --i)
        buf.push_back(static_cast<uint8_t>((v >> (i * 8)) & 0xFF));
}

inline void write_f32(std::vector<uint8_t>& buf, float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    for (int i = 3; i >= 0; --i)
        buf.push_back(static_cast<uint8_t>((bits >> (i * 8)) & 0xFF));
}

inline void write_f64(std::vector<uint8_t>& buf, double v) {
    uint64_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    write_u64(buf, bits);
}

inline void write_str(std::vector<uint8_t>& buf, const std::string& s) {
    write_u64(buf, s.size());
    buf.insert(buf.end(), s.begin(), s.end());
}

struct Reader {
    const uint8_t* data;
    std::size_t    len;
    std::size_t    pos{0};

    bool has(std::size_t n) const { return pos + n <= len; }

    uint8_t read_u8() {
        if (!has(1)) throw std::runtime_error("truncated");
        return data[pos++];
    }

    uint64_t read_u64() {
        if (!has(8)) throw std::runtime_error("truncated");
        uint64_t v = 0;
        for (int i = 0; i < 8; ++i)
            v = (v << 8) | data[pos++];
        return v;
    }

    float read_f32() {
        if (!has(4)) throw std::runtime_error("truncated");
        uint32_t bits = 0;
        for (int i = 0; i < 4; ++i)
            bits = (bits << 8) | data[pos++];
        float v;
        std::memcpy(&v, &bits, sizeof(v));
        return v;
    }

    double read_f64() {
        uint64_t bits = read_u64();
        double v;
        std::memcpy(&v, &bits, sizeof(v));
        return v;
    }

    std::string read_str() {
        auto slen = read_u64();
        if (!has(slen)) throw std::runtime_error("truncated");
        std::string s(reinterpret_cast<const char*>(data + pos), slen);
        pos += slen;
        return s;
    }
};

double now_seconds() {
    auto t = std::chrono::system_clock::now().time_since_epoch();
    return std::chrono::duration<double>(t).count();
}

} // anon namespace

// ============================================================================
// GoalDAG
// ============================================================================

uint64_t GoalDAG::add_goal(Goal goal)
{
    // Validate parent exists if specified
    if (goal.parent_id != 0 && goals_.find(goal.parent_id) == goals_.end()) {
        return 0;
    }

    // Assign ID
    goal.id = next_id_++;
    if (goal.created_at == 0.0) {
        goal.created_at = now_seconds();
    }
    goal.recompute_priority();

    uint64_t id = goal.id;
    goals_.emplace(id, std::move(goal));
    return id;
}

bool GoalDAG::remove_goal(uint64_t id)
{
    auto it = goals_.find(id);
    if (it == goals_.end()) return false;

    // Remove all forward edges from this goal
    if (auto dit = deps_.find(id); dit != deps_.end()) {
        for (uint64_t dep : dit->second) {
            if (auto rit = rdeps_.find(dep); rit != rdeps_.end()) {
                rit->second.erase(id);
                if (rit->second.empty()) rdeps_.erase(rit);
            }
        }
        deps_.erase(dit);
    }

    // Remove all reverse edges to this goal
    if (auto rit = rdeps_.find(id); rit != rdeps_.end()) {
        for (uint64_t dependent : rit->second) {
            if (auto dit = deps_.find(dependent); dit != deps_.end()) {
                dit->second.erase(id);
                if (dit->second.empty()) deps_.erase(dit);
            }
        }
        rdeps_.erase(rit);
    }

    // Re-parent sub-goals to top-level
    for (auto& [gid, g] : goals_) {
        if (g.parent_id == id) {
            g.parent_id = 0;
        }
    }

    goals_.erase(it);
    return true;
}

Goal* GoalDAG::get(uint64_t id)
{
    auto it = goals_.find(id);
    return it != goals_.end() ? &it->second : nullptr;
}

const Goal* GoalDAG::get(uint64_t id) const
{
    auto it = goals_.find(id);
    return it != goals_.end() ? &it->second : nullptr;
}

std::vector<const Goal*> GoalDAG::all_goals() const
{
    std::vector<const Goal*> result;
    result.reserve(goals_.size());
    for (const auto& [id, g] : goals_) {
        result.push_back(&g);
    }
    return result;
}

std::vector<const Goal*> GoalDAG::goals_with_status(GoalStatus status) const
{
    std::vector<const Goal*> result;
    for (const auto& [id, g] : goals_) {
        if (g.status == status) result.push_back(&g);
    }
    return result;
}

bool GoalDAG::add_dependency(uint64_t dependent, uint64_t dependency)
{
    // Both must exist
    if (goals_.find(dependent) == goals_.end() ||
        goals_.find(dependency) == goals_.end()) {
        return false;
    }

    // Self-dependency is a cycle
    if (dependent == dependency) return false;

    // Check if edge already exists
    if (auto it = deps_.find(dependent); it != deps_.end()) {
        if (it->second.count(dependency)) return true;  // Already present
    }

    // Cycle detection: would adding dependent → dependency create a path
    // dependency → ... → dependent?
    if (would_create_cycle(dependent, dependency)) return false;

    deps_[dependent].insert(dependency);
    rdeps_[dependency].insert(dependent);
    return true;
}

bool GoalDAG::remove_dependency(uint64_t dependent, uint64_t dependency)
{
    auto dit = deps_.find(dependent);
    if (dit == deps_.end()) return false;
    auto erased = dit->second.erase(dependency);
    if (erased == 0) return false;
    if (dit->second.empty()) deps_.erase(dit);

    auto rit = rdeps_.find(dependency);
    if (rit != rdeps_.end()) {
        rit->second.erase(dependent);
        if (rit->second.empty()) rdeps_.erase(rit);
    }
    return true;
}

std::vector<uint64_t> GoalDAG::dependencies_of(uint64_t id) const
{
    auto it = deps_.find(id);
    if (it == deps_.end()) return {};
    return {it->second.begin(), it->second.end()};
}

std::vector<uint64_t> GoalDAG::dependents_of(uint64_t id) const
{
    auto it = rdeps_.find(id);
    if (it == rdeps_.end()) return {};
    return {it->second.begin(), it->second.end()};
}

bool GoalDAG::is_blocked(uint64_t id) const
{
    // Check explicit dependencies
    auto dit = deps_.find(id);
    if (dit != deps_.end()) {
        for (uint64_t dep_id : dit->second) {
            const auto* dep = get(dep_id);
            if (dep && !dep->is_terminal()) return true;
        }
    }

    // Check sub-goals
    if (!all_sub_goals_completed(id)) return true;

    return false;
}

std::vector<uint64_t> GoalDAG::sub_goals(uint64_t parent_id) const
{
    std::vector<uint64_t> result;
    for (const auto& [id, g] : goals_) {
        if (g.parent_id == parent_id) result.push_back(id);
    }
    return result;
}

bool GoalDAG::all_sub_goals_completed(uint64_t parent_id) const
{
    for (const auto& [id, g] : goals_) {
        if (g.parent_id == parent_id && !g.is_terminal()) {
            return false;
        }
    }
    return true;
}

bool GoalDAG::would_create_cycle(uint64_t from, uint64_t to) const
{
    // Adding edge from → to creates a cycle if `to` can already reach `from`
    // through existing edges.
    std::unordered_set<uint64_t> visited;
    return can_reach_(to, from, visited);
}

bool GoalDAG::can_reach_(uint64_t from, uint64_t target,
                          std::unordered_set<uint64_t>& visited) const
{
    if (from == target) return true;
    if (!visited.insert(from).second) return false;

    // Follow forward dependency edges
    auto dit = deps_.find(from);
    if (dit != deps_.end()) {
        for (uint64_t next : dit->second) {
            if (can_reach_(next, target, visited)) return true;
        }
    }
    return false;
}

// ── Serialization ──────────────────────────────────────────────────────────

// Wire format:
//   [4] magic "GDAG"
//   [1] version (1)
//   [8] next_id_
//   [8] goal_count
//   For each goal:
//     [8] id, [str] description, [1] tier, [1] status
//     [4] priority, [4] urgency, [4] importance, [4] feasibility, [4] progress
//     [8] parent_id, [8] created_at_bits, [8] completed_at_bits
//   [8] edge_count
//   For each edge:
//     [8] dependent, [8] dependency

std::vector<uint8_t> GoalDAG::serialize() const
{
    std::vector<uint8_t> buf;
    buf.reserve(256 + goals_.size() * 128);

    // Header
    buf.push_back('G'); buf.push_back('D');
    buf.push_back('A'); buf.push_back('G');
    write_u8(buf, 1);  // version
    write_u64(buf, next_id_);
    write_u64(buf, goals_.size());

    // Goals
    for (const auto& [id, g] : goals_) {
        write_u64(buf, g.id);
        write_str(buf, g.description);
        write_u8(buf, static_cast<uint8_t>(g.tier));
        write_u8(buf, static_cast<uint8_t>(g.status));
        write_f32(buf, g.priority);
        write_f32(buf, g.urgency);
        write_f32(buf, g.importance);
        write_f32(buf, g.feasibility);
        write_f32(buf, g.progress);
        write_u64(buf, g.parent_id);
        write_f64(buf, g.created_at);
        write_f64(buf, g.completed_at);
    }

    // Count edges
    std::size_t edge_count = 0;
    for (const auto& [dep, set] : deps_) {
        edge_count += set.size();
    }
    write_u64(buf, edge_count);

    // Edges
    for (const auto& [dependent, set] : deps_) {
        for (uint64_t dependency : set) {
            write_u64(buf, dependent);
            write_u64(buf, dependency);
        }
    }

    return buf;
}

bool GoalDAG::deserialize(const uint8_t* data, std::size_t len)
{
    try {
        Reader r{data, len};

        // Magic
        if (r.read_u8() != 'G' || r.read_u8() != 'D' ||
            r.read_u8() != 'A' || r.read_u8() != 'G') {
            return false;
        }

        uint8_t version = r.read_u8();
        if (version != 1) return false;

        clear();

        next_id_ = r.read_u64();
        uint64_t goal_count = r.read_u64();

        for (uint64_t i = 0; i < goal_count; ++i) {
            Goal g;
            g.id = r.read_u64();
            g.description = r.read_str();
            g.tier = static_cast<GoalTier>(r.read_u8());
            g.status = static_cast<GoalStatus>(r.read_u8());
            g.priority = r.read_f32();
            g.urgency = r.read_f32();
            g.importance = r.read_f32();
            g.feasibility = r.read_f32();
            g.progress = r.read_f32();
            g.parent_id = r.read_u64();
            g.created_at = r.read_f64();
            g.completed_at = r.read_f64();
            goals_.emplace(g.id, std::move(g));
        }

        uint64_t edge_count = r.read_u64();
        for (uint64_t i = 0; i < edge_count; ++i) {
            uint64_t dependent = r.read_u64();
            uint64_t dependency = r.read_u64();
            deps_[dependent].insert(dependency);
            rdeps_[dependency].insert(dependent);
        }

        return true;
    } catch (const std::exception&) {
        return false;
    }
}

void GoalDAG::clear() noexcept
{
    goals_.clear();
    deps_.clear();
    rdeps_.clear();
    next_id_ = 1;
}

// ============================================================================
// GoalSystem
// ============================================================================

GoalSystem::GoalSystem(GoalSystemConfig config)
    : config_(std::move(config))
{}

void GoalSystem::set_reward_fn(RewardFn fn)
{
    std::lock_guard<std::mutex> lock(mutex_);
    reward_fn_ = std::move(fn);
}

uint64_t GoalSystem::create_goal(const std::string& description,
                                  GoalTier tier,
                                  float urgency,
                                  float importance,
                                  uint64_t parent_id)
{
    std::lock_guard<std::mutex> lock(mutex_);

    Goal g;
    g.description = description;
    g.tier = tier;
    g.urgency = urgency;
    g.importance = importance;
    g.parent_id = parent_id;

    uint64_t id = dag_.add_goal(std::move(g));
    if (id != 0) {
        refresh_blocked_status_();
    }
    return id;
}

bool GoalSystem::add_dependency(uint64_t dependent, uint64_t dependency)
{
    std::lock_guard<std::mutex> lock(mutex_);
    bool ok = dag_.add_dependency(dependent, dependency);
    if (ok) {
        refresh_blocked_status_();
    }
    return ok;
}

bool GoalSystem::complete_goal(uint64_t id)
{
    std::lock_guard<std::mutex> lock(mutex_);

    Goal* g = dag_.get(id);
    if (!g || g->is_terminal()) return false;

    // Cannot complete if sub-goals are still active
    if (!dag_.all_sub_goals_completed(id)) return false;

    g->status = GoalStatus::COMPLETED;
    g->progress = 1.0f;
    g->completed_at = now_seconds();
    ++completed_count_;

    // Dopamine reward proportional to tier
    float reward = tier_reward_(g->tier);
    emit_reward_(reward, "goal_completed: " + g->description);

    // Refresh blocked status (dependents may become unblocked)
    refresh_blocked_status_();

    return true;
}

bool GoalSystem::abandon_goal(uint64_t id)
{
    std::lock_guard<std::mutex> lock(mutex_);

    Goal* g = dag_.get(id);
    if (!g || g->is_terminal()) return false;

    g->status = GoalStatus::ABANDONED;
    g->completed_at = now_seconds();
    ++abandoned_count_;

    emit_reward_(config_.abandon_penalty,
                 "goal_abandoned: " + g->description);

    refresh_blocked_status_();
    return true;
}

bool GoalSystem::pause_goal(uint64_t id)
{
    std::lock_guard<std::mutex> lock(mutex_);

    Goal* g = dag_.get(id);
    if (!g || g->status != GoalStatus::ACTIVE) return false;

    g->status = GoalStatus::PAUSED;
    return true;
}

bool GoalSystem::resume_goal(uint64_t id)
{
    std::lock_guard<std::mutex> lock(mutex_);

    Goal* g = dag_.get(id);
    if (!g || g->status != GoalStatus::PAUSED) return false;

    g->status = GoalStatus::ACTIVE;
    refresh_blocked_status_();
    return true;
}

bool GoalSystem::update_progress(uint64_t id, float progress)
{
    std::lock_guard<std::mutex> lock(mutex_);

    Goal* g = dag_.get(id);
    if (!g || g->is_terminal()) return false;

    float old_progress = g->progress;
    g->progress = std::clamp(progress, 0.0f, 1.0f);

    // Proportional dopamine on forward progress
    float delta = g->progress - old_progress;
    if (delta > 0.0f) {
        float reward = delta * config_.progress_dopamine_scale;
        emit_reward_(reward, "goal_progress: " + g->description);
    }

    return true;
}

const Goal* GoalSystem::active_goal() const
{
    std::lock_guard<std::mutex> lock(mutex_);

    const Goal* best = nullptr;
    float best_priority = -1.0f;

    for (const auto* g : dag_.all_goals()) {
        if (g->is_actionable() && !dag_.is_blocked(g->id)) {
            if (g->priority > best_priority) {
                best = g;
                best_priority = g->priority;
            }
        }
    }
    return best;
}

const Goal* GoalSystem::get_goal(uint64_t id) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return dag_.get(id);
}

std::vector<const Goal*> GoalSystem::goals_with_status(GoalStatus status) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return dag_.goals_with_status(status);
}

std::vector<const Goal*> GoalSystem::all_goals() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return dag_.all_goals();
}

std::size_t GoalSystem::goal_count() const noexcept
{
    std::lock_guard<std::mutex> lock(mutex_);
    return dag_.size();
}

void GoalSystem::check_motivation(float boredom, float dopamine, uint64_t tick)
{
    std::lock_guard<std::mutex> lock(mutex_);

    // Track sustained low dopamine
    if (dopamine < config_.low_dopamine_threshold) {
        ++low_dopamine_streak_;
    } else {
        low_dopamine_streak_ = 0;
    }

    // Auto-generate training goal on high boredom
    // Rate limit: at most once per 500 ticks
    if (boredom > config_.boredom_training_threshold &&
        tick - last_training_goal_tick_ > 500) {

        Goal g;
        g.description = "Self-directed training: improve weakest cognitive area";
        g.tier = GoalTier::SHORT;
        g.urgency = boredom;
        g.importance = 0.6f;
        g.feasibility = 0.9f;

        if (dag_.add_goal(std::move(g)) != 0) {
            last_training_goal_tick_ = tick;
        }
    }

    // Auto-generate exploration goal on sustained low dopamine
    if (low_dopamine_streak_ >= config_.low_dopamine_cycles &&
        tick - last_exploration_goal_tick_ > 500) {

        Goal g;
        g.description = "Exploration: seek novel stimuli to recalibrate reward system";
        g.tier = GoalTier::SHORT;
        g.urgency = 0.7f;
        g.importance = 0.5f;
        g.feasibility = 1.0f;

        if (dag_.add_goal(std::move(g)) != 0) {
            last_exploration_goal_tick_ = tick;
            low_dopamine_streak_ = 0;
        }
    }
}

std::vector<uint8_t> GoalSystem::serialize() const
{
    std::lock_guard<std::mutex> lock(mutex_);

    // Serialize DAG + GoalSystem metadata
    auto buf = dag_.serialize();

    // Append system counters
    write_u64(buf, completed_count_);
    write_u64(buf, abandoned_count_);
    write_u64(buf, last_training_goal_tick_);
    write_u64(buf, last_exploration_goal_tick_);

    return buf;
}

bool GoalSystem::deserialize(const uint8_t* data, std::size_t len)
{
    std::lock_guard<std::mutex> lock(mutex_);

    if (!dag_.deserialize(data, len)) return false;

    // Read system counters from after DAG data
    // Re-serialize DAG to find where it ends, then read counters
    auto dag_buf = dag_.serialize();
    std::size_t dag_len = dag_buf.size();

    if (len >= dag_len + 32) {
        Reader r{data, len};
        r.pos = dag_len;
        completed_count_ = r.read_u64();
        abandoned_count_ = r.read_u64();
        last_training_goal_tick_ = r.read_u64();
        last_exploration_goal_tick_ = r.read_u64();
    } else {
        // Just DAG data, no counters (backward compat)
        completed_count_ = 0;
        abandoned_count_ = 0;
        last_training_goal_tick_ = 0;
        last_exploration_goal_tick_ = 0;
    }

    return true;
}

void GoalSystem::emit_reward_(float delta, const std::string& description)
{
    if (reward_fn_) {
        reward_fn_(delta, description);
    }
}

float GoalSystem::tier_reward_(GoalTier tier) const noexcept
{
    switch (tier) {
        case GoalTier::SHORT: return config_.short_reward;
        case GoalTier::MID:   return config_.mid_reward;
        case GoalTier::LONG:  return config_.long_reward;
    }
    return config_.short_reward;
}

void GoalSystem::refresh_blocked_status_()
{
    for (const auto* cg : dag_.all_goals()) {
        Goal* g = dag_.get(cg->id);
        if (!g || g->is_terminal()) continue;

        if (dag_.is_blocked(g->id)) {
            if (g->status == GoalStatus::ACTIVE) {
                g->status = GoalStatus::BLOCKED;
            }
        } else {
            if (g->status == GoalStatus::BLOCKED) {
                g->status = GoalStatus::ACTIVE;
            }
        }
    }
}

} // namespace nikola::autonomy
