/**
 * @file phase13_task_dispatcher_test.cpp
 * @brief Phase 13 unit tests — TaskDispatcher (Priority-aware Orchestrator)
 *
 * Covers:
 *   - task_priority_name(): all 4 values
 *   - Task::operator>: lower priority int = higher urgency
 *   - TaskDispatcher: initial empty
 *   - enqueue() + process_one(): task is called
 *   - process_one() on empty queue → false
 *   - CRITICAL executes before LOW regardless of insertion order
 *   - HIGH executes before NORMAL
 *   - FIFO order preserved within the same priority tier
 *   - process_all() drains the queue + returns count
 *   - process_all(max) limits dispatched count
 *   - process_up_to(HIGH): skips NORMAL and LOW tasks
 *   - size() / empty() correct throughout
 *   - peek_priority() / peek_name() correct
 *   - clear() empties queue
 *   - stats: enqueued per tier, dispatched, errors
 *   - Exception in task fn increments error count, continues
 *   - Thread safety: concurrent enqueue + process_all
 *   - Orchestrator: enqueue_task / process_pending_tasks / task_queue_size
 *   - Orchestrator: task_stats dispatched counter
 *   - null fn → enqueue throws
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/infrastructure/task_dispatcher.hpp>

// Provide Orchestrator::Impl definition (Pimpl header-only implementation gate)
#define NIKOLA_ORCHESTRATOR_IMPL
#include <nikola/infrastructure/orchestrator.hpp>

#include <atomic>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using nikola::infrastructure::TaskDispatcher;
using nikola::infrastructure::TaskPriority;
using nikola::infrastructure::task_priority_name;
using nikola::infrastructure::Orchestrator;
using nikola::infrastructure::OrchestratorConfig;

// ===========================================================================
// task_priority_name
// ===========================================================================

TEST_CASE("task_priority_name: CRITICAL", "[dispatcher][priority]") {
    CHECK(std::string(task_priority_name(TaskPriority::CRITICAL)) == "CRITICAL");
}

TEST_CASE("task_priority_name: HIGH", "[dispatcher][priority]") {
    CHECK(std::string(task_priority_name(TaskPriority::HIGH)) == "HIGH");
}

TEST_CASE("task_priority_name: NORMAL", "[dispatcher][priority]") {
    CHECK(std::string(task_priority_name(TaskPriority::NORMAL)) == "NORMAL");
}

TEST_CASE("task_priority_name: LOW", "[dispatcher][priority]") {
    CHECK(std::string(task_priority_name(TaskPriority::LOW)) == "LOW");
}

// ===========================================================================
// Task ordering
// ===========================================================================

TEST_CASE("Task: CRITICAL < LOW in priority (CRITICAL runs first)", "[dispatcher][ordering]") {
    nikola::infrastructure::Task t_crit, t_low;
    t_crit.priority = TaskPriority::CRITICAL;  t_crit.seq_id = 0;
    t_low.priority  = TaskPriority::LOW;        t_low.seq_id  = 1;
    // In std::priority_queue<T, vector<T>, greater<T>>, the element with the
    // smallest value comes first.  greater means a > b, i.e. lower int = top.
    // t_crit.operator>(t_low) should be false (CRITICAL is "less" urgent in
    // terms of queue position — it rises to the top).
    CHECK_FALSE(t_crit > t_low);  // CRITICAL is NOT greater → sits at top
    CHECK(t_low > t_crit);        // LOW IS greater → pushed down
}

TEST_CASE("Task: same priority FIFO — lower seq_id first", "[dispatcher][ordering]") {
    nikola::infrastructure::Task t1, t2;
    t1.priority = TaskPriority::NORMAL;  t1.seq_id = 5;
    t2.priority = TaskPriority::NORMAL;  t2.seq_id = 10;
    CHECK_FALSE(t1 > t2);   // t1 inserted earlier → not greater
    CHECK(t2 > t1);         // t2 inserted later → greater (pushed down)
}

// ===========================================================================
// TaskDispatcher: basic
// ===========================================================================

TEST_CASE("TaskDispatcher: default state is empty", "[dispatcher][basic]") {
    TaskDispatcher td;
    CHECK(td.empty());
    CHECK(td.size() == 0u);
}

TEST_CASE("TaskDispatcher: process_one on empty queue → false", "[dispatcher][basic]") {
    TaskDispatcher td;
    CHECK_FALSE(td.process_one());
}

TEST_CASE("TaskDispatcher: enqueue task is called by process_one", "[dispatcher][basic]") {
    TaskDispatcher td;
    bool called = false;
    td.enqueue(TaskPriority::NORMAL, "t", [&]{ called = true; });
    REQUIRE(td.process_one());
    CHECK(called);
    CHECK(td.empty());
}

TEST_CASE("TaskDispatcher: null fn throws", "[dispatcher][basic]") {
    TaskDispatcher td;
    REQUIRE_THROWS_AS(
        td.enqueue(TaskPriority::NORMAL, "null", std::function<void()>{}),
        std::invalid_argument);
}

// ===========================================================================
// Priority ordering
// ===========================================================================

TEST_CASE("TaskDispatcher: CRITICAL executes before LOW regardless of insertion order",
          "[dispatcher][priority]") {
    TaskDispatcher td;
    std::vector<int> order;

    // Insert LOW first, then CRITICAL
    td.enqueue(TaskPriority::LOW,      "low",      [&]{ order.push_back(2); });
    td.enqueue(TaskPriority::CRITICAL, "critical", [&]{ order.push_back(1); });
    td.enqueue(TaskPriority::NORMAL,   "normal",   [&]{ order.push_back(3); });

    td.process_all();

    REQUIRE(order.size() == 3u);
    CHECK(order[0] == 1);   // CRITICAL first
    // NORMAL before LOW
    CHECK(order[1] == 3);
    CHECK(order[2] == 2);
}

TEST_CASE("TaskDispatcher: HIGH before NORMAL", "[dispatcher][priority]") {
    TaskDispatcher td;
    std::vector<std::string> order;

    td.enqueue(TaskPriority::NORMAL, "n", [&]{ order.push_back("normal"); });
    td.enqueue(TaskPriority::HIGH,   "h", [&]{ order.push_back("high"); });

    td.process_all();

    REQUIRE(order.size() == 2u);
    CHECK(order[0] == "high");
    CHECK(order[1] == "normal");
}

TEST_CASE("TaskDispatcher: FIFO order within same priority", "[dispatcher][priority]") {
    TaskDispatcher td;
    std::vector<int> order;

    // Same priority: should dispatch in insertion order
    td.enqueue(TaskPriority::NORMAL, "first",  [&]{ order.push_back(1); });
    td.enqueue(TaskPriority::NORMAL, "second", [&]{ order.push_back(2); });
    td.enqueue(TaskPriority::NORMAL, "third",  [&]{ order.push_back(3); });

    td.process_all();

    REQUIRE(order.size() == 3u);
    CHECK(order[0] == 1);
    CHECK(order[1] == 2);
    CHECK(order[2] == 3);
}

// ===========================================================================
// process_all / process_up_to
// ===========================================================================

TEST_CASE("TaskDispatcher: process_all returns dispatched count", "[dispatcher][process]") {
    TaskDispatcher td;
    for (int i = 0; i < 5; ++i)
        td.enqueue(TaskPriority::NORMAL, "t", [](){});
    CHECK(td.process_all() == 5u);
    CHECK(td.empty());
}

TEST_CASE("TaskDispatcher: process_all(max) limits dispatched tasks", "[dispatcher][process]") {
    TaskDispatcher td;
    for (int i = 0; i < 10; ++i)
        td.enqueue(TaskPriority::NORMAL, "t", [](){});
    CHECK(td.process_all(3) == 3u);
    CHECK(td.size() == 7u);
}

TEST_CASE("TaskDispatcher: process_up_to(HIGH) dispatches CRITICAL+HIGH, skips NORMAL+LOW",
          "[dispatcher][process]") {
    TaskDispatcher td;
    std::vector<std::string> ran;

    td.enqueue(TaskPriority::LOW,      "low",  [&]{ ran.push_back("low"); });
    td.enqueue(TaskPriority::NORMAL,   "norm", [&]{ ran.push_back("norm"); });
    td.enqueue(TaskPriority::HIGH,     "high", [&]{ ran.push_back("high"); });
    td.enqueue(TaskPriority::CRITICAL, "crit", [&]{ ran.push_back("crit"); });

    const std::size_t n = td.process_up_to(TaskPriority::HIGH);

    // Should have dispatched CRITICAL + HIGH (2 tasks)
    CHECK(n == 2u);
    CHECK(td.size() == 2u);   // NORMAL + LOW still pending

    // Check which tasks ran
    bool ran_crit = (std::find(ran.begin(), ran.end(), "crit") != ran.end());
    bool ran_high = (std::find(ran.begin(), ran.end(), "high") != ran.end());
    bool ran_norm = (std::find(ran.begin(), ran.end(), "norm") != ran.end());
    bool ran_low  = (std::find(ran.begin(), ran.end(), "low")  != ran.end());

    CHECK(ran_crit);
    CHECK(ran_high);
    CHECK_FALSE(ran_norm);
    CHECK_FALSE(ran_low);
}

// ===========================================================================
// Observation methods
// ===========================================================================

TEST_CASE("TaskDispatcher: size() / empty() track queue depth", "[dispatcher][observe]") {
    TaskDispatcher td;
    CHECK(td.empty());
    td.enqueue(TaskPriority::NORMAL, "a", [](){});
    CHECK(td.size() == 1u);
    CHECK_FALSE(td.empty());
    td.enqueue(TaskPriority::NORMAL, "b", [](){});
    CHECK(td.size() == 2u);
    td.process_one();
    CHECK(td.size() == 1u);
}

TEST_CASE("TaskDispatcher: peek_priority returns front task priority", "[dispatcher][observe]") {
    TaskDispatcher td;
    CHECK_FALSE(td.peek_priority().has_value());

    td.enqueue(TaskPriority::LOW,      "low", [](){});
    td.enqueue(TaskPriority::CRITICAL, "crt", [](){});

    const auto p = td.peek_priority();
    REQUIRE(p.has_value());
    CHECK(*p == TaskPriority::CRITICAL);   // CRITICAL should be at top
}

TEST_CASE("TaskDispatcher: peek_name returns front task name", "[dispatcher][observe]") {
    TaskDispatcher td;
    CHECK_FALSE(td.peek_name().has_value());

    td.enqueue(TaskPriority::CRITICAL, "urgent", [](){});
    td.enqueue(TaskPriority::LOW, "later", [](){});

    const auto n = td.peek_name();
    REQUIRE(n.has_value());
    CHECK(*n == "urgent");
}

TEST_CASE("TaskDispatcher: clear() removes all pending tasks", "[dispatcher][observe]") {
    TaskDispatcher td;
    for (int i = 0; i < 5; ++i)
        td.enqueue(TaskPriority::NORMAL, "t", [](){});
    td.clear();
    CHECK(td.empty());
}

// ===========================================================================
// Statistics
// ===========================================================================

TEST_CASE("TaskDispatcher: stats track enqueued per tier", "[dispatcher][stats]") {
    TaskDispatcher td;
    td.enqueue(TaskPriority::CRITICAL, "c", [](){});
    td.enqueue(TaskPriority::HIGH,     "h", [](){});
    td.enqueue(TaskPriority::HIGH,     "h2",[](){});
    td.enqueue(TaskPriority::NORMAL,   "n", [](){});
    td.enqueue(TaskPriority::LOW,      "l", [](){});

    const auto s = td.stats();
    CHECK(s.enqueued_critical == 1u);
    CHECK(s.enqueued_high     == 2u);
    CHECK(s.enqueued_normal   == 1u);
    CHECK(s.enqueued_low      == 1u);
}

TEST_CASE("TaskDispatcher: stats.dispatched increments per task run", "[dispatcher][stats]") {
    TaskDispatcher td;
    for (int i = 0; i < 4; ++i)
        td.enqueue(TaskPriority::NORMAL, "t", [](){});
    td.process_all();
    CHECK(td.stats().dispatched == 4u);
}

TEST_CASE("TaskDispatcher: stats.errors increments when task throws", "[dispatcher][stats]") {
    TaskDispatcher td;
    td.enqueue(TaskPriority::NORMAL, "bad", []{ throw std::runtime_error("boom"); });
    td.enqueue(TaskPriority::NORMAL, "ok",  [](){});
    td.process_all();
    CHECK(td.stats().errors == 1u);
    CHECK(td.stats().dispatched == 2u);  // still dispatched both
}

// ===========================================================================
// Thread safety
// ===========================================================================

TEST_CASE("TaskDispatcher: concurrent enqueue + process_all", "[dispatcher][thread]") {
    TaskDispatcher td;
    std::atomic<int> counter{0};
    constexpr int kTasks = 200;

    // Producer thread
    std::thread producer([&]{
        for (int i = 0; i < kTasks; ++i) {
            const auto prio = static_cast<TaskPriority>(i % 4);
            td.enqueue(prio, "t", [&]{ counter.fetch_add(1); });
        }
    });

    // Consumer thread
    std::thread consumer([&]{
        int processed = 0;
        while (processed < kTasks) {
            processed += static_cast<int>(td.process_all());
            if (td.empty()) std::this_thread::yield();
        }
    });

    producer.join();
    consumer.join();

    // Drain any remaining
    td.process_all();

    CHECK(counter.load() == kTasks);
}

// ===========================================================================
// Orchestrator integration
// ===========================================================================

TEST_CASE("Orchestrator: enqueue_task + process_pending_tasks", "[dispatcher][orchestrator]") {
    OrchestratorConfig cfg;
    cfg.events_endpoint  = "inproc://test_td_orch1";
    cfg.control_endpoint = "inproc://test_td_ctl1";
    Orchestrator orch(cfg);

    bool ran = false;
    orch.enqueue_task(TaskPriority::HIGH, "test_task", [&]{ ran = true; });

    CHECK(orch.task_queue_size() == 1u);
    orch.process_pending_tasks();
    CHECK(ran);
    CHECK(orch.task_queue_size() == 0u);
}

TEST_CASE("Orchestrator: task_stats dispatched counter after process_pending_tasks",
          "[dispatcher][orchestrator]") {
    OrchestratorConfig cfg;
    cfg.events_endpoint  = "inproc://test_td_orch2";
    cfg.control_endpoint = "inproc://test_td_ctl2";
    Orchestrator orch(cfg);

    for (int i = 0; i < 3; ++i)
        orch.enqueue_task(TaskPriority::NORMAL, "t", [](){});
    orch.process_pending_tasks();
    CHECK(orch.task_stats().dispatched == 3u);
}

TEST_CASE("Orchestrator: CRITICAL dispatched before LOW by process_pending_tasks",
          "[dispatcher][orchestrator]") {
    OrchestratorConfig cfg;
    cfg.events_endpoint  = "inproc://test_td_orch3";
    cfg.control_endpoint = "inproc://test_td_ctl3";
    Orchestrator orch(cfg);

    std::vector<int> order;
    orch.enqueue_task(TaskPriority::LOW,      "l", [&]{ order.push_back(2); });
    orch.enqueue_task(TaskPriority::CRITICAL, "c", [&]{ order.push_back(1); });
    orch.process_pending_tasks();

    REQUIRE(order.size() == 2u);
    CHECK(order[0] == 1);
    CHECK(order[1] == 2);
}
