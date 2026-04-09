/**
 * @file metabolic_lock_test.cpp
 * @brief Unit tests for CF-04 Transactional Metabolic Lock
 * 
 * Tests thermodynamic safety, race condition prevention, and automatic rollback.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <nikola/autonomy/metabolic_lock.hpp>
#include <nikola/autonomy/metabolic_controller.hpp>
#include <thread>
#include <vector>
#include <chrono>

using namespace nikola::autonomy;

TEST_CASE("MetabolicController initialization", "[metabolic][cf-04]") {
    MetabolicController controller(100.0, 10.0);
    
    REQUIRE(controller.get_current_atp() == 100.0);
    REQUIRE_FALSE(controller.needs_nap());
}

TEST_CASE("MetabolicController nap threshold detection", "[metabolic][cf-04]") {
    MetabolicController controller(100.0, 20.0);
    
    // Consume ATP to below nap threshold
    {
        MetabolicLock lock(controller, 85.0);
        lock.commit();  // Consume 85, leaving 15
    }
    
    REQUIRE(controller.needs_nap());
    REQUIRE(controller.get_current_atp() ==15.0);
}

TEST_CASE("MetabolicLock basic reservation and commit", "[metabolic][cf-04]") {
    MetabolicController controller(100.0, 10.0);
    
    SECTION("Successful reservation and commit") {
        {
            MetabolicLock lock(controller, 30.0);
            REQUIRE(controller.get_current_atp() == 70.0);  // Reserved
            lock.commit();
        }
        REQUIRE(controller.get_current_atp() == 70.0);  // Consumed
    }
    
    SECTION("Reservation without commit (rollback)") {
        {
            MetabolicLock lock(controller, 30.0);
            REQUIRE(controller.get_current_atp() == 70.0);  // Reserved
            // Destructor runs without commit
        }
        REQUIRE(controller.get_current_atp() == 100.0);  // Rolled back
    }
}

TEST_CASE("MetabolicLock exhaustion exception", "[metabolic][cf-04]") {
    MetabolicController controller(50.0, 10.0);
    
    // First lock succeeds
    MetabolicLock lock1(controller, 30.0);
    REQUIRE(controller.get_current_atp() == 20.0);
    
    // Second lock exceeds available ATP
    REQUIRE_THROWS_AS(
        MetabolicLock(controller, 30.0),
        MetabolicExhaustionException
    );
    
    // Rollback on exception
    lock1.commit();  // First lock consumed
    REQUIRE(controller.get_current_atp() == 20.0);
}

TEST_CASE("MetabolicLock thread safety (race condition prevention)", "[metabolic][cf-04]") {
    MetabolicController controller(1000.0, 10.0);
    std::atomic<int> successful_locks{0};
    std::atomic<int> failed_locks{0};
    
    // Spawn 10 threads each trying to reserve 150 ATP
    // Total demand: 1500, available: 1000
    // Expected: 6 succeed, 4 fail
    std::vector<std::thread> threads;
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([&]() {
            try {
                MetabolicLock lock(controller, 150.0);
                successful_locks++;
                lock.commit();
            } catch (const MetabolicExhaustionException&) {
                failed_locks++;
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    // Verify no over-consumption (race safety)
    REQUIRE(controller.get_current_atp() >= 0.0);
    REQUIRE(controller.get_current_atp() <= 1000.0);
    
    // Verify correct number of locks
    REQUIRE(successful_locks + failed_locks == 10);
    REQUIRE(successful_locks <= 6);  // At most 6 can succeed (6*150 = 900)
    
    // Verify final ATP state
    double expected_remaining = 1000.0 - (successful_locks * 150.0);
    REQUIRE(controller.get_current_atp() == expected_remaining);
}

TEST_CASE("MetabolicController recharge", "[metabolic][cf-04]") {
    MetabolicController controller(100.0, 10.0);
    
    // Consume ATP
    {
        MetabolicLock lock(controller, 50.0);
        lock.commit();
    }
    REQUIRE(controller.get_current_atp() == 50.0);
    
    // Recharge
    controller.recharge(30.0);
    REQUIRE(controller.get_current_atp() == 80.0);
    
    // Recharge capped at max_atp
    controller.recharge(50.0);
    REQUIRE(controller.get_current_atp() == 100.0);  // Capped
}

TEST_CASE("MetabolicLock complex scenario: nested operations", "[metabolic][cf-04]") {
    MetabolicController controller(200.0, 20.0);
    
    // Simulate complex cognitive operation with sub-tasks
    {
        MetabolicLock main_lock(controller, 100.0);  // Main task
        REQUIRE(controller.get_current_atp() == 100.0);
        
        {
            MetabolicLock subtask_lock(controller, 50.0);  // Subtask
            REQUIRE(controller.get_current_atp() == 50.0);
            subtask_lock.commit();
        }
        
        // Main task continues
        REQUIRE(controller.get_current_atp() == 50.0);
        main_lock.commit();
    }
    
    // Final state: 200 - 100 - 50 = 50
    REQUIRE(controller.get_current_atp() == 50.0);
    REQUIRE_FALSE(controller.needs_nap());  // 50.0 > 20.0 threshold
}

TEST_CASE("MetabolicLock edge cases", "[metabolic][cf-04]") {
    MetabolicController controller(100.0, 10.0);
    
    SECTION("Small ATP reservation") {
        MetabolicLock lock(controller, 0.1);
        REQUIRE_THAT(controller.get_current_atp(), Catch::Matchers::WithinAbs(99.9, 0.01));
        lock.commit();
        REQUIRE_THAT(controller.get_current_atp(), Catch::Matchers::WithinAbs(99.9, 0.01));
    }
    
    SECTION("Exact ATP match") {
        MetabolicLock lock(controller, 100.0);
        REQUIRE(controller.get_current_atp() == 0.0);
        lock.commit();
        REQUIRE(controller.get_current_atp() == 0.0);
        REQUIRE(controller.needs_nap());
    }
    
    SECTION("Multiple rollbacks") {
        {
            MetabolicLock lock1(controller, 30.0);
            // Rollback
        }
        {
            MetabolicLock lock2(controller, 40.0);
            // Rollback
        }
        REQUIRE(controller.get_current_atp() == 100.0);
    }
}

TEST_CASE("MetabolicLock CF-04 stress: 16 threads, non-negative invariant", "[metabolic][cf-04][phase0]") {
    // CF-04 acceptance criterion: ATP budget provably non-negative under 16-thread concurrent access
    MetabolicController controller(10000.0, 10.0);
    std::atomic<int> acquired_locks{0};
    std::atomic<int> failed_locks{0};
    std::atomic<bool> invariant_violated{false};
    
    // 16 threads, each running 100 reserve/commit cycles with random costs
    constexpr int NUM_THREADS = 16;
    constexpr int OPS_PER_THREAD = 100;
    
    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back([&, i]() {
            // Per-thread deterministic RNG (avoid data races on shared rand state)
            unsigned seed = static_cast<unsigned>(i * 7919 + 42);
            for (int op = 0; op < OPS_PER_THREAD; ++op) {
                // LCG: deterministic, no shared state
                seed = seed * 1103515245 + 12345;
                float cost = 1.0f + (seed % 100);  // 1.0 to 100.0
                
                try {
                    MetabolicLock lock(controller, cost);
                    acquired_locks.fetch_add(1, std::memory_order_relaxed);
                    
                    // Check invariant WHILE holding lock
                    float current = controller.get_current_atp();
                    if (current < 0.0f) {
                        invariant_violated.store(true, std::memory_order_relaxed);
                    }
                    
                    // Commit roughly half the time, rollback the other half
                    seed = seed * 1103515245 + 12345;
                    if (seed % 2 == 0) {
                        lock.commit();
                    }
                    // else: destructor rolls back
                } catch (const MetabolicExhaustionException&) {
                    failed_locks.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    // Primary CF-04 invariant: ATP NEVER went negative
    REQUIRE_FALSE(invariant_violated.load());
    
    // ATP must be non-negative at rest
    float final_atp = controller.get_current_atp();
    REQUIRE(final_atp >= 0.0f);
    REQUIRE(final_atp <= 10000.0f);
    
    // Sanity: all operations accounted for
    REQUIRE(acquired_locks + failed_locks == NUM_THREADS * OPS_PER_THREAD);
}

TEST_CASE("MetabolicLock CF-04 contention benchmark", "[metabolic][cf-04][phase0][!benchmark]") {
    // Benchmark: lock contention target <1μs per acquire
    MetabolicController controller(1000000.0, 10.0);
    constexpr int ITERATIONS = 10000;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < ITERATIONS; ++i) {
        MetabolicLock lock(controller, 1.0);
        lock.commit();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double ns_per_op = static_cast<double>(ns) / ITERATIONS;
    
    // Target: <1μs (1000ns) per acquire
    REQUIRE(ns_per_op < 1000.0);
}
