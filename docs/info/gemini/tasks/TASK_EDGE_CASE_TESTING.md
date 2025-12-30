# Gemini Deep Research Task: Edge Case Acceptance Testing Framework

## Problem Statement

**Location**: Multiple sections (Phase 0 validation, Energy conservation, Causal ordering)

**Issue Discovered**: The specification lacks **explicit acceptance criteria** and **test cases** for critical edge cases that could cause system failure.

### Specific Edge Cases Identified

1. **Gershgorin SPD Verification False Positive**:
   - **Scenario**: What if diagonal noise is **exactly zero** due to bad RNG seed?
   - **Impact**: Metric tensor becomes singular (non-invertible)
   - **Current Spec**: "All eigenvalues strictly positive" but no test for zero case

2. **Causal Foliation Stability**:
   - **Scenario**: What if **all nodes have the same timestamp**?
   - **Impact**: Secondary sort (Hilbert order) becomes primary, but is it stable?
   - **Current Spec**: "Temporal ordering primary" but no test for tie-breaking

3. **Metabolic Deadlock**:
   - **Scenario**: What if **all threads simultaneously** request more ATP than available?
   - **Impact**: Livelock - everyone retrying CAS forever
   - **Current Spec**: `try_reserve()` uses CAS loop but no backoff

4. **GPU Out-of-Memory During Bootstrap**:
   - **Scenario**: System allocates 10^5 nodes, GPU only has space for 10^4
   - **Impact**: cudaMalloc fails halfway through SEEDING phase
   - **Current Spec**: No GPU OOM handling mentioned

5. **Zero-Amplitude Pilot Wave**:
   - **Scenario**: What if Pilot Wave amplitude A₀ = 0 due to config error?
   - **Impact**: System enters "Linear Trap" (no nonlinearity)
   - **Current Spec**: "A₀ = 1.0 activates nonlinear term" but no validation

6. **Thermal Bath Entropy Extremes**:
   - **Scenario**: What if thermal noise σ_T is set to 10^6 (extreme)?
   - **Impact**: Velocity field dominates, system thermalizes to noise
   - **Current Spec**: Formula given but no bounds check

## Research Objectives

### Primary Question
**What is a comprehensive acceptance test suite that validates the system handles all edge cases correctly, and what are the explicit pass/fail criteria for each?**

### Sub-Questions to Investigate

1. **Test Coverage Analysis**:
   - What percentage of code paths are exercised by current tests?
   - What are the critical paths that MUST be tested?
   - What are the "happy path" vs "unhappy path" ratios?

2. **Failure Injection Testing**:
   - How do we systematically inject failures to test recovery?
   - What is the equivalent of "fuzzing" for physics simulations?
   - Can we use chaos engineering principles (Netflix Chaos Monkey)?

3. **Property-Based Testing**:
   - What invariants must hold for ALL inputs? (Energy conservation, causality, etc.)
   - Can we use QuickCheck-style property testing for C++?
   - How do we generate random but valid test cases?

4. **Continuous Validation**:
   - Should validation run on every physics timestep? (Too expensive)
   - Should validation run periodically? (Every 1000 steps?)
   - Should validation run only during bootstrap? (Phase 0)

## Required Deliverables

1. **Edge Case Catalog**:
   Comprehensive list of all identified edge cases with:
   - **Description**: What the edge case is
   - **Trigger**: How to reproduce it
   - **Expected Behavior**: What should happen
   - **Current Behavior**: What actually happens (if known)
   - **Test Code**: How to test it
   - **Priority**: P0 (crash), P1 (incorrect), P2 (performance), P3 (cosmetic)

2. **Acceptance Test Suite** (C++23/GoogleTest):
   ```cpp
   // Test 1: Gershgorin Zero Diagonal
   TEST(Bootstrap, GershgorinZeroDiagonal) {
       // Force RNG to produce zero diagonal noise
       std::mt19937 rng(KNOWN_BAD_SEED);
       
       TorusGridSoA grid(1000);
       
       // Attempt seeding
       EXPECT_THROW({
           ManifoldSeeder::seed_metric_tensor(grid, rng);
       }, std::runtime_error);
       
       // Error message should be helpful
       try {
           ManifoldSeeder::seed_metric_tensor(grid, rng);
       } catch (const std::runtime_error& e) {
           EXPECT_THAT(e.what(), ::testing::HasSubstr("diagonal dominance"));
       }
   }
   
   // Test 2: Causal Foliation - All Same Timestamp
   TEST(Mamba9D, CausalFoliationTieBreaking) {
       TorusGridSoA grid;
       
       // Create 1000 nodes all at t=42
       for (size_t i = 0; i < 1000; ++i) {
           grid.coords_t[i] = 42;
           grid.coords_x[i] = i % 100;  // Different spatial coords
       }
       
       CausalFoliationScanner scanner;
       auto sequence = scanner.generate_causal_sequence(grid);
       
       // Verify sequence is deterministic (stable sort)
       auto sequence2 = scanner.generate_causal_sequence(grid);
       EXPECT_EQ(sequence, sequence2);
       
       // Verify spatial locality preserved (Hilbert order)
       // Adjacent indices in sequence should have similar spatial coords
       for (size_t i = 0; i < sequence.size() - 1; ++i) {
           auto coord1 = grid.get_coord(sequence[i]);
           auto coord2 = grid.get_coord(sequence[i+1]);
           auto distance = hilbert_distance(coord1, coord2);
           EXPECT_LT(distance, THRESHOLD);  // Define locality threshold
       }
   }
   
   // Test 3: Metabolic Deadlock
   TEST(ENGS, MetabolicDeadlock) {
       MetabolicController controller(10.0f);  // 10 ATP total
       
       // Launch 100 threads, each requesting 5 ATP
       std::vector<std::thread> threads;
       std::atomic<int> successes{0};
       std::atomic<int> failures{0};
       
       for (int i = 0; i < 100; ++i) {
           threads.emplace_back([&]() {
               try {
                   MetabolicTransaction tx(controller, 5.0f);
                   std::this_thread::sleep_for(1ms);
                   tx.commit();
                   successes++;
               } catch (const MetabolicExhaustionException&) {
                   failures++;
               }
           });
       }
       
       for (auto& t : threads) t.join();
       
       // Exactly 2 should succeed (2 × 5 = 10), 98 should fail
       EXPECT_EQ(successes.load(), 2);
       EXPECT_EQ(failures.load(), 98);
       
       // No deadlock - all threads terminated
       SUCCEED();  // If we reach here, no deadlock occurred
   }
   
   // Test 4: GPU Out-of-Memory
   TEST(Bootstrap, GPUOutOfMemory) {
       // Mock cudaMalloc to fail after N allocations
       set_cuda_allocation_limit(1024 * 1024);  // 1 MB limit
       
       TorusGridSoA grid(1000000);  // Request 100 MB
       
       // Should fail gracefully, not crash
       EXPECT_THROW({
           grid.transfer_to_gpu();
       }, CUDAOutOfMemoryException);
       
       // System should still be in valid state (CPU data intact)
       EXPECT_TRUE(grid.is_cpu_valid());
       EXPECT_FALSE(grid.is_gpu_valid());
       
       // Should be able to fall back to CPU-only mode
       EXPECT_NO_THROW({
           grid.enable_cpu_fallback();
       });
   }
   
   // Test 5: Zero-Amplitude Pilot Wave
   TEST(Bootstrap, ZeroAmplitudePilotWave) {
       Config config;
       config.pilot_wave_amplitude = 0.0;  // Misconfiguration
       
       TorusGridSoA grid;
       ManifoldSeeder seeder(config);
       
       // Should detect and reject
       EXPECT_THROW({
           seeder.inject_pilot_wave(grid);
       }, std::invalid_argument);
       
       // Error should mention nonlinearity requirement
       try {
           seeder.inject_pilot_wave(grid);
       } catch (const std::invalid_argument& e) {
           EXPECT_THAT(e.what(), ::testing::HasSubstr("nonlinear"));
           EXPECT_THAT(e.what(), ::testing::HasSubstr("A0 > 0"));
       }
   }
   
   // Test 6: Thermal Bath Extremes
   TEST(Bootstrap, ThermalBathExtremes) {
       TorusGridSoA grid(1000);
       
       // Test 1: σ_T = 0 (no thermal noise)
       thermalize_velocity_field(grid, 0.0);
       // Should succeed, but system may be too deterministic
       EXPECT_NO_THROW(validate_velocity_field(grid));
       
       // Test 2: σ_T = 10^6 (extreme noise)
       EXPECT_THROW({
           thermalize_velocity_field(grid, 1e6);
       }, std::out_of_range);  // Should reject unreasonable values
   }
   ```

3. **Property-Based Test Generators**:
   ```cpp
   // Generate random valid configurations
   class ConfigGenerator {
   public:
       static Config generate_valid() {
           Config c;
           c.pilot_wave_amplitude = random_in_range(0.1, 10.0);
           c.thermal_sigma = random_in_range(1e-6, 1e-3);
           c.grid_size = random_choice({27, 81, 243});
           return c;
       }
       
       static Config generate_invalid() {
           // Intentionally break one constraint
           Config c = generate_valid();
           switch (random_choice({0, 1, 2})) {
               case 0: c.pilot_wave_amplitude = -1.0; break;  // Negative
               case 1: c.thermal_sigma = 1e10; break;         // Extreme
               case 2: c.grid_size = 13; break;               // Not power of 3
           }
           return c;
       }
   };
   
   TEST(PropertyBased, ValidConfigsAlwaysSucceed) {
       for (int i = 0; i < 1000; ++i) {
           auto config = ConfigGenerator::generate_valid();
           EXPECT_NO_THROW({
               TorusGridSoA grid(config.grid_size);
               ManifoldSeeder::bootstrap(grid, config);
           });
       }
   }
   
   TEST(PropertyBased, InvalidConfigsAlwaysFail) {
       for (int i = 0; i < 1000; ++i) {
           auto config = ConfigGenerator::generate_invalid();
           EXPECT_THROW({
               ManifoldSeeder::validate_config(config);
           }, std::exception);  // Some exception should be thrown
       }
   }
   ```

4. **Chaos Engineering Scenarios**:
   ```cpp
   // Inject random failures during operation
   class ChaosMonkey {
   public:
       void inject_random_failure() {
           switch (random_choice({0, 1, 2, 3, 4})) {
               case 0: kill_random_thread(); break;
               case 1: corrupt_random_memory(); break;
               case 2: delay_random_syscall(100ms); break;
               case 3: fill_disk_to_capacity(); break;
               case 4: trigger_gpu_device_reset(); break;
           }
       }
   };
   
   TEST(ChaosEngineering, SystemResilientToRandomFailures) {
       TorusGridSoA grid(1000);
       ManifoldSeeder::bootstrap(grid);
       
       ChaosMonkey monkey;
       
       // Run for 1000 timesteps with random failures
       for (int t = 0; t < 1000; ++t) {
           if (random_probability(0.01)) {  // 1% chance per step
               monkey.inject_random_failure();
           }
           
           // System should either:
           // 1. Continue running (fault-tolerant), OR
           // 2. Fail-fast with clear error (fail-safe)
           // But NEVER silently corrupt data
           
           try {
               grid.propagate(0.001);
               validate_energy_conservation(grid);
           } catch (const std::exception& e) {
               // Allowed to fail, but must be explicit
               EXPECT_THAT(e.what(), ::testing::Not(::testing::IsEmpty()));
               break;  // Stop test on explicit failure
           }
       }
   }
   ```

## Research Questions

1. **Test-Driven Development in Physics**:
   - How does CERN test particle physics simulations?
   - How does NASA test spacecraft control software?
   - What is the state-of-the-art in scientific software testing?

2. **Formal Verification**:
   - Can we use TLA+ to model the bootstrap state machine?
   - Can we use CBMC (C Bounded Model Checker) to verify no deadlocks?
   - Can we use Frama-C to prove energy conservation mathematically?

3. **Continuous Integration**:
   - Should tests run on every commit? (GitHub Actions)
   - Should we use nightly regression testing?
   - What is the acceptable test execution time? (<10 minutes?)

4. **Test Data Generation**:
   - Should we have a "golden dataset" for regression testing?
   - Should we record production telemetry and replay it?
   - Can we use mutation testing to find missing test cases?

## Success Criteria

- [ ] >90% code coverage (lines)
- [ ] >80% branch coverage (all if/else paths)
- [ ] All P0/P1 edge cases have explicit tests
- [ ] Test suite runs in <5 minutes (CI requirement)
- [ ] Zero false positives (flaky tests)
- [ ] Clear error messages for all failures
- [ ] Property-based tests run 1000+ iterations
- [ ] Chaos engineering tests pass >95% of time

## Output Format

Please provide:
1. **Edge Case Catalog** (3-5 pages): Complete enumeration with priorities
2. **Test Suite Code** (C++23): GoogleTest implementation
3. **Property Generator** (C++23): Random valid/invalid config generator
4. **Chaos Engineering Framework** (C++23): Fault injection library
5. **CI Configuration** (YAML): GitHub Actions / GitLab CI setup
6. **Test Report Template** (Markdown): How to document test results

## Additional Context

This affects ALL sections, but especially:
- Section 8.1: Phase 0 validation (must pass before production)
- Section 9.1: Bootstrap sequence (critical path)
- Section 5.1: ENGS (concurrent ATP management)
- Section 3.2: Mamba-9D (causal ordering)

Industry references:
- Google's Test Certified program (Level 3 = 40-80% coverage)
- NASA's Software Safety Standard (NASA-STD-8739.8)
- DO-178C (avionics software - requires MC/DC coverage)

---

**Priority**: P1 - CRITICAL (Prevents production bugs)
**Estimated Research Time**: 8-10 hours (comprehensive test design)
**Dependencies**: All other tasks (tests must validate their fixes)
