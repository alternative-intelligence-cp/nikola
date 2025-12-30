# Gemini Deep Research Task: Bootstrap Sequence Timing Specification

## Problem Statement

**Location**: Section 9.1 (IMP-03: System Bootstrap Initialization and Startup Sequencing)

**Issue Discovered**: The specification defines **what** happens during bootstrap (SPD initialization, Pilot Wave, Thermal Bath) but lacks **precise timing constraints** and ordering guarantees needed for deterministic startup.

### Specific Details

1. **Current State Machine** (Section 9.1):
   ```
   ALLOCATING → SEEDING → THERMALIZING → IGNITING → STABILIZING → READY
   ```
   Each transition has prerequisites and success criteria, but **no wall-clock time budgets**.

2. **Missing Timing Information**:
   - How long should SEEDING take? (Depends on grid size, but no formula given)
   - What's the maximum allowable stabilization time before timeout?
   - When does GPU transfer happen? (Before SEEDING or after THERMALIZING?)
   - What's the bootstrap token expiration (300 seconds) relative to physics startup?

3. **Concurrency Concerns**:
   - Multiple threads/GPUs may be initializing simultaneously
   - Network services (ZeroMQ) start in parallel with physics
   - Database connections may block during startup
   - Docker healthcheck may timeout before system READY

4. **Real-Time Constraints**:
   - Section 3.1 requires <1ms physics loop once started
   - If stabilization takes too long, real-time guarantee breaks
   - Need to budget time for all startup phases

## Research Objectives

### Primary Question
**What are the precise timing constraints for each bootstrap phase, and how do we enforce them with hard deadlines?**

### Sub-Questions to Investigate

1. **Phase Duration Formulas**:
   - **ALLOCATING**: `malloc()` time for N nodes → typically `O(N)` but depends on system
   - **SEEDING**: SPD generation + Pilot Wave injection → depends on RNG speed
   - **THERMALIZING**: Thermal noise generation → depends on `std::normal_distribution` performance
   - **IGNITING**: Emitter activation → depends on DDS (Direct Digital Synthesis) buffer fill time
   - **STABILIZING**: 100 physics steps → exactly `100 × dt` but how long is dt during warmup?
   - **READY**: Network binding → depends on OS socket creation

2. **Dependency Graph**:
   Which phases can run **in parallel**?
   ```
   Can we do this?
   ┌─ Physics Init (ALLOCATING → SEEDING → THERMALIZING)
   └─ Network Init (ZeroMQ bind, Bootstrap Token) ← Run in parallel?
   
   Or must it be sequential?
   Physics Init → Network Init → READY
   ```

3. **Failure Modes**:
   - What if SEEDING takes >30 seconds? (Timeout? Log warning? Continue anyway?)
   - What if GPU transfer fails? (Retry? Fall back to CPU? Abort?)
   - What if database connection hangs? (Async init? Block? Fail-fast?)

4. **Observability**:
   - How does an external orchestrator (Docker, Kubernetes) know when system is ready?
   - Should we expose a `/health` HTTP endpoint?
   - Should we log timing metrics for each phase? (Prometheus metrics?)

## Required Deliverables

1. **Detailed Timing Budget**:
   Provide formulas for each phase based on system parameters.
   
   Example:
   ```
   Phase           | Formula                          | Typical Value  | Max Allowable
   ----------------|----------------------------------|----------------|--------------
   ALLOCATING      | N × 16 bytes / bandwidth         | 160 MB / 50 GB/s = 3 ms
   SEEDING         | N × (RNG + SPD check)            | 10^5 × 10 μs = 1 s
   THERMALIZING    | N × normal_dist sample           | 10^5 × 5 μs = 0.5 s
   IGNITING        | 8 emitters × buffer_size / freq  | 8 × 4096 / 44.1 kHz = 0.7 s
   STABILIZING     | 100 × dt_warmup                  | 100 × 0.01 s = 1 s
   GPU_TRANSFER    | N × 256 bytes / PCIe bandwidth   | 25.6 MB / 16 GB/s = 1.6 ms
   NETWORK_BIND    | ZeroMQ socket creation (OS call) | 10-100 ms
   ```

2. **Gantt Chart / Timeline Diagram**:
   Visual representation showing:
   - Sequential dependencies (arrows)
   - Parallel execution opportunities (stacked bars)
   - Critical path (longest dependency chain)
   - Timeout thresholds (red lines)

3. **State Machine with Timing**:
   ```cpp
   enum class BootstrapPhase {
       ALLOCATING,
       SEEDING,
       THERMALIZING,
       IGNITING,
       STABILIZING,
       READY,
       TIMEOUT,
       FAILED
   };
   
   struct PhaseTimingConfig {
       std::chrono::milliseconds max_duration;
       bool can_run_parallel;
       std::vector<BootstrapPhase> dependencies;
   };
   
   const std::unordered_map<BootstrapPhase, PhaseTimingConfig> TIMING = {
       {BootstrapPhase::ALLOCATING,   {100ms, false, {}}},
       {BootstrapPhase::SEEDING,      {5000ms, false, {ALLOCATING}}},
       {BootstrapPhase::THERMALIZING, {1000ms, false, {SEEDING}}},
       {BootstrapPhase::IGNITING,     {2000ms, true, {SEEDING}}},  // Can run parallel!
       {BootstrapPhase::STABILIZING,  {3000ms, false, {THERMALIZING, IGNITING}}},
       {BootstrapPhase::READY,        {500ms, false, {STABILIZING}}},
   };
   ```

4. **Timeout and Recovery Strategy**:
   - What happens if a phase exceeds `max_duration`?
   - Retry with exponential backoff?
   - Fail-fast and log error?
   - Degrade gracefully (e.g., skip stabilization if taking too long)?

## Proposed Solutions to Evaluate

### Option 1: Sequential Blocking (Simplest)
```cpp
void bootstrap() {
    auto start = std::chrono::steady_clock::now();
    
    state = ALLOCATING;
    allocate_memory();
    check_timeout(start, 100ms, "Allocation");
    
    state = SEEDING;
    seed_universe();
    check_timeout(start, 5000ms, "Seeding");
    
    // ... etc
}
```
**Pro**: Simple, deterministic
**Con**: Slow, no parallelism

### Option 2: Async with std::future (Parallel)
```cpp
void bootstrap() {
    auto physics_future = std::async(std::launch::async, []{
        allocate_memory();
        seed_universe();
        thermalize();
    });
    
    auto network_future = std::async(std::launch::async, []{
        bind_zeromq();
        generate_bootstrap_token();
    });
    
    physics_future.wait();
    network_future.wait();
    
    stabilize();  // Requires both above complete
}
```
**Pro**: Faster, uses all cores
**Con**: Complex error handling, race conditions

### Option 3: Directed Acyclic Graph (DAG) Scheduler
```cpp
class BootstrapDAG {
    std::unordered_map<Task, std::vector<Task>> dependencies;
    std::unordered_map<Task, std::chrono::milliseconds> timeouts;
    
    void execute() {
        // Topological sort
        // Execute tasks in dependency order
        // Parallelize when possible
    }
};
```
**Pro**: Optimal parallelism, clear dependencies
**Con**: Complex implementation

## Research Questions

1. **Industry Best Practices**:
   - How does PostgreSQL initialize? (Sequential? Parallel?)
   - How does Kubernetes determine pod readiness? (Liveness/readiness probes)
   - How does CUDA handle GPU initialization? (Lazy? Eager? Async?)

2. **Real-Time Systems**:
   - In real-time operating systems (RTOS), how are boot deadlines enforced?
   - What is the standard practice for watchdog timers during startup?
   - Should we use `SCHED_FIFO` or `SCHED_DEADLINE` for bootstrap thread?

3. **Failure Recovery**:
   - Should bootstrap be idempotent? (Can we call it twice if first fails?)
   - Should we checkpoint intermediate states? (Save after SEEDING complete?)
   - What's the recovery strategy if system crashes during STABILIZING?

4. **Cloud Deployment**:
   - Docker healthcheck default timeout is 30 seconds - is that enough?
   - Kubernetes liveness probe defaults to 10 seconds - too short?
   - AWS ECS task definition allows custom healthcheck - what should ours be?

## Critical Path Analysis

Identify the **longest serial chain** of dependencies:

```
Critical Path (cannot parallelize):
ALLOCATING (3 ms) 
  → SEEDING (1000 ms) 
  → THERMALIZING (500 ms) 
  → STABILIZING (1000 ms) 
  → READY (50 ms)

Total: ~2.5 seconds minimum

Meanwhile (parallel):
IGNITING (700 ms) can overlap with THERMALIZING
NETWORK_BIND (100 ms) can overlap with SEEDING

Optimal total: ~2.5 seconds (limited by critical path)
```

## Success Criteria

- [ ] Every phase has explicit max duration
- [ ] Parallel execution paths identified
- [ ] Timeout handling code specified
- [ ] Docker/K8s healthcheck parameters defined
- [ ] Logging/metrics for observability
- [ ] Recovery strategy for each failure mode
- [ ] Total boot time <5 seconds for typical deployment

## Output Format

Please provide:
1. **Timing Budget Spreadsheet** (1 page): Formulas + typical values
2. **Gantt Chart** (1 page): Visual timeline with critical path highlighted
3. **State Machine Code** (C++23): Complete implementation with timeouts
4. **Deployment Guide** (1 page): Docker/K8s configuration
5. **Failure Mode Handbook** (2 pages): What to do when bootstrap fails

## Additional Context

This interacts with:
- Section 4.5: Security (Bootstrap Token 300-second window)
- Section 8.1: Phase 0 blockers (CF-04 requires ATP init before physics)
- Section 10.3: CLI Controller (twi-ctl must connect after READY)

The bootstrap token creates a **hard 300-second deadline** from process start to admin connection. If physics init takes >290 seconds, the admin can't pair before lockdown.

---

**Priority**: P2 - HIGH (Required for production deployment)
**Estimated Research Time**: 4-6 hours
**Dependencies**: None (but informs all other timing-critical sections)
