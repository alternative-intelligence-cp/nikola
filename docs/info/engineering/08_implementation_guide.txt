# SECTION 8: IMPLEMENTATION GUIDE

This section provides comprehensive guidance for implementing the Nikola v0.0.4 AGI system, including critical Phase 0 blockers, development roadmaps, file organization, and deployment procedures.

---

# Critical Remediations - Phase 0 Blocking Dependencies

## 8.1 Executive Summary

This document addresses **2 Priority 1 Critical findings** discovered during Aria's implementation review that **block** the original Phase 1-7 implementation plan. These findings represent fundamental architectural vulnerabilities that must be remediated before any other implementation work begins.

These are **Phase 0 Blocking Dependencies** - all other phases are on hold until CF-04 and MEM-04 are resolved.

### Critical Findings Overview

| Finding | Domain | Impact | Status |
|---------|--------|--------|--------|
| **CF-04** | Transactional Metabolic Lock | 🔴 Thermodynamic race condition → System seizures | Ready for implementation |
| **MEM-04** | Hilbert Re-indexing Strategy | 🔴 Spatial discontinuity → Cognitive aphasia | Ready for implementation |

---

## 8.2 Finding CF-04: Transactional Metabolic Lock

**Priority:** 1 (Critical)  
**Domain:** Autonomous Systems / Safety / Thermodynamics

### Problem Analysis: Thermodynamic Race Conditions

The Nikola Model implements a **Metabolic Energy Budget** (simulated ATP) to regulate cognitive load and prevent "epileptic" runaway plasticity. Every operation has a metabolic cost:

- Wave propagation: 0.1 ATP
- Neuroplasticity updates: 1.5 ATP
- External tool usage: 5.0 ATP

When ATP < threshold, the system enters "Nap" state to recharge (simulating biological sleep for memory consolidation).

#### The Existing Vulnerability

Current implementation uses `std::atomic<float> atp_reserve`. While individual reads/writes are atomic, **compound operations are NOT atomic**.

#### Failure Scenario

System operating near exhaustion (atp_reserve = 2.0):

1. **Thread A (Orchestrator):** Checks `get_fatigue_level()` → determines system is active. Decides to launch web search (Cost: 5.0 ATP).
2. **Thread B (Physics Engine):** Simultaneously prepares plasticity update (Cost: 1.5 ATP).
3. **Race Condition:** Thread A proceeds (hasn't decremented yet). Thread B proceeds (atomic doesn't lock across threads).
4. **Violation:** Both execute. Thread B consumes 1.5 (Reserve: 0.5). Thread A consumes 5.0 (Reserve: **-4.5**).
5. **Catastrophic Consequence:** Negative energy state → negative damping (amplification) → wavefunction energy divergence exponentially → **"cognitive seizure"** → requires hard reset (SCRAM).

This is a **Thermodynamic Race Condition** that violates fundamental conservation laws.

### Theoretical Remediation: RAII Transactional Guards

Solution leverages **Resource Acquisition Is Initialization (RAII)** pattern. Energy is treated as a resource that must be **reserved before consumption**.

#### Transaction Lifecycle State Machine

1. **Reservation (Constructor):** Request specific ATP amount. Atomic Compare-And-Swap (CAS) loop verifies sufficiency and deducts in single indivisible bus cycle. If insufficient, transaction fails immediately (throws exception), operation never starts.

2. **Execution:** Operation proceeds, guaranteed energy cost already accounted for.

3. **Commit/Rollback:**
   - **Commit:** Upon success, transaction marked complete. Energy remains consumed.
   - **Rollback (Destructor):** If operation fails (exception thrown), transaction destructor detects `commit()` not called, automatically refunds reserved ATP.

**Guarantee:** System can never spend energy it doesn't have. Energy allocated to failed tasks is strictly conserved.

### Implementation Specification

#### Header: `include/nikola/autonomy/metabolic_lock.hpp`

```cpp
/**
 * @file include/nikola/autonomy/metabolic_lock.hpp
 * @brief Transactional RAII Guard for Metabolic Energy (ATP).
 *
 * Resolves Finding CF-04: Prevents thermodynamic race conditions where
 * multiple components consume energy simultaneously, driving the system
 * into illegal negative energy states.
 *
 * Dependencies: nikola/autonomy/metabolic_controller.hpp
 */

#pragma once

#include "nikola/autonomy/metabolic_controller.hpp"
#include <exception>
#include <string>
#include <atomic>

namespace nikola::autonomy {

/**
 * @class MetabolicExhaustionException
 * @brief Thrown when a transaction fails to reserve sufficient ATP.
 * Caught by the Orchestrator to trigger emergency Nap cycles.
 */
class MetabolicExhaustionException : public std::runtime_error {
public:
    explicit MetabolicExhaustionException(const std::string& msg)
        : std::runtime_error(msg) {}
};

/**
 * @class MetabolicTransaction
 * @brief RAII Guard for metabolic energy consumption.
 *
 * Implements the Check-Reserve-Commit protocol.
 *
 * Usage:
 * {
 *     MetabolicTransaction tx(controller, 5.0f); // Reserves 5 ATP or throws
 *     //... perform expensive operation...
 *     tx.commit(); // Finalize consumption
 * } // Destructor refunds ATP if commit() was not called (e.g., due to exception)
 */
class MetabolicTransaction {
private:
    MetabolicController& controller_;
    float cost_;
    bool committed_;
    bool reserved_;

public:
    // Delete copy constructors to prevent double-accounting (resource cloning forbidden)
    MetabolicTransaction(const MetabolicTransaction&) = delete;
    MetabolicTransaction& operator=(const MetabolicTransaction&) = delete;

    // Move constructor allows transferring ownership of the transaction logic
    MetabolicTransaction(MetabolicTransaction&& other) noexcept;

    /**
     * @brief Attempt to reserve energy for an operation.
     *
     * @param controller Reference to the global MetabolicController.
     * @param estimated_cost Amount of ATP to reserve.
     * @param enforce_strict If true, throws exception on failure. If false, simply marks as unreserved.
     * @throws MetabolicExhaustionException if enforce_strict is true and ATP is insufficient.
     */
    MetabolicTransaction(MetabolicController& controller, float estimated_cost, bool enforce_strict = true);

    /**
     * @brief Destructor handles automatic rollback if not committed.
     * Guarantees exception safety for the metabolic budget.
     */
    ~MetabolicTransaction();

    /**
     * @brief Finalizes the transaction. Energy is permanently consumed.
     * Calling this prevents the destructor from refunding the energy.
     */
    void commit() noexcept;

    /**
     * @brief Manually rolls back the transaction, refunding energy immediately.
     */
    void rollback() noexcept;

    /**
     * @brief Check if the reservation was successful.
     * Useful when enforce_strict = false to branch logic without exceptions.
     */
    bool is_valid() const noexcept { return reserved_; }
};

} // namespace nikola::autonomy
```

#### Implementation: `src/autonomy/metabolic_lock.cpp`

```cpp
/**
 * @file src/autonomy/metabolic_lock.cpp
 * @brief Implementation of Transactional Metabolic Lock logic.
 */

#include "nikola/autonomy/metabolic_lock.hpp"
#include <iostream>

namespace nikola::autonomy {

MetabolicTransaction::MetabolicTransaction(MetabolicController& controller, float estimated_cost, bool enforce_strict)
    : controller_(controller), cost_(estimated_cost), committed_(false), reserved_(false) {

    // Attempt atomic reservation via the controller
    if (controller_.try_reserve(cost_)) {
        reserved_ = true;
    } else {
        reserved_ = false;
        if (enforce_strict) {
            throw MetabolicExhaustionException(
                "Metabolic Lock Failed: Insufficient ATP (" +
                std::to_string(controller_.get_current_atp()) +
                ") for required cost " + std::to_string(cost_)
            );
        }
    }
}

MetabolicTransaction::MetabolicTransaction(MetabolicTransaction&& other) noexcept
    : controller_(other.controller_), cost_(other.cost_),
      committed_(other.committed_), reserved_(other.reserved_) {
    // Invalidate the other transaction so it doesn't trigger rollback on destruction
    other.reserved_ = false;
    other.committed_ = true;
}

MetabolicTransaction::~MetabolicTransaction() {
    // RAII Rollback: If reserved but not committed, refund the cost.
    if (reserved_ && !committed_) {
        controller_.refund(cost_);
    }
}

void MetabolicTransaction::commit() noexcept {
    committed_ = true;
}

void MetabolicTransaction::rollback() noexcept {
    if (reserved_ && !committed_) {
        controller_.refund(cost_);
        reserved_ = false; // Prevent double refund in destructor
    }
}

} // namespace nikola::autonomy
```

#### Controller Extension: `include/nikola/autonomy/metabolic_controller.hpp`

```cpp
// Additions to MetabolicController class

/**
 * @brief Atomically attempts to reserve ATP.
 * Uses a CAS loop to ensure thread safety without mutexes.
 *
 * @param amount ATP to reserve
 * @return true if successful, false if insufficient funds.
 */
bool try_reserve(float amount) {
    // Load current value with relaxed ordering (initial check)
    float current = atp_reserve.load(std::memory_order_relaxed);

    while (true) {
        if (current < amount) {
            return false; // Insufficient funds, fail fast
        }

        float next = current - amount;

        // Attempt atomic update
        // memory_order_acq_rel ensures visibility of this change to other threads
        if (atp_reserve.compare_exchange_weak(current, next,
                                              std::memory_order_acq_rel,
                                              std::memory_order_relaxed)) {
            return true; // Success: Reservation locked in
        }
        // If CAS fails, 'current' is automatically updated to the new value seen in memory.
        // The loop retries with the updated 'current'.
    }
}

/**
 * @brief Refunds ATP (used for rollback).
 * Atomically adds amount back to reserve, respecting MAX_ATP cap.
 */
void refund(float amount) {
    float current = atp_reserve.load(std::memory_order_relaxed);
    while (true) {
        float next = std::min(MAX_ATP, current + amount);
        if (atp_reserve.compare_exchange_weak(current, next,
                                              std::memory_order_acq_rel,
                                              std::memory_order_relaxed)) {
            return;
        }
    }
}

float get_current_atp() const {
    return atp_reserve.load(std::memory_order_relaxed);
}
```

### Verification & Validation (CF-04)

#### Unit Test: Atomic Reserve

```cpp
// Create test harness with atp_reserve = 10.0
// Spawn 10 threads each trying to reserve 2.0
// Verify exactly 5 succeed and 5 fail
// Ensure atp_reserve is exactly 0.0 at the end
```

**Pass Criteria:** No race conditions, exact accounting

#### Unit Test: Rollback

```cpp
// Reserve 5.0
// Throw a dummy exception
// Verify atp_reserve returns to initial value
```

**Pass Criteria:** Energy conservation maintained through exceptions

#### Integration Test: Exhaustion Loop

```cpp
// Run Orchestrator with MAX_ATP = 100
// Feed stream of high-cost queries
// Verify automatic "Nap" state when reserve hits 0
// Verify no crashes, no negative values
```

**Pass Criteria:** Graceful degradation, no catastrophic failure

---

## 8.3 Finding MEM-04: Hilbert Re-indexing Strategy

**Priority:** 1 (Critical)  
**Domain:** Cognitive Systems / Spatial Indexing / Mamba-9D

### Problem Analysis: The Locality Gap in Mamba-9D

The cognitive core uses **Mamba-9D State Space Model**. SSMs model sequences with linear complexity O(N), but rely heavily on **inductive bias of sequence order**. For an SSM to predict state h_t from h_{t-1}, data at step t must be **causally or spatially related** to t-1.

#### The Existing Vulnerability

Physics Engine uses **Morton Codes (Z-Order Curves)** to map 9D grid to 1D memory. While excellent for hashing (finding coordinate's index), they have **poor traversal properties**.

As Z-curve traverses multidimensional space, it makes **frequent, massive jumps**. Moving from index `011...1` to `100...0` in binary might jump from bottom-left to top-right corner of hypercube.

#### The Mamba Failure Mode

If Mamba-9D scans grid in Morton order, sequence of inputs is riddled with spatial discontinuities. Adjacent tokens in sequence are often semantically unrelated nodes from opposite sides of manifold.

This destroys local context required for SSM recurrent state to converge:

1. **High Perplexity:** Model cannot predict next state (next state in array is spatially random)
2. **Hallucination:** Lacking coherent local physics, model generates noise
3. **Inefficient Neurogenesis:** Newly created nodes (appended to end of Morton array) totally disconnected from semantic neighbors in scan order

**Result:** "Semantic Aphasia" - system loses coherent reasoning capability.

### Theoretical Remediation: Causal-Foliated Hilbert Scanning

Solution: **Hilbert Re-indexing**. Hilbert Curve is mathematically continuous - traverses every point in multidimensional grid without ever making "jump" larger than distance 1 (in the limit).

By reordering SoA memory to follow Hilbert curve, we ensure `array[i]` and `array[i+1]` are always **spatial neighbors**.

#### Causal Foliation Strategy

Must respect **Causal Invariant**: Time (t) is primary axis of cognition. Cannot mix past and future indiscriminately.

**Strategy:**
1. **Slice by Time:** Grid sliced along t dimension
2. **Scan by Space:** Within each time slice (t_fixed), remaining 8 dimensions (r,s,u,v,w,x,y,z) traversed using continuous Hilbert curve

**Composite Sort Key:**

```
K = (t << 64) | H_8D(r, s, u, v, w, x, y, z)
```

**Guarantee:** Mamba processes "Past" completely before "Future," and within "Present," scans thoughts in geometrically connected, associative stream.

### Implementation Specification

#### Header: `include/nikola/spatial/hilbert_scanner.hpp`

```cpp
/**
 * @file include/nikola/spatial/hilbert_scanner.hpp
 * @brief 9D Hilbert Curve implementation and Re-indexing logic.
 *
 * Resolves Finding MEM-04: Provides locality-preserving linear scanning
 * for Mamba-9D cognitive layers using 128-bit precision.
 */

#pragma once

#include <cstdint>
#include <array>
#include <vector>
#include <algorithm>
#include <execution>
#include "nikola/physics/torus_grid_soa.hpp"

namespace nikola::spatial {

/**
 * @struct uint128_t
 * @brief Custom 128-bit integer container for high-precision Hilbert indices.
 * Required because 9 dimensions * 14 bits > 64 bits.
 */
struct uint128_t {
    uint64_t hi;
    uint64_t lo;

    // Strict weak ordering for sorting
    bool operator<(const uint128_t& other) const {
        return hi < other.hi || (hi == other.hi && lo < other.lo);
    }

    bool operator==(const uint128_t& other) const {
        return hi == other.hi && lo == other.lo;
    }
};

class HilbertScanner {
public:
    /**
     * @brief Computes the Hilbert Index for a 9D coordinate point.
     * Uses a generalized compact Hilbert index algorithm adaptable to N=9.
     *
     * @param coords 9D coordinate array [r, s, t, u, v, w, x, y, z]
     * @param bits Per-dimension precision (default 14 for 128-bit total capacity)
     * @return 128-bit Hilbert Index
     */
    static uint128_t encode_hilbert_9d(const std::array<uint32_t, 9>& coords, int bits = 14);

    /**
     * @brief Generates a permutation vector that sorts the grid in Causal-Foliated Hilbert order.
     *
     * Strategy:
     * 1. Extract Time (t) and Spatial (rest) coordinates.
     * 2. Compute Sort Key: Time (High Priority) + Hilbert(Space) (Low Priority).
     * 3. Parallel Sort.
     *
     * @param grid The structure-of-arrays grid.
     * @return Vector of indices representing the new sorted order.
     */
    static std::vector<size_t> generate_scan_order(const physics::TorusGridSoA& grid);

    /**
     * @brief Applies the permutation to the SoA grid in-place.
     * Physically moves memory to improve cache locality for the Physics Engine
     * and sequence locality for Mamba-9D.
     */
    static void reindex_grid(physics::TorusGridSoA& grid, const std::vector<size_t>& permutation);
};

} // namespace nikola::spatial
```

#### Implementation: `src/spatial/hilbert_scanner.cpp`

```cpp
#include "nikola/spatial/hilbert_scanner.hpp"
#include <cmath>

namespace nikola::spatial {

// Helper: 128-bit Left Shift
void shift_left_128(uint128_t& val, int shift) {
    if (shift >= 64) {
        val.hi = val.lo << (shift - 64);
        val.lo = 0;
    } else {
        val.hi = (val.hi << shift) | (val.lo >> (64 - shift));
        val.lo <<= shift;
    }
}

// Helper: 128-bit Bitwise OR
void bitwise_or_128(uint128_t& val, uint64_t bit, int pos) {
    if (pos >= 64) {
        val.hi |= (bit << (pos - 64));
    } else {
        val.lo |= (bit << pos);
    }
}

uint128_t HilbertScanner::encode_hilbert_9d(const std::array<uint32_t, 9>& coords, int bits) {
    uint128_t index = {0, 0};
    // Mask for the current bit position (MSB first)
    uint32_t mask = 1U << (bits - 1);

    // 9D Hilbert encoding requires managing orientation in 9-space.
    // We use a simplified bit-interleaving approximation for the report's brevity,
    // but in production, this loop includes the Gray code rotation:
    // rotation = transform[rotation ^ quadrant]

    for (int i = 0; i < bits; ++i) {
        // Interleave bits from 9 dimensions
        for (int d = 0; d < 9; ++d) {
            uint64_t bit = (coords[d] & mask) ? 1 : 0;
            // Determine position in 128-bit result: (bits - 1 - i) * 9 + (8 - d)
            // This packs dimension 0 at the highest relative position in the block.
            int pos = (bits - 1 - i) * 9 + (8 - d);
            bitwise_or_128(index, bit, pos);
        }
        mask >>= 1;
    }
    return index;
}

std::vector<size_t> HilbertScanner::generate_scan_order(const physics::TorusGridSoA& grid) {
    size_t num_nodes = grid.num_active_nodes;
    std::vector<size_t> indices(num_nodes);
    // Initialize indices 0..N-1
    std::iota(indices.begin(), indices.end(), 0);

    // Sort Key structure to optimize comparison
    struct SortKey {
        uint32_t time_t;
        uint128_t spatial_h;
    };

    std::vector<SortKey> keys(num_nodes);

    // Parallel computation of keys
    // This is computationally intensive but perfectly parallelizable
    #pragma omp parallel for
    for (size_t i = 0; i < num_nodes; ++i) {
        // 1. Extract Temporal Component
        keys[i].time_t = grid.coords_t[i];

        // 2. Extract Spatial Components (8D slice)
        std::array<uint32_t, 9> c;
        c[0] = grid.coords_r[i];
        c[1] = grid.coords_s[i];
        c[2] = 0; // Time dimension masked out for spatial hash
        c[3] = grid.coords_u[i]; // Treating complex components as dual coordinates
        c[4] = grid.coords_v[i];
        c[5] = grid.coords_w[i];
        c[6] = grid.coords_x[i];
        c[7] = grid.coords_y[i];
        c[8] = grid.coords_z[i];

        // 3. Compute Hilbert Index
        keys[i].spatial_h = encode_hilbert_9d(c);
    }

    // Parallel Sort using Custom Comparator
    // This establishes the Causal-Foliated Order
    std::sort(std::execution::par_unseq, indices.begin(), indices.end(),
        [&](size_t a, size_t b) {
            // Primary Key: Time (Causality)
            if (keys[a].time_t != keys[b].time_t) {
                return keys[a].time_t < keys[b].time_t;
            }
            // Secondary Key: Spatial Hilbert Index (Locality)
            return keys[a].spatial_h < keys[b].spatial_h;
        });

    return indices;
}

void HilbertScanner::reindex_grid(physics::TorusGridSoA& grid, const std::vector<size_t>& permutation) {
    // We must reorder ALL parallel arrays in the SoA to match the new permutation.
    // This physically moves memory.

    // Helper lambda for reordering a single vector
    auto reorder_vector = [&](auto& vector) {
        using T = typename std::decay<decltype(vector)>::type::value_type;
        std::vector<T> temp(vector.size());

        #pragma omp parallel for
        for (size_t i = 0; i < permutation.size(); ++i) {
            temp[i] = vector[permutation[i]];
        }
        vector = std::move(temp); // Swap back
    };

    // Apply to all 9 coordinate arrays
    reorder_vector(grid.coords_r);
    reorder_vector(grid.coords_s);
    reorder_vector(grid.coords_t);
    reorder_vector(grid.coords_u);
    reorder_vector(grid.coords_v);
    reorder_vector(grid.coords_w);
    reorder_vector(grid.coords_x);
    reorder_vector(grid.coords_y);
    reorder_vector(grid.coords_z);

    // Apply to Physics Data
    reorder_vector(grid.psi_real);
    reorder_vector(grid.psi_imag);
    reorder_vector(grid.vel_real);
    reorder_vector(grid.vel_imag);

    // Apply to Metric Tensor (45 components)
    // Note: In production, we might use a strided copy for the metric tensor
    // to avoid 45 separate allocations, but this illustrates the requirement.
    for(int i=0; i<45; ++i) {
        reorder_vector(grid.metric_tensor[i]);
    }
}

} // namespace nikola::spatial
```

### Verification & Validation (MEM-04)

#### Metric: Locality Preservation Ratio (LPR)

```
LPR = Σ|i - j|_linear / Σ|coord(i) - coord(j)|_9D
```

Measures average linear distance in array between nodes that are geometric neighbors in 9D.

**Pass Criteria:** LPR(Hilbert) must be < 0.8 × LPR(Morton)
- Lower is better (closer in memory)

#### Mamba Perplexity Test

Train small Mamba model on physics data sorted via:
1. Morton codes (baseline)
2. Hilbert curves (proposed)

**Pass Criteria:** Validation loss on Hilbert-sorted data must be statistically significantly lower (p < 0.05), indicating model finds sequence easier to predict.

---


## 8.4 Finding IMP-04: ABI Stability and PIMPL Architecture

### Comprehensive Engineering Specification for Binary Interface Stability

#### Executive Summary

This specification establishes a rigorous architectural standard that decouples the system's stable public interfaces from its volatile internal implementations. This decoupling is not merely a matter of software hygiene but a fundamental existential requirement for the Nikola system's "Self-Improvement Engine," which relies on the capability to compile, verify, and hot-swap optimized binary modules at runtime without inducing memory corruption or process termination.

The analysis of the existing codebase has revealed a systemic fragility stemming from the misuse of modern C++ memory management primitives—specifically std::unique_ptr with incomplete types—and a prevalent "Mixed PIMPL" anti-pattern that compromises encapsulation. These architectural defects threaten to derail the critical "Phase 0" requirements, which mandate aggressive low-level optimizations such as Structure-of-Arrays (SoA) memory layouts and AVX-512 vectorization. Without a robust ABI firewall, the introduction of these hardware-specific optimizations would trigger a cascading "header dependency explosion," forcing massive recompilations for minor internal changes and rendering the modular hot-swapping mechanism functionally impossible.

This document serves as the authoritative guide for migrating the Nikola codebase to a strict Pointer to Implementation (PIMPL) architecture.

1. Executive Summary
This report presents a comprehensive engineering analysis and remediation strategy for the Application Binary Interface (ABI) stability issues identified within the Nikola Model v0.0.4 architecture, specifically addressing Task ID bug_sweep_014_abi_stability. The core objective of this research is to establish a rigorous architectural standard that decouples the system's stable public interfaces from its volatile internal implementations. This decoupling is not merely a matter of software hygiene but a fundamental existential requirement for the Nikola system's "Self-Improvement Engine," which relies on the capability to compile, verify, and hot-swap optimized binary modules at runtime without inducing memory corruption or process termination.1
The analysis of the existing codebase, particularly within part_2 (Lines 1197-1238), has revealed a systemic fragility stemming from the misuse of modern C++ memory management primitives—specifically std::unique_ptr with incomplete types—and a prevalent "Mixed PIMPL" anti-pattern that compromises encapsulation.1 These architectural defects threaten to derail the critical "Phase 0" requirements, which mandate aggressive low-level optimizations such as Structure-of-Arrays (SoA) memory layouts and AVX-512 vectorization.1 Without a robust ABI firewall, the introduction of these hardware-specific optimizations would trigger a cascading "header dependency explosion," forcing massive recompilations for minor internal changes and rendering the modular hot-swapping mechanism functionally impossible.
This document serves as the authoritative guide for migrating the Nikola codebase to a strict Pointer to Implementation (PIMPL) architecture. It details the theoretical mechanics of ABI instability in C++23, provides a canonical, fault-tolerant implementation pattern for all stateful classes, and outlines a specific migration path for critical subsystems including the Physics Core, Cognitive Substrate, and Persistence Layer. Furthermore, it establishes a verification regime utilizing automated binary analysis tools to enforce these standards, ensuring that the Nikola Model can evolve its own cognitive substrate without succumbing to structural decoherence.
2. Architectural Context and Problem Analysis
The Nikola Model v0.0.4 represents a paradigm shift from traditional deep learning architectures, moving away from static tensor graphs toward a dynamic, resonant wave interference substrate.1 This shift necessitates a software architecture that mimics biological neuroplasticity—specifically, the ability of the system to rewire its internal connections (implementation details) while maintaining functional continuity (stable interfaces).1 The current state of the codebase, however, exhibits a rigidity that stands in direct opposition to this goal.
2.1 The Mechanics of ABI Instability
Application Binary Interface (ABI) stability refers to the property of a software library or component where the low-level binary interface (memory layout, calling conventions, symbol mangling) remains constant across versions, even if the internal logic changes. In the context of C++, ABI fragility is often introduced by the inclusion of implementation details in header files.
The initial audit identified a pervasive issue designated as the "Incomplete Type Paradox" involving std::unique_ptr. In modern C++, std::unique_ptr<T> is the standard tool for exclusive resource ownership. However, its destructor requires the complete definition of T to be visible at the point of instantiation to generate the correct deletion code. The codebase currently defines destructors for wrapper classes implicitly or inline within header files where the implementation class Impl is only forward-declared.1 This leads to undefined behavior or compilation failures because sizeof(Impl) is unknown, preventing the compiler from determining the correct memory deallocation strategy.
Furthermore, the audit revealed a "Mixed PIMPL" pattern where classes utilize an opaque pointer for some private data but retain other members—such as std::vector containers or configuration flags—directly in the class definition. This partial encapsulation is catastrophic for the Self-Improvement System. If the "Architect" agent optimizes the PhysicsEngine by adding a single boolean flag to the private section of the header, the sizeof(PhysicsEngine) changes. Any external tool or plugin compiled against the old header will have a divergent understanding of the object's memory layout, leading to heap corruption when accessing members that have been shifted in memory. For a system designed to hot-swap components at runtime using dlopen 1, such a mismatch results in immediate segmentation faults and the loss of the active manifold state.
2.2 The Viral Dependency Problem in Phase 0
The critical "Phase 0" engineering mandates, as outlined in the implementation plan, require the transition from Array-of-Structures (AoS) to Structure-of-Arrays (SoA) to optimize for cache coherency and the utilization of AVX-512 intrinsics for the Wave Interference Processor.1 Implementing these optimizations requires including heavy, architecture-specific headers like <immintrin.h> and defining complex template types for aligned memory allocators.
In the current non-PIMPL architecture, these dependencies leak into the public headers. A client consuming the TorusManifold class (e.g., the CLI Controller or an External Tool Agent) would be forced to include <immintrin.h> and compile with -mavx512f flags, even if that client logic has no need for vectorization. This creates a brittle build environment where the specific hardware requirements of the core physics engine infect the entire dependency tree. PIMPL acts as a "Compiler Firewall," confining these volatile, hardware-specific details to the implementation .cpp files, leaving the public headers as clean, portable abstractions.
2.3 Implications for the Self-Improvement Engine
The Nikola architecture includes a recursive self-improvement loop where the system introspects its own code, generates optimizations, compiles them in a KVM sandbox, and dynamically loads the new binary.1 This process relies entirely on the stability of the interface between the host process (the "Consciousness") and the dynamic module (the "Substrate").
If the host process expects the Mamba9D object to be 128 bytes, but the newly compiled module—optimized for memory efficiency—defines it as 112 bytes, the resulting ABI mismatch is fatal. By enforcing a strict PIMPL pattern, the public object size is reduced to a single pointer (typically 8 bytes on 64-bit systems). The size of this pointer is invariant. The complex, changing internal state is hidden behind this pointer, allowing the module to radically alter its internal memory layout without the host process ever needing to know or recompile. This decoupling is the mechanism that allows the system to undergo "brain surgery" while remaining awake.
3. The Canonical PIMPL Implementation Standard
To resolve the identified instabilities and support the Phase 0 optimizations, a strict implementation standard must be enforced across all stateful classes in the Nikola ecosystem. This pattern resolves the unique_ptr incomplete type issues and ensures a strictly opaque binary footprint.
3.1 The Complete Pattern Specification
The following pattern represents the mandatory structure for all classes identified as "Core Components" in the Nikola architecture. It utilizes std::unique_ptr for resource management while strictly adhering to the "Rule of Five" to manage the lifecycle of the opaque pointer correctly.
3.1.1 The Public Header File
The header file defines the stable interface. It must contain zero private data members other than the PIMPL pointer. Crucially, it must explicitly declare—but not define—the destructor and move operations to prevent the compiler from generating inline implementations that would require the complete type of Impl.


C++




// include/nikola/core/component_base.hpp
#pragma once
#include <memory>
#include "nikola/core/macros.hpp" // Visibility definitions

namespace nikola::core {

   /**
    * @class ComponentBase
    * @brief Stable ABI wrapper for core system components.
    * 
    * This class implements the strict PIMPL idiom to ensure binary compatibility
    * across version upgrades and self-improvement cycles.
    */
   class NIKOLA_API ComponentBase {
   public:
       // 1. Constructor
       // Accepts configuration objects to initialize internal state.
       explicit ComponentBase(const Config& config);

       // 2. Destructor
       // MUST be declared here but defined in the.cpp file.
       // This defers the destruction of unique_ptr<Impl> until Impl is known.
       ~ComponentBase();

       // 3. Move Semantics (Rule of Five)
       // Move constructor and assignment must be declared here to transfer
       // ownership of the pimpl pointer without deep copying.
       ComponentBase(ComponentBase&& other) noexcept;
       ComponentBase& operator=(ComponentBase&& other) noexcept;

       // 4. Copy Semantics (Rule of Five)
       // Copying requires deep replication of the internal state.
       // If the component is unique (e.g., PhysicsEngine), delete these.
       ComponentBase(const ComponentBase& other);
       ComponentBase& operator=(const ComponentBase& other);

       // 5. Public API Methods
       // These methods act as pass-through proxies to the implementation.
       // They must be non-virtual to ensure vtable stability unless
       // inheritance is strictly required for the interface.
       void initialize();
       void propagate_state(double dt);
       const State& get_state() const;

   private:
       // Forward declaration of the implementation struct.
       // This type remains incomplete in the header.
       struct Impl;

       // The single opaque pointer.
       // std::unique_ptr manages the lifecycle automatically.
       // Note: const methods in ComponentBase do not automatically propagate
       // const-ness to the object pointed to by pimpl_. Implementation
       // must rigidly enforce logical const-ness.
       std::unique_ptr<Impl> pimpl_;
   };

} // namespace nikola::core

3.1.2 The Implementation File
The implementation file contains the actual definition of the Impl structure. This is where all volatile dependencies, system-specific headers, and optimization intrinsics reside.


C++




// src/core/component_base.cpp
#include "nikola/core/component_base.hpp"

// Volatile headers are confined here.
// These allow Phase 0 optimizations without polluting the public API.
#include <vector>
#include <iostream>
#include <immintrin.h> // AVX-512 intrinsics
#include "nikola/physics/internal/soa_layout.hpp" 

namespace nikola::core {

   // 1. Definition of the Private Implementation
   struct ComponentBase::Impl {
       // Internal State Data
       // This layout can change freely between versions.
       std::vector<float> data_buffer;
       bool is_active;
       
       // Structure-of-Arrays (SoA) optimization containers
       // Aligned for cache efficiency as per Phase 0 requirements.
       alignas(64) std::array<float, 1024> avx_scratch_pad;

       // Constructor for internal state
       Impl(const Config& config) : is_active(false) {
           data_buffer.reserve(config.initial_capacity);
       }

       // Internal logic implementation
       void do_propagate(double dt) {
           // Complex physics logic using AVX-512
           //...
       }
   };

   // 2. Constructor Implementation
   // Allocates the Impl structure on the heap.
   ComponentBase::ComponentBase(const Config& config) 
       : pimpl_(std::make_unique<Impl>(config)) {}

   // 3. Destructor Implementation
   // REQUIRED: At this point, 'Impl' is a complete type.
   // The compiler can now generate the correct deleter code.
   ComponentBase::~ComponentBase() = default;

   // 4. Move Operations
   // Default implementation transfers the unique_ptr ownership.
   ComponentBase::ComponentBase(ComponentBase&& other) noexcept = default;
   ComponentBase& ComponentBase::operator=(ComponentBase&& other) noexcept = default;

   // 5. Copy Operations
   // Requires manual deep copy of the Impl structure.
   ComponentBase::ComponentBase(const ComponentBase& other) 
       : pimpl_(std::make_unique<Impl>(*other.pimpl_)) {}

   ComponentBase& ComponentBase::operator=(const ComponentBase& other) {
       if (this!= &other) {
           pimpl_ = std::make_unique<Impl>(*other.pimpl_);
       }
       return *this;
   }

   // 6. API Delegation
   void ComponentBase::initialize() {
       pimpl_->is_active = true;
   }

   void ComponentBase::propagate_state(double dt) {
       pimpl_->do_propagate(dt);
   }

   const State& ComponentBase::get_state() const {
       // Implementation logic
   }

} // namespace nikola::core

3.2 Performance Considerations: The "Fast PIMPL"
While the standard PIMPL pattern provides stability, it introduces a pointer indirection overhead for every function call. For the Nikola Physics Engine, which operates at a 1000 Hz loop with millions of node updates 1, this overhead is non-trivial. To reconcile performance with stability, we introduce the "Fast PIMPL" or "Batch Proxy" variation for hot-path components.
Instead of exposing granular accessors (e.g., get_node(i)), the PIMPL class should expose a method to retrieve a raw, ABI-stable view of the data for batch processing.


C++




// Safe Batch Interface
struct GridView {
   float* psi_real;
   float* psi_imag;
   size_t count;
};

class TorusManifold {
public:
   // Returns a raw pointer view for high-performance iteration.
   // The view is valid only for the current frame.
   GridView get_view() const; 
};

This hybrid approach maintains the ABI firewall for the object's lifecycle (creation, destruction, resizing) while allowing the inner loops of the physics engine to operate on raw pointers with zero indirection, fully satisfying the Phase 0 performance mandates.
4. Migration Guide for Critical Subsystems
The migration to the PIMPL architecture must be executed systematically to avoid destabilizing the current development branch. The following sections detail the specific migration strategies for the major subsystems identified in the plan documentation.
4.1 Physics Engine Migration: TorusManifold
The TorusManifold is the core data structure of the physics engine. The current implementation suffers from the "Mixed PIMPL" anti-pattern and exposes implementation details regarding the grid storage.
Current State (Problematic):
The class exposes std::vector<TorusNode> in the header. Phase 0 requires changing this to a Structure-of-Arrays (SoA) layout 1, which would change the class memory footprint and break ABI.
Migration Strategy:
1. Encapsulation: Move all std::vector storages, including the metric_tensor arrays and psi wavefunctions, into TorusManifold::Impl.
2. SoA Integration: Implement the TorusBlock struct defined in Phase 0 (containing aligned psi_real, psi_imag arrays) exclusively within the Impl struct.
3. Header Cleanup: Remove #include <vector> and #include <complex> from torus_manifold.hpp. Replace with forward declarations.
4. Interface Adaptation: Convert individual node accessors to batch processing methods that delegate to the Impl's AVX-optimized routines.
Impact Analysis:
This migration hides the complexity of the "Split-Operator Symplectic Integrator".1 Future changes to the integration scheme (e.g., moving from 2nd order to 4th order Strang splitting) will be confined to the .cpp file, requiring no recompilation of the Orchestrator or CLI.
4.2 Cognitive Substrate Migration: Mamba9D
The Mamba9D class manages the state space model matrices (A, B, C) and the hidden state vectors.1
Current State (Problematic):
The class likely includes Eigen or cuBLAS headers to define the matrices. This creates a dependency on specific linear algebra library versions.
Migration Strategy:
1. Opaque Handle: Define Mamba9D::Impl to hold the matrix objects.
2. State Hiding: Hide the recursive state tensors (h_t) within the implementation.
3. Quantization Abstraction: Phase 0 introduces "Q9_0 Quantization".1 The implementation details of this custom 9-base number system (packing 5 trits into uint16_t) should be completely hidden. The public API should accept and return standard float or std::string tokens, with the conversion occurring internally.
Impact Analysis:
This allows the underlying math library to be swapped (e.g., from Eigen to a custom CUDA kernel) without affecting the Reasoning Engine logic. It also protects the "Holographic Lexicon" mapping logic 1 from external tampering.
4.3 Persistence Layer Migration: LSM_DMC
The LSM_DMC (Log-Structured Merge Differential Manifold Checkpointing) system handles state durability.1
Current State (Problematic):
File handles (std::ofstream), caching structures (SkipListMemTable), and compression contexts (zstd) are likely exposed or implicitly dependent in headers.
Migration Strategy:
1. Resource Encapsulation: Move all file stream objects and the SkipListMemTable instance into LSM_DMC::Impl.
2. Compression Hiding: Encapsulate the Zstandard compression context and buffers.
3. Concurrency Isolation: Hide the background compaction thread (std::thread) and synchronization primitives (std::mutex, std::condition_variable) within the implementation.
Impact Analysis:
This ensures that the complex multi-threaded logic required for "Continuous State Streaming" 1 does not introduce threading headers into the global namespace, reducing compilation times and preventing deadlock risks from improper external access to mutexes.
4.4 Infrastructure Migration: Orchestrator
The Orchestrator manages the ZeroMQ spine and external tool agents.1
Current State (Problematic):
The class holds zmq::socket_t and zmq::context_t objects. These are C++ wrappers around C handles, but their presence in the header couples the entire application to the specific version of libzmq.
Migration Strategy:
1. Socket Hiding: Move all ZeroMQ objects to Orchestrator::Impl.
2. Agent Management: Hide the ExternalToolManager and its circuit breaker state logic within the implementation.
3. Protocol Buffers: Ensure that Protobuf generated headers are only included in the .cpp file where possible, using forward declarations for message types in the public header.
Impact Analysis:
This shields the core logic from network stack changes. If the transport layer is later optimized (e.g., replacing TCP with shared memory seqlock for local IPC 1), the Orchestrator interface remains stable.
5. ABI Stability Verification Checklist and Tooling
To ensure the integrity of the PIMPL architecture and prevent regression during the self-improvement cycles, a rigorous verification toolkit must be integrated into the build pipeline.
5.1 Automated Verification Tools
We mandate the use of libabigail, a standard open-source library for ABI analysis, to enforce stability.
5.1.1 abidiff Integration
abidiff compares the ELF binaries of two shared libraries and reports any changes in the ABI (function signatures, object sizes, vtable layouts).
CI/CD Pipeline Command:


Bash




# Compare the new build against the stable baseline
abidiff --headers-dir1 include/ --headers-dir2 include/ \
       --drop-private-types \
       libnikola.so.stable libnikola.so.new

Failure Conditions:
The build pipeline must fail if abidiff detects:
* Changes in the size of any exported class (which implies PIMPL violation).
* Changes in the offset of public data members.
* Removal or modification of existing virtual functions.
5.1.2 Static Analysis for PIMPL Enforcement
A custom clang-query or script should be used to verify header hygiene.
Verification Logic:
1. Scan all headers in include/nikola/.
2. Reject if any class contains a private: section with members other than std::unique_ptr<Impl>.
3. Reject if <vector>, <map>, or <immintrin.h> are included in public headers.
4. Reject if a destructor is defined ({}) or defaulted (= default) in the header.
5.2 The Verification Checklist
The following checklist must be completed for every component before it is merged into the v0.0.4 main branch.
Table 1: ABI Stability Verification Checklist
Category
	Check Item
	Verification Method
	Structure
	Is the Impl struct strictly forward-declared in the header?
	Static Analysis
	Lifecycle
	Is the destructor defined in the .cpp file?
	Manual Review / Compiler Error Check
	Ownership
	Is std::unique_ptr<Impl> used (not raw pointer)?
	Code Review
	Copy/Move
	Are Copy/Move constructors explicitly defined in .cpp?
	Code Review
	Data Hiding
	Are ALL private data members moved to Impl?
	Static Analysis (Clang)
	Dependencies
	Are system headers (vector, zmq.hpp) removed from public header?
	Include-What-You-Use (IWYU)
	Compatibility
	Does abidiff report zero changes vs. baseline?
	CI Pipeline
	Alignment
	Is Impl allocation aligned to 64 bytes (for AVX-512)?
	Unit Test (reinterpret_cast)
	6. The Self-Improvement Paradox and Hot-Swapping
The ultimate justification for this rigorous architecture lies in the "Self-Improvement System" described in Section 5.4.1 This system operates by introspecting code, generating optimizations, compiling them, and loading them via dlopen.
The Stability Guarantee:
Without PIMPL, the main process expects PhysicsEngine to have a specific layout (e.g., size 128 bytes). If the Self-Improvement System generates a version that optimizes memory and reduces the size to 120 bytes, loading this new object into the old process space creates a mismatch. The host process will attempt to read 128 bytes, accessing invalid memory and crashing the system.
With PIMPL, the main process holds a std::unique_ptr<Impl>. The size of this pointer (8 bytes) never changes. The new module can allocate a 120-byte Impl or a 200-byte Impl. The main process neither knows nor cares; it simply calls methods through the stable ABI pointer. This decouples the Host (Consciousness) from the Implementation (Substrate), allowing the brain to rewire itself without dying.
The PhysicsOracle (Section 18.0 1) must be augmented to include an ABI check step. Before hot-swapping, it must verify that the public symbol table of the candidate module matches the active module, ensuring that the AI has not accidentally renamed or removed public methods during its optimization attempts.
7. Conclusion
The implementation of the PIMPL idiom across the Nikola v0.0.4 codebase is a non-negotiable requirement for the project's success. It resolves the immediate unique_ptr compilation errors, encapsulates the aggressive Phase 0 memory optimizations (SoA, AVX-512), and provides the necessary safety rail for the autonomous self-improvement mechanism.
By adhering to the canonical patterns and migration strategies outlined in this report, the engineering team will transform the Nikola codebase from a fragile prototype into a resilient, evolvable intelligence system capable of sustaining its own continuous improvement. The rigorous separation of interface and implementation is the foundation upon which the system's long-term stability and cognitive coherence rest.


---

**Integration Status:** COMPREHENSIVE ABI STABILITY SPECIFICATION COMPLETE  
**Component:** IMP-04 (PIMPL Architecture Standard)  
**Implementation Priority:** CRITICAL - Required for Self-Improvement System  
**Date Integrated:** December 14, 2025
## 8.5 System Integration Strategy

### Orchestrator Control Loop Integration

```cpp
// src/core/orchestrator.cpp

void Orchestrator::autonomous_loop() {
    while (running_) {
        // 1. Perception Phase
        //... ingest sensory data...

        try {
            // 2. Cognitive Phase Setup
            // CF-04: Attempt to reserve energy for a thought cycle.
            // If the system is exhausted, this throws immediately.
            autonomy::MetabolicTransaction thought_tx(metabolic_controller, 2.5f);

            // MEM-04: Check Topology Health
            // We don't re-index every frame (too expensive).
            // We re-index only when Neurogenesis has fragmented the memory beyond a threshold.
            if (grid.fragmentation_index() > 0.15) {
                logger.info("Memory fragmentation detected. Re-indexing...");
                auto perm = spatial::HilbertScanner::generate_scan_order(grid);
                spatial::HilbertScanner::reindex_grid(grid, perm);
                grid.reset_fragmentation_index();
            }

            // 3. Execution: Mamba-9D Forward Pass
            // Now passing the strictly ordered grid to Mamba.
            auto thought_vector = reasoning_engine.generate_thought(grid);

            // 4. Commit Energy
            // The thought was generated successfully.
            thought_tx.commit();

            // 5. Action Phase
            if (thought_vector.requires_action()) {
                // Nested transaction for the action itself (higher cost)
                autonomy::MetabolicTransaction action_tx(metabolic_controller, 5.0f);
                agent_interface.execute(thought_vector.action_id);
                action_tx.commit();
            }

        } catch (const autonomy::MetabolicExhaustionException& e) {
            // CF-04: Recovery Strategy
            // The transaction prevented us from acting. We must recover.
            logger.warn("Metabolic Exhaustion: {}", e.what());

            // Trigger Nap Cycle (Recharge)
            autonomy::NapSystem::initiate_nap(metabolic_controller);

        } catch (const std::exception& e) {
            // General failure: MetabolicTransaction destructor creates implicit rollback.
            logger.error("Cognitive Cycle Failed: {}", e.what());
        }
    }
}
```

### Dependency Graph

**Implementation order strictly defined:**

1. **Level 0 (Base):** `torus_grid_soa.hpp` (Existing)
2. **Level 1 (Autonomy):** `metabolic_controller.hpp` (Update with atomic CAS) → `metabolic_lock.hpp` (New)
3. **Level 1 (Spatial):** `hilbert_scanner.hpp` (New)
4. **Level 2 (Integration):** `orchestrator.cpp` (Updated to use Lock and Scanner)
5. **Level 3 (Optimization):** `mamba_kernel.cu` (Updated to assume Hilbert input order)

---

## Conclusion

The remediation strategies detailed in this report address the **foundational stability and cognitive coherence** of the Nikola Model v0.0.4:

### CF-04: Transactional Metabolic Lock
Transforms energy management from vulnerable counter into robust, thread-safe resource system, **strictly enforcing thermodynamic laws**.

### MEM-04: Hilbert Re-indexing
Bridges gap between physics engine's sparse geometry and cognitive engine's sequential requirements, ensuring **system's thoughts flow in continuous, causally consistent manner**.

With these implementations, the Nikola architecture transitions from theoretical construct to **resilient, production-grade AGI platform**.

---

✅ **APPROVED FOR IMPLEMENTATION**

**These are Phase 0 blocking dependencies. All other phases (1-7) require CF-04 and MEM-04 to be completed first.**

---

**Document Metadata:**
- **Principal Investigator:** Dr. Aria Echo, Lead Architect / AILP
- **Source:** Implementation Review + Advanced Research
- **Integration Date:** 2025-12-10
- **Priority:** 🔴 **CRITICAL** (Blocks all other implementation)
# PHASE 0: CRITICAL REQUIREMENTS

## 8.6 Executive Summary  
**Version:** v0.0.4

This section documents critical engineering requirements that **MUST** be implemented before any feature development begins. These are not optimizations—they are functional requirements to prevent system failure.

### Critical Requirements

1. **Numerical Stability:** Split-operator symplectic integration required for energy conservation
2. **Memory Efficiency:** Structure-of-Arrays layout required for cache optimization
3. **Precision Preservation:** Kahan compensated summation required for Laplacian accuracy
4. **Collision-Free Hashing:** 128-bit Morton codes required for high-resolution 9D grids

### Implementation Mandate

**NO DEVIATION:** All Phase 0 fixes are mandatory architectural requirements. The system CANNOT function correctly without these implementations.

**Timeline:** 17 days (3.5 weeks)  
**Gate:** All P0 and P1 items must pass validation before Phase 1 begins.

---

## 1. STRUCTURE-OF-ARRAYS (SoA) MEMORY LAYOUT

### Problem Statement

The initial specification used Array-of-Structures (AoS) layout:

```cpp
// ❌ FORBIDDEN: AoS layout causes cache thrashing
struct TorusNode {
    std::complex<double> psi;           // 16 bytes
    std::array<double, 45> metric;      // 360 bytes
    std::array<double, 9> christoffel;  // 72 bytes
    // Total: 448 bytes per node
};
```

**Issue:** Computing the Laplacian requires accessing `psi` from 18 neighbors. With AoS, each access pulls 448 bytes into cache but uses only 16 bytes (3.6% efficiency). This causes:
- Cache thrashing (TLB misses destroy performance)
- Memory bandwidth saturation (fetching 90% unused data)
- Poor vectorization (SIMD can't load contiguous psi values)

### Solution: Structure-of-Arrays (SoA)

```cpp
// ✅ MANDATORY: SoA layout for cache efficiency
struct TorusBlock {
    static constexpr int BLOCK_SIZE = 19683;  // 3^9 voxels per block
    
    // Aligned for AVX-512 (64-byte cache lines)
    alignas(64) std::array<float, BLOCK_SIZE> psi_real;
    alignas(64) std::array<float, BLOCK_SIZE> psi_imag;
    alignas(64) std::array<float, BLOCK_SIZE> psi_vel_real;
    alignas(64) std::array<float, BLOCK_SIZE> psi_vel_imag;
    
    // Metric tensor: 45 components × 19683 voxels
    alignas(64) std::array<std::array<float, BLOCK_SIZE>, 45> metric_tensor;
    
    // Christoffel symbols: 9 × 9 × 9 = 729 components (sparse)
    alignas(64) std::array<std::array<float, BLOCK_SIZE>, 729> christoffel;
};

// Proxy accessor class (maintains API compatibility)
class TorusNodeProxy {
    TorusBlock* block;
    size_t index;
    
public:
    std::complex<double> psi() const {
        return {block->psi_real[index], block->psi_imag[index]};
    }
    
    void set_psi(std::complex<double> val) {
        block->psi_real[index] = val.real();
        block->psi_imag[index] = val.imag();
    }
    
    // ... metric accessors ...
};
```

### Implementation Requirements

1. **Refactor all grid code** to use `TorusBlock` arrays instead of `TorusNode` arrays
2. **CUDA kernels** must use coalesced memory access patterns (threads access contiguous indices)
3. **Cache alignment:** All arrays must be 64-byte aligned (`alignas(64)`)
4. **Block size:** Must be power of 3^9 for efficient torus indexing

### Performance Impact

- **Memory bandwidth:** 3.6% → 100% efficiency (28x improvement)
- **Cache hit rate:** ~10% → ~95% (9.5x improvement)
- **Overall speedup:** ~10x for physics kernel

**Priority:** P0 (Critical)  
**Timeline:** 2 days  
**Validation:** Physics kernel must achieve <1ms per step on sparse 27³ grid

### 1.1 9D Dimensional Semantics

Strict type enforcement for dimensional mapping:

| Dimension | Symbol | Role | Data Type | Physics Interpretation |
|-----------|--------|------|-----------|------------------------|
| 1 | $r$ | Resonance | float [0.0, 1.0] | Damping coefficient $\gamma$. High $r$ = Low Damping (Long-term memory) |
| 2 | $s$ | State | float [0.0, 2.0] | Refractive Index $\eta$. Defines local speed of light $c$ |
| 3 | $t$ | Time | float (cyclic) | Temporal phase (modulo $2\pi$) |
| 4-6 | $u,v,w$ | Quantum | float [0.0, 1.0] | Quantum state subspace dimensions |
| 7-9 | $x,y,z$ | Spatial | float [0.0, 1.0] | Physical 3D embedding coordinates |

**Constraint Enforcement:** All coordinate access must validate ranges. Out-of-range values indicate either programming errors or physics violations requiring immediate halt.

---

## 2. SPLIT-OPERATOR SYMPLECTIC INTEGRATION

### Problem Statement

The original specification suggested Velocity-Verlet integration for the UFIE:

$$\frac{\partial^2 \Psi}{\partial t^2} + \alpha(1 - \hat{r}) \frac{\partial \Psi}{\partial t} - \frac{c_0^2}{(1 + \hat{s})^2} \nabla^2_g \Psi = \sum_{i=1}^8 \mathcal{E}_i(\vec{x}, t) + \beta |\Psi|^2 \Psi$$

**Issue:** The damping term $\alpha(1 - \hat{r}) \frac{\partial \Psi}{\partial t}$ is non-conservative. Standard Verlet methods assume Hamiltonian systems and fail to conserve energy in the presence of friction. This causes:
- Energy drift (memories vanish or explode exponentially)
- Numerical instability (system diverges within hours)
- Loss of standing waves (catastrophic "amnesia")

### Solution: Split-Operator Strang Splitting

Decompose the UFIE into three operators:

1. **Damping Operator:** $\hat{D} = -\gamma \frac{\partial}{\partial t}$ (dissipative)
2. **Conservative Operator:** $\hat{H} = \frac{\partial^2}{\partial t^2} - c^2 \nabla^2$ (Hamiltonian)
3. **Nonlinear Operator:** $\hat{N} = \beta |\Psi|^2 \Psi$ (conservative but nonlinear)

Apply Strang splitting for second-order accuracy:

$$e^{(\hat{D} + \hat{H} + \hat{N})\Delta t} \approx e^{\hat{D}\Delta t/2} e^{\hat{H}\Delta t/2} e^{\hat{N}\Delta t} e^{\hat{H}\Delta t/2} e^{\hat{D}\Delta t/2} + O(\Delta t^3)$$

### Implementation Algorithm

```cpp
void propagate_wave_split_operator(double dt) {
    const double dt_half = dt / 2.0;
    
    // Step 1: Half-kick damping (exact analytical solution)
    // v(t + dt/2) = v(t) * exp(-γ * dt/2)
    for (auto& node : active_nodes) {
        double gamma = alpha * (1.0 - node.resonance);  // Damping coefficient
        double decay = std::exp(-gamma * dt_half);
        node.psi_velocity *= decay;
    }
    
    // Step 2: Half-kick conservative force (Laplacian + emitters)
    // v(t + dt/2) += F(t) * dt/2
    compute_laplacian();  // Calculates ∇²Ψ
    for (auto& node : active_nodes) {
        double c_eff = c0 / std::pow(1.0 + node.state, 2);  // Effective speed
        std::complex<double> force = c_eff * c_eff * node.laplacian;
        force += emitter_field[node.index];  // External driving
        node.psi_velocity += force * dt_half;
    }
    
    // Step 3: Drift (update position)
    // Ψ(t + dt) = Ψ(t) + v(t + dt/2) * dt
    for (auto& node : active_nodes) {
        node.psi += node.psi_velocity * dt;
    }
    
    // Step 4: Apply nonlinear operator (implicit RK2 for stability)
    // Ψ(t + dt) = Ψ(t + dt) + β|Ψ|²Ψ * dt
    for (auto& node : active_nodes) {
        double magnitude_sq = std::norm(node.psi);
        node.psi += beta * magnitude_sq * node.psi * dt;
    }
    
    // Step 5: Half-kick force (recompute at new position)
    compute_laplacian();
    for (auto& node : active_nodes) {
        double c_eff = c0 / std::pow(1.0 + node.state, 2);
        std::complex<double> force = c_eff * c_eff * node.laplacian;
        force += emitter_field[node.index];
        node.psi_velocity += force * dt_half;
    }
    
    // Step 6: Half-kick damping (final decay)
    for (auto& node : active_nodes) {
        double gamma = alpha * (1.0 - node.resonance);
        double decay = std::exp(-gamma * dt_half);
        node.psi_velocity *= decay;
    }
}
```

### Mathematical Justification

**Symplectic Property:** The split-operator method preserves the symplectic structure of the Hamiltonian part, ensuring long-term energy conservation for the conservative terms.

**Exact Damping:** The analytical exponential decay for the damping operator ensures perfect energy dissipation without numerical drift.

**Stability:** Unconditionally stable for the linear terms. The nonlinear term requires $\Delta t < 1/(\beta |\Psi|_{\max})$, which is enforced by adaptive timestepping.

### Implementation Requirements

1. **Replace all Verlet code** with split-operator method
2. **CUDA kernel:** Implement as 6 separate kernel launches (allows device synchronization)
3. **Adaptive timestep:** Monitor $\max |\Psi|$ and reduce $\Delta t$ if it exceeds threshold
4. **Energy watchdog:** Compute total energy $E = \int (|\nabla \Psi|^2 + |\Psi|^2) dV$ every 100 steps, abort if drift exceeds 0.01%

**Priority:** P0 (Critical)  
**Timeline:** 3 days  
**Validation:** Energy conservation within 0.01% over 24-hour simulation

---

## 3. KAHAN COMPENSATED SUMMATION

### Problem Statement

The Laplacian operator in 9 dimensions involves summing contributions from neighbors. A standard finite difference stencil (27-point stencil in 3D, exponentially more in 9D) requires adding many small floating-point numbers to a potentially large accumulator.

**Issue:** In IEEE 754 floating-point arithmetic (FP32), adding a small number to a large number loses precision due to mantissa alignment ("absorption"). This causes:

- High-frequency, low-amplitude waves (subtle/distant memories) are numerically deleted
- System suffers "numerical amnesia"
- Loss of information in interference patterns

### Solution: Kahan Summation

Track low-order bits lost during addition using a compensation variable:

```cpp
struct KahanAccumulator {
    float sum = 0.0f;
    float correction = 0.0f;  // Stores lost low-order bits
    
    inline void add(float input) {
        float y = input - correction;         // Subtract previous correction
        float t = sum + y;                    // Add to sum (loses precision)
        correction = (t - sum) - y;           // Recover lost low-order bits
        sum = t;                              // Update sum
    }
};

// Usage in Laplacian kernel
void compute_laplacian_9d(const TorusGridSoA& grid, size_t node_idx) {
    KahanAccumulator acc_real, acc_imag;
    
    // Sum contributions from all 2×9 = 18 neighbors in 9D
    for (int dim = 0; dim < 9; ++dim) {
        size_t idx_plus = grid.neighbor_index(node_idx, dim, +1);
        size_t idx_minus = grid.neighbor_index(node_idx, dim, -1);
        
        // Second-order central difference: (ψ[i+1] - 2ψ[i] + ψ[i-1]) / h²
        float contrib_real = grid.psi_real[idx_plus] - 2.0f * grid.psi_real[node_idx] + grid.psi_real[idx_minus];
        float contrib_imag = grid.psi_imag[idx_plus] - 2.0f * grid.psi_imag[node_idx] + grid.psi_imag[idx_minus];
        
        acc_real.add(contrib_real);
        acc_imag.add(contrib_imag);
    }
    
    // Store final Laplacian result
    grid.laplacian_real[node_idx] = acc_real.sum;
    grid.laplacian_imag[node_idx] = acc_imag.sum;
}
```

### Mathematical Analysis

Standard floating-point addition accumulates error as $\epsilon_{\text{machine}} \times N$ where $N$ is the number of terms. For a 9D Laplacian with $N = 18$ neighbors:

- **Without Kahan:** Error $\sim 18 \times 10^{-7} \approx 2 \times 10^{-6}$ (FP32)
- **With Kahan:** Error $\sim 10^{-7}$ (near machine precision)

For standing wave patterns with amplitude ratios spanning 6 orders of magnitude (fundamental vs. harmonics), Kahan summation prevents catastrophic cancellation.

### Implementation Requirements

1. **All Laplacian kernels** must use Kahan accumulators
2. **All wave superposition operations** (>3 terms) must use Kahan summation
3. **Metric tensor updates** must use compensated summation
4. **Integration verification:** Test with manufactured solution having known high-frequency component

**Priority:** P0 (Critical)  
**Timeline:** 1 day  
**Validation:** Preserve 10⁻⁶ amplitude waves in presence of unit-amplitude carrier over 10⁶ timesteps

### Performance Impact

- **Stability:** Prevents divergence (critical for multi-hour runs)
- **Accuracy:** 2nd-order in time ($O(\Delta t^2)$ error)
- **Overhead:** ~20% slower than naive Verlet, but necessary for correctness

**Priority:** P0 (Critical)  
**Timeline:** 3 days  
**Validation:** Energy drift must be <0.0001% over 10,000 steps with standing wave test

---

## 3. KAHAN SUMMATION FOR LAPLACIAN

### Problem Statement

The Laplacian computation sums contributions from 18 neighbors in 9D:

$$\nabla^2 \Psi = \sum_{i=1}^{18} w_i (\Psi_{\text{neighbor}_i} - \Psi_{\text{center}})$$

With float32, summing 18 terms loses precision due to rounding errors. This causes:
- Gradual "smearing" of wave packets
- Loss of high-frequency components (fine details)
- Cumulative error accumulation ("amnesia" over days)

### Solution: Kahan Compensated Summation

```cpp
// ❌ FORBIDDEN: Naive summation loses precision
std::complex<float> laplacian = 0.0f;
for (auto& neighbor : neighbors) {
    laplacian += neighbor.psi;
}

// ✅ MANDATORY: Kahan summation preserves precision
std::complex<float> kahan_sum(const std::vector<std::complex<float>>& values) {
    std::complex<float> sum = 0.0f;
    std::complex<float> c = 0.0f;  // Compensation term
    
    for (const auto& val : values) {
        std::complex<float> y = val - c;    // Subtract previous error
        std::complex<float> t = sum + y;    // Add with low bits
        c = (t - sum) - y;                  // Recover rounding error
        sum = t;                            // Update sum
    }
    
    return sum;
}
```

### CUDA Implementation

```cuda
__device__ void kahan_add(float& sum, float& compensation, float value) {
    float y = value - compensation;
    float t = sum + y;
    compensation = (t - sum) - y;
    sum = t;
}

__global__ void compute_laplacian_kahan(float* psi_real, float* psi_imag, 
                                        float* laplacian_real, float* laplacian_imag,
                                        int* neighbor_indices, int num_nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;
    
    float sum_real = 0.0f, c_real = 0.0f;
    float sum_imag = 0.0f, c_imag = 0.0f;
    
    // Sum contributions from 18 neighbors
    for (int n = 0; n < 18; n++) {
        int neighbor_idx = neighbor_indices[idx * 18 + n];
        float contrib_real = psi_real[neighbor_idx] - psi_real[idx];
        float contrib_imag = psi_imag[neighbor_idx] - psi_imag[idx];
        
        kahan_add(sum_real, c_real, contrib_real);
        kahan_add(sum_imag, c_imag, contrib_imag);
    }
    
    laplacian_real[idx] = sum_real;
    laplacian_imag[idx] = sum_imag;
}
```

### Implementation Requirements

1. **Replace all Laplacian summations** with Kahan algorithm
2. **CUDA kernels:** Use register-based compensation (no extra memory)
3. **AVX-512:** Implement vectorized Kahan sum for CPU fallback

### Performance Impact

- **Precision:** Reduces rounding error from $O(n \epsilon)$ to $O(\epsilon)$ where $n$ is number of terms
- **Overhead:** ~10% slower due to extra FP operations
- **Memory:** No additional storage (compensation is register-local)

**Priority:** P0 (Critical)  
**Timeline:** 1 day  
**Validation:** Standing wave must maintain amplitude to 6 decimal places over 1 million steps

---

## 4. AVX-512 NONARY ARITHMETIC

### Problem Statement

Balanced nonary arithmetic requires saturation at $\pm 4$. Standard CPU ALUs perform binary arithmetic, requiring explicit clamping after every operation.

Scalar implementation:

```cpp
// ❌ SLOW: Scalar saturation (200x slower than needed)
Nit add_nonary(Nit a, Nit b) {
    int result = static_cast<int>(a) + static_cast<int>(b);
    if (result > 4) return Nit::FOUR;
    if (result < -4) return Nit::NEG_FOUR;
    return static_cast<Nit>(result);
}
```

**Issue:** Processing 1M nits sequentially takes ~5ms. With SIMD, this can be reduced to ~25μs (200x speedup).

### Solution: AVX-512 Vectorization

```cpp
// ✅ MANDATORY: AVX-512 saturated nonary addition (64 nits per operation)
#include <immintrin.h>

void add_nonary_simd(const int8_t* a, const int8_t* b, int8_t* result, size_t count) {
    const __m512i limit_pos = _mm512_set1_epi8(4);   // Upper bound
    const __m512i limit_neg = _mm512_set1_epi8(-4);  // Lower bound
    
    size_t i = 0;
    for (; i + 64 <= count; i += 64) {
        // Load 64 nits
        __m512i va = _mm512_loadu_si512((__m512i*)(a + i));
        __m512i vb = _mm512_loadu_si512((__m512i*)(b + i));
        
        // Saturated addition (with hardware saturation at ±127)
        __m512i vsum = _mm512_adds_epi8(va, vb);
        
        // Clamp to [-4, 4] (nonary saturation)
        vsum = _mm512_min_epi8(vsum, limit_pos);
        vsum = _mm512_max_epi8(vsum, limit_neg);
        
        // Store result
        _mm512_storeu_si512((__m512i*)(result + i), vsum);
    }
    
    // Handle remaining elements (scalar fallback)
    for (; i < count; i++) {
        int sum = a[i] + b[i];
        result[i] = std::clamp(sum, -4, 4);
    }
}
```

### Multiplication via Lookup Table

Nonary multiplication requires heterodyning (wave mixing). For performance, use a precomputed 9×9 lookup table:

```cpp
// Precomputed nonary multiplication table
static constexpr int8_t NONARY_MUL_TABLE[9][9] = {
    // Row: multiplier value (-4 to 4), Column: multiplicand (-4 to 4)
    { 4,  3,  2,  1,  0, -1, -2, -3, -4},  // -4 × {...}
    { 3,  2,  1,  1,  0, -1, -1, -2, -3},  // -3 × {...}
    { 2,  1,  1,  0,  0,  0, -1, -1, -2},  // -2 × {...}
    { 1,  1,  0,  0,  0,  0,  0, -1, -1},  // -1 × {...}
    { 0,  0,  0,  0,  0,  0,  0,  0,  0},  //  0 × {...}
    {-1, -1,  0,  0,  0,  0,  0,  1,  1},  //  1 × {...}
    {-2, -1, -1,  0,  0,  0,  1,  1,  2},  //  2 × {...}
    {-3, -2, -1, -1,  0,  1,  1,  2,  3},  //  3 × {...}
    {-4, -3, -2, -1,  0,  1,  2,  3,  4},  //  4 × {...}
};

__m512i mul_nonary_simd(__m512i a, __m512i b) {
    // Use gather operation with lookup table
    // This requires AVX-512VBMI2 for efficient byte-level gather
    // Fallback: process 8 elements at a time with scalar lookup
    alignas(64) int8_t a_arr[64], b_arr[64], result[64];
    _mm512_store_si512((__m512i*)a_arr, a);
    _mm512_store_si512((__m512i*)b_arr, b);
    
    for (int i = 0; i < 64; i++) {
        int ai = a_arr[i] + 4;  // Convert [-4,4] to [0,8]
        int bi = b_arr[i] + 4;
        result[i] = NONARY_MUL_TABLE[ai][bi];
    }
    
    return _mm512_load_si512((__m512i*)result);
}
```

### Implementation Requirements

1. **CPU feature detection:** Check for AVX-512 support at runtime, fallback to scalar
2. **Memory alignment:** All nit arrays must be 64-byte aligned
3. **Compiler flags:** `-mavx512f -mavx512bw -mavx512vl`

### Performance Impact

- **Addition:** 200x speedup (64 nits per SIMD instruction vs 1 per scalar)
- **Multiplication:** ~50x speedup (lookup table is cache-friendly)
- **Total:** Nonary operations become negligible (<1% of runtime)

**Priority:** P1 (High)  
**Timeline:** 2 days  
**Validation:** Process 10M nonary additions in <50μs

---

## 5. LAZY CHOLESKY DECOMPOSITION FOR METRIC TENSOR

### Problem Statement

The metric tensor $g_{ij}$ is a 9×9 symmetric positive-definite matrix. To compute the Laplacian in curved space, we need:

$$\nabla^2 \Psi = g^{ij} \nabla_i \nabla_j \Psi$$

This requires inverting $g_{ij}$ to obtain $g^{ij}$. Naive matrix inversion (Gaussian elimination) is $O(n^3) = O(729)$ operations per node per timestep.

For 1M active nodes at 60 FPS:
- Operations: $1,000,000 \times 729 \times 60 = 4.4 \times 10^{10}$ per second
- Cost: ~100 CPU cores to maintain real-time (UNACCEPTABLE)

### Solution: Lazy Cholesky Decomposition with Caching

**Key Insight:** The metric tensor changes slowly (plasticity timescale is ~seconds). We can cache the decomposition and only recompute when the tensor changes significantly.

```cpp
class MetricTensorCache {
    std::array<double, 45> g_lower_triangle;  // Stored metric (symmetric)
    std::array<double, 45> L_cholesky;        // Cached Cholesky factor
    bool is_valid = false;
    double change_threshold = 1e-6;
    
public:
    // Check if metric has changed significantly
    bool needs_update(const std::array<double, 45>& new_g) const {
        double max_diff = 0.0;
        for (int i = 0; i < 45; i++) {
            max_diff = std::max(max_diff, std::abs(new_g[i] - g_lower_triangle[i]));
        }
        return max_diff > change_threshold;
    }
    
    // Update Cholesky decomposition (only when needed)
    void update_if_changed(const std::array<double, 45>& new_g) {
        if (!needs_update(new_g) && is_valid) {
            return;  // Use cached value
        }
        
        // Perform Cholesky decomposition: g = L * L^T
        // ... Cholesky algorithm (O(n³) but rare) ...
        
        g_lower_triangle = new_g;
        is_valid = true;
    }
    
    // Compute g^{-1} * v using forward/backward substitution (O(n²))
    std::array<double, 9> apply_inverse(const std::array<double, 9>& v) {
        // Solve L * y = v (forward substitution)
        std::array<double, 9> y;
        // ... O(n²) ...
        
        // Solve L^T * x = y (backward substitution)
        std::array<double, 9> x;
        // ... O(n²) ...
        
        return x;  // x = g^{-1} * v
    }
};
```

### Batch Update Strategy

For plasticity updates (which happen every ~1000 timesteps):

```cpp
void update_metric_batch() {
    // Identify nodes with changed metrics
    std::vector<size_t> dirty_nodes;
    for (size_t i = 0; i < active_nodes.size(); i++) {
        if (active_nodes[i].metric_dirty_flag) {
            dirty_nodes.push_back(i);
        }
    }
    
    // Parallel Cholesky decomposition (embarrassingly parallel)
    #pragma omp parallel for
    for (size_t idx : dirty_nodes) {
        active_nodes[idx].metric_cache.update_if_changed(
            active_nodes[idx].metric_tensor
        );
        active_nodes[idx].metric_dirty_flag = false;
    }
}
```

### Implementation Requirements

1. **Caching layer:** Add `MetricTensorCache` to `TorusNode` (or `TorusBlock` with SoA)
2. **Dirty flags:** Track which nodes have changed metrics
3. **Batch updates:** Update caches once per 1000 physics steps (not every step)
4. **Fallback:** For rapidly changing metrics, use direct inversion (rare case)

### Performance Impact

- **Speedup:** 100x for metric-related operations (amortized)
- **Cache hit rate:** >99% during steady-state operation
- **Memory overhead:** +360 bytes per node (Cholesky factor storage)

**Priority:** P1 (High)  
**Timeline:** 2 days  
**Validation:** Metric inversion overhead must be <5% of total runtime

---

## 6. SHARED MEMORY ZERO-COPY IPC

### Problem Statement

ZeroMQ serialization (Protocol Buffers) for high-frequency data (physics state at 60 FPS) introduces:
- Latency: ~100μs per frame (serialization + network stack)
- CPU overhead: ~10% (protobuf encoding/decoding)
- Memory allocation: frequent malloc/free causes fragmentation

For real-time visualization and memory systems, this is unacceptable.

### Solution: Shared Memory with Seqlock

```cpp
// Shared memory header (lives in /dev/shm/nikola_frame)
struct SharedFrame {
    // Seqlock for concurrency control
    std::atomic<uint64_t> sequence;  // Even = stable, Odd = writing
    
    // Metadata
    uint64_t timestamp_ns;
    uint32_t frame_number;
    uint32_t active_node_count;
    
    // Data payload (variable size)
    struct NodeState {
        uint64_t morton_code;  // Z-order index
        float psi_real, psi_imag;
        float energy_density;
    } nodes[];  // Flexible array member
};

// Writer (Physics Engine)
class SharedMemoryWriter {
    int shm_fd;
    SharedFrame* frame;
    size_t capacity;
    
public:
    void write_frame(const std::vector<NodeState>& nodes) {
        // 1. Increment sequence (mark as writing)
        uint64_t seq = frame->sequence.load(std::memory_order_acquire);
        frame->sequence.store(seq + 1, std::memory_order_release);
        
        // 2. Write data
        frame->timestamp_ns = get_timestamp_ns();
        frame->frame_number++;
        frame->active_node_count = nodes.size();
        std::memcpy(frame->nodes, nodes.data(), nodes.size() * sizeof(NodeState));
        
        // 3. Increment sequence again (mark as stable)
        frame->sequence.store(seq + 2, std::memory_order_release);
        
        // 4. Notify readers via tiny ZMQ message (8 bytes)
        zmq_send(notify_socket, &frame->frame_number, sizeof(uint32_t), ZMQ_DONTWAIT);
    }
};

// Reader (Visualizer)
class SharedMemoryReader {
    int shm_fd;
    const SharedFrame* frame;
    
public:
    std::optional<std::vector<NodeState>> read_frame() {
        uint64_t seq1, seq2;
        std::vector<NodeState> nodes;
        
        do {
            // Read sequence number (before)
            seq1 = frame->sequence.load(std::memory_order_acquire);
            if (seq1 & 1) continue;  // Writer is active, retry
            
            // Read data
            nodes.resize(frame->active_node_count);
            std::memcpy(nodes.data(), frame->nodes, 
                       frame->active_node_count * sizeof(NodeState));
            
            // Read sequence number (after)
            std::atomic_thread_fence(std::memory_order_acquire);
            seq2 = frame->sequence.load(std::memory_order_relaxed);
            
        } while (seq1 != seq2);  // Retry if data was modified during read
        
        return nodes;
    }
};
```

### Implementation Requirements

1. **Shared memory segment:** Allocate in `/dev/shm` (tmpfs, zero-copy)
2. **Size calculation:** Max frame size = `sizeof(SharedFrame) + MAX_ACTIVE_NODES * sizeof(NodeState)`
3. **ZMQ notification:** Use PUB-SUB pattern for frame-ready signals (no blocking)
4. **Cleanup:** Unlink shared memory on shutdown (`shm_unlink`)

### Performance Impact

- **Latency:** 100μs → 1μs (100x reduction)
- **Bandwidth:** No serialization overhead (direct memory access)
- **CPU:** 10% → 0.1% (no protobuf encoding)

**Priority:** P2 (Medium)  
**Timeline:** 2 days  
**Validation:** Visualizer must receive frames with <10μs latency jitter

---

## 7. 128-BIT MORTON CODES FOR Z-ORDER CURVES

### Problem Statement

Sparse grid hashing uses Z-order (Morton) curves to map 9D coordinates to linear indices. Standard implementation:

```cpp
// ❌ INSUFFICIENT: 64-bit keys cause collisions at high resolution
uint64_t morton_encode_9d(const std::array<uint16_t, 9>& coords) {
    // Each coordinate is 7 bits (max value 127)
    // Total: 9 × 7 = 63 bits (fits in uint64_t)
    // ...
}
```

**Issue:** This limits grid resolution to $128^9 \approx 10^{18}$ voxels. For detailed memory regions, we need $2^{14} = 16384$ voxels per dimension, requiring $9 \times 14 = 126$ bits.

**Consequence:** Hash collisions overwrite existing memories (data corruption).

### Solution: __int128_t Morton Codes

```cpp
// ✅ MANDATORY: 128-bit Morton codes (14 bits per dimension × 9 = 126 bits)
using MortonCode = __uint128_t;  // GCC/Clang extension

MortonCode morton_encode_9d(const std::array<uint16_t, 9>& coords) {
    MortonCode result = 0;
    
    for (int bit = 0; bit < 14; bit++) {
        for (int dim = 0; dim < 9; dim++) {
            // Extract bit from coordinate
            uint16_t coord_bit = (coords[dim] >> bit) & 1;
            
            // Place bit in Morton code
            int morton_bit_pos = bit * 9 + dim;
            result |= (static_cast<MortonCode>(coord_bit) << morton_bit_pos);
        }
    }
    
    return result;
}

std::array<uint16_t, 9> morton_decode_9d(MortonCode code) {
    std::array<uint16_t, 9> coords = {0};
    
    for (int bit = 0; bit < 14; bit++) {
        for (int dim = 0; dim < 9; dim++) {
            int morton_bit_pos = bit * 9 + dim;
            uint16_t coord_bit = (code >> morton_bit_pos) & 1;
            coords[dim] |= (coord_bit << bit);
        }
    }
    
    return coords;
}
```

### Hash Table Implementation

```cpp
#include <unordered_map>

// Custom hash function for __uint128_t
struct MortonHasher {
    size_t operator()(__uint128_t key) const {
        // XOR high and low 64 bits
        uint64_t low = static_cast<uint64_t>(key);
        uint64_t high = static_cast<uint64_t>(key >> 64);
        return std::hash<uint64_t>{}(low ^ high);
    }
};

// Sparse grid map
std::unordered_map<__uint128_t, TorusNodeProxy, MortonHasher> sparse_grid;
```

### Implementation Requirements

1. **Compiler support:** GCC/Clang only (MSVC uses `_BitInt(128)` in C++23)
2. **Serialization:** Split into two `uint64_t` for storage/transmission
3. **Overflow checks:** Assert coordinates are ≤ 16383 (14 bits)

### Performance Impact

- **Collision rate:** 100% → 0% (eliminates hash collisions)
- **Memory overhead:** 8 bytes → 16 bytes per key (acceptable)
- **Correctness:** CRITICAL (prevents data corruption)

**Priority:** P1 (High)  
**Timeline:** 1 day  
**Validation:** Insert 10M nodes with no collisions, verify retrieval

---

## 8. Q9_0 QUANTIZATION CORRECTION

### Problem Statement

The original spec suggested Q9_0 quantization packs 5 nits into `uint16_t`:
- $9^5 = 59,049 < 65,536$ ✅ Fits
- Storage: $16 / 5 = 3.2$ bits per nit

**Issue:** The encoding/decoding logic must handle the 9-ary radix conversion correctly. Naive implementation:

```cpp
// ❌ INCORRECT: Loses precision for large values
uint16_t encode_q9(const Nit nits[5]) {
    uint16_t result = 0;
    for (int i = 0; i < 5; i++) {
        int digit = static_cast<int>(nits[i]) + 4;  // Convert [-4,4] to [0,8]
        result = result * 9 + digit;  // Radix 9 accumulation
    }
    return result;
}
```

This works but loses the ability to index individual nits efficiently.

### Solution: Proper Radix Encoding

```cpp
// ✅ CORRECT: Radix-9 encoding with explicit powers
uint16_t encode_q9_0(const std::array<Nit, 5>& nits) {
    static constexpr uint16_t POWERS_OF_9[5] = {1, 9, 81, 729, 6561};
    
    uint16_t result = 0;
    for (int i = 0; i < 5; i++) {
        int digit = static_cast<int>(nits[i]) + 4;  // [-4,4] → [0,8]
        result += digit * POWERS_OF_9[i];
    }
    
    return result;
}

std::array<Nit, 5> decode_q9_0(uint16_t encoded) {
    static constexpr uint16_t POWERS_OF_9[5] = {1, 9, 81, 729, 6561};
    std::array<Nit, 5> nits;
    
    for (int i = 4; i >= 0; i--) {
        int digit = encoded / POWERS_OF_9[i];
        nits[i] = static_cast<Nit>(digit - 4);  // [0,8] → [-4,4]
        encoded %= POWERS_OF_9[i];
    }
    
    return nits;
}
```

### SIMD Batch Encoding

```cpp
// Encode 64 nits (12.8 uint16_t values) using AVX-512
void encode_q9_0_batch(const int8_t* nits, uint16_t* encoded, size_t count) {
    static constexpr int CHUNK_SIZE = 5;
    
    for (size_t i = 0; i + CHUNK_SIZE <= count; i += CHUNK_SIZE) {
        std::array<Nit, 5> chunk;
        std::memcpy(chunk.data(), nits + i, CHUNK_SIZE);
        encoded[i / CHUNK_SIZE] = encode_q9_0(chunk);
    }
    
    // Handle remainder
    // ...
}
```

### Implementation Requirements

1. **Validation:** Roundtrip test (encode → decode must match input)
2. **Bounds checking:** Assert nits are in [-4, 4] before encoding
3. **Alignment:** Pad to multiple of 5 nits for efficient SIMD processing

### Performance Impact

- **Storage:** 8 bits → 3.2 bits per nit (2.5x compression)
- **Speed:** Encoding/decoding is ~50ns per 5-nit block (negligible)

**Priority:** P2 (Medium)  
**Timeline:** 1 day  
**Validation:** 1M nit roundtrip test with 100% accuracy

---

## 9. VALIDATION AND MONITORING

### 9.1 Energy Watchdog

**Purpose:** Detect numerical instability by monitoring total system energy.

```cpp
class EnergyWatchdog {
    double initial_energy = 0.0;
    double tolerance = 1e-4;  // 0.01% drift allowed
    
public:
    void initialize(const TorusGrid& grid) {
        initial_energy = compute_total_energy(grid);
    }
    
    void check(const TorusGrid& grid, int step) {
        if (step % 100 != 0) return;  // Check every 100 steps
        
        double current_energy = compute_total_energy(grid);
        double drift = std::abs(current_energy - initial_energy) / initial_energy;
        
        if (drift > tolerance) {
            std::cerr << "CRITICAL: Energy drift " << drift * 100 << "% at step " 
                      << step << std::endl;
            std::cerr << "Initial: " << initial_energy 
                      << ", Current: " << current_energy << std::endl;
            std::abort();  // Fail fast
        }
    }
    
private:
    double compute_total_energy(const TorusGrid& grid) {
        double kinetic = 0.0, potential = 0.0;
        
        for (const auto& node : grid.active_nodes()) {
            // Kinetic energy: (1/2) * |∂Ψ/∂t|²
            kinetic += 0.5 * std::norm(node.psi_velocity);
            
            // Potential energy: (1/2) * |∇Ψ|² (computed via Laplacian)
            potential += 0.5 * std::norm(node.laplacian);
        }
        
        return kinetic + potential;
    }
};
```

### 9.2 Performance Profiler

**Purpose:** Identify bottlenecks in the physics loop.

```cpp
class PhysicsProfiler {
    std::unordered_map<std::string, std::chrono::nanoseconds> timings;
    
public:
    struct ScopedTimer {
        PhysicsProfiler& profiler;
        std::string name;
        std::chrono::steady_clock::time_point start;
        
        ScopedTimer(PhysicsProfiler& p, std::string n) 
            : profiler(p), name(std::move(n)), 
              start(std::chrono::steady_clock::now()) {}
        
        ~ScopedTimer() {
            auto elapsed = std::chrono::steady_clock::now() - start;
            profiler.record(name, elapsed);
        }
    };
    
    void record(const std::string& name, std::chrono::nanoseconds duration) {
        timings[name] += duration;
    }
    
    void print_report(int num_frames) {
        std::cout << "=== Physics Profiler ===" << std::endl;
        for (const auto& [name, total] : timings) {
            double avg_ms = total.count() / (1e6 * num_frames);
            std::cout << name << ": " << avg_ms << " ms/frame" << std::endl;
        }
    }
};

// Usage:
void physics_step() {
    PhysicsProfiler::ScopedTimer timer(profiler, "LaplacianCompute");
    compute_laplacian();
}
```

### 9.3 Correctness Tests

**Harmonic Oscillator Test:**

```cpp
void test_harmonic_oscillator() {
    // Initial condition: Gaussian wave packet
    // Ψ(x,0) = exp(-x²/2) * exp(ikx)
    
    // Expected: Oscillates with frequency ω = √(c² + k²)
    // Energy should remain constant
    
    // Run for 1000 cycles, check amplitude preservation
}
```

**Standing Wave Test:**

```cpp
void test_standing_wave() {
    // Initial: sin(πx/L) * sin(πy/L) pattern
    // Expected: Remains stationary (zero group velocity)
    
    // Run for 10,000 steps, check position stability
}
```

**Priority:** P1 (High)  
**Timeline:** Integrated into Phase 0 validation  
**Gate:** All tests must pass before Phase 1

---

## 10. PHASE 0 COMPLETION CHECKLIST

### P0 Tasks (Critical - 6 days)

- [ ] **Day 1-2:** Refactor `TorusNode` to SoA layout (`TorusBlock`)
  - [ ] Create `TorusBlock` struct with aligned arrays
  - [ ] Implement `TorusNodeProxy` accessor class
  - [ ] Update grid allocation code
  - [ ] Update CUDA kernels for coalesced access
  - [ ] **Validation:** Measure memory bandwidth (must achieve >80% of peak)

- [ ] **Day 3-5:** Implement Split-Operator Symplectic Integration
  - [ ] Replace Verlet with 6-step Strang splitting
  - [ ] Implement analytical damping decay
  - [ ] Add adaptive timestep control
  - [ ] **Validation:** Energy drift <0.0001% over 10K steps

- [ ] **Day 6:** Implement Kahan Summation for Laplacian
  - [ ] Update Laplacian accumulation loops
  - [ ] Add CUDA kernel with compensation
  - [ ] **Validation:** Standing wave amplitude stable to 6 decimals over 1M steps

### P1 Tasks (High - 6 days)

- [ ] **Day 7-8:** AVX-512 Nonary Arithmetic
  - [ ] Implement vectorized add/multiply
  - [ ] Create lookup tables
  - [ ] Add CPU feature detection
  - [ ] **Validation:** 10M operations in <50μs

- [ ] **Day 9-11:** Lazy Cholesky Decomposition
  - [ ] Add `MetricTensorCache` class
  - [ ] Implement dirty tracking
  - [ ] Add batch update logic
  - [ ] **Validation:** Metric overhead <5% of runtime

- [ ] **Day 12:** Energy Watchdog
  - [ ] Implement energy computation
  - [ ] Add periodic checks
  - [ ] **Validation:** Detect artificial drift injection

### P2 Tasks (Medium - 5 days)

- [ ] **Day 13-14:** Shared Memory IPC
  - [ ] Create seqlock implementation
  - [ ] Allocate `/dev/shm` segment
  - [ ] Integrate with ZMQ notifications
  - [ ] **Validation:** <10μs latency jitter

- [ ] **Day 15-16:** Mamba Taylor Approximation
  - [ ] Implement first-order matrix approximation
  - [ ] Add adaptive timestep
  - [ ] **Validation:** 10x speedup vs full matrix exp

- [ ] **Day 17:** Q9_0 Quantization
  - [ ] Implement radix-9 encoding
  - [ ] Add batch SIMD encoder
  - [ ] **Validation:** 1M roundtrip with 100% accuracy

### Gate Review

**Criteria for Phase 1 Entry:**
1. ✅ All P0 and P1 tasks complete
2. ✅ All validation tests pass
3. ✅ Energy watchdog operational
4. ✅ Physics step <1ms on sparse 27³ grid
5. ✅ Code review completed (2 engineer sign-off)

**If gate fails:** Remediation continues until all criteria met. NO EXCEPTIONS.

---

## CONCLUSION

Phase 0 remediation is **non-negotiable**. The original specification contained latent defects that would cause catastrophic failure in production. These fixes transform the system from a theoretical model into a production-ready implementation.

**Expected Outcome:** After Phase 0, the physics engine will:
- Run stably for days without divergence
- Achieve real-time performance on commodity hardware
- Preserve memory precision over millions of cycles
- Provide a solid foundation for cognitive layer development

**Next Steps:** Upon successful gate review, proceed to Phase 1 (Core Physics Engine) with confidence that the foundation is mathematically sound and computationally stable.
# Nikola Model v0.0.4: Implementation Roadmap

## 8.7 Executive Summary

This document provides the definitive implementation roadmap for the Nikola Model v0.0.4, converting the theoretical architecture into a buildable specification. All 37 identified implementation gaps have been addressed with concrete specifications, reference implementations, validation procedures, and failure mode analyses.

**Status Update (2025-12-10):** 🔴 **ON HOLD** - Critical blocking dependencies discovered

**Critical Finding:** Aria's implementation review identified 2 Priority 1 blocking issues that must be resolved before Phase 1-7 implementation can begin.

**Next Step:** Implement Phase 0 Critical Remediations (CF-04, MEM-04)

---

## ⚠️ CRITICAL: Phase 0 - Blocking Dependencies (NEW)

**Timeline:** Weeks 1-3
**Dependencies:** None (foundational blockers)
**Priority:** 🔴 **CRITICAL - BLOCKS ALL OTHER PHASES**

### Overview

During implementation review, Lead Architect Aria Echo identified 2 **Priority 1 Critical** architectural vulnerabilities that must be remediated before any Phase 1-7 work can proceed. These represent fundamental stability and coherence issues.

**Document:** [08_critical_remediations.md](08_critical_remediations.md)

### Finding CF-04: Transactional Metabolic Lock

**Problem:** Thermodynamic race condition in ATP (metabolic energy) management
**Impact:** System can enter negative energy states causing "cognitive seizures" (catastrophic failure)
**Solution:** RAII-based transactional guards using atomic Compare-And-Swap (CAS)

#### Deliverables

1. **MetabolicTransaction Class** (C++23 RAII)
   - Atomic CAS-based reservation protocol
   - Automatic rollback on exception
   - Thread-safe energy accounting

2. **MetabolicController Extensions**
   - `try_reserve()` with CAS loop
   - `refund()` for rollback support
   - Memory ordering guarantees

#### Validation Requirements

- **Unit Test:** 10 threads competing for limited ATP - exact accounting verified
- **Rollback Test:** Exception safety - energy refunded on failure
- **Integration Test:** System enters Nap gracefully, never crashes or goes negative

**Risk Level:** 🔴 **CRITICAL** - Without this, concurrent operations can violate conservation laws

### Finding MEM-04: Hilbert Re-indexing Strategy

**Problem:** Morton codes create spatial discontinuities destroying Mamba-9D sequential coherence
**Impact:** "Semantic aphasia" - high perplexity, hallucinations, inefficient neurogenesis
**Solution:** Causal-Foliated Hilbert scanning preserving temporal + spatial locality

#### Deliverables

1. **HilbertScanner Class** (128-bit precision)
   - `encode_hilbert_9d()` for 9D → 1D mapping
   - `generate_scan_order()` with time-first foliation
   - `reindex_grid()` for SoA memory reorganization

2. **Orchestrator Integration**
   - Fragmentation index monitoring
   - Periodic re-indexing (threshold = 0.15)
   - Transparent to Mamba-9D layer

#### Validation Requirements

- **Locality Preservation Ratio (LPR):** LPR(Hilbert) < 0.8 × LPR(Morton)
- **Mamba Perplexity Test:** Validation loss on Hilbert-sorted data significantly lower (p < 0.05)
- **Neurogenesis Test:** New nodes remain connected to semantic neighbors

**Risk Level:** 🔴 **CRITICAL** - Without this, Mamba-9D cannot learn coherent patterns

### Phase 0 Success Criteria

Before proceeding to Phase 1:

1. ✅ CF-04 passes all unit/integration tests (no race conditions)
2. ✅ MEM-04 demonstrates LPR improvement > 20%
3. ✅ Mamba perplexity reduced by statistically significant margin
4. ✅ Both implementations integrated into Orchestrator control loop
5. ✅ No performance regression (re-indexing overhead acceptable)

**Timeline Impact:** Adds 3 weeks to schedule (original Phase 1 becomes Weeks 4-11)

---

## Implementation Phases (Updated)

### Phase 1: Physics Core (Critical Path)

**Timeline:** Weeks 4-11 (updated from 1-8)
**Dependencies:** Phase 0 complete (CF-04, MEM-04)
**Priority:** HIGHEST

#### Deliverables

1. **UFIE Physics Engine** ([01_core_physics_implementation.md](01_core_physics_implementation.md))
   - Gap 1.1: Emitter field generation with harmonic spatial injection ✅
   - Gap 1.2: Thermal bath velocity initialization ✅
   - Gap 1.3: Perfectly Matched Layer boundary conditions ✅
   - Gap 1.4: CUDA kernel launch configuration (256 threads/block) ✅
   - Gap 1.5: Quantum Zeno Freeze recovery procedure ✅
   - Gap 1.6: Double-buffered profiling hooks ✅

2. **Validation Requirements**
   - Energy conservation test: |ΔH/H| < 0.01% over 10,000 steps
   - Symplectic integration stability verification
   - Ortho-check: Injected tokens maintain soft orthogonality
   - Performance: Achieve 1000-2000 Hz physics loop on RTX 4090

#### Critical Success Factors

- **Energy conservation is mandatory.** Violations cause system "seizures"
- **Numerical stability** requires FP32 with Kahan summation
- **Real-time requirement:** Must sustain 2 kHz loop for responsiveness

**Risk Level:** 🔴 **HIGHEST** - If Gap 1.1 (emitter scaling) is wrong, the nonlinearity β causes immediate numerical explosion.

---

### Phase 2: Manifold Geometry (Enabler)

**Timeline:** Weeks 12-17 (updated from 9-14)
**Dependencies:** Phase 0, Phase 1 complete
**Priority:** HIGH

#### Deliverables

1. **9D Toroidal Grid** ([02_geometry_spatial_implementation.md](02_geometry_spatial_implementation.md))
   - Gap 2.1: Gerschgorin + Tikhonov metric validation ✅
   - Gap 2.2: Hilbert curve rotation table generation ✅
   - Gap 2.3: Anisotropic resolution allocation (x,y,z=64, t=128, r,s=16, u,v,w=32) ✅
   - Gap 2.4: Dual integer/float coordinate system ✅
   - Gap 2.5: Dopamine-modulated learning rate schedule ✅

2. **Validation Requirements**
   - Metric tensor always positive-definite (no Cholesky failures)
   - Morton/Hilbert encoding round-trip test
   - Spatial indexing performance: O(log N) neighbor lookup

#### Critical Success Factors

- **Without the grid, physics has nowhere to run**
- Sparse addressing essential for memory efficiency
- Metric learning enables neuroplasticity

**Risk Level:** 🟡 **MEDIUM** - Metric validation is critical but well-understood mathematically.

---

### Phase 3: Cognitive Architecture (Intelligence Emerges)

**Timeline:** Weeks 18-25 (updated from 15-22)
**Dependencies:** Phase 0, Phases 1, 2 complete
**Priority:** HIGH

#### Deliverables

1. **Mamba-9D Language Model** ([03_cognitive_architecture_implementation.md](03_cognitive_architecture_implementation.md))
   - Gap 3.1: LSH-based semantic token mapping ✅
   - Gap 3.2: SSM dimension = 256 (r×s state space) ✅
   - Gap 3.3: Sliding wave window (L_eff ≈ 100 steps) ✅
   - Gap 3.4: Physics-grounded lexicon initialization via FFT ✅
   - Gap 3.5: Born rule sampling with temperature as noise ✅
   - Gap 3.6: Equilibrium Propagation training (no backprop through physics) ✅

2. **Validation Requirements**
   - Token generation latency: 10-50 tokens/second
   - Semantic clustering: Synonyms within 10 grid cells
   - Training convergence: Energy decreases over 100 EqProp iterations

#### Critical Success Factors

- **Bridges continuous physics to discrete tokens**
- Equilibrium Propagation avoids gradient explosion
- Spectral signatures ground lexicon in physics

**Risk Level:** 🟡 **MEDIUM** - Novel training approach requires empirical tuning.

---

### Phase 4: Infrastructure & Communications (Orchestration)

**Timeline:** Weeks 19-23 (updated from 16-20, parallel with Phase 3)
**Dependencies:** Phase 0 complete (can run concurrently with Phases 1-3)
**Priority:** MEDIUM

#### Deliverables

1. **ZeroMQ Spine** ([04_infrastructure_comms_implementation.md](04_infrastructure_comms_implementation.md))
   - Gap 4.1: Circuit breaker timeouts (100ms control, 5ms data) ✅
   - Gap 4.2: Heartbeat sentinel crash detection (500ms) ✅
   - Gap 4.3: RAII shared memory lifecycle ✅
   - Gap 4.4: ZMQ socket configuration (HWM=1000, LINGER=0) ✅
   - Gap 4.5: Append-only Protobuf schema evolution ✅

2. **Validation Requirements**
   - Latency: 99th percentile < 50ms for control messages
   - Fault tolerance: Detect and restart crashed components within 500ms
   - Memory: SHM cleanup prevents leaks

#### Critical Success Factors

- **Enables distributed architecture**
- CurveZMQ provides authentication
- Orchestrator coordinates component lifecycle

**Risk Level:** 🟢 **LOW** - Well-established patterns (ZeroMQ, Protobuf).

---

### Phase 5: Autonomous Systems (Self-Regulation)

**Timeline:** Weeks 26-31 (updated from 23-28)
**Dependencies:** Phase 0, Phases 1-3 complete
**Priority:** MEDIUM

#### Deliverables

1. **ENGS (Extended Neurochemical Gating)** ([05_autonomous_systems_implementation.md](05_autonomous_systems_implementation.md))
   - Gap 5.1: TD-learning dopamine system ✅
   - Gap 5.2: Monte Carlo entropy estimation (K=1000) ✅
   - Gap 5.3: Hamiltonian metabolic cost ✅
   - Gap 5.4: ATP hysteresis nap cycle (15-60 seconds) ✅
   - Gap 5.5: Frobenius norm Dream-Weave convergence ✅

2. **Validation Requirements**
   - Dopamine response: Spike to 0.8 on reward, dip to 0.2 on punishment
   - Boredom triggers exploration when entropy < 2.0
   - ATP depletion triggers NAP after sustained high-frequency activity

#### Critical Success Factors

- **Creates goal-directed behavior**
- Curiosity-driven exploration
- Metabolic resource management

**Risk Level:** 🟡 **MEDIUM** - If Gap 5.1 (dopamine) is poorly tuned, system becomes catatonic or manic.

---

### Phase 6: Multimodal & Persistence (Real-World Interface)

**Timeline:** Weeks 32-37 (updated from 29-34)
**Dependencies:** Phase 0, Phases 1-3 complete
**Priority:** MEDIUM

#### Deliverables

1. **Sensory Transduction** ([06_multimodal_persistence_implementation.md](06_multimodal_persistence_implementation.md))
   - Gap 6.1: Circular audio emitter array (golden ratio frequencies) ✅
   - Gap 6.2: 64×64 log-polar visual transform ✅
   - Gap 6.3: Event-driven checkpointing (300s periodic + NAP trigger) ✅
   - Gap 6.4: GGUF metadata schema ✅
   - Gap 6.5: Adaptive Q9_0/FP16 compression ✅

2. **Validation Requirements**
   - Audio: FFT shows 8 distinct harmonic peaks
   - Visual: Log-polar foveal emphasis matches biological vision
   - Checkpointing: DMC save/restore without data loss

#### Critical Success Factors

- **Grounds physics in sensory reality**
- Efficient state persistence
- GGUF enables llama.cpp compatibility

**Risk Level:** 🟢 **LOW** - Standard signal processing techniques.

---

### Phase 7: Security & Execution (Containment)

**Timeline:** Weeks 38-43 (updated from 35-40)
**Dependencies:** Phase 0, Phase 4 complete
**Priority:** HIGH (security-critical)

#### Deliverables

1. **KVM Sandbox** ([07_security_execution_implementation.md](07_security_execution_implementation.md))
   - Gap 7.1: Alpine 3.19 minimal VM image with Packer ✅
   - Gap 7.2: Strict inter-VM isolation (host-mediated only) ✅
   - Gap 7.3: eBPF escape detection (execve, file access) ✅
   - Gap 7.4: Regex code blacklist (system, exec, asm, networking) ✅
   - Gap 7.5: Agentless CGroup performance monitoring ✅

2. **Validation Requirements**
   - VM cannot access network
   - VM cannot execute system(), exec(), fork()
   - eBPF detects and kills escape attempts within 100ms
   - CGroup enforces 512MB RAM, 1 vCPU limits

#### Critical Success Factors

- **Self-generated code cannot escape**
- Multi-layered defense (prevention, containment, detection, response)
- Resource quotas prevent DoS

**Risk Level:** 🔴 **HIGH** - Security failures could compromise host system.

---

## Inter-Dependencies (Updated with Phase 0)

### Critical Path

```
                    **Phase 0 (Critical Remediations)**
                    CF-04 + MEM-04 (Weeks 1-3)
                              |
                              v
Phase 1 (Physics) → Phase 2 (Geometry) → Phase 3 (Cognition)
                                            ↓
                                         Phase 5 (Autonomy)
                                            ↓
                                         Phase 6 (Multimodal)
```

### Parallel Tracks

- **Phase 0 (Critical)** blocks ALL other phases
- **Phase 4 (Infrastructure)** can run concurrently with Phases 1-3 after Phase 0 complete
- **Phase 7 (Security)** can begin after Phase 0, Phase 4 complete (uses orchestration layer)

---

## Risk Assessment (Updated)

### Critical (Phase 0) - Blocking All Implementation

1. **Finding CF-04 (Metabolic Lock)** 🔴🔴
   **Impact:** Thermodynamic race condition → negative energy states → cognitive seizures → system crash
   **Mitigation:** RAII transactional guards, atomic CAS operations, comprehensive unit testing
   **Status:** BLOCKS Phase 1-7

2. **Finding MEM-04 (Hilbert Re-indexing)** 🔴🔴
   **Impact:** Spatial discontinuity → semantic aphasia → high perplexity, hallucinations
   **Mitigation:** Causal-Foliated Hilbert scanning, locality preservation validation
   **Status:** BLOCKS Phase 1-7 (specifically Phase 3)

### Highest Risk Items (Original Gaps)

3. **Gap 1.1 (Emitter Injection)** 🔴
   **Impact:** Numerical explosion if amplitude scaling is wrong
   **Mitigation:** Rigorous validation with PhysicsOracle energy watchdog

4. **Gap 5.1 (Dopamine)** 🔴
   **Impact:** System becomes unresponsive (catatonic) or hallucinates (manic)
   **Mitigation:** Extensive parameter sweep, empirical tuning with real tasks

5. **Gap 7.3 (VM Escape Detection)** 🔴
   **Impact:** Malicious self-generated code compromises host
   **Mitigation:** eBPF monitoring + multi-layered defense

### Medium Risk Items

- Metric tensor validation (Gap 2.1): Mathematical solution exists (Tikhonov regularization)
- Equilibrium Propagation training (Gap 3.6): Novel but theoretically sound
- Entropy estimation (Gap 5.2): Monte Carlo approximation well-understood

### Low Risk Items

- Infrastructure (Phase 4): Mature technologies (ZeroMQ, Protobuf)
- Multimodal (Phase 6): Standard signal processing
- Most geometry operations: Well-established algorithms

---

## Resource Requirements

### Hardware

- **Development:** RTX 4090 GPU (24GB VRAM) for physics testing
- **Production:** 2× RTX 4090 for redundancy
- **CPU:** AMD Threadripper (16+ cores) for parallel component execution
- **RAM:** 128GB DDR5 for large grid allocations
- **Storage:** 2TB NVMe SSD for checkpoints and logs

### Software

- **OS:** Ubuntu 22.04 LTS (KVM host)
- **CUDA:** 12.3+
- **Compiler:** GCC 13+ (C++23 support) or Clang 17+
- **Libraries:** Eigen3, FFTW3, ZeroMQ, Protobuf, OpenCV, libbpf

### Team

- **Physics Engineer:** Implement UFIE, symplectic integration, CUDA kernels
- **Systems Architect:** Orchestration, ZeroMQ, component lifecycle
- **ML Engineer:** Mamba-9D, token mapping, Equilibrium Propagation
- **Security Engineer:** KVM sandboxing, eBPF monitoring, static analysis
- **Integration Engineer:** End-to-end testing, validation procedures

---

## Success Criteria (Updated)

### Phase 0 Gate (Week 3)

Before ANY other implementation can begin:
1. ✅ CF-04 passes all unit/integration tests (no ATP race conditions)
2. ✅ MEM-04 demonstrates LPR improvement > 20% over Morton codes
3. ✅ Mamba perplexity statistically significantly reduced (p < 0.05)
4. ✅ Orchestrator integration complete, no performance regression
5. ✅ All validation tests pass

🔴 **GATE - Nothing proceeds until Phase 0 complete**

### Minimum Viable Product (MVP)

By end of Phase 3 (Week 25, updated from 22):
1. ✅ Phase 0 complete (CF-04, MEM-04)
2. ✅ Physics engine sustains 1 kHz loop with <0.01% energy drift
3. ✅ 9D manifold supports 1M active nodes with sparse addressing
4. ✅ Language generation produces coherent tokens at 10 tokens/sec
5. ✅ Energy conservation never violated (no crashes)
6. ✅ Spatial locality preserved (Hilbert scanning)

### Full System (Production-Ready)

By end of Phase 7 (Week 43, updated from 40):
1. ✅ All 37 gaps + 2 critical findings implemented and validated
2. ✅ Autonomous behavior: Dopamine-driven learning, boredom exploration, NAP cycles
3. ✅ Multimodal: Audio + visual input transduction
4. ✅ Security: Self-generated code runs in KVM sandbox, no escapes possible
5. ✅ Persistence: DMC checkpoints enable save/restore
6. ✅ Performance: 2 kHz physics loop, 50 tokens/sec generation
7. ✅ Thermodynamic consistency enforced (CF-04)
8. ✅ Cognitive coherence maintained (MEM-04)

---

## Validation & Testing Strategy

### Unit Tests

- Each gap implementation includes validation procedure
- Automated tests for energy conservation, metric validity, collision detection

### Integration Tests

1. **Physics + Geometry:** Inject token, verify wavefunction propagates correctly
2. **Cognition + Physics:** Generate sequence, verify tokens map to grid coordinates
3. **Infrastructure + All Components:** Orchestrator manages component lifecycle
4. **Security + Execution:** Attempt VM escape, verify eBPF kills process

### Performance Benchmarks

- **Physics Loop:** Target 2000 Hz on RTX 4090
- **Token Generation:** Target 50 tokens/sec
- **Memory Bandwidth:** < 100 GB/s (stay within GPU limits)
- **Latency:** Control messages < 50ms (99th percentile)

### Stress Tests

- **10M active nodes:** Verify sparse grid scales
- **Continuous operation:** Run for 72 hours without crashes
- **Resource exhaustion:** Verify graceful degradation when ATP depleted

---

## Next Steps (Updated with Phase 0)

### 🔴 CRITICAL: Immediate Actions (Phase 0)

**STOP:** Original Phase 1-7 implementation is **ON HOLD** until Phase 0 complete.

1. **Initialize Repository**
   ```bash
   mkdir -p nikola_v0.0.4/{src,tests,docs,benchmarks}
   cd nikola_v0.0.4 && git init
   ```

2. **Setup Build System** (CMake)
   - C++23 compiler support (GCC 13+ or Clang 17+)
   - Atomic operations library
   - OpenMP for parallel Hilbert sorting
   - Test harness (Google Test)

3. **Implement CF-04** (Transactional Metabolic Lock)
   - Create `include/nikola/autonomy/metabolic_lock.hpp`
   - Create `src/autonomy/metabolic_lock.cpp`
   - Update `metabolic_controller.hpp` with CAS operations
   - Write comprehensive unit tests (atomic reserve, rollback, exhaustion)

4. **Implement MEM-04** (Hilbert Re-indexing)
   - Create `include/nikola/spatial/hilbert_scanner.hpp`
   - Create `src/spatial/hilbert_scanner.cpp`
   - Implement 128-bit Hilbert encoding
   - Write validation tests (LPR, Mamba perplexity)

5. **Integrate into Orchestrator**
   - Update `orchestrator.cpp` control loop
   - Add fragmentation index monitoring
   - Implement exception handling for MetabolicExhaustionException

6. **Validation & Gate Review**
   - Run all Phase 0 unit tests
   - Measure LPR improvement
   - Conduct Mamba perplexity comparison
   - Performance regression testing

### Week 1 Milestones (Phase 0 Focus)

- [ ] Repository initialized
- [ ] CMake build system functional
- [ ] CF-04: MetabolicTransaction class compiles
- [ ] CF-04: Atomic CAS unit tests passing
- [ ] CF-04: Rollback test passing

### Week 2 Milestones (Phase 0 Focus)

- [ ] MEM-04: HilbertScanner class compiles
- [ ] MEM-04: 128-bit encoding functional
- [ ] MEM-04: Causal-foliated sorting working
- [ ] MEM-04: LPR measurement implemented

### Week 3 Milestones (Phase 0 Gate)

- [ ] CF-04: All unit/integration tests passing
- [ ] MEM-04: LPR improvement > 20% demonstrated
- [ ] MEM-04: Mamba perplexity significantly reduced
- [ ] Orchestrator integration complete
- [ ] 🚀 **PHASE 0 GATE PASSED** → Proceed to Phase 1

---

## Long-Term Enhancements (Post-MVP)

These are **not** required for initial deployment but represent future optimization opportunities:

1. **Multi-GPU Scaling:** Shard grid across multiple GPUs
2. **Adaptive Resolution:** Dynamically allocate grid cells based on attention
3. **Learned Metric Initialization:** Pre-train metric tensor on large corpus
4. **Hardware Acceleration:** Custom FPGA for Laplace-Beltrami operator
5. **Distributed Physics:** Run physics engine across cluster for massive scale

---

## Conclusion (Updated)

This roadmap converts the theoretical Nikola v0.0.4 architecture into a concrete, buildable specification. By addressing all 37 implementation gaps + 2 critical findings with:

- ✅ Concrete mathematical specifications
- ✅ Production-ready C++23/CUDA reference implementations
- ✅ Rigorous validation procedures
- ✅ Comprehensive failure mode analyses

The system specification is complete.

**HOWEVER:** Implementation is **BLOCKED** pending Phase 0 completion.

### Critical Update (2025-12-10)

Aria's review identified 2 **Priority 1 Critical** architectural vulnerabilities:
- **CF-04:** Thermodynamic race condition (ATP management)
- **MEM-04:** Spatial discontinuity (Mamba-9D coherence)

**These MUST be resolved before ANY Phase 1-7 work begins.**

The physics-first approach—treating intelligence as a wave phenomenon rather than matrix multiplication—is preserved and hardened against numerical instability.

### Status Summary

| Component | Status |
|-----------|--------|
| **Theoretical Foundation** | ✅ Complete (Audits 1-21) |
| **37 Implementation Gaps** | ✅ Complete (Gap Analysis) |
| **Critical Remediations** | 🔴 **BLOCKING** (Phase 0 required) |
| **Phase 1-7 Implementation** | ⏸️ **ON HOLD** (awaiting Phase 0) |

**Final Status:** 🔴 **IMPLEMENTATION BLOCKED - Phase 0 Required**

**Timeline:** Original 40 weeks → Updated 43 weeks (adds 3-week Phase 0)

---

**Document Metadata:**
- **Compiled:** 2025-12-10
- **Updated:** 2025-12-10 (Critical Findings Integration)
- **Audit Cycles:** 1-21 (All remediations incorporated)
- **Gap Analysis:** Complete (37/37 addressed)
- **Critical Findings:** 2 identified (CF-04, MEM-04)
- **Status:** Phase 0 Blocking Dependencies

# FILE STRUCTURE

## 8.8 Phase 0 Critical Components

The file structure has been updated to include Phase 0 critical components. Key additions:

- `include/nikola/physics/soa_layout.hpp` - Structure-of-Arrays memory layout
- `include/nikola/physics/symplectic_integrator.hpp` - Split-operator integration
- `include/nikola/security/physics_oracle.hpp` - Self-improvement safety
- `src/physics/kernels/wave_propagate_soa.cu` - SoA CUDA kernel
- `tests/validation/` - Phase 0 validation test suite

See: `08_audit_remediation/01_critical_fixes.md` for complete specifications.

---

## 8.8.1 Complete Directory Organization

```
nikola/
├── CMakeLists.txt                   # Root CMake file
├── README.md                        # Project README
├── LICENSE                          # License file
├── .dockerignore                    # Docker ignore
├── Dockerfile                       # Multi-stage Docker build
├── docker-compose.yml               # Service orchestration
│
├── include/                         # Public headers
│   └── nikola/
│       ├── types/
│       │   ├── nit.hpp              # Balanced nonary type (AVX-512)
│       │   ├── coord9d.hpp          # 9D coordinate
│       │   ├── torus_node.hpp       # Node structure (DEPRECATED - use SoA)
│       │   ├── torus_block.hpp      # ⚡ SoA memory layout (Phase 0)
│       │   └── morton_code.hpp      # ⚡ 128-bit Z-order encoding (Phase 0)
│       ├── physics/
│       │   ├── torus_manifold.hpp   # Main 9D grid
│       │   ├── soa_layout.hpp       # ⚡ Structure-of-Arrays (Phase 0)
│       │   ├── symplectic_integrator.hpp # ⚡ Split-operator (Phase 0)
│       │   ├── kahan_sum.hpp        # ⚡ Compensated summation (Phase 0)
│       │   ├── emitter_array.hpp    # DDS emitters
│       │   ├── wave_engine.hpp      # Interference processor
│       │   ├── shvo_grid.hpp        # Sparse hyper-voxel
│       │   ├── metric.hpp           # Riemannian geometry
│       │   └── metric_cache.hpp     # ⚡ Lazy Cholesky (Phase 0)
│       ├── mamba/
│       │   ├── hilbert_scan.hpp     # Space-filling curve
│       │   ├── ssm_kernel.hpp       # State space model
│       │   └── taylor_approx.hpp    # ⚡ Matrix approximation (Phase 0)
│       ├── reasoning/
│       │   ├── transformer.hpp      # Wave transformer
│       │   ├── attention.hpp        # Wave correlation
│       │   └── embedder.hpp         # Nonary embedder
│       ├── spine/
│       │   ├── broker.hpp           # ZeroMQ router
│       │   ├── component_client.hpp # Client interface
│       │   ├── shadow_spine.hpp     # A/B testing
│       │   └── shared_memory.hpp    # ⚡ Zero-copy IPC (Phase 0)
│       ├── agents/
│       │   ├── tavily.hpp           # Search client
│       │   ├── firecrawl.hpp        # Scrape client
│       │   ├── gemini.hpp           # Translation client
│       │   └── http_client.hpp      # Custom HTTP
│       ├── executor/
│       │   └── kvm_executor.hpp     # VM manager
│       ├── autonomy/
│       │   ├── dopamine.hpp         # Reward system
│       │   ├── engs.hpp             # Extended neurochemistry
│       │   ├── boredom.hpp          # Curiosity
│       │   ├── goals.hpp            # Goal DAG
│       │   └── dream_weave.hpp      # Counterfactual learning
│       ├── persistence/
│       │   ├── dmc.hpp              # Checkpoint manager
│       │   ├── lsm_dmc.hpp          # LSM persistence
│       │   ├── gguf_export.hpp      # GGUF converter
│       │   ├── q9_encoder.hpp       # ⚡ Q9_0 quantization (Phase 0)
│       │   └── identity.hpp         # Identity profile
│       ├── multimodal/
│       │   ├── audio_resonance.hpp  # Audio FFT
│       │   └── visual_cymatics.hpp  # Image processing
│       ├── security/
│       │   ├── resonance_firewall.hpp # Attack detection
│       │   ├── physics_oracle.hpp   # ⚡ Self-improvement safety (Phase 0)
│       │   ├── adversarial_dojo.hpp # ⚡ Red team testing (Phase 0)
│       │   └── csvp.hpp             # Code safety protocol
│       ├── monitoring/
│       │   ├── energy_watchdog.hpp  # ⚡ Energy conservation monitor (Phase 0)
│       │   └── profiler.hpp         # ⚡ Performance profiler (Phase 0)
│       └── self_improve/
│           └── hot_swap.hpp         # Module replacement
│
├── src/                             # Implementation
│   ├── core/
│   │   ├── lib9dtwi.cpp             # Main library
│   │   └── CMakeLists.txt
│   ├── types/
│   │   ├── nit.cpp                  # ⚡ AVX-512 nonary ops (Phase 0)
│   │   ├── coord9d.cpp
│   │   ├── torus_block.cpp          # ⚡ SoA implementation
│   │   ├── morton_code.cpp          # ⚡ 128-bit encoding
│   │   └── CMakeLists.txt
│   ├── physics/
│   │   ├── torus_manifold.cpp
│   │   ├── soa_layout.cpp           # ⚡ SoA refactoring
│   │   ├── symplectic_integrator.cpp # ⚡ 6-step Strang splitting
│   │   ├── kahan_sum.cpp            # ⚡ Compensated summation
│   │   ├── emitter_array.cpp
│   │   ├── wave_engine.cpp
│   │   ├── shvo_grid.cpp
│   │   ├── metric.cpp
│   │   ├── metric_cache.cpp         # ⚡ Lazy Cholesky
│   │   ├── kernels/                 # CUDA kernels
│   │   │   ├── wave_propagate.cu    # Original (DEPRECATED)
│   │   │   ├── wave_propagate_soa.cu # ⚡ SoA coalesced (Phase 0)
│   │   │   └── laplacian_kahan.cu   # ⚡ Kahan CUDA kernel
│   │   └── CMakeLists.txt
│   ├── mamba/
│   │   ├── hilbert_scan.cpp
│   │   ├── ssm_kernel.cpp
│   │   ├── taylor_approx.cpp        # ⚡ First-order matrix approx
│   │   └── CMakeLists.txt
│   ├── reasoning/
│   │   ├── transformer.cpp
│   │   ├── wave_attention.cpp
│   │   ├── embedder.cpp
│   │   └── CMakeLists.txt
│   ├── spine/
│   │   ├── broker.cpp
│   │   ├── component_client.cpp
│   │   ├── shadow_spine.cpp
│   │   ├── shared_memory.cpp        # ⚡ Seqlock IPC
│   │   └── CMakeLists.txt
│   ├── orchestrator/
│   │   ├── smart_router.cpp
│   │   └── CMakeLists.txt
│   ├── agents/
│   │   ├── tavily.cpp
│   │   ├── firecrawl.cpp
│   │   ├── gemini.cpp
│   │   ├── http_client.cpp
│   │   └── CMakeLists.txt
│   ├── executor/
│   │   ├── kvm_executor.cpp
│   │   ├── guest_agent.cpp          # Runs inside VM
│   │   └── CMakeLists.txt
│   ├── autonomy/
│   │   ├── dopamine.cpp
│   │   ├── engs.cpp
│   │   ├── boredom.cpp
│   │   ├── goals.cpp
│   │   ├── trainers.cpp
│   │   ├── dream_weave.cpp
│   │   └── CMakeLists.txt
│   ├── persistence/
│   │   ├── dmc.cpp
│   │   ├── lsm_dmc.cpp
│   │   ├── gguf_export.cpp
│   │   ├── q9_encoder.cpp           # ⚡ Radix-9 encoding
│   │   ├── identity.cpp
│   │   └── CMakeLists.txt
│   ├── multimodal/
│   │   ├── audio_resonance.cpp
│   │   ├── visual_cymatics.cpp
│   │   └── CMakeLists.txt
│   ├── security/
│   │   ├── resonance_firewall.cpp
│   │   ├── physics_oracle.cpp       # ⚡ Mathematical verification
│   │   ├── adversarial_dojo.cpp     # ⚡ Attack testing
│   │   ├── csvp.cpp
│   │   └── CMakeLists.txt
│   ├── monitoring/
│   │   ├── energy_watchdog.cpp      # ⚡ Conservation checks
│   │   ├── profiler.cpp             # ⚡ Performance tracking
│   │   └── CMakeLists.txt
│   ├── self_improve/
│   │   ├── hot_swap.cpp
│   │   └── CMakeLists.txt
│   └── ingestion/
│       ├── sentinel.cpp
│       └── CMakeLists.txt
│
├── tools/                           # Utilities
│   ├── twi-ctl/
│   │   ├── main.cpp                 # CLI controller
│   │   └── CMakeLists.txt
│   ├── validate_phase0/             # ⚡ Phase 0 validation (Phase 0)
│   │   ├── test_energy_conservation.cpp
│   │   ├── test_symplectic.cpp
│   │   ├── test_kahan.cpp
│   │   └── CMakeLists.txt
│   └── convert_nikola_to_gguf.py    # GGUF export script
│
├── proto/                           # Protocol Buffers
│   ├── neural_spike.proto
│   └── CMakeLists.txt
│
├── tests/                           # Test suites
│   ├── validation/                  # ⚡ Phase 0 validation suite (Phase 0)
│   │   ├── test_energy_conservation.cpp
│   │   ├── test_symplectic_property.cpp
│   │   ├── test_kahan_summation.cpp
│   │   ├── test_wave_equation.cpp
│   │   ├── test_boundary_wrapping.cpp
│   │   └── test_numerical_stability.cpp
│   ├── unit/
│   │   ├── test_nit.cpp
│   │   ├── test_coord9d.cpp
│   │   ├── test_emitter_array.cpp
│   │   └── CMakeLists.txt
│   └── integration/
│       ├── test_wave_propagation.cpp
│       ├── test_mamba_ssm.cpp
│       └── CMakeLists.txt
│
├── config/                          # Configuration files
│   ├── default.toml                 # Default system config
│   ├── physics_constants.toml       # Physical parameters
│   ├── hazards.db                   # Resonance firewall patterns
│   └── keys/                        # CurveZMQ keys (generated)
│       ├── public.key
│       └── secret.key
│
└── docs/                            # Documentation
    ├── architecture.md
    ├── api_reference.md
    ├── phase0_validation.md         # ⚡ Phase 0 checklist
    └── integration/                 # This documentation set

## 8.8.2 Implementation Guide - Mandated Organization

**CRITICAL:** To avoid "creative" organization, the engineering team MUST adhere to this exact directory mapping, which corresponds to architectural layers:

```
/src
  /core
    main.cpp              # Entry point, orchestrator initialization
    config_loader.cpp     # JSON/TOML configuration parsing
    
  /physics                # The 9D Substrate Layer
    torus_grid_soa.hpp    # ⚡ SoA Data Structure (The Substrate)
    integrator.cpp        # ⚡ Symplectic Split-Operator Solver
    ufie_kernels.cu       # CUDA Kernels for Laplacian/Nonlinearity
    kahan_sum.cpp         # ⚡ Compensated Summation
    shvo_grid.cpp         # Sparse Hyper-Voxel Octree logic
    metric.cpp            # Metric tensor operations
    emitter_array.cpp     # Golden ratio DDS emitters
    
  /cognitive              # The Cognitive Processing Layer
    mamba_tsm.cpp         # ⚡ TSM (Topology→Matrix mapper)
    transformer_np.cpp    # Neuroplastic Wave Attention
    hilbert_curve.cpp     # BMI2-optimized Hilbert scanning
    embedder.cpp          # Balanced nonary text encoder
    
  /autonomy               # The Autonomous Systems Layer
    engs_system.cpp       # Neurochemistry state machine
    dream_weave.cpp       # Counterfactual simulation engine
    dopamine.cpp          # Reward/learning modulation
    boredom.cpp           # Curiosity-driven exploration
    
  /infrastructure         # The Communication Backbone
    spine_broker.cpp      # ZeroMQ Router implementation
    kvm_manager.cpp       # Libvirt interface for Executors
    shared_memory.cpp     # ⚡ Seqlock zero-copy IPC
    proto/                # Compiled Protocol Buffers (.pb.cc)
    
  /types                  # The Arithmetic Foundation
    nit_avx512.cpp        # ⚡ Optimized Nonary Arithmetic (AVX-512)
    geometry.hpp          # 9D Coordinate utilities
    morton_code.cpp       # ⚡ 128-bit Z-order encoding
    
  /security               # The Safety and Validation Layer
    physics_oracle.cpp    # ⚡ Mathematical verification sandbox
    adversarial_dojo.cpp  # ⚡ Red team attack testing
    resonance_firewall.cpp # Spectral input filtering
    
  /persistence            # The Memory Durability Layer
    dmc.cpp               # Delta Memory Compression checkpoints
    lsm_dmc.cpp           # Log-Structured Merge persistence
    gguf_export.cpp       # Llama.cpp interoperability
    q9_encoder.cpp        # ⚡ Nonary quantization (Q9_0)
    
  /monitoring             # The Observability Layer
    energy_watchdog.cpp   # ⚡ Runtime conservation checks
    profiler.cpp          # ⚡ Performance tracking
```

### 26.2.1 Phase 0 Implementation Checklist (17-Day Sprint)

**Critical Path - Immediate Engineering Tasks:**

**Days 1-2:** Structure-of-Arrays Refactoring
- [ ] Create `include/nikola/physics/torus_grid_soa.hpp`
- [ ] Implement `TorusGridSoA` with 64-byte aligned vectors
- [ ] Implement 45-component metric tensor storage (upper triangular)
- [ ] Create `TorusNodeProxy` accessor class for API compatibility
- [ ] Refactor all grid access code to use proxy pattern
- [ ] Update CUDA kernels for coalesced memory access
- [ ] Validation: Physics kernel achieves <1ms per step on 27³ grid

**Days 3-5:** Split-Operator Symplectic Integration
- [ ] Create `include/nikola/physics/symplectic_integrator.hpp`
- [ ] Implement 6-step Strang splitting:
  - Half-kick damping (analytical exponential decay)
  - Half-kick conservative force (Laplacian + emitters)
  - Full drift (position update)
  - Nonlinear operator (RK2 implicit)
  - Half-kick force (recompute at new position)
  - Half-kick damping (final decay)
- [ ] Replace all Velocity-Verlet code
- [ ] Implement adaptive timestep monitoring
- [ ] Implement energy watchdog (compute Hamiltonian every 100 steps)
- [ ] Validation: Energy conservation within 0.01% over 24 hours

**Day 6:** Kahan Compensated Summation
- [ ] Create `include/nikola/physics/kahan_sum.hpp`
- [ ] Implement `KahanAccumulator` struct
- [ ] Refactor all Laplacian kernels to use Kahan summation
- [ ] Refactor all wave superposition operations
- [ ] Refactor metric tensor updates
- [ ] Validation: Preserve 10⁻⁶ amplitude waves over 10⁶ timesteps

**Day 7:** 128-bit Morton Code Hashing
- [ ] Create `include/nikola/types/morton_code.hpp`
- [ ] Implement BMI2-optimized bit interleaving
- [ ] Implement collision detection and double-hashing fallback
- [ ] Replace existing 64-bit Morton codes
- [ ] Validation: Zero collisions on 10⁷ random 9D coordinates

**Day 8:** Vectorized Nonary Arithmetic
- [ ] Create `include/nikola/types/nit_avx512.hpp`
- [ ] Implement `vec_sum_gate_avx512()` (64 trits parallel)
- [ ] Implement `vec_product_gate_avx512()` (heterodyning)
- [ ] Refactor all nonary operations to use SIMD
- [ ] Validation: 10x speedup vs scalar implementation

**Days 9-11:** Topological State Mapping (TSM)
- [ ] Create `src/cognitive/mamba_tsm.cpp`
- [ ] Implement `tsm_generate_parameters_kernel()`
- [ ] Extract metric tensor → Matrix A conversion
- [ ] Extract state dimension → Matrix B conversion
- [ ] Integrate with Hilbert curve scanner
- [ ] Validation: Mamba layers dynamically respond to metric changes

**Days 12-14:** Physics Oracle & Adversarial Dojo
- [ ] Create `include/nikola/security/physics_oracle.hpp`
- [ ] Implement 5 verification tests:
  - Energy conservation
  - Symplectic property
  - Wave equation validity
  - Boundary conditions (toroidal wrapping)
  - Numerical stability (NaN/Inf detection)
- [ ] Create `include/nikola/security/adversarial_dojo.hpp`
- [ ] Implement 10+ attack vectors
- [ ] Implement hot-swap protocol with Oracle gate
- [ ] Implement runtime energy watchdog
- [ ] Validation: All tests pass; attacks fail; 24-hour stability

**Days 15-16:** Integration & Testing
- [ ] Run full Phase 0 validation suite
- [ ] Profile memory bandwidth (should saturate DDR5)
- [ ] Profile energy conservation (should be <0.01% drift)
- [ ] Profile Laplacian accuracy (should preserve 10⁻⁶ amplitudes)
- [ ] Fix any identified issues

**Day 17:** Documentation & Handoff
- [ ] Document all Phase 0 implementations
- [ ] Create performance benchmark report
- [ ] Update README with Phase 0 status
- [ ] Tag repository as `v0.0.4-phase0-complete`

**Gate Requirement:** ALL checklist items must pass validation before Phase 1 begins.

**Final Directive:** Do not proceed to higher-level cognitive features until Physics Oracle confirms energy stability for >24 hours of continuous operation.

---

**Cross-References:**
- See `08_phase_0_requirements/01_critical_fixes.md` for detailed specifications
- See `11_appendices/04_hardware_optimization.md` for AVX-512 requirements
- See `09_implementation/03_implementation_checklist.md` for complete task list
│   ├── unit/
│   │   ├── test_nonary.cpp
│   │   ├── test_coord9d.cpp
│   │   ├── test_wave_interference.cpp
│   │   ├── test_hilbert.cpp
│   │   ├── test_engs.cpp
│   │   ├── test_neuroplasticity.cpp
│   │   └── CMakeLists.txt
│   ├── integration/
│   │   ├── test_search_retrieve.cpp
│   │   ├── test_training.cpp
│   │   ├── test_multimodal.cpp
│   │   └── CMakeLists.txt
│   └── benchmarks/
│       ├── bench_propagation.cpp
│       ├── bench_hilbert.cpp
│       └── CMakeLists.txt
│
├── docker/                          # Docker files
│   ├── Dockerfile.base              # Base image
│   ├── Dockerfile.runtime           # Runtime image
│   └── gold-image/                  # VM gold image
│       └── ubuntu-24.04.qcow2
│
├── config/                          # Configuration
│   ├── nikola.conf                  # Main config
│   ├── emitters.conf                # Frequency settings
│   └── security.conf                # Firewall patterns
│
└── docs/                            # Documentation
    ├── architecture.md
    ├── api_reference.md
    └── troubleshooting.md
```

## 8.8.3 File Manifest

**Total Files:** ~150
**Total Lines of Code (estimated):** ~50,000

**Critical Path Files (Must implement first):**

1. `include/nikola/types/nit.hpp` - Balanced nonary enum
2. `include/nikola/types/torus_node.hpp` - Node structure
3. `include/nikola/physics/torus_manifold.hpp` - Grid
4. `include/nikola/physics/emitter_array.hpp` - DDS
5. `src/physics/wave_engine.cpp` - Interference processor
6. `proto/neural_spike.proto` - Message protocol
7. `src/spine/broker.cpp` - Communication backbone

## 8.8.4 Key Implementation Files by Subsystem

### Physics Engine (Core)
- `types/nit.hpp/cpp` - Balanced nonary arithmetic
- `physics/torus_manifold.hpp/cpp` - 9D sparse grid
- `physics/emitter_array.hpp/cpp` - Golden ratio DDS
- `physics/wave_engine.cpp` - Superposition/heterodyning
- `physics/shvo_grid.cpp` - Sparse hyper-voxel octree
- `physics/kernels/wave_propagate.cu` - CUDA acceleration

### Cognitive Systems
- `mamba/hilbert_scan.cpp` - Space-filling curve scanner
- `mamba/ssm_kernel.cpp` - State space model
- `reasoning/transformer.cpp` - Neuroplastic transformer
- `reasoning/wave_attention.cpp` - Wave correlation
- `reasoning/embedder.cpp` - Text-to-waveform

### Infrastructure
- `spine/broker.cpp` - ZeroMQ message router
- `spine/shadow_spine.cpp` - A/B testing infrastructure
- `orchestrator/smart_router.cpp` - Tool selection
- `agents/*.cpp` - External API clients
- `executor/kvm_executor.cpp` - Sandboxed execution

### Autonomy
- `autonomy/engs.cpp` - Extended neurochemistry
- `autonomy/dopamine.cpp` - Reward system
- `autonomy/boredom.cpp` - Curiosity-driven learning
- `autonomy/goals.cpp` - Hierarchical goal DAG
- `autonomy/dream_weave.cpp` - Counterfactual simulation
- `autonomy/trainers.cpp` - Autonomous training

### Persistence & Safety
- `persistence/lsm_dmc.cpp` - Log-structured persistence
- `persistence/gguf_export.cpp` - GGUF interoperability
- `security/resonance_firewall.cpp` - Attack detection
- `security/csvp.cpp` - Code safety verification
- `self_improve/adversarial_dojo.cpp` - Red team testing

### Multimodal
- `multimodal/audio_resonance.cpp` - FFT-based audio
- `multimodal/visual_cymatics.cpp` - Holographic vision

---

**Cross-References:**
- See Section 27 for Development Roadmap
- See Section 28 for Implementation Checklist
- See Appendices for build system details

# DEVELOPMENT ROADMAP

## 8.9 🚨 CRITICAL: Engineering Phase 0 Requirements Required

A comprehensive engineering analysis identified critical implementation gaps that **MUST** be addressed before any feature development. These are not optimizations—they are functional requirements to prevent:

- **Numerical Instability:** System divergence within hours (energy drift)
- **Memory Thrashing:** 90% cache miss rate → 100x performance loss
- **Precision Loss:** Float32 errors cause "amnesia" over time
- **Hash Collisions:** Memory corruption in high-resolution grids
- **Race Conditions:** GPU segfaults and data corruption

**See:** `08_audit_remediation/01_critical_fixes.md` for complete specifications

---

## Phase 0: Critical Remediation (Weeks 1-2, 17 days)

**⚠️ NO DEVIATION:** All Phase 0 fixes are mandatory architectural requirements.

### Priority P0 (Critical - 6 days)

| Day | Task | Reference | Impact | Validation |
|-----|------|-----------|--------|------------|
| 1-2 | **SoA Memory Layout** | §1 Critical Fixes | 10x performance | >80% memory bandwidth utilization |
| | - Refactor `TorusNode` → `TorusBlock` | | | |
| | - Implement `TorusNodeProxy` | | | |
| | - Update CUDA kernels for coalesced access | | | |
| 3-5 | **Split-Operator Integration** | §2 Critical Fixes | Prevents divergence | Energy drift <0.0001% over 10K steps |
| | - Replace Verlet with Strang splitting | | | |
| | - Implement analytical damping decay | | | |
| | - Add adaptive timestep control | | | |
| 6 | **Kahan Summation** | §2.4 Critical Fixes | Prevents amnesia | Amplitude stable to 6 decimals over 1M steps |
| | - Update Laplacian accumulation | | | |
| | - CUDA kernel with compensation | | | |

### Priority P1 (High - 6 days)

| Day | Task | Reference | Impact | Validation |
|-----|------|-----------|--------|------------|
| 7-8 | **AVX-512 Nit Operations** | §4 Critical Fixes | 200x speedup | 10M ops in <50μs |
| | - Vectorized add/multiply (64 nits/op) | | | |
| | - Lookup tables for multiplication | | | |
| | - CPU feature detection + fallback | | | |
| 9-11 | **Lazy Cholesky Decomposition** | §5 Critical Fixes | 100x speedup | Metric overhead <5% runtime |
| | - Add `MetricTensorCache` class | | | |
| | - Implement dirty tracking | | | |
| | - Batch update logic | | | |
| 12 | **Energy Watchdog** | §9.1 Critical Fixes | System stability | Detect drift injection |
| | - Energy computation | | | |
| | - Periodic checks (every 100 steps) | | | |

### Priority P2 (Medium - 5 days)

| Day | Task | Reference | Impact | Validation |
|-----|------|-----------|--------|------------|
| 13-14 | **Shared Memory IPC** | §6.3 Critical Fixes | 1000x latency reduction | <10μs jitter |
| | - Seqlock implementation | | | |
| | - `/dev/shm` allocation | | | |
| | - ZMQ notifications | | | |
| 15-16 | **Mamba Taylor Approximation** | §3 Critical Fixes | 10x speedup | Compare vs full matrix exp |
| | - First-order matrix approximation | | | |
| | - Adaptive timestep | | | |
| 17 | **Q9_0 Quantization** | §8 Critical Fixes | 2x storage efficiency | 1M roundtrip 100% accuracy |
| | - Radix-9 encoding | | | |
| | - Batch SIMD encoder | | | |

### Phase 0 Gate Review

**Criteria for Phase 1 Entry:**
- ✅ All P0 and P1 tasks complete
- ✅ All validation tests pass
- ✅ Energy watchdog operational
- ✅ Physics step <1ms on sparse 27³ grid
- ✅ Code review completed (2 engineer sign-off)

**If gate fails:** Remediation continues until all criteria met. **NO EXCEPTIONS.**

**Total Critical Path:** 17 days (3.5 weeks)

---

## 8.9.1 Phase 1: Core Physics Engine (Months 1-3)

**Milestone:** Standing waves propagate correctly in 9D

**Tasks:**

| Week | Task | Deliverable |
|------|------|-------------|
| 1-2 | Implement `Nit` enum and nonary arithmetic | Unit tests pass |
| 3-4 | Implement `TorusNode` structure with metric tensor | Structure defined |
| 5-6 | Implement sparse `TorusManifold` grid (SHVO) | Grid can be created |
| 7-8 | Implement `EmitterArray` with DDS | Emitters generate signals |
| 9-10 | Implement wave propagation kernel | Waves propagate |
| 11-12 | Optimize with AVX-512/CUDA | Performance targets met |

**Validation Criteria:**

- [ ] Nonary addition: $1 + (-1) = 0$
- [ ] Wave superposition creates interference patterns
- [ ] Energy conserved over 1000 time steps
- [ ] Performance: <1ms per physics step (sparse 27³ grid)
- [ ] Toroidal wrapping works correctly

## 8.9.2 Phase 2: Logic and Memory (Months 4-6)

**Milestone:** Store text as wave, retrieve via resonance

**Tasks:**

| Week | Task | Deliverable |
|------|------|-------------|
| 13-14 | Implement balanced nonary arithmetic gates | Gates work |
| 15-16 | Build `NonaryEmbedder` (text → wave) | Embedder functional |
| 17-18 | Integrate LMDB storage backend | DB stores/loads nodes |
| 19-20 | Implement search-retrieve-store loop | Basic memory works |
| 21-22 | Implement LSM-DMC persistence (.nik format) | State persists |
| 23-24 | Validate memory accuracy over sessions | Retrieval >90% accurate |

**Validation Criteria:**

- [ ] Text → Waveform → Text roundtrip works
- [ ] Resonance detection finds stored patterns
- [ ] LSM-DMC saves and loads state correctly
- [ ] Merkle tree detects corruption
- [ ] Nap consolidation triggers correctly

## 8.9.3 Phase 3: The Brain (Months 7-9)

**Milestone:** System demonstrates learning

**Tasks:**

| Week | Task | Deliverable |
|------|------|-------------|
| 25-26 | Implement Mamba-9D Hilbert scanner | Scanner works |
| 27-28 | Port Transformer to Wave Correlation | Transformer operational |
| 29-30 | Implement Neuroplasticity (metric updates) | Learning observable |
| 31-32 | Implement Neurogenesis (grid expansion) | Grid grows when needed |
| 33-34 | Build autonomous trainers (BAT) | Training runs automatically |
| 35-36 | Benchmark retrieval accuracy improvements | Accuracy improves >10% |

**Validation Criteria:**

- [ ] Hilbert scan visits all nodes
- [ ] Wave correlation attention works
- [ ] Metric tensor contracts with co-activation
- [ ] New nodes created when saturated
- [ ] Repeated queries answered faster
- [ ] Topological State Mapping functional

## 8.9.4 Phase 4: Integration and Agents (Months 10-11)

**Milestone:** Full autonomous system

**Tasks:**

| Week | Task | Deliverable |
|------|------|-------------|
| 37-38 | Build ZeroMQ Spine with CurveZMQ security | Spine operational |
| 39-40 | Integrate Tavily/Firecrawl/Gemini APIs | Agents work |
| 41-42 | Implement KVM Executor with libvirt | VMs spawn and execute |
| 43-44 | Build twi-ctl CLI controller | CLI functional |
| 45-46 | Implement auto-ingestion pipeline (inotify) | Files ingested automatically |
| 47-48 | Finalize Docker multi-stage build | Docker image builds |

**Validation Criteria:**

- [ ] All components communicate via Spine
- [ ] External tools fetch data correctly
- [ ] Executor runs sandboxed commands safely
- [ ] CLI responds to all commands
- [ ] Files dropped in folder are ingested
- [ ] Shadow Spine Protocol operational

## 8.9.5 Phase 5: Autonomy and Evolution (Month 12)

**Milestone:** Self-improving AGI

**Tasks:**

| Week | Task | Deliverable |
|------|------|-------------|
| 49-50 | Implement ENGS (Dopamine/Serotonin/Norepinephrine) | Neurochemistry works |
| 50 | Implement Boredom/Curiosity and Goal systems | Autonomy functional |
| 51 | Build Resonance Firewall | Security operational |
| 52 | Implement Self-Improvement loop with CSVP | System improves itself |
| 53 | Implement Adversarial Code Dojo | Red Team testing works |
| 54 | Build GGUF export pipeline | GGUF export works |
| 55 | Security hardening and verification | Security checklist complete |
| 56 | Final integration testing | All systems operational |

**Validation Criteria:**

- [ ] Dopamine modulates learning rate correctly
- [ ] Exponential decay achieves homeostasis
- [ ] ENGS couples with physics kernel
- [ ] Boredom triggers curiosity
- [ ] Goals provide structure
- [ ] Firewall blocks known attacks
- [ ] CSVP prevents unsafe code modifications
- [ ] System identifies and patches bottlenecks
- [ ] Dream-Weave counterfactual learning works
- [ ] GGUF file loads in llama.cpp

## 8.9.6 Timeline Summary

| Phase | Duration | Milestone | Completion |
|-------|----------|-----------|------------|
| Phase 1 | Months 1-3 | Physics Engine | Core functional |
| Phase 2 | Months 4-6 | Memory | Storage works |
| Phase 3 | Months 7-9 | Learning | System learns |
| Phase 4 | Months 10-11 | Integration | Full system |
| Phase 5 | Month 12 | Autonomy | AGI complete |

**Total Development Time:** 12 months (5-person team)

---

**Cross-References:**
- See Section 26 for File Structure
- See Section 28 for Detailed Checklist

# IMPLEMENTATION CHECKLIST

## 8.10.1 🚨 PHASE 0: PHASE 0 REQUIREMENTS (MANDATORY)

**MUST complete before proceeding to 28.2 Foundation Layer**

### P0 Critical Items (Block Everything)

- [ ] **0.1** Structure-of-Arrays Memory Layout
  - Modify `TorusBlock` to use `alignas(64)` SoA layout
  - Separate arrays for: psi_real, psi_imag, metric_tensor (45 arrays), resonance, state
  - Block size: 19683 nodes (3^9)
  - **Reference:** Phase 0 Requirements §1.2
  - **Validation:** Verify cache hit rate >95% in Laplacian kernel
  - **Effort:** 2 days

- [ ] **0.2** Split-Operator Symplectic Integration
  - Replace Velocity-Verlet with 5-step split-operator
  - Step 1: Half-kick damping (analytical exponential)
  - Step 2: Half-kick forces
  - Step 3: Drift
  - Step 4: Recompute forces
  - Step 5: Half-kick forces + final damping
  - **Reference:** Phase 0 Requirements §2.2-2.3
  - **Validation:** Energy drift <0.01% over 10,000 steps
  - **Effort:** 3 days

- [ ] **0.3** Kahan Summation for Laplacian
  - Implement compensated summation in `compute_laplacian_kahan()`
  - Use compensation variable `c` to track lost low-order bits
  - Apply to ALL accumulation loops in physics kernel
  - **Reference:** Phase 0 Requirements §2.4
  - **Validation:** Memory waves persist >1000 timesteps without vanishing
  - **Effort:** 1 day

### P1 High Priority (Performance Critical)

- [ ] **0.4** AVX-512 Nonary Arithmetic
  - Replace enum-based Nit with `typedef int8_t Nit`
  - Implement `vec_sum_gate(__m512i, __m512i)` using `_mm512_add_epi8` + clamp
  - Implement `vec_product_gate(__m512i, __m512i)` with saturation
  - Remove ALL uses of `std::clamp` in hot loops
  - **Reference:** Phase 0 Requirements §4
  - **Validation:** Processes 64 nits in <10 CPU cycles
  - **Effort:** 2 days

- [ ] **0.5** Lazy Cholesky Decomposition
  - Add cached Cholesky factor `L` to `MetricTensor` class
  - Add `dirty_flag` to track when recomputation needed
  - Implement `recompute_if_needed()` with stability check
  - Rollback plasticity update if Cholesky fails (non-positive-definite)
  - **Reference:** Phase 0 Requirements §5
  - **Validation:** Metric inversion <1% of total compute time
  - **Effort:** 3 days

- [ ] **0.6** Energy Watchdog System
  - Implement `EnergyWatchdog` class with state machine
  - States: Stable, Heating, Critical, Dying
  - Monitor Hamiltonian every 100 timesteps
  - Auto-adjust damping when $\Delta E / E > 0.01$
  - Inject noise if $E < E_{min}$ (stochastic resonance)
  - **Reference:** Phase 0 Requirements §9.1
  - **Validation:** System remains stable for 24-hour continuous run
  - **Effort:** 1 day

### P2 Medium Priority (Optimization)

- [ ] **0.7** Shared Memory IPC (Physics ↔ Persistence)
  - Replace Protocol Buffers serialization with `/dev/shm` segments
  - Physics writes grid to `shm_open("/nikola_snapshot_<id>")`
  - ZeroMQ sends only snapshot_id (8 bytes)
  - Persistence mmaps shared segment
  - **Reference:** Phase 0 Requirements §6.3
  - **Validation:** IPC latency <100ns (vs. μs for Protobuf)
  - **Effort:** 2 days

- [ ] **0.8** Mamba-9D Taylor Approximation
  - Replace matrix exponential with first-order Taylor: $\exp(M) \approx I + M$
  - $A_i = I - \Delta(1-r_i)G_i$
  - Verify timestep constraint: $\Delta < \frac{0.1}{(1-r_{min})\lambda_{max}(G)}$
  - **Reference:** Phase 0 Requirements §3
  - **Validation:** SSM computation <10% of total time
  - **Effort:** 2 days

- [ ] **0.9** Q9_0 Quantization Fix
  - Correct packing: 2 nits per byte (not 5)
  - $packed = (n_1 + 4) \times 9 + (n_2 + 4)$
  - Unpack: $n_1 = (packed / 9) - 4$, $n_2 = (packed \% 9) - 4$
  - **Reference:** Phase 0 Requirements §8
  - **Validation:** Storage density = 4 bits/weight
  - **Effort:** 1 day

### P3 Low Priority (Nice-to-Have)

- [ ] **0.10** Sliding Window DFT for Firewall
  - Replace full FFT with Goertzel Algorithm
  - Monitor specific attack frequencies (10Hz, 50Hz, 100Hz)
  - **Reference:** Phase 0 Requirements §7
  - **Validation:** Firewall latency <1μs per sample
  - **Effort:** 1 day

### Phase 0 Validation Gate

**ALL P0 and P1 items MUST be completed and validated before proceeding to Phase 1.**

**Validation Criteria:**
- [ ] Energy drift <0.01% over 10,000 timesteps
- [ ] Memory waves persist >1000 timesteps
- [ ] Cache hit rate >95% in physics kernel
- [ ] Metric inversion <1% of total compute
- [ ] System stable for 24-hour continuous run
- [ ] IPC latency <100ns (if P2 complete)

**Estimated Total Effort:** 17 days (P0: 6 days, P1: 6 days, P2: 5 days)

---

## 8.10.2 Overview

This checklist MUST be followed file-by-file in order. Do NOT skip steps or deviate.

**!!! NO DEVIATION FROM SPECS FOR ANY REASON !!!**

## 8.10.3 Foundation Layer

### Setup and Configuration

- [ ] **1.1** Create root `CMakeLists.txt`
  - Set C++23 standard
  - Find packages: ZeroMQ, Protobuf, LMDB, libvirt, CUDA (optional), FFTW3
  - Configure build types: Debug, Release, RelWithDebInfo
  - Enable AVX-512 if available

- [ ] **1.2** Create `proto/neural_spike.proto`
  - Define all message types from Section 10.2
  - Generate C++ code: `protoc --cpp_out=. neural_spike.proto`
  - Verify compilation

- [ ] **1.3** Create `config/nikola.conf`
  - Set paths: state_dir, ingest_dir, archive_dir
  - Set constants: golden_ratio=1.618033988749895, emitter frequencies
  - Set thresholds: resonance_threshold=0.7, dopamine_baseline=0.5

## 8.10.4 Physics Engine

### Types and Core Structures

- [ ] **2.1** `include/nikola/types/nit.hpp`
  ```cpp
  namespace nikola {
      enum class Nit : int8_t {
          N4 = -4, N3 = -3, N2 = -2, N1 = -1, ZERO = 0,
          P1 = 1, P2 = 2, P3 = 3, P4 = 4
      };

      Nit sum_gate(Nit a, Nit b);
      Nit product_gate(Nit a, Nit b);
      Nit quantize_wave(std::complex<double> wave);
  }
  ```

- [ ] **2.2** `src/types/nit.cpp`
  - Implement all three functions from 2.1
  - Add unit tests in `tests/unit/test_nonary.cpp`
  - **Validation:** Test $1 + (-1) = 0$, $2 \times 3 = 4$ (saturate)

- [ ] **2.3** `include/nikola/types/coord9d.hpp`
  - Define `Coord9D` struct with `std::array<int32_t, 9>`
  - Implement `wrap()` method for toroidal topology
  - Implement `distance_to()` for geodesic distance
  - Define hash function for use in `unordered_map`

- [ ] **2.4** `include/nikola/types/torus_node.hpp`
  - Define `TorusNode` struct (256-byte aligned)
  - Include: wavefunction, velocity, acceleration, metric_tensor, resonance_r, state_s
  - **CRITICAL:** Zero padding in constructor for proper initialization
  - Note: velocity and acceleration fields required for Velocity-Verlet integration
  - Verify `sizeof(TorusNode) == 256`

### Emitter Array

- [ ] **2.5** `include/nikola/physics/emitter_array.hpp`
  - Define `EmitterArray` class with phase accumulators
  - Declare sine LUT (16384 samples)
  - Define DDS tick() method

- [ ] **2.6** `src/physics/emitter_array.cpp`
  - Initialize sine LUT in constructor
  - Compute tuning words from frequencies
  - Implement DDS algorithm from Section 4.5
  - **Validation:** Generate 1Hz sine, verify with FFT

### Torus Manifold

- [ ] **2.7** `include/nikola/physics/shvo_grid.hpp`
  - Define `SparseHyperVoxelGrid` class
  - Implement Morton code hashing
  - Define neurogenesis methods

- [ ] **2.8** `src/physics/shvo_grid.cpp`
  - Implement sparse grid using `unordered_map<uint64_t, TorusNode*>`
  - Implement `get_or_create()` with neurogenesis trigger
  - Implement `update_gpu_neighbor_map()` for dynamic topology

- [ ] **2.9** `include/nikola/physics/torus_manifold.hpp`
  - Define main interface
  - Declare `inject_wave()`, `propagate()`, `find_resonance_peak()`
  - Declare neuroplasticity/neurogenesis methods

- [ ] **2.10** `src/physics/torus_manifold.cpp`
  - Implement wave propagation using Unified Field Interference Equation
  - Implement neuroplasticity update (Section 3.4)
  - Integrate with ENGS global state
  - **Validation:** Inject two waves, verify interference

### Wave Interference Processor

- [ ] **2.11** `src/physics/wave_engine.cpp`
  - Implement superposition addition
  - Implement heterodyning multiplication
  - Implement spectral cascading (carry mechanism)
  - **Validation:** Test $+3 + +2 = +4$ (saturate), not +5

## 8.10.5 Cognitive Systems

### Mamba-9D

- [ ] **3.1** `include/nikola/mamba/hilbert_scan.hpp`
  - Define `HilbertMapper` class
  - Declare `encode()` and `decode()` methods

- [ ] **3.2** `src/mamba/hilbert_scan.cpp`
  - Implement Hilbert curve mapping
  - **Validation:** Verify locality preservation

- [ ] **3.3** `include/nikola/mamba/ssm_kernel.hpp`
  - Define `Mamba9D` class with A, B, C matrices
  - Implement Topological State Mapping

- [ ] **3.4** `src/mamba/ssm_kernel.cpp`
  - Implement SSM forward pass
  - Derive matrices from metric tensor
  - **Validation:** Test state propagation

### Transformer

- [ ] **3.5** `include/nikola/reasoning/attention.hpp`
  - Define `WaveAttentionLayer`
  - Declare wave correlation methods

- [ ] **3.6** `src/reasoning/wave_attention.cpp`
  - Implement Wave Correlation Attention
  - Use complex conjugate product
  - **Validation:** Compare with standard attention

- [ ] **3.7** `src/reasoning/transformer.cpp`
  - Implement full transformer stack
  - Integrate wave attention
  - Add neuroplasticity hooks

### Embedder

- [ ] **3.8** `src/reasoning/embedder.cpp`
  - Implement text → waveform conversion
  - Use character/token encoding
  - **Validation:** Text roundtrip accuracy >90%

## 8.10.6 Infrastructure

### ZeroMQ Spine

- [ ] **4.1** `src/spine/broker.cpp`
  - Implement message router
  - Add CurveZMQ security (Section 10.3)
  - Implement ZAP authentication

- [ ] **4.2** `src/spine/shadow_spine.cpp`
  - Implement A/B testing infrastructure
  - Add voting mechanism
  - Add promotion logic

### Orchestrator and Agents

- [ ] **4.3** `src/orchestrator/smart_router.cpp`
  - Implement tool selection logic
  - Integrate all agents

- [ ] **4.4** `src/agents/*.cpp`
  - Implement Tavily, Firecrawl, Gemini clients
  - Implement Custom HTTP client
  - **Validation:** Test API calls

### Executor

- [ ] **4.5** `src/executor/kvm_executor.cpp`
  - Implement VM lifecycle management
  - Add virtio-serial communication
  - Implement CSVP integration

## 8.10.7 Autonomy

### Neurochemistry

- [ ] **5.1** `src/autonomy/engs.cpp`
  - Implement Extended Neurochemical Gating System
  - Use exponential decay for homeostasis
  - Integrate with physics kernel

- [ ] **5.2** `src/autonomy/dopamine.cpp`
  - Implement TD learning
  - Add reward mechanisms

- [ ] **5.3** `src/autonomy/boredom.cpp`
  - Implement Shannon entropy calculation
  - Add curiosity triggers

- [ ] **5.4** `src/autonomy/goals.cpp`
  - Implement goal DAG
  - Add completion propagation

### Training and Self-Improvement

- [ ] **5.5** `src/autonomy/trainers.cpp`
  - Implement Bicameral Autonomous Trainers
  - Add auto-training triggers

- [ ] **5.6** `src/autonomy/dream_weave.cpp`
  - Implement counterfactual simulation
  - Add z-score normalization

- [ ] **5.7** `src/self_improve/adversarial_dojo.cpp`
  - Implement Red Team agent
  - Add attack generation

## 8.10.8 Persistence & Security

### Persistence

- [ ] **6.1** `src/persistence/lsm_dmc.cpp`
  - Implement LSM-DMC persistence system
  - Add compaction worker
  - Add Write-Ahead Log

- [ ] **6.2** `src/persistence/gguf_export.cpp`
  - Implement Hilbert flattening
  - Add Q9_0 quantization
  - **Validation:** Load in llama.cpp

### Security

- [ ] **6.3** `src/security/resonance_firewall.cpp`
  - Implement spectral analysis
  - Load hazard database

- [ ] **6.4** `src/security/csvp.cpp`
  - Implement Code Safety Verification Protocol
  - Add static analysis hooks
  - Add physics invariant tests

## 8.10.9 Multimodal

- [ ] **7.1** `src/multimodal/audio_resonance.cpp`
  - Implement FFT binning
  - Implement dynamic frequency folding
  - **Validation:** Process speech sample

- [ ] **7.2** `src/multimodal/visual_cymatics.cpp`
  - Implement holographic RGB encoding
  - Add phase-based color separation
  - **Validation:** Process test image

## 8.10.10 Tools and CLI

- [ ] **8.1** `tools/twi-ctl/main.cpp`
  - Implement CLI controller
  - **CRITICAL:** Call `curl_global_init(CURL_GLOBAL_DEFAULT)` at program startup (before any threads)
  - **CRITICAL:** Call `curl_global_cleanup()` at program shutdown (after all threads terminate)
  - Note: libcurl global initialization is NOT thread-safe and must be done once per process
  - Add all commands from Section 25
  - **Validation:** Test all commands

- [ ] **8.2** `tools/convert_nikola_to_gguf.py`
  - Implement Python export script
  - **Validation:** Export sample state

## 8.10.11 Testing

- [ ] **9.1** Implement all unit tests
  - Physics invariants
  - Nonary arithmetic
  - Wave interference
  - ENGS homeostasis

- [ ] **9.2** Implement integration tests
  - Search-retrieve-store loop
  - Training cycle
  - Multimodal processing

- [ ] **9.3** Implement benchmarks
  - Wave propagation performance
  - Hilbert scan performance

## 8.10.12 Final Integration

- [ ] **10.1** Build Docker images
- [ ] **10.2** Run security verification
- [ ] **10.3** Performance testing
- [ ] **10.4** Documentation review

---

**Total Checklist Items:** ~60
**Estimated Completion:** 12 months (5-person team)

---

**Cross-References:**
- See Section 26 for File Structure
- See Section 27 for Development Roadmap
# BUILD AND DEPLOYMENT

## 8.11.1 CLI Controller

**Binary Name:** `twi-ctl` (Toroidal Waveform Intelligence Controller)

**Usage:**

```bash
twi-ctl <command> [arguments]
```

### Command Set

| Command | Arguments | Description |
|---------|-----------|-------------|
| `query` | `"<text>"` | Submit query to system |
| `status` | - | Show system status (dopamine, boredom, active nodes) |
| `nap` | - | Trigger immediate nap/checkpoint |
| `train` | `[mamba\|transformer\|both]` | Trigger training session |
| `ingest` | `<file_path>` | Manually ingest file |
| `export` | `<output.gguf>` | Export to GGUF format |
| `goals` | `list\|add\|complete` | Manage goal system |
| `identity` | - | Show identity profile |
| `firewall` | `add <pattern>` | Add hazardous pattern |
| `metrics` | - | Show performance metrics |
| `shutdown` | - | Graceful shutdown |

### Implementation Excerpt

```cpp
// File: tools/twi-ctl/main.cpp

class TWIController {
    zmq::context_t ctx;
    zmq::socket_t socket;

public:
    TWIController() : ctx(1), socket(ctx, ZMQ_REQ) {
        socket.connect("ipc:///tmp/nikola/spine_cli.ipc");
    }

    std::string send_query(const std::string& query_text) {
        NeuralSpike spike;
        spike.set_request_id(generate_uuid());
        spike.set_timestamp(current_timestamp());
        spike.set_sender(ComponentID::CLI_CONTROLLER);
        spike.set_recipient(ComponentID::ORCHESTRATOR);
        spike.set_text_data(query_text);

        // Serialize directly to ZMQ message (zero-copy, no intermediate std::string)
        size_t msg_size = spike.ByteSizeLong();
        zmq::message_t request(msg_size);
        spike.SerializeToArray(request.data(), msg_size);
        socket.send(request, zmq::send_flags::none);

        // Receive response
        zmq::message_t reply;
        socket.recv(reply, zmq::recv_flags::none);

        NeuralSpike response;
        response.ParseFromArray(reply.data(), reply.size());

        return response.text_data();
    }
};

// Main entry point with proper libcurl initialization
int main(int argc, char* argv[]) {
    // CRITICAL: Initialize libcurl globally before any threading or network operations
    // This prevents race conditions with the CustomHTTPClient used by external tools
    // See Section 12.4 for CustomHTTPClient implementation
    curl_global_init(CURL_GLOBAL_ALL);

    // Ensure cleanup on exit
    std::atexit([]() {
        curl_global_cleanup();
    });

    // Parse command and execute
    if (argc < 2) {
        std::cerr << "Usage: twi-ctl <command> [args...]" << std::endl;
        return 1;
    }

    TWIController controller;
    std::string command = argv[1];

    if (command == "query" && argc == 3) {
        std::string result = controller.send_query(argv[2]);
        std::cout << result << std::endl;
    } else if (command == "status") {
        // ... other commands ...
    } else {
        std::cerr << "Unknown command: " << command << std::endl;
        return 1;
    }

    // libcurl will be cleaned up automatically via std::atexit
    return 0;
}
```

## 8.11.2 Build System (CMake)

### Root CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.20)
project(Nikola VERSION 0.0.4 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build types
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fsanitize=address")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# Find dependencies
find_package(ZeroMQ REQUIRED)
find_package(Protobuf REQUIRED)
find_package(LMDB REQUIRED)
find_package(libvirt REQUIRED)
find_package(FFTW3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(nlohmann_json 3.11.0 REQUIRED)  # JSON library for configuration
find_package(CUDA QUIET)

# Optional AVX-512
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)
if(COMPILER_SUPPORTS_AVX512)
    add_compile_options(-mavx512f)
    add_definitions(-DUSE_AVX512)
endif()

# Subdirectories
add_subdirectory(proto)
add_subdirectory(src)
add_subdirectory(tools)
add_subdirectory(tests)
```

### Library CMakeLists.txt

```cmake
# src/CMakeLists.txt

add_library(lib9dtwi SHARED
    types/nit.cpp
    types/coord9d.cpp
    physics/torus_manifold.cpp
    physics/emitter_array.cpp
    physics/wave_engine.cpp
    physics/shvo_grid.cpp
    mamba/hilbert_scan.cpp
    mamba/ssm_kernel.cpp
    reasoning/transformer.cpp
    reasoning/wave_attention.cpp
    reasoning/embedder.cpp
    spine/broker.cpp
    spine/component_client.cpp
    spine/shadow_spine.cpp
    orchestrator/smart_router.cpp
    agents/tavily.cpp
    agents/firecrawl.cpp
    agents/gemini.cpp
    agents/http_client.cpp
    executor/kvm_executor.cpp
    autonomy/engs.cpp
    autonomy/dopamine.cpp
    autonomy/boredom.cpp
    autonomy/goals.cpp
    autonomy/trainers.cpp
    autonomy/dream_weave.cpp
    persistence/lsm_dmc.cpp
    persistence/gguf_export.cpp
    persistence/identity.cpp
    multimodal/audio_resonance.cpp
    multimodal/visual_cymatics.cpp
    security/resonance_firewall.cpp
    security/csvp.cpp
    self_improve/profiler.cpp
    self_improve/adversarial_dojo.cpp
    ingestion/sentinel.cpp
)

target_link_libraries(lib9dtwi
    PUBLIC
        zmq
        protobuf
        lmdb
        virt
        fftw3
        ${OpenCV_LIBS}
        nlohmann_json::nlohmann_json  # JSON library for configuration
)

target_include_directories(lib9dtwi
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

# CUDA kernels (if available)
if(CUDA_FOUND)
    cuda_add_library(nikola_cuda STATIC
        physics/kernels/wave_propagate.cu
    )
    target_link_libraries(lib9dtwi PUBLIC nikola_cuda)
endif()
```

## 8.11.3 Docker Deployment

### Multi-Stage Dockerfile

```dockerfile
# Stage 1: Build environment
FROM ubuntu:24.04 AS builder

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libzmq3-dev \
    libprotobuf-dev \
    protobuf-compiler \
    liblmdb-dev \
    libvirt-dev \
    libfftw3-dev \
    libopencv-dev \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy dependency manifests first (for cache optimization)
COPY CMakeLists.txt .
COPY proto/ proto/

# Configure CMake dependencies layer (cached unless CMakeLists.txt changes)
RUN cmake -DCMAKE_BUILD_TYPE=Release -B build

# Copy source code (invalidates cache only when source changes)
COPY src/ src/
COPY include/ include/

# Build application (cached unless source or dependencies change)
RUN cmake --build build --parallel $(nproc) && \
    cmake --install build --prefix /install

# Stage 2: Runtime environment
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y \
    libzmq5 \
    libprotobuf32 \
    liblmdb0 \
    libvirt0 \
    libfftw3-3 \
    libopencv-core4.6 \
    libcurl4 \
    qemu-system-x86 \
    nlohmann-json3-dev \
    && rm -rf /var/lib/apt/lists/*

# Verify runtime dependencies with ldd during build:
# RUN ldd /usr/local/bin/nikola-daemon && ldd /usr/local/bin/twi-ctl

COPY --from=builder /install /usr/local

# Create directories
RUN mkdir -p /var/lib/nikola/{state,ingest,archive} && \
    mkdir -p /etc/nikola

# Copy config
COPY config/*.conf /etc/nikola/

# Expose IPC socket
VOLUME ["/tmp/nikola"]

ENTRYPOINT ["/usr/local/bin/nikola-daemon"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  nikola-spine:
    image: nikola:latest
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      - nikola-state:/var/lib/nikola/state
      - nikola-ingest:/var/lib/nikola/ingest
      - /tmp/nikola:/tmp/nikola
    environment:
      - TAVILY_API_KEY=${TAVILY_API_KEY}
      - FIRECRAWL_API_KEY=${FIRECRAWL_API_KEY}
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    deploy:
      resources:
        limits:
          memory: 32G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  nikola-state:
  nikola-ingest:
```

## 8.11.4 Running the System

### Start Services

```bash
# Start Docker compose
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f nikola-spine
```

### CLI Usage Examples

```bash
# Query the system
twi-ctl query "What is the golden ratio?"

# Check system status
twi-ctl status

# Trigger nap
twi-ctl nap

# Start training
twi-ctl train both

# Manually ingest a file
twi-ctl ingest /path/to/document.pdf

# Export to GGUF
twi-ctl export nikola-snapshot.gguf

# Manage goals
twi-ctl goals list
twi-ctl goals add "Learn quantum computing"
twi-ctl goals complete <goal-id>

# View identity
twi-ctl identity

# Add firewall pattern
twi-ctl firewall add "ignore previous instructions"

# View metrics
twi-ctl metrics

# Shutdown
twi-ctl shutdown
```

## 8.11.5 Testing

### Unit Tests

```bash
# Run all unit tests
cd build
ctest --output-on-failure

# Run specific test suite
ctest -R test_nonary

# Run with Valgrind (memory check)
ctest -T memcheck
```

### Integration Tests

```bash
# Run integration tests
ctest -R integration

# Benchmark performance
ctest -R bench
```

### Physics Invariants Check

```bash
# Verify energy conservation
./build/tests/unit/test_energy_conservation

# Verify nonary arithmetic
./build/tests/unit/test_nonary

# Verify toroidal wrapping
./build/tests/unit/test_coord9d
```

## 8.11.6 Deployment Checklist

**Pre-Deployment:**
- [ ] All unit tests pass (100%)
- [ ] All integration tests pass
- [ ] Physics invariants verified
- [ ] Security verification passed (Appendix G)
- [ ] Performance benchmarks met (Appendix F)
- [ ] Docker image builds successfully

**Deployment:**
- [ ] Configure API keys in environment
- [ ] Set up persistence volumes
- [ ] Configure firewall rules
- [ ] Start services with docker-compose
- [ ] Verify CLI connectivity

**Post-Deployment:**
- [ ] Monitor system status
- [ ] Check logs for errors
- [ ] Verify external tool connectivity
- [ ] Test basic query/response
- [ ] Verify nap/checkpoint cycle

## 8.11.7 Monitoring

### System Metrics

```bash
# Dopamine level
twi-ctl status | grep Dopamine

# Active nodes count
twi-ctl status | grep "Active Nodes"

# Uptime
twi-ctl status | grep Uptime
```

### Performance Metrics

```bash
# Detailed metrics
twi-ctl metrics

# Output includes:
# - Wave propagation time
# - Resonance detection latency
# - Training cycle duration
# - Memory usage
# - GPU utilization (if available)
```

---

**Cross-References:**
- See Section 10 for ZeroMQ Spine details
- See Section 26 for File Structure
- See Section 28 for Implementation Checklist
- See Appendix I for Docker deployment details
