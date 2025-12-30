# Gemini Deep Research Task: PIMPL Destructor Placement Bug

## Problem Statement

**Location**: Section 3.1.8 (PIMPL Pattern for ABI Stability)

**Issue Discovered**: The example code shows **dangerous destructor placement** that will cause compilation errors or undefined behavior with incomplete types.

### Specific Details

1. **Incorrect Code** (from Section 3.1.8):
   ```cpp
   // File: include/nikola/physics/torus_manifold.hpp
   class TorusManifold {
   public:
       TorusManifold(const std::array<int, 9>& dimensions);
       ~TorusManifold();  // ✅ Declared in header (correct)
       
   private:
       struct Impl;
       std::unique_ptr<Impl> pimpl;
   };
   
   // File: src/physics/torus_manifold.cpp
   TorusManifold::~TorusManifold() = default;  // ❌ INCORRECT - Should NOT be defaulted here!
   ```

2. **The Actual Bug**:
   The documentation states:
   > "Constructors/Destructors must be defined in .cpp"
   
   But then shows `= default` in the .cpp file, which is **technically correct** BUT the comment is misleading. The real issue is this would be **wrong** if written in the header:
   
   ```cpp
   // WRONG - This would fail to compile:
   class TorusManifold {
   public:
       ~TorusManifold() = default;  // ❌ Error: incomplete type 'Impl'
   private:
       struct Impl;
       std::unique_ptr<Impl> pimpl;
   };
   ```

3. **Root Cause**:
   - `std::unique_ptr<T>` requires a **complete type** for its deleter at the point of destruction
   - If `Impl` is only forward-declared, the compiler can't generate the destructor
   - The destructor **must be defined** (even if `= default`) in a translation unit where `Impl` is complete

## Research Objectives

### Primary Question
**What is the correct PIMPL idiom for destructors in C++23, and how do we document it to prevent this common mistake?**

### Sub-Questions to Investigate

1. **Rule of Five with PIMPL**:
   - Which of the Rule of Five members can be `= default` in the header?
   - Which must be explicitly defined in the .cpp file?
   - What about `= delete` for non-copyable PIMPL classes?

2. **Compiler Error Messages**:
   - What error does GCC 13 give if destructor is defaulted in header?
   - What error does Clang 16 give?
   - What error does MSVC 19 give?
   - How do we make the error message more helpful?

3. **Alternative Patterns**:
   - Can we use `std::unique_ptr<Impl, custom_deleter>` to avoid the problem?
   - Does `std::shared_ptr<Impl>` have the same restriction? (Answer: No, but has overhead)
   - Can we use `void*` and reinterpret_cast? (Answer: Yes but terrible practice)

4. **Modern C++ Solutions**:
   - Does C++20 `consteval` or C++23 `constexpr` change anything?
   - Can we use `[[deprecated]]` on header-defaulted destructors to warn developers?
   - Can we enforce this with a static analyzer (clang-tidy rule)?

## Required Deliverables

1. **Corrected PIMPL Template**:
   Complete example showing:
   - Header file with proper forward declarations
   - Implementation file with proper definitions
   - Explicit comments explaining WHY each placement is required
   - Common mistakes section showing what NOT to do

2. **Compiler Behavior Matrix**:
   ```
   | Code Pattern              | GCC 13 | Clang 16 | MSVC 19 | Outcome      |
   |---------------------------|--------|----------|---------|--------------|
   | ~T() = default in header  | Error  | Error    | Error   | Won't compile|
   | ~T(); in header, = default| OK     | OK       | OK      | Correct ✅   |
   | ~T() {} in header         | Error  | Error    | Error   | Same issue   |
   | ~T(); in header, {} in cpp| OK     | OK       | OK      | Correct ✅   |
   ```

3. **Best Practices Document**:
   - When to use PIMPL vs other idioms (abstract base class, template, etc.)
   - Copy/move semantics with PIMPL (deep copy vs disabled)
   - Exception safety guarantees
   - Performance overhead (vtable vs PIMPL indirection)

4. **Static Analysis Rules**:
   - Clang-tidy check to detect defaulted destructor with incomplete type
   - Compile-time assertion to verify Impl is complete
   - Documentation generation (Doxygen warnings)

## Example Corrections Needed

### Current (Potentially Misleading):
```cpp
// File: include/nikola/physics/torus_manifold.hpp
class TorusManifold {
public:
    ~TorusManifold();
private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

// File: src/physics/torus_manifold.cpp
TorusManifold::~TorusManifold() = default;
```

### Clarified (With Detailed Comments):
```cpp
// File: include/nikola/physics/torus_manifold.hpp
class TorusManifold {
public:
    // Destructor MUST be declared (not defined) in header
    // because std::unique_ptr<Impl> requires complete type for deletion.
    // Defining it here (even = default) causes compilation error.
    ~TorusManifold();
    
private:
    struct Impl;  // Incomplete type (forward declaration)
    std::unique_ptr<Impl> pimpl;
};

// File: src/physics/torus_manifold.cpp
#include "nikola/physics/torus_manifold.hpp"
#include "torus_manifold_impl.hpp"  // Now Impl is complete

// Destructor definition MUST be in .cpp where Impl is complete.
// = default is fine here because compiler can see Impl's destructor.
TorusManifold::~TorusManifold() = default;

// Alternative: Explicit definition (identical behavior)
// TorusManifold::~TorusManifold() {}
```

## Research Questions

1. **Standard Library Precedent**:
   - How does `std::function` handle type erasure with incomplete types?
   - How does `std::any` achieve similar PIMPL-like behavior?
   - What does Boost.PIMPL recommend?

2. **Advanced Use Cases**:
   - What if Impl is a template? Does the destructor need to be in a .tpp file?
   - What if there are multiple PIMPL members (composition)?
   - Can we use aggregate initialization with PIMPL? (C++20 designated initializers)

3. **Debug vs Release**:
   - Does -O0 vs -O3 change anything about PIMPL performance?
   - Can LTO (Link-Time Optimization) inline through PIMPL barrier?
   - What's the actual cache penalty of pointer indirection?

## Success Criteria

- [ ] Clear explanation of why header-defaulting fails
- [ ] Complete Rule of Five template for PIMPL
- [ ] No undefined behavior or compilation errors
- [ ] Performance benchmarks (PIMPL vs direct member)
- [ ] Automated checking via clang-tidy rule

## Output Format

Please provide:
1. **Bug Report** (1 page): Exact error and why it happens
2. **Corrected Specification** (2-3 pages): Full PIMPL idiom guide
3. **Compiler Test Results** (1 page): Error messages from each compiler
4. **Performance Analysis** (1 page): PIMPL overhead measurement
5. **Code Review Checklist** (1 page): How to spot this in reviews

## Additional Context

This is particularly critical because Section 3.1.8 is referenced as a **mandatory pattern** for:
- `TorusManifold` (Core Physics)
- `Mamba9D` (Cognitive Systems)
- `MultiHeadWaveAttention` (Cognitive Systems)
- `TorusDatabase` (Data Layer)
- `Orchestrator` (Infrastructure)

A mistake in the template will propagate to 5+ critical classes, so getting this 100% correct is essential.

---

**Priority**: P1 - CRITICAL (Affects multiple subsystems)
**Estimated Research Time**: 2-3 hours (well-known C++ gotcha, mostly documentation)
**Dependencies**: None (pure C++ language issue)
