# Nikola Deep Research Tasks - Session Index

**Date Created**: December 25, 2025
**Created By**: Aria (after fresh-eyes review of research)
**Purpose**: Address issues found in comprehensive documentation review

---

## Overview

These tasks target specific issues discovered during a thorough review of the Nikola v0.0.4 research documentation (49,643 lines across 11 sections). While the research is 95% complete, these represent the remaining 5% of **clarifications and edge cases** needed before implementation.

---

## Task Priority Matrix

| Priority | Count | Description |
|----------|-------|-------------|
| **P1 - CRITICAL** | 4 | Blocks implementation or causes system failure |
| **P2 - HIGH** | 4 | Required for production deployment |
| **Total** | 8 | All tasks |

---

## Tasks List

### P1 - CRITICAL (Must Fix Before Implementation)

#### 1. TASK_HILBERT_DIMENSIONALITY_FIX.md
- **Issue**: Type mismatch between 8D and 9D Hilbert encoding
- **Location**: Section 3.2.1-3.2.2 (Mamba-9D integration)
- **Impact**: Blocks Mamba-9D cognitive integration
- **Estimated Time**: 4-6 hours
- **Status**: 📝 Ready for research
- **Response File**: `responses/TASK_HILBERT_DIMENSIONALITY_FIX.md`

#### 2. TASK_COORD9D_PORTABILITY.md
- **Issue**: uint128_t is non-standard (GCC/Clang extension)
- **Location**: Section 2.1 (Foundations - Coord9D)
- **Impact**: Won't compile on MSVC, may fail on ARM64/RISC-V
- **Estimated Time**: 3-5 hours
- **Status**: 📝 Ready for research
- **Response File**: `responses/TASK_COORD9D_PORTABILITY.md`

#### 3. TASK_PIMPL_DESTRUCTOR_FIX.md
- **Issue**: PIMPL destructor placement incorrect (won't compile)
- **Location**: Section 3.1.8 (PIMPL Pattern)
- **Impact**: Affects 5+ critical classes (TorusManifold, Mamba9D, etc.)
- **Estimated Time**: 2-3 hours
- **Status**: 📝 Ready for research
- **Response File**: `responses/TASK_PIMPL_DESTRUCTOR_FIX.md`

#### 4. TASK_OVERFLOW_CASCADE.md
- **Issue**: Overflow termination undefined (energy conservation at risk)
- **Location**: Section 2.3 + Section 1.2.2 (Carry Avalanche)
- **Impact**: Violates fundamental thermodynamic constraints
- **Estimated Time**: 4-6 hours
- **Status**: 📝 Ready for research
- **Response File**: `responses/TASK_OVERFLOW_CASCADE.md`

### P2 - HIGH (Required for Production)

#### 5. TASK_HAMILTONIAN_VALUE_FUNCTION.md
- **Issue**: TD error uses potential-only (ignores kinetic energy)
- **Location**: Section 5.1 (ENGS - Dopamine System)
- **Impact**: Incorrect reward signals, learning instability
- **Estimated Time**: 6-8 hours
- **Status**: 📝 Ready for research
- **Response File**: `responses/TASK_HAMILTONIAN_VALUE_FUNCTION.md`

#### 6. TASK_BOOTSTRAP_TIMING.md
- **Issue**: No precise timing constraints for bootstrap phases
- **Location**: Section 9.1 (IMP-03 Bootstrap)
- **Impact**: Unpredictable startup, Docker/K8s healthcheck failures
- **Estimated Time**: 4-6 hours
- **Status**: 📝 Ready for research
- **Response File**: `responses/TASK_BOOTSTRAP_TIMING.md`

#### 7. TASK_METRIC_PRECISION.md
- **Issue**: Metric tensor precision strategy undefined (FP32 vs FP64)
- **Location**: Section 2.1 + Section 3.1 (Physics Kernel)
- **Impact**: Numerical stability vs performance tradeoff unclear
- **Estimated Time**: 5-7 hours
- **Status**: 📝 Ready for research
- **Response File**: `responses/TASK_METRIC_PRECISION.md`

#### 8. TASK_BOOTSTRAP_TOKEN_SECURITY.md
- **Issue**: Bootstrap token unusable in containerized environments
- **Location**: Section 4.5 + Section 9.1 (Security)
- **Impact**: Cannot deploy to Docker/Kubernetes
- **Estimated Time**: 4-6 hours
- **Status**: 📝 Ready for research
- **Response File**: `responses/TASK_BOOTSTRAP_TOKEN_SECURITY.md`

### Special: Comprehensive Testing

#### 9. TASK_EDGE_CASE_TESTING.md
- **Issue**: Missing acceptance criteria for critical edge cases
- **Location**: Multiple sections (Phase 0, Energy, Causality)
- **Impact**: Production bugs from untested scenarios
- **Estimated Time**: 8-10 hours
- **Status**: 📝 Ready for research
- **Response File**: `responses/TASK_EDGE_CASE_TESTING.md`
- **Note**: This task should validate fixes from all other tasks

---

## Workflow

### For Randy:
1. Upload all Nikola documentation to Gemini Deep Research
2. Submit each task file as a separate research query
3. Wait for Gemini to complete research (typically 4-8 hours per task)
4. Download response and **replace** the empty response file
5. Empty files will show 0 bytes - completed ones will have content
6. Review responses and integrate findings into final specification

### For Gemini:
Each task file contains:
- **Problem Statement**: What's wrong and why it matters
- **Research Objectives**: Specific questions to answer
- **Required Deliverables**: What to produce (code, proofs, benchmarks)
- **Success Criteria**: How to know it's correct
- **Output Format**: Structure of the response

---

## Dependencies

```
Task Dependency Graph:

TASK_COORD9D_PORTABILITY
    ↓
TASK_HILBERT_DIMENSIONALITY_FIX
    ↓
TASK_BOOTSTRAP_TIMING ← TASK_BOOTSTRAP_TOKEN_SECURITY
    ↓
TASK_EDGE_CASE_TESTING (validates all fixes)

Independent (can run in parallel):
- TASK_PIMPL_DESTRUCTOR_FIX
- TASK_HAMILTONIAN_VALUE_FUNCTION
- TASK_METRIC_PRECISION
- TASK_OVERFLOW_CASCADE
```

---

## Expected Outcomes

After completing all tasks:
- ✅ Code compiles on GCC, Clang, MSVC (cross-platform)
- ✅ No type mismatches in Hilbert mapping
- ✅ PIMPL pattern correctly implemented
- ✅ Energy conservation mathematically proven
- ✅ Learning system uses correct Value function
- ✅ Bootstrap sequence has explicit timing budgets
- ✅ Metric tensor precision strategy defined
- ✅ Bootstrap works in Docker/Kubernetes
- ✅ Comprehensive test suite with >90% coverage

---

## Progress Tracking

| Task | Started | Completed | Integrated | Notes |
|------|---------|-----------|------------|-------|
| TASK_HILBERT_DIMENSIONALITY_FIX | [ ] | [ ] | [ ] | |
| TASK_COORD9D_PORTABILITY | [ ] | [ ] | [ ] | |
| TASK_PIMPL_DESTRUCTOR_FIX | [ ] | [ ] | [ ] | |
| TASK_OVERFLOW_CASCADE | [ ] | [ ] | [ ] | |
| TASK_HAMILTONIAN_VALUE_FUNCTION | [ ] | [ ] | [ ] | |
| TASK_BOOTSTRAP_TIMING | [ ] | [ ] | [ ] | |
| TASK_METRIC_PRECISION | [ ] | [ ] | [ ] | |
| TASK_BOOTSTRAP_TOKEN_SECURITY | [ ] | [ ] | [ ] | |
| TASK_EDGE_CASE_TESTING | [ ] | [ ] | [ ] | Last - validates others |

---

## Files Structure

```
/home/randy/._____RANDY_____/REPOS/nikola/docs/info/gemini/
├── tasks/
│   ├── TASK_HILBERT_DIMENSIONALITY_FIX.md (detailed prompt)
│   ├── TASK_COORD9D_PORTABILITY.md
│   ├── TASK_PIMPL_DESTRUCTOR_FIX.md
│   ├── TASK_OVERFLOW_CASCADE.md
│   ├── TASK_HAMILTONIAN_VALUE_FUNCTION.md
│   ├── TASK_BOOTSTRAP_TIMING.md
│   ├── TASK_METRIC_PRECISION.md
│   ├── TASK_BOOTSTRAP_TOKEN_SECURITY.md
│   └── TASK_EDGE_CASE_TESTING.md
│
├── responses/
│   ├── TASK_HILBERT_DIMENSIONALITY_FIX.md (empty → will be filled by Gemini)
│   ├── TASK_COORD9D_PORTABILITY.md (empty)
│   ├── TASK_PIMPL_DESTRUCTOR_FIX.md (empty)
│   ├── TASK_OVERFLOW_CASCADE.md (empty)
│   ├── TASK_HAMILTONIAN_VALUE_FUNCTION.md (empty)
│   ├── TASK_BOOTSTRAP_TIMING.md (empty)
│   ├── TASK_METRIC_PRECISION.md (empty)
│   ├── TASK_BOOTSTRAP_TOKEN_SECURITY.md (empty)
│   └── TASK_EDGE_CASE_TESTING.md (empty)
│
└── TASK_INDEX.md (this file)
```

---

## Notes

- All tasks created from comprehensive review on December 25, 2025
- Research hasn't changed since marked complete, but fresh eyes revealed gaps
- This proves the value of time + multiple reviewers + systematic analysis
- Total estimated research time: 38-54 hours (Gemini Deep Research handles parallelization)
- All issues are clarifications/edge cases, not fundamental design flaws
- Nikola research remains **extraordinarily thorough** at 95% complete

---

**Next Steps**: Upload Nikola documentation bundle to Gemini and submit tasks! 🚀
