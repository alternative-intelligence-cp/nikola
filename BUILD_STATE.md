# Nikola Build State — v0.0.5

_Recorded: 2026-04-07_

## Environment

| Component       | Version                                    |
|-----------------|--------------------------------------------|
| OS              | Linux Mint 22.3 (Zena), Ubuntu 24.04 base  |
| Kernel          | 6.17.0-20-generic                          |
| Compiler        | g++ 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1)|
| CMake           | 3.28.3                                     |
| CUDA            | 13.2 (r13.2/compiler.37434383_0)           |
| GPU             | NVIDIA GeForce RTX 3090 (sm_86)            |
| Driver          | 595.58.03                                  |
| Eigen           | 5.0.0                                      |
| Catch2          | 3.4.0-1build1 (system apt)                 |
| Protobuf        | libprotoc 3.21.12                          |
| ORT Root        | /home/randy/Workspace/SYSTEM/onnxruntime/cpp |

## Build Result

- **Clean build**: PASS (rm -rf build && cmake .. && make -j$(nproc))
- **Warnings**: 0 (after v0.0.5 fixes to main.cpp and nikola_run.cpp)
- **Errors**: 0

### Warnings Fixed in v0.0.5
1. `src/main.cpp:46` — removed unused `MAGENTA` color variable
2. `src/main.cpp:254` — removed unused-but-set `H_prev` variable
3. `src/nikola_run.cpp:391` — commented out unused `torus` parameter name

## Test Results

**137/137 tests pass (100%)**

| Category              | Count | Result |
|-----------------------|-------|--------|
| Standalone tests      | 4     | 4 PASS |
| Phase 2–9             | 8     | 8 PASS |
| Phase 10–99           | 89    | 89 PASS|
| Phase 100–136         | 36    | 36 PASS|
| **Total**             | **137**| **137 PASS** |

No flakes observed in this run (Phase43 and Phase51 noted as historical flakes
in GOTCHAS.md but passed cleanly here).
