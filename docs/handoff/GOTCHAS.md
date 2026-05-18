# Nikola — Gotchas and Known Traps

_Last updated: 2026-04-09_

This document captures the non-obvious problems that will bite you if you
don't know about them. Read this before the first build.

---

## Build System

### `ctest` without a filter hangs forever

The Phase 40s–60s test binaries include tests for blocking infrastructure
(ZMQ daemon sockets, interactive REPL, async monitoring threads) that wait
for external clients that never arrive. Running bare `ctest` will block
indefinitely at one of these phases.

**Always use a filter:**
```bash
ctest -R Phase125 -v       # safe — single phase
ctest -R "Phase13[0-5]" -v # safe — range
ctest -v                   # NEVER DO THIS
```

Known blocking phases: 40s, 50s, 60s range. When in doubt, use `-R Phase1[0-9][0-9]`
to restrict to three-digit phase numbers (Phase 100+).

---

### Catch2 must be v3, not v2

Nikola uses the `Catch2::Catch2WithMain` CMake target which only exists in v3.
If your system has Catch2 v2 installed, CMake will silently find it and then
fail to link with a cryptic error about the missing target.

**Check version:**
```bash
apt show catch2 | grep Version
# Must be ≥ 3.0
```

**If v2 is installed:**
```bash
sudo apt remove catch2
# Install v3 from source:
git clone --branch v3.7.1 https://github.com/catchorg/Catch2
cd Catch2 && cmake -B build -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build build && sudo cmake --install build
```

---

### Eigen3 must be ≥ 3.4 — silent API failure with older versions

The codebase uses `Eigen3::Eigen` target and `reshaped()` / tensor extensions
added in 3.4. With an older version, CMake may find and accept the library but
compilation will fail deep in template instantiations with errors that do not
mention the version number.

**Check version:**
```bash
pkg-config --modversion eigen3
# Must be ≥ 3.4.0
```

---

### ORT path is hardcoded — crashes on a new machine

`CMakeLists.txt` has a hardcoded default:
```
ORT_ROOT = /home/randy/Workspace/SYSTEM/onnxruntime/cpp
```

On any other machine, ORT detection silently fails and BERT embeddings are
disabled (random projection fallback). This is functional but semantically
degraded. Override:

```bash
cmake .. -DORT_ROOT=/path/to/your/onnxruntime-linux-x64-gpu-1.24.2/
```

---

### OpenTelemetry linker expects `/usr/local/lib`

`nikola_run` links against the OTel `.a` archives via:
```cmake
target_link_directories(nikola_run PRIVATE /usr/local/lib)
```

If you compiled OTel and installed it elsewhere, update this path in
`CMakeLists.txt`. The third_party prebuilt is at
`third_party/opentelemetry-cpp-build/` but the `nikola_run` target links
from the system install path because the prebuilt was installed there.

---

### CUDA path compiles but has a pending C++20 issue

`propagator.cu` requires C++17 (nvcc limitation as of CUDA 12.0).
The host codebase is C++23. There is a `std::span` usage in `propagator.cu`
that was patched in Phase 111 but the CUDA SM target must match your GPU:

```bash
cmake .. -DNVCC_ARCH=sm_86   # RTX 3090 (sm_86)
cmake .. -DNVCC_ARCH=sm_89   # RTX 4090 (sm_89)
```

If you don't set this, it defaults to sm_86. Wrong SM target = silently loads
and runs on wrong compute capability with incorrect results on GPU.

---

## Physics / Math

### Emitter spacing — do not change π·φⁿ

The 9 emitters are placed at frequencies `π·φⁿ` for n = 0..8 (φ = golden ratio).
This irrational spacing prevents mode-locking — a state where multiple emitters
fall into resonance and the field develops degenerate stationary patterns.

**Mode-locking is a silent failure.** There is no compile error, no runtime
exception, no NaN. The field will appear to function normally for ~500 ticks
then gradually degrade into a low-entropy standing-wave pattern. Downstream
cognitive outputs become repetitive and nonsensical without any clear error.

If you need different numbers of emitters, or different frequency ranges,
you must preserve the irrationality of the spacing. Any rational ratio between
two emitter frequencies will eventually mode-lock.

---

### Grid size 3⁹ is baked in — not a configurable parameter

The torus is always 3⁹ = 19,683 nodes. SoA layout, AVX-512 alignment, the
Hilbert curve tables, the emitter coefficient arrays — all assume this size.
It is not a runtime parameter. It could theoretically be made configurable but
would require changing at minimum: `soa_layout.hpp`, `hilbert_reference.hpp`,
`torus_block_simd.hpp`, and all `EmitterArray` initialization paths.

---

### Störmer–Verlet must not be replaced with RK4

The Störmer–Verlet / Strang-split integrator is symplectic. This means it
conserves the Hamiltonian structure of the UFIE exactly (up to floating-point).
RK4 is not symplectic — it introduces secular energy drift. Over thousands of
ticks this causes `torus_energy` to grow without bound, eventually overflowing
`NikolaState.torus_energy` and corrupting the dopamine update.

If you need better accuracy, use a higher-order symplectic method (e.g.
Yoshida 4th-order). Do not replace with Runge-Kutta.

---

### `NikolaState` fields must match everywhere

`NikolaState` is serialized for OTel spans, cross-module state passing,
and LMDB persistence. Adding or removing fields requires updating:
- `HomeostasisMonitor::compute_energy()` (uses dopamine, atp, boredom)
- `HomeostasisMonitor::compute_entropy()` (uses entropy)
- `nikola_run.cpp` OTel attribute map
- Any proto/nikola_state.proto definition
- Memory persistence layer deserialization

---

## Testing

### Phase43 and Phase51 are pre-existing timing flakes

`Phase43HebbianUpdate` and `Phase51FailureModeGuards` fail intermittently
under CPU contention (p ≈ 2–5%). These are floating-point timing-sensitive
tests. They are **not regressions** — they have flaked since they were written.

If CI catches them: re-run individually. If they pass on retry, it's the flake.
If they consistently fail, it is a regression.

---

### `docs/architecture/` has 4 specification docs

The directory `/docs/architecture/` contains:
- `memory_schema.md` — LMDB persistence layout (Phase 137)
- `metrics_schema.md` — JSON-Lines telemetry wire format
- `peer_protocol.md` — CurveZMQ peer discovery & handshake
- `physics_calibration_protocol.md` — GAP-030 calibration test spec

The primary architecture documentation lives in `/docs/handoff/ARCHITECTURE.md`.

---

### Phase registry is NOT in git

`META/NITPICK/NIKOLA_PHASE_REGISTRY.md` is outside the repo root (`REPOS/nikola/`)
and is not tracked by git. It lives in the broader workspace at:
```
/home/randy/Workspace/META/NITPICK/NIKOLA_PHASE_REGISTRY.md
```

This is the definitive record of all phases, ctest numbers, commit hashes, and
design rationale. **Do not lose this file.** If you are receiving this handoff
package, ask for this file explicitly — it is not included in anything that
`git clone` will give you.

---

### `TASKS.md` lists items that are NOT implemented

`TASKS.md` in the repo root contains a mix of completed and pending items.
Several items marked with ecosystem integration requirements are **not yet
implemented**:

- `ScopeProfiler` — used in TASKS.md code samples but the class doesn't exist
- `TelemetryDaemon` — `nikola::telemetry::TelemetryDaemon` is referenced but
  not yet a real module
- `DebugAdapter` — similarly referenced but not implemented

These are planning items, not completed work. Do not assume `TASKS.md` entries
are done unless they have a "DONE — Phase NNN" annotation.

---

## Runtime

### `--stream` mode requires the callback to be set

The `--stream` flag works by hooking `DecisionLoop::on_action(callback)`.
If `on_action` is not called before the tick loop starts, streaming silently
produces no output (no error, no fallback to buffered mode).

---

### Memory database path must be writable

LMDB (`--memory /path/to/memory.lmdb`) will crash with a confusing LMDB
error (MDB_LOCK_ERROR or permissions error) if the parent directory is not
writable by the running process. Pre-create the directory.

---

### Interactive mode does not time out

`--interactive` reads from stdin. If you pipe input that closes without EOF,
the process will block waiting for more input. Always send EOF (Ctrl+D) or
use `--prompt` for single-shot use.

---

## GPU

### RTX 3090 sm_86 target

The original development GPU is an RTX 3090 (sm_86). The default NVCC
architecture flag is sm_86. If you run on a different NVIDIA GPU, you must
set `-DNVCC_ARCH=smXXX` or the GPU code will miscompile (wrong instruction
set, possibly segfault on launch).

---

## Miscellaneous

### Build dir is `build/` — not `cmake-build-debug/` or similar

The project assumes `build/` relative to repo root. Using a different build
directory is fine but then binary paths like `./build/nikola-run` will be
wrong in any script or documentation.

---

### The `aria/` and `aria_community/` repos are separate projects

`REPOS/aria/` and `REPOS/aria_community/` contain a different project (the
Aria language). They share the workspace but are not dependencies of Nikola.
Don't confuse them.

---

### Phase142 calibration tests need 300s timeout

The full and long-term physics calibration tests (Phase142PhysicsCalibration,
Phase142PhysicsCalibration_LongTerm) take ~215s. Always use `--timeout 300`
or rely on the CMakeLists.txt `TIMEOUT` property. The quick variant
(Phase142PhysicsCalibration_Quick) completes in ~22s.

---

### Aria SIE library files need `-c` flag

The 7 Aria SIE library files (`aria/sie/nikola_*.aria`) are compilation units
without `failsafe` functions. Compile them with `ariac -c` (library mode).
The 8 test files need `-I aria/sie/ -L aria/sie/shim/ -lnikola_sie` for
module resolution and FFI shim linking.

---

### LMDB environments use directory mode

`CodeProposalStore`, `LmdbMemoryStore`, and `LmdbStateStore` all use LMDB in
directory mode (not `MDB_NOSUBDIR`). Pass a directory path, not a file path.
The directory must be writable and will be created if it doesn't exist.
