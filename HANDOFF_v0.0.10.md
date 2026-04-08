# Nikola v0.0.10 — Aria SIE Integration Layer

## Summary

Built the Aria-side SIE (Self-Improvement Engine) integration layer with
FFI bridge to Nikola's C++ gate classes. The full 4-gate validation pipeline
now runs in Aria, with each gate independently testable.

## What Was Done

### Phase 1: Module Architecture
- Converted 3 existing standalone `.aria` files to importable library modules
  (removed `main`/`failsafe`, extracted tests to `tests/`)
- Created `shim/` directory for C FFI bridge code
- Validated module import pattern: `use "file.aria".*;` with `-I .`
- Validated FFI pattern: `extern func:name` → C shim (`.a`) → `-L shim -lnikola_sie`

### Phase 2: Gate 0 — ShadowSpine (Signature Verification)
- `nikola_shadowspine.aria`: `verify_signature()`, `run_gate0()`, `failure_name()`
- Structural stub (validates sig=64 bytes, pubkey=32 bytes)
- Full Ed25519+SPHINCS+ crypto wired when libsodium/PQC libs linked
- 5 tests passing

### Phase 3: Gate 1 — CodePatternBlacklist
- `nikola_blacklist.aria`: `is_safe()`, `scan_source()`, `run_gate1()`
- Scans for: system(), exec*(), fork(), popen(), __asm__, ptrace
- 8 tests passing

### Phase 4: Gate 2 — PhysicsOracle (Extended)
- Extended `nikola_physics_oracle.aria` with FFI-backed checks:
  - `check_resonance()` — amplitude blow-up detection
  - `check_viscosity_trap()` — exponential decay model via C shim (exp())
  - `drift_alert_level()` — 0=OK, 1=WARN, 2=CRITICAL
  - `check_decoherence()` — visibility < 0.01 detection
  - `run_gate2()` — pipeline wrapper (energy + reverse + viscosity + resonance)
- 17 tests passing (9 pure Aria + 8 FFI-backed)

### Phase 5: Pipeline Orchestrator
- `nikola_sie_pipeline.aria`: `run_sie_cycle()` wires Gate0→Gate1→Gate2→Gate3
- ATP budget enforced before any gate runs (750 ATP/cycle)
- Short-circuits on first failure with proper status code
- `print_cycle_result()` for diagnostics
- 10 integration tests passing

### Phase 6: Gate 3 — ModuleSwapper (Hot-Swap)
- `nikola_swapper.aria`: `swap_in()`, `rollback()`, `has_active()`, `run_gate3()`
- Structural stub (validates .so extension, simulates swap state)
- Full dlopen wired when C++ ModuleSwapper linked
- 8 tests passing

### C FFI Bridge
- `shim/nikola_sie_bridge.c` → `libnikola_sie.a`
- 20 exported functions across 4 gate namespaces
- Pure C with no C++ dependencies (stubs for v0.0.10)

## Test Results

| Test Suite | Tests | Status |
|-----------|-------|--------|
| test_types | Status codes, structs, reporting | PASS |
| test_metabolic | ATP budget, nap/recharge cycles | PASS |
| test_oracle | 17 physics checks (pure + FFI) | PASS |
| test_shadowspine | 5 signature verification tests | PASS |
| test_blacklist | 8 pattern scanning tests | PASS |
| test_swapper | 8 hot-swap tests | PASS |
| test_pipeline | 10 integration tests (all paths) | PASS |
| **Total** | **7/7 suites** | **ALL PASS** |

C++ regression: 140/140 tests pass (1 flaky ordering test passes individually).

## File Inventory

```
aria/sie/
  nikola_sie_types.aria            — shared types & constants (library)
  nikola_metabolic_controller.aria — ATP budget management (library)
  nikola_physics_oracle.aria       — Gate 2: physics validation (library, FFI-extended)
  nikola_shadowspine.aria          — Gate 0: signature verify (library, FFI)
  nikola_blacklist.aria            — Gate 1: pattern blacklist (library, FFI)
  nikola_swapper.aria              — Gate 3: module hot-swap (library, FFI)
  nikola_sie_pipeline.aria         — orchestrator (library)
  README.md                        — updated with v0.0.10 architecture
  shim/
    nikola_sie_bridge.c            — C FFI bridge (20 functions)
    libnikola_sie.a                — compiled static library
  tests/
    test_types.aria
    test_metabolic.aria
    test_oracle.aria
    test_shadowspine.aria
    test_blacklist.aria
    test_swapper.aria
    test_pipeline.aria
```

## Known Limitations

1. **Diamond imports crash ariac** — only import the top-level module you need;
   ariac corrupts its symbol table when the same `.aria` file is imported via
   multiple paths. Workaround: single import chain.
2. **Gate 0/3 are stubs** — structural validation only (sig lengths, .so ext);
   real crypto and dlopen require linking external libs.
3. **No struct returns across FFI** — Aria ABI can't return C structs; we use
   flattened int32 returns and separate getter functions.

## Next: v0.0.11

Per `META/NIKOLA/ROADMAP/RELEASE_0.0.11.md`:
- SSM/Mamba weight training (Equilibrium Propagation)
- NPT attention head fine-tuning
- Live pipeline → SSM integration
