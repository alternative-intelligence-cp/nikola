# Nikola SIE — Nitpick Integration Layer (v0.0.10)

Nitpick modules implementing Nikola's Self-Improvement Engine (SIE) 4-gate
validation pipeline. Gates 0, 1, and 3 call into C++ via FFI; Gate 2
uses both pure Nitpick math and FFI-backed checks.

## Architecture

```
test_*.npk → nikola_sie_pipeline.npk (orchestrator)
                ├── nikola_shadowspine.npk    (Gate 0: signature verify)
                ├── nikola_blacklist.npk       (Gate 1: banned patterns)
                ├── nikola_physics_oracle.npk  (Gate 2: physics checks)
                ├── nikola_swapper.npk         (Gate 3: module hot-swap)
                ├── nikola_metabolic_controller.npk  (ATP budget)
                └── nikola_sie_types.npk       (shared types/constants)
                         ↓ FFI (extern func)
                  shim/nikola_sie_bridge.c → libnikola_sie.a
```

## Files

| File | Purpose | Key Functions |
|------|---------|---------------|
| `nikola_sie_types.npk` | Status codes, CycleReport/CycleStats structs | status_name, print_report |
| `nikola_metabolic_controller.npk` | ATP energy budget (750/cycle) | needs_nap, consume_cycle, recharge |
| `nikola_physics_oracle.npk` | Energy/reversibility/viscosity/resonance | run_gate2, check_resonance, drift_alert_level |
| `nikola_shadowspine.npk` | Gate 0: Ed25519+SPHINCS+ signature (stub) | run_gate0, verify_signature |
| `nikola_blacklist.npk` | Gate 1: Banned syscall pattern scan | run_gate1, is_safe, scan_source |
| `nikola_swapper.npk` | Gate 3: dlopen module hot-swap (stub) | run_gate3, swap_in, rollback |
| `nikola_sie_pipeline.npk` | Orchestrator wiring all 4 gates | run_sie_cycle |
| `shim/nikola_sie_bridge.c` | C FFI bridge to C++ gate classes | nk_verifier_*, nk_blacklist_*, nk_oracle_*, nk_swapper_* |

## Build & Run

Requires `nitpickc` (Nitpick compiler v0.8.4+) and `gcc`:

```bash
# Build FFI shim
cd shim
gcc -O2 -Wall -fPIC -c nikola_sie_bridge.c -o nikola_sie_bridge.o -lm
ar rcs libnikola_sie.a nikola_sie_bridge.o
cd ..

# Run individual module tests (no FFI needed)
nitpickc tests/test_types.npk -I . -o tests/test_types && ./tests/test_types
nitpickc tests/test_metabolic.npk -I . -o tests/test_metabolic && ./tests/test_metabolic

# Run gate tests (FFI required)
nitpickc tests/test_shadowspine.npk -I . -o tests/test_shadowspine -L shim -lnikola_sie && ./tests/test_shadowspine
nitpickc tests/test_blacklist.npk -I . -o tests/test_blacklist -L shim -lnikola_sie && ./tests/test_blacklist
nitpickc tests/test_swapper.npk -I . -o tests/test_swapper -L shim -lnikola_sie && ./tests/test_swapper
nitpickc tests/test_oracle.npk -I . -o tests/test_oracle -L shim -lnikola_sie -lm && ./tests/test_oracle

# Run full pipeline integration test
nitpickc tests/test_pipeline.npk -I . -o tests/test_pipeline -L shim -lnikola_sie -lm && ./tests/test_pipeline
```

## Design

- **Library modules** — no main/failsafe in library files; tests are in `tests/`
- **Single-import rule** — only import the top-level module you need; nitpickc
  doesn't handle diamond imports (same module imported from multiple paths)
- **FFI pattern** — `extern func:name = type(params);` calls C shim functions
  linked via `-L shim -lnikola_sie`
- **Constants from spec** — CANDLE_DRIFT_LIMIT=1e-6, REVERSIBILITY_LIMIT=1e-12,
  ATP_CYCLE_TOTAL=750, RESONANCE_AMPLITUDE_LIMIT=100
- **Stubs** — Gate 0 (crypto) and Gate 3 (dlopen) are structural stubs;
  full implementations wired when libsodium/PQC and C++ ModuleSwapper are linked
- **SIE Gate 2 ready** — physics_oracle.npk implements the full oracle
  verdict logic matching `include/nikola/autonomy/physics_oracle.hpp`

## Status

All 3 files compile and run correctly as of v0.0.5 (nitpickc v0.17.5, 2026-04-07).
