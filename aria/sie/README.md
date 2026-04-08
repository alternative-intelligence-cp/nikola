# Nikola SIE — Aria Integration Layer (v0.0.10)

Aria modules implementing Nikola's Self-Improvement Engine (SIE) 4-gate
validation pipeline. Gates 0, 1, and 3 call into C++ via FFI; Gate 2
uses both pure Aria math and FFI-backed checks.

## Architecture

```
test_*.aria → nikola_sie_pipeline.aria (orchestrator)
                ├── nikola_shadowspine.aria    (Gate 0: signature verify)
                ├── nikola_blacklist.aria       (Gate 1: banned patterns)
                ├── nikola_physics_oracle.aria  (Gate 2: physics checks)
                ├── nikola_swapper.aria         (Gate 3: module hot-swap)
                ├── nikola_metabolic_controller.aria  (ATP budget)
                └── nikola_sie_types.aria       (shared types/constants)
                         ↓ FFI (extern func)
                  shim/nikola_sie_bridge.c → libnikola_sie.a
```

## Files

| File | Purpose | Key Functions |
|------|---------|---------------|
| `nikola_sie_types.aria` | Status codes, CycleReport/CycleStats structs | status_name, print_report |
| `nikola_metabolic_controller.aria` | ATP energy budget (750/cycle) | needs_nap, consume_cycle, recharge |
| `nikola_physics_oracle.aria` | Energy/reversibility/viscosity/resonance | run_gate2, check_resonance, drift_alert_level |
| `nikola_shadowspine.aria` | Gate 0: Ed25519+SPHINCS+ signature (stub) | run_gate0, verify_signature |
| `nikola_blacklist.aria` | Gate 1: Banned syscall pattern scan | run_gate1, is_safe, scan_source |
| `nikola_swapper.aria` | Gate 3: dlopen module hot-swap (stub) | run_gate3, swap_in, rollback |
| `nikola_sie_pipeline.aria` | Orchestrator wiring all 4 gates | run_sie_cycle |
| `shim/nikola_sie_bridge.c` | C FFI bridge to C++ gate classes | nk_verifier_*, nk_blacklist_*, nk_oracle_*, nk_swapper_* |

## Build & Run

Requires `ariac` (Aria compiler v0.8.4+) and `gcc`:

```bash
# Build FFI shim
cd shim
gcc -O2 -Wall -fPIC -c nikola_sie_bridge.c -o nikola_sie_bridge.o -lm
ar rcs libnikola_sie.a nikola_sie_bridge.o
cd ..

# Run individual module tests (no FFI needed)
ariac tests/test_types.aria -I . -o tests/test_types && ./tests/test_types
ariac tests/test_metabolic.aria -I . -o tests/test_metabolic && ./tests/test_metabolic

# Run gate tests (FFI required)
ariac tests/test_shadowspine.aria -I . -o tests/test_shadowspine -L shim -lnikola_sie && ./tests/test_shadowspine
ariac tests/test_blacklist.aria -I . -o tests/test_blacklist -L shim -lnikola_sie && ./tests/test_blacklist
ariac tests/test_swapper.aria -I . -o tests/test_swapper -L shim -lnikola_sie && ./tests/test_swapper
ariac tests/test_oracle.aria -I . -o tests/test_oracle -L shim -lnikola_sie -lm && ./tests/test_oracle

# Run full pipeline integration test
ariac tests/test_pipeline.aria -I . -o tests/test_pipeline -L shim -lnikola_sie -lm && ./tests/test_pipeline
```

## Design

- **Library modules** — no main/failsafe in library files; tests are in `tests/`
- **Single-import rule** — only import the top-level module you need; ariac
  doesn't handle diamond imports (same module imported from multiple paths)
- **FFI pattern** — `extern func:name = type(params);` calls C shim functions
  linked via `-L shim -lnikola_sie`
- **Constants from spec** — CANDLE_DRIFT_LIMIT=1e-6, REVERSIBILITY_LIMIT=1e-12,
  ATP_CYCLE_TOTAL=750, RESONANCE_AMPLITUDE_LIMIT=100
- **Stubs** — Gate 0 (crypto) and Gate 3 (dlopen) are structural stubs;
  full implementations wired when libsodium/PQC and C++ ModuleSwapper are linked
- **SIE Gate 2 ready** — physics_oracle.aria implements the full oracle
  verdict logic matching `include/nikola/autonomy/physics_oracle.hpp`

## Status

All 3 files compile and run correctly as of v0.0.5 (ariac v0.17.5, 2026-04-07).
