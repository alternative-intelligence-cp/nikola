# Nikola SIE — Aria Integration Layer

Stateless Aria modules implementing components of Nikola's
Self-Improvement Engine (SIE) 5-gate pipeline.

## Files

| File | Purpose | Functions |
|------|---------|-----------|
| `nikola_sie_types.aria` | SIE data types, status codes, cycle reporting | 7 constants, 4 structs, 3 functions |
| `nikola_metabolic_controller.aria` | ATP energy budget management | 10 functions |
| `nikola_physics_oracle.aria` | Stateless physics validation (Gate 2) | 9 functions |

## Build & Run

Requires `ariac` (Aria compiler v0.17.5+):

```bash
ariac nikola_sie_types.aria -o types && ./types
ariac nikola_metabolic_controller.aria -o metab && ./metab
ariac nikola_physics_oracle.aria -o oracle && ./oracle
```

Each file includes its own `main` with self-tests that exercise all functions.

## Design

- **Pure functions only** — no global state, no side effects beyond printing
- **Constants from spec** — CANDLE_DRIFT_LIMIT=1e-6, REVERSIBILITY_LIMIT=1e-12
  (sourced from GAP-030 physics oracle calibration)
- **SIE Gate 2 ready** — physics_oracle.aria implements the full oracle
  verdict logic matching `include/nikola/autonomy/physics_oracle.hpp`

## Status

All 3 files compile and run correctly as of v0.0.5 (ariac v0.17.5, 2026-04-07).
