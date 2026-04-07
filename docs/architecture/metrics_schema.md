# Nikola Metrics Schema — v0.0.8

## Transport

All metrics are emitted as **JSON Lines** to **file descriptor 3** (`stddbg`).
Enable with `--telemetry` flag.  The daemon silently no-ops if FD 3 is not open.

Capture example:
```bash
./nikola-run --prompt "hello" --telemetry 3>metrics.jsonl
```

## Metric Types

| Type      | Fields                                            |
|-----------|---------------------------------------------------|
| `gauge`   | `ts`, `type:"gauge"`, `metric`, `value`, `unit`?  |
| `counter` | `ts`, `type:"counter"`, `metric`, `delta`          |
| `event`   | `ts`, `type:"event"`, `metric`, `payload`           |

All records include a monotonic `ts` field (Unix epoch seconds, 3 decimal places).

## Per-Tick Gauges

Emitted once per `DecisionLoop::tick()` via the `on_tick` callback:

| Metric            | Unit  | Description                                     |
|-------------------|-------|-------------------------------------------------|
| `tick.energy`     | `J`   | Total |ψ|² across all active torus nodes          |
| `tick.dopamine`   | —     | Dopamine level ∈ [0, 1]                          |
| `tick.atp`        | —     | Metabolic ATP level ∈ [0, 1]                     |
| `tick.boredom`    | —     | Boredom regulator level ∈ [0, 1]                 |
| `tick.entropy`    | `nat` | Shannon entropy of the torus wavefunction        |
| `tick.duration`   | `us`  | Mean tick wall-clock time (from ScopeProfiler)   |

## Per-Tick Counter

| Metric            | Description                       |
|-------------------|-----------------------------------|
| `tick.count`      | Cumulative tick count (delta = 1) |

## ScopeProfiler Scopes

Available via `--profile` (printed to stderr on exit).  All scopes use RAII
guards (`NIKOLA_PROFILE` macro) and record to the global `ScopeProfiler` singleton.

| Scope Name                   | Location                              |
|------------------------------|---------------------------------------|
| `DecisionLoop::tick`         | decision_loop.cpp — full tick          |
| `torus::run`                 | decision_loop.cpp — physics simulation |
| `torus::step`                | cognitive_torus.hpp — single step      |
| `torus::reseed_check`        | decision_loop.cpp — field liveness     |
| `autonomy::tick`             | decision_loop.cpp — engine update      |
| `autonomy::read_state`       | decision_loop.cpp — state snapshot     |
| `autonomy::score_candidates` | decision_loop.cpp — candidate scoring  |
| `propagator::step`           | propagator.hpp — Strang integrator     |
| `embed::nonary`              | nonary_embedder.hpp — text → nit[128]  |
| `mapper::token_to_coord`     | cognitive_core.hpp — embed → 9D coord  |
| `lmdb::save_state`           | lmdb_state_store.hpp — persist state   |
| `lmdb::load_state`           | lmdb_state_store.hpp — restore state   |
| `lmdb::save_checkpoint`      | lmdb_state_store.hpp — persist Ψ       |
| `lmdb::load_checkpoint`      | lmdb_state_store.hpp — restore Ψ       |
| `lmdb::put`                  | lmdb_state_store.hpp — raw LMDB write  |

## Wire Format Examples

```json
{"ts":1743724800.001,"type":"gauge","metric":"tick.energy","value":68890.5,"unit":"J"}
{"ts":1743724800.001,"type":"gauge","metric":"tick.dopamine","value":0.523}
{"ts":1743724800.001,"type":"gauge","metric":"tick.atp","value":0.847}
{"ts":1743724800.001,"type":"gauge","metric":"tick.boredom","value":0.312}
{"ts":1743724800.001,"type":"gauge","metric":"tick.entropy","value":7.234,"unit":"nat"}
{"ts":1743724800.001,"type":"gauge","metric":"tick.duration","value":142.5,"unit":"us"}
{"ts":1743724800.001,"type":"counter","metric":"tick.count","delta":1}
```

## Analysis

Use `scripts/plot_metrics.py` to visualize a session:
```bash
./nikola-run --prompt "think about numbers" --telemetry --ticks 500 3>session.jsonl
python3 scripts/plot_metrics.py session.jsonl
```
