# Nikola Developer Tooling — v0.0.12

Three new diagnostic binaries ship with v0.0.12.  All build from CMake alongside
`nikola-run` and `nikola-train`.

---

## 1. `nikola-state-dump` — System State Inspector

Boots a WaveFunction + AutonomyEngine, runs N ticks, then prints a full system
snapshot including physics, metabolic state, Ψ field heatmap, and optional LMDB
memory statistics.

### Usage

```bash
nikola-state-dump [OPTIONS]

  --ticks N          Number of physics ticks (default 100)
  --steps N          Propagator steps per tick (default 50)
  --json             Machine-readable JSON output
  --memory-lmdb PATH Load LMDB and show memory stats
  --help             Show usage
```

### Human-readable output

```
╔══════════════════════════════════════════════════════════╗
║          Nikola State Dump — v0.0.12                    ║
╠══════════════════════════════════════════════════════════╣
║  Grid:  3^9 = 19683 nodes                              ║
║  PHYSICS: H₀, H_final, drift, |Ψ|², kinetic E ...     ║
║  METABOLIC: ATP, dopamine, serotonin, boredom ...      ║
║  Ψ FIELD: 2D heatmap (dims 0,1 × 2,3, ░▒▓█)          ║
║  HOT NODES: Top 5 by |Ψ|²                             ║
║  MEMORY: record count, avg strength (if LMDB given)    ║
╚══════════════════════════════════════════════════════════╝
```

### JSON mode

```json
{
  "version": "0.0.12",
  "physics": { "h0": ..., "h_final": ..., "drift_pct": ... },
  "metabolic": { "atp": ..., "dopamine": ..., ... },
  "memory": { "record_count": 0, "avg_strength": 0.0 },
  "hot_nodes": [ { "index": 10474, "psi_sq": 0.92 }, ... ]
}
```

---

## 2. `nikola-dap` — Debug Adapter Protocol Server

Full DAP server over **stdio** for VS Code debugging.  Presents the 9D torus
physics simulation as a debuggable program where each "tick" is a line of
execution.

### Quick start

1. Build: `cd build && cmake .. && make nikola_dap_server`
2. Add `.vscode/launch.json` (shipped in repo):
   ```json
   {
     "name": "Nikola DAP — Physics Debugger",
     "type": "nikola-dap",
     "request": "launch",
     "program": "${workspaceFolder}/build/nikola-dap",
     "args": ["--ticks", "1000", "--steps", "50"],
     "stopOnEntry": true
   }
   ```
3. Press F5

### Supported DAP commands

| Command             | Behavior                                        |
|---------------------|-------------------------------------------------|
| `initialize`        | Returns capabilities + nikola extensions         |
| `configurationDone` | Sends `stopped` event on entry                   |
| `threads`           | Single "Nikola Physics" thread                   |
| `stackTrace`        | Current tick as stack frame                      |
| `scopes`            | Physics (1), Metabolic (2), Memory (3), Ψ (4)   |
| `variables`         | Per-scope variable listing                       |
| `evaluate`          | Expressions: `H`, `drift`, `tick`, `atp`, `dopamine`, `node:N` |
| `setBreakpoints`    | Break on tick N (line = tick number)             |
| `continue`          | Run until breakpoint or max ticks                |
| `next`/`stepIn`     | Advance one tick                                 |
| `pause`             | Stop running                                     |

### Custom extensions

| Request                     | Response                          |
|-----------------------------|-----------------------------------|
| `nikola/gridSnapshot`       | Full Ψ field from DebugAdapter    |
| `nikola/hamiltonianState`   | Energy + drift summary            |

---

## 3. `nikola-diag` — Diagnostic CLI

Quick system health, performance, and physics integrity checks.

### Usage

```bash
nikola-diag [OPTIONS]

  --health         Build config, ORT/CUDA/LMDB availability, bootstrap
  --benchmark      1000-tick performance benchmark (ticks/sec, μs/step)
  --physics-check  Standard Candle (energy) + probability + reversibility
  --all            Run all three checks
  --json           JSON output
  --help           Show usage
```

### Example: `nikola-diag --all`

```
┌─────────────────────────────────────────────┐
│  Health Check                               │
│  Bootstrap: ✓  ORT: ✓  CUDA: ✓  LMDB: ✓   │
├─────────────────────────────────────────────┤
│  Benchmark                                  │
│  1000 ticks → 11.4 ticks/sec, 1753 μs/step │
├─────────────────────────────────────────────┤
│  Physics Check                              │
│  Energy drift: -28.9%  ✗ (expected at 50×)  │
│  Probability:  -99.4%  ✗ (expected at 50×)  │
│  Reversibility: 2.2e-7 ✓                   │
└─────────────────────────────────────────────┘
```

> **Note:** Energy and probability drift at 50 steps/tick × 1000 ticks is
> expected — the current propagator is not symplectic.  The reversibility
> test (forward + backward) passes cleanly, confirming time symmetry.

---

## Build

```bash
cd build && cmake .. && make -j$(nproc) nikola_state_dump nikola_dap_server nikola_diag
```

All three are installed alongside `nikola-run` and `nikola-train` via
`make install`.
