# Nikola v0.0.19 Handoff — Aria Specialist Integration

_Completed: 2026-04-03_

---

## What Was Completed

### Phase 145: Aria Specialist ↔ SIE Pipeline Wiring

v0.0.19 connects the Aria specialist model to Nikola's Self-Improvement Engine
(SIE) so that Nikola can autonomously generate, compile-validate, and persist
Aria code proposals. This is the foundation for the autonomous
generate→compile→evaluate→retrain feedback loop.

### New Components

| File | Purpose |
|------|---------|
| `include/nikola/aria/compile_validator.hpp` | C++ subprocess wrapper for ariac. Writes source to tempfile, invokes ariac, captures exit code + errors, cleans up. Thread-safe (no shared state). Timeout: 30s. |
| `include/nikola/aria/specialist_interface.hpp` | C++ client for the Python specialist server (JSON-Lines over stdin/stdout pipes). Spawns server.py, waits for ready signal, sends request/response pairs. |
| `include/nikola/aria/code_proposal_store.hpp` | LMDB-backed persistence for code proposals + compile results. Auto-increment IDs, big-endian keys, pack/unpack serialisation. 128MB map. |

### Modified Components

| File | Change |
|------|--------|
| `include/nikola/autonomy/decision_loop.hpp` | Added `GENERATE_CODE = 10` to ActionType enum. Added `score_generate_code()` scorer. Added config fields: `min_generate_interval_s`, `specialist_server_path`, `ariac_path`, `proposal_store_path`. Added cooldown timer + `aria_specialist_enabled_` flag. |
| `src/autonomy/decision_loop.cpp` | Implemented `score_generate_code()`: fires when specialist enabled, boredom > 0.4, ATP ≥ 0.30, cooldown elapsed. Score = boredom × ATP × 0.5. Added GENERATE_CODE to candidate array (now 10 candidates). Added `build_payload` case and cooldown update. Constructor initialises generate timer and enabled flag. |
| `CMakeLists.txt` | Added Phase145AriaSpecialist unit test + IntegrationAriaSpecialist integration test. |

### Utility Functions

- `extract_code_block()`: Extracts Aria source from model response — handles fenced ```aria blocks, generic fenced blocks, and raw Aria code detected by keyword (func:, use, extern, int32:, string:).

### Test Summary

| Test Suite | Cases | Assertions | Status |
|-----------|-------|------------|--------|
| Phase145AriaSpecialist (unit) | 60 | ~400 | ✅ All pass |
| IntegrationAriaSpecialist | 14 | ~150 | ✅ All pass |
| **Full CTest suite** | **163** | | **163/163 pass** |

### Test Coverage by Section

- **§A (CompileValidator)**: 12 tests — construction, custom paths, missing compiler, success/fail validation, output parsing, env var override, tempfile cleanup
- **§B (CodeProposalStore)**: 12 tests — CRUD, auto-increment, count/success metrics, export, persistence across reopen, serialisation symmetry, bad magic rejection
- **§C (extract_code_block)**: 12 tests — fenced aria, generic fence, raw keywords, empty input, preference order, multiple blocks
- **§D (GENERATE_CODE ActionType)**: 12 tests — enum value, action_name, distinct from all others, NikolaState default, DecisionResult carrier, config fields, scoring contracts
- **§E (SpecialistInterface)**: 12 tests — construction, path defaults, start failure, ask without start, env override, destructor safety, timeout test, result types
- **Integration §A-E**: 14 tests — validate→store round-trip, mixed success rates, DecisionLoop with specialist disabled/enabled, full extract→validate→store pipeline, metrics accumulation

---

## Architecture

```
 ┌──────────────────┐
 │ DecisionLoop     │   GENERATE_CODE scored when boredom > 0.4 + ATP ≥ 0.30
 │  tick()          │
 └────────┬─────────┘
          │ winner = GENERATE_CODE
          ↓
 ┌──────────────────┐
 │ SpecialistIface  │   JSON-Lines: {"instruction":"...", "context":"..."}
 │  ask()           │   ← spawns server.py as subprocess
 └────────┬─────────┘
          │ raw response text
          ↓
 ┌──────────────────┐
 │ extract_code_block│   regex: ```aria\n...\n``` or keyword detection
 └────────┬─────────┘
          │ Aria source code
          ↓
 ┌──────────────────┐
 │ AriaCompileValid │   subprocess: ariac <tmp>.aria -o <tmp>.out
 │  validate()      │   captures exit code + stderr
 └────────┬─────────┘
          │ CompileResult {success, errors, warnings}
          ↓
 ┌──────────────────┐
 │ CodeProposalStore│   LMDB: proposals DB, 128MB
 │  store()         │   packed binary record
 └──────────────────┘
```

---

## Key Decisions

1. **Header-only implementation**: All three new classes are header-only (inline in .hpp). This avoids adding new .cpp files to the ARIA_SOURCES list, which is currently all stubs. The headers compile cleanly with the existing nikola_core static library.

2. **GENERATE_CODE score formula**: `boredom × ATP × 0.5`. This means GENERATE_CODE only wins over SILENT (0.3 + threshold) when boredom is ≥ 0.64 at full ATP, or when boredom is very high (0.9+) at moderate ATP (0.7+). This prevents code generation from dominating other cognitive actions.

3. **Specialist enabled flag**: `aria_specialist_enabled_` is true when EITHER `specialist_server_path` or `ariac_path` is non-empty in the config. This allows compile-only validation without a running specialist, and vice versa.

4. **LMDB directory mode**: `mdb_env_open()` uses directory mode (not `MDB_NOSUBDIR`) since callers pass directory paths matching the existing LMDB convention.

5. **No full execution pipeline yet**: v0.0.19 provides the infrastructure and scoring integration. The actual GENERATE_CODE side-effect (specialist query → compile → persist → dopamine feedback) will be wired in a follow-up. The scoring and candidate selection work now; the execution path returns a descriptive payload string.

---

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `ARIAC_BIN` | Path to ariac compiler | `~/Workspace/REPOS/aria/build/ariac` |
| `ARIA_SPECIALIST_SERVER` | Path to specialist server.py | `~/Workspace/REPOS/aria-specialist/scripts/server.py` |

---

## What's Next

- Wire GENERATE_CODE execution side-effect: specialist → compile → persist → dopamine
- Connect CodeProposalStore export to self_improve.py training data
- Track compile success rate over iterations (target: 23% → 40%+)
- Integrate with SIE Gate 1 (CodePatternBlacklist) for security scan before compile
- Full autonomous improvement cycle: 100 samples → filter → retrain
