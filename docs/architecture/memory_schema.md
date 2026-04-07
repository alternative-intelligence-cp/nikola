# Nikola Memory Schema — v0.0.6

_Phase 137: LMDB State Persistence_

---

## Overview

Nikola uses **two** independent LMDB environments for persistence:

1. **SemanticMemory** (`--memory-lmdb <dir>`) — Phase 136, unchanged.
   Wave-field memories keyed by Hilbert index. Single unnamed database.

2. **StateStore** (`--state-db <dir>`) — Phase 137, new.
   Full session state across five named databases. Provides cross-session
   continuity for NikolaState, wavefunction checkpoints, and autobiographical
   memory.

Keeping them separate avoids breaking Phase 136 tests and allows independent
backup/migration.

---

## StateStore LMDB Environment

**Map size:** 512 MiB (supports ~3,200 full Ψ checkpoints at 19,683 nodes)
**Max databases:** 6 (5 named + 1 reserved for future use)
**Magic prefix:** `"NKST"` (0x4E4B5354)

### Database 1: `state` — NikolaState snapshots

Stores periodic NikolaState snapshots for metabolic history and session restore.

**Key:** 8 bytes, big-endian uint64_t (tick number)

**Value layout (fixed 37 bytes + variable tokens):**
```
[0..3]    uint32_t  magic       = 0x4E4B5354 ("NKST")
[4..7]    uint32_t  version     = 1
[8..11]   float     time
[12..15]  float     torus_energy
[16..19]  float     dopamine
[20..23]  float     td_error
[24..27]  float     atp
[28..31]  float     boredom
[32..35]  float     entropy
[36]      uint8_t   last_action
[37..40]  uint32_t  n_tokens
[41..]    tokens: for each token: uint16_t len + char[len]
```

### Database 2: `checkpoint` — Ψ field snapshots

Full wavefunction checkpoint for session restore and Hamiltonian verification.

**Key:** 8 bytes, big-endian uint64_t (tick number)

**Value layout (28 bytes header + 16·N floats):**
```
[0..3]    uint32_t  magic       = 0x4E434B50 ("NCKP")
[4..7]    uint32_t  n_nodes
[8..11]   float     time_seconds
[12..19]  double    hamiltonian   (for restore verification)
[20..23]  uint32_t  grid_n        (nodes per dimension, e.g. 3)
[24..27]  uint32_t  reserved      (0, for future schema version)
[28 .. 28 + N*4)           float[] psi_real
[28 + N*4 .. 28 + 2*N*4)  float[] psi_imag
[28 + 2*N*4 .. 28 + 3*N*4) float[] vel_real
[28 + 3*N*4 .. 28 + 4*N*4) float[] vel_imag
```
Total per record: 28 + 16·N bytes (N=19,683 → 315,356 bytes ≈ 308 KiB)

### Database 3: `events` — Autobiographical life events

**Key:** 8 bytes, big-endian uint64_t (tick number)

**Value layout (variable):**
```
[0..3]    uint32_t  magic       = 0x4E455654 ("NEVT")
[4..11]   double    significance
[12..15]  int32_t   affect      (Affect enum as int)
[16..19]  uint32_t  n_tags
[20..23]  uint32_t  desc_len
[24 .. 24+desc_len)   char[] description
then for each tag:    uint16_t tag_len + char[tag_len]
then inline NikolaState snapshot (same layout as `state` DB, minus magic/version)
```

### Database 4: `skills` — Skill tracking

**Key:** variable bytes (skill name as raw UTF-8)

**Value layout (fixed 32 bytes):**
```
[0..7]    double    proficiency
[8..15]   uint64_t  last_tick
[16..23]  uint64_t  practice_count
[24..31]  uint64_t  success_count
```

### Database 5: `values` — Value tracking

**Key:** variable bytes (value name as raw UTF-8)

**Value layout (fixed 16 bytes):**
```
[0..7]    double    importance
[8..15]   uint64_t  update_count
```

---

## CLI Flags

| Flag | Description |
|------|-------------|
| `--state-db <dir>` | LMDB directory for StateStore (all 5 databases) |
| `--checkpoint-interval <N>` | Save Ψ checkpoint every N ticks (default: 100) |
| `--state-dump` | Print latest saved state and exit |

---

## Lifecycle

### Startup
1. Open LMDB environment at `--state-db` path (create if absent)
2. Load latest NikolaState from `state` DB (highest tick key)
3. Load latest Ψ checkpoint from `checkpoint` DB → restore WaveFunction
4. Load all events/skills/values into AutobiographicalMemory

### Runtime
- Save NikolaState every tick (overwrite latest)
- Save Ψ checkpoint every `checkpoint_interval` ticks
- Save autobiography events as they are recorded

### Shutdown
- Final NikolaState + Ψ checkpoint written on clean exit

---

## Version Migration

The `version` field in the state record header (currently 1) enables future
schema evolution. On load, if version > supported, log a warning and skip
the record rather than crashing.
