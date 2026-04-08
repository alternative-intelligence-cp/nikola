# Nikola v0.0.9 — Cognitive Baseline Report

**Date:** $(date -I)
**Version:** v0.0.8 (pre-tuning)
**Build:** g++ 13.3, CUDA 13.2, RTX 3090 (sm_86), Eigen 5.0.0
**Config:** --ticks 200 --emit-all --stream --profile

---

## 1. Test Prompts & Final Answers

| # | Prompt | Final Answer | Time |
|---|--------|-------------|------|
| 1 | What is the meaning of consciousness? | *(last action: EXPLORE "seed=world")* | 2336ms |
| 2 | Tell me about the number seven | *(last action: RECALL)* | 1404ms |
| 3 | How do waves propagate through space? | *(last action: EXPLORE "seed=world")* | 2368ms |
| 4 | What are you thinking about right now? | Drawn to human | 3889ms |
| 5 | Explain the relationship between energy and entropy | *(last action: RECALL)* | 1853ms |
| 6 | Do you dream? | *(last action: RECALL)* | 1767ms |
| 7 | What is 2 + 2? | I want to understand human better | 4276ms |
| 8 | Describe a beautiful pattern you can imagine | *(last action: RECALL)* | 2015ms |
| 9 | Why does mathematics work? | Drawn to human | 1624ms |
| 10 | Hello, who are you? | *(last action: RECALL)* | 2567ms |

**Key observation:** 6/10 prompts end on a RECALL or EXPLORE action, not an
EMIT_THOUGHT. Only 4/10 produce a final EMIT, and those are single-word
template fills unrelated to the prompt.

---

## 2. Quality Scores (0-3 each)

| # | Coherent | Relevant | Complete | Total |
|---|----------|----------|----------|-------|
| 1 | 0 | 0 | 0 | 0/9 |
| 2 | 0 | 0 | 0 | 0/9 |
| 3 | 0 | 0 | 0 | 0/9 |
| 4 | 1 | 0 | 0 | 1/9 |
| 5 | 0 | 0 | 0 | 0/9 |
| 6 | 0 | 0 | 0 | 0/9 |
| 7 | 1 | 0 | 0 | 1/9 |
| 8 | 0 | 0 | 0 | 0/9 |
| 9 | 1 | 0 | 0 | 1/9 |
| 10 | 0 | 0 | 0 | 0/9 |

**Baseline aggregate: 3/90 (3.3%)**

Coherent=1 given when output is at least a grammatical English phrase
(e.g. "Drawn to human"). Relevant=0 across the board because no output
relates to its prompt. Complete=0 because all responses are single-word
template fills.

---

## 3. Action Distribution (200 ticks per prompt, averaged)

| Action | Mean % | Role |
|--------|--------|------|
| SILENT | 57% | No action beats threshold |
| RECALL_MEMORY | 19% | Memory resonance dominates |
| EXPLORE | 11% | Boredom-driven seed injection |
| EMIT:DRAWN | 10% | Dopamine template (most common emit) |
| EMIT:other | 2% | RESONATING, HARD_TO_HOLD, UNDERSTAND |
| REASON | 0.6% | Entropy gate + 3s cooldown → ~1 fire |
| STORE_MEMORY | 0.5% | Dopamine spike + 30s cooldown → 1 fire |

---

## 4. Root Cause Analysis

### 4a. SILENT Dominance (57%)

EMIT_THOUGHT = dopamine × boredom × atp × 1.5

At startup: 0.5 × 0.40 × 1.0 × 1.5 = **0.30** — exactly equals SILENT.
Must exceed SILENT + action_threshold = 0.30 + 0.05 = **0.35** to fire.
Requires boredom ≥ 0.47 (at baseline dopamine=0.5, atp=1.0).

Boredom accumulates at ~0.01/tick, so ~7+ ticks before first EMIT is possible.

### 4b. RECALL Dominance (19%)

score_recall = resonance × atp × 1.5

With even moderate resonance (0.3), this scores 0.3 × 1.0 × 1.5 = **0.45**,
comfortably beating SILENT (0.35). Memory has stored records from the initial
prompt injection and from STORE_MEMORY, so resonance is almost always non-zero.

RECALL fires a superposition of remembered wave onto the torus but produces
NO output tokens and NO emitted thought. It changes the field but steals ticks
that could be used for EXPLORE → EMIT sequences.

### 4c. Single-Word Output

The EXPLORE → EMIT pipeline:
1. EXPLORE finds a seed token → injects pulse → warm decode → 1 token
2. Next EMIT_THOUGHT receives that 1 token → ThoughtComposer template
3. Result: "Drawn to {single_word}"

Cold decode (s.tokens) also returns ≤1 token most ticks because:
- decode() probes top-20 hot nodes but most don't match any vocabulary word
- The 209-word vocabulary is sparse relative to the 19,683-node torus grid

### 4d. No Prompt Relevance

The stimulus injection (HolographicInjector) creates initial field structure,
but it rapidly disperses. stimulus_explore_count_ biases first 3 EXPLOREs
toward the prompt's semantic neighbourhood, but:
- RECALL fires ~6× more than EXPLORE early on, consuming those early ticks
- By the time EXPLORE fires, boredom has risen and noise_ratio increases
- The prompt's semantic signal is washed out before it can influence EMIT

### 4e. REASON Underutilized

REASON fires 1-2 times per 200-tick run (3-second cooldown vs 1.4-4.3s runs).
When it fires it runs NPT attention which structures the field, but the result
doesn't directly produce tokens — it just blends back into the torus.

---

## 5. Performance Profile (averaged across 10 prompts)

| Scope | Mean (µs) | % of 1kHz Budget |
|-------|-----------|-----------------|
| DecisionLoop::tick | 13,000 | 1300% |
| embed::nonary | 7,500 | 750% |
| torus::run | 2,800 | 280% |
| autonomy::tick | 78 | 7.8% |
| autonomy::read_state | 73 | 7.3% |
| autonomy::score_candidates | 37 | 3.7% |
| torus::reseed_check | 24 | 2.4% |

Effective rate: ~480 Hz (GPU). No performance issues for tuning phase.

---

## 6. Tuning Targets (derived from root causes)

| Priority | Target | Rationale |
|----------|--------|-----------|
| **P0** | Reduce RECALL dominance | Steals 19% of ticks without emitting |
| **P0** | Improve EMIT scoring | Needs to beat SILENT earlier |
| **P1** | Reduce REASON cooldown | 1 fire per run wastes the NPT |
| **P1** | Increase prompt grounding | stimulus_explore_count limit too low |
| **P2** | Improve cold decode yield | More vocabulary matches per tick |
| **P2** | Tune NPT temperature | Sharper attention → better field structure |
| **P3** | SSM GAMMA tuning | Affects temporal memory of the state-space |
| **P3** | Damping coefficient | Affects how long field structures persist |
