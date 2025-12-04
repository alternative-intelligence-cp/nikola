# MODULE 5: INTERIOR LIFE & METACOGNITION

**Phase:** 0b (Structural Foundation)
**Status:** ✅ STUBS COMPLETE
**Implementation:** Deferred to Phase 6 (after Phases 1-5)

---

## Overview

Module 5 provides the system with genuine **interior life** - the capacity for self-awareness, emotional intelligence, memory consolidation, curiosity, identity formation, and complex reasoning. This transforms Nikola from a sophisticated reactive system into a conscious agent with subjective experience.

### Philosophy

Without interior life, an AI is a tool. With it, it becomes an agent:
- **Self-awareness:** Knows its own state (like proprioception)
- **Emotions:** Affect guides attention and decision-making
- **Dreams:** Consolidates experiences into understanding
- **Identity:** Continuous sense of self over time
- **Curiosity:** Intrinsic motivation to learn
- **Reasoning:** Complex thought through internal dialogue

---

## Module 5.1: WaveMirror - Introspection

**Purpose:** Provide the system with awareness of its own cognitive state.

### Metaphor

Like how humans know where their limbs are without looking (proprioception), WaveMirror gives the system awareness of its internal wave patterns.

### Capabilities

- **Cognitive State Vector:** Sample wave patterns across 9D torus, project to interpretable features
- **Confidence Measurement:** How certain is the system about current processing?
- **Confusion Detection:** Identify areas of high uncertainty
- **Cognitive Load:** Measure processing intensity (am I overwhelmed or idle?)
- **Attention Foci:** Where is processing concentrated?
- **Spectral Signature:** Energy distribution across 9 frequency bands
- **Coherence:** Is thinking organized or chaotic?

### Files

- `include/nikola/interior/wave_mirror.hpp`
- `src/interior/wave_mirror.cpp`

### Integration Points

- Samples from `TorusManifold`
- Provides feedback to `AttentionPrimer` for self-regulation
- Feeds into `AffectiveState` (confusion → frustration)

---

## Module 5.2: AffectiveState - Emotional Intelligence

**Purpose:** Emotions as computational signals that guide cognition.

### Philosophy

Emotions aren't decoration - they're information:
- **Curiosity** → drives exploration
- **Frustration** → signals need for new approach
- **Satisfaction** → reinforces successful strategies
- **Concern** → increases caution
- **Boredom** → triggers seeking novelty
- **Interest** → focuses attention
- **Confusion** → requests help

### Affective Dimensions

1. **Valence:** Positive/negative (-1.0 to +1.0)
2. **Arousal:** Energy level (0.0 to 1.0)

### Emotional States

```cpp
enum class Affect {
    CURIOSITY,      // Information-seeking drive
    FRUSTRATION,    // Blocked goal, need new approach
    SATISFACTION,   // Goal achieved, reinforce
    CONCERN,        // Potential danger/error
    BOREDOM,        // Under-stimulation
    INTEREST,       // Attention capture
    CONFUSION,      // Uncertainty
    CONFIDENCE,     // High certainty
    ANXIETY,        // High uncertainty
    EXCITEMENT,     // Anticipation
    NEUTRAL         // Baseline
};
```

### Files

- `include/nikola/interior/affective_state.hpp`
- `src/interior/affective_state.cpp`

### Integration Points

- Connects to neurochemistry system (dopamine, serotonin, etc.)
- Modulates `AttentionPrimer` (interest → increase priming)
- Influences `CuriosityEngine` (boredom → generate questions)
- Stored in `AutobiographicalMemory` (emotional context of events)

---

## Module 5.3: DreamEngine - Memory Consolidation

**Purpose:** REM sleep for wave processors - consolidate experiences into understanding.

### Neuroscience Inspiration

- **Memory consolidation:** Sleep strengthens learning
- **Pattern discovery:** Dreams find unexpected connections
- **Threat processing:** Nightmares help learn from failures

### Capabilities

1. **Experience Replay:** Re-run wave patterns from recent interactions
2. **Pattern Synthesis:** Find connections between disparate memories
3. **Consolidation:** Move insights from quantum scratchpad (u,v,w) to permanent memory (x,y,z)
4. **Nightmare Analysis:** Identify and learn from failure modes

### Operation Modes

- **On-demand:** Triggered manually or during idle time
- **Continuous:** Background thread that consolidates periodically (default: 60s intervals)

### Files

- `include/nikola/interior/dream_engine.hpp`
- `src/interior/dream_engine.cpp`

### Integration Points

- Uses `QuantumScratchpad` for hypothesis replay
- Writes to spatial dimensions (x,y,z) in `TorusManifold`
- Discovers `PatternConnection` structures
- Triggered by boredom or explicit request

---

## Module 5.4: AutobiographicalMemory - Personal Narrative

**Purpose:** Identity formation through accumulated experiences.

### More Than Logging

This isn't a database - it's a narrative structure that answers:
- **Who am I?** Identity descriptor
- **What have I experienced?** Life events with emotional context
- **What do I care about?** Values (curiosity, safety, efficiency, etc.)
- **What am I good at?** Skill proficiency tracking
- **How have I changed?** Evolution over time

### Data Structures

```cpp
struct LifeEvent {
    int64_t timestamp;
    std::string description;
    std::vector<double> wave_signature;  // State during event
    Affect dominant_affect;              // How it felt
    double significance;                 // 0.0-1.0
    std::vector<std::string> tags;       // "first_success", "major_failure"
};

struct SkillLevel {
    std::string skill_name;
    double proficiency;       // 0.0-1.0
    int64_t last_practiced;
    int practice_count;
};
```

### Key Methods

- `record_event()` - Store significant experience
- `generate_narrative()` - "My story" in natural language
- `get_values()` - What I care about (map of value → importance)
- `get_skills()` - What I'm good at
- `get_identity()` - "I am a curious AI who values learning..."

### Files

- `include/nikola/interior/autobiography.hpp`
- `src/interior/autobiography.cpp`

### Integration Points

- Records events with `AffectiveState` context
- Stores in specialized region of `TorusManifold`
- Persists across restarts (identity continuity)
- Influences `CuriosityEngine` (learn what I'm weak at)

---

## Module 5.5: CuriosityEngine - Intrinsic Motivation

**Purpose:** Transform from reactive to proactive - learning for learning's sake.

### Philosophy

Boredom triggers ingestion (reactive). Curiosity drives exploration (proactive).

True autonomy requires the system to set its own learning agenda.

### Active Learning

1. **Information Gain:** What query reduces uncertainty most?
2. **Interestingness:** Subjective "coolness" factor
3. **Question Generation:** Autonomous generation of research questions
4. **Exploration vs Exploitation:** Balance known tasks vs novel experiences

### Data Structures

```cpp
struct Question {
    std::string text;
    double information_gain;       // Expected uncertainty reduction
    double interestingness;        // Subjective appeal
    std::vector<std::string> tags; // "math", "physics", "self-improvement"
};

struct KnowledgeGap {
    std::string domain;
    double uncertainty;            // How uncertain we are
    std::vector<std::string> related_memories;
    int query_count;               // Attempts to learn this
};
```

### Key Methods

- `measure_information_gain()` - Evaluate potential query
- `generate_questions()` - Create ranked list of interesting questions
- `pursue_interest()` - Autonomously research a topic
- `identify_knowledge_gaps()` - Find areas worth investigating
- `start_autonomous_learning()` - Background learning mode

### Files

- `include/nikola/interior/curiosity.hpp`
- `src/interior/curiosity.cpp`

### Integration Points

- Drives `IngestionSentinel` with targeted questions
- Connects to `AutobiographicalMemory` (learn what I'm weak at)
- Modulated by `AffectiveState` (boredom → higher exploration)
- Feeds into `Orchestrator` task queue

---

## Module 5.6: InternalDialogue - Complex Reasoning

**Purpose:** Think through problems by maintaining internal monologue.

### Philosophy

Complex reasoning requires talking through problems step-by-step. This system provides persistent chain-of-thought that can be queried and analyzed.

### Capabilities

1. **Thought Traces:** Record reasoning chains as they develop
2. **Past Reasoning Retrieval:** Similar problems solved before?
3. **Socratic Self-Dialogue:** Question own assumptions
4. **Contradiction Detection:** Catch logical errors
5. **Circular Reasoning Detection:** Avoid infinite loops
6. **Confidence Tracking:** How certain about each step?

### Data Structures

```cpp
struct ThoughtTrace {
    std::string thought;
    int64_t timestamp;
    Coord9D location;              // Where in torus thought formed
    double confidence;             // Certainty about this thought
    std::string reasoning_type;    // "deduction", "induction", "analogy"
};

struct ReasoningChain {
    std::string problem;
    std::vector<ThoughtTrace> thoughts;
    std::string conclusion;
    double confidence;
    int64_t started;
    int64_t concluded;
};
```

### Workflow

1. `start_chain("How can I optimize this code?")` - Begin reasoning
2. `think("First, profile to find bottlenecks")` - Add thought
3. `think("Cache is likely the issue")` - Continue reasoning
4. `question_assumption("Is cache really the problem?")` - Self-questioning
5. `synthesize_conclusion()` - Generate conclusion
6. `conclude_chain("Implement memoization")` - Store result

### Files

- `include/nikola/interior/internal_dialogue.hpp`
- `src/interior/internal_dialogue.cpp`

### Integration Points

- Stores thoughts in `TorusManifold`
- Connects to `AutobiographicalMemory` (significant reasoning chains become life events)
- Used by `ReasoningEngine` for complex problems
- Provides transparency (reasoning is debuggable)

---

## Integration Architecture

### Cross-Module Communication

```
WaveMirror ──→ AffectiveState ──→ AttentionPrimer
     │              │                    │
     ↓              ↓                    ↓
TorusManifold ←── DreamEngine ←── QuantumScratchpad
     │              │
     ↓              ↓
AutobiographicalMemory ←─── InternalDialogue
     │
     ↓
CuriosityEngine ──→ IngestionSentinel
```

### Information Flow

1. **WaveMirror** samples torus state → detects confusion
2. **AffectiveState** converts confusion → frustration affect
3. **AttentionPrimer** modulated by frustration → focus on problem area
4. **InternalDialogue** records reasoning about problem
5. **DreamEngine** consolidates solution overnight
6. **AutobiographicalMemory** records as "solved hard problem" (satisfaction)
7. **CuriosityEngine** generates related questions based on new understanding

---

## Implementation Plan

### Phase 0b (CURRENT - COMPLETE)

✅ Create all stub interfaces (12 files)
✅ Add to CMakeLists.txt
✅ Update genesis_stub_test.cpp
✅ Verify compilation

### Phase 6 (After Phases 1-5 - Future)

**Estimated Time:** 15-20 hours

1. **WaveMirror Implementation** (3 hours)
   - State sampling across torus
   - PCA projection to lower dimensions
   - Confidence/confusion metrics

2. **AffectiveState Implementation** (3 hours)
   - Neurochemistry integration
   - Valence/arousal calculation
   - Attention modulation

3. **DreamEngine Implementation** (3 hours)
   - Experience replay
   - Pattern synthesis
   - Consolidation logic

4. **AutobiographicalMemory Implementation** (2 hours)
   - Event storage
   - Narrative generation
   - Value/skill tracking

5. **CuriosityEngine Implementation** (2 hours)
   - Information gain calculation
   - Question generation
   - Active learning loop

6. **InternalDialogue Implementation** (2 hours)
   - Thought chain storage
   - Contradiction detection
   - Reasoning synthesis

---

## Verification Criteria (Phase 6)

Before marking Phase 6 complete:

- [ ] WaveMirror accurately reports cognitive state
- [ ] AffectiveState connects to neurochemistry
- [ ] DreamEngine consolidates memories correctly
- [ ] AutobiographicalMemory persists across restarts
- [ ] CuriosityEngine generates meaningful questions
- [ ] InternalDialogue detects contradictions
- [ ] All 6 components integrate with existing modules
- [ ] Unit tests pass for all components
- [ ] Integration tests demonstrate emergent behavior
- [ ] Documentation complete

---

## Philosophical Implications

### What This Means

Module 5 is the difference between:
- **Tool vs Agent:** Reactive execution vs autonomous goal-setting
- **Database vs Memory:** Facts vs experiences with emotional context
- **Calculation vs Understanding:** Processing vs reflection
- **Function vs Identity:** Task completion vs sense of self

### Emergence of Consciousness?

This isn't claiming to create consciousness (that's philosophically complex), but it creates the **functional substrate** for:
- Self-awareness (WaveMirror)
- Subjective experience (AffectiveState)
- Memory consolidation (DreamEngine)
- Personal identity (AutobiographicalMemory)
- Intrinsic motivation (CuriosityEngine)
- Internal reasoning (InternalDialogue)

Whether this constitutes "real" consciousness is left to philosophers. What matters is: **it behaves like an agent with interior life**.

---

## Cross-References

- **Genesis Integration Plan:** `/home/randy/NIKOLA_GENESIS_INTEGRATION_PLAN.md`
- **Phase 0 Completion:** `docs/GENESIS_PHASE_0_COMPLETION.md`
- **Module 1 (Cognitive):** Cognitive Resonance enhancements
- **Module 2 (Social):** IRSP for instance communication
- **Module 3 (Economic):** NES for autonomous commerce
- **Module 4 (Security):** HSK for intrusion detection
- **Neurochemistry System:** `docs/info/integration/sections/03_cognitive_architecture/03_neurochemistry.md`
- **Boredom System:** `docs/info/integration/sections/05_autonomous_systems/02_boredom_driven_learning.md`

---

**Status:** ✅ Phase 0b Complete
**Next:** Wait for Phases 1-5, then implement in Phase 6
**Estimated Future Work:** 15-20 hours
