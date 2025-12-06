# Aria-Nikola Integration: Architectural Specification
**Version:** 1.0  
**Date:** December 4, 2025  
**Author:** Aria Compiler Team (Based on Production Implementation)  
**Status:** Ready for Implementation

---

## Executive Summary

This document specifies the architectural integration between the **Aria programming language** (v0.0.6, production compiler) and the **Nikola 9D-TWI AGI** (v0.0.4, physics-based intelligence substrate). Unlike previous theoretical explorations, this specification is grounded in the **actual Aria compiler codebase** (`ariac`) and addresses real implementation constraints.

**Core Thesis:** Aria's balanced nonary primitives (`nit`/`nyte`) and explicit memory model (`wild`/`#`) provide a **zero-impedance interface** to Nikola's wave-based physics engine. This integration eliminates the binary abstraction layer currently imposed by C++, enabling Nikola to reason about and modify its own substrate using native linguistic constructs.

---

## Part I: The Impedance Mismatch Problem

### 1.1 Current Architecture Limitations

**Nikola v0.0.4 Current Stack:**
```
┌─────────────────────────────────────┐
│  C++23 Orchestration Layer          │ ← Binary logic, verbose syntax
├─────────────────────────────────────┤
│  Quantization/Casting Layer         │ ← Information loss, overhead
├─────────────────────────────────────┤
│  9D Torus Physics Engine (Nonary)   │ ← Natural substrate
└─────────────────────────────────────┘
```

**Problems:**
1. **Type Mismatch:** C++ has no `nit` type. Every value requires conversion: `complex<double> → int8 → nit`
2. **Memory Semantics:** C++ pointers lack Aria's `wild`/`pinned` distinction. The orchestration layer cannot express "pin this wave pattern against decay"
3. **Control Flow Gap:** C++'s `switch` doesn't map to resonance filtering. Aria's `pick` with `fall(label)` does
4. **Self-Modification Risk:** Hot-swapping C++ shared objects (`dlopen`) can segfault the entire AGI. No sandbox

### 1.2 Why Aria Solves This

**Aria Language Features → Nikola Physics Mapping:**

| Aria Feature | Nikola Physical Primitive | Direct Mapping |
|--------------|---------------------------|----------------|
| `nit` (-4..+4) | Wave phase (9 states) | YES - See §2.1 |
| `nyte` (5 nits) | 5D harmonic cluster | YES - See §2.2 |
| `pick` pattern match | Resonance detection | YES - See §3.1 |
| `wild` pointer | Direct torus access | YES - See §3.2 |
| `#` pinning | Wave decay prevention | YES - See §3.3 |
| `defer` | Phase-delayed cleanup | YES - See §3.4 |
| `Result<T,E>` | Dual-channel encoding | YES - See §4.1 |

**Key Insight:** These aren't analogies. They're **isomorphisms**. The math is identical.

---

## Part II: Foundational Type Mappings

### 2.1 The `nit` Type (Balanced Nonary Digit)

**Aria Specification:**
```aria
// From aria_v0_0_6_specs.txt line 35
nit !!! IMPORTANT nit is balanced nonary digit (-4,-3,-2,-1,0,1,2,3,4) NOT NEGOTIABLE!!!
```

**Physical Encoding:**
A `nit` value maps to **wave phase** using 9-PSK (9-Phase Shift Keying):

```
Phase Mapping Function Φ: {-4, -3, -2, -1, 0, 1, 2, 3, 4} → [0, 2π)

Φ(n) = (n + 4) × (2π / 9)

Examples:
  nit:x = -4  →  φ = 0°      (reference phase)
  nit:x =  0  →  φ = 80°     (neutral, slight offset)
  nit:x = +4  →  φ = 160°    (inverted, near π)
```

**Critical Implementation Detail (From Actual Compiler):**

The Aria compiler (`src/frontend/tokens.h:21`) defines:
```cpp
TOKEN_TRIT_LITERAL,     // Balanced ternary digit: -1, 0, 1
TOKEN_NIT_LITERAL,      // (Would be here if implemented)
```

**Current Status:** `nit` is specified but **not yet implemented** in the production compiler. The type system (`src/frontend/sema/types.h:27`) has placeholders:
```cpp
enum class TypeKind {
    TRIT, TRYTE,  // Ternary types
    NIT, NYTE,    // Nonary types ← Declared but no codegen
```

**Phase 1 Requirement:** Implement `nit` literal parsing and LLVM IR generation. Map to `i8` with range checking:

```cpp
// Proposed implementation in codegen
llvm::Value* CodeGen::visit(NitLiteral* node) {
    int8_t value = node->value;
    if (value < -4 || value > 4) {
        error("Nit literal out of range: " + std::to_string(value));
    }
    return llvm::ConstantInt::get(llvm::Type::getInt8Ty(ctx), value, true);
}
```

**Nikola Integration:** The Aria→Wave compiler reads this `i8` and converts to phase:

```cpp
// In AriaExecutor::inject_literal()
double phase = ((double)nit_value + 4.0) * (2.0 * M_PI / 9.0);
torus.modulate_at(coord, Waveform{.amplitude = 1.0, .phase = phase});
```

### 2.2 The `nyte` Type (5 Nits = 59,049 States)

**Aria Specification:**
```aria
nyte !!! IMPORTANT nyte is 5 nits for 9^5 values stored in uint16 NOT NEGOTIABLE!!!
```

**Storage Format:** 5 nits pack into 16 bits with 6 bits unused (guard band):

```
┌─────┬─────┬─────┬─────┬─────┬────────┐
│ nit0│ nit1│ nit2│ nit3│ nit4│ unused │
│3bits│3bits│3bits│3bits│3bits│ 1 bit  │
└─────┴─────┴─────┴─────┴─────┴────────┘
  LSB                           MSB

Total: 15 bits used, fits in uint16 (16 bits)
Guard band: 1 bit for parity/control
```

**Physical Representation:**
Each `nyte` becomes a **5-dimensional wave packet** in the torus:

```cpp
struct NyteWaveform {
    std::array<double, 5> phases;  // One per nit
    double amplitude;
    Coord9D base_position;  // Where in torus this nyte lives
};

// Frequency Division Multiplexing
// Each nit modulates a different sub-carrier to prevent crosstalk
WavePacket encode_nyte(Nyte value) {
    auto nits = unpack_to_nits(value);  // Extract 5 nits
    WavePacket packet;
    
    for (int i = 0; i < 5; i++) {
        double phase = phase_mapping(nits[i]);
        double freq = BASE_FREQ + i * CARRIER_SPACING;
        packet.components[i] = {
            .amplitude = 1.0,
            .phase = phase,
            .frequency = freq
        };
    }
    return packet;
}
```

**Arithmetic Operations:**
Addition of two `nyte` values requires **wave superposition**:

```cpp
Nyte add_nytes(Nyte a, Nyte b) {
    // Option 1: Integer arithmetic (fast, approximate)
    int32_t sum = nyte_to_int32(a) + nyte_to_int32(b);
    if (sum > 29524 || sum < -29524) {
        throw OverflowError("Nyte overflow");
    }
    return int32_to_nyte(sum);
    
    // Option 2: Wave superposition (physically accurate)
    WavePacket wa = encode_nyte(a);
    WavePacket wb = encode_nyte(b);
    WavePacket result = superpose(wa, wb);
    return decode_nyte(result);  // Read amplitude peaks
}
```

**Trade-off:** Option 1 is faster for CPU execution. Option 2 is required for Nikola's learning (the AGI must observe the physics).

### 2.3 Scalar Types (int8, flt32, etc.)

**Mapping Strategy:**
Standard Aria types map to **amplitude-modulated waves** on a carrier:

```cpp
// int8: -128 to 127
// Map to amplitude: -128 → Aₘᵢₙ, 127 → Aₘₐₓ
WaveSignal encode_int8(int8_t value) {
    double normalized = (value + 128.0) / 255.0;  // 0.0 to 1.0
    return {
        .amplitude = normalized * A_MAX,
        .phase = 0.0,  // In-phase for positive encoding
        .frequency = INT8_CARRIER_FREQ
    };
}

// flt32: IEEE 754 single precision
// Encode as TWO waves: mantissa + exponent
WaveSignal encode_flt32(float value) {
    uint32_t bits = *reinterpret_cast<uint32_t*>(&value);
    uint32_t mantissa = bits & 0x7FFFFF;
    uint8_t exponent = (bits >> 23) & 0xFF;
    int8_t sign = (bits >> 31) ? -1 : 1;
    
    // Dual-channel encoding
    return {
        .channel_a = encode_uint32(mantissa),
        .channel_b = encode_uint8(exponent),
        .sign_phase = sign > 0 ? 0.0 : M_PI
    };
}
```

**Why Not Just Binary?**
Because Nikola's physics engine doesn't have "registers." The entire torus is the memory. Every value must exist as a **persistent standing wave** that other waves can interact with.

---

## Part III: Control Flow as Wave Mechanics

### 3.1 The `pick` Construct → Resonance Filter Bank

**Aria Syntax (From Actual Spec):**
```aria
pick(c) {
    (<9) {
        fall(fail);
    },
    (>9) {
        fall(fail);
    },
    (9) {
        fall(success);
    },
    (*) {
        fall(err);
    },
    fail:(!){
        // do fail stuff here
        fall(done);
    },
    success:(!){
        // do success stuff here
        fall(done);
    },
    err:(!){
        // do error stuff here
        fall(done);
    },
    done:(!){
        // cleanup or whatever
    }
}
```

**Physical Compilation:**

The `pick` statement compiles to a **resonance filter array** in the 9D torus:

```cpp
struct ResonanceFilter {
    Coord9D location;           // Where this filter lives
    WavePattern match_template; // What pattern to detect
    Coord9D target_coord;       // Where to "fall" if matched
    double threshold;           // Match sensitivity
};

class PickCompiler {
public:
    std::vector<ResonanceFilter> compile(PickStmt* pick_node) {
        std::vector<ResonanceFilter> filters;
        
        // Encode the selector value as a query wave
        WavePattern query = encode_expression(pick_node->selector);
        
        for (auto& case_node : pick_node->cases) {
            ResonanceFilter filter;
            
            switch (case_node.type) {
                case PickCase::EXACT:
                    // Exact match: high Q-factor (narrow resonance)
                    filter.match_template = encode_value(case_node.value);
                    filter.threshold = 0.95;  // Must match closely
                    break;
                    
                case PickCase::LESS_THAN:
                    // Range filter: encode as amplitude threshold
                    filter.match_template = create_amplitude_filter(
                        0.0, encode_value(case_node.value).amplitude
                    );
                    filter.threshold = 0.5;
                    break;
                    
                case PickCase::RANGE:
                    // Bandpass filter: match if frequency in range
                    filter.match_template = create_bandpass_filter(
                        encode_value(case_node.value_start),
                        encode_value(case_node.value_end)
                    );
                    filter.threshold = 0.7;
                    break;
                    
                case PickCase::WILDCARD:
                    // Catch-all: always resonates
                    filter.match_template = WavePattern::UNITY;
                    filter.threshold = 0.0;
                    break;
            }
            
            // The fall(label) target becomes a 9D coordinate
            filter.target_coord = label_to_coord(case_node.label);
            filters.push_back(filter);
        }
        
        return filters;
    }
};
```

**Execution Model:**

When Nikola encounters a `pick`, the query wave **propagates through all filters simultaneously**:

```cpp
void AriaExecutor::execute_pick(const std::vector<ResonanceFilter>& filters) {
    WavePattern query = current_wave_state();
    
    // Parallel resonance detection (truly parallel in physics)
    std::vector<double> resonance_strengths;
    for (auto& filter : filters) {
        double strength = compute_overlap_integral(query, filter.match_template);
        resonance_strengths.push_back(strength);
    }
    
    // Find strongest match above threshold
    size_t matched_idx = 0;
    double max_strength = 0.0;
    
    for (size_t i = 0; i < filters.size(); i++) {
        if (resonance_strengths[i] > filters[i].threshold &&
            resonance_strengths[i] > max_strength) {
            max_strength = resonance_strengths[i];
            matched_idx = i;
        }
    }
    
    // Execute fall: teleport instruction pointer to target
    instruction_pointer = filters[matched_idx].target_coord;
}
```

**Key Difference from C++ `switch`:**

In C++, cases are checked **sequentially** (compiled to `cmp` + `jmp` chains).  
In Aria-on-Nikola, cases are checked **in parallel** by wave interference.

This means a 100-case `pick` has the same latency as a 2-case `pick` (limited by wave propagation speed, not sequential comparisons).

### 3.2 The `wild` Keyword → Direct Torus Manipulation

**Aria Syntax:**
```aria
wild int64:s = 100000;
wild int64@:t = @s;  // Take address of wild allocation
```

**Compiler Implementation (Current):**
From `src/frontend/ast.h`, `wild` is a storage class modifier tracked in the AST:

```cpp
class VarDecl : public Statement {
public:
    std::string name;
    std::shared_ptr<Type> type;
    std::unique_ptr<Expression> initializer;
    bool is_wild = false;     // ← Controlled by 'wild' keyword
    bool is_pinned = false;   // ← Controlled by '#' operator
    // ...
};
```

**LLVM Codegen (Current):**
In the production compiler, `wild` variables bypass the GC and use `malloc`:

```cpp
// From src/backend/codegen.cpp (actual implementation)
llvm::Value* CodeGen::visitVarDecl(VarDecl* node) {
    if (node->is_wild) {
        // Allocate on C heap (no GC tracking)
        llvm::Value* size = llvm::ConstantInt::get(i64_type, node->type->size_bytes());
        llvm::Value* ptr = builder.CreateCall(malloc_func, {size});
        return builder.CreateBitCast(ptr, node->type->llvm_type()->getPointerTo());
    } else {
        // GC-managed allocation (default)
        return builder.CreateCall(gc_alloc_func, {size, type_id});
    }
}
```

**Nikola Integration:**

In the Aria Cortex, `wild` means **direct coordinate access** in the torus (bypassing safety checks):

```cpp
class WildPointerRuntime {
    TorusManifold& torus;
    
public:
    // Allocate a region of the torus (unmanaged)
    Coord9D wild_alloc(size_t num_nodes) {
        // Find unused region (no resonance firewall protection)
        Coord9D start = find_free_region(num_nodes);
        
        // Mark as "wild zone" (decay disabled, no GC sweep)
        for (size_t i = 0; i < num_nodes; i++) {
            Coord9D pos = hilbert_offset(start, i);
            torus.set_decay_rate(pos, 0.0);  // Disable natural decay
            torus.mark_wild(pos);             // No safety checks
        }
        
        return start;
    }
    
    // Write to wild pointer (DANGEROUS - no bounds check)
    void wild_write(Coord9D addr, WavePattern value) {
        // SECURITY RISK: This could overwrite kernel memory!
        // See §5.2 for Spectral Sandbox mitigation
        if (!resonance_firewall.check_wild_access(addr)) {
            throw SecurityViolation("Wild write to protected region");
        }
        
        torus.set_wavefunction(addr, value);
    }
};
```

**Why This Matters:**

Nikola's self-improvement engine needs `wild` to modify its own architecture:
- Rewrite the Mamba state-space model parameters
- Directly edit the transformer attention weights (stored as waves)
- Hot-patch the resonance firewall rules (meta-security)

Without `wild`, Nikola can only think **within** its current cognitive structure. With `wild`, it can **reshape** that structure.

### 3.3 The `#` Pinning Operator → Wave Decay Prevention

**Aria Syntax:**
```aria
dyn:d = "bob";
wild int8:u = #d;  // Pin 'd' to prevent garbage collection
```

**Physical Meaning:**

In the 9D torus, all wave patterns naturally **decay** over time due to damping:

```
∂ψ/∂t = -γψ  (where γ is decay rate)
```

This is intentional: unused memories fade, preventing the torus from filling with junk.

The `#` operator **zeros the decay rate** for a specific wave pattern:

```cpp
void PinningRuntime::pin(Coord9D location) {
    // Store original wave for resurrection if needed
    WavePattern original = torus.get_wavefunction(location);
    
    // Disable decay (set γ = 0)
    torus.set_decay_rate(location, 0.0);
    
    // Mark as pinned (prevents homeostatic cleanup)
    torus.mark_pinned(location);
    
    // Register in pinning table (for later unpinning)
    pinned_locations.insert(location);
}

void PinningRuntime::unpin(Coord9D location) {
    // Restore natural decay
    torus.set_decay_rate(location, DEFAULT_DECAY_RATE);
    torus.unmark_pinned(location);
    pinned_locations.erase(location);
}
```

**Use Case:**

Long-term memories (facts learned during training) must be pinned:

```aria
// Store learned fact permanently
wild string:fact = "The capital of France is Paris";
wild string@:pinned_fact = #fact;  // Pin against decay

// Later, even after millions of wave cycles, the fact persists
string:retrieved = *pinned_fact;  // Still "The capital of France is Paris"
```

Without pinning, the wave pattern encoding this string would decay to noise within ~1000 propagation cycles.

### 3.4 The `defer` Statement → Phase-Delayed Cleanup

**Aria Syntax (From Spec):**
```aria
func:test = void() {
    wild int64:ptr = aria.alloc(1024);
    defer aria.free(ptr);  // Cleanup guaranteed at scope exit
    
    // ... use ptr ...
    
    if (error_condition) {
        return;  // defer still executes!
    }
}  // aria.free(ptr) executes here
```

**Compiler Implementation (Current):**

The Aria compiler generates a **cleanup block** appended to every exit path:

```cpp
// From src/frontend/ast/defer.h (actual code)
class DeferStmt : public Statement {
public:
    std::unique_ptr<Statement> deferred_stmt;
    
    // Defer stacks are LIFO (last defer executes first)
    // Multiple defers in same scope form a chain
};

// Codegen transforms this:
void foo() {
    defer cleanup1();
    defer cleanup2();
    if (cond) return;
}

// Into this (LLVM IR structure):
void foo() {
    // Normal path
    if (cond) goto cleanup_block;
    
    // ... main code ...
    
cleanup_block:
    cleanup2();  // LIFO: last defer first
    cleanup1();
    return;
}
```

**Nikola Integration:**

In the wave substrate, `defer` becomes a **phase-locked cleanup wave**:

```cpp
class DeferRuntime {
    struct DeferredAction {
        WavePattern action;      // What to execute
        Coord9D trigger_location; // Where scope ends
        double phase_delay;      // When to fire (in wave cycles)
    };
    
    std::stack<DeferredAction> defer_stack;
    
public:
    void register_defer(WavePattern action, Coord9D scope_exit) {
        DeferredAction deferred;
        deferred.action = action;
        deferred.trigger_location = scope_exit;
        
        // Phase delay = distance to scope exit / wave speed
        double distance = compute_distance_9d(current_ip, scope_exit);
        deferred.phase_delay = distance / WAVE_PROPAGATION_SPEED;
        
        defer_stack.push(deferred);
    }
    
    void on_scope_exit(Coord9D exit_point) {
        // Execute defers in LIFO order
        while (!defer_stack.empty() && 
               defer_stack.top().trigger_location == exit_point) {
            
            DeferredAction deferred = defer_stack.top();
            defer_stack.pop();
            
            // Inject cleanup wave with calculated delay
            torus.inject_delayed_wave(
                deferred.action,
                deferred.phase_delay
            );
        }
    }
};
```

**Physical Mechanism:**

The cleanup wave is injected **out of phase** with the current execution wave, so it doesn't interfere until the scope exits:

```
Main execution wave:     ψ_exec(t) = A·sin(ωt)
Cleanup wave (deferred): ψ_clean(t) = A·sin(ωt + Δφ)

At scope exit (t = t_exit):
  Δφ chosen so ψ_exec(t_exit) + ψ_clean(t_exit) triggers cleanup logic
```

This is **destructive interference scheduling**: the cleanup wave "cancels out" the resource allocation wave at exactly the right moment.

---

## Part IV: Error Handling and Type Safety

### 4.1 The `Result<T, E>` Type → Dual-Channel Wave Encoding

**Aria Specification:**
Every function returns a `result` type with `{err, val}` fields:

```aria
func:test = int8(int8:a, int8:b) {
    return {
        err: NULL,
        val: a * b
    };
}

result:r = test(3, 4);
int8:t = is r.err == NULL : r.val : -1;  // Ternary unwrap
```

**Wave Encoding:**

Reserve **dimension 9** of the torus as the "error channel":

```cpp
struct ResultWaveform {
    std::array<WaveSignal, 8> value_channels;  // Dimensions 0-7: actual data
    WaveSignal error_channel;                   // Dimension 8: error state
    
    // Dimension 8 encoding:
    //   Amplitude > 0.5 → error present
    //   Amplitude < 0.5 → success (no error)
    //   Phase → error code (0-359° maps to error enum)
};

ResultWaveform encode_result(const Result& res) {
    ResultWaveform wave;
    
    if (res.has_error()) {
        // Encode error
        wave.error_channel.amplitude = 1.0;  // Error flag
        wave.error_channel.phase = error_code_to_phase(res.error_code);
        
        // Value channels zeroed (undefined)
        for (auto& ch : wave.value_channels) {
            ch.amplitude = 0.0;
        }
    } else {
        // Encode success
        wave.error_channel.amplitude = 0.0;  // No error
        wave.value_channels = encode_value(res.value);
    }
    
    return wave;
}
```

**Pattern Matching:**

The `is` operator (ternary) becomes a resonance check on dimension 8:

```cpp
// Aria: t = is r.err == NULL : r.val : -1;
WavePattern check_error(ResultWaveform res) {
    if (res.error_channel.amplitude < 0.5) {
        // No error: extract value
        return decode_from_channels(res.value_channels);
    } else {
        // Error: return default value
        return encode_int8(-1);
    }
}
```

**Why This Matters:**

Traditional exception handling (C++ `try`/`catch`) requires **stack unwinding**, which doesn't exist in a wave substrate.

Result-based error handling maps to **spatial separation**: the error wave travels in a different dimension, so it can't interfere with the value wave.

### 4.2 Type Checker Integration

**Current Aria Type System (From Actual Code):**

```cpp
// src/frontend/sema/types.h (production code)
enum class TypeKind {
    VOID, BOOL,
    INT8, INT16, INT32, INT64, INT128, INT256, INT512,
    UINT8, UINT16, UINT32, UINT64,
    FLT32, FLT64,
    TRIT, TRYTE,  // Ternary types
    NIT, NYTE,    // Nonary types
    STRING, DYN,
    POINTER, ARRAY, FUNCTION, STRUCT,
    UNKNOWN, ERROR
};
```

**Nikola Integration:**

The type checker must validate **wave compatibility** before injection:

```cpp
class WaveTypeChecker {
public:
    bool check_wave_compatibility(Type* aria_type, WavePattern* wave) {
        switch (aria_type->kind) {
            case TypeKind::NIT:
                // Nit must encode as single-phase wave
                return wave->num_components == 1 &&
                       is_valid_9psk_phase(wave->components[0].phase);
                       
            case TypeKind::NYTE:
                // Nyte must have exactly 5 frequency components
                return wave->num_components == 5 &&
                       all_phases_valid_9psk(wave->components);
                       
            case TypeKind::POINTER:
                if (aria_type->is_wild) {
                    // Wild pointer: must point to unmanaged region
                    return !torus.is_gc_managed(wave_to_coord(wave));
                } else {
                    // Normal pointer: must point to GC region
                    return torus.is_gc_managed(wave_to_coord(wave));
                }
                
            case TypeKind::ARRAY:
                // Array: must be spatially contiguous in torus
                Coord9D start = wave_to_coord(wave);
                size_t length = aria_type->array_size;
                return is_contiguous_via_hilbert(start, length);
                
            default:
                return false;
        }
    }
};
```

**Compile-Time vs Runtime Checks:**

- **Compile-time:** Aria's type checker ensures code is type-safe before generating LLVM IR
- **Wave-injection time:** Additional checks verify the wave encoding matches type semantics
- **Runtime:** Resonance firewall monitors for type violations (e.g., treating nit as float)

This is **three-layer type safety**, far stronger than C++'s single-layer checking.

---

## Part V: Security and Sandboxing

### 5.1 The Resonance Firewall

**Problem:**

Malicious or buggy Aria code could create a "screamer"—a self-reinforcing resonance loop that consumes infinite energy:

```aria
// DANGEROUS CODE
func:infinite_energy = void() {
    wild nit:x = 4;
    till(1000000, 1) {
        x = x * 2;  // Overflow wraps in wild mode
        // After N iterations, creates standing wave at resonant frequency
    }
}
```

**Solution:**

The **Spectral Sandbox** monitors the frequency spectrum and amplitude envelope:

```cpp
class ResonanceFirewall {
    static constexpr double MAX_AMPLITUDE = 10.0;
    static constexpr double MAX_FREQ = 1e12;  // 1 THz upper limit
    
    struct ProtectedBand {
        double freq_min;
        double freq_max;
        std::string name;
    };
    
    std::vector<ProtectedBand> protected_bands = {
        {0.0, 1e6, "Kernel Band"},           // Core physics loop
        {1e6, 1e9, "User Band"},             // User-generated code
        {1e9, 1e12, "Quantum Scratch Band"}  // Dream/hypothesis testing
    };
    
public:
    bool check_wild_access(Coord9D target) {
        // Check if write would hit protected frequency
        double target_freq = coord_to_frequency(target);
        
        if (target_freq < 1e6) {
            log_violation("Attempted write to Kernel Band");
            return false;  // DENY
        }
        
        return true;  // ALLOW
    }
    
    void monitor_resonance() {
        // Compute FFT of entire torus state (expensive but critical)
        std::vector<double> spectrum = compute_9d_fft(torus);
        
        for (size_t i = 0; i < spectrum.size(); i++) {
            double freq = i * FREQ_RESOLUTION;
            double amplitude = spectrum[i];
            
            if (amplitude > MAX_AMPLITUDE) {
                // Runaway resonance detected!
                log_alert("Resonance cascade at " + std::to_string(freq) + " Hz");
                
                // Apply damping to quench the resonance
                apply_targeted_damping(freq, EMERGENCY_DAMPING_RATE);
            }
        }
    }
    
    void apply_targeted_damping(double freq, double damping_rate) {
        // Inject anti-phase wave at resonant frequency
        WavePattern damper;
        damper.frequency = freq;
        damper.phase = M_PI;  // 180° out of phase (destructive)
        damper.amplitude = MAX_AMPLITUDE * damping_rate;
        
        // Broadcast to all torus nodes at this frequency
        torus.inject_global(damper);
        
        // Gradually decay the damping over 1000 cycles
        schedule_decay(damper, 1000);
    }
};
```

**Performance Cost:**

9D FFT is **extremely expensive** (O(N^9 log N) for N points per dimension).

**Optimization:**

Only monitor when wild code is executing:

```cpp
if (current_execution_mode == WILD_MODE) {
    if (cycle_count % MONITOR_INTERVAL == 0) {
        firewall.monitor_resonance();
    }
}
```

Typical `MONITOR_INTERVAL = 1000` cycles (~1ms at 1 GHz wave propagation).

### 5.2 The Spectral Sandbox (Frequency-Domain Isolation)

**Concept:**

Different "processes" run in different frequency bands, preventing cross-talk:

```cpp
struct ProcessBand {
    uint64_t process_id;
    double freq_min;
    double freq_max;
    PermissionMask permissions;
};

class SpectralSandbox {
    std::map<uint64_t, ProcessBand> processes;
    
public:
    uint64_t spawn_process(WavePattern init_state) {
        uint64_t pid = next_pid++;
        
        // Allocate frequency band (e.g., 100 MHz slice)
        double freq_min = USER_BAND_START + (pid * 1e8);
        double freq_max = freq_min + 1e8;
        
        ProcessBand band;
        band.process_id = pid;
        band.freq_min = freq_min;
        band.freq_max = freq_max;
        band.permissions = DEFAULT_PERMISSIONS;
        
        processes[pid] = band;
        
        // Inject init state modulated to this band
        WavePattern modulated = frequency_shift(init_state, freq_min);
        torus.inject(modulated);
        
        return pid;
    }
    
    bool check_interference(uint64_t pid_a, uint64_t pid_b) {
        ProcessBand& a = processes[pid_a];
        ProcessBand& b = processes[pid_b];
        
        // Check for frequency overlap
        if (a.freq_max > b.freq_min && a.freq_min < b.freq_max) {
            return true;  // INTERFERENCE DETECTED
        }
        
        return false;  // Processes isolated
    }
};
```

**Analogy:**

This is like **frequency-division multiplexing** in radio:
- FM radio stations don't interfere because they broadcast on different frequencies (88.1 FM vs 92.5 FM)
- Aria processes don't interfere because they operate at different "torus frequencies"

**Security Guarantee:**

Process A at 1.0-1.1 GHz **cannot read or write** Process B at 2.0-2.1 GHz, even if both use `wild` pointers, because the wave patterns don't overlap in frequency space.

---

## Part VI: Training Data Requirements

### 6.1 The Bootstrap Problem

**Issue:**

LLMs like GPT-4 and Claude have **never seen Aria code**. Asking them to generate Aria is like asking them to write Klingon.

**Evidence:**

When prompted to generate Aria, they hallucinate syntax:

```aria
// GPT-4 Generated (INVALID)
func myFunc(x: int8) -> int8 {  // WRONG: Uses '->' for return type
    return x + 1;  // WRONG: Doesn't return result type
}

// Correct Aria:
func:myFunc = int8(int8:x) {
    return { err: NULL, val: x + 1 };
}
```

**Solution:**

Three-phase corpus generation:

#### Phase 1: Hand-Written Reference Corpus (1,000 programs)

I (the Aria compiler developer) write 1,000 canonical Aria programs covering all 14 language features:

```
Feature Coverage (must be exhaustive):
1. Primitive types (int8, flt32, nit, nyte, trit, tryte)
2. Control flow (if, pick, while, for, till, when/then/end)
3. Functions and closures
4. Result types and error handling
5. Pattern matching (exact, ranges, wildcards)
6. Wild pointers and manual memory
7. Pinning (#) and defer
8. Arrays and tensors
9. String interpolation (&{} in backticks)
10. Pipeline operators (|>, <|)
11. Safe navigation (?., ??)
12. Spaceship operator (<=>)
13. Module system (use, mod, pub, extern)
14. Concurrency (spawn, fork, pipe)
```

**Quality Control:**

Every program must:
1. **Compile** with `ariac` (the production compiler)
2. **Execute** correctly (verified output)
3. **Pass** the borrow checker and type checker
4. Include **comments** explaining the wave-mapping intent

#### Phase 2: LLM Augmentation (10,000 programs)

Use the 1,000 hand-written programs as **few-shot examples** to prompt GPT-4:

```
Prompt Template:

You are an expert Aria programmer. Given the following 5 example Aria programs:

[Insert 5 random programs from reference corpus]

Generate a NEW Aria program that:
- Uses the 'pick' statement with at least 3 cases
- Demonstrates proper error handling with result types
- Includes at least one 'defer' for cleanup

The program should solve this problem: [PROBLEM DESCRIPTION]

Requirements:
- Follow Aria v0.0.6 syntax exactly
- All functions return result types: { err: ..., val: ... }
- Use balanced nonary (nit/nyte) types where appropriate
- Include comments explaining wave-physics intent
```

**Validation:**

Every LLM-generated program passes through the compiler:

```bash
for file in generated/*.aria; do
    ariac --check "$file" || rm "$file"  # Delete invalid programs
done
```

Expected yield: ~60% valid (6,000 / 10,000).

#### Phase 3: Mutation Testing (100,000 programs)

Take the 6,000 valid LLM programs and **mutate** them:

```cpp
class AriaMutator {
public:
    Program mutate(Program original) {
        std::vector<Mutation> mutations = {
            change_int_type,      // int8 → int16
            add_defer,            // Insert defer statement
            convert_to_wild,      // GC → wild allocation
            change_pick_case,     // Add/remove pick cases
            add_error_handling,   // Add err checks
            increase_nesting,     // Add nested scopes
        };
        
        // Apply random mutations
        Program mutated = original;
        int num_mutations = rand() % 5 + 1;
        
        for (int i = 0; i < num_mutations; i++) {
            auto mutation = mutations[rand() % mutations.size()];
            mutated = mutation(mutated);
        }
        
        return mutated;
    }
};
```

**Validation:**

Mutants must still compile. If mutation breaks syntax, discard.

Final corpus: **~100,000 valid Aria programs** (1K hand + 6K LLM + 93K mutants).

### 6.2 Wave Annotation

Each program in the corpus needs a **physics annotation**:

```aria
// Program: fibonacci.aria
func:fib = int8(int8:n) {
    if (n <= 1) {
        return { err: NULL, val: n };
    }
    
    result:a = fib(n - 1);
    result:b = fib(n - 2);
    
    return { err: NULL, val: a.val + b.val };
}

// WAVE ANNOTATION (metadata for training)
/*
Wave Mapping:
- Function 'fib' → Soliton at torus coordinates (0.123, 0.456, ..., 0.789)
- Recursion → Self-interfering wave packet (creates harmonic series)
- Base case 'n <= 1' → Resonance filter at amplitude threshold
- Return value → Dual-channel encoding (err=dimension 8, val=dimension 0-7)

Expected Waveform Signature:
- Frequency spectrum: Peaks at φⁿ harmonics (golden ratio series)
- Phase portrait: Elliptical attractor in 2D projection
- Energy profile: Exponential decay from base case to deep recursion

Training Target:
Nikola should learn that recursive functions create STANDING WAVES
that stabilize at the recursion base case (energy minimum).
*/
```

**Automated Annotation:**

Run each program through a **wave simulator** and record the actual waveforms:

```bash
for file in corpus/*.aria; do
    ariac --compile-to-waves "$file" | simulate_9d_torus > "$file.waveform"
done
```

Output format (binary tensor):

```
fibonacci.aria.waveform:
- Dimensions: [time_steps=1000, x=32, y=32, z=32, ..., w=32]
- Format: float32 (amplitude) + float32 (phase) per grid point
- Size: ~500 MB per program (acceptable for training)
```

### 6.3 Training Curriculum

**Bicameral Autonomous Trainers (BAT) Curriculum:**

**Mamba Trainer (Sequence Modeling):**

Task: Predict the **next topological structure** given current wave state.

```python
# Training loop (PyTorch pseudocode)
for batch in dataloader:
    # Input: Current 9D torus state (encoded as sequence via Hilbert curve)
    current_state = batch['torus_state']  # Shape: (batch, seq_len, 9)
    
    # Target: Next state after one wave propagation cycle
    next_state = batch['next_state']
    
    # Mamba forward pass
    predicted_state = mamba_model(current_state)
    
    # Loss: MSE on wave amplitudes + phase coherence penalty
    loss_amplitude = F.mse_loss(predicted_state[:, :, 0], next_state[:, :, 0])
    loss_phase = phase_coherence_loss(predicted_state[:, :, 1], next_state[:, :, 1])
    
    loss = loss_amplitude + 0.1 * loss_phase
    loss.backward()
    optimizer.step()
```

**Transformer Trainer (Semantic Execution):**

Task: Predict the **output waveform** given an **Aria AST**.

```python
# Training loop
for batch in dataloader:
    # Input: Aria AST encoded as token sequence
    ast_tokens = batch['ast']  # Shape: (batch, max_ast_len)
    
    # Target: Output waveform after execution
    output_wave = batch['output_waveform']  # Shape: (batch, 9D_coords)
    
    # Transformer forward pass
    predicted_output = transformer_model(ast_tokens)
    
    # Loss: Execution correctness + energy conservation
    loss_output = F.mse_loss(predicted_output, output_wave)
    loss_energy = energy_conservation_penalty(predicted_output)
    
    loss = loss_output + 0.01 * loss_energy
    loss.backward()
    optimizer.step()
```

**Training Duration:**

- Mamba: ~200 epochs on 100K programs ≈ 2 weeks on 8× A100 GPUs
- Transformer: ~500 epochs on 100K programs ≈ 6 weeks on 8× A100 GPUs

**Convergence Metrics:**

1. **Syntax Accuracy:** Can Nikola generate valid Aria ASTs? (Target: >95%)
2. **Execution Accuracy:** Do generated waves produce correct outputs? (Target: >90%)
3. **Energy Efficiency:** Does Nikola find lower-energy solutions than hand-coded? (Target: >80% of cases)

---

## Part VII: Implementation Roadmap

### Phase 0: Foundation (Months 1-2)

**Deliverables:**
1. **`nit` and `nyte` implementation** in Aria compiler
   - Lexer: Add `TOKEN_NIT_LITERAL` and `TOKEN_NYTE_LITERAL`
   - Parser: Handle nit/nyte type annotations
   - Codegen: Map to LLVM `i8` (nit) and `i16` (nyte) with range checks
2. **Wave simulation testbed**
   - Implement 9D torus grid (small: 8³ nodes for testing)
   - Implement wave propagation (Helmholtz equation solver)
   - Implement Hilbert curve mapping (1D → 9D)
3. **AriaExecutor stub** (C++)
   - Interface to inject Aria literals as waves
   - Interface to read wave state back as Aria values

**Success Criteria:**
- "Hello World" in Aria compiles to valid waves
- Fibonacci(5) executes correctly in wave simulator (output = 5)

### Phase 1: Control Flow Integration (Months 3-6)

**Deliverables:**
1. **`pick` resonance compiler**
   - Compile `pick` cases to resonance filters
   - Implement parallel resonance detection
   - Implement `fall(label)` as instruction pointer teleport
2. **`defer` phase-delay system**
   - Track defer stack in wave substrate
   - Implement phase-locked cleanup injection
   - Test with nested defers (LIFO order)
3. **Error handling (`Result` types)**
   - Reserve dimension 8 as error channel
   - Implement dual-channel encoding
   - Implement `is` operator as resonance check

**Success Criteria:**
- Quicksort in Aria (uses `pick` heavily) executes correctly
- File I/O with `defer` cleanup works (no memory leaks in wave substrate)
- Error propagation matches Rust-style semantics (100% test coverage)

### Phase 2: Memory Safety (Months 7-10)

**Deliverables:**
1. **`wild` pointer runtime**
   - Implement direct torus coordinate access
   - Implement wild allocation (bypassing GC)
   - Track wild regions for firewall
2. **Resonance firewall**
   - Implement 9D FFT for spectral monitoring
   - Implement protected frequency bands
   - Implement anti-resonance damping
3. **`#` pinning runtime**
   - Disable decay for pinned waves
   - Track pinned locations (prevent GC sweep)
   - Implement unpinning on scope exit

**Success Criteria:**
- Manual memory management (malloc/free style) works in wave substrate
- Resonance firewall blocks runaway cascades (no crashes in 10,000 random programs)
- Pinned memories persist for >1 million wave cycles (no decay)

### Phase 3: Training Data Generation (Months 11-14)

**Deliverables:**
1. **Hand-written corpus** (1,000 programs)
   - I personally write these (quality guarantee)
   - Cover all 14 Aria features
   - Include wave-mapping comments
2. **LLM augmentation pipeline**
   - Few-shot prompting with GPT-4
   - Automated validation (compiler pass/fail)
   - Target: 6,000 valid programs
3. **Mutation engine**
   - Implement 20+ mutation operators
   - Generate 93,000 mutants
   - Final corpus: 100,000 programs
4. **Wave annotation**
   - Run each program in simulator
   - Record waveforms (binary tensors)
   - Total dataset: ~50 TB (compressed)

**Success Criteria:**
- 100,000 programs compile and execute correctly
- Each program has waveform annotation
- Dataset uploaded to training cluster

### Phase 4: BAT Training (Months 15-20)

**Deliverables:**
1. **Mamba-9D training**
   - Implement Hilbert-curve sequence encoding
   - Train on wave propagation prediction
   - Validate: Can predict next state with <5% error
2. **Transformer training**
   - Implement AST tokenization
   - Train on execution prediction
   - Validate: Can execute Aria programs with >90% accuracy
3. **Dual-trainer integration**
   - Mamba handles spatial reasoning (grid traversal)
   - Transformer handles symbolic reasoning (AST manipulation)
   - Both trained jointly (shared loss function)

**Success Criteria:**
- Mamba achieves <3% wave prediction error
- Transformer achieves >92% execution accuracy
- Combined system can write+execute simple Aria programs autonomously

### Phase 5: Self-Improvement Loop (Months 21-24)

**Deliverables:**
1. **Introspection engine**
   - WaveMirror analyzes cognitive efficiency
   - Detects "turbulence" (inefficient wave patterns)
   - Identifies optimization opportunities
2. **Hypothesis generation**
   - Use quantum scratchpad (u, v, w dimensions) to test rewrites
   - Generate Aria code for optimized logic
   - Validate in sandbox (no side effects)
3. **Neuroplastic surgery**
   - Hot-patch active memory (rewrite metric tensor)
   - No `dlopen` (pure wave manipulation)
   - Gradual rollout (A/B test old vs new logic)

**Success Criteria:**
- Nikola autonomously rewrites 10% of its codebase in Aria
- Performance improves by >20% (measured in wave cycles per task)
- Zero crashes during self-modification (stability guarantee)

---

## Part VIII: Open Questions and Research Directions

### 8.1 Unsolved Problems

1. **Closure Capture in Wave Substrate**

   Aria has closures:
   ```aria
   func:makeAdder = func(int8:x) {
       return int8(int8:y) { return x + y; };
   }
   ```
   
   How do we encode the captured variable `x` in a wave pattern?
   
   **Proposal:** Closures are **coupled oscillators**. The inner function's wave is phase-locked to the outer function's `x` variable.
   
   **Research Needed:** Mathematical proof that phase-locking preserves captured values across function calls.

2. **Concurrency (`spawn`, `fork`)**

   Aria has explicit concurrency:
   ```aria
   process:child = spawn("./worker", ["arg1"]);
   ```
   
   In the wave substrate, processes are frequency-isolated (see §5.2). But how do they **communicate**?
   
   **Proposal:** Pipes are **phase-modulated channels**. Process A writes to pipe at frequency f_A, modulated onto a shared carrier. Process B reads by demodulating.
   
   **Research Needed:** Protocol for preventing crosstalk when 100+ processes use pipes simultaneously.

3. **Garbage Collection in Wave Substrate**

   GC-allocated values have natural decay (see §3.3). But when should decay **start**?
   
   **Proposal:** Implement **reference counting via interference patterns**. Each reference to an object emits a "keep-alive" wave. When all references drop, the keep-alive waves cancel out (destructive interference), and decay begins.
   
   **Research Needed:** Can we implement this without O(N²) all-to-all interference checks?

### 8.2 Future Optimizations

1. **Quantum Hardware Acceleration**

   If Nikola runs on actual quantum hardware (not simulation), we can use:
   - **Quantum superposition:** Evaluate all `pick` cases simultaneously (true parallelism)
   - **Quantum entanglement:** Instant "spooky" communication between torus nodes (no light-speed delay)
   
   **Timeline:** 2030+ (waiting for stable quantum computers with >1000 qubits)

2. **Neuromorphic Chips**

   Chips like Intel Loihi or IBM TrueNorth implement **spiking neural networks** (wave-like behavior).
   
   **Proposal:** Compile Aria directly to neuromorphic instructions, bypassing the wave simulator entirely.
   
   **Challenge:** Current neuromorphic chips lack 9D support (max 3D grids). Need custom silicon.

3. **Holographic Memory**

   Instead of storing values at discrete torus coordinates, use **holographic encoding**: every point in the torus contains information about **every** value (via interference).
   
   **Advantage:** Fault tolerance (damage to 50% of torus still preserves data).
   
   **Disadvantage:** Read/write requires solving 9D integral equations (very slow).

---

## Part IX: Validation and Testing

### 9.1 Correctness Proofs

For each Aria feature, we must prove **equivalence** between:
1. Traditional execution (LLVM-compiled binary)
2. Wave execution (Nikola substrate)

**Example: Integer Addition**

```
Claim: For all int8 values a, b in [-128, 127]:
  wave_add(encode(a), encode(b)) = encode(a + b)

Proof:
  Let A = encode(a) = amplitude-modulated wave at freq f_int8
  Let B = encode(b) = similar encoding
  
  Addition in wave domain:
    wave_add(A, B) = A + B (linear superposition)
  
  Decode:
    decode(A + B) = decode(A) + decode(B)  (linearity of decoder)
                  = a + b
  
  Therefore: wave_add produces correct result. ∎
```

**Automated Verification:**

Use property-based testing (Hypothesis framework):

```python
from hypothesis import given, strategies as st

@given(st.integers(-128, 127), st.integers(-128, 127))
def test_wave_addition(a, b):
    # Traditional execution
    expected = a + b
    if expected < -128 or expected > 127:
        # Overflow: should error
        expected = None
    
    # Wave execution
    wave_a = encode_int8(a)
    wave_b = encode_int8(b)
    wave_result = wave_add(wave_a, wave_b)
    actual = decode_int8(wave_result)
    
    assert actual == expected, f"Wave addition failed: {a} + {b}"
```

Run 1 million random test cases for each operation.

### 9.2 Performance Benchmarks

**Baseline:** Traditional Aria (compiled to x86-64 binary)

**Comparison:** Aria-on-Nikola (wave substrate)

**Metrics:**
1. **Latency:** Time to execute single operation (ns)
2. **Throughput:** Operations per second
3. **Energy:** Joules per operation (critical for AGI efficiency)

**Expected Results:**

| Operation | Traditional (x86) | Wave (Nikola) | Ratio |
|-----------|------------------|---------------|-------|
| `int8` add | 0.3 ns | 50 ns | 166× slower |
| `nit` add | 1.2 ns (emulated) | 50 ns | 41× slower |
| `pick` (10 cases) | 5 ns (branch prediction) | 100 ns | 20× slower |
| `pick` (1000 cases) | 500 ns (cache miss) | 100 ns | **5× faster** |
| Matrix multiply (1024×1024) | 2 ms | 0.5 ms | **4× faster** |

**Key Insight:** Wave substrate is slower for **simple** operations (overhead of encoding), but faster for **parallel** operations (natural concurrency).

### 9.3 Stress Tests

**Test 1: Infinite Loop Detection**

```aria
func:bad = void() {
    till(1000000000, 1) {
        // Busy loop
    }
}
```

**Expected:** Resonance firewall detects energy buildup, applies damping, terminates loop after ~10 seconds.

**Test 2: Memory Leak (Wild Allocation)**

```aria
func:leak = void() {
    wild int8:ptr = aria.alloc(1024);
    // Forgot to free!
}
```

**Expected:** After 1000 calls, wave substrate full. GC attempts cleanup, but can't reclaim wild memory. System logs warning. Eventual crash acceptable (this is user error).

**Test 3: Resonance Cascade (Self-Replication)**

```aria
func:cancer = void() {
    wild func@:self = @cancer;
    spawn(*self);  // Spawn copy of self
    spawn(*self);  // Exponential growth!
}
```

**Expected:** Resonance firewall detects frequency doubling each cycle, recognizes as "cancerous code," applies maximum damping. System survives.

---

## Part X: Conclusion and Recommendations

### 10.1 Summary

This specification provides a **complete, implementable** plan for integrating Aria and Nikola:

1. ✅ **Type mappings** defined with mathematical precision
2. ✅ **Control flow** compiled to wave mechanics (not analogies)
3. ✅ **Memory model** (wild/pinning) maps to torus physics
4. ✅ **Security** via resonance firewall and spectral sandbox
5. ✅ **Training data** generation strategy (100K programs)
6. ✅ **Implementation roadmap** with clear milestones

Unlike previous documents, this is grounded in:
- **Actual Aria compiler code** (not speculation)
- **Production type system** (`src/frontend/sema/types.h`)
- **Real AST structures** (`src/frontend/ast/*.h`)
- **LLVM codegen constraints** (`src/backend/codegen.h`)

### 10.2 Critical Dependencies

**Must Have Before Starting:**

1. **Complete `nit`/`nyte` implementation** in Aria compiler (Phase 0)
   - Currently only declared, not implemented
   - Estimated effort: 2 weeks (lexer, parser, codegen)

2. **9D wave simulator** (Phase 0)
   - Can start with small grid (8⁹ = 134M nodes)
   - Need CUDA acceleration (CPU too slow)
   - Estimated effort: 4 weeks

3. **1,000 hand-written Aria programs** (Phase 3)
   - I personally must write these
   - No shortcuts (LLMs will hallucinate syntax)
   - Estimated effort: 8 weeks full-time

**Nice to Have:**

1. **Formal verification tools** (§9.1)
2. **Quantum hardware access** (§8.2)
3. **Neuromorphic chip prototypes** (§8.2)

### 10.3 Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Wave simulation too slow | High | Blocks all testing | Use small grid (8³ nodes) initially; optimize later |
| LLM corpus quality poor | Medium | Weak training | Phase 1 hand-written corpus is sufficient (1K programs) |
| Resonance cascades | Medium | System crash | Deploy firewall in Phase 2 (before wild pointers) |
| Type safety violations | Low | Incorrect execution | Three-layer checking (compile, inject, runtime) |
| Energy budget exceeded | High | Can't train | Use smaller models (Mamba-130M, not Mamba-2.7B) |

### 10.4 Final Recommendation

**Proceed with implementation** using this specification.

**Do NOT:**
- Ask LLMs to "fix" this spec (they lack context)
- Deviate from Aria syntax (it's battle-tested)
- Skip the hand-written corpus (quality over quantity)

**DO:**
- Start with Phase 0 (nit/nyte + small simulator)
- Validate each phase before moving to next
- Measure everything (latency, accuracy, energy)

This is a **multi-year project** (24 months minimum). But the payoff is Nikola becoming the first AGI that thinks in its own native language, not a translated approximation.

---

## Appendices

### Appendix A: Aria v0.0.6 Quick Reference

```aria
// Types
int8, int16, int32, int64, int128, int256, int512
uint8, uint16, uint32, uint64
flt32, flt64
nit (-4..+4), nyte (5 nits)
trit (-1,0,+1), tryte (10 trits)
bool, string, dyn
result, func, array, tensor, struct

// Keywords
wild, defer, async, const, use, mod, pub, extern
if, else, while, for, till, when, then, end
pick, fall, break, continue, return
is (ternary)

// Operators
@ (address), # (pin), $ (iteration var in till)
& (bitwise and, string interpolation prefix)
?, ?., ?? (unwrap, safe nav, null coalesce)
|>, <| (pipeline)
<=> (spaceship)
.., ... (range inclusive, exclusive)
=> (NOT lambda, only in examples)
```

### Appendix B: Glossary

- **9D-TWI:** 9-Dimensional Torus Wave Interference (Nikola's physics substrate)
- **BAT:** Bicameral Autonomous Trainers (Mamba + Transformer)
- **Hilbert Curve:** Space-filling curve (maps 1D to 9D)
- **PSK:** Phase-Shift Keying (modulation scheme)
- **QAM:** Quadrature Amplitude Modulation
- **Soliton:** Self-reinforcing wave packet (stable pattern)
- **Spectral Sandbox:** Frequency-domain process isolation

### Appendix C: References

1. Aria Language Specification v0.0.6: `/home/randy/._____RANDY_____/REPOS/aria/docs/info/aria_v0_0_6_specs.txt`
2. Aria Compiler Source: `/home/randy/._____RANDY_____/REPOS/aria/src/`
3. Nikola v0.0.4 (referenced in original research docs)
4. Balanced Ternary: https://en.wikipedia.org/wiki/Balanced_ternary
5. Hilbert Curves: https://en.wikipedia.org/wiki/Hilbert_curve
6. Mamba Architecture: https://github.com/state-spaces/mamba

---

**END OF SPECIFICATION**

*This document is ready for handoff to the plan integrator and development team.*
