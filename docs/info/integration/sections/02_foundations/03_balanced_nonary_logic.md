# BALANCED NONARY LOGIC

## 5.1 Radix Economy

### Why Base-9?

The **radix economy** function measures the efficiency of a number base:

$$E(r, N) = r \cdot \lfloor \log_r N \rfloor$$

This is minimized when $r = e \approx 2.718$. Integer bases closest to $e$:
- Base-2 (binary): Inefficient (too many digits)
- Base-3 (ternary): Optimal efficiency
- Base-9 (nonary): Nearly optimal, higher information density

Base-9 = $3^2$, so it retains ternary efficiency while packing two trits per symbol.

### Balanced Representation

**Traditional nonary:** ${0, 1, 2, 3, 4, 5, 6, 7, 8}$

**Balanced nonary:** ${-4, -3, -2, -1, 0, 1, 2, 3, 4}$

**Benefits:**
- Symmetric around zero
- Natural subtraction (no separate operation)
- Direct wave encoding

## 5.2 Wave Encoding

Each balanced nonary digit maps to a wave amplitude and phase:

| Digit | Amplitude | Phase | Wave Representation |
|-------|-----------|-------|---------------------|
| **0** | 0 | N/A | Silence (vacuum) |
| **+1** | 1 | 0° | $\sin(\omega t)$ |
| **+2** | 2 | 0° | $2\sin(\omega t)$ |
| **+3** | 3 | 0° | $3\sin(\omega t)$ |
| **+4** | 4 | 0° | $4\sin(\omega t)$ |
| **-1** | 1 | 180° | $\sin(\omega t + \pi) = -\sin(\omega t)$ |
| **-2** | 2 | 180° | $-2\sin(\omega t)$ |
| **-3** | 3 | 180° | $-3\sin(\omega t)$ |
| **-4** | 4 | 180° | $-4\sin(\omega t)$ |

### C++ Type Definition (AVX-512 Optimized)

```cpp
namespace nikola::types {
    // Use int8_t instead of enum for vectorization
    // AVX-512 can process 64 nits in parallel with _mm512_add_epi8
    typedef int8_t Nit;
    
    // Symbolic constants for readability
    constexpr Nit N4 = -4;
    constexpr Nit N3 = -3;
    constexpr Nit N2 = -2;
    constexpr Nit N1 = -1;
    constexpr Nit ZERO = 0;
    constexpr Nit P1 = 1;
    constexpr Nit P2 = 2;
    constexpr Nit P3 = 3;
    constexpr Nit P4 = 4;
    
    // Vectorized saturation using AVX-512
    // Processes 64 nits in ~3 cycles (vs enum+clamp: 640-960 cycles)
    inline __m512i clamp_nits(__m512i values) {
        const __m512i min_val = _mm512_set1_epi8(-4);
        const __m512i max_val = _mm512_set1_epi8(4);
        return _mm512_max_epi8(min_val, _mm512_min_epi8(values, max_val));
    }
}
```

## 5.3 Arithmetic Operations

### Addition via Superposition

$$\Psi_C = \Psi_A + \Psi_B$$

**Physical example:**
- $A = +1$: $\Psi_A = \sin(\omega t)$
- $B = -1$: $\Psi_B = -\sin(\omega t)$
- $C = \Psi_A + \Psi_B = 0$ (destructive interference)

### Implementation (Scalar Version)

```cpp
Nit sum_gate(Nit a, Nit b) {
    int result = static_cast<int>(a) + static_cast<int>(b);
    // Saturation at ±4
    return static_cast<Nit>(std::clamp(result, -4, 4));
}
```

### Vectorized Implementation (AVX-512)

```cpp
// Add 64 balanced nonary digits in parallel
void vector_add_nits(Nit* result, const Nit* a, const Nit* b, size_t count) {
    for (size_t i = 0; i < count; i += 64) {
        // Load 64 nits (64 bytes) into zmm registers
        __m512i va = _mm512_loadu_si512((__m512i*)&a[i]);
        __m512i vb = _mm512_loadu_si512((__m512i*)&b[i]);
        
        // Add with saturation (3 cycles total)
        __m512i sum = _mm512_adds_epi8(va, vb);  // Signed saturated add
        sum = clamp_nits(sum);  // Additional clamp to ±4 range
        
        // Store result
        _mm512_storeu_si512((__m512i*)&result[i], sum);
    }
}
```

### Subtraction

Already implicit (negative numbers). To compute $A - B$:

```cpp
Nit subtract(Nit a, Nit b) {
    return sum_gate(a, negate(b));
}

Nit negate(Nit x) {
    return static_cast<Nit>(-static_cast<int>(x));
}
```

### Multiplication via Heterodyning

Mixing two sinusoids of frequencies $\omega_1$ and $\omega_2$ through a nonlinear medium (second-order susceptibility $\chi^{(2)}$) generates sidebands:

$$\sin(\omega_1 t) \cdot \sin(\omega_2 t) = \frac{1}{2}[\cos((\omega_1-\omega_2)t) - \cos((\omega_1+\omega_2)t)]$$

The amplitude of the sum-frequency component is proportional to the product.

### Simplified Implementation (Scalar)

```cpp
Nit product_gate(Nit a, Nit b) {
    int result = static_cast<int>(a) * static_cast<int>(b);
    // Saturate to ±4
    return static_cast<Nit>(std::clamp(result, -4, 4));
}
```

### Vectorized Implementation (AVX-512)

```cpp
// Multiply 32 pairs of nits in parallel (requires 16-bit intermediate)
void vector_mul_nits(Nit* result, const Nit* a, const Nit* b, size_t count) {
    for (size_t i = 0; i < count; i += 32) {
        // Load and zero-extend to 16-bit for multiplication
        __m256i va_8 = _mm256_loadu_si256((__m256i*)&a[i]);
        __m256i vb_8 = _mm256_loadu_si256((__m256i*)&b[i]);
        
        // Unpack to 16-bit (signed extension)
        __m512i va = _mm512_cvtepi8_epi16(va_8);
        __m512i vb = _mm512_cvtepi8_epi16(vb_8);
        
        // Multiply 32 pairs (16-bit precision)
        __m512i product = _mm512_mullo_epi16(va, vb);
        
        // Saturate to ±4 and pack back to 8-bit
        const __m512i min_val = _mm512_set1_epi16(-4);
        const __m512i max_val = _mm512_set1_epi16(4);
        product = _mm512_max_epi16(min_val, _mm512_min_epi16(product, max_val));
        
        __m256i result_8 = _mm512_cvtepi16_epi8(product);
        _mm256_storeu_si256((__m256i*)&result[i], result_8);
    }
}
```

### 5.3.1 Nonary Logic and Phase Heterodyning

**[ADDENDUM]**

The requirement for a "Wave Interference Processor rather than binary" necessitates a redefinition of arithmetic operations. Logic gates must be implemented as wave interactions (heterodyning) rather than transistor switches.

#### Mathematical Definition of Nonary Operations

**1. Representation:** A value $v \in \{-4, \dots, 4\}$ is encoded as $\Psi_v = A \cdot e^{i \theta}$, where amplitude $A = |v|$ and phase $\theta = 0$ if $v \ge 0$ else $\pi$.

**2. Superposition (Addition):**

$$\Psi_{sum} = \Psi_A + \Psi_B$$

- **Constructive Interference:** $1 + 1 \to 2$ (Amplitudes add)
- **Destructive Interference:** $1 + (-1) \to 0$ (Waves cancel)
- This naturally implements balanced nonary addition

**3. Heterodyning (Multiplication):**

Multiplication corresponds to the mixing of signals. In the frequency domain, multiplying two sinusoids creates sum and difference frequencies. In our coherent time-domain processor, we model this as:

$$\Psi_{prod} = \Psi_A \cdot \Psi_B$$

- **Magnitudes multiply:** $|A| \cdot |B|$
- **Phases add:** $e^{i\theta_A} \cdot e^{i\theta_B} = e^{i(\theta_A + \theta_B)}$
- **Sign Logic:**
  - $(+) \times (+) \to e^{i0} \cdot e^{i0} = e^{i0} \to (+)$
  - $(-) \times (-) \to e^{i\pi} \cdot e^{i\pi} = e^{i2\pi} \equiv e^{i0} \to (+)$
  - $(+) \times (-) \to e^{i0} \cdot e^{i\pi} = e^{i\pi} \to (-)$
- This physically realizes the sign rules of arithmetic without boolean logic gates

## 5.4 Carry Mechanism: Spectral Cascading

When a node's amplitude exceeds $\pm 4.5$ (saturation), a "carry" occurs:

### Algorithm

1. Detect overflow: $|\Psi| > 4.5$
2. Calculate carry: $\text{carry} = \lfloor |\Psi| / 9 \rfloor$
3. Emit pulse at next higher dimension's frequency
4. Generate cancellation wave: $-(\text{carry} \times 9)$ locally
5. Remainder: $|\Psi| \mod 9$

### Example

If $\Psi = +13$:
- Carry: $\lfloor 13 / 9 \rfloor = 1$
- Emit $+1$ pulse to next dimension
- Local cancellation: $-9$
- Remainder: $13 - 9 = +4$

### Implementation

```cpp
void handle_overflow(TorusNode& node, int next_dim_idx) {
    double mag = std::abs(node.wavefunction);
    if (mag > 4.5) {
        int carry = static_cast<int>(mag / 9.0);
        double phase = std::arg(node.wavefunction);

        // Emit carry to next dimension
        inject_wave(next_dim_coords, std::complex<double>(carry, 0));

        // Local cancellation
        double cancel = carry * 9.0;
        node.wavefunction -= std::polar(cancel, phase);
    }
}
```

---

**Cross-References:**
- See Section 4.4.1 (UFIE) for wave propagation equations
- See Section 6 for Wave Interference Processor implementation
- See Appendix B for mathematical foundations of balanced nonary arithmetic
