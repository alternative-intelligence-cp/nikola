# APPENDIX A: MATHEMATICAL FOUNDATIONS

## A.1 Nonary Arithmetic Examples

### A.1.1 Addition (Superposition)

Balanced nonary addition operates through constructive and destructive interference:

```
Addition Rules:
  +2 + +3 = +4  (saturates at max)
  +1 + (-1) = 0  (destructive interference)
  -3 + -2 = -4  (saturates at min)
  +2 + +1 = +3  (normal addition)
  -2 + (-2) = -4  (normal addition, saturates)
```

**Physical Interpretation:**
- Positive values = In-phase waves
- Negative values = Out-of-phase waves (π phase shift)
- Addition = Superposition of amplitudes

### A.1.2 Multiplication (Heterodyning)

Multiplication represents wave mixing in the frequency domain:

```
Multiplication Rules:
  +2 × +2 = +4
  +3 × +2 = +4  (saturates at +4)
  +1 × (-1) = -1
  +2 × +3 = +4  (6 saturates to max)
  -2 × -3 = +4  (6 saturates to max)
```

**Sign Logic:**
- (+) × (+) → (+)  (phases add: 0 + 0 = 0)
- (-) × (-) → (+)  (phases add: π + π = 2π ≡ 0)
- (+) × (-) → (-)  (phases add: 0 + π = π)

### A.1.3 Carry (Spectral Cascading)

When operations exceed the [-4, +4] range, carry to adjacent dimension:

```
Carry Mechanism:
  If node amplitude = +7:
    Carry = ⌊7/9⌋ = 0
    Remainder = 7 mod 9 = +7 → saturate → +4
    (No carry needed)

  If node amplitude = +13:
    Carry = ⌊13/9⌋ = 1
    Emit +1 to next dimension
    Local remainder = 13 - 9 = +4

  If node amplitude = -11:
    Carry = ⌈-11/9⌉ = -2
    Emit -2 to next dimension
    Local remainder = -11 + 18 = +7 → saturate → +4
```

**Implementation:**

```cpp
// Voronoi quantization in complex plane for balanced nonary distribution
Nit quantize_wave(std::complex<double> wave) {
    // Define Voronoi cell centers for each Nit value in complex plane
    static const std::array<std::complex<double>, 9> voronoi_centers = {{
        {0.0, 0.0},        // ZERO
        {1.0, 0.0},        // P1
        {2.0, 0.0},        // P2
        {3.0, 0.0},        // P3
        {4.0, 0.0},        // P4
        {-1.0, 0.0},       // N1
        {-2.0, 0.0},       // N2
        {-3.0, 0.0},       // N3
        {-4.0, 0.0}        // N4
    }};

    static const std::array<Nit, 9> nit_values = {
        Nit::ZERO, Nit::P1, Nit::P2, Nit::P3, Nit::P4,
        Nit::N1, Nit::N2, Nit::N3, Nit::N4
    };

    // Find nearest Voronoi cell center (minimum Euclidean distance)
    size_t nearest_idx = 0;
    double min_distance = std::abs(wave - voronoi_centers[0]);

    for (size_t i = 1; i < voronoi_centers.size(); ++i) {
        double distance = std::abs(wave - voronoi_centers[i]);
        if (distance < min_distance) {
            min_distance = distance;
            nearest_idx = i;
        }
    }

    return nit_values[nearest_idx];
}
```

---

## A.2 Metric Tensor Index Mapping

### A.2.1 Symmetric Matrix Storage

For a symmetric 9×9 metric tensor, store only the upper triangle to save memory:

**Storage Layout:**

```
Total elements in symmetric matrix = n(n+1)/2 = 9×10/2 = 45 elements
```

**Index Mapping Formula:**

```
For (i, j) where i ≤ j:
    Linear Index = i × 9 - i(i+1)/2 + j
```

**Example Mappings:**

| Matrix Index (i,j) | Linear Index | Value |
|-------------------|--------------|-------|
| (0, 0) | 0 | g₀₀ |
| (0, 1) | 1 | g₀₁ |
| (0, 8) | 8 | g₀₈ |
| (1, 1) | 9 | g₁₁ |
| (1, 2) | 10 | g₁₂ |
| (2, 2) | 17 | g₂₂ |
| (8, 8) | 44 | g₈₈ |

### A.2.2 Access Functions

```cpp
// Convert (i, j) to linear index
inline int metric_index(int i, int j) {
    if (i > j) std::swap(i, j);  // Ensure i ≤ j
    return i * 9 - i * (i + 1) / 2 + j;
}

// Access metric tensor element
double get_metric(const std::array<float, 45>& metric, int i, int j) {
    return metric[metric_index(i, j)];
}

// Set metric tensor element (preserves symmetry)
void set_metric(std::array<float, 45>& metric, int i, int j, double value) {
    metric[metric_index(i, j)] = value;
}
```

---

## A.3 Hilbert Curve Properties

### A.3.1 Definition

The 9D Hilbert curve is a space-filling curve that maps a 1D sequence to 9D space while preserving locality.

**Mathematical Properties:**

For a Hilbert curve with $b$ bits per dimension:

| Property | Formula | Example ($b=10$) |
|----------|---------|------------------|
| Total points | $2^{9b}$ | $2^{90} \approx 1.24 \times 10^{27}$ |
| Index range | $[0, 2^{9b} - 1]$ | $[0, 2^{90} - 1]$ |
| Coordinate range | $[0, 2^b - 1]$ per dim | $[0, 1023]$ |

### A.3.2 Locality Preservation

**Theorem:** If two points are close in Hilbert index space, they are close in 9D Euclidean space.

**Formal Statement:**

$$|\text{index}_A - \text{index}_B| < \delta \implies ||\vec{coord}_A - \vec{coord}_B|| < \epsilon$$

Where:
- $\delta$ = Index distance threshold
- $\epsilon$ = Euclidean distance threshold
- Relationship: $\epsilon \propto \delta^{1/9}$ (fractal dimension)

### A.3.3 Implementation

```cpp
// Encode 9D coordinates to Hilbert index
uint64_t encode_hilbert(const Coord9D& coord, int bits_per_dim) {
    // Uses Gray code and bit interleaving
    // Implementation based on Compact Hilbert Indices algorithm
    // See: https://doc.cgal.org/latest/Spatial_sorting/index.html

    uint64_t index = 0;
    // ... (omitted for brevity - see full implementation in src/mamba/hilbert_scan.cpp)
    return index;
}

// Decode Hilbert index to 9D coordinates
Coord9D decode_hilbert(uint64_t index, int bits_per_dim) {
    Coord9D coord;
    // Reverse Gray code transformation
    // ... (omitted - see implementation)
    return coord;
}
```

---

## A.4 Wave Equations

### A.4.1 Standard Wave Equation

The classical wave equation in $n$ dimensions:

$$\frac{\partial^2 \Psi}{\partial t^2} = c^2 \nabla^2 \Psi$$

Where:
- $\Psi(\vec{x}, t)$ = Wavefunction (complex-valued)
- $c$ = Wave propagation speed
- $\nabla^2$ = Laplacian operator

### A.4.2 Discretized Form (FTDT - Finite Time-Domain Transform)

For numerical simulation, discretize in space and time:

$$\Psi_{i,t+1} = \Psi_{i,t} + \Delta t \left[ c^2 \sum_j (\Psi_{j,t} - \Psi_{i,t}) - \gamma \Psi_{i,t} \right]$$

Where:
- $i$ = Node index in 9D lattice
- $j$ = Neighbors of node $i$ (up to 18 in 9D)
- $\Delta t$ = Time step (typically 0.01)
- $\gamma$ = Damping coefficient

### A.4.3 Damping Term

Damping is controlled by the **resonance dimension** ($r$):

$$\gamma = \alpha (1 - \hat{r})$$

Where:
- $\alpha$ = Baseline damping rate (typically 0.01)
- $\hat{r}$ = Normalized resonance value in [0, 1]
- If $r \to 1$: Damping $\to 0$ (perfect memory retention)
- If $r \to 0$: Damping $\to \alpha$ (rapid forgetting)

### A.4.4 Wave Velocity Modulation

Wave speed is controlled by the **state dimension** ($s$):

$$c_{eff} = \frac{c_0}{1 + \hat{s}}$$

Where:
- $c_0$ = Baseline wave speed
- $\hat{s}$ = Normalized state value in [0, 1]
- If $s \to 1$: Waves slow down (increased interaction time = "focus")
- If $s \to 0$: Waves propagate at full speed

### A.4.5 Unified Field Interference Equation (UFIE)

**Complete Master Equation:**

$$\frac{\partial^2 \Psi}{\partial t^2} + \alpha(1 - \hat{r}) \frac{\partial \Psi}{\partial t} - \frac{c_0^2}{(1 + \hat{s})^2} \nabla^2_g \Psi = \sum_{i=1}^8 \mathcal{E}_i(\vec{x}, t) + \beta |\Psi|^2 \Psi$$

**Term-by-Term Breakdown:**

| Term | Physical Meaning | Engineering Implementation |
|------|------------------|---------------------------|
| $\nabla^2_g \Psi$ | Laplace-Beltrami Operator | Wave propagation over curved metric $g_{ij}$ (neuroplastic manifold) |
| $\alpha(1 - \hat{r})$ | Resonance Damping | If $r \to 1$ (high resonance), damping $\to 0$ (persistent memory) |
| $c_0^2 / (1 + \hat{s})^2$ | Refractive Index | High state $s$ slows waves, increasing interaction time ("attention") |
| $\sum \mathcal{E}_i$ | Emitter Injection | External signal injection from 8 golden ratio harmonic emitters |
| $\beta |\Psi|^2 \Psi$ | Nonlinearity | Self-interaction term (optional, enables solitons) |

---

## A.5 Riemannian Geometry on Torus

### A.5.1 Metric Tensor

Each node has a $9 \times 9$ metric tensor $g_{ij}$ defining local curvature:

$$ds^2 = \sum_{i,j=0}^{8} g_{ij} \, dx^i \, dx^j$$

**Physical Interpretation:**
- Flat space: $g_{ij} = \delta_{ij}$ (identity matrix)
- Curved space: Off-diagonal elements $\neq 0$
- Neuroplasticity: Co-activation → metric contraction

### A.5.2 Geodesic Distance

Distance between two points on curved manifold:

$$d(\vec{x}_A, \vec{x}_B) = \int_{\gamma} \sqrt{g_{ij}(\gamma(s)) \dot{\gamma}^i(s) \dot{\gamma}^j(s)} \, ds$$

**Approximation for Small Distances:**

$$d \approx \sqrt{\sum_{i,j} g_{ij} \Delta x^i \Delta x^j}$$

Where $\Delta x^i = x_B^i - x_A^i$.

### A.5.3 Neuroplastic Update Rule

**Hebbian Learning:** "Neurons that fire together, wire together"

When nodes $A$ and $B$ co-activate:

$$g_{ij}^{new} = g_{ij}^{old} - \eta \cdot \text{activation}_A \cdot \text{activation}_B \cdot (g_{ij}^{old} - g_{ij}^{min})$$

Where:
- $\eta$ = Learning rate (typically 0.01)
- $g_{ij}^{min}$ = Minimum metric value (prevents collapse)
- Effect: Distance between $A$ and $B$ decreases

---

## A.6 Golden Ratio and Ergodicity

### A.6.1 Emitter Frequency Series

**Golden Ratio Series:**

$$f_n = \pi \cdot \phi^n \quad \text{where } \phi = \frac{1 + \sqrt{5}}{2} \approx 1.618033988749895$$

**Emitter Frequencies (Hz):**

| Emitter | $n$ | Frequency ($\pi \phi^n$) | Cognitive Function |
|---------|-----|--------------------------|-------------------|
| 0 | 1 | 5.083 Hz | Metacognitive timing |
| 1 | 2 | 8.225 Hz | Working memory (theta) |
| 2 | 3 | 13.308 Hz | Relaxation (alpha) |
| 3 | 4 | 21.532 Hz | Alertness (beta) |
| 4 | 5 | 34.840 Hz | Low gamma binding |
| 5 | 6 | 56.371 Hz | High gamma attention |
| 6 | 7 | 91.210 Hz | Fast ripples (consolidation) |
| 7 | 8 | 147.580 Hz | X-spatial frequency |

### A.6.2 Ergodicity Proof (Simplified)

**Theorem:** The golden ratio frequency series prevents resonance lock-in.

**Proof Sketch:**

A resonance (stable loop) occurs if:

$$\sum_{n=1}^9 k_n \omega_n = 0 \quad \text{for some } \vec{k} \in \mathbb{Z}^9 \setminus \{\vec{0}\}$$

Substituting $\omega_n = \pi \phi^n$:

$$\sum_{n=1}^9 k_n \phi^n = 0$$

Since $\phi$ is irrational and a Pisot-Vijayaraghavan number (root of $x^2 - x - 1 = 0$), any power $\phi^n$ can be reduced to:

$$\phi^n = F_n \phi + F_{n-1}$$

Where $F_n$ are Fibonacci numbers.

Substituting yields:

$$A + B\phi = 0$$

For integers $A, B$. Since $\phi$ is irrational, this holds **if and only if** $A = 0$ and $B = 0$.

For the range $n \in \{1, \ldots, 8\}$ and reasonable bounds on $k_n$, the only solution is the trivial $\vec{k} = \vec{0}$.

**Engineering Implication:** The system will never hallucinate due to harmonic resonance lock-in. The phase space is fully explored.

---

## A.7 Fourier Transform Properties

### A.7.1 Discrete Fourier Transform (DFT)

Used for audio processing and spectral analysis:

$$X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-i 2\pi k n / N}$$

Where:
- $x[n]$ = Time-domain samples
- $X[k]$ = Frequency-domain bins
- $N$ = FFT size (typically 16384)

### A.7.2 Frequency Bin Calculation

Map FFT bins to emitter frequencies:

$$\text{bin}(f) = \left\lfloor \frac{f \cdot N}{f_s} \right\rfloor$$

Where:
- $f$ = Target frequency (Hz)
- $f_s$ = Sampling rate (Hz, typically 44100)
- $N$ = FFT size

**Example:**

For emitter 4 ($f = 34.840$ Hz, $f_s = 44100$ Hz, $N = 16384$):

$$\text{bin} = \left\lfloor \frac{34.840 \times 16384}{44100} \right\rfloor = 12$$

---

## A.8 Coordinate Wrapping and Toroidal Topology

### A.8.1 Modular Arithmetic for Wrapping

**Toroidal Wrapping Formula:**

```cpp
void Coord9D::wrap(const std::array<int32_t, 9>& dimensions) {
    for (size_t i = 0; i < 9; ++i) {
        if (coords[i] < 0) {
            // Handle negative wrapping
            coords[i] = (coords[i] % dimensions[i] + dimensions[i]) % dimensions[i];
        } else {
            // Handle positive wrapping
            coords[i] = coords[i] % dimensions[i];
        }
    }
}
```

**Mathematical Property:**

For dimension size $D$:
- Coordinate $x = D$ wraps to $x = 0$
- Coordinate $x = -1$ wraps to $x = D-1$

### A.8.2 Geodesic Distance on Torus

**Shortest Path Accounting for Wrapping:**

```cpp
int32_t toroidal_distance_1d(int32_t a, int32_t b, int32_t dim_size) {
    int32_t direct = std::abs(b - a);
    int32_t wrapped = dim_size - direct;
    return std::min(direct, wrapped);
}

double Coord9D::distance_to(const Coord9D& other,
                             const std::array<int32_t, 9>& dims) const {
    double sum = 0.0;
    for (size_t i = 0; i < 9; ++i) {
        int32_t dist = toroidal_distance_1d(coords[i], other.coords[i], dims[i]);
        sum += dist * dist;
    }
    return std::sqrt(sum);
}
```

---

**Cross-References:**
- See Section 2 for Nonary Physics implementation
- See Section 4 for Wave Propagation details
- See Section 6 for Hilbert curve usage in Mamba-9D
- See Appendix H for complete theoretical derivations

