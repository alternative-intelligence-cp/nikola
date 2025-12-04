# MAMBA-9D STATE SPACE MODEL

## 7.1 Hilbert Curve Linearization

The Mamba architecture requires a 1D sequence, but our data is 9D. We use a **9th-order Hilbert curve** to linearize the grid while preserving locality.

### Hilbert Curve Properties

- **Space-filling:** Visits every grid point exactly once
- **Locality-preserving:** Points close in 9D are close in 1D sequence
- **Recursive:** Defined by recursive subdivision

### Algorithm

```cpp
class HilbertMapper {
public:
    static uint64_t encode(const std::array<uint32_t, 9>& coords, int bits) {
        uint64_t h_index = 0;

        for (int level = bits - 1; level >= 0; --level) {
            uint32_t cell_bits = 0;

            // Extract bit from each dimension
            for (int dim = 0; dim < 9; ++dim) {
                uint32_t bit = (coords[dim] >> level) & 1;
                cell_bits |= (bit << dim);
            }

            // Apply Gray code rotation
            cell_bits = apply_hilbert_rotation(cell_bits, level);

            // Append to index
            h_index = (h_index << 9) | cell_bits;
        }

        return h_index;
    }

private:
    static uint32_t apply_hilbert_rotation(uint32_t bits, int level) {
        // Simplified: Use lookup table for 9D rotations
        return hilbert_rotation_lut_9d[level % ROTATION_TABLE_SIZE][bits];
    }
};
```

### Rotation Table

Pre-computed table of Gray code rotations for 9D Hilbert curve (implementation detail, can use existing libraries).

## 7.2 Variable Rate Sampling

The Mamba scanner adjusts its discretization step $\Delta$ based on local information density:

$$\Delta_k = \frac{\Delta_{\text{base}}}{1 + \alpha \cdot \rho_k \cdot \text{Tr}(g_{ij})}$$

Where:
- $\Delta_{\text{base}}$: Baseline time step (e.g., 0.01)
- $\alpha$: Sensitivity parameter (e.g., 10.0)
- $\rho_k$: Information density at position $k$
- $\text{Tr}(g_{ij})$: Trace of metric tensor (measure of curvature)

### Effect

- **Dense regions:** Small $\Delta$ → High resolution (focus)
- **Empty regions:** Large $\Delta$ → Fast skip (saccade)

### Implementation

```cpp
double compute_adaptive_delta(const TorusNode& node, double base_delta) {
    double density = compute_density(node);
    double trace = compute_metric_trace(node.metric_tensor);

    double alpha = 10.0;
    return base_delta / (1.0 + alpha * density * trace);
}
```

## 7.3 SSM Parameter Mapping

Standard Mamba uses State Space Model parameters $(A, B, C, \Delta)$. In 9D-TWI, these map to physical properties:

| SSM Parameter | 9D-TWI Mapping | Physical Meaning |
|---------------|----------------|------------------|
| $A$ (State Matrix) | Metric Tensor $g_{ij}$ + Resonance $r$ | Memory persistence |
| $B$ (Input Matrix) | State dimension $s$ | Input coupling |
| $C$ (Output Matrix) | Read sensitivity | Output strength |
| $\Delta$ (Time Step) | Adaptive (from density) | Scan resolution |

### Parameter Extraction

```cpp
struct MambaParams {
    Eigen::MatrixXd A;  // 9x9 from metric
    Eigen::VectorXd B;  // 9x1 from state dimension
    Eigen::VectorXd C;  // 9x1 from output weights
    double Delta;       // Adaptive time step
};

MambaParams extract_ssm_params(const TorusNode& node) {
    MambaParams params;

    // A matrix: Metric tensor + damping
    params.A = reconstruct_metric_matrix(node.metric_tensor);
    params.A *= (1.0 - node.resonance_r);  // Damping

    // B vector: Input coupling from state dimension
    params.B = Eigen::VectorXd::Constant(9, node.state_s);

    // CRITICAL FIX (Audit 3 Item #7): C vector from QuantumState projection
    // Problem: Was using static Ones(9) instead of projecting wavefunction
    // Solution: Project the node's QuantumState (u, v, w) into output matrix C
    params.C = Eigen::VectorXd::Zero(9);

    // Project quantum state amplitudes into C vector
    // Dimensions 4, 5, 6 (u, v, w) get quantum component magnitudes
    params.C(3) = std::abs(node.quantum.u);  // Quantum 1 magnitude
    params.C(4) = std::abs(node.quantum.v);  // Quantum 2 magnitude
    params.C(5) = std::abs(node.quantum.w);  // Quantum 3 magnitude

    // Other dimensions weighted by total wavefunction strength
    double total_amplitude = std::abs(node.quantum.total_amplitude());
    params.C(0) = total_amplitude * node.resonance_r;  // Resonance-weighted
    params.C(1) = total_amplitude * node.state_s;      // State-weighted
    params.C(2) = total_amplitude;                      // Time component
    params.C(6) = total_amplitude;                      // Spatial X
    params.C(7) = total_amplitude;                      // Spatial Y
    params.C(8) = total_amplitude;                      // Synchronizer

    // Delta: Adaptive
    params.Delta = compute_adaptive_delta(node, 0.01);

    return params;
}
```

### 7.3.1 Topological State Mapping (TSM)

**[ADDENDUM]**

Standard Mamba (State Space Model) relies on learned matrices $A, B, C$ to process sequences. In Nikola v0.0.4, these matrices are not abstract weights; they are **dynamic projections of the torus geometry**.

#### The Isomorphism Protocol

At any time step $t$, the Mamba scanner traverses the Hilbert curve of the active grid. For each node $i$ visited:

**1. Matrix A (State Transition):** Defined by the local Resonance and Metric Curvature.

$$A_i = \exp(-\Delta \cdot (1 - r_i) \cdot \mathbf{G}_i)$$

where $\mathbf{G}_i$ is the local metric tensor.

**Insight:** Regions with high resonance ($r \to 1$) result in an Identity matrix $A \approx I$, meaning the state is preserved perfectly (Long Term Memory). Regions with low resonance result in decay (Forgetting).

**2. Matrix B (Input Sensitivity):** Defined by the local State dimension.

$$B_i = s_i \cdot \vec{u}_{quantum}$$

**Insight:** The "State" dimension ($s$) acts as the input gate. High $s$ means the node is "paying attention" and will accept new information into its hidden state.

**3. Matrix C (Output Projection):** Defined by the Wavefunction.

$$C_i = \text{Project}(\Psi_i)$$

**Insight:** The output of the Mamba layer is the direct observation of the wave interference pattern at that location.

#### Implementation Consequence

The "learning" of the Mamba model is actually the **Neuroplasticity of the torus** (updating $g_{ij}$, $r$, and $s$). There are no separate "weights" for the Mamba layer; **the geometry of the torus IS the weight set**. This fulfills the requirement **"layers ARE the toroid"** literally.

## 7.4 Implementation

### Mamba Forward Pass

```cpp
class Mamba9D {
    Eigen::VectorXd hidden_state;  // Current SSM state

public:
    // Zero-copy forward pass: operate directly on TorusNode memory
    // Fulfills "layers ARE the toroid" requirement
    Eigen::VectorXd forward(const std::vector<TorusNode*>& sequence) {
        hidden_state = Eigen::VectorXd::Zero(9);

        for (const auto* node_ptr : sequence) {
            // Extract SSM params directly from node
            auto params = extract_ssm_params(*node_ptr);

            // ZERO-COPY: Map TorusNode's coordinate array directly into Eigen vector
            // No intermediate allocation - operates on torus memory in-place
            Eigen::Map<const Eigen::VectorXd> input(
                reinterpret_cast<const double*>(&node_ptr->coord.r),
                9
            );

            // SSM recurrence: h_t = A h_{t-1} + B x_t
            // This operates directly on the physical memory of the toroid
            hidden_state = params.A * hidden_state + params.B.cwiseProduct(input);

            // Scale by adaptive delta
            hidden_state *= params.Delta;

            // OPTIONAL: Write output directly back to node (in-place modification)
            // This ensures the computation happens "in memory" without CPU-RAM separation
            node_ptr->mamba_state = hidden_state;
        }

        return hidden_state;
    }

private:
    struct SSMParams {
        Eigen::MatrixXd A;  // State transition matrix (from metric tensor)
        Eigen::VectorXd B;  // Input projection (from resonance r)
        double Delta;       // Adaptive time step (from state s)
    };

    SSMParams extract_ssm_params(const TorusNode& node) const {
        SSMParams params;

        // A matrix: derived from Christoffel symbols of metric tensor
        // Convert upper-triangular storage (45 elements) to full 9x9 matrix
        Eigen::MatrixXd metric_matrix = reconstruct_metric_matrix(node.metric_tensor);
        params.A = Eigen::MatrixXd::Identity(9, 9) + 0.01 * metric_matrix;

        // B vector: modulated by resonance dimension
        params.B = Eigen::VectorXd::Constant(9, node.resonance_r);

        // Delta: adaptive discretization from state dimension
        params.Delta = 1.0 / (1.0 + node.state_s);

        return params;
    }

    // Helper: Reconstruct full 9x9 symmetric matrix from upper-triangular storage
    static Eigen::MatrixXd reconstruct_metric_matrix(const std::array<float, 45>& compressed) {
        Eigen::MatrixXd expanded(9, 9);

        // Upper-triangular storage formula: index(i,j) = i*9 - i*(i+1)/2 + j (for i <= j)
        int idx = 0;
        for (int i = 0; i < 9; ++i) {
            for (int j = i; j < 9; ++j) {
                expanded(i, j) = compressed[idx];
                expanded(j, i) = compressed[idx];  // Symmetric
                ++idx;
            }
        }

        return expanded;
    }
};
```

## 7.5 Architectural Significance

The Mamba-9D architecture represents a fundamental innovation in AI design:

### Traditional Mamba
- Learned weight matrices $(A, B, C)$
- Fixed discretization $\Delta$
- Weights stored separately from data
- Learning = gradient descent on weights

### Mamba-9D
- **Physical matrices** from torus geometry
- **Adaptive discretization** from information density
- Weights = geometry of memory substrate
- Learning = neuroplastic deformation of spacetime

This architecture ensures that the SSM is not an external layer "on top of" the physics, but rather a **natural consequence** of scanning through a curved, dynamic 9D manifold. The "state space" IS the toroidal space itself.

---

**Cross-References:**
- See Section 3.4 for Neuroplasticity mathematics
- See Section 6 for Wave Interference Processor
- See Section 8 for Neuroplastic Transformer
- See Section 8.3 (Work Package 2) for complete TSM implementation
- See Appendix B for Hilbert curve mathematics
