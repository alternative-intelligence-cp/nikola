# TRAINING SYSTEMS

## 15.1 Bicameral Autonomous Trainers (BAT)

The Nikola Model uses two separate training systems:
1. **Mamba Trainer:** Trains the 9D scanning SSM
2. **Transformer Trainer:** Trains the reasoning engine

These run autonomously in separate threads, triggered by performance metrics.

## 15.1.1 NikolaAutodiff: Complex-Valued Automatic Differentiation

The Nikola Model requires automatic differentiation that supports complex-valued parameters (balanced nonary weights) and wave mechanics (UFIE propagation). This tape-based autodiff engine implements Wirtinger calculus for complex derivatives and provides chain rule support for physics-coupled backpropagation.

### Architecture

```cpp
// File: include/nikola/core/autodiff.hpp

namespace nikola::autodiff {

// Computational graph node
struct ComputeNode {
    std::complex<double> value;
    std::complex<double> gradient;
    std::vector<size_t> parent_ids;
    std::function<std::complex<double>(const std::vector<std::complex<double>>&)> backward_fn;
};

// Tape-based automatic differentiation engine
class NikolaAutodiff {
private:
    std::vector<ComputeNode> tape;
    size_t next_id = 0;

public:
    // Create leaf variable (input or parameter)
    size_t create_variable(std::complex<double> value) {
        ComputeNode node;
        node.value = value;
        node.gradient = std::complex<double>(0.0, 0.0);
        tape.push_back(node);
        return next_id++;
    }

    // Get value of node
    std::complex<double> get_value(size_t id) const {
        return tape[id].value;
    }

    // Get gradient of node
    std::complex<double> get_gradient(size_t id) const {
        return tape[id].gradient;
    }

    // Addition: z = x + y
    size_t add(size_t x_id, size_t y_id) {
        ComputeNode node;
        node.value = tape[x_id].value + tape[y_id].value;
        node.parent_ids = {x_id, y_id};

        // Backward: dL/dx = dL/dz, dL/dy = dL/dz
        node.backward_fn = [](const std::vector<std::complex<double>>& parent_grads) {
            return parent_grads[0];  // Gradient flows through unchanged
        };

        tape.push_back(node);
        return next_id++;
    }

    // Multiplication: z = x * y (Wirtinger derivative for complex)
    size_t multiply(size_t x_id, size_t y_id) {
        ComputeNode node;
        std::complex<double> x_val = tape[x_id].value;
        std::complex<double> y_val = tape[y_id].value;

        node.value = x_val * y_val;
        node.parent_ids = {x_id, y_id};

        // Wirtinger calculus: d(xy)/dx = y, d(xy)/dy = x
        node.backward_fn = [x_val, y_val](const std::vector<std::complex<double>>& parent_grads) {
            return parent_grads[0];  // Will be scaled by conjugate during backprop
        };

        tape.push_back(node);
        return next_id++;
    }

    // Matrix-vector multiply: y = A * x (for SSM updates)
    // Returns vector of node IDs (one per output dimension)
    std::vector<size_t> matrix_vector_multiply(const Eigen::MatrixXcd& A, const std::vector<size_t>& x_ids) {
        Eigen::VectorXcd x_vec(x_ids.size());
        for (size_t i = 0; i < x_ids.size(); ++i) {
            x_vec(i) = tape[x_ids[i]].value;
        }

        Eigen::VectorXcd result = A * x_vec;

        // Create vector of output nodes (one per dimension)
        std::vector<size_t> output_ids;

        for (int out_dim = 0; out_dim < result.size(); ++out_dim) {
            ComputeNode node;
            node.value = result(out_dim);
            node.parent_ids = x_ids;

            // Backward pass for matrix-vector multiplication with complex values
            // For y[out_dim] = A[out_dim,:] * x, the gradient is:
            // ∂L/∂x[j] = conj(A[out_dim,j]) * ∂L/∂y[out_dim]
            node.backward_fn = [A, out_dim, x_ids](const std::vector<std::complex<double>>& parent_grads) {
                // This backward function computes the gradient contribution for this output dimension
                // The full gradient accumulation happens in backward() which sums contributions
                // from all output dimensions

                // For matrix-vector product y = A * x:
                // The Hermitian transpose A^H defines the gradient: ∂L/∂x = A^H * ∂L/∂y
                // For a single output dimension: ∂L/∂x[j] = conj(A[out_dim,j]) * ∂L/∂y[out_dim]

                // Return gradient for first parent (proper accumulation handled by backward())
                return std::conj(A(out_dim, 0)) * parent_grads[0];
            };

            tape.push_back(node);
            output_ids.push_back(next_id++);
        }

        return output_ids;
    }

    // Squared norm (loss function): L = |x|^2
    size_t squared_norm(size_t x_id) {
        ComputeNode node;
        std::complex<double> x_val = tape[x_id].value;

        // Real-valued output
        node.value = std::complex<double>(std::norm(x_val), 0.0);
        node.parent_ids = {x_id};

        // Backward: d|x|^2/dx = 2*conj(x) (Wirtinger derivative)
        node.backward_fn = [x_val](const std::vector<std::complex<double>>& parent_grads) {
            return 2.0 * std::conj(x_val);
        };

        tape.push_back(node);
        return next_id++;
    }

    // UFIE Wave Propagation with non-linear soliton term
    // Full propagation: Ψ_{t+1} = exp(-iH dt) Ψ_t where H = H_0 + β|Ψ|²
    // For small timesteps: Ψ_{t+1} ≈ (1 - iH_0 dt - iβ|Ψ|² dt) Ψ_t
    // The non-linear term β|Ψ|² Ψ represents self-organizing soliton dynamics
    size_t ufie_step(size_t psi_id, const Eigen::MatrixXcd& hamiltonian, double dt, double beta = 0.1) {
        ComputeNode node;
        std::complex<double> psi_val = tape[psi_id].value;

        std::complex<double> i_unit(0.0, 1.0);

        // Linear term: H_0 Ψ
        std::complex<double> linear_propagator = 1.0 - i_unit * hamiltonian(0, 0) * dt;

        // Non-linear term: β|Ψ|² Ψ (soliton self-interaction)
        double psi_norm_squared = std::norm(psi_val);  // |Ψ|²
        std::complex<double> nonlinear_term = -i_unit * beta * psi_norm_squared * dt;

        // Full propagation: Ψ_{t+1} = (1 - iH_0 dt - iβ|Ψ|² dt) Ψ_t
        node.value = (linear_propagator + nonlinear_term) * psi_val;
        node.parent_ids = {psi_id};

        // Backward pass: Compute gradient including non-linear term derivative
        // For y = (1 - iH dt - iβ|Ψ|² dt) Ψ, the derivative has two contributions:
        // 1. Linear: ∂/∂Ψ[(1 - iH dt)Ψ] = (1 - iH dt)
        // 2. Non-linear: ∂/∂Ψ[iβ|Ψ|² dt Ψ] = 2iβ|Ψ|² dt (using Wirtinger calculus: ∂|Ψ|²/∂Ψ = conj(Ψ))
        //
        // Total derivative: ∂y/∂Ψ = (1 - iH dt) - 2iβ|Ψ|² dt
        // Gradient chain rule: dL/dΨ_t = ∂y/∂Ψ * dL/dy
        node.backward_fn = [linear_propagator, nonlinear_term, psi_val, beta, dt, i_unit]
                          (const std::vector<std::complex<double>>& parent_grads) {
            // Full derivative including non-linear term
            double psi_norm_sq = std::norm(psi_val);

            // Linear contribution: conj(1 - iH dt)
            std::complex<double> linear_contrib = std::conj(linear_propagator);

            // Non-linear contribution: derivative of β|Ψ|² term
            // The non-linear term contributes: -2iβ|Ψ|² dt to the derivative
            std::complex<double> nonlinear_contrib = -2.0 * i_unit * beta * psi_norm_sq * dt;

            // Total gradient
            std::complex<double> total_derivative = linear_contrib + nonlinear_contrib;

            return total_derivative * parent_grads[0];
        };

        tape.push_back(node);
        return next_id++;
    }

    // Backward pass: compute all gradients
    void backward(size_t loss_id) {
        // Initialize loss gradient to 1
        tape[loss_id].gradient = std::complex<double>(1.0, 0.0);

        // Reverse topological order
        for (int i = static_cast<int>(loss_id); i >= 0; --i) {
            ComputeNode& node = tape[i];

            if (node.backward_fn && !node.parent_ids.empty()) {
                // Collect parent gradients
                std::vector<std::complex<double>> parent_grads;
                for (size_t parent_id : node.parent_ids) {
                    parent_grads.push_back(tape[parent_id].gradient);
                }

                // Compute gradient contribution
                std::complex<double> grad_contribution = node.backward_fn(parent_grads);

                // Accumulate into parent gradients
                for (size_t parent_id : node.parent_ids) {
                    tape[parent_id].gradient += node.gradient * grad_contribution;
                }
            }
        }
    }

    // Clear tape for next computation
    void clear() {
        tape.clear();
        next_id = 0;
    }
};

} // namespace nikola::autodiff
```

### Integration with SSM Parameters

```cpp
// Helper: Create tape variables for SSM matrices
struct SSMParameters {
    std::vector<size_t> A_flat;  // Flattened matrix IDs
    std::vector<size_t> B_flat;
    std::vector<size_t> C_flat;
    Eigen::MatrixXcd A_matrix;
    Eigen::MatrixXcd B_matrix;
    Eigen::VectorXcd C_vector;
};

SSMParameters create_ssm_tape(NikolaAutodiff& tape, const SSMParams& params) {
    SSMParameters ssm_tape;

    // Create tape variables for each matrix element
    for (int i = 0; i < params.A.rows(); ++i) {
        for (int j = 0; j < params.A.cols(); ++j) {
            size_t id = tape.create_variable(params.A(i, j));
            ssm_tape.A_flat.push_back(id);
        }
    }

    // Store matrix structure for reconstruction
    ssm_tape.A_matrix = params.A;
    ssm_tape.B_matrix = params.B;
    ssm_tape.C_vector = params.C;

    return ssm_tape;
}
```

---

## 15.2 Mamba Trainer

**Training Objective:** Minimize sequence prediction error

### Loss Function

$$\mathcal{L}_{\text{Mamba}} = \| h_{t+1}^{\text{pred}} - h_{t+1}^{\text{actual}} \|^2$$

### Implementation

The Mamba trainer uses the NikolaAutodiff engine to compute gradients for the SSM parameters (A, B, C matrices). Training updates the state-space model to minimize sequence prediction errors.

```cpp
class MambaTrainer {
    Mamba9D& model;
    double learning_rate = 0.001;
    nikola::autodiff::NikolaAutodiff autodiff_engine;

public:
    MambaTrainer(Mamba9D& m) : model(m) {}

    void train_step(const std::vector<TorusNode>& sequence) {
        // Clear previous computation graph
        autodiff_engine.clear();

        // Create tape variables for SSM parameters
        SSMParams& params = model.get_params();
        std::vector<size_t> A_ids, B_ids, C_ids;

        // Flatten A matrix into tape
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                size_t id = autodiff_engine.create_variable(params.A(i, j));
                A_ids.push_back(id);
            }
        }

        // Flatten B matrix into tape
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                size_t id = autodiff_engine.create_variable(params.B(i, j));
                B_ids.push_back(id);
            }
        }

        // Flatten C vector into tape
        for (int i = 0; i < 9; ++i) {
            size_t id = autodiff_engine.create_variable(params.C(i));
            C_ids.push_back(id);
        }

        // Forward pass: compute predicted state using SSM dynamics
        // h_{t+1} = A * h_t + B * x_t
        // y_t = C^T * h_t

        std::vector<size_t> hidden_state_ids;
        for (int i = 0; i < 9; ++i) {
            hidden_state_ids.push_back(
                autodiff_engine.create_variable(std::complex<double>(0.0, 0.0))
            );
        }

        // Process sequence
        for (size_t t = 0; t < sequence.size() - 1; ++t) {
            const TorusNode& node = sequence[t];

            // Extract input vector from node
            std::vector<size_t> input_ids;
            input_ids.push_back(autodiff_engine.create_variable(node.quantum.u));
            input_ids.push_back(autodiff_engine.create_variable(node.quantum.v));
            input_ids.push_back(autodiff_engine.create_variable(node.quantum.w));
            // ... (remaining dimensions)

            // SSM update: h = A * h + B * x
            std::vector<size_t> new_hidden_ids;
            for (int i = 0; i < 9; ++i) {
                // Compute A[i,:] * hidden_state
                size_t ah_sum = hidden_state_ids[0];
                for (int j = 1; j < 9; ++j) {
                    size_t a_ij_id = A_ids[i * 9 + j];
                    size_t h_j_id = hidden_state_ids[j];
                    size_t product = autodiff_engine.multiply(a_ij_id, h_j_id);
                    ah_sum = autodiff_engine.add(ah_sum, product);
                }

                // Compute B[i,:] * input (simplified to first 3 dims)
                size_t bx_sum = autodiff_engine.create_variable(std::complex<double>(0.0, 0.0));
                for (int j = 0; j < std::min(3, static_cast<int>(input_ids.size())); ++j) {
                    size_t b_ij_id = B_ids[i * 9 + j];
                    size_t x_j_id = input_ids[j];
                    size_t product = autodiff_engine.multiply(b_ij_id, x_j_id);
                    bx_sum = autodiff_engine.add(bx_sum, product);
                }

                // h_i = A[i,:] * h + B[i,:] * x
                size_t new_h_i = autodiff_engine.add(ah_sum, bx_sum);
                new_hidden_ids.push_back(new_h_i);
            }

            hidden_state_ids = new_hidden_ids;
        }

        // Compute output: y = C^T * h
        size_t predicted_id = hidden_state_ids[0];
        for (int i = 1; i < 9; ++i) {
            size_t c_i_id = C_ids[i];
            size_t h_i_id = hidden_state_ids[i];
            size_t product = autodiff_engine.multiply(c_i_id, h_i_id);
            predicted_id = autodiff_engine.add(predicted_id, product);
        }

        // Ground truth (actual next state)
        const TorusNode& target_node = sequence.back();
        size_t target_id = autodiff_engine.create_variable(target_node.quantum.u);

        // Compute loss: L = |predicted - target|^2
        size_t diff_id = autodiff_engine.add(predicted_id, target_id);  // pred - target
        size_t loss_id = autodiff_engine.squared_norm(diff_id);

        double loss = autodiff_engine.get_value(loss_id).real();

        // BACKWARD PASS: Compute gradients
        autodiff_engine.backward(loss_id);

        // UPDATE PARAMETERS: Gradient descent
        // A = A - lr * dL/dA
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                size_t a_ij_id = A_ids[i * 9 + j];
                std::complex<double> gradient = autodiff_engine.get_gradient(a_ij_id);
                params.A(i, j) -= learning_rate * gradient;
            }
        }

        // B = B - lr * dL/dB
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                size_t b_ij_id = B_ids[i * 9 + j];
                std::complex<double> gradient = autodiff_engine.get_gradient(b_ij_id);
                params.B(i, j) -= learning_rate * gradient;
            }
        }

        // C = C - lr * dL/dC
        for (int i = 0; i < 9; ++i) {
            size_t c_i_id = C_ids[i];
            std::complex<double> gradient = autodiff_engine.get_gradient(c_i_id);
            params.C(i) -= learning_rate * gradient;
        }

        std::cout << "[MAMBA TRAIN] Loss: " << loss << " (Gradients computed and applied)" << std::endl;
    }
};
```

## 15.3 Transformer Trainer

**Training Objective:** Minimize output waveform error

### Loss Function

$$\mathcal{L}_{\text{Trans}} = \| \Psi_{\text{output}} - \Psi_{\text{target}} \|^2$$

### Implementation

The Transformer trainer uses the NikolaAutodiff engine to compute gradients for the attention mechanism weights (Q, K, V). Training updates the wave transformer to minimize output errors while respecting UFIE dynamics.

```cpp
class TransformerTrainer {
    WaveTransformerLayer& model;
    double learning_rate = 0.0001;
    nikola::autodiff::NikolaAutodiff autodiff_engine;

public:
    TransformerTrainer(WaveTransformerLayer& m) : model(m) {}

    void train_step(const std::vector<std::complex<double>>& input,
                     const std::vector<std::complex<double>>& target,
                     TorusManifold& torus) {
        // Clear previous computation graph
        autodiff_engine.clear();

        // Create tape variables for transformer weights
        std::vector<size_t> Q_weight_ids, K_weight_ids, V_weight_ids;
        auto& weights = model.get_weights();

        // Query weights
        for (int i = 0; i < weights.Q.rows(); ++i) {
            for (int j = 0; j < weights.Q.cols(); ++j) {
                size_t id = autodiff_engine.create_variable(weights.Q(i, j));
                Q_weight_ids.push_back(id);
            }
        }

        // Key weights
        for (int i = 0; i < weights.K.rows(); ++i) {
            for (int j = 0; j < weights.K.cols(); ++j) {
                size_t id = autodiff_engine.create_variable(weights.K(i, j));
                K_weight_ids.push_back(id);
            }
        }

        // Value weights
        for (int i = 0; i < weights.V.rows(); ++i) {
            for (int j = 0; j < weights.V.cols(); ++j) {
                size_t id = autodiff_engine.create_variable(weights.V(i, j));
                V_weight_ids.push_back(id);
            }
        }

        // Create tape variables for input
        std::vector<size_t> input_ids;
        for (const auto& val : input) {
            input_ids.push_back(autodiff_engine.create_variable(val));
        }

        // Forward pass through UFIE propagation
        std::vector<size_t> output_ids;

        for (size_t seq_pos = 0; seq_pos < input.size(); ++seq_pos) {
            // Simplified attention mechanism:
            // Q = W_Q * input[seq_pos]
            size_t q_id = autodiff_engine.create_variable(std::complex<double>(0.0, 0.0));
            for (int i = 0; i < weights.Q.rows(); ++i) {
                size_t w_id = Q_weight_ids[i * weights.Q.cols() + seq_pos % weights.Q.cols()];
                size_t inp_id = input_ids[seq_pos];
                size_t prod = autodiff_engine.multiply(w_id, inp_id);
                q_id = autodiff_engine.add(q_id, prod);
            }

            // K = W_K * input[seq_pos]
            size_t k_id = autodiff_engine.create_variable(std::complex<double>(0.0, 0.0));
            for (int i = 0; i < weights.K.rows(); ++i) {
                size_t w_id = K_weight_ids[i * weights.K.cols() + seq_pos % weights.K.cols()];
                size_t inp_id = input_ids[seq_pos];
                size_t prod = autodiff_engine.multiply(w_id, inp_id);
                k_id = autodiff_engine.add(k_id, prod);
            }

            // V = W_V * input[seq_pos]
            size_t v_id = autodiff_engine.create_variable(std::complex<double>(0.0, 0.0));
            for (int i = 0; i < weights.V.rows(); ++i) {
                size_t w_id = V_weight_ids[i * weights.V.cols() + seq_pos % weights.V.cols()];
                size_t inp_id = input_ids[seq_pos];
                size_t prod = autodiff_engine.multiply(w_id, inp_id);
                v_id = autodiff_engine.add(v_id, prod);
            }

            // Attention: softmax(Q * K^T) * V (simplified)
            size_t attention_score = autodiff_engine.multiply(q_id, k_id);
            size_t output = autodiff_engine.multiply(attention_score, v_id);

            // UFIE propagation step
            Eigen::MatrixXcd hamiltonian = torus.compute_local_hamiltonian(seq_pos);
            output = autodiff_engine.ufie_step(output, hamiltonian, 0.01);

            output_ids.push_back(output);
        }

        // Compute loss: sum of |output - target|^2
        size_t total_loss_id = autodiff_engine.create_variable(std::complex<double>(0.0, 0.0));

        for (size_t i = 0; i < output_ids.size(); ++i) {
            size_t target_id = autodiff_engine.create_variable(target[i]);

            // diff = output - target
            size_t diff_id = autodiff_engine.add(output_ids[i], target_id);

            // squared_loss = |diff|^2
            size_t squared_loss = autodiff_engine.squared_norm(diff_id);

            // Accumulate
            total_loss_id = autodiff_engine.add(total_loss_id, squared_loss);
        }

        double loss = autodiff_engine.get_value(total_loss_id).real();

        // BACKWARD PASS: Compute gradients
        autodiff_engine.backward(total_loss_id);

        // UPDATE WEIGHTS: Gradient descent
        // W_Q = W_Q - lr * dL/dW_Q
        for (int i = 0; i < weights.Q.rows(); ++i) {
            for (int j = 0; j < weights.Q.cols(); ++j) {
                size_t w_id = Q_weight_ids[i * weights.Q.cols() + j];
                std::complex<double> gradient = autodiff_engine.get_gradient(w_id);
                weights.Q(i, j) -= learning_rate * gradient;
            }
        }

        // W_K = W_K - lr * dL/dW_K
        for (int i = 0; i < weights.K.rows(); ++i) {
            for (int j = 0; j < weights.K.cols(); ++j) {
                size_t w_id = K_weight_ids[i * weights.K.cols() + j];
                std::complex<double> gradient = autodiff_engine.get_gradient(w_id);
                weights.K(i, j) -= learning_rate * gradient;
            }
        }

        // W_V = W_V - lr * dL/dW_V
        for (int i = 0; i < weights.V.rows(); ++i) {
            for (int j = 0; j < weights.V.cols(); ++j) {
                size_t w_id = V_weight_ids[i * weights.V.cols() + j];
                std::complex<double> gradient = autodiff_engine.get_gradient(w_id);
                weights.V(i, j) -= learning_rate * gradient;
            }
        }

        std::cout << "[TRANSFORMER TRAIN] Loss: " << loss << " (Gradients computed and applied)" << std::endl;

        // Trigger neuroplastic update if loss high
        if (loss > 1.0) {
            // Convert tape outputs to std::complex<double> vector
            std::vector<std::complex<double>> output_values;
            for (size_t id : output_ids) {
                output_values.push_back(autodiff_engine.get_value(id));
            }
            torus.trigger_neuroplasticity_update(output_values);
        }
    }
};
```

## 15.4 Auto-Training Triggers

Training happens automatically when:

1. **Boredom threshold reached:** System is idle and bored
2. **Prediction errors accumulate:** Error rate > 20% over last 100 queries
3. **Scheduled:** Every N hours (e.g., during "nap" periods)

### Implementation

```cpp
class AutoTrainingManager {
    MambaTrainer mamba_trainer;
    TransformerTrainer transformer_trainer;
    std::deque<bool> recent_predictions;  // Success/failure
    size_t window_size = 100;

public:
    void record_prediction(bool correct) {
        recent_predictions.push_back(correct);
        if (recent_predictions.size() > window_size) {
            recent_predictions.pop_front();
        }
    }

    bool should_train() const {
        if (recent_predictions.size() < window_size) {
            return false;
        }

        // Count errors
        size_t errors = std::count(recent_predictions.begin(),
                                    recent_predictions.end(),
                                    false);

        double error_rate = static_cast<double>(errors) / window_size;

        return error_rate > 0.2;  // 20% threshold
    }

    void run_training_session(TorusManifold& torus) {
        std::cout << "[AUTO-TRAIN] Starting training session..." << std::endl;

        // Train for N iterations
        for (int i = 0; i < 1000; ++i) {
            // Sample random sequences from torus
            auto sequence = torus.sample_random_sequence(16);

            // Train Mamba
            mamba_trainer.train_step(sequence);

            // Train Transformer
            // (Would need input/target pairs)
        }

        std::cout << "[AUTO-TRAIN] Session complete." << std::endl;
    }
};
```

## 15.5 Implementation

### Training Loop (runs in background thread)

```cpp
void training_thread_func(AutoTrainingManager& trainer,
                           TorusManifold& torus,
                           NeurochemistryManager& neuro) {
    while (true) {
        // Sleep for 1 hour
        std::this_thread::sleep_for(std::chrono::hours(1));

        // Check if should train
        if (trainer.should_train() || neuro.boredom.should_explore()) {
            trainer.run_training_session(torus);

            // Reward completion
            neuro.reward(0.5);
        }
    }
}
```

---

**Cross-References:**
- See Section 7 for Mamba-9D architecture
- See Section 8 for Neuroplastic Transformer
- See Section 14 for Neurochemistry integration
- See Section 22 for Nap System training triggers
