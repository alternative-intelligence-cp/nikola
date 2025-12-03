# TRAINING SYSTEMS

## 15.1 Bicameral Autonomous Trainers (BAT)

The Nikola Model uses two separate training systems:
1. **Mamba Trainer:** Trains the 9D scanning SSM
2. **Transformer Trainer:** Trains the reasoning engine

These run autonomously in separate threads, triggered by performance metrics.

## 15.2 Mamba Trainer

**Training Objective:** Minimize sequence prediction error

### Loss Function

$$\mathcal{L}_{\text{Mamba}} = \| h_{t+1}^{\text{pred}} - h_{t+1}^{\text{actual}} \|^2$$

### Implementation

```cpp
class MambaTrainer {
    Mamba9D& model;
    double learning_rate = 0.001;

public:
    MambaTrainer(Mamba9D& m) : model(m) {}

    void train_step(const std::vector<TorusNode>& sequence) {
        // Forward pass
        auto predicted_state = model.forward(sequence);

        // Actual next state (ground truth)
        auto actual_state = node_to_vector(sequence.back());

        // Compute loss
        double loss = (predicted_state - actual_state).squaredNorm();

        // Backpropagation (simplified)
        auto gradient = 2.0 * (predicted_state - actual_state);

        // Update parameters (gradient descent)
        // (Actual implementation would update A, B, C matrices)

        std::cout << "[MAMBA TRAIN] Loss: " << loss << std::endl;
    }
};
```

## 15.3 Transformer Trainer

**Training Objective:** Minimize output waveform error

### Loss Function

$$\mathcal{L}_{\text{Trans}} = \| \Psi_{\text{output}} - \Psi_{\text{target}} \|^2$$

### Implementation

```cpp
class TransformerTrainer {
    WaveTransformerLayer& model;
    double learning_rate = 0.0001;

public:
    TransformerTrainer(WaveTransformerLayer& m) : model(m) {}

    void train_step(const std::vector<std::complex<double>>& input,
                     const std::vector<std::complex<double>>& target,
                     TorusManifold& torus) {
        // Forward pass
        auto output = model.forward(input, torus);

        // Compute loss
        double loss = 0.0;
        for (size_t i = 0; i < output.size(); ++i) {
            loss += std::norm(output[i] - target[i]);
        }

        // Backpropagation (simplified)
        // (Actual implementation would compute gradients and update weights)

        std::cout << "[TRANSFORMER TRAIN] Loss: " << loss << std::endl;

        // Trigger neuroplastic update if loss high
        if (loss > 1.0) {
            torus.trigger_neuroplasticity_update(output);
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
