#pragma once

#include <Eigen/Dense>

namespace nikola::cognitive {

// Forward declarations
struct Coord9D {
    int x, y, z;      // Spatial dimensions
    int u, v, w;      // Quantum dimensions
    int audio, visual, sync;
};

class TorusManifold; // Forward declaration

/**
 * @brief Attention Primer for predictive metric tensioning
 *
 * Uses Mamba's hidden state to predict where attention will focus next,
 * then pre-warps the toroidal geometry (increases refractive index) to
 * prime that location for faster Transformer processing.
 *
 * Mechanism: Mamba scans ahead -> predicts focus -> boosts metric at target
 * Result: When Transformer arrives, path is already "shorter" (primed)
 *
 * @status STUB - Implementation deferred to Phase 2
 */
class AttentionPrimer {
public:
    /**
     * @brief Predict next attention focus from Mamba state
     * @param mamba_hidden_state Mamba's current hidden state vector
     * @return 9D coordinates of predicted focus location
     */
    Coord9D predict_next_focus(const Eigen::VectorXd& mamba_hidden_state);

    /**
     * @brief Prime a location by warping local geometry
     * @param torus The 9D toroidal manifold
     * @param target Coordinates to prime
     * @param boost_factor Metric scaling (>1.0 = higher refractive index)
     */
    void prime_location(TorusManifold& torus, const Coord9D& target, double boost_factor = 1.1);

    /**
     * @brief Decay priming over time (restore original geometry)
     * @param torus The 9D toroidal manifold
     * @param decay_rate Exponential decay factor (0.0-1.0)
     */
    void decay_priming(TorusManifold& torus, double decay_rate = 0.95);

private:
    double priming_strength_ = 1.0;
};

} // namespace nikola::cognitive
