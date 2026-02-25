// ============================================================
// nikola/infrastructure/k8s_hpa.hpp
//
// GAP-045: Kubernetes Horizontal Pod Autoscaling for
//          Biological Architectures
//
// Encodes the homeostatic scaling specification:
//   §1   ComponentType enum  (Physics Engine vs. Worker Pool)
//   §2   ScalingAction enum  (SCALE_DOWN / MAINTAIN / SCALE_UP)
//   §3   ATPRegime enum      (CRITICAL / LOW / NOMINAL / HIGH)
//   §4   ATP threshold constants
//   §5   HPA configuration constants
//   §6   Prometheus / scrape-interval constants
//   §7   Resilience & Pod Disruption Budget constants
//   §8   Prometheus metric name constants
//   §9   Kubernetes resource name constants
//   §10  ATP / sigmoid query functions
//   §11  Unified Load Metric compute functions
//   §12  HPA scaling decision helper
//   §13  PDB / scalability queries
//   §14  Label functions
// ============================================================

#pragma once

#include <cmath>
#include <cstdint>
#include <string_view>

namespace nikola::infrastructure {

// ============================================================
// §1  ComponentType Enum
// Distinguishes the singleton Physics Engine (stateful, mind)
// from the elastic Worker Pool (stateless, limbs).
// ============================================================

enum class ComponentType : uint8_t {
    PHYSICS_ENGINE = 0,   ///< Stateful 9D grid — singleton StatefulSet
    WORKER_POOL    = 1    ///< Elastic Deployment — HPA-managed
};

inline constexpr uint8_t COMPONENT_TYPE_COUNT = 2u;

// ============================================================
// §2  ScalingAction Enum
// ============================================================

enum class ScalingAction : uint8_t {
    SCALE_DOWN = 0,   ///< unified_load below half-target
    MAINTAIN   = 1,   ///< within acceptable band
    SCALE_UP   = 2    ///< unified_load above target
};

inline constexpr uint8_t SCALING_ACTION_COUNT = 3u;

// ============================================================
// §3  ATPRegime Enum
// ============================================================

enum class ATPRegime : uint8_t {
    CRITICAL = 0,   ///< ATP < 0.15 → forced Nap State
    LOW      = 1,   ///< 0.15 ≤ ATP < 0.20 → S_atp → 0
    NOMINAL  = 2,   ///< 0.20 ≤ ATP ≤ 0.50 → partial inhibition
    HIGH     = 3    ///< ATP > 0.50 → S_atp ≈ 1, linear scaling
};

inline constexpr uint8_t ATP_REGIME_COUNT = 4u;

// ============================================================
// §4  ATP Threshold Constants
// ============================================================

/// Hard stop: ENGS triggers forced Nap State below this level.
inline constexpr double ATP_HARD_STOP          = 0.15;

/// Sigmoid center for HPA inhibition; safety buffer above hard stop.
inline constexpr double ATP_SIGMOID_THRESHOLD  = 0.30;

/// ATP level above which S_atp ≈ 1 (linear scaling regime).
inline constexpr double ATP_HIGH_THRESHOLD     = 0.50;

/// ATP level below which S_atp → 0 (strong inhibition regime).
inline constexpr double ATP_LOW_THRESHOLD      = 0.20;

/// Steepness coefficient k for sigmoid S_atp calculation.
inline constexpr double SIGMOID_STEEPNESS_K    = 20.0;

// ============================================================
// §5  HPA Configuration Constants
// ============================================================

inline constexpr uint32_t HPA_MIN_REPLICAS          = 2u;
inline constexpr uint32_t HPA_MAX_REPLICAS          = 50u;

/// Target value for HPA controller: 0.5 s adjusted lag (500m).
inline constexpr double   HPA_TARGET_LAG_S          = 0.5;

/// Scale-up stabilisation window (seconds).
inline constexpr uint32_t SCALEUP_STABILIZATION_S   = 30u;

/// Scale-down stabilisation window (seconds); shortened vs. default
/// 300s to allow rapid worker shedding when ATP crashes.
inline constexpr uint32_t SCALEDOWN_STABILIZATION_S = 60u;

/// Scale-up policy: add up to 100% of current replicas per period.
inline constexpr uint32_t SCALEUP_PERCENT_PER_PERIOD   = 100u;
inline constexpr uint32_t SCALEUP_PERIOD_S             = 15u;

/// Scale-down policy: remove at most 5 pods per period.
inline constexpr uint32_t SCALEDOWN_PODS_PER_PERIOD = 5u;
inline constexpr uint32_t SCALEDOWN_PERIOD_S        = 30u;

// ============================================================
// §6  Prometheus / Scrape-Interval Constants
// ============================================================

/// Recommended scrape interval for ATP job — 1s given 1 kHz physics loop.
inline constexpr uint32_t SCRAPE_INTERVAL_S          = 1u;

/// PromQL window for queue-depth rate (responsive to sudden spikes).
inline constexpr uint32_t QUEUE_DEPTH_WINDOW_S       = 10u;

/// PromQL window for processing-rate average (smooths out jitter).
inline constexpr uint32_t PROCESSING_RATE_WINDOW_S   = 30u;

// ============================================================
// §7  Resilience / Pod Disruption Budget Constants
// ============================================================

/// Physics Engine is a singleton — must never be voluntarily evicted.
inline constexpr uint32_t PHYSICS_ENGINE_REPLICAS        = 1u;

/// PDB minAvailable for Physics Engine: 100% (never disrupt).
inline constexpr uint32_t PHYSICS_PDB_MIN_AVAILABLE_PCT  = 100u;

/// PDB minAvailable for Worker Pool: 50% (allows upgrades without
/// queue explosion).
inline constexpr uint32_t WORKER_PDB_MIN_AVAILABLE_PCT   = 50u;

// ============================================================
// §8  Prometheus Metric Name Constants
// ============================================================

/// Raw ZeroMQ receiver queue depth.
inline constexpr std::string_view METRIC_QUEUE_DEPTH    = "nikola_queue_depth";

/// System energy level ∈ [0.0, 1.0].
inline constexpr std::string_view METRIC_ATP_LEVEL      = "nikola_global_atp_level";

/// Monotonic counter of completed NeuralSpike tasks.
inline constexpr std::string_view METRIC_SPIKES_TOTAL   = "nikola_processed_spikes_total";

/// Derived composite: Q(t) / μ(t) — estimated drain time in seconds.
inline constexpr std::string_view METRIC_PROCESSING_LAG = "nikola_processing_lag";

/// Unified Load Metric after metabolic governor: Lag × S_atp.
inline constexpr std::string_view METRIC_LOAD_SCORE     = "nikola_metabolic_load_score";

inline constexpr uint8_t METRIC_COUNT = 5u;

// ============================================================
// §9  Kubernetes Resource Name Constants
// ============================================================

inline constexpr std::string_view K8S_HPA_NAME          = "nikola-worker-hpa";
inline constexpr std::string_view K8S_NAMESPACE         = "nikola-system";
inline constexpr std::string_view K8S_WORKER_DEPLOYMENT = "nikola-worker-pool";
inline constexpr std::string_view K8S_ORCHESTRATOR_SVC  = "nikola-orchestrator";

/// Stable hostname required by Physics Engine's ZeroMQ binding.
inline constexpr std::string_view K8S_PHYSICS_HOSTNAME  = "physics-0";

// ============================================================
// §10  ATP / Sigmoid Query Functions
// ============================================================

/// Classify current ATP level into the four-state regime.
[[nodiscard]] constexpr ATPRegime classify_atp_regime(double atp) noexcept {
    if (atp < ATP_HARD_STOP)       return ATPRegime::CRITICAL;
    if (atp < ATP_LOW_THRESHOLD)   return ATPRegime::LOW;
    if (atp <= ATP_HIGH_THRESHOLD) return ATPRegime::NOMINAL;
    return ATPRegime::HIGH;
}

/// Returns true when ENGS must trigger forced Nap State.
[[nodiscard]] constexpr bool is_atp_hard_stop(double atp) noexcept {
    return atp < ATP_HARD_STOP;
}

/// Returns true when sigmoid inhibition is non-negligible (ATP < 0.30).
[[nodiscard]] constexpr bool should_inhibit_scaling(double atp) noexcept {
    return atp < ATP_SIGMOID_THRESHOLD;
}

/// Returns true when system is in high-ATP linear scaling regime.
[[nodiscard]] constexpr bool is_atp_high_regime(double atp) noexcept {
    return atp > ATP_HIGH_THRESHOLD;
}

/// Returns true when system is in low-ATP strong-inhibition regime.
[[nodiscard]] constexpr bool is_atp_low_regime(double atp) noexcept {
    return atp < ATP_LOW_THRESHOLD;
}

/// ATP Scaling Factor S_atp = 1 / (1 + exp(-k * (ATP - threshold)))
/// Result approaches 1 for high ATP, 0 for low ATP.
[[nodiscard]] inline double sigmoid_atp_factor(double atp) noexcept {
    return 1.0 / (1.0 + std::exp(-SIGMOID_STEEPNESS_K * (atp - ATP_SIGMOID_THRESHOLD)));
}

// ============================================================
// §11  Unified Load Metric Compute
// ============================================================

/// Unified Load Metric: L_unified = Lag × S_atp
/// lag_s: estimated seconds to drain current backlog (nikola_processing_lag)
/// atp:   current normalized ATP ∈ [0.0, 1.0]
[[nodiscard]] inline double unified_load_metric(double lag_s, double atp) noexcept {
    return lag_s * sigmoid_atp_factor(atp);
}

// ============================================================
// §12  HPA Scaling Decision
// ============================================================

/// Derive HPA action from a unified load value.
/// Scale-up if load exceeds target; scale-down if below half-target.
[[nodiscard]] inline ScalingAction
scaling_decision(double unified_load,
                 double target_lag_s = HPA_TARGET_LAG_S) noexcept {
    if (unified_load > target_lag_s)         return ScalingAction::SCALE_UP;
    if (unified_load < target_lag_s * 0.5)   return ScalingAction::SCALE_DOWN;
    return ScalingAction::MAINTAIN;
}

// ============================================================
// §13  PDB / Scalability Queries
// ============================================================

/// Returns the minAvailable percentage for the given component's PDB.
[[nodiscard]] constexpr uint32_t
pdb_min_available_pct(ComponentType c) noexcept {
    return c == ComponentType::PHYSICS_ENGINE
        ? PHYSICS_PDB_MIN_AVAILABLE_PCT
        : WORKER_PDB_MIN_AVAILABLE_PCT;
}

/// Returns true only for components that can be horizontally scaled.
[[nodiscard]] constexpr bool is_horizontally_scalable(ComponentType c) noexcept {
    return c == ComponentType::WORKER_POOL;
}

/// Returns true if the component requires a stable StatefulSet hostname.
[[nodiscard]] constexpr bool requires_stable_identity(ComponentType c) noexcept {
    return c == ComponentType::PHYSICS_ENGINE;
}

// ============================================================
// §14  Label Functions
// ============================================================

[[nodiscard]] constexpr std::string_view
component_type_name(ComponentType c) noexcept {
    switch (c) {
        case ComponentType::PHYSICS_ENGINE: return "physics_engine";
        case ComponentType::WORKER_POOL:    return "worker_pool";
    }
    return "unknown";
}

[[nodiscard]] constexpr std::string_view
scaling_action_name(ScalingAction a) noexcept {
    switch (a) {
        case ScalingAction::SCALE_DOWN: return "scale_down";
        case ScalingAction::MAINTAIN:   return "maintain";
        case ScalingAction::SCALE_UP:   return "scale_up";
    }
    return "unknown";
}

[[nodiscard]] constexpr std::string_view
atp_regime_name(ATPRegime r) noexcept {
    switch (r) {
        case ATPRegime::CRITICAL: return "critical";
        case ATPRegime::LOW:      return "low";
        case ATPRegime::NOMINAL:  return "nominal";
        case ATPRegime::HIGH:     return "high";
    }
    return "unknown";
}

} // namespace nikola::infrastructure
