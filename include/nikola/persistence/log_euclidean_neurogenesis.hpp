#pragma once
/**
 * @file log_euclidean_neurogenesis.hpp
 * @brief v0.3.4 — GAP-M6 Log-Euclidean metric interpolation for neurogenesis.
 *
 * Implements SPD-safe interpolation on the manifold of symmetric positive
 * definite (SPD) tensors:
 *
 *   g(alpha) = exp((1-alpha) * log(g_a) + alpha * log(g_b)), alpha in [0,1]
 *
 * This produces geometry-aware newborn node metrics during topology growth.
 */

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

namespace nikola::persistence {

using NeuroMetricTensor = Eigen::Matrix<double, 9, 9>;

inline constexpr double LE_INTERP_EIGEN_FLOOR = 1.0e-8;
inline constexpr double LE_INTERP_SYM_EPS     = 1.0e-10;

[[nodiscard]] inline bool is_spd_metric(const NeuroMetricTensor& g) noexcept {
    const NeuroMetricTensor sym = (g + g.transpose()) * 0.5;
    if ((sym - sym.transpose()).cwiseAbs().maxCoeff() > LE_INTERP_SYM_EPS) return false;
    const Eigen::LLT<NeuroMetricTensor> llt(sym);
    return llt.info() == Eigen::Success;
}

[[nodiscard]] inline NeuroMetricTensor project_metric_to_spd(
    const NeuroMetricTensor& g,
    double eigen_floor = LE_INTERP_EIGEN_FLOOR)
{
    if (!(eigen_floor > 0.0) || !std::isfinite(eigen_floor)) {
        throw std::invalid_argument("project_metric_to_spd: eigen_floor must be finite and > 0");
    }

    const NeuroMetricTensor sym = (g + g.transpose()) * 0.5;
    const Eigen::SelfAdjointEigenSolver<NeuroMetricTensor> es(sym);
    if (es.info() != Eigen::Success) {
        throw std::runtime_error("project_metric_to_spd: eigendecomposition failed");
    }

    const auto evals = es.eigenvalues().cwiseMax(eigen_floor);
    return es.eigenvectors() * evals.asDiagonal() * es.eigenvectors().transpose();
}

[[nodiscard]] inline NeuroMetricTensor matrix_log_spd(const NeuroMetricTensor& g) {
    const NeuroMetricTensor spd = project_metric_to_spd(g);
    const Eigen::SelfAdjointEigenSolver<NeuroMetricTensor> es(spd);
    if (es.info() != Eigen::Success) {
        throw std::runtime_error("matrix_log_spd: eigendecomposition failed");
    }

    auto evals = es.eigenvalues();
    for (int i = 0; i < evals.size(); ++i) {
        if (!(evals[i] > 0.0) || !std::isfinite(evals[i])) {
            throw std::domain_error("matrix_log_spd: non-positive eigenvalue");
        }
        evals[i] = std::log(evals[i]);
    }

    return es.eigenvectors() * evals.asDiagonal() * es.eigenvectors().transpose();
}

[[nodiscard]] inline NeuroMetricTensor matrix_exp_sym(const NeuroMetricTensor& a) {
    const NeuroMetricTensor sym = (a + a.transpose()) * 0.5;
    const Eigen::SelfAdjointEigenSolver<NeuroMetricTensor> es(sym);
    if (es.info() != Eigen::Success) {
        throw std::runtime_error("matrix_exp_sym: eigendecomposition failed");
    }

    auto evals = es.eigenvalues();
    for (int i = 0; i < evals.size(); ++i) evals[i] = std::exp(evals[i]);

    return es.eigenvectors() * evals.asDiagonal() * es.eigenvectors().transpose();
}

[[nodiscard]] inline NeuroMetricTensor log_euclidean_interpolate(
    const NeuroMetricTensor& parent_a,
    const NeuroMetricTensor& parent_b,
    double alpha)
{
    if (!std::isfinite(alpha) || alpha < 0.0 || alpha > 1.0) {
        throw std::invalid_argument("log_euclidean_interpolate: alpha must be finite and in [0, 1]");
    }

    const NeuroMetricTensor log_a = matrix_log_spd(parent_a);
    const NeuroMetricTensor log_b = matrix_log_spd(parent_b);

    const NeuroMetricTensor blend = (1.0 - alpha) * log_a + alpha * log_b;
    return project_metric_to_spd(matrix_exp_sym(blend));
}

[[nodiscard]] inline double log_euclidean_distance(
    const NeuroMetricTensor& a,
    const NeuroMetricTensor& b)
{
    const NeuroMetricTensor delta = matrix_log_spd(a) - matrix_log_spd(b);
    return delta.norm();
}

} // namespace nikola::persistence
