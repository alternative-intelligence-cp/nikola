#pragma once
// ============================================================
// nikola/validation/conversion_validator.hpp
// GAP-023b  DMC <-> GGUF Bidirectional Conversion Validation
// Namespace: nikola::validation
// C++23 — header-only, stateless
// ============================================================
//
// Implements the round-trip fidelity standard for the DMC -> GGUF -> DMC
// conversion pipeline.  Three corruption sources must be validated:
//   1. Topological Decoherence  (Hilbert linearisation artefacts)
//   2. Spectral Distortion       (Q9_0 quantisation noise)
//   3. Vacuum Noise              (zero-padding sparse regions)
//
// Validation passes iff ALL four criteria are simultaneously satisfied:
//   energy_drift          < ENERGY_DRIFT_LIMIT        (0.001)
//   spectral_correlation  > SPECTRAL_CORRELATION_MIN  (0.999)
//   max_phase_error       < PHASE_ERROR_LIMIT_RAD     (0.03)
//   topology_match        == true                     (Jaccard = 1.0)
// ============================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace nikola::validation {

// ────────────────────────────────────────────────────────────────────────────
// §1  Fidelity thresholds (spec §GAP-023b "Round-Trip Fidelity Standard")
// ────────────────────────────────────────────────────────────────────────────

/// Maximum permitted relative energy drift: |E_orig − E_rt| / E_orig < 0.001
constexpr double ENERGY_DRIFT_LIMIT        = 0.001;

/// Minimum Pearson r between original and reconstructed amplitude vectors
constexpr double SPECTRAL_CORRELATION_MIN  = 0.999;

/// Maximum per-sample phase error (radians); spec: π/100 ≈ 0.0314 → 0.03
constexpr double PHASE_ERROR_LIMIT_RAD     = 0.03;

/// KL divergence upper bound when proper attention mask is present
constexpr double KL_WITH_MASK_LIMIT        = 0.1;

/// KL divergence lower bound expected WITHOUT attention mask (vacuum dilution)
constexpr double KL_WITHOUT_MASK_FLOOR     = 5.0;

/// Minimum compression ratio required by CI/CD validation test
constexpr double MIN_COMPRESSION_RATIO     = 10.0;

// ────────────────────────────────────────────────────────────────────────────
// §2  Q9_0 quantisation constants
// ────────────────────────────────────────────────────────────────────────────

/// Balanced nonary alphabet: {-4, -3, -2, -1, 0, +1, +2, +3, +4}
constexpr int8_t Q9_0_MIN_VALUE            = -4;
constexpr int8_t Q9_0_MAX_VALUE            = +4;
constexpr int    Q9_0_STATES               = 9;   ///< cardinality of alphabet

/// Weights per quantisation block
constexpr size_t Q9_0_BLOCK_WEIGHTS        = 32;

/// Bytes per block: 32 × 0.5 B (packed nibbles) + 4 B (FP32 scale) = 20 B
constexpr size_t Q9_0_BLOCK_BYTES          = 20;

/// Base compression ratio FP32 → Q9_0 (block overhead included):
///   FP32 = 32 weights × 4 B = 128 B; Q9_0 = 20 B  →  128/20 = 6.4
constexpr float  Q9_0_BASE_COMPRESSION     = 6.4f;

/// Reference sparsity fraction used in the spec example (90% vacant nodes)
constexpr float  Q9_0_REFERENCE_SPARSITY   = 0.90f;

/// Effective compression at reference sparsity: 6.4 / 0.10 = 64 ≈ 62.5 with
/// minor header overhead; spec quotes 62.5:1
constexpr float  Q9_0_SPARSE_COMPRESSION   = 62.5f;

// ────────────────────────────────────────────────────────────────────────────
// §3  GGUF version constants
// ────────────────────────────────────────────────────────────────────────────

constexpr uint32_t GGUF_VERSION_LEGACY     = 1u;  ///< v0.0.3 – deprecated
constexpr uint32_t GGUF_VERSION_CURRENT    = 2u;  ///< v0.0.4 – active
constexpr uint32_t GGUF_VERSION_FUTURE     = 3u;  ///< v0.0.5+ – planned

// ────────────────────────────────────────────────────────────────────────────
// §4  Enumeration types
// ────────────────────────────────────────────────────────────────────────────

/// Lifecycle status of a DMC/GGUF version pair
enum class ConversionStatus : uint8_t {
    DEPRECATED,   ///< v1 – no Q9_0, no attention mask; trigger migration
    ACTIVE,       ///< v2 – Q9_0 + attention mask; standard operations
    PLANNED       ///< v3 – future; forward compatibility only
};

/// Reason a round-trip migration must be triggered
enum class MigrationTrigger : uint8_t {
    NONE,           ///< current format, no migration needed
    LEGACY_FORMAT,  ///< GGUF v1 detected; rebuild with Q9_0 + mask
    UNKNOWN_FORMAT  ///< unrecognised version; conservative re-export
};

// ────────────────────────────────────────────────────────────────────────────
// §5  Validation report
// ────────────────────────────────────────────────────────────────────────────

/// Aggregated result of one DMC → GGUF → DMC round-trip validation
struct ValidationReport {
    double energy_drift;          ///< |E_orig - E_rt| / E_orig
    double spectral_correlation;  ///< Pearson r(A_orig, A_rt)
    double max_phase_error_rad;   ///< max_i |θ_orig[i] - θ_rt[i]|
    bool   topology_match;        ///< Jaccard(active_orig, active_rt) == 1.0
    bool   passed;                ///< all four criteria satisfied
};

// ────────────────────────────────────────────────────────────────────────────
// §6  Energy fidelity
// ────────────────────────────────────────────────────────────────────────────

/// Compute relative energy drift between original and round-tripped state.
///
/// @param E_orig   Total Hamiltonian (sum of squared amplitudes) of original
/// @param E_rt     Total Hamiltonian of reconstructed state
/// @return         |E_orig − E_rt| / E_orig
/// @throws std::invalid_argument if E_orig <= 0
[[nodiscard]] inline double energy_drift_fraction(double E_orig, double E_rt)
{
    if (E_orig <= 0.0)
        throw std::invalid_argument(
            "energy_drift_fraction: E_orig must be > 0");
    return std::abs(E_orig - E_rt) / E_orig;
}

/// Returns true when the energy drift satisfies the fidelity standard.
[[nodiscard]] inline bool passes_energy_criterion(double drift) noexcept
{
    return drift < ENERGY_DRIFT_LIMIT;
}

// ────────────────────────────────────────────────────────────────────────────
// §7  Spectral fidelity (Pearson correlation)
// ────────────────────────────────────────────────────────────────────────────

/// Compute Pearson correlation coefficient between two amplitude vectors.
///
/// @param x   Original amplitude samples
/// @param y   Reconstructed amplitude samples
/// @return    Pearson r ∈ [-1, 1];  1.0 when x == y element-wise
/// @throws std::invalid_argument on size mismatch or empty span
/// @throws std::domain_error if either vector is constant (zero std dev)
[[nodiscard]] inline double pearson_correlation(
    std::span<const float> x,
    std::span<const float> y)
{
    if (x.empty())
        throw std::invalid_argument("pearson_correlation: spans must not be empty");
    if (x.size() != y.size())
        throw std::invalid_argument("pearson_correlation: span size mismatch");

    const auto n = static_cast<double>(x.size());

    double sum_x  = 0.0, sum_y  = 0.0;
    double sum_x2 = 0.0, sum_y2 = 0.0;
    double sum_xy = 0.0;

    for (size_t i = 0; i < x.size(); ++i) {
        const double xi = static_cast<double>(x[i]);
        const double yi = static_cast<double>(y[i]);
        sum_x  += xi;
        sum_y  += yi;
        sum_x2 += xi * xi;
        sum_y2 += yi * yi;
        sum_xy += xi * yi;
    }

    const double num   = n * sum_xy - sum_x * sum_y;
    const double den_x = n * sum_x2 - sum_x * sum_x;
    const double den_y = n * sum_y2 - sum_y * sum_y;
    const double den   = std::sqrt(den_x * den_y);

    if (den < 1e-30)
        throw std::domain_error(
            "pearson_correlation: constant vector (zero standard deviation)");

    return num / den;
}

/// Returns true when the spectral correlation satisfies the fidelity standard.
[[nodiscard]] inline bool passes_spectral_criterion(double r) noexcept
{
    return r > SPECTRAL_CORRELATION_MIN;
}

// ────────────────────────────────────────────────────────────────────────────
// §8  Phase coherence
// ────────────────────────────────────────────────────────────────────────────

/// Compute the maximum absolute phase difference between two phase vectors.
///
/// @param theta_orig   Phase angles of original wavefunction (radians)
/// @param theta_rt     Phase angles of reconstructed wavefunction (radians)
/// @return             max_i |θ_orig[i] − θ_rt[i]|
/// @throws std::invalid_argument on size mismatch or empty span
[[nodiscard]] inline double max_phase_error(
    std::span<const float> theta_orig,
    std::span<const float> theta_rt)
{
    if (theta_orig.empty())
        throw std::invalid_argument("max_phase_error: spans must not be empty");
    if (theta_orig.size() != theta_rt.size())
        throw std::invalid_argument("max_phase_error: span size mismatch");

    double max_err = 0.0;
    for (size_t i = 0; i < theta_orig.size(); ++i) {
        const double diff = std::abs(
            static_cast<double>(theta_orig[i]) -
            static_cast<double>(theta_rt[i]));
        if (diff > max_err) max_err = diff;
    }
    return max_err;
}

/// Returns true when the maximum phase error satisfies the fidelity standard.
[[nodiscard]] inline bool passes_phase_criterion(double err) noexcept
{
    return err < PHASE_ERROR_LIMIT_RAD;
}

// ────────────────────────────────────────────────────────────────────────────
// §9  Topological fidelity (Jaccard index)
// ────────────────────────────────────────────────────────────────────────────

/// Compute Jaccard index of two active-node ID sets.
///
/// Spec requirement: both sets must be identical (Jaccard = 1.0) —
/// "active node sets and neighbour lists identical."
///
/// @param nodes_orig  Active node IDs from original manifold
/// @param nodes_rt    Active node IDs from round-tripped manifold
/// @return            |A ∩ B| / |A ∪ B|; returns 1.0 if both empty
/// @throws std::invalid_argument if exactly one set is empty
[[nodiscard]] inline double jaccard_index(
    std::span<const uint64_t> nodes_orig,
    std::span<const uint64_t> nodes_rt)
{
    const bool a_empty = nodes_orig.empty();
    const bool b_empty = nodes_rt.empty();

    if (a_empty && b_empty)  return 1.0;   // vacuously identical
    if (a_empty || b_empty)
        throw std::invalid_argument(
            "jaccard_index: one set is empty, the other is not — topological mismatch");

    const std::unordered_set<uint64_t> set_a(nodes_orig.begin(), nodes_orig.end());
    const std::unordered_set<uint64_t> set_b(nodes_rt.begin(),   nodes_rt.end());

    size_t intersection = 0;
    for (const auto& id : set_a)
        if (set_b.count(id)) ++intersection;

    const size_t union_size = set_a.size() + set_b.size() - intersection;
    return static_cast<double>(intersection) / static_cast<double>(union_size);
}

/// Returns true when the Jaccard index indicates perfect topological match.
[[nodiscard]] inline bool passes_topology_criterion(double j) noexcept
{
    // Jaccard = 1.0 requires exact set equality; use tight epsilon for
    // floating-point output of jaccard_index()
    return j >= (1.0 - 1e-12);
}

// ────────────────────────────────────────────────────────────────────────────
// §10  Q9_0 quantisation / de-quantisation
// ────────────────────────────────────────────────────────────────────────────

/// Quantise a float weight to the balanced nonary alphabet {-4…+4}.
///
/// @param value   Raw float weight
/// @param scale   Block scale factor (must be > 0)
/// @return        Nearest integer in {-4, …, +4} as int8_t
/// @throws std::invalid_argument if scale <= 0
[[nodiscard]] inline int8_t quantize_q9_0(float value, float scale)
{
    if (scale <= 0.0f)
        throw std::invalid_argument("quantize_q9_0: scale must be > 0");

    const int rounded = static_cast<int>(std::round(value / scale));
    const int clamped = std::clamp(rounded, static_cast<int>(Q9_0_MIN_VALUE),
                                            static_cast<int>(Q9_0_MAX_VALUE));
    return static_cast<int8_t>(clamped);
}

/// De-quantise a Q9_0 integer back to a float approximation.
[[nodiscard]] inline float dequantize_q9_0(int8_t q, float scale) noexcept
{
    return static_cast<float>(q) * scale;
}

/// Bytes per Q9_0 block: 32 packed nibbles (16 B) + 1 FP32 scale (4 B) = 20 B
[[nodiscard]] inline constexpr size_t q9_0_block_bytes() noexcept
{
    return Q9_0_BLOCK_BYTES;
}

/// Base compression ratio of Q9_0 vs FP32 (block overhead included): 6.4
[[nodiscard]] inline constexpr float q9_0_base_compression_ratio() noexcept
{
    return Q9_0_BASE_COMPRESSION;
}

/// Effective compression ratio for a given active-node fraction.
///
/// At 10% occupancy (90% sparsity): 6.4 / 0.1 = 64 ≈ 62.5 with header.
/// Spec quotes 62.5:1; this formula gives exact 64× at ideal 10% →
/// the spec approximation accounts for file header overhead.
///
/// @param active_fraction   Fraction of grid nodes that are active (0 < f ≤ 1)
/// @throws std::invalid_argument if fraction out of (0, 1]
[[nodiscard]] inline float q9_0_sparse_compression_ratio(float active_fraction)
{
    if (active_fraction <= 0.0f || active_fraction > 1.0f)
        throw std::invalid_argument(
            "q9_0_sparse_compression_ratio: active_fraction must be in (0, 1]");
    return Q9_0_BASE_COMPRESSION / active_fraction;
}

// ────────────────────────────────────────────────────────────────────────────
// §11  GGUF version matrix (spec §GAP-023b "Compatibility Matrix")
// ────────────────────────────────────────────────────────────────────────────

/// Return the lifecycle status of a GGUF major version.
/// @throws std::invalid_argument for unrecognised versions
[[nodiscard]] inline ConversionStatus version_status(uint32_t gguf_version)
{
    switch (gguf_version) {
        case GGUF_VERSION_LEGACY:  return ConversionStatus::DEPRECATED;
        case GGUF_VERSION_CURRENT: return ConversionStatus::ACTIVE;
        case GGUF_VERSION_FUTURE:  return ConversionStatus::PLANNED;
        default:
            throw std::invalid_argument(
                "version_status: unrecognised GGUF version");
    }
}

/// Returns true if the version supports Q9_0 quantisation (v2+).
[[nodiscard]] inline bool has_q9_0_support(uint32_t gguf_version) noexcept
{
    return gguf_version >= GGUF_VERSION_CURRENT;
}

/// Returns true if the version mandates an attention mask (v2+).
[[nodiscard]] inline bool requires_attention_mask(uint32_t gguf_version) noexcept
{
    return gguf_version >= GGUF_VERSION_CURRENT;
}

/// Returns true if the version is deprecated and must trigger migration.
[[nodiscard]] inline bool is_deprecated_version(uint32_t gguf_version) noexcept
{
    return gguf_version == GGUF_VERSION_LEGACY;
}

/// Determine what migration (if any) is required for a given GGUF version.
[[nodiscard]] inline MigrationTrigger migration_trigger(uint32_t gguf_version) noexcept
{
    if (gguf_version == GGUF_VERSION_LEGACY)  return MigrationTrigger::LEGACY_FORMAT;
    if (gguf_version == GGUF_VERSION_CURRENT ||
        gguf_version == GGUF_VERSION_FUTURE)  return MigrationTrigger::NONE;
    return MigrationTrigger::UNKNOWN_FORMAT;
}

// ────────────────────────────────────────────────────────────────────────────
// §12  KL divergence and compression validation
// ────────────────────────────────────────────────────────────────────────────

/// Returns true when the KL divergence is acceptable (correct attention mask
/// suppresses vacuum noise so distributions align within 0.1 nats).
[[nodiscard]] inline bool kl_divergence_passes_with_mask(double kl) noexcept
{
    return kl < KL_WITH_MASK_LIMIT;
}

/// Returns true when the KL divergence indicates the expected vacuum-noise
/// degradation in the absence of an attention mask (D_KL > 5.0).
[[nodiscard]] inline bool kl_divergence_fails_without_mask(double kl) noexcept
{
    return kl > KL_WITHOUT_MASK_FLOOR;
}

/// Returns true when the effective compression ratio meets the CI requirement.
[[nodiscard]] inline bool meets_compression_requirement(double ratio) noexcept
{
    return ratio > MIN_COMPRESSION_RATIO;
}

// ────────────────────────────────────────────────────────────────────────────
// §13  ValidationReport construction and assessment
// ────────────────────────────────────────────────────────────────────────────

/// Build a ValidationReport from the four measured metrics.
[[nodiscard]] inline ValidationReport make_report(
    double energy_drift,
    double spectral_correlation,
    double max_phase_err_rad,
    bool   topology_match) noexcept
{
    const bool ok = passes_energy_criterion(energy_drift)
                 && passes_spectral_criterion(spectral_correlation)
                 && passes_phase_criterion(max_phase_err_rad)
                 && topology_match;
    return { energy_drift, spectral_correlation, max_phase_err_rad,
             topology_match, ok };
}

/// Convenience predicate: true iff all four criteria in the report pass.
[[nodiscard]] inline bool report_passes(const ValidationReport& r) noexcept
{
    return r.passed;
}

// ────────────────────────────────────────────────────────────────────────────
// §14  Diagnostic name functions
// ────────────────────────────────────────────────────────────────────────────

[[nodiscard]] inline std::string_view
conversion_status_name(ConversionStatus s) noexcept
{
    switch (s) {
        case ConversionStatus::DEPRECATED: return "DEPRECATED";
        case ConversionStatus::ACTIVE:     return "ACTIVE";
        case ConversionStatus::PLANNED:    return "PLANNED";
        default:                           return "UNKNOWN";
    }
}

[[nodiscard]] inline std::string_view
migration_trigger_name(MigrationTrigger t) noexcept
{
    switch (t) {
        case MigrationTrigger::NONE:           return "NONE";
        case MigrationTrigger::LEGACY_FORMAT:  return "LEGACY_FORMAT";
        case MigrationTrigger::UNKNOWN_FORMAT: return "UNKNOWN_FORMAT";
        default:                               return "UNKNOWN";
    }
}

} // namespace nikola::validation
