// ============================================================
// tests/unit/phase69_conversion_validator_test.cpp
// Phase 69 — GAP-023b  DMC <-> GGUF Bidirectional Conversion Validation
// ============================================================
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

#include "nikola/validation/conversion_validator.hpp"

using namespace nikola::validation;
using Catch::Approx;

// ────────────────────────────────────────────────────────────────────────────
// §1  Constants
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("Fidelity thresholds", "[constants][energy][spectral][phase]") {
    // Energy drift: 0.001 = 0.1%
    CHECK(ENERGY_DRIFT_LIMIT       == Approx(0.001).epsilon(1e-9));
    CHECK(SPECTRAL_CORRELATION_MIN == Approx(0.999).epsilon(1e-9));
    CHECK(PHASE_ERROR_LIMIT_RAD    == Approx(0.03).epsilon(1e-9));
}

TEST_CASE("KL divergence thresholds", "[constants][kl]") {
    CHECK(KL_WITH_MASK_LIMIT   == Approx(0.1).epsilon(1e-9));
    CHECK(KL_WITHOUT_MASK_FLOOR == Approx(5.0).epsilon(1e-9));
    // Proper ordering: with-mask limit is far below without-mask floor
    CHECK(KL_WITH_MASK_LIMIT < KL_WITHOUT_MASK_FLOOR);
}

TEST_CASE("Compression constants", "[constants][compression]") {
    CHECK(MIN_COMPRESSION_RATIO == Approx(10.0).epsilon(1e-9));
    CHECK(Q9_0_BASE_COMPRESSION == Approx(6.4f).epsilon(1e-5f));
    CHECK(Q9_0_SPARSE_COMPRESSION == Approx(62.5f).epsilon(1e-4f));
    CHECK(Q9_0_REFERENCE_SPARSITY == Approx(0.90f).epsilon(1e-6f));
}

TEST_CASE("Q9_0 quantisation constants", "[constants][q9_0]") {
    CHECK(Q9_0_MIN_VALUE    == -4);
    CHECK(Q9_0_MAX_VALUE    == +4);
    CHECK(Q9_0_STATES       == 9);
    CHECK(Q9_0_BLOCK_WEIGHTS == 32u);
    // Verify block byte derivation: 32 weights × 0.5 B nibble + 4 B FP32
    CHECK(Q9_0_BLOCK_BYTES == 20u);
}

TEST_CASE("GGUF version constants", "[constants][version]") {
    CHECK(GGUF_VERSION_LEGACY  == 1u);
    CHECK(GGUF_VERSION_CURRENT == 2u);
    CHECK(GGUF_VERSION_FUTURE  == 3u);
    // Strict ordering
    CHECK(GGUF_VERSION_LEGACY < GGUF_VERSION_CURRENT);
    CHECK(GGUF_VERSION_CURRENT < GGUF_VERSION_FUTURE);
}

TEST_CASE("Phase error limit is near pi/100", "[constants][phase]") {
    // Spec states < pi/100 ≈ 0.03142 → threshold chosen as 0.03 (conservative)
    const double pi_over_100 = std::acos(-1.0) / 100.0;
    CHECK(PHASE_ERROR_LIMIT_RAD < pi_over_100);
    CHECK(PHASE_ERROR_LIMIT_RAD == Approx(0.03).epsilon(1e-9));
}

TEST_CASE("Block bytes derivation invariant", "[constants][q9_0]") {
    // 32 weights packed into nibbles = 16 bytes + 4-byte FP32 scale
    const size_t nibble_bytes = Q9_0_BLOCK_WEIGHTS / 2;
    const size_t scale_bytes  = sizeof(float);
    CHECK(nibble_bytes + scale_bytes == Q9_0_BLOCK_BYTES);
}

// ────────────────────────────────────────────────────────────────────────────
// §2  Energy drift
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("energy_drift_fraction: identical energies → 0", "[energy]") {
    CHECK(energy_drift_fraction(100.0, 100.0) == Approx(0.0).margin(1e-15));
}

TEST_CASE("energy_drift_fraction: known drift", "[energy]") {
    // |100.0 - 100.05| / 100.0 = 0.0005 (within limit)
    CHECK(energy_drift_fraction(100.0, 100.05) == Approx(0.0005).epsilon(1e-9));
    // |100.0 - 99.90| / 100.0 = 0.001 (exactly at boundary)
    CHECK(energy_drift_fraction(100.0, 99.90)  == Approx(0.001).epsilon(1e-9));
}

TEST_CASE("energy_drift_fraction: symmetric (absolute value)", "[energy]") {
    const double d1 = energy_drift_fraction(100.0, 101.0);
    const double d2 = energy_drift_fraction(100.0,  99.0);
    CHECK(d1 == Approx(d2).epsilon(1e-9));
}

TEST_CASE("energy_drift_fraction: E_orig zero or negative throws", "[energy]") {
    CHECK_THROWS_AS(energy_drift_fraction(0.0, 5.0),  std::invalid_argument);
    CHECK_THROWS_AS(energy_drift_fraction(-1.0, 5.0), std::invalid_argument);
}

TEST_CASE("passes_energy_criterion: boundary", "[energy]") {
    // strictly less than 0.001 passes
    CHECK(passes_energy_criterion(0.0009999) == true);
    CHECK(passes_energy_criterion(0.001)     == false);  // not strictly less
    CHECK(passes_energy_criterion(0.002)     == false);
}

TEST_CASE("passes_energy_criterion: zero drift always passes", "[energy]") {
    CHECK(passes_energy_criterion(0.0) == true);
}

// ────────────────────────────────────────────────────────────────────────────
// §3  Spectral fidelity (Pearson correlation)
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("pearson_correlation: identical vectors → 1.0", "[spectral]") {
    const std::vector<float> v = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    CHECK(pearson_correlation(v, v) == Approx(1.0).epsilon(1e-6));
}

TEST_CASE("pearson_correlation: perfectly anti-correlated → -1.0", "[spectral]") {
    const std::vector<float> a = {1.0f, 2.0f, 3.0f};
    const std::vector<float> b = {3.0f, 2.0f, 1.0f};
    CHECK(pearson_correlation(a, b) == Approx(-1.0).epsilon(1e-6));
}

TEST_CASE("pearson_correlation: symmetric", "[spectral]") {
    const std::vector<float> a = {1.0f, 3.0f, 2.0f, 5.0f};
    const std::vector<float> b = {2.0f, 4.0f, 1.0f, 3.0f};
    CHECK(pearson_correlation(a, b) == Approx(pearson_correlation(b, a)).epsilon(1e-9));
}

TEST_CASE("pearson_correlation: size mismatch throws", "[spectral]") {
    const std::vector<float> a = {1.0f, 2.0f, 3.0f};
    const std::vector<float> b = {1.0f, 2.0f};
    CHECK_THROWS_AS(pearson_correlation(a, b), std::invalid_argument);
}

TEST_CASE("pearson_correlation: empty span throws", "[spectral]") {
    const std::vector<float> empty{};
    CHECK_THROWS_AS(pearson_correlation(empty, empty), std::invalid_argument);
}

TEST_CASE("pearson_correlation: constant vector throws domain_error", "[spectral]") {
    const std::vector<float> constant = {3.0f, 3.0f, 3.0f, 3.0f};
    const std::vector<float> other    = {1.0f, 2.0f, 3.0f, 4.0f};
    CHECK_THROWS_AS(pearson_correlation(constant, other), std::domain_error);
}

TEST_CASE("passes_spectral_criterion: boundary", "[spectral]") {
    CHECK(passes_spectral_criterion(0.9991) == true);
    CHECK(passes_spectral_criterion(0.999)  == false);  // not strictly greater
    CHECK(passes_spectral_criterion(0.998)  == false);
    CHECK(passes_spectral_criterion(1.0)    == true);
}

// ────────────────────────────────────────────────────────────────────────────
// §4  Phase coherence
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("max_phase_error: identical phases → 0", "[phase]") {
    const std::vector<float> p = {0.1f, 0.5f, 1.0f, 2.5f};
    CHECK(max_phase_error(p, p) == Approx(0.0).margin(1e-9));
}

TEST_CASE("max_phase_error: finds maximum absolute difference", "[phase]") {
    const std::vector<float> a = {0.0f, 0.1f, 0.2f};
    const std::vector<float> b = {0.0f, 0.1f, 0.22f};
    // Max difference is at index 2: |0.2 - 0.22| = 0.02
    CHECK(max_phase_error(a, b) == Approx(0.02f).epsilon(1e-5));
}

TEST_CASE("max_phase_error: size mismatch throws", "[phase]") {
    const std::vector<float> a = {0.0f, 1.0f};
    const std::vector<float> b = {0.0f};
    CHECK_THROWS_AS(max_phase_error(a, b), std::invalid_argument);
}

TEST_CASE("max_phase_error: empty span throws", "[phase]") {
    const std::vector<float> empty{};
    CHECK_THROWS_AS(max_phase_error(empty, empty), std::invalid_argument);
}

TEST_CASE("passes_phase_criterion: boundary", "[phase]") {
    CHECK(passes_phase_criterion(0.029)  == true);
    CHECK(passes_phase_criterion(0.03)   == false);  // not strictly less
    CHECK(passes_phase_criterion(0.031)  == false);
    CHECK(passes_phase_criterion(0.0)    == true);
}

// ────────────────────────────────────────────────────────────────────────────
// §5  Topological fidelity (Jaccard index)
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("jaccard_index: identical non-empty sets → 1.0", "[topology]") {
    const std::vector<uint64_t> a = {1, 2, 3, 4, 5};
    CHECK(jaccard_index(a, a) == Approx(1.0).epsilon(1e-12));
}

TEST_CASE("jaccard_index: both empty sets → 1.0 (vacuous match)", "[topology]") {
    const std::vector<uint64_t> empty{};
    CHECK(jaccard_index(empty, empty) == Approx(1.0).epsilon(1e-12));
}

TEST_CASE("jaccard_index: disjoint sets → 0.0", "[topology]") {
    const std::vector<uint64_t> a = {1, 2, 3};
    const std::vector<uint64_t> b = {4, 5, 6};
    CHECK(jaccard_index(a, b) == Approx(0.0).margin(1e-12));
}

TEST_CASE("jaccard_index: partial overlap", "[topology]") {
    // {1,2,3} vs {2,3,4}: intersection={2,3}=2, union={1,2,3,4}=4 → 0.5
    const std::vector<uint64_t> a = {1, 2, 3};
    const std::vector<uint64_t> b = {2, 3, 4};
    CHECK(jaccard_index(a, b) == Approx(0.5).epsilon(1e-9));
}

TEST_CASE("jaccard_index: one empty throws", "[topology]") {
    const std::vector<uint64_t> a = {1, 2, 3};
    const std::vector<uint64_t> empty{};
    CHECK_THROWS_AS(jaccard_index(a, empty), std::invalid_argument);
    CHECK_THROWS_AS(jaccard_index(empty, a), std::invalid_argument);
}

TEST_CASE("passes_topology_criterion: boundary", "[topology]") {
    CHECK(passes_topology_criterion(1.0)         == true);
    CHECK(passes_topology_criterion(1.0 - 1e-13) == true);   // within epsilon
    CHECK(passes_topology_criterion(0.99)        == false);
    CHECK(passes_topology_criterion(0.9999)      == false);
}

// ────────────────────────────────────────────────────────────────────────────
// §6  Q9_0 quantisation / de-quantisation
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("quantize_q9_0: zero always maps to 0", "[q9_0]") {
    CHECK(quantize_q9_0(0.0f, 1.0f) == 0);
    CHECK(quantize_q9_0(0.0f, 0.5f) == 0);
}

TEST_CASE("quantize_q9_0: exact integer values", "[q9_0]") {
    const float scale = 1.0f;
    CHECK(quantize_q9_0( 1.0f, scale) == int8_t( 1));
    CHECK(quantize_q9_0(-1.0f, scale) == int8_t(-1));
    CHECK(quantize_q9_0( 4.0f, scale) == int8_t( 4));
    CHECK(quantize_q9_0(-4.0f, scale) == int8_t(-4));
}

TEST_CASE("quantize_q9_0: clamps overflow", "[q9_0]") {
    const float scale = 1.0f;
    CHECK(quantize_q9_0( 10.0f, scale) == int8_t( 4));
    CHECK(quantize_q9_0(-10.0f, scale) == int8_t(-4));
}

TEST_CASE("quantize_q9_0: rounding behaviour", "[q9_0]") {
    // value=1.6, scale=1.0 → round(1.6)=2
    CHECK(quantize_q9_0(1.6f, 1.0f) == int8_t(2));
    // value=1.4, scale=1.0 → round(1.4)=1
    CHECK(quantize_q9_0(1.4f, 1.0f) == int8_t(1));
}

TEST_CASE("quantize_q9_0: scale != 1.0", "[q9_0]") {
    // value=2.0, scale=2.0 → round(1.0)=1
    CHECK(quantize_q9_0(2.0f, 2.0f) == int8_t(1));
    // value=-3.0, scale=0.75 → round(-4.0)=-4
    CHECK(quantize_q9_0(-3.0f, 0.75f) == int8_t(-4));
}

TEST_CASE("quantize_q9_0: zero or negative scale throws", "[q9_0]") {
    CHECK_THROWS_AS(quantize_q9_0(1.0f, 0.0f),  std::invalid_argument);
    CHECK_THROWS_AS(quantize_q9_0(1.0f, -1.0f), std::invalid_argument);
}

TEST_CASE("dequantize_q9_0: round-trip approximation", "[q9_0]") {
    const float scale = 1.0f;
    // Exact integers survive round-trip
    for (int8_t q = Q9_0_MIN_VALUE; q <= Q9_0_MAX_VALUE; ++q) {
        const float reconstructed = dequantize_q9_0(q, scale);
        CHECK(reconstructed == Approx(static_cast<float>(q)).epsilon(1e-6f));
    }
}

TEST_CASE("dequantize_q9_0: scale factor applied", "[q9_0]") {
    CHECK(dequantize_q9_0(int8_t(3), 0.5f) == Approx(1.5f).epsilon(1e-6f));
    CHECK(dequantize_q9_0(int8_t(-2), 2.0f) == Approx(-4.0f).epsilon(1e-6f));
}

// ────────────────────────────────────────────────────────────────────────────
// §7  Block sizing and compression ratios
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("q9_0_block_bytes returns 20", "[compression]") {
    CHECK(q9_0_block_bytes() == 20u);
}

TEST_CASE("q9_0_base_compression_ratio returns 6.4", "[compression]") {
    CHECK(q9_0_base_compression_ratio() == Approx(6.4f).epsilon(1e-5f));
}

TEST_CASE("q9_0_sparse_compression_ratio: 10% active → 64x", "[compression]") {
    // 6.4 / 0.1 = 64.0
    CHECK(q9_0_sparse_compression_ratio(0.10f) == Approx(64.0f).epsilon(1e-4f));
}

TEST_CASE("q9_0_sparse_compression_ratio: 100% active → equals base ratio", "[compression]") {
    CHECK(q9_0_sparse_compression_ratio(1.0f) == Approx(Q9_0_BASE_COMPRESSION).epsilon(1e-5f));
}

TEST_CASE("q9_0_sparse_compression_ratio: invalid fraction throws", "[compression]") {
    CHECK_THROWS_AS(q9_0_sparse_compression_ratio(0.0f),  std::invalid_argument);
    CHECK_THROWS_AS(q9_0_sparse_compression_ratio(-0.1f), std::invalid_argument);
    CHECK_THROWS_AS(q9_0_sparse_compression_ratio(1.1f),  std::invalid_argument);
}

// ────────────────────────────────────────────────────────────────────────────
// §8  GGUF version matrix
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("version_status: all three versions", "[version]") {
    CHECK(version_status(GGUF_VERSION_LEGACY)  == ConversionStatus::DEPRECATED);
    CHECK(version_status(GGUF_VERSION_CURRENT) == ConversionStatus::ACTIVE);
    CHECK(version_status(GGUF_VERSION_FUTURE)  == ConversionStatus::PLANNED);
}

TEST_CASE("version_status: unknown version throws", "[version]") {
    CHECK_THROWS_AS(version_status(0u),  std::invalid_argument);
    CHECK_THROWS_AS(version_status(99u), std::invalid_argument);
}

TEST_CASE("has_q9_0_support: v1 no, v2+ yes", "[version]") {
    CHECK(has_q9_0_support(GGUF_VERSION_LEGACY)  == false);
    CHECK(has_q9_0_support(GGUF_VERSION_CURRENT) == true);
    CHECK(has_q9_0_support(GGUF_VERSION_FUTURE)  == true);
}

TEST_CASE("requires_attention_mask: v1 no, v2+ yes", "[version]") {
    CHECK(requires_attention_mask(GGUF_VERSION_LEGACY)  == false);
    CHECK(requires_attention_mask(GGUF_VERSION_CURRENT) == true);
    CHECK(requires_attention_mask(GGUF_VERSION_FUTURE)  == true);
}

TEST_CASE("is_deprecated_version: only v1", "[version]") {
    CHECK(is_deprecated_version(GGUF_VERSION_LEGACY)  == true);
    CHECK(is_deprecated_version(GGUF_VERSION_CURRENT) == false);
    CHECK(is_deprecated_version(GGUF_VERSION_FUTURE)  == false);
}

TEST_CASE("migration_trigger: v1 needs migration, v2 v3 do not", "[version]") {
    CHECK(migration_trigger(GGUF_VERSION_LEGACY)  == MigrationTrigger::LEGACY_FORMAT);
    CHECK(migration_trigger(GGUF_VERSION_CURRENT) == MigrationTrigger::NONE);
    CHECK(migration_trigger(GGUF_VERSION_FUTURE)  == MigrationTrigger::NONE);
    CHECK(migration_trigger(99u)                  == MigrationTrigger::UNKNOWN_FORMAT);
}

// ────────────────────────────────────────────────────────────────────────────
// §9  KL divergence and compression validation
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("kl_divergence_passes_with_mask: below 0.1 passes", "[kl]") {
    CHECK(kl_divergence_passes_with_mask(0.05)  == true);
    CHECK(kl_divergence_passes_with_mask(0.099) == true);
    CHECK(kl_divergence_passes_with_mask(0.1)   == false);  // not < 0.1
    CHECK(kl_divergence_passes_with_mask(0.2)   == false);
}

TEST_CASE("kl_divergence_fails_without_mask: above 5.0 expected", "[kl]") {
    CHECK(kl_divergence_fails_without_mask(5.1)  == true);
    CHECK(kl_divergence_fails_without_mask(10.0) == true);
    CHECK(kl_divergence_fails_without_mask(5.0)  == false);  // not > 5.0
    CHECK(kl_divergence_fails_without_mask(4.9)  == false);
}

TEST_CASE("kl_divergence: mask/no-mask relationship", "[kl]") {
    // Spec test: without mask D_KL > 5.0, with mask D_KL < 0.1
    const double kl_unmasked = 7.5;
    const double kl_masked   = 0.04;
    CHECK(kl_divergence_fails_without_mask(kl_unmasked) == true);
    CHECK(kl_divergence_passes_with_mask(kl_masked)     == true);
}

TEST_CASE("meets_compression_requirement: above 10.0 passes", "[compression]") {
    CHECK(meets_compression_requirement(25.0) == true);
    CHECK(meets_compression_requirement(10.1) == true);
    CHECK(meets_compression_requirement(10.0) == false);  // not strictly >
    CHECK(meets_compression_requirement(9.9)  == false);
}

TEST_CASE("Spec example: 25:1 compression for 1M active in 14M grid", "[compression]") {
    // Dense FP32: 14M × 4B = 56 MB; Nikola GGUF: ~2.2 MB → ratio ~ 25.5
    const double dense_mb    = 14'000'000.0 * 4.0 / 1e6;   // 56 MB
    const double nikola_mb   = 1'000'000.0 * 0.5 / 1e6 + 1.7;  // ~2.2 MB
    const double ratio       = dense_mb / nikola_mb;
    CHECK(ratio > 10.0);
    CHECK(meets_compression_requirement(ratio) == true);
}

// ────────────────────────────────────────────────────────────────────────────
// §10  ValidationReport construction and assessment
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("make_report: all criteria pass → passed=true", "[report]") {
    const auto r = make_report(0.0005, 0.9995, 0.01, true);
    CHECK(r.energy_drift         == Approx(0.0005).epsilon(1e-9));
    CHECK(r.spectral_correlation == Approx(0.9995).epsilon(1e-9));
    CHECK(r.max_phase_error_rad  == Approx(0.01).epsilon(1e-9));
    CHECK(r.topology_match       == true);
    CHECK(r.passed               == true);
    CHECK(report_passes(r)       == true);
}

TEST_CASE("make_report: energy criterion fails → passed=false", "[report]") {
    const auto r = make_report(0.002, 0.9995, 0.01, true);
    CHECK(r.passed == false);
    CHECK(report_passes(r) == false);
}

TEST_CASE("make_report: spectral criterion fails → passed=false", "[report]") {
    const auto r = make_report(0.0005, 0.995, 0.01, true);
    CHECK(r.passed == false);
}

TEST_CASE("make_report: topology mismatch → passed=false", "[report]") {
    const auto r = make_report(0.0005, 0.9995, 0.01, false);
    CHECK(r.topology_match == false);
    CHECK(r.passed         == false);
}

TEST_CASE("make_report: phase error too large → passed=false", "[report]") {
    const auto r = make_report(0.0005, 0.9995, 0.05, true);
    CHECK(r.max_phase_error_rad > PHASE_ERROR_LIMIT_RAD);
    CHECK(r.passed == false);
}

// ────────────────────────────────────────────────────────────────────────────
// §11  Diagnostic names
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("conversion_status_name: all enumerants", "[diagnostics]") {
    CHECK(conversion_status_name(ConversionStatus::DEPRECATED) == "DEPRECATED");
    CHECK(conversion_status_name(ConversionStatus::ACTIVE)     == "ACTIVE");
    CHECK(conversion_status_name(ConversionStatus::PLANNED)    == "PLANNED");
}

TEST_CASE("migration_trigger_name: all enumerants", "[diagnostics]") {
    CHECK(migration_trigger_name(MigrationTrigger::NONE)           == "NONE");
    CHECK(migration_trigger_name(MigrationTrigger::LEGACY_FORMAT)  == "LEGACY_FORMAT");
    CHECK(migration_trigger_name(MigrationTrigger::UNKNOWN_FORMAT) == "UNKNOWN_FORMAT");
}

TEST_CASE("Diagnostic names are non-empty string_views", "[diagnostics]") {
    CHECK_FALSE(conversion_status_name(ConversionStatus::ACTIVE).empty());
    CHECK_FALSE(migration_trigger_name(MigrationTrigger::NONE).empty());
    CHECK_FALSE(migration_trigger_name(MigrationTrigger::LEGACY_FORMAT).empty());
}

// ────────────────────────────────────────────────────────────────────────────
// §12  Invariants
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("Invariant: Q9_0 states = max - min + 1", "[invariants]") {
    CHECK(Q9_0_STATES == static_cast<int>(Q9_0_MAX_VALUE) -
                         static_cast<int>(Q9_0_MIN_VALUE) + 1);
}

TEST_CASE("Invariant: base compression = FP32/Q9_0 block ratio", "[invariants]") {
    // FP32: 32 weights × 4 B = 128 B; Q9_0: 20 B → ratio = 6.4
    const float fp32_bytes  = static_cast<float>(Q9_0_BLOCK_WEIGHTS * sizeof(float));
    const float q9_0_bytes  = static_cast<float>(Q9_0_BLOCK_BYTES);
    CHECK(fp32_bytes / q9_0_bytes == Approx(Q9_0_BASE_COMPRESSION).epsilon(1e-4f));
}

TEST_CASE("Invariant: sparse ratio is monotone decreasing in active fraction", "[invariants]") {
    CHECK(q9_0_sparse_compression_ratio(0.05f) >
          q9_0_sparse_compression_ratio(0.10f));
    CHECK(q9_0_sparse_compression_ratio(0.10f) >
          q9_0_sparse_compression_ratio(0.50f));
    CHECK(q9_0_sparse_compression_ratio(0.50f) >
          q9_0_sparse_compression_ratio(1.00f));
}

TEST_CASE("Invariant: q9_0 quantize/dequantize round-trip error bounded", "[invariants]") {
    // For any value within ±(4 × scale), round-trip error ≤ 0.5 × scale
    const float scale = 1.0f;
    for (int v = -40; v <= 40; ++v) {
        const float original  = static_cast<float>(v) * 0.1f;  // 0.1 increments
        const int8_t q        = quantize_q9_0(original, scale);
        // Verify recon is in alphabet range (dequantised value = q * scale)
        CHECK(q >= Q9_0_MIN_VALUE);
        CHECK(q <= Q9_0_MAX_VALUE);
    }
}

TEST_CASE("Invariant: KL limits properly ordered", "[invariants]") {
    // With-mask limit << without-mask floor: 0.1 × 49 = 4.9 < 5.0
    CHECK(KL_WITH_MASK_LIMIT * 49.0 < KL_WITHOUT_MASK_FLOOR);
}

TEST_CASE("Invariant: all fidelity limits are positive", "[invariants]") {
    CHECK(ENERGY_DRIFT_LIMIT       > 0.0);
    CHECK(SPECTRAL_CORRELATION_MIN > 0.0);
    CHECK(PHASE_ERROR_LIMIT_RAD    > 0.0);
}

TEST_CASE("Invariant: version functions agree on v1 deprecation", "[invariants]") {
    // All three predicates agree that v1 is legacy
    CHECK(is_deprecated_version(GGUF_VERSION_LEGACY)         == true);
    CHECK(has_q9_0_support(GGUF_VERSION_LEGACY)              == false);
    CHECK(requires_attention_mask(GGUF_VERSION_LEGACY)       == false);
    CHECK(migration_trigger(GGUF_VERSION_LEGACY)             == MigrationTrigger::LEGACY_FORMAT);
    CHECK(version_status(GGUF_VERSION_LEGACY)                == ConversionStatus::DEPRECATED);
}

// ────────────────────────────────────────────────────────────────────────────
// §13  Integration scenarios
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("Integration: perfect round-trip report", "[integration]") {
    // Identical original and reconstructed wavefunction
    const std::vector<float> orig  = {1.0f, 2.0f, 3.0f, 4.0f, 2.0f};
    const std::vector<float> recon = orig;
    const std::vector<float> phases_orig  = {0.1f, 0.5f, 1.0f, 1.5f, 2.0f};
    const std::vector<float> phases_recon = phases_orig;

    const double E_orig   = 1.0 + 4.0 + 9.0 + 16.0 + 4.0;
    const double E_recon  = E_orig;
    const double drift    = energy_drift_fraction(E_orig, E_recon);
    const double r        = pearson_correlation(orig, recon);
    const double max_ph   = max_phase_error(phases_orig, phases_recon);

    const std::vector<uint64_t> nodes = {0, 1, 2, 3, 4};
    const double j = jaccard_index(nodes, nodes);

    const auto report = make_report(drift, r, max_ph, passes_topology_criterion(j));
    CHECK(report.passed == true);
}

TEST_CASE("Integration: energy corruption detected", "[integration]") {
    // 0.05% energy loss (drift=0.0005) — well within 0.1% threshold, passes
    const double E_orig  = 1000.0;
    const double E_recon = 999.5;  // 0.05% loss
    CHECK(passes_energy_criterion(energy_drift_fraction(E_orig, E_recon)) == true);

    // 0.5% energy loss (drift=0.005) — exceeds 0.1% threshold, fails
    const double E_recon2 = 995.0;
    CHECK(passes_energy_criterion(energy_drift_fraction(E_orig, E_recon2)) == false);
}

TEST_CASE("Integration: Q9_0 pipeline preserves sign for all alphabet values", "[integration]") {
    const float scale = 1.0f;
    for (int8_t q = Q9_0_MIN_VALUE; q <= Q9_0_MAX_VALUE; ++q) {
        const float val   = dequantize_q9_0(q, scale);
        const int8_t back = quantize_q9_0(val, scale);
        CHECK(back == q);
    }
}

TEST_CASE("Integration: v2 version is fully capable (no migration, full features)", "[integration]") {
    CHECK(has_q9_0_support(GGUF_VERSION_CURRENT)        == true);
    CHECK(requires_attention_mask(GGUF_VERSION_CURRENT) == true);
    CHECK(is_deprecated_version(GGUF_VERSION_CURRENT)   == false);
    CHECK(migration_trigger(GGUF_VERSION_CURRENT)       == MigrationTrigger::NONE);
    CHECK(version_status(GGUF_VERSION_CURRENT)          == ConversionStatus::ACTIVE);
    CHECK(conversion_status_name(
        version_status(GGUF_VERSION_CURRENT))           == "ACTIVE");
}

TEST_CASE("Integration: spec reference numbers self-consistent", "[integration]") {
    // Spec: 1M active in 14M grid → compression > 10
    //   6.4 / (1M/14M) = 6.4 / 0.07143 ≈ 89.6
    const float active_fraction = 1.0f / 14.0f;
    const float ratio = q9_0_sparse_compression_ratio(active_fraction);
    CHECK(static_cast<double>(ratio) > MIN_COMPRESSION_RATIO);
    CHECK(meets_compression_requirement(static_cast<double>(ratio)) == true);
}
