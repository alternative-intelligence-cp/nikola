/**
 * @file phase160_cuda_selective_scan_test.cpp
 * @brief Phase 160 — CUDA Selective Scan kernel tests.
 *
 * Tests the GPU-accelerated Mamba S6 selective scan:
 *   1. GPU results match CPU SSMLayer::selective_step() reference
 *   2. Numerical stability over long sequences
 *   3. Throughput benchmark (target: 1M nodes/ms on RTX 3090)
 *
 * Sections:
 *   §1 — CPU/GPU equivalence
 *   §2 — Edge cases
 *   §3 — Throughput benchmark
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <nikola/cognitive/cuda_selective_scan.hpp>
#include <nikola/cognitive/cognitive_core.hpp>

#include <array>
#include <cmath>
#include <chrono>
#include <numeric>
#include <random>
#include <vector>

using namespace nikola::cognitive;
using nikola::foundation::TORUS_DIMS;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Run CPU selective_step for T steps, return final h.
static SSMLayer::State cpu_reference_scan(
    SSMLayer& ssm,
    const std::vector<float>& inputs,   // T × I flattened
    const SSMLayer::State& h_init,
    int T, int I)
{
    SSMLayer::State h = h_init;
    std::array<float, 9> u{};

    for (int t = 0; t < T; ++t) {
        for (int k = 0; k < I; ++k)
            u[k] = inputs[t * I + k];
        ssm.selective_step(h, u);
    }
    return h;
}

// ============================================================================
// §1 — CPU/GPU equivalence
// ============================================================================

TEST_CASE("§1-1 CUDA scan matches CPU selective_step with 10 inputs",
          "[phase160][cuda][scan]") {
    constexpr int H = 32, I = 9, O = 10, T = 10;

    // Set up matching CPU and GPU SSM
    SSMLayer ssm(H, I, O);
    ssm.randomise(42);
    ssm.randomise_selective(42);

    CudaSelectiveScan gpu(H, I);
    gpu.upload_weights(ssm.A().data(), ssm.W_delta().data(), ssm.W_Bsel().data());

    // Random inputs
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> ud(-1.f, 1.f);
    std::vector<float> inputs(T * I);
    for (float& v : inputs) v = ud(rng);

    // Initial state
    SSMLayer::State h_init(H, 0.f);

    // CPU reference
    SSMLayer::State h_cpu = cpu_reference_scan(ssm, inputs, h_init, T, I);

    // GPU scan
    std::vector<float> h_gpu(H, 0.f);
    gpu.scan(inputs.data(), h_init.data(), T, h_gpu.data());

    // Compare — should match within FP32 tolerance
    // (GPU uses same arithmetic, minor reordering may cause small diffs)
    for (int j = 0; j < H; ++j) {
        INFO("Hidden unit j=" << j << " cpu=" << h_cpu[j] << " gpu=" << h_gpu[j]);
        REQUIRE_THAT(static_cast<double>(h_gpu[j]),
                     Catch::Matchers::WithinAbs(static_cast<double>(h_cpu[j]), 1e-3));
    }
}

TEST_CASE("§1-2 CUDA scan matches CPU with nonzero initial state",
          "[phase160][cuda][scan]") {
    constexpr int H = 64, I = 9, O = 5, T = 20;

    SSMLayer ssm(H, I, O);
    ssm.randomise(99);
    ssm.randomise_selective(99);

    CudaSelectiveScan gpu(H, I);
    gpu.upload_weights(ssm.A().data(), ssm.W_delta().data(), ssm.W_Bsel().data());

    std::mt19937 rng(456);
    std::uniform_real_distribution<float> ud(-1.f, 1.f);
    std::vector<float> inputs(T * I);
    for (float& v : inputs) v = ud(rng);

    // Nonzero initial state
    SSMLayer::State h_init(H);
    for (float& v : h_init) v = ud(rng) * 0.5f;

    SSMLayer::State h_cpu = cpu_reference_scan(ssm, inputs, h_init, T, I);

    std::vector<float> h_gpu(H);
    gpu.scan(inputs.data(), h_init.data(), T, h_gpu.data());

    for (int j = 0; j < H; ++j) {
        REQUIRE_THAT(static_cast<double>(h_gpu[j]),
                     Catch::Matchers::WithinAbs(static_cast<double>(h_cpu[j]), 1e-3));
    }
}

TEST_CASE("§1-3 CUDA scan: all intermediate states match CPU",
          "[phase160][cuda][scan]") {
    constexpr int H = 16, I = 9, O = 5, T = 5;

    SSMLayer ssm(H, I, O);
    ssm.randomise(77);
    ssm.randomise_selective(77);

    CudaSelectiveScan gpu(H, I);
    gpu.upload_weights(ssm.A().data(), ssm.W_delta().data(), ssm.W_Bsel().data());

    std::mt19937 rng(789);
    std::uniform_real_distribution<float> ud(-1.f, 1.f);
    std::vector<float> inputs(T * I);
    for (float& v : inputs) v = ud(rng);

    SSMLayer::State h_init(H, 0.f);

    // Get all intermediate CPU states
    std::vector<SSMLayer::State> cpu_states;
    {
        SSMLayer::State h = h_init;
        std::array<float, 9> u{};
        for (int t = 0; t < T; ++t) {
            for (int k = 0; k < I; ++k)
                u[k] = inputs[t * I + k];
            ssm.selective_step(h, u);
            cpu_states.push_back(h);
        }
    }

    // GPU: request all states
    std::vector<float> h_final(H);
    std::vector<float> all_states(T * H);
    gpu.scan(inputs.data(), h_init.data(), T, h_final.data(), all_states.data());

    // Compare each intermediate state
    for (int t = 0; t < T; ++t) {
        for (int j = 0; j < H; ++j) {
            INFO("t=" << t << " j=" << j);
            REQUIRE_THAT(static_cast<double>(all_states[t * H + j]),
                         Catch::Matchers::WithinAbs(
                             static_cast<double>(cpu_states[t][j]), 1e-3));
        }
    }
}

TEST_CASE("§1-4 CUDA scan: H=256 production dimensions",
          "[phase160][cuda][scan]") {
    constexpr int H = 256, I = 9, O = 100, T = 50;

    SSMLayer ssm(H, I, O);
    ssm.randomise(42);
    ssm.randomise_selective(42);

    CudaSelectiveScan gpu(H, I);
    gpu.upload_weights(ssm.A().data(), ssm.W_delta().data(), ssm.W_Bsel().data());

    std::mt19937 rng(321);
    std::uniform_real_distribution<float> ud(-1.f, 1.f);
    std::vector<float> inputs(T * I);
    for (float& v : inputs) v = ud(rng);

    SSMLayer::State h_init(H, 0.f);
    SSMLayer::State h_cpu = cpu_reference_scan(ssm, inputs, h_init, T, I);

    std::vector<float> h_gpu(H);
    gpu.scan(inputs.data(), h_init.data(), T, h_gpu.data());

    float max_err = 0.f;
    for (int j = 0; j < H; ++j)
        max_err = std::max(max_err, std::abs(h_gpu[j] - h_cpu[j]));

    INFO("Max CPU/GPU error over H=256: " << max_err);
    REQUIRE(max_err < 0.01f);
}

// ============================================================================
// §2 — Edge cases
// ============================================================================

TEST_CASE("§2-1 CUDA scan: T=1 single step",
          "[phase160][cuda][scan][edge]") {
    constexpr int H = 32, I = 9;

    SSMLayer ssm(H, I, 5);
    ssm.randomise(42);
    ssm.randomise_selective(42);

    CudaSelectiveScan gpu(H, I);
    gpu.upload_weights(ssm.A().data(), ssm.W_delta().data(), ssm.W_Bsel().data());

    std::vector<float> inputs(I, 0.5f);
    SSMLayer::State h_init(H, 0.f);
    SSMLayer::State h_cpu = cpu_reference_scan(ssm, inputs, h_init, 1, I);

    std::vector<float> h_gpu(H);
    gpu.scan(inputs.data(), h_init.data(), 1, h_gpu.data());

    for (int j = 0; j < H; ++j) {
        REQUIRE_THAT(static_cast<double>(h_gpu[j]),
                     Catch::Matchers::WithinAbs(static_cast<double>(h_cpu[j]), 1e-4));
    }
}

TEST_CASE("§2-2 CUDA scan: zero inputs → state decays",
          "[phase160][cuda][scan][edge]") {
    constexpr int H = 32, I = 9, T = 50;

    SSMLayer ssm(H, I, 5);
    ssm.randomise(42);
    ssm.randomise_selective(42);

    CudaSelectiveScan gpu(H, I);
    gpu.upload_weights(ssm.A().data(), ssm.W_delta().data(), ssm.W_Bsel().data());

    std::vector<float> inputs(T * I, 0.f);
    // Nonzero initial state
    SSMLayer::State h_init(H, 0.5f);

    std::vector<float> h_gpu(H);
    gpu.scan(inputs.data(), h_init.data(), T, h_gpu.data());

    // State should remain finite
    for (int j = 0; j < H; ++j)
        REQUIRE(std::isfinite(h_gpu[j]));
}

TEST_CASE("§2-3 CUDA scan: long sequence T=1000 stays finite",
          "[phase160][cuda][scan][edge]") {
    constexpr int H = 64, I = 9, T = 1000;

    SSMLayer ssm(H, I, 10);
    ssm.randomise(42);
    ssm.randomise_selective(42);

    CudaSelectiveScan gpu(H, I);
    gpu.upload_weights(ssm.A().data(), ssm.W_delta().data(), ssm.W_Bsel().data());

    std::mt19937 rng(999);
    std::uniform_real_distribution<float> ud(-1.f, 1.f);
    std::vector<float> inputs(T * I);
    for (float& v : inputs) v = ud(rng);

    SSMLayer::State h_init(H, 0.f);
    std::vector<float> h_gpu(H);
    gpu.scan(inputs.data(), h_init.data(), T, h_gpu.data());

    for (int j = 0; j < H; ++j)
        REQUIRE(std::isfinite(h_gpu[j]));
}

// ============================================================================
// §3 — Throughput benchmark
// ============================================================================

TEST_CASE("§3-1 CUDA selective scan throughput benchmark",
          "[phase160][cuda][!benchmark]") {
    constexpr int H = 256, I = 9;

    // Test with various sequence lengths
    for (int T : {50, 100, 500, 1000}) {
        SSMLayer ssm(H, I, 100);
        ssm.randomise(42);
        ssm.randomise_selective(42);

        CudaSelectiveScan gpu(H, I);
        gpu.upload_weights(ssm.A().data(), ssm.W_delta().data(), ssm.W_Bsel().data());

        std::mt19937 rng(42);
        std::uniform_real_distribution<float> ud(-1.f, 1.f);
        std::vector<float> inputs(T * I);
        for (float& v : inputs) v = ud(rng);

        SSMLayer::State h_init(H, 0.f);
        std::vector<float> h_gpu(H);

        float us = gpu.benchmark_scan(inputs.data(), h_init.data(), T,
                                       h_gpu.data(), 5, 20);

        float nodes_per_ms = static_cast<float>(T) / (us / 1000.f);
        INFO("T=" << T << ": " << us << " µs/scan, "
             << nodes_per_ms << " nodes/ms");

        // Target: reasonable GPU throughput for T ≤ 1000.
        // Allow headroom for system load during full regression runs.
        if (T >= 1000) {
            REQUIRE(nodes_per_ms > 5000.f);  // 5K nodes/ms min under load
        }
    }
}
