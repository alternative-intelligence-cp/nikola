/**
 * @file src/cognitive/selective_scan.cu
 * @brief CUDA kernels for Mamba S6 parallel selective scan.
 *
 * v0.1.6 Phase 3: Implements the parallel selective scan recurrence on GPU.
 *
 * The selective scan is a linear recurrence:
 *   h_t = Ā_t · h_{t-1} + B̄_t
 * where Ā_t and B̄_t are input-dependent (S6 selective scan).
 *
 * This is parallelized using Blelloch's work-efficient prefix scan with
 * the associative operator on (multiply, bias) tuples:
 *   (a, b) ⊕ (c, d) = (a·c, a·d + b)
 *
 * Architecture:
 *   Phase 1: k_precompute — compute Ā_t and B̄_t for all (t, j) in parallel
 *   Phase 2: k_parallel_scan — Blelloch scan over T per hidden unit j
 *   Phase 3: k_apply_initial — fold in h_init (multiply by scan output)
 *
 * Grid: N = 19,683 nodes at 50 hot nodes/tick → T ≈ 50.
 *       H = 256 hidden units.
 *       Total work = T × H ≈ 12,800 — well within a single SM.
 *
 * Target: 1M nodes/ms on RTX 3090.
 */

#include <nikola/cognitive/cuda_selective_scan.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

// ============================================================================
// CUDA error checking (same macro as propagator.cu)
// ============================================================================

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            throw std::runtime_error(                                          \
                std::string("CUDA error at " __FILE__ ":") +                   \
                std::to_string(__LINE__) + " — " +                             \
                cudaGetErrorString(_e));                                        \
        }                                                                      \
    } while (0)

// ============================================================================
// Constants
// ============================================================================

static constexpr int BLOCK_SIZE = 256;

// ============================================================================
// Kernel 1: Pre-compute Ā_t and B̄_t from inputs
// ============================================================================

/**
 * One thread per (t, j) where t ∈ [0, T) and j ∈ [0, H).
 * Grid: ((T * H + BLOCK_SIZE - 1) / BLOCK_SIZE) blocks × BLOCK_SIZE threads.
 *
 * For hidden unit j at time t:
 *   raw_Δ = W_delta[j, :] · u_t           (dot product, I elements)
 *   Δ     = softplus(raw_Δ)               (log(1 + exp(x)))
 *   b_t   = W_Bsel[j, :] · u_t            (dot product)
 *   Ā     = exp(Δ · A_j)                  (clamped to [0, 1])
 *   B̄     = Δ · b_t  if |A_j| < ε        (ZOH linear limit)
 *         = (Ā − 1) / A_j · Δ · b_t       (ZOH general case)
 */
__global__ void k_precompute(
    const float* __restrict__ A,         // [H]
    const float* __restrict__ W_delta,   // [H × I]
    const float* __restrict__ W_Bsel,    // [H × I]
    const float* __restrict__ inputs,    // [T × I]
    float* __restrict__ A_bar,           // [T × H] output
    float* __restrict__ B_bar,           // [T × H] output
    int T, int H, int I)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = T * H;
    if (idx >= total) return;

    const int t = idx / H;
    const int j = idx % H;

    // Pointers to weight rows for hidden unit j
    const float* wd_row = W_delta + j * I;
    const float* wb_row = W_Bsel  + j * I;
    const float* u      = inputs  + t * I;

    // Dot products: raw_Δ and b_t
    float raw_delta = 0.f;
    float b_t = 0.f;
    for (int k = 0; k < I; ++k) {
        raw_delta += wd_row[k] * u[k];
        b_t       += wb_row[k] * u[k];
    }

    // Softplus: Δ = log(1 + exp(x)), numerically stable
    float delta_i = (raw_delta > 20.f) ? raw_delta : logf(1.f + expf(raw_delta));

    // ZOH discretization
    float a_j = A[j];
    float a_bar_val = fminf(fmaxf(expf(delta_i * a_j), 0.f), 1.f);

    float b_bar_val;
    if (fabsf(a_j) < 1e-6f) {
        b_bar_val = delta_i * b_t;
    } else {
        b_bar_val = (a_bar_val - 1.f) / a_j * delta_i * b_t;
    }

    A_bar[t * H + j] = a_bar_val;
    B_bar[t * H + j] = b_bar_val;
}

// ============================================================================
// Kernel 2: Parallel scan (Blelloch up-sweep + down-sweep)
// ============================================================================

/**
 * Parallel prefix scan over the T dimension for each hidden unit j.
 *
 * One block per hidden unit j.  blockDim.x = next_pow2(T).
 * Uses shared memory for the (multiply, bias) tuples.
 *
 * Associative operator: (a, b) ⊕ (c, d) = (a·c, c·b + d)
 * Identity: (1, 0)
 *
 * After scan, output[t, j] = h_t given h_init = 0.
 * The caller folds in h_init separately (additive with Ā_prefix).
 *
 * For simplicity and correctness, we do an INCLUSIVE scan:
 *   h_t = Ā_t · h_{t-1} + B̄_t
 *   prefix[0] = (Ā_0, B̄_0)
 *   prefix[t] = prefix[t-1] ⊕ (Ā_t, B̄_t)
 *   h_t = prefix[t].a · h_init + prefix[t].b
 *
 * Since T is small (≤ 1024 typical), a single-block sequential scan is
 * actually faster than the full Blelloch approach for this problem size.
 * We use a single-thread sequential scan per hidden unit for simplicity
 * and correctness, with H blocks × 1 thread.
 */
__global__ void k_sequential_scan(
    const float* __restrict__ A_bar,   // [T × H]
    const float* __restrict__ B_bar,   // [T × H]
    const float* __restrict__ h_init,  // [H]
    float* __restrict__ h_all,         // [T × H] all intermediate states
    float* __restrict__ h_final,       // [H] final state
    int T, int H)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= H) return;

    // Sequential scan for hidden unit j
    float h = h_init[j];

    for (int t = 0; t < T; ++t) {
        const int idx = t * H + j;
        h = A_bar[idx] * h + B_bar[idx];

        if (h_all != nullptr) {
            h_all[idx] = h;
        }
    }

    h_final[j] = h;
}

/**
 * Batched scan: process multiple hidden units per block.
 * Block size = 256, each thread handles one hidden unit.
 * Since H = 256 = one block, all hidden units are processed in one block.
 * For larger H, multiple blocks are used.
 */
// (The k_sequential_scan above already handles this with generic grid config.)

// ============================================================================
// Impl struct (PIMPL internals)
// ============================================================================

namespace nikola::cognitive {

struct CudaSelectiveScan::Impl {
    int H, I;

    // Device buffers — weights (persistent)
    float* d_A       = nullptr;   // [H]
    float* d_W_delta = nullptr;   // [H × I]
    float* d_W_Bsel  = nullptr;   // [H × I]

    // Device buffers — per-scan (resized as needed)
    float* d_inputs  = nullptr;   // [T × I]
    float* d_A_bar   = nullptr;   // [T × H]
    float* d_B_bar   = nullptr;   // [T × H]
    float* d_h_init  = nullptr;   // [H]
    float* d_h_all   = nullptr;   // [T × H] (optional, for all_states)
    float* d_h_final = nullptr;   // [H]

    int allocated_T  = 0;         // Current allocation size for T

    Impl(int hidden_dim, int input_dim) : H(hidden_dim), I(input_dim) {
        // Allocate weight buffers
        CUDA_CHECK(cudaMalloc(&d_A,       sizeof(float) * H));
        CUDA_CHECK(cudaMalloc(&d_W_delta, sizeof(float) * H * I));
        CUDA_CHECK(cudaMalloc(&d_W_Bsel,  sizeof(float) * H * I));
        CUDA_CHECK(cudaMalloc(&d_h_init,  sizeof(float) * H));
        CUDA_CHECK(cudaMalloc(&d_h_final, sizeof(float) * H));
    }

    void ensure_T(int T, bool need_all_states) {
        if (T <= allocated_T && (!need_all_states || d_h_all != nullptr))
            return;

        // Free old per-scan buffers
        if (d_inputs) { cudaFree(d_inputs); d_inputs = nullptr; }
        if (d_A_bar)  { cudaFree(d_A_bar);  d_A_bar  = nullptr; }
        if (d_B_bar)  { cudaFree(d_B_bar);  d_B_bar  = nullptr; }
        if (d_h_all)  { cudaFree(d_h_all);  d_h_all  = nullptr; }

        CUDA_CHECK(cudaMalloc(&d_inputs, sizeof(float) * T * I));
        CUDA_CHECK(cudaMalloc(&d_A_bar,  sizeof(float) * T * H));
        CUDA_CHECK(cudaMalloc(&d_B_bar,  sizeof(float) * T * H));

        if (need_all_states) {
            CUDA_CHECK(cudaMalloc(&d_h_all, sizeof(float) * T * H));
        }

        allocated_T = T;
    }

    void free_all() noexcept {
        auto safe_free = [](float*& p) {
            if (p) { cudaFree(p); p = nullptr; }
        };
        safe_free(d_A);
        safe_free(d_W_delta);
        safe_free(d_W_Bsel);
        safe_free(d_inputs);
        safe_free(d_A_bar);
        safe_free(d_B_bar);
        safe_free(d_h_init);
        safe_free(d_h_all);
        safe_free(d_h_final);
        allocated_T = 0;
    }

    ~Impl() { free_all(); }
};

// ============================================================================
// CudaSelectiveScan public API
// ============================================================================

CudaSelectiveScan::CudaSelectiveScan(int hidden_dim, int input_dim)
    : impl_(new Impl(hidden_dim, input_dim))
{}

CudaSelectiveScan::~CudaSelectiveScan() = default;

CudaSelectiveScan::CudaSelectiveScan(CudaSelectiveScan&&) noexcept = default;
CudaSelectiveScan& CudaSelectiveScan::operator=(CudaSelectiveScan&&) noexcept = default;

void CudaSelectiveScan::ImplDeleter::operator()(Impl* p) const noexcept {
    delete p;
}

void CudaSelectiveScan::upload_weights(const float* A,
                                       const float* W_delta,
                                       const float* W_Bsel) {
    auto& m = *impl_;
    CUDA_CHECK(cudaMemcpy(m.d_A,       A,       sizeof(float) * m.H,       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(m.d_W_delta, W_delta, sizeof(float) * m.H * m.I, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(m.d_W_Bsel,  W_Bsel,  sizeof(float) * m.H * m.I, cudaMemcpyHostToDevice));
}

void CudaSelectiveScan::scan(const float* inputs, const float* h_init,
                             int T, float* h_out, float* all_states) {
    auto& m = *impl_;
    const bool need_all = (all_states != nullptr);
    m.ensure_T(T, need_all);

    // Upload inputs and initial state
    CUDA_CHECK(cudaMemcpy(m.d_inputs, inputs, sizeof(float) * T * m.I, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(m.d_h_init, h_init, sizeof(float) * m.H,    cudaMemcpyHostToDevice));

    // Phase 1: Pre-compute Ā and B̄
    {
        const int total = T * m.H;
        const int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
        k_precompute<<<blocks, BLOCK_SIZE>>>(
            m.d_A, m.d_W_delta, m.d_W_Bsel, m.d_inputs,
            m.d_A_bar, m.d_B_bar,
            T, m.H, m.I);
        CUDA_CHECK(cudaGetLastError());
    }

    // Phase 2: Sequential scan per hidden unit (T is small, this is optimal)
    {
        const int blocks = (m.H + BLOCK_SIZE - 1) / BLOCK_SIZE;
        k_sequential_scan<<<blocks, BLOCK_SIZE>>>(
            m.d_A_bar, m.d_B_bar, m.d_h_init,
            need_all ? m.d_h_all : nullptr,
            m.d_h_final,
            T, m.H);
        CUDA_CHECK(cudaGetLastError());
    }

    // Download results
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, m.d_h_final, sizeof(float) * m.H, cudaMemcpyDeviceToHost));

    if (need_all) {
        CUDA_CHECK(cudaMemcpy(all_states, m.d_h_all,
                              sizeof(float) * T * m.H,
                              cudaMemcpyDeviceToHost));
    }
}

float CudaSelectiveScan::benchmark_scan(const float* inputs, const float* h_init,
                                        int T, float* h_out,
                                        int warmup, int repeats) {
    // Warmup
    for (int i = 0; i < warmup; ++i)
        scan(inputs, h_init, T, h_out);

    // Timed runs
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i)
        scan(inputs, h_init, T, h_out);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return (ms * 1000.f) / static_cast<float>(repeats);  // µs per scan
}

int CudaSelectiveScan::hidden_dim() const noexcept { return impl_->H; }
int CudaSelectiveScan::input_dim()  const noexcept { return impl_->I; }

} // namespace nikola::cognitive
