/**
 * @file wave_function.hpp
 * @brief Quantum-state wrapper for the 9D toroidal manifold.
 *
 * WaveFunction owns and manages a TorusGrid and provides physics-level
 * factory methods (pilot wave seeding, thermal bath) together with common
 * observables (total probability, total energy density, norm).
 *
 * The underlying TorusGrid stores the primary fields:
 *   - Ψ   (psi)  — wavefunction, complex FP32
 *   - ∂Ψ/∂t (vel) — velocity field (canonical momentum), complex FP32
 *   - r         — resonance / damping field, real FP32 (0=erase, 1=freeze)
 *   - s         — state / refractive-index field, real FP32
 *
 * Reference: nikola engineering report, Phase 1, IMP-03 (Manifold Seeder),
 *            Gap 1.2 (thermal bath), Gap 1.1 (harmonic spatial injection).
 */
#pragma once

#include <nikola/foundation/toroidal_grid.hpp>
#include <nikola/foundation/complex_field.hpp>

#include <memory>
#include <cmath>
#include <cstring>
#include <random>
#include <cassert>
#include <stdexcept>

namespace nikola::physics {

using foundation::Complex;
using foundation::TorusGrid;
using foundation::GridConfig;
using foundation::TorusNode;
using foundation::TORUS_DIMS;
using foundation::VACUUM_NODE;

// ============================================================================
// Physical constants  (natural units: c₀ = 1, ħ = 1, β = 1)
// ============================================================================

inline constexpr float DEFAULT_C0   = 1.f;    ///< Wave speed in vacuum
inline constexpr float DEFAULT_BETA = 1.f;    ///< Nonlinearity coefficient
inline constexpr float DEFAULT_ALPHA= 0.01f;  ///< Damping (resonance coupling)

// ============================================================================
// WaveFunction
// ============================================================================

/**
 * @brief 9D wavefunction on a sparse toroidal manifold.
 *
 * Owns the grid; physics operations (propagation) are delegated to Propagator.
 */
class WaveFunction {
public:
    // ------------------------------------------------------------------ construction

    /**
     * @brief Construct with a given grid configuration.
     *
     * Creates an empty grid (no nodes allocated).  Call seed_*() or
     * add_node() to populate before propagating.
     */
    explicit WaveFunction(GridConfig config = GridConfig::uniform(3))
        : grid_(std::make_unique<TorusGrid>(std::move(config)))
        , time_(0.f)
    {}

    // Non-copyable (grid contains significant heap data; use clone() explicitly).
    WaveFunction(const WaveFunction&) = delete;
    WaveFunction& operator=(const WaveFunction&) = delete;
    WaveFunction(WaveFunction&&) = default;
    WaveFunction& operator=(WaveFunction&&) = default;

    // ------------------------------------------------------------------ factory: manifold seeder

    /**
     * @brief Seed the manifold for a dense hypercubic grid (IMP-03).
     *
     * Allocates n^9 nodes and initialises them according to the Manifold Seeder
     * specification:
     *   1. Pilot wave in dimension @p pilot_dim with wavenumber @p k_mode.
     *   2. Velocity field: thermal bath (Gap 1.2).
     *   3. Resonance r = 0.5 (neutral plasticity).
     *   4. State field s = 0.0 (vacuum refractive index).
     *
     * @param n           Nodes per dimension (side length).
     * @param pilot_dim   Dimension that carries the pilot wave (default: 3 = time).
     * @param k_mode      Wavenumber mode (integer → satisfies periodic BC).
     * @param amplitude   Pilot wave amplitude A₀ (default 1.0 activates nonlinearity).
     * @param seed        RNG seed for reproducible thermal noise.
     */
    void seed_manifold(int n, int pilot_dim = 3, int k_mode = 1,
                       float amplitude = 1.f, uint32_t seed = 42)
    {
        grid_ = std::make_unique<TorusGrid>(GridConfig::uniform(n));
        time_ = 0.f;

        std::mt19937 rng(seed);

        // Allocate all nodes with default state, then overwrite fields.
        std::array<int, TORUS_DIMS> c{};
        seed_recursive(c, 0, n, 0, pilot_dim, k_mode, amplitude, rng);
    }

    /**
     * @brief Seed with an existing (already-allocated) grid.
     *
     * Takes ownership.
     */
    void set_grid(std::unique_ptr<TorusGrid> grid) {
        assert(grid);
        grid_ = std::move(grid);
        time_ = 0.f;
    }

    // ------------------------------------------------------------------ observables

    /**
     * @brief Total probability  P = Σᵢ |Ψᵢ|²
     *
     * This is the discrete approximation of ∫|Ψ|² dV.
     */
    [[nodiscard]]
    double total_probability() const noexcept {
        const size_t N = grid_->num_active_nodes();
        const float* pr = grid_->psi_real();
        const float* pi = grid_->psi_imag();
        double sum = 0.0;
        for (size_t i = 0; i < N; ++i) {
            sum += static_cast<double>(pr[i]*pr[i] + pi[i]*pi[i]);
        }
        return sum;
    }

    /**
     * @brief Total kinetic energy  K = Σᵢ |Vᵢ|²
     */
    [[nodiscard]]
    double total_kinetic_energy() const noexcept {
        const size_t N = grid_->num_active_nodes();
        const float* vr = grid_->vel_real();
        const float* vi = grid_->vel_imag();
        double sum = 0.0;
        for (size_t i = 0; i < N; ++i) {
            sum += static_cast<double>(vr[i]*vr[i] + vi[i]*vi[i]);
        }
        return sum;
    }

    /**
     * @brief Maximum wavefunction amplitude max_i |Ψᵢ|.
     */
    [[nodiscard]]
    float max_amplitude() const noexcept {
        const size_t N = grid_->num_active_nodes();
        const float* pr = grid_->psi_real();
        const float* pi = grid_->psi_imag();
        float mx = 0.f;
        for (size_t i = 0; i < N; ++i) {
            const float a = std::sqrt(pr[i]*pr[i] + pi[i]*pi[i]);
            if (a > mx) mx = a;
        }
        return mx;
    }

    /**
     * @brief Check that all fields are numerically finite.
     * @return true if every node has finite Ψ and V.
     */
    [[nodiscard]]
    bool is_finite() const noexcept {
        const size_t N = grid_->num_active_nodes();
        const float* pr = grid_->psi_real();
        const float* pi = grid_->psi_imag();
        const float* vr = grid_->vel_real();
        const float* vi = grid_->vel_imag();
        for (size_t i = 0; i < N; ++i) {
            if (!std::isfinite(pr[i]) || !std::isfinite(pi[i]) ||
                !std::isfinite(vr[i]) || !std::isfinite(vi[i]))
                return false;
        }
        return true;
    }

    // ------------------------------------------------------------------ inject emitter  (Gap 1.1)

    /**
     * @brief Inject a standing wave at a specific node (harmonic spatial injection).
     *
     * The injected amplitude is safety-clamped to prevent nonlinear explosion.
     * Physically: atomicAdd(psi_real, safe_amp) — linear superposition.
     *
     * @param node_idx  Target node (must be allocated).
     * @param amplitude Complex injection amplitude.
     * @param max_amp   Safety clamp maximum (sqrt(MAX_ENERGY / beta)).
     */
    void inject(size_t node_idx, Complex amplitude,
                float max_amp = std::sqrt(0.1f / DEFAULT_BETA))
    {
        assert(node_idx < grid_->num_active_nodes());
        auto node = grid_->get_node(node_idx);
        node.psi += foundation::clamp_amplitude(amplitude, max_amp);
        grid_->set_node(node_idx, node);
    }

    // ------------------------------------------------------------------ grid access

    /// Read-access to the underlying grid.
    const TorusGrid& grid() const noexcept { return *grid_; }

    /// Write-access to the underlying grid (for propagator).
    TorusGrid& grid() noexcept { return *grid_; }

    /// Current simulation time.
    float time() const noexcept { return time_; }

    /// Advance simulation time (called by Propagator).
    void advance_time(float dt) noexcept { time_ += dt; }

    /// Number of active nodes.
    size_t num_nodes() const noexcept { return grid_->num_active_nodes(); }

    // ------------------------------------------------------------------ Quantum Zeno SCRAM  (Gap 1.5)

    /**
     * @brief Emergency damping: multiply all velocities by factor γ.
     *
     * Called by the energy watchdog when |ΔH/H| > tolerance.
     * Drains excess energy without zeroing the wavefunction.
     *
     * @param gamma  Damping factor (0.5 recommended for soft SCRAM).
     */
    void apply_emergency_damping(float gamma = 0.5f) noexcept {
        const size_t N = grid_->num_active_nodes();
        float* vr = grid_->vel_real();
        float* vi = grid_->vel_imag();
        for (size_t i = 0; i < N; ++i) {
            vr[i] *= gamma;
            vi[i] *= gamma;
        }
    }

    /**
     * @brief Add a scaled copy of another WaveFunction's ψ-field into this one.
     *
     * Computes:  this->psi[i] += alpha * other.psi[i]  for all i.
     * The two grids must have the same number of active nodes; if they differ
     * this is a silent no-op (grids of different sizes represent distinct
     * physical systems and mixing them is undefined).
     * Used by NPT and RECALL_MEMORY to blend candidate fields into the live torus.
     *
     * @param other  Source WaveFunction (read-only).
     * @param alpha  Blend weight (e.g. top_head_score × 0.3).
     */
    void add_scaled(const WaveFunction& other, float alpha) noexcept
    {
        const size_t N = grid_->num_active_nodes();
        const size_t M = other.grid_->num_active_nodes();
        if (N != M) return;  // size mismatch → silent no-op
        float*       pr  = grid_->psi_real();
        float*       pi  = grid_->psi_imag();
        const float* opr = other.grid_->psi_real();
        const float* opi = other.grid_->psi_imag();
        for (size_t i = 0; i < N; ++i) {
            pr[i] += alpha * opr[i];
            pi[i] += alpha * opi[i];
        }
    }

    /**
     * @brief Real part of the ψ-field inner product ⟨this|other⟩.
     *
     * Computes Σᵢ (psi_real[i] · other.psi_real[i] + psi_imag[i] · other.psi_imag[i]).
     * This is the wave-correlation kernel used by NPT attention heads.
     *
     * Returns 0.0 immediately when the two grids have different node counts:
     * grids of different sizes represent physically distinct systems and
     * their overlap is undefined.
     *
     * @param other  Key WaveFunction.
     * @return Σᵢ Re(Ψᵢ* · Kᵢ), or 0.0 when num_nodes() != other.num_nodes().
     */
    [[nodiscard]]
    double inner_product_re(const WaveFunction& other) const noexcept
    {
        const size_t N = grid_->num_active_nodes();
        const size_t M = other.grid_->num_active_nodes();
        if (N != M) return 0.0;  // size mismatch → undefined overlap
        const float* pr  = grid_->psi_real();
        const float* pi  = grid_->psi_imag();
        const float* opr = other.grid_->psi_real();
        const float* opi = other.grid_->psi_imag();
        double sum = 0.0;
        for (size_t i = 0; i < N; ++i)
            sum += static_cast<double>(pr[i] * opr[i] + pi[i] * opi[i]);
        return sum;
    }

    /**
     * @brief Scale the ψ-field by a real scalar in-place.
     *
     * Computes: psi[i] *= alpha  for all i.
     * Used by the NPT heterodyne aggregation step (output = Σᵢ score[i] · V[i]).
     *
     * @param alpha  Scaling factor.
     */
    void scale_by(float alpha) noexcept
    {
        const size_t N = grid_->num_active_nodes();
        float* pr = grid_->psi_real();
        float* pi = grid_->psi_imag();
        for (size_t i = 0; i < N; ++i) {
            pr[i] *= alpha;
            pi[i] *= alpha;
        }
    }

    /**
     * @brief Rotate every ψᵢ by a global phase θ in-place.
     *
     * Computes: ψᵢ ← ψᵢ · e^{i·θ} = (ψ_r cosθ − ψ_i sinθ) + i(ψ_r sinθ + ψ_i cosθ).
     * Used by NPT head projection (V_i = torus_wf · e^{i·2π·f_i·t}).
     *
     * @param theta  Phase angle in radians.
     */
    void phase_rotate_psi(float theta) noexcept
    {
        const float c = std::cos(theta);
        const float s = std::sin(theta);
        const size_t N = grid_->num_active_nodes();
        float* pr = grid_->psi_real();
        float* pi = grid_->psi_imag();
        for (size_t i = 0; i < N; ++i) {
            const float r = pr[i];
            const float im = pi[i];
            pr[i] = r * c - im * s;
            pi[i] = r * s + im * c;
        }
    }

    /**
     * @brief Riemannian curvature proxy: mean plasticity = mean(1 − resonance[i]).
     *
     * The resonance field r[i] ∈ [0, 1] encodes neural plasticity:
     *   r = 0  → fully plastic  (maximum curvature, R̄ = 1)
     *   r = 1  → fully frozen   (zero curvature,    R̄ = 0)
     *
     * Mean curvature R̄ = (1/N) Σᵢ (1 − r[i])
     *
     * After seed_manifold() all nodes have resonance = 0.5 → R̄ = 0.5.
     * Used by NPT curvature bias to weight attention heads.
     *
     * @return Non-negative curvature proxy ∈ [0, 1].
     */
    [[nodiscard]]
    double mean_curvature() const noexcept
    {
        const size_t N = grid_->num_active_nodes();
        if (N == 0) return 0.0;
        const float* r = grid_->resonance();
        double sum = 0.0;
        for (size_t i = 0; i < N; ++i)
            sum += static_cast<double>(1.0f - r[i]);
        return sum / static_cast<double>(N);
    }

    /**
     * @brief Return a deep copy of this WaveFunction (explicit clone).
     *
     * Allocates a fresh grid with the same topology as this one, then
     * blast-copies psi and vel arrays.  The cloned time_ is preserved.
     * Used by NPT project_heads() to create per-head V fields.
     *
     * @return A new WaveFunction owning a copy of this one's field data.
     */
    [[nodiscard]]
    WaveFunction clone() const
    {
        const int n = grid_->grid_n();
        WaveFunction result;
        result.seed_manifold(n, 0, 1, 0.f, 0);  // allocates n^9 nodes (vacuuum)
        const size_t N = grid_->num_active_nodes();
        // Blast-copy psi and vel arrays.
        std::memcpy(result.grid_->psi_real(), grid_->psi_real(), N * sizeof(float));
        std::memcpy(result.grid_->psi_imag(), grid_->psi_imag(), N * sizeof(float));
        std::memcpy(result.grid_->vel_real(), grid_->vel_real(), N * sizeof(float));
        std::memcpy(result.grid_->vel_imag(), grid_->vel_imag(), N * sizeof(float));
        result.time_ = time_;
        return result;
    }

    /**
     * @brief Renormalise wavefunction to target total probability.
     *
     * Called as a last resort if emergency damping fails.
     * @param target_prob  Desired Σ|Ψᵢ|² (default 1.0 for unit norm).
     */
    void renormalize(double target_prob = 1.0) noexcept {
        const double current = total_probability();
        if (current < 1e-30) return;  // nothing to normalise
        const float scale = static_cast<float>(std::sqrt(target_prob / current));
        const size_t N = grid_->num_active_nodes();
        float* pr = grid_->psi_real();
        float* pi = grid_->psi_imag();
        for (size_t i = 0; i < N; ++i) {
            pr[i] *= scale;
            pi[i] *= scale;
        }
    }

private:
    // ------------------------------------------------------------------ seeding helpers

    void seed_recursive(std::array<int,TORUS_DIMS>& c, int depth,
                        int n, int offset,
                        int pilot_dim, int k_mode, float amplitude,
                        std::mt19937& rng)
    {
        if (depth == TORUS_DIMS) {
            // Pilot wave: in the chosen dimension, set standing wave.
            const float sigma = foundation::thermal_sigma(9.f); // flat metric Tr = 9
            TorusNode node{};
            node.psi         = foundation::pilot_wave(c[pilot_dim], n, k_mode, amplitude);
            node.vel         = foundation::sample_thermal(sigma, rng);
            node.resonance   = 0.5f;
            node.state_field = 0.f;
            grid_->add_node(c, node);
            return;
        }
        for (int i = 0; i < n; ++i) {
            c[depth] = i + offset;
            seed_recursive(c, depth + 1, n, offset, pilot_dim, k_mode, amplitude, rng);
        }
    }

    // ------------------------------------------------------------------ data
    std::unique_ptr<TorusGrid> grid_;
    float time_{0.f};
};

} // namespace nikola::physics
