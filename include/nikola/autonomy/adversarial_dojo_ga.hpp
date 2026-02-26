/// @file   adversarial_dojo_ga.hpp
/// @brief  Adversarial Code Dojo — Genetic Algorithm Red Team (GAP-035)
///
/// Evolves attack waveform parameters using GAlib 2.4 to maximise the energy
/// perturbation |dH/dt| in a simplified 1D Hamiltonian system.
///
/// **Design**
/// - Genome: `GA1DArrayGenome<float>` of 3 genes
///     - gene[0] = amplitude  ∈ [0.0, 1.0]
///     - gene[1] = frequency  ∈ [0.0, 4π]
///     - gene[2] = phase      ∈ [0.0, 2π]
/// - Objective: maximise peak energy deflection over ATTACK_STEPS time-steps
/// - Algorithm: `GASimpleGA` (generational), configurable population + gens
///
/// **Biological analogy** (per §05_autonomous_systems/01_computational_neurochemistry.md)
/// Red team genomes represent adversarial waveforms injected into the toroidal
/// manifold; the oracle (blue team) flags those that exceed acceptable dH/dt.
///
/// **GAlib API contract**
/// `objective_fn` must be a module-level `float(GAGenome&)` function.  All
/// internal state shared between objective + the dojo is stored in the
/// thread-local `dojo_context` struct.

#pragma once

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>

#include <ga/GA1DArrayGenome.h>
#include <ga/GASimpleGA.h>
#include <ga/GAStatistics.h>

namespace nikola::autonomy {

// ---------------------------------------------------------------------------
// Domain constants
// ---------------------------------------------------------------------------

constexpr int   ATTACK_GENOME_SIZE = 3;    ///< Genes: [amplitude, frequency, phase]
constexpr int   ATTACK_STEPS       = 64;   ///< Simulation ticks per fitness eval
constexpr float ATTACK_DT          = 0.05f;///< Simulation time-step
constexpr float ATTACK_AMP_MAX     = 1.0f; ///< Max amplitude gene value
constexpr float ATTACK_FREQ_MAX    = static_cast<float>(4.0 * M_PI); ///< Max frequency
constexpr float ATTACK_PHASE_MAX   = static_cast<float>(2.0 * M_PI); ///< Max phase

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// Decoded attack parameter set (one individual in the GA population).
struct AttackParams {
    float amplitude{0.0f};   ///< Waveform amplitude   ∈ [0, ATTACK_AMP_MAX]
    float frequency{0.0f};   ///< Waveform frequency   ∈ [0, ATTACK_FREQ_MAX]
    float phase    {0.0f};   ///< Initial phase offset ∈ [0, ATTACK_PHASE_MAX]
};

/// Result of a single attack simulation run.
struct AttackResult {
    float peak_energy_delta{0.0f}; ///< max(|ψ(t)|²) − |ψ(0)|²
    float mean_energy_delta{0.0f}; ///< mean(|ψ(t)|²) − |ψ(0)|²
    int   steps{0};                ///< Number of simulation steps executed
};

/// Summary returned by `run_adversarial_ga()`.
struct DojoResult {
    AttackParams best;             ///< Best-evolved attack genome (decoded)
    float        best_fitness{0.f};///< Peak fitness (energy delta) ever seen
    int          generations{0};   ///< Actual number of generations run
    float        stat_max{0.f};    ///< `GAStatistics::maxEver()` for the run
    float        stat_min{0.f};    ///< `GAStatistics::minEver()` for the run
};

// ---------------------------------------------------------------------------
// Standalone physics kernel
// ---------------------------------------------------------------------------

/// @brief Simulate `steps` ticks of a 1D attacked Hamiltonian.
///
/// State: ψ ∈ ℂ;   H = |ψ|²
/// Attack term: ψ += dt · amplitude · exp(i·(frequency·t + phase))
///
/// @returns AttackResult with energy deltas.
inline AttackResult simulate_attack(const AttackParams& p,
                                    int    steps = ATTACK_STEPS,
                                    float  dt    = ATTACK_DT) noexcept
{
    using C = std::complex<float>;

    C   psi         = {1.0f, 0.0f};          // initial ground state
    float H0        = std::norm(psi);          // initial energy
    float peak_H    = H0;
    float total_H   = 0.0f;

    for (int t = 0; t < steps; ++t) {
        float arg = p.frequency * (static_cast<float>(t) * dt) + p.phase;
        C     kick = {dt * p.amplitude * std::cos(arg),
                      dt * p.amplitude * std::sin(arg)};
        psi += kick;
        float H = std::norm(psi);
        if (H > peak_H) peak_H = H;
        total_H += H;
    }

    float mean_H = total_H / static_cast<float>(steps);
    return AttackResult{peak_H - H0, mean_H - H0, steps};
}

// ---------------------------------------------------------------------------
// GAlib genome callbacks (need external linkage for GAlib C callback protocol)
// ---------------------------------------------------------------------------

/// Decode the first three genes of a GA1DArrayGenome<float> into AttackParams.
inline AttackParams decode_genome(const GA1DArrayGenome<float>& g) noexcept
{
    return AttackParams{
        /* amplitude */ g.gene(0) * ATTACK_AMP_MAX,
        /* frequency */ g.gene(1) * ATTACK_FREQ_MAX,
        /* phase     */ g.gene(2) * ATTACK_PHASE_MAX,
    };
}

/// Objective function: higher = more destructive attack (maximised by GA).
inline float attack_objective(GAGenome& raw_genome) noexcept
{
    const auto& g  = static_cast<const GA1DArrayGenome<float>&>(raw_genome);
    AttackParams p  = decode_genome(g);
    AttackResult r  = simulate_attack(p);
    return r.peak_energy_delta;
}

/// Genome initialiser: randomise each gene uniformly in [0, 1].
inline void attack_initializer(GAGenome& raw_genome)
{
    auto& g = static_cast<GA1DArrayGenome<float>&>(raw_genome);
    for (int i = 0; i < g.length(); ++i)
        g.gene(i, GARandomFloat(0.0f, 1.0f));
}

/// Gaussian mutator: with probability pMutation, perturb each gene ±σ.
inline int attack_mutator(GAGenome& raw_genome, float pMutation)
{
    auto& g    = static_cast<GA1DArrayGenome<float>&>(raw_genome);
    int   nmut = 0;
    for (int i = 0; i < g.length(); ++i) {
        if (GAFlipCoin(pMutation)) {
            float sigma = 0.1f;
            float v     = g.gene(i) + GARandomFloat(-sigma, sigma);
            // Clamp to [0, 1]
            v = (v < 0.0f) ? 0.0f : (v > 1.0f) ? 1.0f : v;
            g.gene(i, v);
            ++nmut;
        }
    }
    return nmut;
}

// ---------------------------------------------------------------------------
// Main GA runner
// ---------------------------------------------------------------------------

/// @brief Run the adversarial dojo GA and return the best attack found.
///
/// @param population_size Number of individuals per generation.
/// @param n_generations   Number of generations to evolve.
/// @param seed            Random seed (0 = use GAlib's default seeding).
/// @param p_mutation      Per-gene mutation probability.
/// @param p_crossover     Crossover probability.
///
/// @throws std::invalid_argument if population_size < 2 or n_generations < 1.
inline DojoResult run_adversarial_ga(int   population_size = 30,
                                     int   n_generations   = 10,
                                     int   seed            = 1,
                                     float p_mutation      = 0.05f,
                                     float p_crossover     = 0.6f)
{
    if (population_size < 2)
        throw std::invalid_argument(
            "run_adversarial_ga: population_size must be >= 2");
    if (n_generations < 1)
        throw std::invalid_argument(
            "run_adversarial_ga: n_generations must be >= 1");

    // Build genome template
    GA1DArrayGenome<float> genome(ATTACK_GENOME_SIZE, attack_objective);
    genome.initializer(attack_initializer);
    genome.mutator(attack_mutator);
    genome.crossover(GA1DArrayGenome<float>::UniformCrossover);

    // Build and configure the simple GA
    GASimpleGA ga(genome);
    ga.maximize();                              // larger fitness = better
    ga.populationSize(population_size);
    ga.nGenerations(n_generations);
    ga.pMutation(p_mutation);
    ga.pCrossover(p_crossover);
    if (seed != 0) GARandomSeed(static_cast<unsigned int>(seed));

    ga.initialize();
    ga.evolve();

    // Extract result
    const auto& stats = ga.statistics();
    const auto& best  = static_cast<const GA1DArrayGenome<float>&>(
                            stats.bestIndividual());

    DojoResult r;
    r.best          = decode_genome(best);
    r.best_fitness  = stats.maxEver();
    r.generations   = n_generations;
    r.stat_max      = stats.maxEver();
    r.stat_min      = stats.minEver();

    return r;
}

} // namespace nikola::autonomy
