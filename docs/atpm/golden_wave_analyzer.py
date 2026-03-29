#!/usr/bin/env python3
"""
Golden-Prime Wave Superposition Analyzer

Calculates the average, minimum, and maximum amplitudes of a 13-wave
superposition system with:
- Amplitudes decaying by Golden Ratio (φ^-n)
- Frequencies following prime number sequence
- Phase shifts inversely proportional to frequency (π/f)

Based on S-URAM (Simulated Unified Resultant Amplitude Method) theory
as independently identified by Gemini from raw frequency/phase data.
"""

import numpy as np
from scipy.optimize import minimize_scalar, differential_evolution
import matplotlib.pyplot as plt
from typing import Tuple, Optional

# Mathematical constants
PHI = (1 + np.sqrt(5)) / 2  # Golden Ratio: 1.618033988749...
PI = np.pi

# System constants (from the paper)
JACKSON_SOKAL = 5/13  # Topological constant: 0.384615...
PHASE_ASYMMETRY = 179.0  # degrees (CP-violation analog)
REFERENCE_PHASE = 0.0  # Base clock (can be modified)

# Prime frequencies for the 13 waves
# Wave 0 has f=0 (DC component that evaluates to zero)
# Waves 1-12 use primes: 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31
FREQUENCIES = [0, 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

class GoldenWaveAnalyzer:
    """Analyzer for 13-wave golden-ratio prime-frequency superposition."""
    
    def __init__(self, apply_jackson_sokal: bool = False, 
                 apply_phase_asymmetry: bool = False,
                 reference_phase: float = 0.0):
        """
        Initialize the wave analyzer.
        
        Args:
            apply_jackson_sokal: Scale domain by Jackson-Sokal constant
            apply_phase_asymmetry: Apply 179° asymmetry to wave 2
            reference_phase: Global phase shift (base clock)
        """
        self.apply_js = apply_jackson_sokal
        self.apply_asym = apply_phase_asymmetry
        self.ref_phase = reference_phase
        
        # Fundamental period is 2π (lowest non-zero frequency is 1)
        self.fundamental_period = 2 * PI
        
        # Build wave parameters
        self.n_waves = 13
        self.amplitudes = [PHI**(-n) for n in range(self.n_waves)]
        self.frequencies = FREQUENCIES
        
        # Phase shifts: π/f_n (with special handling for f=0)
        self.phases = []
        for i, f in enumerate(self.frequencies):
            if f == 0:
                self.phases.append(0.0)
            else:
                phase = PI / f
                # Apply 179° asymmetry to wave 2 if requested
                if self.apply_asym and i == 2:
                    phase = np.deg2rad(PHASE_ASYMMETRY)
                self.phases.append(phase)
    
    def wave_component(self, n: int, x: np.ndarray) -> np.ndarray:
        """
        Calculate individual wave component p_n(x).
        
        Args:
            n: Wave index (0-12)
            x: Evaluation points
            
        Returns:
            Wave values at x
        """
        A_n = self.amplitudes[n]
        f_n = self.frequencies[n]
        phi_n = self.phases[n]
        
        # For f=0, sin(0*x + 0) = 0 always (DC offset is zero)
        if f_n == 0:
            return np.zeros_like(x)
        
        return A_n * np.sin(f_n * x + phi_n)
    
    def superposition(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate total superposition S(x) = Σ p_n(x).
        
        Args:
            x: Evaluation points (can be array or scalar)
            
        Returns:
            Superposed wave values
        """
        x = np.atleast_1d(x)
        
        # Apply reference phase (base clock)
        x_shifted = x + self.ref_phase
        
        # Apply Jackson-Sokal domain scaling if requested
        if self.apply_js:
            x_shifted = x_shifted * JACKSON_SOKAL
        
        # Sum all wave components
        result = np.zeros_like(x_shifted)
        for n in range(self.n_waves):
            result += self.wave_component(n, x_shifted)
        
        return result
    
    def calculate_average_analytical(self) -> float:
        """
        Calculate RMS average amplitude using Parseval's theorem.
        
        This is exact and doesn't require sampling.
        
        Returns:
            RMS average amplitude
        """
        # RMS = sqrt(Σ (A_n^2 / 2)) for n=1 to 12 (skip n=0 since f=0)
        # This forms a geometric series with ratio r = φ^-2
        
        sum_of_squares = 0.0
        for n in range(1, self.n_waves):  # Start at 1, skip DC component
            A_n = self.amplitudes[n]
            sum_of_squares += (A_n ** 2) / 2
        
        rms_average = np.sqrt(sum_of_squares)
        
        # The paper shows this simplifies to approximately:
        # RMS ≈ (sqrt(2)/2) * φ^-1 ≈ 0.564
        return rms_average
    
    def find_extrema_grid_search(self, n_points: int = 10000) -> Tuple[float, float, float, float]:
        """
        Find min/max using dense grid search over one fundamental period.
        
        Args:
            n_points: Number of grid points (paper recommends ~10,000)
            
        Returns:
            (x_min, min_value, x_max, max_value)
        """
        # Evaluate over one fundamental period [0, 2π]
        x_grid = np.linspace(0, self.fundamental_period, n_points)
        y_grid = self.superposition(x_grid)
        
        # Find grid extrema
        min_idx = np.argmin(y_grid)
        max_idx = np.argmax(y_grid)
        
        x_min_approx = x_grid[min_idx]
        x_max_approx = x_grid[max_idx]
        
        return (x_min_approx, y_grid[min_idx], 
                x_max_approx, y_grid[max_idx])
    
    def find_extrema_optimized(self, grid_points: int = 10000) -> Tuple[float, float, float, float]:
        """
        Find precise min/max using grid search + gradient optimization (BFGS).
        
        This is the recommended approach from the paper.
        
        Args:
            grid_points: Initial grid density
            
        Returns:
            (x_min, min_value, x_max, max_value)
        """
        # Phase 1: Grid search to find approximate locations
        x_min_approx, min_approx, x_max_approx, max_approx = \
            self.find_extrema_grid_search(grid_points)
        
        # Phase 2: Local refinement using gradient optimization
        # For minimum: minimize S(x)
        result_min = minimize_scalar(
            lambda x: self.superposition(np.array([x]))[0],
            bounds=(max(0, x_min_approx - 0.1), 
                   min(self.fundamental_period, x_min_approx + 0.1)),
            method='bounded'
        )
        
        # For maximum: minimize -S(x)
        result_max = minimize_scalar(
            lambda x: -self.superposition(np.array([x]))[0],
            bounds=(max(0, x_max_approx - 0.1), 
                   min(self.fundamental_period, x_max_approx + 0.1)),
            method='bounded'
        )
        
        x_min_refined = result_min.x
        min_refined = self.superposition(np.array([x_min_refined]))[0]
        
        x_max_refined = result_max.x
        max_refined = self.superposition(np.array([x_max_refined]))[0]
        
        return (x_min_refined, min_refined, x_max_refined, max_refined)
    
    def analyze(self, use_optimization: bool = True, 
                grid_points: int = 10000) -> dict:
        """
        Perform complete analysis of the wave system.
        
        Args:
            use_optimization: Use BFGS refinement (slower but precise)
            grid_points: Grid density
            
        Returns:
            Dictionary with all metrics
        """
        # Calculate average (analytical - instant and exact)
        avg = self.calculate_average_analytical()
        
        # Find extrema
        if use_optimization:
            x_min, min_val, x_max, max_val = self.find_extrema_optimized(grid_points)
            method = "Grid + BFGS"
        else:
            x_min, min_val, x_max, max_val = self.find_extrema_grid_search(grid_points)
            method = "Grid Search"
        
        # Calculate crest factor (peak-to-RMS ratio)
        crest_factor = max(abs(max_val), abs(min_val)) / avg
        
        return {
            'average_rms': avg,
            'minimum': min_val,
            'maximum': max_val,
            'x_at_minimum': x_min,
            'x_at_maximum': x_max,
            'peak_to_peak': max_val - min_val,
            'crest_factor': crest_factor,
            'method': method,
            'grid_points': grid_points
        }
    
    def plot_waveform(self, n_points: int = 5000, 
                     show_components: bool = False,
                     filename: Optional[str] = None):
        """
        Plot the superposition waveform.
        
        Args:
            n_points: Number of plot points
            show_components: Also plot individual wave components
            filename: If provided, save plot to file
        """
        x = np.linspace(0, self.fundamental_period, n_points)
        y = self.superposition(x)
        
        fig, axes = plt.subplots(2 if show_components else 1, 1, 
                                 figsize=(14, 10 if show_components else 6))
        
        if not show_components:
            axes = [axes]
        
        # Main superposition plot
        ax = axes[0]
        ax.plot(x, y, 'b-', linewidth=1.5, label='Total Superposition S(x)')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Mark extrema
        results = self.analyze(use_optimization=True)
        ax.plot(results['x_at_maximum'], results['maximum'], 
               'ro', markersize=10, label=f"Max: {results['maximum']:.4f}")
        ax.plot(results['x_at_minimum'], results['minimum'], 
               'go', markersize=10, label=f"Min: {results['minimum']:.4f}")
        ax.axhline(y=results['average_rms'], color='purple', 
                  linestyle=':', linewidth=2, 
                  label=f"RMS Avg: {results['average_rms']:.4f}")
        ax.axhline(y=-results['average_rms'], color='purple', 
                  linestyle=':', linewidth=2, alpha=0.5)
        
        ax.set_xlabel('x (radians)', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        ax.set_title('13-Wave Golden-Prime Superposition (φ-decay, Prime frequencies)', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Component waves plot
        if show_components:
            ax2 = axes[1]
            
            # Plot first 6 waves (dominant components)
            colors = plt.cm.viridis(np.linspace(0, 1, 6))
            for n in range(1, 7):  # Skip n=0 (zero DC)
                y_component = self.wave_component(n, x)
                ax2.plot(x, y_component, alpha=0.7, linewidth=1, 
                        color=colors[n-1],
                        label=f'p{n}: A={PHI**(-n):.3f}, f={FREQUENCIES[n]}')
            
            ax2.set_xlabel('x (radians)', fontsize=12)
            ax2.set_ylabel('Amplitude', fontsize=12)
            ax2.set_title('Dominant Wave Components (n=1 to 6)', 
                         fontsize=12, fontweight='bold')
            ax2.legend(loc='upper right', fontsize=9)
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {filename}")
        
        plt.show()


def main():
    """Main analysis routine."""
    
    print("=" * 80)
    print("GOLDEN-PRIME WAVE SUPERPOSITION ANALYZER")
    print("=" * 80)
    print()
    print("System Configuration:")
    print(f"  - 13 waves with φ^-n amplitude decay")
    print(f"  - Prime number frequencies: {FREQUENCIES}")
    print(f"  - Phase shifts: π/f_n")
    print(f"  - Golden Ratio φ = {PHI:.10f}")
    print()
    
    # Standard analysis (no topology modifiers)
    print("Standard Analysis (Pure Golden-Prime System)")
    print("-" * 80)
    
    analyzer = GoldenWaveAnalyzer(
        apply_jackson_sokal=False,
        apply_phase_asymmetry=False,
        reference_phase=0.0
    )
    
    results = analyzer.analyze(use_optimization=True, grid_points=10000)
    
    print(f"\n📊 Results ({results['method']}, {results['grid_points']:,} grid points):")
    print(f"   Average (RMS):     {results['average_rms']:.10f}")
    print(f"   Minimum:           {results['minimum']:.10f}")
    print(f"   Maximum:           {results['maximum']:.10f}")
    print(f"   Peak-to-Peak:      {results['peak_to_peak']:.10f}")
    print(f"   Crest Factor:      {results['crest_factor']:.6f}")
    print()
    print(f"   x at minimum:      {results['x_at_minimum']:.10f} rad")
    print(f"   x at maximum:      {results['x_at_maximum']:.10f} rad")
    print()
    
    # Theoretical comparison
    theoretical_avg = (np.sqrt(2) / 2) * (1 / PHI)
    print(f"📐 Theoretical RMS (from paper): {theoretical_avg:.10f}")
    print(f"   Error: {abs(results['average_rms'] - theoretical_avg):.2e}")
    print()
    
    # S-URAM connection
    print("🌀 S-URAM Physical Interpretation:")
    print("   - 13 waves → 13 Planck Spherical Units (cuboctahedron)")
    print("   - φ decay → Golden ratio quantum geometry")
    print("   - Prime frequencies → Non-harmonic interference patterns")
    print(f"   - Crest factor {results['crest_factor']:.3f} → "
          f"Peak is {results['crest_factor']:.1f}× RMS average")
    print()
    
    # Generate visualization
    print("Generating visualization...")
    analyzer.plot_waveform(
        n_points=5000, 
        show_components=True,
        filename="/home/randy/Workspace/SYSTEM/Downloads/atpm/golden_wave_plot.png"
    )
    
    # Optional: Analysis with topology modifiers
    print("\n" + "=" * 80)
    print("Advanced Analysis (With Topological Modifiers)")
    print("=" * 80)
    
    analyzer_advanced = GoldenWaveAnalyzer(
        apply_jackson_sokal=True,
        apply_phase_asymmetry=True,
        reference_phase=0.0
    )
    
    results_adv = analyzer_advanced.analyze(use_optimization=True, grid_points=10000)
    
    print(f"\n📊 Results with Jackson-Sokal ({JACKSON_SOKAL}) & 179° Asymmetry:")
    print(f"   Average (RMS):     {results_adv['average_rms']:.10f}")
    print(f"   Minimum:           {results_adv['minimum']:.10f}")
    print(f"   Maximum:           {results_adv['maximum']:.10f}")
    print(f"   Peak-to-Peak:      {results_adv['peak_to_peak']:.10f}")
    print(f"   Crest Factor:      {results_adv['crest_factor']:.6f}")
    print()
    
    print("✅ Analysis complete!")
    

if __name__ == '__main__':
    main()
