#!/usr/bin/env python3
"""
Golden-Prime Wave Superposition Analyzer (Non-Interactive Mode)
Outputs detailed numerical analysis without interactive plots.
"""

import numpy as np
from scipy.optimize import minimize_scalar
import sys

# Mathematical constants
PHI = (1 + np.sqrt(5)) / 2  # Golden Ratio: 1.618033988749...
PI = np.pi

# System constants (from the paper)
JACKSON_SOKAL = 5/13  # Topological constant: 0.384615...
PHASE_ASYMMETRY = 179.0  # degrees (CP-violation analog)
REFERENCE_PHASE = 0.0  # Base clock (can be modified)

# Prime frequencies for the 13 waves
FREQUENCIES = [0, 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

class GoldenWaveAnalyzer:
    """Analyzer for 13-wave golden-ratio prime-frequency superposition."""
    
    def __init__(self, apply_jackson_sokal=False, apply_phase_asymmetry=False, reference_phase=0.0):
        self.apply_js = apply_jackson_sokal
        self.apply_asym = apply_phase_asymmetry
        self.ref_phase = reference_phase
        self.fundamental_period = 2 * PI
        self.n_waves = 13
        self.amplitudes = [PHI**(-n) for n in range(self.n_waves)]
        self.frequencies = FREQUENCIES
        
        # Phase shifts: π/f_n
        self.phases = []
        for i, f in enumerate(self.frequencies):
            if f == 0:
                self.phases.append(0.0)
            else:
                phase = PI / f
                if self.apply_asym and i == 2:
                    phase = np.deg2rad(PHASE_ASYMMETRY)
                self.phases.append(phase)
    
    def wave_component(self, n, x):
        """Calculate individual wave component p_n(x)."""
        A_n = self.amplitudes[n]
        f_n = self.frequencies[n]
        phi_n = self.phases[n]
        
        if f_n == 0:
            return np.zeros_like(x)
        
        return A_n * np.sin(f_n * x + phi_n)
    
    def superposition(self, x):
        """Calculate total superposition S(x) = Σ p_n(x)."""
        x = np.atleast_1d(x)
        x_shifted = x + self.ref_phase
        
        if self.apply_js:
            x_shifted = x_shifted * JACKSON_SOKAL
        
        result = np.zeros_like(x_shifted)
        for n in range(self.n_waves):
            result += self.wave_component(n, x_shifted)
        
        return result
    
    def calculate_average_analytical(self):
        """Calculate RMS average amplitude using Parseval's theorem."""
        sum_of_squares = 0.0
        for n in range(1, self.n_waves):
            A_n = self.amplitudes[n]
            sum_of_squares += (A_n ** 2) / 2
        
        return np.sqrt(sum_of_squares)
    
    def find_extrema_grid_search(self, n_points=10000):
        """Find min/max using dense grid search."""
        x_grid = np.linspace(0, self.fundamental_period, n_points)
        y_grid = self.superposition(x_grid)
        
        min_idx = np.argmin(y_grid)
        max_idx = np.argmax(y_grid)
        
        return (x_grid[min_idx], y_grid[min_idx], x_grid[max_idx], y_grid[max_idx])
    
    def find_extrema_optimized(self, grid_points=10000):
        """Find precise min/max using grid search + gradient optimization."""
        x_min_approx, min_approx, x_max_approx, max_approx = self.find_extrema_grid_search(grid_points)
        
        # Refine minimum
        result_min = minimize_scalar(
            lambda x: self.superposition(np.array([x]))[0],
            bounds=(max(0, x_min_approx - 0.1), min(self.fundamental_period, x_min_approx + 0.1)),
            method='bounded'
        )
        
        # Refine maximum
        result_max = minimize_scalar(
            lambda x: -self.superposition(np.array([x]))[0],
            bounds=(max(0, x_max_approx - 0.1), min(self.fundamental_period, x_max_approx + 0.1)),
            method='bounded'
        )
        
        x_min_refined = result_min.x
        min_refined = self.superposition(np.array([x_min_refined]))[0]
        x_max_refined = result_max.x
        max_refined = self.superposition(np.array([x_max_refined]))[0]
        
        return (x_min_refined, min_refined, x_max_refined, max_refined)
    
    def analyze(self, use_optimization=True, grid_points=10000):
        """Perform complete analysis."""
        avg = self.calculate_average_analytical()
        
        if use_optimization:
            x_min, min_val, x_max, max_val = self.find_extrema_optimized(grid_points)
            method = "Grid + BFGS"
        else:
            x_min, min_val, x_max, max_val = self.find_extrema_grid_search(grid_points)
            method = "Grid Search"
        
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


def print_detailed_analysis(title, analyzer, show_wave_details=False):
    """Print comprehensive numerical analysis."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    
    if show_wave_details:
        print("\n📐 Wave Component Details:")
        print(f"   {'n':<4} {'Freq':<6} {'Amplitude (φ^-n)':<20} {'Phase (rad)':<15} {'Phase (°)':<12}")
        print("   " + "-" * 75)
        for n in range(analyzer.n_waves):
            amp = analyzer.amplitudes[n]
            freq = analyzer.frequencies[n]
            phase_rad = analyzer.phases[n]
            phase_deg = np.rad2deg(phase_rad)
            print(f"   {n:<4} {freq:<6} {amp:<20.15f} {phase_rad:<15.10f} {phase_deg:<12.6f}")
    
    results = analyzer.analyze(use_optimization=True, grid_points=10000)
    
    print(f"\n📊 Statistical Results ({results['method']}, {results['grid_points']:,} grid points):")
    print(f"   {'Metric':<25} {'Value':<20} {'Details'}")
    print("   " + "-" * 75)
    print(f"   {'Average (RMS)':<25} {results['average_rms']:<20.15f}")
    print(f"   {'Minimum':<25} {results['minimum']:<20.15f} (at x = {results['x_at_minimum']:.10f} rad)")
    print(f"   {'Maximum':<25} {results['maximum']:<20.15f} (at x = {results['x_at_maximum']:.10f} rad)")
    print(f"   {'Peak-to-Peak':<25} {results['peak_to_peak']:<20.15f}")
    print(f"   {'Crest Factor':<25} {results['crest_factor']:<20.10f} (peak/RMS ratio)")
    print(f"   {'|Min| / RMS':<25} {abs(results['minimum'])/results['average_rms']:<20.10f}")
    print(f"   {'|Max| / RMS':<25} {abs(results['maximum'])/results['average_rms']:<20.10f}")
    
    print(f"\n📍 Extrema Locations:")
    print(f"   Minimum at x = {results['x_at_minimum']:.15f} rad ({np.rad2deg(results['x_at_minimum']):.10f}°)")
    print(f"   Maximum at x = {results['x_at_maximum']:.15f} rad ({np.rad2deg(results['x_at_maximum']):.10f}°)")
    
    # Sample the waveform at key points
    print(f"\n🔍 Waveform Samples:")
    sample_points = [0, PI/4, PI/2, 3*PI/4, PI, 5*PI/4, 3*PI/2, 7*PI/4, 2*PI]
    for x in sample_points:
        y = analyzer.superposition(np.array([x]))[0]
        print(f"   S({x:.10f}) = {y:20.15f}   [x = {np.rad2deg(x):7.3f}°]")
    
    return results


def main():
    print("=" * 80)
    print("GOLDEN-PRIME WAVE SUPERPOSITION: COMPLETE NUMERICAL ANALYSIS")
    print("=" * 80)
    print(f"\nMathematical Constants:")
    print(f"   Golden Ratio (φ)       = {PHI:.20f}")
    print(f"   φ^-1                   = {1/PHI:.20f}")
    print(f"   φ^-2                   = {PHI**-2:.20f}")
    print(f"   Jackson-Sokal (5/13)   = {JACKSON_SOKAL:.20f}")
    print(f"   Phase Asymmetry        = {PHASE_ASYMMETRY}° = {np.deg2rad(PHASE_ASYMMETRY):.15f} rad")
    print(f"   π                      = {PI:.20f}")
    
    # STANDARD ANALYSIS
    print("\n\n" + "█" * 80)
    print("CONFIGURATION 1: PURE GOLDEN-PRIME SYSTEM")
    print("█" * 80)
    print("   Jackson-Sokal:     DISABLED")
    print("   Phase Asymmetry:   DISABLED (standard π/f phases)")
    print("   Reference Phase:   0.0")
    
    analyzer_std = GoldenWaveAnalyzer(
        apply_jackson_sokal=False,
        apply_phase_asymmetry=False,
        reference_phase=0.0
    )
    
    results_std = print_detailed_analysis(
        "Standard Golden-Prime Configuration", 
        analyzer_std,
        show_wave_details=True
    )
    
    # ADVANCED ANALYSIS
    print("\n\n" + "█" * 80)
    print("CONFIGURATION 2: ADVANCED TOPOLOGY (S-URAM PHYSICAL MODEL)")
    print("█" * 80)
    print(f"   Jackson-Sokal:     ENABLED ({JACKSON_SOKAL:.15f})")
    print(f"   Phase Asymmetry:   ENABLED ({PHASE_ASYMMETRY}° = {np.deg2rad(PHASE_ASYMMETRY):.15f} rad)")
    print("   Reference Phase:   0.0")
    print("\n   Physical Interpretation:")
    print("   - Jackson-Sokal scales the temporal domain by 5/13")
    print("   - 179° asymmetry simulates CP-violation (weak force analog)")
    print("   - Wave 2 (f=2) phase changed from π to 179° offset")
    
    analyzer_adv = GoldenWaveAnalyzer(
        apply_jackson_sokal=True,
        apply_phase_asymmetry=True,
        reference_phase=0.0
    )
    
    results_adv = print_detailed_analysis(
        "Advanced Topology Configuration",
        analyzer_adv,
        show_wave_details=True
    )
    
    # COMPARATIVE ANALYSIS
    print("\n\n" + "█" * 80)
    print("COMPARATIVE ANALYSIS: Standard vs. Advanced Topology")
    print("█" * 80)
    
    print(f"\n{'Metric':<30} {'Standard':<25} {'Advanced':<25} {'Δ (Adv - Std)':<25}")
    print("-" * 105)
    
    metrics = ['average_rms', 'minimum', 'maximum', 'peak_to_peak', 'crest_factor']
    for metric in metrics:
        std_val = results_std[metric]
        adv_val = results_adv[metric]
        delta = adv_val - std_val
        pct = (delta / std_val * 100) if std_val != 0 else 0
        print(f"{metric:<30} {std_val:<25.15f} {adv_val:<25.15f} {delta:<+25.15f} ({pct:+7.3f}%)")
    
    print(f"\n{'Position Metric':<30} {'Standard (rad)':<25} {'Advanced (rad)':<25} {'Δ (rad)':<25}")
    print("-" * 105)
    print(f"{'x at minimum':<30} {results_std['x_at_minimum']:<25.15f} {results_adv['x_at_minimum']:<25.15f} {results_adv['x_at_minimum']-results_std['x_at_minimum']:<+25.15f}")
    print(f"{'x at maximum':<30} {results_std['x_at_maximum']:<25.15f} {results_adv['x_at_maximum']:<25.15f} {results_adv['x_at_maximum']-results_std['x_at_maximum']:<+25.15f}")
    
    # THEORETICAL PREDICTIONS
    print("\n\n" + "█" * 80)
    print("THEORETICAL PREDICTIONS vs. COMPUTED VALUES")
    print("█" * 80)
    
    theoretical_rms = (np.sqrt(2) / 2) * (1 / PHI)
    print(f"\nRMS Average:")
    print(f"   Theoretical (√2/2 × φ^-1):  {theoretical_rms:.20f}")
    print(f"   Computed (Standard):        {results_std['average_rms']:.20f}")
    print(f"   Error:                      {abs(results_std['average_rms'] - theoretical_rms):.2e}")
    
    theoretical_max_limit = PHI  # If all waves aligned perfectly
    print(f"\nTheoretical Maximum Limit (perfect coherence):")
    print(f"   Limit (φ):                  {theoretical_max_limit:.20f}")
    print(f"   Computed (Standard):        {results_std['maximum']:.20f}")
    print(f"   Ratio (Computed/Limit):     {results_std['maximum']/theoretical_max_limit:.10f}")
    print(f"   % of theoretical max:       {results_std['maximum']/theoretical_max_limit*100:.6f}%")
    
    print("\n" + "=" * 80)
    print("✅ COMPLETE NUMERICAL ANALYSIS FINISHED")
    print("=" * 80)


if __name__ == '__main__':
    main()
