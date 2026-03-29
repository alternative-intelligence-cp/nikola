#!/usr/bin/env python3
"""spherical_residual_proxy_dipole.py

A tiny, testable “residual proxy on a sphere” toy model.

Purpose
-------
You said the word “spherical” made your brain itch (in a good way). This script makes a
simple *sky-like* scalar field Q(n̂) on the unit sphere and extracts its best-fit dipole.

It stays non-metaphysical:
- We do not claim what the substrate is.
- We assume an effective relative phase Δ(n̂) that is near anti-phase (≈ π) but with a
  small offset ε(n̂).
- The residual amplitude factor for two equal components is:
      R/A = 2|cos(Δ/2)|
  and near anti-phase (Δ = π - ε), we have R/A ≈ ε (with ε in radians).

We then define a proxy observable Q(n̂) using either:
- H1 (linear):    Q ∝ ε
- H2 (quadratic): Q ∝ ε²

and show that this naturally produces a dipole if ε varies slightly across the sphere.

Run
---
    python3 spherical_residual_proxy_dipole.py

Outputs
-------
- Prints the fitted dipole direction/amplitude
- Saves an image: spherical_residual_proxy_dipole.png
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n != 0 else v


def sph_to_cart(lon_rad: float, lat_rad: float) -> np.ndarray:
    """lon: [-pi, pi], lat: [-pi/2, pi/2]"""
    clat = math.cos(lat_rad)
    return np.array([
        clat * math.cos(lon_rad),
        clat * math.sin(lon_rad),
        math.sin(lat_rad),
    ])


def cart_to_sph(n: np.ndarray) -> tuple[float, float]:
    """Return (lon, lat) in radians."""
    x, y, z = n
    lon = math.atan2(y, x)
    lat = math.asin(max(-1.0, min(1.0, z / (np.linalg.norm(n) + 1e-12))))
    return lon, lat


def sample_sphere(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform random unit vectors, shape (n, 3)."""
    u = rng.uniform(0.0, 1.0, size=n)
    v = rng.uniform(0.0, 1.0, size=n)
    lon = 2 * math.pi * u - math.pi
    z = 2 * v - 1
    r_xy = np.sqrt(np.maximum(0.0, 1 - z * z))
    x = r_xy * np.cos(lon)
    y = r_xy * np.sin(lon)
    return np.stack([x, y, z], axis=1)


def fit_dipole(nhat: np.ndarray, Q: np.ndarray) -> tuple[float, np.ndarray]:
    """Fit Q(n̂) ≈ a + d·n̂ via least squares.

    Returns (a, dvec).
    """
    A = np.concatenate([np.ones((nhat.shape[0], 1)), nhat], axis=1)  # (N,4)
    coeff, *_ = np.linalg.lstsq(A, Q, rcond=None)
    a = float(coeff[0])
    d = coeff[1:4]
    return a, d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--eps0-deg', type=float, default=1.0, help='base offset ε0 in degrees (near anti-phase)')
    ap.add_argument('--eps1-deg', type=float, default=0.15, help='dipole modulation ε1 in degrees')
    ap.add_argument('--dipole-lon-deg', type=float, default=0.0, help='dipole axis longitude (deg)')
    ap.add_argument('--dipole-lat-deg', type=float, default=30.0, help='dipole axis latitude (deg)')
    ap.add_argument('--mapping', choices=['H1', 'H2'], default='H1', help='H1: Q∝ε, H2: Q∝ε²')
    ap.add_argument('--grid', type=int, default=361, help='grid resolution for visualization')
    ap.add_argument('--samples', type=int, default=20000, help='random samples for dipole fit')
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    eps0 = math.radians(args.eps0_deg)
    eps1 = math.radians(args.eps1_deg)

    dip_axis = sph_to_cart(math.radians(args.dipole_lon_deg), math.radians(args.dipole_lat_deg))
    dip_axis = unit(dip_axis)

    # Define ε(n̂) = ε0 + ε1 (dip_axis · n̂)
    nhat = sample_sphere(args.samples, rng)
    eps = eps0 + eps1 * (nhat @ dip_axis)

    # Keep ε positive in this toy (avoids sign flips in the small-angle approx discussion).
    # In a more serious model you’d decide what the sign means.
    eps = np.maximum(eps, 1e-8)

    if args.mapping == 'H1':
        Q = eps
    else:
        Q = eps * eps

    a, d = fit_dipole(nhat, Q)

    d_amp = float(np.linalg.norm(d))
    d_dir = unit(d)
    lon, lat = cart_to_sph(d_dir)

    print('=== Residual proxy dipole fit ===')
    print(f'mapping: {args.mapping}')
    print(f'ε0: {args.eps0_deg:.4f}°   ε1: {args.eps1_deg:.4f}°')
    print(f'fit monopole a: {a:.6e}')
    print(f'fit dipole |d|: {d_amp:.6e}')
    print(f'fit dipole direction lon,lat: {math.degrees(lon):.2f}°, {math.degrees(lat):.2f}°')

    # Visualize on a lon/lat grid with Mollweide projection.
    grid = args.grid
    lon_grid = np.linspace(-math.pi, math.pi, grid)
    lat_grid = np.linspace(-math.pi / 2, math.pi / 2, grid // 2)
    LON, LAT = np.meshgrid(lon_grid, lat_grid)

    # Compute Q on the grid.
    x = np.cos(LAT) * np.cos(LON)
    y = np.cos(LAT) * np.sin(LON)
    z = np.sin(LAT)
    n_grid = np.stack([x, y, z], axis=2)

    eps_grid = eps0 + eps1 * (n_grid @ dip_axis)
    eps_grid = np.maximum(eps_grid, 1e-8)
    if args.mapping == 'H1':
        Q_grid = eps_grid
    else:
        Q_grid = eps_grid * eps_grid

    # Normalize for visualization.
    Qv = (Q_grid - Q_grid.min()) / (Q_grid.max() - Q_grid.min() + 1e-12)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='mollweide')
    im = ax.pcolormesh(LON, LAT, Qv, shading='auto', cmap='viridis')
    ax.grid(True, alpha=0.35)
    ax.set_title('Residual proxy Q(n̂) on sphere (normalized)')
    cb = plt.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label('Q (normalized)')

    # Mark the intended dipole axis.
    ax.plot([math.radians(args.dipole_lon_deg)], [math.radians(args.dipole_lat_deg)], 'ro', markersize=6, label='input axis')
    ax.plot([lon], [lat], 'wx', markersize=8, markeredgewidth=2, label='fit axis')
    ax.legend(loc='lower left')

    out = '/home/randy/Workspace/SYSTEM/Downloads/atpm/spherical_residual_proxy_dipole.png'
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
