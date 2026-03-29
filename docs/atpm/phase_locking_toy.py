#!/usr/bin/env python3
"""phase_locking_toy.py

A minimal demonstration of how "perfect anti-phase" (180°) does NOT have to stay perfect.

This does NOT assume a physical "medium" for waves. It models only a phase difference Δ
between two oscillatory degrees of freedom.

Model 1 (continuous time):
    dΔ/dt = Δω + K sin(Δ)

- If Δω != 0 (tiny mismatch) and K > 0 (weak coupling), the system can phase-lock.
- With this sign convention, the stable locked solution is near anti-phase (Δ ≈ π).
  Locked points satisfy:
      sin(Δ*) = -Δω / K
  The one near π is:
      Δ* = π - arcsin(-Δω/K)

If we want a 1° offset from perfect anti-phase:
    ε = 1° = 0.0174533 rad
we can choose -Δω/K ≈ ε (i.e., Δω ≈ -Kε), which makes Δ* ≈ π - ε.

Model 2 (discrete "per lap" update):
    Δ_{n+1} = (Δ_n + ε - K sin(Δ_n)) mod 2π

This corresponds to the intuition "each pass around a torus gives a small drift + a grazing kick".

Run:
    python3 phase_locking_toy.py
"""

from __future__ import annotations

import math


def rad_to_deg(x: float) -> float:
    return x * 180.0 / math.pi


def residual_amplitude_factor(delta: float) -> float:
    """For two equal-amplitude sinusoids A sin(t) and A sin(t+Δ),
    the resultant amplitude is R = 2A cos(Δ/2).

    This returns R/A (dimensionless). Near anti-phase (Δ≈π), this is small.
    """
    return 2.0 * abs(math.cos(delta / 2.0))


def wrap_pi(x: float) -> float:
    """Wrap angle to (-pi, pi]."""
    y = (x + math.pi) % (2 * math.pi) - math.pi
    # put -pi on +pi for nicer printing consistency
    if abs(y + math.pi) < 1e-12:
        return math.pi
    return y


def simulate_continuous(delta0: float, delta_omega: float, K: float, dt: float, steps: int) -> float:
    delta = delta0
    for _ in range(steps):
        delta += dt * (delta_omega + K * math.sin(delta))
        delta = (delta + 2 * math.pi) % (2 * math.pi)
    return delta


def simulate_discrete(delta0: float, eps: float, K: float, steps: int) -> float:
    delta = delta0
    for _ in range(steps):
        # Discrete analogue: drift per step (eps) plus a weak coupling “kick”.
        delta = (delta + eps + K * math.sin(delta)) % (2 * math.pi)
    return delta


def main() -> None:
    # Start exactly anti-phase.
    delta0 = math.pi  # 180°

    # Target offset from anti-phase.
    eps_deg = 1.0
    eps = math.radians(eps_deg)

    # Choose parameters so that -Δω/K ≈ eps.
    # Then the locked solution near π is Δ* = π - arcsin(-Δω/K) ≈ π - eps.
    K = 1.0
    delta_omega = -K * eps

    # Predicted locked point near anti-phase.
    delta_star = math.pi - math.asin(-delta_omega / K)

    print("=== Continuous-time phase locking ===")
    print(f"Start Δ0:           {rad_to_deg(delta0):9.4f}°")
    print(f"Chosen Δω/K:        {delta_omega / K:9.6f} rad  (~ {rad_to_deg(delta_omega / K):.4f}°)")
    print(f"Predicted Δ*:       {rad_to_deg(delta_star):9.4f}°  (should be ~ 179°)")

    # Integrate.
    dt = 0.05
    steps = 4000
    delta_end = simulate_continuous(delta0, delta_omega, K, dt, steps)

    # Show how close it got to Δ* and to (π - 1°).
    err_to_star = wrap_pi(delta_end - delta_star)
    err_to_179 = wrap_pi(delta_end - (math.pi - eps))

    # Report both the raw [0, 360) angle and its distance from 180°.
    print(f"Simulated Δ_end:    {rad_to_deg(delta_end):9.4f}°")
    print(f"Δ_end - Δ*:         {rad_to_deg(err_to_star): .6f}°")
    print(f"Δ_end - (π-1°):     {rad_to_deg(err_to_179): .6f}°")

    raf = residual_amplitude_factor(delta_end)
    print(f"Residual R/A @Δ_end:{raf:9.6f}   (e.g. Δ=179° → ~0.01745)")

    print("\n=== Discrete 'per lap' update (drift + grazing kick) ===")
    # Choose a small negative drift so the stable fixed point sits at π - 1°.
    delta_end_d = simulate_discrete(delta0, eps=-eps, K=1.0, steps=5000)
    err_to_179_d = wrap_pi(delta_end_d - (math.pi - eps))
    print(f"Simulated Δ_end:    {rad_to_deg(delta_end_d):9.4f}°")
    print(f"Δ_end - (π-1°):     {rad_to_deg(err_to_179_d): .6f}°")

    raf_d = residual_amplitude_factor(delta_end_d)
    print(f"Residual R/A @Δ_end:{raf_d:9.6f}   (e.g. Δ=179° → ~0.01745)")

    print("\nInterpretation:")
    print("- This shows you do NOT have to 'start at 179°'. Starting at 180° is fine.")
    print("- Any tiny mismatch (Δω) breaks perfect cancellation immediately.")
    print("- Weak coupling (K) can make the system lock near anti-phase with a small stable offset.")
    print("- In a torus picture, repeated passes can act like discrete updates ('kicks') that accumulate.")


if __name__ == "__main__":
    main()
