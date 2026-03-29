# CMB Dipole Wave Interference Visualization
# Demonstrates how 13 wave pairs with 179° phase offset create the observed cosmic dipole

import numpy as np
import matplotlib.pyplot as plt

# System Constants
PHI = (1 + np.sqrt(5)) / 2  # Golden Ratio = 1.618...
PI = np.pi
T_SYMBOL = 32 / 27  # Jackson-Sokal Constant
PHASE_OFFSET = 179  # degrees (1° from perfect cancellation)

# Prime number sequence for phase offsets (first 13, excluding 2)
PRIMES = [0, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]


def get_amplitude(n):
    if n == 0:
        return PI * (1 / PHI) * np.sqrt(2) * T_SYMBOL
    else:
        return PI * (PHI**n)


# Generate x-axis
x = np.linspace(0, 45, 2000)
individual_waves = []
superposition = np.zeros_like(x)

# Generate 13 wave pairs
for i in range(13):
    amplitude = get_amplitude(i)
    phase_deg = PRIMES[i]
    phase_rad = np.deg2rad(phase_deg)
    phase_offset_rad = np.deg2rad(PHASE_OFFSET)

    wave1 = amplitude * np.sin(x + phase_rad)
    wave2 = amplitude * np.sin(x + phase_rad + phase_offset_rad)

    superposition += wave1 + wave2
    individual_waves.append((wave1, wave2, i))

# Create visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Individual waves
ax1.set_title("CMB Dipole: 13 Wave Pairs (179° Offset)", fontsize=14, fontweight="bold")
for wave1, wave2, idx in individual_waves:
    alpha = 0.3 if idx > 0 else 0.5
    ax1.plot(x, wave1, color="steelblue", alpha=alpha, linewidth=0.8)
    ax1.plot(x, wave2, color="cornflowerblue", alpha=alpha, linewidth=0.8)

ax1.set_xlabel("Position (x)")
ax1.set_ylabel("Amplitude")
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-8, 8)
ax1.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)

# Plot 2: Superposition
ax2.set_title(
    "Coherent Superposition → Observable Dipole", fontsize=14, fontweight="bold"
)
ax2.plot(x, superposition, color="darkred", linewidth=2.5, label="Observable")
ax2.fill_between(x, 0, superposition, alpha=0.2, color="darkred")
ax2.set_xlabel("Position (x)")
ax2.set_ylabel("Amplitude")
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)

plt.tight_layout()
plt.savefig(
    "/home/randy/Desktop/cmb_dipole_wave_interference.png", dpi=300, bbox_inches="tight"
)
print("✅ Chart saved: cmb_dipole_wave_interference.png")
# plt.show()  # Commented out for headless execution
