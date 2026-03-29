#!/usr/bin/env python3
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt

# --- Constants Definitions  ---
PHI = (1 + np.sqrt(5)) / 2  # golden ratio
PI = np.pi
T_SYMBOL = 32 / 27
REF_PHASE = 0  # ※ treated as 0

# --- Domain Setup ---
# Total period 45 units
x = np.linspace(0, 45, 5000)

# --- Wave Parameter definitions from Sources 2 and 3 ---
# Format: (Amplitude Expression, Base Phase Offset in Degrees)
# Amplitudes derived literally from mathematical expressions in sources.
wave_params = [
    (PI * (1 / PHI) * np.sqrt(2) * T_SYMBOL, 0),  # p0 [cite: 2]
    (PI * PHI**1, 3),  # p1 [cite: 2]
    (PI * PHI**2, 5),  # p2 [cite: 2]
    (PI * PHI**3, 7),  # p3 [cite: 2]
    (PI * PHI**4, 11),  # p4 [cite: 2]
    (PI * PHI**5, 13),  # p5 [cite: 2]
    (PI * PHI**6, 17),  # p6 [cite: 2]
    (PI * PHI**7, 19),  # p7 [cite: 2]
    (PI * PHI**8, 23),  # p8 [cite: 2]
    (PI * PHI**9, 29),  # p9 [cite: 2]
    (PI * PHI**10, 31),  # p10 [cite: 2]
    (PI * PHI**11, 37),  # p11 [cite: 2]
    (PI * PHI**12, 41),  # p12
]

# --- Generating and Plotting Waves ---
plt.figure(figsize=(15, 8))
superposition_wave = np.zeros_like(x)

# Iterate through p0-p12 parameters
for i, (amp, base_phase_deg) in enumerate(wave_params):
    # Convert base phase to radians, adding reference phase
    theta1 = np.deg2rad(REF_PHASE + base_phase_deg)
    # Define second component with 179 degree offset
    theta2 = np.deg2rad(REF_PHASE + base_phase_deg + 179)

    # Assuming normalized frequency of 1 as none was provided
    wave1 = amp * np.sin(x + theta1)
    wave2 = amp * np.sin(x + theta2)

    # Plot individual pair components (thin, semi-transparent)
    # Showing all 13 pairs
    plt.plot(x, wave1, color="gray", alpha=0.3, linewidth=0.8)
    plt.plot(x, wave2, color="gray", alpha=0.3, linewidth=0.8)

    # Add to superposition
    superposition_wave += wave1 + wave2

# Plot the superposition wave (distinct style)
# Easily distinguishable superposition wave
plt.plot(
    x,
    superposition_wave,
    "b-",
    linewidth=2.5,
    label="Superposition Wave (Sum of all 13 pairs)",
)

# --- Graph Formatting ---
# -4 min <-> 4 max for amplitude
plt.ylim(-4, 4)
plt.xlim(0, 45)

# Add major grid lines at every integer x-value
ax = plt.gca()
ax.set_xticks(np.arange(0, 46, 1))  # Major ticks at every integer
ax.set_xticks(np.arange(0, 45.5, 0.5), minor=True)  # Minor ticks at half-integers
ax.set_yticks(np.arange(-4, 5, 1))  # Major ticks at integer amplitudes

# Grid styling - prominent vertical lines at integers
plt.grid(
    True,
    which="major",
    axis="x",
    linestyle="-",
    alpha=0.6,
    linewidth=0.8,
    color="black",
)
plt.grid(
    True, which="minor", axis="x", linestyle=":", alpha=0.3, linewidth=0.5, color="gray"
)
plt.grid(
    True,
    which="major",
    axis="y",
    linestyle="--",
    alpha=0.4,
    linewidth=0.6,
    color="gray",
)

plt.title(
    "Superposition of 13 Wave Pairs based on Source Parameters\n(View restricted to Amplitude range [-4, 4])"
)
plt.xlabel("Units (Domain 0-45)")
plt.ylabel("Amplitude")
plt.legend(loc="upper right")

plt.tight_layout()
# Save to file instead of showing (which can hang)
output_file = "/home/randy/Downloads/ai/wave_superposition.png"
plt.savefig(output_file, dpi=150, bbox_inches="tight")
print(f"Graph saved to: {output_file}")
# plt.show()  # Commented out - use savefig instead
