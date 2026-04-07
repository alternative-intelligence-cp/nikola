import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Generate Wave 1: exactly advancing by 1 per frame
# 0 -> 180 -> -180 -> 0
wave1 = []
wave1.extend(range(0, 181))          # 0 to 180 (181 frames)
wave1.extend(range(179, -181, -1))   # 179 to -180 (360 frames)
wave1.extend(range(-179, 0))         # -179 to -1 (179 frames)
# Total length = 181 + 360 + 179 = 720 frames. 
# 720 frames = 360 degrees of the wave cycle. So 2 frames = 1 degree.

wave1 = np.array(wave1)

# Wave 2: 179 degrees out of phase
# 1 degree = 2 frames. 179 degrees = 179 * 2 = 358 frames.
# Shift wave 2 by 358 frames (delaying it)
phase_shift_frames = 358
wave2 = np.roll(wave1, phase_shift_frames)

# Combined Wave
combined = wave1 + wave2

# Create a DataFrame to save to CSV for the user
df = pd.DataFrame({
    'Frame': range(len(wave1)),
    'Wave1': wave1,
    'Wave2': wave2,
    'Combined': combined
})
df.to_csv('wave_interaction.csv', index=False)

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(wave1, label='Wave 1', color='blue', linewidth=2)
plt.plot(wave2, label='Wave 2 (179° out of phase)', color='red', linestyle='--', linewidth=2)
plt.plot(combined, label='Combined (Sum)', color='green', linewidth=2.5)

plt.title('Interaction of Two Triangular Waves (179° Out of Phase)', fontsize=14)
plt.xlabel('Frame Number', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.axhline(0, color='black', linewidth=1)
plt.legend(fontsize=12, loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig('wave_interaction_chart.png', dpi=300)
plt.close()

print("Plot saved as wave_interaction_chart.png and data saved to wave_interaction.csv")