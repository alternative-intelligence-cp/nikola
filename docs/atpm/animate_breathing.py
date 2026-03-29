import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.decomposition import PCA

# NOTE ABOUT INTERPRETATION
# -------------------------
# This script computes two trajectories (traj1, traj2) separated by a fixed phase offset
# and uses both to compute an interference amplitude.
#
# However, the PCA visualization is built from an embedding of traj1 ONLY.
# That means any “two halves merging / orbiting” look in the 3D plot is a projection/trail
# effect of a single high-dimensional trajectory, not necessarily two separate points
# physically approaching.
#
# For an animation that explicitly shows BOTH members of the pair and tracks true vs
# projected distance, see: animate_pair_breathing.py

def generate_wave_pair_evolution(steps=500, dim=9, phase_offset=179):
    """
    Generates evolution of two interfering waves on 9D torus.
    Returns amplitudes at each time step for animation.
    """
    phi = (1 + np.sqrt(5)) / 2
    frequencies = np.array([phi**i for i in range(dim)])
    
    t = np.linspace(0, 100, steps)
    
    # Wave 1 trajectory
    traj1 = np.outer(t, frequencies) % (2 * np.pi)
    
    # Wave 2 trajectory with phase offset
    phase_rad = np.deg2rad(phase_offset)
    traj2 = (np.outer(t, frequencies) + phase_rad) % (2 * np.pi)
    
    # Calculate interference at each time step
    amplitude_evolution = []
    for i in range(steps):
        wave1 = np.cos(traj1[i, :])
        wave2 = np.cos(traj2[i, :])
        # Total amplitude changes over time as waves evolve
        amp = np.sum(wave1 + wave2)
        amplitude_evolution.append(amp)
    
    # Embed for PCA
    embedded = []
    for d in range(dim):
        embedded.append(np.sin(traj1[:, d]))
        embedded.append(np.cos(traj1[:, d]))
    
    return np.array(embedded).T, np.array(amplitude_evolution), t

print("Generating wave evolution data...")
X, amplitudes, times = generate_wave_pair_evolution(steps=500, dim=9, phase_offset=179)

print("Computing PCA projection...")
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# Normalize amplitudes for color mapping
amp_norm = (amplitudes - amplitudes.min()) / (amplitudes.max() - amplitudes.min())

print("Creating animation...")

# Set up the figure and 3D axis
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)

# Initial empty plot
sc = ax.scatter([], [], [], c=[], cmap='RdBu_r', s=5, alpha=0.6)
line, = ax2.plot([], [], 'b-', linewidth=2)
ax2.set_xlim(0, 100)
ax2.set_ylim(amplitudes.min() * 1.1, amplitudes.max() * 1.1)
ax2.set_xlabel('Time', fontsize=12)
ax2.set_ylabel('Total Amplitude', fontsize=12)
ax2.set_title('Interference "Breathing" Over Time', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)

# Text for current amplitude
amp_text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes, fontsize=11,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.set_xlabel('PC1', fontsize=10)
ax.set_ylabel('PC2', fontsize=10)
ax.set_zlabel('PC3', fontsize=10)
ax.set_title('9D Toroidal Wave Evolution\n179° Phase Offset', fontsize=14)

# Color bar
cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
cbar.set_label('Amplitude', fontsize=10)

def init():
    sc._facecolors3d = np.array([])
    sc._edgecolors3d = np.array([])
    line.set_data([], [])
    amp_text.set_text('')
    return sc, line, amp_text

def update(frame):
    # Show accumulated trajectory up to current frame
    current_points = X_pca[:frame+1]
    current_amps = amp_norm[:frame+1]
    
    # Update 3D scatter plot
    ax.clear()
    sc = ax.scatter(current_points[:, 0], current_points[:, 1], current_points[:, 2],
                    c=current_amps, cmap='RdBu_r', s=5, alpha=0.6, vmin=0, vmax=1)
    
    # Highlight current position
    if frame > 0:
        ax.scatter([X_pca[frame, 0]], [X_pca[frame, 1]], [X_pca[frame, 2]],
                  c='yellow', s=100, edgecolors='black', linewidths=2, marker='o')
    
    ax.set_xlabel('PC1', fontsize=10)
    ax.set_ylabel('PC2', fontsize=10)
    ax.set_zlabel('PC3', fontsize=10)
    ax.set_title(f'9D Toroidal Wave Evolution (t={times[frame]:.1f})\n179° Phase Offset', fontsize=14)
    
    # Update amplitude time series
    line.set_data(times[:frame+1], amplitudes[:frame+1])
    
    # Update amplitude text
    amp_text.set_text(f'Current Amplitude: {amplitudes[frame]:.4f}\nFrame: {frame+1}/500')
    
    # Add marker on time series
    ax2.plot([times[frame]], [amplitudes[frame]], 'ro', markersize=8)
    
    return sc, line, amp_text

# Create animation
print("Rendering animation frames (this will take a minute)...")
anim = FuncAnimation(fig, update, init_func=init, frames=500, interval=50, blit=False)

# Save as GIF
output_file = '/home/randy/Workspace/SYSTEM/Downloads/atpm/wave_breathing_animation.gif'
writer = PillowWriter(fps=20)
anim.save(output_file, writer=writer)

print(f"\nAnimation saved to: {output_file}")
print(f"Total frames: 500")
print(f"Duration: 25 seconds at 20 fps")
print("\nThe animation shows:")
print("  - Left: 3D trajectory through phase space")
print("  - Right: Amplitude oscillation over time")
print("  - Yellow dot: Current position")
print("  - Color: Red (destructive) → Blue (constructive)")
