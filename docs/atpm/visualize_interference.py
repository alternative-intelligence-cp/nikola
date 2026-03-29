import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def generate_wave_pair(steps=5000, dim=9, phase_offset=179):
    """
    Generates TWO waves on 9D torus:
    - Wave 1: starting at 0°
    - Wave 2: starting at phase_offset degrees
    
    Shows actual interference pattern
    """
    phi = (1 + np.sqrt(5)) / 2
    frequencies = np.array([phi**i for i in range(dim)])
    
    t = np.linspace(0, 100, steps)
    
    # Wave 1: standard trajectory
    traj1 = np.outer(t, frequencies) % (2 * np.pi)
    
    # Wave 2: offset by phase_offset degrees (converted to radians)
    phase_rad = np.deg2rad(phase_offset)
    traj2 = (np.outer(t, frequencies) + phase_rad) % (2 * np.pi)
    
    # Calculate interference amplitude at each point
    # Sum of two waves: A1 + A2 where phases differ by phase_offset
    # Using cos for simplicity: cos(θ1) + cos(θ2)
    amplitude = np.zeros(steps)
    for i in range(steps):
        wave1_sum = np.sum(np.cos(traj1[i, :]))
        wave2_sum = np.sum(np.cos(traj2[i, :]))
        amplitude[i] = wave1_sum + wave2_sum
    
    # Normalize amplitude to [0, 1] for coloring
    amplitude_norm = (amplitude - amplitude.min()) / (amplitude.max() - amplitude.min())
    
    # Embed wave 1 in R^18
    embedded = []
    for d in range(dim):
        embedded.append(np.sin(traj1[:, d]))
        embedded.append(np.cos(traj1[:, d]))
    
    return np.array(embedded).T, amplitude_norm, amplitude

# Generate with 179° offset
print("Generating wave pair with 179° phase offset...")
X, amplitude_norm, amplitude_raw = generate_wave_pair(steps=2000, dim=9, phase_offset=179)

print("Computing PCA...")
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

print("Computing t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

# Create figure with 3 subplots
fig = plt.figure(figsize=(20, 6))

# Plot 1: PCA colored by interference amplitude
ax1 = fig.add_subplot(131, projection='3d')
sc1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
                  c=amplitude_norm, cmap='RdBu_r', s=3)
ax1.set_title("Global Structure (PCA)\nColor = Interference Amplitude\n(Red=Destructive, Blue=Constructive)")
ax1.set_xlabel("PC1")
ax1.set_ylabel("PC2")
ax1.set_zlabel("PC3")
plt.colorbar(sc1, ax=ax1, label='Amplitude')

# Plot 2: t-SNE colored by interference amplitude
ax2 = fig.add_subplot(132)
sc2 = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                  c=amplitude_norm, cmap='RdBu_r', s=8, alpha=0.7)
ax2.set_title("Recurrence Map (t-SNE)\nColor = Interference Amplitude\n(Red=Voids, Blue=Peaks)")
ax2.set_xlabel("Dimension 1")
ax2.set_ylabel("Dimension 2")
plt.colorbar(sc2, ax=ax2, label='Amplitude')

# Plot 3: Amplitude distribution histogram
ax3 = fig.add_subplot(133)
ax3.hist(amplitude_raw, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax3.axvline(amplitude_raw.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {amplitude_raw.mean():.2f}')
ax3.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
ax3.set_xlabel('Interference Amplitude')
ax3.set_ylabel('Frequency')
ax3.set_title('Amplitude Distribution\n(Negative = Destructive, Positive = Constructive)')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.suptitle(r"Two Waves on $T^9$ with 179° Phase Offset", fontsize=18, y=1.02)
plt.tight_layout()

output_file = '/home/randy/Workspace/SYSTEM/Downloads/atpm/interference_pattern_179deg.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to: {output_file}")

# Print statistics
print(f"\nInterference Statistics:")
print(f"  Mean amplitude: {amplitude_raw.mean():.4f}")
print(f"  Std deviation: {amplitude_raw.std():.4f}")
print(f"  Min (most destructive): {amplitude_raw.min():.4f}")
print(f"  Max (most constructive): {amplitude_raw.max():.4f}")
print(f"  Amplitude range: {amplitude_raw.max() - amplitude_raw.min():.4f}")

# Calculate percentage of destructive vs constructive
destructive = np.sum(amplitude_raw < amplitude_raw.mean())
constructive = np.sum(amplitude_raw >= amplitude_raw.mean())
print(f"\nDistribution:")
print(f"  Below mean (voids): {destructive/len(amplitude_raw)*100:.1f}%")
print(f"  Above mean (peaks): {constructive/len(amplitude_raw)*100:.1f}%")

plt.show()
