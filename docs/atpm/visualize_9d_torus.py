import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def generate_golden_trajectory(steps=5000, dim=9):
    """
    Generates a trajectory on a 9-dimensional torus using irrational frequencies
    based on the Golden Ratio (phi). This ensures ergodicity.
    """
    phi = (1 + np.sqrt(5)) / 2
    # Frequencies: 1, phi, phi^2,..., phi^8
    frequencies = np.array([phi**i for i in range(dim)])
    
    # Time vector
    t = np.linspace(0, 100, steps)
    
    # Trajectory in phase space
    trajectory = np.outer(t, frequencies) % (2 * np.pi)
    
    # Embed in R^18 (sin/cos for each dimension)
    embedded_data = []
    for d in range(dim):
        embedded_data.append(np.sin(trajectory[:, d]))
        embedded_data.append(np.cos(trajectory[:, d]))
    
    return np.array(embedded_data).T, t

# Generate the data
print("Generating 9D Toroidal Trajectory...")
X, time_steps = generate_golden_trajectory(steps=2000, dim=9)

# Dimensionality Reduction
# 1. PCA (Global Structure)
print("Computing PCA...")
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# 2. t-SNE (Local Interactions/Brushes)
print("Computing t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

# Plotting
fig = plt.figure(figsize=(16, 7))

# Plot 1: PCA - Shows the cyclical nature
ax1 = fig.add_subplot(121, projection='3d')
sc1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=time_steps, cmap='plasma', s=2)
ax1.set_title("Global Structure (PCA Projection)\nColor = Time Evolution")
ax1.set_xlabel("PC1")
ax1.set_ylabel("PC2")
ax1.set_zlabel("PC3")

# Plot 2: t-SNE - Shows recurrence and 'brushes'
# Points close together in t-SNE space are close in the 9D torus
ax2 = fig.add_subplot(122)
sc2 = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=time_steps, cmap='plasma', s=5, alpha=0.7)
ax2.set_title("Interaction Map (t-SNE)\nClusters indicate 'Brushes' or Recurrence")
ax2.set_xlabel("Dimension 1")
ax2.set_ylabel("Dimension 2")

plt.colorbar(sc2, label='Time')
plt.suptitle(r"Ergodic Flow on $T^9$ with Golden Ratio Frequencies", fontsize=16)
plt.tight_layout()

# Save the figure
output_file = '/home/randy/Workspace/SYSTEM/Downloads/atpm/9d_torus_visualization.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to: {output_file}")

plt.show()
