import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import EMBEDDINGS_PATH, CLUSTER_MEMBERSHIPS_PATH

def visualize_embeddings(n_samples: int = 2000):
    """
    Visualizes document embeddings in 2D using t-SNE, colored by cluster.
    """
    if not EMBEDDINGS_PATH.exists() or not CLUSTER_MEMBERSHIPS_PATH.exists():
        print(f"Error: Required files not found. Run clustering scripts first.")
        return

    print(f"Loading embeddings and cluster data (sampling {n_samples} points)...")
    embeddings = np.load(EMBEDDINGS_PATH)
    memberships = np.load(CLUSTER_MEMBERSHIPS_PATH)
    
    # Randomly sample to speed up t-SNE
    indices = np.random.choice(len(embeddings), min(n_samples, len(embeddings)), replace=False)
    sampled_emb = embeddings[indices]
    sampled_clusters = np.argmax(memberships[indices], axis=1)
    
    print("Running t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    embeddings_2d = tsne.fit_transform(sampled_emb)
    
    # Plotting
    plt.figure(figsize=(12, 10))
    # Use a diverse colormap for clusters
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=sampled_clusters, cmap='tab20', alpha=0.6, s=15)
    
    plt.colorbar(scatter, label='Cluster ID')
    plt.title(f'2D Visualization of Newsgroups Embeddings (t-SNE)\nColored by Dominant Cluster')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    output_file = 'embedding_visualization.png'
    plt.savefig(output_file, dpi=300)
    print(f"Visualization saved to {output_file}")

if __name__ == "__main__":
    visualize_embeddings()
