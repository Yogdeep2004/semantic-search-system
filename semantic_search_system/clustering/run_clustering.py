import numpy as np
import sys
import pickle
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import (
    EMBEDDINGS_PATH, CLUSTER_MEMBERSHIPS_PATH, CLUSTER_CENTERS_PATH, 
    CLUSTERING_CONFIG_PATH, DEFAULT_K
)
from clustering.fuzzy_clusterer import FuzzyClusterer

def run_final_clustering():
    """
    Fits the FuzzyClusterer on all embeddings using the optimal k (if available)
    and saves membership probabilities and centers.
    """
    if not EMBEDDINGS_PATH.exists():
        print(f"Error: Embeddings not found at {EMBEDDINGS_PATH}. Run embedding_builder.py first.")
        return

    # Load optimal k if evaluation was run
    n_clusters = DEFAULT_K
    if CLUSTERING_CONFIG_PATH.exists():
        with open(CLUSTERING_CONFIG_PATH, 'rb') as f:
            config = pickle.load(f)
            n_clusters = config.get('optimal_k', DEFAULT_K)
            print(f"Using optimal k = {n_clusters} from {CLUSTERING_CONFIG_PATH}")
    else:
        print(f"Evaluation config not found. Using default k = {DEFAULT_K}")

    embeddings = np.load(EMBEDDINGS_PATH)
    print(f"Fitting Fuzzy C-Means with k={n_clusters} on {embeddings.shape[0]} embeddings...")
    
    clusterer = FuzzyClusterer(n_clusters=n_clusters)
    memberships = clusterer.fit(embeddings)
    
    np.save(CLUSTER_MEMBERSHIPS_PATH, memberships)
    np.save(CLUSTER_CENTERS_PATH, clusterer.cntr)
    print(f"Saved cluster memberships to {CLUSTER_MEMBERSHIPS_PATH}")
    print(f"Saved cluster centers to {CLUSTER_CENTERS_PATH}")

if __name__ == "__main__":
    run_final_clustering()
