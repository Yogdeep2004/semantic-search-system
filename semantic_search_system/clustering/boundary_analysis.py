import numpy as np
import pickle
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import CLEANED_DOCS_PATH, CLUSTER_MEMBERSHIPS_PATH

def analyze_boundaries(threshold: float = 0.1, top_n: int = 10):
    """
    Identifies documents with high cluster ambiguity (boundary documents).
    Ambiguity is defined as a small difference between the top two membership values.
    """
    if not CLEANED_DOCS_PATH.exists() or not CLUSTER_MEMBERSHIPS_PATH.exists():
        print(f"Error: Required files not found. Run clustering scripts first.")
        return

    print("Loading documents and memberships...")
    with open(CLEANED_DOCS_PATH, 'rb') as f:
        documents = pickle.load(f)
    
    # memberships: (n_samples, n_clusters)
    memberships = np.load(CLUSTER_MEMBERSHIPS_PATH)
    
    # Sort memberships per row in descending order
    sorted_m = np.sort(memberships, axis=1)[:, ::-1]
    
    # Compute difference between top two memberships
    diff = sorted_m[:, 0] - sorted_m[:, 1]
    
    # Find documents where diff is small (ambiguous)
    ambiguous_indices = np.where(diff < threshold)[0]
    
    # Sort those indices by the smallest diff
    sorted_ambiguous = ambiguous_indices[np.argsort(diff[ambiguous_indices])]
    
    print(f"\nBOUNDARY DOCUMENT ANALYSIS (Threshold: {threshold})")
    print(f"Total documents analyzed:   {len(documents)}")
    print(f"Ambiguous documents found:  {len(ambiguous_indices)}")
    print("-" * 60)
    
    for i in range(min(top_n, len(sorted_ambiguous))):
        idx = sorted_ambiguous[i]
        top_two_clusters = np.argsort(memberships[idx])[::-1][:2]
        top_two_vals = sorted_m[idx, :2]
        
        print(f"Rank {i+1} Ambiguity (Difference: {diff[idx]:.4f})")
        print(f"Document Index: {idx}")
        print(f"Primary Cluster:   {top_two_clusters[0]:02d} (Prob: {top_two_vals[0]:.4f})")
        print(f"Secondary Cluster: {top_two_clusters[1]:02d} (Prob: {top_two_vals[1]:.4f})")
        snippet = " ".join(documents[idx].split()[:40]) + "..."
        print(f"Snippet: {snippet}")
        print("-" * 60)

if __name__ == "__main__":
    analyze_boundaries()
