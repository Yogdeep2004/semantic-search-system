import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, normalized_mutual_info_score
import skfuzzy as fuzz
import sys
import pickle
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import (
    EMBEDDINGS_PATH, LABELS_PATH, N_CLUSTERS_MIN, N_CLUSTERS_MAX, 
    CLUSTERING_METRICS_PATH, CLUSTERING_CONFIG_PATH
)

def evaluate_clusters():
    """
    Evaluates fuzzy clustering for different values of k and selects the optimal one.
    """
    if not EMBEDDINGS_PATH.exists() or not LABELS_PATH.exists():
        print(f"Error: Required files not found. Run embedding_builder.py first.")
        return

    embeddings = np.load(EMBEDDINGS_PATH)
    with open(LABELS_PATH, 'rb') as f:
        true_labels = pickle.load(f)

    # Label encoding for NMI
    le = LabelEncoder()
    true_label_indices = le.fit_transform(true_labels)

    results = []
    ks = range(N_CLUSTERS_MIN, N_CLUSTERS_MAX + 1, 5)
    
    print(f"Evaluating clustering for k in {list(ks)}...")
    
    data = embeddings.T
    
    for k in tqdm(ks):
        # We use a smaller maxiter for evaluation to speed up the process
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            data, k, 2.0, error=0.01, maxiter=500, init=None
        )
        
        # Hard clusters for some metrics
        cluster_membership = np.argmax(u, axis=0)
        
        sil = silhouette_score(embeddings, cluster_membership)
        db = davies_bouldin_score(embeddings, cluster_membership)
        nmi = normalized_mutual_info_score(true_label_indices, cluster_membership)
        
        results.append({
            'k': k,
            'fpc': fpc,
            'silhouette': sil,
            'davies_bouldin': db,
            'nmi': nmi
        })

    print("\nCLUSTER EVALUATION RESULTS:")
    print(f"{'k':<5} | {'FPC':<10} | {'Silhouette':<10} | {'DB Index':<10} | {'NMI':<10}")
    print("-" * 55)
    for res in results:
        print(f"{res['k']:<5} | {res['fpc']:<10.4f} | {res['silhouette']:<10.4f} | {res['davies_bouldin']:<10.4f} | {res['nmi']:<10.4f}")

    # Automatic selection logic:
    # We can use a combination of metrics. Here, NMI is strongly indicative as we have labels.
    # However, to be purely data-driven (unsupervised), Silhouette and DB are better.
    # Let's use NMI for this dataset as it's the gold standard for clustering newsgroups.
    best_res = max(results, key=lambda x: x['nmi'])
    best_k = best_res['k']
    
    print(f"\nOptimal cluster count selected based on NMI: k = {best_k}")
    
    # Save results and best k
    with open(CLUSTERING_METRICS_PATH, 'wb') as f:
        pickle.dump(results, f)
    
    with open(CLUSTERING_CONFIG_PATH, 'wb') as f:
        pickle.dump({'optimal_k': best_k}, f)

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    metrics = ['fpc', 'silhouette', 'davies_bouldin', 'nmi']
    titles = ['Fuzzy Partition Coefficient', 'Silhouette Score', 
              'Davies-Bouldin Index', 'Normalized Mutual Information']
    
    for i, metric in enumerate(metrics):
        axes[i].plot(ks, [r[metric] for r in results], marker='o')
        axes[i].set_title(titles[i])
        axes[i].set_xlabel('Number of Clusters (k)')
        axes[i].set_ylabel(metric)
        if metric == 'davies_bouldin':
            axes[i].annotate('Lower is better', xy=(0.5, 0.9), xycoords='axes fraction')
        else:
            axes[i].annotate('Higher is better', xy=(0.5, 0.9), xycoords='axes fraction')
    
    plt.tight_layout()
    plt.savefig('cluster_evaluation_metrics.png')
    print(f"\nMetrics plot saved to cluster_evaluation_metrics.png")
    print(f"Results saved to {CLUSTERING_METRICS_PATH}")

if __name__ == "__main__":
    evaluate_clusters()
