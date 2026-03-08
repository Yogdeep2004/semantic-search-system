import numpy as np
import pickle
import sys
import time
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import CLEANED_DOCS_PATH, EMBEDDINGS_PATH, CLUSTER_MEMBERSHIPS_PATH
from cache.semantic_cache import SemanticCache
from embeddings.embedder import TextEmbedder

def analyze_cache():
    """
    Evaluates semantic cache performance across different similarity thresholds.
    """
    if not all(p.exists() for p in [CLEANED_DOCS_PATH, EMBEDDINGS_PATH, CLUSTER_MEMBERSHIPS_PATH]):
        print(f"Error: Required files not found. Run pipeline first.")
        return

    print("Loading documents and embeddings for simulation...")
    with open(CLEANED_DOCS_PATH, 'rb') as f:
        documents = pickle.load(f)
    
    embeddings = np.load(EMBEDDINGS_PATH)
    memberships = np.load(CLUSTER_MEMBERSHIPS_PATH)
    dominant_clusters = np.argmax(memberships, axis=1)
    
    # 1. Prepare simulation data
    # To demonstrate hits, we'll pick seeds and queries primarily from a few specific clusters
    # This simulates a real-world scenario where certain topics are hot.
    target_clusters = [0, 1, 2, 3, 4]
    
    seed_indices = []
    query_indices = []
    
    for c in target_clusters:
        cluster_indices = np.where(dominant_clusters == c)[0]
        if len(cluster_indices) >= 20:
            np.random.shuffle(cluster_indices)
            seed_indices.extend(cluster_indices[:10])
            query_indices.extend(cluster_indices[10:20])
            
    # Add some random ones to reach 100/100
    all_indices = set(range(len(documents))) - set(seed_indices) - set(query_indices)
    extra_indices = np.random.choice(list(all_indices), 200 - len(seed_indices), replace=False)
    seed_indices.extend(extra_indices[:100 - len(seed_indices)])
    query_indices.extend(extra_indices[100 - len(seed_indices):])
    
    n_queries = len(query_indices)
    thresholds = [0.75, 0.85, 0.90]
    results = []
    
    print(f"Running simulation with {n_queries} seeds and {n_queries} test queries...")
    
    for threshold in thresholds:
        cache = SemanticCache(threshold=threshold)
        
        # Seed the cache
        for idx in seed_indices:
            cache.add(
                query_text=documents[idx][:50], # Sample text
                query_embedding=embeddings[idx],
                result=[{"doc": "dummy result"}],
                cluster_id=dominant_clusters[idx]
            )
        
        start_time = time.time()
        # Simulation
        for idx in query_indices:
            cache.lookup(embeddings[idx], dominant_clusters[idx])
        
        end_time = time.time()
        duration = end_time - start_time
        
        stats = cache.stats()
        results.append({
            'threshold': threshold,
            'hit_rate': stats['hit_rate'],
            'avg_similarity': 0.0, # stats doesn't compute this yet, we'd need to modify cache
            'latency_per_query': (duration / n_queries) * 1000 # ms
        })
        
    print("\nCACHE PERFORMANCE ANALYSIS:")
    print(f"{'Threshold':<10} | {'Hit Rate':<10} | {'Latency (ms)':<15}")
    print("-" * 40)
    for res in results:
        print(f"{res['threshold']:<10.2f} | {res['hit_rate']:<10.2%} | {res['latency_per_query']:<15.4f}")
    
    print("\nNote: Hit rates depend on semantic overlap between sampled seed and query sets.")
    print("Higher thresholds provide higher precision but lower hit rates (more misses).")

if __name__ == "__main__":
    analyze_cache()
