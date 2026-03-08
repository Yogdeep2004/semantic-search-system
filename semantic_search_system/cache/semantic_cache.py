import numpy as np
from typing import List, Dict, Optional, Any
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import CACHE_SIMILARITY_THRESHOLD

class SemanticCache:
    def __init__(self, threshold: float = CACHE_SIMILARITY_THRESHOLD):
        """
        Manual implementation of a cluster-aware semantic cache.
        cache = { cluster_id: [entries] }
        """
        self.cache: Dict[int, List[Dict[str, Any]]] = {}
        self.threshold = threshold
        self.hits = 0
        self.misses = 0

    def lookup(self, query_embedding: np.ndarray, cluster_id: int) -> Optional[Dict]:
        """
        Looks up a result in the cache for the given cluster.
        Returns cached result if cosine similarity > threshold.
        """
        if cluster_id not in self.cache:
            self.misses += 1
            return None

        # Normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)

        for entry in self.cache[cluster_id]:
            cached_emb = entry['query_embedding']
            cached_norm = cached_emb / (np.linalg.norm(cached_emb) + 1e-10)
            
            similarity = np.dot(query_norm, cached_norm)
            
            if similarity >= self.threshold:
                self.hits += 1
                return {
                    "result": entry['result'],
                    "matched_query": entry['query_text'],
                    "similarity_score": float(similarity)
                }
        
        self.misses += 1
        return None

    def add(self, query_text: str, query_embedding: np.ndarray, result: Any, cluster_id: int):
        """
        Adds a new entry to the cache.
        """
        if cluster_id not in self.cache:
            self.cache[cluster_id] = []
        
        self.cache[cluster_id].append({
            "query_text": query_text,
            "query_embedding": query_embedding,
            "result": result
        })

    def clear(self):
        """Flushes the cache."""
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def stats(self):
        try:
            total_entries = sum(len(v) for v in self.cache.values())

            hits = getattr(self, "hits", 0)
            misses = getattr(self, "misses", 0)

            total_requests = hits + misses

            hit_rate = hits / total_requests if total_requests > 0 else 0

            return {
                "total_entries": total_entries,
                "hits": hits,
                "misses": misses,
                "hit_rate": hit_rate
            }

        except Exception as e:
            return {"error": str(e)}
