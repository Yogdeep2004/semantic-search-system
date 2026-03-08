import numpy as np
from typing import List, Dict, Optional
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from keyword_index.bm25_index import BM25Indexer
from vector_store.faiss_index import FaissVectorIndex
from embeddings.embedder import TextEmbedder

class HybridRetriever:
    def __init__(self, bm25_index: BM25Indexer, vector_index: FaissVectorIndex, 
                 embedder: TextEmbedder, documents: List[str], embeddings: np.ndarray,
                 dominant_clusters: np.ndarray = None):
        self.bm25_index = bm25_index
        self.vector_index = vector_index
        self.embedder = embedder
        self.documents = documents
        self.embeddings = embeddings
        self.dominant_clusters = dominant_clusters

    def retrieve(self, query: str, query_embedding: np.ndarray, 
                 bm25_k: int = 20, final_k: int = 5,
                 use_cluster_restriction: bool = False,
                 target_cluster: Optional[int] = None) -> List[Dict]:
        """
        Retrieval workflow:
        1. (Optional) Filter candidates to a specific cluster.
        2. BM25 keyword retrieval from the (filtered) candidates.
        3. FAISS semantic re-ranking of results.
        """
        
        # Determine the set of indices to search over
        if use_cluster_restriction and target_cluster is not None and self.dominant_clusters is not None:
            # Find indices of documents belonging to the target cluster
            search_indices = np.where(self.dominant_clusters == target_cluster)[0]
            if len(search_indices) == 0:
                # Fallback to global if cluster is empty (should not happen)
                search_indices = np.arange(len(self.documents))
        else:
            search_indices = np.arange(len(self.documents))

        # Stage 1: BM25 Search
        # Note: Our BM25Indexer is built on the whole corpus. 
        # For simplicity and efficiency, we search globally and then filter for cluster membership,
        # OR we search only the subset. 
        # rank-bm25 doesn't easily support subset search without re-indexing.
        # So we'll get more candidates and filter them.
        
        bm25_indices, _ = self.bm25_index.search(query, top_k=len(self.documents)) # Get all scores
        
        # Filter bm25_indices by search_indices
        search_indices_set = set(search_indices)
        filtered_bm25_indices = [idx for idx in bm25_indices if idx in search_indices_set][:bm25_k]
        
        if not filtered_bm25_indices:
            # Fallback if no matching docs in cluster (unlikely but possible with rare keywords)
            # Re-run with global search_indices
            filtered_bm25_indices = bm25_indices[:bm25_k]

        # Stage 2: Semantic Re-ranking
        candidate_embeddings = self.embeddings[filtered_bm25_indices]
        
        # Normalize candidates
        candidate_norms = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
        candidate_norms[candidate_norms == 0] = 1e-10
        norm_candidates = candidate_embeddings / candidate_norms
        
        # Normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        
        # Similarities
        similarities = np.dot(norm_candidates, query_norm)
        
        # Sort and take top final_k
        top_rel_indices = np.argsort(similarities)[::-1][:final_k]
        
        results = []
        for idx in top_rel_indices:
            global_idx = filtered_bm25_indices[idx]
            results.append({
                "document": self.documents[global_idx],
                "score": float(similarities[idx]),
                "index": int(global_idx),
                "cluster": int(self.dominant_clusters[global_idx]) if self.dominant_clusters is not None else None
            })
            
        return results
