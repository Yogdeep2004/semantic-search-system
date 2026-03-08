"""
hybrid_retriever.py
-------------------
Hybrid retrieval: BM25 candidate generation + FAISS semantic re-ranking.

Pipeline
--------
1. BM25 retrieves the top-N candidate documents by keyword overlap.
2. Each candidate embedding is compared to the query embedding using
   cosine similarity (FAISS inner-product on normalised vectors).
3. Candidates are re-ranked by semantic similarity and the top-k are returned.

This two-stage design balances speed (BM25 is O(N) but fast) with semantic
precision (FAISS re-ranks based on meaning, not just term overlap).
"""

import logging
import numpy as np
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Combines BM25 first-stage retrieval with FAISS semantic re-ranking.

    Parameters
    ----------
    bm25_retriever  : fitted BM25Retriever instance
    vector_store    : fitted FAISSVectorStore instance
    embeddings      : (N, D) float32 array of all document embeddings
    bm25_candidates : number of BM25 candidates to re-rank
    """

    def __init__(
        self,
        bm25_retriever,
        vector_store,
        embeddings: np.ndarray,
        bm25_candidates: int = 100,
    ):
        self.bm25 = bm25_retriever
        self.faiss = vector_store
        self.embeddings = embeddings
        self.bm25_candidates = bm25_candidates

    def retrieve(
        self,
        query: str,
        query_vec: np.ndarray,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the *top_k* most relevant documents for *query*.

        Steps
        -----
        1. BM25 → top-N candidate indices
        2. Semantic re-ranking of candidates using query_vec
        3. Return top-k results with metadata

        Parameters
        ----------
        query     : raw query string (used for BM25)
        query_vec : (1, D) float32 query embedding (used for re-ranking)
        top_k     : number of results to return

        Returns
        -------
        List of dicts with keys: rank, doc_id, bm25_score, semantic_score,
        combined_score, text_snippet.
        """
        # Stage 1 — BM25 candidates
        bm25_results = self.bm25.retrieve(query, top_k=self.bm25_candidates)
        if not bm25_results:
            logger.warning("BM25 returned no results.")
            return []

        candidate_indices = [r[0] for r in bm25_results]
        bm25_scores = {r[0]: r[1] for r in bm25_results}

        # Stage 2 — Semantic re-ranking
        q_vec = query_vec.flatten().astype(np.float32)
        candidate_vecs = self.embeddings[candidate_indices]   # (N_cand, D)
        semantic_scores = candidate_vecs @ q_vec               # dot product (= cosine for normalised)

        # Normalise BM25 scores to [0, 1] for combination
        bm25_arr = np.array([bm25_scores[i] for i in candidate_indices])
        bm25_max = bm25_arr.max() if bm25_arr.max() > 0 else 1.0
        bm25_norm = bm25_arr / bm25_max

        # Combined score (equal weight)
        alpha = 0.5
        combined = alpha * semantic_scores + (1 - alpha) * bm25_norm

        # Sort by combined score
        order = np.argsort(combined)[::-1][:top_k]

        results = []
        for rank, oi in enumerate(order, start=1):
            doc_idx = candidate_indices[oi]
            text = self.bm25._texts[doc_idx]
            doc_id = self.bm25._ids[doc_idx]
            results.append({
                "rank": rank,
                "doc_id": doc_id,
                "bm25_score": float(bm25_norm[oi]),
                "semantic_score": float(semantic_scores[oi]),
                "combined_score": float(combined[oi]),
                "text_snippet": text[:400],
            })

        return results
