"""
vector_store.py
---------------
FAISS-backed vector database for fast approximate nearest-neighbour (ANN)
retrieval of document embeddings.
"""

import os
import logging
import numpy as np
from typing import List, Tuple

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    Wraps a FAISS flat inner-product index (suitable for normalised vectors,
    where inner-product equals cosine similarity).

    Parameters
    ----------
    dim          : embedding dimension
    index_path   : optional path to load/save the FAISS index
    """

    def __init__(self, dim: int, index_path: str = None):
        import faiss  # imported here so the class can be imported without faiss
        self.dim = dim
        self.index_path = index_path
        self._index = None
        self._texts: List[str] = []
        self._ids: List[str] = []

        if index_path and os.path.exists(index_path):
            self._load(index_path)

    # ------------------------------------------------------------------

    def build(self, embeddings: np.ndarray, texts: List[str], ids: List[str]) -> None:
        """
        Build the index from a (N, D) float32 array of *embeddings*.

        Also stores the raw *texts* and *ids* for retrieval.
        """
        import faiss
        assert embeddings.shape[1] == self.dim, (
            f"Embedding dim mismatch: expected {self.dim}, got {embeddings.shape[1]}"
        )
        self._index = faiss.IndexFlatIP(self.dim)   # inner-product (cosine on normalised vecs)
        self._index.add(embeddings)
        self._texts = list(texts)
        self._ids = list(ids)
        logger.info(f"FAISS index built with {self._index.ntotal} vectors (dim={self.dim}).")

        if self.index_path:
            self._save(self.index_path)

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 10,
    ) -> List[Tuple[int, float, str, str]]:
        """
        Search for the *top_k* nearest neighbours of *query_vec*.

        Returns a list of (index, score, text, doc_id) tuples sorted by
        descending similarity.
        """
        if self._index is None:
            raise RuntimeError("Index has not been built. Call build() first.")

        query_vec = query_vec.reshape(1, -1).astype(np.float32)
        scores, indices = self._index.search(query_vec, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append((int(idx), float(score), self._texts[idx], self._ids[idx]))
        return results

    # ------------------------------------------------------------------

    def _save(self, path: str) -> None:
        import faiss
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self._index, path)
        logger.info(f"FAISS index saved to: {path}")

    def _load(self, path: str) -> None:
        import faiss
        self._index = faiss.read_index(path)
        logger.info(f"FAISS index loaded from: {path} ({self._index.ntotal} vectors)")

    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return self._index.ntotal if self._index else 0
