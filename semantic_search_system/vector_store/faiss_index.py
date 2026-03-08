import faiss
import numpy as np
import sys
from pathlib import Path
from typing import Tuple

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import EMBEDDING_DIM

class FaissVectorIndex:
    def __init__(self, dimension: int = EMBEDDING_DIM):
        """
        Initializes a FAISS FlatIP index for cosine similarity.
        Note: For IP index, embeddings should be normalized for true cosine similarity.
        """
        self.dimension = dimension
        # IndexFlatIP is used for inner product
        self.index = faiss.IndexFlatIP(dimension)

    def build_index(self, embeddings: np.ndarray):
        """
        Normalizes embeddings and adds them to the FAISS index.
        """
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.dimension}, got {embeddings.shape[1]}")
        
        # Normalize for cosine similarity using numpy to avoid FAISS segfault
        normalized_embeddings = np.ascontiguousarray(embeddings, dtype=np.float32).copy()
        norms = np.linalg.norm(normalized_embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1e-10
        normalized_embeddings = normalized_embeddings / norms
        
        self.index.add(normalized_embeddings)
        print(f"Added {embeddings.shape[0]} embeddings to FAISS index.")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Searches the index for the top_k most similar embeddings.
        Returns:
            distances: cosine similarity scores
            indices: indices of the matched documents
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        # Normalize query for cosine similarity using numpy
        query_norm = np.ascontiguousarray(query_embedding, dtype=np.float32).copy()
        norms = np.linalg.norm(query_norm, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        query_norm = query_norm / norms
        
        distances, indices = self.index.search(query_norm, top_k)
        return distances[0], indices[0]
