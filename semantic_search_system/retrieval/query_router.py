import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from embeddings.embedder import TextEmbedder
from clustering.fuzzy_clusterer import FuzzyClusterer

class QueryRouter:
    def __init__(self, embedder: TextEmbedder, clusterer: FuzzyClusterer):
        self.embedder = embedder
        self.clusterer = clusterer

    def route_query(self, query: str):
        """
        Embeds the query and detects the dominant cluster.
        Returns:
            query_embedding: np.ndarray
            dominant_cluster: int
        """
        query_embedding = self.embedder.encode_query(query)
        dominant_cluster = self.clusterer.dominant_cluster(query_embedding)
        return query_embedding, dominant_cluster
