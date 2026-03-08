import numpy as np
import skfuzzy as fuzz
import sys
from pathlib import Path
from typing import Tuple

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import DEFAULT_K

class FuzzyClusterer:
    def __init__(self, n_clusters: int = DEFAULT_K, m: float = 2.0, error: float = 0.005, maxiter: int = 1000):
        """
        Initializes the Fuzzy C-Means clusterer.
        Args:
            n_clusters: Number of clusters (c)
            m: Fuzziness parameter (typically 2.0)
            error: Stopping criterion
            maxiter: Maximum number of iterations
        """
        self.n_clusters = n_clusters
        self.m = m
        self.error = error
        self.maxiter = maxiter
        self.cntr = None # Cluster centers
        self.u = None    # Membership matrix

    def fit(self, embeddings: np.ndarray):
        """
        Fits Fuzzy C-Means to the provided embeddings.
        Input embeddings should be (n_samples, n_features).
        skfuzzy expects (n_features, n_samples).
        """
        data = embeddings.T
        self.cntr, self.u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            data, self.n_clusters, self.m, error=self.error, maxiter=self.maxiter, init=None
        )
        print(f"Fuzzy C-Means fitted with {self.n_clusters} clusters. FPC: {fpc:.4f}")
        return self.u.T # Return memberships as (n_samples, n_clusters)

    def predict_membership(self, embedding: np.ndarray) -> np.ndarray:
        """
        Predicts cluster memberships for a new embedding.
        """
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        data = embedding.T
        u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
            data, self.cntr, self.m, error=self.error, maxiter=self.maxiter
        )
        return u.T[0] # Return memberships for the single sample

    def dominant_cluster(self, embedding: np.ndarray) -> int:
        """
        Returns the ID of the cluster with maximum membership probability.
        """
        memberships = self.predict_membership(embedding)
        return int(np.argmax(memberships))
