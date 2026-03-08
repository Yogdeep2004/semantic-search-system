"""
embeddings.py
-------------
Sentence-Transformer embedding module.

Wraps the `sentence-transformers` library to produce dense vector
representations of text documents. Embeddings are cached to disk to
avoid recomputation across runs.
"""

import os
import logging
import numpy as np
from typing import List, Optional

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Wraps a SentenceTransformer model to embed lists of texts.

    Parameters
    ----------
    model_name   : HuggingFace model identifier
    cache_path   : if provided, load/save embeddings from this .npy file
    batch_size   : batch size used during encoding
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        cache_path: Optional[str] = None,
        batch_size: int = 64,
    ):
        self.model_name = model_name
        self.cache_path = cache_path
        self.batch_size = batch_size
        self._model = None  # lazy load

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Encode *texts* to a (N, D) float32 numpy array.

        If *cache_path* is set and the file exists the cached embeddings are
        returned directly.  Otherwise embeddings are computed and, when
        *cache_path* is set, saved to disk.
        """
        if self.cache_path and os.path.exists(self.cache_path):
            logger.info(f"Loading cached embeddings from: {self.cache_path}")
            return np.load(self.cache_path)

        self._load_model()
        logger.info(f"Encoding {len(texts)} documents …")
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            np.save(self.cache_path, embeddings)
            logger.info(f"Embeddings saved to: {self.cache_path}")

        return embeddings.astype(np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        """Encode a single query string → (1, D) float32 array."""
        self._load_model()
        vec = self._model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec.astype(np.float32)
