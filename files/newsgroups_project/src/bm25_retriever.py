"""
bm25_retriever.py
-----------------
BM25 (Best Match 25) keyword retrieval module.

Uses the `rank_bm25` library to build a probabilistic keyword index
over the document corpus.  BM25 is used as the first-stage (candidate)
retriever in the hybrid pipeline.
"""

import re
import logging
import pickle
import os
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Simple stop-word list for tokenisation
_STOPWORDS = {
    "the", "a", "an", "is", "it", "in", "on", "at", "of", "and", "or",
    "to", "for", "with", "that", "this", "was", "are", "be", "by", "as",
    "from", "but", "not", "have", "had", "has", "he", "she", "they",
    "we", "you", "i", "do", "did", "will", "would", "could", "should",
    "been", "were", "its", "if", "so", "up", "out", "about", "than",
}


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


class BM25Retriever:
    """
    Wraps BM25Okapi from rank_bm25.

    Parameters
    ----------
    cache_path : optional pickle path for persisting the index
    """

    def __init__(self, cache_path: str = None):
        self.cache_path = cache_path
        self._bm25 = None
        self._texts: List[str] = []
        self._ids: List[str] = []

        if cache_path and os.path.exists(cache_path):
            self._load(cache_path)

    # ------------------------------------------------------------------

    def build(self, texts: List[str], ids: List[str]) -> None:
        from rank_bm25 import BM25Okapi
        logger.info(f"Building BM25 index over {len(texts)} documents …")
        tokenized = [_tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(tokenized)
        self._texts = list(texts)
        self._ids = list(ids)
        logger.info("BM25 index ready.")

        if self.cache_path:
            self._save(self.cache_path)

    def retrieve(self, query: str, top_k: int = 50) -> List[Tuple[int, float, str, str]]:
        """
        Return the *top_k* documents most relevant to *query* by BM25 score.

        Returns list of (index, score, text, doc_id).
        """
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built. Call build() first.")

        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            (i, float(scores[i]), self._texts[i], self._ids[i])
            for i in top_indices
        ]

    # ------------------------------------------------------------------

    def _save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self._bm25, "texts": self._texts, "ids": self._ids}, f)
        logger.info(f"BM25 index saved to: {path}")

    def _load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._texts = data["texts"]
        self._ids = data["ids"]
        logger.info(f"BM25 index loaded from: {path} ({len(self._texts)} docs)")
