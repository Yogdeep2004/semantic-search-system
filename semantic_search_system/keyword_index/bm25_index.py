from rank_bm25 import BM25Okapi
from typing import List, Tuple
import numpy as np

class BM25Indexer:
    def __init__(self):
        self.bm25 = None
        self.corpus_size = 0

    def build_index(self, documents: List[str]):
        """
        Tokenizes documents and builds the BM25 index.
        """
        tokenized_corpus = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.corpus_size = len(documents)
        print(f"BM25 index built with {self.corpus_size} documents.")

    def search(self, query: str, top_k: int = 20) -> Tuple[List[int], np.ndarray]:
        """
        Searches the BM25 index for the top_k most relevant documents.
        Returns:
            indices: indices of the matched documents
            scores: BM25 scores
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        # get top_k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[top_indices]
        return top_indices.tolist(), top_scores
