from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import EMBEDDING_MODEL_NAME

class TextEmbedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        """
        Initializes the SentenceTransformer model.
        """
        self.model = SentenceTransformer(model_name)

    def encode_documents(self, docs: List[str]) -> np.ndarray:
        """
        Generates embeddings for a list of documents.
        """
        # show_progress_bar=True helps visualize progress for large batches
        return self.model.encode(docs, show_progress_bar=True, convert_to_numpy=True)

    def encode_query(self, query: str) -> np.ndarray:
        """
        Generates an embedding for a single query.
        """
        return self.model.encode([query], convert_to_numpy=True)[0]
