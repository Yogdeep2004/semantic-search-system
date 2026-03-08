import pickle
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import CLEANED_DOCS_PATH, EMBEDDINGS_PATH, PROCESSED_DATA_DIR
from embeddings.embedder import TextEmbedder

def build_embeddings():
    """
    Loads cleaned documents, generates embeddings, and saves them to disk.
    """
    if not CLEANED_DOCS_PATH.exists():
        print(f"Error: Cleaned documents not found at {CLEANED_DOCS_PATH}. Run dataset_loader.py first.")
        return

    print(f"Loading documents from {CLEANED_DOCS_PATH}...")
    with open(CLEANED_DOCS_PATH, 'rb') as f:
        documents = pickle.load(f)

    print(f"Generating embeddings for {len(documents)} documents...")
    embedder = TextEmbedder()
    embeddings = embedder.encode_documents(documents)

    print(f"Embeddings generated with shape: {embeddings.shape}")
    
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"Saved embeddings to {EMBEDDINGS_PATH}")

if __name__ == "__main__":
    build_embeddings()
