"""
build_index.py
--------------
One-time index build pipeline.

Run this script once to:
  1. Extract the dataset (ZIP → TAR.GZ → plain text documents)
  2. Compute sentence-transformer embeddings
  3. Build the FAISS vector index
  4. Build the BM25 keyword index
  5. Fit the Fuzzy C-Means clustering model

All artefacts are written to ./data/cache/ and re-used on subsequent
API starts, so this step only needs to be repeated when the corpus changes.

Usage
-----
  python build_index.py --zip_path ./data/twenty_newsgroups_dataset.zip
"""

import argparse
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(__file__))

from src.data_extractor import prepare_dataset, load_newsgroups_documents
from src.embeddings import EmbeddingModel
from src.vector_store import FAISSVectorStore
from src.bm25_retriever import BM25Retriever
from src.fuzzy_clustering import FuzzyClusterer

CACHE_DIR = "./data/cache"


def main(zip_path: str, extract_root: str = "./data/raw"):
    os.makedirs(CACHE_DIR, exist_ok=True)

    # ---------------------------------------------------------------
    # Step 1 — Dataset extraction (or bootstrap via sklearn if no zip)
    # ---------------------------------------------------------------
    data_dir = os.path.join(extract_root, "20_newsgroups")
    if not os.path.exists(data_dir):
        if os.path.isfile(zip_path):
            logger.info("Extracting dataset …")
            texts, labels, ids = prepare_dataset(zip_path, extract_root=extract_root)
        else:
            logger.info("No dataset zip found; bootstrapping via sklearn …")
            from bootstrap_data import bootstrap_20_newsgroups
            bootstrap_20_newsgroups(extract_root=extract_root)
            texts, labels, ids = load_newsgroups_documents(data_dir)
    else:
        logger.info("Dataset already extracted, loading documents …")
        texts, labels, ids = load_newsgroups_documents(data_dir)

    logger.info(f"Corpus size: {len(texts)} documents")

    # ---------------------------------------------------------------
    # Step 2 — Embeddings
    # ---------------------------------------------------------------
    embed_cache = os.path.join(CACHE_DIR, "embeddings.npy")
    embed_model = EmbeddingModel(cache_path=embed_cache)
    embeddings = embed_model.embed(texts)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    # ---------------------------------------------------------------
    # Step 3 — FAISS vector store
    # ---------------------------------------------------------------
    dim = embeddings.shape[1]
    faiss_store = FAISSVectorStore(dim=dim)
    faiss_store.build(embeddings, texts, ids)
    logger.info(f"FAISS index: {faiss_store.size} vectors")

    # ---------------------------------------------------------------
    # Step 4 — BM25 index
    # ---------------------------------------------------------------
    bm25_cache = os.path.join(CACHE_DIR, "bm25.pkl")
    bm25 = BM25Retriever(cache_path=bm25_cache)
    bm25.build(texts, ids)

    # ---------------------------------------------------------------
    # Step 5 — Fuzzy clustering
    # ---------------------------------------------------------------
    cluster_cache = os.path.join(CACHE_DIR, "clusters.pkl")
    clusterer = FuzzyClusterer(n_clusters=20, cache_path=cluster_cache)
    clusterer.fit(embeddings)

    # Print cluster keyword summary
    logger.info("Extracting cluster keywords …")
    keywords = clusterer.cluster_top_keywords(texts, n_keywords=10)
    logger.info("\n=== Cluster Interpretation ===")
    for cluster_id, kws in sorted(keywords.items()):
        logger.info(f"  Cluster {cluster_id:2d}: {', '.join(kws[:5])}")

    logger.info("Index build complete. The API is ready to start.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build all search indices.")
    parser.add_argument(
        "--zip_path",
        default="./data/twenty_newsgroups_dataset.zip",
        help="Path to the twenty_newsgroups_dataset.zip file",
    )
    parser.add_argument(
        "--extract_root",
        default="./data/raw",
        help="Directory to extract dataset archives into",
    )
    args = parser.parse_args()
    main(args.zip_path, args.extract_root)
