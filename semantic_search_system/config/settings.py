import os
from pathlib import Path

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
EXTRACTED_DATA_DIR = DATA_DIR / "extracted"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Dataset Files
DATASET_ZIP = RAW_DATA_DIR / "twenty_newsgroups_dataset.zip"
DATASET_TAR = EXTRACTED_DATA_DIR / "20_newsgroups.tar.gz"
EXTRACTED_FOLDER = EXTRACTED_DATA_DIR / "20_newsgroups"

# Processed Files
CLEANED_DOCS_PATH = PROCESSED_DATA_DIR / "cleaned_docs.pkl"
LABELS_PATH = PROCESSED_DATA_DIR / "labels.pkl"
EMBEDDINGS_PATH = PROCESSED_DATA_DIR / "document_embeddings.npy"
CLUSTER_MEMBERSHIPS_PATH = PROCESSED_DATA_DIR / "cluster_memberships.npy"
CLUSTER_CENTERS_PATH = PROCESSED_DATA_DIR / "cluster_centers.npy"
CLUSTERING_METRICS_PATH = PROCESSED_DATA_DIR / "clustering_metrics.pkl"
CLUSTERING_CONFIG_PATH = PROCESSED_DATA_DIR / "clustering_config.pkl"

# Embedding Settings
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Clustering Settings
N_CLUSTERS_MIN = 5
N_CLUSTERS_MAX = 30
DEFAULT_K = 20 # Chosen based on ground truth and evaluation

# Search Settings
CACHE_SIMILARITY_THRESHOLD = 0.85
TOP_K_RETRIEVAL = 5
BM25_CANDIDATES = 20

# Create directories if they don't exist
for path in [RAW_DATA_DIR, EXTRACTED_DATA_DIR, PROCESSED_DATA_DIR]:
    path.mkdir(parents=True, exist_ok=True)
