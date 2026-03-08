# Hybrid Semantic Search System with Fuzzy Clustering

A research-grade semantic retrieval system combining keyword matching, dense vector search, and fuzzy clustering for high precision and scalability.

## Architecture & Scientific Justification

This system implements a multi-stage retrieval pipeline designed to balance flexibility, speed, and semantic accuracy:

1.  **Semantic Routing**: Every query is first embedded and routed to a dominant cluster discovered during unsupervised clustering. This allows for **Cluster-Restricted Search**, reducing the search space and speeding up retrieval in large-scale environments.
2.  **Hybrid Retrieval**:
    *   **BM25 (Stage 1)**: Robust keyword matching retrieves high-recall candidates.
    *   **FAISS (Stage 2)**: Semantic re-ranking using Sentence Transformer embeddings (`all-MiniLM-L6-v2`) ensures high-precision relevance.
3.  **Cluster-Aware Semantic Cache**: A novel caching strategy where results are partitioned by cluster ID. This reduces collision probability and enables faster lookups compared to global caches.
4.  **Data-Driven Topic Discovery**: Instead of fixed categories, the system uses fuzzy clustering metrics (NMI, Silhouette) to automatically discover the natural topic structure of the corpus.

## Core Features
- **Unsupervised Topic Discovery**: Automatic selection of optimal cluster count ($k$) using NMI and Silhouette metrics.
- **Explainable Clustering**: Top keywords and representative documents per cluster for human interpretability.
- **Ambiguity Analysis**: Detection of "boundary documents" with high semantic overlap between topics.
- **Performance Evaluation**: Built-in simulators for cache hit-rate and retrieval latency.
- **Visualization**: 2D projection of high-dimensional embeddings using t-SNE.

## Project Structure
- `api/`: FastAPI service and query orchestration.
- `cache/`: Semantic caching and performance analysis.
- `clustering/`: Evaluation ($k$ selection), interpretation, boundary analysis, and visualization.
- `embeddings/`: Model wrappers and embedding generation.
- `retrieval/`: Query routing and BM25+FAISS hybrid logic.
- `preprocessing/`: Specialized logic for **ZIP and TAR.GZ** document extraction and cleaning.

## Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Full Pipeline Execution
```bash
# 1. Preprocess and Embed
python3 preprocessing/dataset_extractor.py
python3 preprocessing/dataset_loader.py
python3 embeddings/embedding_builder.py

# 2. Scientific Evaluation & Clustering
python3 clustering/cluster_evaluation.py   # Finds optimal k
python3 clustering/run_clustering.py       # Fits clusters
python3 clustering/cluster_interpretation.py
python3 clustering/embedding_visualization.py
```

### 3. Start the API
```bash
./run_server.sh
```

## Advanced Analysis
- **Boundary Analysis**: `python3 clustering/boundary_analysis.py` - Find documents at the intersection of topics.
- **Cache Simulation**: `python3 cache/cache_analysis.py` - Test hit rates at thresholds 0.75, 0.85, 0.90.
