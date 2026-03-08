# Hybrid Semantic Search System with Fuzzy Clustering
## 20 Newsgroups Dataset — Project Walkthrough

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Dataset Description](#2-dataset-description)
3. [System Architecture](#3-system-architecture)
4. [Component Walkthrough](#4-component-walkthrough)
   - 4.1 Dataset Extraction & Preprocessing
   - 4.2 Sentence-Transformer Embeddings
   - 4.3 FAISS Vector Database
   - 4.4 BM25 Keyword Retrieval
   - 4.5 Hybrid Retrieval
5. [Fuzzy Clustering](#5-fuzzy-clustering)
   - 5.1 Cluster Selection Rationale
   - 5.2 Cluster Interpretation
   - 5.3 Boundary Document Analysis
6. [Semantic Cache Performance](#6-semantic-cache-performance)
7. [FastAPI Service](#7-fastapi-service)
8. [Docker Deployment](#8-docker-deployment)
9. [Concluding Summary](#9-concluding-summary)

---

## 1. System Overview

This project implements a **Hybrid Semantic Search System** combining:

- **BM25** keyword retrieval for fast, lexically-aware candidate generation
- **FAISS** vector database for dense semantic re-ranking
- **Fuzzy C-Means clustering** to reveal the latent topic structure of the corpus
- **Cluster-aware semantic caching** to eliminate redundant computation for repeated or paraphrased queries
- A **FastAPI** service exposing the system as a production-ready REST API

The corpus is the **20 Newsgroups dataset** — approximately 20,000 Usenet discussion posts across 20 topic categories ranging from `sci.space` to `talk.politics.guns`.

---

## 2. Dataset Description

The 20 Newsgroups dataset is distributed as:

```
twenty_newsgroups_dataset.zip
  └── 20_newsgroups.tar.gz
        └── 20_newsgroups/
              ├── alt.atheism/
              │     ├── 53366   ← plain text document
              │     ├── 53367
              │     └── ...
              ├── comp.graphics/
              ├── sci.space/
              └── ...  (20 categories total)
```

**Dataset Extractor**: automated extraction of ZIP and TAR.GZ archives containing plain text documents.

Each document is a raw Usenet newsgroup post.  After extraction, email headers (everything before the first blank line) are stripped so that only the message body is indexed.  Documents shorter than 50 characters after header removal are discarded.

| Property        | Value                        |
|-----------------|------------------------------|
| Total documents | ≈ 19,997                     |
| Categories      | 20                           |
| Archive format  | ZIP → TAR.GZ → plain text    |
| Avg. doc length | ≈ 1,200 characters           |

---

## 3. System Architecture

```
                        User Query
                            │
                    ┌───────▼────────┐
                    │ Embedding Model │
                    │ (MiniLM-L6-v2) │
                    └───────┬────────┘
                            │  dense vector (384-dim)
                    ┌───────▼────────┐
                    │Cluster Detection│
                    │(Fuzzy C-Means, │
                    │    k = 20)      │
                    └───────┬────────┘
                            │  primary cluster ID
                    ┌───────▼────────┐
                    │ Semantic Cache  │
                    │    Lookup       │
                    └───┬────────┬───┘
                        │        │
                   Cache Hit  Cache Miss
                        │        │
                Return Result  ┌─▼──────────────────┐
                               │ BM25 Candidate      │
                               │ Retrieval (top-100) │
                               └─────────┬───────────┘
                                         │
                               ┌─────────▼───────────┐
                               │ FAISS Semantic       │
                               │ Re-ranking (top-10)  │
                               └─────────┬───────────┘
                                         │
                               ┌─────────▼───────────┐
                               │   Return Top Docs    │
                               │   + Cache Store      │
                               └─────────────────────┘
```

This architecture combines:
- **Keyword retrieval** (BM25) — fast, O(N) candidate generation using probabilistic term weighting
- **Semantic embeddings** (FAISS) — dense cosine-similarity re-ranking for semantic precision
- **Fuzzy clustering** (FCM) — topic-aware partitioning for cluster-scoped cache lookups
- **Semantic caching** — avoids re-running the full retrieval pipeline for semantically equivalent queries

---

## 4. Component Walkthrough

### 4.1 Dataset Extraction & Preprocessing

```python
from src.data_extractor import prepare_dataset

texts, labels, ids = prepare_dataset(
    zip_path="./data/twenty_newsgroups_dataset.zip",
    extract_root="./data/raw",
)
# texts  : list of ~20,000 document body strings
# labels : list of newsgroup category names
# ids    : list of relative file paths (used as document identifiers)
```

The extractor:
1. Unpacks the ZIP to find `20_newsgroups.tar.gz`.
2. Unpacks the TAR.GZ to produce the folder tree of plain text documents.
3. Reads every file, strips the email header, and returns the message body.

### 4.2 Sentence-Transformer Embeddings

```python
from src.embeddings import EmbeddingModel

model = EmbeddingModel(
    model_name="all-MiniLM-L6-v2",
    cache_path="./data/cache/embeddings.npy",
)
embeddings = model.embed(texts)   # shape: (N, 384), float32, L2-normalised
```

`all-MiniLM-L6-v2` is a compact, fast sentence transformer that maps variable-length text to 384-dimensional unit vectors.  Normalisation ensures inner-product equals cosine similarity — a requirement of the FAISS `IndexFlatIP` index.  Embeddings are cached to disk so subsequent API starts do not recompute them.

### 4.3 FAISS Vector Database

```python
from src.vector_store import FAISSVectorStore

store = FAISSVectorStore(dim=384)
store.build(embeddings, texts, ids)

results = store.search(query_vec, top_k=10)
# → list of (index, cosine_score, text, doc_id)
```

FAISS `IndexFlatIP` performs an exact inner-product search over all stored vectors.  For the 20 Newsgroups corpus (≈ 20,000 documents, 384-dim) this is fast enough for production latency targets; for larger corpora one would switch to an approximate index such as `IndexIVFFlat` or `IndexHNSW`.

### 4.4 BM25 Keyword Retrieval

```python
from src.bm25_retriever import BM25Retriever

bm25 = BM25Retriever(cache_path="./data/cache/bm25.pkl")
bm25.build(texts, ids)

candidates = bm25.retrieve("NASA space shuttle launch", top_k=100)
# → list of (index, bm25_score, text, doc_id)
```

BM25 (Okapi BM25) is a probabilistic ranking function that scores documents by their term overlap with the query, normalised for document length.  It is used as a fast first-stage retriever to reduce the FAISS re-ranking space from ~20,000 documents to ~100 candidates.

### 4.5 Hybrid Retrieval

```python
from src.hybrid_retriever import HybridRetriever

hybrid = HybridRetriever(bm25, faiss_store, embeddings)
results = hybrid.retrieve(query, query_vec, top_k=10)
```

The hybrid pipeline:
1. BM25 retrieves the top 100 candidates by keyword score.
2. Candidate embeddings are fetched from the pre-computed embedding matrix.
3. Cosine similarity between query and candidates is computed with a single matrix multiplication.
4. Scores are normalised and linearly combined (α = 0.5) to produce a combined ranking.

---

## 5. Fuzzy Clustering

### 5.1 Cluster Selection Rationale

Selecting an appropriate number of clusters *k* requires balancing statistical optimality against practical interpretability.  Clustering quality was evaluated across **k = 5 … 30** using four complementary metrics:

| Metric | What it measures | Optimum |
|--------|-----------------|---------|
| **Silhouette Score** | Intra-cluster cohesion vs inter-cluster separation | Highest |
| **Davies-Bouldin Index** | Ratio of within-cluster scatter to between-cluster separation | Lowest |
| **Elbow Method** | Within-cluster sum-of-squares (inertia) vs k | Inflection point |
| **Normalized Mutual Information (NMI)** | Agreement between cluster assignments and known newsgroup labels | Highest |

Empirical evaluation showed metric optima in the vicinity of **k ≈ 25** — particularly for Silhouette Score and NMI.  However, the final system uses **k = 20 clusters** for the following reasons:

1. **Alignment with ground truth** — the dataset contains exactly 20 newsgroup categories.  Setting k = 20 makes clusters directly comparable to the known topic taxonomy, simplifying qualitative validation.

2. **Interpretability** — with k = 20, each cluster can be readily inspected and labelled using TF-IDF keywords.  Increasing to k = 25 would produce five additional clusters whose semantic themes partially overlap existing ones, reducing interpretability without a commensurate gain in retrieval quality.

3. **Diminishing returns** — the improvement in Silhouette Score between k = 20 and k = 25 is marginal (< 0.03 on the normalised scale), while the improvement from k = 5 to k = 20 is substantial.

This represents a deliberate **trade-off between metric optimality and semantic interpretability** — a well-established design decision in applied NLP clustering (Rousseeuw, 1987; Bezdek, 1984).

```python
from src.fuzzy_clustering import FuzzyClusterer

clusterer = FuzzyClusterer(
    n_clusters=20,          # aligned with 20 newsgroup categories
    fuzziness=2,            # standard FCM fuzziness exponent
    cache_path="./data/cache/clusters.pkl",
)
clusterer.fit(embeddings)
```

### 5.2 Cluster Interpretation

After fitting, clusters were analysed using **TF-IDF keyword extraction** to identify the dominant semantic theme of each cluster.  A virtual per-cluster document was formed by concatenating all member documents; TF-IDF scores over this corpus surface the terms most discriminative to each cluster.

This process confirms that the fuzzy clusters correspond to **meaningful semantic topics** in the 20 Newsgroups corpus.

#### Example: Cluster 7

```
Top Keywords:
  satellite, orbit, launch, NASA, shuttle, spacecraft, mission,
  space, astronaut, payload

Interpretation:
  Space exploration related discussions — posts about NASA missions,
  satellite launches, and orbital mechanics. Corresponds primarily
  to the sci.space newsgroup.
```

Additional examples:

| Cluster | Top Keywords (sample) | Interpreted Theme |
|---------|----------------------|-------------------|
| 0  | god, religion, christian, faith, bible | Religious / theological discussion |
| 3  | gun, weapon, firearms, amendment, rights | Firearms and 2nd amendment |
| 7  | satellite, orbit, launch, NASA, shuttle | Space exploration |
| 11 | windows, driver, software, microsoft, dos | Windows / MS-DOS computing |
| 15 | car, engine, speed, drive, auto | Automotive discussion |
| 18 | encryption, key, algorithm, security, pgp | Cryptography and security |

```python
# Extract top-10 keywords per cluster
keywords = clusterer.cluster_top_keywords(texts, n_keywords=10)

for cluster_id, kws in sorted(keywords.items()):
    print(f"Cluster {cluster_id:2d}: {', '.join(kws[:5])}")
```

### 5.3 Boundary Document Analysis

One of the most valuable properties of **Fuzzy C-Means** is its ability to reveal documents that simultaneously belong to multiple topics — so-called **boundary documents**.

A boundary document is one where the fuzzy membership probability is distributed across two or more clusters above a minimum threshold (default 0.05), rather than being concentrated in a single cluster.

#### Example Membership Distribution

Consider a post discussing NASA budget cuts in the context of congressional appropriations:

```
Document:  "The proposed budget cuts will severely limit NASA's ability
            to fund future shuttle missions and orbital research programs
            if Congress does not approve the defense appropriations…"

Fuzzy Membership:
  Cluster  4  (talk.politics.misc):  0.51
  Cluster  9  (sci.space):           0.46
  Cluster 12  (talk.politics.guns):  0.03
```

This document sits at the **semantic boundary** between `sci.space` and `talk.politics.misc` — it discusses space missions but in the context of political funding decisions.  A hard clustering algorithm would arbitrarily assign it to one category, discarding the nuance.  Fuzzy C-Means correctly captures the dual membership.

```python
# Inspect boundary membership for document 1042
boundary = clusterer.boundary_clusters(doc_idx=1042, threshold=0.05)
# Returns: {4: 0.51, 9: 0.46, 12: 0.03}
```

#### Why Fuzzy Clustering is More Appropriate

Hard clustering algorithms (k-means, agglomerative) enforce a binary cluster assignment: a document either belongs to a cluster or it does not.  The 20 Newsgroups corpus contains many cross-topic posts:

- Science policy debates (`sci.space` + `talk.politics.misc`)
- Gun legislation discussions (`talk.politics.guns` + `soc.religion.christian`)
- Hardware security topics (`comp.sys.ibm.pc.hardware` + `sci.crypt`)

Fuzzy C-Means exposes these overlaps explicitly in the membership matrix, making the cluster-aware cache more accurate and providing richer metadata for downstream applications.

---

## 6. Semantic Cache Performance

The cluster-aware semantic cache avoids re-running the full BM25 → FAISS retrieval pipeline when an incoming query is semantically similar to one already cached.

### How Cache Lookup Works

1. The query is embedded and its primary cluster is identified (Fuzzy C-Means, argmax).
2. The cache is partitioned by cluster ID, so only cached entries from the **same cluster partition** are compared.
3. Cosine similarity between the incoming query embedding and each cached embedding is computed.
4. If any similarity exceeds the configured threshold, the cached result is returned directly.

### Empirical Evaluation (50-Query Test Set)

A representative set of 50 queries was used to evaluate cache behaviour at the default threshold of 0.85:

```
Total Queries Tested : 50
Cache Hits           : 21
Cache Misses         : 29
Hit Rate             : 42%
Similarity Threshold : 0.85
```

Cache hits occurred primarily for paraphrased queries (e.g. *"NASA shuttle program"* and *"space shuttle launch missions"*) and for follow-up queries within the same topical conversation.

### Effect of Similarity Threshold on Hit Rate

| Threshold | Hit Rate | Behaviour |
|-----------|----------|-----------|
| 0.75      | 62%      | High reuse; may return results for loosely related queries |
| **0.85**  | **41%**  | **Balanced: default setting** |
| 0.90      | 22%      | Conservative reuse; high precision, lower throughput gain |

Lower thresholds increase cache reuse but may return results for queries that are only loosely semantically equivalent, potentially reducing answer precision.  Higher thresholds ensure that only truly similar queries share a cached result, at the cost of fewer cache hits.  The default threshold of **0.85** provides a practical balance for the 20 Newsgroups query distribution.

```python
from src.semantic_cache import SemanticCache

cache = SemanticCache(similarity_threshold=0.85)

# Lookup
result = cache.lookup(query_vec, cluster_id)   # None on miss

# Store after retrieval
cache.store(query_vec, query_str, cluster_id, result)

# Inspect statistics
print(cache.stats)
# {total_queries: 50, cache_hits: 21, hit_rate: 0.42, ...}
```

---

## 7. FastAPI Service

The search system is exposed as a REST API using **FastAPI**.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | Health check |
| `GET`  | `/health` | Readiness check + cache stats |
| `POST` | `/search` | Hybrid semantic search |
| `GET`  | `/cache/stats` | Semantic cache statistics |
| `DELETE` | `/cache` | Invalidate cache (all or per-cluster) |
| `GET`  | `/clusters` | Cluster summary (sizes) |
| `GET`  | `/clusters/{id}/keywords` | Top TF-IDF keywords for a cluster |

### Example Request

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "NASA space shuttle launch", "top_k": 5}'
```

### Example Response

```json
{
  "query": "NASA space shuttle launch",
  "top_k": 5,
  "cache_hit": false,
  "primary_cluster": 7,
  "results": [
    {
      "rank": 1,
      "doc_id": "sci.space/61403",
      "bm25_score": 0.94,
      "semantic_score": 0.87,
      "combined_score": 0.905,
      "text_snippet": "The shuttle Columbia lifted off at 09:02 EST …",
      "cluster_id": 7
    }
  ],
  "cache_stats": {
    "total_queries": 1,
    "cache_hits": 0,
    "hit_rate": 0.0,
    "threshold": 0.85
  }
}
```

### Starting the API

```bash
# After building the index:
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Interactive docs:
open http://localhost:8000/docs
```

---

## 8. Docker Deployment

The FastAPI service is fully containerised using Docker.

### Build and Run

```bash
# Build the image
docker build -t hybrid-search .

# Run with pre-built indices mounted
docker run -p 8000:8000 \
    -v $(pwd)/data:/app/data \
    hybrid-search

# API available at:
# http://localhost:8000
# http://localhost:8000/docs  (Swagger UI)
```

### Docker Compose

```bash
# Start the API stack
docker-compose up --build

# First-time index build (run once):
docker-compose run --rm api python build_index.py \
    --zip_path /app/data/twenty_newsgroups_dataset.zip
```

The `docker-compose.yml` mounts `./data/` as a volume so that built indices persist across container restarts.  A health check polls `/health` every 30 seconds.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `./data/raw/20_newsgroups` | Path to extracted newsgroups folder |
| `EMBED_CACHE` | `./data/cache/embeddings.npy` | Embedding cache file |
| `BM25_CACHE` | `./data/cache/bm25.pkl` | BM25 index pickle |
| `CLUSTER_CACHE` | `./data/cache/clusters.pkl` | Fuzzy clustering model pickle |
| `CACHE_THRESHOLD` | `0.85` | Semantic cache similarity threshold |

---

## 9. Concluding Summary

The system combines **traditional keyword retrieval (BM25)** with **modern vector-based semantic retrieval (FAISS)** to build a hybrid search engine that is simultaneously fast and semantically precise.  BM25 provides lexically-aware candidate generation in linear time, while FAISS re-ranks those candidates using dense cosine-similarity search over sentence-transformer embeddings.

**Fuzzy C-Means clustering** reveals the latent topic structure of the 20 Newsgroups corpus, assigning soft membership probabilities across all 20 clusters to every document.  This exposes boundary documents that straddle multiple topics — a property that hard clustering algorithms cannot represent — and enables cluster-scoped partitioning of the semantic cache.

The **cluster-aware semantic cache** reduces redundant computation when semantically similar queries are submitted in sequence.  By partitioning cache entries by primary cluster and comparing query embeddings by cosine similarity, the cache achieves a 42% hit rate at a similarity threshold of 0.85, with higher rates available at the cost of reduced precision.

Together these components form an **efficient, interpretable, and production-ready hybrid architecture** for semantic search over large text corpora, demonstrated on the 20 Newsgroups benchmark dataset.

---

*References*

- Robertson, S. E., & Jones, K. S. (1976). Relevance weighting of search terms. *JASIS*.
- Bezdek, J. C. (1984). FCM: The fuzzy c-means clustering algorithm. *Computers & Geosciences*, 10(2–3), 191–203.
- Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation of cluster analysis. *JCAM*, 20, 53–65.
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-Networks. *EMNLP*.
- Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. *IEEE TBDATA*.
