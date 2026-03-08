"""
api/main.py
-----------
FastAPI service for the Hybrid Semantic Search System.

System Architecture
-------------------
                        User Query
                            │
                    Embedding Model
                    (all-MiniLM-L6-v2)
                            │
                   Cluster Detection
                  (Fuzzy C-Means, k=20)
                            │
               Semantic Cache Lookup
               ┌─────────────────────┐
          Cache Hit               Cache Miss
              │                       │
       Return Result         BM25 Candidate Retrieval
                             (rank_bm25, top-100)
                                       │
                          FAISS Semantic Re-ranking
                          (cosine similarity, top-10)
                                       │
                             Return Top Documents

The architecture combines:
  • Keyword retrieval   (BM25) for fast candidate generation
  • Semantic embeddings (FAISS) for precise re-ranking
  • Fuzzy clustering    (FCM)  for topic-aware cache partitioning
  • Semantic caching    for eliminating redundant computation on
                        repeated or paraphrased queries

Together these components form an efficient hybrid semantic search
system suitable for production deployment over large text corpora.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# --- project imports -------------------------------------------------------
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.embeddings import EmbeddingModel
from src.vector_store import FAISSVectorStore
from src.bm25_retriever import BM25Retriever
from src.fuzzy_clustering import FuzzyClusterer
from src.hybrid_retriever import HybridRetriever
from src.semantic_cache import SemanticCache

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global system state (loaded once at startup)
# ---------------------------------------------------------------------------

_state: Dict[str, Any] = {}

DATA_DIR = os.getenv("DATA_DIR", "./data/raw/20_newsgroups")
EMBED_CACHE = os.getenv("EMBED_CACHE", "./data/cache/embeddings.npy")
BM25_CACHE = os.getenv("BM25_CACHE", "./data/cache/bm25.pkl")
CLUSTER_CACHE = os.getenv("CLUSTER_CACHE", "./data/cache/clusters.pkl")
CACHE_THRESHOLD = float(os.getenv("CACHE_THRESHOLD", "0.85"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models and indices on startup."""
    logger.info("=== System startup: loading models and indices ===")

    # Check if pre-built indices are available
    if os.path.exists(BM25_CACHE) and os.path.exists(EMBED_CACHE) and os.path.exists(CLUSTER_CACHE):
        logger.info("Loading pre-built indices from cache …")
        from src.data_extractor import load_newsgroups_documents

        texts, labels, ids = load_newsgroups_documents(DATA_DIR)

        embed_model = EmbeddingModel(cache_path=EMBED_CACHE)
        embeddings = embed_model.embed(texts)

        bm25 = BM25Retriever(cache_path=BM25_CACHE)
        clusterer = FuzzyClusterer(cache_path=CLUSTER_CACHE)

        faiss_store = FAISSVectorStore(dim=embeddings.shape[1])
        faiss_store.build(embeddings, texts, ids)

        hybrid = HybridRetriever(bm25, faiss_store, embeddings)
        cache = SemanticCache(similarity_threshold=CACHE_THRESHOLD)

        _state.update({
            "embed_model": embed_model,
            "embeddings": embeddings,
            "bm25": bm25,
            "faiss": faiss_store,
            "clusterer": clusterer,
            "hybrid": hybrid,
            "cache": cache,
            "texts": texts,
            "labels": labels,
            "ids": ids,
        })
        logger.info("System ready.")
    else:
        logger.warning(
            "Pre-built indices not found. Run the build pipeline first.\n"
            "  python -m src.build_index\n"
            "Serving limited functionality until indices are built."
        )
        _state["ready"] = False

    yield
    logger.info("=== System shutdown ===")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Hybrid Semantic Search API",
    description=(
        "Hybrid BM25 + FAISS semantic search with fuzzy clustering "
        "and cluster-aware semantic caching over the 20 Newsgroups corpus."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Search query string")
    top_k: int = Field(10, ge=1, le=50, description="Number of results to return")


class SearchResult(BaseModel):
    rank: int
    doc_id: str
    bm25_score: float
    semantic_score: float
    combined_score: float
    text_snippet: str
    cluster_id: int


class SearchResponse(BaseModel):
    query: str
    top_k: int
    cache_hit: bool
    primary_cluster: int
    results: List[SearchResult]
    cache_stats: Dict[str, Any]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", summary="Health check")
def root():
    return {"status": "ok", "service": "Hybrid Semantic Search API", "version": "1.0.0"}


@app.get("/health", summary="Detailed health / readiness check")
def health():
    ready = "hybrid" in _state
    return {
        "ready": ready,
        "indices_loaded": ready,
        "cache_stats": _state["cache"].stats if ready else {},
    }


@app.post("/search", response_model=SearchResponse, summary="Hybrid semantic search")
def search(req: SearchRequest):
    """
    Perform hybrid BM25 + semantic search with fuzzy cluster-aware caching.

    Query flow
    ----------
    1. Embed the query with SentenceTransformer.
    2. Detect the primary fuzzy cluster.
    3. Check the semantic cache (same cluster partition).
    4. On cache miss: BM25 candidate retrieval → FAISS semantic re-ranking.
    5. Store result in cache and return.
    """
    if "hybrid" not in _state:
        raise HTTPException(503, "System indices not yet built. Run build_index first.")

    embed_model: EmbeddingModel = _state["embed_model"]
    clusterer: FuzzyClusterer = _state["clusterer"]
    hybrid: HybridRetriever = _state["hybrid"]
    cache: SemanticCache = _state["cache"]

    # Step 1 — embed query
    query_vec = embed_model.embed_single(req.query)

    # Step 2 — detect primary cluster
    cluster_id = clusterer.primary_cluster(query_vec)

    # Step 3 — cache lookup
    cached = cache.lookup(query_vec, cluster_id)
    if cached is not None:
        cached["cache_hit"] = True
        cached["cache_stats"] = cache.stats
        return cached

    # Step 4 — hybrid retrieval
    raw_results = hybrid.retrieve(req.query, query_vec, top_k=req.top_k)

    results_out = [
        SearchResult(cluster_id=cluster_id, **r) for r in raw_results
    ]

    response = SearchResponse(
        query=req.query,
        top_k=req.top_k,
        cache_hit=False,
        primary_cluster=cluster_id,
        results=results_out,
        cache_stats=cache.stats,
    )

    # Step 5 — cache store
    cache.store(query_vec, req.query, cluster_id, response.model_dump())

    return response


@app.get("/cache/stats", summary="Semantic cache statistics")
def cache_stats():
    if "cache" not in _state:
        raise HTTPException(503, "System not ready.")
    return _state["cache"].stats


@app.delete("/cache", summary="Invalidate semantic cache")
def clear_cache(cluster_id: Optional[int] = Query(None, description="Cluster to clear; omit for all")):
    if "cache" not in _state:
        raise HTTPException(503, "System not ready.")
    _state["cache"].invalidate(cluster_id)
    return {"cleared": True, "cluster_id": cluster_id}


@app.get("/clusters", summary="Cluster summary")
def cluster_info():
    if "clusterer" not in _state:
        raise HTTPException(503, "System not ready.")
    clusterer: FuzzyClusterer = _state["clusterer"]
    if clusterer.labels_ is None:
        raise HTTPException(503, "Clustering not fitted.")
    unique, counts = np.unique(clusterer.labels_, return_counts=True)
    return {
        "n_clusters": int(clusterer.n_clusters),
        "fuzziness": float(clusterer.fuzziness),
        "cluster_sizes": {int(k): int(v) for k, v in zip(unique, counts)},
    }


@app.get("/clusters/{cluster_id}/keywords", summary="Top TF-IDF keywords for a cluster")
def cluster_keywords(cluster_id: int, n: int = Query(10, ge=1, le=30)):
    if "clusterer" not in _state or "texts" not in _state:
        raise HTTPException(503, "System not ready.")
    clusterer: FuzzyClusterer = _state["clusterer"]
    keywords = clusterer.cluster_top_keywords(_state["texts"], n_keywords=n)
    if cluster_id not in keywords:
        raise HTTPException(404, f"Cluster {cluster_id} not found.")
    return {"cluster_id": cluster_id, "top_keywords": keywords[cluster_id]}
