import pickle
import sys
import os
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import (
    CLEANED_DOCS_PATH, EMBEDDINGS_PATH, CLUSTER_MEMBERSHIPS_PATH, CLUSTER_CENTERS_PATH,
    DEFAULT_K, TOP_K_RETRIEVAL, BM25_CANDIDATES
)
from embeddings.embedder import TextEmbedder
from vector_store.faiss_index import FaissVectorIndex
from clustering.fuzzy_clusterer import FuzzyClusterer
from keyword_index.bm25_index import BM25Indexer
from retrieval.query_router import QueryRouter
from retrieval.hybrid_retriever import HybridRetriever
from cache.semantic_cache import SemanticCache

app = FastAPI(title="Hybrid Semantic Search API")

# --- Global Initialization ---
print("Initializing system components at global scope...", flush=True)

# 1. Load data
if not CLEANED_DOCS_PATH.exists():
    raise RuntimeError(f"Cleaned documents not found at {CLEANED_DOCS_PATH}")

with open(CLEANED_DOCS_PATH, "rb") as f:
    documents = pickle.load(f)

# 2. Components
print("Initializing Embedder...", flush=True)
embedder = TextEmbedder()

print("Loading raw embeddings...", flush=True)
embeddings = np.load(EMBEDDINGS_PATH)

print("Building FAISS index...", flush=True)
faiss_index = FaissVectorIndex()
faiss_index.build_index(embeddings)

print("Building BM25 index...", flush=True)
bm25_index = BM25Indexer()
bm25_index.build_index(documents)

# 3. Clustering
print("Loading Fuzzy Clusterer...", flush=True)
clusterer = FuzzyClusterer(n_clusters=DEFAULT_K)
if CLUSTER_CENTERS_PATH.exists():
    clusterer.cntr = np.load(CLUSTER_CENTERS_PATH)

# 4. Retrieval & Router
print("Initializing Router...", flush=True)
router = QueryRouter(embedder, clusterer)

print("Loading Memberships...", flush=True)
memberships = np.load(CLUSTER_MEMBERSHIPS_PATH)
dominant_clusters = np.argmax(memberships, axis=1)

print("Initializing Hybrid Retriever...", flush=True)
retriever = HybridRetriever(
    bm25_index, faiss_index, embedder, documents, embeddings, dominant_clusters
)

# 5. Cache
print("Initializing Cache...", flush=True)
cache = SemanticCache()

print("System components initialized successfully!", flush=True)

class QueryRequest(BaseModel):
    query: str
    top_k: int = TOP_K_RETRIEVAL
    use_cache: bool = True
    use_cluster_restriction: bool = False

class CacheStatsResponse(BaseModel):
    total_entries: int
    hits: int
    misses: int
    hit_rate: float

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    try:
        # 1 & 2. Embed Query and Detect Cluster
        query_embedding, cluster_id = router.route_query(request.query)
        
        # 3. Cache Lookup
        if request.use_cache:
            cached_result = cache.lookup(query_embedding, cluster_id)
            if cached_result:
                return {"results": cached_result, "cache_hit": True, "cluster_id": int(cluster_id)}
        
        # 4. Hybrid Retrieval
        results = retriever.retrieve(
            request.query, 
            query_embedding, 
            bm25_k=BM25_CANDIDATES, 
            final_k=request.top_k,
            use_cluster_restriction=request.use_cluster_restriction,
            target_cluster=cluster_id
        )
        
        # 5. Update Cache
        if request.use_cache:
            cache.add(request.query, query_embedding, results, cluster_id)
            
        return {"results": results, "cache_hit": False, "cluster_id": int(cluster_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats():
    return cache.stats()

@app.delete("/cache")
async def clear_cache():
    cache.clear()
    return {"message": "Cache cleared successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
