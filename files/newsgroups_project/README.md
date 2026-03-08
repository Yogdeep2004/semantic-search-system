# Hybrid Semantic Search System
### 20 Newsgroups · BM25 + FAISS + Fuzzy Clustering + Semantic Cache

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Build all indices (one-time setup)

```bash
python build_index.py --zip_path ./data/twenty_newsgroups_dataset.zip
```

This will:
- Extract the ZIP → TAR.GZ → plain text documents
- Compute sentence-transformer embeddings (cached to `data/cache/`)
- Build the BM25 keyword index
- Build the FAISS vector index
- Fit Fuzzy C-Means clustering (k=20)

### 3. Start the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- API:       http://localhost:8000
- Swagger:   http://localhost:8000/docs
- ReDoc:     http://localhost:8000/redoc

### 4. Search

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "NASA space shuttle", "top_k": 5}'
```

---

## Docker

```bash
# Build + run
docker build -t hybrid-search .
docker run -p 8000:8000 -v $(pwd)/data:/app/data hybrid-search

# Or with docker-compose:
docker-compose up --build
```

---

## Project Structure

```
.
├── api/
│   └── main.py              # FastAPI application
├── src/
│   ├── data_extractor.py    # ZIP/TAR.GZ extraction, document loading
│   ├── embeddings.py        # SentenceTransformer wrapper
│   ├── vector_store.py      # FAISS index (build + search)
│   ├── bm25_retriever.py    # BM25 keyword retrieval
│   ├── hybrid_retriever.py  # BM25 + FAISS hybrid pipeline
│   ├── fuzzy_clustering.py  # Fuzzy C-Means clustering
│   └── semantic_cache.py    # Cluster-aware semantic cache
├── build_index.py           # One-time index build script
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── WALKTHROUGH.md           # Full technical walkthrough
```

---

See **[WALKTHROUGH.md](WALKTHROUGH.md)** for the full technical documentation.
