# ============================================================
# Dockerfile — Hybrid Semantic Search API
# ============================================================
# Builds a container that serves the FastAPI application using
# uvicorn on port 8000.
#
# Usage
# -----
#   docker build -t hybrid-search .
#   docker run -p 8000:8000 hybrid-search
#
# The API will be available at: http://localhost:8000
#
# Pre-built index artefacts (data/cache/) can be mounted at
# runtime to avoid rebuilding on every container start:
#   docker run -p 8000:8000 \
#     -v $(pwd)/data:/app/data \
#     hybrid-search
# ============================================================

FROM python:3.11-slim

# --- system dependencies -------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# --- working directory ---------------------------------------------------
WORKDIR /app

# --- Python dependencies -------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# --- application source --------------------------------------------------
COPY . .

# --- runtime config ------------------------------------------------------
ENV DATA_DIR=/app/data/raw/20_newsgroups
ENV EMBED_CACHE=/app/data/cache/embeddings.npy
ENV BM25_CACHE=/app/data/cache/bm25.pkl
ENV CLUSTER_CACHE=/app/data/cache/clusters.pkl
ENV CACHE_THRESHOLD=0.85
ENV PYTHONUNBUFFERED=1

# --- expose API port ------------------------------------------------------
EXPOSE 8000

# --- entrypoint -----------------------------------------------------------
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
