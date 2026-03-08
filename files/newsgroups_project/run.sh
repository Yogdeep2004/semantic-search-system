#!/usr/bin/env bash
# Run the Hybrid Semantic Search project: install deps, build indices (with
# auto dataset bootstrap if needed), then start the API.

set -e
cd "$(dirname "$0")"

echo "==> Installing dependencies..."
pip install -q -r requirements.txt

echo "==> Building indices (dataset will be downloaded via sklearn if needed)..."
python build_index.py

echo "==> Starting API on http://0.0.0.0:8000"
exec uvicorn api.main:app --host 0.0.0.0 --port 8000
