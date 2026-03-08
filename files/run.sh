#!/usr/bin/env bash
# Run the entire Hybrid Semantic Search project from repo root.
# Uses the extracted newsgroups_project (extract hybrid_search_system.zip if needed).

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
PROJECT="${ROOT}/newsgroups_project"

if [[ ! -d "$PROJECT/api" ]]; then
  echo "Project not found. Extracting hybrid_search_system.zip..."
  unzip -o -q "${ROOT}/hybrid_search_system.zip" -d "${ROOT}"
fi

cd "$PROJECT"
exec ./run.sh
