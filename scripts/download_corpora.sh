#!/usr/bin/env bash
# Download corpus data for EigenDialectos
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PROJECT_DIR}/data/raw"

echo "EigenDialectos Corpus Download"
echo "=============================="
echo "Output: ${DATA_DIR}"
echo ""

mkdir -p "${DATA_DIR}"

# Synthetic data (always available)
echo "[1/3] Generating synthetic data..."
python -m eigendialectos.cli.main corpus generate-synthetic --output-dir "${PROJECT_DIR}/data/synthetic"

# Wikipedia (freely available)
echo "[2/3] Downloading Wikipedia data..."
python -c "
from eigendialectos.corpus.sources.wikipedia import WikipediaSource
src = WikipediaSource()
src.download('${DATA_DIR}/wikipedia')
" 2>/dev/null || echo "  Skipped (source not configured)"

echo "[3/3] Done."
echo ""
echo "To download additional sources, configure API keys in .env"
echo "and run: eigendialectos corpus download --source=<name>"
