#!/usr/bin/env bash
# Setup EigenDialectos development environment
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "EigenDialectos Environment Setup"
echo "================================"

cd "$PROJECT_DIR"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing package in editable mode..."
pip install -e ".[dev]" 2>/dev/null || pip install -e .

echo "Creating data directories..."
mkdir -p data/{raw,processed,synthetic,embeddings,spectral,tensor,experiments}
mkdir -p models/{fasttext,word2vec,beto_finetuned,lora_adapters,dial}
mkdir -p outputs/{figures,tables,reports,logs}

echo "Copying .env template..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "  Created .env from template. Edit it to add your API keys."
else
    echo "  .env already exists, skipping."
fi

echo ""
echo "Setup complete! Activate with: source .venv/bin/activate"
echo "Run tests with: make test"
echo "Validate project: make validate"
