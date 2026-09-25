#!/usr/bin/env bash
# Setup script for Linux / macOS / WSL
set -e

echo "===================================================="
echo "Setting up Business Entity Resolution Environment"
echo "===================================================="

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements-lock.txt

echo ""
echo "Running test suite to verify setup..."
pytest tests/ -v

echo ""
echo "Virtual environment setup complete!"
echo "To activate: source .venv/bin/activate"
echo "To run pipeline: python run_pipeline.py --mode all"
