#!/usr/bin/env bash
set -e

echo "Starting Speech Understanding dashboard..."
echo "Working directory: $(pwd)"

mkdir -p /app/data/raw
mkdir -p /app/data/processed
mkdir -p /app/models
mkdir -p /app/outputs
mkdir -p /app/.cache/huggingface

exec streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0