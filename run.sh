#!/bin/bash
set -e

# ClaimVision IA Service - Start Script

MODELS_SRC="${1:-../Mineria de Datos/EDA/models}"

echo "=== ClaimVision IA Service ==="
echo ""

# Check if models exist
if [ ! -f "models/encoder_best.pth" ] || [ ! -f "models/kmeans.pkl" ]; then
    echo "[!] Model files not found in models/"
    echo "    Copying from: $MODELS_SRC"
    mkdir -p models
    cp "$MODELS_SRC"/encoder_best.pth models/
    cp "$MODELS_SRC"/encoder_config.json models/ 2>/dev/null || true
    cp "$MODELS_SRC"/kmeans.pkl models/
    echo "    Models copied."
fi

echo "Starting service..."
echo "  API Docs: http://localhost:8000/docs"
echo "  Health:   http://localhost:8000/api/v1/health"
echo ""

# Run with Docker or local
if command -v docker &> /dev/null && [ -f "docker-compose.yml" ]; then
    echo "[1] Start with Docker Compose..."
    docker compose up --build
else
    echo "[1] Start locally with Uvicorn..."
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
fi
