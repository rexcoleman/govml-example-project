#!/usr/bin/env bash
# reproduce.sh — Run the entire govML example pipeline
set -euo pipefail

echo "=== govML Example: Iris Classification ==="
echo ""

echo "[1/6] Training 3 algorithms x 5 seeds..."
python scripts/train.py
echo ""

echo "[2/6] Running sanity baselines..."
python scripts/run_sanity_baselines.py
echo ""

echo "[3/6] Generating learning curves..."
python scripts/run_learning_curves.py
echo ""

echo "[4/6] Creating report figures..."
python scripts/make_report_figures.py
echo ""

echo "[5/6] Final test-set evaluation..."
python scripts/final_eval.py
echo ""

echo "[6/6] Running governance tests..."
pytest tests/ -v
echo ""

echo "=== Pipeline complete. Check outputs/ for results. ==="
