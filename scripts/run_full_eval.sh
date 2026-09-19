#!/usr/bin/env bash
set -e

echo "============================================================"
echo " Running Full Continual Evaluation across Q1-Q4 & Confusion Set"
echo "============================================================"

for REGIME in Q1 Q2 Q3 Q4; do
    echo "Updating ${REGIME}..."
    python -m src.offline.pipeline --regime "${REGIME}"
done

echo "Running pytest verification..."
pytest tests/ -v

echo "Full Evaluation & Verification Completed."
