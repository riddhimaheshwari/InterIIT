#!/usr/bin/env bash
set -e

REGIME=${1:-"Q1"}
echo "============================================================"
echo " Running Quarterly Update for Regime: ${REGIME}"
echo "============================================================"

python -m src.offline.pipeline --regime "${REGIME}"

echo "Update complete for ${REGIME}."
