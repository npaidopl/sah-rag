#!/usr/bin/env bash
# SAH-RAG Ablation: Master GPU Execution Script (Linux server)
set -euo pipefail
export PYTHONIOENCODING=utf-8
export HF_HUB_OFFLINE=1

SAH="${SAH_ROOT:-$HOME/SAH-RAG}"
PY="$SAH/.venv/bin/python"
START=$(date +%s)
echo "=========================================================="
echo " SAH-RAG Ablation: Master GPU Run"
echo " Start: $(date)"
echo "=========================================================="

cd "$SAH"

echo ">>> STEP 1/5: Encode Tier-1 chunks (49 pairs)"
$PY -m ablation.embedding.encode_chunks --tier 1 --batch-size 32
echo ""
echo ">>> STEP 2/5: Encode Tier-3 chunks (10 pairs)"
$PY -m ablation.embedding.encode_chunks --tier 3 --batch-size 32
echo ""
echo ">>> STEP 3/5: Encode queries (all 17 datasets)"
$PY -m ablation.embedding.encode_queries --all
echo ""
echo ">>> STEP 4/5: Run Tier-1 evaluations (49 cells)"
$PY -m ablation.run_all --tier 1
echo ""
echo ">>> STEP 5/5: Run Tier-2 evaluations (272 free re-ranks)"
$PY -m ablation.run_all --tier 2
echo ""
echo ">>> FINAL: Generate reports"
$PY -m ablation.reporting.all

ELAPSED=$(( $(date +%s) - START ))
echo ""
echo "=========================================================="
echo " DONE in $((ELAPSED/3600))h $((ELAPSED%3600/60))m $((ELAPSED%60))s"
echo "=========================================================="