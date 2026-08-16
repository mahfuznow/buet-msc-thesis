#!/usr/bin/env bash
# Run TituLLM-3B baseline + CoT on all 4 tasks (GQA, Summarization, Reasoning, Codemixed)
# and both tracks (ground-truth + hallucinated). Sequential; each script is resumable.

set -euo pipefail

cd "$(dirname "$0")/.."      # repo root

MODEL="${TITULLM_MODEL:-hishab/titulm-llama-3.2-3b-v1.1}"
LOG_DIR="scripts/results_titullm_3b_logs"
mkdir -p "$LOG_DIR"

echo "[$(date)] === Phase 1: baseline (yes/no) on all 4 tasks x both tracks ==="
python -u scripts/evaluate_titullm_3b.py --task all --model "$MODEL" \
  2>&1 | tee -a "$LOG_DIR/baseline_$(date +%Y%m%d_%H%M%S).log"
echo "[$(date)] baseline complete"

echo "[$(date)] === Phase 2: CoT on all 4 tasks x both tracks ==="
python -u scripts/evaluate_cot_titullm_3b.py --task all --model "$MODEL" \
  2>&1 | tee -a "$LOG_DIR/cot_$(date +%Y%m%d_%H%M%S).log"
echo "[$(date)] CoT complete"

echo "[$(date)] All TituLLM-3B runs finished."
echo "Outputs:"
echo "  Baseline: scripts/results_titullm_3b/*.csv"
echo "  CoT:      scripts/results_titullm_3b_cot/*.csv"
