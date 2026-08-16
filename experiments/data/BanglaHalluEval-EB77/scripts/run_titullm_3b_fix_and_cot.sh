#!/usr/bin/env bash
# TituLLM-3B: Fix baseline unknowns → push → clean corrupt CoT → run CoT
# Run on RunPod. Adjust PY path if not using the attention conda env.
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root

# ── Adjust this to your RunPod Python ──
PY="${PY:-python3}"
MODEL="${TITULLM_MODEL:-hishab/titulm-llama-3.2-3b-v1.1}"
LOG_DIR="scripts/results_titullm_3b_logs"
mkdir -p "$LOG_DIR"

# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Fix baseline unknowns (288 rows across gqa_gt, gqa_hallu,
#           reason_gt, reason_hallu)
# ══════════════════════════════════════════════════════════════════════════════
echo "[$(date)] === Phase 1: Rerunning baseline unknowns ==="
"$PY" -u scripts/rerun_unknowns_titullm_3b.py --all --model "$MODEL" \
  2>&1 | tee -a "$LOG_DIR/rerun_unknowns_$(date +%Y%m%d_%H%M%S).log"
echo "[$(date)] Phase 1 (unknowns) complete."

# ── Push fixed baseline results ──
echo "[$(date)] Pushing fixed baseline results..."
git add scripts/results_titullm_3b/
git commit -m "fix: rerun 288 unknown baseline labels (titullm-3b)" || echo "(nothing to commit)"
git push || echo "(push failed — check SSH key / credentials)"
echo "[$(date)] Push done."

# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: Clean corrupted CoT results and run fresh CoT
# ══════════════════════════════════════════════════════════════════════════════
echo "[$(date)] === Phase 2: Cleaning corrupt CoT files ==="
rm -f scripts/results_titullm_3b_cot/gqa_gt_cot.csv
rm -f scripts/results_titullm_3b_cot/gqa_hallu_cot.csv
rm -f scripts/results_titullm_3b_cot/summ_gt_cot.csv
echo "  Deleted 3 corrupted CoT files (99% unknowns)."

echo "[$(date)] === Phase 2: Running CoT on all 6 tasks ==="
"$PY" -u scripts/evaluate_cot_titullm_3b.py --task all --model "$MODEL" \
  2>&1 | tee -a "$LOG_DIR/cot_$(date +%Y%m%d_%H%M%S).log"
echo "[$(date)] Phase 2 (CoT) complete."

# ── Push CoT results ──
echo "[$(date)] Pushing CoT results..."
git add scripts/results_titullm_3b_cot/
git commit -m "feat: titullm-3b CoT evaluation results" || echo "(nothing to commit)"
git push || echo "(push failed — check SSH key / credentials)"

echo "[$(date)] All done. Outputs:"
echo "  Baseline (fixed): scripts/results_titullm_3b/*.csv"
echo "  CoT:              scripts/results_titullm_3b_cot/*.csv"
