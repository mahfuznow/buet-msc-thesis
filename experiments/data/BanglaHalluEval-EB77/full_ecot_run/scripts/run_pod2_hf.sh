#!/usr/bin/env bash
# Pod 2 — HuggingFace judges: BanglaLLaMA-13B (4-bit) + TigerLLM-9B (bf16)
#
# Run this entire file inside tmux on the RunPod pod:
#   tmux new -s pod2
#   bash full_ecot_run/scripts/run_pod2_hf.sh
#   Ctrl-b d   (detach; pod keeps running)
#
# No Ollama needed here — purely HuggingFace Transformers.
# BanglaLLaMA runs in 4-bit (~8 GB VRAM), TigerLLM in bfloat16 (~18 GB VRAM).
# Both fit on a single 24 GB GPU running one at a time.

set -euo pipefail
cd /workspace/BanglaHalluEval

LOG_DIR="full_ecot_run/results/_logs"
mkdir -p "$LOG_DIR"

# ── 1. Python dependencies ────────────────────────────────────────────────────
# The RunPod pytorch image ships with torch 2.4+cu124 preinstalled.
# Do NOT reinstall torch — it breaks the CUDA build.
echo "[setup] Installing Python deps (transformers, accelerate, bitsandbytes)..."
pip install -q --upgrade pandas python-dotenv
pip install -q transformers accelerate
pip install -q bitsandbytes   # required for BanglaLLaMA 4-bit quantisation

echo "[setup] Verifying CUDA..."
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

# ── 2. BanglaLLaMA-13B (4-bit, ~8 GB VRAM) ───────────────────────────────────
echo ""
echo "================================================================"
echo " [1/2] BanglaLLaMA-13B  4-bit  (~5 h)"
echo "       HuggingFace model: BanglaLLM/bangla-llama-13b-instruct-v0.1"
echo "       First run downloads ~7 GB from HuggingFace (~5-10 min)"
echo "================================================================"
ts=$(date +%Y%m%d_%H%M%S)
python -u full_ecot_run/scripts/02_run_bangla_llama_13b.py --task all --track both \
  2>&1 | tee -a "$LOG_DIR/bangla_llama_13b_${ts}.log"
echo "[done] BanglaLLaMA-13B finished at $(date)"

# ── 3. TigerLLM-9B (bfloat16, ~18 GB VRAM) ──────────────────────────────────
echo ""
echo "================================================================"
echo " [2/2] TigerLLM-9B  bfloat16  (~7 h)"
echo "       HuggingFace model: md-nishat-008/TigerLLM-9B-it"
echo "       First run downloads ~18 GB from HuggingFace (~10-20 min)"
echo "================================================================"
ts=$(date +%Y%m%d_%H%M%S)
python -u full_ecot_run/scripts/02_run_tigerllm_9b.py --task all --track both \
  2>&1 | tee -a "$LOG_DIR/tigerllm_9b_${ts}.log"
echo "[done] TigerLLM-9B finished at $(date)"

echo ""
echo "================================================================"
echo " Pod 2 COMPLETE. Pull results with rsync/runpodctl then STOP this pod."
echo "================================================================"
ls -lh full_ecot_run/results/*/*.csv 2>/dev/null || true
