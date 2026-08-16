#!/usr/bin/env bash
# Run E-CoT (Variant C) full benchmark on the 6 open-weight judges.
# Designed for the lab PC. GPT-4.1 mini is handled separately.
#
# Prereqs:
#   * Ollama running: `ollama serve` (or daemonised)
#   * Ollama models pulled:
#       ollama pull qwen2.5:32b-instruct
#       ollama pull gemma2:27b
#       ollama pull deepseek-r1:14b
#       ollama pull mistral-nemo:latest
#       ollama pull llama3.1:8b
#   * TigerLLM-9B downloaded by HuggingFace (auto on first run)
#   * Python deps: requests, openai, python-dotenv, pandas, torch, transformers
#
# All scripts support --resume; safe to re-run after interruption.

set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root

LOG_DIR="full_ecot_run/results/_logs"
mkdir -p "$LOG_DIR"

# Ollama judges (smallest/fastest first)
OLLAMA_JUDGES=(
  "llama3_1_8b"
  "deepseek_r1_14b"
  "gemma2_27b"
  "qwen2_5_32b"
)

# HuggingFace judges (need GPU free of Ollama)
HF_JUDGES=(
  "bangla_llama_13b"
  "tigerllm_9b"
)

for slug in "${OLLAMA_JUDGES[@]}"; do
  ts=$(date +%Y%m%d_%H%M%S)
  log="$LOG_DIR/${slug}_${ts}.log"
  echo "[$(date)] starting $slug -> $log"
  python -u "full_ecot_run/scripts/02_run_${slug}.py" --task all --track both 2>&1 | tee "$log"
  echo "[$(date)] finished $slug"
done

# Stop Ollama before loading HuggingFace models into GPU
echo "[$(date)] Stopping Ollama to free GPU VRAM for HuggingFace models..."
pkill -f "ollama serve" 2>/dev/null || true
sleep 5
nvidia-smi --query-gpu=memory.used --format=noheader,nounits 2>/dev/null || true

for slug in "${HF_JUDGES[@]}"; do
  ts=$(date +%Y%m%d_%H%M%S)
  log="$LOG_DIR/${slug}_${ts}.log"
  echo "[$(date)] starting $slug -> $log"
  python -u "full_ecot_run/scripts/02_run_${slug}.py" --task all --track both 2>&1 | tee "$log"
  echo "[$(date)] finished $slug"
done

echo "[$(date)] computing metrics ..."
python full_ecot_run/scripts/03_compute_metrics.py
python full_ecot_run/scripts/04_build_paper_tables.py
echo "[$(date)] done."
