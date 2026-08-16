#!/usr/bin/env bash
# Pod 1 — Ollama judges: LLaMA-3.1-8B (resume) + DeepSeek-R1-14B + Gemma-2-27B + Qwen2.5-32B
#
# Run this entire file inside tmux on the RunPod pod:
#   tmux new -s pod1
#   bash full_ecot_run/scripts/run_pod1_ollama.sh
#   Ctrl-b d   (detach; pod keeps running)
#
# Every judge is resumable — re-run the same script after an interruption.
# Already-written rows are skipped automatically.

set -euo pipefail
cd /workspace/BanglaHalluEval

LOG_DIR="full_ecot_run/results/_logs"
mkdir -p "$LOG_DIR"

# ── 1. System + Python dependencies ──────────────────────────────────────────
echo "[setup] Installing system and Python deps..."
apt-get update -qq && apt-get install -y -qq zstd curl
pip install -q --upgrade pandas requests python-dotenv

# ── 2. Install Ollama (idempotent) ───────────────────────────────────────────
if ! command -v ollama &>/dev/null; then
  echo "[setup] Installing Ollama..."
  curl -fsSL https://ollama.com/install.sh | sh
fi

# ── 3. Start Ollama daemon ───────────────────────────────────────────────────
if ! curl -s http://localhost:11434/api/tags &>/dev/null; then
  echo "[setup] Starting Ollama daemon..."
  nohup ollama serve > /workspace/ollama.log 2>&1 &
  sleep 5
fi
echo "[setup] Ollama ready: $(curl -s http://localhost:11434/api/tags | python3 -c 'import sys,json; d=json.load(sys.stdin); print(len(d.get("models",[])), "models loaded")')"

# ── 4. Pull models in parallel (skip if already present) ─────────────────────
echo "[setup] Pulling Ollama models in parallel (this takes 20-35 min first time)..."
(ollama pull llama3.1:8b          2>&1 | tee -a /workspace/pull.log) &
(ollama pull deepseek-r1:14b      2>&1 | tee -a /workspace/pull.log) &
(ollama pull gemma2:27b           2>&1 | tee -a /workspace/pull.log) &
(ollama pull qwen2.5:32b-instruct 2>&1 | tee -a /workspace/pull.log) &
wait
echo "[setup] All models pulled."
ollama list

# ── 5. Run judges (smallest/fastest first) ───────────────────────────────────

echo ""
echo "================================================================"
echo " [1/4] LLaMA-3.1-8B   (~45 min, resuming from partial results)"
echo "================================================================"
ts=$(date +%Y%m%d_%H%M%S)
python -u full_ecot_run/scripts/02_run_llama3_1_8b.py --task all --track both \
  2>&1 | tee -a "$LOG_DIR/llama3_1_8b_${ts}.log"
echo "[done] LLaMA-3.1-8B finished at $(date)"

echo ""
echo "================================================================"
echo " [2/4] DeepSeek-R1-14B  (~6 h, thinking model — be patient)"
echo "================================================================"
ts=$(date +%Y%m%d_%H%M%S)
python -u full_ecot_run/scripts/02_run_deepseek_r1_14b.py --task all --track both \
  2>&1 | tee -a "$LOG_DIR/deepseek_r1_14b_${ts}.log"
echo "[done] DeepSeek-R1-14B finished at $(date)"

echo ""
echo "================================================================"
echo " [3/4] Gemma-2-27B  (~2.5 h)"
echo "================================================================"
ts=$(date +%Y%m%d_%H%M%S)
python -u full_ecot_run/scripts/02_run_gemma2_27b.py --task all --track both \
  2>&1 | tee -a "$LOG_DIR/gemma2_27b_${ts}.log"
echo "[done] Gemma-2-27B finished at $(date)"

echo ""
echo "================================================================"
echo " [4/4] Qwen2.5-32B-Instruct  (~3 h)"
echo "================================================================"
ts=$(date +%Y%m%d_%H%M%S)
python -u full_ecot_run/scripts/02_run_qwen2_5_32b.py --task all --track both \
  2>&1 | tee -a "$LOG_DIR/qwen2_5_32b_${ts}.log"
echo "[done] Qwen2.5-32B finished at $(date)"

echo ""
echo "================================================================"
echo " Pod 1 COMPLETE. Pull results with rsync/runpodctl then STOP this pod."
echo " ls full_ecot_run/results/*/*.csv | wc -l"
echo "================================================================"
ls -lh full_ecot_run/results/*/*.csv 2>/dev/null || true
