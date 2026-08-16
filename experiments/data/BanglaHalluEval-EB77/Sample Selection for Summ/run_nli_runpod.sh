#!/usr/bin/env bash
# One-shot RunPod runner for the NLI-vs-BERTScore summarization experiment.
# Assumes an NVIDIA pytorch pod (torch already installed) and that this repo is
# cloned at /workspace/BanglaHalluEval.  Run from anywhere:
#   bash "Sample Selection for Summ/run_nli_runpod.sh"
set -euo pipefail

cd "$(dirname "$0")/.."          # repo root
echo "== repo: $(pwd)"

echo "== GPU check"
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

echo "== installing deps"
pip install -q -r "Sample Selection for Summ/nli_requirements.txt"

echo "== smoke test (200 pairs)"
python "Sample Selection for Summ/nli_vs_bertscore.py" --limit 100

echo "== full run"
python "Sample Selection for Summ/nli_vs_bertscore.py" 2>&1 | tee "Sample Selection for Summ/nli_vs_bertscore/run.log"

echo "== done. outputs in 'Sample Selection for Summ/nli_vs_bertscore/'"
ls -la "Sample Selection for Summ/nli_vs_bertscore/"
