# RunPod Guide — Full E-CoT Run on a $17 Budget

Target: run the full BenHalluEval benchmark (11,000 candidates per judge) across the **6 open-weight judges** on a single RunPod GPU pod, staying inside a **$17 credit ceiling** and finishing in **≈ 14–18 wall-clock hours**. GPT-4.1 mini is *not* run here — it goes on your local dev machine via the OpenAI API.

---

## 1. Recommended pod configuration

### GPU: **RTX 4090 (24 GB VRAM)** or **A6000 (48 GB VRAM)** — Community Cloud

| Option | ≈ $/hr | Fits all our models? | Recommended if... |
|---|---|---|---|
| **RTX 4090 24 GB** (community) | ~$0.34–0.50 | Yes (Q4 quantised) | You want lowest cost per hour |
| **A6000 48 GB** (community) | ~$0.49–0.79 | Yes, comfortably | You want headroom + a bit more speed on the 32 B model |
| A100 40 GB SXM | ~$1.19–1.89 | Yes | Skip — kills budget |
| H100 80 GB | ~$2.49+ | Overkill | Skip |

**Pick RTX 4090 24 GB** unless it's out of stock. If both are stocked, A6000 is worth the ~40 % premium *only* for `qwen2.5:32b-instruct` — you can spot-provision a 4090 for the small judges and switch to A6000 just for Qwen.

### Template
- Image: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (or any recent `runpod/pytorch:*-cuda12.*` image)
- Container disk: **75 GB** (Ollama models totalling ~55 GB + HuggingFace TigerLLM ~18 GB + code + logs)
- Volume disk: **20 GB** (persistent — mount to `/workspace`)
- Ports: **22** (SSH), **11434** (Ollama, optional — only if you SSH-tunnel Ollama to your laptop)
- Environment variables: none required at boot; you'll set `OPENAI_API_KEY` only if you also decide to run GPT-4.1 mini from the pod (not recommended — do that locally).

---

## 2. Budget arithmetic

Estimated wall-clock per judge (RTX 4090, Ollama Q4 unless noted), **11,000 candidates each**:

| Judge | Backend | Throughput | Wall-clock | Cost @ $0.50/hr |
|---|---|---|---|---|
| LLaMA-3.1-8B (Q4) | Ollama | ~4 req/s | ~45 min | $0.38 |
| Mistral-Nemo-12B (Q4) | Ollama | ~3 req/s | ~60 min | $0.50 |
| DeepSeek-R1-14B (Q4) | Ollama | ~0.5 req/s (thinking) | ~6 h | $3.00 |
| Gemma-2-27B (Q4) | Ollama | ~1.2 req/s | ~2.5 h | $1.25 |
| Qwen2.5-32B (Q4) | Ollama | ~1.0 req/s | ~3 h | $1.50 |
| TigerLLM-9B (bf16) | HuggingFace | ~0.4 req/s | ~7 h | $3.50 |
| **Subtotal (runtime)** | | | **~20 h** | **~$10.13** |
| Pod up-time buffer (setup, model pulls, idle) | | | ~4 h | ~$2.00 |
| **Grand total** | | | **~24 h** | **≈ $12.15** |

That leaves **~$5 slack**. Real numbers depend on the exact GPU tier you get; watch the RunPod billing dashboard.

**Cost-saving tips:**
- **Stop the pod** the moment TigerLLM finishes (not just detach the terminal). RunPod charges for the whole hour a pod is running.
- Use **Community Cloud** (cheaper than Secure Cloud) unless your data has restrictions.
- Set a **max spend limit** in RunPod → *Settings* → *Billing*.
- Pull models in parallel (`ollama pull ... &`) to shorten the setup phase.

---

## 3. Pod setup checklist (one-time, ~30–45 min)

Assumes you've spun up the pod, opened the *Connect* → *Web Terminal* (or SSH via `runpodctl` / your key).

### 3.1 Get your repo onto the pod
```bash
cd /workspace
git clone https://github.com/<YOUR-USER>/BanglaHalluEval.git
cd BanglaHalluEval
```
If the repo is private, either use a **GitHub PAT** (`git clone https://<PAT>@github.com/...`) or upload a tarball via `runpodctl send`.

### 3.2 System deps + Python deps
```bash
# zstd is required by the Ollama install script; not preinstalled in the
# runpod/pytorch:2.4.0 image, so install it first or the ollama installer
# fails with "This version requires zstd for extraction".
apt-get update && apt-get install -y zstd

pip install --upgrade pandas requests openai python-dotenv bitsandbytes

# The base image already has torch 2.4.1+cu124 preinstalled — do NOT reinstall.
# Only transformers and accelerate are needed for TigerLLM; install them from
# the regular PyPI (the pytorch cu121 index does not host transformers).
pip install transformers accelerate
```

### 3.3 Install Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
which ollama && ollama --version   # sanity — should print a path and version
# Start the daemon (backgrounded)
nohup ollama serve > /workspace/ollama.log 2>&1 &
sleep 3
curl -s http://localhost:11434/api/tags   # should return {"models":[]}
```

### 3.4 Pull the 5 Ollama judges (55 GB total)
Run all pulls **in parallel** to save wall-clock:
```bash
(ollama pull llama3.1:8b            2>&1 | tee -a /workspace/pull.log) &
(ollama pull mistral-nemo:latest    2>&1 | tee -a /workspace/pull.log) &
(ollama pull deepseek-r1:14b        2>&1 | tee -a /workspace/pull.log) &
(ollama pull gemma2:27b             2>&1 | tee -a /workspace/pull.log) &
(ollama pull qwen2.5:32b-instruct   2>&1 | tee -a /workspace/pull.log) &
wait
ollama list
```
On RunPod's usual bandwidth this finishes in **20–35 min**.

### 3.5 Verify Ollama is happy
```bash
curl -s http://localhost:11434/api/tags | head -c 300
```

You should see JSON listing all 5 models.

---

## 4. Runbook — recommended order (fastest budget path)

Optimal order: **small judges first**, then the slow thinking model, then the 27B/32B tier, then TigerLLM last on its own GPU allocation. This front-loads quick wins so if you run out of budget mid-way, you still have coverage on the smaller judges.

```bash
cd /workspace/BanglaHalluEval

# 1. LLaMA-3.1-8B                                 ~45 min
python -u full_ecot_run/scripts/02_run_llama3_1_8b.py --task all --track both \
  2>&1 | tee -a full_ecot_run/results/_logs/llama3_1_8b.log

# 2. Mistral-Nemo-12B                             ~1 h
python -u full_ecot_run/scripts/02_run_mistral_nemo.py --task all --track both \
  2>&1 | tee -a full_ecot_run/results/_logs/mistral_nemo.log

# 3. DeepSeek-R1-14B  (long thinking blocks)      ~6 h — run overnight
python -u full_ecot_run/scripts/02_run_deepseek_r1_14b.py --task all --track both \
  2>&1 | tee -a full_ecot_run/results/_logs/deepseek_r1_14b.log

# 4. Gemma-2-27B                                  ~2.5 h
python -u full_ecot_run/scripts/02_run_gemma2_27b.py --task all --track both \
  2>&1 | tee -a full_ecot_run/results/_logs/gemma2_27b.log

# 5. Qwen2.5-32B                                  ~3 h
python -u full_ecot_run/scripts/02_run_qwen2_5_32b.py --task all --track both \
  2>&1 | tee -a full_ecot_run/results/_logs/qwen2_5_32b.log

# 6. Stop Ollama and free the GPU for TigerLLM
pkill -f "ollama serve"
sleep 5
nvidia-smi   # confirm GPU is free

# 7. TigerLLM-9B (HuggingFace bfloat16)           ~7 h
python -u full_ecot_run/scripts/02_run_tigerllm_9b.py --task all --track both \
  2>&1 | tee -a full_ecot_run/results/_logs/tigerllm_9b.log

# Optional replacement: BanglaLLama-13B (4-bit)
python -u full_ecot_run/scripts/02_run_bangla_llama_13b.py --task all --track both \
  2>&1 | tee -a full_ecot_run/results/_logs/bangla_llama_13b.log
```

**Prefer to just kick off the full sequence?**
```bash
mkdir -p full_ecot_run/results/_logs
bash full_ecot_run/scripts/run_all_open_weight.sh
```
That wrapper runs judges 1→5 with Ollama, then explicitly stops Ollama before step 7, in the exact order above.

### Detaching so you can close your laptop
Every judge takes hours, so run them inside `tmux` (or `screen`):
```bash
tmux new -s ecot
# ... start the commands above ...
# Ctrl-b then d       # detach; the pod keeps running
# Reattach later:
tmux attach -t ecot
```

### If a run is interrupted
Every runner defaults to `--resume` — just re-invoke the same command. Rows already written are skipped, and `os.fsync` after every write means at most **one** row is ever lost.

---

## 5. Monitoring during the run

```bash
# Live progress from the current script
tail -f full_ecot_run/results/_logs/qwen2_5_32b.log

# Row count in the current output file (how far through are we?)
wc -l full_ecot_run/results/qwen2_5_32b/qa_hallu_ecot.csv

# GPU utilisation
watch -n 2 nvidia-smi

# Ollama daemon health
curl -s http://localhost:11434/api/ps
```

If a run visibly stalls, check `full_ecot_run/results/_logs/<slug>.log` and look for `Ollama unreachable at ...`. The runner will block-and-retry when Ollama returns, but if the daemon crashed you need `nohup ollama serve > /workspace/ollama.log 2>&1 &` again.

---

## 6. Syncing results back to your local machine

At any checkpoint (or when everything finishes), pull the compact CSV outputs down:

```bash
# From your laptop
runpodctl send POD_ID:/workspace/BanglaHalluEval/full_ecot_run/results ./results_from_runpod
# or, using rsync over SSH
rsync -avz --progress \
  root@<pod-ip>:/workspace/BanglaHalluEval/full_ecot_run/results/ \
  ./full_ecot_run/results/
```

Results per judge total ~5 MB, so this is fast even on modest bandwidth.

Then compute metrics locally:
```bash
python full_ecot_run/scripts/03_compute_metrics.py
python full_ecot_run/scripts/04_build_paper_tables.py
```

---

## 7. Shutting the pod down properly

The moment TigerLLM finishes, **stop the pod** from the RunPod dashboard — do not just close the browser tab. Otherwise you pay for the idle hours until you notice.

```bash
# Optional: last confirmation before shutdown
ls -la full_ecot_run/results/*/*.csv
```

Then on the RunPod dashboard: pod → **⋯** → **Stop**.

---

## 8. What to do if you run out of budget mid-way

The runners resume from wherever they left off. If the pod is terminated with judges 5 and 6 unfinished:

1. Spin up a new pod (same template).
2. `git pull` the repo, re-copy your `full_ecot_run/results/` back to `/workspace/BanglaHalluEval/full_ecot_run/results/` **first** (so the resume logic sees the partial CSVs).
3. Continue with the pending judge — the runner will skip everything already done.

Keep the partial results locally between sessions.

---

## 9. Quick reference — files you touch on the pod

| Purpose | File |
|---|---|
| Lab-guide (general) | `full_ecot_run/LAB_SETUP_GUIDE.md` |
| RunPod-specific guide | `full_ecot_run/RUNPOD_GUIDE.md` (this file) |
| Per-judge runners | `full_ecot_run/scripts/02_run_*.py` |
| Convenience sequential run | `full_ecot_run/scripts/run_all_open_weight.sh` |
| Metrics + paper tables | `full_ecot_run/scripts/03_compute_metrics.py`, `04_build_paper_tables.py` |
| Ollama daemon log | `/workspace/ollama.log` |
| Per-judge stdout logs | `full_ecot_run/results/_logs/<slug>.log` |
| Per-judge output CSVs | `full_ecot_run/results/<slug>/{qa,summarization,reasoning}_{gt,hallu}_ecot.csv` |

---

## 10. TL;DR

1. Spin up **RTX 4090 24 GB** pod on Community Cloud with a 75 GB container disk.
2. Clone repo → install Ollama + Python deps → pull 5 models in parallel (~35 min).
3. Run judges in this order inside `tmux`:
   `llama3_1_8b → mistral_nemo → deepseek_r1_14b → gemma2_27b → qwen2_5_32b → (stop Ollama) → tigerllm_9b`
4. Every writeback flushes + fsyncs to disk, and every runner `--resume`s by default → interruption is free.
5. `rsync` results back, run `03_compute_metrics.py` + `04_build_paper_tables.py` **locally**, then **stop the pod**.

Expected: **~14–18 hours runtime, ~$10–13 total on $17 credit.**
