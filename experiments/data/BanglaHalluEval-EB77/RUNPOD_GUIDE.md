# RunPod Migration Guide — BanglaHalluEval

Migrating from local Ollama (`localhost:11434`) to RunPod GPU pods.

---

## Current Setup (Ollama Local)

All scripts call `http://localhost:11434/api/generate` or `/api/chat` with these models:

| Model | Task | VRAM Required |
|-------|------|---------------|
| `qwen2.5:32b-instruct` | QA/Reasoning labeling | ~20GB |
| `qwen2.5:14b` | QA zero-shot, summarization | ~10GB |
| `deepseek-r1:14b` | Reasoning, QA | ~10GB |
| `gemma2:27b` | Reasoning, summarization | ~16GB |
| `gemma2:9b` | QA zero-shot | ~6GB |
| `mistral-nemo:latest` | QA labeling | ~8GB |

---

## Strategy: Run Ollama on RunPod Pods

Deploy Ollama on a RunPod GPU pod. Scripts need only **one line change** — the base URL.

---

## Step 1: Choose the Right GPU

| GPU | VRAM | $/hr | Best For |
|-----|------|------|----------|
| RTX 4090 | 24GB | ~$0.44–0.74 | All 14B models, gemma2:9b |
| RTX A6000 | 48GB | ~$0.79 | qwen2.5:32b, gemma2:27b |
| L40S | 48GB | ~$1.19 | Fast 32B inference |
| A100 SXM 80GB | 80GB | ~$2.49 | All models in one pod |

**Recommended: RTX A6000 (48GB)** — runs all models comfortably at a good price.

---

## Step 2: Deploy Ollama on RunPod

### Option A: Community Template (Easiest)

1. Go to **runpod.io → Deploy → GPU Cloud**
2. Search template: **"Ollama"**
3. Select GPU (A6000 recommended)
4. Set **Container Disk**: 100GB
5. **Expose HTTP Port**: `11434`
6. Deploy

### Option B: Custom Docker

- **Image:** `ollama/ollama:latest`
- **Start Command:** `ollama serve`
- **Exposed Port:** `11434`
- **Volume mount:** `/root/.ollama` → 100GB persistent storage

---

## Step 3: Pull Models on the Pod

SSH into the pod or use RunPod's web terminal:

```bash
ollama pull qwen2.5:32b-instruct   # ~20GB
ollama pull qwen2.5:14b            # ~9GB
ollama pull deepseek-r1:14b        # ~9GB
ollama pull gemma2:27b             # ~16GB
ollama pull gemma2:9b              # ~5.5GB
ollama pull mistral-nemo           # ~7GB

ollama list  # verify
```

Total download: ~66GB — use 100GB container disk.

---

## Step 4: Get Your Pod URL

RunPod exposes Ollama via a proxy URL:

```
https://{YOUR-POD-ID}-11434.proxy.runpod.net
```

Find this in: **RunPod dashboard → Your Pod → Connect → HTTP Service → Port 11434**

---

## Step 5: The Only Code Change Needed

In every script, change the base URL:

```python
# OLD (local)
url = "http://localhost:11434/api/generate"

# NEW (RunPod)
RUNPOD_URL = "https://abc123xyz-11434.proxy.runpod.net"
url = f"{RUNPOD_URL}/api/generate"
```

### Cleaner approach — use an environment variable

Create a `.env` file in the project root:

```
OLLAMA_BASE_URL=https://abc123xyz-11434.proxy.runpod.net
```

Update scripts to read it:

```python
import os
from dotenv import load_dotenv
load_dotenv()

BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
url = f"{BASE_URL}/api/generate"
```

Or set it inline before running:

```powershell
$env:OLLAMA_BASE_URL = "https://abc123xyz-11434.proxy.runpod.net"
python scripts/label_hallucinations_ollama.py --input ...
```

---

## Step 6: Upload Scripts and Data to the Pod

### Option A: Clone from GitHub

```bash
git clone https://github.com/AASani29/BanglaHalluEval.git /workspace/BanglaHalluEval
cd /workspace/BanglaHalluEval
pip install pandas requests python-dotenv tqdm
```

### Option B: SCP from local machine

```bash
scp -r "C:\Users\bashi\...\BanglaHalluEval" root@{pod-ip}:/workspace/
```

### Option C: RunPod Network Volume

Mount a persistent volume, upload once, reuse across pod restarts.

---

## Step 7: Run the Scripts

```bash
cd /workspace/BanglaHalluEval

# QA labeling — Qwen 32B
python scripts/label_hallucinations_ollama.py \
  --input "Results/qa_4000.csv" \
  --output "Results/qa_4000_qwen32b_labels.csv" \
  --model qwen2.5:32b-instruct

# QA labeling — Mistral
python scripts/label_hallucinations_mistral.py \
  --input "Results/qa_4000.csv" \
  --output "Results/qa_4000_mistral_labels.csv"

# Reasoning evaluation — DeepSeek
python "Reasoning/Evaluation Script/evaluate_reasoning_deepseek.py"

# Reasoning evaluation — Gemma2
python "Reasoning/Evaluation Script/evaluate_reasoning_gemma2.py"

# Summarization generation — Qwen
python "Sample Selection for Summ/generate_summaries.py"

# Summarization generation — Gemma
python "Sample Selection for Summ/generate_summaries_gemma.py"
```

---

## Cost Estimates

### Processing time on A6000:

| Task | Samples | Model | Time/sample | Total time |
|------|---------|-------|-------------|------------|
| QA labeling | 4,000 | qwen2.5:32b | ~2–3s | ~3 hrs |
| QA labeling | 4,000 | deepseek-r1:14b | ~1–2s | ~2 hrs |
| QA labeling | 4,000 | gemma2:9b | ~0.5s | ~1 hr |
| Summarization gen | 3,000 | qwen2.5:14b | ~5–10s | ~5–8 hrs |
| Summarization gen | 3,000 | gemma2:27b | ~8–12s | ~8–10 hrs |
| Reasoning eval | 1,000 | qwen2.5:32b | ~3–5s | ~1.5 hrs |
| Reasoning eval | 1,000 | deepseek-r1:14b | ~2–4s | ~1 hr |

### Total cost on A6000 ($0.79/hr):

| Run | Hours | Cost |
|-----|-------|------|
| All QA labeling (3 models) | ~6 hrs | ~$4.70 |
| All summarization (2 models) | ~18 hrs | ~$14.20 |
| All reasoning (2 models) | ~2.5 hrs | ~$2.00 |
| **Total** | **~26.5 hrs** | **~$21** |

### Cost-saving tips

- Use **spot instances** (~30–50% cheaper). Your scripts already have checkpoint/resume logic so interruptions are safe.
- Use **RTX 4090 ($0.44/hr)** for all 14B models, A6000 only for 32B.
- Use a **RunPod Network Volume** (~$0.07/GB/month) to persist model weights across pod restarts — avoids re-downloading 66GB every session.

---

## Recommended Workflow

```
Day 1:
  1. Spin up A6000 pod
  2. Pull all models (~1–1.5 hrs, ~$1 cost)
  3. Run QA labeling for all models back to back
  4. Stop pod

Day 2:
  1. Restart same pod (models cached on volume)
  2. Run summarization generation
  3. Stop pod

Day 3:
  1. Run reasoning evaluation
  2. Download all results via scp or git push
  3. Terminate pod
```

---

## Persistent Volume Setup (Recommended)

Create a **RunPod Network Volume**: 100GB at ~$0.07/GB/month = **~$7/month**.

Mount at `/root/.ollama` so model weights survive pod restarts. Start/stop pods freely without re-downloading models.

---

## Summary

| | Old (Ollama local) | New (RunPod) |
|---|---|---|
| Base URL | `http://localhost:11434` | `https://{pod-id}-11434.proxy.runpod.net` |
| GPU | Local hardware | A6000 48GB (rented) |
| Total cost | Electricity | ~$21–30 for full eval |
| Script changes | — | One env var change |

Everything else — Ollama API calls, model names, JSON parsing, checkpoint logic — works exactly the same.

---

## Keeping Runs Alive After Disconnection (tmux)

**Always use tmux when running long jobs on RunPod.** If your device goes off, the internet drops, or credits temporarily run out and the pod is paused/restarted, tmux keeps the process running inside the pod and lets you reattach.

### First-time setup on a fresh pod

```bash
apt-get update && apt-get install -y tmux git python3 python3-pip
pip3 install requests pandas python-dotenv tqdm --break-system-packages
ollama serve &   # start Ollama in background if not already running
ollama list      # verify models are loaded
```

### Starting a run inside tmux

```bash
# Create a named session
tmux new -s <session-name>

# Inside the session, run your script
python3 scripts/your_script.py

# Detach (leave script running) — press:
Ctrl+B  then  D
```

Your device can now go off. The pod keeps running.

### Reattaching to a running session

```bash
tmux attach -t <session-name>
```

### Listing all active sessions

```bash
tmux ls
```

### Killing a session when done

```bash
tmux kill-session -t <session-name>
```

---

## Standard Run Order for Reasoning Evaluation

Run candidates first, then main. Each in its own tmux session.

### Step 1 — Candidates (hallucinated 1000)

```bash
tmux new -s candidates
python3 scripts/evaluate_reasoning_deepseek_candidates.py
# Ctrl+B then D to detach
```

Output: `Reasoning/Results/reasoning_1000_candidates_deepseek.csv`

### Step 2 — Main ground truth (1000)

```bash
tmux new -s main
python3 scripts/evaluate_reasoning_deepseek_main.py
# Ctrl+B then D to detach
```

Output: `Reasoning/Results/reasoning_main_1000_deepseek.csv`

Both scripts checkpoint after **every row** — if the pod is interrupted, re-run the same command and it resumes automatically.

---

## Pushing Results Back to GitHub

Git is not installed by default. Install it and authenticate once per pod:

```bash
apt-get install -y git
git config --global user.email "shefayatadib@iut-dhaka.edu"
git config --global user.name "Shefwef"
git remote set-url origin https://Shefwef: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@github.com/AASani29/BanglaHalluEval.git
```

Then push results:

```bash
git add Reasoning/Results/reasoning_1000_candidates_deepseek.csv
git add Reasoning/Results/reasoning_main_1000_deepseek.csv
git commit -m "Add deepseek reasoning eval results"
git push
```

**Important:** Use a token from an account that has **Write** access to the repo. Generate at:
`https://github.com/settings/tokens/new` → check `repo` scope.

Pull on your local machine after pushing:

```powershell
git pull
```

---

## If SSH / SCP Fails (Outbound Network Restricted)

RunPod pods may block outbound connections. Use the GitHub API via Python to upload files directly:

```bash
python3 - <<'EOF'
import base64, json, urllib.request

TOKEN = "YOUR_PAT"
FILE  = "path/to/your/file.csv"
REPO_PATH = "path/in/repo/file.csv"   # must match exact path in repo

with open(FILE, "rb") as f:
    content = base64.b64encode(f.read()).decode()

payload = json.dumps({
    "message": "Upload result file",
    "content": content
}).encode()

req = urllib.request.Request(
    f"https://api.github.com/repos/AASani29/BanglaHalluEval/contents/{REPO_PATH}",
    data=payload,
    headers={"Authorization": f"token {TOKEN}", "Content-Type": "application/json"},
    method="PUT"
)
with urllib.request.urlopen(req) as r:
    print("Success:", r.status)
EOF
```

Then `git pull` locally to get the file.

---

## Post-Run Cleanup (Deduplication)

If a script was interrupted and restarted multiple times, the output CSV may contain duplicate rows. Clean up locally after downloading:

```python
import pandas as pd

path = "path/to/output.csv"
df = pd.read_csv(path)
df = df[df["is_hallucinated"].isin(["Yes", "No", "yes", "no"])]
df = df.drop_duplicates(subset=["id"], keep="last")   # use "question_id" if no "id" column
df.to_csv(path, index=False)
print(f"Clean row count: {len(df)}")  # should match dataset size (e.g. 1000 or 3000)
```
