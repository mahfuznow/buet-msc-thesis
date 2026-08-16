# Full-Benchmark E-CoT Run — Lab PC Operating Guide

This folder runs E-CoT (Variant C: Evidence + CoT + Citation Forcing) on the **entire BenHalluEval** (3,000 GT + 8,000 hallucinated = **11,000 candidates per judge**) across **all 7 judges**.

If you are picking this up cold, **read [`../HALLUCINATION_MITIGATION.md`](../HALLUCINATION_MITIGATION.md) for the project-wide overview** and [`../pilot_50_samples/ECOT_PILOT_REPORT.md`](../pilot_50_samples/ECOT_PILOT_REPORT.md) for the pilot report that froze every design decision used here.

---

## 1. What this folder does

For each (judge × task × track) combination it:

1. Loads the canonical input CSV (GT or hallucinated) from the existing BenHalluEval data folders.
2. Builds an evidence-augmented Citation-Forcing prompt (Variant C).
3. Calls the judge model with `temperature=0` and a structured JSON response.
4. Parses the JSON, computes both `verdict_model` and `verdict_agg`, and selects the **task-aware** `is_hallucinated` value (aggregated for QA + Reasoning; model self-report for Summarization — the policy frozen in the pilot).
5. Streams the row to `full_ecot_run/results/<slug>/<task>_<gt|hallu>_ecot.csv`, checkpoint-flushing every 20 rows so a crash never loses more than 20 rows.
6. On re-invocation it `--resume`s automatically (skips rows whose `id_key` already appears in the output CSV).

After all judges finish, `03_compute_metrics.py` + `04_build_paper_tables.py` aggregate into the rebuttal tables.

---

## 2. Folder layout

```
full_ecot_run/
├── README.md                                ← this file
├── prompts/
│   ├── ecot_qa.txt
│   ├── ecot_summarization.txt
│   └── ecot_reasoning.txt
├── scripts/
│   ├── _ecot_core.py                        ← shared driver + policies
│   ├── _backend_ollama.py                   ← Ollama HTTP client (5 judges)
│   ├── _backend_openai.py                   ← OpenAI client (GPT-4.1 mini)
│   ├── _backend_tigerllm.py                 ← HuggingFace transformers (TigerLLM)
│   ├── 02_run_gpt4_1_mini.py                ← API judge — run on dev machine
│   ├── 02_run_qwen2_5_32b.py                ← Ollama judge
│   ├── 02_run_gemma2_27b.py                 ← Ollama judge
│   ├── 02_run_deepseek_r1_14b.py            ← Ollama judge (thinking model)
│   ├── 02_run_mistral_nemo.py               ← Ollama judge
│   ├── 02_run_llama3_1_8b.py                ← Ollama judge
│   ├── 02_run_tigerllm_9b.py                ← HuggingFace judge (single GPU)
│   ├── 03_compute_metrics.py                ← BHS vs Baseline + CoT
│   ├── 04_build_paper_tables.py             ← markdown + LaTeX tables
│   └── run_all_open_weight.sh               ← convenience: runs the 6 open-weight judges in series
└── results/                                 ← auto-populated; one subfolder per judge
    └── <slug>/
        ├── qa_gt_ecot.csv            qa_hallu_ecot.csv
        ├── summarization_gt_ecot.csv summarization_hallu_ecot.csv
        ├── reasoning_gt_ecot.csv     reasoning_hallu_ecot.csv
        └── metrics.csv               (produced by 03_compute_metrics.py)
```

---

## 3. Lab PC prerequisites

### 3.1 Hardware
| Judge | GPU memory needed | Notes |
|---|---|---|
| Qwen2.5-32B (Q4 quant via Ollama) | ~22 GB | 24 GB GPU OK |
| Gemma-2-27B (Q4) | ~18 GB | |
| DeepSeek-R1-14B (Q4) | ~10 GB | needs large context for thinking |
| Mistral-Nemo-12B (Q4) | ~9 GB | |
| LLaMA-3.1-8B (Q4) | ~6 GB | |
| TigerLLM-9B (bfloat16) | ~18 GB | **HuggingFace, not Ollama** — give it a dedicated GPU |

A single 24 GB GPU is enough; only one judge runs at a time.

### 3.2 Software
- Python 3.10+
- `pip install pandas requests openai python-dotenv`
- For TigerLLM only: `pip install torch transformers accelerate` (matching CUDA build)
- Ollama installed and running: `curl -fsSL https://ollama.com/install.sh | sh && ollama serve`

### 3.3 Pull all Ollama judges (one-time, ~50 GB total)
```bash
ollama pull qwen2.5:32b-instruct
ollama pull gemma2:27b
ollama pull deepseek-r1:14b
ollama pull mistral-nemo:latest
ollama pull llama3.1:8b
```

### 3.4 .env (only needed for GPT-4.1 mini)
```
OPENAI_API_KEY=sk-...
```
At the repo root. The Ollama and TigerLLM scripts don't need it.

---

## 4. Running on the lab PC

### 4.1 Sanity-check the data files exist
```bash
ls "BanglaHalluEval Datasets/banglahallueval_qa_1000.csv" "Summarization/1000 Selected Samples/banglahallueval_summarization_dataset_1000.csv" "Reasoning/1000 Selected Samples/somadhan_1000_main_ordered.csv"
ls "Hallucination Generated Answers"/{qa_4000.csv,summarization_3000_corrected.csv,reasoning_1000.csv}
```

### 4.2 Quick smoke test — 1 row per task, 1 judge
Edit the registry in `_ecot_core.py` if you want a one-row sample, OR set a low task limit by hand. Easier: run the pilot path first (50 samples per task) to confirm the env:
```bash
python pilot_50_samples/scripts/02_run_ecot.py --task qa --track A
```
If that succeeds, the lab env is correctly configured.

### 4.3 Run a single judge end-to-end
```bash
# Ollama judge
python full_ecot_run/scripts/02_run_qwen2_5_32b.py --task all --track both

# TigerLLM (HuggingFace) — make sure no other model is using the GPU first
python full_ecot_run/scripts/02_run_tigerllm_9b.py --task all --track both

# OpenAI judge (typically run from a dev machine, not the lab box)
python full_ecot_run/scripts/02_run_gpt4_1_mini.py --task all --track both
```
Each script has `--task qa|summarization|reasoning|all`, `--track A|B|both`, `--no-resume`.

### 4.4 Run all 6 open-weight judges in series
```bash
bash full_ecot_run/scripts/run_all_open_weight.sh
```
Logs go to `full_ecot_run/results/_logs/<slug>_<timestamp>.log`. The script invokes `03_compute_metrics.py` and `04_build_paper_tables.py` at the end.

### 4.5 Resume after a crash / reboot
Every script always defaults to `--resume`. To force a fresh run instead:
```bash
python full_ecot_run/scripts/02_run_<slug>.py --task all --track both --no-resume
```

### 4.6 Build the paper tables
```bash
python full_ecot_run/scripts/03_compute_metrics.py
python full_ecot_run/scripts/04_build_paper_tables.py
```
Produces:
- `results/<slug>/metrics.csv`
- `results/_all_judges_metrics.csv`
- `results/_paper_tables/{01_headline_bhs.md, 02_cot_regression_recovery.md, 03_ab_breakdown.md, 04_main_table.tex}`

---

## 5. Expected run time and disk

Estimates per judge for the **full 11,000 candidates** at temperature 0:

| Backend | Throughput | Wall-clock | GPU memory |
|---|---|---|---|
| Ollama Q4 (8B–14B) | ~2 req/s | ~90 min | 6–10 GB |
| Ollama Q4 (27B–32B) | ~0.4 req/s | ~7 h | 18–22 GB |
| Ollama Q4 DeepSeek-R1-14B | ~0.3 req/s (long thinking) | ~10 h | 10 GB + KV |
| TigerLLM HF (bfloat16) | ~0.3 req/s | ~10 h | 18 GB |
| OpenAI GPT-4.1 mini | sequential ~0.5 req/s; concurrent 10–15× faster | ~30 min sequential | n/a (API) |

Output disk: ~200 KB per CSV → ~6 MB per judge → ~50 MB total. Logs add ~10 MB.

---

## 6. Recommended execution order on the lab PC

Pre-flight on a workstation (cheap, prove the pipeline works at scale):
```bash
python full_ecot_run/scripts/02_run_gpt4_1_mini.py --task all --track both
python full_ecot_run/scripts/03_compute_metrics.py --judges gpt4_1_mini
```
This confirms the full pipeline produces sensible numbers before committing the GPU.

Then on the lab PC, in this order (smallest first → keeps you unblocked while big ones run):

1. `mistral_nemo`        (~90 min)
2. `llama3_1_8b`         (~90 min)  ← high-leverage judge: CoT regressed it most
3. `deepseek_r1_14b`     (~10 h)    ← run overnight
4. `gemma2_27b`          (~7 h)     ← run overnight
5. `qwen2_5_32b`         (~7 h)     ← run overnight
6. Stop Ollama (`pkill ollama`), then `tigerllm_9b` (~10 h, dedicated GPU)

The full sweep finishes in 3–4 working days if you start a large run overnight each day.

---

## 7. What "done" looks like

A completed run satisfies all of these:

- `full_ecot_run/results/<slug>/{qa,summarization,reasoning}_{gt,hallu}_ecot.csv` exist for all 7 judges (42 CSVs total).
- Each CSV has rows = input dataset rows (1,000 / 3,000 / 4,000) with `is_hallucinated` populated.
- `parse_error` column is empty or low (< 2%) across all CSVs.
- `_all_judges_metrics.csv` has 21 rows (7 judges × 3 tasks) with non-NaN `ecot_bhs`.
- `_paper_tables/04_main_table.tex` compiles in your paper draft.

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Ollama unreachable at http://localhost:11434` | Ollama daemon not running | `ollama serve &` then re-run script (it auto-retries) |
| Many `parse_error` rows | Smaller judges struggling with JSON | Increase `num_predict_for` budget; verify Ollama supports `format=json` |
| `CUDA out of memory` on TigerLLM | Another model still loaded | Restart Python; ensure Ollama is not also using the GPU |
| Run crashes mid-task | Network / OOM / etc. | Re-invoke the same command — `--resume` is on by default and picks up where it stopped |
| One judge's CSV has odd column count | Source CSV column names changed | Confirm `TASKS` registry in `_ecot_core.py` matches the actual column headers |
| GPT-4.1 mini run hits rate limit | Free / tier-1 keys | Switch to a higher tier or add a sleep between calls; see `_backend_openai.py` |

---

## 9. After the runs land — rebuttal next steps

See [`../pilot_50_samples/ECOT_PILOT_REPORT.md`](../pilot_50_samples/ECOT_PILOT_REPORT.md) §7.5–7.9 for:

- The three tables to extract for the rebuttal body
- The five ablations reviewers will ask for (most are free re-aggregations)
- Failure-mode analysis script template (`04_failure_modes.py`)
- Bootstrap CI + McNemar test code paths
- The rebuttal artefact registry
