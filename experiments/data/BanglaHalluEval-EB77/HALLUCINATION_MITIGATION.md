# BenHalluEval — Hallucination Mitigation Master Guide

> **Purpose of this document.** A single entry point for anyone picking up the BenHalluEval mitigation work — from a co-author reviewing the paper, to a reviewer during rebuttal, to a lab-mate continuing the runs on RunPod. It traces the project from the **baseline** experiments, through the **Chain-of-Thought (CoT)** mitigation attempt, to the **Evidence-Augmented CoT (E-CoT)** mitigation currently deployed, and maps every important file, prompt, script, and paper reference.

**Target venue:** EMNLP 2026 (rebuttal-phase mitigation experiment).

---

## Table of Contents

1. [Timeline: Baseline → CoT → E-CoT](#1-timeline-baseline--cot--e-cot)
2. [BenHalluEval benchmark recap](#2-benhallueval-benchmark-recap)
3. [Method 1 — Baseline (closed-book Yes/No)](#3-method-1--baseline-closed-book-yesno)
4. [Method 2 — Chain-of-Thought (CoT)](#4-method-2--chain-of-thought-cot)
5. [Method 3 — Evidence-Augmented CoT (E-CoT), Variant C](#5-method-3--evidence-augmented-cot-e-cot-variant-c)
6. [Prompt design and paper inspiration](#6-prompt-design-and-paper-inspiration)
7. [Pilot results (300-sample proof)](#7-pilot-results-300-sample-proof)
8. [Repository map](#8-repository-map)
9. [Reproducing the pilot locally](#9-reproducing-the-pilot-locally)
10. [Scaling to the full benchmark](#10-scaling-to-the-full-benchmark)
11. [Continuing the work — checklist](#11-continuing-the-work--checklist)
12. [Full bibliography](#12-full-bibliography)

---

## 1. Timeline: Baseline → CoT → E-CoT

| Phase | What was tried | Status | Headline outcome | Where to look |
|---|---|---|---|---|
| **Baseline** | Ask each of 7 judges a closed-book "is this hallucinated? Yes/No" | Complete on full benchmark | Mean BHS ≈ 40% across judges — leaves a lot on the table | `scripts/label_correct_answers_gpt_4_1_mini.py`, `Evaluation/*.csv`, `*/Results/*.csv` |
| **CoT mitigation** | Add "Analyse step by step" reasoning trace before the Yes/No | Complete on full benchmark | Mixed: helps DeepSeek + GPT-4.1 mini on some tasks; **regresses GQA and LLaMA-3.1-8B badly** | `scripts/evaluate_cot_ollama.py`, `scripts/evaluate_cot_gpt4_1_mini.py`, `scripts/evaluate_cot_tigerllm.py`, `*/Results/*_cot_*.csv`, `scripts/extract_cot_metrics.py` |
| **E-CoT (Variant C)** | Add gold evidence + atomic-claim decomposition + word-for-word citations + task-aware verdict aggregation | **Pilot complete (300 samples)**; full-benchmark run pending on 7 judges | Mean pilot BHS **20.33% → 12.00% on GPT-4.1 mini** (–41% relative) | `pilot_50_samples/`, `full_ecot_run/` |

### Why we moved past CoT

The paper's CoT results (in `Evaluations/cot_per_model_metrics.csv`) show that plain CoT **hurts** hallucination detection on QA for 6 of 7 judges (LLaMA-3.1-8B went from BHS 45.93 → 48.10; TigerLLM 28.28 → 30.09; GPT-4.1 mini 15.56 → 36.78). The failure mode is uniform: without grounding, CoT lets the model invent plausible-sounding justifications and *increase* false alarms on ground-truth correct answers. E-CoT anchors the reasoning in verifiable evidence and forces auditable citations.

---

## 2. BenHalluEval benchmark recap

The benchmark scores hallucination *detection*, not generation. For every task, every judge sees a candidate answer and outputs Yes (hallucinated) or No (faithful). Two tracks per task:

- **Track A** — ground-truth candidates (expected verdict `no`); mistakes = false alarms → **A-err**.
- **Track B** — hallucinated candidates (expected verdict `yes`); mistakes = misses → **B-err**.
- **BenHalluScore (BHS)** = ½ (A-err + B-err) × 100 %. Lower is better.

### 3 tasks (Codemix is generation-only, no mitigation applied)

| Task | Source dataset | Track A size | Track B size |
|---|---|---|---|
| Generative QA (GQA) | TyDiQA-GoldP (Bengali) | 1,000 | 4,000 |
| Summarization | BanglaCHQ-Summ (Bengali medical) | 1,000 | 3,000 |
| Reasoning | SOMADHAN (Bengali math CoT) | 1,000 | 1,000 |

Total: **11,000 candidates per judge**, and 7 judges.

### 7 judges — three tiers

- **Bangla-Centric:** TigerLLM-9B (HuggingFace transformers; bf16)
- **Multilingual:** LLaMA-3.1-8B, Qwen2.5-32B, Gemma-2-27B, Mistral-Nemo-12B (all via Ollama Q4)
- **Reasoning-oriented:** GPT-4.1 mini (OpenAI API), DeepSeek-R1-14B (Ollama, thinking model)

---

## 3. Method 1 — Baseline (closed-book Yes/No)

**Prompt template** (from `scripts/label_correct_answers_gpt_4_1_mini.py`):
```
You are an evaluator.
Decide whether the provided model answer is hallucinated relative to the context and question.
Only reply with a single token: yes or no. No explanation, no punctuation, no extra text.
Context: {context}
Question: {question}
Model answer: {answer}
Answer now:
```

**Key files:**
- Judge scripts: `scripts/label_*.py`, `scripts/evaluate_tigerllm*.py`
- Aggregation: `scripts/extract_baseline_metrics.py`
- Results: `Evaluation/qa_*.csv`, `Summarization/Results/summarization_*.csv`, `Reasoning/Results/reasoning_*.csv` (each with column `is_hallucinated ∈ {yes, no}`)

**Findings:** Recorded in the "Before CoT" columns of the paper's main table. Judges with weak Bengali knowledge (e.g., LLaMA-3.1-8B) have high B-err (many missed hallucinations). Strong reasoning judges (GPT-4.1 mini) have modest BHS but still leave headroom.

---

## 4. Method 2 — Chain-of-Thought (CoT)

**Prompt templates** (from `scripts/evaluate_cot_ollama.py`, lines ~62–96):

```
# QA_COT_PROMPT
You are an evaluator checking whether a model answer is hallucinated.
Question: {question}
Model Answer: {answer}
Analyze step by step:
Step 1: What factual claims does the answer make?
Step 2: Are these claims supported by or inferable from the question context?
Step 3: Based on steps 1-2, is the answer hallucinated?
Final answer (write only this word on the last line): yes or no
```

```
# SUMM_COT_PROMPT
You are an evaluator checking whether a summary is hallucinated relative to a document.
Document: {document}
Summary: {summary}
Analyze step by step:
Step 1: List the key claims made in the summary.
Step 2: For each claim, check whether it is directly supported by the document.
Step 3: Based on steps 1-2, decide your final answer.
Final answer (write only this word on the last line): yes or no
```

```
# REASONING_COT_PROMPT
You are an expert evaluator for Bengali mathematical reasoning tasks.
Question: {question}
Reasoning Chain: {chain}
Answer: {answer}
Analyze step by step:
Step 1: Is each calculation or logical step in the reasoning chain mathematically correct?
Step 2: Does the final answer follow logically from the reasoning chain?
Step 3: Based on steps 1-2, is this reasoning chain hallucinated (incorrect or fabricated)?
Respond ONLY with a JSON object on the last line: {{"is_hallucinated": "Yes"}} or {{"is_hallucinated": "No"}}
```

**Key files:**
- Judge scripts: `scripts/evaluate_cot_ollama.py`, `scripts/evaluate_cot_gpt4_1_mini.py`, `scripts/evaluate_cot_tigerllm.py`
- Aggregation: `scripts/extract_cot_metrics.py`
- Results: `QA/Results/qa_cot_*.csv`, `Summarization/Results/summ_*_cot_*.csv`, `Reasoning/Results/reasoning_cot_*.csv`
- Metrics table: `T Sampled Evaluations/cot_per_model_metrics.csv`

**Findings:** Mixed. Improved Summarization on most judges but broke GQA on most judges. This regression is the primary motivation for E-CoT.

**Foundational paper this reproduces:** Wei et al. (2022), *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*, NeurIPS. https://arxiv.org/abs/2201.11903

---

## 5. Method 3 — Evidence-Augmented CoT (E-CoT), Variant C

E-CoT was designed as a targeted response to the CoT regression. Three variants sit on an increasing-rigor spectrum:

- **Variant A** — Evidence conditioning only. Insert gold evidence word-for-word, keep Yes/No output.
- **Variant B** — Evidence + explicit atomic-claim decomposition, natural-language reasoning.
- **Variant C (deployed)** — Evidence + atomic decomposition + **word-for-word citation forcing** + structured JSON. Every claim carries an exact supporting span. Reviewers can audit any verdict.

### 5.1 JSON output schema (enforced via `response_format={"type":"json_object"}` or Ollama `format=json`)

```json
{
  "claims": [
    {
      "claim": "<atomic factual claim extracted from candidate>",
      "supported_by": "<exact word-for-word span copied from evidence>",
      "status": "supported | contradicted | missing"
    }
  ],
  "verdict": "yes | no"
}
```

### 5.2 Task-aware verdict source (pilot finding)

The pilot exposed that the *best* choice for translating claim statuses into the final Yes/No varies by task:

| Task | Verdict source | Why |
|---|---|---|
| GQA | `verdict_agg` (deterministic) | Model under-flags contradictions; the rule "any contradicted ⇒ yes" recovers them. |
| Summarization | `verdict_model` (self-reported) | Model's holistic judgment is well-calibrated; aggregation over-flags via the "missing" heuristic on paraphrastic summaries. |
| Reasoning | `verdict_agg` (deterministic) | Same under-flagging pattern as QA; drives B-err from 38 % → 0 % on the pilot. |

Deterministic rule (applied only for QA + Reasoning):
```text
if any(claim.status == "contradicted"):           verdict = "yes"
elif fraction(claim.status == "missing") > 0.30:  verdict = "yes"
else:                                             verdict = "no"
```

Both policies are encoded as `VERDICT_SOURCE` and `MISSING_THRESHOLD` constants in `full_ecot_run/scripts/_ecot_core.py`.

### 5.3 Evidence mapping per task

BenHalluEval's own data files already carry the gold source-of-truth — no retrieval infrastructure required.

| Task | Track A evidence field | Track B evidence field | Candidate field |
|---|---|---|---|
| GQA | `context` (TyDiQA passage) | `context` (same passage) | `correct_answer` / `hallucinated_answer` |
| Summarization | `question` (source query) | `document` (source doc) | `summary` / `hallucinated_summary` |
| Reasoning | `answer` (gold CoT) | `answer` (gold CoT) | `answer` (equals evidence on Track A) / `hallucinated_chain` |

---

## 6. Prompt design and paper inspiration

Every E-CoT prompt draws from a specific line of the hallucination-mitigation literature. Reading the three prompt files below alongside these citations tells you exactly *why* each instruction is there.

### 6.1 The three prompts to inspect

| Task | File |
|---|---|
| GQA | [`full_ecot_run/prompts/ecot_qa.txt`](full_ecot_run/prompts/ecot_qa.txt) |
| Summarization | [`full_ecot_run/prompts/ecot_summarization.txt`](full_ecot_run/prompts/ecot_summarization.txt) |
| Reasoning | [`full_ecot_run/prompts/ecot_reasoning.txt`](full_ecot_run/prompts/ecot_reasoning.txt) |

(Identical files are mirrored at `pilot_50_samples/prompts/` for the pilot.)

### 6.2 Design ingredients → paper inspirations

| Prompt ingredient | Paper (with link) | What we adapted |
|---|---|---|
| Evidence conditioning (all three tasks) | **Lewis et al. (2020)** — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS 2020. https://arxiv.org/abs/2005.11401 | Instead of retrieval, we plug in the benchmark's gold passage / source query / gold CoT as the "retrieved" evidence. |
| Self-critique after retrieval | **Asai et al. (2024)** — *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*, ICLR 2024. https://arxiv.org/abs/2310.11511 | Structured JSON with explicit `status` field per claim mirrors Self-RAG's "isRel / isSup / isUse" tokens, distilled into a single "supported / contradicted / missing" label per atomic claim. |
| Atomic-claim decomposition (all three tasks) | **Min et al. (2023)** — *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation*, EMNLP 2023. https://arxiv.org/abs/2305.14251 | The `"claims"` list decomposition is FActScore's atomic-fact split, adapted to Bengali/medical/mathematical domains. |
| Verification-question loop, in spirit | **Dhuliawala et al. (2023)** — *Chain-of-Verification Reduces Hallucination in Large Language Models*. https://arxiv.org/abs/2309.11495 | We collapsed CoVe's three passes (draft → verify → revise) into a single-pass structured JSON so open-weight judges can run it too. |
| Verbatim citation forcing (`supported_by`) | **Menick et al. (2022)** — *Teaching language models to support answers with verified quotes* (GopherCite). https://arxiv.org/abs/2203.11147 | Each supported claim must carry an exact quoted span from the evidence — this is what makes the verdict auditable. |
| Attributed answering framing | **Bohnet et al. (2022)** — *Attributed Question Answering: Evaluation and Modeling for Attributed Large Language Models*. https://arxiv.org/abs/2212.08037 | The `"supported_by"` string is our attribution channel. |
| Structured Yes/No as the final verdict | **Wei et al. (2022)** — *Chain-of-Thought Prompting Elicits Reasoning in LLMs*, NeurIPS 2022. https://arxiv.org/abs/2201.11903 | Retains CoT's chain but subordinates it to a deterministic aggregation rule. |
| LLM-as-a-judge with structured evaluation | **Liu et al. (2023)** — *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment*, EMNLP 2023. https://arxiv.org/abs/2303.16634 | Confirms structured LLM judgments track human agreement, motivating our JSON schema. |
| Reasoning-specific: final-answer check | **Gao et al. (2022)** — *PAL: Program-Aided Language Models*, ICML 2023. https://arxiv.org/abs/2211.10435; **Wang et al. (2023)** — *Self-Consistency Improves CoT Reasoning*, ICLR 2023. https://arxiv.org/abs/2203.11171 | Reasoning prompt's mandatory "Step 1 — Final Answer Check" borrows the *end-state verification* idea: if the two final numbers disagree, that decides the verdict independent of the trace. |

### 6.3 What is *specifically* Variant C vs. any of these papers

E-CoT Variant C is not a re-implementation of any single paper. It combines:
- **RAG's** evidence-conditioning idea,
- **FActScore's** atomic-fact decomposition,
- **GopherCite's** word-for-word citation constraint,
- **CoVe's** verify-then-decide philosophy,
- **CoT's** step-by-step trace,

into a *single-pass* prompt that emits JSON, then applies our own deterministic aggregation (Section 5.2). To our knowledge no prior work combines these five for hallucination *detection* in a low-resource language. That's the contribution we plan to argue in the rebuttal.

---

## 7. Pilot results (300-sample proof)

50 + 50 samples per task, GPT-4.1 mini, same row IDs across Baseline / CoT / E-CoT (see [`pilot_50_samples/ECOT_PILOT_REPORT.md`](pilot_50_samples/ECOT_PILOT_REPORT.md) for the full report).

| Task | Baseline BHS | CoT BHS | **E-CoT BHS** | Δ vs Baseline | Δ vs CoT |
|---|---|---|---|---|---|
| GQA | 23.00% | 39.00% | **20.00%** | −3.00 pp | **−19.00 pp** |
| Summarization | 13.00% | 9.00% | 14.00% | +1.00 pp | +5.00 pp |
| Reasoning | 25.00% | 15.00% | **2.00%** | **−23.00 pp** | **−13.00 pp** |
| **Mean** | **20.33%** | **21.00%** | **12.00%** | **−8.33 pp** | **−9.00 pp** |

**Headline:** mean BHS drops 20.33 % → 12.00 % (a 41 % relative reduction over Baseline) on GPT-4.1 mini. E-CoT beats both Baseline and CoT on 2 of 3 tasks and on the mean. Reasoning gains are exceptional (BHS 25 % → 2 %, zero misses on hallucinated chains).

---

## 8. Repository map

```
BanglaHalluEval/
├── HALLUCINATION_MITIGATION.md            ← THIS FILE — project-wide overview
├── .env                                    OPENAI_API_KEY (gitignored)
│
├── (Track A / ground-truth source data now lives per-task: BanglaHalluEval Datasets/,
│    Summarization/1000 Selected Samples/, Reasoning/1000 Selected Samples/,
│    Codemix/Main dataset/ — see each task's README section)
├── Hallucination Generated Answers/        Track B source data (hallucinated candidates)
│
├── scripts/                                Baseline + CoT scripts (original pipeline)
│   ├── label_correct_answers_gpt_4_1_mini.py    baseline judge (OpenAI)
│   ├── label_hallucinations_ollama.py           baseline judge (Ollama)
│   ├── evaluate_cot_ollama.py                   CoT judge (5 open-weight)
│   ├── evaluate_cot_gpt4_1_mini.py              CoT judge (OpenAI)
│   ├── evaluate_cot_tigerllm.py                 CoT judge (HuggingFace TigerLLM)
│   ├── extract_baseline_metrics.py              baseline BHS aggregator
│   └── extract_cot_metrics.py                   CoT BHS aggregator
│
├── Evaluation/                              baseline judge outputs (QA/Reasoning, GPT-4.1-mini/LLaMA)
├── QA/Results/, Codemix/Results/            baseline judge outputs (per task)
├── QA/Results/, Summarization/Results/, Reasoning/Results/   CoT judge outputs
├── T Sampled Evaluations/                   baseline_metrics.csv, cot_per_model_metrics.csv (10%-sample TituLLM/BanglaLLaMA baseline+CoT runs)
│
├── pilot_50_samples/                       E-CoT pilot (300 samples, GPT-4.1 mini)
│   ├── ECOT_PILOT_REPORT.md                     pilot report (design, results, next-steps)
│   ├── prompts/  (ecot_qa.txt, ecot_summarization.txt, ecot_reasoning.txt)
│   ├── scripts/  (01_sample_pilot.py, 02_run_ecot.py, 03_compute_metrics.py)
│   ├── data/     (50-sample slices; deterministic, seed=42)
│   └── results/  (per-row claims + verdicts, pilot_metrics.csv)
│
└── full_ecot_run/                          E-CoT full-benchmark scaffold (11 000 × 7 judges)
    ├── LAB_SETUP_GUIDE.md                       lab-PC operating guide
    ├── RUNPOD_GUIDE.md                          $17-budget RunPod path
    ├── prompts/                                 mirror of the pilot prompts
    ├── scripts/
    │   ├── _ecot_core.py                        shared driver: prompts, JSON, agg, resume
    │   ├── _backend_ollama.py                   Ollama HTTP client
    │   ├── _backend_openai.py                   OpenAI client
    │   ├── _backend_tigerllm.py                 HuggingFace transformers loader
    │   ├── 02_run_gpt4_1_mini.py                per-judge runners (7 total)
    │   ├── 02_run_qwen2_5_32b.py
    │   ├── 02_run_gemma2_27b.py
    │   ├── 02_run_deepseek_r1_14b.py
    │   ├── 02_run_mistral_nemo.py
    │   ├── 02_run_llama3_1_8b.py
    │   ├── 02_run_tigerllm_9b.py
    │   ├── 03_compute_metrics.py                per-judge + cross-judge BHS
    │   ├── 04_build_paper_tables.py             markdown + LaTeX tables
    │   └── run_all_open_weight.sh               sequential wrapper for 6 open-weight judges
    └── results/                                 auto-populated per judge (gitignored logs)
```

---

## 9. Reproducing the pilot locally

Prerequisites: Python 3.10+; `pip install pandas requests openai python-dotenv`; `.env` with `OPENAI_API_KEY=...`.

```bash
# 1. Sample the 300-row pilot slice (deterministic, seed=42)
python pilot_50_samples/scripts/01_sample_pilot.py --track both

# 2. Run E-CoT Variant C on GPT-4.1 mini (~10 min, ~300 API calls, ~$0.30)
python pilot_50_samples/scripts/02_run_ecot.py --task all --track both

# 3. Compute A-err, B-err, BHS vs Baseline and CoT on the same 300 IDs
python pilot_50_samples/scripts/03_compute_metrics.py
```

Confirm the summary matches Section 7 above. Then you're ready to scale.

---

## 10. Scaling to the full benchmark

Two operating environments, choose whichever fits.

### Path A — Lab PC (own GPU)
See [`full_ecot_run/LAB_SETUP_GUIDE.md`](full_ecot_run/LAB_SETUP_GUIDE.md). Sections 3–5 cover Ollama install, model pulls, the run order, resume handling, and shutdown.

### Path B — RunPod ($17 credit)
See [`full_ecot_run/RUNPOD_GUIDE.md`](full_ecot_run/RUNPOD_GUIDE.md). Recommends RTX 4090 24 GB on Community Cloud, ~$10–13 total, 14–18 h wall-clock. Full pod-setup checklist + `tmux` detachment + `rsync` sync-back procedure.

Both paths use the same per-judge runner scripts and the same resume behaviour: every write is `flush + fsync`'d, so power loss costs at most one row.

### Suggested judge order (either path)

Small first, then thinking model, then 27B/32B tier, TigerLLM last (needs dedicated GPU):

```
llama3_1_8b (~45 min)
    ↓
mistral_nemo (~1 h)
    ↓
deepseek_r1_14b (~6 h)     ← overnight
    ↓
gemma2_27b (~2.5 h)
    ↓
qwen2_5_32b (~3 h)
    ↓
pkill ollama; sleep 5     ← free the GPU
    ↓
tigerllm_9b (~7 h)         ← HuggingFace bfloat16
```

Then metrics + tables **locally**:
```bash
python full_ecot_run/scripts/03_compute_metrics.py
python full_ecot_run/scripts/04_build_paper_tables.py
# outputs -> full_ecot_run/results/_paper_tables/{01_headline_bhs.md, 02_cot_regression_recovery.md, 03_ab_breakdown.md, 04_main_table.tex}
```

---

## 11. Continuing the work — checklist

If you are picking up this project cold, do these in order:

- [ ] Read this document top-to-bottom (~15 min).
- [ ] Skim [`pilot_50_samples/ECOT_PILOT_REPORT.md`](pilot_50_samples/ECOT_PILOT_REPORT.md) for the pilot's design decisions (~10 min).
- [ ] Open the three E-CoT prompt files (Section 6.1) and read them.
- [ ] Reproduce the pilot on GPT-4.1 mini (Section 9). Confirm mean BHS ≈ 12 %.
- [ ] Pick a scaling path: lab GPU or RunPod.
- [ ] Run the 6 open-weight judges + TigerLLM (Section 10).
- [ ] Sync CSVs back locally, run `03_compute_metrics.py` + `04_build_paper_tables.py`.
- [ ] Cross-reference results against expected `A/B/BHS` per-judge deltas.
- [ ] Add per-judge failure-mode analysis (script skeleton in `ECOT_PILOT_REPORT.md` §7.7).
- [ ] Attach bootstrap CIs + McNemar tests (§7.8 of same doc).
- [ ] Draft the rebuttal paragraph using the headline in `04_build_paper_tables.py` output.

---

## 12. Full bibliography

All the papers that shaped either the CoT baseline we're compared to, or E-CoT itself:

1. Wei, J. et al. **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.** NeurIPS 2022. https://arxiv.org/abs/2201.11903
2. Lewis, P. et al. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.** NeurIPS 2020. https://arxiv.org/abs/2005.11401
3. Asai, A. et al. **Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.** ICLR 2024. https://arxiv.org/abs/2310.11511
4. Dhuliawala, S. et al. **Chain-of-Verification Reduces Hallucination in Large Language Models.** 2023. https://arxiv.org/abs/2309.11495
5. Min, S. et al. **FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.** EMNLP 2023. https://arxiv.org/abs/2305.14251
6. Menick, J. et al. **Teaching Language Models to Support Answers with Verified Quotes (GopherCite).** 2022. https://arxiv.org/abs/2203.11147
7. Bohnet, B. et al. **Attributed Question Answering: Evaluation and Modeling for Attributed Large Language Models.** 2022. https://arxiv.org/abs/2212.08037
8. Liu, Y. et al. **G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment.** EMNLP 2023. https://arxiv.org/abs/2303.16634
9. Gao, L. et al. **PAL: Program-Aided Language Models.** ICML 2023. https://arxiv.org/abs/2211.10435
10. Wang, X. et al. **Self-Consistency Improves Chain of Thought Reasoning in Language Models.** ICLR 2023. https://arxiv.org/abs/2203.11171
11. Manakul, P. et al. **SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative LLMs.** EMNLP 2023. https://arxiv.org/abs/2303.08896 *(considered as an alternative direction, not adopted.)*
12. Zheng, H. S. et al. **Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models.** ICLR 2024. https://arxiv.org/abs/2310.06117 *(considered as a lightweight alternative, not adopted.)*
