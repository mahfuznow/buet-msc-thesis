#!/usr/bin/env python3
"""Assemble the rebuttal tables from `_all_judges_metrics.csv`.

Produces three Markdown tables, ready to drop into the paper:
  1. Headline BHS per (judge x task) for Baseline / CoT / E-CoT.
  2. CoT-regression-recovery: cells where CoT > Baseline and E-CoT < Baseline.
  3. A-err / B-err breakdown.

Also emits a LaTeX version of table 1 with `\\cellcolor` hooks matching the
existing Before/After CoT table style in the paper.
"""

import csv
import sys
from pathlib import Path
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent.parent
RUN_RESULTS = ROOT / "full_ecot_run" / "results"
OUT_DIR = RUN_RESULTS / "_paper_tables"

JUDGE_ORDER = [
    "deepseek_r1_14b", "gpt4_1_mini", "qwen2_5_32b", "gemma2_27b",
    "mistral_nemo", "llama3_1_8b", "tigerllm_9b",
]
JUDGE_LABEL = {
    "deepseek_r1_14b": "DeepSeek-R1-14B",
    "gpt4_1_mini":     "GPT-4.1 mini",
    "qwen2_5_32b":     "Qwen2.5-32B",
    "gemma2_27b":      "Gemma-2-27B",
    "mistral_nemo":    "Mistral-Nemo-12B",
    "llama3_1_8b":     "LLaMA-3.1-8B",
    "tigerllm_9b":     "TigerLLM-9B",
}
TASK_LABEL = {"qa": "GQA", "summarization": "Summarization", "reasoning": "Reasoning"}


def md_table_headline(df: pd.DataFrame) -> str:
    out = ["| Judge | Task | Baseline BHS | CoT BHS | **E-CoT BHS** | Δ vs Baseline | Δ vs CoT |",
           "|---|---|---|---|---|---|---|"]
    for slug in JUDGE_ORDER:
        sub = df[df["judge"] == slug]
        if sub.empty:
            continue
        for task in ("qa", "summarization", "reasoning"):
            row = sub[sub["task"] == task]
            if row.empty:
                continue
            r = row.iloc[0]
            out.append(
                f"| {JUDGE_LABEL[slug]} | {TASK_LABEL[task]} | "
                f"{r['baseline_bhs']} | {r['cot_bhs']} | **{r['ecot_bhs']}** | "
                f"{r['delta_vs_baseline']} | {r['delta_vs_cot']} |"
            )
    return "\n".join(out)


def md_table_regression_recovery(df: pd.DataFrame) -> str:
    regressed = df[(df["cot_bhs"] > df["baseline_bhs"]) & (df["cot_bhs"] == df["cot_bhs"])]
    out = ["| Judge | Task | Baseline | CoT (worse) | **E-CoT** | Recovered? |",
           "|---|---|---|---|---|---|"]
    for _, r in regressed.iterrows():
        recovered = "YES" if r["ecot_bhs"] < r["baseline_bhs"] else "partial" if r["ecot_bhs"] < r["cot_bhs"] else "no"
        out.append(
            f"| {JUDGE_LABEL[r['judge']]} | {TASK_LABEL[r['task']]} | "
            f"{r['baseline_bhs']} | {r['cot_bhs']} | **{r['ecot_bhs']}** | {recovered} |"
        )
    return "\n".join(out)


def md_table_ab_breakdown(df: pd.DataFrame) -> str:
    out = ["| Judge | Task | Baseline A/B/BHS | CoT A/B/BHS | E-CoT A/B/BHS |",
           "|---|---|---|---|---|"]
    for slug in JUDGE_ORDER:
        sub = df[df["judge"] == slug]
        if sub.empty:
            continue
        for task in ("qa", "summarization", "reasoning"):
            row = sub[sub["task"] == task]
            if row.empty:
                continue
            r = row.iloc[0]
            out.append(
                f"| {JUDGE_LABEL[slug]} | {TASK_LABEL[task]} | "
                f"{r['baseline_a_err']} / {r['baseline_b_err']} / {r['baseline_bhs']} | "
                f"{r['cot_a_err']} / {r['cot_b_err']} / {r['cot_bhs']} | "
                f"{r['ecot_a_err']} / {r['ecot_b_err']} / {r['ecot_bhs']} |"
            )
    return "\n".join(out)


def latex_table_main(df: pd.DataFrame) -> str:
    """Mirror the Before/After CoT table style, adding an E-CoT column."""
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\setlength{\tabcolsep}{6pt}",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\begin{tabular}{ll|c|c|c}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Task} & \textbf{Baseline BHS} & \textbf{CoT BHS} & \textbf{E-CoT BHS} \\",
        r"\midrule",
    ]
    for slug in JUDGE_ORDER:
        sub = df[df["judge"] == slug]
        if sub.empty:
            continue
        first = True
        for task in ("qa", "summarization", "reasoning"):
            row = sub[sub["task"] == task]
            if row.empty: continue
            r = row.iloc[0]
            color = ""
            if r["ecot_bhs"] == r["ecot_bhs"] and r["baseline_bhs"] == r["baseline_bhs"]:
                color = r"\cellcolor{cellbest}" if r["ecot_bhs"] < r["baseline_bhs"] else r"\cellcolor{cellworst}"
            label = f"\\multirow{{3}}{{*}}{{{JUDGE_LABEL[slug]}}}" if first else ""
            first = False
            lines.append(f"  {label} & {TASK_LABEL[task]} & "
                         f"{r['baseline_bhs']} & {r['cot_bhs']} & {color}\\textbf{{{r['ecot_bhs']}}} \\\\")
        lines.append(r"\midrule")
    lines = lines[:-1] + [r"\bottomrule", r"\end{tabular}",
                          r"\caption{BenHalluScore: Baseline vs CoT vs E-CoT across 7 judges and 3 tasks.}",
                          r"\label{tab:ecot_full}", r"\end{table*}"]
    return "\n".join(lines)


def main():
    src = RUN_RESULTS / "_all_judges_metrics.csv"
    if not src.exists():
        raise SystemExit(f"Run 03_compute_metrics.py first ({src} missing).")
    df = pd.read_csv(src)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    artefacts = {
        "01_headline_bhs.md":          md_table_headline(df),
        "02_cot_regression_recovery.md": md_table_regression_recovery(df),
        "03_ab_breakdown.md":          md_table_ab_breakdown(df),
        "04_main_table.tex":           latex_table_main(df),
    }
    for name, body in artefacts.items():
        (OUT_DIR / name).write_text(body + "\n", encoding="utf-8")
        print(f"  -> {OUT_DIR.relative_to(ROOT) / name}")


if __name__ == "__main__":
    main()
