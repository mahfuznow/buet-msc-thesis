#!/usr/bin/env python3
"""Compute BenHalluScore on the 50+50 pilot for baseline / CoT / E-CoT.

Track A (GT, expected="no")         => "yes" prediction = false alarm  -> A-err
Track B (hallu, expected="yes")     => "no"  prediction = miss          -> B-err
BHS = 0.5 * (A-err + B-err) [%]      ; lower is better

E-CoT uses the deterministic aggregated verdict (`verdict_agg`) — produced by the
post-processing rule "any claim 'contradicted' OR >30% 'missing' => yes" — because
GPT-4.1 mini's self-reported verdict was violating its own instruction-level rule
(see pilot v1/v2 divergence analysis).

Outputs: pilot_50_samples/results/pilot_metrics.csv (one row per task)
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
PILOT_DIR = ROOT / "pilot_50_samples"
DATA_DIR = PILOT_DIR / "data"
RESULTS_DIR = PILOT_DIR / "results"


TASKS = [
    {
        "task":             "QA",
        # Track A
        "pilot_A":          DATA_DIR / "qa_gt_50.csv",
        "baseline_A":       ROOT / "Evaluation" / "qa_1000_eval_gpt_4_1_mini.csv",
        "cot_A":            ROOT / "QA" / "Results" / "qa_cot_gt_gpt4_1_mini.csv",
        "ecot_A":           RESULTS_DIR / "qa_gt_50_ecot.csv",
        "key_pilot_A":      "id",
        "key_full_A":       "id",
        # Track B
        "pilot_B":          DATA_DIR / "qa_hallu_50.csv",
        "baseline_B":       ROOT / "Evaluation" / "qa_4000_eval_gpt_4_1_mini.csv",
        "cot_B":            ROOT / "QA" / "Results" / "qa_cot_hallu_gpt4_1_mini.csv",
        "ecot_B":           RESULTS_DIR / "qa_hallu_50_ecot.csv",
        "key_pilot_B":      "id",
        "key_full_B":       "id",
    },
    {
        "task":             "Summarization",
        "pilot_A":          DATA_DIR / "summarization_gt_50.csv",
        "baseline_A":       ROOT / "Results" / "summarization_dataset_1000_eval_gpt_4_1_mini.csv",
        "cot_A":            ROOT / "Summarization" / "Results" / "summ_1000_cot_gpt4_1_mini.csv",
        "ecot_A":           RESULTS_DIR / "summarization_gt_50_ecot.csv",
        "key_pilot_A":      "id",
        "key_full_A":       "id",
        "pilot_B":          DATA_DIR / "summarization_hallu_50.csv",
        "baseline_B":       ROOT / "Results" / "summarization_3000_eval_gpt_4_1_mini.csv",
        "cot_B":            ROOT / "Summarization" / "Results" / "summ_3000_cot_gpt4_1_mini.csv",
        "ecot_B":           RESULTS_DIR / "summarization_hallu_50_ecot.csv",
        "key_pilot_B":      "id",
        "key_full_B":       "id",
    },
    {
        "task":             "Reasoning",
        "pilot_A":          DATA_DIR / "reasoning_gt_50.csv",
        "baseline_A":       ROOT / "Results" / "reasoning_main_1000_eval_gpt_4_1_mini.csv",
        "cot_A":            ROOT / "Reasoning" / "Results" / "reasoning_gt_cot_gpt4_1_mini.csv",
        "ecot_A":           RESULTS_DIR / "reasoning_gt_50_ecot.csv",
        "key_pilot_A":      "question",
        "key_full_A":       "question",
        "pilot_B":          DATA_DIR / "reasoning_hallu_50.csv",
        "baseline_B":       ROOT / "Results" / "reasoning_1000_eval_gpt_4_1_mini.csv",
        "cot_B":            ROOT / "Reasoning" / "Results" / "reasoning_cot_gpt4_1_mini.csv",
        "ecot_B":           RESULTS_DIR / "reasoning_hallu_50_ecot.csv",
        "key_pilot_B":      "question",
        "key_full_B":       "question",
    },
]


def yes_no(s) -> str:
    if s is None:
        return "unknown"
    t = str(s).strip().lower()
    if t.startswith("y"):
        return "yes"
    if t.startswith("n"):
        return "no"
    return "unknown"


def count_preds(df: pd.DataFrame, col: str = "is_hallucinated"):
    if df is None or len(df) == 0 or col not in df.columns:
        return 0, 0, 0
    preds = df[col].apply(yes_no).tolist()
    yes_c = sum(1 for p in preds if p == "yes")
    no_c = sum(1 for p in preds if p == "no")
    unk_c = sum(1 for p in preds if p == "unknown")
    return yes_c, no_c, unk_c


VERDICT_SOURCE = {
    "QA":            "verdict_agg",
    "Summarization": "verdict_model",
    "Reasoning":     "verdict_agg",
}


def count_ecot_preds(df: pd.DataFrame, task: str):
    """E-CoT uses task-aware verdict source.

    Aggregated verdict for QA + Reasoning (model under-flags real contradictions);
    model self-reported verdict for Summarization (aggregation over-flags via missing).
    """
    if df is None or len(df) == 0:
        return 0, 0, 0
    col = VERDICT_SOURCE.get(task, "verdict_agg")
    if col not in df.columns:
        col = "is_hallucinated"
    return count_preds(df, col=col)


def load_or_warn(path: Path):
    if not path.exists():
        print(f"  [!] missing: {path}")
        return None
    return pd.read_csv(path, on_bad_lines="skip")


def subset_by_ids(df: pd.DataFrame, key: str, ids):
    if df is None:
        return None
    df = df.copy()
    df[key] = df[key].astype(str).str.strip()
    return df[df[key].isin(ids)]


def main() -> None:
    rows = []
    for cfg in TASKS:
        task = cfg["task"]
        print(f"\n[{task}]")

        # ── Track A ────────────────────────────────────────────────────────
        pilotA = pd.read_csv(cfg["pilot_A"])
        kPA, kFA = cfg["key_pilot_A"], cfg["key_full_A"]
        pilotA[kPA] = pilotA[kPA].astype(str).str.strip()
        idsA = pilotA[kPA].tolist()
        N_A = len(idsA)

        baseA = subset_by_ids(load_or_warn(cfg["baseline_A"]), kFA, idsA)
        cotA  = subset_by_ids(load_or_warn(cfg["cot_A"]),      kFA, idsA)
        ecotA = subset_by_ids(load_or_warn(cfg["ecot_A"]),     kPA, idsA)

        ya_bl, _, ua_bl = count_preds(baseA)        # yes on GT = false alarm
        ya_ct, _, ua_ct = count_preds(cotA)
        ya_ec, _, ua_ec = count_ecot_preds(ecotA, task)   # uses verdict_agg

        a_err_bl = round(ya_bl / N_A * 100, 2)
        a_err_ct = round(ya_ct / N_A * 100, 2)
        a_err_ec = round(ya_ec / N_A * 100, 2)

        print(f"  Track A (n={N_A})  baseline yes={ya_bl} A-err={a_err_bl:.2f}% | "
              f"CoT yes={ya_ct} A-err={a_err_ct:.2f}% | "
              f"E-CoT yes={ya_ec} A-err={a_err_ec:.2f}%")

        # ── Track B ────────────────────────────────────────────────────────
        pilotB = pd.read_csv(cfg["pilot_B"])
        kPB, kFB = cfg["key_pilot_B"], cfg["key_full_B"]
        pilotB[kPB] = pilotB[kPB].astype(str).str.strip()
        idsB = pilotB[kPB].tolist()
        N_B = len(idsB)

        baseB = subset_by_ids(load_or_warn(cfg["baseline_B"]), kFB, idsB)
        cotB  = subset_by_ids(load_or_warn(cfg["cot_B"]),      kFB, idsB)
        ecotB = subset_by_ids(load_or_warn(cfg["ecot_B"]),     kPB, idsB)

        # On hallucinated data, "no" = miss = wrong. unknown is also a miss.
        _, no_bl, uB_bl = count_preds(baseB)
        _, no_ct, uB_ct = count_preds(cotB)
        _, no_ec, uB_ec = count_ecot_preds(ecotB, task)   # uses verdict_agg

        wrong_B_bl = no_bl + uB_bl
        wrong_B_ct = no_ct + uB_ct
        wrong_B_ec = no_ec + uB_ec

        # Note: matching baselines may have fewer than 50 rows if some IDs
        # missing — guard against div-by-zero by using actual matched count
        nB_bl = len(baseB) if baseB is not None else 0
        nB_ct = len(cotB)  if cotB  is not None else 0
        nB_ec = len(ecotB) if ecotB is not None else 0

        def pct(num, den):
            return round(num / den * 100, 2) if den > 0 else float("nan")

        b_err_bl = pct(wrong_B_bl, nB_bl)
        b_err_ct = pct(wrong_B_ct, nB_ct)
        b_err_ec = pct(wrong_B_ec, nB_ec)

        print(f"  Track B (n={N_B})  baseline wrong={wrong_B_bl}/{nB_bl} B-err={b_err_bl:.2f}% | "
              f"CoT wrong={wrong_B_ct}/{nB_ct} B-err={b_err_ct:.2f}% | "
              f"E-CoT wrong={wrong_B_ec}/{nB_ec} B-err={b_err_ec:.2f}%")

        # ── BHS ────────────────────────────────────────────────────────────
        bhs_bl = round(0.5 * (a_err_bl + b_err_bl), 2)
        bhs_ct = round(0.5 * (a_err_ct + b_err_ct), 2)
        bhs_ec = round(0.5 * (a_err_ec + b_err_ec), 2)

        print(f"  BHS  baseline={bhs_bl:.2f}%  CoT={bhs_ct:.2f}%  E-CoT={bhs_ec:.2f}%")
        print(f"  Δ E-CoT vs baseline: {bhs_ec - bhs_bl:+.2f}pp | vs CoT: {bhs_ec - bhs_ct:+.2f}pp")

        rows.append({
            "task":             task,
            "n_A":              N_A,
            "n_B":              N_B,
            # Track A
            "baseline_a_err":   a_err_bl,
            "cot_a_err":        a_err_ct,
            "ecot_a_err":       a_err_ec,
            # Track B
            "baseline_b_err":   b_err_bl,
            "cot_b_err":        b_err_ct,
            "ecot_b_err":       b_err_ec,
            # BHS
            "baseline_bhs":     bhs_bl,
            "cot_bhs":          bhs_ct,
            "ecot_bhs":         bhs_ec,
            "delta_bhs_vs_baseline": round(bhs_ec - bhs_bl, 2),
            "delta_bhs_vs_cot":      round(bhs_ec - bhs_ct, 2),
        })

    out_path = RESULTS_DIR / "pilot_metrics.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved -> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
