"""
build_qa_1000_with_context.py
-----------------------------
Reads:
  - banglahallueval_qa_dataset_1000.csv  (id, question, ..., correct_answer, ...)
  - tydiqa_goldp_bengali.csv             (split, language, id, title, context, ...)

Outputs:
  - banglahallueval_qa_1000.csv          (id, context, question, correct_answer)

The two files are joined on the 'id' column.
Rows where the id is not found in tydiqa are kept with empty context and reported.
"""

import csv
import sys
from pathlib import Path

# Force UTF-8 output so Bengali text prints correctly on Windows
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# -- paths -------------------------------------------------------------------
THIS_DIR   = Path(__file__).resolve().parent
REPO_ROOT  = THIS_DIR.parent

QA_CSV     = THIS_DIR / "banglahallueval_qa_dataset_1000.csv"
TYDIQA_CSV = REPO_ROOT / "Sample Selection for QA" / "Datasets" / "tydiqa_goldp_bengali.csv"
OUT_CSV    = THIS_DIR / "banglahallueval_qa_1000.csv"

# -- load tydiqa context lookup ----------------------------------------------
print(f"Loading TyDiQA contexts from:\n  {TYDIQA_CSV}")
context_map: dict[str, str] = {}

with TYDIQA_CSV.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        row_id  = row.get("id", "").strip()
        context = row.get("context", "").strip()
        if row_id:
            context_map[row_id] = context

print(f"  Loaded {len(context_map):,} context entries.")

# -- process QA dataset ------------------------------------------------------
print(f"\nLoading QA dataset from:\n  {QA_CSV}")
out_rows: list[dict] = []
missing_ids: list[str] = []

with QA_CSV.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        row_id      = row.get("id", "").strip()
        question    = row.get("question", "").strip()
        correct_ans = row.get("correct_answer", "").strip()

        context = context_map.get(row_id, "")
        if not context:
            missing_ids.append(row_id)

        out_rows.append({
            "id":             row_id,
            "context":        context,
            "question":       question,
            "correct_answer": correct_ans,
        })

# -- write output ------------------------------------------------------------
fieldnames = ["id", "context", "question", "correct_answer"]
with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(out_rows)

print(f"\nWrote {len(out_rows):,} rows to:\n  {OUT_CSV}")

if missing_ids:
    print(f"\nWARNING: {len(missing_ids)} IDs had no matching context in TyDiQA:")
    for mid in missing_ids[:20]:
        print(f"  {mid}")
    if len(missing_ids) > 20:
        print(f"  ... and {len(missing_ids) - 20} more.")
else:
    print("\nAll IDs matched successfully.")
