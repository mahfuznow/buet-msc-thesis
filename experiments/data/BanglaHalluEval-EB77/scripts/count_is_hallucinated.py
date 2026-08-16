import csv
from collections import Counter
from pathlib import Path

def main():
    p = Path("Results/summarization_3000_eval_gpt_4_1_mini.csv")
    if not p.exists():
        print(f"ERROR: file not found: {p}")
        return
    c = Counter()
    with p.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            v = r.get("is_hallucinated", "").strip().lower()
            c[v] += 1
    total = sum(c.values())
    yes = c.get("yes", 0)
    no = c.get("no", 0)
    print(f"total,{total}")
    print(f"yes,{yes},{(yes/total*100) if total else 0:.2f}%")
    print(f"no,{no},{(no/total*100) if total else 0:.2f}%")

if __name__ == '__main__':
    main()
