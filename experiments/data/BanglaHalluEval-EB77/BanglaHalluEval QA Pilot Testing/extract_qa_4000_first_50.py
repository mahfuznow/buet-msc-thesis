import csv
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    input_csv = root.parent / "Hallucination Generated Answers" / "qa_4000.csv"
    output_csv = root / "qa_4000_first_50.csv"

    if not input_csv.exists():
        raise SystemExit(f"Input CSV not found: {input_csv}")

    with input_csv.open("r", encoding="utf-8-sig", newline="") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames or []
        rows = []
        for _, row in zip(range(50), reader):
            rows.append(row)

    if not rows:
        raise SystemExit("No rows found in input CSV.")

    with output_csv.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {output_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
