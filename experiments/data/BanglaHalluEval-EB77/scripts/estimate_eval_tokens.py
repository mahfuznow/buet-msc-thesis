import csv
from pathlib import Path

import tiktoken

PROMPT_TEMPLATE = (
    "You are an evaluator.\n"
    "Decide whether the provided model answer is hallucinated relative to the question.\n"
    "Only reply with a single token: yes or no. No explanation, no punctuation, no extra text.\n"
    "Interpretation: 'yes' means the answer contains information not supported by the question/context or is factually incorrect (hallucinated).\n"
    "Provide the answer in English only: yes or no.\n\n"
    "Question: {question}\n"
    "Model answer: {answer}\n\n"
    "Answer now:"
)


def get_encoding(model: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def build_prompt(question: str, answer: str) -> str:
    return PROMPT_TEMPLATE.format(question=question, answer=answer)


def count_tokens(text: str, enc: tiktoken.Encoding) -> int:
    return len(enc.encode(text))


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    input_csv = root / "BanglaHalluEval QA Pilot Testing" / "qa_4000_first_50.csv"
    output_csv = root / "BanglaHalluEval QA Pilot Testing" / "qa_4000_first_50_eval_gpt_5_4_20.csv"
    model = "gpt-5.4"

    enc = get_encoding(model)

    with input_csv.open(newline="", encoding="utf-8-sig") as f_in:
        rows = list(csv.DictReader(f_in))

    with output_csv.open(newline="", encoding="utf-8-sig") as f_out:
        labeled = list(csv.DictReader(f_out))

    # Only use the first 20 rows to match the pilot run
    rows = rows[:20]
    labeled = labeled[:20]

    input_tokens = []
    output_tokens = []

    for row, lab in zip(rows, labeled):
        question = row.get("question", "")
        answer = row.get("hallucinated_answer", "")
        prompt = build_prompt(question, answer)
        input_tokens.append(count_tokens(prompt, enc))

        label = lab.get("is_hallucinated", "")
        output_tokens.append(count_tokens(label, enc))

    avg_in = sum(input_tokens) / len(input_tokens)
    avg_out = sum(output_tokens) / len(output_tokens)

    total_in_4000 = avg_in * 4000
    total_out_4000 = avg_out * 4000

    print(f"Avg input tokens (20): {avg_in:.2f}")
    print(f"Avg output tokens (20): {avg_out:.2f}")
    print(f"Estimated input tokens for 4000: {total_in_4000:.0f}")
    print(f"Estimated output tokens for 4000: {total_out_4000:.0f}")


if __name__ == "__main__":
    main()
