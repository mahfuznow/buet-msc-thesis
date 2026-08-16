# BanglaHalluEval Process Log

## QA Hallucination Pipeline
- Source dataset: BanglaHalluEval QA 1000 sample.
- Sampling: filtered rows with all-zero model scores, then sampled 1,000.
- Context injection: merged TyDiQA context into QA rows.
- Patterns used: factualness, comprehension, specificity, inference.
- Output: 4,000 hallucinated QA answers (4 per item).

## Summarization Hallucination Pilot Setup
- Source summaries: Sample Selection for Summ/lowest_1000_summaries.csv.
- Patterns used: Factual, Non-factual, Intrinsic.
- Example selection: 3 examples per pattern with reasons.
- Output: summarization_hallucination_examples_3per_pattern.csv.

## Next Steps
- Confirm 3 exemplar summaries (one per pattern).
- Run summarization pilot generator (3 prompts, GPT-5.4).
- Prepare 10–20 pilot summaries for end-to-end run.
