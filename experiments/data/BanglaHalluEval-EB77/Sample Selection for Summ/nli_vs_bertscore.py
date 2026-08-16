"""NLI-entailment vs BERTScore-F1 as a proxy for selecting hallucination-prone summaries.

Motivation (reviewer 1dkm, weakness 2): BERTScore F1 measures similarity of a candidate
summary to a *reference* summary.  It never looks at the source document, so it cannot
express factual consistency.  An NLI model scores P(entail | premise = source document,
hypothesis = summary) and therefore measures faithfulness to the source directly.

Two experiments:

  E1  Validity.  On the finished benchmark (1,000 gold + 3,000 GPT-5.4 hallucinated
      summaries, labels known by construction), does the NLI score separate hallucinated
      from faithful summaries?  Reported as AUC + per-hallucination-type breakdown.

  E2  Selection divergence.  On the 1,880-item BanglaCHQ-Summ pool, rank items by
      (a) mean BERTScore F1 across the 3 zero-shot model summaries (the paper's criterion)
      and (b) mean NLI entailment of those same summaries against the source document.
      Report Spearman rho and the overlap of the two bottom-1000 selections.

Model: MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
  mDeBERTa-v3-base is pretrained on CC100 (includes Bengali) and fine-tuned on XNLI +
  multilingual-NLI-26lang.  Verified: 0% UNK on Bengali; contradiction prob 0.98 on a
  reversed Bengali claim.  (The English-only cross-encoder/nli-deberta-v3-* family is
  not usable here.)
"""
import argparse
import csv
import os
import sys
import time

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.stdout.reconfigure(encoding="utf-8")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=0, help="0 = auto (64 on GPU, 16 on CPU)")
ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
ap.add_argument("--limit", type=int, default=0, help="smoke test: cap pairs per experiment")
ARGS = ap.parse_args()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "Sample Selection for Summ", "nli_vs_bertscore")
os.makedirs(OUT, exist_ok=True)

MID = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
MAXLEN = 512
MODELS = ["deepseek", "qwen", "gemma"]

DEV = ("cuda" if torch.cuda.is_available() else "cpu") if ARGS.device == "auto" else ARGS.device
BATCH = ARGS.batch or (64 if DEV == "cuda" else 16)
# DeBERTa-v3 disentangled attention can overflow in fp16; prefer bf16, else fp32.
USE_BF16 = DEV == "cuda" and torch.cuda.is_bf16_supported()

if DEV == "cpu":
    torch.set_num_threads(os.cpu_count() or 8)

tok = AutoTokenizer.from_pretrained(MID)
mdl = AutoModelForSequenceClassification.from_pretrained(MID).eval().to(DEV)
L = {v: k for k, v in mdl.config.id2label.items()}
I_ENT, I_NEU, I_CON = L["entailment"], L["neutral"], L["contradiction"]

print(f"device={DEV}  batch={BATCH}  bf16={USE_BF16}"
      + (f"  gpu={torch.cuda.get_device_name(0)}" if DEV == "cuda" else ""), flush=True)


def clean(s):
    s = (s or "").strip()
    return s if s else " "


def _forward(enc):
    """bf16 autocast on GPU, with an fp32 fallback if any non-finite logit appears."""
    if USE_BF16:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = mdl(**enc).logits
        logits = logits.float()
        if not torch.isfinite(logits).all():
            logits = mdl(**enc).logits.float()
    else:
        logits = mdl(**enc).logits.float()
    return torch.softmax(logits, -1)


def nli(pairs, tag=""):
    """pairs: list of (premise, hypothesis).  Returns list of (p_ent, p_neu, p_con).
    Length-sorted batching to minimise padding waste."""
    order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
    out = [None] * len(pairs)
    t0 = time.time()
    with torch.inference_mode():
        for bi in range(0, len(order), BATCH):
            idx = order[bi:bi + BATCH]
            prem = [clean(pairs[i][0]) for i in idx]
            hyp = [clean(pairs[i][1]) for i in idx]
            enc = tok(prem, hyp, return_tensors="pt", truncation=True,
                      max_length=MAXLEN, padding=True).to(DEV)
            p = _forward(enc).cpu()
            for j, i in enumerate(idx):
                out[i] = (float(p[j][I_ENT]), float(p[j][I_NEU]), float(p[j][I_CON]))
            if bi % (BATCH * 25) == 0:
                done = bi + len(idx)
                el = time.time() - t0
                eta = el / max(done, 1) * (len(order) - done) / 60
                print(f"  [{tag}] {done}/{len(order)}  eta {eta:.1f} min", flush=True)
    return out


def load(p):
    with open(os.path.join(ROOT, p), encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def auc(scores, labels):
    """Mann-Whitney AUC. label 1 = positive (hallucinated)."""
    pairs = sorted(zip(scores, labels))
    n1 = sum(labels); n0 = len(labels) - n1
    if n0 == 0 or n1 == 0:
        return float("nan")
    # rank with ties averaged
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    s1 = sum(r for r, (_, l) in zip(ranks, pairs) if l == 1)
    return (s1 - n1 * (n1 + 1) / 2) / (n0 * n1)


def spearman(a, b):
    def rank(x):
        o = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and x[o[j + 1]] == x[o[i]]:
                j += 1
            rr = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[o[k]] = rr
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** .5
    db = sum((y - mb) ** 2 for y in rb) ** .5
    return num / (da * db) if da and db else float("nan")


# ===================================================================== E1
print("=" * 78)
print("E1  Does NLI entailment detect hallucination?  (labels known by construction)")
print("=" * 78, flush=True)

gold = load("Summarization/1000 Selected Samples/banglahallueval_summarization_dataset_1000.csv")
hallu = load("Hallucination Generated Answers/summarization_3000_corrected.csv")

if ARGS.limit:
    gold, hallu = gold[:ARGS.limit], hallu[:ARGS.limit]

e1_pairs = [(r["question"], r["summary"]) for r in gold] + \
           [(r["document"], r["hallucinated_summary"]) for r in hallu]
e1_lab = [0] * len(gold) + [1] * len(hallu)
e1_pat = ["gold"] * len(gold) + [r["pattern"] for r in hallu]

e1 = nli(e1_pairs, "E1")

with open(os.path.join(OUT, "e1_nli_scores.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f); w.writerow(["idx", "label", "pattern", "p_entail", "p_neutral", "p_contra"])
    for i, (lab, pat, (pe, pn, pc)) in enumerate(zip(e1_lab, e1_pat, e1)):
        w.writerow([i, lab, pat, round(pe, 5), round(pn, 5), round(pc, 5)])

ent = [x[0] for x in e1]; con = [x[2] for x in e1]
faith = [e - c for e, c in zip(ent, con)]          # faithfulness score
g = [i for i, l in enumerate(e1_lab) if l == 0]
h = [i for i, l in enumerate(e1_lab) if l == 1]
mean = lambda xs: sum(xs) / len(xs)
print(f"\n  gold (n={len(g)})        entail={mean([ent[i] for i in g]):.3f}  contra={mean([con[i] for i in g]):.3f}")
print(f"  hallucinated (n={len(h)}) entail={mean([ent[i] for i in h]):.3f}  contra={mean([con[i] for i in h]):.3f}")
print(f"\n  AUC  P(contradiction)      = {auc(con, e1_lab):.3f}")
print(f"  AUC  1 - P(entailment)     = {auc([-e for e in ent], e1_lab):.3f}")
print(f"  AUC  -(entail - contra)    = {auc([-x for x in faith], e1_lab):.3f}")

print("\n  By hallucination type (AUC of -(entail-contra) vs gold):")
for pat in sorted(set(e1_pat) - {"gold"}):
    sub = [i for i in range(len(e1_lab)) if e1_pat[i] in (pat, "gold")]
    print(f"    {pat:26s} n={sum(1 for i in sub if e1_lab[i]==1):4d}  "
          f"AUC={auc([-faith[i] for i in sub], [e1_lab[i] for i in sub]):.3f}  "
          f"mean_contra={mean([con[i] for i in sub if e1_lab[i]==1]):.3f}", flush=True)

# ===================================================================== E2
print("\n" + "=" * 78)
print("E2  Selection divergence on the 1,880-item BanglaCHQ-Summ pool")
print("=" * 78, flush=True)

pool = load("Sample Selection for Summ/combined_summaries_bertscore.csv")
if ARGS.limit:
    pool = pool[:ARGS.limit]
e2_pairs, meta = [], []
for r in pool:
    for m in MODELS:
        e2_pairs.append((r["question"], r[f"{m}_summary"]))
        meta.append((r["id"], m))
e2 = nli(e2_pairs, "E2")

by_item = {}
for (rid, m), (pe, pn, pc) in zip(meta, e2):
    by_item.setdefault(rid, {})[m] = (pe, pn, pc)

rows = []
for r in pool:
    d = by_item[r["id"]]
    ents = [d[m][0] for m in MODELS]
    cons = [d[m][2] for m in MODELS]
    try:
        bs = float(r["average"])
    except (ValueError, TypeError):
        bs = float("nan")
    rows.append(dict(id=r["id"], bertscore_avg=bs,
                     nli_entail_avg=mean(ents), nli_contra_avg=mean(cons),
                     nli_faith_avg=mean(ents) - mean(cons),
                     **{f"{m}_entail": d[m][0] for m in MODELS},
                     **{f"{m}_contra": d[m][2] for m in MODELS}))

rows = [r for r in rows if r["bertscore_avg"] == r["bertscore_avg"]]  # drop NaN bertscore
with open(os.path.join(OUT, "e2_pool_scores.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

bs = [r["bertscore_avg"] for r in rows]
en = [r["nli_entail_avg"] for r in rows]
fa = [r["nli_faith_avg"] for r in rows]
print(f"\n  n = {len(rows)} items with both metrics")
print(f"  Spearman rho (BERTScore avg , NLI entail avg) = {spearman(bs, en):+.3f}")
print(f"  Spearman rho (BERTScore avg , NLI faith  avg) = {spearman(bs, fa):+.3f}")

K = 1000
sel_bs = {r["id"] for r in sorted(rows, key=lambda r: r["bertscore_avg"])[:K]}
sel_nli = {r["id"] for r in sorted(rows, key=lambda r: r["nli_faith_avg"])[:K]}
ov = len(sel_bs & sel_nli)
print(f"\n  bottom-{K} by BERTScore  vs  bottom-{K} by NLI faithfulness")
print(f"    overlap        = {ov}/{K}  ({ov/K*100:.1f}%)")
print(f"    Jaccard        = {ov/len(sel_bs | sel_nli):.3f}")
print(f"    would change   = {K-ov} items ({(K-ov)/K*100:.1f}% of the seed set)")

exp = K * K / len(rows)
print(f"    overlap expected by chance = {exp:.0f} ({exp/K*100:.1f}%)")

with open(os.path.join(OUT, "e2_selection_overlap.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "in_bertscore_bottom1000", "in_nli_bottom1000"])
    for r in rows:
        w.writerow([r["id"], int(r["id"] in sel_bs), int(r["id"] in sel_nli)])

print(f"\nwrote outputs to {OUT}")
