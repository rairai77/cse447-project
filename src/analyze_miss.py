#!/usr/bin/env python3
"""
Diagnostic: see why we miss. For each (prefix, gold) pair, check:
  (A) Is gold in at least one order's stored top-K (candidate set)?
  (B) If not, we can never rank it (TOP_NEXT / coverage limit).
  (C) If yes but we still miss, it's a ranking/weight issue.

Usage (from repo root):
    python src/analyze_miss.py
    python src/analyze_miss.py --input example2/input.txt --answer example2/answer.txt
"""

import unicodedata
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from myprogram import MyModel

WORK_DIR = REPO / "work"
DEFAULT_INPUT = REPO / "example2" / "input.txt"
DEFAULT_ANSWER = REPO / "example2" / "answer.txt"


def main():
    import argparse
    p = argparse.ArgumentParser(description="Analyze where we miss: candidate coverage vs ranking")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input lines (prefixes)")
    p.add_argument("--answer", type=Path, default=DEFAULT_ANSWER, help="Answer lines (gold next char = first char)")
    p.add_argument("--work_dir", type=Path, default=WORK_DIR, help="Model work dir")
    p.add_argument("--max_pairs", type=int, default=None, help="Cap number of pairs to analyze")
    args = p.parse_args()

    if not args.work_dir.joinpath("ngram_model.db").exists():
        raise SystemExit("Train first: python src/myprogram.py train --data_dir data/wiki --work_dir work")

    inputs = args.input.read_text(encoding="utf-8").strip().split("\n")
    answers = args.answer.read_text(encoding="utf-8").strip().split("\n")
    if len(inputs) != len(answers):
        raise SystemExit(f"Length mismatch: input {len(inputs)}, answer {len(answers)}")
    pairs = []
    for inp, a in zip(inputs, answers):
        inp = inp.strip()
        gold = (a.strip() or " ")[:1]
        if not gold or gold in MyModel.INVALID_PRED_CHARS:
            continue
        pairs.append((inp, gold))
    if args.max_pairs is not None:
        pairs = pairs[: args.max_pairs]
    print(f"Loaded {len(pairs):,} pairs from {args.input.name} / {args.answer.name}")

    print("Loading model...")
    model = MyModel.load(str(args.work_dir))

    n_total = len(pairs)
    gold_in_any_order = 0
    gold_in_no_order = 0
    hit_overall = 0
    hit_when_in_any = 0  # when gold was in at least one order's candidate set

    for prefix, gold in pairs:
        text = unicodedata.normalize("NFKC", prefix.lower())
        in_any = False
        for order in range(1, model.max_order + 1):
            ctx_len = order - 1
            if len(text) < ctx_len:
                continue
            context = text[-ctx_len:] if ctx_len > 0 else ""
            dist = model._db_lookup(order, context)
            if dist:
                chars = [c for c, _ in dist]
                if gold in chars:
                    in_any = True
                    break
        if in_any:
            gold_in_any_order += 1
        else:
            gold_in_no_order += 1

        top3 = model.predict_next_chars(prefix)
        if gold in top3:
            hit_overall += 1
            if in_any:
                hit_when_in_any += 1
        elif in_any:
            pass  # gold in candidates but not in top3 (ranking/weight issue)

    model.close()

    print()
    print("=" * 60)
    print("Diagnostic: candidate coverage vs ranking")
    print("=" * 60)
    print(f"  Total pairs:                    {n_total:,}")
    print(f"  Gold in at least one order:     {gold_in_any_order:,}  ({100 * gold_in_any_order / n_total:.1f}%)")
    print(f"  Gold in NO order (cap):        {gold_in_no_order:,}  ({100 * gold_in_no_order / n_total:.1f}%)")
    print()
    print(f"  Top-3 hit (overall):            {hit_overall:,}  ({100 * hit_overall / n_total:.1f}%)")
    if gold_in_any_order:
        print(f"  Top-3 hit when gold in candidates: {hit_when_in_any:,} / {gold_in_any_order:,}  ({100 * hit_when_in_any / gold_in_any_order:.1f}%)")
    print()
    print("Interpretation:")
    print("  - If 'Gold in NO order' is large, increase TOP_NEXT or add smoothing.")
    print("  - If 'Top-3 hit when gold in candidates' is much below 100%, tune weights or smoothing.")


if __name__ == "__main__":
    main()
