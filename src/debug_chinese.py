#!/usr/bin/env python3
"""
Debug why Chinese accuracy is ~1% after converting zh to Simplified and retraining.
Run from repo root: python src/debug_chinese.py

Prints for the first few test lines: input tail, gold, predicted, and whether
the model found any context in the DB (or fell back to unigram).
"""

import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from myprogram import MyModel

WORK_DIR = REPO / "work"
INPUT = REPO / "tests" / "chinese" / "input.txt"
GOLD = REPO / "tests" / "chinese" / "answer.txt"


def main():
    if not (WORK_DIR / "ngram_model.db").exists():
        print("Train first. work/ngram_model.db not found.")
        return
    inputs = INPUT.read_text(encoding="utf-8").strip().split("\n")[:15]
    golds = GOLD.read_text(encoding="utf-8").strip().split("\n")[:15]
    model = MyModel.load(str(WORK_DIR))
    max_order = model.max_order
    print(f"Model max_order={max_order}")
    print("-" * 60)
    for i, (inp, gold) in enumerate(zip(inputs, golds)):
        text = unicodedata.normalize("NFKC", inp.lower())
        pred = model.predict_next_chars(inp)
        # Check if we got any DB hit for this input (any order)
        got_hit = False
        for order in range(1, max_order + 1):
            ctx_len = order - 1
            if len(text) >= ctx_len:
                ctx = text[-ctx_len:] if ctx_len > 0 else ""
                dist = model._get_context_distribution(order, ctx)
                if dist:
                    got_hit = True
                    break
        tail = text[-12:] if len(text) >= 12 else text
        status = "HIT" if got_hit else "FALLBACK"
        ok = "OK" if (gold.strip() and gold.strip()[0] in pred) else "WRONG"
        print(f"{i+1}. tail={repr(tail)} gold={repr(gold[:1])} pred={repr(pred)} [{status}] {ok}")
    model.close()
    print("-" * 60)
    print("If you see FALLBACK for all, the DB has no matching Chinese context (wrong script or not in train).")
    print("If you see HIT but WRONG, the model is predicting Chinese but the wrong character.")


if __name__ == "__main__":
    main()
