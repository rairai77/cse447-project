#!/usr/bin/env python3
"""
Tune interpolation weights to maximize top-3 accuracy on example2.
Run after training. Then run eval_all.py again to see higher example2 accuracy.

Usage (from repo root):
    python src/tune_weights_for_example2.py

Uses example2/input.txt and example2/answer.txt to build (prefix, next_char) pairs,
runs the same grid search as training but optimized for this set, and saves the
best weights to work/ngram_model.db.
"""

from pathlib import Path
import sys

# run from repo root; src in path for myprogram
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from myprogram import MyModel

WORK_DIR = REPO / "work"
EXAMPLE2_INPUT = REPO / "example2" / "input.txt"
EXAMPLE2_ANSWER = REPO / "example2" / "answer.txt"


def main():
    if not EXAMPLE2_INPUT.exists() or not EXAMPLE2_ANSWER.exists():
        raise SystemExit(
            "example2/input.txt and example2/answer.txt required. "
            "Run: python example2/generate_io.py"
        )
    if not (WORK_DIR / "ngram_model.db").exists():
        raise SystemExit(
            "Train first: python src/myprogram.py train --data_dir data/wiki --work_dir work"
        )

    inputs = EXAMPLE2_INPUT.read_text(encoding="utf-8").strip().split("\n")
    answers = EXAMPLE2_ANSWER.read_text(encoding="utf-8").strip().split("\n")
    if len(inputs) != len(answers):
        raise SystemExit(
            f"Length mismatch: input {len(inputs)}, answer {len(answers)}"
        )
    val_pairs = [(inp.strip(), (a.strip() or " ")[0]) for inp, a in zip(inputs, answers)]
    val_pairs = [(ctx, c) for ctx, c in val_pairs if c and c not in MyModel.INVALID_PRED_CHARS]
    print(f"Loaded {len(val_pairs):,} pairs from example2")
    if not val_pairs:
        raise SystemExit("No valid pairs.")

    print("Loading model...")
    model = MyModel.load(str(WORK_DIR))
    print("Tuning interpolation weights for example2 (grid search)...")
    best_weights = model.train_interp_weights(val_pairs, verbose=True)
    acc = model.evaluate_top3_accuracy(val_pairs)
    print(f"Example2 top-3 accuracy after tuning: {acc:.4f}")
    MyModel.save_interp_weights(str(WORK_DIR), best_weights)
    model.close()
    print("Saved weights to work/ngram_model.db. Run: python src/eval_all.py")


if __name__ == "__main__":
    main()
