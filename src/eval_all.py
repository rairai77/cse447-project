#!/usr/bin/env python3
"""
Run the trained model on all built-in evaluation sets and summarize metrics.

Datasets covered:
- example/          (input.txt, answer.txt)
- example2/         (input.txt, answer.txt)  – if generated via generate_io.py
- tests/<lang>/     (input.txt, answer.txt)  – one folder per language

Usage (from repo root, after training to work/ngram_model.db):

    python src/eval_all.py

This prints a table with:
- dataset name
- number of examples
- success rate (fraction where gold char is in top-3 prediction)
- optional prediction file path
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from myprogram import MyModel


REPO_ROOT = Path(__file__).resolve().parents[1]
WORK_DIR = REPO_ROOT / "work"


@dataclass
class EvalResult:
    name: str
    n_examples: int
    n_correct: int

    @property
    def success_rate(self) -> float:
        if self.n_examples == 0:
            return 0.0
        return self.n_correct / self.n_examples


def load_lines(fname: Path) -> List[str]:
    with fname.open(encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def score_predictions(pred_path: Path, gold_path: Path) -> EvalResult:
    pred = load_lines(pred_path)
    gold = [g.lower() for g in load_lines(gold_path)]

    # Align lengths (pad predictions if too short)
    if len(pred) < len(gold):
        pred.extend([""] * (len(gold) - len(pred)))

    correct = 0
    for p, g in zip(pred, gold):
        p = (p or "").lower()
        # Only care about first 3 chars of prediction
        p = p[:3]
        if g and g[0] in p:
            correct += 1

    return EvalResult(
        name=gold_path.parent.name,
        n_examples=len(gold),
        n_correct=correct,
    )


def run_dataset(model: MyModel, input_path: Path, pred_path: Path, gold_path: Path) -> EvalResult:
    # Load inputs
    data = MyModel.load_test_data(str(input_path))

    # Run model
    pred = model.run_pred(data)

    # Sanity check and write predictions
    assert len(pred) == len(
        data
    ), f"Expected {len(data)} predictions for {input_path}, got {len(pred)}"
    MyModel.write_pred(pred, str(pred_path))

    # Score
    return score_predictions(pred_path, gold_path)


def find_test_datasets() -> list[tuple[str, Path, Path, Path]]:
    """
    Return list of (name, input_path, pred_path, gold_path).

    Includes:
    - example/
    - example2/ (if answer.txt exists)
    - tests/<lang>/ for all languages with input.txt + answer.txt
    """
    datasets: list[tuple[str, Path, Path, Path]] = []

    # example/
    example_dir = REPO_ROOT / "example"
    if (example_dir / "input.txt").exists() and (example_dir / "answer.txt").exists():
        datasets.append(
            (
                "example",
                example_dir / "input.txt",
                example_dir / "pred.txt",
                example_dir / "answer.txt",
            )
        )

    # example2/
    example2_dir = REPO_ROOT / "example2"
    if (example2_dir / "input.txt").exists() and (example2_dir / "answer.txt").exists():
        datasets.append(
            (
                "example2",
                example2_dir / "input.txt",
                example2_dir / "pred.txt",
                example2_dir / "answer.txt",
            )
        )

    # tests/<lang>/
    tests_root = REPO_ROOT / "tests"
    if tests_root.is_dir():
        for lang_dir in sorted(p for p in tests_root.iterdir() if p.is_dir()):
            name = f"tests/{lang_dir.name}"
            input_path = lang_dir / "input.txt"
            gold_path = lang_dir / "answer.txt"
            pred_path = lang_dir / "pred.txt"
            if input_path.exists() and gold_path.exists():
                datasets.append((name, input_path, pred_path, gold_path))

    return datasets


def format_rate(x: float) -> str:
    return f"{x*100:6.2f}%"


def main() -> None:
    if not WORK_DIR.exists():
        raise SystemExit(
            f"work/ directory not found at {WORK_DIR}. "
            "Train first with: python src/myprogram.py train --data_dir data/wiki --work_dir work"
        )

    datasets = find_test_datasets()
    if not datasets:
        raise SystemExit("No evaluation datasets found (example/, example2/, or tests/*).")

    print(f"Loading model from {WORK_DIR}...")
    model = MyModel.load(str(WORK_DIR))

    results: list[EvalResult] = []
    print()
    print("Running evaluations...")
    for name, input_path, pred_path, gold_path in datasets:
        print(f"  - {name}: {input_path} (gold: {gold_path})")
        res = run_dataset(model, input_path, pred_path, gold_path)
        results.append(res)

    # Optional explicit close
    model.close()

    # Summaries
    print()
    print("=" * 72)
    print(f"{'Dataset':30} {'N':>8} {'Success':>10}")
    print("-" * 72)

    total_examples = 0
    total_correct = 0
    for res in results:
        total_examples += res.n_examples
        total_correct += res.n_correct
        print(
            f"{res.name:30} {res.n_examples:8d} {format_rate(res.success_rate):>10}"
        )

    print("-" * 72)
    macro_avg = sum(r.success_rate for r in results) / len(results)
    micro_rate = total_correct / total_examples if total_examples else 0.0
    print(
        f"{'MACRO avg':30} {' ':8} {format_rate(macro_avg):>10}"
    )
    print(
        f"{'MICRO overall':30} {total_examples:8d} {format_rate(micro_rate):>10}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()

