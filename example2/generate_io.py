#!/usr/bin/env python3
"""Generate input/answer files from phrase lines.

Each output pair follows the example format:
- input line: phrase prefix
- answer line: immediate next character after that prefix
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phrases",
        type=Path,
        default=Path("example2/phrases.txt"),
        help="Path to source phrases file (one phrase per line).",
    )
    parser.add_argument(
        "--input-out",
        type=Path,
        default=Path("example2/input.txt"),
        help="Path to write generated input data.",
    )
    parser.add_argument(
        "--answer-out",
        type=Path,
        default=Path("example2/answer.txt"),
        help="Path to write generated expected answers.",
    )
    parser.add_argument(
        "--mode",
        choices=("fractions", "all"),
        default="fractions",
        help=(
            "fractions: create a few split points per phrase; "
            "all: create every possible split."
        ),
    )
    parser.add_argument(
        "--fractions",
        default="0.5,0.75,0.9",
        help="Comma-separated fractions used in fractions mode.",
    )
    parser.add_argument(
        "--min-prefix",
        type=int,
        default=2,
        help="Minimum prefix length to keep.",
    )
    return parser.parse_args()


def split_points_all(length: int, min_prefix: int) -> list[int]:
    return list(range(min_prefix, length))


def split_points_fractions(length: int, min_prefix: int, fractions: list[float]) -> list[int]:
    points = []
    max_prefix = length - 1
    for frac in fractions:
        idx = round(length * frac)
        idx = max(min_prefix, min(idx, max_prefix))
        points.append(idx)
    return sorted(set(points))


def main() -> None:
    args = parse_args()

    fractions = []
    if args.mode == "fractions":
        fractions = [float(x.strip()) for x in args.fractions.split(",") if x.strip()]
        if not fractions:
            raise ValueError("At least one fraction is required in fractions mode.")

    phrases = args.phrases.read_text(encoding="utf-8").splitlines()
    inputs: list[str] = []
    answers: list[str] = []

    for phrase in phrases:
        if len(phrase) <= args.min_prefix:
            continue

        if args.mode == "all":
            points = split_points_all(len(phrase), args.min_prefix)
        else:
            points = split_points_fractions(len(phrase), args.min_prefix, fractions)

        for idx in points:
            inputs.append(phrase[:idx])
            answers.append(phrase[idx])

    args.input_out.write_text("\n".join(inputs) + "\n", encoding="utf-8")
    args.answer_out.write_text("\n".join(answers) + "\n", encoding="utf-8")

    print(
        f"Wrote {len(inputs)} examples to {args.input_out} and {args.answer_out}."
    )


if __name__ == "__main__":
    main()
