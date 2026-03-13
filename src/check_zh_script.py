#!/usr/bin/env python3
"""
Check whether Chinese test data and training data use the same script
(Simplified vs Traditional) and similar character set. Run from repo root:

    python src/check_zh_script.py
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEST_INPUT = REPO / "tests" / "chinese" / "input.txt"
TEST_ANSWER = REPO / "tests" / "chinese" / "answer.txt"
TRAIN_ZH = REPO / "data" / "wiki" / "zh.txt"


def sample_chars(path: Path, label: str, max_lines: int = 20, max_chars: int = 500) -> None:
    if not path.exists():
        print(f"  {label}: FILE NOT FOUND {path}")
        return
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    total_chars = sum(len(line) for line in lines)
    # First few hundred chars as sample
    sample = text.replace("\n", " ")[:max_chars]
    print(f"  {label}:")
    print(f"    path: {path}")
    print(f"    lines: {len(lines):,}  total chars: {total_chars:,}")
    print(f"    sample: {repr(sample[:200])}...")
    print()


def main() -> None:
    print("Chinese script / data check")
    print("=" * 60)
    sample_chars(TEST_INPUT, "Test input (tests/chinese/input.txt)")
    sample_chars(TEST_ANSWER, "Test answers (tests/chinese/answer.txt)", max_chars=200)
    sample_chars(TRAIN_ZH, "Training data (data/wiki/zh.txt)")
    print("If test sample looks Traditional (e.g. 國, 語) and train looks Simplified (e.g. 国, 语),")
    print("they don't match — add Traditional zh data or convert one side (e.g. OpenCC).")


if __name__ == "__main__":
    main()
