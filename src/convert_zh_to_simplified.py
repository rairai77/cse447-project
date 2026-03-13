#!/usr/bin/env python3
"""
Convert data/wiki/zh.txt from Traditional to Simplified Chinese so it matches
the test set (tests/chinese uses Simplified). Run from repo root:

    pip install opencc-python-reimplemented
    python src/convert_zh_to_simplified.py

Then retrain: python src/myprogram.py train --data_dir data/wiki --work_dir work ...
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ZH_PATH = REPO / "data" / "wiki" / "zh.txt"


def main() -> None:
    if not ZH_PATH.exists():
        raise SystemExit(f"Not found: {ZH_PATH}")

    try:
        from opencc import OpenCC
    except ImportError:
        raise SystemExit(
            "Install OpenCC first: pip install opencc-python-reimplemented"
        )

    cc = OpenCC("t2s")  # Traditional → Simplified

    print(f"Reading {ZH_PATH}...")
    text = ZH_PATH.read_text(encoding="utf-8")
    n_chars = len(text)
    print(f"Converting {n_chars:,} chars (Traditional → Simplified)...")
    out = cc.convert(text)
    print(f"Writing back to {ZH_PATH}...")
    ZH_PATH.write_text(out, encoding="utf-8")
    print("Done. Retrain with: python src/myprogram.py train --data_dir data/wiki --work_dir work ...")


if __name__ == "__main__":
    main()
