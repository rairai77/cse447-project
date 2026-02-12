#!/usr/bin/env python
"""
Download multilingual Wikipedia text for n-gram training.
Streams articles and saves lowercased plain text to disk.

Usage:
    # Install datasets library:
    pip install datasets

    # Then run:
    python src/download_data.py --output_dir data/wiki
    python src/download_data.py --output_dir data/wiki
"""

import os
import argparse
import time
from datasets import load_dataset

# Snapshot date - using November 2023 which should be available for all languages
SNAPSHOT = "20231101"

# Language configs: (huggingface_wiki_config, target_num_chars)
# Total ~20M chars. English-heavy, with coverage of major scripts.
# Using newer "wikimedia/wikipedia" dataset with recent snapshot date
LANGUAGE_CONFIGS = {
    "en": (f"{SNAPSHOT}.en", 20_000_000),
    "ceb": (f"{SNAPSHOT}.ceb", 5_000_000),
    "de": (f"{SNAPSHOT}.de", 5_000_000),
    "fr": (f"{SNAPSHOT}.fr", 5_000_000),
    "sv": (f"{SNAPSHOT}.sv", 5_000_000),
    "nl": (f"{SNAPSHOT}.nl", 5_000_000),
    "es": (f"{SNAPSHOT}.es", 5_000_000),
    "ru": (f"{SNAPSHOT}.ru", 5_000_000),
    "it": (f"{SNAPSHOT}.it", 5_000_000),
    "pl": (f"{SNAPSHOT}.pl", 5_000_000),
    "arz": (f"{SNAPSHOT}.arz", 5_000_000),
    "zh": (f"{SNAPSHOT}.zh", 5_000_000),
    "ja": (f"{SNAPSHOT}.ja", 5_000_000),
    "uk": (f"{SNAPSHOT}.uk", 5_000_000),
    "ar": (f"{SNAPSHOT}.ar", 5_000_000),
    "vi": (f"{SNAPSHOT}.vi", 5_000_000),
    "war": (f"{SNAPSHOT}.war", 5_000_000),
    "pt": (f"{SNAPSHOT}.pt", 5_000_000),
    "fa": (f"{SNAPSHOT}.fa", 5_000_000),
    "ce": (f"{SNAPSHOT}.ce", 5_000_000),
    "ca": (f"{SNAPSHOT}.ca", 5_000_000),
    "id": (f"{SNAPSHOT}.id", 5_000_000),
    "ko": (f"{SNAPSHOT}.ko", 5_000_000),
    "sr": (f"{SNAPSHOT}.sr", 5_000_000),
    "no": (f"{SNAPSHOT}.no", 5_000_000),
    "tr": (f"{SNAPSHOT}.tr", 5_000_000),
    "fi": (f"{SNAPSHOT}.fi", 5_000_000),
    "tt": (f"{SNAPSHOT}.tt", 5_000_000),
    "cs": (f"{SNAPSHOT}.cs", 5_000_000),
    "hu": (f"{SNAPSHOT}.hu", 5_000_000),
    "ro": (f"{SNAPSHOT}.ro", 5_000_000),
    "eu": (f"{SNAPSHOT}.eu", 5_000_000),
    "sh": (f"{SNAPSHOT}.sh", 5_000_000),
    "ms": (f"{SNAPSHOT}.ms", 5_000_000),
    "zh-min-nan": (f"{SNAPSHOT}.zh-min-nan", 5_000_000),
    "he": (f"{SNAPSHOT}.he", 5_000_000),
    "eo": (f"{SNAPSHOT}.eo", 5_000_000),
    "uz": (f"{SNAPSHOT}.uz", 5_000_000),
    "hy": (f"{SNAPSHOT}.hy", 5_000_000),
    "da": (f"{SNAPSHOT}.da", 5_000_000),
    "bg": (f"{SNAPSHOT}.bg", 5_000_000),
    "cy": (f"{SNAPSHOT}.cy", 5_000_000),
    "el": (f"{SNAPSHOT}.el", 5_000_000),
    "be": (f"{SNAPSHOT}.be", 5_000_000),
    "sk": (f"{SNAPSHOT}.sk", 5_000_000),
    "et": (f"{SNAPSHOT}.et", 5_000_000),
    "azb": (f"{SNAPSHOT}.azb", 5_000_000),
    "kk": (f"{SNAPSHOT}.kk", 5_000_000),
    "ur": (f"{SNAPSHOT}.ur", 5_000_000),
}


def download_language(lang, wiki_id, target_chars, output_dir):
    """Download text for one language and save to a file."""
    output_path = os.path.join(output_dir, f"{lang}.txt")
    print(f"[{lang}] Downloading (target: {target_chars:,} chars)...")

    try:
        # Use newer "wikimedia/wikipedia" dataset
        ds = load_dataset(
            "wikimedia/wikipedia",
            wiki_id,
            split="train",
            streaming=True,
        )

        chars_collected = 0
        articles_collected = 0

        with open(output_path, "w", encoding="utf-8") as f:
            for article in ds:
                text = article["text"].lower()
                f.write(text)
                f.write("\n")
                chars_collected += len(text) + 1
                articles_collected += 1

                # Progress update every 500 articles
                if articles_collected % 500 == 0:
                    print(
                        f"  [{lang}] {articles_collected} articles, {chars_collected:,} chars..."
                    )

                if chars_collected >= target_chars:
                    break

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(
            f"[{lang}] Done: {chars_collected:,} chars, {articles_collected} articles ({size_mb:.1f} MB)"
        )
        return True

    except Exception as e:
        print(f"[{lang}] FAILED: {e}")
        return False


def download_all(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    configs = LANGUAGE_CONFIGS

    start = time.time()
    results = {}

    for lang, (wiki_id, target_chars) in configs.items():
        success = download_language(lang, wiki_id, target_chars, output_dir)
        results[lang] = success

    elapsed = time.time() - start

    # Summary
    print("\n" + "=" * 50)
    print("Download Summary")
    print("=" * 50)
    for lang, success in results.items():
        status = "OK" if success else "FAILED"
        filepath = os.path.join(output_dir, f"{lang}.txt")
        if success and os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  {lang}: {status} ({size_mb:.1f} MB)")
        else:
            print(f"  {lang}: {status}")

    print(f"\nTotal time: {elapsed:.1f}s ({elapsed / 60:.1f}m)")
    print(f"Files saved to: {output_dir}/")


def main(output_dir=None):
    parser = argparse.ArgumentParser(
        description="Download Wikipedia text for n-gram training"
    )
    parser.add_argument(
        "--output_dir",
        default="data/wiki",
        help="Directory to save downloaded text files",
    )
    args = parser.parse_args()

    resolved_output_dir = args.output_dir if output_dir is None else output_dir
    download_all(resolved_output_dir)


if __name__ == "__main__":
    main()
