#!/usr/bin/env python
"""
Download multilingual Wikipedia text for n-gram training.
Streams articles and saves lowercased plain text to disk.

Usage:
    # First, install compatible datasets version:
    pip install "datasets<3.0.0"

    # Then run:
    python src/download_data.py --output_dir data/wiki
    python src/download_data.py --output_dir data/wiki --english_only
"""

import os
import argparse
import time

# Language configs: (huggingface_wiki_config, target_num_chars)
# Total ~20M chars. English-heavy, with coverage of major scripts.
LANGUAGE_CONFIGS = {
    "en": ("20220301.en", 14_000_000),  # 70% - Latin script
    "es": ("20220301.es", 1_000_000),  #  5% - Latin script
    "fr": ("20220301.fr", 800_000),  #  4% - Latin + accents
    "de": ("20220301.de", 800_000),  #  4% - Latin + umlauts
    "ru": ("20220301.ru", 1_000_000),  #  5% - Cyrillic
    "zh": ("20220301.zh", 1_000_000),  #  5% - CJK
    "ja": ("20220301.ja", 600_000),  #  3% - Hiragana/Katakana/CJK
    "ar": ("20220301.ar", 400_000),  #  2% - Arabic script
    "ko": ("20220301.ko", 400_000),  #  2% - Hangul
}


def download_language(lang, wiki_id, target_chars, output_dir):
    """Download text for one language and save to a file."""
    try:
        from datasets import load_dataset
    except ImportError:
        print(f"[{lang}] FAILED: datasets library not installed.")
        print("  Install with: pip install 'datasets<3.0.0'")
        return False

    output_path = os.path.join(output_dir, f"{lang}.txt")
    print(f"[{lang}] Downloading (target: {target_chars:,} chars)...")

    try:
        # Use trust_remote_code for older datasets versions (<3.0.0)
        ds = load_dataset(
            "wikipedia",
            wiki_id,
            split="train",
            streaming=True,
            trust_remote_code=True,
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


def main():
    parser = argparse.ArgumentParser(
        description="Download Wikipedia text for n-gram training"
    )
    parser.add_argument(
        "--output_dir",
        default="data/wiki",
        help="Directory to save downloaded text files",
    )
    parser.add_argument(
        "--english_only",
        action="store_true",
        help="Only download English (faster for testing)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.english_only:
        configs = {"en": LANGUAGE_CONFIGS["en"]}
    else:
        configs = LANGUAGE_CONFIGS

    start = time.time()
    results = {}

    for lang, (wiki_id, target_chars) in configs.items():
        success = download_language(lang, wiki_id, target_chars, args.output_dir)
        results[lang] = success

    elapsed = time.time() - start

    # Summary
    print("\n" + "=" * 50)
    print("Download Summary")
    print("=" * 50)
    for lang, success in results.items():
        status = "OK" if success else "FAILED"
        filepath = os.path.join(args.output_dir, f"{lang}.txt")
        if success and os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  {lang}: {status} ({size_mb:.1f} MB)")
        else:
            print(f"  {lang}: {status}")

    print(f"\nTotal time: {elapsed:.1f}s ({elapsed / 60:.1f}m)")
    print(f"Files saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
