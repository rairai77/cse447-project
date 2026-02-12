#!/usr/bin/env python
"""
Build character n-gram model from downloaded Wikipedia text.
Processes one n-gram order at a time to control memory usage.

Usage:
    python src/build_ngrams.py --data_dir data/wiki --work_dir work
"""

import os
import pickle
import collections
import argparse
import time
import glob

# ============================================================
# Configuration
# ============================================================

MAX_ORDER = 7  # Up to 7-grams (6 chars of context → predict 7th)

# Minimum count thresholds per order
# Higher orders: lower threshold (they're naturally rarer)
# Lower orders: higher threshold (keep only strong patterns)
MIN_COUNTS = {
    1: 0,  # Keep all unigrams (character frequencies)
    2: 5,  # Bigrams: need at least 5 occurrences
    3: 5,
    4: 3,
    5: 3,
    6: 3,
    7: 3,
}


# ============================================================
# N-gram Counting (one order at a time)
# ============================================================


def load_all_text(data_dir):
    """Load all .txt files from data_dir and combine into one string."""
    text_files = glob.glob(os.path.join(data_dir, "*.txt"))
    if not text_files:
        raise ValueError(f"No .txt files found in {data_dir}")

    print(f"Loading text from {len(text_files)} files...")
    all_text = []
    total_chars = 0

    for filepath in sorted(text_files):
        lang = os.path.basename(filepath).replace(".txt", "")
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
            all_text.append(text)
            total_chars += len(text)
            print(f"  {lang}: {len(text):,} chars")

    combined = "\n".join(all_text)
    print(f"Total: {total_chars:,} chars ({total_chars / 1_000_000:.1f} MB)")
    return combined


def count_ngrams_for_order(text, order, verbose=True):
    """
    Count all n-grams of a given order in the text.

    Order N means: (N-1) chars of context + 1 predicted char = N chars total.
    E.g., order=7 counts 7-char substrings; first 6 are context, last is prediction.
    """
    n = len(text)
    if order > n:
        return collections.Counter()

    if verbose:
        t0 = time.time()
        print(f"  Counting {order}-grams...")

    # Optimized: use defaultdict for faster counting, process in chunks
    counter = collections.defaultdict(int)

    # Process in chunks to reduce memory pressure
    chunk_size = min(10_000_000, n)  # Process 10M chars at a time

    for start in range(0, n - order + 1, chunk_size):
        end = min(start + chunk_size + order - 1, n)
        chunk = text[start:end]

        # Count n-grams in this chunk
        for i in range(len(chunk) - order + 1):
            ngram = chunk[i : i + order]
            counter[ngram] += 1

    # Convert to Counter for compatibility
    result = collections.Counter(counter)

    if verbose:
        elapsed = time.time() - t0
        print(f"    {len(result):,} unique {order}-grams ({elapsed:.1f}s)")

    return result


def build_top3_for_order(counter, min_count, verbose=True):
    """
    From an n-gram counter, extract top-3 most frequent next chars per context.

    Returns: {context_string: "abc"} where "abc" are the top-3 predicted chars.
    """
    # Group by context
    context_chars = collections.defaultdict(list)
    for ngram, count in counter.items():
        if count >= min_count:
            context = ngram[:-1]  # All but last char
            next_char = ngram[-1]  # Last char (the one we predict)
            context_chars[context].append((count, next_char))

    # For each context, sort by count and keep top 3
    order_model = {}
    for context, char_counts in context_chars.items():
        char_counts.sort(reverse=True)
        top3 = "".join(c for _, c in char_counts[:3])
        order_model[context] = top3

    if verbose:
        print(
            f"    {len(order_model):,} contexts after pruning (min_count={min_count})"
        )

    return order_model


# ============================================================
# Main Training Pipeline
# ============================================================


def build_model(data_dir, work_dir, max_order=MAX_ORDER, min_counts=MIN_COUNTS):
    """Full training pipeline: load text → count → prune → save."""

    # Step 1: Load all text
    print("=" * 60)
    print("STEP 1: Loading text files")
    print("=" * 60)
    text = load_all_text(data_dir)
    print()

    # Step 2: Build model (one order at a time to control memory)
    print("=" * 60)
    print("STEP 2: Building n-gram model")
    print("=" * 60)
    model = {}

    for order in range(max_order, 0, -1):  # Start from highest order
        print(f"\n--- Order {order} ---")
        counter = count_ngrams_for_order(text, order)
        min_count = min_counts.get(order, 3)
        order_model = build_top3_for_order(counter, min_count)
        model[order] = order_model
        del counter  # Free memory immediately

    del text  # Free memory
    print()

    # Step 3: Save model
    print("=" * 60)
    print("STEP 3: Saving model")
    print("=" * 60)
    os.makedirs(work_dir, exist_ok=True)
    model_path = os.path.join(work_dir, "ngram_model.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Model saved to {model_path} ({size_mb:.1f} MB)")

    # Print model stats
    total_contexts = sum(len(m) for m in model.values())
    print(f"Total contexts across all orders: {total_contexts:,}")
    for order in range(1, max_order + 1):
        print(f"  Order {order}: {len(model[order]):,} contexts")

    # Show the fallback prediction (unigram top-3)
    if 1 in model and "" in model[1]:
        fallback = model[1][""]
        print(f"Fallback prediction (unigram top-3): '{fallback}'")
    else:
        # Compute fallback from unigram counts if not in model
        print("Note: Computing fallback from character frequencies...")

    return model


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build character n-gram model from Wikipedia text"
    )
    parser.add_argument(
        "--data_dir",
        default="data/wiki",
        help="Directory containing downloaded .txt files",
    )
    parser.add_argument(
        "--work_dir",
        default="work",
        help="Directory to save trained model",
    )
    parser.add_argument(
        "--max_order",
        type=int,
        default=MAX_ORDER,
        help=f"Maximum n-gram order (default: {MAX_ORDER})",
    )
    args = parser.parse_args()

    start = time.time()

    build_model(args.data_dir, args.work_dir, max_order=args.max_order)

    elapsed = time.time() - start
    print(f"\nTotal training time: {elapsed:.1f}s ({elapsed / 60:.1f}m)")
