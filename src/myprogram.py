#!/usr/bin/env python
import os
import pickle
import collections
import glob
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# ============================================================
# Training Configuration
# ============================================================

MAX_ORDER = 7  # Up to 7-grams (6 chars of context → predict 7th)

# Minimum count thresholds per order
MIN_COUNTS = {
    1: 0,  # Keep all unigrams (character frequencies)
    2: 5,  # Bigrams: need at least 5 occurrences
    3: 5,
    4: 3,
    5: 3,
    6: 3,
    7: 3,
}


class MyModel:
    """
    Character n-gram model with backoff for next-character prediction.
    """

    def __init__(self, ngram_model=None):
        """Initialize with n-gram model dictionary."""
        self.model = ngram_model or {}
        self.max_order = max(self.model.keys()) if self.model else 0

    @classmethod
    def load_training_data(cls, data_dir):
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

    @classmethod
    def load_test_data(cls, fname):
        """Load test data from file."""
        data = []
        with open(fname, encoding="utf-8") as f:
            for line in f:
                inp = line.rstrip("\n")  # Remove trailing newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        """Write predictions to file."""
        with open(fname, "wt", encoding="utf-8") as f:
            for p in preds:
                f.write("{}\n".format(p))

    @staticmethod
    def _count_ngrams_for_order(text, order, verbose=True):
        """Count all n-grams of a given order in the text."""
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

    @staticmethod
    def _build_top3_for_order(counter, min_count, verbose=True):
        """From an n-gram counter, extract top-3 most frequent next chars per context."""
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

    def run_train(self, text, work_dir, max_order=MAX_ORDER, min_counts=MIN_COUNTS):
        """Full training pipeline: count n-grams → prune → save."""
        # Step 1: Build model (one order at a time to control memory)
        print("=" * 60)
        print("STEP 2: Building n-gram model")
        print("=" * 60)
        model = {}

        for order in range(max_order, 0, -1):  # Start from highest order
            print(f"\n--- Order {order} ---")
            counter = self._count_ngrams_for_order(text, order)
            min_count = min_counts.get(order, 3)
            order_model = self._build_top3_for_order(counter, min_count)
            model[order] = order_model
            del counter  # Free memory immediately

        # Update model
        self.model = model
        self.max_order = max(model.keys()) if model else 0

        print()
        return model

    def predict_next_chars(self, text):
        """
        Predict top-3 next characters for given text using backoff.

        Args:
            text: Input string (will be lowercased)

        Returns:
            String of 3 characters (top-3 predictions)
        """
        # Lowercase input (grader is case-insensitive)
        text = text.lower()

        # Try backoff from highest order to lowest
        for order in range(self.max_order, 0, -1):
            if order not in self.model:
                continue

            # Context length = order - 1
            context_len = order - 1
            if len(text) < context_len:
                continue

            # Extract context (last N characters)
            context = text[-context_len:] if context_len > 0 else ""

            # Look up in model
            if context in self.model[order]:
                predictions = self.model[order][context]
                # Ensure we return exactly 3 characters (pad if needed)
                if len(predictions) >= 3:
                    return predictions[:3]
                # If we have fewer than 3, pad with fallback
                fallback = self._get_fallback()
                return (predictions + fallback)[:3]

        # Fallback: use unconditional top-3
        return self._get_fallback()

    def _get_fallback(self):
        """Get fallback prediction (unigram top-3)."""
        if 1 in self.model and "" in self.model[1]:
            fallback = self.model[1][""]
            # Ensure exactly 3 characters
            if len(fallback) >= 3:
                return fallback[:3]
            # Pad with common chars if needed
            return (fallback + " et")[:3]
        # Ultimate fallback if model is broken
        return " et"

    def run_pred(self, data):
        """Make predictions for all inputs."""
        preds = []
        for inp in data:
            top3 = self.predict_next_chars(inp)
            preds.append(top3)
        return preds

    def save(self, work_dir):
        """Save n-gram model to pickle file."""
        os.makedirs(work_dir, exist_ok=True)
        model_path = os.path.join(work_dir, "ngram_model.pkl")

        with open(model_path, "wb") as f:
            pickle.dump(self.model, f, protocol=pickle.HIGHEST_PROTOCOL)

        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"Model saved to {model_path} ({size_mb:.1f} MB)")

        # Print model stats
        total_contexts = sum(len(m) for m in self.model.values())
        print(f"Total contexts across all orders: {total_contexts:,}")
        for order in range(1, self.max_order + 1):
            if order in self.model:
                print(f"  Order {order}: {len(self.model[order]):,} contexts")

        # Show the fallback prediction (unigram top-3)
        if 1 in self.model and "" in self.model[1]:
            fallback = self.model[1][""]
            print(f"Fallback prediction (unigram top-3): '{fallback}'")

    @classmethod
    def load(cls, work_dir):
        """Load n-gram model from pickle file."""
        model_path = os.path.join(work_dir, "ngram_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "Run: python src/myprogram.py train --data_dir data/wiki --work_dir work"
            )

        print(f"Loading model from {model_path}...")
        with open(model_path, "rb") as f:
            ngram_model = pickle.load(f)

        print(
            f"Model loaded: {len(ngram_model)} orders, max_order={max(ngram_model.keys())}"
        )
        return cls(ngram_model=ngram_model)


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("mode", choices=("train", "test"), help="what to run")
    parser.add_argument("--work_dir", help="where to save", default="work")
    parser.add_argument(
        "--data_dir", help="directory with training .txt files", default="data/wiki"
    )
    parser.add_argument(
        "--test_data", help="path to test data", default="example/input.txt"
    )
    parser.add_argument(
        "--test_output", help="path to write test predictions", default="pred.txt"
    )
    args = parser.parse_args()

    if args.mode == "train":
        if not os.path.isdir(args.work_dir):
            print("Making working directory {}".format(args.work_dir))
            os.makedirs(args.work_dir)
        print("Instantiating model")
        model = MyModel()
        print("Loading training data")
        print("=" * 60)
        print("STEP 1: Loading text files")
        print("=" * 60)
        train_text = MyModel.load_training_data(args.data_dir)
        print()
        print("Training")
        model.run_train(train_text, args.work_dir)
        del train_text  # Free memory
        print("Saving model")
        print("=" * 60)
        print("STEP 3: Saving model")
        print("=" * 60)
        model.save(args.work_dir)
    elif args.mode == "test":
        print("Loading model")
        model = MyModel.load(args.work_dir)
        print("Loading test data from {}".format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print("Making predictions")
        pred = model.run_pred(test_data)
        print("Writing predictions to {}".format(args.test_output))
        assert len(pred) == len(test_data), "Expected {} predictions but got {}".format(
            len(test_data), len(pred)
        )
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError("Unknown mode {}".format(args.mode))
