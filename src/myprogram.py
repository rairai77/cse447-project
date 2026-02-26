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

MAX_ORDER = 10  # Up to 10-grams (9 chars of context → predict 10th)
TOP_K = 10      # Store top-K candidates with probabilities per context

# Minimum count thresholds per order (higher orders are rarer, so lower thresholds)
MIN_COUNTS = {
    1: 0,   # Keep all unigrams (character frequencies)
    2: 5,   # Bigrams: need at least 5 occurrences
    3: 5,
    4: 3,
    5: 3,
    6: 3,
    7: 3,
    8: 3,
    9: 2,
    10: 2,
}

# Default interpolation weights (used if we don't train weights).
# During prediction, only orders with matching contexts contribute.
INTERP_WEIGHTS = {
    1: 1, 2: 4, 3: 9, 4: 16, 5: 25, 6: 36, 7: 49, 8: 64, 9: 81, 10: 100,
}

# Fraction of training data held out for tuning interpolation weights (0 = skip tuning)
HELDOUT_FRACTION = 0.10
# Max held-out positions to use when tuning (keeps tuning fast)
MAX_HELDOUT_POSITIONS = 500_000


class MyModel:
    """
    Character n-gram model with interpolated scoring for next-character prediction.
    """
    INVALID_PRED_CHARS = {"\n", "\r"}

    def __init__(self, ngram_model=None, interp_weights=None):
        """Initialize with n-gram model and optional interpolation weights."""
        self.model = ngram_model or {}
        self.max_order = max(self.model.keys()) if self.model else 0
        self.interp_weights = dict(interp_weights) if interp_weights else dict(INTERP_WEIGHTS)

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
        """Count all n-grams of a given order in the text.

        For orders >= 7, periodically prunes singletons to keep memory
        manageable on large corpora.  This is safe because MIN_COUNTS for
        those orders are >= 2, so singletons would be discarded anyway.
        """
        n = len(text)
        if order > n:
            return collections.Counter()

        if verbose:
            t0 = time.time()
            print(f"  Counting {order}-grams...")

        counter = collections.defaultdict(int)
        chunk_size = min(10_000_000, n)
        use_pruning = order >= 7
        chunks_since_prune = 0
        prune_every = 5  # prune every N chunks (~50M chars)

        for start in range(0, n - order + 1, chunk_size):
            end = min(start + chunk_size + order - 1, n)
            chunk = text[start:end]

            for i in range(len(chunk) - order + 1):
                counter[chunk[i : i + order]] += 1

            chunks_since_prune += 1
            if use_pruning and chunks_since_prune >= prune_every and len(counter) > 10_000_000:
                before = len(counter)
                counter = collections.defaultdict(
                    int, {k: v for k, v in counter.items() if v > 1}
                )
                chunks_since_prune = 0
                if verbose:
                    print(f"    Memory: pruned singletons {before:,} → {len(counter):,}")

        result = collections.Counter(counter)

        if verbose:
            elapsed = time.time() - t0
            print(f"    {len(result):,} unique {order}-grams ({elapsed:.1f}s)")

        return result

    @staticmethod
    def _build_topk_for_order(counter, min_count, top_k=TOP_K, verbose=True):
        """From n-gram counter, extract top-K next chars with probabilities per context.

        Returns dict mapping context -> {char: probability, ...} for the top-K
        most frequent next characters.  Probabilities are normalized over the
        characters that survived the min_count filter.
        """
        context_chars = collections.defaultdict(list)
        for ngram, count in counter.items():
            if count >= min_count:
                context = ngram[:-1]
                next_char = ngram[-1]
                context_chars[context].append((count, next_char))

        order_model = {}
        for context, char_counts in context_chars.items():
            total = sum(c for c, _ in char_counts)
            char_counts.sort(reverse=True)
            top_entries = char_counts[:top_k]
            order_model[context] = {ch: cnt / total for cnt, ch in top_entries}

        if verbose:
            print(
                f"    {len(order_model):,} contexts after pruning (min_count={min_count})"
            )

        return order_model

    def run_train(self, text, work_dir, max_order=MAX_ORDER, min_counts=MIN_COUNTS):
        """Full training pipeline: count n-grams -> build probability tables -> save."""
        print("=" * 60)
        print("STEP 2: Building interpolated n-gram model")
        print("=" * 60)
        model = {}

        for order in range(max_order, 0, -1):
            print(f"\n--- Order {order} ---")
            counter = self._count_ngrams_for_order(text, order)
            min_count = min_counts.get(order, 3)
            order_model = self._build_topk_for_order(counter, min_count)
            model[order] = order_model
            del counter

        self.model = model
        self.max_order = max(model.keys()) if model else 0

        print()
        return model

    def _tune_interp_weights(self, heldout_text, verbose=True):
        """
        Learn interpolation weights by maximizing held-out log-likelihood.

        Uses scipy.optimize if available; otherwise keeps default INTERP_WEIGHTS.
        """
        try:
            from scipy.optimize import minimize
        except ImportError:
            if verbose:
                print("  scipy not installed; using default interpolation weights.")
                print("  Install with: pip install scipy")
            return

        text = heldout_text.lower()
        n = len(text)
        max_order = self.max_order
        if n <= max_order:
            if verbose:
                print("  Held-out too short for weight tuning.")
            return

        # Build matrix: for each position, prob of true next char from each order
        num_positions = min(n - max_order, MAX_HELDOUT_POSITIONS)
        step = max(1, (n - max_order) // num_positions)
        positions = list(range(max_order, n - 1, step))[:num_positions]

        P = []  # P[i] = (p1, p2, ..., p_max_order) for position i
        for i in positions:
            true_char = text[i]
            if true_char in self.INVALID_PRED_CHARS:
                continue
            row = []
            for order in range(1, max_order + 1):
                if order not in self.model:
                    row.append(0.0)
                    continue
                context_len = order - 1
                context = text[i - context_len : i] if context_len > 0 else ""
                dist = self.model[order].get(context, {})
                row.append(dist.get(true_char, 0.0))
            if sum(row) > 0:  # at least one order had this context
                P.append(row)

        if len(P) < 100:
            if verbose:
                print("  Too few held-out positions with matching contexts; skip tuning.")
            return

        import math

        def neg_log_likelihood(y):
            # y unconstrained; weights = softmax(y) so they sum to 1
            exp_y = [math.exp(yk) for yk in y]
            s = sum(exp_y)
            x = [e / s for e in exp_y]
            total = 0.0
            for row in P:
                interp = sum(x[o] * row[o] for o in range(max_order))
                total += math.log(max(1e-12, interp))
            return -total

        x0 = [0.0] * max_order  # softmax(0,...,0) = uniform
        result = minimize(neg_log_likelihood, x0, method="L-BFGS-B")
        if not result.success:
            if verbose:
                print("  Weight optimization did not converge; using defaults.")
            return

        exp_y = [math.exp(yk) for yk in result.x]
        s = sum(exp_y)
        self.interp_weights = {o + 1: float(exp_y[o] / s) for o in range(max_order)}
        if verbose:
            print("  Learned interpolation weights:", {k: round(v, 4) for k, v in self.interp_weights.items()})

    def predict_next_chars(self, text):
        """
        Predict top-3 next characters using interpolated n-gram scoring.

        Combines probability estimates from every order that has a matching
        context, weighted by INTERP_WEIGHTS.  Weights are renormalized over
        the orders that actually matched so missing high-order contexts don't
        dilute the signal.
        """
        text = text.lower()

        scores = collections.defaultdict(float)
        total_weight = 0.0

        for order in range(1, self.max_order + 1):
            if order not in self.model:
                continue
            context_len = order - 1
            if len(text) < context_len:
                continue
            context = text[-context_len:] if context_len > 0 else ""
            dist = self.model[order].get(context)
            if not dist:
                continue

            w = self.interp_weights.get(order, 1)
            total_weight += w
            for ch, prob in dist.items():
                if ch not in self.INVALID_PRED_CHARS:
                    scores[ch] += w * prob

        if scores and total_weight > 0:
            sorted_chars = sorted(scores.items(), key=lambda x: -x[1])
            candidates = [
                ch for ch, _ in sorted_chars if ch not in self.INVALID_PRED_CHARS
            ][:3]
            if len(candidates) >= 3:
                return "".join(candidates)
            seen = set(candidates)
            for ch in self._get_fallback():
                if ch not in seen and ch not in self.INVALID_PRED_CHARS:
                    candidates.append(ch)
                    seen.add(ch)
                    if len(candidates) == 3:
                        return "".join(candidates)

        return self._get_fallback()

    def _get_fallback(self):
        """Get fallback prediction (unigram top-3)."""
        if 1 in self.model and "" in self.model[1]:
            dist = self.model[1][""]
            if isinstance(dist, dict):
                sorted_chars = sorted(dist.items(), key=lambda x: -x[1])
                fallback = "".join(
                    ch for ch, _ in sorted_chars if ch not in self.INVALID_PRED_CHARS
                )
            else:
                fallback = "".join(
                    ch for ch in dist if ch not in self.INVALID_PRED_CHARS
                )
            if len(fallback) >= 3:
                return fallback[:3]
            return (fallback + " et")[:3]
        return " et"

    def run_pred(self, data):
        """Make predictions for all inputs."""
        preds = []
        for inp in data:
            top3 = self.predict_next_chars(inp)
            preds.append(top3)
        return preds

    def save(self, work_dir):
        """Save n-gram model and interpolation weights to pickle file."""
        os.makedirs(work_dir, exist_ok=True)
        model_path = os.path.join(work_dir, "ngram_model.pkl")

        payload = {"ngram_model": self.model, "interp_weights": self.interp_weights}
        with open(model_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"Model saved to {model_path} ({size_mb:.1f} MB)")

        # Print model stats
        total_contexts = sum(len(m) for m in self.model.values())
        print(f"Total contexts across all orders: {total_contexts:,}")
        for order in range(1, self.max_order + 1):
            if order in self.model:
                print(f"  Order {order}: {len(self.model[order]):,} contexts")

        if 1 in self.model and "" in self.model[1]:
            dist = self.model[1][""]
            if isinstance(dist, dict):
                top3 = sorted(dist.items(), key=lambda x: -x[1])[:3]
                fallback = "".join(ch for ch, _ in top3)
            else:
                fallback = dist
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
            data = pickle.load(f)

        if isinstance(data, dict) and "ngram_model" in data:
            ngram_model = data["ngram_model"]
            interp_weights = data.get("interp_weights")
        else:
            ngram_model = data
            interp_weights = None

        print(
            f"Model loaded: {len(ngram_model)} orders, max_order={max(ngram_model.keys())}"
        )
        return cls(ngram_model=ngram_model, interp_weights=interp_weights)


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
        full_text = MyModel.load_training_data(args.data_dir)
        # Hold out a fraction for tuning interpolation weights
        if HELDOUT_FRACTION > 0 and len(full_text) > 100_000:
            split = int(len(full_text) * (1 - HELDOUT_FRACTION))
            train_text = full_text[:split]
            heldout_text = full_text[split:]
            print(f"  Held out {len(heldout_text):,} chars ({HELDOUT_FRACTION*100:.0f}%) for weight tuning")
        else:
            train_text = full_text
            heldout_text = None
        del full_text
        print()
        print("Training")
        model.run_train(train_text, args.work_dir)
        del train_text
        if heldout_text is not None:
            print("=" * 60)
            print("Tuning interpolation weights on held-out data")
            print("=" * 60)
            model._tune_interp_weights(heldout_text)
            del heldout_text
        print()
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
