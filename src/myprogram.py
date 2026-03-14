#!/usr/bin/env python
import os
import json
import unicodedata
import collections
import glob
import itertools
import time
import sqlite3
from functools import lru_cache
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# ============================================================
# Training Configuration
# ============================================================

MAX_ORDER = 9

# Interpolation weights: P(c|text) = sum over orders k of lambda_k * P_k(c|context_k).
# Higher orders get more weight; sum must be 1.
_INTERP_RAW = {
    1: 0.02,
    2: 0.03,
    3: 0.05,
    4: 0.10,
    5: 0.15,
    6: 0.20,
    7: 0.20,
    8: 0.15,
    9: 0.10,
    10: 0.10,
}
_total = sum(_INTERP_RAW.values())
INTERP_WEIGHTS = {k: v / _total for k, v in _INTERP_RAW.items()}

# Minimum count thresholds per order (lower high-order = keep more rare scripts)
MIN_COUNTS = {
    1: 0,  # Keep all unigrams (character frequencies)
    2: 5,  # Bigrams: need at least 5 occurrences
    3: 5,
    4: 3,
    5: 3,
    6: 3,
    7: 3,
    8: 3,
    9: 3,
}


class MyModel:
    """
    Character n-gram model with backoff for next-character prediction.

    Drop-in replacement for the pickle-backed version, but persists and reads
    from SQLite with one table per order and an LRU cache for DB lookups.
    """

    INVALID_PRED_CHARS = {"\n", "\r"}

    def __init__(self, ngram_model=None, db_path=None):
        # In training, we keep the in-memory dict model (fast to build).
        self.model = ngram_model or {}
        self.max_order = max(self.model.keys()) if self.model else 0

        # Interpolation weights (order -> weight); used in predict_next_chars.
        self.interp_weights = dict(INTERP_WEIGHTS)

        # In test/inference, we use SQLite
        self.db_path = db_path
        self.conn = None
        self._db_ready = False  # becomes True after load()

    # -----------------------------
    # Data I/O
    # -----------------------------
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
            print(f"  {lang}: ", end="", flush=True)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            text = unicodedata.normalize("NFKC", text)
            all_text.append(text)
            total_chars += len(text)
            print(f"{len(text):,} chars", flush=True)

        print("Joining corpus...", flush=True)
        combined = "\n".join(all_text)
        print(f"Total: {total_chars:,} chars ({total_chars / 1_000_000:.1f} MB)")
        return combined

    @classmethod
    def load_training_data_split(cls, data_dir, val_ratio=0.2):
        """
        Load all .txt files and split per-file into train/val so every language
        contributes to training. Returns (train_text, val_text).
        (Fixes Chinese ~1% when zh was last file and ended up entirely in val.)
        """
        text_files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
        if not text_files:
            raise ValueError(f"No .txt files found in {data_dir}")
        print(f"Loading text from {len(text_files)} files (per-file {1-val_ratio:.0%} train / {val_ratio:.0%} val)...")
        train_parts = []
        val_parts = []
        total_train = 0
        total_val = 0
        for filepath in text_files:
            lang = os.path.basename(filepath).replace(".txt", "")
            print(f"  {lang}: ", end="", flush=True)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            text = unicodedata.normalize("NFKC", text)
            idx = int(len(text) * (1 - val_ratio))
            train_part = text[:idx]
            val_part = text[idx:]
            train_parts.append(train_part)
            val_parts.append(val_part)
            total_train += len(train_part)
            total_val += len(val_part)
            print(f"{len(train_part):,} train, {len(val_part):,} val", flush=True)
        train_text = "\n".join(train_parts)
        val_text = "\n".join(val_parts)
        print(f"Total: {total_train:,} train chars, {total_val:,} val chars")
        return train_text, val_text

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

    # -----------------------------
    # Validation for interpolation weight tuning
    # -----------------------------
    @staticmethod
    def build_validation_pairs(text, max_order, max_pairs=50000, invalid_chars=None):
        """
        Build (context, target_char) pairs from text for top-3 accuracy evaluation.
        context is the preceding characters (length up to max_order-1).
        """
        invalid_chars = invalid_chars or MyModel.INVALID_PRED_CHARS
        n = len(text)
        ctx_len = max_order - 1
        if n <= ctx_len:
            return []
        positions = list(range(ctx_len, n))
        if len(positions) > max_pairs:
            step = max(1, len(positions) // max_pairs)
            positions = positions[::step][:max_pairs]
        pairs = []
        for i in positions:
            target = text[i]
            if target in invalid_chars:
                continue
            context = text[max(0, i - ctx_len):i]
            pairs.append((context, target))
        return pairs

    def evaluate_top3_accuracy(self, pairs):
        """Compute top-3 accuracy: fraction of pairs where target is in predicted top-3."""
        if not pairs:
            return 0.0
        correct = 0
        for context, target in pairs:
            pred = self.predict_next_chars(context)
            if target in pred:
                correct += 1
        return correct / len(pairs)

    def train_interp_weights(self, val_pairs, verbose=True):
        """
        Grid search over interpolation weights to maximize top-3 accuracy on val_pairs.
        Returns best weight dict (order -> float).
        """
        best_acc = -1.0
        best_weights = dict(INTERP_WEIGHTS)
        # Bins: orders 1-2 (low), 3-5 (mid), 6-max_order (high)
        max_order = self.max_order
        low_orders = [1, 2]
        mid_orders = [3, 4, 5]
        high_orders = list(range(6, max_order + 1))
        # Finer grid: more low/mid points to find better weights
        low_vals = [0.02, 0.04, 0.06, 0.08, 0.10]
        mid_vals = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
        for w_low, w_mid in itertools.product(low_vals, mid_vals):
            w_high = 1.0 - w_low - w_mid
            if w_high < 0.05:
                continue
            weights = {}
            n_low, n_mid, n_high = len(low_orders), len(mid_orders), len(high_orders)
            for k in low_orders:
                weights[k] = w_low / n_low
            for k in mid_orders:
                weights[k] = w_mid / n_mid
            for k in high_orders:
                weights[k] = w_high / n_high
            # Restrict to orders present in model and renormalize
            weights = {k: weights.get(k, 0) for k in range(1, max_order + 1)}
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}
            self.interp_weights = weights
            acc = self.evaluate_top3_accuracy(val_pairs)
            if acc > best_acc:
                best_acc = acc
                best_weights = dict(weights)
                if verbose:
                    print(f"  New best top-3 acc: {acc:.4f} (low={w_low}, mid={w_mid}, high={w_high:.2f})")
        self.interp_weights = best_weights
        return best_weights

    @staticmethod
    def save_interp_weights(work_dir, weights):
        """Update interp_weights in the existing DB meta table."""
        db_path = os.path.join(work_dir, "ngram_model.db")
        if not os.path.exists(db_path):
            return
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM meta WHERE k='interp_weights'")
            if cur.fetchone():
                cur.execute("UPDATE meta SET v=? WHERE k='interp_weights'", (json.dumps({str(k): v for k, v in weights.items()}),))
            else:
                cur.execute("INSERT INTO meta (k, v) VALUES ('interp_weights', ?)", (json.dumps({str(k): v for k, v in weights.items()}),))
            conn.commit()
        finally:
            conn.close()

    # -----------------------------
    # Training helpers
    # -----------------------------
    @staticmethod
    def _count_ngrams_for_order(text, order, verbose=True):
        """Count all n-grams of a given order in the text."""
        n = len(text)
        if order > n:
            return collections.Counter()

        if verbose:
            t0 = time.time()
            print(f"  Counting {order}-grams...")

        counter = collections.defaultdict(int)

        # Process in chunks to reduce memory pressure
        chunk_size = min(10_000_000, n)  # Process 10M chars at a time

        for start in range(0, n - order + 1, chunk_size):
            end = min(start + chunk_size + order - 1, n)
            chunk = text[start:end]

            for i in range(len(chunk) - order + 1):
                ngram = chunk[i : i + order]
                counter[ngram] += 1

        result = collections.Counter(counter)

        if verbose:
            elapsed = time.time() - t0
            print(f"    {len(result):,} unique {order}-grams ({elapsed:.1f}s)")

        return result

    TOP_NEXT = 10  # Store top-K next chars per context; prediction still returns top-3 by interpolated score

    @staticmethod
    def _build_top3_for_order(counter, min_count, verbose=True):
        """From an n-gram counter, extract top-K next chars per context with normalized probs."""
        k = MyModel.TOP_NEXT
        context_chars = collections.defaultdict(list)
        for ngram, count in counter.items():
            if count >= min_count:
                context = ngram[:-1]
                next_char = ngram[-1]
                context_chars[context].append((count, next_char))

        order_model = {}
        for context, char_counts in context_chars.items():
            char_counts.sort(reverse=True)
            topk_counts = char_counts[:k]
            total = sum(c for c, _ in topk_counts)
            if total <= 0:
                continue
            order_model[context] = [
                (ch, cnt / total) for cnt, ch in topk_counts
            ]

        if verbose:
            print(
                f"    {len(order_model):,} contexts after pruning (min_count={min_count})"
            )

        return order_model

    def run_train(self, text, work_dir, max_order=MAX_ORDER, min_counts=MIN_COUNTS):
        """Full training pipeline: count n-grams -> prune -> keep in-memory model."""
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

        self.model = model
        self.max_order = max(model.keys()) if model else 0

        print()
        return model

    # -----------------------------
    # SQLite persistence (one table per order)
    # -----------------------------
    @staticmethod
    def _apply_write_pragmas(conn):
        # Faster bulk writes (safe for this use case)
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA temp_store=MEMORY")

    @staticmethod
    def _apply_read_pragmas(conn):
        # Faster reads for inference
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute(
            "PRAGMA cache_size=200000"
        )  # number of pages; negative = KiB. Here keep default style.
        # mmap_size helps a lot on large DBs if OS allows it
        conn.execute("PRAGMA mmap_size=30000000000")  # ~30GB (OS will cap)

    def save(self, work_dir):
        """Save n-gram model to a SQLite DB, one table per order."""
        os.makedirs(work_dir, exist_ok=True)
        db_path = os.path.join(work_dir, "ngram_model.db")

        # Recreate DB
        if os.path.exists(db_path):
            os.remove(db_path)

        conn = sqlite3.connect(db_path, isolation_level=None)
        try:
            self._apply_write_pragmas(conn)
            cur = conn.cursor()

            # Create a tiny metadata table
            cur.execute("CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT)")
            cur.execute(
                "INSERT INTO meta (k, v) VALUES (?, ?)",
                ("max_order", str(self.max_order)),
            )
            cur.execute(
                "INSERT INTO meta (k, v) VALUES (?, ?)",
                ("interp_weights", json.dumps(self.interp_weights)),
            )

            # Create per-order tables: context -> preds_chars, preds_probs (for interpolation)
            for order in range(1, self.max_order + 1):
                cur.execute(
                    f"""
                    CREATE TABLE ngrams_{order} (
                        context TEXT PRIMARY KEY,
                        preds_chars TEXT,
                        preds_probs TEXT
                    )
                    """
                )

            # Bulk insert
            print("Writing model to SQLite (per-order tables)...")
            t0 = time.time()

            cur.execute("BEGIN")
            for order, contexts in self.model.items():
                rows = []
                for ctx, entries in contexts.items():
                    # entries: [(char, prob), ...]
                    preds_chars = "".join(c for c, _ in entries)
                    preds_probs = ",".join(f"{p:.6f}" for _, p in entries)
                    rows.append((ctx, preds_chars, preds_probs))
                cur.executemany(
                    f"INSERT INTO ngrams_{order} (context, preds_chars, preds_probs) VALUES (?, ?, ?)",
                    rows,
                )
            conn.commit()

            elapsed = time.time() - t0
            size_mb = os.path.getsize(db_path) / (1024 * 1024)

            print(
                f"Model saved to {db_path} ({size_mb:.1f} MB, wrote in {elapsed:.1f}s)"
            )

            # Print model stats
            total_contexts = sum(len(m) for m in self.model.values())
            print(f"Total contexts across all orders: {total_contexts:,}")
            for order in range(1, self.max_order + 1):
                if order in self.model:
                    print(f"  Order {order}: {len(self.model[order]):,} contexts")

            # Show the fallback prediction (unigram top-3)
            if 1 in self.model and "" in self.model[1]:
                fallback = "".join(c for c, _ in self.model[1][""][:3])
                print(f"Fallback prediction (unigram top-3): '{fallback}'")
        finally:
            conn.close()

    @classmethod
    def load(cls, work_dir):
        """Load SQLite-backed model (does not load all contexts into RAM)."""
        db_path = os.path.join(work_dir, "ngram_model.db")
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"Model file not found: {db_path}\n"
                "Run: python src/myprogram.py train --data_dir data/wiki --work_dir work"
            )

        print(f"Loading model from {db_path}...")
        obj = cls(ngram_model=None, db_path=db_path)

        conn = sqlite3.connect(db_path)
        obj._apply_read_pragmas(conn)
        obj.conn = conn
        obj._db_ready = True

        # Read max_order from meta (fallback to discovering existing tables)
        cur = conn.cursor()
        try:
            cur.execute("SELECT v FROM meta WHERE k='max_order'")
            row = cur.fetchone()
            if row and row[0].strip().isdigit():
                obj.max_order = int(row[0])
            else:
                raise sqlite3.Error("meta.max_order missing or invalid")
        except sqlite3.Error:
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'ngrams_%'"
            )
            names = [r[0] for r in cur.fetchall()]
            orders = []
            for name in names:
                try:
                    orders.append(int(name.split("_", 1)[1]))
                except Exception:
                    pass
            obj.max_order = max(orders) if orders else 0

        # Interpolation weights from meta (optional)
        cur.execute("SELECT v FROM meta WHERE k='interp_weights'")
        row = cur.fetchone()
        if row:
            try:
                obj.interp_weights = json.loads(row[0])
                # keys may be strings from JSON
                obj.interp_weights = {int(k): float(v) for k, v in obj.interp_weights.items()}
            except (json.JSONDecodeError, ValueError):
                pass

        # Detect schema: new DBs have preds_chars/preds_probs, old have preds
        cur.execute("PRAGMA table_info(ngrams_1)")
        columns = [r[1] for r in cur.fetchall()]
        obj._schema_has_probs = "preds_probs" in columns

        print(f"Model loaded (SQLite): max_order={obj.max_order}")
        return obj

    def close(self):
        """Close SQLite connection (optional, but good hygiene)."""
        if self.conn is not None:
            try:
                self.conn.close()
            finally:
                self.conn = None
                self._db_ready = False

    def __del__(self):
        # Best-effort close
        try:
            self.close()
        except Exception:
            pass

    # -----------------------------
    # Fast SQLite lookups with LRU cache
    # -----------------------------
    @lru_cache(maxsize=400000)
    def _db_lookup(self, order, context):
        """
        Cached SQLite lookup: returns list of (char, prob) or None.
        Supports both new schema (preds_chars, preds_probs) and old (preds only).
        """
        if not self._db_ready or self.conn is None:
            return None
        if order < 1 or order > self.max_order:
            return None

        has_probs = getattr(self, "_schema_has_probs", True)
        if has_probs:
            cur = self.conn.execute(
                f"SELECT preds_chars, preds_probs FROM ngrams_{order} WHERE context=?",
                (context,),
            )
        else:
            cur = self.conn.execute(
                f"SELECT preds FROM ngrams_{order} WHERE context=?",
                (context,),
            )
        row = cur.fetchone()
        if not row:
            return None
        if has_probs:
            chars, probs_str = row[0], row[1]
            if not chars or not probs_str:
                return None
            probs = [float(x) for x in probs_str.split(",")]
            return list(zip(chars, probs[:len(chars)]))
        preds = row[0] or ""
        if not preds:
            return None
        n = len(preds)
        return [(c, 1.0 / n) for c in preds]

    # -----------------------------
    # Prediction (interpolation: P(c) = sum_k lambda_k * P_k(c|context_k))
    # -----------------------------
    def _get_context_distribution(self, order, context):
        """Return list of (char, prob) for this order/context, or None."""
        if self._db_ready:
            return self._db_lookup(order, context)
        return self.model.get(order, {}).get(context)

    def predict_next_chars(self, text):
        """
        Predict top-3 next characters using interpolated n-gram probabilities:
        score(c) = sum over orders k of lambda_k * P_k(c|context_k).
        """
        text = unicodedata.normalize("NFKC", text.lower())
        scores = {}

        for order in range(1, self.max_order + 1):
            context_len = order - 1
            if len(text) < context_len:
                continue
            context = text[-context_len:] if context_len > 0 else ""

            dist = self._get_context_distribution(order, context)
            if not dist:
                continue
            lk = self.interp_weights.get(order, 0.0)
            if lk <= 0:
                continue
            for ch, p in dist:
                if ch in self.INVALID_PRED_CHARS:
                    continue
                scores[ch] = scores.get(ch, 0.0) + lk * p

        if not scores:
            return self._get_fallback()

        top3 = sorted(scores.items(), key=lambda x: -x[1])[:3]
        return "".join(c for c, _ in top3)

    def _get_fallback(self):
        """Get fallback prediction (unigram top-3) when no interpolated scores."""
        if self._db_ready:
            preds = self._db_lookup(1, "")
            if preds:
                chars = "".join(c for c, _ in preds[:3] if c not in self.INVALID_PRED_CHARS)
                return (chars + " et")[:3]
            return " et"
        if 1 in self.model and "" in self.model[1]:
            entries = self.model[1][""]
            chars = "".join(c for c, _ in entries[:3] if c not in self.INVALID_PRED_CHARS)
            return (chars + " et")[:3]
        return " et"

    def run_pred(self, data):
        """Make predictions for all inputs."""
        preds = []
        for inp in data:
            top3 = self.predict_next_chars(inp)
            preds.append(top3)
        return preds


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
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="fraction of data held out for tuning interpolation weights (default: 0.1)",
    )
    parser.add_argument(
        "--max_val_pairs",
        type=int,
        default=50000,
        help="max (context, target) pairs for weight tuning (default: 50000)",
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
        print("STEP 1: Loading text files (per-file train/val split)")
        print("=" * 60)
        train_text, val_text = MyModel.load_training_data_split(
            args.data_dir, val_ratio=args.val_ratio
        )
        print()

        print("Training n-gram model (order 9 first; each order may take several minutes)...", flush=True)
        model.run_train(train_text, args.work_dir)
        del train_text

        print("Saving model")
        print("=" * 60)
        print("STEP 3: Saving model")
        print("=" * 60)
        model.save(args.work_dir)

        print()
        print("STEP 4: Tuning interpolation weights (grid search)")
        print("=" * 60)
        val_pairs = MyModel.build_validation_pairs(
            val_text, model.max_order, max_pairs=args.max_val_pairs
        )
        del val_text
        print(f"Validation pairs: {len(val_pairs):,}")
        if val_pairs:
            model = MyModel.load(args.work_dir)
            best_weights = model.train_interp_weights(val_pairs)
            MyModel.save_interp_weights(args.work_dir, best_weights)
            print(f"Best top-3 accuracy: {model.evaluate_top3_accuracy(val_pairs):.4f}")
            print("Saved optimized interp_weights to model DB.")
            model.close()
        else:
            print("No validation pairs; keeping default interp_weights.")

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

        # Optional explicit close
        model.close()

    else:
        raise NotImplementedError("Unknown mode {}".format(args.mode))
