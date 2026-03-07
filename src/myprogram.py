#!/usr/bin/env python
import os
import collections
import glob
import time
import sqlite3
from functools import lru_cache
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# ============================================================
# Training Configuration
# ============================================================

MAX_ORDER = 9  # Up to 10-grams (9 chars of context → predict 10th)

# Minimum count thresholds per order
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
    10: 3,
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

    @staticmethod
    def _build_top3_for_order(counter, min_count, verbose=True):
        """From an n-gram counter, extract top-3 most frequent next chars per context."""
        context_chars = collections.defaultdict(list)
        for ngram, count in counter.items():
            if count >= min_count:
                context = ngram[:-1]  # All but last char
                next_char = ngram[-1]  # Last char (the one we predict)
                context_chars[context].append((count, next_char))

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

            # Create per-order tables: context -> preds
            for order in range(1, self.max_order + 1):
                cur.execute(
                    f"""
                    CREATE TABLE ngrams_{order} (
                        context TEXT PRIMARY KEY,
                        preds   TEXT
                    )
                    """
                )

            # Bulk insert
            print("Writing model to SQLite (per-order tables)...")
            t0 = time.time()

            cur.execute("BEGIN")
            for order, contexts in self.model.items():
                rows = [(ctx, preds) for ctx, preds in contexts.items()]
                cur.executemany(
                    f"INSERT INTO ngrams_{order} (context, preds) VALUES (?, ?)",
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
                fallback = self.model[1][""]
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
            # Fallback: find max existing ngrams_{k} table
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
        Cached SQLite lookup: returns preds (string) or None.

        NOTE: This cache is process-local and does not persist across runs.
        """
        if not self._db_ready or self.conn is None:
            return None
        if order < 1 or order > self.max_order:
            return None

        # Per-order table lookup (fastest path in SQLite)
        cur = self.conn.execute(
            f"SELECT preds FROM ngrams_{order} WHERE context=?",
            (context,),
        )
        row = cur.fetchone()
        return row[0] if row else None

    # -----------------------------
    # Prediction
    # -----------------------------
    def predict_next_chars(self, text):
        """
        Predict top-3 next characters for given text using backoff.

        Args:
            text: Input string (will be lowercased)

        Returns:
            String of 3 characters (top-3 predictions)
        """
        text = text.lower()

        candidates = []
        seen = set()

        for order in range(self.max_order, 0, -1):
            context_len = order - 1
            if len(text) < context_len:
                continue

            context = text[-context_len:] if context_len > 0 else ""

            # Choose DB-backed or in-memory lookup transparently
            if self._db_ready:
                predictions = self._db_lookup(order, context)
            else:
                predictions = self.model.get(order, {}).get(context)

            if not predictions:
                continue

            for ch in predictions:
                if ch in self.INVALID_PRED_CHARS:
                    continue
                if ch not in seen:
                    seen.add(ch)
                    candidates.append(ch)
                    if len(candidates) == 3:
                        return "".join(candidates)

        # Last resort: fill any remaining slots from global unigram fallback.
        for ch in self._get_fallback():
            if ch in self.INVALID_PRED_CHARS:
                continue
            if ch not in seen:
                seen.add(ch)
                candidates.append(ch)
                if len(candidates) == 3:
                    return "".join(candidates)

        # Extremely defensive: keep output length exactly 3 even in degenerate cases.
        return ("".join(candidates) + " et")[:3]

    def _get_fallback(self):
        """Get fallback prediction (unigram top-3)."""
        if self._db_ready:
            preds = self._db_lookup(1, "")
            if preds:
                # Ensure at least 3 chars, pad if needed
                return (preds + " et")[:3]
            return " et"

        # In-memory fallback
        if 1 in self.model and "" in self.model[1]:
            fallback = "".join(
                ch for ch in self.model[1][""] if ch not in self.INVALID_PRED_CHARS
            )
            return (fallback + " et")[:3]
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

        # Optional explicit close
        model.close()

    else:
        raise NotImplementedError("Unknown mode {}".format(args.mode))
