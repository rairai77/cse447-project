#!/usr/bin/env python
import os
import pickle
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


class MyModel:
    """
    Character n-gram model with backoff for next-character prediction.
    """

    def __init__(self, ngram_model=None):
        """Initialize with n-gram model dictionary."""
        self.model = ngram_model or {}
        self.max_order = max(self.model.keys()) if self.model else 0

    @classmethod
    def load_training_data(cls):
        # Training is done separately via build_ngrams.py
        return []

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

    def run_train(self, data, work_dir):
        """Training is done separately via build_ngrams.py."""
        pass

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
        """Model is already saved by build_ngrams.py."""
        pass

    @classmethod
    def load(cls, work_dir):
        """Load n-gram model from pickle file."""
        model_path = os.path.join(work_dir, "ngram_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "Run: python src/build_ngrams.py --data_dir data/wiki --work_dir work"
            )

        print(f"Loading model from {model_path}...")
        with open(model_path, "rb") as f:
            ngram_model = pickle.load(f)

        print(f"Model loaded: {len(ngram_model)} orders, max_order={max(ngram_model.keys())}")
        return cls(ngram_model=ngram_model)


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("mode", choices=("train", "test"), help="what to run")
    parser.add_argument("--work_dir", help="where to save", default="work")
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
        train_data = MyModel.load_training_data()
        print("Training")
        model.run_train(train_data, args.work_dir)
        print("Saving model")
        model.save(args.work_dir)
    elif args.mode == "test":
        print("Loading model")
        model = MyModel.load(args.work_dir)
        print("Loading test data from {}".format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print("Making predictions")
        pred = model.run_pred(test_data)
        print("Writing predictions to {}".format(args.test_output))
        assert len(pred) == len(
            test_data
        ), "Expected {} predictions but got {}".format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError("Unknown mode {}".format(args.mode))
