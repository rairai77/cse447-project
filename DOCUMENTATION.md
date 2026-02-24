# Character N-gram Model Documentation

## Overview

This project implements a character-level n-gram language model for predicting the next character in a text sequence. The model is designed for the CSE447 project, where astronauts need to communicate via eye-tracking by selecting from 3 predicted characters at each time step.

The system uses a **character n-gram model with backoff** trained on multilingual Wikipedia text. It achieves high accuracy while maintaining fast inference speed through efficient dictionary lookups.

## Architecture

### Model Type
- **Character-level n-grams** (orders 1-7)
- **Backoff strategy**: Try longest context first, fall back to shorter contexts
- **Top-3 prediction**: Stores only the 3 most frequent next characters per context to minimize model size

### Training Data
- **Source**: Wikipedia dumps (multilingual)
- **Languages**: English (70%), French (4%), German (4%), and others
- **Total size**: ~15-20 million characters
- **Format**: Lowercased plain text

### Model Structure
```
model = {
    7: {context_6_chars: "abc", ...},  # 7-grams: 6-char context → top 3 chars
    6: {context_5_chars: "def", ...},  # 6-grams: 5-char context → top 3 chars
    ...
    1: {"": " ea"}                      # Unigrams: unconditional top 3
}
```

## File Structure

```
cse447-project/
├── src/
│   ├── download_data.py      # Downloads Wikipedia text
│   ├── build_ngrams.py        # Trains n-gram model
│   ├── myprogram.py           # Inference script
│   └── predict.sh             # Docker prediction entry point
├── data/
│   └── wiki/                  # Downloaded Wikipedia text files
├── work/
│   └── ngram_model.pkl        # Trained model (9.2 MB)
├── Dockerfile                  # Container definition
└── DOCUMENTATION.md           # This file
```

## Usage

### Step 1: Download Training Data

Download Wikipedia text for training:

```bash
# Install dependencies
pip install "datasets<3.0.0"

# Download all languages (takes ~10-20 min)
python src/download_data.py --output_dir data/wiki

# Or download English only (faster, ~2-5 min)
python src/download_data.py --output_dir data/wiki --english_only
```

This creates files like `data/wiki/en.txt`, `data/wiki/fr.txt`, etc.

### Step 2: Train the Model

Build the n-gram model from downloaded text:

```bash
python src/build_ngrams.py --data_dir data/wiki --work_dir work
```

This will:
- Load all `.txt` files from `data/wiki/`
- Count n-grams for orders 1-7
- Prune rare n-grams based on MIN_COUNT thresholds
- Extract top-3 predictions per context
- Save model to `work/ngram_model.pkl`

**Expected output:**
- Model size: ~5-15 MB (depends on data size)
- Training time: 30-60 seconds for 15MB text
- Total contexts: ~500K-1M across all orders

### Step 3: Test Locally

Test the model on example data:

```bash
# Make predictions
python src/myprogram.py test --work_dir work --test_data example/input.txt --test_output pred.txt

# Evaluate accuracy
python grader/grade.py pred.txt example/answer.txt --verbose
```

### Step 4: Test Docker Pipeline

Test the full Docker pipeline (as graders will run it):

```bash
bash grader/grade.sh example
```

This builds the Docker image and runs predictions inside the container.

## How It Works

### Training Process

1. **Text Loading**: Combine all Wikipedia text files into one string
2. **N-gram Counting**: For each order (1-7), count all n-gram occurrences
   - Order N = (N-1) chars of context + 1 predicted char
   - Example: Order 7 counts 7-char substrings; first 6 are context, last is prediction
3. **Pruning**: Remove n-grams below MIN_COUNT threshold
   - Higher orders: lower threshold (they're naturally rarer)
   - Lower orders: higher threshold (keep only strong patterns)
4. **Top-3 Extraction**: For each context, keep only the 3 most frequent next characters
5. **Serialization**: Save as pickle file

### Inference Process

For each input string:

1. **Lowercase**: Convert input to lowercase (grader is case-insensitive)
2. **Backoff**: Try contexts from longest to shortest:
   - Try 7-gram: last 6 chars → lookup in `model[7]`
   - If not found, try 6-gram: last 5 chars → lookup in `model[6]`
   - Continue down to 2-gram, then 1-gram
3. **Fallback**: If no context matches, use unconditional top-3 (`model[1][""]`)
4. **Output**: Return exactly 3 characters (pad if needed)

### Example

Input: `"Happy Ne"`

1. Lowercase: `"happy ne"`
2. Try 7-gram context: `"ppy ne"` (last 6 chars) → lookup in `model[7]`
3. If found: return top-3, e.g., `"w e"`
4. If not found: try 6-gram `"py ne"` → lookup in `model[6]`
5. Continue until match or fallback

## Configuration

### N-gram Parameters

Edit `src/build_ngrams.py` to adjust:

- **MAX_ORDER**: Maximum n-gram order (default: 7)
- **MIN_COUNTS**: Pruning thresholds per order
  ```python
  MIN_COUNTS = {
      1: 0,   # Keep all unigrams
      2: 5,   # Bigrams: min 5 occurrences
      3: 5,
      4: 3,
      5: 3,
      6: 3,
      7: 3,
  }
  ```

### Language Selection

Edit `src/download_data.py` to change language mix:

```python
LANGUAGE_CONFIGS = {
    "en": ("20220301.en", 14_000_000),  # English: 70%
    "es": ("20220301.es", 1_000_000),   # Spanish: 5%
    # ... add more languages
}
```

## Performance

### Accuracy
- **Example data**: ~92% top-3 accuracy
- **Expected on test set**: 60-75% (depends on test data characteristics)

### Speed
- **Training**: 30-60 seconds for 15MB text
- **Inference**: <1ms per prediction (dictionary lookup)
- **Model loading**: 1-3 seconds (pickle deserialization)

### Model Size
- **Current**: 9.2 MB (15MB training data)
- **Limit**: 3 GB (project requirement)
- **Scaling**: ~0.5-1 MB per 1MB of training text

## Troubleshooting

### "Model file not found" error
- Run `build_ngrams.py` first to create `work/ngram_model.pkl`
- Ensure `--work_dir` matches between training and inference

### Low accuracy
- Add more training data (more languages or more text per language)
- Decrease MIN_COUNT thresholds (keep more n-grams)
- Increase MAX_ORDER (use longer contexts)

### Model too large (>3GB)
- Increase MIN_COUNT thresholds (prune more aggressively)
- Reduce MAX_ORDER (use shorter contexts)
- Reduce training data size

### Docker build fails
- Ensure Dockerfile uses correct base image
- Check that `predict.sh` has execute permissions: `chmod +x src/predict.sh`

## Design Decisions

### Why Character-Level?
- Handles any language/script (Unicode support)
- No vocabulary limitations
- Captures sub-word patterns

### Why N-grams (not Neural)?
- **Speed**: Dictionary lookups are O(1), extremely fast
- **Simplicity**: No GPU needed, easy to debug
- **Size**: Compact model (top-3 only, not full distributions)
- **Reliability**: Deterministic, no training instability

### Why Top-3 Only?
- Project requirement: predict exactly 3 characters
- Massive storage reduction: 80-90x smaller than full distributions
- Fast inference: no sorting or probability computation needed

### Why Backoff?
- Handles unseen contexts gracefully
- Uses most specific information available
- Guarantees a prediction (never fails)

## Future Improvements

If time permits:
1. **More languages**: Add Spanish, Russian, Chinese, Arabic, etc.
2. **Dialogue data**: Add conversational corpora (OpenSubtitles, Reddit)
3. **Smoothing**: Implement Kneser-Ney smoothing for better accuracy
4. **Adaptive thresholds**: Tune MIN_COUNT per language/order
5. **Compression**: Use trie data structure for even smaller model size

## References

- Character n-gram language models
- Wikipedia dumps: https://dumps.wikimedia.org/
- HuggingFace datasets: https://huggingface.co/datasets/wikipedia
