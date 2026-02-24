#!/usr/bin/env bash
set -euo pipefail
set -x

DATA=${1:?Usage: grader/grade_local.sh DATA_DIR [OUT_DIR]}
OUT=${2:-output}

mkdir -p "$OUT"

function run() {
  bash src/predict.sh "$DATA/input.txt" "$OUT/pred.txt"
}

(time run) > "$OUT/output" 2> "$OUT/runtime"
uv run grader/grade.py "$OUT/pred.txt" "$DATA/answer.txt" > "$OUT/success"
