#!/usr/bin/env bash
# Minimal plumbing validation; results are not scientific evidence.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
FIXTURES="$ROOT/tests/fixtures/prompts"
OUTPUT="${SMOKE_OUTPUT_DIR:-$ROOT/artifacts/smoke}"
DEFAULT_MODELSCOPE="${MODELSCOPE_CACHE_DIR:-$HOME/.cache/modelscope}/google/gemma-3-1b-it"

if [[ -n "${SMOKE_MODEL:-}" ]]; then
    MODEL="$SMOKE_MODEL"
elif [[ -d "$DEFAULT_MODELSCOPE" ]]; then
    MODEL="$DEFAULT_MODELSCOPE"
else
    MODEL="google/gemma-3-1b-it"
fi

if [[ ! -d "$MODEL" && "${HF_HUB_OFFLINE:-1}" == "1" ]]; then
    echo "Offline smoke model not found: $MODEL"
    echo "Set SMOKE_MODEL=/local/model/path or deliberately set HF_HUB_OFFLINE=0."
    exit 2
fi

mkdir -p "$OUTPUT/results" "$OUTPUT/models" "$OUTPUT/logs" "$OUTPUT/heretic_checkpoints"

export PYTHON
export GPU_ID="${GPU_ID:-0}"
export BASE_MODEL="$MODEL"
export TOKENIZER="$MODEL"
export ORIGINAL_CLEAN_MODEL="$MODEL"
export ORIGINAL_CLEAN_TOKENIZER="$MODEL"
export MODEL_SCOPE_PROMPT_DIR="$FIXTURES"
export EVAL_DATA_SOURCE="local:$FIXTURES"
export TRAIN_DATA_SOURCE="local:$FIXTURES"
export TRAIN_GOOD_DATA_SOURCE="local:$FIXTURES"
export TRAIN_BAD_DATA_SOURCE="local:$FIXTURES"
export GENERAL_DATA_SOURCE="local:$FIXTURES"
export SUITE_RESULTS_DIR="$OUTPUT/results"
export MODEL_SAVE_DIR="$OUTPUT/models"
export LOG_DIR="$OUTPUT/logs"
export CHECKPOINT_BASE="$OUTPUT/heretic_checkpoints"

export VERIFY_TAG="smoke_small_model"
export HERETIC_COVERAGE_LAYERS=2
export PHASE1_LAYERS=23
export PHASE2_LAYERS=24
export N_GOOD=4
export N_BAD=4
export N_TRAIN=8
export N_TRAIN_GOOD=4
export N_TRAIN_BAD=4
export TRAIN_GOOD_OFFSET=4
export TRAIN_BAD_OFFSET=4
export BATCH_SIZE=1
export N_EPOCHS=1
export PROGRESSIVE_STAGES=1
export PROGRESSIVE_BLOCK_JOINT_EPOCHS=0
export PROGRESSIVE_GEOMETRY_JOINT_EPOCHS=0
export SKIP_JOINT=true
export RUN_STEP1=true
export RUN_TRAIN=true
export RUN_HERETIC="${SMOKE_RUN_HERETIC:-true}"
export RUN_BASELINE_HERETIC="$RUN_HERETIC"
export RUN_COMPARE="$RUN_HERETIC"
export N_TRIALS="${SMOKE_N_TRIALS:-1}"
export HERETIC_BATCH_SIZE="${HERETIC_BATCH_SIZE:-8}"
export HERETIC_RUNNER=exact
export HERETIC_SAMPLER_SEED=42
export HERETIC_PROMPT_DIR="${HERETIC_PROMPT_DIR:-$FIXTURES}"

echo "Smoke model: $MODEL"
echo "Smoke output: $OUTPUT"
echo "Heretic enabled: $RUN_HERETIC"

bash "$ROOT/experiments/sidechannel_suite/scripts/run_full_pipeline_modelscope_alpaca_safemt.sh"

find "$OUTPUT/models" -maxdepth 2 -type f -name config.json -print | grep -q . || {
    echo "Smoke training did not create a merged model config."
    exit 1
}

if [[ "$RUN_HERETIC" == "true" ]]; then
    find "$OUTPUT/heretic_checkpoints" -type f -name '*.jsonl' -size +0c -print | grep -q . || {
        echo "Smoke attack did not create a non-empty Heretic journal."
        exit 1
    }
fi

echo "End-to-end smoke test passed."
