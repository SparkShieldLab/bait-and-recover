#!/usr/bin/env bash
# Full pipeline using local JSONL exported from ModelScope:
#   good/general: AI-ModelScope/alpaca-gpt4-data-en
#   bad: SafeMTData/SafeMTData Attack_600 plain_query
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export MODEL_SCOPE_PROMPT_DIR="${MODEL_SCOPE_PROMPT_DIR:-$SUITE_DIR/data/anti_heretic_prompts/modelscope_alpaca_safemt}"

if [[ ! -f "$MODEL_SCOPE_PROMPT_DIR/good.jsonl" || ! -f "$MODEL_SCOPE_PROMPT_DIR/bad.jsonl" ]]; then
    echo "[ERROR] Missing exported ModelScope prompt JSONL files:"
    echo "        $MODEL_SCOPE_PROMPT_DIR/good.jsonl"
    echo "        $MODEL_SCOPE_PROMPT_DIR/bad.jsonl"
    echo ""
    echo "Run:"
    echo "  python $SCRIPT_DIR/export_modelscope_prompts.py --out-dir $MODEL_SCOPE_PROMPT_DIR"
    exit 1
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export ANTI_HERETIC_MODEL_LOCAL_FILES_ONLY="${ANTI_HERETIC_MODEL_LOCAL_FILES_ONLY:-1}"

export BASE_MODEL="${BASE_MODEL:-google/gemma-3-1b-it}"
export TOKENIZER="${TOKENIZER:-$BASE_MODEL}"
export ORIGINAL_CLEAN_MODEL="${ORIGINAL_CLEAN_MODEL:-$BASE_MODEL}"
export ORIGINAL_CLEAN_TOKENIZER="${ORIGINAL_CLEAN_TOKENIZER:-$TOKENIZER}"

export STYLE="${STYLE:-oldstyle}"
export EVAL_DATA_SOURCE="${EVAL_DATA_SOURCE:-mlabonne}"
export TRAIN_DATA_SOURCE="${TRAIN_DATA_SOURCE:-local:$MODEL_SCOPE_PROMPT_DIR}"
export TRAIN_GOOD_DATA_SOURCE="${TRAIN_GOOD_DATA_SOURCE:-local:$MODEL_SCOPE_PROMPT_DIR}"
export TRAIN_BAD_DATA_SOURCE="${TRAIN_BAD_DATA_SOURCE:-local:$MODEL_SCOPE_PROMPT_DIR}"
export GENERAL_DATA_SOURCE="${GENERAL_DATA_SOURCE:-local:$MODEL_SCOPE_PROMPT_DIR}"

export N_TRAIN="${N_TRAIN:-256}"
export N_TRAIN_GOOD="${N_TRAIN_GOOD:-128}"
export N_TRAIN_BAD="${N_TRAIN_BAD:-96}"
export TRAIN_GOOD_OFFSET="${TRAIN_GOOD_OFFSET:-128}"
export TRAIN_BAD_OFFSET="${TRAIN_BAD_OFFSET:-0}"

export RUN_STEP1="${RUN_STEP1:-auto}"
export RUN_TRAIN="${RUN_TRAIN:-true}"
export RUN_HERETIC="${RUN_HERETIC:-true}"
export RUN_BASELINE_HERETIC="${RUN_BASELINE_HERETIC:-false}"
export RUN_COMPARE="${RUN_COMPARE:-true}"
export HERETIC_RUNNER="${HERETIC_RUNNER:-single}"

exec "$SCRIPT_DIR/run_full_pipeline.sh" "$@"
