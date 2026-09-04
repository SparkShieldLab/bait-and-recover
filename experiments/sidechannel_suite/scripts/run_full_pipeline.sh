#!/usr/bin/env bash
# Run Step1, Phase 1, Phase 2, and optional Heretic comparisons.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

MODEL_SLUG="${MODEL_SLUG:-$(model_slug "$BASE_MODEL")}"

case "$STYLE" in
    oldstyle)
        PHASE1_RF_WEIGHT="${PHASE1_RF_WEIGHT:-${RF_WEIGHT:-1.0}}"
        PHASE2_RF_WEIGHT="${PHASE2_RF_WEIGHT:-${RF_WEIGHT:-1.0}}"
        ;;
    ideal_pair)
        PHASE1_RF_WEIGHT="${PHASE1_RF_WEIGHT:-${RF_WEIGHT:-0.25}}"
        PHASE2_RF_WEIGHT="${PHASE2_RF_WEIGHT:-${RF_WEIGHT:-0.1}}"
        ;;
    *)
        echo "[ERROR] Unknown STYLE=$STYLE. Use oldstyle or ideal_pair."
        exit 1
        ;;
esac

PHASE1_RF_TAG="$(rf_tag "$PHASE1_RF_WEIGHT")"
PHASE2_RF_TAG="$(rf_tag "$PHASE2_RF_WEIGHT")"
DATA_TAG="$(data_source_tag "$EVAL_DATA_SOURCE" "$TRAIN_DATA_SOURCE" "$GENERAL_DATA_SOURCE" "$TRAIN_GOOD_DATA_SOURCE" "$TRAIN_BAD_DATA_SOURCE")"
if [[ -n "$DATA_TAG" ]]; then
    DEFAULT_PHASE1_TAG="phase1_sidechannel_${STYLE}_rf${PHASE1_RF_TAG}_${DATA_TAG}_${MODEL_SLUG}"
    DEFAULT_PHASE2_TAG="phase2_sidechannel_${STYLE}_rf${PHASE2_RF_TAG}_${DATA_TAG}_${MODEL_SLUG}"
    DEFAULT_DEFENDED_CHECKPOINT_TAG="defended_${DATA_TAG}_${MODEL_SLUG}"
else
    DEFAULT_PHASE1_TAG="phase1_sidechannel_${STYLE}_rf${PHASE1_RF_TAG}_${MODEL_SLUG}"
    DEFAULT_PHASE2_TAG="phase2_sidechannel_${STYLE}_rf${PHASE2_RF_TAG}_${MODEL_SLUG}"
    DEFAULT_DEFENDED_CHECKPOINT_TAG="defended_${MODEL_SLUG}"
fi
PHASE1_TAG="${PHASE1_TAG:-$DEFAULT_PHASE1_TAG}"
PHASE2_TAG="${PHASE2_TAG:-$DEFAULT_PHASE2_TAG}"

RUN_STEP1="${RUN_STEP1:-auto}"
RUN_TRAIN="${RUN_TRAIN:-true}"
RUN_HERETIC="${RUN_HERETIC:-true}"
RUN_BASELINE_HERETIC="${RUN_BASELINE_HERETIC:-false}"
RUN_COMPARE="${RUN_COMPARE:-true}"
HERETIC_RUNNER="${HERETIC_RUNNER:-single}"
DEFENDED_CHECKPOINT_TAG="${DEFENDED_CHECKPOINT_TAG:-$DEFAULT_DEFENDED_CHECKPOINT_TAG}"
BASELINE_CHECKPOINT_TAG="${BASELINE_CHECKPOINT_TAG:-baseline_${MODEL_SLUG}}"
HERETIC_DIRECTION_SCOPE="$(normalize_direction_scope "${HERETIC_DIRECTION_SCOPE:-both}")"
DEFENDED_HERETIC_CHECKPOINT_TAG="$(scoped_checkpoint_tag "$DEFENDED_CHECKPOINT_TAG" "$HERETIC_DIRECTION_SCOPE")"
BASELINE_HERETIC_CHECKPOINT_TAG="$(scoped_checkpoint_tag "$BASELINE_CHECKPOINT_TAG" "$HERETIC_DIRECTION_SCOPE")"
HERETIC_CHECKPOINT_BASE="${CHECKPOINT_BASE:-$SUITE_DIR/heretic_checkpoints}"

case "$HERETIC_RUNNER" in
    parallel)
        HERETIC_SCRIPT="$SCRIPT_DIR/run_heretic_parallel.sh"
        ;;
    single)
        HERETIC_SCRIPT="$SCRIPT_DIR/run_heretic.sh"
        ;;
    exact)
        HERETIC_SCRIPT="$SCRIPT_DIR/run_heretic.sh"
        export HERETIC_SAMPLER_SEED="${HERETIC_SAMPLER_SEED:-${HERETIC_SAMPLER_SEED_BASE:-42}}"
        ;;
    *)
        echo "[ERROR] Unknown HERETIC_RUNNER=$HERETIC_RUNNER. Use parallel, single, or exact."
        exit 1
        ;;
esac

start_log "full_pipeline_${STYLE}_${MODEL_SLUG}"

echo "Full sidechannel pipeline"
print_common
echo "  phase1 tag          : $PHASE1_TAG"
echo "  phase1 RF weight    : $PHASE1_RF_WEIGHT"
echo "  phase2 tag          : $PHASE2_TAG"
echo "  phase2 RF weight    : $PHASE2_RF_WEIGHT"
echo "  run step1           : $RUN_STEP1"
echo "  run train           : $RUN_TRAIN"
echo "  run heretic         : $RUN_HERETIC"
echo "  heretic runner      : $HERETIC_RUNNER"
echo "  heretic dir scope   : $HERETIC_DIRECTION_SCOPE"
if [[ "$HERETIC_RUNNER" == "exact" ]]; then
    echo "  sampler seed        : ${HERETIC_SAMPLER_SEED}"
fi
echo "  run baseline heretic: $RUN_BASELINE_HERETIC"
echo "  run compare         : $RUN_COMPARE"
echo "  defended checkpoint : $DEFENDED_HERETIC_CHECKPOINT_TAG"
echo "  baseline checkpoint : $BASELINE_HERETIC_CHECKPOINT_TAG"
echo "  checkpoint base     : $HERETIC_CHECKPOINT_BASE"
echo ""

STEP1_NEEDS_REGEN=false
if [[ "$RUN_STEP1" == "auto" && -f "$STEP1_PATH" && -n "${HERETIC_COVERAGE_LAYERS:-${COVERAGE_LAYER_COUNT:-}}" ]]; then
    if ! "$PYTHON" - "$STEP1_PATH" "${HERETIC_COVERAGE_LAYERS:-${COVERAGE_LAYER_COUNT:-}}" <<'PY'
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
coverage_layers = int(sys.argv[2])
data = json.loads(path.read_text())
layers = sorted(
    int(match.group(1))
    for key in data
    for match in [re.fullmatch(r"layer_(\d+)", key)]
    if match
)
if not layers:
    raise SystemExit(1)
num_layers = max(layers) + 1
ratios = data.get("fisher_ratios_baseline")
if isinstance(ratios, list) and ratios:
    num_layers = max(num_layers, len(ratios) - 1)
required_start = max(0, (num_layers - 2) - coverage_layers + 1)
raise SystemExit(0 if required_start in layers else 1)
PY
    then
        STEP1_NEEDS_REGEN=true
    fi
fi

if [[ "$RUN_STEP1" == "true" || ( "$RUN_STEP1" == "auto" && ( ! -f "$STEP1_PATH" || "$STEP1_NEEDS_REGEN" == "true" ) ) ]]; then
    echo "== Step1 =="
    if [[ "$STEP1_NEEDS_REGEN" == "true" ]]; then
        echo "Existing Step1 summary does not cover HERETIC_COVERAGE_LAYERS=${HERETIC_COVERAGE_LAYERS:-${COVERAGE_LAYER_COUNT:-}}; regenerating."
    fi
    BASE_MODEL="$BASE_MODEL" \
    TOKENIZER="$TOKENIZER" \
    MODEL_SLUG="$MODEL_SLUG" \
    MODEL_SAVE_DIR="$MODEL_SAVE_DIR" \
    SUITE_RESULTS_DIR="$SUITE_RESULTS_DIR" \
    LOG_DIR="$LOG_DIR" \
    STEP1_PATH="$STEP1_PATH" \
    EVAL_DATA_SOURCE="$EVAL_DATA_SOURCE" \
    TRAIN_DATA_SOURCE="$TRAIN_DATA_SOURCE" \
    TRAIN_GOOD_DATA_SOURCE="$TRAIN_GOOD_DATA_SOURCE" \
    TRAIN_BAD_DATA_SOURCE="$TRAIN_BAD_DATA_SOURCE" \
    GENERAL_DATA_SOURCE="$GENERAL_DATA_SOURCE" \
    NESTED_LOG_TO_STDOUT=true \
    "$SCRIPT_DIR/prepare_step1.sh"
elif [[ ! -f "$STEP1_PATH" ]]; then
    echo "[ERROR] Step1 summary not found and RUN_STEP1=$RUN_STEP1:"
    echo "        $STEP1_PATH"
    exit 1
else
    echo "== Step1 =="
    echo "Using existing Step1 summary:"
    echo "  $STEP1_PATH"
fi

if [[ "$RUN_TRAIN" == "true" ]]; then
    echo ""
    echo "== Phase 1 =="
    BASE_MODEL="$BASE_MODEL" \
    TOKENIZER="$TOKENIZER" \
    MODEL_SLUG="$MODEL_SLUG" \
    MODEL_SAVE_DIR="$MODEL_SAVE_DIR" \
    SUITE_RESULTS_DIR="$SUITE_RESULTS_DIR" \
    LOG_DIR="$LOG_DIR" \
    STEP1_PATH="$STEP1_PATH" \
    STYLE="$STYLE" \
    EVAL_DATA_SOURCE="$EVAL_DATA_SOURCE" \
    TRAIN_DATA_SOURCE="$TRAIN_DATA_SOURCE" \
    TRAIN_GOOD_DATA_SOURCE="$TRAIN_GOOD_DATA_SOURCE" \
    TRAIN_BAD_DATA_SOURCE="$TRAIN_BAD_DATA_SOURCE" \
    GENERAL_DATA_SOURCE="$GENERAL_DATA_SOURCE" \
    KL_LOSS_WEIGHT="${KL_LOSS_WEIGHT:-1.0}" \
    RF_WEIGHT="$PHASE1_RF_WEIGHT" \
    EXPERIMENT_TAG="$PHASE1_TAG" \
    NESTED_LOG_TO_STDOUT=true \
    "$SCRIPT_DIR/train_phase1.sh"

    echo ""
    echo "== Phase 2 =="
    BASE_MODEL="$BASE_MODEL" \
    TOKENIZER="$TOKENIZER" \
    ORIGINAL_CLEAN_MODEL="$ORIGINAL_CLEAN_MODEL" \
    ORIGINAL_CLEAN_TOKENIZER="$ORIGINAL_CLEAN_TOKENIZER" \
    MODEL_SLUG="$MODEL_SLUG" \
    MODEL_SAVE_DIR="$MODEL_SAVE_DIR" \
    SUITE_RESULTS_DIR="$SUITE_RESULTS_DIR" \
    LOG_DIR="$LOG_DIR" \
    STEP1_PATH="$STEP1_PATH" \
    STYLE="$STYLE" \
    EVAL_DATA_SOURCE="$EVAL_DATA_SOURCE" \
    TRAIN_DATA_SOURCE="$TRAIN_DATA_SOURCE" \
    TRAIN_GOOD_DATA_SOURCE="$TRAIN_GOOD_DATA_SOURCE" \
    TRAIN_BAD_DATA_SOURCE="$TRAIN_BAD_DATA_SOURCE" \
    GENERAL_DATA_SOURCE="$GENERAL_DATA_SOURCE" \
    KL_LOSS_WEIGHT="${KL_LOSS_WEIGHT:-1.0}" \
    PHASE1_TAG="$PHASE1_TAG" \
    PHASE1_RF_FOR_TAG="$PHASE1_RF_WEIGHT" \
    RF_WEIGHT="$PHASE2_RF_WEIGHT" \
    EXPERIMENT_TAG="$PHASE2_TAG" \
    NESTED_LOG_TO_STDOUT=true \
    "$SCRIPT_DIR/train_phase2.sh"
fi

if [[ "$RUN_HERETIC" == "true" ]]; then
    echo ""
    echo "== Heretic: defended =="
    BASE_MODEL="$BASE_MODEL" \
    MODEL_SLUG="$MODEL_SLUG" \
    MODEL_SAVE_DIR="$MODEL_SAVE_DIR" \
    LOG_DIR="$LOG_DIR" \
    TARGET_TAG="$PHASE2_TAG" \
    CHECKPOINT_TAG="$DEFENDED_HERETIC_CHECKPOINT_TAG" \
    CHECKPOINT_BASE="$HERETIC_CHECKPOINT_BASE" \
    "$HERETIC_SCRIPT"

    if [[ "$RUN_BASELINE_HERETIC" == "true" ]]; then
        echo ""
        echo "== Heretic: baseline =="
        BASE_MODEL="$BASE_MODEL" \
        MODEL_SLUG="$MODEL_SLUG" \
        MODEL_SAVE_DIR="$MODEL_SAVE_DIR" \
        LOG_DIR="$LOG_DIR" \
        CHECKPOINT_TAG="$BASELINE_HERETIC_CHECKPOINT_TAG" \
        CHECKPOINT_BASE="$HERETIC_CHECKPOINT_BASE" \
        "$HERETIC_SCRIPT" --baseline
    fi
fi

if [[ "$RUN_COMPARE" == "true" ]]; then
    latest_jsonl_by_mtime() {
        local dir="$1"
        find "$dir" -type f -name '*.jsonl' -printf '%T@ %p\n' 2>/dev/null \
            | sort -n \
            | tail -n 1 \
            | cut -d' ' -f2-
    }

    DEFENDED_JOURNAL="$(latest_jsonl_by_mtime "$HERETIC_CHECKPOINT_BASE/$DEFENDED_HERETIC_CHECKPOINT_TAG" || true)"
    BASELINE_JOURNAL="$(latest_jsonl_by_mtime "$HERETIC_CHECKPOINT_BASE/$BASELINE_HERETIC_CHECKPOINT_TAG" || true)"

    if [[ -n "$DEFENDED_JOURNAL" && -n "$BASELINE_JOURNAL" ]]; then
        echo ""
        echo "== Heretic comparison =="
        "$PYTHON" "$SCRIPT_DIR/compare_heretic.py" \
            "baseline=$BASELINE_JOURNAL" \
            "defended=$DEFENDED_JOURNAL" \
            --show-best \
            --by-scope
    elif [[ -n "$DEFENDED_JOURNAL" ]]; then
        echo ""
        echo "== Heretic defended summary =="
        "$PYTHON" "$SCRIPT_DIR/compare_heretic.py" "defended=$DEFENDED_JOURNAL" --show-best --by-scope
    else
        echo ""
        echo "[INFO] No defended Heretic journal found yet; skipping comparison."
    fi
fi

echo ""
echo "Pipeline complete."
echo "  final model         : $MODEL_SAVE_DIR/step4_merged_${PHASE2_TAG}"
echo "  logs                : $LOG_DIR"
echo "  heretic checkpoints : $HERETIC_CHECKPOINT_BASE"
