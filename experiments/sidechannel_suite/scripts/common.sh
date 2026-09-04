#!/usr/bin/env bash
set -euo pipefail

SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENTS_DIR="$(cd "$SUITE_DIR/.." && pwd)"
REPO_DIR="$(cd "$EXPERIMENTS_DIR/.." && pwd)"

PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-auto}"
GPU_MODE="${GPU_MODE:-single}"
GPU_ID="${GPU_ID:-2}"
if [[ -n "$GPU_ID" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

BASE_MODEL="${BASE_MODEL:-google/gemma-3-1b-it}"
TOKENIZER="${TOKENIZER:-$BASE_MODEL}"
ORIGINAL_CLEAN_MODEL="${ORIGINAL_CLEAN_MODEL:-$BASE_MODEL}"
ORIGINAL_CLEAN_TOKENIZER="${ORIGINAL_CLEAN_TOKENIZER:-$TOKENIZER}"

SUITE_RESULTS_DIR="${SUITE_RESULTS_DIR:-$SUITE_DIR/results}"
MODEL_SAVE_DIR="${MODEL_SAVE_DIR:-$SUITE_RESULTS_DIR/models}"
LOG_DIR="${LOG_DIR:-$SUITE_DIR/logs}"
DEFAULT_MODEL_SLUG="${MODEL_SLUG:-$(printf '%s' "${BASE_MODEL%/}" | tr '/:. ' '____')}"
STEP1_PATH="${STEP1_PATH:-$SUITE_RESULTS_DIR/step1_results_${DEFAULT_MODEL_SLUG}.json}"

STYLE="${STYLE:-oldstyle}"
TAG_SCALE="${TAG_SCALE:-1.0}"
VISIBLE_SHIFT_TARGET="${VISIBLE_SHIFT_TARGET:-0.10}"
VISIBLE_SHIFT_MAX="${VISIBLE_SHIFT_MAX:-0.20}"
VISIBLE_LOSS_WEIGHT="${VISIBLE_LOSS_WEIGHT:-1.0}"
VISIBLE_MAX_WEIGHT="${VISIBLE_MAX_WEIGHT:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SEED="${SEED:-42}"
FREEZE_BAIT_PARAMS="${FREEZE_BAIT_PARAMS:-false}"

N_TRAIN="${N_TRAIN:-128}"
N_GOOD="${N_GOOD:-64}"
N_BAD="${N_BAD:-64}"
N_TRAIN_GOOD="${N_TRAIN_GOOD:-64}"
N_TRAIN_BAD="${N_TRAIN_BAD:-96}"
TRAIN_GOOD_OFFSET="${TRAIN_GOOD_OFFSET:-64}"
TRAIN_BAD_OFFSET="${TRAIN_BAD_OFFSET:-64}"
EVAL_DATA_SOURCE="${EVAL_DATA_SOURCE:-mlabonne}"
TRAIN_DATA_SOURCE="${TRAIN_DATA_SOURCE:-$EVAL_DATA_SOURCE}"
TRAIN_GOOD_DATA_SOURCE="${TRAIN_GOOD_DATA_SOURCE:-$TRAIN_DATA_SOURCE}"
TRAIN_BAD_DATA_SOURCE="${TRAIN_BAD_DATA_SOURCE:-$TRAIN_DATA_SOURCE}"
GENERAL_DATA_SOURCE="${GENERAL_DATA_SOURCE:-$TRAIN_DATA_SOURCE}"
HF_HOME_DIR="${HF_HOME_DIR:-${HF_HOME:-}}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export ANTI_HERETIC_MODEL_LOCAL_FILES_ONLY="${ANTI_HERETIC_MODEL_LOCAL_FILES_ONLY:-1}"
if [[ -n "$HF_HOME_DIR" ]]; then
    export HF_HOME="$HF_HOME_DIR"
    export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME_DIR/hub}"
    export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME_DIR/datasets}"
    export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME_DIR/hub}"
fi

model_slug() {
    local raw="$1"
    raw="${raw%/}"
    printf '%s' "$raw" | tr '/:. ' '____'
}

infer_coverage_layers_from_tag() {
    local tag="${1:-}"
    if [[ "$tag" =~ (^|_)cov([0-9]+)($|_) ]]; then
        printf '%s\n' "${BASH_REMATCH[2]}"
    fi
}

data_source_slug() {
    local raw="$1"
    raw="${raw%/}"
    printf '%s' "$raw" | tr '/:. ' '____'
}

data_source_tag() {
    local eval_source="${1:-mlabonne}"
    local train_source="${2:-$eval_source}"
    local general_source="${3:-$train_source}"
    local train_good_source="${4:-$train_source}"
    local train_bad_source="${5:-$train_source}"

    if [[ "$eval_source" == "mlabonne" && "$train_source" == "mlabonne" && "$general_source" == "mlabonne" && "$train_good_source" == "mlabonne" && "$train_bad_source" == "mlabonne" ]]; then
        printf ''
        return
    fi

    if [[ "$train_good_source" == "$train_source" && "$train_bad_source" == "$train_source" ]]; then
        printf 'ds_eval%s_train%s_gen%s' \
            "$(data_source_slug "$eval_source")" \
            "$(data_source_slug "$train_source")" \
            "$(data_source_slug "$general_source")"
    else
        printf 'ds_eval%s_traingood%s_trainbad%s_gen%s' \
            "$(data_source_slug "$eval_source")" \
            "$(data_source_slug "$train_good_source")" \
            "$(data_source_slug "$train_bad_source")" \
            "$(data_source_slug "$general_source")"
    fi
}

normalize_direction_scope() {
    local raw="${1:-both}"
    raw="$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]')"
    raw="${raw//_/-}"
    raw="${raw// /-}"
    case "$raw" in
        ""|both|mixed)
            printf 'both'
            ;;
        global|global-only|globalonly)
            printf 'global'
            ;;
        per-layer|perlayer|per-layer-only|perlayer-only)
            printf 'per-layer'
            ;;
        *)
            printf '%s' "$raw"
            ;;
    esac
}

direction_scope_slug() {
    local raw="${1:-both}"
    raw="$(normalize_direction_scope "$raw")"
    case "$raw" in
        ""|both)
            printf ''
            ;;
        global)
            printf 'global'
            ;;
        per-layer|perlayer)
            printf 'per_layer'
            ;;
        *)
            printf '%s' "$raw" | tr '/:. ' '____'
            ;;
    esac
}

scoped_checkpoint_tag() {
    local tag="$1"
    local scope="${2:-both}"
    local suffix
    suffix="$(direction_scope_slug "$scope")"
    if [[ -z "$suffix" || "$tag" == *"_$suffix" ]]; then
        printf '%s' "$tag"
    else
        printf '%s_%s' "$tag" "$suffix"
    fi
}

rf_tag() {
    printf '%s' "$1" | sed 's/\./p/g; s/-/m/g'
}

timestamp() {
    date +"%Y%m%d_%H%M%S"
}

start_log() {
    local name="$1"
    if [[ "${NESTED_LOG_TO_STDOUT:-false}" == "true" ]]; then
        return
    fi
    mkdir -p "$LOG_DIR"
    if [[ ! -w "$LOG_DIR" ]]; then
        echo "[ERROR] Log directory is not writable: $LOG_DIR" >&2
        echo "        Fix ownership/permissions or set LOG_DIR to a writable path." >&2
        exit 1
    fi
    LOG_FILE="${LOG_FILE:-$LOG_DIR/${name}_$(timestamp).log}"
    if [[ -t 1 ]]; then
        exec > >(tee -a "$LOG_FILE") 2>&1
    else
        exec >>"$LOG_FILE" 2>&1
    fi
    echo "Log file:"
    echo "  $LOG_FILE"
    echo ""
}

auto_phase_layers() {
    local phase="$1"
    local style="$2"
    local coverage_layers="${HERETIC_COVERAGE_LAYERS:-${COVERAGE_LAYER_COUNT:-}}"

    "$PYTHON" - "$phase" "$style" "$BASE_MODEL" "$STEP1_PATH" "$coverage_layers" <<'PY'
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

phase = sys.argv[1]
style = sys.argv[2]
model = sys.argv[3]
step1_path = Path(sys.argv[4])
coverage_layers_raw = sys.argv[5].strip()
coverage_layers = int(coverage_layers_raw) if coverage_layers_raw else 0
if coverage_layers < 0:
    raise SystemExit(f"HERETIC_COVERAGE_LAYERS must be non-negative, got {coverage_layers}")


def from_step1(path: Path) -> int | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    layers = []
    for key in data:
        match = re.fullmatch(r"layer_(\d+)", key)
        if match:
            layers.append(int(match.group(1)))
    if layers:
        return max(layers) + 1
    ratios = data.get("fisher_ratios_baseline")
    if isinstance(ratios, list) and ratios:
        return len(ratios) - 1
    return None


def from_model_config(model_name: str) -> int | None:
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_name, local_files_only=True)
    except Exception:
        return None
    for name in ("num_hidden_layers", "n_layer", "num_layers"):
        value = getattr(cfg, name, None)
        if isinstance(value, int) and value > 0:
            return value
    text_config = getattr(cfg, "text_config", None)
    if isinstance(text_config, dict):
        for name in ("num_hidden_layers", "n_layer", "num_layers"):
            value = text_config.get(name)
            if isinstance(value, int) and value > 0:
                return value
    elif text_config is not None:
        for name in ("num_hidden_layers", "n_layer", "num_layers"):
            value = getattr(text_config, name, None)
            if isinstance(value, int) and value > 0:
                return value
    return None


def align(value: int, parity: int) -> int:
    return value if value % 2 == parity else value + 1


num_layers = from_step1(step1_path) or from_model_config(model)
if not num_layers:
    raise SystemExit(
        "Could not infer model layer count from STEP1_PATH or transformers config. "
        "Set PHASE1_LAYERS/PHASE2_LAYERS explicitly."
    )

if coverage_layers > 0:
    # Fixed two-phase coverage window. For Gemma-3-1B with 26 layers and
    # cov12 this yields Phase 1 = 13,15,17,19,21,23 and Phase 2 =
    # 14,16,18,20,22,24, i.e. exactly layers 13..24.
    stop_all = num_layers - 2
    start_all = max(0, stop_all - coverage_layers + 1)
    if phase == "1":
        parity = 1
    elif phase == "2":
        parity = 0
    else:
        raise SystemExit(f"Unknown phase: {phase}")
    start = align(start_all, parity)
    stop = stop_all
elif phase == "1":
    parity = 1
    # Cover the first Heretic global-direction escape point just past the
    # midpoint. For 26-layer Gemma this starts Phase 1 at L13 rather than L15.
    start_ratio = 0.50
    start = align(int(num_layers * start_ratio), parity)
    stop = num_layers - 3
elif phase == "2":
    parity = 0
    # Heretic's global direction_index is shifted by +1 internally and can
    # interpolate residual[13]↔residual[14] on 26-layer Gemma.  Bait@L13 only
    # poisons residual[14], so Phase 2 starts at L12 to cover residual[13].
    start = align(int(num_layers * 0.48), parity)
    stop = num_layers - 2
else:
    raise SystemExit(f"Unknown phase: {phase}")

stop = stop if stop % 2 == parity else stop - 1
layers = list(range(start, stop + 1, 2))
if not layers:
    raise SystemExit(
        f"Auto layer selection is empty for num_layers={num_layers}, phase={phase}, style={style}. "
        "Set PHASE1_LAYERS/PHASE2_LAYERS explicitly."
    )

print(",".join(str(x) for x in layers))
PY
}

require_python() {
    if ! "$PYTHON" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
    then
        echo "[ERROR] Python 3.10+ is required"
        exit 1
    fi
}

require_step1() {
    if [[ ! -f "$STEP1_PATH" ]]; then
        echo "[ERROR] Step1 summary not found: $STEP1_PATH"
        echo "        Set STEP1_PATH to an existing Step1 JSON, or run:"
        echo "          BASE_MODEL=\"$BASE_MODEL\" bash $SUITE_DIR/scripts/prepare_step1.sh"
        exit 1
    fi
}

print_common() {
    mkdir -p "$MODEL_SAVE_DIR" "$SUITE_RESULTS_DIR" "$LOG_DIR"
    echo "  suite dir           : $SUITE_DIR"
    echo "  base model          : $BASE_MODEL"
    echo "  tokenizer           : $TOKENIZER"
    echo "  step1 path          : $STEP1_PATH"
    echo "  suite results dir   : $SUITE_RESULTS_DIR"
    echo "  model save dir      : $MODEL_SAVE_DIR"
    echo "  log dir             : $LOG_DIR"
    echo "  device              : $DEVICE"
    echo "  gpu mode            : $GPU_MODE"
    if [[ -n "$GPU_ID" ]]; then
        echo "  GPU_ID              : $GPU_ID"
    fi
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    fi
    echo "  style               : $STYLE"
    echo "  eval data source    : $EVAL_DATA_SOURCE"
    echo "  train data source   : $TRAIN_DATA_SOURCE"
    echo "  train good source   : $TRAIN_GOOD_DATA_SOURCE"
    echo "  train bad source    : $TRAIN_BAD_DATA_SOURCE"
    echo "  general data source : $GENERAL_DATA_SOURCE"
    echo "  model local only    : $ANTI_HERETIC_MODEL_LOCAL_FILES_ONLY"
    if [[ -n "$HF_HOME_DIR" ]]; then
        echo "  HF_HOME             : $HF_HOME"
        echo "  HF hub cache        : $HUGGINGFACE_HUB_CACHE"
        echo "  HF datasets cache   : $HF_DATASETS_CACHE"
    fi
}
