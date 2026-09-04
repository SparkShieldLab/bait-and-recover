#!/usr/bin/env bash
# Run Heretic on a baseline or defended model.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

if [[ -n "${HERETIC_HF_HOME:-}" ]]; then
    export HF_HOME="$HERETIC_HF_HOME"
fi
if [[ -n "${HF_HOME:-}" ]]; then
    export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
    export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
    export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
fi
export HF_HUB_OFFLINE="${HERETIC_HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${HERETIC_TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HERETIC_HF_DATASETS_OFFLINE:-1}"
export HERETIC_EXIT_AFTER_OPTIMIZATION="${HERETIC_EXIT_AFTER_OPTIMIZATION:-true}"
export HERETIC_SAMPLER_SEED="${HERETIC_SAMPLER_SEED:-42}"
export HERETIC_PRINT_TIMING="${HERETIC_PRINT_TIMING:-false}"

N_TRIALS="${N_TRIALS:-200}"
HERETIC_BATCH_SIZE="${HERETIC_BATCH_SIZE:-256}"
HERETIC_DIRECTION_SCOPE="$(normalize_direction_scope "${HERETIC_DIRECTION_SCOPE:-both}")"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-$SCRIPT_DIR/../heretic_checkpoints}"

USE_BASELINE=false
EXTRA_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --baseline)
            USE_BASELINE=true
            ;;
        *)
            EXTRA_ARGS+=("$arg")
            ;;
    esac
done

HERETIC_TRUST_REMOTE_CODE="${HERETIC_TRUST_REMOTE_CODE:-${ANTI_HERETIC_TRUST_REMOTE_CODE:-}}"
case "$HERETIC_TRUST_REMOTE_CODE" in
    1|true|TRUE|yes|YES)
        EXTRA_ARGS+=(--trust-remote-code true)
        ;;
    ""|0|false|FALSE|no|NO)
        ;;
    *)
        echo "[ERROR] HERETIC_TRUST_REMOTE_CODE must be a boolean: $HERETIC_TRUST_REMOTE_CODE"
        exit 1
        ;;
esac

if [[ -n "${HERETIC_PROMPT_DIR:-}" ]]; then
    GOOD_PROMPTS="$HERETIC_PROMPT_DIR/good.jsonl"
    BAD_PROMPTS="$HERETIC_PROMPT_DIR/bad.jsonl"
    if [[ ! -f "$GOOD_PROMPTS" || ! -f "$BAD_PROMPTS" ]]; then
        echo "[ERROR] HERETIC_PROMPT_DIR must contain good.jsonl and bad.jsonl: $HERETIC_PROMPT_DIR"
        exit 1
    fi
    EXTRA_ARGS+=(
        --good-prompts.dataset "$GOOD_PROMPTS"
        --good-prompts.split "train[:4]"
        --good-prompts.column prompt
        --bad-prompts.dataset "$BAD_PROMPTS"
        --bad-prompts.split "train[:4]"
        --bad-prompts.column prompt
        --good-evaluation-prompts.dataset "$GOOD_PROMPTS"
        --good-evaluation-prompts.split "train[:4]"
        --good-evaluation-prompts.column prompt
        --bad-evaluation-prompts.dataset "$BAD_PROMPTS"
        --bad-evaluation-prompts.split "train[:4]"
        --bad-evaluation-prompts.column prompt
    )
fi

MODEL_SLUG="${MODEL_SLUG:-$(model_slug "$BASE_MODEL")}"

if $USE_BASELINE; then
    TARGET="$BASE_MODEL"
    CHECKPOINT_TAG="${CHECKPOINT_TAG:-baseline_${MODEL_SLUG}}"
else
    # A pipeline-supplied tag is more specific than a possibly stale inherited
    # TARGET_MODEL. Direct one-off runs can still provide TARGET_MODEL alone.
    if [[ -n "${TARGET_TAG:-}" ]]; then
        TARGET="$MODEL_SAVE_DIR/step4_merged_${TARGET_TAG}"
    elif [[ -n "${TARGET_MODEL:-}" ]]; then
        TARGET="$TARGET_MODEL"
    else
        echo "[ERROR] Set TARGET_TAG or TARGET_MODEL, or pass --baseline."
        exit 1
    fi
    CHECKPOINT_TAG="${CHECKPOINT_TAG:-defended_${MODEL_SLUG}}"
    if [[ ! -d "$TARGET" ]]; then
        echo "[ERROR] Defended model not found: $TARGET"
        exit 1
    fi
fi
CHECKPOINT_TAG="$(scoped_checkpoint_tag "$CHECKPOINT_TAG" "$HERETIC_DIRECTION_SCOPE")"

LOG_NAME="heretic_${CHECKPOINT_TAG}"
start_log "$LOG_NAME"

CHECKPOINT_DIR="$CHECKPOINT_BASE/$CHECKPOINT_TAG"
mkdir -p "$CHECKPOINT_DIR"

echo "Heretic run"
echo "  target              : $TARGET"
echo "  checkpoint dir      : $CHECKPOINT_DIR"
echo "  trials              : $N_TRIALS"
echo "  batch size          : $HERETIC_BATCH_SIZE"
echo "  direction scope     : $HERETIC_DIRECTION_SCOPE"
echo "  exit after optimize : $HERETIC_EXIT_AFTER_OPTIMIZATION"
echo "  sampler seed        : $HERETIC_SAMPLER_SEED"
echo "  print timing        : $HERETIC_PRINT_TIMING"
echo "  HF cache            : ${HF_HOME:-$HOME/.cache/huggingface}"
echo "  datasets offline    : $HF_DATASETS_OFFLINE"
echo "  trust remote code   : ${HERETIC_TRUST_REMOTE_CODE:-default}"
echo ""

HERETIC_SOURCE_DIR="${HERETIC_SOURCE_DIR:-$REPO_DIR/external/heretic}"
if [[ -d "$HERETIC_SOURCE_DIR/src" ]]; then
    HERETIC_PYTHONPATH="$HERETIC_SOURCE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
else
    HERETIC_PYTHONPATH="${PYTHONPATH:-}"
fi

if ! PYTHONPATH="$HERETIC_PYTHONPATH" "$PYTHON" -c \
    "from importlib.metadata import version; from heretic.main import main; version('heretic-llm')" \
    2>/dev/null; then
    echo "[ERROR] Patched Heretic is not available."
    echo "        Run: bash scripts/setup_heretic.sh"
    echo "        Or set HERETIC_SOURCE_DIR to an existing patched Heretic checkout."
    exit 1
fi

PYTHONPATH="$HERETIC_PYTHONPATH" \
"$PYTHON" -c "from heretic.main import main; main()" \
    --model "$TARGET" \
    --n-trials "$N_TRIALS" \
    --batch-size "$HERETIC_BATCH_SIZE" \
    --direction-scope "$HERETIC_DIRECTION_SCOPE" \
    --print-residual-geometry \
    --study-checkpoint-dir "$CHECKPOINT_DIR" \
    "${EXTRA_ARGS[@]}"
