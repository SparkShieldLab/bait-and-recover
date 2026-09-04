#!/usr/bin/env bash
# Run one Heretic study with multiple worker processes.
#
# All workers share the same CHECKPOINT_TAG / JournalStorage and cooperate toward
# a single global N_TRIALS budget. This is for using multiple GPUs on one target
# model without changing the final journal format.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

MODEL_SLUG="${MODEL_SLUG:-$(model_slug "$BASE_MODEL")}"

HERETIC_CUDA_DEVICES="${HERETIC_CUDA_DEVICES:-0,1,2,3}"
IFS=',' read -r -a DEVICE_ARRAY <<<"$HERETIC_CUDA_DEVICES"

HERETIC_WORKERS="${HERETIC_WORKERS:-${#DEVICE_ARRAY[@]}}"
N_TRIALS="${N_TRIALS:-200}"
HERETIC_DIRECTION_SCOPE="$(normalize_direction_scope "${HERETIC_DIRECTION_SCOPE:-both}")"
HERETIC_SAMPLER_SEED_BASE="${HERETIC_SAMPLER_SEED_BASE:-42}"
HERETIC_SAMPLER_SEED_MODE="${HERETIC_SAMPLER_SEED_MODE:-offset}"
CHECKPOINT_TAG="${CHECKPOINT_TAG:-defended_${MODEL_SLUG}_parallel_seed${HERETIC_SAMPLER_SEED_BASE}}"
CHECKPOINT_TAG="$(scoped_checkpoint_tag "$CHECKPOINT_TAG" "$HERETIC_DIRECTION_SCOPE")"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-$SUITE_DIR/heretic_checkpoints}"
CHECKPOINT_DIR="$CHECKPOINT_BASE/$CHECKPOINT_TAG"
HERETIC_INIT_TIMEOUT_SECONDS="${HERETIC_INIT_TIMEOUT_SECONDS:-300}"

if (( HERETIC_WORKERS < 1 )); then
    echo "[ERROR] HERETIC_WORKERS must be >= 1"
    exit 1
fi

if [[ "$HERETIC_SAMPLER_SEED_MODE" == "exact" && "$HERETIC_WORKERS" != "1" ]]; then
    echo "[ERROR] HERETIC_SAMPLER_SEED_MODE=exact reproduces the single-process sampler path."
    echo "        It requires HERETIC_WORKERS=1 because TPE suggestions depend on completed trial results."
    echo "        For multi-worker search, use HERETIC_SAMPLER_SEED_MODE=offset or same."
    exit 1
fi

if (( HERETIC_WORKERS > ${#DEVICE_ARRAY[@]} )); then
    echo "[ERROR] HERETIC_WORKERS=$HERETIC_WORKERS exceeds HERETIC_CUDA_DEVICES count=${#DEVICE_ARRAY[@]}"
    exit 1
fi

if [[ -z "${TARGET_TAG:-}" && -z "${TARGET_MODEL:-}" ]]; then
    case " $* " in
        *" --baseline "*) ;;
        *)
            echo "[ERROR] Set TARGET_TAG or TARGET_MODEL, or pass --baseline."
            exit 1
            ;;
    esac
fi

start_log "heretic_parallel_${CHECKPOINT_TAG}"

echo "Parallel Heretic run"
print_common
echo "  checkpoint tag      : $CHECKPOINT_TAG"
echo "  checkpoint dir      : $CHECKPOINT_DIR"
echo "  total trials        : $N_TRIALS"
echo "  direction scope     : $HERETIC_DIRECTION_SCOPE"
echo "  workers             : $HERETIC_WORKERS"
echo "  cuda devices        : $HERETIC_CUDA_DEVICES"
echo "  sampler seed base   : $HERETIC_SAMPLER_SEED_BASE"
echo "  sampler seed mode   : $HERETIC_SAMPLER_SEED_MODE"
if [[ "$HERETIC_SAMPLER_SEED_MODE" == "same" && "$HERETIC_WORKERS" != "1" ]]; then
    echo "  seed note           : same seed per worker; not identical to single-process TPE"
elif [[ "$HERETIC_SAMPLER_SEED_MODE" == "exact" ]]; then
    echo "  seed note           : strict single-process TPE reproduction"
fi
echo ""

pids=()
start_worker() {
    local worker="$1"
    shift
    local device="${DEVICE_ARRAY[$worker]}"
    local seed
    case "$HERETIC_SAMPLER_SEED_MODE" in
        offset)
            seed=$((HERETIC_SAMPLER_SEED_BASE + worker))
            ;;
        same)
            seed="$HERETIC_SAMPLER_SEED_BASE"
            ;;
        exact)
            seed="$HERETIC_SAMPLER_SEED_BASE"
            ;;
        *)
            echo "[ERROR] Unknown HERETIC_SAMPLER_SEED_MODE=$HERETIC_SAMPLER_SEED_MODE. Use offset, same, or exact."
            exit 1
            ;;
    esac
    local worker_log="$LOG_DIR/heretic_${CHECKPOINT_TAG}_worker${worker}_$(timestamp).log"

    echo "Starting worker $worker on CUDA_VISIBLE_DEVICES=$device with HERETIC_SAMPLER_SEED=$seed"
    (
        export CUDA_VISIBLE_DEVICES="$device"
        export CHECKPOINT_TAG="$CHECKPOINT_TAG"
        export N_TRIALS="$N_TRIALS"
        export HERETIC_DIRECTION_SCOPE="$HERETIC_DIRECTION_SCOPE"
        export HERETIC_SAMPLER_SEED="$seed"
        export LOG_FILE="$worker_log"
        export TARGET_TAG="${TARGET_TAG:-}"
        export TARGET_MODEL="${TARGET_MODEL:-}"
        export BASE_MODEL TOKENIZER MODEL_SAVE_DIR LOG_DIR PYTHON
        export CHECKPOINT_BASE
        bash "$SCRIPT_DIR/run_heretic.sh" "$@"
    ) &
    pids+=("$!")
}

count_studies() {
    "$PYTHON" - "$CHECKPOINT_DIR" <<'PY'
import json
import sys
from pathlib import Path

checkpoint_dir = Path(sys.argv[1])
count = 0
for path in checkpoint_dir.glob("*.jsonl"):
    try:
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                if json.loads(line).get("op_code") == 0:
                    count += 1
    except json.JSONDecodeError:
        pass
print(count)
PY
}

wait_for_study_init() {
    local first_pid="$1"
    local waited=0
    mkdir -p "$CHECKPOINT_DIR"

    while (( waited < HERETIC_INIT_TIMEOUT_SECONDS )); do
        local study_count
        study_count="$(count_studies)"
        if (( study_count == 1 )); then
            return 0
        fi
        if (( study_count > 1 )); then
            echo "[ERROR] Checkpoint journal contains $study_count Optuna studies; delete it and rerun:"
            echo "        rm -rf \"$CHECKPOINT_DIR\""
            return 1
        fi

        if ! kill -0 "$first_pid" 2>/dev/null; then
            echo "[ERROR] Worker 0 exited before creating the Optuna study."
            return 1
        fi

        sleep 2
        waited=$((waited + 2))
    done

    echo "[ERROR] Timed out waiting for worker 0 to initialize the Optuna study."
    return 1
}

mkdir -p "$CHECKPOINT_DIR"
existing_study_count="$(count_studies)"
if (( existing_study_count > 1 )); then
    echo "[ERROR] Checkpoint journal already contains $existing_study_count Optuna studies; delete it and rerun:"
    echo "        rm -rf \"$CHECKPOINT_DIR\""
    exit 1
fi

start_worker 0 "$@"
if (( HERETIC_WORKERS > 1 && existing_study_count == 0 )); then
    wait_for_study_init "${pids[0]}"
    echo "Optuna study initialized; starting remaining workers."
elif (( HERETIC_WORKERS > 1 )); then
    echo "Existing Optuna study found; starting remaining workers."
fi

for ((worker = 1; worker < HERETIC_WORKERS; worker++)); do
    start_worker "$worker" "$@"
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done

echo ""
if (( status == 0 )); then
    echo "Parallel Heretic complete."
else
    echo "[ERROR] One or more Heretic workers failed."
fi
echo "  checkpoint tag      : $CHECKPOINT_TAG"
echo "  checkpoint dir      : $CHECKPOINT_DIR"
echo "  logs                : $LOG_DIR/heretic_${CHECKPOINT_TAG}_worker*.log"

exit "$status"
