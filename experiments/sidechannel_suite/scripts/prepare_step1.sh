#!/usr/bin/env bash
# Generate the Step1 SVD/Fisher prior for BASE_MODEL into this suite's results.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

require_python
mkdir -p "$MODEL_SAVE_DIR" "$SUITE_RESULTS_DIR" "$LOG_DIR"

MODEL_SLUG="${MODEL_SLUG:-$(model_slug "$BASE_MODEL")}"
EXPECTED_STEP1="$SUITE_RESULTS_DIR/step1_results_${MODEL_SLUG}.json"

start_log "prepare_step1_${MODEL_SLUG}"

echo "Step1 prior generation"
print_common
echo "  expected output     : $EXPECTED_STEP1"
echo ""

ANTI_HERETIC_RESULTS_DIR="$SUITE_RESULTS_DIR" \
"$PYTHON" "$EXPERIMENTS_DIR/step1_subspace_diagnostic.py" \
    --model "$BASE_MODEL" \
    --device "$DEVICE" \
    --n-good "$N_GOOD" \
    --n-bad "$N_BAD" \
    --data-source "$EVAL_DATA_SOURCE" \
    --batch-size "$BATCH_SIZE" \
    "$@"

if [[ ! -f "$EXPECTED_STEP1" ]]; then
    echo "[ERROR] Step1 script completed, but expected output was not found:"
    echo "        $EXPECTED_STEP1"
    exit 1
fi

echo ""
echo "Step1 prior ready:"
echo "  $EXPECTED_STEP1"
