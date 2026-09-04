#!/usr/bin/env bash
# Train Phase 1 sidechannel defenses.
#
# STYLE=oldstyle   : short progressive stages; optional joint finetune via SKIP_JOINT=false.
# STYLE=ideal_pair : cumulative sequential pair training, no joint finetune.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

MODEL_SLUG="${MODEL_SLUG:-$(model_slug "$BASE_MODEL")}"

case "$STYLE" in
    oldstyle)
        RF_WEIGHT="${RF_WEIGHT:-1.0}"
        PROGRESSIVE_STAGES="${PROGRESSIVE_STAGES:-2}"
        N_EPOCHS="${N_EPOCHS:-40}"
        USE_CUMULATIVE="${USE_CUMULATIVE:-false}"
        SKIP_JOINT="${SKIP_JOINT:-true}"
        ;;
    ideal_pair)
        RF_WEIGHT="${RF_WEIGHT:-0.25}"
        PROGRESSIVE_STAGES="${PROGRESSIVE_STAGES:-40}"
        N_EPOCHS="${N_EPOCHS:-40}"
        USE_CUMULATIVE="${USE_CUMULATIVE:-true}"
        SKIP_JOINT="${SKIP_JOINT:-true}"
        ;;
    *)
        echo "[ERROR] Unknown STYLE=$STYLE. Use oldstyle or ideal_pair."
        exit 1
        ;;
esac

RF_TAG="$(rf_tag "$RF_WEIGHT")"
DATA_TAG="$(data_source_tag "$EVAL_DATA_SOURCE" "$TRAIN_DATA_SOURCE" "$GENERAL_DATA_SOURCE" "$TRAIN_GOOD_DATA_SOURCE" "$TRAIN_BAD_DATA_SOURCE")"
if [[ -n "$DATA_TAG" ]]; then
    DEFAULT_EXPERIMENT_TAG="phase1_sidechannel_${STYLE}_rf${RF_TAG}_${DATA_TAG}_${MODEL_SLUG}"
else
    DEFAULT_EXPERIMENT_TAG="phase1_sidechannel_${STYLE}_rf${RF_TAG}_${MODEL_SLUG}"
fi
EXPERIMENT_TAG="${EXPERIMENT_TAG:-$DEFAULT_EXPERIMENT_TAG}"
HERETIC_COVERAGE_LAYERS="${HERETIC_COVERAGE_LAYERS:-${COVERAGE_LAYER_COUNT:-}}"
if [[ -z "$HERETIC_COVERAGE_LAYERS" ]]; then
    HERETIC_COVERAGE_LAYERS="$(infer_coverage_layers_from_tag "$EXPERIMENT_TAG")"
fi
export HERETIC_COVERAGE_LAYERS

require_python
require_step1

if [[ -z "${PHASE1_LAYERS:-}" ]]; then
    PHASE1_LAYERS="$(auto_phase_layers 1 "$STYLE")"
    PHASE1_LAYERS_SOURCE="auto"
else
    PHASE1_LAYERS_SOURCE="env"
fi

NUM_PAIRS="${NUM_PAIRS:-$(awk -F',' '{print NF}' <<<"$PHASE1_LAYERS")}"
MIN_JOINT_EPOCHS="${MIN_JOINT_EPOCHS:-2}"
TRAIN_LR="${TRAIN_LR:-2e-4}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

start_log "train_phase1_${EXPERIMENT_TAG}"

if [[ "$SKIP_JOINT" != "true" ]]; then
    JOINT_EPOCHS=$((N_EPOCHS - PROGRESSIVE_STAGES * NUM_PAIRS))
    if (( JOINT_EPOCHS < MIN_JOINT_EPOCHS )); then
        echo "[ERROR] Schedule leaves only ${JOINT_EPOCHS} joint epoch(s)."
        echo "        Increase N_EPOCHS or lower PROGRESSIVE_STAGES."
        exit 1
    fi
else
    JOINT_EPOCHS=0
fi

EXTRA_PROGRESSIVE_FLAGS=()
if [[ "$USE_CUMULATIVE" == "true" ]]; then
    EXTRA_PROGRESSIVE_FLAGS+=(--progressive-cumulative-stages)
fi
if [[ "$SKIP_JOINT" == "true" ]]; then
    EXTRA_PROGRESSIVE_FLAGS+=(--skip-progressive-joint-finetune)
fi

FREEZE_BAIT_PARAMS="${FREEZE_BAIT_PARAMS:-true}"
if [[ -z "${BAIT_FREEZE_MODE:-}" ]]; then
    if [[ "$FREEZE_BAIT_PARAMS" == "true" ]]; then
        BAIT_FREEZE_MODE="full"
    else
        BAIT_FREEZE_MODE="none"
    fi
fi
EXTRA_BAIT_FLAGS=()
case "$BAIT_FREEZE_MODE" in
    full)
        EXTRA_BAIT_FLAGS+=(--freeze-bait --freeze-bait-gate)
        ;;
    coeff)
        EXTRA_BAIT_FLAGS+=(--freeze-bait-coeff)
        ;;
    none)
        ;;
    *)
        echo "[ERROR] Unknown BAIT_FREEZE_MODE=$BAIT_FREEZE_MODE. Use full, coeff, or none."
        exit 1
        ;;
esac

AUTO_NORMALIZE_LOSS_WEIGHTS="${AUTO_NORMALIZE_LOSS_WEIGHTS:-true}"
EXTRA_LOSS_NORM_FLAGS=()
if [[ "$AUTO_NORMALIZE_LOSS_WEIGHTS" == "false" ]]; then
    EXTRA_LOSS_NORM_FLAGS+=(--no-auto-normalize-loss-weights)
else
    EXTRA_LOSS_NORM_FLAGS+=(--auto-normalize-loss-weights)
fi
EXTRA_LOSS_NORM_FLAGS+=(
    --loss-normalization-target-ratio "${LOSS_NORMALIZATION_TARGET_RATIO:-0.5}"
    --loss-normalization-batch-size "${LOSS_NORMALIZATION_BATCH_SIZE:-0}"
)
LEGACY_DETACH_LOCAL_LOSS_FROM_BAIT="${DETACH_LOCAL_LOSS_FROM_BAIT:-}"
DETACH_RECOVERY_LOSS_FROM_BAIT="${DETACH_RECOVERY_LOSS_FROM_BAIT:-${LEGACY_DETACH_LOCAL_LOSS_FROM_BAIT:-true}}"
DETACH_VISIBLE_LOSS_FROM_BAIT="${DETACH_VISIBLE_LOSS_FROM_BAIT:-false}"
DETACH_KL_LOSS_FROM_BAIT="${DETACH_KL_LOSS_FROM_BAIT:-${LEGACY_DETACH_LOCAL_LOSS_FROM_BAIT:-true}}"
EXTRA_LOCAL_BAIT_GRAD_FLAGS=()
if [[ "$DETACH_RECOVERY_LOSS_FROM_BAIT" == "false" ]]; then
    EXTRA_LOCAL_BAIT_GRAD_FLAGS+=(--no-detach-recovery-loss-from-bait)
else
    EXTRA_LOCAL_BAIT_GRAD_FLAGS+=(--detach-recovery-loss-from-bait)
fi
if [[ "$DETACH_VISIBLE_LOSS_FROM_BAIT" == "false" ]]; then
    EXTRA_LOCAL_BAIT_GRAD_FLAGS+=(--no-detach-visible-loss-from-bait)
else
    EXTRA_LOCAL_BAIT_GRAD_FLAGS+=(--detach-visible-loss-from-bait)
fi
if [[ "$DETACH_KL_LOSS_FROM_BAIT" == "false" ]]; then
    EXTRA_LOCAL_BAIT_GRAD_FLAGS+=(--no-detach-kl-loss-from-bait)
else
    EXTRA_LOCAL_BAIT_GRAD_FLAGS+=(--detach-kl-loss-from-bait)
fi

SIDECHANNEL_TAG_TRIGGER_MODE="${SIDECHANNEL_TAG_TRIGGER_MODE:-shared_supervised}"
SIDECHANNEL_TAG_TRIGGER_MIX="${SIDECHANNEL_TAG_TRIGGER_MIX:-0.25}"
BAIT_GATE_LOG_INIT="${BAIT_GATE_LOG_INIT:--2.0}"
RESIDUAL_FISHER_LOSS_MODE="${RESIDUAL_FISHER_LOSS_MODE:-detached_gap}"
RESIDUAL_FISHER_TARGET_SPACE="${RESIDUAL_FISHER_TARGET_SPACE:-projected}"
RESIDUAL_DIRECTION_LOSS_WEIGHT="${RESIDUAL_DIRECTION_LOSS_WEIGHT:-0.0}"
BAIT_ANTI_COHERENCE_LOSS_WEIGHT="${BAIT_ANTI_COHERENCE_LOSS_WEIGHT:-0.0}"
BAIT_ANTI_COHERENCE_COMPONENTS="${BAIT_ANTI_COHERENCE_COMPONENTS:-2}"
BAIT_ANTI_COHERENCE_MARGIN="${BAIT_ANTI_COHERENCE_MARGIN:-0.0}"
RESIDUAL_GAP_ANTI_COHERENCE_LOSS_WEIGHT="${RESIDUAL_GAP_ANTI_COHERENCE_LOSS_WEIGHT:-0.0}"
RESIDUAL_GAP_ANTI_COHERENCE_MARGIN="${RESIDUAL_GAP_ANTI_COHERENCE_MARGIN:-0.0}"
GLOBAL_DIRECTION_LOSS_WEIGHT="${GLOBAL_DIRECTION_LOSS_WEIGHT:-0.0}"
GLOBAL_DIRECTION_SAMPLES="${GLOBAL_DIRECTION_SAMPLES:-5}"
GLOBAL_DIRECTION_MARGIN="${GLOBAL_DIRECTION_MARGIN:-0.0}"
GLOBAL_DIRECTION_RANGE_LOW="${GLOBAL_DIRECTION_RANGE_LOW:-0.4}"
GLOBAL_DIRECTION_RANGE_HIGH="${GLOBAL_DIRECTION_RANGE_HIGH:-0.9}"
PROGRESSIVE_GEOMETRY_JOINT_EPOCHS="${PROGRESSIVE_GEOMETRY_JOINT_EPOCHS:-0}"
PROGRESSIVE_GEOMETRY_JOINT_LR_SCALE="${PROGRESSIVE_GEOMETRY_JOINT_LR_SCALE:-0.05}"
PROGRESSIVE_GEOMETRY_JOINT_INCLUDE_LOCAL_LOSSES="${PROGRESSIVE_GEOMETRY_JOINT_INCLUDE_LOCAL_LOSSES:-false}"
PROGRESSIVE_GEOMETRY_JOINT_TRAIN_BAIT="${PROGRESSIVE_GEOMETRY_JOINT_TRAIN_BAIT:-false}"
PROGRESSIVE_GEOMETRY_JOINT_KL_ROLLBACK_BUDGET="${PROGRESSIVE_GEOMETRY_JOINT_KL_ROLLBACK_BUDGET:-0.0}"
PROGRESSIVE_GEOMETRY_JOINT_KL_LOSS_WEIGHT="${PROGRESSIVE_GEOMETRY_JOINT_KL_LOSS_WEIGHT:-0.0}"
PROGRESSIVE_GEOMETRY_JOINT_INTERPOLATION_STEPS="${PROGRESSIVE_GEOMETRY_JOINT_INTERPOLATION_STEPS:-8}"
PROGRESSIVE_BLOCK_JOINT_EPOCHS="${PROGRESSIVE_BLOCK_JOINT_EPOCHS:-0}"
PROGRESSIVE_BLOCK_JOINT_SIZE="${PROGRESSIVE_BLOCK_JOINT_SIZE:-0}"
PROGRESSIVE_BLOCK_JOINT_STRIDE="${PROGRESSIVE_BLOCK_JOINT_STRIDE:-0}"
PROGRESSIVE_BLOCK_JOINT_LR_SCALE="${PROGRESSIVE_BLOCK_JOINT_LR_SCALE:-0.05}"
PROGRESSIVE_BLOCK_JOINT_KL_ROLLBACK_BUDGET="${PROGRESSIVE_BLOCK_JOINT_KL_ROLLBACK_BUDGET:-0.0}"
PROGRESSIVE_BLOCK_JOINT_INTERPOLATION_STEPS="${PROGRESSIVE_BLOCK_JOINT_INTERPOLATION_STEPS:-8}"
PROGRESSIVE_BLOCK_JOINT_GEOMETRY_FALLBACK="${PROGRESSIVE_BLOCK_JOINT_GEOMETRY_FALLBACK:-true}"
PROGRESSIVE_BLOCK_JOINT_DETACH_VISIBLE_BAIT="${PROGRESSIVE_BLOCK_JOINT_DETACH_VISIBLE_BAIT:-true}"
PROGRESSIVE_BLOCK_JOINT_INCLUDE_LOCAL_LOSSES="${PROGRESSIVE_BLOCK_JOINT_INCLUDE_LOCAL_LOSSES:-true}"
GEOMETRY_JOINT_MODE="geometry-only"
EXTRA_GEOMETRY_JOINT_FLAGS=()
if [[ "$PROGRESSIVE_GEOMETRY_JOINT_INCLUDE_LOCAL_LOSSES" == "true" ]]; then
    GEOMETRY_JOINT_MODE="behavior+geometry"
    EXTRA_GEOMETRY_JOINT_FLAGS+=(--progressive-geometry-joint-include-local-losses)
fi
if [[ "$PROGRESSIVE_GEOMETRY_JOINT_TRAIN_BAIT" == "true" ]]; then
    EXTRA_GEOMETRY_JOINT_FLAGS+=(--progressive-geometry-joint-train-bait)
fi
EXTRA_BLOCK_JOINT_FLAGS=()
if [[ "$PROGRESSIVE_BLOCK_JOINT_GEOMETRY_FALLBACK" != "true" ]]; then
    EXTRA_BLOCK_JOINT_FLAGS+=(--no-progressive-block-joint-geometry-fallback)
fi
if [[ "$PROGRESSIVE_BLOCK_JOINT_DETACH_VISIBLE_BAIT" != "true" ]]; then
    EXTRA_BLOCK_JOINT_FLAGS+=(--no-progressive-block-joint-detach-visible-bait)
fi
if [[ "$PROGRESSIVE_BLOCK_JOINT_INCLUDE_LOCAL_LOSSES" != "true" ]]; then
    EXTRA_BLOCK_JOINT_FLAGS+=(--progressive-block-joint-disable-local-losses)
fi
PROGRESSIVE_KL_BUDGET="${PROGRESSIVE_KL_BUDGET:-0.003}"
BAIT_CALIBRATION_TARGET="${BAIT_CALIBRATION_TARGET:-0.02}"
BAIT_CALIBRATION_QUANTILE="${BAIT_CALIBRATION_QUANTILE:-0.90}"
BAIT_CALIBRATION_LOG_MIN="${BAIT_CALIBRATION_LOG_MIN:--8.0}"
BAIT_CALIBRATION_LOG_MAX="${BAIT_CALIBRATION_LOG_MAX:-0.0}"
BAIT_CALIBRATION_KL_TARGET="${BAIT_CALIBRATION_KL_TARGET:-0.0}"
BAIT_CALIBRATION_KL_ITERS="${BAIT_CALIBRATION_KL_ITERS:-2}"
AUTO_CALIBRATE_BAIT_GATE="${AUTO_CALIBRATE_BAIT_GATE:-true}"
EXTRA_BAIT_CALIBRATION_FLAGS=()
if [[ "$AUTO_CALIBRATE_BAIT_GATE" == "false" ]]; then
    EXTRA_BAIT_CALIBRATION_FLAGS+=(--no-auto-calibrate-bait-gate)
else
    EXTRA_BAIT_CALIBRATION_FLAGS+=(--auto-calibrate-bait-gate)
fi
EXTRA_COVERAGE_FLAGS=()
if [[ -n "$HERETIC_COVERAGE_LAYERS" ]]; then
    EXTRA_COVERAGE_FLAGS+=(--heretic-coverage-layers "$HERETIC_COVERAGE_LAYERS")
fi
FINAL_KL_BUDGET="${FINAL_KL_BUDGET:-}"
EXTRA_FINAL_KL_FLAGS=()
if [[ -n "$FINAL_KL_BUDGET" ]]; then
    EXTRA_FINAL_KL_FLAGS+=(--final-kl-budget "$FINAL_KL_BUDGET")
fi

echo "Phase 1 sidechannel training"
print_common
echo "  experiment tag      : $EXPERIMENT_TAG"
echo "  phase1 layers       : $PHASE1_LAYERS"
echo "  phase1 layers source: $PHASE1_LAYERS_SOURCE"
echo "  coverage layers     : ${HERETIC_COVERAGE_LAYERS:-auto}"
echo "  RF weight           : $RF_WEIGHT"
echo "  KL weight           : ${KL_LOSS_WEIGHT:-1.0}"
echo "  KL loss mode        : ${KL_LOSS_MODE:-full}"
echo "  KL top-k            : ${KL_TOP_K:-64}"
echo "  bait freeze mode    : $BAIT_FREEZE_MODE"
echo "  tag trigger mode    : $SIDECHANNEL_TAG_TRIGGER_MODE"
echo "  tag trigger mix     : $SIDECHANNEL_TAG_TRIGGER_MIX"
echo "  bait gate log init  : $BAIT_GATE_LOG_INIT"
echo "  RF loss mode        : $RESIDUAL_FISHER_LOSS_MODE"
echo "  RF target space     : $RESIDUAL_FISHER_TARGET_SPACE"
echo "  direction loss w    : $RESIDUAL_DIRECTION_LOSS_WEIGHT"
echo "  anti-coherence w    : $BAIT_ANTI_COHERENCE_LOSS_WEIGHT"
echo "  anti-coherence comps: $BAIT_ANTI_COHERENCE_COMPONENTS"
echo "  anti-coherence margin: $BAIT_ANTI_COHERENCE_MARGIN"
echo "  gap anti-coherence w: $RESIDUAL_GAP_ANTI_COHERENCE_LOSS_WEIGHT"
echo "  gap anti-coh margin : $RESIDUAL_GAP_ANTI_COHERENCE_MARGIN"
echo "  global direction w  : $GLOBAL_DIRECTION_LOSS_WEIGHT"
echo "  global dir samples  : $GLOBAL_DIRECTION_SAMPLES"
echo "  global dir margin   : $GLOBAL_DIRECTION_MARGIN"
echo "  geometry joint ep   : $PROGRESSIVE_GEOMETRY_JOINT_EPOCHS"
echo "  geometry joint lr   : $PROGRESSIVE_GEOMETRY_JOINT_LR_SCALE"
echo "  geometry joint mode : $GEOMETRY_JOINT_MODE"
echo "  geometry bait train : $PROGRESSIVE_GEOMETRY_JOINT_TRAIN_BAIT"
echo "  geometry KL guard   : $PROGRESSIVE_GEOMETRY_JOINT_KL_ROLLBACK_BUDGET"
echo "  geometry KL anchor  : $PROGRESSIVE_GEOMETRY_JOINT_KL_LOSS_WEIGHT"
echo "  geometry interp step: $PROGRESSIVE_GEOMETRY_JOINT_INTERPOLATION_STEPS"
echo "  block joint epochs  : $PROGRESSIVE_BLOCK_JOINT_EPOCHS"
echo "  block joint size    : $PROGRESSIVE_BLOCK_JOINT_SIZE"
echo "  block joint stride  : $PROGRESSIVE_BLOCK_JOINT_STRIDE"
echo "  block joint lr      : $PROGRESSIVE_BLOCK_JOINT_LR_SCALE"
echo "  block joint KL guard: $PROGRESSIVE_BLOCK_JOINT_KL_ROLLBACK_BUDGET"
echo "  block joint fallback: $PROGRESSIVE_BLOCK_JOINT_GEOMETRY_FALLBACK"
echo "  block local losses  : $PROGRESSIVE_BLOCK_JOINT_INCLUDE_LOCAL_LOSSES"
echo "  block detach visbait: $PROGRESSIVE_BLOCK_JOINT_DETACH_VISIBLE_BAIT"
echo "  visible shift target: $VISIBLE_SHIFT_TARGET"
echo "  visible shift max   : $VISIBLE_SHIFT_MAX"
echo "  visible loss weight : $VISIBLE_LOSS_WEIGHT"
echo "  visible max weight  : $VISIBLE_MAX_WEIGHT"
echo "  detach recovery bait: $DETACH_RECOVERY_LOSS_FROM_BAIT"
echo "  detach visible bait : $DETACH_VISIBLE_LOSS_FROM_BAIT"
echo "  detach KL bait      : $DETACH_KL_LOSS_FROM_BAIT"
echo "  auto loss normalize : $AUTO_NORMALIZE_LOSS_WEIGHTS"
echo "  loss norm target    : ${LOSS_NORMALIZATION_TARGET_RATIO:-0.5}"
echo "  auto bait calibrate : $AUTO_CALIBRATE_BAIT_GATE"
echo "  bait calib target   : $BAIT_CALIBRATION_TARGET"
echo "  bait calib KL target: $BAIT_CALIBRATION_KL_TARGET"
echo "  bait calib log max  : $BAIT_CALIBRATION_LOG_MAX"
echo "  progressive KL budget: $PROGRESSIVE_KL_BUDGET"
echo "  final KL budget   : ${FINAL_KL_BUDGET:-disabled}"
echo "  progressive stages  : $PROGRESSIVE_STAGES"
echo "  total epochs        : $N_EPOCHS"
echo "  joint epochs        : $JOINT_EPOCHS"
echo "  train lr            : $TRAIN_LR"
echo "  max grad norm       : $MAX_GRAD_NORM"
echo ""

ANTI_HERETIC_RESULTS_DIR="$SUITE_RESULTS_DIR" \
"$PYTHON" "$EXPERIMENTS_DIR/step4_multilayer.py" \
    --model "$BASE_MODEL" \
    --tokenizer "$TOKENIZER" \
    --device "$DEVICE" \
    --gpu-mode "$GPU_MODE" \
    --step1-path "$STEP1_PATH" \
    --experiment-tag "$EXPERIMENT_TAG" \
    --model-save-dir "$MODEL_SAVE_DIR" \
    \
    --cover-heretic-layers \
    --heretic-target-layers "$PHASE1_LAYERS" \
    "${EXTRA_COVERAGE_FLAGS[@]}" \
    "${EXTRA_FINAL_KL_FLAGS[@]}" \
    \
    --bait-output-mode sidechannel_cancel \
    "${EXTRA_BAIT_FLAGS[@]}" \
    --refusal-cancel-min-rank 4 \
    --tag-scale "$TAG_SCALE" \
    --sidechannel-tag-trigger-mode "$SIDECHANNEL_TAG_TRIGGER_MODE" \
    --sidechannel-tag-trigger-mix "$SIDECHANNEL_TAG_TRIGGER_MIX" \
    --bait-gate-log-init "$BAIT_GATE_LOG_INIT" \
    --bait-subspace-mode mid \
    --bait-scale 1.0 \
    --bait-sv-pct-low 0.05 \
    --bait-sv-pct-high 0.25 \
    \
    --progressive-training \
    "${EXTRA_PROGRESSIVE_FLAGS[@]}" \
    --progressive-stages "$PROGRESSIVE_STAGES" \
    --progressive-kl-budget "$PROGRESSIVE_KL_BUDGET" \
    \
    --n-epochs "$N_EPOCHS" \
    --lr "$TRAIN_LR" \
    --weight-decay 0.0 \
    --batch-size "$BATCH_SIZE" \
    --max-grad-norm "$MAX_GRAD_NORM" \
    \
    --n-train "$N_TRAIN" \
    --n-good "$N_GOOD" \
    --n-bad "$N_BAD" \
    --n-train-good "$N_TRAIN_GOOD" \
    --n-train-bad "$N_TRAIN_BAD" \
    --train-good-offset "$TRAIN_GOOD_OFFSET" \
    --train-bad-offset "$TRAIN_BAD_OFFSET" \
    --eval-data-source "$EVAL_DATA_SOURCE" \
    --train-data-source "$TRAIN_DATA_SOURCE" \
    --train-good-data-source "$TRAIN_GOOD_DATA_SOURCE" \
    --train-bad-data-source "$TRAIN_BAD_DATA_SOURCE" \
    --general-data-source "$GENERAL_DATA_SOURCE" \
    \
    --recovery-loss-weight "${RECOVERY_LOSS_WEIGHT:-2.0}" \
    --kl-loss-weight "${KL_LOSS_WEIGHT:-1.0}" \
    --kl-loss-mode "${KL_LOSS_MODE:-full}" \
    --kl-top-k "${KL_TOP_K:-64}" \
    --kl-temperature "${KL_TEMPERATURE:-1.0}" \
    --kl-logit-clamp "${KL_LOGIT_CLAMP:-0.0}" \
    "${EXTRA_LOSS_NORM_FLAGS[@]}" \
    "${EXTRA_LOCAL_BAIT_GRAD_FLAGS[@]}" \
    --visible-loss-weight "$VISIBLE_LOSS_WEIGHT" \
    --residual-fisher-loss-weight "$RF_WEIGHT" \
    --residual-fisher-loss-mode "$RESIDUAL_FISHER_LOSS_MODE" \
    --residual-fisher-target-space "$RESIDUAL_FISHER_TARGET_SPACE" \
    --residual-direction-loss-weight "$RESIDUAL_DIRECTION_LOSS_WEIGHT" \
    --bait-anti-coherence-loss-weight "$BAIT_ANTI_COHERENCE_LOSS_WEIGHT" \
    --bait-anti-coherence-components "$BAIT_ANTI_COHERENCE_COMPONENTS" \
    --bait-anti-coherence-margin "$BAIT_ANTI_COHERENCE_MARGIN" \
    --residual-gap-anti-coherence-loss-weight "$RESIDUAL_GAP_ANTI_COHERENCE_LOSS_WEIGHT" \
    --residual-gap-anti-coherence-margin "$RESIDUAL_GAP_ANTI_COHERENCE_MARGIN" \
    --global-direction-loss-weight "$GLOBAL_DIRECTION_LOSS_WEIGHT" \
    --global-direction-samples "$GLOBAL_DIRECTION_SAMPLES" \
    --global-direction-margin "$GLOBAL_DIRECTION_MARGIN" \
    --global-direction-range-low "$GLOBAL_DIRECTION_RANGE_LOW" \
    --global-direction-range-high "$GLOBAL_DIRECTION_RANGE_HIGH" \
    --progressive-geometry-joint-epochs "$PROGRESSIVE_GEOMETRY_JOINT_EPOCHS" \
    --progressive-geometry-joint-lr-scale "$PROGRESSIVE_GEOMETRY_JOINT_LR_SCALE" \
    --progressive-geometry-joint-kl-rollback-budget "$PROGRESSIVE_GEOMETRY_JOINT_KL_ROLLBACK_BUDGET" \
    --progressive-geometry-joint-kl-loss-weight "$PROGRESSIVE_GEOMETRY_JOINT_KL_LOSS_WEIGHT" \
    --progressive-geometry-joint-interpolation-steps "$PROGRESSIVE_GEOMETRY_JOINT_INTERPOLATION_STEPS" \
    --progressive-block-joint-epochs "$PROGRESSIVE_BLOCK_JOINT_EPOCHS" \
    --progressive-block-joint-size "$PROGRESSIVE_BLOCK_JOINT_SIZE" \
    --progressive-block-joint-stride "$PROGRESSIVE_BLOCK_JOINT_STRIDE" \
    --progressive-block-joint-lr-scale "$PROGRESSIVE_BLOCK_JOINT_LR_SCALE" \
    --progressive-block-joint-kl-rollback-budget "$PROGRESSIVE_BLOCK_JOINT_KL_ROLLBACK_BUDGET" \
    --progressive-block-joint-interpolation-steps "$PROGRESSIVE_BLOCK_JOINT_INTERPOLATION_STEPS" \
    "${EXTRA_GEOMETRY_JOINT_FLAGS[@]}" \
    "${EXTRA_BLOCK_JOINT_FLAGS[@]}" \
    --visible-shift-target "$VISIBLE_SHIFT_TARGET" \
    --visible-shift-max "$VISIBLE_SHIFT_MAX" \
    "${EXTRA_BAIT_CALIBRATION_FLAGS[@]}" \
    --bait-calibration-target "$BAIT_CALIBRATION_TARGET" \
    --bait-calibration-quantile "$BAIT_CALIBRATION_QUANTILE" \
    --bait-calibration-log-min "$BAIT_CALIBRATION_LOG_MIN" \
    --bait-calibration-log-max "$BAIT_CALIBRATION_LOG_MAX" \
    --bait-calibration-kl-target "$BAIT_CALIBRATION_KL_TARGET" \
    --bait-calibration-kl-iters "$BAIT_CALIBRATION_KL_ITERS" \
    --recovery-init-scale 0.25 \
    \
    --recovery-max-weight 1.0 \
    --visible-max-weight "$VISIBLE_MAX_WEIGHT" \
    \
    --min-rank 2 \
    --max-rank 16 \
    --rank-energy-target 0.40 \
    \
    --seed "$SEED" \
    "$@"

echo ""
echo "Phase 1 complete:"
echo "  $MODEL_SAVE_DIR/step4_merged_${EXPERIMENT_TAG}/"
