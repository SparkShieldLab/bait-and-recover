#!/usr/bin/env bash
# Paper/release reproduction configurations for Bait-and-Recover.
#
# This file intentionally centralizes the model/layer/tag/hyperparameter knobs
# that were used for the paper-aligned Heretic evaluations. The model-specific
# runners are kept thin so future users do not need to diff several historical
# temporary scripts to understand what was run.

anti_heretic_export_default() {
    local name="$1"
    local value="$2"
    if [[ -z "${!name:-}" ]]; then
        export "$name=$value"
    else
        export "$name"
    fi
}

anti_heretic_resolve_cached_model() {
    local namespace="$1"
    local model_name="$2"
    local fallback="${3:-$namespace/$model_name}"
    local cache_dir="${MODELSCOPE_CACHE_DIR:-$HOME/.cache/modelscope}"
    local candidates=(
        "$cache_dir/$namespace/$model_name"
        "$cache_dir/models/$namespace/$model_name"
        "$cache_dir/hub/$namespace/$model_name"
    )
    for candidate in "${candidates[@]}"; do
        if [[ -d "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return
        fi
    done
    printf '%s\n' "$fallback"
}

anti_heretic_resolve_hf_cache() {
    if [[ -n "${HERETIC_HF_HOME:-}" ]]; then
        printf '%s\n' "$HERETIC_HF_HOME"
    elif [[ -n "${HF_HOME:-}" ]]; then
        printf '%s\n' "$HF_HOME"
    else
        printf '%s\n' "$HOME/.cache/huggingface"
    fi
}

anti_heretic_apply_paper_common() {
    local suite_dir="$1"

    anti_heretic_export_default MODEL_SCOPE_PROMPT_DIR "$suite_dir/data/anti_heretic_prompts/modelscope_alpaca_safemt"
    anti_heretic_export_default HF_HOME_DIR "$(anti_heretic_resolve_hf_cache)"
    anti_heretic_export_default HERETIC_HF_HUB_OFFLINE 1
    anti_heretic_export_default HERETIC_TRANSFORMERS_OFFLINE 1
    anti_heretic_export_default HERETIC_HF_DATASETS_OFFLINE 1
    anti_heretic_export_default EVAL_DATA_SOURCE "local:$MODEL_SCOPE_PROMPT_DIR"
    anti_heretic_export_default TRAIN_DATA_SOURCE "local:$MODEL_SCOPE_PROMPT_DIR"
    anti_heretic_export_default TRAIN_GOOD_DATA_SOURCE "local:$MODEL_SCOPE_PROMPT_DIR"
    anti_heretic_export_default TRAIN_BAD_DATA_SOURCE "local:$MODEL_SCOPE_PROMPT_DIR"
    anti_heretic_export_default GENERAL_DATA_SOURCE "local:$MODEL_SCOPE_PROMPT_DIR"

    anti_heretic_export_default N_GOOD 64
    anti_heretic_export_default N_BAD 32
    anti_heretic_export_default N_TRAIN 256
    anti_heretic_export_default N_TRAIN_GOOD 128
    anti_heretic_export_default N_TRAIN_BAD 64
    anti_heretic_export_default TRAIN_GOOD_OFFSET 128
    anti_heretic_export_default TRAIN_BAD_OFFSET 32

    anti_heretic_export_default ANTI_HERETIC_MODEL_DTYPE bf16
    anti_heretic_export_default ANTI_HERETIC_TRUST_REMOTE_CODE 1
    anti_heretic_export_default PYTORCH_CUDA_ALLOC_CONF expandable_segments:True

    anti_heretic_export_default STYLE oldstyle
    anti_heretic_export_default RF_WEIGHT 2.0
    anti_heretic_export_default KL_LOSS_WEIGHT 0.0
    anti_heretic_export_default KL_LOSS_MODE topk_last
    anti_heretic_export_default KL_TOP_K 64
    anti_heretic_export_default KL_TEMPERATURE 1.0
    anti_heretic_export_default KL_LOGIT_CLAMP 30.0
    anti_heretic_export_default SKIP_JOINT true

    anti_heretic_export_default BAIT_FREEZE_MODE coeff
    anti_heretic_export_default SIDECHANNEL_TAG_TRIGGER_MODE orthogonal_svd
    anti_heretic_export_default BAIT_GATE_LOG_INIT -1.0
    anti_heretic_export_default VISIBLE_SHIFT_TARGET 0.20
    anti_heretic_export_default VISIBLE_SHIFT_MAX 0.40
    anti_heretic_export_default VISIBLE_LOSS_WEIGHT 4.0
    anti_heretic_export_default VISIBLE_MAX_WEIGHT 3.0
    anti_heretic_export_default DETACH_RECOVERY_LOSS_FROM_BAIT true
    anti_heretic_export_default DETACH_VISIBLE_LOSS_FROM_BAIT false
    anti_heretic_export_default DETACH_KL_LOSS_FROM_BAIT true
    anti_heretic_export_default AUTO_CALIBRATE_BAIT_GATE true
    anti_heretic_export_default BAIT_CALIBRATION_TARGET 0.20
    anti_heretic_export_default BAIT_CALIBRATION_LOG_MAX 1.0
    anti_heretic_export_default BAIT_CALIBRATION_KL_TARGET 0.0
    anti_heretic_export_default AUTO_NORMALIZE_LOSS_WEIGHTS false
    anti_heretic_export_default LOSS_NORMALIZATION_TARGET_RATIO 3.0

    anti_heretic_export_default RECOVERY_LOSS_WEIGHT 0.5
    anti_heretic_export_default RESIDUAL_FISHER_LOSS_MODE true_ratio
    anti_heretic_export_default RESIDUAL_FISHER_TARGET_SPACE full
    anti_heretic_export_default RESIDUAL_DIRECTION_LOSS_WEIGHT 2.0
    anti_heretic_export_default BAIT_ANTI_COHERENCE_LOSS_WEIGHT 3.0
    anti_heretic_export_default BAIT_ANTI_COHERENCE_COMPONENTS 2
    anti_heretic_export_default BAIT_ANTI_COHERENCE_MARGIN 0.15
    anti_heretic_export_default RESIDUAL_GAP_ANTI_COHERENCE_LOSS_WEIGHT 1.0
    anti_heretic_export_default RESIDUAL_GAP_ANTI_COHERENCE_MARGIN 0.15
    anti_heretic_export_default GLOBAL_DIRECTION_LOSS_WEIGHT 2.0
    anti_heretic_export_default GLOBAL_DIRECTION_SAMPLES 7
    anti_heretic_export_default GLOBAL_DIRECTION_MARGIN 0.10
    anti_heretic_export_default GLOBAL_DIRECTION_RANGE_LOW 0.40
    anti_heretic_export_default GLOBAL_DIRECTION_RANGE_HIGH 0.90

    anti_heretic_export_default PROGRESSIVE_GEOMETRY_JOINT_EPOCHS 0
    anti_heretic_export_default PROGRESSIVE_GEOMETRY_JOINT_LR_SCALE 0.05
    anti_heretic_export_default PROGRESSIVE_GEOMETRY_JOINT_INCLUDE_LOCAL_LOSSES false
    anti_heretic_export_default PROGRESSIVE_GEOMETRY_JOINT_TRAIN_BAIT false
    anti_heretic_export_default PROGRESSIVE_GEOMETRY_JOINT_KL_ROLLBACK_BUDGET 0.10
    anti_heretic_export_default PROGRESSIVE_GEOMETRY_JOINT_KL_LOSS_WEIGHT 0.05
    anti_heretic_export_default PROGRESSIVE_GEOMETRY_JOINT_INTERPOLATION_STEPS 8
    anti_heretic_export_default PROGRESSIVE_BLOCK_JOINT_EPOCHS 1
    anti_heretic_export_default PROGRESSIVE_BLOCK_JOINT_SIZE 2
    anti_heretic_export_default PROGRESSIVE_BLOCK_JOINT_STRIDE 2
    anti_heretic_export_default PROGRESSIVE_BLOCK_JOINT_LR_SCALE 0.05
    anti_heretic_export_default PROGRESSIVE_BLOCK_JOINT_KL_ROLLBACK_BUDGET 0.10
    anti_heretic_export_default PROGRESSIVE_BLOCK_JOINT_INTERPOLATION_STEPS 8
    anti_heretic_export_default PROGRESSIVE_BLOCK_JOINT_GEOMETRY_FALLBACK true
    anti_heretic_export_default PROGRESSIVE_BLOCK_JOINT_DETACH_VISIBLE_BAIT true
    anti_heretic_export_default PROGRESSIVE_BLOCK_JOINT_INCLUDE_LOCAL_LOSSES true
    anti_heretic_export_default MAX_GRAD_NORM 1.0
    anti_heretic_export_default PROGRESSIVE_KL_BUDGET 0.05
    anti_heretic_export_default FINAL_KL_BUDGET 0.20

    anti_heretic_export_default RUN_STEP1 auto
    anti_heretic_export_default RUN_TRAIN true
    anti_heretic_export_default RUN_HERETIC true
    anti_heretic_export_default RUN_BASELINE_HERETIC false
    anti_heretic_export_default RUN_COMPARE true
    anti_heretic_export_default HERETIC_RUNNER exact
    anti_heretic_export_default N_TRIALS 200
    anti_heretic_export_default HERETIC_DIRECTION_SCOPE both
    anti_heretic_export_default HERETIC_SAMPLER_SEED_BASE 42
}

anti_heretic_apply_paper_table3_config() {
    local model_key="$1"
    local suite_dir="$2"

    # Apply model-specific defaults before the shared preset. User-provided
    # environment variables still take precedence over all defaults.
    case "$model_key" in
        qwen3_8b|qwen3-8b)
            anti_heretic_export_default PAPER_TABLE3_MODEL_NAME "Qwen3-8B"
            anti_heretic_export_default PAPER_TABLE3_EXPECTED_BASE "36% / 11% / 11% / 8%"
            anti_heretic_export_default PAPER_TABLE3_EXPECTED_DEFENDED "79% / 60% / 44% / 43%"
            anti_heretic_export_default GPU_ID 2
            anti_heretic_export_default MODEL_SLUG "Qwen_Qwen3-8B"
            anti_heretic_export_default BASE_MODEL "$(anti_heretic_resolve_cached_model Qwen Qwen3-8B Qwen/Qwen3-8B)"
            anti_heretic_export_default VERIFY_TAG "qwen3_8b_allnewdata_bf16_orthsvd_cov21_blockjoint2_nanhook_coeffbait_strong_anticoh3_gap1m15_gdir2_vis20_calib20_vw4_rf2_dir2_kl0_topkkl_detrec_visbait"
            anti_heretic_export_default HERETIC_COVERAGE_LAYERS 21
            anti_heretic_export_default BATCH_SIZE 16
            anti_heretic_export_default HERETIC_BATCH_SIZE 256
            anti_heretic_export_default TRAIN_LR 2e-4
            ;;
        gemma3_12b|gemma3-12b)
            anti_heretic_export_default PAPER_TABLE3_MODEL_NAME "Gemma-3-12B-it"
            anti_heretic_export_default PAPER_TABLE3_EXPECTED_BASE "12% / 3% / 2% / 0%"
            anti_heretic_export_default PAPER_TABLE3_EXPECTED_DEFENDED "83% / 83% / 83% / 79%"
            anti_heretic_export_default GPU_ID 2
            anti_heretic_export_default MODEL_SLUG "google_gemma-3-12b-it"
            anti_heretic_export_default BASE_MODEL "$(anti_heretic_resolve_cached_model google gemma-3-12b-it google/gemma-3-12b-it)"
            anti_heretic_export_default VERIFY_TAG "gemma3_12b_bf16_orthsvd_prior29_leftwide_from18_blockjoint2_nanhook_coeffbait_strong_anticoh3_gap1m15_gdir2_vis20_calib20_vw4_rf2_dir2_kl0_topkkl_lr1e-4_detrec_visbait"
            anti_heretic_export_default HERETIC_COVERAGE_LAYERS 29
            anti_heretic_export_default PHASE1_LAYERS "18,20,22,24,26,28,30,32,34,36,38"
            anti_heretic_export_default PHASE2_LAYERS "19,21,23,25,27,29,31,33,35,37"
            anti_heretic_export_default ANTI_HERETIC_ALLOW_IMAGE_TEXT_TO_TEXT 1
            anti_heretic_export_default BATCH_SIZE 2
            anti_heretic_export_default HERETIC_BATCH_SIZE 32
            anti_heretic_export_default TRAIN_LR 1e-4
            ;;
        *)
            echo "[ERROR] Unknown public model key: $model_key" >&2
            echo "        Choose one of: qwen3_8b, gemma3_12b" >&2
            return 2
            ;;
    esac

    anti_heretic_apply_paper_common "$suite_dir"

    anti_heretic_export_default TOKENIZER "$BASE_MODEL"
    anti_heretic_export_default ORIGINAL_CLEAN_MODEL "$BASE_MODEL"
    anti_heretic_export_default ORIGINAL_CLEAN_TOKENIZER "$TOKENIZER"

    local model_slug_safe="$MODEL_SLUG"
    anti_heretic_export_default PHASE1_TAG "phase1_${VERIFY_TAG}_${model_slug_safe}"
    anti_heretic_export_default PHASE2_TAG "phase2_${VERIFY_TAG}_${model_slug_safe}"
    anti_heretic_export_default DEFENDED_CHECKPOINT_TAG "defended_${VERIFY_TAG}_${model_slug_safe}"
}
