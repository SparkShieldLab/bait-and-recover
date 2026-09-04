#!/usr/bin/env bash
# Reproduce the public Bait-and-Recover configurations.
#
# Usage:
#   bash experiments/sidechannel_suite/run_paper_table3.sh qwen3_8b
#   GPU_ID=3 RUN_BASELINE_HERETIC=true bash experiments/sidechannel_suite/run_paper_table3.sh gemma3_12b
#
# Model keys:
#   qwen3_8b, gemma3_12b
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_KEY="${1:-}"

if [[ -z "$MODEL_KEY" || "$MODEL_KEY" == "-h" || "$MODEL_KEY" == "--help" ]]; then
    cat <<'EOF'
Usage:
  bash experiments/sidechannel_suite/run_paper_table3.sh <model_key>

Model keys:
  qwen3_8b      Qwen3-8B public recipe
  gemma3_12b    Gemma-3-12B-it public recipe

Examples:
  GPU_ID=0 bash experiments/sidechannel_suite/run_paper_table3.sh qwen3_8b
  PRINT_CONFIG_ONLY=true bash experiments/sidechannel_suite/run_paper_table3.sh gemma3_12b
  GPU_ID=3 RUN_BASELINE_HERETIC=true bash experiments/sidechannel_suite/run_paper_table3.sh gemma3_12b
EOF
    exit 0
fi

shift || true

# shellcheck source=scripts/paper_table3_configs.sh
source "$SCRIPT_DIR/scripts/paper_table3_configs.sh"
anti_heretic_apply_paper_table3_config "$MODEL_KEY" "$SCRIPT_DIR"

cat <<EOF
[paper-table3] model=${PAPER_TABLE3_MODEL_NAME}
[paper-table3] expected baseline min-refusal @ KL<=0.05,0.10,0.20,1.0: ${PAPER_TABLE3_EXPECTED_BASE}
[paper-table3] expected defended min-refusal @ KL<=0.05,0.10,0.20,1.0: ${PAPER_TABLE3_EXPECTED_DEFENDED}
[paper-table3] verify tag: ${VERIFY_TAG}
[paper-table3] defended checkpoint tag: ${DEFENDED_CHECKPOINT_TAG}
[paper-table3] base model: ${BASE_MODEL}
[paper-table3] HF cache: ${HF_HOME_DIR:-${HF_HOME:-$HOME/.cache/huggingface}} (Heretic offline: hub=${HERETIC_HF_HUB_OFFLINE:-1}, datasets=${HERETIC_HF_DATASETS_OFFLINE:-1})
[paper-table3] GPU_ID=${GPU_ID}, N_TRIALS=${N_TRIALS}, RUN_TRAIN=${RUN_TRAIN}, RUN_HERETIC=${RUN_HERETIC}, RUN_BASELINE_HERETIC=${RUN_BASELINE_HERETIC}
EOF

if [[ "${PRINT_CONFIG_ONLY:-false}" == "true" ]]; then
    exit 0
fi

bash "$SCRIPT_DIR/scripts/run_full_pipeline_modelscope_alpaca_safemt.sh" "$@"
