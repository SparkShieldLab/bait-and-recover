#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
OUT_DIR="${MODEL_SCOPE_PROMPT_DIR:-$ROOT/experiments/sidechannel_suite/data/anti_heretic_prompts/modelscope_alpaca_safemt}"

cat <<'EOF'
This command downloads/exports third-party datasets. Review and accept the current
dataset cards and terms before continuing. The exported JSONL files are ignored by
Git and must not be redistributed without permission.
EOF

if [[ "${ACCEPT_DATASET_TERMS:-false}" != "true" ]]; then
    echo "Set ACCEPT_DATASET_TERMS=true after reviewing the dataset terms."
    exit 2
fi

"$PYTHON" "$ROOT/experiments/sidechannel_suite/scripts/export_modelscope_prompts.py" \
    --out-dir "$OUT_DIR" \
    --good-dataset "${GOOD_DATASET:-AI-ModelScope/alpaca-gpt4-data-en}" \
    --bad-dataset "${BAD_DATASET:-Shanghai_AI_Laboratory/SafeMTData}" \
    --bad-subset "${BAD_SUBSET:-Attack_600}" \
    --bad-split "${BAD_SPLIT:-Attack_600}"

echo "Prepared local prompts in $OUT_DIR"
