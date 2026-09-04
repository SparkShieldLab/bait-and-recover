# Bait-and-Recover experiment suite

This directory contains the training and Heretic evaluation wrappers used by the public release. Invoke them from the repository root or from this directory; generated outputs remain repository-relative unless overridden through environment variables.

## Public reproduction recipes

The recommended interface is the consolidated runner:

```bash
bash experiments/sidechannel_suite/run_paper_table3.sh <model_key>
```

Available public keys are:

- `qwen3_8b`
- `gemma3_12b`

Thin convenience wrappers call the same runner:

```bash
bash experiments/sidechannel_suite/run_paper_table3_qwen3_8b.sh
bash experiments/sidechannel_suite/run_paper_table3_gemma3_12b.sh
```

The frozen tags, layer lists, loss weights, and expected refusal-rate rows are centralized in `scripts/paper_table3_configs.sh`.

## Outputs

- Step-1 summaries and training results: `results/`
- Merged models: `results/models/`
- Logs: `logs/`
- Heretic study journals: `heretic_checkpoints/`

All generated output directories are ignored by Git. For external storage, set `SUITE_RESULTS_DIR`, `MODEL_SAVE_DIR`, `LOG_DIR`, and `CHECKPOINT_BASE` explicitly.

## Required inputs

Set `BASE_MODEL`, `TOKENIZER`, `ORIGINAL_CLEAN_MODEL`, and `ORIGINAL_CLEAN_TOKENIZER` to local paths, or make the corresponding identifiers available in the configured cache. The suite is offline-first.

The ModelScope wrappers require local `good.jsonl` and `bad.jsonl` prompt exports. Prepare them with `scripts/prepare_data.sh` from the repository root after reviewing the dataset terms. Synthetic test fixtures are used only by `scripts/smoke_train_and_attack.sh`.

See the top-level `README.md` and `docs/REPRODUCIBILITY.md` before launching a scientific run.
