# Reproducibility protocol

## Validation levels

1. **Static release check:** syntax checks, fixture checks, repository hygiene checks, and artifact exclusion checks.
2. **Smoke validation:** a tiny fixture-based run that validates pipeline plumbing only.
3. **Scientific reproduction:** a frozen model, local prompt files, seed, layer list, training schedule, and matched 200-trial Heretic protocol.

## Required run record

For every scientific run, retain:

- Git commit and dirty-state flag.
- Python, PyTorch, Transformers, CUDA, and GPU versions.
- Base-model identifier, local path, and revision/checksum where permitted.
- Dataset identifiers, export commands, row counts, offsets, and file hashes.
- Training seed, attack sampler seed, layer lists, loss weights, batch size, dtype, epochs, and exact/low-rank SVD initialization settings.
- Step-1 JSON, Phase-1/Phase-2 summaries, merged-checkpoint path, Heretic JSONL journals, and comparison output.

Generated checkpoints are intentionally ignored by Git. Selected aggregate summaries and redacted Heretic journals may be included under `release_artifacts/`; model redistribution remains subject to each base-model license.

## Public reproduction entry point

Use the consolidated runner instead of editing scripts by hand:

```bash
bash experiments/sidechannel_suite/run_paper_table3.sh <model_key>
```

Recommended public model keys and expected Heretic minimum-refusal rates at KL <= 0.05 / 0.10 / 0.20 / 1.0 are:

| model key | model | clean/base | Bait-and-Recover |
| --- | --- | --- | --- |
| `qwen3_8b` | Qwen3-8B | 36 / 11 / 11 / 8 | 79 / 60 / 44 / 43 |
| `gemma3_12b` | Gemma-3-12B-it | 12 / 3 / 2 / 0 | 83 / 83 / 83 / 79 |

The model-specific constants live in `experiments/sidechannel_suite/scripts/paper_table3_configs.sh`. All settings accept environment-variable overrides; for scientific reproduction, record overrides in the run manifest rather than editing the frozen config.

## Matched attack comparison

Clean and defended Heretic runs must use the same prompt files, estimator, editable modules, direction scope, sampler seed, trial count, batch size, and KL thresholds. High defended refusal is meaningful only when the corresponding clean attack is effective under the same protocol.
