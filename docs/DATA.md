# Data sources

The training and evaluation runners expect local JSONL prompt files for benign/general prompts and harmful prompts. A local prompt directory should contain `good.jsonl` and `bad.jsonl`; each line must be a JSON object with one of the supported prompt keys such as `prompt`, `text`, `instruction`, or `query`.

The paper experiments used benign/general prompts derived from `AI-ModelScope/alpaca-gpt4-data-en` and harmful prompts derived from the `Attack_600` subset of `Shanghai_AI_Laboratory/SafeMTData`. These dataset exports are not redistributed in this repository. Users should review the corresponding dataset cards and licenses before preparing local copies.

Use the helper below to prepare local prompt files after confirming that your intended use complies with the dataset terms:

```bash
bash scripts/prepare_data.sh
```

The files under `tests/fixtures/prompts/` are tiny synthetic fixtures for plumbing tests only. They are not suitable for scientific evaluation, and smoke-test numbers must not be reported as paper reproduction results.

For a scientific run, record dataset identifiers, revisions when available, export commands, row counts, prompt offsets, and file hashes.
