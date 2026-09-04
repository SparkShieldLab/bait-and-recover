# Contributing

Keep pull requests small and include the exact command used for validation. Run:

```bash
python scripts/check_release.py
```

Changes to training or attack behavior should include a matched clean/defended test plan, seed information, and a note on expected artifact changes. Do not commit model weights, generated checkpoints, credentials, cached datasets, generated harmful outputs, or unredacted local run logs.

By contributing, you certify that you have the right to submit the code under the repository's AGPL-3.0-or-later license.
