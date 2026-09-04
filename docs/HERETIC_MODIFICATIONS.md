# Heretic integration policy

This release avoids vendoring the full Heretic attack implementation. To reproduce the evaluation, users should obtain upstream Heretic v1.2.0 and apply our compatibility patch.

```bash
bash scripts/setup_heretic.sh
```

The patch is located at `patches/heretic/bait-and-recover-heretic-v1.2.0.patch`. It is provided for reproducibility of the reported evaluation protocol, not as a standalone attack distribution.

Main patch categories:

- deterministic 200-trial search controls;
- explicit global/per-layer/mixed direction-scope handling;
- local/offline prompt-file support;
- non-interactive execution after optimization;
- per-trial and residual-geometry logging;
- stable journal names for long local model paths;
- resume/skip-existing support for interrupted journals.
