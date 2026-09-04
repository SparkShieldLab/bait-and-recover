# Heretic compatibility patch

This release does not vendor the full Heretic attack implementation. Instead, it references the upstream Heretic project and provides the minimal patch used to reproduce the Bait-and-Recover evaluation protocol.

Upstream reference:

- Repository: https://github.com/p-e-w/heretic
- Base tag: `v1.2.0`
- Base commit: `27097bfe8e60d5e65f0200785f1e96a52afcbc2a`

Patch:

- `bait-and-recover-heretic-v1.2.0.patch`

The patch contains compatibility/evaluation-runner changes needed by our experiments, including local/offline prompt files, deterministic search controls, direction-scope controls, stable long-path journal names, non-interactive execution, residual-geometry logging, and resume/skip-existing support for interrupted 200-trial runs.

Use `scripts/setup_heretic.sh` from the repository root to clone upstream Heretic into `external/heretic`, apply the patch, and install it in editable mode.
