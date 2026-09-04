# Code provenance

This repository is the cleaned public research-code release for **Bait-and-Recover: Poisoning Internal Refusal Signals to Defend LLMs against White-Box Editing Jailbreaks**, accepted at AACL-IJCNLP 2026 Main.

The public tree contains:

- Bait-and-Recover training and merging code;
- reproducible runners for the two highlighted public recipes, Qwen3-8B and Gemma-3-12B-it;
- selected structured validation artifacts;
- a compatibility patch against upstream Heretic v1.2.0.

## Heretic integration

This repository does not vendor the full Heretic attack source tree. Reproduction uses the upstream Heretic project plus the patch stored under `patches/heretic/`.

```text
repository: https://github.com/p-e-w/heretic
tag:        v1.2.0
commit:     27097bfe8e60d5e65f0200785f1e96a52afcbc2a
license:    AGPL-3.0-or-later
```

The patch adds evaluation controls needed for the reported protocol, including local prompt loading, deterministic search settings, direction-scope controls, residual-geometry logging, and resume/skip-existing support for interrupted 200-trial searches.

## Exclusions

The Git repository does not include model weights, generated checkpoints, local caches, credentials, third-party dataset exports, or generated harmful model outputs.
