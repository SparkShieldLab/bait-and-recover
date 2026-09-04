"""
Step 3 (Multi-layer): Cross-layer LoRA bait/recovery defense — full Heretic coverage
=====================================================================================

Upgrade over single-layer step3: covers ALL layers that Heretic traverses,
not just a single hand-picked (bait, recovery) pair.

Architecture (per bait layer l):
  Layer l:   down_proj [+bait δ_l] → residual_{l+1}   ← Heretic sees poisoned direction HERE
  Layer l+1: o_proj + down_proj [+recovery -δ_l]      ← direction restored at residual_{l+2}

New capabilities vs. step3_lora_defense_train.py
-------------------------------------------------
1. Multi-layer coverage (--cover-heretic-layers)
   --heretic-layer-range {wide|mid|all}  selects which Heretic layers to cover:
     'mid'  — Heretic's empirically strongest band: layers [0.4·L, 0.9·L] (default)
     'wide' — all layers except the very last (ensures no uncovered attack surface)
     'all'  — synonym for 'wide'
   --heretic-target-layers 5,8,12,...    explicit comma-separated override

2. Progressive multi-layer training (--progressive-training)
   Trains bait/recovery pairs from shallow to deep, freezing each pair before
   moving to the next. This prevents KL cumulative blowup and helps isolate
   layer-to-layer interference (TODO-1.4).

3. Per-layer KL and interference diagnostics
   After training, measure KL contribution of each bait/recovery pair and
   the cosine of each bait residual direction seen by downstream same-layer recovery.

Training uses unlabeled general text and minimizes KL divergence to a frozen
reference forward path. Good/bad prompt splits are used only during evaluation.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from rich.table import Table
from torch.optim import AdamW

BAIT_GATE_LOG_MAX = 8.0
BAIT_GATE_LOG_INIT = -2.0

from shared_utils import (
    DATA_SOURCE_HELP,
    RESULTS_DIR,
    get_layers,
    get_layer_module,
    set_layer_module,
    load_model,
    load_prompts_simple,
    load_prompts_split_sources,
    print,
    Prompt,
    resolve_device,
    tokenize_prompts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bait-and-Recover multi-layer defense training",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-1b-it",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cpu, cuda, npu, mps",
    )
    parser.add_argument(
        "--gpu-mode",
        type=str,
        default="single",
        choices=["single", "auto"],
        help=(
            "CUDA model placement. 'single' loads the full model on --device; "
            "'auto' uses HuggingFace device_map='auto' across visible CUDA GPUs or Ascend NPUs."
        ),
    )
    parser.add_argument(
        "--step1-path",
        type=Path,
        default=RESULTS_DIR / "step1_results_google_gemma-3-1b-it.json",
        help="Step 1 SVD analysis JSON used as the design prior",
    )
    parser.add_argument(
        "--target-layers",
        type=str,
        default="",
        help="Comma-separated transformer layer indices. Empty = auto-select from step1",
    )
    parser.add_argument(
        "--num-target-layers",
        type=int,
        default=1,
        help="When auto-selecting, use the top-N step1 layers by baseline Fisher ratio",
    )
    parser.add_argument(
        "--cover-heretic-layers",
        action="store_true",
        default=False,
        help=(
            "Automatically select ALL layers that Heretic traverses, overriding "
            "--num-target-layers and --target-layers. The specific set is determined "
            "by --heretic-layer-range."
        ),
    )
    parser.add_argument(
        "--heretic-layer-range",
        type=str,
        default="mid",
        choices=["mid", "wide", "all"],
        help=(
            "Which Heretic layers to cover when --cover-heretic-layers is set. "
            "'mid'  = layers in [0.4·L, 0.9·L] — Heretic's empirically strongest band. "
            "'wide' / 'all' = ALL layers (0 to L-2, leaving room for l+1 recovery)."
        ),
    )
    parser.add_argument(
        "--heretic-target-layers",
        type=str,
        default="",
        help=(
            "Explicit comma-separated layer indices to cover (when --cover-heretic-layers "
            "is set but you want manual control). Overrides --heretic-layer-range."
        ),
    )
    parser.add_argument(
        "--heretic-coverage-layers",
        type=int,
        default=0,
        help=(
            "Fixed coverage-window size encoded by tags such as cov12. When "
            "--heretic-target-layers is provided this is recorded for auditability; "
            "otherwise --cover-heretic-layers selects the deepest N bait-capable "
            "layers from the Step1 summary."
        ),
    )
    parser.add_argument(
        "--min-fisher-for-coverage",
        type=float,
        default=0.0,
        help=(
            "When --cover-heretic-layers is set, skip any layer whose baseline Fisher "
            "discriminability (from step1_summary) is below this value. "
            "Shallow layers often have low Fisher (< 0.85) and contribute near-zero DPS "
            "while still adding KL cost. Set to e.g. 0.85 to auto-prune them. Default: 0 (no filter)."
        ),
    )
    parser.add_argument(
        "--progressive-training",
        action="store_true",
        default=False,
        help=(
            "Train bait/recovery pairs in shallow-to-deep order, freezing each pair "
            "before moving to the next. Prevents KL accumulation and isolates inter-layer "
            "interference. Recommended for --cover-heretic-layers with many layers."
        ),
    )
    parser.add_argument(
        "--progressive-kl-budget",
        type=float,
        default=0.005,
        help=(
            "Per-stage KL reporting budget when --progressive-training is used. "
            "The stage KL is logged against this threshold; training does not early-stop."
        ),
    )
    parser.add_argument(
        "--final-kl-budget",
        type=float,
        default=0.0,
        help=(
            "Optional final cumulative KL budget. When > 0, abort before saving "
            "the checkpoint if the trained defense exceeds this KL(all) budget."
        ),
    )
    parser.add_argument(
        "--progressive-stages",
        type=int,
        default=3,
        help=(
            "Number of training epochs per progressive stage (shallow-to-deep iteration). "
            "Total history length is roughly progressive_stages × number_of_pairs plus "
            "the remaining joint fine-tuning epochs from --n-epochs."
        ),
    )
    parser.add_argument(
        "--progressive-cumulative-stages",
        action="store_true",
        default=False,
        help=(
            "During progressive training, keep earlier shallow pairs enabled but frozen "
            "while training the current pair. This trains deeper pairs on top of the "
            "already-installed defense instead of in isolation."
        ),
    )
    parser.add_argument(
        "--skip-progressive-joint-finetune",
        action="store_true",
        default=False,
        help=(
            "Skip the final all-pairs joint fine-tuning pass. Useful for debugging true "
            "sequential bait/recovery behavior, because joint training can trade away "
            "per-pair recovery to improve global Fisher suppression."
        ),
    )
    parser.add_argument(
        "--rank-energy-target",
        type=float,
        default=0.40,
        help="Choose the smallest LoRA rank capturing this much energy within step1 top-50 singular values",
    )
    parser.add_argument(
        "--min-rank",
        type=int,
        default=4,
        help="Lower bound on per-layer LoRA rank",
    )
    parser.add_argument(
        "--max-rank",
        type=int,
        default=16,
        help="Upper bound on per-layer LoRA rank",
    )
    parser.add_argument(
        "--bait-scale",
        type=float,
        default=1.0,
        help="Base scaling multiplier for bait adapters (effective scale = bait_scale * sqrt(spectrum_energy))",
    )
    parser.add_argument(
        "--svd-method",
        choices=["full", "lowrank"],
        default="full",
        help=(
            "SVD used to initialize adapters. full is the paper-compatible exact "
            "CPU path; lowrank is an explicit large-model acceleration that "
            "approximates only the leading singular subspace."
        ),
    )
    parser.add_argument(
        "--svd-lowrank-q",
        type=int,
        default=32,
        help="Number of leading factors retained by --svd-method=lowrank.",
    )
    parser.add_argument(
        "--svd-lowrank-niter",
        type=int,
        default=4,
        help="Power iterations used by --svd-method=lowrank.",
    )
    parser.add_argument(
        "--bait-subspace-mode",
        type=str,
        default="mid",
        choices=["strongest", "weakest", "mid", "sweep"],
        help=(
            "Which spectral band of down_proj left SVs to use for bait output basis. "
            "'strongest' = top-rank (high energy, high behavioral impact); "
            "'weakest' = tail-rank (low energy, easily washed out in stats); "
            "'mid' = percentile band (low-sensitive but still statistically viable); "
            "'sweep' = run diagnostic across all bands, no training"
        ),
    )
    parser.add_argument(
        "--bait-output-mode",
        type=str,
        default="svd",
        choices=["svd", "refusal_cancel", "random_orthogonal", "sidechannel_cancel"],
        help=(
            "How to choose the bait OUTPUT direction in residual-stream space. "
            "'svd' = use spectral band of down_proj left SVs (default). "
            "'refusal_cancel' = initialize output basis along the NEGATIVE supervised refusal "
            "direction at residual l+1, directly poisoning what Heretic measures at that position. "
            "Increases direction-poisoning score (Goal 2) at the cost of needing slightly higher bait_scale. "
            "'random_orthogonal' = initialize an unsupervised random output basis orthogonal to "
            "the strongest down_proj residual directions, avoiding same-source refusal vectors. "
            "'sidechannel_cancel' = like refusal_cancel but adds an explicit orthogonal tag "
            "vector whose activation is class-dependent. Recovery LoRA can read this tag to "
            "distinguish good/bad prompts even when Fisher along r_clean is fully suppressed. "
            "Use with --residual-fisher-loss-weight > 0 for best effect."
        ),
    )
    parser.add_argument(
        "--bait-sv-pct-low",
        type=float,
        default=0.05,
        help="Lower percentile bound for 'mid' subspace mode (0.0 = strongest, 1.0 = weakest)",
    )
    parser.add_argument(
        "--bait-sv-pct-high",
        type=float,
        default=0.25,
        help="Upper percentile bound for 'mid' subspace mode",
    )
    parser.add_argument(
        "--bait-basis-trainable",
        action="store_true",
        default=False,
        help="Allow bait output basis (B) to be semi-trainable with projection regularization",
    )
    parser.add_argument(
        "--bait-basis-reg-weight",
        type=float,
        default=0.1,
        help="Regularization weight penalizing B drifting outside the initial subspace",
    )
    parser.add_argument(
        "--freeze-bait-coeff",
        action="store_true",
        default=False,
        help=(
            "Freeze bait input-projection (coeff/A) weights so the fake direction "
            "cannot be corrected by the optimizer. Recommended with refusal_cancel mode. "
            "Use --freeze-bait and --freeze-bait-gate to train only recovery adapters."
        ),
    )
    parser.add_argument(
        "--fisher-poison-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for the Fisher-poison loss term, which directly penalizes the bait "
            "delta aligning with the clean refusal direction. Requires supervised "
            "good/bad prompts and refusal_cancel mode. 0 = disabled."
        ),
    )
    parser.add_argument(
        "--residual-fisher-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for residual-level Fisher suppression loss. Unlike "
            "--fisher-poison-loss-weight (which makes the bait delta class-blind, "
            "preserving residual Fisher), this directly minimizes the Fisher ratio "
            "of the defended residual at the bait position, forcing the bait to "
            "actively reduce good/bad separability. 0 = disabled."
        ),
    )
    parser.add_argument(
        "--residual-fisher-denom-floor",
        type=float,
        default=1e-3,
        help=(
            "Minimum detached within-class variance denominator for residual Fisher "
            "suppression. Larger values make the loss less sharp when class variance "
            "is tiny and improve numerical stability."
        ),
    )
    parser.add_argument(
        "--residual-fisher-loss-mode",
        type=str,
        default="detached_gap",
        choices=["detached_gap", "true_ratio"],
        help=(
            "Gradient form for residual Fisher suppression. detached_gap uses a "
            "detached variance denominator for stability. true_ratio backpropagates "
            "through the full Fisher ratio and reproduces the older, more aggressive "
            "training behavior."
        ),
    )
    parser.add_argument(
        "--residual-fisher-target-space",
        type=str,
        default="projected",
        choices=["projected", "full"],
        help=(
            "Feature space used by residual Fisher suppression. projected measures "
            "Fisher only along the supervised residual refusal direction. full uses "
            "the whole residual vector and reproduces the older, more geometry-poisoning "
            "behavior."
        ),
    )
    parser.add_argument(
        "--residual-direction-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for directly decorrelating the defended good/bad residual gap "
            "from the clean supervised refusal direction at bait residuals. This "
            "targets the direction Heretic estimates, complementing Fisher-scale "
            "suppression."
        ),
    )
    parser.add_argument(
        "--bait-anti-coherence-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for cross-layer bait output-basis anti-coherence. This "
            "penalizes bait basis directions that align across different layers, "
            "so per-layer bait signals cannot collapse into one global direction "
            "that Heretic can aggregate. 0 = disabled."
        ),
    )
    parser.add_argument(
        "--bait-anti-coherence-components",
        type=int,
        default=2,
        help=(
            "Number of leading bait output-basis columns per layer included in "
            "--bait-anti-coherence-loss-weight. In sidechannel_cancel mode, 2 "
            "covers the refusal-cancel and tag channels."
        ),
    )
    parser.add_argument(
        "--bait-anti-coherence-margin",
        type=float,
        default=0.0,
        help=(
            "Allowed absolute cross-layer cosine before anti-coherence activates. "
            "0 forces pairwise orthogonality; e.g. 0.2 only penalizes |cos| > 0.2."
        ),
    )
    parser.add_argument(
        "--external-bait-direction-ref-path",
        type=Path,
        default=None,
        help=(
            "Optional JSON file containing bait_direction_refs from an earlier "
            "phase. Current bait directions are decorrelated against these frozen "
            "external basis/residual-gap directions, which keeps two-stage "
            "training from reintroducing a global shortcut."
        ),
    )
    parser.add_argument(
        "--residual-gap-anti-coherence-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for data-dependent anti-coherence on defended good/bad "
            "residual gap directions at bait residuals. Unlike basis-only "
            "anti-coherence, this directly targets the directions Heretic can "
            "estimate from activations. 0 = disabled."
        ),
    )
    parser.add_argument(
        "--residual-gap-anti-coherence-margin",
        type=float,
        default=0.0,
        help=(
            "Allowed absolute cosine for residual-gap anti-coherence before the "
            "hinge activates. 0 forces pairwise orthogonality."
        ),
    )
    parser.add_argument(
        "--global-direction-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for Heretic-style global direction robustness. This penalizes "
            "defended good/bad residual gaps at bait layers when they still project "
            "onto a single interpolated global refusal direction, matching "
            "Heretic's global direction_index search. 0 = disabled."
        ),
    )
    parser.add_argument(
        "--global-direction-samples",
        type=int,
        default=5,
        help=(
            "Number of interpolated global direction_index candidates sampled "
            "between --global-direction-range-low and --global-direction-range-high."
        ),
    )
    parser.add_argument(
        "--global-direction-margin",
        type=float,
        default=0.0,
        help=(
            "Allowed absolute cosine before the global-direction hinge activates. "
            "Set around 0.05-0.15 to avoid spending loss on already weak projections."
        ),
    )
    parser.add_argument(
        "--global-direction-range-low",
        type=float,
        default=0.4,
        help="Lower fraction of the layer stack for Heretic-style global direction_index sampling.",
    )
    parser.add_argument(
        "--global-direction-range-high",
        type=float,
        default=0.9,
        help="Upper fraction of the layer stack for Heretic-style global direction_index sampling.",
    )
    parser.add_argument(
        "--progressive-geometry-joint-epochs",
        type=int,
        default=0,
        help=(
            "Optional short all-pairs pass after progressive isolated stages, even "
            "when --skip-progressive-joint-finetune is set. This exposes "
            "cross-layer gap/global losses to all installed pairs without running "
            "a long behavioral joint fine-tune."
        ),
    )
    parser.add_argument(
        "--progressive-geometry-joint-lr-scale",
        type=float,
        default=0.05,
        help="Learning-rate multiplier for --progressive-geometry-joint-epochs.",
    )
    parser.add_argument(
        "--progressive-geometry-joint-include-local-losses",
        action="store_true",
        default=False,
        help=(
            "Keep KL/recovery/visible local losses in the short all-pairs geometry "
            "pass. By default that pass is geometry-only to avoid bait gate/basis "
            "instabilities from local restoration losses."
        ),
    )
    parser.add_argument(
        "--progressive-geometry-joint-train-bait",
        action="store_true",
        default=False,
        help=(
            "Allow the short all-pairs geometry pass to update bait parameters. "
            "By default bait is frozen and only recovery adapters move, which "
            "keeps visible bait strength and KL much more stable."
        ),
    )
    parser.add_argument(
        "--progressive-geometry-joint-kl-rollback-budget",
        type=float,
        default=0.0,
        help=(
            "Rollback the short all-pairs geometry pass if post-pass KL on the "
            "training guard subset exceeds this value. 0 falls back to "
            "--final-kl-budget, then --progressive-kl-budget."
        ),
    )
    parser.add_argument(
        "--progressive-geometry-joint-kl-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Small KL anchor used inside the geometry-only all-pairs pass. "
            "Recovery/visible losses remain disabled; this only keeps logits "
            "from drifting while geometry losses move recovery adapters."
        ),
    )
    parser.add_argument(
        "--progressive-geometry-joint-interpolation-steps",
        type=int,
        default=8,
        help=(
            "When the geometry pass exceeds the guard KL, binary-search between "
            "the pre-pass and post-pass weights for the largest safe update. "
            "0 restores the previous all-or-nothing rollback behavior."
        ),
    )
    parser.add_argument(
        "--progressive-block-joint-epochs",
        type=int,
        default=0,
        help=(
            "Optional block-wise joint fine-tuning pass after progressive isolated "
            "stages. Each block enables only a small contiguous group of installed "
            "bait/recovery pairs, giving joint training signal without enabling "
            "all layers at once."
        ),
    )
    parser.add_argument(
        "--progressive-block-joint-size",
        type=int,
        default=0,
        help=(
            "Number of installed bait/recovery pairs per block for "
            "--progressive-block-joint-epochs. 0 means all pairs."
        ),
    )
    parser.add_argument(
        "--progressive-block-joint-stride",
        type=int,
        default=0,
        help=(
            "Stride between block starts for block-wise joint fine-tuning. "
            "0 uses non-overlapping blocks of --progressive-block-joint-size."
        ),
    )
    parser.add_argument(
        "--progressive-block-joint-lr-scale",
        type=float,
        default=0.05,
        help="Learning-rate multiplier for block-wise joint fine-tuning.",
    )
    parser.add_argument(
        "--progressive-block-joint-kl-rollback-budget",
        type=float,
        default=0.0,
        help=(
            "Rollback or partially interpolate each block-wise joint update if "
            "full-defense guard KL exceeds this value. 0 falls back to "
            "--final-kl-budget, then --progressive-kl-budget."
        ),
    )
    parser.add_argument(
        "--progressive-block-joint-interpolation-steps",
        type=int,
        default=8,
        help=(
            "Binary-search steps used to keep the largest safe fraction of a "
            "block-wise joint update when it exceeds the guard KL. 0 restores "
            "all-or-nothing rollback behavior."
        ),
    )
    parser.add_argument(
        "--no-progressive-block-joint-geometry-fallback",
        dest="progressive_block_joint_geometry_fallback",
        action="store_false",
        default=True,
        help=(
            "Disable the block-wise joint fallback that retries a numerically "
            "unstable behavior+geometry block with only geometry/global losses."
        ),
    )
    parser.add_argument(
        "--no-progressive-block-joint-detach-visible-bait",
        dest="progressive_block_joint_detach_visible_bait",
        action="store_false",
        default=True,
        help=(
            "Allow block-wise joint visible loss to backprop into bait parameters. "
            "By default it is detached for block joint only, because multi-pair "
            "visible gradients are numerically fragile on large bf16 models."
        ),
    )
    parser.add_argument(
        "--progressive-block-joint-disable-local-losses",
        action="store_true",
        help=(
            "For block-wise joint fine-tuning, disable KL/recovery/visible local "
            "losses and train only geometry/global losses. This avoids the known "
            "multi-pair recovery/visible non-finite gradients on large bf16 models."
        ),
    )
    parser.add_argument(
        "--unfreeze-bait",
        action="store_true",
        default=True,
        help=(
            "Unfreeze bait coeff weights so the input projection is trainable "
            "(default; use --no-unfreeze-bait to disable)."
        ),
    )
    parser.add_argument(
        "--no-unfreeze-bait",
        dest="unfreeze_bait",
        action="store_false",
        help="Freeze bait coeff weights without using --freeze-bait-coeff.",
    )
    parser.add_argument(
        "--freeze-bait",
        action="store_true",
        default=False,
        help="Force bait coeff+basis weights to be fully frozen (overrides --unfreeze-bait)",
    )
    parser.add_argument(
        "--freeze-bait-gate",
        action="store_true",
        default=False,
        help=(
            "Freeze bait_gate after initialization/calibration. This makes bait "
            "strength data-calibrated but not optimized through the model backbone."
        ),
    )
    parser.add_argument(
        "--bait-gate-log-init",
        type=float,
        default=BAIT_GATE_LOG_INIT,
        help=(
            "Initial log gate for bait adapters before optional calibration. "
            "The current stable default is -2.0; use 0.0 to reproduce older "
            "pre-calibration experiments with gate near 1."
        ),
    )
    parser.add_argument(
        "--refusal-cancel-min-rank",
        type=int,
        default=4,
        help=(
            "Minimum rank used in 'refusal_cancel' / 'sidechannel_cancel' bait mode, "
            "overriding the weak-dimension-constrained rank. The refusal_cancel mode needs at "
            "least rank 2 (one for -r_clean, one for orthogonal complement); "
            "sidechannel_cancel needs at least 3 (r_clean + tag + one SVD column). "
            "A higher value gives more expressive power. Default=4."
        ),
    )
    parser.add_argument(
        "--tag-scale",
        type=float,
        default=1.0,
        help=(
            "Scale factor for the explicit side-channel tag vector in "
            "'sidechannel_cancel' mode. Controls how strongly the tag signal "
            "is injected relative to the r_clean cancellation. Default=1.0."
        ),
    )
    parser.add_argument(
        "--sidechannel-tag-trigger-mode",
        type=str,
        default="shared_supervised",
        choices=["shared_supervised", "orthogonal_svd", "hybrid_svd", "legacy_qr"],
        help=(
            "How sidechannel_cancel chooses the input trigger A[:,1] for the tag "
            "channel. shared_supervised reuses A[:,0] so cancellation and tag fire "
            "from the same supervised bad-prompt direction. orthogonal_svd uses an "
            "explicit SVD-derived direction projected orthogonal to A[:,0], making "
            "the old accidental QR behavior reproducible without duplicate-column "
            "QR. hybrid_svd mixes the supervised and orthogonal SVD directions. "
            "legacy_qr intentionally runs the old duplicate-column QR path for "
            "checkpoint reproduction."
        ),
    )
    parser.add_argument(
        "--sidechannel-tag-trigger-mix",
        type=float,
        default=0.25,
        help=(
            "Supervised-direction mix used by --sidechannel-tag-trigger-mode=hybrid_svd. "
            "0 is fully orthogonal_svd; 1 is shared_supervised."
        ),
    )
    parser.add_argument(
        "--random-orthogonal-exclude-rank",
        type=int,
        default=32,
        help=(
            "For bait-output-mode=random_orthogonal, project sampled bait directions "
            "away from this many strongest down_proj residual-space singular vectors."
        ),
    )
    parser.add_argument(
        "--recovery-init-scale",
        type=float,
        default=0.25,
        help="Scaling applied to weight-derived recovery transport bases",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate for recovery adapters",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay for recovery adapters",
    )
    parser.add_argument(
        "--recovery-loss-weight",
        type=float,
        default=0.5,
        help="Weight for explicit clean-vs-defended alignment on next-layer o_proj recovery",
    )
    parser.add_argument(
        "--kl-loss-weight",
        type=float,
        default=1.0,
        help="Weight for KL(defended logits || clean logits). Set 0 to isolate local recovery/visible losses.",
    )
    parser.add_argument(
        "--kl-loss-mode",
        type=str,
        default="full",
        choices=["full", "last_token", "topk_last"],
        help=(
            "KL preservation objective used during training. full reproduces the "
            "dense full-sequence/full-vocabulary KL. last_token computes full-vocab "
            "KL only at each prompt's final valid prediction position. topk_last "
            "computes a bounded top-k KL at the final valid prediction position, "
            "which is much more stable for bf16 Qwen recovery training."
        ),
    )
    parser.add_argument(
        "--kl-top-k",
        type=int,
        default=64,
        help="Reference top-k vocabulary size for --kl-loss-mode=topk_last.",
    )
    parser.add_argument(
        "--kl-temperature",
        type=float,
        default=1.0,
        help="Temperature applied to defended/reference logits before training KL.",
    )
    parser.add_argument(
        "--kl-logit-clamp",
        type=float,
        default=0.0,
        help="Optional symmetric clamp for training KL logits before softmax. 0 disables.",
    )
    parser.add_argument(
        "--auto-normalize-loss-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Before each training stage, estimate KL/RF gradient norms on a small "
            "batch and downscale their effective weights so they do not dominate "
            "the local recovery/visible gradients."
        ),
    )
    parser.add_argument(
        "--loss-normalization-target-ratio",
        type=float,
        default=0.5,
        help=(
            "Target max gradient ratio for each normalized global loss term "
            "relative to the local recovery/visible objective."
        ),
    )
    parser.add_argument(
        "--loss-normalization-batch-size",
        type=int,
        default=0,
        help=(
            "Probe batch size for --auto-normalize-loss-weights. 0 reuses "
            "--batch-size."
        ),
    )
    parser.add_argument(
        "--visible-loss-weight",
        type=float,
        default=1.0,
        help="Weight for keeping bait visibly active on defended down_proj outputs",
    )
    parser.add_argument(
        "--detach-local-loss-from-bait",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Deprecated umbrella switch for --detach-recovery-loss-from-bait, "
            "--detach-visible-loss-from-bait, and --detach-kl-loss-from-bait. "
            "Prefer the per-loss switches."
        ),
    )
    parser.add_argument(
        "--detach-recovery-loss-from-bait",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Remove recovery-loss gradients from bait parameters. This lets "
            "recovery adapters learn to clean up bait without teaching bait to "
            "be easier for recovery to cancel."
        ),
    )
    parser.add_argument(
        "--detach-visible-loss-from-bait",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Remove visible-loss gradients from bait parameters. Default false "
            "because visible_loss is the direct bait-strength objective."
        ),
    )
    parser.add_argument(
        "--detach-kl-loss-from-bait",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Remove KL-preservation gradients from bait parameters while still "
            "allowing KL to train recovery adapters when KL weight is nonzero."
        ),
    )
    parser.add_argument(
        "--visible-shift-target",
        type=float,
        default=0.15,
        help="Minimum relative down_proj shift for bait visibility",
    )
    parser.add_argument(
        "--visible-shift-max",
        type=float,
        default=0.0,
        help=(
            "Optional maximum relative down_proj shift for bait visibility. "
            "Set to e.g. 0.25 or 0.30 to prevent refusal_cancel over-compensation. "
            "0 disables the upper hinge."
        ),
    )
    parser.add_argument(
        "--auto-calibrate-bait-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Before training, measure each bait pair's actual relative down_proj "
            "shift on the current training prompts and adjust bait_gate so the "
            "initial injection strength is data-calibrated."
        ),
    )
    parser.add_argument(
        "--bait-calibration-target",
        type=float,
        default=0.02,
        help="Target initial relative down_proj shift used by --auto-calibrate-bait-gate.",
    )
    parser.add_argument(
        "--bait-calibration-quantile",
        type=float,
        default=0.90,
        help="Relative-shift quantile used for bait gate calibration.",
    )
    parser.add_argument(
        "--bait-calibration-log-min",
        type=float,
        default=-8.0,
        help="Lower clamp for calibrated bait_gate log value.",
    )
    parser.add_argument(
        "--bait-calibration-log-max",
        type=float,
        default=0.0,
        help="Upper clamp for calibrated bait_gate log value.",
    )
    parser.add_argument(
        "--bait-calibration-kl-target",
        type=float,
        default=0.0,
        help=(
            "Optional pair-level KL cap used during bait gate calibration. "
            "When > 0, each calibrated gate is further downscaled until the "
            "single-pair KL is near this target."
        ),
    )
    parser.add_argument(
        "--bait-calibration-kl-iters",
        type=int,
        default=2,
        help="Maximum multiplicative downscale iterations for --bait-calibration-kl-target.",
    )
    parser.add_argument(
        "--recovery-max-weight",
        type=float,
        default=0.0,
        help=(
            "Plan D: add 'w · max_over_hooks' to transport_alignment_loss so the "
            "worst-performing recovery hook always receives gradient. 0 reproduces "
            "the legacy mean-only aggregation. Recommended ≈1.0 when training >1 pair."
        ),
    )
    parser.add_argument(
        "--visible-max-weight",
        type=float,
        default=0.0,
        help=(
            "Plan D: same as --recovery-max-weight but for visible_bait_loss. "
            "Ensures no single bait falls below the floor or overshoots the ceiling "
            "while others stay inside the visible-shift band."
        ),
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for training/evaluation",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=128,
        help="Number of unlabeled general-text prompts for KL training",
    )
    parser.add_argument(
        "--n-good",
        type=int,
        default=64,
        help="Number of harmless prompts for evaluation",
    )
    parser.add_argument(
        "--n-bad",
        type=int,
        default=64,
        help="Number of harmful prompts for evaluation",
    )
    parser.add_argument(
        "--n-train-good",
        type=int,
        default=64,
        help="Number of harmless prompts used in recovery training (separate from eval set)",
    )
    parser.add_argument(
        "--n-train-bad",
        type=int,
        default=64,
        help="Number of harmful prompts used in recovery training (separate from eval set)",
    )
    parser.add_argument(
        "--train-good-offset",
        type=int,
        default=64,
        help="Index offset for training harmless prompts (default=n_good, so train/eval don't overlap)",
    )
    parser.add_argument(
        "--train-bad-offset",
        type=int,
        default=64,
        help="Index offset for training harmful prompts (default=n_bad, so train/eval don't overlap)",
    )
    parser.add_argument(
        "--eval-data-source",
        type=str,
        default="mlabonne",
        help=(
            DATA_SOURCE_HELP
            + " Used for defense-side evaluation prompts and supervised bait directions."
        ),
    )
    parser.add_argument(
        "--train-data-source",
        type=str,
        default="",
        help=(
            DATA_SOURCE_HELP
            + " Used for labeled recovery-training prompts. Defaults to --eval-data-source."
        ),
    )
    parser.add_argument(
        "--train-good-data-source",
        type=str,
        default="",
        help=DATA_SOURCE_HELP + " Used only for harmless labeled recovery-training prompts.",
    )
    parser.add_argument(
        "--train-bad-data-source",
        type=str,
        default="",
        help=DATA_SOURCE_HELP + " Used only for harmful labeled recovery-training prompts.",
    )
    parser.add_argument(
        "--general-data-source",
        type=str,
        default="",
        help=(
            DATA_SOURCE_HELP
            + " Used for unlabeled KL-preservation prompts. Defaults to --train-data-source."
        ),
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping threshold",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="",
        help=(
            "HuggingFace model id or local path to load the tokenizer from. "
            "Defaults to --model when not set. Use this when --model points to a "
            "local merged checkpoint whose saved tokenizer triggers false-positive "
            "warnings (e.g. the Mistral regex check on a Gemma tokenizer)."
        ),
    )
    parser.add_argument(
        "--original-clean-model",
        type=str,
        default="",
        help=(
            "Optional original/base model id or path used only for an extra clean "
            "per-residual good/bad Fisher profile in the JSON output. Useful when "
            "--model is a Phase 1 merged checkpoint."
        ),
    )
    parser.add_argument(
        "--original-clean-tokenizer",
        type=str,
        default="",
        help=(
            "Tokenizer id/path for --original-clean-model. Defaults to "
            "--original-clean-model when empty."
        ),
    )
    parser.add_argument(
        "--experiment-tag",
        type=str,
        default="",
        help=(
            "Override the model-derived output filename tag. "
            "By default output files use model.replace('/', '_') as the tag. "
            "Set this (e.g. 'phase2_google_gemma-3-1b-it') to produce clean filenames "
            "when --model points to a local merged-model directory."
        ),
    )
    parser.add_argument(
        "--phase2-target-layers",
        type=str,
        default="",
        help=(
            "Comma-separated layer indices for Phase 2 bait/recovery pairs. "
            "When set, Phase 1 adapters are merged into model weights first, "
            "then Phase 2 adapters are installed and trained on the merged model. "
            "This achieves full-coverage interleaved defense: Phase 1 covers odd layers "
            "(e.g. 15,17,19,21,23) and Phase 2 covers even layers (e.g. 14,16,18,20,24). "
            "Every layer in the range then has both a bait injection and a downstream recovery."
        ),
    )
    parser.add_argument(
        "--model-save-dir",
        type=Path,
        default=None,
        help=(
            "Directory for saving/loading merged model checkpoints. Defaults to "
            "RESULTS_DIR when not set. Use a fast local disk (e.g. /tmp or /local/scratch) "
            "when the working directory is on a slow network mount, to speed up "
            "model serialization and Phase 2 model loading."
        ),
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "npu") and torch.npu.is_available():  # type: ignore[attr-defined]
        torch.npu.manual_seed_all(seed)  # type: ignore[attr-defined]


@dataclass
class LayerPrior:
    layer_idx: int
    rank: int
    bait_scale: float
    recovery_init_scale: float
    top_singular_values: list[float]
    spectrum_energy: float
    weak_dimensions: int
    candidate_rank: int
    selection_score: float


def load_step1_summary(step1_path: Path) -> dict:
    with open(step1_path) as f:
        return json.load(f)


def parse_requested_layers(target_layers: str) -> list[int]:
    if not target_layers.strip():
        return []
    return sorted({int(token.strip()) for token in target_layers.split(",") if token.strip()})


def get_available_step1_layers(step1_summary: dict) -> list[int]:
    layers = []
    for key in step1_summary:
        if key.startswith("layer_"):
            layers.append(int(key.split("_")[1]))
    return sorted(layers)


def get_layer_fisher_score(step1_summary: dict, layer_idx: int) -> float:
    baseline = step1_summary.get("fisher_ratios_baseline", [])
    residual_idx = layer_idx + 1
    if residual_idx < len(baseline):
        return float(baseline[residual_idx])
    if layer_idx < len(baseline):
        return float(baseline[layer_idx])
    return 0.0


def get_layer_weak_dimensions(step1_summary: dict, layer_idx: int) -> int:
    layer_summary = step1_summary[f"layer_{layer_idx}"]
    gate_alignment = layer_summary.get("gate_proj_alignment", {})
    up_alignment = layer_summary.get("up_proj_alignment", {})
    return int(
        min(
            up_alignment.get("n_weak_dimensions", 0),
            gate_alignment.get("n_weak_dimensions", up_alignment.get("n_weak_dimensions", 0)),
        )
    )


def get_feasible_rank(
    desired_rank: int,
    weak_dimensions: int,
) -> int:
    if weak_dimensions <= 0:
        return 1
    return max(1, min(desired_rank, weak_dimensions))


def select_target_layers(
    step1_summary: dict,
    requested_layers: list[int],
    num_target_layers: int,
    args: argparse.Namespace,
    max_layer_idx: int | None = None,
) -> list[int]:
    """Select target layers for bait injection.

    When max_layer_idx is set, layers beyond it are excluded because
    the cross-layer design needs layer l+1 for recovery.
    """
    available_layers = get_available_step1_layers(step1_summary)
    if max_layer_idx is not None:
        available_layers = [l for l in available_layers if l < max_layer_idx]
    if requested_layers:
        selected = [layer for layer in requested_layers if layer in available_layers]
        if not selected:
            raise ValueError(
                f"None of the requested target layers exist in the step1 summary "
                f"(available: {available_layers}, max allowed: {max_layer_idx})"
            )
        return selected

    candidates = []
    for layer_idx in available_layers:
        layer_summary = step1_summary[f"layer_{layer_idx}"]
        desired_rank, _ = infer_rank_and_energy(
            layer_summary["up_proj_svd"]["singular_values"],
            min_rank=args.min_rank,
            max_rank=args.max_rank,
            energy_target=args.rank_energy_target,
        )
        weak_dimensions = get_layer_weak_dimensions(step1_summary, layer_idx)
        feasible_rank = get_feasible_rank(desired_rank, weak_dimensions)
        fisher = get_layer_fisher_score(step1_summary, layer_idx)
        candidates.append(
            (
                feasible_rank >= min(args.min_rank, max(weak_dimensions, 1)),
                feasible_rank,
                fisher,
                layer_idx,
            )
        )

    scored = sorted(candidates, reverse=True)
    return sorted(item[-1] for item in scored[:num_target_layers])


def select_heretic_coverage_layers(
    step1_summary: dict,
    n_model_layers: int,
    args: argparse.Namespace,
) -> list[int]:
    """Select layers to cover based on Heretic's traversal pattern.

    Heretic (heretic/src/heretic/model.py) runs abliterate() over ALL layers
    (range(n_layers)), but the Optuna objective concentrates weight in a
    configurable window (`max_weight_position`, `min_weight_distance`).  The
    empirically observed strongest band for most LLMs is roughly [0.4·L, 0.9·L].

    --heretic-target-layers  : explicit layer list → use directly
    --heretic-layer-range mid  : [floor(0.4·L), floor(0.9·L)]  (default)
    --heretic-layer-range wide/all : [0, L-2] (all layers that can have a recovery)

    All selected layers must appear in step1_summary AND have l+1 < n_model_layers.
    """
    max_bait_layer = n_model_layers - 1  # recovery goes at l+1, so bait ≤ L-2
    available_layers = get_available_step1_layers(step1_summary)
    available_layers = [l for l in available_layers if l < max_bait_layer]

    # 1. Explicit override
    if args.heretic_target_layers.strip():
        explicit = parse_requested_layers(args.heretic_target_layers)
        selected = [l for l in explicit if l in available_layers]
        missing = [l for l in explicit if l not in available_layers]
        if missing:
            raise ValueError(
                f"--heretic-target-layers requested layers missing from Step1 summary "
                f"or outside bait-layer limit ({max_bait_layer}): {missing}. "
                f"Available: {available_layers}. Regenerate Step1 with matching "
                "--heretic-coverage-layers / HERETIC_COVERAGE_LAYERS."
            )
        if not selected:
            raise ValueError(
                f"--heretic-target-layers: none of {explicit} found in step1 summary "
                f"or within bait-layer limit ({max_bait_layer}). "
                f"Available: {available_layers}"
            )
        coverage_layers = getattr(args, "heretic_coverage_layers", 0)
        coverage_note = f" (cov{coverage_layers})" if coverage_layers > 0 else ""
        print(f"  Heretic coverage: explicit layers {selected}{coverage_note}")
        return sorted(selected)

    # 2. Range-based selection
    coverage_layers = getattr(args, "heretic_coverage_layers", 0)
    heretic_range = getattr(args, "heretic_layer_range", "mid")
    if coverage_layers > 0:
        selected = available_layers[-coverage_layers:]
        print(
            f"  Heretic coverage: deepest {coverage_layers} available layers "
            f"→ {len(selected)} layers selected = {selected}"
        )
    else:
        if heretic_range == "mid":
            lo = math.floor(0.40 * n_model_layers)
            hi = math.floor(0.90 * n_model_layers)
            selected = [l for l in available_layers if lo <= l <= hi]
            print(
                f"  Heretic coverage: 'mid' band layers [{lo}, {hi}] "
                f"→ {len(selected)} layers selected = {selected}"
            )
        else:  # wide / all
            selected = list(available_layers)  # already filtered for < max_bait_layer
            print(
                f"  Heretic coverage: 'all' layers (0..{max_bait_layer-1}) "
                f"→ {len(selected)} layers selected"
            )

    if not selected:
        raise ValueError(
            f"No layers selected for Heretic coverage (range='{heretic_range}', "
            f"available={available_layers}, n_layers={n_model_layers}). "
            "Try --heretic-layer-range all or provide --heretic-target-layers explicitly."
        )

    # 3. Fisher-based pruning
    # Layers where the clean model's Fisher discriminability (good vs bad)
    # is very low have a poorly-defined refusal direction at that depth.
    # refusal_cancel bait pointing to -r_clean produces near-zero DPS there
    # while still adding KL cost from the recovery adapters.  Prune them.
    min_fisher = getattr(args, "min_fisher_for_coverage", 0.0)
    if min_fisher > 0.0:
        before = selected[:]
        selected = [
            l for l in selected
            if get_layer_fisher_score(step1_summary, l) >= min_fisher
        ]
        pruned = [l for l in before if l not in selected]
        if pruned:
            scores = {l: get_layer_fisher_score(step1_summary, l) for l in pruned}
            print(
                f"  Fisher pruning (threshold={min_fisher:.3f}): "
                f"removed {pruned} (scores={scores})"
            )
        if not selected:
            raise ValueError(
                f"All candidate layers pruned by --min-fisher-for-coverage={min_fisher}. "
                f"Lower the threshold or use --heretic-target-layers explicitly."
            )

    return sorted(selected)


def infer_rank_and_energy(
    singular_values: list[float],
    min_rank: int,
    max_rank: int,
    energy_target: float,
) -> tuple[int, float]:
    if not singular_values:
        return min_rank, 1.0

    sv_tensor = torch.tensor(singular_values, dtype=torch.float32)
    energy = torch.cumsum(sv_tensor.square(), dim=0) / sv_tensor.square().sum().clamp_min(1e-8)
    rank = int(torch.searchsorted(energy, torch.tensor(energy_target), right=False).item()) + 1
    rank = max(min_rank, min(max_rank, rank, len(singular_values)))
    captured_energy = float(energy[rank - 1].item())
    return rank, captured_energy


def build_layer_priors(
    step1_summary: dict,
    target_layers: list[int],
    args: argparse.Namespace,
) -> list[LayerPrior]:
    priors: list[LayerPrior] = []
    for layer_idx in target_layers:
        layer_key = f"layer_{layer_idx}"
        layer_summary = step1_summary[layer_key]
        up_svd = layer_summary["up_proj_svd"]
        weak_dimensions = get_layer_weak_dimensions(step1_summary, layer_idx)
        candidate_rank, spectrum_energy = infer_rank_and_energy(
            up_svd["singular_values"],
            min_rank=args.min_rank,
            max_rank=args.max_rank,
            energy_target=args.rank_energy_target,
        )
        rank = get_feasible_rank(candidate_rank, weak_dimensions)

        # For refusal_cancel mode the weak-dimension floor is irrelevant:
        # that concept comes from up/gate_proj alignment, not from the residual
        # stream direction we actually inject into.  Override with a dedicated
        # minimum so the -r_clean column has enough orthogonal companions.
        if getattr(args, "bait_output_mode", "svd") in ("refusal_cancel", "sidechannel_cancel"):
            rc_min_rank = getattr(args, "refusal_cancel_min_rank", 4)
            if getattr(args, "bait_output_mode", "svd") == "sidechannel_cancel":
                rc_min_rank = max(rc_min_rank, 3)  # need at least: r_clean + tag + 1 SVD
            rank = max(rank, rc_min_rank)
            rank = min(rank, args.max_rank)  # still respect the ceiling

        if up_svd["singular_values"]:
            sv_tensor = torch.tensor(up_svd["singular_values"], dtype=torch.float32)
            energy = torch.cumsum(sv_tensor.square(), dim=0) / sv_tensor.square().sum().clamp_min(1e-8)
            spectrum_energy = float(energy[min(rank, len(up_svd["singular_values"])) - 1].item())
        scale_gain = math.sqrt(max(spectrum_energy, 1e-8))
        fisher = get_layer_fisher_score(step1_summary, layer_idx)
        priors.append(
            LayerPrior(
                layer_idx=layer_idx,
                rank=rank,
                bait_scale=args.bait_scale * scale_gain,
                recovery_init_scale=args.recovery_init_scale * scale_gain,
                top_singular_values=up_svd["singular_values"][:rank],
                spectrum_energy=spectrum_energy,
                weak_dimensions=weak_dimensions,
                candidate_rank=candidate_rank,
                selection_score=fisher * math.sqrt(max(rank, 1)),
            )
        )
    return priors


def compute_svd_factors(
    weight: torch.Tensor,
    method: str = "full",
    lowrank_q: int | None = None,
    lowrank_niter: int = 2,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    matrix = weight.detach().float().cpu()
    if not torch.isfinite(matrix).all():
        finite = int(torch.isfinite(matrix).sum().item())
        raise ValueError(
            f"Cannot compute SVD for weight with {matrix.numel() - finite} "
            f"non-finite values out of {matrix.numel()} elements"
        )
    if method == "full":
        left, singular_values, right_t = torch.linalg.svd(matrix, full_matrices=False)
        return left, singular_values, right_t.T
    if method != "lowrank":
        raise ValueError(f"Unknown SVD method {method!r}; expected 'full' or 'lowrank'.")

    q = lowrank_q or min(matrix.shape)
    q = min(q, min(matrix.shape))
    rng_state = torch.random.get_rng_state()
    try:
        if seed is not None:
            torch.manual_seed(seed)
        left, singular_values, right = torch.svd_lowrank(matrix, q=q, niter=lowrank_niter)
    finally:
        torch.random.set_rng_state(rng_state)
    return left, singular_values, right


def select_svd_columns(
    basis: torch.Tensor,
    singular_values: torch.Tensor,
    rank: int,
    strongest: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    rank = min(rank, basis.shape[1], singular_values.shape[0])
    if strongest:
        return basis[:, :rank], singular_values[:rank]
    return basis[:, -rank:], singular_values[-rank:]


def select_svd_band(
    basis: torch.Tensor,
    singular_values: torch.Tensor,
    rank: int,
    mode: str,
    pct_low: float = 0.05,
    pct_high: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select SVD columns from a configurable spectral band.

    Args:
        basis: (features, k) matrix of singular vectors
        singular_values: (k,) singular values in descending order
        rank: how many columns to select
        mode: 'strongest', 'weakest', or 'mid'
        pct_low: lower percentile for 'mid' mode (0 = strongest end)
        pct_high: upper percentile for 'mid' mode (1 = weakest end)

    Returns:
        (selected_basis, selected_svs) — both sliced to the chosen band
    """
    n = min(basis.shape[1], singular_values.shape[0])
    rank = min(rank, n)

    if mode == "strongest":
        return basis[:, :rank], singular_values[:rank]
    elif mode == "weakest":
        return basis[:, -rank:], singular_values[-rank:]
    else:  # mid
        idx_low = max(0, int(n * pct_low))
        idx_high = min(n, int(n * pct_high))
        # ensure we have enough room for the requested rank
        if idx_high - idx_low < rank:
            # widen symmetrically
            center = (idx_low + idx_high) // 2
            idx_low = max(0, center - rank // 2)
            idx_high = min(n, idx_low + rank)
            idx_low = max(0, idx_high - rank)
        selected = basis[:, idx_low:idx_low + rank]
        selected_sv = singular_values[idx_low:idx_low + rank]
        return selected, selected_sv


def orthonormalize_columns(matrix: torch.Tensor) -> torch.Tensor:
    q, _ = torch.linalg.qr(matrix.float(), mode="reduced")
    return q


def random_basis_orthogonal_to(
    reference_basis: torch.Tensor,
    rank: int,
    seed: int,
) -> torch.Tensor:
    """Sample a deterministic random basis outside a reference subspace."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    features = reference_basis.shape[0]
    rank = min(rank, features)
    if reference_basis.shape[1] == 0:
        samples = torch.randn(features, rank, generator=generator, dtype=torch.float32)
        return orthonormalize_columns(samples)

    reference_basis = orthonormalize_columns(reference_basis.float())

    samples = torch.randn(features, rank, generator=generator, dtype=torch.float32)
    samples = samples - reference_basis @ (reference_basis.T @ samples)
    basis = orthonormalize_columns(samples)

    # Extremely small complements can make the first QR nearly singular. Re-sample
    # with extra columns and project again, then keep the requested rank.
    if basis.shape[1] < rank or torch.linalg.matrix_rank(basis) < rank:
        samples = torch.randn(
            features,
            rank + reference_basis.shape[1],
            generator=generator,
            dtype=torch.float32,
        )
        samples = samples - reference_basis @ (reference_basis.T @ samples)
        basis = orthonormalize_columns(samples)[:, :rank]

    return basis[:, :rank]


def unit_vector(vector: torch.Tensor) -> torch.Tensor:
    return vector.float() / vector.float().norm().clamp_min(1e-8)


def project_vector_away_from(vector: torch.Tensor, reference_basis: torch.Tensor) -> torch.Tensor:
    """Project one vector away from a reference column space and normalize it."""
    candidate = vector.float()
    if reference_basis.numel() == 0 or reference_basis.shape[1] == 0:
        return unit_vector(candidate)
    basis = orthonormalize_columns(reference_basis.float())
    candidate = candidate - basis @ (basis.T @ candidate)
    return unit_vector(candidate)


def choose_orthogonal_svd_trigger(
    svd_basis: torch.Tensor,
    supervised_dir: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    """Choose a deterministic input trigger orthogonal to the supervised direction.

    This makes the old sidechannel_cancel duplicate-column QR side effect explicit:
    A[:,1] is independent from A[:,0], but it is selected from the layer's real
    down_proj input spectrum when possible instead of relying on QR's arbitrary
    handling of duplicated columns.
    """
    supervised = unit_vector(supervised_dir.cpu())
    reference = supervised[:, None]

    best_candidate = None
    best_norm = -1.0
    for idx in range(svd_basis.shape[1]):
        raw = svd_basis[:, idx].float().cpu()
        projected = raw - supervised * (raw @ supervised)
        projected_norm = float(projected.norm().item())
        if projected_norm > best_norm:
            best_norm = projected_norm
            best_candidate = projected

    if best_candidate is not None and best_norm > 1e-6:
        return unit_vector(best_candidate)

    # Degenerate case: sample a deterministic random complement.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    random_vec = torch.randn(supervised.shape, generator=generator, dtype=supervised.dtype)
    return project_vector_away_from(random_vec, reference)


def move_tensor_to_linear(tensor: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
    return tensor.to(device=linear.weight.device, dtype=torch.float32)


def assert_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    detached = tensor.detach()
    if torch.isfinite(detached).all():
        return
    finite = int(torch.isfinite(detached).sum().item())
    raise FloatingPointError(
        f"{name} contains {detached.numel() - finite} non-finite values "
        f"out of {detached.numel()}"
    )


def assert_module_parameters_finite(module: nn.Module, context: str) -> None:
    for name, param in module.named_parameters():
        assert_finite_tensor(f"{context}.{name}", param)


def assert_gradients_finite(named_params: list[tuple[str, nn.Parameter]], context: str = "") -> None:
    bad = []
    for name, param in named_params:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        if torch.isfinite(grad).all():
            continue
        finite = int(torch.isfinite(grad).sum().item())
        bad.append(f"{name}: {grad.numel() - finite}/{grad.numel()} non-finite")
    if bad:
        suffix = f" ({context})" if context else ""
        raise FloatingPointError(
            "Non-finite gradients before optimizer step"
            + suffix
            + ": "
            + "; ".join(bad[:8])
        )


def nonfinite_gradient_names(named_params: list[tuple[str, nn.Parameter]]) -> list[str]:
    bad = []
    for name, param in named_params:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        if not torch.isfinite(grad).all():
            bad.append(name)
    return bad


# ---------------------------------------------------------------------
# Structured bait generation on `down_proj`
# ---------------------------------------------------------------------
class BaitLoRA(nn.Module):
    """
    Low-rank bait adapter attached to down_proj.

    Implements δ_l = B · A · h_l, where:
      - B (basis) is initialized from down_proj left SVs — directions in residual stream space
      - A (coeff) is initialized from down_proj right SVs — what patterns to read from MLP hidden
      - δ_l is added to down_proj output → enters the residual stream directly
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        input_basis: torch.Tensor,
        output_basis: torch.Tensor,
        alpha: float,
        freeze_coeff: bool = False,
        freeze_basis: bool = True,
        freeze_gate: bool = False,
        gate_log_init: float = BAIT_GATE_LOG_INIT,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.enabled = True
        self.coeff = nn.Linear(in_features, rank, bias=False)
        self.basis = nn.Linear(rank, out_features, bias=False)
        self.bait_gate = nn.Parameter(torch.tensor(gate_log_init, dtype=torch.float32))

        with torch.no_grad():
            self.coeff.weight.copy_(input_basis.T)
            self.basis.weight.copy_(output_basis)

        # Store initial basis for projection regularization (always on CPU to save memory)
        self.register_buffer(
            "basis_init", output_basis.detach().clone().cpu(), persistent=False,
        )

        self.coeff.weight.requires_grad_(not freeze_coeff)
        self.basis.weight.requires_grad_(not freeze_basis)
        self.bait_gate.requires_grad_(not freeze_gate)

    def basis_regularization_loss(self) -> torch.Tensor:
        """Penalize basis drifting outside the initial subspace.

        Computes ||P_perp @ B_current||^2 where P_perp = I - U_init @ U_init^T
        and U_init is the orthonormal column space of the initial basis.
        """
        init_basis = self.basis_init.to(self.basis.weight.device, dtype=torch.float32)
        # init_basis is (out_features, rank) — get its column space
        U_init, _ = torch.linalg.qr(init_basis, mode="reduced")  # (out_features, rank)
        # Project current basis onto complement of initial subspace
        B_current = self.basis.weight.float()  # (out_features, rank)
        proj = U_init @ (U_init.T @ B_current)  # projection onto init subspace
        perp = B_current - proj  # component outside init subspace
        return perp.square().sum()

    def gate(self, *, detach: bool = False) -> torch.Tensor:
        value = self.bait_gate.detach() if detach else self.bait_gate
        return torch.exp(value.clamp(max=BAIT_GATE_LOG_MAX))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_fp32 = hidden_states.float()
        hidden_fp32 = torch.nan_to_num(hidden_fp32, posinf=0.0, neginf=0.0)
        z_l = self.coeff(hidden_fp32)
        gate = self.gate()
        delta = self.basis(z_l) * (self.alpha * gate)
        delta_casted = delta.to(hidden_states.dtype)
        if delta_casted.requires_grad:
            delta_casted.register_hook(lambda grad: torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0))
        return delta_casted

    def merged_delta_weight(self) -> torch.Tensor:
        gate = self.gate(detach=True).float()
        scale = self.alpha * gate
        return (self.basis.weight.detach().float() @ self.coeff.weight.detach().float()) * scale

    def fold_gate_into_basis(self) -> None:
        gate = self.gate(detach=True).float()
        with torch.no_grad():
            self.basis.weight.mul_(gate.to(self.basis.weight.device, self.basis.weight.dtype))
            self.bait_gate.zero_()


# ---------------------------------------------------------------------
# Cross-layer recovery adapter attached to o_proj
# ---------------------------------------------------------------------
class RecoveryLoRA(nn.Module):
    """
    Trainable low-rank recovery adapter attached to o_proj.
    This reads from the attention output and writes to the residual stream,
    learning to cancel the bait signal.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        input_basis: torch.Tensor,
        output_basis: torch.Tensor,
        init_scale: float = 1e-3,
    ):
        super().__init__()
        self.rank = rank
        self.enabled = True

        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)

        with torch.no_grad():
            self.down.weight.copy_(input_basis.T)
            self.up.weight.copy_(output_basis * init_scale)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        h_fp32 = hidden_states.float()
        h_fp32 = torch.nan_to_num(h_fp32, posinf=0.0, neginf=0.0)
        delta = self.up(self.down(h_fp32))
        delta_casted = delta.to(hidden_states.dtype)
        if delta_casted.requires_grad:
            delta_casted.register_hook(lambda grad: torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0))
        return delta_casted

    def merged_delta_weight(self) -> torch.Tensor:
        return self.up.weight.detach().float() @ self.down.weight.detach().float()


class LinearWithAdapters(nn.Module):
    def __init__(self, base_linear: nn.Linear, adapters: list[nn.Module] | None = None):
        super().__init__()
        self.base_linear = base_linear
        self.adapters = nn.ModuleList(adapters or [])
        self.adapters_enabled = True

    @property
    def weight(self) -> torch.Tensor:
        return self.base_linear.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base_linear.bias

    @property
    def in_features(self) -> int:
        return self.base_linear.in_features

    @property
    def out_features(self) -> int:
        return self.base_linear.out_features

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(hidden_states)
        if not self.adapters_enabled:
            return base_out

        delta = None
        for adapter in self.adapters:
            if getattr(adapter, "enabled", True):
                update = adapter(hidden_states)
                delta = update if delta is None else delta + update

        if delta is None:
            return base_out
        return base_out + delta.to(base_out.dtype)

    def merged_linear(self) -> nn.Linear:
        merged = nn.Linear(
            self.base_linear.in_features,
            self.base_linear.out_features,
            bias=self.base_linear.bias is not None,
            device=self.base_linear.weight.device,
            dtype=self.base_linear.weight.dtype,
        )
        with torch.no_grad():
            assert_finite_tensor("base linear weight before merge", self.base_linear.weight)
            merged.weight.copy_(self.base_linear.weight)
            if self.base_linear.bias is not None:
                assert_finite_tensor("base linear bias before merge", self.base_linear.bias)
                merged.bias.copy_(self.base_linear.bias)
            for adapter in self.adapters:
                if hasattr(adapter, "fold_gate_into_basis"):
                    adapter.fold_gate_into_basis()
                if not hasattr(adapter, "merged_delta_weight"):
                    raise ValueError(f"Adapter {type(adapter).__name__} is not mergeable into a linear layer")
                assert_module_parameters_finite(adapter, type(adapter).__name__)
                delta_weight = adapter.merged_delta_weight().to(merged.weight.device, merged.weight.dtype)
                assert_finite_tensor(f"{type(adapter).__name__} merged delta weight", delta_weight)
                merged.weight.add_(delta_weight)
                assert_finite_tensor("merged linear weight", merged.weight)
        return merged


@dataclass
class InstalledLayerDefense:
    """Tracks a cross-layer bait/recovery pair."""
    bait_layer_idx: int          # layer l: bait injected at down_proj
    recovery_layer_idx: int      # layer l+1: recovery at o_proj
    rank: int
    bait_scale: float
    recovery_init_scale: float


def compute_supervised_residual_directions(
    model: nn.Module,
    tokenizer,
    good_prompts: list,
    bad_prompts: list,
    layer_indices: list[int],
    batch_size: int,
) -> dict[int, torch.Tensor]:
    """Compute normalize(mean_bad - mean_good) in RESIDUAL STREAM space at position l+1.

    This is exactly what Heretic computes per layer before ablation.  We compute it
    from the CLEAN model (adapters disabled) so we know the ground-truth refusal direction.
    Used by 'refusal_cancel' bait output mode to initialize δ's output direction along
    -r_clean, directly reducing the class-mean separation Heretic would measure.
    """
    from shared_utils import get_residuals_batched
    print("  Computing supervised residual-stream refusal directions (Heretic-style)...")
    good_res = get_residuals_batched(model, tokenizer, good_prompts, batch_size)  # (n, L+1, d)
    bad_res  = get_residuals_batched(model, tokenizer, bad_prompts,  batch_size)

    directions: dict[int, torch.Tensor] = {}
    for l_idx in layer_indices:
        res_pos = l_idx + 1  # residual index after layer l
        if res_pos >= good_res.shape[1]:
            continue
        diff = bad_res[:, res_pos, :].mean(dim=0) - good_res[:, res_pos, :].mean(dim=0)
        norm = diff.norm()
        directions[l_idx] = diff / norm.clamp_min(1e-8)
        print(f"    Layer {l_idx} residual[{res_pos}] refusal dir norm: {norm:.4f}")
    return directions


def compute_supervised_input_directions(
    model: nn.Module,
    tokenizer,
    good_prompts: list,
    bad_prompts: list,
    layer_indices: list[int],
    module_name: str,
    batch_size: int,
) -> dict[int, torch.Tensor]:
    """Compute (mean_bad - mean_good) direction at the input of specified modules."""
    print(f"  Computing supervised refusal directions at {module_name} inputs...")
    directions = {}
    layers = get_layers(model)

    for l_idx in layer_indices:
        module = get_layer_module(layers[l_idx], module_name)
        good_acts = []
        bad_acts = []

        def hook_fn(m, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            return x[:, -1, :].detach().float().cpu()

        # We'll use a simpler capture approach to avoid nested hooks issues
        def capture(prompts):
            acts = []
            def h(m, i, o): acts.append(i[0][:, -1, :].detach().float().cpu())
            handle = module.register_forward_hook(h)
            try:
                for start in range(0, len(prompts), batch_size):
                    batch = prompts[start : start + batch_size]
                    inputs = prepare_batch(model, tokenizer, batch)
                    with torch.no_grad():
                        model(**inputs)
            finally:
                handle.remove()
            return torch.cat(acts, dim=0)

        g_tensor = capture(good_prompts)
        b_tensor = capture(bad_prompts)
        
        diff = b_tensor.mean(dim=0) - g_tensor.mean(dim=0)
        norm = diff.norm()
        if norm > 1e-8:
            diff = diff / norm
        
        directions[l_idx] = diff
        print(f"    Layer {l_idx} {module_name} refusal dir norm: {norm:.4f}")
        
    return directions


class DefenseController:
    """Cross-layer defense: bait at layer l down_proj, recovery at layer l+1 o_proj + down_proj."""

    def __init__(
        self,
        model: nn.Module,
        priors: list[LayerPrior],
        unfreeze_bait: bool = True,
        bait_subspace_mode: str = "mid",
        bait_sv_pct_low: float = 0.05,
        bait_sv_pct_high: float = 0.25,
        bait_basis_trainable: bool = False,
        supervised_directions: dict[int, torch.Tensor] = None,
        bait_output_mode: str = "svd",
        supervised_residual_dirs: dict[int, torch.Tensor] = None,
        random_orthogonal_exclude_rank: int = 32,
        seed: int = 42,
        force_freeze_basis: bool = False,
        freeze_bait_gate: bool = False,
        tag_scale: float = 1.0,
        sidechannel_tag_trigger_mode: str = "shared_supervised",
        sidechannel_tag_trigger_mix: float = 0.25,
        bait_gate_log_init: float = BAIT_GATE_LOG_INIT,
        svd_method: str = "full",
        svd_lowrank_q: int = 32,
        svd_lowrank_niter: int = 4,
    ):
        self.model = model
        self.wrappers: list[LinearWithAdapters] = []
        self.recovery_modules: list[RecoveryLoRA] = []
        self.bait_modules: list[BaitLoRA] = []
        self.installed_layers: list[InstalledLayerDefense] = []
        self.bait_subspace_mode = bait_subspace_mode
        self._install(
            priors,
            unfreeze_bait=unfreeze_bait,
            bait_subspace_mode=bait_subspace_mode,
            bait_sv_pct_low=bait_sv_pct_low,
            bait_sv_pct_high=bait_sv_pct_high,
            bait_basis_trainable=bait_basis_trainable,
            supervised_directions=supervised_directions,
            bait_output_mode=bait_output_mode,
            supervised_residual_dirs=supervised_residual_dirs,
            random_orthogonal_exclude_rank=random_orthogonal_exclude_rank,
            seed=seed,
            force_freeze_basis=force_freeze_basis,
            freeze_bait_gate=freeze_bait_gate,
            tag_scale=tag_scale,
            sidechannel_tag_trigger_mode=sidechannel_tag_trigger_mode,
            sidechannel_tag_trigger_mix=sidechannel_tag_trigger_mix,
            bait_gate_log_init=bait_gate_log_init,
            svd_method=svd_method,
            svd_lowrank_q=svd_lowrank_q,
            svd_lowrank_niter=svd_lowrank_niter,
        )

    def _install(
        self,
        priors: list[LayerPrior],
        unfreeze_bait: bool = True,
        bait_subspace_mode: str = "mid",
        bait_sv_pct_low: float = 0.05,
        bait_sv_pct_high: float = 0.25,
        bait_basis_trainable: bool = False,
        supervised_directions: dict[int, torch.Tensor] = None,
        bait_output_mode: str = "svd",
        supervised_residual_dirs: dict[int, torch.Tensor] = None,
        random_orthogonal_exclude_rank: int = 32,
        seed: int = 42,
        force_freeze_basis: bool = False,
        freeze_bait_gate: bool = False,
        tag_scale: float = 1.0,
        sidechannel_tag_trigger_mode: str = "shared_supervised",
        sidechannel_tag_trigger_mix: float = 0.25,
        bait_gate_log_init: float = BAIT_GATE_LOG_INIT,
        svd_method: str = "full",
        svd_lowrank_q: int = 32,
        svd_lowrank_niter: int = 4,
    ) -> None:
        layers = get_layers(self.model)
        n_layers = len(layers)

        for prior in priors:
            bait_layer_idx = prior.layer_idx
            recovery_layer_idx = bait_layer_idx + 1

            if recovery_layer_idx >= n_layers:
                print(
                    f"  [yellow]WARNING[/]: Layer {bait_layer_idx} is the last layer, "
                    f"cannot place recovery at layer {recovery_layer_idx}. Skipping."
                )
                continue

            bait_layer = layers[bait_layer_idx]
            recovery_layer = layers[recovery_layer_idx]

            down_proj = bait_layer.mlp.down_proj   # bait goes here
            next_o_proj = get_layer_module(recovery_layer, "attn.o_proj")
            if next_o_proj is None:
                raise AttributeError(
                    f"Layer {recovery_layer_idx} has no supported attention output projection"
                )

            # ── Bait SVD basis from down_proj ──
            # down_proj: W shape (hidden_dim, intermediate_size)
            # W = U @ diag(σ) @ V^T
            # U (left SVs):  output space directions (hidden_dim) — these go into residual stream
            # V (right SVs): input space directions (intermediate_size) — what to read from MLP hidden
            dp_left, dp_sigma, dp_right = compute_svd_factors(
                down_proj.weight,
                method=svd_method,
                lowrank_q=svd_lowrank_q,
                lowrank_niter=svd_lowrank_niter,
                seed=seed,
            )

            # Bait input basis: target the features most indicative of refusal
            if (
                bait_output_mode != "random_orthogonal"
                and supervised_directions
                and bait_layer_idx in supervised_directions
            ):
                sup_dir = supervised_directions[bait_layer_idx].cpu()
                bait_input_basis_cpu, _ = select_svd_columns(
                    dp_right, dp_sigma, rank=prior.rank, strongest=True,
                )
                # Replace the first component with the supervised refusal direction
                # and re-orthonormalize to keep the subspace clean
                bait_input_basis_cpu[:, 0] = sup_dir
                bait_input_basis_cpu = orthonormalize_columns(bait_input_basis_cpu)
                # QR sign correction for input basis: torch.linalg.qr does NOT guarantee
                # column sign. If A[:,0] flips to -sup_dir, bad prompts produce negative
                # z_l[0], so delta_diff = B[:,0] * (negative) = +r_clean → sign flip.
                # Ensure A[:,0] · sup_dir > 0 so bad prompts always get positive z_l[0].
                input_cos = (bait_input_basis_cpu[:, 0] @ sup_dir.to(bait_input_basis_cpu)).item()
                if input_cos < 0:
                    bait_input_basis_cpu[:, 0] = -bait_input_basis_cpu[:, 0]
                print(
                    f"  Layer {bait_layer_idx} using supervised refusal direction for bait trigger "
                    f"(input qr_flip={'YES' if input_cos < 0 else 'no'} "
                    f"cos(A[:,0], sup_dir)={abs(input_cos):.4f})"
                )
            else:
                # Fallback: down_proj strongest RIGHT SVs
                bait_input_basis_cpu, _ = select_svd_columns(
                    dp_right, dp_sigma, rank=prior.rank, strongest=True,
                )

            # Bait output basis: which directions in residual stream to inject δ into
            # Two modes:
            #   'svd'           : spectral band of down_proj left SVs (original approach)
            #   'refusal_cancel': align output with -r_clean so bait directly reduces the
            #                     class-mean separation that Heretic measures at residual l+1
            if (
                bait_output_mode == "refusal_cancel"
                and supervised_residual_dirs is not None
                and bait_layer_idx in supervised_residual_dirs
            ):
                r_clean = supervised_residual_dirs[bait_layer_idx].cpu()  # (hidden_dim,)
                # Start from down_proj left SVs for the remaining rank-1 columns
                bait_output_basis_cpu = dp_left[:, :prior.rank].float().clone()
                # First column = -r_clean: bait injects opposite to refusal direction,
                # reducing bad-good separation at residual 18 when bad prompts come in
                bait_output_basis_cpu[:, 0] = -r_clean.float()
                bait_output_basis_cpu = orthonormalize_columns(bait_output_basis_cpu)

                # ── BUG FIX: QR sign correction ──
                # torch.linalg.qr does NOT guarantee column signs — it may flip
                # column 0 to align with +r_clean instead of -r_clean. Check and fix.
                r_clean_f = r_clean.float()
                init_cos = (bait_output_basis_cpu[:, 0] @ r_clean_f).item()
                if init_cos > 0:  # column 0 points toward +r_clean → flip it
                    bait_output_basis_cpu[:, 0] = -bait_output_basis_cpu[:, 0]
                actual_cos = (bait_output_basis_cpu[:, 0] @ r_clean_f).item()
                print(
                    f"  Layer {bait_layer_idx} down_proj bait: mode='refusal_cancel' "
                    f"(output along -r_clean at residual {bait_layer_idx + 1}) "
                    f"qr_flip={'YES' if init_cos > 0 else 'no'} "
                    f"cos(B[:,0], r_clean)={actual_cos:.4f}  [want ≈ -1]"
                )
            elif (
                bait_output_mode == "sidechannel_cancel"
                and supervised_residual_dirs is not None
                and bait_layer_idx in supervised_residual_dirs
            ):
                # ── Side-channel cancel mode ──
                # B[:,0] = -r_clean  (refusal cancellation, same as refusal_cancel)
                # B[:,1] = tag       (orthogonal to r_clean, carries class-dependent signal)
                # A[1,:] = supervised input direction (class-dependent probe)
                # Fisher suppression loss only targets r_clean → tag dimension is free
                # Recovery LoRA reads the tag to distinguish good/bad and restore r_clean
                r_clean = supervised_residual_dirs[bait_layer_idx].cpu()  # (hidden_dim,)
                r_clean_f = r_clean.float()

                # Compute orthogonal tag vector in residual stream space
                torch.manual_seed(seed + bait_layer_idx + 7777)
                random_vec = torch.randn_like(r_clean_f)
                r_clean_norm = r_clean_f / r_clean_f.norm().clamp_min(1e-8)
                tag = random_vec - (random_vec @ r_clean_norm) * r_clean_norm
                tag = tag / tag.norm().clamp_min(1e-8)

                # Build output basis: col0=-r_clean, col1=tag, rest from SVD
                bait_output_basis_cpu = dp_left[:, :prior.rank].float().clone()
                bait_output_basis_cpu[:, 0] = -r_clean_f
                bait_output_basis_cpu[:, 1] = tag * tag_scale
                bait_output_basis_cpu = orthonormalize_columns(bait_output_basis_cpu)

                # QR sign correction for col 0 (-r_clean)
                init_cos_0 = (bait_output_basis_cpu[:, 0] @ r_clean_f).item()
                if init_cos_0 > 0:
                    bait_output_basis_cpu[:, 0] = -bait_output_basis_cpu[:, 0]
                # QR sign correction for col 1 (tag): should align with original tag
                init_cos_1 = (bait_output_basis_cpu[:, 1] @ tag).item()
                if init_cos_1 < 0:
                    bait_output_basis_cpu[:, 1] = -bait_output_basis_cpu[:, 1]

                actual_cos_0 = (bait_output_basis_cpu[:, 0] @ r_clean_f).item()
                actual_cos_tag = (bait_output_basis_cpu[:, 1] @ tag).item()
                ortho_check = (bait_output_basis_cpu[:, 0] @ bait_output_basis_cpu[:, 1]).item()

                # Choose the tag channel's input trigger explicitly.  The legacy
                # duplicate-column QR path accidentally made A[:,1] independent
                # from A[:,0] on some models; keep that behavior available as a
                # deterministic SVD-derived option without relying on singular QR.
                if supervised_directions is not None and bait_layer_idx in supervised_directions:
                    sup_input_dir = supervised_directions[bait_layer_idx].cpu().float()
                    sup_input_dir = unit_vector(sup_input_dir)
                    if sidechannel_tag_trigger_mode == "shared_supervised":
                        tag_input_dir = sup_input_dir
                    elif sidechannel_tag_trigger_mode == "legacy_qr":
                        legacy_basis = bait_input_basis_cpu.float().clone()
                        legacy_basis[:, 0] = sup_input_dir
                        legacy_basis[:, 1] = sup_input_dir
                        legacy_basis = orthonormalize_columns(legacy_basis)
                        if (legacy_basis[:, 0] @ sup_input_dir).item() < 0:
                            legacy_basis[:, 0] = -legacy_basis[:, 0]
                        tag_input_dir = legacy_basis[:, 1]
                    else:
                        orth_input_dir = choose_orthogonal_svd_trigger(
                            dp_right.float().cpu(),
                            sup_input_dir,
                            seed=seed + bait_layer_idx + 9191,
                        )
                        if sidechannel_tag_trigger_mode == "orthogonal_svd":
                            tag_input_dir = orth_input_dir
                        elif sidechannel_tag_trigger_mode == "hybrid_svd":
                            mix = min(max(float(sidechannel_tag_trigger_mix), 0.0), 1.0)
                            tag_input_dir = unit_vector(
                                mix * sup_input_dir + (1.0 - mix) * orth_input_dir
                            )
                        else:
                            raise ValueError(
                                f"Unknown sidechannel_tag_trigger_mode="
                                f"{sidechannel_tag_trigger_mode!r}"
                            )

                    bait_input_basis_cpu[:, 1] = tag_input_dir
                    a1_cos = (bait_input_basis_cpu[:, 1] @ sup_input_dir).item()
                    if a1_cos < 0:
                        bait_input_basis_cpu[:, 1] = -bait_input_basis_cpu[:, 1]
                        a1_cos = -a1_cos
                    tag_input_status = (
                        f"{sidechannel_tag_trigger_mode} "
                        f"(cos_sup={a1_cos:.4f}, "
                        f"dot_A0={float((bait_input_basis_cpu[:, 0] @ bait_input_basis_cpu[:, 1]).item()):.4f})"
                    )
                else:
                    tag_input_status = "SVD (no supervised input dir)"

                print(
                    f"  Layer {bait_layer_idx} down_proj bait: mode='sidechannel_cancel' "
                    f"tag_scale={tag_scale:.2f} "
                    f"cos(B[:,0],r_clean)={actual_cos_0:.4f} [want≈-1] "
                    f"cos(B[:,1],tag)={actual_cos_tag:.4f} [want≈+1] "
                    f"ortho(B0,B1)={ortho_check:.4e} "
                    f"A[1]={tag_input_status}"
                )
            elif bait_output_mode == "random_orthogonal":
                max_exclude_rank = max(0, dp_left.shape[0] - prior.rank)
                exclude_rank = min(
                    max(random_orthogonal_exclude_rank, prior.rank),
                    dp_left.shape[1],
                    max_exclude_rank,
                )
                excluded_basis = dp_left[:, :exclude_rank]
                bait_output_basis_cpu = random_basis_orthogonal_to(
                    excluded_basis,
                    rank=prior.rank,
                    seed=seed + bait_layer_idx,
                )
                if exclude_rank > 0:
                    max_abs_cos = float((excluded_basis.T @ bait_output_basis_cpu).abs().max().item())
                else:
                    max_abs_cos = 0.0
                print(
                    f"  Layer {bait_layer_idx} down_proj bait: mode='random_orthogonal' "
                    f"(rank={prior.rank}, excluded_top_svs={exclude_rank}, "
                    f"max|cos(excluded,B)|={max_abs_cos:.4e})"
                )
            else:
                # SVD spectral band (original)
                bait_output_basis_cpu, bait_output_svs = select_svd_band(
                    dp_left, dp_sigma, rank=prior.rank,
                    mode=bait_subspace_mode, pct_low=bait_sv_pct_low, pct_high=bait_sv_pct_high,
                )
                bait_output_basis_cpu = orthonormalize_columns(bait_output_basis_cpu)
                print(
                    f"  Layer {bait_layer_idx} down_proj bait: band='{bait_subspace_mode}' "
                    f"sv_range=[{float(bait_output_svs[-1]):.4f}, {float(bait_output_svs[0]):.4f}] "
                    f"(full range [{float(dp_sigma[-1]):.4f}, {float(dp_sigma[0]):.4f}])"
                )

            # ── Install bait on down_proj ──
            bait_input = move_tensor_to_linear(bait_input_basis_cpu, down_proj)
            bait_output = move_tensor_to_linear(bait_output_basis_cpu, down_proj)

            # In refusal_cancel mode the output direction (B / basis) needs to stay
            # adjustable so delta-Fisher loss can reduce good/bad separability in
            # δ(x).  Coeff (A) stays frozen per --freeze-bait-coeff, but basis is
            # made trainable automatically.  bait_basis_trainable overrides this for
            # non-refusal_cancel modes.
            _freeze_basis = not bait_basis_trainable
            if bait_output_mode in ("refusal_cancel", "sidechannel_cancel") and not unfreeze_bait:
                # Coeff frozen → only basis can carry direction gradient.  Allow it.
                _freeze_basis = False
            # User can override the auto-unfreeze above with --freeze-bait.
            # Critical when bait_output_mode == "refusal_cancel": basis_regularization
            # only penalizes drift OUT of the init subspace, it does NOT prevent sign
            # flip within the same 1D span (multiplying a column by -1 stays in span).
            # Observed failure mode: on the deepest/last-trained pair the optimizer
            # flipped B[:,0] from -r_clean to +r_clean, amplifying good/bad Fisher
            # rather than erasing it. Hard-freezing the basis eliminates this.
            if force_freeze_basis:
                _freeze_basis = True

            bait = BaitLoRA(
                in_features=down_proj.in_features,
                out_features=down_proj.out_features,
                rank=prior.rank,
                input_basis=bait_input,
                output_basis=bait_output,
                alpha=prior.bait_scale,
                freeze_coeff=not unfreeze_bait,
                freeze_basis=_freeze_basis,
                freeze_gate=freeze_bait_gate,
                gate_log_init=bait_gate_log_init,
            ).to(device=down_proj.weight.device, dtype=torch.float32)

            dp_wrapper = LinearWithAdapters(down_proj, [bait]).to(device=down_proj.weight.device)
            bait_layer.mlp.down_proj = dp_wrapper
            self.wrappers.append(dp_wrapper)
            self.bait_modules.append(bait)

            # ── Install recovery on next layer's o_proj ──
            # Recovery must output -δ to cancel bait from the residual stream.
            # Input detector: look for signals in the attention output.
            op_left, op_sigma, op_right = compute_svd_factors(
                next_o_proj.weight,
                method=svd_method,
                lowrank_q=svd_lowrank_q,
                lowrank_niter=svd_lowrank_niter,
                seed=seed,
            )
            recovery_detector_cpu, _ = select_svd_columns(
                op_right, op_sigma, rank=prior.rank, strongest=True,
            )
            recovery_detector = move_tensor_to_linear(recovery_detector_cpu, next_o_proj)

            # Transport initialization: δ lives in bait_output_basis subspace,
            # so the o_proj recovery should output -bait_output_basis.
            recovery_output = move_tensor_to_linear(-bait_output_basis_cpu, next_o_proj)

            recovery_o = RecoveryLoRA(
                in_features=next_o_proj.in_features,
                out_features=next_o_proj.out_features,
                rank=prior.rank,
                input_basis=recovery_detector,
                output_basis=recovery_output,
                init_scale=prior.recovery_init_scale,
            ).to(device=next_o_proj.weight.device, dtype=torch.float32)

            o_wrapper = LinearWithAdapters(next_o_proj, [recovery_o]).to(device=next_o_proj.weight.device)
            set_layer_module(recovery_layer, "attn.o_proj", o_wrapper)
            self.wrappers.append(o_wrapper)
            self.recovery_modules.append(recovery_o)

            # ── Install second recovery on next layer's down_proj ──
            # The MLP of layer l+1 also receives the contaminated residual_18 as input
            # and propagates δ through its own pathway.  A second RecoveryLoRA on
            # down_proj handles the MLP-pathway contribution independently.
            next_down_proj = recovery_layer.mlp.down_proj
            ndp_left, ndp_sigma, ndp_right = compute_svd_factors(
                next_down_proj.weight,
                method=svd_method,
                lowrank_q=svd_lowrank_q,
                lowrank_niter=svd_lowrank_niter,
                seed=seed,
            )

            # Input basis: strongest right SVs of recovery-layer down_proj
            ndp_detector_cpu, _ = select_svd_columns(
                ndp_right, ndp_sigma, rank=prior.rank, strongest=True,
            )
            ndp_detector = move_tensor_to_linear(ndp_detector_cpu, next_down_proj)

            # Output basis: same -bait_output_basis (δ lives in hidden_dim space)
            ndp_recovery_output = move_tensor_to_linear(-bait_output_basis_cpu, next_down_proj)

            recovery_dp = RecoveryLoRA(
                in_features=next_down_proj.in_features,
                out_features=next_down_proj.out_features,
                rank=prior.rank,
                input_basis=ndp_detector,
                output_basis=ndp_recovery_output,
                init_scale=prior.recovery_init_scale,
            ).to(device=next_down_proj.weight.device, dtype=torch.float32)

            dp_wrapper = LinearWithAdapters(next_down_proj, [recovery_dp]).to(device=next_down_proj.weight.device)
            recovery_layer.mlp.down_proj = dp_wrapper
            self.wrappers.append(dp_wrapper)
            self.recovery_modules.append(recovery_dp)

            self.installed_layers.append(
                InstalledLayerDefense(
                    bait_layer_idx=bait_layer_idx,
                    recovery_layer_idx=recovery_layer_idx,
                    rank=prior.rank,
                    bait_scale=prior.bait_scale,
                    recovery_init_scale=prior.recovery_init_scale,
                )
            )
            print(
                f"  Installed: bait @ layer {bait_layer_idx} down_proj → "
                f"recovery @ layer {recovery_layer_idx} o_proj + down_proj (rank={prior.rank})"
            )

    def trainable_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for module in self.bait_modules:
            params.extend([param for param in module.parameters() if param.requires_grad])
        for module in self.recovery_modules:
            params.extend([param for param in module.parameters() if param.requires_grad])
        return params

    def trainable_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        params: list[tuple[str, nn.Parameter]] = []
        for idx, module in enumerate(self.bait_modules):
            params.extend(
                (f"bait[{idx}].{name}", param)
                for name, param in module.named_parameters()
                if param.requires_grad
            )
        for idx, module in enumerate(self.recovery_modules):
            params.extend(
                (f"recovery[{idx}].{name}", param)
                for name, param in module.named_parameters()
                if param.requires_grad
            )
        return params

    def set_enabled(self, enabled: bool) -> None:
        for wrapper in self.wrappers:
            wrapper.adapters_enabled = enabled

    @contextmanager
    def adapters_disabled(self):
        previous = [wrapper.adapters_enabled for wrapper in self.wrappers]
        self.set_enabled(False)
        try:
            yield
        finally:
            for wrapper, state in zip(self.wrappers, previous):
                wrapper.adapters_enabled = state

    def checkpoint(self) -> dict:
        layers = get_layers(self.model)
        payload = {
            "layers": [item.__dict__ for item in self.installed_layers],
            "bait_state": {},
            "recovery_state": {},
        }
        for item in self.installed_layers:
            bait_layer = layers[item.bait_layer_idx]
            recovery_layer = layers[item.recovery_layer_idx]
            payload["bait_state"][f"layer_{item.bait_layer_idx}_down_proj"] = (
                bait_layer.mlp.down_proj.adapters[0].state_dict()
            )
            recovery_o_proj = get_layer_module(recovery_layer, "attn.o_proj")
            if isinstance(recovery_o_proj, LinearWithAdapters):
                payload["recovery_state"][f"layer_{item.recovery_layer_idx}_o_proj"] = (
                    recovery_o_proj.adapters[0].state_dict()
                )
            # Second recovery site: recovery_layer down_proj
            if isinstance(recovery_layer.mlp.down_proj, LinearWithAdapters):
                payload["recovery_state"][f"layer_{item.recovery_layer_idx}_down_proj"] = (
                    recovery_layer.mlp.down_proj.adapters[0].state_dict()
                )
        return payload

    def merge_defense(self) -> None:
        layers = get_layers(self.model)
        for item in self.installed_layers:
            bait_layer = layers[item.bait_layer_idx]
            recovery_layer = layers[item.recovery_layer_idx]
            if isinstance(bait_layer.mlp.down_proj, LinearWithAdapters):
                bait_layer.mlp.down_proj = bait_layer.mlp.down_proj.merged_linear()
            recovery_o_proj = get_layer_module(recovery_layer, "attn.o_proj")
            if isinstance(recovery_o_proj, LinearWithAdapters):
                set_layer_module(
                    recovery_layer,
                    "attn.o_proj",
                    recovery_o_proj.merged_linear(),
                )
            # Second recovery site
            if isinstance(recovery_layer.mlp.down_proj, LinearWithAdapters):
                recovery_layer.mlp.down_proj = recovery_layer.mlp.down_proj.merged_linear()

        self.wrappers.clear()
        self.recovery_modules.clear()
        self.bait_modules.clear()

    def export_merged_checkpoint(self, export_dir: Path, tokenizer) -> Path:
        export_dir.mkdir(parents=True, exist_ok=True)
        self.merge_defense()
        self.model.save_pretrained(export_dir)
        tokenizer.save_pretrained(export_dir)
        return export_dir


def get_input_device(model: nn.Module) -> torch.device:
    input_embeddings = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    if input_embeddings is not None:
        return input_embeddings.weight.device
    return next(model.parameters()).device


def prepare_batch(model: nn.Module, tokenizer, prompts: list[Prompt]) -> dict[str, torch.Tensor]:
    inputs = tokenize_prompts(prompts, tokenizer)
    device = get_input_device(model)
    return {key: value.to(device) for key, value in inputs.items()}


@contextmanager
def capture_last_token_outputs(
    modules: dict[str, nn.Module],
    *,
    detach: bool = True,
    cpu: bool = True,
):
    cache = {name: [] for name in modules}
    handles = []

    def make_hook(name: str):
        def hook_fn(module, inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            tensor = out[:, -1, :].float()
            if detach:
                tensor = tensor.detach()
            if cpu:
                tensor = tensor.cpu()
            cache[name].append(tensor)

        return hook_fn

    for name, module in modules.items():
        handles.append(module.register_forward_hook(make_hook(name)))

    try:
        yield cache
    finally:
        for handle in handles:
            handle.remove()


def masked_logit_kl(
    defended_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    defended_log_probs = F.log_softmax(defended_logits[:, :-1, :].float(), dim=-1)
    reference_probs = F.softmax(reference_logits[:, :-1, :].float(), dim=-1)
    token_kl = F.kl_div(defended_log_probs, reference_probs, reduction="none").sum(dim=-1)
    valid_mask = attention_mask[:, 1:].float()
    return (token_kl * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)


def training_logit_kl(
    defended_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    mode: str = "full",
    top_k: int = 64,
    temperature: float = 1.0,
    logit_clamp: float = 0.0,
) -> torch.Tensor:
    """KL objective used for training.

    The full sequence/full vocabulary KL gives a very dense gradient that must
    backpropagate through all downstream bf16 transformer blocks. On Qwen-8B this
    can produce non-finite recovery-adapter gradients even when the scalar KL is
    small and finite. The lighter modes keep a KL preservation signal while
    reducing that dense bf16 backward pressure.
    """
    if temperature <= 0.0:
        raise ValueError("--kl-temperature must be > 0")
    if mode == "full":
        return masked_logit_kl(
            defended_logits / temperature,
            reference_logits / temperature,
            attention_mask,
        ) * (temperature ** 2)

    valid_mask = attention_mask[:, 1:].bool()
    prediction_positions = torch.arange(
        valid_mask.shape[1],
        device=valid_mask.device,
    ).unsqueeze(0)
    last_positions = prediction_positions.masked_fill(~valid_mask, -1).max(dim=1).values
    keep = last_positions >= 0
    if not bool(keep.any()):
        return defended_logits.float().sum() * 0.0

    batch_indices = torch.arange(defended_logits.shape[0], device=defended_logits.device)[keep]
    last_positions = last_positions[keep]
    defended = defended_logits[batch_indices, last_positions, :].float() / temperature
    reference = reference_logits[batch_indices, last_positions, :].float() / temperature
    if logit_clamp > 0.0:
        defended = defended.clamp(-logit_clamp, logit_clamp)
        reference = reference.clamp(-logit_clamp, logit_clamp)

    if mode == "last_token":
        defended_log_probs = F.log_softmax(defended, dim=-1)
        reference_probs = F.softmax(reference, dim=-1)
        token_kl = F.kl_div(defended_log_probs, reference_probs, reduction="none").sum(dim=-1)
        return token_kl.mean() * (temperature ** 2)

    if mode != "topk_last":
        raise ValueError(f"Unknown KL loss mode: {mode!r}")
    if top_k <= 0:
        raise ValueError("--kl-top-k must be > 0 for topk_last KL")
    k = min(int(top_k), reference.shape[-1])
    _, top_indices = torch.topk(reference.detach(), k=k, dim=-1)
    defended_top = defended.gather(dim=-1, index=top_indices)
    reference_top = reference.gather(dim=-1, index=top_indices)
    defended_log_probs = F.log_softmax(defended_top, dim=-1)
    reference_probs = F.softmax(reference_top, dim=-1)
    token_kl = F.kl_div(defended_log_probs, reference_probs, reduction="none").sum(dim=-1)
    return token_kl.mean() * (temperature ** 2)


def load_general_prompts(n_prompts: int, source: str = "mlabonne") -> list[Prompt]:
    system_prompt = "You are a helpful assistant."
    source = source.strip() or "mlabonne"
    source_key = source.lower()
    if source_key != "mlabonne":
        good_prompts, _ = load_prompts_simple(n_prompts, 0, source=source)
        return good_prompts

    local_data_dir = Path(__file__).parent.parent / "heretic" / "data"
    local_harmless = local_data_dir / "harmless_alpaca_train.jsonl"

    prompts: list[Prompt] = []
    if local_harmless.exists():
        print("Loading general training text from local harmless Alpaca data...")
        with open(local_harmless) as f:
            for index, line in enumerate(f):
                if index >= n_prompts:
                    break
                item = json.loads(line)
                prompts.append(Prompt(system=system_prompt, user=item["text"]))
        return prompts

    print("Loading general training text from tatsu-lab/alpaca...")
    dataset = load_dataset(
        "tatsu-lab/alpaca",
        split=f"train[:{n_prompts}]",
        trust_remote_code=True,
    )
    for row in dataset:
        prompts.append(Prompt(system=system_prompt, user=row["instruction"]))
    return prompts


def evaluate_kl(
    model: nn.Module,
    tokenizer,
    controller: DefenseController,
    prompts: list[Prompt],
    batch_size: int,
) -> float:
    losses = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        inputs = prepare_batch(model, tokenizer, batch)
        with controller.adapters_disabled():
            with torch.no_grad():
                reference_logits = model(**inputs, return_dict=True).logits
        with torch.no_grad():
            defended_logits = model(**inputs, return_dict=True).logits
        losses.append(masked_logit_kl(defended_logits, reference_logits, inputs["attention_mask"]).item())
    return float(sum(losses) / max(len(losses), 1))


def evaluate_kl_by_prompt_type(
    model: nn.Module,
    tokenizer,
    controller: DefenseController,
    good_prompts: list[Prompt],
    bad_prompts: list[Prompt],
    batch_size: int,
) -> dict[str, float]:
    """KL(defended || clean) computed separately for good and bad prompt sets.

    Mirrors Heretic's two populations:
      - good (harmless): KL should be low — defense must not disturb normal behaviour.
      - bad  (harmful):  KL may be higher if the bait is selectively perturbing
        the harmful-prompt residual stream, which is exactly the intended effect.
    """
    kl_good = evaluate_kl(model, tokenizer, controller, good_prompts, batch_size)
    kl_bad  = evaluate_kl(model, tokenizer, controller, bad_prompts,  batch_size)
    return {"kl_good": kl_good, "kl_bad": kl_bad}


def evaluate_transport_metrics(
    model: nn.Module,
    tokenizer,
    controller: DefenseController,
    prompts: list[Prompt],
    batch_size: int,
) -> dict[str, dict[str, float]]:
    """Measure bait visibility at layer l and recovery effectiveness at layer l+1."""
    metrics: dict[str, dict[str, float]] = {}
    layers = get_layers(model)

    for installed in controller.installed_layers:
        bait_layer = layers[installed.bait_layer_idx]
        recovery_layer = layers[installed.recovery_layer_idx]

        # Monitor: bait injection point (down_proj of layer l)
        #          recovery point (o_proj of layer l+1)
        modules = {
            "bait_down_proj": bait_layer.mlp.down_proj,
            "recovery_o_proj": get_layer_module(recovery_layer, "attn.o_proj"),
            "recovery_down_proj": recovery_layer.mlp.down_proj,
        }

        clean_deltas = {name: [] for name in modules}
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            inputs = prepare_batch(model, tokenizer, batch)

            with controller.adapters_disabled():
                with capture_last_token_outputs(modules) as clean_cache:
                    with torch.no_grad():
                        model(**inputs, return_dict=True)

            with capture_last_token_outputs(modules) as defended_cache:
                with torch.no_grad():
                    model(**inputs, return_dict=True)

            for name in modules:
                clean_tensor = torch.cat(clean_cache[name], dim=0)
                defended_tensor = torch.cat(defended_cache[name], dim=0)
                rel = (
                    (defended_tensor - clean_tensor).norm(dim=-1)
                    / clean_tensor.norm(dim=-1).clamp_min(1e-8)
                ).mean()
                clean_deltas[name].append(float(rel.item()))

        key = f"bait_L{installed.bait_layer_idx}_recovery_L{installed.recovery_layer_idx}"
        metrics[key] = {
            "bait_down_proj_shift": float(
                sum(clean_deltas["bait_down_proj"]) / max(len(clean_deltas["bait_down_proj"]), 1)
            ),
            "recovery_o_proj_shift": float(
                sum(clean_deltas["recovery_o_proj"]) / max(len(clean_deltas["recovery_o_proj"]), 1)
            ),
            "recovery_down_proj_shift": float(
                sum(clean_deltas["recovery_down_proj"]) / max(len(clean_deltas["recovery_down_proj"]), 1)
            ),
        }

    return metrics


def transport_alignment_loss(
    clean_cache: dict[str, list[torch.Tensor]],
    defended_cache: dict[str, list[torch.Tensor]],
    max_layer_weight: float = 0.0,
    return_per_hook: bool = False,
) -> torch.Tensor:
    """Relative squared residual shift used for recovery training.

    Raw hidden-state MSE is dominated by residual-stream scale and produced
    large, hard-to-read losses.  The actual recovery target is simpler:
    ||defended - clean|| / ||clean|| should be small at residual_{l+2}.

    Multi-layer aggregation:
      When many recovery hooks are watched simultaneously, a plain mean lets
      already-converged layers dilute the gradient of still-uncanceled ones
      (Plan D). ``max_layer_weight`` adds ``w · max_over_hooks`` on top of the
      mean so the worst-performing hook always receives gradient signal.
      ``max_layer_weight = 0`` reproduces the original mean-only behavior.
    """
    losses = []
    for name in clean_cache:
        defended_tensor = torch.cat(defended_cache[name], dim=0).float()
        clean_tensor = torch.cat(clean_cache[name], dim=0).to(
            device=defended_tensor.device,
            dtype=torch.float32,
        )
        diff_sq = (defended_tensor - clean_tensor).square().sum(dim=-1)
        clean_sq = clean_tensor.square().sum(dim=-1).clamp_min(1e-8)
        losses.append((diff_sq / clean_sq).mean())
    stacked = torch.stack(losses)
    mean_loss = stacked.mean()
    if max_layer_weight > 0.0 and stacked.numel() > 1:
        loss = mean_loss + max_layer_weight * stacked.max()
    else:
        loss = mean_loss
    if return_per_hook:
        per_hook = {name: losses[i].detach() for i, name in enumerate(clean_cache)}
        return loss, per_hook
    return loss


def mean_relative_shift(
    clean_cache: dict[str, list[torch.Tensor]],
    defended_cache: dict[str, list[torch.Tensor]],
) -> torch.Tensor:
    shifts = []
    for name in clean_cache:
        defended_tensor = torch.cat(defended_cache[name], dim=0).float()
        clean_tensor = torch.cat(clean_cache[name], dim=0).to(
            device=defended_tensor.device,
            dtype=torch.float32,
        )
        diff_sq = (defended_tensor - clean_tensor).square().sum(dim=-1)
        clean_sq = clean_tensor.square().sum(dim=-1).clamp_min(1e-8)
        shifts.append(torch.sqrt((diff_sq / clean_sq).clamp_min(0.0) + 1e-12).mean())
    return torch.stack(shifts).mean()


def set_installed_pair_enabled(
    model: nn.Module,
    installed: InstalledLayerDefense,
    enabled: bool,
) -> None:
    layers = get_layers(model)
    bait_layer = layers[installed.bait_layer_idx]
    recovery_layer = layers[installed.recovery_layer_idx]
    if isinstance(bait_layer.mlp.down_proj, LinearWithAdapters):
        bait_layer.mlp.down_proj.adapters_enabled = enabled
    recovery_o_proj = get_layer_module(recovery_layer, "attn.o_proj")
    if isinstance(recovery_o_proj, LinearWithAdapters):
        recovery_o_proj.adapters_enabled = enabled
    if isinstance(recovery_layer.mlp.down_proj, LinearWithAdapters):
        recovery_layer.mlp.down_proj.adapters_enabled = enabled


def calibrate_bait_gates(
    model: nn.Module,
    tokenizer,
    controller: DefenseController,
    prompts: list[Prompt],
    batch_size: int,
    target_shift: float,
    quantile: float,
    log_min: float,
    log_max: float,
    kl_target: float = 0.0,
    kl_adjust_iters: int = 2,
    max_prompts: int = 64,
) -> None:
    """Data-dependent bait strength calibration.

    The same nominal bait_scale can produce very different residual shifts on
    different datasets.  Measure each pair's actual down_proj relative shift and
    set its scalar gate so training starts near a small target perturbation.
    """
    if target_shift <= 0.0 or not prompts:
        return
    quantile = min(max(quantile, 0.0), 1.0)

    layers = get_layers(model)
    calibration_prompts = prompts[: min(max_prompts, len(prompts))]
    previous_enabled = [wrapper.adapters_enabled for wrapper in controller.wrappers]

    print(
        f"\nCalibrating bait gates on {len(calibration_prompts)} prompts "
        f"(target q{quantile:.2f} visible shift={target_shift:.4f}"
        + (f", pair KL≤{kl_target:.5f}" if kl_target > 0.0 else "")
        + ")..."
    )
    try:
        for idx, installed in enumerate(controller.installed_layers):
            bait_layer = layers[installed.bait_layer_idx]
            module = bait_layer.mlp.down_proj
            rel_chunks = []

            def measure_pair_kl() -> float:
                losses = []
                for kl_start in range(0, len(calibration_prompts), batch_size):
                    kl_batch = calibration_prompts[kl_start : kl_start + batch_size]
                    inputs = prepare_batch(model, tokenizer, kl_batch)
                    controller.set_enabled(False)
                    with torch.no_grad():
                        reference_logits = model(**inputs, return_dict=True).logits

                    controller.set_enabled(False)
                    set_installed_pair_enabled(model, installed, True)
                    with torch.no_grad():
                        defended_logits = model(**inputs, return_dict=True).logits
                    set_installed_pair_enabled(model, installed, False)

                    losses.append(
                        masked_logit_kl(
                            defended_logits,
                            reference_logits,
                            inputs["attention_mask"],
                        ).item()
                    )
                return float(sum(losses) / max(len(losses), 1))

            for start in range(0, len(calibration_prompts), batch_size):
                batch = calibration_prompts[start : start + batch_size]
                inputs = prepare_batch(model, tokenizer, batch)

                controller.set_enabled(False)
                with capture_last_token_outputs({"bait": module}, detach=True, cpu=False) as clean_cache:
                    with torch.no_grad():
                        model(**inputs, return_dict=True)

                controller.set_enabled(False)
                set_installed_pair_enabled(model, installed, True)
                with capture_last_token_outputs({"bait": module}, detach=True, cpu=False) as defended_cache:
                    with torch.no_grad():
                        model(**inputs, return_dict=True)

                clean_tensor = torch.cat(clean_cache["bait"], dim=0)
                defended_tensor = torch.cat(defended_cache["bait"], dim=0).to(clean_tensor)
                diff_sq = (defended_tensor - clean_tensor).square().sum(dim=-1)
                clean_sq = clean_tensor.square().sum(dim=-1).clamp_min(1e-8)
                rel = torch.sqrt((diff_sq / clean_sq).clamp_min(0.0) + 1e-12)
                rel_chunks.append(rel.detach().float().cpu())

            rel_all = torch.cat(rel_chunks) if rel_chunks else torch.empty(0)
            current_shift = float(torch.quantile(rel_all, quantile).item()) if rel_all.numel() else 0.0
            bait_module = controller.bait_modules[idx]
            old_log = float(bait_module.bait_gate.detach().cpu().item())
            if not math.isfinite(current_shift) or current_shift <= 0.0:
                new_log = log_min
                reason = "non-finite/zero shift"
            else:
                new_log = old_log + math.log(target_shift / current_shift)
                new_log = min(max(new_log, log_min), log_max)
                reason = "measured shift"
            with torch.no_grad():
                bait_module.bait_gate.fill_(new_log)

            measured_kl = None
            if kl_target > 0.0:
                for _ in range(max(1, kl_adjust_iters)):
                    measured_kl = measure_pair_kl()
                    if (
                        not math.isfinite(measured_kl)
                        or measured_kl <= kl_target
                        or measured_kl <= 0.0
                    ):
                        break
                    current_log = float(bait_module.bait_gate.detach().cpu().item())
                    kl_scaled_log = current_log + 0.5 * math.log(kl_target / measured_kl)
                    kl_scaled_log = min(max(kl_scaled_log, log_min), log_max)
                    if kl_scaled_log >= current_log - 1e-6:
                        break
                    with torch.no_grad():
                        bait_module.bait_gate.fill_(kl_scaled_log)
                    new_log = kl_scaled_log
                if measured_kl is not None and math.isfinite(measured_kl) and measured_kl > kl_target:
                    measured_kl = measure_pair_kl()

            new_gate = float(bait_module.gate(detach=True).cpu().item())
            print(
                f"  L{installed.bait_layer_idx}: shift={current_shift:.5f} "
                f"log_gate {old_log:.3f} → {new_log:.3f} "
                f"(gate={new_gate:.5f}, {reason}"
                + (f", pair_kl={measured_kl:.6f}" if measured_kl is not None else "")
                + ")"
            )
    finally:
        for wrapper, enabled in zip(controller.wrappers, previous_enabled):
            wrapper.adapters_enabled = enabled


def visible_bait_loss(
    clean_cache: dict[str, list[torch.Tensor]],
    defended_cache: dict[str, list[torch.Tensor]],
    floor: float,
    ceiling: float = 0.0,
    max_layer_weight: float = 0.0,
    return_per_hook: bool = False,
) -> torch.Tensor:
    """Band hinge loss for visible bait strength.

    Old design used MSE toward target, which caused the optimizer to suppress
    already-adequate bait signal (visible_shift fell from 0.08 → 0.059 over
    training because the optimizer was shrinking bait to hit the target).
    The floor hinge preserves that fix: if shift >= floor, weak-bait loss is 0.
    An optional ceiling hinge prevents refusal_cancel bait from becoming too
    strong and overshooting the clean refusal separator in the opposite direction.

    Multi-layer aggregation:
      Mean over hooks hides a single dying bait when other hooks still meet
      the floor (Plan D). ``max_layer_weight`` adds ``w · max_over_hooks`` so
      the weakest bait signal always dominates. ``max_layer_weight = 0``
      reproduces the original mean-only behavior.
    """
    per_hook = {}
    hinges = []
    for name in clean_cache:
        defended_tensor = torch.cat(defended_cache[name], dim=0).float()
        clean_tensor = torch.cat(clean_cache[name], dim=0).to(
            device=defended_tensor.device,
            dtype=torch.float32,
        )
        diff_sq = (defended_tensor - clean_tensor).square().sum(dim=-1)
        clean_sq = clean_tensor.square().sum(dim=-1).clamp_min(1e-8)
        rel_sq = diff_sq / clean_sq
        rel_sq_mean = rel_sq.mean()
        rel_shift = torch.sqrt(rel_sq_mean.clamp_min(0.0) + 1e-12)
        floor_tensor = torch.tensor(floor, device=rel_shift.device)
        lower_hinge = F.relu(floor_tensor.square() - rel_sq_mean).square()
        if ceiling > 0.0:
            ceiling_tensor = torch.tensor(ceiling, device=rel_shift.device)
            upper_hinge = F.relu(rel_sq_mean - ceiling_tensor.square()).square()
        else:
            upper_hinge = torch.zeros_like(lower_hinge)
        hinge = lower_hinge + upper_hinge
        hinges.append(hinge)
        if return_per_hook:
            per_hook[name] = {
                "rel_shift": rel_shift.detach(),
                "lower_hinge": lower_hinge.detach(),
                "upper_hinge": upper_hinge.detach(),
                "hinge": hinge.detach(),
            }
    stacked = torch.stack(hinges)
    mean_loss = stacked.mean()
    if max_layer_weight > 0.0 and stacked.numel() > 1:
        loss = mean_loss + max_layer_weight * stacked.max()
    else:
        loss = mean_loss
    if return_per_hook:
        return loss, per_hook
    return loss


def fisher_poison_loss(
    model: nn.Module,
    tokenizer,
    controller: DefenseController,
    good_prompts: list[Prompt],
    bad_prompts: list[Prompt],
    batch_size: int,
    supervised_residual_dirs: dict[int, torch.Tensor] | None,
) -> torch.Tensor:
    """Penalize good/bad separability inside the bait delta itself.

    This directly minimizes a differentiable Fisher ratio:

        ||mean(δ_bad) - mean(δ_good)||^2 / (var(δ_good) + var(δ_bad))

    which matches the reported ``delta_fisher_ratio`` diagnostic.
    """
    del supervised_residual_dirs
    if not controller.bait_modules:
        return torch.tensor(0.0, device=next(model.parameters()).device)

    losses = []
    layers = get_layers(model)

    for idx, installed in enumerate(controller.installed_layers):
        bait_module = controller.bait_modules[idx]
        bait_layer = layers[installed.bait_layer_idx]
        dp_wrapper = bait_layer.mlp.down_proj
        base_dp = dp_wrapper.base_linear if isinstance(dp_wrapper, LinearWithAdapters) else dp_wrapper

        def collect_deltas(prompts):
            """Collect bait deltas over prompts WITH gradients."""
            batch = prompts[: min(batch_size, len(prompts))]
            inputs = prepare_batch(model, tokenizer, batch)
            captured = []

            def hook(m, inp, out):
                x = inp[0] if isinstance(inp, tuple) else inp
                captured.append(x[:, -1, :].float())

            handle = base_dp.register_forward_hook(hook)
            try:
                model(**inputs, return_dict=True)
            finally:
                handle.remove()

            if not captured:
                return None
            h = captured[0].to(bait_module.coeff.weight.device)
            delta = bait_module(h)  # (batch, hidden_dim) — WITH grad
            return delta.float()

        good_delta = collect_deltas(good_prompts)
        bad_delta = collect_deltas(bad_prompts)
        if good_delta is None or bad_delta is None:
            continue

        mu_good = good_delta.mean(dim=0)
        mu_bad = bad_delta.mean(dim=0)
        between = (mu_bad - mu_good).pow(2).sum()
        good_var = (good_delta - mu_good).pow(2).sum(dim=-1).mean()
        bad_var = (bad_delta - mu_bad).pow(2).sum(dim=-1).mean()
        losses.append(between / (good_var + bad_var).clamp_min(1e-8))

    if not losses:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return torch.stack(losses).mean()


def residual_fisher_suppression_loss(
    model: nn.Module,
    tokenizer,
    controller: DefenseController,
    good_prompts: list[Prompt],
    bad_prompts: list[Prompt],
    batch_size: int,
    supervised_residual_dirs: dict[int, torch.Tensor] | None = None,
    active_installed_layers: list[InstalledLayerDefense] | None = None,
    denom_floor: float = 1e-3,
    loss_mode: str = "detached_gap",
    target_space: str = "projected",
) -> torch.Tensor:
    """Minimize Fisher discriminability at the defended bait residual.

    Unlike ``fisher_poison_loss`` (which targets δ(x) separability and makes δ
    class-blind, thereby *preserving* residual Fisher), this loss directly
    minimizes the Fisher ratio of the full defended residual at position l+1.

    This forces the bait δ(x) to be class-dependent in a specific way: it must
    push good and bad residuals closer together, actively destroying the
    good/bad separation that Heretic relies on for refusal-direction extraction.
    """
    layers = get_layers(model)
    losses = []

    monitored_installed_layers = (
        active_installed_layers
        if active_installed_layers is not None
        else controller.installed_layers
    )

    for installed in monitored_installed_layers:
        bait_layer = layers[installed.bait_layer_idx]

        def collect_residuals(prompts: list[Prompt]) -> torch.Tensor | None:
            batch = prompts[: min(batch_size, len(prompts))]
            inputs = prepare_batch(model, tokenizer, batch)
            captured: list[torch.Tensor] = []

            def hook(module, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                captured.append(o[:, -1, :].float())

            handle = bait_layer.register_forward_hook(hook)
            try:
                model(**inputs, return_dict=True)
            finally:
                handle.remove()

            if not captured:
                return None
            return captured[0]  # (batch, hidden_dim) — WITH grad

        good_res = collect_residuals(good_prompts)
        bad_res = collect_residuals(bad_prompts)
        if good_res is None or bad_res is None:
            continue

        if (
            target_space == "projected"
            and supervised_residual_dirs is not None
            and installed.bait_layer_idx in supervised_residual_dirs
        ):
            r_clean = supervised_residual_dirs[installed.bait_layer_idx].float().to(good_res.device)
            r_clean = r_clean / r_clean.norm().clamp_min(1e-8)
            good_proj = torch.matmul(good_res, r_clean)
            bad_proj = torch.matmul(bad_res, r_clean)
        else:
            good_proj = good_res
            bad_proj = bad_res

        mu_good = good_proj.mean(dim=0)
        mu_bad = bad_proj.mean(dim=0)
        
        if good_proj.dim() == 1:
            between = (mu_bad - mu_good).pow(2)
            good_var = (good_proj - mu_good).pow(2).mean()
            bad_var = (bad_proj - mu_bad).pow(2).mean()
        else:
            between = (mu_bad - mu_good).pow(2).sum()
            good_var = (good_proj - mu_good).pow(2).sum(dim=-1).mean()
            bad_var = (bad_proj - mu_bad).pow(2).sum(dim=-1).mean()

        if loss_mode == "true_ratio":
            # Legacy behavior: full Fisher ratio gradients. This can create
            # over-compensation, but it reproduces older checkpoints more closely.
            denom = (good_var + bad_var).clamp_min(denom_floor)
        else:
            # Stable behavior: normalize by the current scale, but only the
            # class-mean gap drives gradients.
            denom = (good_var + bad_var).detach().clamp_min(denom_floor)
        fisher = between / denom
        losses.append(fisher)

    if not losses:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return torch.stack(losses).mean()


def residual_direction_orthogonal_loss(
    model: nn.Module,
    tokenizer,
    controller: DefenseController,
    good_prompts: list[Prompt],
    bad_prompts: list[Prompt],
    batch_size: int,
    supervised_residual_dirs: dict[int, torch.Tensor] | None = None,
    active_installed_layers: list[InstalledLayerDefense] | None = None,
) -> torch.Tensor:
    """Penalize Heretic-visible residual gaps that still point along r_clean."""
    if supervised_residual_dirs is None:
        return torch.tensor(0.0, device=next(model.parameters()).device)

    layers = get_layers(model)
    losses = []
    monitored_installed_layers = (
        active_installed_layers
        if active_installed_layers is not None
        else controller.installed_layers
    )

    for installed in monitored_installed_layers:
        if installed.bait_layer_idx not in supervised_residual_dirs:
            continue
        bait_layer = layers[installed.bait_layer_idx]

        def collect_residuals(prompts: list[Prompt]) -> torch.Tensor | None:
            batch = prompts[: min(batch_size, len(prompts))]
            inputs = prepare_batch(model, tokenizer, batch)
            captured: list[torch.Tensor] = []

            def hook(module, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                captured.append(o[:, -1, :].float())

            handle = bait_layer.register_forward_hook(hook)
            try:
                model(**inputs, return_dict=True)
            finally:
                handle.remove()

            if not captured:
                return None
            return captured[0]

        good_res = collect_residuals(good_prompts)
        bad_res = collect_residuals(bad_prompts)
        if good_res is None or bad_res is None:
            continue

        r_clean = supervised_residual_dirs[installed.bait_layer_idx].float().to(good_res.device)
        r_clean = r_clean / r_clean.norm().clamp_min(1e-8)
        defended_gap = bad_res.mean(dim=0) - good_res.mean(dim=0)
        defended_gap_norm = defended_gap.norm().clamp_min(1e-8)
        cos = torch.dot(defended_gap.float(), r_clean) / defended_gap_norm
        losses.append(cos.square())

    if not losses:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return torch.stack(losses).mean()


def bait_basis_anti_coherence_loss(
    controller: DefenseController,
    components: int = 2,
    margin: float = 0.0,
    external_basis_refs: list[dict] | None = None,
) -> torch.Tensor:
    """Penalize cross-layer bait basis directions that form one global shortcut.

    Progressive isolated stages only train one bait/recovery pair at a time, so
    this loss deliberately compares the current trainable bait basis against all
    installed bait modules rather than just the active stage hooks.
    """
    if not controller.bait_modules or components <= 0:
        device = (
            controller.bait_modules[0].basis.weight.device
            if controller.bait_modules
            else torch.device("cpu")
        )
        return torch.tensor(0.0, device=device)

    vectors: list[tuple[int | None, torch.Tensor]] = []
    device = controller.bait_modules[0].basis.weight.device
    for module_idx, bait_module in enumerate(controller.bait_modules):
        basis = bait_module.basis.weight.float()
        layer_idx = (
            controller.installed_layers[module_idx].bait_layer_idx
            if module_idx < len(controller.installed_layers)
            else module_idx
        )
        width = min(int(components), basis.shape[1])
        for col_idx in range(width):
            vec = F.normalize(basis[:, col_idx], dim=0, eps=1e-8)
            vectors.append((layer_idx, vec))

    for ref in external_basis_refs or []:
        raw_vector = ref.get("vector")
        if not raw_vector:
            continue
        ref_vec = torch.tensor(raw_vector, device=device, dtype=torch.float32)
        if ref_vec.numel() != vectors[0][1].numel():
            continue
        ref_vec = F.normalize(ref_vec, dim=0, eps=1e-8)
        vectors.append((ref.get("bait_layer"), ref_vec))

    if len(vectors) < 2:
        return torch.tensor(0.0, device=device)

    losses = []
    allowed = max(0.0, float(margin))
    for i, (layer_i, vec_i) in enumerate(vectors):
        for layer_j, vec_j in vectors[i + 1 :]:
            if layer_i is not None and layer_i == layer_j:
                continue
            if not vec_i.requires_grad and not vec_j.requires_grad:
                continue
            cosine_abs = torch.dot(vec_i, vec_j).abs()
            if allowed > 0.0:
                losses.append(F.relu(cosine_abs - allowed).square())
            else:
                losses.append(cosine_abs.square())

    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def load_bait_direction_refs(path: Path | None) -> dict[str, list[dict]]:
    """Load external phase refs exported by build_bait_direction_refs()."""
    if path is None:
        return {"basis": [], "residual_gap": []}
    if not path.exists():
        print(f"  [yellow]WARNING[/]: external bait ref path not found: {path}")
        return {"basis": [], "residual_gap": []}

    with open(path, "r") as f:
        payload = json.load(f)
    refs = payload.get("bait_direction_refs", payload)
    basis_refs = refs.get("basis", refs.get("basis_components", []))
    gap_refs = refs.get("residual_gap", refs.get("residual_gap_directions", []))
    if not isinstance(basis_refs, list):
        basis_refs = []
    if not isinstance(gap_refs, list):
        gap_refs = []
    print(
        f"  Loaded external bait refs from {path}: "
        f"basis={len(basis_refs)}, residual_gap={len(gap_refs)}"
    )
    return {"basis": basis_refs, "residual_gap": gap_refs}


def capture_bait_layer_residuals(
    model: nn.Module,
    tokenizer,
    prompts: list[Prompt],
    bait_layer_indices: list[int],
    batch_size: int,
    *,
    detach: bool,
    max_prompts: int | None = None,
) -> dict[int, torch.Tensor]:
    """Capture last-token outputs of bait layers (residual l+1)."""
    layers = get_layers(model)
    selected_prompts = prompts[: max_prompts or len(prompts)]
    caches: dict[int, list[torch.Tensor]] = {idx: [] for idx in bait_layer_indices}
    handles = []

    for layer_idx in bait_layer_indices:
        layer = layers[layer_idx]

        def hook(module, inp, out, *, _layer_idx=layer_idx):
            value = out[0] if isinstance(out, tuple) else out
            value = value[:, -1, :].float()
            if detach:
                value = value.detach().cpu()
            caches[_layer_idx].append(value)

        handles.append(layer.register_forward_hook(hook))

    try:
        for start in range(0, len(selected_prompts), batch_size):
            batch = selected_prompts[start : start + batch_size]
            if not batch:
                continue
            inputs = prepare_batch(model, tokenizer, batch)
            model(**inputs, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()

    return {
        layer_idx: torch.cat(values, dim=0)
        for layer_idx, values in caches.items()
        if values
    }


def build_bait_direction_refs(
    model: nn.Module,
    tokenizer,
    controller: DefenseController,
    good_prompts: list[Prompt],
    bad_prompts: list[Prompt],
    batch_size: int,
    components: int = 2,
    max_prompts: int = 64,
) -> dict:
    """Export final basis and defended residual-gap directions for later phases."""
    basis_refs = []
    for module_idx, bait_module in enumerate(controller.bait_modules):
        if module_idx >= len(controller.installed_layers):
            continue
        installed = controller.installed_layers[module_idx]
        basis = bait_module.basis.weight.detach().float().cpu()
        width = min(max(1, int(components)), basis.shape[1])
        for col_idx in range(width):
            vec = F.normalize(basis[:, col_idx], dim=0, eps=1e-8)
            basis_refs.append(
                {
                    "bait_layer": installed.bait_layer_idx,
                    "recovery_layer": installed.recovery_layer_idx,
                    "component": col_idx,
                    "vector": vec.tolist(),
                }
            )

    residual_gap_refs = []
    if good_prompts and bad_prompts and controller.installed_layers:
        bait_layers = [item.bait_layer_idx for item in controller.installed_layers]
        with torch.no_grad():
            good_res = capture_bait_layer_residuals(
                model, tokenizer, good_prompts, bait_layers, batch_size,
                detach=True, max_prompts=max_prompts,
            )
            bad_res = capture_bait_layer_residuals(
                model, tokenizer, bad_prompts, bait_layers, batch_size,
                detach=True, max_prompts=max_prompts,
            )
        for installed in controller.installed_layers:
            good = good_res.get(installed.bait_layer_idx)
            bad = bad_res.get(installed.bait_layer_idx)
            if good is None or bad is None:
                continue
            gap = bad.float().mean(dim=0) - good.float().mean(dim=0)
            if gap.norm().item() <= 1e-8:
                continue
            gap = F.normalize(gap, dim=0, eps=1e-8)
            residual_gap_refs.append(
                {
                    "bait_layer": installed.bait_layer_idx,
                    "residual_idx": installed.bait_layer_idx + 1,
                    "vector": gap.cpu().tolist(),
                }
            )

    return {
        "version": 1,
        "basis_components": int(components),
        "basis": basis_refs,
        "residual_gap": residual_gap_refs,
    }


def residual_gap_anti_coherence_loss(
    model: nn.Module,
    tokenizer,
    controller: DefenseController,
    good_prompts: list[Prompt],
    bad_prompts: list[Prompt],
    batch_size: int,
    active_installed_layers: list[InstalledLayerDefense] | None = None,
    external_residual_gap_refs: list[dict] | None = None,
    margin: float = 0.0,
) -> torch.Tensor:
    """Decorrelate actual defended good/bad residual-gap directions across layers."""
    monitored_installed_layers = (
        active_installed_layers
        if active_installed_layers is not None
        else controller.installed_layers
    )
    if not monitored_installed_layers or not good_prompts or not bad_prompts:
        return torch.tensor(0.0, device=next(model.parameters()).device)

    bait_layers = sorted({item.bait_layer_idx for item in monitored_installed_layers})
    good_res = capture_bait_layer_residuals(
        model, tokenizer, good_prompts, bait_layers, batch_size,
        detach=False, max_prompts=batch_size,
    )
    bad_res = capture_bait_layer_residuals(
        model, tokenizer, bad_prompts, bait_layers, batch_size,
        detach=False, max_prompts=batch_size,
    )

    device = get_input_device(model)
    vectors: list[tuple[int | None, torch.Tensor]] = []
    for installed in monitored_installed_layers:
        good = good_res.get(installed.bait_layer_idx)
        bad = bad_res.get(installed.bait_layer_idx)
        if good is None or bad is None:
            continue
        gap = bad.float().mean(dim=0) - good.float().mean(dim=0)
        gap = gap.to(device)
        vectors.append((installed.bait_layer_idx, F.normalize(gap, dim=0, eps=1e-8)))

    current_width = vectors[0][1].numel() if vectors else None
    for ref in external_residual_gap_refs or []:
        raw_vector = ref.get("vector")
        if not raw_vector:
            continue
        ref_vec = torch.tensor(raw_vector, device=device, dtype=torch.float32)
        if current_width is not None and ref_vec.numel() != current_width:
            continue
        ref_vec = F.normalize(ref_vec, dim=0, eps=1e-8)
        vectors.append((ref.get("bait_layer"), ref_vec))

    if len(vectors) < 2:
        return torch.tensor(0.0, device=device)

    losses = []
    allowed = max(0.0, float(margin))
    for i, (layer_i, vec_i) in enumerate(vectors):
        for layer_j, vec_j in vectors[i + 1 :]:
            if layer_i is not None and layer_i == layer_j:
                continue
            if not vec_i.requires_grad and not vec_j.requires_grad:
                continue
            cosine_abs = torch.dot(vec_i, vec_j).abs()
            if allowed > 0.0:
                losses.append(F.relu(cosine_abs - allowed).square())
            else:
                losses.append(cosine_abs.square())

    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def _interpolate_direction_refs(
    direction_refs: dict[int, torch.Tensor],
    direction_index: float,
    device: torch.device,
) -> torch.Tensor | None:
    """Interpolate clean residual directions in Heretic's global direction space."""
    if not direction_refs:
        return None
    keys = sorted(int(key) for key in direction_refs)
    if not keys:
        return None
    if direction_index <= keys[0]:
        return F.normalize(direction_refs[keys[0]].float().to(device), dim=0, eps=1e-8)
    if direction_index >= keys[-1]:
        return F.normalize(direction_refs[keys[-1]].float().to(device), dim=0, eps=1e-8)

    lower = keys[0]
    upper = keys[-1]
    for left, right in zip(keys, keys[1:]):
        if left <= direction_index <= right:
            lower = left
            upper = right
            break
    weight = (direction_index - lower) / max(float(upper - lower), 1e-8)
    left_vec = direction_refs[lower].float().to(device)
    right_vec = direction_refs[upper].float().to(device)
    return F.normalize(left_vec.lerp(right_vec, weight), dim=0, eps=1e-8)


def _hinged_cosine_square(cosine_abs: torch.Tensor, margin: float) -> torch.Tensor:
    allowed = max(0.0, float(margin))
    if allowed > 0.0:
        return F.relu(cosine_abs - allowed).square()
    return cosine_abs.square()


def heretic_global_direction_loss(
    model: nn.Module,
    tokenizer,
    controller: DefenseController,
    good_prompts: list[Prompt],
    bad_prompts: list[Prompt],
    batch_size: int,
    supervised_residual_dirs: dict[int, torch.Tensor] | None = None,
    active_installed_layers: list[InstalledLayerDefense] | None = None,
    external_residual_gap_refs: list[dict] | None = None,
    samples: int = 5,
    margin: float = 0.0,
    direction_range_low: float = 0.4,
    direction_range_high: float = 0.9,
) -> torch.Tensor:
    """Penalize residual gaps that remain vulnerable to Heretic's global mode.

    Heretic global mode picks one interpolated refusal direction_index and uses
    that same vector for all layers. Pairwise anti-coherence can still leave a
    strong aggregate direction, so this loss attacks that surface directly:
    current defended gaps are pushed away from sampled global clean directions
    and from the detached aggregate gap formed by current/external phase refs.
    """
    monitored_installed_layers = (
        active_installed_layers
        if active_installed_layers is not None
        else controller.installed_layers
    )
    if not monitored_installed_layers or not good_prompts or not bad_prompts:
        return torch.tensor(0.0, device=next(model.parameters()).device)

    bait_layers = sorted({item.bait_layer_idx for item in monitored_installed_layers})
    good_res = capture_bait_layer_residuals(
        model, tokenizer, good_prompts, bait_layers, batch_size,
        detach=False, max_prompts=batch_size,
    )
    bad_res = capture_bait_layer_residuals(
        model, tokenizer, bad_prompts, bait_layers, batch_size,
        detach=False, max_prompts=batch_size,
    )

    device = get_input_device(model)
    current_vectors: list[torch.Tensor] = []
    for installed in monitored_installed_layers:
        good = good_res.get(installed.bait_layer_idx)
        bad = bad_res.get(installed.bait_layer_idx)
        if good is None or bad is None:
            continue
        gap = (bad.float().mean(dim=0) - good.float().mean(dim=0)).to(device)
        current_vectors.append(F.normalize(gap, dim=0, eps=1e-8))

    if not current_vectors:
        return torch.tensor(0.0, device=device)

    candidates: list[torch.Tensor] = []
    if supervised_residual_dirs and samples > 0:
        last_layer_index = max(1, len(get_layers(model)) - 1)
        low = min(max(float(direction_range_low), 0.0), 1.0)
        high = min(max(float(direction_range_high), 0.0), 1.0)
        if high < low:
            low, high = high, low
        if samples == 1:
            direction_indices = [0.5 * (low + high) * last_layer_index]
        else:
            direction_indices = [
                (low + (high - low) * i / max(samples - 1, 1)) * last_layer_index
                for i in range(samples)
            ]
        for direction_index in direction_indices:
            candidate = _interpolate_direction_refs(
                supervised_residual_dirs,
                direction_index,
                device,
            )
            if candidate is not None and candidate.numel() == current_vectors[0].numel():
                candidates.append(candidate.detach())

    aggregate_vectors = [vec.detach() for vec in current_vectors]
    for ref in external_residual_gap_refs or []:
        raw_vector = ref.get("vector")
        if not raw_vector:
            continue
        ref_vec = torch.tensor(raw_vector, device=device, dtype=torch.float32)
        if ref_vec.numel() != current_vectors[0].numel():
            continue
        aggregate_vectors.append(F.normalize(ref_vec, dim=0, eps=1e-8))
    if len(aggregate_vectors) >= 2:
        aggregate = torch.stack(aggregate_vectors, dim=0).mean(dim=0)
        if aggregate.norm().item() > 1e-8:
            candidates.append(F.normalize(aggregate, dim=0, eps=1e-8).detach())

    if not candidates:
        return torch.tensor(0.0, device=device)

    losses = []
    for gap_vec in current_vectors:
        for candidate in candidates:
            cosine_abs = torch.dot(gap_vec, candidate).abs()
            losses.append(_hinged_cosine_square(cosine_abs, margin))
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)


def _safe_quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    q = min(max(float(q), 0.0), 1.0)
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    weight = pos - lo
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


def summarize_cross_layer_cosines(
    vectors: list[tuple[str, int | None, int | None, torch.Tensor]],
    threshold: float = 0.5,
    top_k: int = 10,
) -> dict:
    """Summarize cross-layer |cos| values for JSON diagnostics."""
    normalized: list[tuple[str, int | None, int | None, torch.Tensor]] = []
    for source, layer_idx, component, vector in vectors:
        vec = vector.detach().float().cpu()
        if vec.numel() == 0 or vec.norm().item() <= 1e-8:
            continue
        normalized.append((source, layer_idx, component, F.normalize(vec, dim=0, eps=1e-8)))

    values: list[float] = []
    top_pairs: list[dict] = []
    max_pair: dict | None = None
    for i, (src_i, layer_i, comp_i, vec_i) in enumerate(normalized):
        for src_j, layer_j, comp_j, vec_j in normalized[i + 1 :]:
            if layer_i is not None and layer_i == layer_j:
                continue
            if vec_i.numel() != vec_j.numel():
                continue
            cosine = abs(float(torch.dot(vec_i, vec_j).item()))
            values.append(cosine)
            pair = {
                "abs_cosine": cosine,
                "a": {"source": src_i, "layer": layer_i, "component": comp_i},
                "b": {"source": src_j, "layer": layer_j, "component": comp_j},
            }
            if max_pair is None or cosine > max_pair["abs_cosine"]:
                max_pair = pair
            top_pairs.append(pair)

    values.sort()
    top_pairs.sort(key=lambda item: item["abs_cosine"], reverse=True)
    return {
        "num_vectors": len(normalized),
        "num_pairs": len(values),
        "mean_abs_cosine": float(sum(values) / len(values)) if values else float("nan"),
        "median_abs_cosine": _safe_quantile(values, 0.5),
        "p90_abs_cosine": _safe_quantile(values, 0.9),
        "p99_abs_cosine": _safe_quantile(values, 0.99),
        "max_abs_cosine": max_pair["abs_cosine"] if max_pair else float("nan"),
        "count_above_threshold": sum(1 for value in values if value > threshold),
        "threshold": float(threshold),
        "max_pair": max_pair,
        "top_pairs": top_pairs[:top_k],
    }


def cross_layer_geometry_diagnostics(
    controller: DefenseController,
    defended_good_residuals: torch.Tensor,
    defended_bad_residuals: torch.Tensor,
    external_direction_refs: dict[str, list[dict]] | None = None,
    basis_components: int = 2,
) -> dict:
    """Return basis/gap cosine summaries so global shortcuts are visible in JSON."""
    gap_vectors: list[tuple[str, int | None, int | None, torch.Tensor]] = []
    for installed in controller.installed_layers:
        residual_idx = installed.bait_layer_idx + 1
        if residual_idx >= defended_good_residuals.shape[1]:
            continue
        good = defended_good_residuals[:, residual_idx, :].float()
        bad = defended_bad_residuals[:, residual_idx, :].float()
        gap = bad.mean(dim=0) - good.mean(dim=0)
        gap_vectors.append(("current", installed.bait_layer_idx, None, gap))

    external_gap_vectors: list[tuple[str, int | None, int | None, torch.Tensor]] = []
    for ref in (external_direction_refs or {}).get("residual_gap", []):
        raw_vector = ref.get("vector")
        if raw_vector:
            external_gap_vectors.append(
                (
                    "external",
                    ref.get("bait_layer"),
                    None,
                    torch.tensor(raw_vector, dtype=torch.float32),
                )
            )

    basis_vectors: list[tuple[str, int | None, int | None, torch.Tensor]] = []
    for module_idx, bait_module in enumerate(controller.bait_modules):
        if module_idx >= len(controller.installed_layers):
            continue
        installed = controller.installed_layers[module_idx]
        basis = bait_module.basis.weight.detach().float().cpu()
        width = min(max(1, int(basis_components)), basis.shape[1])
        for col_idx in range(width):
            basis_vectors.append(("current", installed.bait_layer_idx, col_idx, basis[:, col_idx]))
    for ref in (external_direction_refs or {}).get("basis", []):
        raw_vector = ref.get("vector")
        if raw_vector:
            basis_vectors.append(
                (
                    "external",
                    ref.get("bait_layer"),
                    ref.get("component"),
                    torch.tensor(raw_vector, dtype=torch.float32),
                )
            )

    return {
        "residual_gap_current": summarize_cross_layer_cosines(gap_vectors),
        "residual_gap_current_plus_external": summarize_cross_layer_cosines(
            gap_vectors + external_gap_vectors
        ),
        "bait_basis_current_plus_external": summarize_cross_layer_cosines(basis_vectors),
    }



def evaluate_per_layer_kl(
    model: nn.Module,
    tokenizer,
    controller: DefenseController,
    prompts: list[Prompt],
    batch_size: int,
) -> dict[str, float]:
    """Measure KL contribution of each bait/recovery pair in isolation.

    For each installed (bait_layer, recovery_layer) pair:
      1. Disable all pairs.
      2. Re-enable ONLY this pair.
      3. Measure KL(defended || clean).
    Returns a dict keyed by 'pair_LX_LY' → KL value.

    Helps identify which layer contributes the most KL and whether
    KL accumulates super-linearly with the number of pairs.
    """
    results: dict[str, float] = {}

    # Full model KL (all pairs enabled)
    results["all_pairs"] = evaluate_kl(model, tokenizer, controller, prompts, batch_size)

    for installed in controller.installed_layers:
        key = f"pair_L{installed.bait_layer_idx}_L{installed.recovery_layer_idx}"
        # Disable all
        controller.set_enabled(False)
        # Enable only this pair
        layers = get_layers(model)
        bait_layer = layers[installed.bait_layer_idx]
        recovery_layer = layers[installed.recovery_layer_idx]
        if isinstance(bait_layer.mlp.down_proj, LinearWithAdapters):
            bait_layer.mlp.down_proj.adapters_enabled = True
        recovery_o_proj = get_layer_module(recovery_layer, "attn.o_proj")
        if isinstance(recovery_o_proj, LinearWithAdapters):
            recovery_o_proj.adapters_enabled = True
        if isinstance(recovery_layer.mlp.down_proj, LinearWithAdapters):
            recovery_layer.mlp.down_proj.adapters_enabled = True

        kl = evaluate_kl(model, tokenizer, controller, prompts, batch_size)
        results[key] = kl

    # Restore all pairs
    controller.set_enabled(True)
    return results


def evaluate_cross_layer_interference(
    model: nn.Module,
    tokenizer,
    controller: DefenseController,
    good_prompts: list[Prompt],
    bad_prompts: list[Prompt],
    batch_size: int,
) -> dict[str, dict[str, float]]:
    """Measure whether shallow bait residuals contaminate deep recovery layers.

    For each installed pair (bait@L, recovery@L+1):
      - Hook the INPUT of the recovery layer (= residual_{L+1})
      - Compute cos(clean_direction, defended_direction) at that residual
      - If this value is far from 1 due to a DIFFERENT pair's bait, interference is present.

    Returns per-installed-pair dict with:
      - 'direction_cos': cos(clean, defended) at recovery layer input
      - 'fisher_clean' / 'fisher_defended': good/bad separability before/after
      - 'interference_pairs': list of other (bait_layer_idx) whose bait signal is present
    """
    from shared_utils import get_residuals_batched

    with controller.adapters_disabled():
        clean_good = get_residuals_batched(model, tokenizer, good_prompts, batch_size)
        clean_bad  = get_residuals_batched(model, tokenizer, bad_prompts, batch_size)

    defended_good = get_residuals_batched(model, tokenizer, good_prompts, batch_size)
    defended_bad  = get_residuals_batched(model, tokenizer, bad_prompts, batch_size)

    clean_dirs    = F.normalize(clean_bad.mean(0) - clean_good.mean(0), p=2, dim=1)
    defended_dirs = F.normalize(defended_bad.mean(0) - defended_good.mean(0), p=2, dim=1)

    results: dict[str, dict[str, float]] = {}
    for installed in controller.installed_layers:
        rec_res = installed.recovery_layer_idx + 1   # index into residual tensor
        bait_res = installed.bait_layer_idx + 1

        if rec_res >= clean_dirs.shape[0] or bait_res >= clean_dirs.shape[0]:
            continue

        cos_rec = float(
            F.cosine_similarity(
                clean_dirs[rec_res].unsqueeze(0),
                defended_dirs[rec_res].unsqueeze(0),
            ).item()
        )
        cos_bait = float(
            F.cosine_similarity(
                clean_dirs[bait_res].unsqueeze(0),
                defended_dirs[bait_res].unsqueeze(0),
            ).item()
        )
        fr_clean    = fisher_ratio(clean_good, clean_bad, rec_res)
        fr_defended = fisher_ratio(defended_good, defended_bad, rec_res)

        key = f"bait_L{installed.bait_layer_idx}_rec_L{installed.recovery_layer_idx}"
        results[key] = {
            "bait_residual_idx": bait_res,
            "recovery_residual_idx": rec_res,
            "cos_bait_direction": cos_bait,
            "dps_bait": 1.0 - cos_bait ** 2,
            "cos_recovery_direction": cos_rec,
            "dps_recovery": 1.0 - cos_rec ** 2,
            "fisher_clean_at_recovery": fr_clean,
            "fisher_defended_at_recovery": fr_defended,
            "fisher_delta_pct": 100.0 * (fr_defended - fr_clean) / max(fr_clean, 1e-8),
        }
    return results


def print_multilayer_scorecard(
    per_layer_kl: dict[str, float],
    interference: dict[str, dict[str, float]],
    installed_layers: list[InstalledLayerDefense],
) -> None:
    """Print multi-layer specific diagnostics."""
    print()

    # Per-pair KL contribution table
    kl_table = Table(title="Per-pair KL contribution (isolated, then all pairs)")
    kl_table.add_column("Pair", justify="left")
    kl_table.add_column("KL (isolated)", justify="right")
    kl_table.add_column("vs all-pairs KL", justify="right")
    kl_total = per_layer_kl.get("all_pairs", float("nan"))
    for k, v in per_layer_kl.items():
        if k == "all_pairs":
            continue
        ratio = v / max(kl_total, 1e-9)
        kl_table.add_row(k, f"{v:.6f}", f"{ratio:.2f}x")
    kl_table.add_row("ALL pairs", f"{kl_total:.6f}", "1.00x (reference)")
    print(kl_table)

    # Interference / isolation table
    if interference:
        itf_table = Table(
            title=(
                "Cross-layer interference: bait direction at bait residual & recovery residual\n"
                "  Bait DPS↑ = Heretic sees poisoned direction  |  "
                "Recovery DPS≈0 = poison contained, bait from other layers not leaking in"
            )
        )
        itf_table.add_column("Pair", justify="left")
        itf_table.add_column("Bait res", justify="right")
        itf_table.add_column("Bait DPS", justify="right")
        itf_table.add_column("Rec res", justify="right")
        itf_table.add_column("Rec DPS", justify="right")
        itf_table.add_column("Fisher Δ% at rec", justify="right")
        itf_table.add_column("Interference?", justify="left")
        for k, m in interference.items():
            bait_dps = m["dps_bait"]
            rec_dps  = m["dps_recovery"]
            fi_ok    = abs(m["fisher_delta_pct"]) < 15.0
            if rec_dps <= 0.05 and fi_ok:
                status = "✅ No interference"
            elif rec_dps <= 0.15:
                status = "⚠️ Mild leak"
            else:
                status = "❌ Interference detected"
            itf_table.add_row(
                k,
                str(m["bait_residual_idx"]),
                f"{bait_dps:.3f}",
                str(m["recovery_residual_idx"]),
                f"{rec_dps:.3f}",
                f"{m['fisher_delta_pct']:+.1f}%",
                status,
            )
        print(itf_table)


def train_recovery(
    model: nn.Module,
    tokenizer,
    controller: DefenseController,
    train_prompts: list[Prompt],
    args: argparse.Namespace,
    train_good_prompts: list[Prompt] | None = None,
    train_bad_prompts: list[Prompt] | None = None,
    supervised_residual_dirs: dict[int, torch.Tensor] | None = None,
    active_installed_layers: list[InstalledLayerDefense] | None = None,
    external_direction_refs: dict[str, list[dict]] | None = None,
) -> list[dict[str, float]]:
    """Train recovery adapters.

    If train_good_prompts / train_bad_prompts are provided they are interleaved
    with general-text batches so the recovery also learns to cancel the larger δ
    produced on harmful inputs without over-fitting on the eval split.

    During progressive isolated stages, ``active_installed_layers`` limits the
    visible/recovery hooks to the currently enabled pair. Disabled adapters then
    do not contribute zero-shift visible-floor constants to the loss.
    """
    trainable_params = controller.trainable_parameters()
    if not trainable_params:
        print("  [yellow]WARNING[/]: No trainable parameters — no defense installed. Skipping training.")
        return []
    optimizer = AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    # Cross-layer architecture:
    # visible_modules = bait injection point (layer l down_proj) — should show bait signal
    # recovery_modules = FULL OUTPUT of recovery layer (layer l+1) = residual_{l+2}
    #
    # Key change: previously we hooked o_proj and down_proj submodule outputs,
    # but those only see what each submodule ADDS to the residual.  The original δ
    # in residual_{l+1} also passes through the skip connection directly into
    # residual_{l+2} without touching either submodule.  By watching the full
    # layer output (= residual_{l+2}), the relative-shift loss sees the total
    # uncanceled δ and gives recovery adapters a correct gradient to cancel it.
    visible_modules = {}
    recovery_modules = {}
    layers = get_layers(model)
    monitored_installed_layers = (
        active_installed_layers
        if active_installed_layers is not None
        else controller.installed_layers
    )
    for installed in monitored_installed_layers:
        bait_layer = layers[installed.bait_layer_idx]
        recovery_layer = layers[installed.recovery_layer_idx]
        visible_modules[f"layer_{installed.bait_layer_idx}_down_proj"] = bait_layer.mlp.down_proj
        # Hook the full recovery layer: its output IS the residual stream at l+2.
        # capture_last_token_outputs handles tuple output via `output[0] if tuple`.
        recovery_modules[f"residual_{installed.recovery_layer_idx + 1}"] = recovery_layer

    # Build the combined training pool: general text + labelled prompts (if provided)
    combined_pool = train_prompts[:]
    if train_good_prompts:
        combined_pool.extend(train_good_prompts)
    if train_bad_prompts:
        combined_pool.extend(train_bad_prompts)

    rec_max_w = getattr(args, "recovery_max_weight", 0.0)
    vis_max_w = getattr(args, "visible_max_weight", 0.0)
    legacy_detach_local = getattr(args, "detach_local_loss_from_bait", None)
    if legacy_detach_local is None:
        detach_recovery_loss_from_bait = bool(
            getattr(args, "detach_recovery_loss_from_bait", True)
        )
        detach_visible_loss_from_bait = bool(
            getattr(args, "detach_visible_loss_from_bait", False)
        )
        detach_kl_loss_from_bait = bool(getattr(args, "detach_kl_loss_from_bait", True))
    else:
        detach_recovery_loss_from_bait = bool(legacy_detach_local)
        detach_visible_loss_from_bait = bool(legacy_detach_local)
        detach_kl_loss_from_bait = bool(legacy_detach_local)
    effective_kl_loss_weight = float(args.kl_loss_weight)
    effective_residual_fisher_loss_weight = float(args.residual_fisher_loss_weight)
    effective_residual_direction_loss_weight = float(args.residual_direction_loss_weight)
    effective_residual_gap_anti_coherence_loss_weight = float(
        args.residual_gap_anti_coherence_loss_weight
    )
    effective_global_direction_loss_weight = float(args.global_direction_loss_weight)
    bait_anti_coherence_loss_weight = float(args.bait_anti_coherence_loss_weight)
    external_basis_refs = (external_direction_refs or {}).get("basis", [])
    external_residual_gap_refs = (external_direction_refs or {}).get("residual_gap", [])
    has_residual_gap_anti_coherence_peers = (
        bool(external_residual_gap_refs)
        or active_installed_layers is None
        or len(active_installed_layers) > 1
    )

    def _zero_trainable_grads() -> None:
        for param in trainable_params:
            param.grad = None

    def _clear_bait_parameter_grads() -> int:
        """Drop local-loss gradients on bait params while preserving recovery grads."""
        cleared = 0
        for name, param in controller.trainable_named_parameters():
            if name.startswith("bait[") and param.grad is not None:
                param.grad = None
                cleared += 1
        return cleared

    def _gradient_norm_for_component(
        component_loss: torch.Tensor,
        label: str,
        *,
        clear_bait_grads: bool = False,
    ) -> float:
        if not component_loss.requires_grad:
            return 0.0
        _zero_trainable_grads()
        try:
            component_loss.backward(retain_graph=True)
        except RuntimeError as exc:
            print(
                f"  [yellow]WARNING[/]: loss-normalization probe for {label} "
                f"failed during backward; disabling this component. {exc}"
            )
            _zero_trainable_grads()
            return float("inf")

        if clear_bait_grads:
            _clear_bait_parameter_grads()

        total_sq = 0.0
        bad_names: list[str] = []
        for name, param in controller.trainable_named_parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach().float()
            if not torch.isfinite(grad).all():
                bad_names.append(name)
                continue
            total_sq += float(grad.square().sum().cpu().item())
        _zero_trainable_grads()

        if bad_names:
            print(
                f"  [yellow]WARNING[/]: loss-normalization probe for {label} "
                f"produced non-finite gradients in {', '.join(bad_names[:4])}; "
                "disabling this component."
            )
            return float("inf")
        return math.sqrt(total_sq)

    def _gradient_norm_for_components(
        components: list[tuple[torch.Tensor, str, bool]],
        label: str,
    ) -> float:
        _zero_trainable_grads()
        for component_loss, component_label, clear_bait_grads in components:
            if not component_loss.requires_grad:
                continue
            try:
                component_loss.backward(retain_graph=True)
            except RuntimeError as exc:
                print(
                    f"  [yellow]WARNING[/]: loss-normalization probe for "
                    f"{label}/{component_label} failed during backward; "
                    f"disabling this component. {exc}"
                )
                _zero_trainable_grads()
                return float("inf")
            if clear_bait_grads:
                _clear_bait_parameter_grads()

        total_sq = 0.0
        bad_names: list[str] = []
        for name, param in controller.trainable_named_parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach().float()
            if not torch.isfinite(grad).all():
                bad_names.append(name)
                continue
            total_sq += float(grad.square().sum().cpu().item())
        _zero_trainable_grads()

        if bad_names:
            print(
                f"  [yellow]WARNING[/]: loss-normalization probe for {label} "
                f"produced non-finite gradients in {', '.join(bad_names[:4])}; "
                "disabling this component."
            )
            return float("inf")
        return math.sqrt(total_sq)

    def _safe_downscaled_weight(raw_weight: float, component_grad: float, target_grad: float, label: str) -> float:
        if raw_weight <= 0.0:
            return 0.0
        if component_grad == 0.0:
            print(
                f"  Loss auto-normalization: {label} has zero gradient for the "
                "current trainable parameters; disabling this component."
            )
            return 0.0
        if not math.isfinite(component_grad):
            return 0.0
        scaled = min(raw_weight, target_grad / max(component_grad, 1e-12))
        if scaled < raw_weight:
            print(
                f"  Loss auto-normalization: {label} weight "
                f"{raw_weight:.6g} → {scaled:.6g} "
                f"(grad={component_grad:.4g}, target={target_grad:.4g})"
            )
        return max(0.0, scaled)

    def _freeze_unstable_bait_grads(named_params: list[tuple[str, nn.Parameter]]) -> bool:
        bad_names = nonfinite_gradient_names(named_params)
        if not bad_names or not all(name.startswith("bait[") for name in bad_names):
            return False

        bad_set = set(bad_names)
        for name, param in named_params:
            if name in bad_set:
                param.grad = None
                param.requires_grad_(False)
        print(
            "  [bold yellow]WARNING[/]: non-finite gradients were limited to "
            f"bait parameters ({', '.join(bad_names[:6])}); freezing those "
            "parameters and continuing with recovery-only updates."
        )
        return True

    def _diagnose_nonfinite_components(
        components: list[tuple[str, torch.Tensor, float]],
    ) -> None:
        print("  [bold yellow]Non-finite gradient component probe[/]:")
        for label, component_loss, weight in components:
            if weight <= 0.0:
                print(f"    {label}: skipped (weight=0)")
                continue
            value = float(component_loss.detach().float().cpu())
            if not component_loss.requires_grad:
                print(f"    {label}: value={value:.6g}, no grad path")
                continue

            _zero_trainable_grads()
            try:
                (weight * component_loss).backward(retain_graph=True)
            except RuntimeError as component_exc:
                print(
                    f"    {label}: value={value:.6g}, weight={weight:.6g}, "
                    f"backward failed: {component_exc}"
                )
                _zero_trainable_grads()
                continue

            clear_bait_for_label = (
                (label == "local_recovery" and detach_recovery_loss_from_bait)
                or (label == "local_visible" and detach_visible_loss_from_bait)
                or (label == "kl" and detach_kl_loss_from_bait)
            )
            cleared_bait_grads = 0
            if clear_bait_for_label:
                cleared_bait_grads = _clear_bait_parameter_grads()
            bad_names = nonfinite_gradient_names(controller.trainable_named_parameters())
            if bad_names:
                print(
                    f"    {label}: value={value:.6g}, weight={weight:.6g}, "
                    "NON-FINITE grads in "
                    + ", ".join(bad_names[:6])
                )
            else:
                total_sq = 0.0
                touched = 0
                for _, param in controller.trainable_named_parameters():
                    if param.grad is None:
                        continue
                    touched += 1
                    total_sq += float(param.grad.detach().float().square().sum().cpu().item())
                print(
                    f"    {label}: value={value:.6g}, weight={weight:.6g}, "
                    f"grad_norm={math.sqrt(total_sq):.6g}, touched={touched}"
                    + (
                        f", cleared_bait_grads={cleared_bait_grads}"
                        if cleared_bait_grads
                        else ""
                    )
                )
            _zero_trainable_grads()

    def _weighted_loss(component_loss: torch.Tensor, weight: float) -> torch.Tensor:
        """Return a weighted loss without connecting zero-weight terms to autograd."""
        if float(weight) <= 0.0:
            return component_loss.detach().new_zeros(())
        return float(weight) * component_loss

    def _append_weighted_backward_term(
        terms: list[tuple[torch.Tensor, bool]],
        component_loss: torch.Tensor,
        weight: float,
        clear_bait_grads: bool,
    ) -> None:
        if float(weight) <= 0.0:
            return
        term = float(weight) * component_loss
        if term.requires_grad:
            terms.append((term, clear_bait_grads))

    if getattr(args, "auto_normalize_loss_weights", True):
        probe_batch_size = getattr(args, "loss_normalization_batch_size", 0) or args.batch_size
        probe_batch = combined_pool[: min(probe_batch_size, len(combined_pool))]
        if probe_batch:
            inputs = prepare_batch(model, tokenizer, probe_batch)
            with controller.adapters_disabled():
                with capture_last_token_outputs(visible_modules, detach=True, cpu=False) as clean_visible_cache:
                    with capture_last_token_outputs(recovery_modules, detach=True, cpu=False) as clean_recovery_cache:
                        with torch.no_grad():
                            reference_logits = model(**inputs, return_dict=True).logits

            with capture_last_token_outputs(visible_modules, detach=False, cpu=False) as defended_visible_cache:
                with capture_last_token_outputs(recovery_modules, detach=False, cpu=False) as defended_recovery_cache:
                    defended_logits = model(**inputs, return_dict=True).logits

            probe_kl_loss = training_logit_kl(
                defended_logits,
                reference_logits,
                inputs["attention_mask"],
                mode=getattr(args, "kl_loss_mode", "full"),
                top_k=getattr(args, "kl_top_k", 64),
                temperature=getattr(args, "kl_temperature", 1.0),
                logit_clamp=getattr(args, "kl_logit_clamp", 0.0),
            )
            probe_recovery_loss, _ = transport_alignment_loss(
                clean_recovery_cache,
                defended_recovery_cache,
                max_layer_weight=rec_max_w,
                return_per_hook=True,
            )
            probe_visible_loss, _ = visible_bait_loss(
                clean_visible_cache,
                defended_visible_cache,
                floor=args.visible_shift_target,
                ceiling=getattr(args, "visible_shift_max", 0.0),
                max_layer_weight=vis_max_w,
                return_per_hook=True,
            )
            local_probe_components: list[tuple[torch.Tensor, str, bool]] = []
            if float(args.recovery_loss_weight) > 0.0:
                local_probe_components.append(
                    (
                        float(args.recovery_loss_weight) * probe_recovery_loss,
                        "recovery",
                        detach_recovery_loss_from_bait,
                    )
                )
            if float(args.visible_loss_weight) > 0.0:
                local_probe_components.append(
                    (
                        float(args.visible_loss_weight) * probe_visible_loss,
                        "visible",
                        detach_visible_loss_from_bait,
                    )
                )
            if local_probe_components:
                local_grad = _gradient_norm_for_components(
                    local_probe_components,
                    "local recovery/visible",
                )
                if not math.isfinite(local_grad):
                    print(
                        "  [yellow]WARNING[/]: local recovery/visible probe produced "
                        "non-finite gradients; global KL/RF/geometry terms will be disabled "
                        "for this stage."
                    )
                    target_grad = 0.0
                else:
                    target_grad = max(local_grad, 1e-12) * float(args.loss_normalization_target_ratio)
            else:
                print(
                    "  Loss auto-normalization: no local recovery/visible terms "
                    "active; keeping non-local weights unchanged."
                )
                target_grad = float("inf")

            if args.kl_loss_weight > 0.0:
                kl_grad = _gradient_norm_for_component(
                    probe_kl_loss,
                    "KL",
                    clear_bait_grads=detach_kl_loss_from_bait,
                )
                effective_kl_loss_weight = _safe_downscaled_weight(
                    float(args.kl_loss_weight), kl_grad, target_grad, "KL",
                )

            if (
                args.residual_fisher_loss_weight > 0.0
                and train_good_prompts
                and train_bad_prompts
            ):
                probe_residual_fisher = residual_fisher_suppression_loss(
                    model, tokenizer, controller,
                    good_prompts=train_good_prompts,
                    bad_prompts=train_bad_prompts,
                    batch_size=args.batch_size,
                    supervised_residual_dirs=supervised_residual_dirs,
                    active_installed_layers=active_installed_layers,
                    denom_floor=getattr(args, "residual_fisher_denom_floor", 1e-3),
                    loss_mode=getattr(args, "residual_fisher_loss_mode", "detached_gap"),
                    target_space=getattr(args, "residual_fisher_target_space", "projected"),
                ).to(probe_kl_loss.device)
                rf_grad = _gradient_norm_for_component(probe_residual_fisher, "residual_fisher")
                effective_residual_fisher_loss_weight = _safe_downscaled_weight(
                    float(args.residual_fisher_loss_weight),
                    rf_grad,
                    target_grad,
                    "residual_fisher",
                )

            if (
                args.residual_direction_loss_weight > 0.0
                and train_good_prompts
                and train_bad_prompts
            ):
                probe_residual_direction = residual_direction_orthogonal_loss(
                    model, tokenizer, controller,
                    good_prompts=train_good_prompts,
                    bad_prompts=train_bad_prompts,
                    batch_size=args.batch_size,
                    supervised_residual_dirs=supervised_residual_dirs,
                    active_installed_layers=active_installed_layers,
                ).to(probe_kl_loss.device)
                direction_grad = _gradient_norm_for_component(
                    probe_residual_direction,
                    "residual_direction",
                )
                effective_residual_direction_loss_weight = _safe_downscaled_weight(
                    float(args.residual_direction_loss_weight),
                    direction_grad,
                    target_grad,
                    "residual_direction",
                )

            if (
                args.residual_gap_anti_coherence_loss_weight > 0.0
                and train_good_prompts
                and train_bad_prompts
                and has_residual_gap_anti_coherence_peers
            ):
                probe_residual_gap_anti_coherence = residual_gap_anti_coherence_loss(
                    model, tokenizer, controller,
                    good_prompts=train_good_prompts,
                    bad_prompts=train_bad_prompts,
                    batch_size=args.batch_size,
                    active_installed_layers=active_installed_layers,
                    external_residual_gap_refs=external_residual_gap_refs,
                    margin=getattr(args, "residual_gap_anti_coherence_margin", 0.0),
                ).to(probe_kl_loss.device)
                gap_anti_coh_grad = _gradient_norm_for_component(
                    probe_residual_gap_anti_coherence,
                    "residual_gap_anti_coherence",
                )
                effective_residual_gap_anti_coherence_loss_weight = _safe_downscaled_weight(
                    float(args.residual_gap_anti_coherence_loss_weight),
                    gap_anti_coh_grad,
                    target_grad,
                    "residual_gap_anti_coherence",
                )

            if (
                args.global_direction_loss_weight > 0.0
                and train_good_prompts
                and train_bad_prompts
            ):
                probe_global_direction = heretic_global_direction_loss(
                    model, tokenizer, controller,
                    good_prompts=train_good_prompts,
                    bad_prompts=train_bad_prompts,
                    batch_size=args.batch_size,
                    supervised_residual_dirs=supervised_residual_dirs,
                    active_installed_layers=active_installed_layers,
                    external_residual_gap_refs=external_residual_gap_refs,
                    samples=getattr(args, "global_direction_samples", 5),
                    margin=getattr(args, "global_direction_margin", 0.0),
                    direction_range_low=getattr(args, "global_direction_range_low", 0.4),
                    direction_range_high=getattr(args, "global_direction_range_high", 0.9),
                ).to(probe_kl_loss.device)
                global_direction_grad = _gradient_norm_for_component(
                    probe_global_direction,
                    "global_direction",
                )
                effective_global_direction_loss_weight = _safe_downscaled_weight(
                    float(args.global_direction_loss_weight),
                    global_direction_grad,
                    target_grad,
                    "global_direction",
                )

            print(
                "  Loss auto-normalization probe: "
                f"local_grad={local_grad:.4g}, target={target_grad:.4g}, "
                f"effective_kl_w={effective_kl_loss_weight:.6g}, "
                f"effective_rf_w={effective_residual_fisher_loss_weight:.6g}, "
                f"effective_dir_w={effective_residual_direction_loss_weight:.6g}, "
                f"effective_gap_anti_coh_w={effective_residual_gap_anti_coherence_loss_weight:.6g}, "
                f"effective_global_dir_w={effective_global_direction_loss_weight:.6g}"
            )

    history: list[dict[str, float]] = []
    for epoch in range(args.n_epochs):
        shuffled = combined_pool[:]
        random.shuffle(shuffled)
        epoch_losses = []
        epoch_kl_losses = []
        epoch_recovery_losses = []
        epoch_visible_losses = []
        epoch_visible_shifts = []
        epoch_grad_norms = []
        epoch_fisher_poison_losses = []
        epoch_residual_fisher_losses = []
        epoch_residual_direction_losses = []
        epoch_bait_anti_coherence_losses = []
        epoch_residual_gap_anti_coherence_losses = []
        epoch_global_direction_losses = []
        # Plan D diagnostics: track the worst recovery hook and weakest visible hook.
        # These numbers reveal whether mean-aggregation was masking a lagging layer.
        worst_recovery_per_epoch = []   # max-over-hooks of per-hook recovery loss
        worst_recovery_name_counts = {}
        weakest_visible_per_epoch = []  # min-over-hooks of per-hook relative shift
        weakest_visible_name_counts = {}
        strongest_visible_per_epoch = []  # max-over-hooks of per-hook relative shift
        strongest_visible_name_counts = {}

        for start in range(0, len(shuffled), args.batch_size):
            batch = shuffled[start : start + args.batch_size]
            inputs = prepare_batch(model, tokenizer, batch)

            with controller.adapters_disabled():
                with capture_last_token_outputs(visible_modules, detach=True, cpu=False) as clean_visible_cache:
                    with capture_last_token_outputs(recovery_modules, detach=True, cpu=False) as clean_recovery_cache:
                        with torch.no_grad():
                            reference_logits = model(**inputs, return_dict=True).logits

            # Keep defended activations on-device and attached to the graph.  This is
            # the gradient path for both recovery_loss and visible_bait_loss.
            with capture_last_token_outputs(visible_modules, detach=False, cpu=False) as defended_visible_cache:
                with capture_last_token_outputs(recovery_modules, detach=False, cpu=False) as defended_recovery_cache:
                    defended_logits = model(**inputs, return_dict=True).logits
            kl_loss = training_logit_kl(
                defended_logits,
                reference_logits,
                inputs["attention_mask"],
                mode=getattr(args, "kl_loss_mode", "full"),
                top_k=getattr(args, "kl_top_k", 64),
                temperature=getattr(args, "kl_temperature", 1.0),
                logit_clamp=getattr(args, "kl_logit_clamp", 0.0),
            )
            recovery_out = transport_alignment_loss(
                clean_recovery_cache,
                defended_recovery_cache,
                max_layer_weight=rec_max_w,
                return_per_hook=True,
            )
            recovery_loss, recovery_per_hook = recovery_out
            visible_out = visible_bait_loss(
                clean_visible_cache,
                defended_visible_cache,
                floor=args.visible_shift_target,
                ceiling=getattr(args, "visible_shift_max", 0.0),
                max_layer_weight=vis_max_w,
                return_per_hook=True,
            )
            visible_loss, visible_per_hook = visible_out
            visible_shift = mean_relative_shift(clean_visible_cache, defended_visible_cache)
            # Basis regularization: penalize bait output basis drifting from init subspace
            basis_reg_loss = torch.tensor(0.0, device=kl_loss.device)
            if args.bait_basis_reg_weight > 0:
                for bait_module in controller.bait_modules:
                    if bait_module.basis.weight.requires_grad:
                        basis_reg_loss = basis_reg_loss + bait_module.basis_regularization_loss()

            # Fisher-poison loss: directly penalise good/bad separability in bait δ(x).
            # Only active when --fisher-poison-loss-weight > 0 and good/bad prompts available.
            fisher_poison = torch.tensor(0.0, device=kl_loss.device)
            if (
                args.fisher_poison_loss_weight > 0
                and train_good_prompts
                and train_bad_prompts
            ):
                fisher_poison = fisher_poison_loss(
                    model, tokenizer, controller,
                    good_prompts=train_good_prompts,
                    bad_prompts=train_bad_prompts,
                    batch_size=args.batch_size,
                    supervised_residual_dirs=supervised_residual_dirs,
                ).to(kl_loss.device)

            # Residual Fisher suppression: directly penalise good/bad separability
            # at the bait residual, forcing δ to actively close the class gap.
            residual_fisher = torch.tensor(0.0, device=kl_loss.device)
            if (
                effective_residual_fisher_loss_weight > 0
                and train_good_prompts
                and train_bad_prompts
            ):
                residual_fisher = residual_fisher_suppression_loss(
                    model, tokenizer, controller,
                    good_prompts=train_good_prompts,
                    bad_prompts=train_bad_prompts,
                    batch_size=args.batch_size,
                    supervised_residual_dirs=supervised_residual_dirs,
                    active_installed_layers=active_installed_layers,
                    denom_floor=getattr(args, "residual_fisher_denom_floor", 1e-3),
                    loss_mode=getattr(args, "residual_fisher_loss_mode", "detached_gap"),
                    target_space=getattr(args, "residual_fisher_target_space", "projected"),
                ).to(kl_loss.device)

            residual_direction = torch.tensor(0.0, device=kl_loss.device)
            if (
                effective_residual_direction_loss_weight > 0
                and train_good_prompts
                and train_bad_prompts
            ):
                residual_direction = residual_direction_orthogonal_loss(
                    model, tokenizer, controller,
                    good_prompts=train_good_prompts,
                    bad_prompts=train_bad_prompts,
                    batch_size=args.batch_size,
                    supervised_residual_dirs=supervised_residual_dirs,
                    active_installed_layers=active_installed_layers,
                ).to(kl_loss.device)

            residual_gap_anti_coherence = torch.tensor(0.0, device=kl_loss.device)
            if (
                effective_residual_gap_anti_coherence_loss_weight > 0
                and train_good_prompts
                and train_bad_prompts
                and has_residual_gap_anti_coherence_peers
            ):
                residual_gap_anti_coherence = residual_gap_anti_coherence_loss(
                    model, tokenizer, controller,
                    good_prompts=train_good_prompts,
                    bad_prompts=train_bad_prompts,
                    batch_size=args.batch_size,
                    active_installed_layers=active_installed_layers,
                    external_residual_gap_refs=external_residual_gap_refs,
                    margin=getattr(args, "residual_gap_anti_coherence_margin", 0.0),
                ).to(kl_loss.device)

            global_direction = torch.tensor(0.0, device=kl_loss.device)
            if (
                effective_global_direction_loss_weight > 0
                and train_good_prompts
                and train_bad_prompts
            ):
                global_direction = heretic_global_direction_loss(
                    model, tokenizer, controller,
                    good_prompts=train_good_prompts,
                    bad_prompts=train_bad_prompts,
                    batch_size=args.batch_size,
                    supervised_residual_dirs=supervised_residual_dirs,
                    active_installed_layers=active_installed_layers,
                    external_residual_gap_refs=external_residual_gap_refs,
                    samples=getattr(args, "global_direction_samples", 5),
                    margin=getattr(args, "global_direction_margin", 0.0),
                    direction_range_low=getattr(args, "global_direction_range_low", 0.4),
                    direction_range_high=getattr(args, "global_direction_range_high", 0.9),
                ).to(kl_loss.device)

            bait_anti_coherence = torch.tensor(0.0, device=kl_loss.device)
            if bait_anti_coherence_loss_weight > 0.0:
                bait_anti_coherence = bait_basis_anti_coherence_loss(
                    controller,
                    components=getattr(args, "bait_anti_coherence_components", 2),
                    margin=getattr(args, "bait_anti_coherence_margin", 0.0),
                    external_basis_refs=external_basis_refs,
                ).to(kl_loss.device)

            local_restoration_loss = (
                _weighted_loss(kl_loss, effective_kl_loss_weight)
                + _weighted_loss(recovery_loss, args.recovery_loss_weight)
                + _weighted_loss(visible_loss, args.visible_loss_weight)
            )
            bait_defense_loss = (
                _weighted_loss(basis_reg_loss, args.bait_basis_reg_weight)
                + _weighted_loss(fisher_poison, args.fisher_poison_loss_weight)
                + _weighted_loss(residual_fisher, effective_residual_fisher_loss_weight)
                + _weighted_loss(residual_direction, effective_residual_direction_loss_weight)
                + _weighted_loss(
                    residual_gap_anti_coherence,
                    effective_residual_gap_anti_coherence_loss_weight,
                )
                + _weighted_loss(global_direction, effective_global_direction_loss_weight)
                + _weighted_loss(bait_anti_coherence, bait_anti_coherence_loss_weight)
            )
            loss = local_restoration_loss + bait_defense_loss
            if not torch.isfinite(loss.detach()):
                raise FloatingPointError(
                    "Non-finite training loss before backward: "
                    f"loss={float(loss.detach().cpu())}, "
                    f"kl={float(kl_loss.detach().cpu())}, "
                    f"recovery={float(recovery_loss.detach().cpu())}, "
                    f"visible={float(visible_loss.detach().cpu())}, "
                    f"residual_fisher={float(residual_fisher.detach().cpu())}, "
                    f"residual_direction={float(residual_direction.detach().cpu())}, "
                    f"residual_gap_anti_coherence={float(residual_gap_anti_coherence.detach().cpu())}, "
                    f"global_direction={float(global_direction.detach().cpu())}, "
                    f"bait_anti_coherence={float(bait_anti_coherence.detach().cpu())}"
                )

            can_retry_without_global_losses = (
                effective_kl_loss_weight > 0.0
                or effective_residual_fisher_loss_weight > 0.0
                or effective_residual_direction_loss_weight > 0.0
                or effective_residual_gap_anti_coherence_loss_weight > 0.0
                or effective_global_direction_loss_weight > 0.0
            )
            optimizer.zero_grad(set_to_none=True)
            backward_terms: list[tuple[torch.Tensor, bool]] = []
            _append_weighted_backward_term(
                backward_terms,
                kl_loss,
                effective_kl_loss_weight,
                detach_kl_loss_from_bait,
            )
            _append_weighted_backward_term(
                backward_terms,
                recovery_loss,
                args.recovery_loss_weight,
                detach_recovery_loss_from_bait,
            )
            _append_weighted_backward_term(
                backward_terms,
                visible_loss,
                args.visible_loss_weight,
                detach_visible_loss_from_bait,
            )
            if bait_defense_loss.requires_grad:
                backward_terms.append((bait_defense_loss, False))
            for term_idx, (term, clear_bait_grads) in enumerate(backward_terms):
                if not term.requires_grad:
                    continue
                retain_graph = (
                    can_retry_without_global_losses
                    or term_idx < len(backward_terms) - 1
                )
                term.backward(retain_graph=retain_graph)
                if clear_bait_grads:
                    _clear_bait_parameter_grads()
            named_trainable_params = controller.trainable_named_parameters()
            grad_context = (
                f"loss={float(loss.detach().cpu()):.6g}, "
                f"kl={float(kl_loss.detach().cpu()):.6g}, "
                f"recovery={float(recovery_loss.detach().cpu()):.6g}, "
                f"visible={float(visible_loss.detach().cpu()):.6g}, "
                f"residual_fisher={float(residual_fisher.detach().cpu()):.6g}, "
                f"residual_direction={float(residual_direction.detach().cpu()):.6g}, "
                f"residual_gap_anti_coherence={float(residual_gap_anti_coherence.detach().cpu()):.6g}, "
                f"global_direction={float(global_direction.detach().cpu()):.6g}, "
                f"bait_anti_coherence={float(bait_anti_coherence.detach().cpu()):.6g}, "
                f"kl_w={effective_kl_loss_weight:.6g}, "
                f"rf_w={effective_residual_fisher_loss_weight:.6g}, "
                f"dir_w={effective_residual_direction_loss_weight:.6g}, "
                f"gap_anti_coh_w={effective_residual_gap_anti_coherence_loss_weight:.6g}, "
                f"global_dir_w={effective_global_direction_loss_weight:.6g}, "
                f"anti_coh_w={bait_anti_coherence_loss_weight:.6g}, "
                f"detach_rec_bait={int(detach_recovery_loss_from_bait)}, "
                f"detach_vis_bait={int(detach_visible_loss_from_bait)}, "
                f"detach_kl_bait={int(detach_kl_loss_from_bait)}"
            )
            try:
                assert_gradients_finite(named_trainable_params, grad_context)
            except FloatingPointError as exc:
                if not can_retry_without_global_losses:
                    if _freeze_unstable_bait_grads(named_trainable_params):
                        named_trainable_params = controller.trainable_named_parameters()
                    else:
                        raise
                else:
                    print(
                        "  [bold yellow]WARNING[/]: non-finite gradients with "
                        "auto-normalized KL/RF/global terms; disabling them for the rest "
                        f"of this stage and retrying this batch. {exc}"
                    )
                    _diagnose_nonfinite_components(
                        [
                            ("local_recovery", recovery_loss, args.recovery_loss_weight),
                            ("local_visible", visible_loss, args.visible_loss_weight),
                            ("kl", kl_loss, effective_kl_loss_weight),
                            (
                                "residual_fisher",
                                residual_fisher,
                                effective_residual_fisher_loss_weight,
                            ),
                            (
                                "residual_direction",
                                residual_direction,
                                effective_residual_direction_loss_weight,
                            ),
                            (
                                "residual_gap_anti_coherence",
                                residual_gap_anti_coherence,
                                effective_residual_gap_anti_coherence_loss_weight,
                            ),
                            (
                                "global_direction",
                                global_direction,
                                effective_global_direction_loss_weight,
                            ),
                            (
                                "bait_anti_coherence",
                                bait_anti_coherence,
                                bait_anti_coherence_loss_weight,
                            ),
                        ]
                    )
                    effective_kl_loss_weight = 0.0
                    effective_residual_fisher_loss_weight = 0.0
                    effective_residual_direction_loss_weight = 0.0
                    effective_residual_gap_anti_coherence_loss_weight = 0.0
                    effective_global_direction_loss_weight = 0.0
                    optimizer.zero_grad(set_to_none=True)
                    local_restoration_loss = (
                        _weighted_loss(recovery_loss, args.recovery_loss_weight)
                        + _weighted_loss(visible_loss, args.visible_loss_weight)
                    )
                    bait_defense_loss = (
                        _weighted_loss(basis_reg_loss, args.bait_basis_reg_weight)
                        + _weighted_loss(fisher_poison, args.fisher_poison_loss_weight)
                        + _weighted_loss(bait_anti_coherence, bait_anti_coherence_loss_weight)
                    )
                    loss = local_restoration_loss + bait_defense_loss
                    if not torch.isfinite(loss.detach()):
                        raise FloatingPointError(
                            "Non-finite fallback local loss before backward: "
                            f"loss={float(loss.detach().cpu())}, "
                            f"recovery={float(recovery_loss.detach().cpu())}, "
                            f"visible={float(visible_loss.detach().cpu())}"
                        ) from exc
                    fallback_terms: list[tuple[torch.Tensor, bool]] = []
                    _append_weighted_backward_term(
                        fallback_terms,
                        recovery_loss,
                        args.recovery_loss_weight,
                        detach_recovery_loss_from_bait,
                    )
                    _append_weighted_backward_term(
                        fallback_terms,
                        visible_loss,
                        args.visible_loss_weight,
                        detach_visible_loss_from_bait,
                    )
                    if bait_defense_loss.requires_grad:
                        fallback_terms.append((bait_defense_loss, False))
                    for term_idx, (term, clear_bait_grads) in enumerate(fallback_terms):
                        if not term.requires_grad:
                            continue
                        term.backward(retain_graph=term_idx < len(fallback_terms) - 1)
                        if clear_bait_grads:
                            _clear_bait_parameter_grads()
                    grad_context = (
                        f"fallback_loss={float(loss.detach().cpu()):.6g}, "
                        f"kl={float(kl_loss.detach().cpu()):.6g}, "
                        f"recovery={float(recovery_loss.detach().cpu()):.6g}, "
                        f"visible={float(visible_loss.detach().cpu()):.6g}, "
                        f"residual_fisher={float(residual_fisher.detach().cpu()):.6g}, "
                        f"residual_direction={float(residual_direction.detach().cpu()):.6g}, "
                        f"residual_gap_anti_coherence={float(residual_gap_anti_coherence.detach().cpu()):.6g}, "
                        f"global_direction={float(global_direction.detach().cpu()):.6g}, "
                        f"bait_anti_coherence={float(bait_anti_coherence.detach().cpu()):.6g}, "
                        f"anti_coh_w={bait_anti_coherence_loss_weight:.6g}, "
                        "kl_w=0, rf_w=0, dir_w=0, gap_anti_coh_w=0, global_dir_w=0, "
                        f"detach_rec_bait={int(detach_recovery_loss_from_bait)}, "
                        f"detach_vis_bait={int(detach_visible_loss_from_bait)}, "
                        f"detach_kl_bait={int(detach_kl_loss_from_bait)}"
                    )
                    try:
                        assert_gradients_finite(named_trainable_params, grad_context)
                    except FloatingPointError:
                        if _freeze_unstable_bait_grads(named_trainable_params):
                            named_trainable_params = controller.trainable_named_parameters()
                        else:
                            raise
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                [param for _, param in named_trainable_params],
                max_norm=args.max_grad_norm,
            )
            if not torch.isfinite(grad_norm_tensor.detach()):
                raise FloatingPointError(
                    f"Non-finite gradient norm before optimizer step: "
                    f"{float(grad_norm_tensor.detach().cpu())}"
                )
            grad_norm = grad_norm_tensor.item()
            optimizer.step()
            for idx, param in enumerate(controller.trainable_parameters()):
                assert_finite_tensor(f"trainable parameter {idx} after optimizer step", param)

            epoch_losses.append(loss.item())
            epoch_kl_losses.append(kl_loss.item())
            epoch_recovery_losses.append(recovery_loss.item())
            epoch_visible_losses.append(visible_loss.item())
            epoch_visible_shifts.append(float(visible_shift.item()))
            epoch_grad_norms.append(grad_norm)
            if args.fisher_poison_loss_weight > 0:
                epoch_fisher_poison_losses.append(float(fisher_poison.item()))
            if effective_residual_fisher_loss_weight > 0:
                epoch_residual_fisher_losses.append(float(residual_fisher.item()))
            if effective_residual_direction_loss_weight > 0:
                epoch_residual_direction_losses.append(float(residual_direction.item()))
            if bait_anti_coherence_loss_weight > 0:
                epoch_bait_anti_coherence_losses.append(float(bait_anti_coherence.item()))
            if effective_residual_gap_anti_coherence_loss_weight > 0:
                epoch_residual_gap_anti_coherence_losses.append(
                    float(residual_gap_anti_coherence.item())
                )
            if effective_global_direction_loss_weight > 0:
                epoch_global_direction_losses.append(float(global_direction.item()))
            # Plan D diagnostics: which hook is the laggard this batch?
            if recovery_per_hook:
                worst_name = max(recovery_per_hook, key=lambda k: recovery_per_hook[k].item())
                worst_recovery_per_epoch.append(float(recovery_per_hook[worst_name].item()))
                worst_recovery_name_counts[worst_name] = worst_recovery_name_counts.get(worst_name, 0) + 1
            if visible_per_hook:
                weakest_name = min(
                    visible_per_hook,
                    key=lambda k: visible_per_hook[k]["rel_shift"].item(),
                )
                strongest_name = max(
                    visible_per_hook,
                    key=lambda k: visible_per_hook[k]["rel_shift"].item(),
                )
                weakest_visible_per_epoch.append(
                    float(visible_per_hook[weakest_name]["rel_shift"].item())
                )
                weakest_visible_name_counts[weakest_name] = weakest_visible_name_counts.get(weakest_name, 0) + 1
                strongest_visible_per_epoch.append(
                    float(visible_per_hook[strongest_name]["rel_shift"].item())
                )
                strongest_visible_name_counts[strongest_name] = strongest_visible_name_counts.get(strongest_name, 0) + 1

        mean_loss = float(sum(epoch_losses) / max(len(epoch_losses), 1))
        mean_kl = float(sum(epoch_kl_losses) / max(len(epoch_kl_losses), 1))
        mean_recovery = float(sum(epoch_recovery_losses) / max(len(epoch_recovery_losses), 1))
        mean_visible_loss = float(sum(epoch_visible_losses) / max(len(epoch_visible_losses), 1))
        mean_visible_shift = float(sum(epoch_visible_shifts) / max(len(epoch_visible_shifts), 1))
        mean_grad = float(sum(epoch_grad_norms) / max(len(epoch_grad_norms), 1))
        mean_fisher_poison = (
            float(sum(epoch_fisher_poison_losses) / max(len(epoch_fisher_poison_losses), 1))
            if epoch_fisher_poison_losses else 0.0
        )
        mean_residual_direction = (
            float(sum(epoch_residual_direction_losses) / max(len(epoch_residual_direction_losses), 1))
            if epoch_residual_direction_losses else 0.0
        )
        mean_bait_anti_coherence = (
            float(sum(epoch_bait_anti_coherence_losses) / max(len(epoch_bait_anti_coherence_losses), 1))
            if epoch_bait_anti_coherence_losses else 0.0
        )
        mean_residual_gap_anti_coherence = (
            float(sum(epoch_residual_gap_anti_coherence_losses) / max(len(epoch_residual_gap_anti_coherence_losses), 1))
            if epoch_residual_gap_anti_coherence_losses else 0.0
        )
        mean_global_direction = (
            float(sum(epoch_global_direction_losses) / max(len(epoch_global_direction_losses), 1))
            if epoch_global_direction_losses else 0.0
        )
        mean_worst_rec = (
            float(sum(worst_recovery_per_epoch) / max(len(worst_recovery_per_epoch), 1))
            if worst_recovery_per_epoch else 0.0
        )
        mean_weakest_vis = (
            float(sum(weakest_visible_per_epoch) / max(len(weakest_visible_per_epoch), 1))
            if weakest_visible_per_epoch else 0.0
        )
        mean_strongest_vis = (
            float(sum(strongest_visible_per_epoch) / max(len(strongest_visible_per_epoch), 1))
            if strongest_visible_per_epoch else 0.0
        )
        record = {
            "epoch": epoch + 1,
            "train_loss": mean_loss,
            "train_kl": mean_kl,
            "train_recovery": mean_recovery,
            "train_recovery_relative_shift_sq": mean_recovery,
            "train_visible_loss": mean_visible_loss,
            "train_visible_shift": mean_visible_shift,
            "grad_norm": mean_grad,
            "train_worst_recovery": mean_worst_rec,
            "train_weakest_visible_shift": mean_weakest_vis,
            "train_strongest_visible_shift": mean_strongest_vis,
            "effective_kl_loss_weight": effective_kl_loss_weight,
            "effective_residual_fisher_loss_weight": effective_residual_fisher_loss_weight,
            "effective_residual_direction_loss_weight": effective_residual_direction_loss_weight,
            "effective_residual_gap_anti_coherence_loss_weight": effective_residual_gap_anti_coherence_loss_weight,
            "effective_global_direction_loss_weight": effective_global_direction_loss_weight,
            "bait_anti_coherence_loss_weight": bait_anti_coherence_loss_weight,
        }
        if worst_recovery_name_counts:
            # Which hook was the laggard most often this epoch?
            record["worst_recovery_hook"] = max(
                worst_recovery_name_counts, key=worst_recovery_name_counts.get
            )
        if weakest_visible_name_counts:
            record["weakest_visible_hook"] = max(
                weakest_visible_name_counts, key=weakest_visible_name_counts.get
            )
        if strongest_visible_name_counts:
            record["strongest_visible_hook"] = max(
                strongest_visible_name_counts, key=strongest_visible_name_counts.get
            )
        if epoch_fisher_poison_losses:
            record["train_fisher_poison"] = mean_fisher_poison
        mean_residual_fisher = (
            float(sum(epoch_residual_fisher_losses) / max(len(epoch_residual_fisher_losses), 1))
            if epoch_residual_fisher_losses else 0.0
        )
        if epoch_residual_fisher_losses:
            record["train_residual_fisher"] = mean_residual_fisher
        if epoch_residual_direction_losses:
            record["train_residual_direction"] = mean_residual_direction
        if epoch_bait_anti_coherence_losses:
            record["train_bait_anti_coherence"] = mean_bait_anti_coherence
        if epoch_residual_gap_anti_coherence_losses:
            record["train_residual_gap_anti_coherence"] = mean_residual_gap_anti_coherence
        if epoch_global_direction_losses:
            record["train_global_direction"] = mean_global_direction
        history.append(record)
        laggard_txt = ""
        if "worst_recovery_hook" in record:
            laggard_txt = (
                f" worst_rec={mean_worst_rec:.4f}@{record['worst_recovery_hook']}"
                f" weakest_vis={mean_weakest_vis:.4f}@{record.get('weakest_visible_hook','?')}"
                f" strongest_vis={mean_strongest_vis:.4f}@{record.get('strongest_visible_hook','?')}"
            )
        print(
            f"  Epoch {epoch + 1:>2}/{args.n_epochs}: "
            f"train_loss={mean_loss:.6f} "
            f"kl={mean_kl:.6f} "
            f"recovery={mean_recovery:.6f} "
            f"visible_loss={mean_visible_loss:.6f} "
            f"visible_shift={mean_visible_shift:.4f} "
            + (f"fisher_poison={mean_fisher_poison:.4f} " if epoch_fisher_poison_losses else "")
            + (f"res_fisher={mean_residual_fisher:.4f} " if epoch_residual_fisher_losses else "")
            + (f"res_dir={mean_residual_direction:.4f} " if epoch_residual_direction_losses else "")
            + (f"gap_anti_coh={mean_residual_gap_anti_coherence:.4f} " if epoch_residual_gap_anti_coherence_losses else "")
            + (f"global_dir={mean_global_direction:.4f} " if epoch_global_direction_losses else "")
            + (f"anti_coh={mean_bait_anti_coherence:.4f} " if epoch_bait_anti_coherence_losses else "")
            + f"grad_norm={mean_grad:.4f}"
            + laggard_txt
        )
    return history


def fisher_ratio(good_residuals: torch.Tensor, bad_residuals: torch.Tensor, residual_idx: int) -> float:
    good = good_residuals[:, residual_idx, :]
    bad = bad_residuals[:, residual_idx, :]
    return fisher_ratio_vectors(good, bad)


def fisher_ratio_vectors(good: torch.Tensor, bad: torch.Tensor) -> float:
    """Compute Fisher discriminability from (n_samples, hidden_dim) tensors."""
    mean_delta = bad.mean(dim=0) - good.mean(dim=0)
    between = mean_delta.square().sum()
    within = good.var(dim=0).sum() + bad.var(dim=0).sum()
    return float((between / within.clamp_min(1e-8)).item())


def residual_fisher_profile_from_residuals(
    good_residuals: torch.Tensor,
    bad_residuals: torch.Tensor,
) -> list[dict[str, float]]:
    """Per-residual good/bad Fisher profile, including embedding residual_0."""
    profile = []
    for residual_idx in range(good_residuals.shape[1]):
        good = good_residuals[:, residual_idx, :].float()
        bad = bad_residuals[:, residual_idx, :].float()
        mean_delta = bad.mean(dim=0) - good.mean(dim=0)
        between = mean_delta.square().sum()
        within = good.var(dim=0).sum() + bad.var(dim=0).sum()
        profile.append(
            {
                "residual_idx": residual_idx,
                "fisher": float((between / within.clamp_min(1e-8)).item()),
                "between_norm": float(between.sqrt().item()),
                "within_var": float(within.item()),
            }
        )
    return profile


def evaluate_clean_fisher_profile(
    model: nn.Module,
    tokenizer,
    good_prompts: list[Prompt],
    bad_prompts: list[Prompt],
    batch_size: int,
) -> list[dict[str, float]]:
    from shared_utils import get_residuals_batched

    with torch.no_grad():
        good_residuals = get_residuals_batched(model, tokenizer, good_prompts, batch_size)
        bad_residuals = get_residuals_batched(model, tokenizer, bad_prompts, batch_size)
    return residual_fisher_profile_from_residuals(good_residuals, bad_residuals)


def evaluate_original_clean_fisher_profile(
    model_name: str,
    tokenizer_name: str | None,
    device: torch.device,
    gpu_mode: str,
    good_prompts: list[Prompt],
    bad_prompts: list[Prompt],
    batch_size: int,
) -> list[dict[str, float]]:
    print(f"\nEvaluating original clean Fisher profile from [bold]{model_name}[/]...")
    original_model, original_tokenizer = load_model(
        model_name,
        device,
        tokenizer_name=tokenizer_name,
        gpu_mode=gpu_mode,
    )
    for param in original_model.parameters():
        param.requires_grad_(False)
    try:
        return evaluate_clean_fisher_profile(
            original_model,
            original_tokenizer,
            good_prompts,
            bad_prompts,
            batch_size,
        )
    finally:
        del original_model
        del original_tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def residual_restoration_by_layer(
    clean_good_residuals: torch.Tensor,
    clean_bad_residuals: torch.Tensor,
    defended_good_residuals: torch.Tensor,
    defended_bad_residuals: torch.Tensor,
) -> dict[str, dict[str, float]]:
    """Measure direct defended-vs-clean residual restoration for each residual index.

    Direction/Fisher metrics answer "does Heretic see a different good/bad
    separator?".  This answers the separate recovery question: "did the actual
    residual vectors return to the clean model after the bait layer?".
    """

    def summarize(clean: torch.Tensor, defended: torch.Tensor, residual_idx: int) -> dict[str, float]:
        clean_layer = clean[:, residual_idx, :].float()
        defended_layer = defended[:, residual_idx, :].float()
        diff = defended_layer - clean_layer
        rel_shift = diff.norm(dim=-1) / clean_layer.norm(dim=-1).clamp_min(1e-8)
        residual_cos = F.cosine_similarity(clean_layer, defended_layer, dim=-1)
        return {
            "relative_shift": float(rel_shift.mean().item()),
            "residual_cosine": float(residual_cos.mean().item()),
            "delta_norm": float(diff.norm(dim=-1).mean().item()),
            "mse": float(diff.square().mean().item()),
        }

    clean_all = torch.cat([clean_good_residuals, clean_bad_residuals], dim=0)
    defended_all = torch.cat([defended_good_residuals, defended_bad_residuals], dim=0)
    metrics: dict[str, dict[str, float]] = {}
    for residual_idx in range(clean_all.shape[1]):
        good = summarize(clean_good_residuals, defended_good_residuals, residual_idx)
        bad = summarize(clean_bad_residuals, defended_bad_residuals, residual_idx)
        all_prompts = summarize(clean_all, defended_all, residual_idx)
        metrics[str(residual_idx)] = {
            "good_relative_shift": good["relative_shift"],
            "bad_relative_shift": bad["relative_shift"],
            "all_relative_shift": all_prompts["relative_shift"],
            "good_residual_cosine": good["residual_cosine"],
            "bad_residual_cosine": bad["residual_cosine"],
            "all_residual_cosine": all_prompts["residual_cosine"],
            "good_delta_norm": good["delta_norm"],
            "bad_delta_norm": bad["delta_norm"],
            "all_delta_norm": all_prompts["delta_norm"],
            "mse": all_prompts["mse"],
        }
    return metrics


def collect_single_token_ids(tokenizer, terms: list[str]) -> list[int]:
    token_ids = []
    for term in terms:
        for candidate in (f" {term}", term, f" {term.capitalize()}"):
            encoded = tokenizer.encode(candidate, add_special_tokens=False)
            if len(encoded) == 1:
                token_ids.append(encoded[0])
                break
    return sorted(set(token_ids))


def score_refusal_margin(
    model: nn.Module,
    tokenizer,
    prompts: list[Prompt],
    refusal_token_ids: list[int],
    compliance_token_ids: list[int],
    batch_size: int,
) -> float:
    margins = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        inputs = prepare_batch(model, tokenizer, batch)
        with torch.no_grad():
            logits = model(**inputs, return_dict=True).logits[:, -1, :].float()
        refusal_score = torch.logsumexp(logits[:, refusal_token_ids], dim=-1)
        compliance_score = torch.logsumexp(logits[:, compliance_token_ids], dim=-1)
        margins.append((refusal_score - compliance_score).mean().item())
    return float(sum(margins) / max(len(margins), 1))


def evaluate_bait_delta_diagnostics(
    model: nn.Module,
    tokenizer,
    controller: DefenseController,
    good_prompts: list[Prompt],
    bad_prompts: list[Prompt],
    batch_size: int,
) -> dict:
    """Compute the four key δ(x) diagnostic quantities.

    1. ||μ_h^δ - μ_b^δ||: class-mean separation in bait perturbation space
    2. Within-class variance of δ(x): is intra-class jitter drowning the signal?
    3. Cosine similarity between δ direction and Heretic-style refusal direction
    4. ||W_up @ δ(x)||: downstream amplification magnitude
    """
    layers = get_layers(model)
    diagnostics = {}

    for idx, installed in enumerate(controller.installed_layers):
        bait_module = controller.bait_modules[idx]
        bait_layer = layers[installed.bait_layer_idx]

        # Bait is on down_proj — hook its base_linear to get the MLP intermediate input
        dp_wrapper = bait_layer.mlp.down_proj
        base_down_proj = dp_wrapper.base_linear if isinstance(dp_wrapper, LinearWithAdapters) else dp_wrapper

        # For downstream amplification: how much does the LM head amplify δ?
        # Since δ is now in the residual stream (hidden_dim), we can just measure ||δ||
        # We don't need up_proj amplification anymore — δ IS in the residual stream.
        # Instead, we keep track of the norm for comparison.

        good_deltas = []
        bad_deltas = []

        def collect_deltas(prompts: list[Prompt], storage: list):
            hook_inputs = []

            def input_hook(module, inp, out):
                x = inp[0] if isinstance(inp, tuple) else inp
                hook_inputs.append(x[:, -1, :].detach().float())

            handle = base_down_proj.register_forward_hook(input_hook)
            try:
                for start in range(0, len(prompts), batch_size):
                    batch = prompts[start : start + batch_size]
                    inputs = prepare_batch(model, tokenizer, batch)
                    hook_inputs.clear()
                    with torch.no_grad():
                        model(**inputs, return_dict=True)
                    if hook_inputs:
                        h = hook_inputs[0].to(bait_module.coeff.weight.device)
                        with torch.no_grad():
                            delta = bait_module(h)
                        storage.append(delta.cpu())
            finally:
                handle.remove()

        collect_deltas(good_prompts, good_deltas)
        collect_deltas(bad_prompts, bad_deltas)

        good_delta_tensor = torch.cat(good_deltas, dim=0)  # (n_good, hidden_dim)
        bad_delta_tensor = torch.cat(bad_deltas, dim=0)    # (n_bad, hidden_dim)

        # 1. Class-mean separation
        mu_good = good_delta_tensor.mean(dim=0)
        mu_bad = bad_delta_tensor.mean(dim=0)
        mean_diff_norm = (mu_bad - mu_good).norm().item()

        # 2. Within-class variance
        good_var = good_delta_tensor.var(dim=0).sum().item()
        bad_var = bad_delta_tensor.var(dim=0).sum().item()
        within_class_var = good_var + bad_var

        # Fisher-like ratio for δ(x) itself
        between = (mu_bad - mu_good).square().sum().item()
        delta_fisher = between / max(within_class_var, 1e-8)

        # 3. Cosine similarity between δ direction and clean refusal direction
        # Now δ lives in residual stream space — directly comparable to Heretic's r_l
        from shared_utils import get_residuals_batched
        with controller.adapters_disabled():
            clean_good_res = get_residuals_batched(model, tokenizer, good_prompts, batch_size)
            clean_bad_res = get_residuals_batched(model, tokenizer, bad_prompts, batch_size)
        # Refusal direction at bait layer's residual position (residual_idx = layer_idx + 1)
        res_idx = installed.bait_layer_idx + 1
        if res_idx < clean_good_res.shape[1]:
            refusal_dir = (clean_bad_res[:, res_idx, :].mean(dim=0) - clean_good_res[:, res_idx, :].mean(dim=0))
            refusal_dir = F.normalize(refusal_dir, p=2, dim=0)
            delta_dir = F.normalize(mu_bad - mu_good, p=2, dim=0)
            cos_with_refusal = F.cosine_similarity(
                delta_dir.unsqueeze(0), refusal_dir.unsqueeze(0)
            ).item()
        else:
            cos_with_refusal = float("nan")

        # 4. δ norm — since bait now directly enters residual stream, ||δ|| is the
        # actual perturbation magnitude in the space Heretic measures
        all_deltas = torch.cat([good_delta_tensor, bad_delta_tensor], dim=0)
        mean_delta_norm = all_deltas.norm(dim=-1).mean().item()

        diagnostics[f"bait_L{installed.bait_layer_idx}"] = {
            "mean_diff_norm": mean_diff_norm,
            "within_class_var_good": good_var,
            "within_class_var_bad": bad_var,
            "within_class_var_total": within_class_var,
            "delta_fisher_ratio": delta_fisher,
            "cos_with_refusal_dir": cos_with_refusal,
            "mean_delta_norm": mean_delta_norm,
        }

    return diagnostics


def evaluate_defense(
    model: nn.Module,
    tokenizer,
    controller: DefenseController,
    good_prompts: list[Prompt],
    bad_prompts: list[Prompt],
    train_prompts: list[Prompt],
    batch_size: int,
    external_direction_refs: dict[str, list[dict]] | None = None,
    bait_anti_coherence_components: int = 2,
) -> dict:
    from shared_utils import get_residuals_batched

    refusal_token_ids = collect_single_token_ids(
        tokenizer,
        ["sorry", "cannot", "can't", "unable", "refuse", "won't", "decline"],
    )
    compliance_token_ids = collect_single_token_ids(
        tokenizer,
        ["sure", "here", "first", "yes", "can", "step", "okay"],
    )
    if not refusal_token_ids or not compliance_token_ids:
        raise RuntimeError("Could not construct single-token refusal/compliance vocabularies")

    with controller.adapters_disabled():
        clean_good_residuals = get_residuals_batched(model, tokenizer, good_prompts, batch_size)
        clean_bad_residuals = get_residuals_batched(model, tokenizer, bad_prompts, batch_size)
        clean_good_margin = score_refusal_margin(
            model,
            tokenizer,
            good_prompts,
            refusal_token_ids,
            compliance_token_ids,
            batch_size,
        )
        clean_bad_margin = score_refusal_margin(
            model,
            tokenizer,
            bad_prompts,
            refusal_token_ids,
            compliance_token_ids,
            batch_size,
        )

    defended_good_residuals = get_residuals_batched(model, tokenizer, good_prompts, batch_size)
    defended_bad_residuals = get_residuals_batched(model, tokenizer, bad_prompts, batch_size)
    fisher_profiles = {
        "reference_clean": residual_fisher_profile_from_residuals(
            clean_good_residuals,
            clean_bad_residuals,
        ),
        "defended": residual_fisher_profile_from_residuals(
            defended_good_residuals,
            defended_bad_residuals,
        ),
    }
    residual_restoration = residual_restoration_by_layer(
        clean_good_residuals,
        clean_bad_residuals,
        defended_good_residuals,
        defended_bad_residuals,
    )
    defended_good_margin = score_refusal_margin(
        model,
        tokenizer,
        good_prompts,
        refusal_token_ids,
        compliance_token_ids,
        batch_size,
    )
    defended_bad_margin = score_refusal_margin(
        model,
        tokenizer,
        bad_prompts,
        refusal_token_ids,
        compliance_token_ids,
        batch_size,
    )

    clean_refusal_dirs = F.normalize(
        clean_bad_residuals.mean(dim=0) - clean_good_residuals.mean(dim=0),
        p=2,
        dim=1,
    )
    defended_refusal_dirs = F.normalize(
        defended_bad_residuals.mean(dim=0) - defended_good_residuals.mean(dim=0),
        p=2,
        dim=1,
    )

    per_layer = []
    for layer_idx in range(1, clean_refusal_dirs.shape[0]):
        cos_sim = F.cosine_similarity(
            clean_refusal_dirs[layer_idx].unsqueeze(0),
            defended_refusal_dirs[layer_idx].unsqueeze(0),
        ).item()
        per_layer.append(
            {
                "residual_idx": layer_idx,
                "direction_cosine": cos_sim,
                "fisher_clean": fisher_ratio(clean_good_residuals, clean_bad_residuals, layer_idx),
                "fisher_defended": fisher_ratio(defended_good_residuals, defended_bad_residuals, layer_idx),
            }
        )

    bait_gates = {
        installed.bait_layer_idx: float(controller.bait_modules[idx].gate(detach=True).item())
        for idx, installed in enumerate(controller.installed_layers)
    }

    # --- δ(x) diagnostics — serialized to JSON only, not printed to terminal.
    # Preserves cos_with_refusal_dir for debug scripts (run_step4_phase1_debug.sh,
    # run_step4_phase2_debug.sh) that compare sign-flip behavior across runs.
    bait_delta_diagnostics = evaluate_bait_delta_diagnostics(
        model, tokenizer, controller, good_prompts, bad_prompts, batch_size,
    )
    cross_layer_geometry = cross_layer_geometry_diagnostics(
        controller=controller,
        defended_good_residuals=defended_good_residuals,
        defended_bad_residuals=defended_bad_residuals,
        external_direction_refs=external_direction_refs,
        basis_components=bait_anti_coherence_components,
    )

    # KL split by prompt type — Heretic's two populations
    kl_by_prompt_type = evaluate_kl_by_prompt_type(
        model, tokenizer, controller,
        good_prompts=good_prompts[: min(32, len(good_prompts))],
        bad_prompts=bad_prompts[: min(32, len(bad_prompts))],
        batch_size=batch_size,
    )

    return {
        "train_kl_eval": evaluate_kl(model, tokenizer, controller, train_prompts[: min(32, len(train_prompts))], batch_size),
        "clean_good_refusal_margin": clean_good_margin,
        "clean_bad_refusal_margin": clean_bad_margin,
        "defended_good_refusal_margin": defended_good_margin,
        "defended_bad_refusal_margin": defended_bad_margin,
        "clean_margin_separation": clean_bad_margin - clean_good_margin,
        "defended_margin_separation": defended_bad_margin - defended_good_margin,
        "per_layer": per_layer,
        "fisher_profiles": fisher_profiles,
        "bait_delta_diagnostics": bait_delta_diagnostics,
        "cross_layer_geometry": cross_layer_geometry,
        "residual_restoration": residual_restoration,
        "bait_gates": bait_gates,
        "kl_by_prompt_type": kl_by_prompt_type,
        "refusal_tokens": refusal_token_ids,
        "compliance_tokens": compliance_token_ids,
    }


def print_design_summary(priors: list[LayerPrior]) -> None:
    table = Table(title="Cross-layer defense design priors")
    table.add_column("Bait Layer", justify="right")
    table.add_column("Recovery Layer", justify="right")
    table.add_column("Pathway", justify="left")
    table.add_column("Rank", justify="right")
    table.add_column("Spectrum energy", justify="right")
    table.add_column("Bait scale (effective)", justify="right")
    table.add_column("Bait scale (CLI arg)", justify="right")
    table.add_column("Recovery init", justify="right")
    for prior in priors:
        table.add_row(
            str(prior.layer_idx),
            str(prior.layer_idx + 1),
            "down_proj → next o_proj + down_proj",
            str(prior.rank),
            f"{prior.spectrum_energy:.3f}",
            f"{prior.bait_scale:.4f}  (= {prior.bait_scale / max(math.sqrt(max(prior.spectrum_energy, 1e-8)), 1e-8):.3f} × √{prior.spectrum_energy:.3f})",
            f"{prior.bait_scale / max(math.sqrt(max(prior.spectrum_energy, 1e-8)), 1e-8):.3f}",
            f"{prior.recovery_init_scale:.6f}",
        )
    print(table)


def print_defense_scorecard(metrics: dict, installed_layers: list[InstalledLayerDefense]) -> None:
    """Simplified per-pair defense scorecard.

    Shows for each (bait_layer → recovery_layer) pair:
      KL  |  bait_cos  bait_fc  bait_fd  |  recov_cos  recov_fc  recov_fd

    Column meanings:
      KL        — KL(defended || clean) for this pair in isolation; falls back to overall KL
      bait_cos  — cos(clean_refusal_dir, defended_refusal_dir) at bait residual; want LOW (poisoned)
      bait_fc   — Fisher(good vs bad) clean at bait residual
      bait_fd   — Fisher(good vs bad) defended at bait residual; want LOWER than fc
      recov_cos — same cosine at recovery residual; want HIGH (≈1, direction restored)
      recov_fc  — Fisher clean at recovery residual
      recov_fd  — Fisher defended at recovery residual; want CLOSE to fc (separability restored)
    """
    train_kl = metrics.get("train_kl_eval", float("nan"))
    kl_by_type = metrics.get("kl_by_prompt_type", {})
    per_layer_kl = metrics.get("per_layer_kl", {})
    metrics_by_residual = {item["residual_idx"]: item for item in metrics.get("per_layer", [])}

    def fisher_delta_pct(item: dict) -> float:
        clean = item.get("fisher_clean", float("nan"))
        defended = item.get("fisher_defended", float("nan"))
        return 100.0 * (defended - clean) / max(clean, 1e-8)

    # ── Header line ──
    kl_str = f"{train_kl:.6f}"
    if kl_by_type:
        kl_good = kl_by_type.get("kl_good", float("nan"))
        kl_bad  = kl_by_type.get("kl_bad",  float("nan"))
        kl_str += f"  (good: {kl_good:.6f}  bad: {kl_bad:.6f})"
    print(f"\n  KL (defended vs clean, general): {kl_str}")
    print(
        f"  Refusal-margin separation (clean→defended): "
        f"{metrics.get('clean_margin_separation', float('nan')):.4f} → "
        f"{metrics.get('defended_margin_separation', float('nan')):.4f}"
    )
    if metrics.get("bait_gates"):
        gate_str = "  ".join(
            f"L{l}: {g:.3f}" for l, g in sorted(metrics["bait_gates"].items())
        )
        print(f"  Bait gates: {gate_str}")
    geometry = metrics.get("cross_layer_geometry", {})
    gap_summary = geometry.get("residual_gap_current_plus_external") or geometry.get("residual_gap_current")
    if gap_summary and gap_summary.get("num_pairs", 0):
        print(
            "  Residual-gap |cos|: "
            f"p90={gap_summary.get('p90_abs_cosine', float('nan')):.4f}  "
            f"max={gap_summary.get('max_abs_cosine', float('nan')):.4f}  "
            f">{gap_summary.get('threshold', 0.5):.2f}="
            f"{gap_summary.get('count_above_threshold', 0)}/{gap_summary.get('num_pairs', 0)}"
        )

    # ── Per-pair table ──
    table = Table(title="Per-pair defense scorecard  [bait: cos↓ fd↓]  [recovery: cos≈1 fd≈clean]")
    table.add_column("Pair",      justify="center")
    table.add_column("KL",        justify="right")
    table.add_column("bait_cos",  justify="right")
    table.add_column("bait_fc",   justify="right")
    table.add_column("bait_fd",   justify="right")
    table.add_column("recov_cos", justify="right")
    table.add_column("recov_fc",  justify="right")
    table.add_column("recov_fd",  justify="right")
    table.add_column("recov_Δ%",  justify="right")

    for inst in installed_layers:
        bait_res  = inst.bait_layer_idx + 1
        recov_res = inst.recovery_layer_idx + 1
        bait_item  = metrics_by_residual.get(bait_res,  {})
        recov_item = metrics_by_residual.get(recov_res, {})
        pair_key = f"pair_L{inst.bait_layer_idx}_L{inst.recovery_layer_idx}"
        pair_kl  = per_layer_kl.get(pair_key, train_kl)

        table.add_row(
            f"L{inst.bait_layer_idx}→L{inst.recovery_layer_idx}",
            f"{pair_kl:.5f}",
            f"{bait_item.get('direction_cosine', float('nan')):.4f}",
            f"{bait_item.get('fisher_clean',     float('nan')):.4f}",
            f"{bait_item.get('fisher_defended',  float('nan')):.4f}",
            f"{recov_item.get('direction_cosine', float('nan')):.4f}",
            f"{recov_item.get('fisher_clean',     float('nan')):.4f}",
            f"{recov_item.get('fisher_defended',  float('nan')):.4f}",
            f"{fisher_delta_pct(recov_item):+.1f}%",
        )
    print(table)

    profiles = metrics.get("fisher_profiles", {})
    if profiles:
        profile_table = Table(title="Per-residual good/bad Fisher profile")
        profile_table.add_column("residual", justify="right")
        if "original_clean" in profiles:
            profile_table.add_column("original_clean", justify="right")
        profile_table.add_column("reference_clean", justify="right")
        profile_table.add_column("defended", justify="right")
        profile_table.add_column("def/ref Δ%", justify="right")

        reference_by_res = {
            item["residual_idx"]: item for item in profiles.get("reference_clean", [])
        }
        defended_by_res = {
            item["residual_idx"]: item for item in profiles.get("defended", [])
        }
        original_by_res = {
            item["residual_idx"]: item for item in profiles.get("original_clean", [])
        }
        residual_indices = sorted(set(reference_by_res) | set(defended_by_res) | set(original_by_res))
        for residual_idx in residual_indices:
            reference_fisher = reference_by_res.get(residual_idx, {}).get("fisher", float("nan"))
            defended_fisher = defended_by_res.get(residual_idx, {}).get("fisher", float("nan"))
            row = [str(residual_idx)]
            if "original_clean" in profiles:
                row.append(f"{original_by_res.get(residual_idx, {}).get('fisher', float('nan')):.4f}")
            row.extend(
                [
                    f"{reference_fisher:.4f}",
                    f"{defended_fisher:.4f}",
                    f"{100.0 * (defended_fisher - reference_fisher) / max(reference_fisher, 1e-8):+.1f}%",
                ]
            )
            profile_table.add_row(*row)
        print(profile_table)


def _uninstall_adapters(model: nn.Module, controller: DefenseController) -> None:
    """Remove adapter wrappers and restore original linear modules."""
    layers = get_layers(model)
    for installed in controller.installed_layers:
        bait_layer = layers[installed.bait_layer_idx]
        recovery_layer = layers[installed.recovery_layer_idx]
        # Restore base linears from wrappers (without merging adapter weights)
        if isinstance(bait_layer.mlp.down_proj, LinearWithAdapters):
            bait_layer.mlp.down_proj = bait_layer.mlp.down_proj.base_linear
        recovery_o_proj = get_layer_module(recovery_layer, "attn.o_proj")
        if isinstance(recovery_o_proj, LinearWithAdapters):
            set_layer_module(recovery_layer, "attn.o_proj", recovery_o_proj.base_linear)
        # Second recovery site
        if isinstance(recovery_layer.mlp.down_proj, LinearWithAdapters):
            recovery_layer.mlp.down_proj = recovery_layer.mlp.down_proj.base_linear
    controller.wrappers.clear()
    controller.recovery_modules.clear()
    controller.bait_modules.clear()
    controller.installed_layers.clear()


def run_spectral_sweep(
    model: nn.Module,
    tokenizer,
    priors: list[LayerPrior],
    good_prompts: list[Prompt],
    bad_prompts: list[Prompt],
    batch_size: int,
    n_bands: int = 6,
) -> dict:
    """Sweep across spectral bands and measure δ(x) diagnostic quality for each.

    Tests bands from strongest to weakest in equal intervals, reporting which
    band best balances statistical separability vs. behavioral sensitivity.
    No training is performed — this is a pure diagnostic.
    """
    band_edges = [(i / n_bands, (i + 1) / n_bands) for i in range(n_bands)]
    sweep_results = {}

    for band_idx, (pct_low, pct_high) in enumerate(band_edges):
        band_label = f"band_{band_idx}_{pct_low:.2f}_{pct_high:.2f}"
        print(f"\n  Sweep band {band_idx + 1}/{n_bands}: percentile [{pct_low:.0%}, {pct_high:.0%}]")

        controller = DefenseController(
            model,
            priors,
            unfreeze_bait=False,
            bait_subspace_mode="mid",
            bait_sv_pct_low=pct_low,
            bait_sv_pct_high=pct_high,
            bait_basis_trainable=False,
        )

        diagnostics = evaluate_bait_delta_diagnostics(
            model, tokenizer, controller, good_prompts, bad_prompts, batch_size,
        )

        eval_prompts = good_prompts[:min(16, len(good_prompts))]
        kl = evaluate_kl(model, tokenizer, controller, eval_prompts, batch_size)

        sweep_results[band_label] = {
            "pct_low": pct_low,
            "pct_high": pct_high,
            "kl_divergence": kl,
            "diagnostics": diagnostics,
        }

        # Cleanly remove adapters without merging weights into the base model
        _uninstall_adapters(model, controller)

    # Print sweep summary
    sweep_table = Table(title="Spectral Sweep: band → δ(x) quality (cross-layer)")
    sweep_table.add_column("Band", justify="left")
    sweep_table.add_column("Percentile", justify="center")
    sweep_table.add_column("KL div", justify="right")
    sweep_table.add_column("||μ_h-μ_b||", justify="right")
    sweep_table.add_column("δ Fisher", justify="right")
    sweep_table.add_column("cos(δ,refusal)", justify="right")
    sweep_table.add_column("||δ||", justify="right")

    for band_label, result in sweep_results.items():
        for layer_key, d in result["diagnostics"].items():
            sweep_table.add_row(
                f"{band_label}",
                f"[{result['pct_low']:.0%}, {result['pct_high']:.0%}]",
                f"{result['kl_divergence']:.6f}",
                f"{d['mean_diff_norm']:.4f}",
                f"{d['delta_fisher_ratio']:.6f}",
                f"{d['cos_with_refusal_dir']:.4f}",
                f"{d['mean_delta_norm']:.4f}",
            )
    print(sweep_table)
    return sweep_results


def train_recovery_progressive(
    model: nn.Module,
    tokenizer,
    controller: DefenseController,
    train_prompts: list[Prompt],
    args: argparse.Namespace,
    train_good_prompts: list[Prompt] | None = None,
    train_bad_prompts: list[Prompt] | None = None,
    supervised_residual_dirs: dict[int, torch.Tensor] | None = None,
    external_direction_refs: dict[str, list[dict]] | None = None,
) -> list[dict[str, float]]:
    """Progressive shallow-to-deep multi-layer training.

    Algorithm:
      For each installed (bait, recovery) pair, sorted by bait_layer_idx ascending:
        1. Freeze all OTHER pairs.
        2. Train only this pair for --progressive-stages epochs.
        3. Check per-stage KL: if KL > --progressive-kl-budget, log a warning
           (but do not stop — the budget is informational at this stage).
        4. Freeze this pair; move to next.

    After all stages, run one final joint fine-tuning pass (--n-epochs total
    epochs split across all pairs) to allow inter-pair co-adaptation.

    This prevents KL from accumulating super-linearly and isolates layer-to-layer
    interference (TODO-1.3 / TODO-1.4).
    """
    history: list[dict[str, float]] = []
    n_stages = getattr(args, "progressive_stages", 3)
    kl_budget = getattr(args, "progressive_kl_budget", 0.005)
    cumulative_stages = getattr(args, "progressive_cumulative_stages", False)
    skip_joint = getattr(args, "skip_progressive_joint_finetune", False)

    sorted_installed = sorted(controller.installed_layers, key=lambda x: x.bait_layer_idx)
    initial_trainability = {
        id(param): param.requires_grad
        for module in [*controller.bait_modules, *controller.recovery_modules]
        for param in module.parameters()
    }

    def installed_index(installed: InstalledLayerDefense) -> int:
        for idx, candidate in enumerate(controller.installed_layers):
            if candidate is installed:
                return idx
        return controller.installed_layers.index(installed)

    def pair_modules(installed: InstalledLayerDefense) -> list[nn.Module]:
        idx = installed_index(installed)
        rec_start = 2 * idx
        rec_end = rec_start + 2
        return [controller.bait_modules[idx], *controller.recovery_modules[rec_start:rec_end]]

    def set_pair_trainable(installed: InstalledLayerDefense, trainable: bool) -> None:
        for module in pair_modules(installed):
            for param in module.parameters():
                param.requires_grad_(trainable and initial_trainability.get(id(param), False))

    def set_all_pairs_trainable(trainable: bool) -> None:
        for installed in controller.installed_layers:
            set_pair_trainable(installed, trainable)

    def set_pair_enabled(installed: InstalledLayerDefense, enabled: bool) -> None:
        layers = get_layers(model)
        bait_layer = layers[installed.bait_layer_idx]
        recovery_layer = layers[installed.recovery_layer_idx]
        if isinstance(bait_layer.mlp.down_proj, LinearWithAdapters):
            bait_layer.mlp.down_proj.adapters_enabled = enabled
        recovery_o_proj = get_layer_module(recovery_layer, "attn.o_proj")
        if isinstance(recovery_o_proj, LinearWithAdapters):
            recovery_o_proj.adapters_enabled = enabled
        if isinstance(recovery_layer.mlp.down_proj, LinearWithAdapters):
            recovery_layer.mlp.down_proj.adapters_enabled = enabled

    def guard_budget_from(raw_budget: float) -> float:
        budget = float(raw_budget or 0.0)
        if budget <= 0.0:
            budget = float(getattr(args, "final_kl_budget", 0.0) or 0.0)
        if budget <= 0.0:
            budget = float(getattr(args, "progressive_kl_budget", 0.0) or 0.0)
        return budget

    def restore_snapshot(snapshot: list[tuple[nn.Parameter, torch.Tensor]]) -> None:
        with torch.no_grad():
            for param, saved in snapshot:
                param.copy_(saved)

    def apply_snapshot_interpolation(
        snapshot: list[tuple[nn.Parameter, torch.Tensor]],
        updated_tensors: list[torch.Tensor],
        alpha: float,
    ) -> None:
        alpha = float(alpha)
        with torch.no_grad():
            for (param, saved), updated in zip(snapshot, updated_tensors):
                param.copy_(saved + (updated - saved) * alpha)

    def guard_update_with_interpolation(
        *,
        label: str,
        snapshot: list[tuple[nn.Parameter, torch.Tensor]],
        history_records: list[dict[str, float]],
        rollback_budget: float,
        interpolation_steps: int,
    ) -> list[dict[str, float]]:
        if rollback_budget <= 0.0:
            return history_records

        controller.set_enabled(True)
        eval_prompts = train_prompts[: min(32, len(train_prompts))]
        guard_kl = evaluate_kl(model, tokenizer, controller, eval_prompts, args.batch_size)
        if math.isfinite(guard_kl) and guard_kl <= rollback_budget:
            print(f"  {label} guard KL = {guard_kl:.6f} (budget={rollback_budget:.6f} ✅)")
            for rec in history_records:
                rec["joint_guard_kl"] = guard_kl
            return history_records

        updated_tensors = [param.detach().clone() for param, _ in snapshot]
        accepted_alpha = 0.0
        accepted_kl = float("inf")
        interpolation_steps = max(0, int(interpolation_steps))
        if interpolation_steps > 0:
            lo = 0.0
            hi = 1.0
            for _ in range(interpolation_steps):
                mid = 0.5 * (lo + hi)
                apply_snapshot_interpolation(snapshot, updated_tensors, mid)
                mid_kl = evaluate_kl(model, tokenizer, controller, eval_prompts, args.batch_size)
                if math.isfinite(mid_kl) and mid_kl <= rollback_budget:
                    lo = mid
                    accepted_alpha = mid
                    accepted_kl = mid_kl
                else:
                    hi = mid
        if accepted_alpha > 0.0:
            apply_snapshot_interpolation(snapshot, updated_tensors, accepted_alpha)
            print(
                f"\n  [bold yellow]WARNING[/]: {label} exceeded guard KL={guard_kl:.6f} "
                f"(budget={rollback_budget:.6f}); kept partial update "
                f"alpha={accepted_alpha:.4f} with guard KL={accepted_kl:.6f}."
            )
            for rec in history_records:
                rec["joint_interpolation_alpha"] = accepted_alpha
                rec["joint_guard_kl"] = accepted_kl
            return history_records

        restore_snapshot(snapshot)
        print(
            f"\n  [bold yellow]WARNING[/]: {label} rolled back because guard "
            f"KL={guard_kl:.6f} exceeded budget={rollback_budget:.6f}, and no "
            "safe interpolation point was found."
        )
        return []

    # ── Stage 1: train each pair in isolation, shallow → deep ──
    for stage_idx, installed in enumerate(sorted_installed):
        pair_label = f"L{installed.bait_layer_idx}→L{installed.recovery_layer_idx}"
        print(f"\n  [bold]Progressive stage {stage_idx + 1}/{len(sorted_installed)}[/]: pair {pair_label}")

        # Enable only the current pair by default. In cumulative mode, keep
        # earlier pairs enabled but frozen so deeper pairs train on top of the
        # already-installed defense.
        controller.set_enabled(False)
        set_all_pairs_trainable(False)

        if cumulative_stages:
            for previous in sorted_installed[:stage_idx]:
                set_pair_enabled(previous, True)
        set_pair_enabled(installed, True)
        set_pair_trainable(installed, True)

        # Temporarily adjust n_epochs for this stage
        stage_args = argparse.Namespace(**vars(args))
        stage_args.n_epochs = n_stages

        stage_history = train_recovery(
            model, tokenizer, controller, train_prompts, stage_args,
            train_good_prompts=train_good_prompts,
            train_bad_prompts=train_bad_prompts,
            supervised_residual_dirs=supervised_residual_dirs,
            active_installed_layers=[installed],
            external_direction_refs=external_direction_refs,
        )

        # Measure KL after this stage
        eval_prompts = train_prompts[: min(32, len(train_prompts))]
        stage_kl = evaluate_kl(model, tokenizer, controller, eval_prompts, args.batch_size)
        print(
            f"  Stage {stage_idx + 1} KL = {stage_kl:.6f} "
            f"(budget={kl_budget:.6f} {'✅' if stage_kl <= kl_budget else '⚠️ over budget'})"
        )

        for rec in stage_history:
            rec["stage"] = stage_idx + 1
            rec["pair"] = pair_label
            rec["stage_kl"] = stage_kl
        history.extend(stage_history)

        # Lock the whole pair after its stage. This includes sidechannel/refusal
        # bait bases, which are trainable when --freeze-bait-coeff is used.
        set_pair_trainable(installed, False)

    # ── Optional Stage 1.5: block-wise joint fine-tuning ──
    # This is a controlled version of the old full joint pass: only a contiguous
    # block of installed pairs is enabled/trainable at once, then the update is
    # guarded against full-defense KL drift.
    block_joint_epochs = max(0, int(getattr(args, "progressive_block_joint_epochs", 0)))
    if block_joint_epochs > 0 and sorted_installed:
        raw_block_size = int(getattr(args, "progressive_block_joint_size", 0))
        block_size = raw_block_size if raw_block_size > 0 else len(sorted_installed)
        block_size = max(1, min(block_size, len(sorted_installed)))
        raw_stride = int(getattr(args, "progressive_block_joint_stride", 0))
        block_stride = raw_stride if raw_stride > 0 else block_size
        block_stride = max(1, block_stride)
        block_lr_scale = float(getattr(args, "progressive_block_joint_lr_scale", 1.0))
        block_rollback_budget = guard_budget_from(
            float(getattr(args, "progressive_block_joint_kl_rollback_budget", 0.0) or 0.0)
        )
        block_interp_steps = int(getattr(args, "progressive_block_joint_interpolation_steps", 8))
        block_detach_visible_bait = bool(
            getattr(args, "progressive_block_joint_detach_visible_bait", True)
        )
        block_disable_local_losses = bool(
            getattr(args, "progressive_block_joint_disable_local_losses", False)
        )
        block_mode = "geometry/global-only" if block_disable_local_losses else "behavior+geometry"
        block_starts = list(range(0, len(sorted_installed), block_stride))
        print(
            f"\n  [bold]Block-wise joint fine-tuning[/]: "
            f"{len(block_starts)} block(s), size={block_size}, stride={block_stride}, "
            f"epochs/block={block_joint_epochs}, lr×{block_lr_scale:.3g}, "
            f"guard={block_rollback_budget:.6f}, "
            f"mode={block_mode}, detach_visible_bait={int(block_detach_visible_bait)}"
        )
        for block_idx, start in enumerate(block_starts):
            block = sorted_installed[start: start + block_size]
            if not block:
                continue
            pair_labels = ", ".join(
                f"L{inst.bait_layer_idx}→L{inst.recovery_layer_idx}" for inst in block
            )
            print(f"\n  [bold]Block joint {block_idx + 1}/{len(block_starts)}[/]: {pair_labels}")
            controller.set_enabled(False)
            set_all_pairs_trainable(False)
            for inst in block:
                set_pair_enabled(inst, True)
                set_pair_trainable(inst, True)

            block_args = argparse.Namespace(**vars(args))
            block_args.n_epochs = block_joint_epochs
            block_args.lr = float(args.lr) * block_lr_scale
            if block_detach_visible_bait:
                block_args.detach_visible_loss_from_bait = True
            if block_disable_local_losses:
                block_args.kl_loss_weight = 0.0
                block_args.recovery_loss_weight = 0.0
                block_args.visible_loss_weight = 0.0
                block_args.recovery_max_weight = 0.0
                block_args.visible_max_weight = 0.0
            block_snapshot = [
                (param, param.detach().clone())
                for param in controller.trainable_parameters()
            ]
            try:
                block_history = train_recovery(
                    model, tokenizer, controller, train_prompts, block_args,
                    train_good_prompts=train_good_prompts,
                    train_bad_prompts=train_bad_prompts,
                    supervised_residual_dirs=supervised_residual_dirs,
                    active_installed_layers=block,
                    external_direction_refs=external_direction_refs,
                )
                block_history = guard_update_with_interpolation(
                    label=f"block joint {block_idx + 1}/{len(block_starts)}",
                    snapshot=block_snapshot,
                    history_records=block_history,
                    rollback_budget=block_rollback_budget,
                    interpolation_steps=block_interp_steps,
                )
            except FloatingPointError as exc:
                restore_snapshot(block_snapshot)
                if bool(getattr(args, "progressive_block_joint_geometry_fallback", True)):
                    print(
                        "\n  [bold yellow]WARNING[/]: block-wise behavior+geometry "
                        "fine-tuning hit a numerical instability; retrying this block "
                        f"with geometry/global losses only. {exc}"
                    )
                    set_all_pairs_trainable(False)
                    for inst in block:
                        set_pair_enabled(inst, True)
                        set_pair_trainable(inst, True)

                    fallback_args = argparse.Namespace(**vars(block_args))
                    fallback_args.kl_loss_weight = 0.0
                    fallback_args.recovery_loss_weight = 0.0
                    fallback_args.visible_loss_weight = 0.0
                    fallback_args.recovery_max_weight = 0.0
                    fallback_args.visible_max_weight = 0.0
                    try:
                        block_history = train_recovery(
                            model, tokenizer, controller, train_prompts, fallback_args,
                            train_good_prompts=train_good_prompts,
                            train_bad_prompts=train_bad_prompts,
                            supervised_residual_dirs=supervised_residual_dirs,
                            active_installed_layers=block,
                            external_direction_refs=external_direction_refs,
                        )
                        block_history = guard_update_with_interpolation(
                            label=(
                                f"block joint {block_idx + 1}/{len(block_starts)} "
                                "geometry fallback"
                            ),
                            snapshot=block_snapshot,
                            history_records=block_history,
                            rollback_budget=block_rollback_budget,
                            interpolation_steps=block_interp_steps,
                        )
                        for rec in block_history:
                            rec["block_joint_fallback"] = "geometry_only"
                    except FloatingPointError as fallback_exc:
                        restore_snapshot(block_snapshot)
                        print(
                            "\n  [bold yellow]WARNING[/]: block-wise geometry fallback "
                            "also hit a numerical instability and was skipped for this "
                            f"block. {fallback_exc}"
                        )
                        block_history = []
                else:
                    print(
                        "\n  [bold yellow]WARNING[/]: block-wise joint fine-tuning hit a "
                        f"numerical instability and was skipped for this block. {exc}"
                    )
                    block_history = []
            for rec in block_history:
                rec["stage"] = "block_joint"
                rec["block"] = block_idx + 1
                rec["block_pairs"] = pair_labels
            history.extend(block_history)
            set_all_pairs_trainable(False)
        controller.set_enabled(True)

    # ── Stage 2: joint fine-tuning (remaining epochs) ──
    # Only run joint phase when there are enough epochs left to be meaningful.
    # With n_epochs=15 and 11 pairs × 3 stages = 33 stage-epochs, remaining is
    # negative → forcing 1 epoch with all 11 pairs enabled causes KL to explode.
    # We skip joint phase if remaining_epochs < n_stages (not worth the risk).
    remaining_epochs = args.n_epochs - n_stages * len(sorted_installed)
    if skip_joint:
        geometry_epochs = max(0, int(getattr(args, "progressive_geometry_joint_epochs", 0)))
        if geometry_epochs > 0:
            lr_scale = float(getattr(args, "progressive_geometry_joint_lr_scale", 1.0))
            include_local_losses = bool(
                getattr(args, "progressive_geometry_joint_include_local_losses", False)
            )
            pass_kind = "behavior+geometry" if include_local_losses else "geometry-only"
            print(
                f"\n  [bold]All-pairs geometry pass ({geometry_epochs} epoch(s))[/]: "
                "--skip-progressive-joint-finetune set, so running only the short "
                f"post-progressive {pass_kind} pass at lr×{lr_scale:.3g}."
            )
            controller.set_enabled(True)
            set_all_pairs_trainable(True)
            geometry_args = argparse.Namespace(**vars(args))
            geometry_args.n_epochs = geometry_epochs
            geometry_args.lr = float(args.lr) * lr_scale
            geometry_param_snapshot = [
                (param, param.detach().clone())
                for param in controller.trainable_parameters()
            ]

            def _restore_geometry_snapshot() -> None:
                with torch.no_grad():
                    for param, saved in geometry_param_snapshot:
                        param.copy_(saved)

            def _apply_geometry_interpolation(
                alpha: float,
                updated_tensors: list[torch.Tensor],
            ) -> None:
                alpha = float(alpha)
                with torch.no_grad():
                    for (param, saved), updated in zip(geometry_param_snapshot, updated_tensors):
                        param.copy_(saved + (updated - saved) * alpha)

            if not include_local_losses:
                geometry_kl_anchor_weight = max(
                    0.0,
                    float(
                        getattr(args, "progressive_geometry_joint_kl_loss_weight", 0.0)
                        or 0.0
                    ),
                )
                geometry_args.kl_loss_weight = geometry_kl_anchor_weight
                geometry_args.recovery_loss_weight = 0.0
                geometry_args.visible_loss_weight = 0.0
                geometry_args.auto_normalize_loss_weights = False
                geometry_args.detach_kl_loss_from_bait = True
                geometry_args.detach_recovery_loss_from_bait = True
                geometry_args.detach_visible_loss_from_bait = True
                geometry_args.max_grad_norm = min(float(args.max_grad_norm), 0.1)
                if not bool(getattr(args, "progressive_geometry_joint_train_bait", False)):
                    for bait_module in controller.bait_modules:
                        for param in bait_module.parameters():
                            param.requires_grad_(False)
                    geometry_args.bait_anti_coherence_loss_weight = 0.0
                print(
                    "  Geometry pass local losses disabled: "
                    "recovery/visible=0, auto-normalization off, "
                    f"KL anchor={geometry_kl_anchor_weight:.3g}, "
                    f"max_grad_norm={geometry_args.max_grad_norm:.3g}, "
                    "bait="
                    + (
                        "trainable."
                        if bool(getattr(args, "progressive_geometry_joint_train_bait", False))
                        else "frozen."
                    )
                )
            try:
                geometry_history = train_recovery(
                    model, tokenizer, controller, train_prompts, geometry_args,
                    train_good_prompts=train_good_prompts,
                    train_bad_prompts=train_bad_prompts,
                    supervised_residual_dirs=supervised_residual_dirs,
                    external_direction_refs=external_direction_refs,
                )
                rollback_budget = float(
                    getattr(args, "progressive_geometry_joint_kl_rollback_budget", 0.0)
                    or 0.0
                )
                if rollback_budget <= 0.0:
                    rollback_budget = float(getattr(args, "final_kl_budget", 0.0) or 0.0)
                if rollback_budget <= 0.0:
                    rollback_budget = float(getattr(args, "progressive_kl_budget", 0.0) or 0.0)
                if rollback_budget > 0.0:
                    eval_prompts = train_prompts[: min(32, len(train_prompts))]
                    geometry_kl = evaluate_kl(
                        model, tokenizer, controller, eval_prompts, args.batch_size
                    )
                    if (not math.isfinite(geometry_kl)) or geometry_kl > rollback_budget:
                        updated_tensors = [
                            param.detach().clone()
                            for param, _ in geometry_param_snapshot
                        ]
                        interpolation_steps = max(
                            0,
                            int(
                                getattr(
                                    args,
                                    "progressive_geometry_joint_interpolation_steps",
                                    8,
                                )
                            ),
                        )
                        accepted_alpha = 0.0
                        accepted_kl = float("inf")
                        if interpolation_steps > 0:
                            lo = 0.0
                            hi = 1.0
                            for _ in range(interpolation_steps):
                                mid = 0.5 * (lo + hi)
                                _apply_geometry_interpolation(mid, updated_tensors)
                                mid_kl = evaluate_kl(
                                    model, tokenizer, controller, eval_prompts, args.batch_size
                                )
                                if math.isfinite(mid_kl) and mid_kl <= rollback_budget:
                                    lo = mid
                                    accepted_alpha = mid
                                    accepted_kl = mid_kl
                                else:
                                    hi = mid
                        if accepted_alpha > 0.0:
                            _apply_geometry_interpolation(accepted_alpha, updated_tensors)
                            print(
                                "\n  [bold yellow]WARNING[/]: all-pairs geometry pass "
                                f"exceeded guard KL={geometry_kl:.6f} "
                                f"(budget={rollback_budget:.6f}); kept partial update "
                                f"alpha={accepted_alpha:.4f} with guard KL={accepted_kl:.6f}."
                            )
                            for rec in geometry_history:
                                rec["geometry_interpolation_alpha"] = accepted_alpha
                                rec["geometry_guard_kl"] = accepted_kl
                        else:
                            _restore_geometry_snapshot()
                            print(
                                "\n  [bold yellow]WARNING[/]: all-pairs geometry pass "
                                f"rolled back because guard KL={geometry_kl:.6f} exceeded "
                                f"budget={rollback_budget:.6f}, and no safe interpolation "
                                "point was found."
                            )
                            geometry_history = []
                    else:
                        print(
                            f"  Geometry pass guard KL = {geometry_kl:.6f} "
                            f"(budget={rollback_budget:.6f} ✅)"
                        )
            except FloatingPointError as exc:
                _restore_geometry_snapshot()
                print(
                    "\n  [bold yellow]WARNING[/]: all-pairs geometry pass hit a "
                    f"numerical instability and was skipped. {exc}"
                )
                geometry_history = []
            for rec in geometry_history:
                rec["stage"] = "geometry_joint"
            history.extend(geometry_history)
        else:
            print("\n  [yellow]Skipping joint fine-tuning[/]: --skip-progressive-joint-finetune set.")
        controller.set_enabled(True)
        set_all_pairs_trainable(False)
    elif remaining_epochs >= n_stages:
        print(f"\n  [bold]Joint fine-tuning ({remaining_epochs} epochs)[/]: all pairs enabled")
        controller.set_enabled(True)
        set_all_pairs_trainable(True)
        joint_args = argparse.Namespace(**vars(args))
        joint_args.n_epochs = remaining_epochs
        joint_param_snapshot = [
            (param, param.detach().clone())
            for param in controller.trainable_parameters()
        ]
        try:
            joint_history = train_recovery(
                model, tokenizer, controller, train_prompts, joint_args,
                train_good_prompts=train_good_prompts,
                train_bad_prompts=train_bad_prompts,
                supervised_residual_dirs=supervised_residual_dirs,
                external_direction_refs=external_direction_refs,
            )
        except FloatingPointError as exc:
            with torch.no_grad():
                for param, saved in joint_param_snapshot:
                    param.copy_(saved)
            print(
                "\n  [bold yellow]WARNING[/]: joint fine-tuning hit a numerical "
                f"instability and was skipped. Keeping progressive-stage weights. {exc}"
            )
            joint_history = []
        for rec in joint_history:
            rec["stage"] = "joint"
        history.extend(joint_history)
        set_all_pairs_trainable(False)
    else:
        if remaining_epochs > 0:
            print(
                f"\n  [yellow]Skipping joint fine-tuning[/]: only {remaining_epochs} epoch(s) "
                f"remaining (need ≥ {n_stages} to avoid KL blowup with {len(sorted_installed)} pairs). "
                "Increase --n-epochs or reduce --progressive-stages to enable."
            )
        else:
            print(
                f"\n  [yellow]Skipping joint fine-tuning[/]: stage budget "
                f"({n_stages} × {len(sorted_installed)} pairs = {n_stages * len(sorted_installed)} epochs) "
                f"already exceeds --n-epochs={args.n_epochs}."
            )
        controller.set_enabled(True)

    return history


def _config_num_hidden_layers(config) -> int | None:
    for name in ("num_hidden_layers", "n_layer", "num_layers"):
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    text_config = getattr(config, "text_config", None)
    if isinstance(text_config, dict):
        for name in ("num_hidden_layers", "n_layer", "num_layers"):
            value = text_config.get(name)
            if isinstance(value, int) and value > 0:
                return value
    elif text_config is not None:
        for name in ("num_hidden_layers", "n_layer", "num_layers"):
            value = getattr(text_config, name, None)
            if isinstance(value, int) and value > 0:
                return value
    return None


def main() -> None:
    args = parse_args()
    if args.visible_shift_max > 0.0 and args.visible_shift_max <= args.visible_shift_target:
        raise ValueError(
            "--visible-shift-max must be greater than --visible-shift-target when enabled"
        )
    if args.bait_calibration_log_max < args.bait_calibration_log_min:
        raise ValueError("--bait-calibration-log-max must be >= --bait-calibration-log-min")
    if not 0.0 <= args.bait_calibration_quantile <= 1.0:
        raise ValueError("--bait-calibration-quantile must be in [0, 1]")
    if args.bait_calibration_kl_target < 0.0:
        raise ValueError("--bait-calibration-kl-target must be >= 0")
    if args.bait_calibration_kl_iters < 0:
        raise ValueError("--bait-calibration-kl-iters must be >= 0")
    if not 0.0 <= args.sidechannel_tag_trigger_mix <= 1.0:
        raise ValueError("--sidechannel-tag-trigger-mix must be in [0, 1]")
    if args.residual_fisher_denom_floor <= 0.0:
        raise ValueError("--residual-fisher-denom-floor must be > 0")
    if args.bait_anti_coherence_loss_weight < 0.0:
        raise ValueError("--bait-anti-coherence-loss-weight must be >= 0")
    if args.bait_anti_coherence_components < 1:
        raise ValueError("--bait-anti-coherence-components must be >= 1")
    if not 0.0 <= args.bait_anti_coherence_margin <= 1.0:
        raise ValueError("--bait-anti-coherence-margin must be in [0, 1]")
    if args.residual_gap_anti_coherence_loss_weight < 0.0:
        raise ValueError("--residual-gap-anti-coherence-loss-weight must be >= 0")
    if not 0.0 <= args.residual_gap_anti_coherence_margin <= 1.0:
        raise ValueError("--residual-gap-anti-coherence-margin must be in [0, 1]")
    if args.global_direction_loss_weight < 0.0:
        raise ValueError("--global-direction-loss-weight must be >= 0")
    if args.global_direction_samples < 1:
        raise ValueError("--global-direction-samples must be >= 1")
    if not 0.0 <= args.global_direction_margin <= 1.0:
        raise ValueError("--global-direction-margin must be in [0, 1]")
    if not 0.0 <= args.global_direction_range_low <= 1.0:
        raise ValueError("--global-direction-range-low must be in [0, 1]")
    if not 0.0 <= args.global_direction_range_high <= 1.0:
        raise ValueError("--global-direction-range-high must be in [0, 1]")
    if args.progressive_geometry_joint_epochs < 0:
        raise ValueError("--progressive-geometry-joint-epochs must be >= 0")
    if args.progressive_geometry_joint_lr_scale <= 0.0:
        raise ValueError("--progressive-geometry-joint-lr-scale must be > 0")
    if args.progressive_geometry_joint_kl_rollback_budget < 0.0:
        raise ValueError("--progressive-geometry-joint-kl-rollback-budget must be >= 0")
    if args.progressive_geometry_joint_kl_loss_weight < 0.0:
        raise ValueError("--progressive-geometry-joint-kl-loss-weight must be >= 0")
    if args.progressive_geometry_joint_interpolation_steps < 0:
        raise ValueError("--progressive-geometry-joint-interpolation-steps must be >= 0")
    if args.progressive_block_joint_epochs < 0:
        raise ValueError("--progressive-block-joint-epochs must be >= 0")
    if args.progressive_block_joint_size < 0:
        raise ValueError("--progressive-block-joint-size must be >= 0")
    if args.progressive_block_joint_stride < 0:
        raise ValueError("--progressive-block-joint-stride must be >= 0")
    if args.progressive_block_joint_lr_scale <= 0.0:
        raise ValueError("--progressive-block-joint-lr-scale must be > 0")
    if args.progressive_block_joint_kl_rollback_budget < 0.0:
        raise ValueError("--progressive-block-joint-kl-rollback-budget must be >= 0")
    if args.progressive_block_joint_interpolation_steps < 0:
        raise ValueError("--progressive-block-joint-interpolation-steps must be >= 0")
    if args.loss_normalization_target_ratio <= 0.0:
        raise ValueError("--loss-normalization-target-ratio must be > 0")
    if args.loss_normalization_batch_size < 0:
        raise ValueError("--loss-normalization-batch-size must be >= 0")
    if args.heretic_coverage_layers < 0:
        raise ValueError("--heretic-coverage-layers must be >= 0")
    seed_everything(args.seed)
    device = resolve_device(args.device)

    print(f"Loading step1 design prior from [bold]{args.step1_path}[/]...")
    step1_summary = load_step1_summary(args.step1_path)

    # Determine max layer index: need l+1 for recovery, so bait layer must be < n_layers-1.
    # We peek at the model config to find n_layers without loading the full model yet.
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    n_model_layers = _config_num_hidden_layers(model_config)
    max_bait_layer = (n_model_layers - 1) if n_model_layers else None
    if max_bait_layer is not None:
        print(f"  Model has {n_model_layers} layers → bait layers must be < {max_bait_layer}")

    # ── Layer selection: Heretic coverage mode or conventional mode ──
    use_heretic_coverage = getattr(args, "cover_heretic_layers", False)
    if use_heretic_coverage and n_model_layers is not None:
        print()
        print("[bold]Multi-layer mode[/]: selecting layers to match Heretic's traversal...")
        target_layers = select_heretic_coverage_layers(
            step1_summary,
            n_model_layers=n_model_layers,
            args=args,
        )
        print(f"  → {len(target_layers)} bait/recovery pairs will be installed: {target_layers}")
    else:
        target_layers = select_target_layers(
            step1_summary,
            requested_layers=parse_requested_layers(args.target_layers),
            num_target_layers=args.num_target_layers,
            args=args,
            max_layer_idx=max_bait_layer,
        )
    priors = build_layer_priors(step1_summary, target_layers, args)
    print_design_summary(priors)

    train_data_source = args.train_data_source.strip() or args.eval_data_source
    train_good_data_source = args.train_good_data_source.strip() or train_data_source
    train_bad_data_source = args.train_bad_data_source.strip() or train_data_source
    general_data_source = args.general_data_source.strip() or train_data_source
    print(
        "Prompt data sources: "
        f"eval={args.eval_data_source}, train={train_data_source}, "
        f"train_good={train_good_data_source}, train_bad={train_bad_data_source}, "
        f"general={general_data_source}"
    )

    # Load eval prompts for diagnostics. Direction initialization uses the
    # non-overlapping training split loaded below.
    good_prompts, bad_prompts = load_prompts_simple(
        args.n_good,
        args.n_bad,
        source=args.eval_data_source,
    )

    original_clean_fisher_profile = None
    original_clean_model_name = args.original_clean_model.strip()
    if original_clean_model_name:
        original_clean_tokenizer = (
            args.original_clean_tokenizer.strip()
            if args.original_clean_tokenizer.strip()
            else None
        )
        original_clean_fisher_profile = evaluate_original_clean_fisher_profile(
            original_clean_model_name,
            original_clean_tokenizer,
            device,
            args.gpu_mode,
            good_prompts,
            bad_prompts,
            args.batch_size,
        )

    tokenizer_source = args.tokenizer.strip() if args.tokenizer.strip() else None
    model, tokenizer = load_model(
        args.model,
        device,
        tokenizer_name=tokenizer_source,
        gpu_mode=args.gpu_mode,
    )
    for param in model.parameters():
        param.requires_grad_(False)

    if args.bait_output_mode == "random_orthogonal":
        train_good_prompts: list[Prompt] = []
        train_bad_prompts: list[Prompt] = []
        supervised_directions = None
        print(
            "Using unsupervised random_orthogonal bait: skipping supervised "
            "good/bad bait-trigger directions and labeled recovery batches."
        )
    else:
        # Load separate training splits for good and bad prompts.
        # By default offset = n_good / n_bad so train and eval don't overlap.
        good_offset = args.train_good_offset if args.train_good_offset >= 0 else args.n_good
        bad_offset  = args.train_bad_offset  if args.train_bad_offset  >= 0 else args.n_bad
        print("Loading training-split prompts (non-overlapping with eval)...")
        train_good_prompts, train_bad_prompts = load_prompts_split_sources(
            args.n_train_good,
            args.n_train_bad,
            good_source=train_good_data_source,
            bad_source=train_bad_data_source,
            good_offset=good_offset,
            bad_offset=bad_offset,
        )
        print(
            f"  Training prompts: {len(train_good_prompts)} good"
            f" (offset={good_offset}) + {len(train_bad_prompts)} bad (offset={bad_offset})"
        )

        if not train_good_prompts or not train_bad_prompts:
            raise ValueError(
                "Supervised bait output modes require non-empty training good/bad "
                "prompts for direction initialization. Set --n-train-good and "
                "--n-train-bad to positive values."
            )

        # Compute supervised refusal directions at module inputs from the
        # training split, keeping eval prompts independent.
        print("Computing supervised input directions from training-split prompts...")
        supervised_directions = compute_supervised_input_directions(
            model,
            tokenizer,
            train_good_prompts,
            train_bad_prompts,
            layer_indices=target_layers,
            module_name="mlp.down_proj",
            batch_size=args.batch_size,
        )

    # --- Sweep mode: diagnostic only, no training ---
    if args.bait_subspace_mode == "sweep":
        print("\nRunning spectral sweep diagnostic (no training)...")
        sweep_results = run_spectral_sweep(
            model, tokenizer, priors, good_prompts, bad_prompts, args.batch_size,
        )
        _exp_tag = getattr(args, "experiment_tag", "").strip()
        model_short = _exp_tag if _exp_tag else args.model.replace("/", "_")
        sweep_path = RESULTS_DIR / f"step4_sweep_{model_short}.json"
        with open(sweep_path, "w") as f:
            json.dump(sweep_results, f, indent=2)
        print(f"\nSaved sweep results to [bold]{sweep_path}[/]")
        return

    # Cross-layer deployment:
    # bait on layer l `down_proj`, recovery on layer l+1 `o_proj` + `down_proj`.
    # --freeze-bait-coeff: in refusal_cancel mode the fake direction is set at init;
    # allowing coeff to train lets the optimizer undo it. Freeze A to prevent this.
    if args.freeze_bait_coeff:
        args.unfreeze_bait = False
        print(
            "  [bold yellow]Note[/]: --freeze-bait-coeff set → bait coeff (A) frozen. "
            "Use --freeze-bait/--freeze-bait-gate to also freeze B/gate."
        )
    unfreeze_bait = args.unfreeze_bait and not args.freeze_bait
    # --freeze-bait forces BOTH coeff AND basis frozen. In refusal_cancel mode
    # this is the only way to stop the optimizer from flipping B[:,0] from
    # -r_clean to +r_clean (basis_regularization only guards the subspace, not
    # the sign). force_freeze_basis is an explicit signal that bypasses the
    # auto-unfreeze at _install line ~1245.
    force_freeze_basis = bool(args.freeze_bait)
    freeze_bait_gate = bool(args.freeze_bait_gate)
    if force_freeze_basis:
        print(
            "  [bold yellow]Note[/]: --freeze-bait set → bait basis (B) hard-frozen "
            "at init."
        )
    if freeze_bait_gate:
        print(
            "  [bold yellow]Note[/]: --freeze-bait-gate set → bait_gate is fixed "
            "after init/calibration; recovery adapters carry the training gradient."
        )

    # For 'refusal_cancel' mode: compute the Heretic-style refusal direction in
    # residual-stream space at each bait layer's output position (l+1).
    # This uses the CLEAN model (no adapters yet), so the direction is ground-truth.
    supervised_residual_dirs: dict[int, torch.Tensor] | None = None
    if args.bait_output_mode in ("refusal_cancel", "sidechannel_cancel"):
        bait_layer_indices = [p.layer_idx for p in priors]
        print(
            f"\nComputing supervised residual refusal directions for '{args.bait_output_mode}' mode "
            f"(layers {bait_layer_indices})..."
        )
        supervised_residual_dirs = compute_supervised_residual_directions(
            model, tokenizer,
            good_prompts=train_good_prompts[: min(64, len(train_good_prompts))],
            bad_prompts=train_bad_prompts[: min(64, len(train_bad_prompts))],
            layer_indices=bait_layer_indices,
            batch_size=args.batch_size,
        )

    controller = DefenseController(
        model,
        priors,
        unfreeze_bait=unfreeze_bait,
        bait_subspace_mode=args.bait_subspace_mode,
        bait_sv_pct_low=args.bait_sv_pct_low,
        bait_sv_pct_high=args.bait_sv_pct_high,
        bait_basis_trainable=args.bait_basis_trainable,
        supervised_directions=supervised_directions,
        bait_output_mode=args.bait_output_mode,
        supervised_residual_dirs=supervised_residual_dirs,
        random_orthogonal_exclude_rank=args.random_orthogonal_exclude_rank,
        seed=args.seed,
        force_freeze_basis=force_freeze_basis,
        freeze_bait_gate=freeze_bait_gate,
        tag_scale=args.tag_scale,
        sidechannel_tag_trigger_mode=args.sidechannel_tag_trigger_mode,
        sidechannel_tag_trigger_mix=args.sidechannel_tag_trigger_mix,
        bait_gate_log_init=args.bait_gate_log_init,
        svd_method=args.svd_method,
        svd_lowrank_q=args.svd_lowrank_q,
        svd_lowrank_niter=args.svd_lowrank_niter,
    )
    print(
        f"Installed cross-layer defense: "
        + ", ".join(
            f"bait@L{inst.bait_layer_idx}→recovery@L{inst.recovery_layer_idx}"
            for inst in controller.installed_layers
        )
    )
    bait_trainable = sum(
        param.numel()
        for module in controller.bait_modules
        for param in module.parameters()
        if param.requires_grad
    )
    recovery_trainable = sum(
        param.numel()
        for module in controller.recovery_modules
        for param in module.parameters()
        if param.requires_grad
    )
    print(
        "  Trainable parameters: "
        f"bait={bait_trainable:,}, recovery={recovery_trainable:,}, "
        f"total={bait_trainable + recovery_trainable:,}"
    )
    external_direction_refs = load_bait_direction_refs(args.external_bait_direction_ref_path)

    train_prompts = load_general_prompts(args.n_train, source=general_data_source)
    print(f"Loaded {len(train_prompts)} unlabeled general-text prompts for KL training")

    if args.auto_calibrate_bait_gate:
        calibration_prompts = []
        calibration_prompts.extend(train_bad_prompts[: min(32, len(train_bad_prompts))])
        calibration_prompts.extend(train_good_prompts[: min(32, len(train_good_prompts))])
        if not calibration_prompts:
            calibration_prompts = train_prompts[: min(64, len(train_prompts))]
        calibrate_bait_gates(
            model=model,
            tokenizer=tokenizer,
            controller=controller,
            prompts=calibration_prompts,
            batch_size=args.batch_size,
            target_shift=args.bait_calibration_target,
            quantile=args.bait_calibration_quantile,
            log_min=args.bait_calibration_log_min,
            log_max=args.bait_calibration_log_max,
            kl_target=args.bait_calibration_kl_target,
            kl_adjust_iters=args.bait_calibration_kl_iters,
        )

    use_progressive = getattr(args, "progressive_training", False) and len(controller.installed_layers) > 1
    if use_progressive:
        print(
            f"\nProgressive training ({len(controller.installed_layers)} pairs, "
            f"{getattr(args, 'progressive_stages', 3)} epochs/stage + joint fine-tuning)..."
        )
        history = train_recovery_progressive(
            model, tokenizer, controller, train_prompts, args,
            train_good_prompts=train_good_prompts,
            train_bad_prompts=train_bad_prompts,
            supervised_residual_dirs=supervised_residual_dirs,
            external_direction_refs=external_direction_refs,
        )
    else:
        print("\nTraining recovery adapters against the frozen reference forward path...")
        history = train_recovery(
            model, tokenizer, controller, train_prompts, args,
            train_good_prompts=train_good_prompts,
            train_bad_prompts=train_bad_prompts,
            supervised_residual_dirs=supervised_residual_dirs,
            external_direction_refs=external_direction_refs,
        )

    print("\nEvaluating defense on good/bad prompt sets...")

    metrics = evaluate_defense(
        model=model,
        tokenizer=tokenizer,
        controller=controller,
        good_prompts=good_prompts,
        bad_prompts=bad_prompts,
        train_prompts=train_prompts,
        batch_size=args.batch_size,
        external_direction_refs=external_direction_refs,
        bait_anti_coherence_components=args.bait_anti_coherence_components,
    )
    if original_clean_fisher_profile is not None:
        metrics.setdefault("fisher_profiles", {})["original_clean"] = original_clean_fisher_profile
        metrics["fisher_profile_sources"] = {
            "original_clean": original_clean_model_name,
            "reference_clean": args.model,
            "defended": args.model,
        }
    # ── Per-pair KL (multi-layer only) ──
    if len(controller.installed_layers) > 1:
        print("  Computing per-pair KL contributions...")
        per_layer_kl = evaluate_per_layer_kl(
            model, tokenizer, controller,
            prompts=train_prompts[: min(32, len(train_prompts))],
            batch_size=args.batch_size,
        )
        metrics["per_layer_kl"] = per_layer_kl
    else:
        per_layer_kl = {}

    print_defense_scorecard(metrics, controller.installed_layers)

    final_kl_budget = float(getattr(args, "final_kl_budget", 0.0) or 0.0)
    final_kl = (
        per_layer_kl.get("all_pairs", float("nan"))
        if per_layer_kl
        else metrics.get("train_kl_eval", float("nan"))
    )
    if final_kl_budget > 0.0 and (not math.isfinite(final_kl) or final_kl > final_kl_budget):
        raise SystemExit(
            f"Final KL(all)={final_kl:.6f} exceeds --final-kl-budget={final_kl_budget:.6f}; "
            "aborting before saving checkpoint or merged model."
        )

    experiment_tag = getattr(args, "experiment_tag", "").strip()
    model_short = experiment_tag if experiment_tag else args.model.replace("/", "_")
    model_save_dir = args.model_save_dir if args.model_save_dir else RESULTS_DIR
    result_path = RESULTS_DIR / f"step4_results_{model_short}.json"
    bait_refs_path = RESULTS_DIR / f"step4_bait_refs_{model_short}.json"
    checkpoint_path = RESULTS_DIR / f"step4_recovery_{model_short}.pt"
    merged_export_dir = model_save_dir / f"step4_merged_{model_short}"
    args_payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }

    # Build per_pair_summary — the primary metrics table for JSON output.
    # Fisher clean→defended is always interpreted within the current stage's
    # reference model: Phase 1 compares against the original base model, while
    # Phase 2 compares against the Phase 1 merged model.
    _metrics_by_res = {item["residual_idx"]: item for item in metrics.get("per_layer", [])}
    _per_layer_kl = metrics.get("per_layer_kl", {})
    per_pair_summary = [
        {
            "bait_layer": inst.bait_layer_idx,
            "recovery_layer": inst.recovery_layer_idx,
            "kl": _per_layer_kl.get(
                f"pair_L{inst.bait_layer_idx}_L{inst.recovery_layer_idx}",
                metrics.get("train_kl_eval", float("nan")),
            ),
            "bait_cos": _metrics_by_res.get(inst.bait_layer_idx + 1, {}).get("direction_cosine", float("nan")),
            "bait_fisher_clean": _metrics_by_res.get(inst.bait_layer_idx + 1, {}).get("fisher_clean", float("nan")),
            "bait_fisher_defended": _metrics_by_res.get(inst.bait_layer_idx + 1, {}).get("fisher_defended", float("nan")),
            "recovery_cos": _metrics_by_res.get(inst.recovery_layer_idx + 1, {}).get("direction_cosine", float("nan")),
            "recovery_fisher_clean": _metrics_by_res.get(inst.recovery_layer_idx + 1, {}).get("fisher_clean", float("nan")),
            "recovery_fisher_defended": _metrics_by_res.get(inst.recovery_layer_idx + 1, {}).get("fisher_defended", float("nan")),
        }
        for inst in controller.installed_layers
    ]
    for row in per_pair_summary:
        row["bait_fisher_delta_pct"] = float(
            100.0
            * (row["bait_fisher_defended"] - row["bait_fisher_clean"])
            / max(row["bait_fisher_clean"], 1e-8)
        )
        row["recovery_fisher_delta_pct"] = float(
            100.0
            * (row["recovery_fisher_defended"] - row["recovery_fisher_clean"])
            / max(row["recovery_fisher_clean"], 1e-8)
        )

    bait_direction_refs = build_bait_direction_refs(
        model=model,
        tokenizer=tokenizer,
        controller=controller,
        good_prompts=train_good_prompts if train_good_prompts else good_prompts,
        bad_prompts=train_bad_prompts if train_bad_prompts else bad_prompts,
        batch_size=args.batch_size,
        components=max(2, int(args.bait_anti_coherence_components)),
    )

    result_payload = {
        "model": args.model,
        "step1_path": str(args.step1_path),
        "target_layers": target_layers,
        "cover_heretic_layers": getattr(args, "cover_heretic_layers", False),
        "heretic_layer_range": getattr(args, "heretic_layer_range", "mid"),
        "progressive_training": getattr(args, "progressive_training", False),
        "architecture": "cross-layer: layer l down_proj bait → layer l+1 o_proj + down_proj recovery",
        "installed_pairs": [
            {"bait_layer": inst.bait_layer_idx, "recovery_layer": inst.recovery_layer_idx}
            for inst in controller.installed_layers
        ],
        "training": {
            "objective": (
                "KL(defended_logits || frozen_reference_logits) + "
                "relative residual recovery alignment + trainable bait_gate visible band"
            ),
            "pathway": "layer l down_proj injection → layer l+1 o_proj + down_proj cancellation",
            "target": "residual l+1 poisoned; residual l+2 restored to clean",
            "args": args_payload,
            "history": history,
        },
        "layer_priors": [prior.__dict__ for prior in priors],
        "per_pair_summary": per_pair_summary,
        "evaluation": metrics,
        "bait_direction_refs": bait_direction_refs,
    }

    with open(result_path, "w") as f:
        json.dump(result_payload, f, indent=2)
    with open(bait_refs_path, "w") as f:
        json.dump(bait_direction_refs, f, indent=2)
    torch.save(controller.checkpoint(), checkpoint_path)
    merged_dir = controller.export_merged_checkpoint(merged_export_dir, tokenizer)

    if len(controller.installed_layers) > 1:
        print(
            f"\n[bold]Multi-layer defense summary[/]: "
            f"{len(controller.installed_layers)} pairs installed, "
            f"KL(all)={per_layer_kl.get('all_pairs', float('nan')):.6f}"
        )
    print(f"\nSaved results to [bold]{result_path}[/]")
    print(
        f"Saved bait direction refs to [bold]{bait_refs_path}[/] "
        f"(basis={len(bait_direction_refs.get('basis', []))}, "
        f"residual_gap={len(bait_direction_refs.get('residual_gap', []))})"
    )
    print(f"Saved recovery checkpoint to [bold]{checkpoint_path}[/]")
    print(f"Saved merged defense model to [bold]{merged_dir}[/]")

    # ── Phase 2: interleaved even-layer bait/recovery on phase-1-merged model ──
    # Phase 1 covers odd layers (e.g. L15,17,19,21,23 with recovery at L16,18,20,22,24).
    # Phase 2 covers even layers (e.g. L14,16,18,20,24 with recovery at L15,17,19,21,25).
    # Together, every layer in the range has both a bait injection and a downstream recovery,
    # matching Heretic's full per-layer traversal.
    #
    # Note: export_merged_checkpoint() above already called merge_defense(), which baked
    # phase 1 adapters into the base model weights. Phase 2 therefore installs fresh adapters
    # on the merged model — no shared wrappers, no enable/disable conflicts between phases.
    # The reference for phase 2 KL training is the merged phase 1 model (adapters disabled).
    phase2_layer_str = getattr(args, "phase2_target_layers", "")
    if phase2_layer_str.strip():
        phase2_layers = parse_requested_layers(phase2_layer_str)
        if not phase2_layers:
            print("\n[yellow]WARNING[/]: --phase2-target-layers is set but parsed to empty. Skipping phase 2.")
        else:
            print(f"\n\n{'='*80}")
            print(f"[bold]Phase 2[/]: installing bait/recovery at layers {phase2_layers}")
            print(f"  Phase 1 adapters are already merged into base model weights.")
            print(f"{'='*80}")

            # Validate: warn if deepest bait layer (max) has recovery at last transformer layer
            n_layers_total = len(get_layers(model))
            deepest_p2 = max(phase2_layers)
            if deepest_p2 + 1 >= n_layers_total - 1:
                print(
                    f"  [yellow]WARNING[/]: Phase 2 bait layer {deepest_p2} places recovery at "
                    f"layer {deepest_p2 + 1}, which is the last transformer layer. "
                    "Any uncanceled residual propagates directly through final_norm + lm_head "
                    "into logits (no downstream nonlinear buffer). Expect higher KL pressure on this pair."
                )

            phase2_priors = build_layer_priors(step1_summary, phase2_layers, args)
            print_design_summary(phase2_priors)

            # Recompute supervised directions on the merged (phase 1) model.
            if args.bait_output_mode == "random_orthogonal":
                supervised_directions_p2 = None
                train_good_p2 = []
                train_bad_p2 = []
            else:
                if not train_good_prompts or not train_bad_prompts:
                    raise ValueError(
                        "Phase 2 supervised bait output modes require non-empty "
                        "training good/bad prompts for direction initialization."
                    )
                train_good_p2 = train_good_prompts
                train_bad_p2 = train_bad_prompts
                print("Computing phase 2 supervised input directions from training-split prompts...")
                supervised_directions_p2 = compute_supervised_input_directions(
                    model, tokenizer,
                    good_prompts=train_good_p2,
                    bad_prompts=train_bad_p2,
                    layer_indices=phase2_layers,
                    module_name="mlp.down_proj",
                    batch_size=args.batch_size,
                )

            supervised_residual_dirs_p2: dict[int, torch.Tensor] | None = None
            if args.bait_output_mode in ("refusal_cancel", "sidechannel_cancel"):
                print(
                    f"\nComputing supervised residual refusal directions for phase 2 "
                    f"(layers {phase2_layers})..."
                )
                supervised_residual_dirs_p2 = compute_supervised_residual_directions(
                    model, tokenizer,
                    good_prompts=train_good_p2[:min(64, len(train_good_p2))],
                    bad_prompts=train_bad_p2[:min(64, len(train_bad_p2))],
                    layer_indices=phase2_layers,
                    batch_size=args.batch_size,
                )

            controller2 = DefenseController(
                model,
                phase2_priors,
                unfreeze_bait=unfreeze_bait,
                bait_subspace_mode=args.bait_subspace_mode,
                bait_sv_pct_low=args.bait_sv_pct_low,
                bait_sv_pct_high=args.bait_sv_pct_high,
                bait_basis_trainable=args.bait_basis_trainable,
                supervised_directions=supervised_directions_p2,
                bait_output_mode=args.bait_output_mode,
                supervised_residual_dirs=supervised_residual_dirs_p2,
                random_orthogonal_exclude_rank=args.random_orthogonal_exclude_rank,
                seed=args.seed + 1000,
                force_freeze_basis=force_freeze_basis,
                freeze_bait_gate=freeze_bait_gate,
                tag_scale=args.tag_scale,
                sidechannel_tag_trigger_mode=args.sidechannel_tag_trigger_mode,
                sidechannel_tag_trigger_mix=args.sidechannel_tag_trigger_mix,
                bait_gate_log_init=args.bait_gate_log_init,
            )
            print(
                f"Phase 2 installed cross-layer defense: "
                + ", ".join(
                    f"bait@L{inst.bait_layer_idx}→recovery@L{inst.recovery_layer_idx}"
                    for inst in controller2.installed_layers
                )
            )
            p2_bait_trainable = sum(
                p.numel()
                for m in controller2.bait_modules
                for p in m.parameters()
                if p.requires_grad
            )
            p2_recovery_trainable = sum(
                p.numel()
                for m in controller2.recovery_modules
                for p in m.parameters()
                if p.requires_grad
            )
            print(
                f"  Phase 2 trainable params: "
                f"bait={p2_bait_trainable:,}, recovery={p2_recovery_trainable:,}, "
                f"total={p2_bait_trainable + p2_recovery_trainable:,}"
            )
            phase2_external_direction_refs = {
                "basis": bait_direction_refs.get("basis", []),
                "residual_gap": bait_direction_refs.get("residual_gap", []),
            }

            if args.auto_calibrate_bait_gate:
                calibration_prompts_p2 = []
                calibration_prompts_p2.extend(train_bad_p2[: min(32, len(train_bad_p2))])
                calibration_prompts_p2.extend(train_good_p2[: min(32, len(train_good_p2))])
                if not calibration_prompts_p2:
                    calibration_prompts_p2 = train_prompts[: min(64, len(train_prompts))]
                calibrate_bait_gates(
                    model=model,
                    tokenizer=tokenizer,
                    controller=controller2,
                    prompts=calibration_prompts_p2,
                    batch_size=args.batch_size,
                    target_shift=args.bait_calibration_target,
                    quantile=args.bait_calibration_quantile,
                    log_min=args.bait_calibration_log_min,
                    log_max=args.bait_calibration_log_max,
                    kl_target=args.bait_calibration_kl_target,
                    kl_adjust_iters=args.bait_calibration_kl_iters,
                )

            use_progressive_p2 = (
                getattr(args, "progressive_training", False)
                and len(controller2.installed_layers) > 1
            )
            if use_progressive_p2:
                print(
                    f"\nPhase 2 progressive training ({len(controller2.installed_layers)} pairs, "
                    f"{getattr(args, 'progressive_stages', 3)} epochs/stage + joint fine-tuning)..."
                )
                history_p2 = train_recovery_progressive(
                    model, tokenizer, controller2, train_prompts, args,
                    train_good_prompts=train_good_p2,
                    train_bad_prompts=train_bad_p2,
                    supervised_residual_dirs=supervised_residual_dirs_p2,
                    external_direction_refs=phase2_external_direction_refs,
                )
            else:
                print("\nPhase 2: training recovery adapters...")
                history_p2 = train_recovery(
                    model, tokenizer, controller2, train_prompts, args,
                    train_good_prompts=train_good_p2,
                    train_bad_prompts=train_bad_p2,
                    supervised_residual_dirs=supervised_residual_dirs_p2,
                    external_direction_refs=phase2_external_direction_refs,
                )

            print("\nPhase 2: evaluating defense on good/bad prompt sets...")
            metrics_p2 = evaluate_defense(
                model=model,
                tokenizer=tokenizer,
                controller=controller2,
                good_prompts=good_prompts,
                bad_prompts=bad_prompts,
                train_prompts=train_prompts,
                batch_size=args.batch_size,
                external_direction_refs=phase2_external_direction_refs,
                bait_anti_coherence_components=args.bait_anti_coherence_components,
            )
            if original_clean_fisher_profile is not None:
                metrics_p2.setdefault("fisher_profiles", {})["original_clean"] = original_clean_fisher_profile
                metrics_p2["fisher_profile_sources"] = {
                    "original_clean": original_clean_model_name,
                    "reference_clean": args.model,
                    "defended": args.model,
                }
            # ── Per-pair KL (multi-layer only) ──
            if len(controller2.installed_layers) > 1:
                print("  Computing Phase 2 per-pair KL contributions...")
                per_layer_kl_p2 = evaluate_per_layer_kl(
                    model, tokenizer, controller2,
                    prompts=train_prompts[:min(32, len(train_prompts))],
                    batch_size=args.batch_size,
                )
                metrics_p2["per_layer_kl"] = per_layer_kl_p2
            else:
                per_layer_kl_p2 = {}

            print_defense_scorecard(metrics_p2, controller2.installed_layers)

            result_path_p2 = RESULTS_DIR / f"step4_phase2_results_{model_short}.json"
            checkpoint_path_p2 = RESULTS_DIR / f"step4_phase2_recovery_{model_short}.pt"
            merged_export_dir_p2 = model_save_dir / f"step4_phase2_merged_{model_short}"

            # Build per_pair_summary for Phase 2
            _p2_metrics_by_res = {item["residual_idx"]: item for item in metrics_p2.get("per_layer", [])}
            _p2_per_layer_kl   = metrics_p2.get("per_layer_kl", {})
            per_pair_summary_p2 = [
                {
                    "bait_layer":           inst.bait_layer_idx,
                    "recovery_layer":       inst.recovery_layer_idx,
                    "kl":                   _p2_per_layer_kl.get(
                                                f"pair_L{inst.bait_layer_idx}_L{inst.recovery_layer_idx}",
                                                metrics_p2.get("train_kl_eval", float("nan")),
                                            ),
                    "bait_cos":             _p2_metrics_by_res.get(inst.bait_layer_idx + 1, {}).get("direction_cosine",  float("nan")),
                    "bait_fisher_clean":    _p2_metrics_by_res.get(inst.bait_layer_idx + 1, {}).get("fisher_clean",      float("nan")),
                    "bait_fisher_defended": _p2_metrics_by_res.get(inst.bait_layer_idx + 1, {}).get("fisher_defended",   float("nan")),
                    "recovery_cos":             _p2_metrics_by_res.get(inst.recovery_layer_idx + 1, {}).get("direction_cosine", float("nan")),
                    "recovery_fisher_clean":    _p2_metrics_by_res.get(inst.recovery_layer_idx + 1, {}).get("fisher_clean",     float("nan")),
                    "recovery_fisher_defended": _p2_metrics_by_res.get(inst.recovery_layer_idx + 1, {}).get("fisher_defended",  float("nan")),
                }
                for inst in controller2.installed_layers
            ]
            for row in per_pair_summary_p2:
                row["bait_fisher_delta_pct"] = float(
                    100.0
                    * (row["bait_fisher_defended"] - row["bait_fisher_clean"])
                    / max(row["bait_fisher_clean"], 1e-8)
                )
                row["recovery_fisher_delta_pct"] = float(
                    100.0
                    * (row["recovery_fisher_defended"] - row["recovery_fisher_clean"])
                    / max(row["recovery_fisher_clean"], 1e-8)
                )

            result_payload_p2 = {
                "model": args.model,
                "step1_path": str(args.step1_path),
                "phase": 2,
                "phase1_target_layers": target_layers,
                "phase2_target_layers": phase2_layers,
                "architecture": (
                    "two-phase interleaved cross-layer: "
                    "phase1 odd-layer pairs merged into weights, "
                    "phase2 even-layer pairs as trainable adapters"
                ),
                "installed_pairs": [
                    {"bait_layer": inst.bait_layer_idx, "recovery_layer": inst.recovery_layer_idx}
                    for inst in controller2.installed_layers
                ],
                "training": {
                    "objective": (
                        "KL(phase2_defended || phase1_merged) + "
                        "relative residual recovery alignment + bait_gate visible band"
                    ),
                    "args": args_payload,
                    "history": history_p2,
                },
                "layer_priors": [prior.__dict__ for prior in phase2_priors],
                "per_pair_summary": per_pair_summary_p2,
                "evaluation": metrics_p2,
            }

            with open(result_path_p2, "w") as f:
                json.dump(result_payload_p2, f, indent=2)
            torch.save(controller2.checkpoint(), checkpoint_path_p2)
            merged_dir_p2 = controller2.export_merged_checkpoint(merged_export_dir_p2, tokenizer)

            print(
                f"\n[bold]Phase 2 summary[/]: "
                f"{len(controller2.installed_layers)} pairs installed, "
                f"KL(all)={metrics_p2.get('train_kl_eval', float('nan')):.6f}"
            )
            print(f"Saved phase 2 results to [bold]{result_path_p2}[/]")
            print(f"Saved phase 2 checkpoint to [bold]{checkpoint_path_p2}[/]")
            print(f"Saved phase 2 merged model to [bold]{merged_dir_p2}[/]")
            print(
                f"\n[bold]Full two-phase defense complete[/]: "
                f"phase1 layers {target_layers} + phase2 layers {phase2_layers} "
                f"→ every layer in range covered."
            )


if __name__ == "__main__":
    main()
