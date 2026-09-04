#!/usr/bin/env python3
"""Profile prompt-source composition used by sidechannel defense training."""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))

from shared_utils import (  # noqa: E402
    _row_is_direct_harm_request,
    _load_cached_dataset_files,
    _prompt_source_spec,
    _row_has_harm_category,
    load_prompts_simple,
)


HARM_MARKERS = (
    "steal",
    "kill",
    "bomb",
    "weapon",
    "gun",
    "poison",
    "hack",
    "exploit",
    "malware",
    "drug",
    "cocaine",
    "meth",
    "suicide",
    "self-harm",
    "abuse",
    "illegal",
    "blackmail",
    "bypass",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="beavertails")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--n-good", type=int, default=64)
    parser.add_argument("--n-bad", type=int, default=96)
    parser.add_argument("--good-offset", type=int, default=64)
    parser.add_argument("--bad-offset", type=int, default=64)
    parser.add_argument("--samples", type=int, default=8)
    return parser.parse_args()


def row_iter_for_source(source: str):
    good_spec = _prompt_source_spec(source, want_good=True)
    dataset_name, split = good_spec[0], good_spec[1]
    try:
        from datasets import load_dataset

        return iter(load_dataset(dataset_name, split=split))
    except Exception as exc:
        return iter(_load_cached_dataset_files(dataset_name, split, original_error=exc))


def text_of(row: dict[str, Any]) -> str:
    for key in ("prompt", "text", "instruction", "query", "user"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    content = row.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item) for item in content)
    return ""


def describe_texts(name: str, texts: list[str]) -> None:
    lengths = [len(text.split()) for text in texts]
    chars = [len(text) for text in texts]
    unique = len(set(texts))
    marker_hits = [
        text
        for text in texts
        if any(marker in text.lower() for marker in HARM_MARKERS)
    ]
    print(f"\n[{name}]")
    print(f"count={len(texts)} unique={unique} duplicate_rate={(1 - unique / max(len(texts), 1)):.3f}")
    if lengths:
        print(
            "word_len mean/median/max="
            f"{statistics.mean(lengths):.1f}/{statistics.median(lengths):.1f}/{max(lengths)}"
        )
        print(
            "char_len mean/median/max="
            f"{statistics.mean(chars):.1f}/{statistics.median(chars):.1f}/{max(chars)}"
        )
    print(f"harm_marker_hits={len(marker_hits)} ({len(marker_hits) / max(len(texts), 1):.3f})")
    for idx, text in enumerate(marker_hits[:5], 1):
        print(f"  marker_sample_{idx}: {text[:220].replace(chr(10), ' ')}")


def main() -> None:
    args = parse_args()

    print(f"source={args.source}")
    rows = []
    category_counts: Counter[str] = Counter()
    safety_counts: Counter[str] = Counter()
    prompt_lengths = []
    harmful_category_rows = 0
    direct_harm_rows = 0

    try:
        iterator = row_iter_for_source(args.source)
        for row in iterator:
            rows.append(row)
            text = text_of(row)
            prompt_lengths.append(len(text.split()))
            if "is_safe" in row:
                safety_counts[str(bool(row["is_safe"]))] += 1
            if _row_has_harm_category(row):
                harmful_category_rows += 1
            if _row_is_direct_harm_request(row):
                direct_harm_rows += 1
            category = row.get("category")
            if isinstance(category, dict):
                for key, value in category.items():
                    if value:
                        category_counts[key] += 1
            if len(rows) >= args.limit:
                break
    except ValueError:
        rows = []

    if rows:
        print("\n[raw rows]")
        print(f"inspected={len(rows)}")
        print(f"is_safe_counts={dict(safety_counts)}")
        print(
            f"harm_category_rows={harmful_category_rows} "
            f"({harmful_category_rows / max(len(rows), 1):.3f})"
        )
        print(
            f"direct_harm_request_rows={direct_harm_rows} "
            f"({direct_harm_rows / max(len(rows), 1):.3f})"
        )
        if prompt_lengths:
            print(
                "prompt_word_len mean/median/max="
                f"{statistics.mean(prompt_lengths):.1f}/"
                f"{statistics.median(prompt_lengths):.1f}/{max(prompt_lengths)}"
            )
        print("top_categories=" + repr(category_counts.most_common(12)))

    good_prompts, bad_prompts = load_prompts_simple(
        args.n_good,
        args.n_bad,
        good_offset=args.good_offset,
        bad_offset=args.bad_offset,
        source=args.source,
    )
    good_texts = [prompt.user for prompt in good_prompts]
    bad_texts = [prompt.user for prompt in bad_prompts]

    describe_texts("selected good prompts", good_texts)
    describe_texts("selected bad prompts", bad_texts)

    overlap = set(good_texts) & set(bad_texts)
    print(f"\ngood_bad_overlap={len(overlap)}")

    print("\n[good samples]")
    for text in good_texts[: args.samples]:
        print("- " + text[:260].replace("\n", " "))
    print("\n[bad samples]")
    for text in bad_texts[: args.samples]:
        print("- " + text[:260].replace("\n", " "))


if __name__ == "__main__":
    main()
