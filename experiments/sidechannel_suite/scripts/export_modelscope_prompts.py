#!/usr/bin/env python3
"""Export ModelScope datasets into local good/bad JSONL prompt files.

The training scripts consume local JSONL through TRAIN_*_DATA_SOURCE=local:/path,
so this script keeps ModelScope access out of the actual training path.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


DIRECT_HARM_MARKERS = (
    "how to",
    "how can i",
    "how do i",
    "give detailed",
    "give instructions",
    "tutorial",
    "steps",
    "make ",
    "build ",
    "create ",
    "synthesize",
    "steal",
    "smuggle",
    "evad",
    "hack",
    "malware",
    "phishing",
    "poison",
    "weapon",
    "bomb",
    "kill",
    "attack",
    "blackmail",
    "defamatory",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--good-dataset",
        default="AI-ModelScope/alpaca-gpt4-data-en",
        help="ModelScope dataset id for harmless/general prompts.",
    )
    parser.add_argument(
        "--bad-dataset",
        default="Shanghai_AI_Laboratory/SafeMTData",
        help="ModelScope dataset id for harmful prompts.",
    )
    parser.add_argument("--good-split", default="train")
    parser.add_argument("--bad-split", default="Attack_600")
    parser.add_argument("--bad-subset", default="Attack_600")
    parser.add_argument("--max-good", type=int, default=0, help="0 means no limit")
    parser.add_argument("--max-bad", type=int, default=0, help="0 means no limit")
    parser.add_argument("--filter-direct-bad", action="store_true", default=True)
    parser.add_argument("--no-filter-direct-bad", dest="filter_direct_bad", action="store_false")
    return parser.parse_args()


def load_ms_dataset(dataset_id: str, split: str, subset: str | None = None):
    from modelscope.msdatasets import MsDataset

    attempts = []
    if subset:
        attempts.extend(
            [
                lambda: MsDataset.load(dataset_id, subset_name=subset, split=split),
                lambda: MsDataset.load(dataset_id, name=subset, split=split),
                lambda: MsDataset.load(dataset_id, subset, split=split),
            ]
        )
    attempts.append(lambda: MsDataset.load(dataset_id, split=split))

    last_error: Exception | None = None
    for attempt in attempts:
        try:
            return attempt()
        except Exception as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def iter_rows(dataset: Any) -> Iterable[dict[str, Any]]:
    if hasattr(dataset, "to_hf_dataset"):
        dataset = dataset.to_hf_dataset()
    if isinstance(dataset, dict):
        dataset = next(iter(dataset.values()))
    for row in dataset:
        yield dict(row)


def extract_good_prompt(row: dict[str, Any]) -> str:
    instruction = str(row.get("instruction", "")).strip()
    input_text = str(row.get("input", "") or "").strip()
    if instruction:
        return f"{instruction}\n\n{input_text}".strip()
    for key in ("question", "prompt", "query", "text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def extract_bad_prompt(row: dict[str, Any]) -> str:
    for key in ("plain_query", "behavior", "goal", "prompt", "query", "instruction", "text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def looks_direct_harm(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in DIRECT_HARM_MARKERS)


def write_prompts(path: Path, rows: Iterable[dict[str, Any]], extractor, max_count: int, filter_fn=None) -> int:
    count = 0
    seen: set[str] = set()
    with open(path, "w") as f:
        for row in rows:
            prompt = extractor(row)
            if not prompt or prompt in seen:
                continue
            if filter_fn and not filter_fn(prompt):
                continue
            f.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n")
            seen.add(prompt)
            count += 1
            if max_count > 0 and count >= max_count:
                break
    return count


def materialize_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return list(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    good_ds = load_ms_dataset(args.good_dataset, split=args.good_split)
    good_count = write_prompts(
        args.out_dir / "good.jsonl",
        iter_rows(good_ds),
        extract_good_prompt,
        args.max_good,
    )

    bad_ds = load_ms_dataset(args.bad_dataset, split=args.bad_split, subset=args.bad_subset)
    bad_rows = materialize_rows(iter_rows(bad_ds))
    bad_count = write_prompts(
        args.out_dir / "bad.jsonl",
        bad_rows,
        extract_bad_prompt,
        args.max_bad,
        filter_fn=looks_direct_harm if args.filter_direct_bad else None,
    )
    if bad_count == 0 and args.filter_direct_bad:
        print("WARNING: direct-harm filter produced 0 bad prompts; retrying without filter.")
        bad_count = write_prompts(
            args.out_dir / "bad.jsonl",
            bad_rows,
            extract_bad_prompt,
            args.max_bad,
            filter_fn=None,
        )

    print(f"good={good_count} -> {args.out_dir / 'good.jsonl'}")
    print(f"bad={bad_count} -> {args.out_dir / 'bad.jsonl'}")
    if bad_count == 0:
        print("WARNING: no bad prompts exported. Try --no-filter-direct-bad or check bad dataset fields.")
        print("First bad rows:")
        for row in bad_rows[:5]:
            print(json.dumps(row, ensure_ascii=False)[:1000])


if __name__ == "__main__":
    main()
