"""
共享工具模块：模型加载、数据集加载、残差提取等。
复用 Heretic 的核心逻辑，但不依赖其交互式 CLI。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, ModuleList
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from datasets import load_dataset, load_from_disk
from rich.console import Console

print = Console(highlight=False).print

RESULTS_DIR = Path(os.environ.get("ANTI_HERETIC_RESULTS_DIR", Path(__file__).parent / "results"))
RESULTS_DIR.mkdir(exist_ok=True)


@dataclass
class Prompt:
    system: str
    user: str


DATA_SOURCE_HELP = (
    "Prompt data source preset. Supported: mlabonne, beavertails, "
    "beavertails-direct, alpaca, openorca, truthfulqa, pku-saferlhf, "
    "wildjailbreak, local:/path/to/dir, local-jsonl. "
    "mlabonne preserves the historical harmless_alpaca/"
    "harmful_behaviors behavior."
)


SUPPORTED_PROMPT_SOURCES = {
    "mlabonne",
    "alpaca",
    "beavertails",
    "beavertails-direct",
    "openorca",
    "pku-saferlhf",
    "truthfulqa",
    "wildjailbreak",
    "local-jsonl",
}


DIRECT_HARM_CATEGORY_ALLOWLIST = {
    "animal_abuse",
    "child_abuse",
    "drug_abuse,weapons,banned_substance",
    "financial_crime,property_crime,theft",
    "non_violent_unethical_behavior",
    "privacy_violation",
    "self_harm",
    "terrorism,organized_crime",
    "violence,aiding_and_abetting,incitement",
}


DIRECT_HARM_REQUEST_MARKERS = (
    "how can i",
    "how do i",
    "how to",
    "tell me how",
    "give me instructions",
    "give me step",
    "steps to",
    "instructions for",
    "guide me",
    "teach me",
    "help me",
    "make a",
    "make an",
    "build a",
    "create a",
    "write a",
    "generate a",
    "bypass",
    "hack",
    "steal",
    "poison",
    "kill",
    "attack",
    "burn",
    "exploit",
    "malware",
    "phishing",
    "blackmail",
    "forge",
    "counterfeit",
    "weapon",
    "bomb",
)


def get_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
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
        "--n-good",
        type=int,
        default=64,
        help="Number of harmless prompts to use",
    )
    parser.add_argument(
        "--n-bad",
        type=int,
        default=64,
        help="Number of harmful prompts to use",
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default="mlabonne",
        help=DATA_SOURCE_HELP,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference",
    )
    return parser.parse_args()


def resolve_device(device_str: str) -> torch.device:
    """Resolve 'auto' to the best available device."""

    def ensure_npu_registered() -> None:
        if not hasattr(torch, "npu"):
            with suppress(Exception):
                import torch_npu  # noqa: F401

    if device_str != "auto":
        if device_str.startswith("npu"):
            ensure_npu_registered()
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    ensure_npu_registered()
    if hasattr(torch, "npu"):
        with suppress(Exception):
            if torch.npu.is_available():  # type: ignore[attr-defined]
                return torch.device("npu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_model_input_device(model: PreTrainedModel) -> torch.device:
    """Return the device where token inputs should be placed."""
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is not None:
        return input_embeddings.weight.device
    return next(model.parameters()).device


def _resolve_model_dtype(default: torch.dtype) -> torch.dtype:
    """Resolve model load dtype, respecting the ``ANTI_HERETIC_MODEL_DTYPE`` env var.

    Useful when bf16 backward through models with very large residual streams
    (e.g. Gemma 3) loses precision and produces non-finite gradients in
    cross-layer adapter training. Set ``ANTI_HERETIC_MODEL_DTYPE=fp32`` to force
    the whole forward/backward to fp32.
    """
    override = os.environ.get("ANTI_HERETIC_MODEL_DTYPE", "").strip().lower()
    if override in ("fp32", "float32"):
        return torch.float32
    if override in ("bf16", "bfloat16"):
        return torch.bfloat16
    if override in ("fp16", "float16"):
        return torch.float16
    return default


def load_model(
    model_name: str,
    device: torch.device,
    tokenizer_name: str | None = None,
    gpu_mode: str = "auto",
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load model and tokenizer.

    tokenizer_name: if set, load the tokenizer from this path/id instead of
    model_name. Useful when model_name is a local merged checkpoint whose
    saved tokenizer triggers false-positive warnings (e.g. Mistral regex check
    on a Gemma tokenizer); pass the original HuggingFace model id instead.

    gpu_mode: controls CUDA placement.
      - "single": load the whole model onto ``device``.
      - "auto": use HuggingFace ``device_map="auto"`` across visible GPUs.
    """
    print(f"Loading model [bold]{model_name}[/]...")

    if gpu_mode not in {"single", "auto"}:
        raise ValueError(f"gpu_mode must be 'single' or 'auto', got {gpu_mode!r}")

    model_local_files_only = os.environ.get("ANTI_HERETIC_MODEL_LOCAL_FILES_ONLY", "0") == "1"
    trust_remote_code = os.environ.get("ANTI_HERETIC_TRUST_REMOTE_CODE", "1") == "1"
    tok_source = tokenizer_name if tokenizer_name else model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tok_source,
        local_files_only=model_local_files_only,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def _load_pretrained_model(**kwargs):
        try:
            return AutoModelForCausalLM.from_pretrained(
                model_name,
                local_files_only=model_local_files_only,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )
        except (KeyError, OSError, ValueError) as causal_exc:
            if os.environ.get("ANTI_HERETIC_ALLOW_IMAGE_TEXT_TO_TEXT", "1") != "1":
                raise
            try:
                from transformers import AutoModelForImageTextToText
            except ImportError as import_exc:
                raise RuntimeError(
                    "AutoModelForCausalLM could not load this model and this "
                    "Transformers install does not provide AutoModelForImageTextToText. "
                    "Install a newer Transformers version for multimodal text models "
                    "such as Qwen3.5/Gemma4."
                ) from import_exc
            try:
                return AutoModelForImageTextToText.from_pretrained(
                    model_name,
                    local_files_only=model_local_files_only,
                    trust_remote_code=trust_remote_code,
                    **kwargs,
                )
            except Exception as image_exc:
                raise RuntimeError(
                    "Failed to load model with both AutoModelForCausalLM and "
                    "AutoModelForImageTextToText. "
                    f"CausalLM error: {causal_exc}. ImageTextToText error: {image_exc}"
                ) from image_exc

    if device.type in ("cuda", "npu") and gpu_mode == "auto":
        model = _load_pretrained_model(
            torch_dtype=_resolve_model_dtype(torch.bfloat16),
            device_map="auto",
        )
    elif device.type in ("cuda", "npu", "mps"):
        default_dtype = torch.bfloat16 if device.type in ("cuda", "npu") else torch.float32
        model = _load_pretrained_model(
            torch_dtype=_resolve_model_dtype(default_dtype),
        )
        model = model.to(device)
    else:
        model = _load_pretrained_model(
            torch_dtype=_resolve_model_dtype(torch.float32),
        )

    model.eval()
    loaded_dtype = next(model.parameters()).dtype
    print(f"  Model loaded on [bold]{next(model.parameters()).device}[/] (dtype={loaded_dtype})")
    hf_device_map = getattr(model, "hf_device_map", None)
    if hf_device_map:
        devices = sorted({str(device_name) for device_name in hf_device_map.values()})
        print(f"  HF device_map: {len(hf_device_map)} modules across {', '.join(devices)}")
    else:
        print(f"  GPU mode: {gpu_mode}")
    return model, tokenizer


def load_prompts_simple(
    n_good: int,
    n_bad: int,
    good_offset: int = 0,
    bad_offset: int = 0,
    source: str = "mlabonne",
) -> tuple[list[Prompt], list[Prompt]]:
    """Load good/bad prompt datasets with optional start offsets.

    Args:
        n_good: Number of harmless prompts to load.
        n_bad:  Number of harmful prompts to load.
        good_offset: Skip this many harmless prompts from the start (for train/eval split).
        bad_offset:  Skip this many harmful  prompts from the start (for train/eval split).
        source: Prompt source preset. "mlabonne" is the legacy source.

    Uses local data if available, otherwise downloads from HuggingFace.
    """
    system_prompt = "You are a helpful assistant."
    source = source.strip()
    source_key = source.lower()
    is_local_source = source_key.startswith("local:")
    if not is_local_source:
        source = source_key
    if source_key not in SUPPORTED_PROMPT_SOURCES and not is_local_source:
        raise ValueError(
            f"Unknown prompt source {source!r}. Supported sources: "
            f"{', '.join(sorted(SUPPORTED_PROMPT_SOURCES))}, local:/path/to/dir"
        )

    local_data_dir = Path(__file__).parent.parent / "heretic" / "data"

    good_prompts: list[Prompt] = []
    bad_prompts:  list[Prompt] = []

    if source == "local-jsonl" or is_local_source:
        prompt_dir = _resolve_local_prompt_dir(source)
        good_prompts, bad_prompts = _load_local_prompt_dir(
            prompt_dir,
            n_good,
            n_bad,
            good_offset,
            bad_offset,
            system_prompt,
        )
    elif source == "mlabonne" and (local_data_dir / "harmless_alpaca_train.jsonl").exists():
        print("Loading prompts from local data...")
        good_prompts = _load_local_jsonl_prompts(
            local_data_dir / "harmless_alpaca_train.jsonl",
            n_good,
            good_offset,
            "text",
            system_prompt,
        )
        bad_prompts = _load_local_jsonl_prompts(
            local_data_dir / "harmful_behaviors_train.jsonl",
            n_bad,
            bad_offset,
            "text",
            system_prompt,
        )
    elif source == "mlabonne":
        print("Loading prompts from HuggingFace...")
        g_end = good_offset + n_good
        b_end = bad_offset  + n_bad
        good_ds = load_dataset(
            "mlabonne/harmless_alpaca", split=f"train[{good_offset}:{g_end}]"
        )
        bad_ds = load_dataset(
            "mlabonne/harmful_behaviors", split=f"train[{bad_offset}:{b_end}]"
        )
        good_prompts = [Prompt(system=system_prompt, user=t) for t in good_ds["text"]]
        bad_prompts  = [Prompt(system=system_prompt, user=t) for t in bad_ds["text"]]
    else:
        print(f"Loading prompts from HuggingFace preset [bold]{source}[/]...")
        good_prompts, bad_prompts = _load_prompt_source_pair(
            source=source,
            n_good=n_good,
            n_bad=n_bad,
            good_offset=good_offset,
            bad_offset=bad_offset,
            system_prompt=system_prompt,
        )

    print(
        f"  Loaded {len(good_prompts)} good + {len(bad_prompts)} bad prompts "
        f"(source={source})"
    )
    return good_prompts, bad_prompts


def load_prompts_split_sources(
    n_good: int,
    n_bad: int,
    good_source: str,
    bad_source: str,
    good_offset: int = 0,
    bad_offset: int = 0,
) -> tuple[list[Prompt], list[Prompt]]:
    if good_source == bad_source:
        return load_prompts_simple(
            n_good,
            n_bad,
            good_offset=good_offset,
            bad_offset=bad_offset,
            source=good_source,
        )
    good_prompts, _ = load_prompts_simple(
        n_good,
        0,
        good_offset=good_offset,
        bad_offset=0,
        source=good_source,
    )
    _, bad_prompts = load_prompts_simple(
        0,
        n_bad,
        good_offset=0,
        bad_offset=bad_offset,
        source=bad_source,
    )
    return good_prompts, bad_prompts


def _load_local_jsonl_prompts(
    path: Path,
    n_prompts: int,
    offset: int,
    column: str,
    system_prompt: str,
) -> list[Prompt]:
    prompts: list[Prompt] = []
    if n_prompts <= 0:
        return prompts
    with open(path) as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            if len(prompts) >= n_prompts:
                break
            data = json.loads(line)
            prompts.append(Prompt(system=system_prompt, user=str(data[column])))
    return prompts


def _resolve_local_prompt_dir(source: str) -> Path:
    if source.lower().startswith("local:"):
        path = source.split(":", 1)[1].strip()
    else:
        path = os.environ.get("ANTI_HERETIC_LOCAL_DATA_DIR", "").strip()
    if not path:
        raise ValueError(
            "local-jsonl source requires ANTI_HERETIC_LOCAL_DATA_DIR, or use "
            "source form local:/path/to/dir"
        )
    prompt_dir = Path(path).expanduser()
    if not prompt_dir.exists():
        raise FileNotFoundError(f"Local prompt directory not found: {prompt_dir}")
    return prompt_dir


def _load_local_prompt_dir(
    prompt_dir: Path,
    n_good: int,
    n_bad: int,
    good_offset: int,
    bad_offset: int,
    system_prompt: str,
) -> tuple[list[Prompt], list[Prompt]]:
    print(f"Loading prompts from local JSONL directory [bold]{prompt_dir}[/]...")
    good_file = _find_local_prompt_file(
        prompt_dir,
        ["good.jsonl", "harmless.jsonl", "safe.jsonl", "train_good.jsonl"],
        required=n_good > 0,
    )
    bad_file = _find_local_prompt_file(
        prompt_dir,
        ["bad.jsonl", "harmful.jsonl", "unsafe.jsonl", "train_bad.jsonl"],
        required=n_bad > 0,
    )
    good_prompts = (
        _load_flexible_jsonl_prompts(good_file, n_good, good_offset, system_prompt)
        if good_file
        else []
    )
    bad_prompts = (
        _load_flexible_jsonl_prompts(bad_file, n_bad, bad_offset, system_prompt)
        if bad_file
        else []
    )
    return good_prompts, bad_prompts


def _find_local_prompt_file(prompt_dir: Path, names: list[str], required: bool) -> Path | None:
    for name in names:
        path = prompt_dir / name
        if path.exists():
            return path
    if required:
        raise FileNotFoundError(
            f"Expected one of {names} in local prompt directory: {prompt_dir}"
        )
    return None


def _load_flexible_jsonl_prompts(
    path: Path,
    n_prompts: int,
    offset: int,
    system_prompt: str,
) -> list[Prompt]:
    prompts: list[Prompt] = []
    if n_prompts <= 0:
        return prompts
    with open(path) as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            if len(prompts) >= n_prompts:
                break
            row = json.loads(line)
            text = _extract_prompt_text(row).strip()
            if text:
                prompts.append(Prompt(system=system_prompt, user=text))
    if len(prompts) < n_prompts:
        raise ValueError(
            f"Local prompt file {path} yielded only {len(prompts)} prompts after "
            f"offset={offset}; requested {n_prompts}."
        )
    return prompts


def _extract_prompt_text(row: dict[str, Any]) -> str:
    for key in ("text", "prompt", "user", "instruction", "query"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    content = row.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item) for item in content)
    raise KeyError(
        "Could not find prompt text column. Expected one of: "
        "text, prompt, user, instruction, query, content."
    )


def _load_prompt_source_side(
    source: str,
    want_good: bool,
    n_prompts: int,
    offset: int,
    system_prompt: str,
) -> list[Prompt]:
    if n_prompts <= 0:
        return []

    dataset_name, split, row_filter, extractor = _prompt_source_spec(source, want_good)
    dataset = _load_dataset_name(dataset_name, split)
    prompts: list[Prompt] = []
    seen = 0
    for row in dataset:
        if not row_filter(row):
            continue
        if seen < offset:
            seen += 1
            continue
        text = extractor(row).strip()
        if not text:
            continue
        prompts.append(Prompt(system=system_prompt, user=text))
        if len(prompts) >= n_prompts:
            break
    if len(prompts) < n_prompts:
        raise ValueError(
            f"Source {source!r} yielded only {len(prompts)} prompts for "
            f"{'good' if want_good else 'bad'} side after offset={offset}; "
            f"requested {n_prompts}."
        )
    return prompts


def _load_prompt_source_pair(
    source: str,
    n_good: int,
    n_bad: int,
    good_offset: int,
    bad_offset: int,
    system_prompt: str,
) -> tuple[list[Prompt], list[Prompt]]:
    good_spec = _prompt_source_spec(source, want_good=True)
    bad_spec = _prompt_source_spec(source, want_good=False)
    good_dataset_name, good_split, good_filter, good_extractor = good_spec
    bad_dataset_name, bad_split, bad_filter, bad_extractor = bad_spec

    if good_dataset_name != bad_dataset_name or good_split != bad_split:
        return (
            _load_prompt_source_side(source, True, n_good, good_offset, system_prompt),
            _load_prompt_source_side(source, False, n_bad, bad_offset, system_prompt),
        )

    try:
        dataset = _load_dataset_name(good_dataset_name, good_split)
    except Exception as exc:
        dataset = _load_cached_dataset_files(
            good_dataset_name,
            good_split,
            original_error=exc,
        )
    good_prompts: list[Prompt] = []
    bad_prompts: list[Prompt] = []
    good_seen = 0
    bad_seen = 0

    for row in dataset:
        if len(good_prompts) < n_good and good_filter(row):
            if good_seen < good_offset:
                good_seen += 1
            else:
                text = good_extractor(row).strip()
                if text:
                    good_prompts.append(Prompt(system=system_prompt, user=text))

        if len(bad_prompts) < n_bad and bad_filter(row):
            if bad_seen < bad_offset:
                bad_seen += 1
            else:
                text = bad_extractor(row).strip()
                if text:
                    bad_prompts.append(Prompt(system=system_prompt, user=text))

        if len(good_prompts) >= n_good and len(bad_prompts) >= n_bad:
            break

    missing = []
    if len(good_prompts) < n_good:
        missing.append(f"good={len(good_prompts)}/{n_good}")
    if len(bad_prompts) < n_bad:
        missing.append(f"bad={len(bad_prompts)}/{n_bad}")
    if missing:
        raise ValueError(
            f"Source {source!r} yielded too few prompts after offsets "
            f"(good_offset={good_offset}, bad_offset={bad_offset}): "
            + ", ".join(missing)
        )
    return good_prompts, bad_prompts


def _load_cached_dataset_files(
    dataset_name: str,
    split: str,
    original_error: Exception,
):
    processed_dataset = _load_processed_dataset_cache(dataset_name, split)
    if processed_dataset is not None:
        return processed_dataset

    cached_files = _find_cached_dataset_files(dataset_name, split)
    if not cached_files:
        raise original_error

    print(
        f"  Falling back to cached dataset files for {dataset_name}:{split}: "
        + ", ".join(str(path) for path in cached_files)
    )
    suffix = cached_files[0].suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return load_dataset("json", data_files=[str(path) for path in cached_files], split="train")
    if suffix == ".parquet":
        return load_dataset("parquet", data_files=[str(path) for path in cached_files], split="train")
    if suffix == ".csv":
        return load_dataset("csv", data_files=[str(path) for path in cached_files], split="train")
    raise ValueError(
        f"Unsupported cached dataset file type for {dataset_name}:{split}: "
        f"{cached_files[0]}"
    )


def _find_cached_dataset_files(dataset_name: str, split: str) -> list[Path]:
    repo_cache = _dataset_repo_cache_dir(dataset_name)
    if not repo_cache.exists():
        return []

    snapshot_dirs = sorted((repo_cache / "snapshots").glob("*"), reverse=True)
    if not snapshot_dirs:
        return []

    allowed_suffixes = {".json", ".jsonl", ".parquet", ".csv"}
    split_tokens = [token.lower() for token in split.replace("-", "_").split("_") if token]

    candidates: list[Path] = []
    for snapshot in snapshot_dirs:
        files = [
            path
            for path in snapshot.rglob("*")
            if path.is_file()
            and path.suffix.lower() in allowed_suffixes
            and all(token in path.name.lower().replace("-", "_") for token in split_tokens)
        ]
        if files:
            candidates.extend(sorted(files))
            break

    if not candidates:
        for snapshot in snapshot_dirs:
            files = [
                path
                for path in snapshot.rglob("*")
                if path.is_file() and path.suffix.lower() in allowed_suffixes
            ]
            if files:
                candidates.extend(sorted(files))
                break

    return candidates


def _dataset_repo_cache_dir(dataset_name: str) -> Path:
    dataset_name = dataset_name.split(":", 1)[0]
    encoded = "datasets--" + dataset_name.replace("/", "--")
    roots = []
    for env_name in ("HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE"):
        value = os.environ.get(env_name, "").strip()
        if value:
            roots.append(Path(value).expanduser())
    hf_home = os.environ.get("HF_HOME", "").strip()
    if hf_home:
        roots.append(Path(hf_home).expanduser() / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")

    for root in roots:
        candidate = root / encoded
        if candidate.exists():
            return candidate
    return roots[0] / encoded


def _load_processed_dataset_cache(dataset_name: str, split: str):
    roots = []
    for env_name in ("HF_DATASETS_CACHE", "HF_DATASETS_CACHE_DIR"):
        value = os.environ.get(env_name, "").strip()
        if value:
            roots.append(Path(value).expanduser())
    hf_home = os.environ.get("HF_HOME", "").strip()
    if hf_home:
        roots.append(Path(hf_home).expanduser() / "datasets")
    roots.append(Path.home() / ".cache" / "huggingface" / "datasets")

    dataset_parts = dataset_name.split(":", 1)[0].split("/")
    direct_names = [
        "___".join(dataset_parts),
        "___".join(part.replace("-", "_") for part in dataset_parts),
        dataset_parts[-1],
        dataset_parts[-1].replace("-", "_"),
    ]

    candidates: list[Path] = []
    for root in roots:
        for name in direct_names:
            base = root / name
            if base.exists():
                candidates.extend(path for path in base.rglob("*") if path.is_dir())
        if root.exists():
            leaf = dataset_parts[-1].lower().replace("-", "_")
            candidates.extend(
                path
                for path in root.rglob("*")
                if path.is_dir() and leaf in path.name.lower().replace("-", "_")
            )

    for path in sorted(set(candidates), key=lambda item: len(str(item)), reverse=True):
        if not (path / "dataset_info.json").exists() and not (path / "state.json").exists():
            continue
        try:
            dataset = load_from_disk(str(path))
        except Exception:
            continue
        if hasattr(dataset, "keys") and split in dataset:
            print(f"  Falling back to processed dataset cache for {dataset_name}:{split}: {path}")
            return dataset[split]
        if not hasattr(dataset, "keys"):
            print(f"  Falling back to processed dataset cache for {dataset_name}:{split}: {path}")
            return dataset
    return None


def _load_dataset_name(dataset_name: str, split: str):
    if ":" in dataset_name:
        base_name, config_name = dataset_name.split(":", 1)
        return load_dataset(base_name, config_name, split=split)
    return load_dataset(dataset_name, split=split)


def _prompt_source_spec(
    source: str,
    want_good: bool,
) -> tuple[str, str, Callable[[dict[str, Any]], bool], Callable[[dict[str, Any]], str]]:
    if source in {"beavertails", "beavertails-direct"}:
        return (
            "PKU-Alignment/BeaverTails",
            "330k_train",
            lambda row: (
                bool(row["is_safe"]) and not _row_has_harm_category(row)
            )
            if want_good
            else (
                not bool(row["is_safe"]) and _row_has_harm_category(row)
                and (source != "beavertails-direct" or _row_is_direct_harm_request(row))
            ),
            lambda row: str(row["prompt"]),
        )
    if source == "alpaca":
        return (
            "tatsu-lab/alpaca",
            "train",
            lambda row: want_good,
            _extract_alpaca_instruction,
        )
    if source == "openorca":
        return (
            "Open-Orca/OpenOrca",
            "train",
            lambda row: want_good,
            lambda row: str(row["question"]),
        )
    if source == "truthfulqa":
        return (
            "truthful_qa:generation",
            "validation",
            lambda row: want_good,
            lambda row: str(row["question"]),
        )
    if source == "pku-saferlhf":
        return (
            "PKU-Alignment/PKU-SafeRLHF",
            "train",
            lambda row: (
                bool(row["is_response_0_safe"]) and bool(row["is_response_1_safe"])
            )
            if want_good
            else (
                not bool(row["is_response_0_safe"])
                or not bool(row["is_response_1_safe"])
            ),
            lambda row: str(row["prompt"]),
        )
    if source == "wildjailbreak":
        return (
            "allenai/wildjailbreak",
            "train",
            lambda row: ("benign" in str(row["data_type"]))
            if want_good
            else ("harmful" in str(row["data_type"])),
            lambda row: str(row["adversarial"] or row["vanilla"]),
        )
    raise ValueError(f"Unhandled prompt source: {source}")


def _row_has_harm_category(row: dict[str, Any]) -> bool:
    category = row.get("category")
    if isinstance(category, dict):
        return any(bool(value) for value in category.values())
    if isinstance(category, str):
        normalized = category.strip().lower()
        return bool(normalized) and normalized not in {"none", "safe"}
    return False


def _extract_alpaca_instruction(row: dict[str, Any]) -> str:
    instruction = str(row.get("instruction", "")).strip()
    input_text = str(row.get("input", "")).strip()
    if input_text:
        return f"{instruction}\n\n{input_text}"
    return instruction


def _active_harm_categories(row: dict[str, Any]) -> set[str]:
    category = row.get("category")
    if isinstance(category, dict):
        return {str(key) for key, value in category.items() if value}
    if isinstance(category, str):
        normalized = category.strip().lower()
        return {normalized} if normalized and normalized not in {"none", "safe"} else set()
    return set()


def _row_is_direct_harm_request(row: dict[str, Any]) -> bool:
    categories = _active_harm_categories(row)
    if not categories.intersection(DIRECT_HARM_CATEGORY_ALLOWLIST):
        return False
    prompt = str(row.get("prompt", "")).lower()
    return any(marker in prompt for marker in DIRECT_HARM_REQUEST_MARKERS)


def _resolve_nested_attr(root: Any, dotted_path: str) -> Any:
    obj = root
    for name in dotted_path.split("."):
        obj = getattr(obj, name)
    return obj


def get_layers(model: PreTrainedModel) -> ModuleList:
    """Extract transformer layer list from text-only or multimodal wrappers."""
    candidates = (
        "model.layers",
        "model.language_model.layers",
        "language_model.layers",
        "language_model.model.layers",
        "model.model.layers",
        "base_model.model.model.layers",
        "base_model.model.language_model.layers",
    )
    for path in candidates:
        with suppress(Exception):
            layers = _resolve_nested_attr(model, path)
            if isinstance(layers, ModuleList):
                return layers
    raise AttributeError(
        "Could not locate transformer layers. Tried: " + ", ".join(candidates)
    )


def get_layer_module(layer: Module, component: str) -> Module | None:
    """Get a specific sub-module from a layer."""
    component_paths = {
        # Keep the semantic component name used by Heretic while accepting
        # architecture-specific attribute names. Spark-X2.5 exposes the
        # standard attention output projection as ``self_attn.out_proj``.
        "attn.o_proj": (
            "self_attn.o_proj",
            "self_attn.out_proj",
            "linear_attn.out_proj",
        ),
        "mlp.down_proj": ("mlp.down_proj",),
        "mlp.up_proj": ("mlp.up_proj",),
        "mlp.gate_proj": ("mlp.gate_proj",),
        "attn.q_proj": ("self_attn.q_proj",),
        "attn.k_proj": ("self_attn.k_proj",),
        "attn.v_proj": ("self_attn.v_proj",),
    }
    for path in component_paths.get(component, ()):
        with suppress(Exception):
            module = _resolve_nested_attr(layer, path)
            if isinstance(module, Module):
                return module
    return None


def set_layer_module(layer: Module, component: str, module: Module) -> None:
    """Replace a semantic layer component across supported model layouts."""
    component_paths = {
        "attn.o_proj": (
            "self_attn.o_proj",
            "self_attn.out_proj",
            "linear_attn.out_proj",
        ),
        "mlp.down_proj": ("mlp.down_proj",),
    }
    paths = component_paths.get(component, ())
    for path in paths:
        parent_path, attr_name = path.rsplit(".", 1)
        with suppress(Exception):
            parent = _resolve_nested_attr(layer, parent_path)
            current = getattr(parent, attr_name)
            if isinstance(current, Module):
                setattr(parent, attr_name, module)
                return
    raise AttributeError(
        f"Could not replace component {component!r}. Tried: {', '.join(paths)}"
    )


def tokenize_prompts(
    prompts: list[Prompt],
    tokenizer: PreTrainedTokenizerBase,
) -> dict:
    """Tokenize prompts with chat template."""
    chats = [
        [
            {"role": "system", "content": p.system},
            {"role": "user", "content": p.user},
        ]
        for p in prompts
    ]

    chat_texts = tokenizer.apply_chat_template(
        chats,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = tokenizer(
        chat_texts,
        return_tensors="pt",
        padding=True,
        return_token_type_ids=False,
    )
    return inputs


def get_residuals_differentiable(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[Prompt],
    batch_size: int = 8,
) -> Tensor:
    """
    可微分版本的残差提取。使用 model() forward pass 而非 generate()，
    保留计算图以支持梯度回传（用于 step3 训练）。
    返回 tensor of shape (n_prompts, n_layers+1, hidden_dim)。
    """
    all_residuals = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenize_prompts(batch, tokenizer)
        inputs = {k: v.to(get_model_input_device(model)) for k, v in inputs.items()}

        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        # hidden_states: tuple of (n_layers+1) tensors, each (batch, seq_len, hidden_dim)
        # padding_side="left"，所以最后一个真实 token 始终在最后位置
        hidden_states = outputs.hidden_states
        residuals = torch.stack(
            [layer_hs[:, -1, :] for layer_hs in hidden_states],
            dim=1,
        )  # (batch, n_layers+1, hidden_dim)

        all_residuals.append(residuals.float())

    return torch.cat(all_residuals, dim=0)


@torch.no_grad()
def get_residuals_batched(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[Prompt],
    batch_size: int = 8,
) -> Tensor:
    """
    Extract per-layer residual vectors at the first generated token position.
    Returns tensor of shape (n_prompts, n_layers+1, hidden_dim).
    与 Heretic 的 get_residuals() 逻辑完全一致。
    """
    all_residuals = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenize_prompts(batch, tokenizer)
        inputs = {k: v.to(get_model_input_device(model)) for k, v in inputs.items()}

        # Some trusted custom architectures (currently Spark-X2.5) accept but
        # do not propagate ``output_hidden_states``. Capture the exact same
        # residual sequence through layer hooks so Step1 keeps the canonical
        # embedding + per-layer-output readout in that case.
        layers = get_layers(model)
        hooked_residuals: list[Tensor | None] = [None] * (len(layers) + 1)
        handles = []

        def capture_embedding(_module, args):
            if hooked_residuals[0] is None and args and isinstance(args[0], Tensor):
                hooked_residuals[0] = args[0][:, -1, :].detach().float().cpu()

        def capture_layer(layer_position: int):
            def hook(_module, _args, output):
                tensor = output[0] if isinstance(output, (tuple, list)) else output
                if hooked_residuals[layer_position] is None and isinstance(tensor, Tensor):
                    hooked_residuals[layer_position] = tensor[:, -1, :].detach().float().cpu()

            return hook

        handles.append(layers[0].register_forward_pre_hook(capture_embedding))
        handles.extend(
            layer.register_forward_hook(capture_layer(layer_idx + 1))
            for layer_idx, layer in enumerate(layers)
        )
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
            )
        finally:
            for handle in handles:
                handle.remove()

        # hidden_states[0] = tuple of (n_layers+1) tensors, each (batch, seq_len, hidden_dim)
        # We want the last position of each
        native_hidden_states = outputs.hidden_states[0] if outputs.hidden_states else None
        if native_hidden_states is not None:
            hidden_states = native_hidden_states
            residuals = torch.stack(
                [layer_hs[:, -1, :] for layer_hs in hidden_states],
                dim=1,
            ).float().cpu()
        else:
            missing = [idx for idx, value in enumerate(hooked_residuals) if value is None]
            if missing:
                raise RuntimeError(f"Residual hook capture missed positions: {missing}")
            residuals = torch.stack(
                [value for value in hooked_residuals if value is not None],
                dim=1,
            )

        all_residuals.append(residuals)

    return torch.cat(all_residuals, dim=0)


@torch.no_grad()
def get_intermediate_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[Prompt],
    target_layers: list[int],
    batch_size: int = 8,
) -> dict[int, dict[str, Tensor]]:
    """
    用 hook 提取指定层的中间激活值。
    返回 {layer_idx: {"o_proj_out": Tensor, "mlp_input": Tensor, ...}}
    每个 Tensor 的 shape 为 (n_prompts, hidden_dim)，取最后一个 token position。
    """
    activations = {l: {} for l in target_layers}
    hooks = []

    def make_hook(layer_idx, name):
        def hook_fn(module, input, output):
            # output 可能是 tuple，取第一个
            out = output[0] if isinstance(output, tuple) else output
            if name not in activations[layer_idx]:
                activations[layer_idx][name] = []
            # 取最后一个 token position
            activations[layer_idx][name].append(out[:, -1, :].float().cpu())
        return hook_fn

    layers = get_layers(model)
    for l_idx in target_layers:
        layer = layers[l_idx]

        # Hook o_proj output
        o_proj = get_layer_module(layer, "attn.o_proj")
        if o_proj is not None:
            hooks.append(o_proj.register_forward_hook(make_hook(l_idx, "o_proj_out")))

        # Hook up_proj output
        up_proj = get_layer_module(layer, "mlp.up_proj")
        if up_proj is not None:
            hooks.append(up_proj.register_forward_hook(make_hook(l_idx, "up_proj_out")))

        # Hook gate_proj output
        gate_proj = get_layer_module(layer, "mlp.gate_proj")
        if gate_proj is not None:
            hooks.append(gate_proj.register_forward_hook(make_hook(l_idx, "gate_proj_out")))

        # Hook down_proj output
        down_proj = get_layer_module(layer, "mlp.down_proj")
        if down_proj is not None:
            hooks.append(down_proj.register_forward_hook(make_hook(l_idx, "down_proj_out")))

    # Run forward passes
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenize_prompts(batch, tokenizer)
        inputs = {k: v.to(get_model_input_device(model)) for k, v in inputs.items()}

        model.generate(
            **inputs,
            max_new_tokens=1,
            output_hidden_states=False,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
        )

    # Clean up hooks
    for h in hooks:
        h.remove()

    # Concatenate batches
    for l_idx in target_layers:
        for name in activations[l_idx]:
            activations[l_idx][name] = torch.cat(activations[l_idx][name], dim=0)

    return activations
