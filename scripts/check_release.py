#!/usr/bin/env python3
"""Dependency-free release hygiene checks for the public repository."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "SECURITY.md",
    "CITATION.cff",
    "docs/PROVENANCE.md",
    "docs/DATA.md",
    "requirements/runtime.txt",
    "experiments/step4_multilayer.py",
    "patches/heretic/README.md",
    "patches/heretic/bait-and-recover-heretic-v1.2.0.patch",
    "scripts/setup_heretic.sh",
    "release_artifacts/README.md",
)
SKIP_PARTS = {".git", ".venv", "artifacts", "logs", "results", "kernel_meta", "__pycache__"}
SKIP_NAMES = {"fusion_result.json"}
TEXT_SUFFIXES = {"", ".md", ".txt", ".py", ".sh", ".toml", ".yaml", ".yml", ".cff", ".json", ".jsonl", ".log"}
PROHIBITED_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".bin"}
SENSITIVE_PATTERNS = {
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "OpenAI-style token": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private key": re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
    "developer home path": re.compile(r"/(?:Users|home)/gaotian(?:/|\b)"),
    "internal data path": re.compile(r"/(?:data|mnt/t4_data)/tiangao5(?:/|\b)"),
    "root-only path": re.compile(r"/root/(?:miniconda|anaconda|\.cache)(?:/|\b)"),
}


def iter_files() -> list[Path]:
    """Return release-relevant files, excluding ignored local caches/artifacts."""
    try:
        proc = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        candidates = [ROOT / item.decode() for item in proc.stdout.split(b"\0") if item]
    except (OSError, subprocess.CalledProcessError):
        candidates = list(ROOT.rglob("*"))
    return sorted(
        path
        for path in candidates
        if (
            path.is_file()
            and path.name not in SKIP_NAMES
            and not any(part in SKIP_PARTS for part in path.relative_to(ROOT).parts)
        )
    )


def is_allowed_release_log(path: Path) -> bool:
    relative = path.relative_to(ROOT).as_posix()
    return relative.startswith("release_artifacts/") and path.suffix == ".log"


def check_python(path: Path, errors: list[str]) -> None:
    try:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError) as exc:
        errors.append(f"Python syntax: {path.relative_to(ROOT)}: {exc}")


def check_shell(path: Path, errors: list[str]) -> None:
    proc = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    if proc.returncode:
        errors.append(f"Shell syntax: {path.relative_to(ROOT)}: {proc.stderr.strip()}")


def check_jsonl(path: Path, errors: list[str]) -> None:
    rows = 0
    try:
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or not any(row.get(key) for key in ("prompt", "text", "instruction", "query")):
                    errors.append(f"JSONL schema: {path.relative_to(ROOT)}:{line_no}")
                rows += 1
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"JSONL parse: {path.relative_to(ROOT)}: {exc}")
    if rows < 4:
        errors.append(f"JSONL fixture too small: {path.relative_to(ROOT)} has {rows} rows")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-only", action="store_true", help="Retained for CI compatibility")
    parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append(f"Missing required file: {relative}")

    files = iter_files()
    for path in files:
        relative = path.relative_to(ROOT)
        if path.suffix in PROHIBITED_SUFFIXES:
            errors.append(f"Prohibited artifact: {relative}")
        if path.stat().st_size > 10 * 1024 * 1024:
            errors.append(f"Unexpected file larger than 10 MiB: {relative}")
        if path.suffix == ".py":
            check_python(path, errors)
        elif path.suffix == ".sh":
            check_shell(path, errors)
        elif path.suffix == ".jsonl" and "tests/fixtures" in relative.as_posix():
            check_jsonl(path, errors)

        if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > 10 * 1024 * 1024:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in SENSITIVE_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"{label}: {relative}")
        if "REPLACE_WITH_FINAL" in text and relative.as_posix() != "scripts/check_release.py":
            warnings.append(f"Release placeholder remains: {relative}")

    redistributed = sorted(
        path for path in files
        if path.relative_to(ROOT).as_posix().startswith(
            "experiments/sidechannel_suite/data/anti_heretic_prompts/modelscope_alpaca_safemt/"
        ) and path.suffix == ".jsonl"
    )
    if redistributed:
        errors.append(
            "Third-party ModelScope exports must not be committed: "
            + ", ".join(str(p.relative_to(ROOT)) for p in redistributed)
        )

    for warning in sorted(set(warnings)):
        print(f"WARNING: {warning}")
    if errors:
        for error in sorted(set(errors)):
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Release checks passed for {len(files)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
