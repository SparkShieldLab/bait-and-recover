import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from shared_utils import load_prompts_simple  # noqa: E402
from step4_multilayer import load_general_prompts  # noqa: E402


def _write_prompt(path: Path, text: str) -> None:
    path.write_text(json.dumps({"prompt": text}) + "\n", encoding="utf-8")


def test_local_prompt_paths_preserve_case(tmp_path: Path) -> None:
    prompt_dir = tmp_path / "MixedCasePrompts"
    prompt_dir.mkdir()
    _write_prompt(prompt_dir / "good.jsonl", "A benign request")
    _write_prompt(prompt_dir / "bad.jsonl", "A synthetic unsafe request")

    source = f"local:{prompt_dir}"
    good, bad = load_prompts_simple(1, 1, source=source)
    general = load_general_prompts(1, source=source)

    assert len(good) == 1
    assert len(bad) == 1
