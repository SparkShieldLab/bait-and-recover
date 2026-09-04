from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HERETIC_RUNNER = (
    ROOT / "experiments" / "sidechannel_suite" / "scripts" / "run_heretic.sh"
)


def test_target_tag_overrides_stale_target_model(tmp_path: Path) -> None:
    """A pipeline tag must not be shadowed by an inherited target path."""
    stale_target = tmp_path / "stale_target"
    stale_target.mkdir()
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    environment = os.environ.copy()
    environment.update(
        {
            "BASE_MODEL": str(stale_target),
            "GPU_ID": "",
            "LOG_DIR": str(tmp_path / "logs"),
            "MODEL_SAVE_DIR": str(model_dir),
            "PYTHON": sys.executable,
            "TARGET_MODEL": str(stale_target),
            "TARGET_TAG": "fresh",
        }
    )
    result = subprocess.run(
        ["bash", str(HERETIC_RUNNER)],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    output = result.stdout + result.stderr
    assert (
        f"Defended model not found: {model_dir}/step4_merged_fresh" in output
    )
