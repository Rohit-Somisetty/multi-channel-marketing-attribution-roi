"""Runs the full synthetic data + modeling pipelines in sequence."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

STEPS = [
    "build_synthetic_data.py",
    "run_heuristic_attribution.py",
    "run_mta_models.py",
    "run_incrementality.py",
]


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    scripts_dir = project_root / "scripts"
    env = os.environ.copy()
    mode_label = "FAST" if env.get("FAST") == "1" else "FULL"
    print(f"Running marketing analytics pipeline in {mode_label} mode...")

    for script in STEPS:
        cmd = [sys.executable, str(scripts_dir / script)]
        print(f"➡️  {script}")
        subprocess.run(cmd, check=True, env=env)

    print("✅ All pipeline steps completed")


if __name__ == "__main__":
    main()
