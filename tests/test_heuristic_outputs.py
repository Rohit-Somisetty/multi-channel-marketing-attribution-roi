from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _run_script(script_name: str) -> None:
    env = dict(os.environ)
    env["FAST"] = env.get("FAST", "1")
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    subprocess.run([sys.executable, str(SCRIPTS_DIR / script_name)], check=True, env=env)


def _ensure_data() -> None:
    if not (DATA_DIR / "touchpoints.csv").exists():
        _run_script("build_synthetic_data.py")


def test_channel_metrics_has_roi_columns() -> None:
    _ensure_data()
    channel_metrics = DATA_DIR / "attribution_channel_metrics.csv"
    if not channel_metrics.exists():
        _run_script("run_heuristic_attribution.py")
    df = pd.read_csv(channel_metrics)
    expected = {"model_name", "channel", "ROI", "ROAS", "CPA"}
    missing = expected.difference(df.columns)
    assert df.shape[0] > 0, "heuristic channel metrics should not be empty"
    assert not missing, f"Missing columns: {missing}"


def test_touch_level_created() -> None:
    _ensure_data()
    touch_level = DATA_DIR / "attribution_touch_level.csv"
    if not touch_level.exists():
        _run_script("run_heuristic_attribution.py")
    assert touch_level.exists(), "Touch-level attribution output is required for downstream models"
