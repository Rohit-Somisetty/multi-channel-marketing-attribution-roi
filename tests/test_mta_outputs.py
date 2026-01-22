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


def _ensure_heuristics() -> None:
    if not (DATA_DIR / "attribution_channel_metrics.csv").exists():
        if not (DATA_DIR / "touchpoints.csv").exists():
            _run_script("build_synthetic_data.py")
        _run_script("run_heuristic_attribution.py")


def test_markov_and_regression_outputs() -> None:
    _ensure_heuristics()
    markov_path = DATA_DIR / "attribution_markov_channel_metrics.csv"
    regression_path = DATA_DIR / "attribution_regression_channel_metrics.csv"
    removal_path = DATA_DIR / "markov_removal_effects.csv"
    if not (markov_path.exists() and regression_path.exists() and removal_path.exists()):
        _run_script("run_mta_models.py")
    for artifact in (markov_path, regression_path, removal_path):
        assert artifact.exists(), f"Missing MTA artifact: {artifact.name}"

    markov_df = pd.read_csv(markov_path)
    regression_df = pd.read_csv(regression_path)
    assert markov_df.shape[0] > 0
    assert regression_df.shape[0] > 0

    total_revenue = pd.read_csv(DATA_DIR / "conversions.csv")["conversion_value"].sum()
    attributed = markov_df["attributed_revenue"].sum()
    tolerance = max(0.05 * total_revenue, 1.0)
    assert abs(attributed - total_revenue) <= tolerance, "Markov revenue should reconcile with total conversions"
