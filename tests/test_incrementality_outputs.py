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


def _ensure_incrementality_outputs() -> None:
    required = [
        DATA_DIR / "incremental_roi_summary.csv",
        DATA_DIR / "naive_incrementality_metrics.csv",
    ]
    if not all(path.exists() for path in required):
        if not (DATA_DIR / "touchpoints.csv").exists():
            _run_script("build_synthetic_data.py")
        _run_script("run_heuristic_attribution.py")
        _run_script("run_mta_models.py")
        _run_script("run_incrementality.py")


def test_incremental_roi_files_exist_and_columns() -> None:
    _ensure_incrementality_outputs()
    roi_summary = pd.read_csv(DATA_DIR / "incremental_roi_summary.csv")
    naive_metrics = pd.read_csv(DATA_DIR / "naive_incrementality_metrics.csv")
    assert {"incremental_ROI", "incremental_ROAS"}.issubset(roi_summary.columns)
    assert {"incremental_ROI", "incremental_ROAS"}.issubset(naive_metrics.columns)
    assert roi_summary.shape[0] > 0
    assert naive_metrics.shape[0] > 0
