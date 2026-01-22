"""Builds synthetic marketing touchpoints and conversions datasets."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    current = start
    for _ in range(6):
        candidate_src = current / "src"
        if candidate_src.is_dir() and (candidate_src / "data_generation.py").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not locate repo root containing src/data_generation.py")


def _ensure_repo_root_on_path() -> None:
    repo_root = _find_repo_root(Path(__file__).resolve())
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def main(output_dir: str = "data") -> None:
    _ensure_repo_root_on_path()

    from src.data_generation import GenerationConfig, build_synthetic_marketing_data

    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / output_dir
    fast_mode = os.getenv("FAST") == "1"
    if fast_mode:
        config = GenerationConfig(
            num_users=3000,
            target_touchpoints=15000,
            target_conversions=2500,
            seed=7,
        )
    else:
        config = GenerationConfig()
    touchpoints, conversions = build_synthetic_marketing_data(data_dir, config=config)

    mode_label = "FAST" if fast_mode else "FULL"
    print(f"[{mode_label}] ✅ Wrote touchpoints to {data_dir / 'touchpoints.csv'} ({len(touchpoints):,} rows)")
    print(f"[{mode_label}] ✅ Wrote conversions to {data_dir / 'conversions.csv'} ({len(conversions):,} rows)")


if __name__ == "__main__":
    main()
