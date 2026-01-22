"""Builds synthetic marketing touchpoints and conversions datasets."""

from __future__ import annotations

import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from src.data_generation import GenerationConfig, build_synthetic_marketing_data


def main(output_dir: str = "data") -> None:
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
