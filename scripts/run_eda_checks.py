"""Quick diagnostics for synthetic marketing datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import data_prep


def main(data_dir: str = "data") -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / data_dir

    touches = data_prep.load_touchpoints(data_path / "touchpoints.csv")
    conversions = data_prep.load_conversions(data_path / "conversions.csv")

    journey_lengths = data_prep.journey_length_distribution(touches)
    channel_rates = data_prep.conversion_rates_by_channel(touches, conversions)
    channel_costs = data_prep.cost_by_channel(touches)
    naive_roi = data_prep.naive_roi_by_channel(touches, conversions)

    print("Dataset overview:")
    print(f"  Touchpoints: {touches.shape[0]:,} rows | {touches['user_id'].nunique():,} users")
    print(f"  Conversions: {conversions.shape[0]:,} rows")
    print(f"  Avg journey length: {journey_lengths.mean():.2f} touches")

    print("\nConversion rates by channel:")
    print(channel_rates.round(3).to_string())

    print("\nCost by channel:")
    print(channel_costs.round(2).to_string())

    print("\nNaive ROI by channel:")
    print(naive_roi.round(3).to_string())


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    main()
