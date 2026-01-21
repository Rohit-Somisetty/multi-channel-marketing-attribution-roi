"""Data loading and preparation helpers for marketing attribution EDA."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _ensure_datetime(df: pd.DataFrame, column: str) -> pd.Series:
    if not np.issubdtype(df[column].dtype, np.datetime64):
        return pd.to_datetime(df[column])
    return df[column]


def load_touchpoints(path: str | Path = "data/touchpoints.csv") -> pd.DataFrame:
    touchpoints = pd.read_csv(path)
    touchpoints["timestamp"] = _ensure_datetime(touchpoints, "timestamp")
    return touchpoints


def load_conversions(path: str | Path = "data/conversions.csv") -> pd.DataFrame:
    conversions = pd.read_csv(path)
    conversions["conversion_timestamp"] = _ensure_datetime(conversions, "conversion_timestamp")
    return conversions


def prepare_attribution_view(
    touchpoints: pd.DataFrame,
    conversions: pd.DataFrame,
    window_days: int = 30,
) -> pd.DataFrame:
    journeys = touchpoints.merge(
        conversions[["user_id", "conversion_timestamp", "conversion_value"]],
        on="user_id",
        how="left",
    )

    if "conversion_timestamp" in journeys:
        journeys["within_window"] = (
            journeys["conversion_timestamp"].notna()
            & (journeys["timestamp"] <= journeys["conversion_timestamp"])
            & (journeys["timestamp"] >= journeys["conversion_timestamp"] - pd.to_timedelta(window_days, unit="d"))
        )
    else:
        journeys["within_window"] = False
    return journeys


def journey_length_distribution(touchpoints: pd.DataFrame) -> pd.Series:
    return touchpoints.groupby("user_id").size()


def conversion_rates_by_channel(
    touchpoints: pd.DataFrame,
    conversions: pd.DataFrame,
) -> pd.DataFrame:
    converted_users = set(conversions["user_id"].unique())
    channel_users = touchpoints.groupby("channel")["user_id"].nunique().rename("touch_users")
    channel_conversions = (
        touchpoints[touchpoints["user_id"].isin(converted_users)]
        .groupby("channel")["user_id"]
        .nunique()
        .rename("converted_users")
    )
    rates = pd.concat([channel_users, channel_conversions], axis=1).fillna(0)
    rates["conversion_rate"] = rates["converted_users"] / rates["touch_users"].clip(lower=1)
    return rates.sort_values("conversion_rate", ascending=False)


def cost_by_channel(touchpoints: pd.DataFrame) -> pd.DataFrame:
    summary = touchpoints.groupby("channel").agg(
        touches=("user_id", "count"),
        spend=("cost", "sum"),
    )
    summary["avg_cost_per_touch"] = summary["spend"] / summary["touches"].clip(lower=1)
    return summary.sort_values("spend", ascending=False)


def naive_roi_by_channel(
    touchpoints: pd.DataFrame,
    conversions: pd.DataFrame,
) -> pd.DataFrame:
    conv_values = conversions.set_index("user_id")["conversion_value"]
    touches = touchpoints.copy()
    touches["journey_length"] = touches.groupby("user_id")["user_id"].transform("count")
    touches["attributed_value"] = touches["user_id"].map(conv_values).fillna(0)
    touches.loc[:, "attributed_value"] = touches["attributed_value"] / touches["journey_length"].clip(lower=1)

    channel_value = touches.groupby("channel")["attributed_value"].sum()
    channel_cost = touchpoints.groupby("channel")["cost"].sum()
    roi = (channel_value - channel_cost) / channel_cost.replace(0, np.nan)

    frame = pd.DataFrame(
        {
            "attributed_value": channel_value,
            "spend": channel_cost,
            "naive_roi": roi,
        }
    )
    return frame.sort_values("naive_roi", ascending=False)


def sample_user_journeys(
    touchpoints: pd.DataFrame,
    conversions: pd.DataFrame,
    n_samples: int = 5,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    converting_users = conversions["user_id"].unique()
    sample_size = min(n_samples, len(converting_users))
    sampled_users = rng.choice(converting_users, size=sample_size, replace=False)
    subset = touchpoints[touchpoints["user_id"].isin(sampled_users)].copy()
    subset.sort_values(["user_id", "timestamp"], inplace=True)
    return subset


def run_basic_checks(data_dir: str | Path = "data") -> dict[str, tuple[int, int]]:
    data_path = Path(data_dir)
    touches = load_touchpoints(data_path / "touchpoints.csv")
    conversions = load_conversions(data_path / "conversions.csv")

    metrics = {
        "touchpoints_shape": touches.shape,
        "conversions_shape": conversions.shape,
    }
    return metrics


__all__ = [
    "load_touchpoints",
    "load_conversions",
    "prepare_attribution_view",
    "journey_length_distribution",
    "conversion_rates_by_channel",
    "cost_by_channel",
    "naive_roi_by_channel",
    "sample_user_journeys",
    "run_basic_checks",
]
