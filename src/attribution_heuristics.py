"""Heuristic attribution utilities for marketing ROI analysis."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AttributionConfig:
    """Configuration for building attribution-ready tables."""

    window_days: int = 30
    time_decay_half_life_days: float = 7.0


def _ensure_datetime(series: pd.Series) -> pd.Series:
    if not np.issubdtype(series.dtype, np.datetime64):
        return pd.to_datetime(series)
    return series


def build_attribution_tables(
    touchpoints: pd.DataFrame,
    conversions: pd.DataFrame,
    config: AttributionConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct journey-level and touch-level tables limited to the attribution window."""

    cfg = config or AttributionConfig()
    convs = conversions.copy().reset_index(drop=True)
    convs["conversion_timestamp"] = _ensure_datetime(convs["conversion_timestamp"])
    convs["conversion_id"] = np.arange(1, len(convs) + 1)

    touches = touchpoints.copy()
    touches["timestamp"] = _ensure_datetime(touches["timestamp"])

    merged = touches.merge(
        convs[
            [
                "conversion_id",
                "user_id",
                "conversion_timestamp",
                "conversion_value",
            ]
        ],
        on="user_id",
        how="inner",
        suffixes=("", "_conversion"),
    )

    window = pd.to_timedelta(cfg.window_days, unit="d")
    mask = (merged["timestamp"] <= merged["conversion_timestamp"]) & (
        merged["timestamp"] >= merged["conversion_timestamp"] - window
    )
    filtered = merged.loc[mask].copy()

    if filtered.empty:
        return pd.DataFrame(), pd.DataFrame()

    filtered.sort_values(["conversion_id", "timestamp", "campaign_id"], inplace=True)
    filtered["touch_order"] = filtered.groupby("conversion_id").cumcount() + 1
    filtered["time_to_conversion_days"] = (
        filtered["conversion_timestamp"] - filtered["timestamp"]
    ).dt.total_seconds() / 86400.0

    touch_cols = [
        "conversion_id",
        "user_id",
        "conversion_timestamp",
        "conversion_value",
        "channel",
        "campaign_id",
        "timestamp",
        "cost",
        "touch_order",
        "time_to_conversion_days",
    ]
    touch_level = filtered[touch_cols].copy()
    touch_level["journey_length"] = touch_level.groupby("conversion_id")["conversion_id"].transform("count")

    journeys = (
        touch_level.groupby("conversion_id")
        .agg(
            user_id=("user_id", "first"),
            conversion_timestamp=("conversion_timestamp", "first"),
            conversion_value=("conversion_value", "first"),
            journey_length=("journey_length", "first"),
            touches=("timestamp", lambda x: [ts.isoformat() for ts in x]),
            channels=("channel", list),
            campaigns=("campaign_id", list),
            costs=("cost", lambda x: [float(c) for c in x]),
        )
        .reset_index()
    )

    return journeys, touch_level


def _first_touch_weights(group: pd.DataFrame) -> np.ndarray:
    weights = np.zeros(len(group), dtype=float)
    if len(group):
        weights[0] = 1.0
    return weights


def _last_touch_weights(group: pd.DataFrame) -> np.ndarray:
    weights = np.zeros(len(group), dtype=float)
    if len(group):
        weights[-1] = 1.0
    return weights


def _linear_weights(group: pd.DataFrame) -> np.ndarray:
    if not len(group):
        return np.array([])
    return np.full(len(group), 1.0 / len(group))


def _time_decay_weights(group: pd.DataFrame, half_life_days: float) -> np.ndarray:
    if not len(group):
        return np.array([])
    decay = 0.5 ** (group["time_to_conversion_days"].values / half_life_days)
    total = decay.sum()
    if total == 0:
        return np.full(len(group), 1.0 / len(group))
    return decay / total


MODEL_BUILDERS: dict[str, Callable[[pd.DataFrame, float], np.ndarray]] = {
    "first_touch": lambda g, _: _first_touch_weights(g),
    "last_touch": lambda g, _: _last_touch_weights(g),
    "linear": lambda g, _: _linear_weights(g),
    "time_decay": lambda g, half_life: _time_decay_weights(g, half_life),
}


def apply_heuristic_model(
    touch_level: pd.DataFrame,
    model_name: str,
    half_life_days: float = 7.0,
) -> pd.DataFrame:
    """Attach credit weights for a given heuristic model to the touch-level table."""

    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Unsupported model: {model_name}")

    groups = []
    for _, group in touch_level.groupby("conversion_id", sort=False):
        weights = MODEL_BUILDERS[model_name](group, half_life_days)
        result = group.copy()
        result["credit_weight"] = weights
        groups.append(result)

    if not groups:
        return pd.DataFrame()

    weighted = pd.concat(groups, ignore_index=True)
    # Conversions without touches will be absent; weights already sum to 1 per conversion.
    weighted["model_name"] = model_name
    weighted["conversion_credit"] = weighted["credit_weight"]
    weighted["attributed_revenue"] = weighted["credit_weight"] * weighted["conversion_value"]
    return weighted


def run_all_heuristics(
    touch_level: pd.DataFrame,
    config: AttributionConfig | None = None,
) -> pd.DataFrame:
    """Compute touch-level credit for all supported heuristic models."""

    cfg = config or AttributionConfig()
    outputs = []
    for model in MODEL_BUILDERS.keys():
        outputs.append(apply_heuristic_model(touch_level, model, cfg.time_decay_half_life_days))
    if not outputs:
        return pd.DataFrame()
    combined = pd.concat(outputs, ignore_index=True)
    combined["timestamp_iso"] = combined["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return combined


def aggregate_roi_metrics(
    attribution_touches: pd.DataFrame,
    level: str,
) -> pd.DataFrame:
    """Aggregate touch-level credits into ROI metrics for a channel or campaign level.

    Costs are aggregated as the full spend recorded on each touch (i.e., spend is not
    prorated by attribution credit) so ROI reflects the true dollars paid for the
    impressions/emails/events that contributed to each conversion.
    """

    if level not in {"channel", "campaign_id"}:
        raise ValueError("level must be 'channel' or 'campaign_id'")

    group_cols = ["model_name", level]
    grouped = (
        attribution_touches.groupby(group_cols, dropna=False)
        .agg(
            conversions=("conversion_credit", "sum"),
            attributed_revenue=("attributed_revenue", "sum"),
            cost=("cost", "sum"),
        )
        .reset_index()
    )

    grouped["ROI"] = (grouped["attributed_revenue"] - grouped["cost"]) / grouped["cost"].replace(0, np.nan)
    grouped["ROAS"] = grouped["attributed_revenue"] / grouped["cost"].replace(0, np.nan)
    grouped["CPA"] = grouped["cost"] / grouped["conversions"].replace(0, np.nan)

    grouped["revenue_share"] = grouped["attributed_revenue"] / grouped.groupby("model_name")[
        "attributed_revenue"
    ].transform("sum")

    ordered_cols = [
        "model_name",
        level,
        "conversions",
        "attributed_revenue",
        "cost",
        "ROI",
        "ROAS",
        "CPA",
        "revenue_share",
    ]
    return grouped[ordered_cols]


__all__ = [
    "AttributionConfig",
    "build_attribution_tables",
    "run_all_heuristics",
    "aggregate_roi_metrics",
    "apply_heuristic_model",
]
