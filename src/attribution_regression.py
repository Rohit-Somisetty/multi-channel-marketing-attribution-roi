"""Logistic-regression-based multi-touch attribution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

CHANNEL_PREFIX = "touch_count_"
SPEND_PREFIX = "touch_spend_"


@dataclass
class RegressionConfig:
    window_days: int = 30
    negative_sample_ratio: float = 1.0
    random_state: int = 7
    test_size: float = 0.2


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _summarize_window(
    user_id: int,
    touches: pd.DataFrame,
    anchor_time: pd.Timestamp,
    conversion_value: float,
    converted: bool,
    market: str,
    channels: list[str],
) -> dict[str, object]:
    if touches.empty:
        return {}
    ordered = touches.sort_values("timestamp")
    last_touch = ordered.iloc[-1]
    data: dict[str, object] = {
        "user_id": user_id,
        "converted": int(converted),
        "conversion_value": conversion_value,
        "last_touch_channel": last_touch["channel"],
        "journey_length": len(touches),
        "device": last_touch.get("device", "unknown"),
        "geo": last_touch.get("geo", "unknown"),
        "segment": last_touch.get("segment", "unknown"),
        "market": market,
        "days_since_last_touch": (anchor_time - last_touch["timestamp"]).days,
    }
    channel_counts = touches["channel"].value_counts()
    channel_spend = touches.groupby("channel")["cost"].sum()
    for channel in channels:
        data[f"{CHANNEL_PREFIX}{channel}"] = float(channel_counts.get(channel, 0.0))
        data[f"{SPEND_PREFIX}{channel}"] = float(channel_spend.get(channel, 0.0))
    return data


def build_regression_dataset(
    touch_level: pd.DataFrame,
    touchpoints: pd.DataFrame,
    conversions: pd.DataFrame,
    config: RegressionConfig | None = None,
) -> pd.DataFrame:
    cfg = config or RegressionConfig()
    channels = sorted(touchpoints["channel"].unique())
    touchpoints = touchpoints.copy()
    touchpoints["timestamp"] = pd.to_datetime(touchpoints["timestamp"])
    conversions = conversions.copy().reset_index(drop=True)
    if "conversion_id" not in conversions.columns:
        conversions["conversion_id"] = np.arange(1, len(conversions) + 1)

    rows: list[dict[str, object]] = []
    # Positive samples
    grouped = touch_level.sort_values(["conversion_id", "touch_order"]).groupby("conversion_id")
    for conv_id, group in grouped:
        anchor_time = group["conversion_timestamp"].iloc[0]
        touches = group.copy()
        market_row = conversions.loc[conversions["conversion_id"] == conv_id, "market"]
        market = market_row.iloc[0] if not market_row.empty else "NA"
        row = _summarize_window(
            user_id=int(group["user_id"].iloc[0]),
            touches=touches,
            anchor_time=anchor_time,
            conversion_value=float(group["conversion_value"].iloc[0]),
            converted=True,
            market=market,
            channels=channels,
        )
        if row:
            rows.append(row)

    converted_users = set(conversions["user_id"].unique())
    negative_candidates = touchpoints[~touchpoints["user_id"].isin(converted_users)]["user_id"].unique()
    if len(negative_candidates) == 0:
        negative_sample = []
    else:
        rng = _rng(cfg.random_state)
        desired = int(len(rows) * cfg.negative_sample_ratio)
        if cfg.negative_sample_ratio > 0 and desired == 0:
            desired = 1
        sample_size = min(len(negative_candidates), len(rows), desired)
        negative_sample = rng.choice(negative_candidates, size=sample_size, replace=False)

    for user_id in negative_sample:
        user_touches = touchpoints[touchpoints["user_id"] == user_id].sort_values("timestamp")
        if user_touches.empty:
            continue
        anchor_time = user_touches["timestamp"].max()
        window_start = anchor_time - pd.to_timedelta(cfg.window_days, unit="d")
        window_touches = user_touches[user_touches["timestamp"] >= window_start]
        if window_touches.empty:
            continue
        row = _summarize_window(
            user_id=int(user_id),
            touches=window_touches,
            anchor_time=anchor_time,
            conversion_value=0.0,
            converted=False,
            market="NA",
            channels=channels,
        )
        if row:
            rows.append(row)

    if not rows:
        raise ValueError("Regression dataset is empty.")

    df = pd.DataFrame(rows)
    df.fillna({"market": "NA", "device": "unknown", "geo": "unknown", "segment": "unknown"}, inplace=True)
    return df


def _train_logistic(df: pd.DataFrame, cfg: RegressionConfig) -> tuple[LogisticRegression, list[str], float]:
    feature_cols = [c for c in df.columns if c not in {"user_id", "converted", "conversion_value"}]
    X = pd.get_dummies(df[feature_cols], drop_first=False)
    y = df["converted"].astype(int)
    feature_names = X.columns.tolist()

    rng = _rng(cfg.random_state)
    user_ids = df["user_id"].unique()
    rng.shuffle(user_ids)
    split_idx = int(len(user_ids) * (1 - cfg.test_size))
    train_users = set(user_ids[:split_idx])
    train_mask = df["user_id"].isin(train_users)
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    if len(y_test) > 0:
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = float("nan")

    return model, feature_names, auc


def run_regression_attribution(
    touch_level: pd.DataFrame,
    touchpoints: pd.DataFrame,
    conversions: pd.DataFrame,
    cost_by_channel: dict[str, float],
    config: RegressionConfig | None = None,
    dataset: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config or RegressionConfig()
    if dataset is None:
        dataset = build_regression_dataset(touch_level, touchpoints, conversions, cfg)

    model, feature_names, auc = _train_logistic(dataset, cfg)
    coefficients = pd.Series(model.coef_[0], index=feature_names)
    odds = np.exp(coefficients)
    odds_df = pd.DataFrame(
        {
            "feature": coefficients.index,
            "coefficient": coefficients.values,
            "odds_ratio": odds.values,
        }
    ).sort_values("odds_ratio", ascending=False)

    channel_cols = [c for c in feature_names if c.startswith(CHANNEL_PREFIX)]
    total_revenue = conversions["conversion_value"].sum()
    total_conversions = len(conversions)
    raw_scores = {}
    for col in channel_cols:
        channel = col.replace(CHANNEL_PREFIX, "")
        mean_count = dataset[col].mean() if col in dataset.columns else 0.0
        coef_odds = odds.get(col, 1.0)
        weight = mean_count * (coef_odds - 1)
        raw_scores[channel] = max(weight, 0.0)

    total_score = sum(raw_scores.values())
    if total_score == 0:
        raw_scores = {c.replace(CHANNEL_PREFIX, ""): dataset[c].mean() for c in channel_cols}
        total_score = sum(raw_scores.values()) or 1.0

    metrics_rows = []
    for channel, score in raw_scores.items():
        share = score / total_score if total_score else 0.0
        attributed_value = share * total_revenue
        conversions_attr = share * total_conversions
        cost = cost_by_channel.get(channel, 0.0)
        roi = (attributed_value - cost) / cost if cost else np.nan
        roas = attributed_value / cost if cost else np.nan
        cpa = cost / conversions_attr if conversions_attr else np.nan
        revenue_share = attributed_value / total_revenue if total_revenue else np.nan
        metrics_rows.append(
            {
                "model_name": "regression_logit",
                "channel": channel,
                "conversions": conversions_attr,
                "attributed_revenue": attributed_value,
                "cost": cost,
                "ROI": roi,
                "ROAS": roas,
                "CPA": cpa,
                "revenue_share": revenue_share,
            }
        )

    metrics_df = pd.DataFrame(metrics_rows).sort_values("attributed_revenue", ascending=False)
    odds_df["model_auc"] = auc
    return metrics_df, odds_df


__all__ = [
    "RegressionConfig",
    "build_regression_dataset",
    "run_regression_attribution",
]
