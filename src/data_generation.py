"""Utilities to build synthetic multi-channel marketing datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CHANNELS = ["paid", "email", "organic", "event"]
DEVICES = ["mobile", "desktop", "tablet"]
GEOS = ["NA", "EMEA", "APAC", "LATAM"]
SEGMENTS = ["SMB", "Mid-Market", "Enterprise"]
PRODUCTS = ["Core", "Analytics", "Premium"]
MARKETS = ["US", "Canada", "UK", "Germany", "Australia", "Brazil", "Singapore"]

_CAMPAIGNS = {
    "paid": ["search_brand", "search_generic", "display_ret", "social_acq"],
    "email": ["welcome_nurture", "promo_drop", "event_invite"],
    "organic": ["blog_content", "webinar_on_demand", "community_post"],
    "event": ["field_dinner", "trade_show", "webinar_live"],
}
_TOUCH_TYPES = {
    "paid": ["impression", "click"],
    "email": ["open", "click"],
    "organic": ["impression", "click"],
    "event": ["register", "attend"],
}


@dataclass
class GenerationConfig:
    num_users: int = 10000
    min_touches: int = 2
    max_touches: int = 8
    target_touchpoints: int = 50000
    target_conversions: int = 8000
    attribution_days: int = 30
    seed: int = 42


def _rng(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_user_profiles(config: GenerationConfig) -> pd.DataFrame:
    rng = _rng(config.seed)
    user_ids = np.arange(1, config.num_users + 1)

    is_high_intent = rng.random(config.num_users) < 0.35
    segment = rng.choice(SEGMENTS, size=config.num_users, p=[0.45, 0.35, 0.2])
    geo = rng.choice(GEOS, size=config.num_users, p=[0.5, 0.2, 0.2, 0.1])

    intent_base = 0.15 + rng.normal(0, 0.03, config.num_users)
    intent_score = intent_base + np.where(is_high_intent, 0.35, 0.0)

    users = pd.DataFrame(
        {
            "user_id": user_ids,
            "is_high_intent": is_high_intent,
            "segment": segment,
            "geo": geo,
            "intent_score": intent_score,
        }
    )
    return users


def _channel_probs(is_high_intent: bool) -> np.ndarray:
    if is_high_intent:
        return np.array([0.45, 0.28, 0.17, 0.10])
    return np.array([0.28, 0.20, 0.37, 0.15])


def _sample_cost(channel: str, rng: np.random.Generator) -> float:
    if channel == "paid":
        return float(rng.normal(25, 8))
    if channel == "email":
        return float(rng.normal(2, 0.7))
    if channel == "organic":
        return float(max(rng.normal(0.8, 0.3), 0.1))
    if channel == "event":
        return float(rng.normal(75, 20))
    return float(rng.normal(10, 3))


def generate_touchpoints(users: pd.DataFrame, config: GenerationConfig) -> pd.DataFrame:
    rng = _rng(config.seed + 1)
    rows = []
    base_start = pd.Timestamp("2024-01-01")
    time_horizon_days = 120

    for _, row in users.iterrows():
        num_touches = int(rng.integers(config.min_touches, config.max_touches + 1))
        relative_positions = np.sort(rng.beta(2, 1.5, num_touches))
        start_offset = int(rng.integers(0, time_horizon_days // 2))
        start_time = base_start + pd.Timedelta(days=start_offset)
        channels_probs = _channel_probs(bool(row.is_high_intent))

        for rel in relative_positions:
            timestamp = start_time + pd.Timedelta(days=float(rel * time_horizon_days))
            channel = rng.choice(CHANNELS, p=channels_probs)
            touch_type = rng.choice(_TOUCH_TYPES[channel])
            campaign = rng.choice(_CAMPAIGNS[channel])
            device = rng.choice(DEVICES, p=[0.55, 0.35, 0.10])
            cost = max(_sample_cost(channel, rng), 0.05)

            rows.append(
                {
                    "user_id": row.user_id,
                    "timestamp": timestamp,
                    "channel": channel,
                    "campaign_id": campaign,
                    "touch_type": touch_type,
                    "cost": round(cost, 2),
                    "device": device,
                    "geo": row.geo,
                    "segment": row.segment,
                }
            )

    touchpoints = pd.DataFrame(rows)
    touchpoints.sort_values(["user_id", "timestamp"], inplace=True)

    if len(touchpoints) < config.target_touchpoints:
        # Slightly increase sample density by duplicating late-stage touches with jitter.
        deficit = config.target_touchpoints - len(touchpoints)
        extra = touchpoints.sample(deficit, replace=True, random_state=config.seed + 2).copy()
        extra["timestamp"] = extra["timestamp"] + pd.to_timedelta(
            _rng(config.seed + 3).integers(1, 4, len(extra)), unit="h"
        )
        touchpoints = pd.concat([touchpoints, extra], ignore_index=True)
        touchpoints.sort_values(["user_id", "timestamp"], inplace=True)

    touchpoints.reset_index(drop=True, inplace=True)
    return touchpoints


def generate_conversions(
    users: pd.DataFrame,
    touchpoints: pd.DataFrame,
    config: GenerationConfig,
) -> pd.DataFrame:
    rng = _rng(config.seed + 4)
    last_touch = touchpoints.groupby("user_id")["timestamp"].max().rename("last_touch_ts")
    users = users.merge(last_touch, on="user_id", how="left")
    users["last_touch_ts"] = users["last_touch_ts"].fillna(pd.Timestamp("2024-01-01"))

    noise = rng.normal(0, 0.08, len(users))
    users["conversion_score"] = users["intent_score"] + noise
    target = min(config.target_conversions, len(users))
    converters = users.nlargest(target, "conversion_score").copy()

    offsets = rng.integers(1, config.attribution_days + 1, len(converters))
    converters["conversion_timestamp"] = converters["last_touch_ts"] + pd.to_timedelta(offsets, unit="h")

    value_scale = {
        "SMB": (150, 40),
        "Mid-Market": (450, 120),
        "Enterprise": (1200, 400),
    }
    values = []
    products = []
    markets = []
    for seg in converters["segment"].values:
        mean, std = value_scale.get(seg, (300, 80))
        values.append(max(rng.normal(mean, std), 50))
        products.append(rng.choice(PRODUCTS, p=[0.5, 0.3, 0.2]))
        markets.append(rng.choice(MARKETS))

    converters["conversion_value"] = np.round(values, 2)
    converters["product"] = products
    converters["market"] = markets

    conversions = converters[
        [
            "user_id",
            "conversion_timestamp",
            "conversion_value",
            "product",
            "market",
        ]
    ].copy()
    conversions.sort_values("conversion_timestamp", inplace=True)
    conversions.reset_index(drop=True, inplace=True)
    return conversions


def build_synthetic_marketing_data(
    output_dir: str | Path,
    config: GenerationConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config or GenerationConfig()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    users = generate_user_profiles(cfg)
    touchpoints = generate_touchpoints(users, cfg)
    conversions = generate_conversions(users, touchpoints, cfg)

    touchpoints.to_csv(output_path / "touchpoints.csv", index=False)
    conversions.to_csv(output_path / "conversions.csv", index=False)
    return touchpoints, conversions


__all__ = [
    "GenerationConfig",
    "build_synthetic_marketing_data",
    "generate_user_profiles",
    "generate_touchpoints",
    "generate_conversions",
]
