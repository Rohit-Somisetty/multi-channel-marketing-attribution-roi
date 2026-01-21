"""Markov-chain removal effect attribution for marketing channels."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

START_STATE = "START"
CONVERT_STATE = "CONVERSION"
NULL_STATE = "NULL"


@dataclass
class MarkovConfig:
    """Configuration for building Markov paths and computation."""

    window_days: int = 30
    negative_sample_ratio: float = 1.0
    random_state: int = 44


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _build_positive_paths(touch_level: pd.DataFrame) -> pd.DataFrame:
    paths = (
        touch_level.sort_values(["conversion_id", "touch_order"])
        .groupby("conversion_id")
        .agg(
            user_id=("user_id", "first"),
            conversion_value=("conversion_value", "first"),
            channel_sequence=("channel", list),
        )
        .reset_index()
    )
    paths["path"] = paths["channel_sequence"].apply(lambda seq: [START_STATE] + seq + [CONVERT_STATE])
    paths["converted"] = True
    return paths


def _sample_negative_users(
    touchpoints: pd.DataFrame,
    conversions: pd.DataFrame,
    sample_size: int,
    window_days: int,
    seed: int,
) -> pd.DataFrame:
    converted_users = set(conversions["user_id"].unique())
    nonconv_users = [u for u in touchpoints["user_id"].unique() if u not in converted_users]
    if not nonconv_users or sample_size <= 0:
        return pd.DataFrame(columns=["user_id", "path"])

    rng = _rng(seed)
    sample_size = min(sample_size, len(nonconv_users))
    sampled = rng.choice(nonconv_users, size=sample_size, replace=False)

    rows = []
    for user_id in sampled:
        user_touches = touchpoints[touchpoints["user_id"] == user_id].sort_values("timestamp")
        if user_touches.empty:
            continue
        anchor_time = user_touches["timestamp"].max()
        window_start = anchor_time - pd.to_timedelta(window_days, unit="d")
        window_touches = user_touches[user_touches["timestamp"] >= window_start]
        if window_touches.empty:
            continue
        seq = window_touches["channel"].tolist()
        path = [START_STATE] + seq + [NULL_STATE]
        rows.append(
            {
                "conversion_id": np.nan,
                "user_id": user_id,
                "conversion_value": 0.0,
                "channel_sequence": seq,
                "path": path,
                "converted": False,
            }
        )
    return pd.DataFrame(rows)


def build_markov_paths(
    touch_level: pd.DataFrame,
    touchpoints: pd.DataFrame,
    conversions: pd.DataFrame,
    config: MarkovConfig | None = None,
) -> pd.DataFrame:
    """Return positive and sampled negative paths for Markov modeling."""

    cfg = config or MarkovConfig()
    positives = _build_positive_paths(touch_level)
    negatives = _sample_negative_users(
        touchpoints,
        conversions,
        sample_size=int(len(positives) * cfg.negative_sample_ratio),
        window_days=cfg.window_days,
        seed=cfg.random_state,
    )
    paths = pd.concat([positives, negatives], ignore_index=True)
    return paths


def _transition_counts(paths: Iterable[Sequence[str]]) -> dict[tuple[str, str], float]:
    counts: dict[tuple[str, str], float] = {}
    for path in paths:
        if not path or len(path) < 2:
            continue
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])
            counts[edge] = counts.get(edge, 0.0) + 1.0
    return counts


def _build_transition_matrix(states: list[str], counts: dict[tuple[str, str], float]) -> np.ndarray:
    index = {state: i for i, state in enumerate(states)}
    matrix = np.zeros((len(states), len(states)), dtype=float)
    for (src, dst), value in counts.items():
        if src in index and dst in index:
            matrix[index[src], index[dst]] += value

    row_sums = matrix.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        probs = np.divide(matrix, row_sums, where=row_sums != 0)
    for i, state in enumerate(states):
        if state in {CONVERT_STATE, NULL_STATE}:
            probs[i, :] = 0.0
            probs[i, i] = 1.0
        elif row_sums[i, 0] == 0:
            probs[i, i] = 1.0
    return probs


def _conversion_probability(matrix: np.ndarray, states: list[str]) -> float:
    absorbing = [CONVERT_STATE, NULL_STATE]
    transient = [s for s in states if s not in absorbing]
    idx = {state: i for i, state in enumerate(states)}

    if START_STATE not in transient:
        return 0.0

    transient_idx = [idx[s] for s in transient]
    absorbing_idx = [idx[s] for s in absorbing]

    Q = matrix[np.ix_(transient_idx, transient_idx)]
    R = matrix[np.ix_(transient_idx, absorbing_idx)]

    identity = np.eye(len(Q))
    try:
        N = np.linalg.inv(identity - Q)
    except np.linalg.LinAlgError:
        return 0.0

    B = N @ R
    start_pos = transient.index(START_STATE)
    convert_pos = absorbing.index(CONVERT_STATE)
    return float(B[start_pos, convert_pos])


def _remove_channel(paths: pd.DataFrame, channel: str) -> list[list[str]]:
    stripped_paths: list[list[str]] = []
    for path in paths["path"]:
        if channel not in path:
            stripped_paths.append(path)
            continue
        idx = path.index(channel)
        truncated = path[:idx] + [NULL_STATE]
        if len(truncated) < 2:
            truncated = [START_STATE, NULL_STATE]
        stripped_paths.append(truncated)
    return stripped_paths


def run_markov_attribution(
    touch_level: pd.DataFrame,
    touchpoints: pd.DataFrame,
    conversions: pd.DataFrame,
    cost_by_channel: dict[str, float],
    config: MarkovConfig | None = None,
    paths: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Compute channel-level Markov removal effect attribution and diagnostics."""

    cfg = config or MarkovConfig()
    if paths is None:
        paths = build_markov_paths(touch_level, touchpoints, conversions, cfg)
    if paths.empty:
        raise ValueError("Markov paths are empty.")

    channel_states = sorted({c for seq in paths["channel_sequence"] for c in seq})
    states = [START_STATE] + channel_states + [CONVERT_STATE, NULL_STATE]

    counts = _transition_counts(paths["path"])
    matrix = _build_transition_matrix(states, counts)
    baseline_prob = _conversion_probability(matrix, states)
    total_revenue = conversions["conversion_value"].sum()
    total_conversions = len(conversions)
    avg_conversion_value = total_revenue / total_conversions if total_conversions else 0.0

    removal_rows = []
    metrics_rows = []

    if baseline_prob <= 0:
        raise ValueError("Baseline conversion probability is zero; Markov chain degenerate.")

    for channel in channel_states:
        stripped = _remove_channel(paths, channel)
        stripped_counts = _transition_counts(stripped)
        stripped_states = [START_STATE] + [c for c in channel_states if c != channel] + [CONVERT_STATE, NULL_STATE]
        stripped_matrix = _build_transition_matrix(stripped_states, stripped_counts)
        removal_prob = _conversion_probability(stripped_matrix, stripped_states)
        effect_prob = max(baseline_prob - removal_prob, 0.0)
        attributed_value = (effect_prob / baseline_prob) * total_revenue
        conversions_lost = attributed_value / avg_conversion_value if avg_conversion_value else 0.0
        removal_rows.append(
            {
                "channel": channel,
                "removal_effect_pct": effect_prob,
                "removal_value": attributed_value,
            }
        )
        metrics_rows.append(
            {
                "model_name": "markov_chain",
                "channel": channel,
                "conversions": conversions_lost,
                "attributed_revenue": attributed_value,
                "cost": cost_by_channel.get(channel, 0.0),
            }
        )

    removal_df = pd.DataFrame(removal_rows).sort_values("removal_value", ascending=False)
    if total_revenue:
        removal_df["removal_share"] = removal_df["removal_value"] / total_revenue

    metrics_df = pd.DataFrame(metrics_rows)
    total_attr = metrics_df["attributed_revenue"].sum()
    if total_revenue and total_attr:
        scale = total_revenue / total_attr
        metrics_df["attributed_revenue"] *= scale
        metrics_df["conversions"] *= scale

    metrics_df["ROI"] = (metrics_df["attributed_revenue"] - metrics_df["cost"]) / metrics_df["cost"].replace(0, np.nan)
    metrics_df["ROAS"] = metrics_df["attributed_revenue"] / metrics_df["cost"].replace(0, np.nan)
    metrics_df["CPA"] = metrics_df["cost"] / metrics_df["conversions"].replace(0, np.nan)
    metrics_df["revenue_share"] = metrics_df["attributed_revenue"] / total_revenue if total_revenue else np.nan
    metrics_df = metrics_df.sort_values("attributed_revenue", ascending=False)
    diagnostics = {
        "transition_matrix": matrix,
        "states": states,
        "baseline_conversion_prob": baseline_prob,
    }
    return metrics_df, removal_df, diagnostics


__all__ = [
    "MarkovConfig",
    "build_markov_paths",
    "run_markov_attribution",
]
