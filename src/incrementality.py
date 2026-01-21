"""Incrementality (causal ROI) modeling helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import NearestNeighbors


@dataclass
class IncrementalityConfig:
    """Configuration for building user-level incrementality datasets."""

    index_date: str = "2024-05-01"
    pre_window_days: int = 60
    post_window_days: int = 30
    caliper: float = 0.05
    bootstrap_samples: int = 200
    random_state: int = 42


def _mode(series: pd.Series, default: str = "Unknown") -> str:
    if series.empty:
        return default
    modes = series.mode(dropna=True)
    if modes.empty:
        return str(series.iloc[-1]) if len(series) else default
    return str(modes.iloc[0])


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    weights = np.asarray(weights)
    denom = weights.sum()
    if denom == 0:
        return float("nan")
    return float(np.sum(values * weights) / denom)


def _weighted_var(values: np.ndarray, weights: np.ndarray, mean: float) -> float:
    if values.size == 0:
        return float("nan")
    weights = np.asarray(weights)
    denom = weights.sum()
    if denom == 0:
        return float("nan")
    return float(np.sum(weights * (values - mean) ** 2) / denom)


def build_incrementality_dataset(
    touchpoints: pd.DataFrame,
    conversions: pd.DataFrame,
    config: IncrementalityConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Create user-level features, treatments, and outcomes for causal analysis."""

    cfg = config or IncrementalityConfig()
    index_ts = pd.Timestamp(cfg.index_date)
    pre_start = index_ts - pd.to_timedelta(cfg.pre_window_days, unit="d")
    post_end = index_ts + pd.to_timedelta(cfg.post_window_days, unit="d")

    touches = touchpoints.copy()
    touches["timestamp"] = pd.to_datetime(touches["timestamp"])
    conversions = conversions.copy()
    conversions["conversion_timestamp"] = pd.to_datetime(conversions["conversion_timestamp"])

    touches_pre = touches[(touches["timestamp"] >= pre_start) & (touches["timestamp"] < index_ts)]
    conversions_pre = conversions[conversions["conversion_timestamp"] < index_ts]
    conversions_post = conversions[
        (conversions["conversion_timestamp"] >= index_ts) & (conversions["conversion_timestamp"] < post_end)
    ]

    user_ids = sorted(set(touches_pre["user_id"].unique()) | set(conversions_post["user_id"].unique()))
    dataset = pd.DataFrame({"user_id": user_ids})

    behavior = touches_pre.groupby("user_id").agg(
        journey_length_pre=("user_id", "count"),
        total_spend_pre=("cost", "sum"),
        last_touch_ts=("timestamp", "max"),
    )
    dataset = dataset.merge(behavior, on="user_id", how="left")
    dataset["journey_length_pre"] = dataset["journey_length_pre"].fillna(0)
    dataset["total_spend_pre"] = dataset["total_spend_pre"].fillna(0.0)
    dataset["recency_pre_days"] = (index_ts - dataset["last_touch_ts"]).dt.total_seconds() / 86400.0
    dataset.drop(columns=["last_touch_ts"], inplace=True)
    dataset["recency_pre_days"] = dataset["recency_pre_days"].fillna(cfg.pre_window_days + 1)
    dataset["recency_pre_days"] = dataset["recency_pre_days"].clip(lower=0, upper=365)

    channel_counts = (
        touches_pre.pivot_table(index="user_id", columns="channel", values="timestamp", aggfunc="count")
        .fillna(0)
        .rename_axis(None, axis=1)
    )
    core_channels = list(channel_counts.columns)
    for channel in core_channels:
        column = f"touches_pre_{channel}"
        series = channel_counts.get(channel)
        if series is None:
            dataset[column] = 0
        else:
            dataset[column] = dataset["user_id"].map(series).fillna(0)
    if core_channels:
        dataset["touches_pre_other"] = dataset["journey_length_pre"] - dataset[
            [f"touches_pre_{c}" for c in core_channels]
        ].sum(axis=1)
    else:
        dataset["touches_pre_other"] = dataset["journey_length_pre"]
    dataset["touches_pre_other"] = dataset["touches_pre_other"].clip(lower=0)

    dataset["treated_paid"] = (dataset.get("touches_pre_paid", 0) > 0).astype(int)
    dataset["treated_email"] = (dataset.get("touches_pre_email", 0) > 0).astype(int)

    mode_features = touches_pre.groupby("user_id").agg(
        device_mode=("device", _mode),
        geo_mode=("geo", _mode),
        segment_mode=("segment", _mode),
    )
    dataset = dataset.merge(mode_features, on="user_id", how="left")

    conversion_lookup = (
        conversions_pre.sort_values("conversion_timestamp")
        .groupby("user_id")
        .agg(market_mode=("market", "last"), product_mode=("product", "last"))
    )
    dataset = dataset.merge(conversion_lookup, on="user_id", how="left")
    for col in ["device_mode", "geo_mode", "segment_mode", "market_mode", "product_mode"]:
        dataset[col] = dataset[col].fillna("Unknown")

    outcomes = conversions_post.groupby("user_id").agg(
        y_rev=("conversion_value", "sum"),
        conversion_count=("conversion_value", "count"),
    )
    dataset = dataset.merge(outcomes, on="user_id", how="left")
    dataset["y_rev"] = dataset["y_rev"].fillna(0.0)
    dataset["y_conv"] = (dataset["conversion_count"].fillna(0) > 0).astype(int)
    dataset.drop(columns=["conversion_count"], inplace=True)

    numeric_cols = [
        "journey_length_pre",
        "total_spend_pre",
        "recency_pre_days",
    ] + [col for col in dataset.columns if col.startswith("touches_pre_")]
    categorical_cols = ["device_mode", "geo_mode", "segment_mode", "market_mode", "product_mode"]

    feature_meta = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "channels": core_channels,
        "channel_costs": touches_pre.groupby("channel")["cost"].sum().to_dict(),
    }

    return dataset, feature_meta


def prepare_design_matrix(
    dataset: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Return a design matrix with one-hot encoded categorical variables."""

    X_num = dataset.reindex(columns=numeric_cols, fill_value=0.0).astype(float)
    X_cat = pd.get_dummies(
        dataset.reindex(columns=categorical_cols).fillna("Unknown"),
        drop_first=False,
        dtype=float,
    )
    matrix = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    matrix = matrix.rename(columns={col: f"feat__{col}" for col in matrix.columns})
    feature_cols = matrix.columns.tolist()
    matrix.insert(0, "user_id", dataset["user_id"].values)
    return matrix, feature_cols


def fit_propensity_model(
    dataset: pd.DataFrame,
    treatment_col: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    random_state: int = 42,
    feature_matrix: pd.DataFrame | None = None,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, LogisticRegression, pd.DataFrame, list[str]]:
    """Train a logistic regression propensity model for a given treatment."""

    if feature_matrix is None or feature_cols is None:
        feature_matrix, feature_cols = prepare_design_matrix(dataset, numeric_cols, categorical_cols)
    channel_key = treatment_col.replace("treated_", "")
    excluded_cols = [col for col in feature_cols if col.endswith(f"touches_pre_{channel_key}")]
    model_features = [col for col in feature_cols if col not in excluded_cols]
    X = feature_matrix[model_features]
    y = dataset[treatment_col].astype(int)
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X, y)
    scores = model.predict_proba(X)[:, 1]
    prop_df = dataset[["user_id", treatment_col]].copy()
    prop_df["propensity_score"] = scores
    return prop_df, model, feature_matrix, feature_cols


def diagnose_overlap(
    propensity_entries: dict[str, tuple[pd.DataFrame, str]],
    output_path: Path,
    clip_bounds: tuple[float, float] = (0.02, 0.98),
) -> dict[str, pd.DataFrame]:
    """Plot propensity overlap and trim extreme scores for each treatment."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    entries = list(propensity_entries.items())
    n_rows = len(entries)
    fig, axes = plt.subplots(n_rows, 1, figsize=(8, 4 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    trimmed: dict[str, pd.DataFrame] = {}
    clip_low, clip_high = clip_bounds
    for ax, (label, (frame, treatment_col)) in zip(axes, entries, strict=False):
        treated_mask = frame[treatment_col] == 1
        treated_scores = frame.loc[treated_mask, "propensity_score"]
        control_scores = frame.loc[~treated_mask, "propensity_score"]
        ax.hist(control_scores, bins=20, alpha=0.6, label="Control", color="#c7d2eb")
        ax.hist(treated_scores, bins=20, alpha=0.6, label="Treated", color="#3572b0")
        ax.set_title(f"Propensity overlap: {label}")
        ax.set_ylabel("Users")
        ax.set_xlim(0, 1)
        ax.axvline(clip_low, color="#ffb347", linestyle="--", linewidth=1)
        ax.axvline(clip_high, color="#ffb347", linestyle="--", linewidth=1)
        ax.legend()
        mask = frame["propensity_score"].between(clip_low, clip_high)
        trimmed[label] = frame.loc[mask].copy()

    axes[-1].set_xlabel("Propensity score")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    return trimmed


def _nearest_neighbor_pairs(
    treated_df: pd.DataFrame,
    control_df: pd.DataFrame,
    propensity_col: str,
    caliper: float | None,
    feature_cols: list[str],
    match_feature_subset: Iterable[str] | None = None,
) -> pd.DataFrame:
    if treated_df.empty or control_df.empty:
        raise ValueError("Insufficient units for matching.")
    subset = [col for col in (match_feature_subset or []) if col in feature_cols]
    feature_list = [propensity_col] + subset
    combined = pd.concat([treated_df[feature_list], control_df[feature_list]], axis=0)
    means = combined.mean()
    stds = combined.std().replace(0, 1.0)
    treated_scaled = (treated_df[feature_list] - means) / stds
    control_scaled = (control_df[feature_list] - means) / stds
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control_scaled)
    distances, indices = nn.kneighbors(treated_scaled)
    matches = []
    for idx, (_dist, ctrl_idx) in enumerate(zip(distances[:, 0], indices[:, 0], strict=False)):
        prop_diff = abs(treated_df.iloc[idx][propensity_col] - control_df.iloc[ctrl_idx][propensity_col])
        if caliper is not None and prop_diff > caliper:
            continue
        row = {
            "treated_user_id": treated_df.iloc[idx]["user_id"],
            "control_user_id": control_df.iloc[ctrl_idx]["user_id"],
        }
        matches.append(row)
    return pd.DataFrame(matches)


def _compute_smd_table(
    df: pd.DataFrame,
    group_col: str,
    feature_cols: list[str],
    weight_col: str | None = None,
) -> pd.Series:
    weights = df[weight_col] if weight_col else pd.Series(1.0, index=df.index)
    treated_mask = df[group_col] == 1
    smd_values = {}
    for feature in feature_cols:
        treated_vals = df.loc[treated_mask, feature].astype(float).values
        control_vals = df.loc[~treated_mask, feature].astype(float).values
        treated_weights = weights.loc[treated_mask].values
        control_weights = weights.loc[~treated_mask].values
        mean_t = _weighted_mean(treated_vals, treated_weights)
        mean_c = _weighted_mean(control_vals, control_weights)
        var_t = _weighted_var(treated_vals, treated_weights, mean_t)
        var_c = _weighted_var(control_vals, control_weights, mean_c)
        pooled = np.sqrt(((var_t + var_c) / 2) + 1e-8)
        smd_values[feature] = 0.0 if pooled == 0 else (mean_t - mean_c) / pooled
    return pd.Series(smd_values)


def run_psm_matching(
    dataset: pd.DataFrame,
    treatment_col: str,
    propensity_col: str,
    outcome_cols: Iterable[str],
    feature_matrix: pd.DataFrame,
    feature_cols: list[str],
    caliper: float = 0.05,
    bootstrap_samples: int = 200,
    random_state: int = 42,
    match_feature_subset: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute nearest-neighbor matching and compute ATT with balance diagnostics."""

    df = dataset.merge(feature_matrix, on="user_id", how="left")
    treated_df = df[df[treatment_col] == 1].reset_index(drop=True)
    control_df = df[df[treatment_col] == 0].reset_index(drop=True)
    match_pairs = _nearest_neighbor_pairs(
        treated_df,
        control_df,
        propensity_col,
        caliper,
        feature_cols,
        match_feature_subset,
    )
    if match_pairs.empty and caliper < 0.2:
        match_pairs = _nearest_neighbor_pairs(
            treated_df,
            control_df,
            propensity_col,
            0.2,
            feature_cols,
            match_feature_subset,
        )
    if match_pairs.empty and caliper < 0.5:
        match_pairs = _nearest_neighbor_pairs(
            treated_df,
            control_df,
            propensity_col,
            0.5,
            feature_cols,
            match_feature_subset,
        )
    if match_pairs.empty:
        match_pairs = _nearest_neighbor_pairs(
            treated_df,
            control_df,
            propensity_col,
            None,
            feature_cols,
            match_feature_subset,
        )
    if match_pairs.empty:
        raise RuntimeError(f"No valid matches within caliper for {treatment_col}.")

    merged_pairs = match_pairs.merge(
        treated_df[["user_id", propensity_col] + list(outcome_cols)],
        left_on="treated_user_id",
        right_on="user_id",
        how="left",
        suffixes=("", "_treated"),
    ).drop(columns=["user_id"])
    merged_pairs = merged_pairs.merge(
        control_df[["user_id", propensity_col] + list(outcome_cols)],
        left_on="control_user_id",
        right_on="user_id",
        how="left",
        suffixes=("_treated", "_control"),
    ).drop(columns=["user_id"])

    rng = np.random.default_rng(random_state)
    att_rows = []
    for outcome in outcome_cols:
        treated_vals = merged_pairs[f"{outcome}_treated"].values
        control_vals = merged_pairs[f"{outcome}_control"].values
        diffs = treated_vals - control_vals
        estimate = float(np.mean(diffs))
        boot = []
        if len(diffs) > 1:
            for _ in range(bootstrap_samples):
                sample = rng.choice(diffs, size=len(diffs), replace=True)
                boot.append(np.mean(sample))
        else:
            boot = [estimate]
        ci_low, ci_high = np.percentile(boot, [2.5, 97.5]) if len(boot) > 1 else (estimate, estimate)
        att_rows.append(
            {
                "treatment": treatment_col,
                "outcome": outcome,
                "method": "PSM_ATT",
                "estimate": estimate,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "matched_pairs": len(diffs),
            }
        )

    balance_features = [col for col in feature_cols if "touches_pre_" not in col]
    before_input = df[[treatment_col] + balance_features].copy()
    before_table = _compute_smd_table(before_input, treatment_col, balance_features)
    matched_rows = pd.concat(
        [
            match_pairs[["treated_user_id"]].rename(columns={"treated_user_id": "user_id"}).assign(group=1),
            match_pairs[["control_user_id"]].rename(columns={"control_user_id": "user_id"}).assign(group=0),
        ],
        ignore_index=True,
    )
    matched_rows["weight"] = 1.0
    matched_rows = matched_rows.groupby(["user_id", "group"], as_index=False)["weight"].sum()
    matched_df = matched_rows.merge(feature_matrix, on="user_id", how="left")
    after_table = _compute_smd_table(
        matched_df.drop(columns=["user_id"]),
        "group",
        balance_features,
        weight_col="weight",
    )

    balance = pd.DataFrame(
        {
            "feature": balance_features,
            "smd_before": before_table[balance_features].values,
            "smd_after": after_table[balance_features].values,
            "treatment": treatment_col,
        }
    )

    return pd.DataFrame(att_rows), balance


def _clip_scores(scores: np.ndarray, eps: float = 0.01) -> np.ndarray:
    return np.clip(scores, eps, 1 - eps)


def ipw_ate(
    dataset: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    propensity_col: str,
    bootstrap_samples: int = 200,
    random_state: int = 42,
) -> dict[str, float]:
    """Inverse-probability weighted ATE estimate with bootstrap CI."""

    data = dataset[[treatment_col, outcome_col, propensity_col]].dropna()
    if data.empty:
        raise RuntimeError("No data available for IPW estimate.")
    t = data[treatment_col].values.astype(float)
    y = data[outcome_col].values.astype(float)
    p = _clip_scores(data[propensity_col].values.astype(float))

    def _estimate(sample_idx: np.ndarray) -> float:
        tt = t[sample_idx]
        yy = y[sample_idx]
        pp = p[sample_idx]
        w1 = tt / pp
        w0 = (1 - tt) / (1 - pp)
        mu1 = np.sum(w1 * yy) / np.sum(w1)
        mu0 = np.sum(w0 * yy) / np.sum(w0)
        return float(mu1 - mu0)

    rng = np.random.default_rng(random_state)
    point = _estimate(np.arange(len(data)))
    boot = []
    if len(data) > 1:
        for _ in range(bootstrap_samples):
            idx = rng.integers(0, len(data), len(data))
            boot.append(_estimate(idx))
    else:
        boot = [point]
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5]) if len(boot) > 1 else (point, point)
    return {
        "estimate": point,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def aipw_estimate(
    dataset: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    propensity_col: str,
    feature_matrix: pd.DataFrame,
    feature_cols: list[str],
    bootstrap_samples: int = 200,
    random_state: int = 42,
) -> dict[str, float]:
    """Augmented inverse-propensity weighted (doubly robust) estimator."""

    merged = dataset.merge(feature_matrix, on="user_id", how="left")
    t = merged[treatment_col].values.astype(float)
    y = merged[outcome_col].values.astype(float)
    p = _clip_scores(merged[propensity_col].values.astype(float))
    X = merged[feature_cols].values
    treated_mask = t == 1
    control_mask = t == 0
    if treated_mask.sum() == 0 or control_mask.sum() == 0:
        raise RuntimeError("Need treated and control examples for AIPW.")

    model1 = LinearRegression().fit(X[treated_mask], y[treated_mask])
    model0 = LinearRegression().fit(X[control_mask], y[control_mask])
    mu1 = model1.predict(X)
    mu0 = model0.predict(X)

    influence = mu1 + t * (y - mu1) / p - (mu0 + (1 - t) * (y - mu0) / (1 - p))
    point = float(np.mean(influence))

    rng = np.random.default_rng(random_state)
    boot = []
    if len(merged) > 1:
        for _ in range(bootstrap_samples):
            idx = rng.integers(0, len(merged), len(merged))
            boot.append(float(np.mean(influence[idx])))
    else:
        boot = [point]
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5]) if len(boot) > 1 else (point, point)
    return {
        "estimate": point,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def summarize_incremental_roi(
    lift_results: pd.DataFrame,
    dataset: pd.DataFrame,
    channel_costs: dict[str, float],
) -> pd.DataFrame:
    """Translate per-user lifts into incremental ROI/ROAS for stakeholders."""

    summary_rows = []
    population = len(dataset)
    for treatment_col in sorted(col for col in dataset.columns if col.startswith("treated_")):
        channel = treatment_col.replace("treated_", "")
        channel_spend = channel_costs.get(channel, 0.0)
        treated_population = float(dataset[treatment_col].sum())
        subset = lift_results[lift_results["treatment"] == treatment_col]
        for method in subset["method"].unique():
            method_rows = subset[subset["method"] == method]
            conv_row = method_rows[method_rows["outcome"] == "y_conv"].head(1)
            rev_row = method_rows[method_rows["outcome"] == "y_rev"].head(1)
            if rev_row.empty:
                continue
            lift_conv = float(conv_row["estimate"].iloc[0]) if not conv_row.empty else float("nan")
            lift_rev = float(rev_row["estimate"].iloc[0])
            ci_low = float(rev_row["ci_low"].iloc[0])
            ci_high = float(rev_row["ci_high"].iloc[0])
            population_base = treated_population if "ATT" in method else population
            incremental_conversions = lift_conv * population_base if not np.isnan(lift_conv) else float("nan")
            incremental_revenue = lift_rev * population_base
            incremental_cost = channel_spend
            incremental_roas = incremental_revenue / incremental_cost if incremental_cost else float("nan")
            incremental_roi = (
                (incremental_revenue - incremental_cost) / incremental_cost if incremental_cost else float("nan")
            )
            summary_rows.append(
                {
                    "treatment": channel,
                    "method": method,
                    "lift_conv": lift_conv,
                    "lift_rev": lift_rev,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "incremental_conversions": incremental_conversions,
                    "incremental_revenue": incremental_revenue,
                    "incremental_cost": incremental_cost,
                    "incremental_ROI": incremental_roi,
                    "incremental_ROAS": incremental_roas,
                }
            )

    return pd.DataFrame(summary_rows)


def compute_naive_incrementality_metrics(
    dataset: pd.DataFrame,
    channel_costs: dict[str, float],
) -> pd.DataFrame:
    """Quick treated vs control diff-in-means benchmark for each treatment."""

    rows = []
    for treatment_col in sorted(col for col in dataset.columns if col.startswith("treated_")):
        channel = treatment_col.replace("treated_", "")
        treated_mask = dataset[treatment_col] == 1
        control_mask = dataset[treatment_col] == 0
        treated_count = int(treated_mask.sum())
        control_count = int(control_mask.sum())
        if treated_count == 0 or control_count == 0:
            continue
        treated_conv = dataset.loc[treated_mask, "y_conv"].mean()
        control_conv = dataset.loc[control_mask, "y_conv"].mean()
        treated_rev = dataset.loc[treated_mask, "y_rev"].mean()
        control_rev = dataset.loc[control_mask, "y_rev"].mean()
        lift_conv = float(treated_conv - control_conv)
        lift_rev = float(treated_rev - control_rev)
        incremental_conversions = lift_conv * treated_count
        incremental_revenue = lift_rev * treated_count
        incremental_cost = channel_costs.get(channel, 0.0)
        incremental_roas = incremental_revenue / incremental_cost if incremental_cost else float("nan")
        incremental_roi = (
            (incremental_revenue - incremental_cost) / incremental_cost if incremental_cost else float("nan")
        )
        rows.append(
            {
                "treatment": channel,
                "method": "Naive_diff_in_means",
                "treated_users": treated_count,
                "control_users": control_count,
                "lift_conv": lift_conv,
                "lift_rev": lift_rev,
                "incremental_conversions": incremental_conversions,
                "incremental_revenue": incremental_revenue,
                "incremental_cost": incremental_cost,
                "incremental_ROAS": incremental_roas,
                "incremental_ROI": incremental_roi,
            }
        )

    return pd.DataFrame(rows)


__all__ = [
    "IncrementalityConfig",
    "build_incrementality_dataset",
    "prepare_design_matrix",
    "fit_propensity_model",
    "diagnose_overlap",
    "run_psm_matching",
    "ipw_ate",
    "aipw_estimate",
    "summarize_incremental_roi",
    "compute_naive_incrementality_metrics",
]
