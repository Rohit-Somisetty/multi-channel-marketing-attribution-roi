"""Runs heuristic attribution pipeline and writes ROI outputs + figures."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def _find_repo_root(start: Path) -> Path:
    current = start
    for _ in range(8):
        if (current / "src").is_dir() and (current / "pyproject.toml").exists():
            return current
        if current.parent == current:
            break
        current = current.parent
    return start


def _ensure_repo_root_on_path() -> Path:
    repo_root = _find_repo_root(Path(__file__).resolve())
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


_REPO_ROOT = _ensure_repo_root_on_path()


from src import attribution_heuristics as heuristics  # noqa: E402
from src import data_prep  # noqa: E402

plt.style.use("seaborn-v0_8")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_roi_by_channel(channel_metrics: pd.DataFrame, output_path: Path) -> None:
    pivot = channel_metrics.pivot(index="channel", columns="model_name", values="ROI")
    pivot = pivot.sort_values(by=list(pivot.columns), ascending=False)
    pivot.plot(kind="bar", figsize=(10, 6))
    plt.ylabel("ROI")
    plt.title("Channel ROI by heuristic model")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_top_campaigns_roas(campaign_metrics: pd.DataFrame, output_path: Path, top_n: int = 10) -> None:
    baseline = campaign_metrics[campaign_metrics["model_name"] == "linear"]
    top_campaigns = baseline.nlargest(top_n, "attributed_revenue")["campaign_id"]
    top_df = campaign_metrics[campaign_metrics["campaign_id"].isin(top_campaigns)]
    pivot = top_df.pivot_table(
        index="campaign_id",
        columns="model_name",
        values="ROAS",
        fill_value=0,
    )
    pivot = pivot.loc[top_campaigns]
    pivot.plot(kind="bar", figsize=(12, 6))
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("ROAS")
    plt.title(f"Top {top_n} campaigns by linear attributed revenue")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_credit_distribution(attribution_touches: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    models = attribution_touches["model_name"].unique().tolist()
    data = [
        attribution_touches.loc[attribution_touches["model_name"] == model, "credit_weight"].values for model in models
    ]
    plt.boxplot(data, tick_labels=models)
    plt.ylabel("Credit weight per touch")
    plt.title("Distribution of touch-level credit across heuristics")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main(data_dir: str = "data", figures_dir: str = "reports/figures") -> None:
    data_path = _REPO_ROOT / data_dir
    figures_path = _REPO_ROOT / figures_dir
    ensure_dir(figures_path)
    fast_mode = os.getenv("FAST") == "1"

    touchpoints = data_prep.load_touchpoints(data_path / "touchpoints.csv")
    conversions = data_prep.load_conversions(data_path / "conversions.csv")

    journeys, touch_level = heuristics.build_attribution_tables(touchpoints, conversions)
    if touch_level.empty:
        raise RuntimeError("No touches fall within the attribution window; cannot run heuristics.")

    attribution_touches = heuristics.run_all_heuristics(touch_level)

    channel_metrics = heuristics.aggregate_roi_metrics(attribution_touches, level="channel")
    campaign_metrics = heuristics.aggregate_roi_metrics(attribution_touches, level="campaign_id")

    touch_output = attribution_touches[
        [
            "model_name",
            "conversion_id",
            "user_id",
            "channel",
            "campaign_id",
            "timestamp_iso",
            "touch_order",
            "journey_length",
            "time_to_conversion_days",
            "credit_weight",
            "conversion_value",
            "attributed_revenue",
            "cost",
        ]
    ]

    channel_output_path = data_path / "attribution_channel_metrics.csv"
    campaign_output_path = data_path / "attribution_campaign_metrics.csv"
    touch_output_path = data_path / "attribution_touch_level.csv"

    channel_metrics.to_csv(channel_output_path, index=False)
    campaign_metrics.to_csv(campaign_output_path, index=False)
    touch_output.to_csv(touch_output_path, index=False)

    plot_roi_by_channel(channel_metrics, figures_path / "roi_by_channel_models.png")
    plot_top_campaigns_roas(
        campaign_metrics,
        figures_path / "top_campaigns_roas.png",
        top_n=5 if fast_mode else 10,
    )
    if not fast_mode:
        plot_credit_distribution(attribution_touches, figures_path / "attribution_credit_distribution.png")

    mode_label = "FAST" if fast_mode else "FULL"
    print(f"[{mode_label}] Saved {channel_output_path}")
    print(f"[{mode_label}] Saved {campaign_output_path}")
    print(f"[{mode_label}] Saved {touch_output_path}")


if __name__ == "__main__":
    main()
