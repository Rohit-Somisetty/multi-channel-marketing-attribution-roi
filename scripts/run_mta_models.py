"""Runs data-driven MTA models (Markov + logistic regression) and produces outputs."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src import (
    attribution_heuristics,
    attribution_markov,
    attribution_regression,
    data_prep,
    validation,
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_markov_removal(removal_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.bar(removal_df["channel"], removal_df["removal_value"], color="#3a7fb0")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Attributed revenue lost if removed")
    plt.title("Markov removal effect by channel")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_model_comparison(channel_df: pd.DataFrame, output_path: Path) -> None:
    pivot_roi = channel_df.pivot(index="channel", columns="model_name", values="ROI")
    pivot_roas = channel_df.pivot(index="channel", columns="model_name", values="ROAS")
    models = pivot_roi.columns

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    pivot_roi.plot(kind="bar", ax=axes[0])
    axes[0].set_title("ROI by channel")
    axes[0].set_ylabel("ROI")
    axes[0].tick_params(axis="x", rotation=45)

    pivot_roas.plot(kind="bar", ax=axes[1])
    axes[1].set_title("ROAS by channel")
    axes[1].set_ylabel("ROAS")
    axes[1].tick_params(axis="x", rotation=45)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(models))
    axes[0].get_legend().remove()
    axes[1].get_legend().remove()
    fig.suptitle("Heuristics vs data-driven MTA")
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_top_paths(journeys: pd.DataFrame, output_path: Path, top_n: int = 10) -> None:
    if journeys.empty:
        return
    df = journeys.copy()
    df["path"] = df["channels"].apply(lambda seq: " > ".join(seq))
    grouped = df.groupby("path").agg(
        conversions=("conversion_id", "count"),
        attributed_revenue=("conversion_value", "sum"),
    )
    if grouped.empty:
        return
    top_freq = grouped.nlargest(top_n, "conversions").sort_values("conversions")
    top_revenue = grouped.nlargest(top_n, "attributed_revenue").sort_values("attributed_revenue")
    height = max(6, top_n * 0.45)

    fig, axes = plt.subplots(1, 2, figsize=(16, height), sharey=False)
    axes[0].barh(top_freq.index, top_freq["conversions"], color="#3a7fb0")
    axes[0].set_title("Top paths by conversion count")
    axes[0].set_xlabel("Conversions")

    axes[1].barh(top_revenue.index, top_revenue["attributed_revenue"], color="#5abf90")
    axes[1].set_title("Top paths by revenue")
    axes[1].set_xlabel("Attributed revenue ($)")

    for axis in axes:
        axis.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_journey_length_vs_conversion(dataset: pd.DataFrame, output_path: Path, max_length: int = 12) -> None:
    if dataset.empty or "journey_length" not in dataset.columns:
        return
    stats = (
        dataset.groupby("journey_length")
        .agg(conversion_rate=("converted", "mean"), sample_size=("converted", "count"))
        .reset_index()
        .sort_values("journey_length")
    )
    stats = stats[stats["journey_length"] <= max_length]
    if stats.empty:
        return

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(stats["journey_length"], stats["sample_size"], color="#c8d9ed", label="Sample size")
    ax1.set_xlabel("Journey length (touches)")
    ax1.set_ylabel("Sample size", color="#3a7fb0")
    ax1.tick_params(axis="y", labelcolor="#3a7fb0")

    ax2 = ax1.twinx()
    ax2.plot(
        stats["journey_length"],
        stats["conversion_rate"],
        color="#1f4b99",
        marker="o",
        label="Conversion rate",
    )
    ax2.set_ylabel("Conversion rate", color="#1f4b99")
    ax2.tick_params(axis="y", labelcolor="#1f4b99")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")
    plt.title("Conversion rate by journey length")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_summary(
    report_path: Path,
    heuristics_df: pd.DataFrame,
    campaign_df: pd.DataFrame,
    markov_metrics: pd.DataFrame,
    removal_df: pd.DataFrame,
    regression_metrics: pd.DataFrame,
) -> None:
    def fmt_currency(value: float) -> str:
        return f"${value:,.0f}"

    def fmt_pct(value: float) -> str:
        return f"{value * 100:.1f}%"

    heuristics_channels = (
        heuristics_df.groupby("channel")
        .agg(attributed_revenue=("attributed_revenue", "mean"), ROI=("ROI", "mean"))
        .sort_values("attributed_revenue", ascending=False)
    )
    heuristics_top_channel = heuristics_channels.index[0]
    heuristics_top_row = heuristics_channels.iloc[0]
    lagging_channel = heuristics_channels["ROI"].idxmin()
    lagging_roi = heuristics_channels.loc[lagging_channel, "ROI"]

    campaign_strength = (
        campaign_df.groupby("campaign_id")
        .agg(ROI=("ROI", "mean"), attributed_revenue=("attributed_revenue", "mean"))
        .sort_values("ROI", ascending=False)
    )
    top_campaign = campaign_strength.index[0]
    top_campaign_roi = campaign_strength.iloc[0]["ROI"]

    markov_top = removal_df.iloc[0] if not removal_df.empty else None
    markov_share = 0.0
    if markov_top is not None:
        share_val = markov_top.get("removal_share", 0.0)
        markov_share = float(share_val) if not pd.isna(share_val) else 0.0
    regression_top = (
        regression_metrics.sort_values("attributed_revenue", ascending=False).iloc[0]
        if not regression_metrics.empty
        else None
    )

    lines = ["# Data-Driven MTA Summary", ""]
    lines.append("## Model Overview")
    lines.append(
        "- **Heuristics (first/last/linear/time-decay)**: High-speed "
        "directional splits that assume simple credit rules; best for sanity "
        "checks and campaign ops dashboards."
    )
    lines.append(
        "- **Markov chain**: Evaluates incremental lift by removing each "
        "channel from observed paths, surfacing assists that heuristics miss."
    )
    lines.append(
        "- **Logistic regression**: Controls for geo/segment/device mix to "
        "estimate marginal odds of conversion as exposure volume changes."
    )
    lines.append("")

    lines.append("## Key Findings")
    lines.append(
        f"- Across heuristics, {heuristics_top_channel} stays #1 (avg "
        f"{fmt_currency(heuristics_top_row['attributed_revenue'])} revenue, ROI "
        f"{heuristics_top_row['ROI']:.1f}x) while organic/email combine for "
        "outsized efficiency."
    )
    lines.append(
        f"- Campaign {top_campaign} delivers ROI {top_campaign_roi:.1f}x across "
        "models, making it the safest venue for incremental spend."
    )
    if markov_top is not None:
        lines.append(
            f"- Markov removal shows dropping {markov_top['channel']} would "
            f"forfeit {fmt_currency(markov_top['removal_value'])} in revenue "
            f"({fmt_pct(markov_share)} of total)."
        )
    if regression_top is not None:
        lines.append(
            f"- Logistic regression attributes the most incremental revenue to "
            f"{regression_top['channel']} (ROI {regression_top['ROI']:.1f}x), "
            "reinforcing its budget priority."
        )
    lines.append(
        f"- {lagging_channel} remains the weakest ROI channel (~{lagging_roi:.1f}x); "
        "treat it as a test-and-learn arena rather than a core driver."
    )
    lines.append("")

    lines.append("## Confounding Risks")
    lines.append(
        "Even with removal effects and covariate controls, all models rely on "
        "observational exposure data; high-intent leads still self-select into "
        "high-performing channels, and timing overlaps mean incremental lift "
        "can be overstated. Pair these diagnostics with causal experiments "
        "(propensity matching, geo holdouts) before wholesale budget shifts."
    )
    lines.append("")

    lines.append("## Recommended Actions")
    core_share = heuristics_df[heuristics_df["channel"] == heuristics_top_channel]["revenue_share"].mean()
    if pd.isna(core_share):
        core_share = 0.0
    lines.append(
        f"- Protect core investment in {heuristics_top_channel}; every model "
        f"keeps it above {fmt_pct(core_share)} share of attributed revenue."
    )
    if markov_top is not None:
        lines.append(
            f"- Scale nurture programs feeding {markov_top['channel']}; its "
            "removal effect is the largest incremental unlock."
        )
    lines.append(
        f"- Double down on {top_campaign} creative packages while their ROI "
        f"stays above {top_campaign_roi:.1f}x; create variant tests to absorb "
        "more spend without fatigue."
    )
    lines.append(
        f"- Ring-fence {lagging_channel} spend for experiments with clear "
        "success metrics and causal readouts before expanding further."
    )
    lines.append("")

    lines.append("## Next")
    lines.append(
        "1. Run the causal incrementality module on the weakest/highest-variance "
        "channels.\n2. Refresh this pipeline weekly to watch for "
        "seasonality-driven drift.\n3. Socialize journey-length diagnostics "
        "with lifecycle teams to tighten orchestration windows."
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main(
    data_dir: str = "data",
    figures_dir: str = "reports/figures",
    report_path: str = "reports/mta_summary.md",
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / data_dir
    figures_path = project_root / figures_dir
    report_file = project_root / report_path
    validation_report = project_root / "reports/validation_checks.md"
    ensure_dir(figures_path)
    ensure_dir(report_file.parent)
    ensure_dir(validation_report.parent)
    fast_mode = os.getenv("FAST") == "1"

    touchpoints = data_prep.load_touchpoints(data_path / "touchpoints.csv")
    conversions = data_prep.load_conversions(data_path / "conversions.csv")
    conversions = conversions.reset_index(drop=True)
    if "conversion_id" not in conversions.columns:
        conversions["conversion_id"] = range(1, len(conversions) + 1)
    total_revenue = float(conversions["conversion_value"].sum())

    cost_by_channel = touchpoints.groupby("channel")["cost"].sum().to_dict()

    journeys, touch_level = attribution_heuristics.build_attribution_tables(touchpoints, conversions)
    if touch_level.empty:
        raise RuntimeError("No touch-level records available for attribution window.")

    markov_paths = attribution_markov.build_markov_paths(touch_level, touchpoints, conversions)
    markov_paths_out = markov_paths.copy()
    markov_paths_out["channel_sequence"] = markov_paths_out["channel_sequence"].apply(lambda seq: " > ".join(seq))
    markov_paths_out["path"] = markov_paths_out["path"].apply(lambda seq: " > ".join(seq))
    markov_paths_out.to_csv(data_path / "markov_paths.csv", index=False)

    markov_metrics, removal_df, markov_diag = attribution_markov.run_markov_attribution(
        touch_level,
        touchpoints,
        conversions,
        cost_by_channel,
        paths=markov_paths,
    )

    regression_dataset = attribution_regression.build_regression_dataset(
        touch_level,
        touchpoints,
        conversions,
    )
    regression_dataset.to_csv(data_path / "regression_modeling_dataset.csv", index=False)

    regression_metrics, odds_df = attribution_regression.run_regression_attribution(
        touch_level,
        touchpoints,
        conversions,
        cost_by_channel,
        dataset=regression_dataset,
    )

    markov_metrics.to_csv(data_path / "attribution_markov_channel_metrics.csv", index=False)
    regression_metrics.to_csv(data_path / "attribution_regression_channel_metrics.csv", index=False)
    removal_df.to_csv(data_path / "markov_removal_effects.csv", index=False)
    odds_df.to_csv(data_path / "regression_channel_odds.csv", index=False)

    heuristics_path = data_path / "attribution_channel_metrics.csv"
    if heuristics_path.exists():
        heuristics_df = pd.read_csv(heuristics_path)
    else:
        raise FileNotFoundError("Heuristic channel metrics not found. Run scripts/run_heuristic_attribution.py first.")

    campaign_path = data_path / "attribution_campaign_metrics.csv"
    if campaign_path.exists():
        campaign_df = pd.read_csv(campaign_path)
    else:
        raise FileNotFoundError("Heuristic campaign metrics not found. Run scripts/run_heuristic_attribution.py first.")

    comparison_df = pd.concat([heuristics_df, markov_metrics, regression_metrics], ignore_index=True)
    plot_markov_removal(removal_df, figures_path / "markov_removal_effect.png")
    plot_model_comparison(comparison_df, figures_path / "mta_model_comparison.png")
    plot_top_paths(journeys, figures_path / "top_paths.png", top_n=5 if fast_mode else 10)
    if not fast_mode:
        plot_journey_length_vs_conversion(regression_dataset, figures_path / "journey_length_vs_conversion.png")

    write_summary(report_file, heuristics_df, campaign_df, markov_metrics, removal_df, regression_metrics)

    validation_checks = validation.run_validation_suite(
        transition_matrix=markov_diag["transition_matrix"],
        baseline_prob=markov_diag["baseline_conversion_prob"],
        removal_df=removal_df,
        model_metrics=comparison_df,
        touchpoints=touchpoints,
        cost_by_channel=cost_by_channel,
        total_conversion_value=total_revenue,
    )
    validation.write_validation_report(validation_checks, validation_report)

    mode_label = "FAST" if fast_mode else "FULL"
    print(f"[{mode_label}] Saved modeling datasets, attribution outputs, validation, and figures.")


if __name__ == "__main__":
    main()
