"""Lightweight validation helpers for the MTA pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ValidationCheck:
    """Container for a single validation result."""

    name: str
    passed: bool
    value: str
    threshold: str
    notes: str


def _format_float(value: float | None, precision: int = 4) -> str:
    if value is None or np.isnan(value):
        return "n/a"
    return f"{value:.{precision}f}"


def run_validation_suite(
    transition_matrix: np.ndarray,
    baseline_prob: float,
    removal_df: pd.DataFrame,
    model_metrics: pd.DataFrame,
    touchpoints: pd.DataFrame,
    cost_by_channel: dict[str, float],
    total_conversion_value: float,
    row_tolerance: float = 5e-3,
    revenue_tolerance: float | None = None,
) -> list[ValidationCheck]:
    """Evaluate core attribution assumptions and return structured checks."""

    checks: list[ValidationCheck] = []
    row_sums = transition_matrix.sum(axis=1)
    mask = row_sums > 0
    deviation = float(np.max(np.abs(row_sums[mask] - 1.0))) if np.any(mask) else 0.0
    checks.append(
        ValidationCheck(
            name="Transition row sums",
            passed=deviation <= row_tolerance,
            value=_format_float(deviation, precision=6),
            threshold=f"<= {row_tolerance:.3e}",
            notes="Ensures all Markov transition rows normalize to 1.",
        )
    )

    prob_pass = 0 < baseline_prob <= 1
    checks.append(
        ValidationCheck(
            name="Baseline conversion probability",
            passed=prob_pass,
            value=f"{baseline_prob:.4%}",
            threshold="0 < p <= 1",
            notes="Probability of converting from the Markov chain's START state.",
        )
    )

    noise_tolerance = max(1.0, total_conversion_value * 1e-4)
    if removal_df.empty:
        min_removal = 0.0
    else:
        min_removal = float(removal_df["removal_value"].min())
    removal_pass = min_removal >= -noise_tolerance
    notes = (
        "All removal effects are non-negative."
        if min_removal >= 0
        else f"Small negative ({min_removal:.2f}) tolerated as sampling noise."
    )
    checks.append(
        ValidationCheck(
            name="Removal effects",
            passed=removal_pass,
            value=_format_float(min_removal),
            threshold=f">= -{noise_tolerance:.2f}",
            notes=notes,
        )
    )

    if model_metrics.empty:
        revenue_gap = 0.0
        worst_model_msg = "No model metrics available."
    else:
        model_totals = model_metrics.groupby("model_name")["attributed_revenue"].sum()
        gaps = (model_totals - total_conversion_value).abs()
        revenue_gap = float(gaps.max())
        if gaps.empty:
            worst_model_msg = "No models found."
        else:
            culprit = gaps.idxmax()
            worst_model_msg = f"Largest gap from {culprit}: {_format_float(gaps.max())}."
    revenue_tolerance = revenue_tolerance or max(1.0, total_conversion_value * 1e-3)
    checks.append(
        ValidationCheck(
            name="Revenue reconciliation",
            passed=revenue_gap <= revenue_tolerance,
            value=_format_float(revenue_gap),
            threshold=f"<= {revenue_tolerance:.2f}",
            notes=worst_model_msg,
        )
    )

    total_spend = float(touchpoints["cost"].sum())
    channel_spend = float(sum(cost_by_channel.values()))
    touch_vs_channel_gap = abs(total_spend - channel_spend)
    model_cost_totals = (
        model_metrics.groupby("model_name")["cost"].sum() if not model_metrics.empty else pd.Series(dtype=float)
    )
    data_driven_models = {"markov_chain", "regression_logit"}
    dd_gaps = []
    heur_spread = 0.0
    heur_msg = ""
    heur_models: list[str] = []
    if not model_cost_totals.empty:
        for model in data_driven_models:
            if model in model_cost_totals.index:
                dd_gaps.append(abs(float(model_cost_totals.loc[model]) - total_spend))
        heur_models = [m for m in model_cost_totals.index if m not in data_driven_models]
        if heur_models:
            heur_costs = model_cost_totals.loc[heur_models]
            heur_spread = float(heur_costs.max() - heur_costs.min())
            heur_msg = f"Heuristic model cost spread {_format_float(heur_spread)}."
    dd_gap = max(dd_gaps) if dd_gaps else 0.0
    cost_tolerance = max(1.0, total_spend * 1e-3)
    heur_reference = float(model_cost_totals.loc[heur_models].mean()) if heur_models else 0.0
    heur_tolerance = max(1.0, heur_reference * 1e-3) if heur_reference else 1.0
    passed = (touch_vs_channel_gap <= cost_tolerance) and (dd_gap <= cost_tolerance) and (heur_spread <= heur_tolerance)
    notes = (
        f"Touch vs. channel gap {_format_float(touch_vs_channel_gap)}; data-driven max gap {_format_float(dd_gap)}."
        f" {heur_msg or 'Heuristic models share consistent spend base.'}"
    )
    checks.append(
        ValidationCheck(
            name="Cost consistency",
            passed=passed,
            value=_format_float(max(touch_vs_channel_gap, dd_gap, heur_spread)),
            threshold=f"DD <= {cost_tolerance:.2f}; heur <= {heur_tolerance:.2f}",
            notes=notes,
        )
    )

    return checks


def write_validation_report(checks: list[ValidationCheck], output_path: Path) -> None:
    lines = ["# Validation Checks", ""]
    for idx, check in enumerate(checks, start=1):
        status = "PASS" if check.passed else "FAIL"
        lines.append(f"## {idx}. {check.name}")
        lines.append(f"- Status: **{status}**")
        lines.append(f"- Value: {check.value}")
        lines.append(f"- Threshold: {check.threshold}")
        lines.append(f"- Notes: {check.notes}")
        lines.append("")
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


__all__ = [
    "ValidationCheck",
    "run_validation_suite",
    "write_validation_report",
]
