from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"

st.set_page_config(page_title="Marketing ROI Control Tower", layout="wide")
st.title("🧭 Marketing Attribution & Incrementality Control Tower")


@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


heuristic_df = load_csv("attribution_channel_metrics.csv")
markov_df = load_csv("attribution_markov_channel_metrics.csv")
regression_df = load_csv("attribution_regression_channel_metrics.csv")
removal_df = load_csv("markov_removal_effects.csv")
roi_summary = load_csv("incremental_roi_summary.csv")
naive_incrementality = load_csv("naive_incrementality_metrics.csv")

plots = {
    "heuristic_roi": FIG_DIR / "roi_by_channel_models.png",
    "mta_comparison": FIG_DIR / "mta_model_comparison.png",
    "markov_removal": FIG_DIR / "markov_removal_effect.png",
    "propensity_overlap": FIG_DIR / "propensity_overlap.png",
    "balance_plot": FIG_DIR / "psm_balance_plot.png",
}


def show_image(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_column_width=True)
    else:
        st.info(f"{caption} pending. Run `python scripts/run_all.py` to regenerate.")


tab_heuristic, tab_mta, tab_incrementality, tab_budget = st.tabs(
    [
        "Heuristic Attribution ROI",
        "Data-driven MTA",
        "Incrementality",
        "Budget Allocation",
    ]
)

with tab_heuristic:
    st.subheader("Channel ROI across heuristic models")
    if heuristic_df.empty:
        st.warning("No heuristic outputs found. Run `python scripts/run_heuristic_attribution.py`.")
    else:
        latest = heuristic_df.pivot_table(index="channel", columns="model_name", values="ROI")
        st.dataframe(latest.style.format("{:.2f}"))
    show_image(plots["heuristic_roi"], "ROI by channel across heuristic models")

with tab_mta:
    st.subheader("Markov and regression insights")
    if markov_df.empty or regression_df.empty:
        st.warning("Run `python scripts/run_mta_models.py` to populate MTA artifacts.")
    else:
        col1, col2 = st.columns(2)
        col1.metric("Markov total revenue", f"${markov_df['attributed_revenue'].sum():,.0f}")
        col2.metric("Regression total revenue", f"${regression_df['attributed_revenue'].sum():,.0f}")
        st.markdown("**Markov removal effects**")
        st.dataframe(
            removal_df.sort_values("removal_value", ascending=False)
            .head(10)
            .style.format(
                {
                    "removal_value": "${:,.0f}",
                    "removal_share": "{:.1%}",
                }
            )
        )
    show_image(plots["mta_comparison"], "Model comparison: heuristics vs. MTA")
    show_image(plots["markov_removal"], "Markov removal effect by channel")

with tab_incrementality:
    st.subheader("Naive vs. causal ROI")
    if roi_summary.empty or naive_incrementality.empty:
        st.warning("Run `python scripts/run_incrementality.py` to refresh causal outputs.")
    else:
        merged = roi_summary.merge(
            naive_incrementality.rename(
                columns={
                    "treatment": "channel",
                    "incremental_ROI": "naive_incremental_ROI",
                    "incremental_ROAS": "naive_incremental_ROAS",
                }
            ),
            on="channel",
            how="left",
        )
        st.dataframe(
            merged[
                [
                    "channel",
                    "method",
                    "incremental_ROI",
                    "incremental_ROAS",
                    "naive_incremental_ROI",
                    "naive_incremental_ROAS",
                ]
            ]
            .sort_values(["channel", "method"])
            .style.format("{:.2f}")
        )
    col1, col2 = st.columns(2)
    with col1:
        show_image(plots["propensity_overlap"], "Propensity overlap diagnostics")
    with col2:
        show_image(plots["balance_plot"], "Post-matching balance")

with tab_budget:
    st.subheader("Budget allocation recommender")
    if roi_summary.empty:
        st.info("Causal ROI not available yet. Run the incrementality pipeline.")
    else:
        best_rows = (
            roi_summary.groupby("channel", group_keys=False)
            .apply(lambda df: df.sort_values("incremental_ROI", ascending=False).iloc[0])
            .reset_index(drop=True)
        )
        st.write("Prioritize channels with positive incremental ROI and strong ROAS.")
        st.dataframe(
            best_rows[["channel", "method", "incremental_ROI", "incremental_ROAS", "incremental_revenue"]]
            .sort_values("incremental_ROI", ascending=False)
            .style.format(
                {
                    "incremental_ROI": "{:.2f}",
                    "incremental_ROAS": "{:.2f}",
                    "incremental_revenue": "${:,.0f}",
                }
            )
        )
        st.caption("Tip: layer incremental ROI with Markov removal share to balance assists vs. net lift.")
