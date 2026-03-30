"""
UncertaintyLens — Streamlit interactive application.

Upload a CSV or use sample data to get a full uncertainty analysis report.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uncertainty_lens.pipeline import UncertaintyPipeline
from uncertainty_lens.visualizers import (
    create_uncertainty_heatmap,
    create_uncertainty_bar,
    create_confidence_plot,
    create_distribution_comparison,
    create_info_loss_sankey,
)


# ========== Page Config ==========
st.set_page_config(
    page_title="UncertaintyLens",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 UncertaintyLens")
st.markdown("**Reveal what your data doesn't know — and how much that ignorance costs.**")
st.markdown("---")


# ========== Sidebar: Data Input & Config ==========
with st.sidebar:
    st.header("📁 Data Input")

    data_source = st.radio(
        "Choose data source",
        ["Upload CSV", "Use sample data"],
    )

    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            df = None
    else:
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            "channel": np.random.choice(
                ["Search Ads", "Social Media", "Video", "Feed", "Email"],
                n,
            ),
            "impressions": np.random.lognormal(8, 1.5, n).astype(int),
            "clicks": np.where(
                np.random.random(n) > 0.1,
                np.random.lognormal(5, 1.2, n).astype(int),
                np.nan,
            ),
            "conversions": np.where(
                np.random.random(n) > 0.25,
                np.random.poisson(10, n),
                np.nan,
            ),
            "spend": np.concatenate([
                np.random.lognormal(6, 0.8, n - 30),
                np.random.lognormal(9, 0.5, 30),
            ]),
            "attributed_revenue": np.where(
                np.random.random(n) > 0.35,
                np.random.lognormal(7, 1.5, n),
                np.nan,
            ),
        })
        st.success("Loaded sample advertising data (1,000 records)")

    if df is not None:
        st.markdown("---")
        st.header("⚙️ Configuration")

        string_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        group_col = st.selectbox(
            "Group column (e.g. channel, category)",
            ["None"] + string_cols,
        )
        group_col = None if group_col == "None" else group_col

        st.markdown("---")
        st.header("🎛️ Weights")
        w_missing = st.slider("Missing weight", 0.0, 1.0, 0.4, 0.05)
        w_anomaly = st.slider("Anomaly weight", 0.0, 1.0, 0.3, 0.05)
        w_variance = st.slider("Variance weight", 0.0, 1.0, 0.3, 0.05)

        total_w = w_missing + w_anomaly + w_variance
        if total_w > 0:
            w_missing /= total_w
            w_anomaly /= total_w
            w_variance /= total_w


# ========== Main Area: Analysis Results ==========
if df is not None:
    with st.expander("📊 Data Preview", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", f"{df.shape[0]:,}")
        col2.metric("Total Columns", f"{df.shape[1]}")
        col3.metric("Missing Values", f"{df.isna().sum().sum():,}")

    with st.spinner("Analyzing data uncertainty..."):
        pipeline = UncertaintyPipeline(
            weights={
                "missing": w_missing,
                "anomaly": w_anomaly,
                "variance": w_variance,
            }
        )
        report = pipeline.analyze(df, group_col=group_col)

    # ===== Summary Cards =====
    st.markdown("## 📋 Analysis Summary")
    summary = report["summary"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Overall Uncertainty",
        f"{summary['overall_uncertainty']:.1%}",
        delta=summary["overall_level"],
    )
    col2.metric("Features Analyzed", summary["total_features_analyzed"])
    col3.metric("High Uncertainty", len(summary["high_uncertainty_features"]))
    col4.metric("Low Uncertainty", len(summary["low_uncertainty_features"]))

    # ===== Heatmap =====
    st.markdown("## 🌡️ Uncertainty Heatmap")
    st.markdown("Redder = higher uncertainty. Red zones are where your data needs the most attention.")

    fig_heatmap = create_uncertainty_heatmap(report["uncertainty_index"])
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # ===== Stacked Bar =====
    st.markdown("## 📊 Uncertainty Composition")
    st.markdown("What drives each feature's uncertainty? Blue = missing, Yellow = anomaly, Red = variance.")

    fig_bar = create_uncertainty_bar(report["uncertainty_index"])
    st.plotly_chart(fig_bar, use_container_width=True)

    # ===== Sankey =====
    st.markdown("## 🔀 Information Loss Flow")
    st.markdown("From raw data to reliable data — how much information is lost?")

    missing_rows = int(df.isnull().any(axis=1).sum())
    anomaly_count = sum(
        report["anomaly_analysis"].get("consensus_anomalies", {}).values()
    )
    high_var_count = sum(
        1
        for v in report["variance_analysis"].get("cv_analysis", {}).values()
        if isinstance(v, dict) and v.get("is_high_variance", False)
    ) * (df.shape[0] // 5)

    fig_sankey = create_info_loss_sankey(
        total_records=df.shape[0],
        missing_records=missing_rows,
        anomaly_records=min(anomaly_count, df.shape[0] // 3),
        high_variance_records=min(high_var_count, df.shape[0] // 4),
    )
    st.plotly_chart(fig_sankey, use_container_width=True)

    # ===== Group Analysis =====
    if group_col:
        st.markdown(f"## 📈 Analysis by {group_col}")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_col = st.selectbox("Select numeric feature to analyze", numeric_cols)

        if selected_col:
            tab1, tab2 = st.tabs(["Confidence Intervals", "Distribution Comparison"])

            with tab1:
                fig_ci = create_confidence_plot(df, selected_col, group_col)
                st.plotly_chart(fig_ci, use_container_width=True)
                st.markdown("*Wider error bars = higher uncertainty for that group*")

            with tab2:
                fig_violin = create_distribution_comparison(
                    df, selected_col, group_col
                )
                st.plotly_chart(fig_violin, use_container_width=True)
                st.markdown("*Wider/more irregular distributions = higher uncertainty*")

    # ===== Detailed Report =====
    with st.expander("🔬 View Detailed Analysis Data", expanded=False):
        st.json(report["summary"])

else:
    st.info("👈 Upload data or select sample data in the sidebar to begin analysis")
