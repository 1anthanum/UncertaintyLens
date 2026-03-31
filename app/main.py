"""
UncertaintyLens — Streamlit interactive application.

Upload a CSV or use sample data to get a full uncertainty analysis report.
"""

from html import escape as html_escape

import streamlit as st
import pandas as pd
import numpy as np

from uncertainty_lens.pipeline import UncertaintyPipeline
from uncertainty_lens.quantifiers import MonteCarloQuantifier
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
    initial_sidebar_state="expanded",
)

# ========== Custom CSS ==========
st.markdown(
    """
<style>
    /* Hero section */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        color: #6b7280;
        margin-top: 0.25rem;
        margin-bottom: 1.5rem;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
    }
    div[data-testid="stMetric"] label {
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
    }

    /* Section headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1e293b;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.4rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .section-desc {
        color: #64748b;
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }
    section[data-testid="stSidebar"] .stRadio > label {
        font-weight: 600;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 600;
    }

    /* Info cards */
    .info-card {
        background: #f8fafc;
        border-left: 4px solid #667eea;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin-bottom: 1rem;
        color: #334155;
        font-size: 0.9rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.8rem;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid #e2e8f0;
        margin-top: 3rem;
    }
    .footer a { color: #667eea; text-decoration: none; }
</style>
""",
    unsafe_allow_html=True,
)


# ========== Hero Section ==========
st.markdown('<p class="hero-title">UncertaintyLens</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">'
    "Reveal what your data doesn't know &mdash; and how much that ignorance costs."
    "</p>",
    unsafe_allow_html=True,
)


# ========== Sidebar ==========
with st.sidebar:
    st.markdown("### Data Source")

    data_source = st.radio(
        "Choose data source",
        ["Sample Data", "Upload CSV"],
        label_visibility="collapsed",
    )

    if data_source == "Upload CSV":
        MAX_FILE_MB = 50
        uploaded_file = st.file_uploader(
            f"Upload your CSV file (max {MAX_FILE_MB} MB)", type=["csv"]
        )
        if uploaded_file:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > MAX_FILE_MB:
                st.error(f"File too large ({file_size_mb:.1f} MB). Maximum is {MAX_FILE_MB} MB.")
                df = None
            else:
                try:
                    df = pd.read_csv(uploaded_file)
                    if df.empty:
                        st.warning("The uploaded CSV is empty.")
                        df = None
                    else:
                        st.success(f"Loaded **{uploaded_file.name}** ({df.shape[0]:,} rows, {df.shape[1]} columns)")
                except Exception as e:
                    st.error(f"Failed to parse CSV: {e}")
                    df = None
        else:
            df = None
    else:
        rng = np.random.default_rng(42)
        n = 1000
        df = pd.DataFrame(
            {
                "channel": rng.choice(
                    ["Search Ads", "Social Media", "Video", "Feed", "Email"],
                    n,
                ),
                "impressions": rng.lognormal(8, 1.5, n).astype(int),
                "clicks": np.where(
                    rng.random(n) > 0.1,
                    rng.lognormal(5, 1.2, n).astype(int),
                    np.nan,
                ),
                "conversions": np.where(
                    rng.random(n) > 0.25,
                    rng.poisson(10, n),
                    np.nan,
                ),
                "spend": np.concatenate(
                    [
                        rng.lognormal(6, 0.8, n - 30),
                        rng.lognormal(9, 0.5, 30),
                    ]
                ),
                "attributed_revenue": np.where(
                    rng.random(n) > 0.35,
                    rng.lognormal(7, 1.5, n),
                    np.nan,
                ),
            }
        )
        st.success("Loaded sample ad data (1,000 rows)")

    if df is not None:
        st.markdown("---")
        st.markdown("### Configuration")

        string_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        group_col = st.selectbox(
            "Group column",
            ["None"] + string_cols,
            help="Select a categorical column to enable group-level analysis.",
        )
        group_col = None if group_col == "None" else group_col

        st.markdown("---")
        st.markdown("### Detector Weights")

        w_missing = st.slider("Missing", 0.0, 1.0, 0.4, 0.05, format="%.2f")
        w_anomaly = st.slider("Anomaly", 0.0, 1.0, 0.3, 0.05, format="%.2f")
        w_variance = st.slider("Variance", 0.0, 1.0, 0.3, 0.05, format="%.2f")

        # Display normalized weights (actual normalization happens inside pipeline)
        total_w = w_missing + w_anomaly + w_variance
        if total_w > 0:
            st.caption(
                f"Normalized: missing={w_missing / total_w:.0%}, "
                f"anomaly={w_anomaly / total_w:.0%}, "
                f"variance={w_variance / total_w:.0%}"
            )
        else:
            st.warning("At least one weight must be greater than zero.")


# ========== Main Content ==========
if df is not None:
    # ----- Data Preview -----
    with st.expander("Data Preview", expanded=False):
        st.dataframe(df.head(20), use_container_width=True, height=300)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", f"{df.shape[1]}")
        c3.metric("Missing Cells", f"{df.isna().sum().sum():,}")
        missing_pct = df.isna().sum().sum() / (df.shape[0] * df.shape[1])
        c4.metric("Missing Rate", f"{missing_pct:.1%}")

    # ----- Run Analysis -----
    @st.cache_data(show_spinner="Running uncertainty analysis...")
    def _run_analysis(_df, _w_missing, _w_anomaly, _w_variance, _group_col):
        pipe = UncertaintyPipeline(
            weights={"missing": _w_missing, "anomaly": _w_anomaly, "variance": _w_variance}
        )
        result = pipe.analyze(_df, group_col=_group_col)
        # Drop non-serializable vote_matrix before caching
        result["anomaly_analysis"].pop("vote_matrix", None)
        return result

    report = _run_analysis(df, w_missing, w_anomaly, w_variance, group_col)

    summary = report["summary"]

    # ===== Section 1: Summary =====
    st.markdown('<p class="section-header">Overview</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Overall Uncertainty", f"{summary['overall_uncertainty']:.1%}")
    col2.metric("Risk Level", summary["overall_level"])
    col3.metric("High-Risk Features", len(summary["high_uncertainty_features"]))
    col4.metric("Reliable Features", len(summary["low_uncertainty_features"]))

    # Top-3 summary
    if summary.get("top_3_uncertain"):
        st.markdown(
            '<div class="info-card">'
            "<strong>Top uncertain features:</strong> "
            + " &bull; ".join(
                f'{item["feature"]} ({item["composite_score"]:.1%})'
                for item in summary["top_3_uncertain"]
            )
            + "</div>",
            unsafe_allow_html=True,
        )

    # ===== Section 2: Heatmap + Bar side by side =====
    st.markdown('<p class="section-header">Uncertainty Breakdown</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-desc">'
        "Left: heatmap across dimensions. Right: stacked composition per feature."
        "</p>",
        unsafe_allow_html=True,
    )

    tab_heat, tab_bar = st.tabs(["Heatmap", "Composition"])

    with tab_heat:
        fig_heatmap = create_uncertainty_heatmap(report["uncertainty_index"], title="")
        fig_heatmap.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            height=max(350, len(report["uncertainty_index"]) * 55),
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    with tab_bar:
        fig_bar = create_uncertainty_bar(report["uncertainty_index"], title="")
        fig_bar.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=420)
        st.plotly_chart(fig_bar, use_container_width=True)

    # ===== Section 3: Information Flow =====
    st.markdown(
        '<p class="section-header">Information Loss Flow</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="section-desc">'
        "How much data survives from raw input to reliable, decision-ready records?"
        "</p>",
        unsafe_allow_html=True,
    )

    missing_rows = int(df.isnull().any(axis=1).sum())

    # Estimate anomaly rows from consensus counts (max across features as upper bound)
    consensus = report["anomaly_analysis"].get("consensus_anomalies", {})
    anomaly_rows = max(consensus.values()) if consensus else 0

    # Count high-variance features, estimate affected rows proportionally
    cv_analysis = report["variance_analysis"].get("cv_analysis", {})
    n_numeric = max(1, len(cv_analysis))
    n_high_var_features = sum(
        1
        for v in cv_analysis.values()
        if isinstance(v, dict) and v.get("is_high_variance", False)
    )
    high_var_rows = int(df.shape[0] * n_high_var_features / n_numeric)

    fig_sankey = create_info_loss_sankey(
        total_records=df.shape[0],
        missing_records=missing_rows,
        anomaly_records=anomaly_rows,
        high_variance_records=high_var_rows,
        title="",
    )
    fig_sankey.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=420)
    st.plotly_chart(fig_sankey, use_container_width=True)

    # ===== Section 4: Group Analysis =====
    if group_col:
        st.markdown(
            f'<p class="section-header">Group Analysis: {html_escape(str(group_col))}</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="section-desc">'
            "Compare uncertainty across groups. Wider intervals or distributions indicate higher uncertainty."
            "</p>",
            unsafe_allow_html=True,
        )

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_col = st.selectbox("Select feature", numeric_cols, key="group_feature")

        if selected_col:
            tab_ci, tab_violin = st.tabs(["Confidence Intervals", "Distributions"])

            with tab_ci:
                fig_ci = create_confidence_plot(df, selected_col, group_col)
                fig_ci.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=420)
                st.plotly_chart(fig_ci, use_container_width=True)

            with tab_violin:
                fig_violin = create_distribution_comparison(df, selected_col, group_col)
                fig_violin.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=420)
                st.plotly_chart(fig_violin, use_container_width=True)

    # ===== Section 5: Monte Carlo =====
    st.markdown(
        '<p class="section-header">Monte Carlo Sensitivity</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="section-desc">'
        "If we re-impute missing values and add small noise 200 times, "
        "how much does the mean shift? High sensitivity = less trustworthy statistic."
        "</p>",
        unsafe_allow_html=True,
    )

    numeric_cols_mc = df.select_dtypes(include=[np.number]).columns.tolist()
    mc_col = st.selectbox("Select feature", numeric_cols_mc, key="mc_col")

    if mc_col:
        with st.spinner("Running 200 Monte Carlo trials..."):
            quantifier = MonteCarloQuantifier(n_simulations=200)
            mc_result = quantifier.estimate(df, statistic_fn=lambda d: d[mc_col].mean())

        if "error" not in mc_result:
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Point Estimate", f"{mc_result['point_estimate']:.2f}")
            ci_width = (
                mc_result["confidence_interval_95"][1] - mc_result["confidence_interval_95"][0]
            )
            mc2.metric("95% CI Width", f"{ci_width:.2f}")
            mc3.metric("Sensitivity", f"{mc_result['sensitivity_ratio']:.2%}")

            ci_lo = mc_result["confidence_interval_95"][0]
            ci_hi = mc_result["confidence_interval_95"][1]

            if mc_result["sensitivity_ratio"] > 0.5:
                st.warning(
                    f"The mean of **{mc_col}** is highly sensitive to data uncertainty. "
                    f"95% CI: [{ci_lo:.2f}, {ci_hi:.2f}]"
                )
            elif mc_result["sensitivity_ratio"] > 0.1:
                st.info(
                    f"The mean of **{mc_col}** has moderate sensitivity. "
                    f"95% CI: [{ci_lo:.2f}, {ci_hi:.2f}]"
                )
            else:
                st.success(
                    f"The mean of **{mc_col}** is robust to data uncertainty. "
                    f"95% CI: [{ci_lo:.2f}, {ci_hi:.2f}]"
                )
        else:
            st.warning(f"Monte Carlo analysis could not run: {mc_result['error']}")

    # ===== Detailed Report =====
    with st.expander("Raw Analysis Data (JSON)", expanded=False):
        st.json(report["summary"])

    # ===== Footer =====
    st.markdown(
        '<div class="footer">'
        "Built with UncertaintyLens &bull; "
        '<a href="https://github.com/xuyangchen/UncertaintyLens" target="_blank">GitHub</a>'
        "</div>",
        unsafe_allow_html=True,
    )

else:
    # ===== Landing Page =====
    st.markdown("---")

    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("### How it works")
        st.markdown("""
1. **Upload** your CSV or use the built-in sample data
2. **Configure** group columns and detector weights in the sidebar
3. **Explore** interactive heatmaps, Sankey diagrams, and confidence intervals
4. **Quantify** how sensitive your statistics are with Monte Carlo simulation
""")

    with col_r:
        st.markdown("### Three uncertainty dimensions")
        st.markdown("""
| Detector | What it finds |
|----------|--------------|
| **Missing** | Gaps & whether they're random |
| **Anomaly** | Outliers via IQR + IsoForest + LOF |
| **Variance** | Unexplained dispersion hotspots |
""")

    st.info("Select **Sample Data** or **Upload CSV** in the sidebar to begin.")
