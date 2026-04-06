"""
UncertaintyLens v1.0 — Streamlit interactive application.

Upload a CSV or use sample data to get a full uncertainty analysis report
with 10 detectors, attribution breakdown, and actionable recommendations.
"""

from html import escape as html_escape

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from uncertainty_lens.pipeline import UncertaintyPipeline
from uncertainty_lens.detectors import (
    ConformalShiftDetector,
    UncertaintyDecomposer,
    JackknifePlusDetector,
    MMDShiftDetector,
    ZeroInflationDetector,
    DeepEnsembleDetector,
)
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
    page_title="UncertaintyLens v1.0",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========== Custom CSS ==========
st.markdown(
    """
<style>
    .hero-title {
        font-size: 2.8rem; font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0; line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.15rem; color: #6b7280;
        margin-top: 0.25rem; margin-bottom: 1.5rem;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 16px 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    div[data-testid="stMetric"] label {
        color: #64748b; font-size: 0.85rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.03em;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.8rem; font-weight: 700; color: #1e293b;
    }
    .section-header {
        font-size: 1.4rem; font-weight: 700; color: #1e293b;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.4rem; margin-top: 2rem; margin-bottom: 1rem;
    }
    .section-desc {
        color: #64748b; font-size: 0.95rem; margin-bottom: 1rem;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }
    .info-card {
        background: #f8fafc; border-left: 4px solid #667eea;
        border-radius: 0 8px 8px 0; padding: 12px 16px;
        margin-bottom: 1rem; color: #334155; font-size: 0.9rem;
    }
    .action-card {
        background: #fffbeb; border-left: 4px solid #f59e0b;
        border-radius: 0 8px 8px 0; padding: 12px 16px;
        margin-bottom: 0.6rem; color: #78350f; font-size: 0.88rem;
    }
    .action-card.high {
        background: #fef2f2; border-left-color: #ef4444; color: #7f1d1d;
    }
    .action-card.low {
        background: #f0fdf4; border-left-color: #22c55e; color: #14532d;
    }
    .detector-badge {
        display: inline-block; padding: 2px 10px; border-radius: 12px;
        font-size: 0.78rem; font-weight: 600; margin-right: 4px;
        background: #e0e7ff; color: #3730a3;
    }
    .footer {
        text-align: center; color: #94a3b8; font-size: 0.8rem;
        padding: 2rem 0 1rem; border-top: 1px solid #e2e8f0; margin-top: 3rem;
    }
    .footer a { color: #667eea; text-decoration: none; }
</style>
""",
    unsafe_allow_html=True,
)


# ========== Hero ==========
st.markdown('<p class="hero-title">UncertaintyLens</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">'
    "Reveal what your data doesn't know &mdash; and how much that ignorance costs. "
    "<strong>v1.0</strong> &bull; 10 detectors &bull; auto-attribution"
    "</p>",
    unsafe_allow_html=True,
)


# ========== Available extra detectors ==========
EXTRA_DETECTORS = {
    "conformal_shift": {
        "label": "Distribution Shift",
        "desc": "Conformal p-value test for group distribution differences",
        "factory": lambda seed: ConformalShiftDetector(seed=seed),
        "default_weight": 0.1,
        "needs_group": True,
    },
    "decomposition": {
        "label": "Uncertainty Decomposition",
        "desc": "Splits uncertainty into aleatoric (noise) vs epistemic (knowledge gap)",
        "factory": lambda seed: UncertaintyDecomposer(n_bootstrap=100, seed=seed),
        "default_weight": 0.15,
        "needs_group": False,
    },
    "jackknife_plus": {
        "label": "Jackknife+ Prediction Intervals",
        "desc": "Leave-one-out conformal prediction interval width",
        "factory": lambda seed: JackknifePlusDetector(seed=seed),
        "default_weight": 0.1,
        "needs_group": False,
    },
    "mmd_shift": {
        "label": "MMD Distribution Drift",
        "desc": "Multi-dimensional distribution drift with adaptive kernel",
        "factory": lambda seed: MMDShiftDetector(n_permutations=200, seed=seed),
        "default_weight": 0.1,
        "needs_group": True,
    },
    "zero_inflation": {
        "label": "Zero Inflation",
        "desc": "Detects columns with abnormally high zero counts",
        "factory": lambda _: ZeroInflationDetector(),
        "default_weight": 0.2,
        "needs_group": False,
    },
    "deep_ensemble": {
        "label": "Deep Ensemble",
        "desc": "Neural network ensemble disagreement as uncertainty proxy",
        "factory": lambda seed: DeepEnsembleDetector(n_ensemble=3, seed=seed),
        "default_weight": 0.1,
        "needs_group": False,
    },
}

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
                st.error(f"File too large ({file_size_mb:.1f} MB). Max {MAX_FILE_MB} MB.")
                df = None
            else:
                try:
                    df = pd.read_csv(uploaded_file)
                    if df.empty:
                        st.warning("The uploaded CSV is empty.")
                        df = None
                    else:
                        st.success(
                            f"Loaded **{uploaded_file.name}** "
                            f"({df.shape[0]:,} rows, {df.shape[1]} columns)"
                        )
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
                "channel": rng.choice(["Search Ads", "Social Media", "Video", "Feed", "Email"], n),
                "impressions": rng.lognormal(8, 1.5, n).astype(int),
                "clicks": np.where(
                    rng.random(n) > 0.1, rng.lognormal(5, 1.2, n).astype(int), np.nan
                ),
                "conversions": np.where(rng.random(n) > 0.25, rng.poisson(10, n), np.nan),
                "spend": np.concatenate([rng.lognormal(6, 0.8, n - 30), rng.lognormal(9, 0.5, 30)]),
                "attributed_revenue": np.where(
                    rng.random(n) > 0.35, rng.lognormal(7, 1.5, n), np.nan
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
            help="Categorical column for group-level comparison.",
        )
        group_col = None if group_col == "None" else group_col

        st.markdown("---")
        st.markdown("### Core Detector Weights")
        w_missing = st.slider("Missing", 0.0, 1.0, 0.4, 0.05, format="%.2f")
        w_anomaly = st.slider("Anomaly", 0.0, 1.0, 0.3, 0.05, format="%.2f")
        w_variance = st.slider("Variance", 0.0, 1.0, 0.3, 0.05, format="%.2f")

        total_w = w_missing + w_anomaly + w_variance
        if total_w > 0:
            st.caption(
                f"Normalized: missing={w_missing / total_w:.0%}, "
                f"anomaly={w_anomaly / total_w:.0%}, "
                f"variance={w_variance / total_w:.0%}"
            )
        else:
            st.warning("At least one weight must be > 0.")

        st.markdown("---")
        st.markdown("### Extra Detectors")
        st.caption("Enable additional detectors for deeper analysis.")

        enabled_extras = {}
        for key, info in EXTRA_DETECTORS.items():
            if info["needs_group"] and group_col is None:
                continue
            checked = st.checkbox(info["label"], value=False, help=info["desc"], key=f"det_{key}")
            if checked:
                enabled_extras[key] = info


# ========== Helper: build and run pipeline ==========
def _build_and_run(df, w_missing, w_anomaly, w_variance, group_col, enabled_extras):
    pipe = UncertaintyPipeline(
        weights={"missing": w_missing, "anomaly": w_anomaly, "variance": w_variance}
    )
    for key, info in enabled_extras.items():
        pipe.register(key, info["factory"](42), weight=info["default_weight"])
    report = pipe.analyze(df, group_col=group_col)
    # Drop non-serializable items before caching
    if "anomaly_analysis" in report:
        report["anomaly_analysis"].pop("vote_matrix", None)
    return report


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
    extras_key = tuple(sorted(enabled_extras.keys()))

    with st.spinner("Running uncertainty analysis..."):
        report = _build_and_run(df, w_missing, w_anomaly, w_variance, group_col, enabled_extras)

    summary = report["summary"]

    # ===== Section 1: Overview =====
    st.markdown('<p class="section-header">Overview</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Overall Uncertainty", f"{summary['overall_uncertainty']:.1%}")
    col2.metric("Risk Level", summary["overall_level"])
    col3.metric("High-Risk Features", len(summary["high_uncertainty_features"]))
    col4.metric("Reliable Features", len(summary["low_uncertainty_features"]))

    # Active detectors badge row
    active_names = ["Missing", "Anomaly", "Variance"] + [
        EXTRA_DETECTORS[k]["label"] for k in enabled_extras
    ]
    badges_html = " ".join(f'<span class="detector-badge">{n}</span>' for n in active_names)
    st.markdown(
        f'<div class="info-card"><strong>Active detectors ({len(active_names)}):</strong> '
        f"{badges_html}</div>",
        unsafe_allow_html=True,
    )

    # Top-3 summary
    if summary.get("top_3_uncertain"):
        st.markdown(
            '<div class="info-card"><strong>Top uncertain features:</strong> '
            + " &bull; ".join(
                f'{item["feature"]} ({item["composite_score"]:.1%})'
                for item in summary["top_3_uncertain"]
            )
            + "</div>",
            unsafe_allow_html=True,
        )

    # ===== Section 2: Heatmap + Bar =====
    st.markdown('<p class="section-header">Uncertainty Breakdown</p>', unsafe_allow_html=True)

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

    # ===== Section 3: Attribution & Action Plan =====
    explanation = report.get("explanation")
    if explanation:
        st.markdown(
            '<p class="section-header">Attribution Analysis</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="section-desc">'
            "Each feature's uncertainty decomposed by detector contribution."
            "</p>",
            unsafe_allow_html=True,
        )

        # Build attribution bar chart
        feat_expls = explanation.get("feature_explanations", {})
        if feat_expls:
            features = list(feat_expls.keys())
            selected_feat = st.selectbox("Select feature", features, key="attr_feat")

            if selected_feat and selected_feat in feat_expls:
                fe = feat_expls[selected_feat]
                st.markdown(
                    f'<div class="info-card">{html_escape(fe.get("summary", ""))}</div>',
                    unsafe_allow_html=True,
                )

                # Contribution bar
                contribs = fe.get("contributions", {})
                if contribs:
                    det_names = list(contribs.keys())
                    det_vals = [contribs[d].get("contribution", 0) for d in det_names]
                    det_pcts = [contribs[d].get("pct", 0) for d in det_names]

                    fig_attr = go.Figure()
                    fig_attr.add_trace(
                        go.Bar(
                            x=det_vals,
                            y=det_names,
                            orientation="h",
                            marker_color=[
                                (
                                    "#ef4444"
                                    if contribs[d].get("severity") == "high"
                                    else (
                                        "#f59e0b"
                                        if contribs[d].get("severity") == "moderate"
                                        else "#22c55e"
                                    )
                                )
                                for d in det_names
                            ],
                            text=[f"{p:.0%}" for p in det_pcts],
                            textposition="auto",
                        )
                    )
                    fig_attr.update_layout(
                        title=f"Contribution to '{selected_feat}' uncertainty",
                        xaxis_title="Contribution",
                        height=max(250, len(det_names) * 40),
                        margin=dict(l=20, r=20, t=40, b=20),
                    )
                    st.plotly_chart(fig_attr, use_container_width=True)

        # Action plan
        action_plan = explanation.get("action_plan", [])
        if action_plan:
            st.markdown(
                '<p class="section-header">Action Plan</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<p class="section-desc">'
                "Prioritized recommendations based on detected issues."
                "</p>",
                unsafe_allow_html=True,
            )

            for action in action_plan[:8]:
                severity = action.get("severity", "moderate")
                css_class = "high" if severity == "high" else ("low" if severity == "low" else "")
                label = html_escape(action.get("label", ""))
                text = html_escape(action.get("action", ""))
                st.markdown(
                    f'<div class="action-card {css_class}">'
                    f"<strong>[{severity.upper()}] {label}</strong><br>{text}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # Global insights — radar chart
        global_insights = explanation.get("global_insights", {})
        if global_insights:
            st.markdown(
                '<p class="section-header">Global Health Radar</p>',
                unsafe_allow_html=True,
            )
            radar_names = list(global_insights.keys())
            radar_vals = [global_insights[n].get("avg_score", 0) for n in radar_names]

            fig_radar = go.Figure()
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=radar_vals + [radar_vals[0]],
                    theta=radar_names + [radar_names[0]],
                    fill="toself",
                    fillcolor="rgba(102,126,234,0.2)",
                    line=dict(color="#667eea", width=2),
                    name="Current",
                )
            )
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=[0.2] * (len(radar_names) + 1),
                    theta=radar_names + [radar_names[0]],
                    line=dict(color="#22c55e", width=1, dash="dot"),
                    name="Healthy baseline",
                )
            )
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(range=[0, 1])),
                height=420,
                margin=dict(l=60, r=60, t=30, b=30),
                showlegend=True,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    # ===== Section 4: Information Flow =====
    st.markdown(
        '<p class="section-header">Information Loss Flow</p>',
        unsafe_allow_html=True,
    )

    missing_rows = int(df.isnull().any(axis=1).sum())
    consensus = report.get("anomaly_analysis", {}).get("consensus_anomalies", {})
    anomaly_rows = max(consensus.values()) if consensus else 0

    cv_analysis = report.get("variance_analysis", {}).get("cv_analysis", {})
    n_numeric = max(1, len(cv_analysis))
    n_high_var = sum(
        1 for v in cv_analysis.values() if isinstance(v, dict) and v.get("is_high_variance", False)
    )
    high_var_rows = int(df.shape[0] * n_high_var / n_numeric)

    fig_sankey = create_info_loss_sankey(
        total_records=df.shape[0],
        missing_records=missing_rows,
        anomaly_records=anomaly_rows,
        high_variance_records=high_var_rows,
        title="",
    )
    fig_sankey.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=420)
    st.plotly_chart(fig_sankey, use_container_width=True)

    # ===== Section 5: Group Analysis =====
    if group_col:
        st.markdown(
            f'<p class="section-header">Group Analysis: {html_escape(str(group_col))}</p>',
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

    # ===== Section 6: Monte Carlo =====
    st.markdown(
        '<p class="section-header">Monte Carlo Sensitivity</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="section-desc">'
        "Re-impute missing values and add noise 200 times to test how stable the mean is."
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
                    f"**{mc_col}** is highly sensitive. " f"95% CI: [{ci_lo:.2f}, {ci_hi:.2f}]"
                )
            elif mc_result["sensitivity_ratio"] > 0.1:
                st.info(
                    f"**{mc_col}** has moderate sensitivity. " f"95% CI: [{ci_lo:.2f}, {ci_hi:.2f}]"
                )
            else:
                st.success(f"**{mc_col}** is robust. " f"95% CI: [{ci_lo:.2f}, {ci_hi:.2f}]")
        else:
            st.warning(f"Monte Carlo could not run: {mc_result['error']}")

    # ===== Raw JSON =====
    with st.expander("Raw Analysis Data (JSON)", expanded=False):
        st.json(report["summary"])

    # ===== Footer =====
    st.markdown(
        '<div class="footer">'
        "UncertaintyLens v1.0 &bull; 10 detectors &bull; "
        '<a href="https://github.com/1anthanum/UncertaintyLens" target="_blank">GitHub</a>'
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
2. **Configure** group columns and enable extra detectors in the sidebar
3. **Explore** heatmaps, attribution charts, radar diagrams, and Sankey flows
4. **Act** on prioritized recommendations to improve your data quality
""")

    with col_r:
        st.markdown("### 10 Uncertainty Detectors")
        st.markdown("""
| Detector | What it finds |
|----------|--------------|
| **Missing** | Gaps & whether they're random (MCAR/MAR) |
| **Anomaly** | Outliers via IQR + IsoForest + LOF ensemble |
| **Variance** | Unexplained dispersion & instability |
| **Conformal Shift** | Group distribution differences |
| **Decomposition** | Aleatoric vs epistemic uncertainty |
| **Jackknife+** | Prediction interval width |
| **MMD Drift** | Multi-dimensional distribution shift |
| **Zero Inflation** | Excess zero counts |
| **Deep Ensemble** | Neural network disagreement |
| **Streaming** | Online drift detection |
""")

    st.info("Select **Sample Data** or **Upload CSV** in the sidebar to begin.")
