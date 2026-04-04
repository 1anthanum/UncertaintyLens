"""
Integrated HTML decision report generator.

Produces a single self-contained HTML file that combines all uncertainty
analyses into an interactive decision dashboard.  No external dependencies
required to view the report — Plotly JS is loaded from CDN.

Usage::

    from uncertainty_lens.visualizers.report import generate_decision_report

    pipeline = UncertaintyPipeline()
    pipeline.register("decomposition", UncertaintyDecomposer(), weight=0.15)
    pipeline.register("conformal_pred", ConformalPredictor(), weight=0.15)
    report = pipeline.analyze(df, group_col="region")

    generate_decision_report(report, df, output_path="uncertainty_report.html")
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from uncertainty_lens.visualizers.heatmap import (
    create_uncertainty_heatmap,
    create_uncertainty_bar,
)
from uncertainty_lens.visualizers.sankey import create_info_loss_sankey
from uncertainty_lens.visualizers.decision import (
    create_decomposition_scatter,
    create_action_priority_chart,
    create_decision_table,
    create_conformal_intervals,
    create_shift_overview,
)


def _fig_to_html_div(fig: go.Figure, div_id: str = "") -> str:
    """Convert a Plotly figure to an HTML div string (no full page wrapper)."""
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=False,
        div_id=div_id,
    )


def _build_summary_cards(summary: Dict[str, Any]) -> str:
    """Build HTML for summary metric cards."""
    overall = summary.get("overall_uncertainty", 0)
    level = summary.get("overall_level", "—")
    n_features = summary.get("total_features_analyzed", 0)
    high_risk = summary.get("high_uncertainty_features", [])
    low_risk = summary.get("low_uncertainty_features", [])

    level_color = {
        "Low": "#27ae60",
        "Medium-Low": "#2ecc71",
        "Medium": "#f39c12",
        "Medium-High": "#e67e22",
        "High": "#e74c3c",
    }.get(level, "#95a5a6")

    cards = f"""
    <div class="cards-row">
        <div class="card">
            <div class="card-value" style="color: {level_color}">{overall:.2f}</div>
            <div class="card-label">Overall Uncertainty</div>
            <div class="card-sublabel">{level}</div>
        </div>
        <div class="card">
            <div class="card-value">{n_features}</div>
            <div class="card-label">Features Analyzed</div>
        </div>
        <div class="card">
            <div class="card-value" style="color: #e74c3c">{len(high_risk)}</div>
            <div class="card-label">High-Risk Features</div>
            <div class="card-sublabel">{', '.join(high_risk[:3]) or 'None'}</div>
        </div>
        <div class="card">
            <div class="card-value" style="color: #27ae60">{len(low_risk)}</div>
            <div class="card-label">Reliable Features</div>
            <div class="card-sublabel">{', '.join(low_risk[:3]) or 'None'}</div>
        </div>
    </div>
    """
    return cards


def _build_recommendations_html(
    recommendations: Dict[str, Dict[str, str]],
) -> str:
    """Build HTML for actionable recommendations section."""
    if not recommendations:
        return "<p>No decomposition data available. Register an UncertaintyDecomposer to get recommendations.</p>"

    action_icons = {
        "collect_more_data": ("📊", "#e74c3c", "Collect More Data"),
        "improve_measurement": ("🔧", "#3498db", "Improve Measurement"),
        "both": ("📊🔧", "#9b59b6", "Both Actions Needed"),
        "none": ("✅", "#27ae60", "No Action Needed"),
    }

    html_parts = []
    for feat, rec in recommendations.items():
        action = rec.get("action", "none")
        explanation = rec.get("explanation", "")
        icon, color, label = action_icons.get(action, ("❓", "#95a5a6", action))

        html_parts.append(f"""
        <div class="rec-item" style="border-left: 4px solid {color}">
            <div class="rec-header">
                <span class="rec-icon">{icon}</span>
                <span class="rec-feature">{feat}</span>
                <span class="rec-action" style="color: {color}">{label}</span>
            </div>
            <div class="rec-explanation">{explanation}</div>
        </div>
        """)

    return "\n".join(html_parts)


def generate_decision_report(
    report: Dict[str, Any],
    df: Optional[pd.DataFrame] = None,
    output_path: str = "uncertainty_decision_report.html",
    title: str = "Uncertainty Decision Report",
) -> str:
    """
    Generate a self-contained HTML decision report.

    Parameters
    ----------
    report : dict
        Full pipeline report from ``UncertaintyPipeline.analyze()``.
    df : pd.DataFrame, optional
        Original DataFrame (used for data summary stats).
    output_path : str
        Path to write the HTML file.
    title : str
        Report title.

    Returns
    -------
    str
        Absolute path to the generated report file.
    """
    uncertainty_index = report.get("uncertainty_index", {})
    summary = report.get("summary", {})

    # Extract optional analysis results
    decomp_analysis = report.get("decomposition_analysis", {})
    decomposition = decomp_analysis.get("decomposition", {})
    recommendations = decomp_analysis.get("recommendation", {})

    conformal_analysis = report.get("conformal_pred_analysis", {})
    conformal_results = conformal_analysis.get("conformal_results", {})

    shift_analysis = report.get("conformal_shift_analysis", {})

    missing_analysis = report.get("missing_analysis", {})
    anomaly_analysis = report.get("anomaly_analysis", {})

    # ── Build chart HTML divs ──────────────────────────────────────────

    charts_html = {}

    # 1. Action priority chart
    fig_priority = create_action_priority_chart(
        uncertainty_index,
        recommendations,
        title="Feature Risk Priority",
    )
    charts_html["priority"] = _fig_to_html_div(fig_priority, "chart-priority")

    # 2. Heatmap
    fig_heatmap = create_uncertainty_heatmap(
        uncertainty_index,
        title="Uncertainty Heatmap",
    )
    charts_html["heatmap"] = _fig_to_html_div(fig_heatmap, "chart-heatmap")

    # 3. Stacked bar
    fig_bar = create_uncertainty_bar(
        uncertainty_index,
        title="Uncertainty Composition",
    )
    charts_html["bar"] = _fig_to_html_div(fig_bar, "chart-bar")

    # 4. Sankey (information loss)
    missing_summary = missing_analysis.get("summary", {})
    anomaly_counts = anomaly_analysis.get("consensus_anomalies", {})
    variance_cv = report.get("variance_analysis", {}).get("cv_analysis", {})

    total_records = (
        missing_summary.get("total_rows", 0)
        if missing_summary
        else (len(df) if df is not None else 0)
    )
    missing_records = (
        missing_summary.get("total_rows", 0) - missing_summary.get("complete_rows", 0)
        if missing_summary
        else 0
    )
    anomaly_records = (
        sum(anomaly_counts.values()) // max(len(anomaly_counts), 1) if anomaly_counts else 0
    )
    high_var_count = (
        sum(1 for v in variance_cv.values() if v.get("is_high_variance", False))
        if variance_cv
        else 0
    )
    # Scale high_var to approximate record count
    high_var_records = int(total_records * 0.05 * high_var_count) if total_records > 0 else 0

    if total_records > 0:
        fig_sankey = create_info_loss_sankey(
            total_records=total_records,
            missing_records=missing_records,
            anomaly_records=anomaly_records,
            high_variance_records=high_var_records,
            title="Data Quality Flow",
        )
        charts_html["sankey"] = _fig_to_html_div(fig_sankey, "chart-sankey")
    else:
        charts_html["sankey"] = "<p>No data summary available for Sankey diagram.</p>"

    # 5. Decomposition scatter (if available)
    if decomposition:
        fig_decomp = create_decomposition_scatter(
            decomposition,
            title="Epistemic vs. Aleatoric: What Can You Control?",
        )
        charts_html["decomp"] = _fig_to_html_div(fig_decomp, "chart-decomp")
    else:
        charts_html["decomp"] = ""

    # 6. Decision table
    fig_table = create_decision_table(
        uncertainty_index,
        recommendations,
        decomposition,
        title="Decision Summary",
    )
    charts_html["table"] = _fig_to_html_div(fig_table, "chart-table")

    # 7. Conformal intervals (if available)
    if conformal_results:
        fig_conformal = create_conformal_intervals(
            conformal_results,
            title="Prediction Interval Widths",
        )
        charts_html["conformal"] = _fig_to_html_div(fig_conformal, "chart-conformal")
    else:
        charts_html["conformal"] = ""

    # 8. Shift overview (if available)
    if shift_analysis and shift_analysis.get("group_shift"):
        fig_shift = create_shift_overview(
            shift_analysis,
            title="Distribution Shift Detection",
        )
        charts_html["shift"] = _fig_to_html_div(fig_shift, "chart-shift")
    else:
        charts_html["shift"] = ""

    # ── Build recommendations HTML ─────────────────────────────────────
    rec_html = _build_recommendations_html(recommendations)

    # ── Build summary cards ────────────────────────────────────────────
    cards_html = _build_summary_cards(summary)

    # ── Data overview ──────────────────────────────────────────────────
    data_overview = ""
    if df is not None:
        n_rows, n_cols = df.shape
        n_numeric = df.select_dtypes(include=["number"]).shape[1]
        n_missing = int(df.isna().sum().sum())
        data_overview = f"""
        <div class="data-overview">
            <span><b>Rows:</b> {n_rows:,}</span>
            <span><b>Columns:</b> {n_cols}</span>
            <span><b>Numeric:</b> {n_numeric}</span>
            <span><b>Missing cells:</b> {n_missing:,}</span>
        </div>
        """

    # ── Build optional sections ────────────────────────────────────────
    decomp_section = ""
    if charts_html["decomp"]:
        decomp_section = f"""
        <section class="report-section">
            <h2>Uncertainty Decomposition</h2>
            <p class="section-desc">
                Each feature's uncertainty is decomposed into <b>epistemic</b>
                (reducible with more data) and <b>aleatoric</b> (irreducible noise).
                Features in the top-left quadrant benefit most from additional data collection.
            </p>
            {charts_html["decomp"]}
        </section>
        """

    conformal_section = ""
    if charts_html["conformal"]:
        conformal_section = f"""
        <section class="report-section">
            <h2>Prediction Intervals</h2>
            <p class="section-desc">
                Conformal prediction intervals provide distribution-free coverage
                guarantees. Wider intervals indicate higher predictive uncertainty.
            </p>
            {charts_html["conformal"]}
        </section>
        """

    shift_section = ""
    if charts_html["shift"]:
        shift_section = f"""
        <section class="report-section">
            <h2>Distribution Shift</h2>
            <p class="section-desc">
                Detects whether subgroups in the data follow different distributions.
                Red cells indicate statistically significant shifts that may require
                group-specific handling.
            </p>
            {charts_html["shift"]}
        </section>
        """

    rec_section = ""
    if recommendations:
        rec_section = f"""
        <section class="report-section">
            <h2>Actionable Recommendations</h2>
            <p class="section-desc">
                Based on the decomposition analysis, each feature receives a
                specific recommendation for reducing uncertainty.
            </p>
            <div class="rec-list">
                {rec_html}
            </div>
        </section>
        """

    # ── Assemble full HTML ─────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #f5f6fa;
            color: #2c3e50;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            padding: 24px;
        }}
        header {{
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 32px 0;
            margin-bottom: 24px;
        }}
        header h1 {{
            font-size: 28px;
            font-weight: 600;
        }}
        header .subtitle {{
            font-size: 14px;
            opacity: 0.8;
            margin-top: 4px;
        }}
        .data-overview {{
            display: flex;
            gap: 24px;
            font-size: 13px;
            color: rgba(255,255,255,0.7);
            margin-top: 12px;
        }}
        .cards-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }}
        .card-value {{
            font-size: 32px;
            font-weight: 700;
        }}
        .card-label {{
            font-size: 13px;
            color: #7f8c8d;
            margin-top: 4px;
        }}
        .card-sublabel {{
            font-size: 11px;
            color: #95a5a6;
            margin-top: 2px;
        }}
        .report-section {{
            background: white;
            border-radius: 10px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }}
        .report-section h2 {{
            font-size: 18px;
            color: #2c3e50;
            margin-bottom: 8px;
            padding-bottom: 8px;
            border-bottom: 2px solid #ecf0f1;
        }}
        .section-desc {{
            font-size: 13px;
            color: #7f8c8d;
            margin-bottom: 16px;
        }}
        .tab-container {{
            display: flex;
            gap: 8px;
            margin-bottom: 16px;
        }}
        .tab-btn {{
            padding: 8px 16px;
            border: 1px solid #ddd;
            border-radius: 6px;
            background: white;
            cursor: pointer;
            font-size: 13px;
            transition: all 0.2s;
        }}
        .tab-btn.active {{
            background: #34495e;
            color: white;
            border-color: #34495e;
        }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
        .rec-list {{
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}
        .rec-item {{
            padding: 12px 16px;
            background: #fafbfc;
            border-radius: 6px;
        }}
        .rec-header {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 4px;
        }}
        .rec-icon {{ font-size: 16px; }}
        .rec-feature {{
            font-weight: 600;
            font-size: 14px;
        }}
        .rec-action {{
            font-size: 12px;
            font-weight: 500;
            margin-left: auto;
        }}
        .rec-explanation {{
            font-size: 12px;
            color: #7f8c8d;
        }}
        footer {{
            text-align: center;
            padding: 20px;
            font-size: 12px;
            color: #95a5a6;
        }}
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1>{title}</h1>
            <div class="subtitle">Generated by UncertaintyLens</div>
            {data_overview}
        </div>
    </header>

    <div class="container">

        {cards_html}

        <section class="report-section">
            <h2>Feature Risk Overview</h2>
            <p class="section-desc">
                Features ranked by composite uncertainty score.
                Color indicates recommended action based on uncertainty decomposition.
            </p>
            {charts_html["priority"]}
        </section>

        <section class="report-section">
            <h2>Decision Summary</h2>
            <p class="section-desc">
                Complete feature-by-feature decision table with risk levels,
                dominant uncertainty type, and recommended actions.
            </p>
            {charts_html["table"]}
        </section>

        {decomp_section}

        {conformal_section}

        {shift_section}

        {rec_section}

        <section class="report-section">
            <h2>Uncertainty Breakdown</h2>
            <p class="section-desc">
                Detailed view of uncertainty composition per feature across
                all registered detectors.
            </p>
            <div class="tab-container">
                <button class="tab-btn active" onclick="switchTab(event, 'tab-heatmap')">Heatmap</button>
                <button class="tab-btn" onclick="switchTab(event, 'tab-bar')">Stacked Bar</button>
                <button class="tab-btn" onclick="switchTab(event, 'tab-sankey')">Data Flow</button>
            </div>
            <div id="tab-heatmap" class="tab-content active">
                {charts_html["heatmap"]}
            </div>
            <div id="tab-bar" class="tab-content">
                {charts_html["bar"]}
            </div>
            <div id="tab-sankey" class="tab-content">
                {charts_html["sankey"]}
            </div>
        </section>

    </div>

    <footer>
        UncertaintyLens Decision Report
    </footer>

    <script>
        function switchTab(evt, tabId) {{
            var section = evt.target.closest('.report-section');
            section.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            section.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            section.querySelector('#' + tabId).classList.add('active');
            evt.target.classList.add('active');
            // Trigger Plotly resize for newly visible charts
            var plotDiv = section.querySelector('#' + tabId + ' .plotly-graph-div');
            if (plotDiv) Plotly.Plots.resize(plotDiv);
        }}
    </script>
</body>
</html>"""

    output = Path(output_path)
    output.write_text(html, encoding="utf-8")
    return str(output.resolve())
