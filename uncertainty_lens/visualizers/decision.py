"""
Decision-oriented uncertainty visualizations.

These charts answer "what should I do?" rather than just "how much uncertainty?"
They map uncertainty types to actionable recommendations, helping analysts
make informed decisions about data collection, measurement, and modeling.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional, List

# ═══════════════════════════════════════════════════════════════════════
# 1. Epistemic vs. Aleatoric scatter (quadrant chart)
# ═══════════════════════════════════════════════════════════════════════


def create_decomposition_scatter(
    decomposition: Dict[str, Dict[str, Any]],
    title: str = "Uncertainty Decomposition: What Can You Control?",
) -> go.Figure:
    """
    Quadrant scatter plot of epistemic vs. aleatoric uncertainty per feature.

    Quadrants:
      - Top-left:     High epistemic, low aleatoric → Collect more data
      - Top-right:    High epistemic, high aleatoric → Both actions needed
      - Bottom-left:  Low epistemic, low aleatoric  → Feature is reliable
      - Bottom-right: Low epistemic, high aleatoric → Improve measurement

    Parameters
    ----------
    decomposition : dict
        From UncertaintyDecomposer results: ``results["decomposition"]``.
        Each entry needs ``aleatoric_score`` and ``epistemic_score``.
    title : str
        Chart title.

    Returns
    -------
    go.Figure
    """
    features = []
    ale_scores = []
    epi_scores = []
    dominants = []

    for feat, d in decomposition.items():
        if d.get("dominant") == "insufficient_data":
            continue
        features.append(feat)
        ale_scores.append(d["aleatoric_score"])
        epi_scores.append(d["epistemic_score"])
        dominants.append(d.get("dominant", "mixed"))

    if not features:
        fig = go.Figure()
        fig.add_annotation(text="No decomposition data available", showarrow=False)
        return fig

    # Color by dominant type
    color_map = {
        "epistemic": "#e74c3c",  # red
        "aleatoric": "#3498db",  # blue
        "mixed": "#9b59b6",  # purple
    }
    colors = [color_map.get(d, "#95a5a6") for d in dominants]

    fig = go.Figure()

    # Quadrant background shading
    for x0, x1, y0, y1, color, label in [
        (0, 0.5, 0.5, 1.0, "rgba(231,76,60,0.06)", "Collect More Data"),
        (0.5, 1.0, 0.5, 1.0, "rgba(155,89,182,0.06)", "Both Actions Needed"),
        (0, 0.5, 0, 0.5, "rgba(46,204,113,0.06)", "Reliable"),
        (0.5, 1.0, 0, 0.5, "rgba(52,152,219,0.06)", "Improve Measurement"),
    ]:
        fig.add_shape(
            type="rect",
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            fillcolor=color,
            line=dict(width=0),
            layer="below",
        )
        fig.add_annotation(
            x=(x0 + x1) / 2,
            y=(y0 + y1) / 2,
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(size=11, color="rgba(0,0,0,0.25)"),
        )

    # Scatter points
    fig.add_trace(
        go.Scatter(
            x=ale_scores,
            y=epi_scores,
            mode="markers+text",
            marker=dict(
                size=14,
                color=colors,
                line=dict(width=1, color="white"),
                opacity=0.85,
            ),
            text=features,
            textposition="top center",
            textfont=dict(size=10),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Aleatoric: %{x:.3f}<br>"
                "Epistemic: %{y:.3f}<br>"
                "<extra></extra>"
            ),
        )
    )

    # Threshold lines
    fig.add_hline(y=0.5, line_dash="dot", line_color="rgba(0,0,0,0.2)")
    fig.add_vline(x=0.5, line_dash="dot", line_color="rgba(0,0,0,0.2)")

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(
            title="Aleatoric Score (irreducible noise)",
            range=[-0.05, 1.05],
            dtick=0.2,
        ),
        yaxis=dict(
            title="Epistemic Score (reducible with more data)",
            range=[-0.05, 1.05],
            dtick=0.2,
        ),
        width=700,
        height=600,
        showlegend=False,
        template="plotly_white",
    )

    return fig


# ═══════════════════════════════════════════════════════════════════════
# 2. Action priority waterfall
# ═══════════════════════════════════════════════════════════════════════


def create_action_priority_chart(
    uncertainty_index: Dict[str, Dict[str, Any]],
    recommendations: Optional[Dict[str, Dict[str, str]]] = None,
    title: str = "Feature Action Priority",
) -> go.Figure:
    """
    Horizontal bar chart ranking features by composite uncertainty score,
    color-coded by recommended action.

    Parameters
    ----------
    uncertainty_index : dict
        Pipeline's ``report["uncertainty_index"]``.
    recommendations : dict, optional
        From decomposer: ``results["recommendation"]``.
        If not provided, action colors default to grey.
    title : str
        Chart title.

    Returns
    -------
    go.Figure
    """
    # Sort by composite score descending
    sorted_items = sorted(
        uncertainty_index.items(),
        key=lambda x: x[1]["composite_score"],
        reverse=True,
    )

    features = [item[0] for item in sorted_items]
    scores = [item[1]["composite_score"] for item in sorted_items]
    levels = [item[1].get("level", "") for item in sorted_items]

    # Map actions to colors
    action_colors = {
        "collect_more_data": "#e74c3c",
        "improve_measurement": "#3498db",
        "both": "#9b59b6",
        "none": "#2ecc71",
    }

    if recommendations:
        colors = [
            action_colors.get(recommendations.get(f, {}).get("action", "none"), "#bdc3c7")
            for f in features
        ]
        hover_texts = [
            recommendations.get(f, {}).get("explanation", "No recommendation") for f in features
        ]
    else:
        colors = ["#e74c3c" if s >= 0.6 else "#f39c12" if s >= 0.4 else "#2ecc71" for s in scores]
        hover_texts = [f"Score: {s:.3f} ({lv})" for s, lv in zip(scores, levels)]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=features,
            x=scores,
            orientation="h",
            marker=dict(color=colors, line=dict(width=1, color="white")),
            text=[f"{s:.2f}" for s in scores],
            textposition="outside",
            hovertext=hover_texts,
            hoverinfo="text",
        )
    )

    # Add threshold lines
    for threshold, label, color in [
        (0.2, "Low", "#2ecc71"),
        (0.6, "High", "#e74c3c"),
    ]:
        fig.add_vline(
            x=threshold,
            line_dash="dot",
            line_color=color,
            annotation_text=label,
            annotation_position="top",
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(title="Composite Uncertainty Score", range=[0, 1.15]),
        yaxis=dict(autorange="reversed"),
        width=750,
        height=max(350, 40 * len(features) + 100),
        template="plotly_white",
        margin=dict(l=120),
    )

    return fig


# ═══════════════════════════════════════════════════════════════════════
# 3. Decision summary table (Plotly table)
# ═══════════════════════════════════════════════════════════════════════


def create_decision_table(
    uncertainty_index: Dict[str, Dict[str, Any]],
    recommendations: Optional[Dict[str, Dict[str, str]]] = None,
    decomposition: Optional[Dict[str, Dict[str, Any]]] = None,
    title: str = "Decision Summary",
) -> go.Figure:
    """
    Interactive table summarizing each feature's uncertainty profile
    and recommended action.

    Parameters
    ----------
    uncertainty_index : dict
        Pipeline's ``report["uncertainty_index"]``.
    recommendations : dict, optional
        From decomposer's ``results["recommendation"]``.
    decomposition : dict, optional
        From decomposer's ``results["decomposition"]``.
    title : str

    Returns
    -------
    go.Figure
    """
    sorted_items = sorted(
        uncertainty_index.items(),
        key=lambda x: x[1]["composite_score"],
        reverse=True,
    )

    features = []
    composite_scores = []
    risk_levels = []
    dominant_types = []
    actions = []

    action_labels = {
        "collect_more_data": "📊 Collect More Data",
        "improve_measurement": "🔧 Improve Measurement",
        "both": "📊🔧 Both Needed",
        "none": "✅ No Action",
    }

    level_colors = {
        "Low": "#27ae60",
        "Medium-Low": "#2ecc71",
        "Medium": "#f39c12",
        "Medium-High": "#e67e22",
        "High": "#e74c3c",
    }

    for feat, vals in sorted_items:
        features.append(feat)
        composite_scores.append(f"{vals['composite_score']:.3f}")
        level = vals.get("level", "Medium")
        risk_levels.append(level)

        if decomposition and feat in decomposition:
            dom = decomposition[feat].get("dominant", "—")
            dominant_types.append(dom.capitalize())
        else:
            dominant_types.append("—")

        if recommendations and feat in recommendations:
            act = recommendations[feat].get("action", "none")
            actions.append(action_labels.get(act, act))
        else:
            actions.append("—")

    # Color cells by risk level
    level_cell_colors = [level_colors.get(lv, "#bdc3c7") for lv in risk_levels]

    fig = go.Figure(
        data=[
            go.Table(
                columnwidth=[120, 80, 90, 90, 160],
                header=dict(
                    values=[
                        "<b>Feature</b>",
                        "<b>Score</b>",
                        "<b>Risk</b>",
                        "<b>Dominant</b>",
                        "<b>Recommended Action</b>",
                    ],
                    fill_color="#34495e",
                    font=dict(color="white", size=12),
                    align="left",
                    height=35,
                ),
                cells=dict(
                    values=[features, composite_scores, risk_levels, dominant_types, actions],
                    fill_color=[
                        ["white"] * len(features),
                        ["white"] * len(features),
                        [
                            [
                                f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.15)"
                                for c in level_cell_colors
                            ]
                        ],
                        ["white"] * len(features),
                        ["white"] * len(features),
                    ],
                    font=dict(size=11),
                    align="left",
                    height=30,
                ),
            )
        ]
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        width=750,
        height=max(300, 35 * len(features) + 100),
        margin=dict(l=10, r=10, t=50, b=10),
    )

    return fig


# ═══════════════════════════════════════════════════════════════════════
# 4. Conformal interval visualization
# ═══════════════════════════════════════════════════════════════════════


def create_conformal_intervals(
    conformal_results: Dict[str, Dict[str, Any]],
    title: str = "Prediction Interval Width (Conformal)",
) -> go.Figure:
    """
    Bar chart showing conformal prediction interval widths per feature,
    with coverage annotations.

    Parameters
    ----------
    conformal_results : dict
        From ConformalPredictor: ``results["conformal_results"]``.
    title : str

    Returns
    -------
    go.Figure
    """
    features = []
    widths = []
    coverages = []

    for feat, cr in conformal_results.items():
        if "interval_width" not in cr:
            continue
        features.append(feat)
        widths.append(cr["interval_width"])
        coverages.append(cr.get("empirical_coverage_cal", cr.get("empirical_coverage", 0)))

    if not features:
        fig = go.Figure()
        fig.add_annotation(text="No conformal results available", showarrow=False)
        return fig

    # Color by width (wider = more uncertain)
    max_w = max(widths) if widths else 1.0
    colors = [
        f"rgba({int(220 * w / max_w)}, {int(100 * (1 - w / max_w))}, {int(50 * (1 - w / max_w))}, 0.8)"
        for w in widths
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=features,
            y=widths,
            marker=dict(color=colors, line=dict(width=1, color="white")),
            text=[f"±{w:.2f}" for w in widths],
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Interval Width: %{y:.3f}<br>"
                "Coverage: %{customdata:.1%}<br>"
                "<extra></extra>"
            ),
            customdata=coverages,
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(title="Feature"),
        yaxis=dict(title="Prediction Interval Width"),
        width=700,
        height=450,
        template="plotly_white",
    )

    return fig


# ═══════════════════════════════════════════════════════════════════════
# 5. Shift detection group comparison
# ═══════════════════════════════════════════════════════════════════════


def create_shift_overview(
    shift_results: Dict[str, Any],
    title: str = "Distribution Shift Detection",
) -> go.Figure:
    """
    Heatmap showing which group × feature combinations exhibit
    distributional shifts, with p-values as hover data.

    Parameters
    ----------
    shift_results : dict
        From ConformalShiftDetector: full ``results`` dict.
    title : str

    Returns
    -------
    go.Figure
    """
    group_shift = shift_results.get("group_shift", {})
    if not group_shift:
        fig = go.Figure()
        fig.add_annotation(text="No shift data available", showarrow=False)
        return fig

    groups = sorted(group_shift.keys())
    # Collect all features across groups
    all_features = set()
    for g_data in group_shift.values():
        all_features.update(g_data.keys())
    features = sorted(all_features)

    # Build matrix: 1 = shifted, 0 = not shifted
    z_matrix = []
    hover_matrix = []
    for group in groups:
        row = []
        hover_row = []
        for feat in features:
            entry = group_shift.get(group, {}).get(feat, {})
            shifted = entry.get("shifted", False)
            p_val = entry.get("p_value", None)
            row.append(1.0 if shifted else 0.0)
            p_str = f"p={p_val:.4f}" if p_val is not None else "N/A"
            hover_row.append(
                f"Group: {group}<br>Feature: {feat}<br>"
                f"Shifted: {'Yes' if shifted else 'No'}<br>{p_str}"
            )
        z_matrix.append(row)
        hover_matrix.append(hover_row)

    fig = go.Figure(
        data=go.Heatmap(
            z=z_matrix,
            x=features,
            y=groups,
            colorscale=[[0, "#2ecc71"], [1, "#e74c3c"]],
            showscale=False,
            hovertext=hover_matrix,
            hoverinfo="text",
            zmin=0,
            zmax=1,
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(title="Feature"),
        yaxis=dict(title="Group"),
        width=700,
        height=max(300, 50 * len(groups) + 100),
        template="plotly_white",
    )

    return fig
