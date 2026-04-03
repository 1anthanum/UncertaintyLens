"""
Confidence interval visualization module.

Uses error bands to display uncertainty ranges, letting users intuitively
understand "how confident are we about this value."
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional, List


def create_confidence_plot(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    confidence_level: float = 0.95,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Create a grouped confidence interval plot.

    Wider confidence intervals = higher uncertainty.
    """
    if value_col not in df.columns:
        raise ValueError(f"value_col '{value_col}' not found in DataFrame")
    if group_col not in df.columns:
        raise ValueError(f"group_col '{group_col}' not found in DataFrame")

    groups = df[group_col].unique()
    means = []
    ci_lower = []
    ci_upper = []
    group_names = []
    n_samples = []

    for group in sorted(groups):
        data = df[df[group_col] == group][value_col].dropna()
        if len(data) < 2:
            continue

        mean = data.mean()
        se = stats.sem(data)
        ci = stats.t.interval(confidence_level, df=len(data) - 1, loc=mean, scale=se)

        group_names.append(str(group))
        means.append(mean)
        ci_lower.append(ci[0])
        ci_upper.append(ci[1])
        n_samples.append(len(data))

    ci_widths = [u - l for u, l in zip(ci_upper, ci_lower)]

    fig = go.Figure()

    if not group_names:
        fig.add_annotation(
            text="No groups with enough data (need ≥ 2 observations per group)",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="#64748b"),
        )
        return fig

    fig.add_trace(
        go.Scatter(
            x=group_names,
            y=means,
            error_y=dict(
                type="data",
                symmetric=False,
                array=[u - m for u, m in zip(ci_upper, means)],
                arrayminus=[m - l for m, l in zip(means, ci_lower)],
                color="rgba(214, 39, 40, 0.6)",
                thickness=2,
                width=10,
            ),
            mode="markers",
            marker=dict(size=10, color="#1f77b4"),
            name=f"Mean +/- {confidence_level:.0%} CI",
            hovertemplate=(
                "Group: %{x}<br>"
                "Mean: %{y:.2f}<br>"
                "CI Width: %{customdata[0]:.2f}<br>"
                "Sample Size: %{customdata[1]}<br>"
                "<extra></extra>"
            ),
            customdata=list(zip(ci_widths, n_samples)),
        )
    )

    if not title:
        title = f"{value_col} — Grouped Confidence Intervals ({confidence_level:.0%} CI)"

    fig.update_layout(
        title={"text": title, "font": {"size": 16}},
        xaxis_title=group_col,
        yaxis_title=value_col,
        width=800,
        height=500,
        font={"family": "Arial"},
        showlegend=True,
    )

    return fig


def create_distribution_comparison(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Create a grouped distribution comparison (violin plot).

    Violin plots convey more information than box plots:
    - Distribution shape (multi-modal, skewness)
    - Density concentration regions
    - Tail behavior
    """
    if value_col not in df.columns:
        raise ValueError(f"value_col '{value_col}' not found in DataFrame")
    if group_col not in df.columns:
        raise ValueError(f"group_col '{group_col}' not found in DataFrame")

    groups = sorted(df[group_col].unique())

    fig = go.Figure()

    colors = px.colors.qualitative.Set2

    def _to_rgba(color_str: str, alpha: float = 0.3) -> str:
        """Convert any Plotly color string (hex or rgb) to rgba."""
        if color_str.startswith("#"):
            rgb = px.colors.hex_to_rgb(color_str)
            return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})"
        elif color_str.startswith("rgb("):
            return color_str.replace("rgb(", "rgba(").replace(")", f", {alpha})")
        return color_str

    traces_added = 0
    for i, group in enumerate(groups):
        data = df[df[group_col] == group][value_col].dropna()
        if len(data) < 5:
            continue

        color = colors[i % len(colors)]
        fig.add_trace(
            go.Violin(
                y=data,
                name=str(group),
                box_visible=True,
                meanline_visible=True,
                line_color=color,
                fillcolor=_to_rgba(color),
            )
        )
        traces_added += 1

    if traces_added == 0:
        fig.add_annotation(
            text="No groups with enough data (need ≥ 5 observations per group)",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="#64748b"),
        )

    if not title:
        title = f"{value_col} — Distribution Comparison by {group_col}"

    fig.update_layout(
        title={"text": title, "font": {"size": 16}},
        yaxis_title=value_col,
        xaxis_title=group_col,
        width=800,
        height=500,
        font={"family": "Arial"},
    )

    return fig
