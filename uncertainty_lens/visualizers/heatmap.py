"""
Uncertainty heatmap module.

Visualizes the uncertainty index as an interactive heatmap so users
can instantly see which dimensions carry the most uncertainty.
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


def create_uncertainty_heatmap(
    uncertainty_index: Dict[str, Dict],
    title: str = "Data Uncertainty Heatmap",
) -> go.Figure:
    features = list(uncertainty_index.keys())
    dimensions = ["Missing", "Anomaly", "Variance", "Composite"]

    z_data = []
    for col in features:
        vals = uncertainty_index[col]
        z_data.append([
            vals["missing_score"],
            vals["anomaly_score"],
            vals["variance_score"],
            vals["composite_score"],
        ])

    z_array = np.array(z_data)

    fig = go.Figure(data=go.Heatmap(
        z=z_array,
        x=dimensions,
        y=features,
        colorscale=[
            [0.0, "#1a9641"],
            [0.25, "#a6d96a"],
            [0.5, "#ffffbf"],
            [0.75, "#fdae61"],
            [1.0, "#d7191c"],
        ],
        zmin=0,
        zmax=1,
        text=[[f"{v:.3f}" for v in row] for row in z_data],
        texttemplate="%{text}",
        textfont={"size": 12},
        hovertemplate=(
            "Feature: %{y}<br>"
            "Dimension: %{x}<br>"
            "Score: %{z:.3f}<br>"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        title={"text": title, "font": {"size": 18}},
        xaxis_title="Uncertainty Dimension",
        yaxis_title="Feature",
        width=800,
        height=max(400, len(features) * 50),
        font={"family": "Arial"},
    )

    return fig


def create_uncertainty_bar(
    uncertainty_index: Dict[str, Dict],
    title: str = "Uncertainty Score Breakdown by Feature",
) -> go.Figure:
    features = list(uncertainty_index.keys())

    missing_scores = [uncertainty_index[f]["missing_score"] for f in features]
    anomaly_scores = [uncertainty_index[f]["anomaly_score"] for f in features]
    variance_scores = [uncertainty_index[f]["variance_score"] for f in features]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Missing",
        x=features,
        y=missing_scores,
        marker_color="#3288bd",
    ))
    fig.add_trace(go.Bar(
        name="Anomaly",
        x=features,
        y=anomaly_scores,
        marker_color="#fee08b",
    ))
    fig.add_trace(go.Bar(
        name="Variance",
        x=features,
        y=variance_scores,
        marker_color="#d53e4f",
    ))

    fig.update_layout(
        barmode="stack",
        title={"text": title, "font": {"size": 18}},
        xaxis_title="Feature",
        yaxis_title="Uncertainty Score",
        yaxis={"range": [0, 1.2]},
        width=800,
        height=500,
        legend={"orientation": "h", "y": -0.15},
        font={"family": "Arial"},
    )

    return fig
