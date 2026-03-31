"""
Information loss Sankey diagram module.

Shows how much information is lost at each stage from raw data to
decision-ready data.
"""

import plotly.graph_objects as go
from typing import Dict, List, Optional


def create_info_loss_sankey(
    total_records: int,
    missing_records: int,
    anomaly_records: int,
    high_variance_records: int,
    title: str = "Data Information Loss Analysis",
) -> go.Figure:
    clean_records = max(
        0,
        total_records - missing_records - anomaly_records - high_variance_records,
    )

    uncertain_total = total_records - clean_records

    labels = [
        f"Raw Data\n({total_records:,})",
        f"Missing\n({missing_records:,})",
        f"Anomalous\n({anomaly_records:,})",
        f"High Variance\n({high_variance_records:,})",
        f"Reliable\n({clean_records:,})",
        f"Uncertain\n({uncertain_total:,})",
    ]

    source = [0, 0, 0, 0, 1, 2, 3]
    target = [1, 2, 3, 4, 5, 5, 5]
    value = [
        missing_records,
        anomaly_records,
        high_variance_records,
        clean_records,
        missing_records,
        anomaly_records,
        high_variance_records,
    ]

    node_colors = [
        "#2196F3",  # Raw — blue
        "#FF9800",  # Missing — orange
        "#F44336",  # Anomaly — red
        "#9C27B0",  # High variance — purple
        "#4CAF50",  # Reliable — green
        "#E91E63",  # Uncertain — pink
    ]

    link_colors = [
        "rgba(255,152,0,0.3)",
        "rgba(244,67,54,0.3)",
        "rgba(156,39,176,0.3)",
        "rgba(76,175,80,0.3)",
        "rgba(255,152,0,0.2)",
        "rgba(244,67,54,0.2)",
        "rgba(156,39,176,0.2)",
    ]

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=20,
                    thickness=30,
                    line=dict(color="black", width=0.5),
                    label=labels,
                    color=node_colors,
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value,
                    color=link_colors,
                ),
            )
        ]
    )

    loss_rate = uncertain_total / total_records if total_records > 0 else 0

    fig.update_layout(
        title={
            "text": f"{title}<br>"
            f"<sub>Loss Rate: {loss_rate:.1%} | "
            f"Reliable: {clean_records:,}/{total_records:,}</sub>",
            "font": {"size": 16},
        },
        width=900,
        height=500,
        font={"family": "Arial", "size": 12},
    )

    return fig
