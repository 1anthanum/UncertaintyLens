"""
Example: Advertising Data Uncertainty Analysis

Demonstrates how to use UncertaintyLens to analyze uncertainty in
marketing/advertising data across multiple channels.
"""

import pandas as pd
import numpy as np
from uncertainty_lens import UncertaintyPipeline
from uncertainty_lens.visualizers import (
    create_uncertainty_heatmap,
    create_uncertainty_bar,
    create_confidence_plot,
    create_info_loss_sankey,
)


def generate_ad_data(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic advertising data with realistic uncertainty patterns."""
    np.random.seed(seed)

    channels = np.random.choice(
        ["Search Ads", "Social Media", "Video", "Feed", "Email"], n
    )

    df = pd.DataFrame(
        {
            "channel": channels,
            "impressions": np.random.lognormal(8, 1.5, n).astype(int),
            # ~10% missing clicks (tracking failures)
            "clicks": np.where(
                np.random.random(n) > 0.1,
                np.random.lognormal(5, 1.2, n).astype(int),
                np.nan,
            ),
            # ~25% missing conversions (attribution gaps)
            "conversions": np.where(
                np.random.random(n) > 0.25,
                np.random.poisson(10, n),
                np.nan,
            ),
            # Spend has outliers (30 abnormally high-spend campaigns)
            "spend": np.concatenate(
                [
                    np.random.lognormal(6, 0.8, n - 30),
                    np.random.lognormal(9, 0.5, 30),
                ]
            ),
            # ~35% missing revenue (hard to attribute)
            "attributed_revenue": np.where(
                np.random.random(n) > 0.35,
                np.random.lognormal(7, 1.5, n),
                np.nan,
            ),
        }
    )

    return df


def main():
    print("Generating advertising dataset...")
    df = generate_ad_data()
    print(f"  Shape: {df.shape}")
    print(f"  Missing values: {df.isna().sum().sum()}")
    print()

    # Run the pipeline
    pipeline = UncertaintyPipeline(
        weights={"missing": 0.4, "anomaly": 0.3, "variance": 0.3}
    )
    report = pipeline.analyze(df, group_col="channel")

    # Print uncertainty index
    print("Uncertainty Index (sorted high -> low):")
    print("-" * 70)
    for col, vals in report["uncertainty_index"].items():
        print(
            f"  {col:25s} | composite: {vals['composite_score']:.3f} | "
            f"missing: {vals['missing_score']:.3f} | "
            f"anomaly: {vals['anomaly_score']:.3f} | "
            f"variance: {vals['variance_score']:.3f} | "
            f"{vals['level']}"
        )
    print()

    # Print summary
    summary = report["summary"]
    print(f"Overall uncertainty: {summary['overall_uncertainty']:.1%} ({summary['overall_level']})")
    print(f"High uncertainty features: {summary['high_uncertainty_features']}")
    print(f"Most reliable features: {[f['feature'] for f in summary['most_reliable']]}")
    print()

    # Generate visualizations (saved as HTML)
    print("Generating visualizations...")

    fig = create_uncertainty_heatmap(report["uncertainty_index"])
    fig.write_html("uncertainty_heatmap.html")
    print("  -> uncertainty_heatmap.html")

    fig = create_uncertainty_bar(report["uncertainty_index"])
    fig.write_html("uncertainty_breakdown.html")
    print("  -> uncertainty_breakdown.html")

    fig = create_confidence_plot(df, "spend", "channel")
    fig.write_html("spend_confidence.html")
    print("  -> spend_confidence.html")

    print("\nDone! Open the HTML files in a browser to explore interactively.")


if __name__ == "__main__":
    main()
