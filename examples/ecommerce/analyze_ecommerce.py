"""
Example: E-Commerce Data Uncertainty Analysis

Analyzes uncertainty in e-commerce transaction data, focusing on
revenue attribution and customer behavior patterns.
"""

import pandas as pd
import numpy as np
from uncertainty_lens import UncertaintyPipeline
from uncertainty_lens.visualizers import (
    create_uncertainty_heatmap,
    create_uncertainty_bar,
    create_info_loss_sankey,
)


def generate_ecommerce_data(n: int = 2000, seed: int = 123) -> pd.DataFrame:
    """Generate synthetic e-commerce data with realistic uncertainty patterns."""
    np.random.seed(seed)

    categories = np.random.choice(
        ["Electronics", "Clothing", "Home & Garden", "Books", "Food"], n,
        p=[0.25, 0.30, 0.15, 0.15, 0.15],
    )

    df = pd.DataFrame(
        {
            "category": categories,
            "page_views": np.random.poisson(12, n),
            # ~5% missing session duration (tracking script blocked)
            "session_duration_sec": np.where(
                np.random.random(n) > 0.05,
                np.random.lognormal(5, 1.0, n),
                np.nan,
            ),
            # Cart value: high variance in Electronics, low in Books
            "cart_value": np.where(
                np.isin(categories, ["Electronics"]),
                np.random.lognormal(5, 1.8, n),
                np.random.lognormal(3, 0.6, n),
            ),
            # ~15% missing (guest checkouts with no follow-up)
            "items_purchased": np.where(
                np.random.random(n) > 0.15,
                np.random.poisson(2.5, n),
                np.nan,
            ),
            # ~40% missing (many visitors don't convert)
            "revenue": np.where(
                np.random.random(n) > 0.4,
                np.random.lognormal(4, 1.2, n),
                np.nan,
            ),
            # Satisfaction score: some anomalous 0s from bots
            "satisfaction_score": np.concatenate(
                [
                    np.clip(np.random.normal(4.0, 0.8, n - 50), 1, 5),
                    np.zeros(50),  # bot/spam ratings
                ]
            ),
        }
    )

    return df


def main():
    print("Generating e-commerce dataset...")
    df = generate_ecommerce_data()
    print(f"  Shape: {df.shape}")
    print(f"  Categories: {df['category'].unique().tolist()}")
    print(f"  Missing values: {df.isna().sum().sum()}")
    print()

    pipeline = UncertaintyPipeline()
    report = pipeline.analyze(df, group_col="category")

    print("Uncertainty Index:")
    print("-" * 70)
    for col, vals in report["uncertainty_index"].items():
        print(
            f"  {col:25s} | composite: {vals['composite_score']:.3f} | {vals['level']}"
        )
    print()

    summary = report["summary"]
    print(f"Overall uncertainty: {summary['overall_uncertainty']:.1%}")
    print(f"High uncertainty features: {summary['high_uncertainty_features']}")
    print()

    # Save visualizations
    fig = create_uncertainty_heatmap(report["uncertainty_index"])
    fig.write_html("ecommerce_heatmap.html")

    fig = create_uncertainty_bar(report["uncertainty_index"])
    fig.write_html("ecommerce_breakdown.html")

    missing_rows = int(df.isnull().any(axis=1).sum())
    fig = create_info_loss_sankey(
        total_records=df.shape[0],
        missing_records=missing_rows,
        anomaly_records=50,
        high_variance_records=200,
    )
    fig.write_html("ecommerce_sankey.html")

    print("Visualizations saved as HTML files.")


if __name__ == "__main__":
    main()
