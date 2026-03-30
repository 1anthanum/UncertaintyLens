"""
Example: Supply Chain Data Uncertainty Analysis

Analyzes uncertainty in supply chain/logistics data, where variance
in lead times and demand is a critical business risk.
"""

import pandas as pd
import numpy as np
from uncertainty_lens import UncertaintyPipeline
from uncertainty_lens.visualizers import (
    create_uncertainty_heatmap,
    create_confidence_plot,
    create_distribution_comparison,
)


def generate_supply_chain_data(n: int = 1500, seed: int = 77) -> pd.DataFrame:
    """Generate synthetic supply chain data with realistic uncertainty patterns."""
    np.random.seed(seed)

    regions = np.random.choice(
        ["North America", "Europe", "Asia Pacific", "Latin America"], n,
        p=[0.35, 0.30, 0.25, 0.10],
    )

    # Lead time varies drastically by region
    lead_time = np.where(
        np.isin(regions, ["Asia Pacific", "Latin America"]),
        np.random.lognormal(3.0, 0.8, n),  # higher variance
        np.random.lognormal(2.5, 0.3, n),  # lower variance
    )

    df = pd.DataFrame(
        {
            "region": regions,
            "lead_time_days": lead_time,
            "order_quantity": np.random.poisson(500, n),
            # ~8% missing (sensor failures in warehouse)
            "inventory_level": np.where(
                np.random.random(n) > 0.08,
                np.random.lognormal(6, 0.5, n).astype(int),
                np.nan,
            ),
            # Demand has seasonal outliers
            "demand_units": np.concatenate(
                [
                    np.random.poisson(450, n - 100),
                    np.random.poisson(1200, 100),  # holiday spikes
                ]
            ),
            # ~12% missing (delayed cost reporting)
            "shipping_cost": np.where(
                np.random.random(n) > 0.12,
                np.random.lognormal(5, 0.7, n),
                np.nan,
            ),
            # Supplier reliability score (some suspicious perfect 100s)
            "supplier_score": np.concatenate(
                [
                    np.clip(np.random.normal(82, 8, n - 40), 50, 99),
                    np.full(40, 100.0),  # suspiciously perfect scores
                ]
            ),
        }
    )

    return df


def main():
    print("Generating supply chain dataset...")
    df = generate_supply_chain_data()
    print(f"  Shape: {df.shape}")
    print(f"  Regions: {df['region'].unique().tolist()}")
    print(f"  Missing values: {df.isna().sum().sum()}")
    print()

    pipeline = UncertaintyPipeline(
        weights={"missing": 0.3, "anomaly": 0.35, "variance": 0.35}
    )
    report = pipeline.analyze(df, group_col="region")

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
    fig.write_html("supply_chain_heatmap.html")

    fig = create_confidence_plot(df, "lead_time_days", "region")
    fig.write_html("lead_time_confidence.html")

    fig = create_distribution_comparison(df, "demand_units", "region")
    fig.write_html("demand_distribution.html")

    print("Visualizations saved as HTML files.")


if __name__ == "__main__":
    main()
