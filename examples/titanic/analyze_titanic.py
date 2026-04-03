"""
Case Study: Titanic Passenger Data — What Uncertainty Hides in a "Clean" Dataset

Demonstrates how UncertaintyLens reveals hidden uncertainty in a well-known
dataset that most analysts treat as straightforward.

Dataset: Titanic passenger list (from seaborn's built-in datasets)
         891 rows × 15 columns, with real missing values and structural anomalies.

Usage:
    pip install seaborn   # one-time, for dataset access
    python examples/titanic/analyze_titanic.py
"""

import numpy as np
import pandas as pd
from uncertainty_lens import UncertaintyPipeline
from uncertainty_lens.quantifiers import MonteCarloQuantifier
from uncertainty_lens.visualizers import (
    create_uncertainty_heatmap,
    create_uncertainty_bar,
    create_confidence_plot,
    create_distribution_comparison,
    create_info_loss_sankey,
)


def load_titanic() -> pd.DataFrame:
    """
    Load the Titanic dataset.

    Tries (in order):
    1. seaborn.load_dataset (if seaborn is installed and has network)
    2. Direct CSV download from GitHub
    3. Synthetic replica with matching statistical properties
    """
    # Try seaborn first
    try:
        import seaborn as sns

        return sns.load_dataset("titanic")
    except Exception:
        pass

    # Try direct download
    try:
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
        return pd.read_csv(url)
    except Exception:
        pass

    # Fallback: generate a synthetic replica
    print("  (Using synthetic Titanic replica — install seaborn for real data)")
    rng = np.random.default_rng(42)
    n = 891

    pclass = rng.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55])
    sex = rng.choice(["male", "female"], n, p=[0.65, 0.35])
    survived = np.where(
        (pclass == 1) & (sex == "female"),
        rng.choice([0, 1], n, p=[0.03, 0.97]),
        np.where(
            (pclass == 3) & (sex == "male"),
            rng.choice([0, 1], n, p=[0.86, 0.14]),
            rng.choice([0, 1], n, p=[0.4, 0.6]),
        ),
    )

    # Age: ~20% missing, correlated with pclass (non-random)
    age = np.where(
        pclass == 1,
        rng.normal(38, 12, n),
        np.where(pclass == 2, rng.normal(30, 11, n), rng.normal(25, 10, n)),
    )
    age = np.clip(age, 0.5, 80)
    age_missing = rng.random(n) < np.where(pclass == 3, 0.28, 0.12)
    age = np.where(age_missing, np.nan, age)

    # Fare: log-normal with class-dependent mean, outliers in 1st class
    fare = np.where(
        pclass == 1,
        rng.lognormal(3.8, 0.8, n),
        np.where(pclass == 2, rng.lognormal(2.6, 0.4, n), rng.lognormal(2.0, 0.6, n)),
    )

    sibsp = rng.choice([0, 1, 2, 3, 4, 5], n, p=[0.68, 0.23, 0.05, 0.02, 0.01, 0.01])
    parch = rng.choice([0, 1, 2, 3, 4, 5], n, p=[0.76, 0.12, 0.08, 0.02, 0.01, 0.01])

    return pd.DataFrame(
        {
            "survived": survived,
            "pclass": pclass,
            "sex": sex,
            "age": age,
            "sibsp": sibsp,
            "parch": parch,
            "fare": fare,
        }
    )


def print_section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def main():
    # ── 1. Load & inspect ────────────────────────────────────────────────
    print_section("1. Loading Titanic Dataset")
    df = load_titanic()
    print(f"Shape: {df.shape[0]} passengers × {df.shape[1]} features")
    print(f"\nMissing values per column:")
    missing = df.isnull().sum()
    for col in missing[missing > 0].sort_values(ascending=False).index:
        pct = missing[col] / len(df) * 100
        print(f"  {col:20s}  {missing[col]:4d}  ({pct:.1f}%)")

    # ── 2. Run UncertaintyLens pipeline ──────────────────────────────────
    print_section("2. Running Uncertainty Analysis")

    pipeline = UncertaintyPipeline(weights={"missing": 0.4, "anomaly": 0.3, "variance": 0.3})
    report = pipeline.analyze(df, group_col="pclass")

    # ── 3. Uncertainty Index ─────────────────────────────────────────────
    print_section("3. Uncertainty Index (sorted high → low)")
    print(
        f"  {'Feature':20s} | {'Composite':>9s} | {'Missing':>7s} | {'Anomaly':>7s} | {'Variance':>8s} | Level"
    )
    print(f"  {'-'*20}-+-{'-'*9}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}-+--------")
    for col, vals in report["uncertainty_index"].items():
        print(
            f"  {col:20s} | {vals['composite_score']:9.3f} | "
            f"{vals['missing_score']:7.3f} | "
            f"{vals['anomaly_score']:7.3f} | "
            f"{vals['variance_score']:8.3f} | {vals['level']}"
        )

    # ── 4. Key findings ─────────────────────────────────────────────────
    print_section("4. Key Findings")
    summary = report["summary"]
    print(
        f"Overall dataset uncertainty : {summary['overall_uncertainty']:.1%} ({summary['overall_level']})"
    )
    print(f"Features analyzed           : {summary['total_features_analyzed']}")
    print(f"High-uncertainty features   : {summary['high_uncertainty_features']}")
    print(f"Low-uncertainty features    : {summary['low_uncertainty_features']}")

    # ── 5. Monte Carlo: How robust is the survival rate? ─────────────
    print_section("5. Monte Carlo: How Robust Is the Mean Fare?")

    quantifier = MonteCarloQuantifier(n_simulations=500, random_state=42)
    mc_result = quantifier.estimate(
        df,
        statistic_fn=lambda d: d["fare"].mean(),
        columns=["fare"],
    )

    if "error" not in mc_result:
        ci = mc_result["confidence_interval_95"]
        print(f"Point estimate     : ${mc_result['point_estimate']:.2f}")
        print(f"MC mean            : ${mc_result['mean']:.2f}")
        print(f"95% CI             : [${ci[0]:.2f}, ${ci[1]:.2f}]")
        print(f"CI width           : ${ci[1] - ci[0]:.2f}")
        print(f"Sensitivity ratio  : {mc_result['sensitivity_ratio']:.2%}")
        print(f"Simulations used   : {mc_result['successful_simulations']}/500")

        if mc_result["sensitivity_ratio"] > 0.1:
            print("\n⚠  The mean fare is sensitive to data uncertainty.")
            print("   Decisions based on this statistic should account for the wide CI.")
        else:
            print("\n✓  The mean fare is robust — data uncertainty has minimal impact.")
    else:
        print(f"Monte Carlo could not run: {mc_result['error']}")

    # ── 6. Interpretation ────────────────────────────────────────────────
    print_section("6. What This Means for Analysis")
    print(
        "The Titanic dataset is often treated as clean and ready for modeling.\n"
        "UncertaintyLens reveals a more nuanced picture:\n"
        "\n"
        "• 'age' has ~20% missing values. The MCAR test shows whether this\n"
        "  missingness is random or correlated with survival — which changes\n"
        "  whether simple mean imputation is valid.\n"
        "\n"
        "• 'fare' has extreme outliers (max $512 vs median $14). The ensemble\n"
        "  anomaly detector flags these, and Monte Carlo shows how much they\n"
        "  shift the mean.\n"
        "\n"
        "• Between-class variance decomposition shows that Pclass explains\n"
        "  a large portion of fare variance — but within-class variance in\n"
        "  1st class is still very high, meaning fare alone is an unreliable\n"
        "  predictor even within a class.\n"
        "\n"
        "Bottom line: a model trained on this data without acknowledging these\n"
        "uncertainty sources will produce overconfident predictions.\n"
    )

    # ── 7. Generate visualizations ──────────────────────────────────────
    print_section("7. Generating Visualizations")

    fig = create_uncertainty_heatmap(report["uncertainty_index"])
    fig.write_html("titanic_heatmap.html")
    print("  → titanic_heatmap.html")

    fig = create_uncertainty_bar(report["uncertainty_index"])
    fig.write_html("titanic_breakdown.html")
    print("  → titanic_breakdown.html")

    fig = create_confidence_plot(df, "fare", "pclass")
    fig.write_html("titanic_fare_ci.html")
    print("  → titanic_fare_ci.html")

    fig = create_distribution_comparison(df, "fare", "pclass")
    fig.write_html("titanic_fare_dist.html")
    print("  → titanic_fare_dist.html")

    missing_rows = int(df.isnull().any(axis=1).sum())
    consensus = report["anomaly_analysis"].get("consensus_anomalies", {})
    anomaly_rows = max(consensus.values()) if consensus else 0
    fig = create_info_loss_sankey(
        total_records=df.shape[0],
        missing_records=missing_rows,
        anomaly_records=anomaly_rows,
        high_variance_records=50,
        title="Titanic — Information Loss Flow",
    )
    fig.write_html("titanic_sankey.html")
    print("  → titanic_sankey.html")

    print("\nDone! Open the HTML files in a browser to explore interactively.")


if __name__ == "__main__":
    main()
