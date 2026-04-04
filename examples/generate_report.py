"""
Generate an interactive decision report for the Titanic dataset.

Demonstrates the full UncertaintyLens pipeline with all detectors
and the integrated HTML report generator.

Usage:
    python examples/generate_report.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

from uncertainty_lens import UncertaintyPipeline
from uncertainty_lens.detectors import (
    ConformalShiftDetector,
    UncertaintyDecomposer,
    ConformalPredictor,
)


def load_titanic() -> pd.DataFrame:
    """Load or generate the Titanic dataset."""
    try:
        import seaborn as sns

        return sns.load_dataset("titanic")
    except Exception:
        pass

    try:
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
        return pd.read_csv(url)
    except Exception:
        pass

    # Synthetic fallback
    print("  (Using synthetic Titanic replica)")
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

    age = np.where(
        pclass == 1,
        rng.normal(38, 12, n),
        np.where(pclass == 2, rng.normal(30, 11, n), rng.normal(25, 10, n)),
    )
    age = np.clip(age, 0.5, 80)
    age_missing = rng.random(n) < np.where(pclass == 3, 0.28, 0.12)
    age = np.where(age_missing, np.nan, age)

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


def main():
    print("Loading dataset...")
    df = load_titanic()
    print(f"  Shape: {df.shape[0]} × {df.shape[1]}")

    print("Setting up pipeline with all detectors...")
    pipeline = UncertaintyPipeline(weights={"missing": 0.35, "anomaly": 0.25, "variance": 0.25})

    # Register Tier 1 detectors
    pipeline.register(
        "conformal_shift",
        ConformalShiftDetector(seed=42),
        weight=0.1,
    )
    pipeline.register(
        "decomposition",
        UncertaintyDecomposer(n_bootstrap=200, seed=42),
        weight=0.15,
    )

    # Register Tier 2 detector
    pipeline.register(
        "conformal_pred",
        ConformalPredictor(coverage=0.9, seed=42),
        weight=0.1,
    )

    print(f"  Registered detectors: {[name for name, _, _ in pipeline.registered_detectors]}")

    print("Running analysis...")
    report = pipeline.analyze(df, group_col="pclass")

    print("Generating decision report...")
    output_dir = Path(__file__).parent
    output_path = output_dir / "titanic_decision_report.html"

    report_path = pipeline.generate_report(
        df=df,
        output_path=str(output_path),
        title="Titanic Uncertainty Decision Report",
    )
    print(f"  Report saved to: {report_path}")

    # Print summary
    summary = report["summary"]
    print(f"\n{'=' * 50}")
    print(
        f"  Overall Uncertainty: {summary['overall_uncertainty']:.1%} ({summary['overall_level']})"
    )
    print(f"  Features analyzed:  {summary['total_features_analyzed']}")
    print(f"  High-risk features: {summary['high_uncertainty_features']}")
    print(f"  Reliable features:  {summary['low_uncertainty_features']}")
    print(f"{'=' * 50}")

    print("\nDone! Open the HTML report in a browser.")


if __name__ == "__main__":
    main()
