"""
生成演示用 HTML 报告 — 使用 Insurance 盲测数据集。

PYTHONPATH=. python examples/generate_demo_report.py
"""

import sys
import warnings
import numpy as np
import pandas as pd

from uncertainty_lens import UncertaintyPipeline
from uncertainty_lens.detectors import (
    ConformalShiftDetector,
    UncertaintyDecomposer,
    JackknifePlusDetector,
    MMDShiftDetector,
    ZeroInflationDetector,
)

warnings.filterwarnings("ignore", category=UserWarning)


def generate_insurance(n=8000, seed=123):
    rng = np.random.default_rng(seed)
    age = rng.uniform(18, 80, n)
    age_group = np.where(age < 35, "young", np.where(age < 55, "middle", "senior"))
    bmi = rng.normal(27, 4, n).clip(15, 50)
    premium = 200 + age * 15 + bmi * 10 + rng.normal(0, 50, n)
    claim_amount = np.zeros(n)
    has_claim = rng.random(n) < 0.15
    claim_amount[has_claim] = rng.lognormal(7, 1.5, has_claim.sum())
    income = rng.lognormal(10.5, 0.6, n)
    missing_prob = np.where(income > np.percentile(income, 70), 0.30, 0.05)
    income[rng.random(n) < missing_prob] = np.nan
    prev_claims = np.zeros(n, dtype=float)
    has_prev = rng.random(n) < 0.30
    prev_claims[has_prev] = rng.poisson(2, has_prev.sum()).astype(float)
    satisfaction = rng.integers(1, 6, n).astype(float)
    noise_mask = rng.random(n) < 0.10
    satisfaction[noise_mask] = rng.integers(1, 6, noise_mask.sum())
    risk_score = premium * 0.3 + age * 0.1
    return pd.DataFrame(
        {
            "claim_amount": claim_amount,
            "premium": premium,
            "age": age,
            "bmi": bmi,
            "income": income,
            "prev_claims": prev_claims,
            "satisfaction": satisfaction,
            "risk_score": risk_score,
            "age_group": age_group,
        }
    )


def main():
    print("Generating Insurance dataset...")
    df = generate_insurance()

    print("Building pipeline...")
    pipeline = UncertaintyPipeline(weights={"missing": 0.35, "anomaly": 0.25, "variance": 0.25})
    pipeline.register("conformal_shift", ConformalShiftDetector(seed=42), weight=0.1)
    pipeline.register("decomposition", UncertaintyDecomposer(n_bootstrap=200, seed=42), weight=0.15)
    pipeline.register("jackknife_plus", JackknifePlusDetector(n_folds=10, seed=42), weight=0.1)
    pipeline.register("mmd_shift", MMDShiftDetector(n_permutations=200, seed=42), weight=0.1)
    pipeline.register("zero_inflation", ZeroInflationDetector(zero_threshold=0.5), weight=0.2)

    print("Running analysis...")
    report = pipeline.analyze(df, group_col="age_group")

    print("Generating HTML report...")
    output = pipeline.generate_report(
        df=df,
        output_path="uncertainty_demo_report.html",
        title="UncertaintyLens — Insurance Data Quality Report",
    )
    print(f"Report saved to: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
