"""Benchmark DeepEnsembleDetector on all 3 datasets."""

import warnings
import time
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from uncertainty_lens import UncertaintyPipeline
from uncertainty_lens.detectors import (
    ConformalShiftDetector,
    UncertaintyDecomposer,
    ConformalPredictor,
    DeepEnsembleDetector,
)
from examples.benchmark_real_data import generate_housing, generate_wine, generate_census

SEP = "-" * 60


def run_with_ensemble(name, df, group_col=None):
    pipeline = UncertaintyPipeline(weights={"missing": 0.30, "anomaly": 0.20, "variance": 0.20})
    pipeline.register("conformal_shift", ConformalShiftDetector(seed=42), weight=0.08)
    pipeline.register("decomposition", UncertaintyDecomposer(n_bootstrap=100, seed=42), weight=0.10)
    pipeline.register("conformal_pred", ConformalPredictor(coverage=0.9, seed=42), weight=0.08)
    pipeline.register(
        "deep_ensemble", DeepEnsembleDetector(n_ensemble=5, max_iter=150, seed=42), weight=0.15
    )

    t0 = time.time()
    report = pipeline.analyze(df, group_col=group_col)
    elapsed = time.time() - t0

    print(f"\n{SEP}")
    print(f"  {name} ({len(df):,} rows, {elapsed:.1f}s)")
    print(SEP)

    # Learnability results
    lea = report.get("deep_ensemble_analysis", {}).get("learnability", {})
    epi = report.get("deep_ensemble_analysis", {}).get("epistemic", {})
    rec = report.get("deep_ensemble_analysis", {}).get("recommendations", {})

    header = f"  {'Feature':22s} {'R2':>6s} {'Disagr':>6s} {'Learn':>5s}  Action"
    print(header)
    print(f"  {'='*22} {'='*6} {'='*6} {'='*5}  {'='*25}")

    for feat in sorted(lea.keys(), key=lambda f: lea[f]["ensemble_r2"], reverse=True):
        l = lea[feat]
        e = epi.get(feat, {})
        r = rec.get(feat, {})
        flag = "Y" if l["is_learnable"] else "N"
        print(
            f"  {feat:22s} {l['ensemble_r2']:6.3f} "
            f"{e.get('mean_disagreement', 0):6.3f} "
            f"  {flag:>3s}  {r.get('action', '?')}"
        )

    return report


def main():
    print("=" * 60)
    print("  DeepEnsemble Feature Learnability Benchmark")
    print("=" * 60)

    run_with_ensemble("Housing", generate_housing(), group_col="region")
    run_with_ensemble("Wine Quality", generate_wine(), group_col="wine_type")
    run_with_ensemble("Census", generate_census(), group_col="sex")

    print(f"\n{SEP}")
    print("  Done.")


if __name__ == "__main__":
    main()
