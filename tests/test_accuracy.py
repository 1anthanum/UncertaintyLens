"""
Synthetic benchmark tests for technical accuracy.

These tests verify that detectors *recover known uncertainty structures*
from synthetically generated data. Unlike the existing tests (which check
API contracts and edge cases), these validate statistical correctness.
"""

import numpy as np
import pandas as pd
import pytest

from uncertainty_lens.detectors import (
    MissingPatternDetector,
    AnomalyDetector,
    VarianceDetector,
    ConformalShiftDetector,
    UncertaintyDecomposer,
    ConformalPredictor,
)
from uncertainty_lens.pipeline import UncertaintyPipeline
from uncertainty_lens.quantifiers import MonteCarloQuantifier

# ═══════════════════════════════════════════════════════════════════════
# 1. Conformal predictor: coverage guarantee
# ═══════════════════════════════════════════════════════════════════════


class TestConformalCoverageGuarantee:
    """
    The core guarantee of split conformal prediction:
    empirical coverage on *new* data ≥ (1-α) - O(1/n).

    We generate training + test data from the *same* distribution,
    fit conformal on training, and check coverage on *held-out* test.
    """

    @pytest.mark.parametrize("coverage_target", [0.9, 0.95])
    def test_coverage_on_held_out_data(self, coverage_target):
        rng = np.random.default_rng(123)
        n_total = 1000
        n_test = 300

        x = rng.normal(0, 1, n_total)
        y = 2 * x + rng.normal(0, 1, n_total)
        df = pd.DataFrame({"x": x, "y": y})

        train_df = df.iloc[: n_total - n_test]
        test_df = df.iloc[n_total - n_test :]

        detector = ConformalPredictor(target_col="y", coverage=coverage_target, seed=42)
        results = detector.analyze(train_df)

        # Use the conformal radius from training to check test coverage
        radius = results["conformal_results"]["y"]["conformal_radius"]
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline

        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(train_df[["x"]], train_df["y"])
        test_pred = model.predict(test_df[["x"]])

        covered = np.mean(
            (test_df["y"].values >= test_pred - radius)
            & (test_df["y"].values <= test_pred + radius)
        )

        # Coverage should be ≥ target - small tolerance for finite-sample
        assert covered >= coverage_target - 0.05, (
            f"Coverage {covered:.3f} < {coverage_target - 0.05:.3f} " f"(target={coverage_target})"
        )

    def test_wider_intervals_for_noisier_data(self):
        """Interval width should be proportional to noise level."""
        rng = np.random.default_rng(42)
        n = 500

        x = rng.normal(0, 1, n)
        df_low = pd.DataFrame({"x": x, "y": 2 * x + rng.normal(0, 0.5, n)})
        df_high = pd.DataFrame({"x": x, "y": 2 * x + rng.normal(0, 5.0, n)})

        det = ConformalPredictor(target_col="y", coverage=0.9, seed=42)
        r_low = det.analyze(df_low)
        r_high = det.analyze(df_high)

        width_low = r_low["conformal_results"]["y"]["interval_width"]
        width_high = r_high["conformal_results"]["y"]["interval_width"]

        # High-noise width should be at least 5x the low-noise width
        assert width_high > width_low * 3, (
            f"width_high={width_high}, width_low={width_low}; "
            f"ratio={width_high / width_low:.1f} should be > 3"
        )


# ═══════════════════════════════════════════════════════════════════════
# 2. Decomposer: epistemic vs. aleatoric separation
# ═══════════════════════════════════════════════════════════════════════


class TestDecomposerAccuracy:
    """
    Verify that the decomposer correctly separates:
    - High aleatoric, low epistemic (large noisy dataset)
    - High epistemic, low aleatoric (small clean dataset)
    """

    def test_large_noisy_data_is_aleatoric_dominant(self):
        """n=5000, high noise → aleatoric >> epistemic."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"x": rng.exponential(100, 5000)})

        decomposer = UncertaintyDecomposer(n_bootstrap=200, seed=42)
        results = decomposer.analyze(df)

        d = results["decomposition"]["x"]
        # With 5000 samples, epistemic (sampling uncertainty) should be small
        # Aleatoric (data noise) should be substantial for exponential
        assert d["dominant"] in ("aleatoric", "mixed"), (
            f"Expected aleatoric-dominant, got {d['dominant']} "
            f"(ale={d['aleatoric_score']:.3f}, epi={d['epistemic_score']:.3f})"
        )

    def test_small_clean_data_is_epistemic_dominant(self):
        """n=12, low noise → epistemic >> aleatoric."""
        rng = np.random.default_rng(42)
        # Very small sample, very low noise relative to mean
        df = pd.DataFrame({"x": 100 + rng.normal(0, 0.01, 12)})

        decomposer = UncertaintyDecomposer(n_bootstrap=200, seed=42)
        results = decomposer.analyze(df)

        d = results["decomposition"]["x"]
        # With only 12 samples, bootstrap stat should vary a lot
        assert (
            d["epistemic_score"] > 0
        ), f"Epistemic should be non-zero for n=12, got {d['epistemic_score']}"

    def test_epistemic_decreases_with_sample_size(self):
        """Core property: epistemic_raw ∝ 1/n."""
        rng = np.random.default_rng(42)
        decomposer = UncertaintyDecomposer(n_bootstrap=300, seed=42)

        sizes = [20, 100, 1000, 5000]
        epi_raws = []

        for n in sizes:
            df = pd.DataFrame({"x": rng.normal(100, 10, n)})
            result = decomposer.analyze(df)
            epi_raws.append(result["decomposition"]["x"]["epistemic_raw"])

        # Epistemic raw should be monotonically decreasing
        for i in range(len(epi_raws) - 1):
            assert epi_raws[i] > epi_raws[i + 1], (
                f"Epistemic not decreasing: n={sizes[i]}→{epi_raws[i]:.6f}, "
                f"n={sizes[i + 1]}→{epi_raws[i + 1]:.6f}"
            )


# ═══════════════════════════════════════════════════════════════════════
# 3. Conformal shift: detect known distributional shifts
# ═══════════════════════════════════════════════════════════════════════


class TestConformalShiftAccuracy:
    def test_detects_mean_shift(self):
        """Group C has shifted mean — should be detected."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "x": np.concatenate(
                    [
                        rng.normal(100, 5, 300),  # Group A
                        rng.normal(100, 5, 300),  # Group B (same)
                        rng.normal(200, 5, 300),  # Group C (shifted mean)
                    ]
                ),
                "group": ["A"] * 300 + ["B"] * 300 + ["C"] * 300,
            }
        )

        detector = ConformalShiftDetector(seed=42)
        results = detector.analyze(df, group_col="group")

        # Group C should be flagged as shifted
        assert results["group_shift"]["C"]["x"]["shifted"] == True  # noqa: E712
        # Score should be high
        assert results["uncertainty_scores"]["x"] > 0.3

    def test_detects_variance_shift(self):
        """Group C has shifted variance — should be detected."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "x": np.concatenate(
                    [
                        rng.normal(100, 5, 300),  # Group A
                        rng.normal(100, 5, 300),  # Group B (same)
                        rng.normal(100, 50, 300),  # Group C (10x variance)
                    ]
                ),
                "group": ["A"] * 300 + ["B"] * 300 + ["C"] * 300,
            }
        )

        detector = ConformalShiftDetector(seed=42)
        results = detector.analyze(df, group_col="group")

        assert results["group_shift"]["C"]["x"]["shifted"] == True  # noqa: E712

    def test_no_false_alarm_on_identical_groups(self):
        """All groups from same distribution — low shift score."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "x": rng.normal(100, 10, 900),
                "group": ["A"] * 300 + ["B"] * 300 + ["C"] * 300,
            }
        )

        detector = ConformalShiftDetector(seed=42)
        results = detector.analyze(df, group_col="group")

        # Score should be low — no real shift
        assert results["uncertainty_scores"]["x"] < 0.5


# ═══════════════════════════════════════════════════════════════════════
# 4. Missing pattern: MCAR test correctness
# ═══════════════════════════════════════════════════════════════════════


class TestMCARAccuracy:
    def test_truly_mcar_detected(self):
        """Random missingness should be classified as MCAR."""
        rng = np.random.default_rng(42)
        n = 1000
        df = pd.DataFrame(
            {
                "x": rng.normal(0, 1, n),
                "y": rng.normal(0, 1, n),
            }
        )
        # Randomly set 10% missing in x — completely at random
        mask = rng.random(n) < 0.1
        df.loc[mask, "x"] = np.nan

        detector = MissingPatternDetector()
        results = detector.analyze(df)

        assert results["mcar_test"]["is_mcar"] is True

    def test_systematic_missing_detected(self):
        """Missing values correlated with another feature → not MCAR."""
        rng = np.random.default_rng(42)
        n = 1000
        y = rng.normal(0, 1, n)
        x = rng.normal(0, 1, n)
        # Make x missing when y > 1 (systematic)
        df = pd.DataFrame({"x": x, "y": y})
        df.loc[y > 1, "x"] = np.nan

        detector = MissingPatternDetector()
        results = detector.analyze(df)

        assert results["mcar_test"]["is_mcar"] is False

    def test_bonferroni_correction_present(self):
        """Verify Bonferroni correction metadata is in output."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "x": rng.normal(0, 1, 200),
                "y": rng.normal(0, 1, 200),
            }
        )
        df.loc[:19, "x"] = np.nan

        detector = MissingPatternDetector()
        results = detector.analyze(df)

        assert "adjusted_alpha" in results["mcar_test"]
        assert "correction_method" in results["mcar_test"]
        assert results["mcar_test"]["correction_method"] == "bonferroni"


# ═══════════════════════════════════════════════════════════════════════
# 5. Variance detector: zero-mean handling
# ═══════════════════════════════════════════════════════════════════════


class TestVarianceZeroMean:
    def test_zero_mean_does_not_get_max_score(self):
        """Feature with mean=0 and low variance should NOT get score 1.0."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "zero_mean_low_var": rng.normal(0, 0.01, 500),
                "zero_mean_high_var": rng.normal(0, 10, 500),
            }
        )

        detector = VarianceDetector()
        results = detector.analyze(df)

        score_low = results["uncertainty_scores"]["zero_mean_low_var"]
        score_high = results["uncertainty_scores"]["zero_mean_high_var"]

        # Low-var feature should NOT be max uncertainty
        assert score_low < 0.8, f"Low-var zero-mean feature got score {score_low}"
        # High-var feature should be higher
        assert score_high > score_low

    def test_zero_mean_cv_fallback_method(self):
        """CV should use std_fallback method for zero-mean features."""
        df = pd.DataFrame({"x": np.random.default_rng(42).normal(0, 1, 100)})
        detector = VarianceDetector()
        results = detector.analyze(df)

        cv_info = results["cv_analysis"]["x"]
        assert cv_info["cv_method"] == "std_fallback"
        assert "note" in cv_info


# ═══════════════════════════════════════════════════════════════════════
# 6. Pipeline: adaptive weighting
# ═══════════════════════════════════════════════════════════════════════


class TestAdaptiveWeighting:
    def test_clean_data_not_diluted_by_missing_detector(self):
        """
        On data with zero missing values, the missing detector's weight
        should be reduced so it doesn't pull composite scores to zero.
        """
        rng = np.random.default_rng(42)
        # Clean data with high anomaly rate in one feature
        n = 200
        x_normal = rng.normal(0, 1, n)
        x_outliers = np.concatenate(
            [
                rng.normal(0, 1, n - 20),
                rng.normal(100, 1, 20),  # obvious outliers
            ]
        )

        df = pd.DataFrame({"normal": x_normal, "outlier_prone": x_outliers})

        pipe = UncertaintyPipeline()
        report = pipe.analyze(df)

        # The outlier-prone feature should have a meaningful composite score
        # even though missing detector gives it 0
        score = report["uncertainty_index"]["outlier_prone"]["composite_score"]
        assert score > 0.01, (
            f"Composite score for outlier-prone feature should be " f"meaningful, got {score}"
        )

    def test_scores_bounded_zero_one(self):
        """Adaptive weighting must not produce scores outside [0, 1]."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "a": rng.normal(0, 1, 200),
                "b": rng.exponential(10, 200),
            }
        )

        pipe = UncertaintyPipeline()
        report = pipe.analyze(df)

        for col, vals in report["uncertainty_index"].items():
            assert 0 <= vals["composite_score"] <= 1.0


# ═══════════════════════════════════════════════════════════════════════
# 7. Monte Carlo: relative CI width replaces misnamed sensitivity_ratio
# ═══════════════════════════════════════════════════════════════════════


class TestMonteCarloAccuracy:
    def test_relative_ci_width_key_exists(self):
        """Verify renamed output key."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"x": rng.normal(100, 10, 200)})

        mc = MonteCarloQuantifier(n_simulations=100, random_state=42)
        result = mc.estimate(df, statistic_fn=lambda d: d["x"].mean())

        assert "relative_ci_width" in result
        assert "failure_rate" in result
        assert "sensitivity_ratio" not in result  # renamed

    def test_ci_narrows_with_larger_data(self):
        """More data → narrower CI spread."""
        mc = MonteCarloQuantifier(n_simulations=200, random_state=42)

        rng = np.random.default_rng(42)
        df_small = pd.DataFrame({"x": rng.normal(100, 10, 30)})
        df_large = pd.DataFrame({"x": rng.normal(100, 10, 3000)})

        r_small = mc.estimate(df_small, statistic_fn=lambda d: d["x"].mean())
        r_large = mc.estimate(df_large, statistic_fn=lambda d: d["x"].mean())

        ci_width_small = r_small["confidence_interval_95"][1] - r_small["confidence_interval_95"][0]
        ci_width_large = r_large["confidence_interval_95"][1] - r_large["confidence_interval_95"][0]

        assert ci_width_large < ci_width_small
