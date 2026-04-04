"""Tests for JackknifePlusDetector (CV+ prediction intervals)."""

import numpy as np
import pandas as pd
import pytest

from uncertainty_lens.detectors import JackknifePlusDetector


class TestJackknifePlusBasic:
    """Core functionality tests."""

    def test_output_structure(self):
        """Returned dict has required keys and correct types."""
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame(
            {
                "x1": rng.normal(0, 1, n),
                "x2": rng.normal(0, 1, n),
                "y": np.zeros(n),
            }
        )
        df["y"] = 2 * df["x1"] + 0.5 * df["x2"] + rng.normal(0, 0.1, n)

        det = JackknifePlusDetector(n_folds=5, coverage=0.9, seed=42)
        result = det.analyze(df)

        assert "uncertainty_scores" in result
        assert "prediction_intervals" in result
        assert "method" in result
        assert result["method"] == "cv_plus_jackknife"
        assert "comparison_to_split_conformal" in result

        for col in ["x1", "x2", "y"]:
            assert col in result["uncertainty_scores"]
            score = result["uncertainty_scores"][col]
            assert 0 <= score <= 1, f"{col} score {score} out of [0,1]"

    def test_interval_statistics_present(self):
        """Each target should have width distribution and residual stats."""
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame(
            {
                "x": rng.normal(0, 1, n),
                "y": rng.normal(0, 1, n),
            }
        )

        det = JackknifePlusDetector(n_folds=5, seed=42)
        result = det.analyze(df)

        for target, info in result["prediction_intervals"].items():
            assert "conformal_radius" in info
            assert "interval_width_mean" in info
            assert "adaptivity_ratio" in info
            assert "empirical_coverage" in info
            assert "loo_r2" in info
            assert "width_distribution" in info
            assert "residual_stats" in info

    def test_coverage_achieved(self):
        """Empirical coverage should be close to target for clean data."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.normal(0, 1, n)
        y = 2 * x + rng.normal(0, 0.5, n)
        df = pd.DataFrame({"x": x, "y": y})

        det = JackknifePlusDetector(n_folds=10, coverage=0.9, seed=42)
        result = det.analyze(df)

        # CV+ has guarantee P >= 1 - 2α, so for α=0.1,
        # coverage should be >= 0.8 in theory, typically > 0.85
        for target, info in result["prediction_intervals"].items():
            cov = info["empirical_coverage"]
            assert cov >= 0.75, f"{target}: coverage {cov:.3f} too low for 90% target"

    def test_learnable_vs_noise_intervals(self):
        """Learnable target should have tighter intervals than noise."""
        rng = np.random.default_rng(42)
        n = 500
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        learnable = 3 * x1 + 2 * x2 + rng.normal(0, 0.2, n)
        noise = rng.normal(0, 1, n)

        df = pd.DataFrame(
            {
                "x1": x1,
                "x2": x2,
                "learnable": learnable,
                "noise": noise,
            }
        )

        det = JackknifePlusDetector(n_folds=10, seed=42)
        result = det.analyze(df)

        learn_width = result["prediction_intervals"]["learnable"]["normalized_width"]
        noise_width = result["prediction_intervals"]["noise"]["normalized_width"]

        assert learn_width < noise_width, (
            f"Learnable width ({learn_width:.3f}) should be less than "
            f"noise width ({noise_width:.3f})"
        )

    def test_learnable_higher_r2(self):
        """Learnable target should have higher LOO R² than noise."""
        rng = np.random.default_rng(42)
        n = 500
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        learnable = 3 * x1 + 2 * x2 + rng.normal(0, 0.2, n)
        noise = rng.normal(0, 1, n)

        df = pd.DataFrame(
            {
                "x1": x1,
                "x2": x2,
                "learnable": learnable,
                "noise": noise,
            }
        )

        det = JackknifePlusDetector(n_folds=10, seed=42)
        result = det.analyze(df)

        learn_r2 = result["prediction_intervals"]["learnable"]["loo_r2"]
        noise_r2 = result["prediction_intervals"]["noise"]["loo_r2"]

        assert (
            learn_r2 > noise_r2
        ), f"Learnable R²={learn_r2:.3f} should exceed noise R²={noise_r2:.3f}"


class TestJackknifePlusEdgeCases:
    """Edge case and validation tests."""

    def test_insufficient_data(self):
        """With very few rows, should return default scores."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        det = JackknifePlusDetector(n_folds=5, seed=42)
        result = det.analyze(df)

        # < 20 rows → default
        assert all(v == 0.5 for v in result["uncertainty_scores"].values())

    def test_single_feature(self):
        """Single feature → default scores."""
        df = pd.DataFrame({"x": np.arange(100)})
        det = JackknifePlusDetector(n_folds=5, seed=42)
        result = det.analyze(df)

        assert result["uncertainty_scores"]["x"] == 0.5

    def test_missing_values_handled(self):
        """NaN rows should be dropped without crashing."""
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame(
            {
                "x": rng.normal(0, 1, n),
                "y": rng.normal(0, 1, n),
            }
        )
        df.loc[rng.choice(n, 20, replace=False), "x"] = np.nan

        det = JackknifePlusDetector(n_folds=5, seed=42)
        result = det.analyze(df)

        assert "x" in result["uncertainty_scores"]
        assert "y" in result["uncertainty_scores"]

    def test_invalid_coverage(self):
        """coverage outside (0,1) should raise ValueError."""
        with pytest.raises(ValueError, match="coverage"):
            JackknifePlusDetector(coverage=0.0)
        with pytest.raises(ValueError, match="coverage"):
            JackknifePlusDetector(coverage=1.0)

    def test_invalid_n_folds(self):
        """n_folds < 2 should raise ValueError."""
        with pytest.raises(ValueError, match="n_folds"):
            JackknifePlusDetector(n_folds=1)

    def test_empty_dataframe(self):
        """Empty DataFrame should raise ValueError."""
        det = JackknifePlusDetector()
        with pytest.raises(ValueError, match="empty"):
            det.analyze(pd.DataFrame())

    def test_non_dataframe(self):
        """Non-DataFrame input should raise TypeError."""
        det = JackknifePlusDetector()
        with pytest.raises(TypeError, match="DataFrame"):
            det.analyze(np.array([[1, 2], [3, 4]]))


class TestJackknifePlusIntegration:
    """Integration with UncertaintyPipeline."""

    def test_pipeline_registration(self):
        """JackknifePlusDetector works with pipeline.register()."""
        from uncertainty_lens import UncertaintyPipeline

        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame(
            {
                "a": rng.normal(0, 1, n),
                "b": rng.normal(0, 1, n),
                "c": rng.normal(0, 1, n),
            }
        )

        pipeline = UncertaintyPipeline()
        pipeline.register(
            "jackknife_plus",
            JackknifePlusDetector(n_folds=5, seed=42),
            weight=0.15,
        )

        report = pipeline.analyze(df)

        assert "jackknife_plus_analysis" in report
        assert "uncertainty_index" in report

        for col_data in report["uncertainty_index"].values():
            assert "jackknife_plus_score" in col_data

    def test_replaces_split_conformal(self):
        """Can replace split conformal in the pipeline."""
        from uncertainty_lens import UncertaintyPipeline

        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame(
            {
                "a": rng.normal(0, 1, n),
                "b": rng.normal(0, 1, n),
            }
        )

        pipeline = UncertaintyPipeline()
        # Replace split conformal with Jackknife+
        pipeline.unregister("missing")
        pipeline.unregister("anomaly")
        pipeline.unregister("variance")
        pipeline.register("jackknife_plus", JackknifePlusDetector(seed=42), weight=1.0)

        report = pipeline.analyze(df)
        assert "jackknife_plus_analysis" in report

    def test_comparison_output(self):
        """Comparison to split conformal is present."""
        rng = np.random.default_rng(42)
        n = 300
        df = pd.DataFrame(
            {
                "x": rng.normal(0, 1, n),
                "y": rng.normal(0, 1, n),
            }
        )

        det = JackknifePlusDetector(n_folds=10, seed=42)
        result = det.analyze(df)

        comp = result["comparison_to_split_conformal"]
        assert "mean_normalized_width" in comp
        assert "mean_empirical_coverage" in comp
        assert "advantage_over_split" in comp
