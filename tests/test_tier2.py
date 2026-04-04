"""
Tests for Tier 2 model-aware detectors:
- ConformalPredictor (split conformal with sklearn models)
- CatBoostUncertainty (RMSEWithUncertainty, optional dependency)
- Pipeline integration
"""

import pandas as pd
import numpy as np
import pytest

from uncertainty_lens.detectors import ConformalPredictor, CatBoostUncertainty
from uncertainty_lens.pipeline import UncertaintyPipeline

# Check if catboost is actually importable (class exists but __init__ checks)
try:
    import catboost  # noqa: F401

    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False


# ========== Fixtures ==========


@pytest.fixture
def regression_df():
    """Dataset with a clear linear signal plus noise."""
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    noise = rng.normal(0, 0.5, n)
    y = 3 * x1 + 2 * x2 + noise

    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


@pytest.fixture
def noisy_regression_df():
    """Dataset with very high noise — wide prediction intervals expected."""
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.normal(0, 1, n)
    noise = rng.normal(0, 10, n)  # 20x the signal
    y = 0.5 * x1 + noise

    return pd.DataFrame({"x1": x1, "target": y})


@pytest.fixture
def multi_feature_df():
    """Multi-feature dataset for leave-one-out mode."""
    rng = np.random.default_rng(42)
    n = 300
    return pd.DataFrame(
        {
            "a": rng.normal(100, 10, n),
            "b": rng.normal(50, 5, n),
            "c": rng.normal(0, 1, n),
            "d": rng.exponential(20, n),
        }
    )


# ========== ConformalPredictor ==========


class TestConformalPredictor:
    def test_single_target(self, regression_df):
        detector = ConformalPredictor(target_col="target", coverage=0.9)
        results = detector.analyze(regression_df)

        assert "uncertainty_scores" in results
        assert "conformal_results" in results
        assert "target" in results["conformal_results"]

        cr = results["conformal_results"]["target"]
        # Empirical coverage should be close to target
        assert cr["empirical_coverage"] >= 0.8  # allow some slack

        # Interval width should be positive
        assert cr["interval_width"] > 0

    def test_leave_one_out_columns(self, multi_feature_df):
        """Without target_col, each column takes a turn."""
        detector = ConformalPredictor(coverage=0.9)
        results = detector.analyze(multi_feature_df)

        # Should have results for all 4 features
        assert len(results["conformal_results"]) == 4
        for col in ["a", "b", "c", "d"]:
            assert col in results["uncertainty_scores"]

    def test_noisy_data_higher_score(self, regression_df, noisy_regression_df):
        """Noisier data should produce higher uncertainty scores."""
        detector = ConformalPredictor(target_col="target", coverage=0.9)

        r_clean = detector.analyze(regression_df)
        r_noisy = detector.analyze(noisy_regression_df)

        clean_score = r_clean["uncertainty_scores"]["target"]
        noisy_score = r_noisy["uncertainty_scores"]["target"]

        assert noisy_score > clean_score

    def test_coverage_parameter(self, regression_df):
        """Higher coverage → wider intervals."""
        det_90 = ConformalPredictor(target_col="target", coverage=0.9)
        det_99 = ConformalPredictor(target_col="target", coverage=0.99)

        r_90 = det_90.analyze(regression_df)
        r_99 = det_99.analyze(regression_df)

        width_90 = r_90["conformal_results"]["target"]["interval_width"]
        width_99 = r_99["conformal_results"]["target"]["interval_width"]

        assert width_99 > width_90

    def test_custom_model(self, regression_df):
        """Test with a custom sklearn model."""
        from sklearn.ensemble import GradientBoostingRegressor

        model = GradientBoostingRegressor(n_estimators=50, max_depth=3)
        detector = ConformalPredictor(model=model, target_col="target", coverage=0.9)
        results = detector.analyze(regression_df)

        assert results["conformal_results"]["target"]["empirical_coverage"] >= 0.8

    def test_input_validation(self):
        detector = ConformalPredictor()
        with pytest.raises(TypeError):
            detector.analyze("not a dataframe")
        with pytest.raises(ValueError, match="empty"):
            detector.analyze(pd.DataFrame())

    def test_constructor_validation(self):
        with pytest.raises(ValueError):
            ConformalPredictor(coverage=0)
        with pytest.raises(ValueError):
            ConformalPredictor(calibration_fraction=1.5)

    def test_insufficient_data(self):
        """Tiny dataset should return gracefully."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        detector = ConformalPredictor()
        results = detector.analyze(df)

        assert "note" in results
        assert "Insufficient" in results["note"]

    def test_single_numeric_column(self):
        """Only one numeric column — can't do conformal."""
        df = pd.DataFrame({"x": np.arange(100, dtype=float)})
        detector = ConformalPredictor()
        results = detector.analyze(df)

        assert "note" in results

    def test_with_missing_values(self):
        """Missing values should be handled (rows dropped)."""
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame(
            {
                "x": rng.normal(0, 1, n),
                "y": rng.normal(0, 1, n),
            }
        )
        # Add some NaNs
        df.loc[rng.choice(n, 20, replace=False), "x"] = np.nan

        detector = ConformalPredictor()
        results = detector.analyze(df)
        assert "uncertainty_scores" in results


# ========== CatBoostUncertainty ==========


@pytest.mark.skipif(not _HAS_CATBOOST, reason="catboost not installed")
class TestCatBoostUncertainty:
    def test_single_target(self, regression_df):
        detector = CatBoostUncertainty(target_col="target", iterations=50)
        results = detector.analyze(regression_df)

        assert "uncertainty_scores" in results
        assert "feature_results" in results
        assert "target" in results["feature_results"]

        fr = results["feature_results"]["target"]
        assert fr["mean_predicted_variance"] > 0
        assert fr["uncertainty_score"] >= 0

    def test_leave_one_out(self, multi_feature_df):
        detector = CatBoostUncertainty(iterations=50)
        results = detector.analyze(multi_feature_df)

        assert len(results["feature_results"]) == 4

    def test_noisy_higher_uncertainty(self, regression_df, noisy_regression_df):
        detector = CatBoostUncertainty(target_col="target", iterations=100)

        r_clean = detector.analyze(regression_df)
        r_noisy = detector.analyze(noisy_regression_df)

        assert (
            r_noisy["feature_results"]["target"]["mean_predicted_variance"]
            > r_clean["feature_results"]["target"]["mean_predicted_variance"]
        )

    def test_input_validation(self):
        detector = CatBoostUncertainty()
        with pytest.raises(TypeError):
            detector.analyze("bad")
        with pytest.raises(ValueError, match="empty"):
            detector.analyze(pd.DataFrame())


class TestCatBoostOptionalImport:
    """Test that CatBoost import failure is handled cleanly."""

    def test_instantiation_raises_without_catboost(self):
        """Instantiating CatBoostUncertainty should raise ImportError."""
        if _HAS_CATBOOST:
            pytest.skip("catboost is installed")

        from uncertainty_lens.detectors.catboost_uncertainty import (
            CatBoostUncertainty as CB,
        )

        with pytest.raises(ImportError, match="catboost"):
            CB()


# ========== Pipeline Integration ==========


class TestPipelineWithTier2:
    def test_register_conformal_predictor(self, regression_df):
        pipe = UncertaintyPipeline()
        pipe.register(
            "conformal_pred",
            ConformalPredictor(target_col="target"),
            weight=0.2,
        )

        report = pipe.analyze(regression_df)

        assert "conformal_pred_analysis" in report
        assert "conformal_pred_score" in list(report["uncertainty_index"].values())[0]

    def test_full_pipeline_with_tier1_and_tier2(self, multi_feature_df):
        """All detectors together: 3 built-in + 2 Tier 1 + 1 Tier 2."""
        from uncertainty_lens.detectors import (
            ConformalShiftDetector,
            UncertaintyDecomposer,
        )

        df = multi_feature_df.copy()
        rng = np.random.default_rng(42)
        df["group"] = rng.choice(["X", "Y"], len(df))

        pipe = UncertaintyPipeline()
        pipe.register("conformal_shift", ConformalShiftDetector(), weight=0.1)
        pipe.register("decomposition", UncertaintyDecomposer(n_bootstrap=50), weight=0.1)
        pipe.register("conformal_pred", ConformalPredictor(), weight=0.1)

        assert len(pipe.registered_detectors) == 6

        report = pipe.analyze(df, group_col="group")

        # All analysis keys present
        for name in [
            "missing",
            "anomaly",
            "variance",
            "conformal_shift",
            "decomposition",
            "conformal_pred",
        ]:
            assert f"{name}_analysis" in report

        # Composite scores bounded
        for vals in report["uncertainty_index"].values():
            assert 0 <= vals["composite_score"] <= 1.0

    @pytest.mark.skipif(not _HAS_CATBOOST, reason="catboost not installed")
    def test_catboost_in_pipeline(self, regression_df):
        pipe = UncertaintyPipeline()
        pipe.register(
            "catboost",
            CatBoostUncertainty(target_col="target", iterations=50),
            weight=0.15,
        )

        report = pipe.analyze(regression_df)
        assert "catboost_analysis" in report
