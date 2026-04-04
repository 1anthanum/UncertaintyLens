"""
Tests for UncertaintyLens core modules.

Covers normal operation and edge cases (empty data, all-missing columns,
single-row data, constant features, etc.).
"""

import pandas as pd
import numpy as np
import pytest

from uncertainty_lens.pipeline import UncertaintyPipeline
from uncertainty_lens.detectors import (
    MissingPatternDetector,
    AnomalyDetector,
    VarianceDetector,
)
from uncertainty_lens.quantifiers import MonteCarloQuantifier

# ========== Fixtures ==========


@pytest.fixture
def sample_df():
    """Standard test dataset with known uncertainty characteristics."""
    np.random.seed(42)
    n = 500

    return pd.DataFrame(
        {
            "clean_feature": np.random.normal(100, 5, n),
            "missing_feature": np.where(
                np.random.random(n) > 0.7,
                np.nan,
                np.random.normal(50, 10, n),
            ),
            "anomaly_feature": np.concatenate(
                [
                    np.random.normal(200, 10, n - 20),
                    np.random.normal(500, 50, 20),
                ]
            ),
            "high_variance_feature": np.random.exponential(100, n),
            "channel": np.random.choice(["A", "B", "C", "D"], n),
        }
    )


@pytest.fixture
def edge_case_df():
    """Dataset with extreme edge cases."""
    return pd.DataFrame(
        {
            "all_missing": [np.nan] * 100,
            "constant": [42.0] * 100,
            "single_value_rest_nan": [1.0] + [np.nan] * 99,
            "normal": np.random.normal(0, 1, 100),
            "group": (["A"] * 50) + (["B"] * 50),
        }
    )


@pytest.fixture
def tiny_df():
    """Very small dataset (single row)."""
    return pd.DataFrame({"x": [1.0], "y": [2.0]})


@pytest.fixture
def no_numeric_df():
    """Dataset with no numeric columns."""
    return pd.DataFrame({"a": ["foo", "bar", "baz"], "b": ["x", "y", "z"]})


# ========== MissingPatternDetector ==========


class TestMissingPatternDetector:
    def test_basic_analysis(self, sample_df):
        detector = MissingPatternDetector()
        results = detector.analyze(sample_df)

        assert "missing_rates" in results
        assert "uncertainty_scores" in results
        assert "mcar_test" in results

        # missing_feature should have highest missing rate among numeric cols
        numeric_rates = {k: v for k, v in results["missing_rates"].items() if k != "channel"}
        max_col = max(numeric_rates, key=numeric_rates.get)
        assert max_col == "missing_feature"

    def test_no_missing_values(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        detector = MissingPatternDetector()
        results = detector.analyze(df)

        assert results["summary"]["total_missing"] == 0
        for score in results["uncertainty_scores"].values():
            assert score < 0.1  # near-zero uncertainty

    def test_all_missing_column(self, edge_case_df):
        detector = MissingPatternDetector()
        results = detector.analyze(edge_case_df)

        assert results["missing_rates"]["all_missing"] == 1.0
        assert results["uncertainty_scores"]["all_missing"] > 0.5

    def test_constant_column(self, edge_case_df):
        detector = MissingPatternDetector()
        results = detector.analyze(edge_case_df)

        assert results["missing_rates"]["constant"] == 0.0

    def test_single_row(self, tiny_df):
        detector = MissingPatternDetector()
        results = detector.analyze(tiny_df)

        assert results["summary"]["total_rows"] == 1
        assert "uncertainty_scores" in results


# ========== AnomalyDetector ==========


class TestAnomalyDetector:
    def test_basic_analysis(self, sample_df):
        detector = AnomalyDetector()
        results = detector.analyze(sample_df)

        assert "anomaly_rates" in results
        assert "uncertainty_scores" in results

        # anomaly_feature should have a non-trivial anomaly rate
        assert results["anomaly_rates"]["anomaly_feature"] > 0

    def test_insufficient_data(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        detector = AnomalyDetector()
        results = detector.analyze(df)

        assert "note" in results
        assert "Insufficient" in results["note"]

    def test_constant_data(self):
        df = pd.DataFrame({"x": [5.0] * 50})
        detector = AnomalyDetector()
        results = detector.analyze(df)

        # Constant data should have zero or very low anomaly rate
        if "anomaly_rates" in results and "x" in results["anomaly_rates"]:
            assert results["anomaly_rates"]["x"] < 0.5

    def test_configurable_params(self, sample_df):
        detector = AnomalyDetector(iqr_factor=3.0, contamination=0.01, min_votes=3)
        results = detector.analyze(sample_df)

        # Stricter params => fewer anomalies
        assert "uncertainty_scores" in results

    def test_all_missing_numeric(self):
        df = pd.DataFrame({"a": [np.nan] * 20, "b": [np.nan] * 20})
        detector = AnomalyDetector()
        results = detector.analyze(df)

        assert "note" in results


# ========== VarianceDetector ==========


class TestVarianceDetector:
    def test_basic_analysis(self, sample_df):
        detector = VarianceDetector()
        results = detector.analyze(sample_df, group_col="channel")

        assert "cv_analysis" in results
        assert "variance_decomposition" in results
        assert "uncertainty_scores" in results

    def test_high_cv_detection(self, sample_df):
        detector = VarianceDetector(cv_threshold=0.5)
        results = detector.analyze(sample_df)

        # Exponential distribution has CV=1, should be flagged
        cv_info = results["cv_analysis"]["high_variance_feature"]
        assert cv_info["is_high_variance"] is True

    def test_constant_column(self, edge_case_df):
        detector = VarianceDetector()
        results = detector.analyze(edge_case_df)

        # Constant column has zero variance -> CV is inf or special-cased
        cv_info = results["cv_analysis"]["constant"]
        assert cv_info["cv"] == 0 or cv_info["level"] in ("low", "N/A")

    def test_no_group_col(self, sample_df):
        detector = VarianceDetector()
        results = detector.analyze(sample_df)

        assert "variance_decomposition" not in results

    def test_single_row(self, tiny_df):
        detector = VarianceDetector()
        results = detector.analyze(tiny_df)

        assert "uncertainty_scores" in results


# ========== UncertaintyPipeline ==========


class TestPipeline:
    def test_full_pipeline(self, sample_df):
        pipeline = UncertaintyPipeline()
        report = pipeline.analyze(sample_df, group_col="channel")

        assert "uncertainty_index" in report
        assert "summary" in report
        assert "missing_analysis" in report
        assert "anomaly_analysis" in report
        assert "variance_analysis" in report

        # Composite scores should be between 0 and 1
        for col, vals in report["uncertainty_index"].items():
            assert 0 <= vals["composite_score"] <= 1

    def test_custom_weights(self, sample_df):
        pipeline = UncertaintyPipeline(weights={"missing": 1.0, "anomaly": 0.0, "variance": 0.0})
        report = pipeline.analyze(sample_df)

        # With only missing weight, composite should equal missing score
        for col, vals in report["uncertainty_index"].items():
            assert abs(vals["composite_score"] - vals["missing_score"]) < 0.01

    def test_summary_structure(self, sample_df):
        pipeline = UncertaintyPipeline()
        report = pipeline.analyze(sample_df)
        summary = report["summary"]

        assert "overall_uncertainty" in summary
        assert "overall_level" in summary
        assert "total_features_analyzed" in summary
        assert "high_uncertainty_features" in summary
        assert "low_uncertainty_features" in summary
        assert "top_3_uncertain" in summary
        assert "most_reliable" in summary

    def test_no_numeric_features(self, no_numeric_df):
        pipeline = UncertaintyPipeline()
        with pytest.raises(ValueError, match="no numeric columns"):
            pipeline.analyze(no_numeric_df)

    def test_edge_case_data(self, edge_case_df):
        pipeline = UncertaintyPipeline()
        report = pipeline.analyze(edge_case_df, group_col="group")

        assert "uncertainty_index" in report

    def test_level_labels(self):
        pipeline = UncertaintyPipeline()
        assert pipeline._score_to_level(0.1) == "Low"
        assert pipeline._score_to_level(0.3) == "Medium-Low"
        assert pipeline._score_to_level(0.5) == "Medium"
        assert pipeline._score_to_level(0.7) == "Medium-High"
        assert pipeline._score_to_level(0.9) == "High"


# ========== MonteCarloQuantifier ==========


class TestMonteCarloQuantifier:
    def test_basic_estimation(self, sample_df):
        quantifier = MonteCarloQuantifier(n_simulations=100)
        result = quantifier.estimate(sample_df, statistic_fn=lambda d: d["clean_feature"].mean())

        assert "point_estimate" in result
        assert "confidence_interval_95" in result
        assert result["successful_simulations"] >= 50

    def test_with_missing_data(self, sample_df):
        quantifier = MonteCarloQuantifier(n_simulations=100)
        result = quantifier.estimate(sample_df, statistic_fn=lambda d: d["missing_feature"].mean())

        # CI should be wider due to missing data uncertainty
        ci = result["confidence_interval_95"]
        assert ci[1] > ci[0]

    def test_relative_ci_width(self, sample_df):
        quantifier = MonteCarloQuantifier(n_simulations=100)
        result = quantifier.estimate(sample_df, statistic_fn=lambda d: d["clean_feature"].mean())

        # Clean feature should have narrow CI relative to mean
        assert result["relative_ci_width"] < 1.0

    def test_small_dataset(self, tiny_df):
        quantifier = MonteCarloQuantifier(n_simulations=50)
        result = quantifier.estimate(tiny_df, statistic_fn=lambda d: d["x"].mean())

        assert "point_estimate" in result
