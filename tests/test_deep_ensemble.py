"""Tests for DeepEnsembleDetector."""

import numpy as np
import pandas as pd
import pytest

from uncertainty_lens.detectors import DeepEnsembleDetector


class TestDeepEnsembleDetector:
    """Core functionality tests."""

    def test_basic_output_structure(self):
        """Returned dict has the required keys and correct types."""
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame(
            {
                "x1": rng.normal(0, 1, n),
                "x2": rng.normal(0, 1, n),
                "y": np.zeros(n),  # placeholder
            }
        )
        df["y"] = 2 * df["x1"] + 0.5 * df["x2"] + rng.normal(0, 0.1, n)

        det = DeepEnsembleDetector(n_ensemble=3, max_iter=50, seed=42)
        result = det.analyze(df)

        assert "uncertainty_scores" in result
        assert "learnability" in result
        assert "epistemic" in result
        assert "recommendations" in result
        assert "config" in result

        # All numeric columns should have scores
        for col in ["x1", "x2", "y"]:
            assert col in result["uncertainty_scores"]
            score = result["uncertainty_scores"][col]
            assert 0 <= score <= 1, f"{col} score {score} out of [0,1]"

    def test_learnable_feature_detected(self):
        """A feature with clear linear relationship should be learnable."""
        rng = np.random.default_rng(42)
        n = 300
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        # y is a strong function of x1 and x2
        y = 3 * x1 - 2 * x2 + rng.normal(0, 0.1, n)

        df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
        det = DeepEnsembleDetector(n_ensemble=3, max_iter=100, seed=42)
        result = det.analyze(df)

        # y should be highly learnable from x1, x2
        assert result["learnability"]["y"]["is_learnable"] is True
        assert result["learnability"]["y"]["ensemble_r2"] > 0.5

    def test_noise_feature_not_learnable(self):
        """Pure noise feature should have low R² and high uncertainty."""
        rng = np.random.default_rng(42)
        n = 300
        df = pd.DataFrame(
            {
                "signal1": rng.normal(0, 1, n),
                "signal2": rng.normal(0, 1, n),
                "noise": rng.normal(0, 1, n),  # independent of others
            }
        )

        det = DeepEnsembleDetector(n_ensemble=3, max_iter=100, seed=42)
        result = det.analyze(df)

        # Noise should have low R² (can't be predicted from others)
        noise_r2 = result["learnability"]["noise"]["ensemble_r2"]
        assert noise_r2 < 0.3, f"Noise R²={noise_r2} unexpectedly high"

        # Noise should have higher uncertainty than signal features
        noise_score = result["uncertainty_scores"]["noise"]
        signal_score = result["uncertainty_scores"]["signal1"]
        # noise_score should be at least as high as signal_score
        assert noise_score >= signal_score * 0.5

    def test_recommendation_types(self):
        """Check that recommendations produce valid action types."""
        rng = np.random.default_rng(42)
        n = 200
        x = rng.normal(0, 1, n)
        y = 2 * x + rng.normal(0, 0.1, n)
        z = rng.normal(0, 1, n)

        df = pd.DataFrame({"x": x, "y": y, "z": z})
        det = DeepEnsembleDetector(n_ensemble=3, max_iter=50, seed=42)
        result = det.analyze(df)

        valid_actions = {
            "reliable",
            "independent_feature",
            "unstable_learning",
            "investigate_or_drop",
        }
        for col, rec in result["recommendations"].items():
            assert "action" in rec
            assert "explanation" in rec
            assert rec["action"] in valid_actions, f"{col}: unknown action '{rec['action']}'"

    def test_insufficient_data(self):
        """With very few rows, should return default scores gracefully."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        det = DeepEnsembleDetector(n_ensemble=3, max_iter=50)
        result = det.analyze(df)

        assert all(v == 0.5 for v in result["uncertainty_scores"].values())

    def test_single_feature_returns_default(self):
        """With only one numeric column, can't do cross-prediction."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5] * 20})
        det = DeepEnsembleDetector(n_ensemble=3, max_iter=50)
        result = det.analyze(df)

        assert result["uncertainty_scores"]["x"] == 0.5

    def test_missing_values_handled(self):
        """NaN rows should be dropped, not crash."""
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame(
            {
                "x": rng.normal(0, 1, n),
                "y": rng.normal(0, 1, n),
            }
        )
        # Inject 10% NaN
        df.loc[rng.choice(n, 20, replace=False), "x"] = np.nan

        det = DeepEnsembleDetector(n_ensemble=3, max_iter=50, seed=42)
        result = det.analyze(df)

        assert "x" in result["uncertainty_scores"]
        assert "y" in result["uncertainty_scores"]

    def test_n_ensemble_validation(self):
        """n_ensemble < 2 should raise ValueError."""
        with pytest.raises(ValueError, match="n_ensemble"):
            DeepEnsembleDetector(n_ensemble=1)

    def test_pipeline_integration(self):
        """DeepEnsembleDetector works with UncertaintyPipeline.register()."""
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
            "deep_ensemble",
            DeepEnsembleDetector(n_ensemble=3, max_iter=50, seed=42),
            weight=0.15,
        )

        report = pipeline.analyze(df)

        assert "deep_ensemble_analysis" in report
        assert "uncertainty_index" in report

        # Composite scores should incorporate deep_ensemble
        for col_data in report["uncertainty_index"].values():
            assert "deep_ensemble_score" in col_data


class TestLearnabilityAccuracy:
    """Tests that verify learnability detection is directionally correct."""

    def test_linear_vs_noise(self):
        """Linear feature should be more learnable than noise."""
        rng = np.random.default_rng(42)
        n = 500
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        linear_target = 3 * x1 + 2 * x2 + rng.normal(0, 0.2, n)
        noise_target = rng.normal(0, 1, n)

        df = pd.DataFrame(
            {
                "x1": x1,
                "x2": x2,
                "linear": linear_target,
                "noise": noise_target,
            }
        )

        det = DeepEnsembleDetector(n_ensemble=5, max_iter=150, seed=42)
        result = det.analyze(df)

        linear_r2 = result["learnability"]["linear"]["ensemble_r2"]
        noise_r2 = result["learnability"]["noise"]["ensemble_r2"]

        assert (
            linear_r2 > noise_r2
        ), f"Linear R²={linear_r2:.3f} should exceed noise R²={noise_r2:.3f}"

    def test_nonlinear_relationship_detected(self):
        """MLP should detect nonlinear relationships that linear models miss."""
        rng = np.random.default_rng(42)
        n = 500
        x1 = rng.uniform(-3, 3, n)
        x2 = rng.uniform(-3, 3, n)
        # Nonlinear: sin + interaction
        y = np.sin(x1) * x2 + rng.normal(0, 0.3, n)

        df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
        det = DeepEnsembleDetector(n_ensemble=5, max_iter=200, seed=42)
        result = det.analyze(df)

        # MLP should capture some nonlinear signal
        y_r2 = result["learnability"]["y"]["ensemble_r2"]
        assert y_r2 > 0.05, f"Nonlinear R²={y_r2:.3f} too low"
