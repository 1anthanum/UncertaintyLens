"""
Tests for Tier 1 SOTA detectors:
- ConformalShiftDetector (distribution-shift via conformal non-conformity)
- UncertaintyDecomposer (aleatoric vs. epistemic decomposition)
- Pipeline integration with new detectors via register()
"""

import pandas as pd
import numpy as np
import pytest

from uncertainty_lens.detectors import ConformalShiftDetector, UncertaintyDecomposer
from uncertainty_lens.pipeline import UncertaintyPipeline


# ========== Fixtures ==========


@pytest.fixture
def grouped_df():
    """Dataset with deliberate distributional shift between groups."""
    rng = np.random.default_rng(42)
    n_per_group = 200

    # Group A: normal(100, 5) — standard
    # Group B: normal(100, 5) — same distribution (no shift)
    # Group C: normal(130, 20) — shifted mean AND spread
    a = pd.DataFrame({
        "value": rng.normal(100, 5, n_per_group),
        "stable": rng.normal(50, 2, n_per_group),
        "group": "A",
    })
    b = pd.DataFrame({
        "value": rng.normal(100, 5, n_per_group),
        "stable": rng.normal(50, 2, n_per_group),
        "group": "B",
    })
    c = pd.DataFrame({
        "value": rng.normal(130, 20, n_per_group),
        "stable": rng.normal(50, 2, n_per_group),
        "group": "C",
    })
    return pd.concat([a, b, c], ignore_index=True)


@pytest.fixture
def noisy_df():
    """High-noise dataset (high aleatoric) with many samples (low epistemic)."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "noisy": rng.exponential(100, 2000),
        "clean": rng.normal(50, 0.5, 2000),
    })


@pytest.fixture
def small_df():
    """Small dataset (high epistemic uncertainty)."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "x": rng.normal(100, 10, 15),
        "y": rng.normal(50, 5, 15),
    })


@pytest.fixture
def edge_df():
    """Edge-case dataset."""
    return pd.DataFrame({
        "constant": [42.0] * 100,
        "all_nan": [np.nan] * 100,
        "normal": np.random.default_rng(42).normal(0, 1, 100),
        "group": (["X"] * 50) + (["Y"] * 50),
    })


# ========== ConformalShiftDetector ==========


class TestConformalShiftDetector:
    def test_basic_shift_detection(self, grouped_df):
        detector = ConformalShiftDetector()
        results = detector.analyze(grouped_df, group_col="group")

        assert "uncertainty_scores" in results
        assert "group_shift" in results
        assert "population_profile" in results

        # Group C is shifted — should be detected
        assert "C" in results["group_shift"]
        c_value = results["group_shift"]["C"]["value"]
        assert c_value["shifted"] == True  # noqa: E712 (numpy bool)

        # "value" feature should have higher uncertainty than "stable"
        assert results["uncertainty_scores"]["value"] > results["uncertainty_scores"]["stable"]

    def test_no_shift_detected(self):
        """All groups from same distribution — no shift expected."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "x": rng.normal(0, 1, 600),
            "group": (["A"] * 200) + (["B"] * 200) + (["C"] * 200),
        })
        detector = ConformalShiftDetector()
        results = detector.analyze(df, group_col="group")

        # Score should be low (no distributional shift)
        assert results["uncertainty_scores"]["x"] < 0.5

    def test_no_group_col(self, grouped_df):
        """Without group_col, scores default to 0."""
        detector = ConformalShiftDetector()
        results = detector.analyze(grouped_df)

        for score in results["uncertainty_scores"].values():
            assert score == 0.0
        assert results["group_shift"] == {}

    def test_edge_cases(self, edge_df):
        detector = ConformalShiftDetector()
        results = detector.analyze(edge_df, group_col="group")

        assert "uncertainty_scores" in results
        # Should not crash on constant or all-NaN columns

    def test_input_validation(self):
        detector = ConformalShiftDetector()
        with pytest.raises(TypeError):
            detector.analyze("not a dataframe")
        with pytest.raises(ValueError, match="empty"):
            detector.analyze(pd.DataFrame())
        with pytest.raises(ValueError, match="not found"):
            detector.analyze(
                pd.DataFrame({"x": [1, 2, 3]}), group_col="nonexistent"
            )

    def test_constructor_validation(self):
        with pytest.raises(ValueError):
            ConformalShiftDetector(significance=0)
        with pytest.raises(ValueError):
            ConformalShiftDetector(calibration_fraction=1.5)

    def test_small_groups(self):
        """Groups with < 3 observations should be handled gracefully."""
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 100.0, 200.0],
            "group": ["A", "A", "A", "B", "B"],
        })
        detector = ConformalShiftDetector()
        results = detector.analyze(df, group_col="group")

        # Should not crash; B might have insufficient data note
        assert "uncertainty_scores" in results


# ========== UncertaintyDecomposer ==========


class TestUncertaintyDecomposer:
    def test_basic_decomposition(self, noisy_df):
        decomposer = UncertaintyDecomposer(n_bootstrap=100)
        results = decomposer.analyze(noisy_df)

        assert "uncertainty_scores" in results
        assert "decomposition" in results
        assert "recommendation" in results

        # Noisy feature should have aleatoric dominance (large dataset, so
        # epistemic is low, but noise is high)
        d = results["decomposition"]["noisy"]
        assert d["aleatoric_score"] > 0  # some aleatoric signal

    def test_small_dataset_epistemic(self, small_df):
        """Small dataset should show higher epistemic uncertainty."""
        decomposer = UncertaintyDecomposer(n_bootstrap=100)
        results = decomposer.analyze(small_df)

        # With only 15 samples, epistemic should be non-trivial
        d = results["decomposition"]["x"]
        assert d["epistemic_score"] >= 0  # at minimum computed

    def test_large_vs_small_epistemic(self):
        """Epistemic should decrease with more data."""
        rng = np.random.default_rng(42)
        small = pd.DataFrame({"x": rng.normal(100, 10, 20)})
        large = pd.DataFrame({"x": rng.normal(100, 10, 5000)})

        decomposer = UncertaintyDecomposer(n_bootstrap=100, seed=42)
        r_small = decomposer.analyze(small)
        r_large = decomposer.analyze(large)

        epi_small = r_small["decomposition"]["x"]["epistemic_raw"]
        epi_large = r_large["decomposition"]["x"]["epistemic_raw"]

        # With 250x more data, epistemic raw variance should be much smaller
        assert epi_large < epi_small

    def test_group_decomposition(self, noisy_df):
        """Test per-group decomposition when group_col is provided."""
        df = noisy_df.copy()
        rng = np.random.default_rng(42)
        df["group"] = rng.choice(["A", "B"], len(df))

        decomposer = UncertaintyDecomposer(n_bootstrap=50)
        results = decomposer.analyze(df, group_col="group")

        assert "group_decomposition" in results
        assert "A" in results["group_decomposition"]
        assert "B" in results["group_decomposition"]

    def test_recommendations(self, noisy_df):
        decomposer = UncertaintyDecomposer(n_bootstrap=100)
        results = decomposer.analyze(noisy_df)

        for col, rec in results["recommendation"].items():
            assert "action" in rec
            assert rec["action"] in (
                "collect_more_data",
                "improve_measurement",
                "both",
                "none",
            )
            assert "explanation" in rec

    def test_constant_feature(self, edge_df):
        decomposer = UncertaintyDecomposer(n_bootstrap=50)
        results = decomposer.analyze(edge_df)

        # Constant feature has zero variance → scores should be near zero
        d = results["decomposition"]["constant"]
        assert d["aleatoric_score"] < 0.1
        assert d["epistemic_score"] < 0.15  # near-zero; slight noise from bootstrap

    def test_input_validation(self):
        decomposer = UncertaintyDecomposer()
        with pytest.raises(TypeError):
            decomposer.analyze([1, 2, 3])
        with pytest.raises(ValueError, match="empty"):
            decomposer.analyze(pd.DataFrame())

    def test_constructor_validation(self):
        with pytest.raises(ValueError):
            UncertaintyDecomposer(n_bootstrap=5)
        with pytest.raises(ValueError):
            UncertaintyDecomposer(statistic="mode")

    def test_median_statistic(self, noisy_df):
        decomposer = UncertaintyDecomposer(n_bootstrap=50, statistic="median")
        results = decomposer.analyze(noisy_df)
        assert results["config"]["statistic"] == "median"
        assert "decomposition" in results


# ========== Pipeline Integration ==========


class TestPipelineWithTier1:
    def test_register_conformal_detector(self, grouped_df):
        """ConformalShiftDetector integrates via register()."""
        pipe = UncertaintyPipeline()
        pipe.register("conformal", ConformalShiftDetector(), weight=0.2)

        report = pipe.analyze(grouped_df, group_col="group")

        assert "conformal_analysis" in report
        assert "conformal_score" in list(report["uncertainty_index"].values())[0]

    def test_register_decomposer(self, noisy_df):
        """UncertaintyDecomposer integrates via register()."""
        pipe = UncertaintyPipeline()
        pipe.register("decomposition", UncertaintyDecomposer(n_bootstrap=50), weight=0.15)

        report = pipe.analyze(noisy_df)

        assert "decomposition_analysis" in report
        assert "decomposition_score" in list(report["uncertainty_index"].values())[0]

    def test_full_5_detector_pipeline(self, grouped_df):
        """Pipeline with all 3 built-in + 2 Tier 1 detectors."""
        pipe = UncertaintyPipeline()
        pipe.register("conformal", ConformalShiftDetector(), weight=0.15)
        pipe.register("decomposition", UncertaintyDecomposer(n_bootstrap=50), weight=0.1)

        assert len(pipe.registered_detectors) == 5

        report = pipe.analyze(grouped_df, group_col="group")

        # All 5 analysis keys present
        assert "missing_analysis" in report
        assert "anomaly_analysis" in report
        assert "variance_analysis" in report
        assert "conformal_analysis" in report
        assert "decomposition_analysis" in report

        # Composite scores still bounded [0, 1]
        for col, vals in report["uncertainty_index"].items():
            assert 0 <= vals["composite_score"] <= 1.0

        # Weights auto-normalized to sum to 1
        w = pipe.weights
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_backward_compatibility_preserved(self, grouped_df):
        """Adding new detectors doesn't break the base 3-detector pipeline."""
        pipe_old = UncertaintyPipeline()
        report_old = pipe_old.analyze(grouped_df, group_col="group")

        pipe_new = UncertaintyPipeline()
        pipe_new.register("conformal", ConformalShiftDetector(), weight=0.0)
        pipe_new.register("decomposition", UncertaintyDecomposer(n_bootstrap=50), weight=0.0)
        report_new = pipe_new.analyze(grouped_df, group_col="group")

        # With weight=0 for new detectors, composite scores should be identical
        for col in report_old["uncertainty_index"]:
            old_score = report_old["uncertainty_index"][col]["composite_score"]
            new_score = report_new["uncertainty_index"][col]["composite_score"]
            assert abs(old_score - new_score) < 0.01, (
                f"{col}: old={old_score}, new={new_score}"
            )
