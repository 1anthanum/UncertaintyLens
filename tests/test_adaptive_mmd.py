"""Tests for adaptive multi-bandwidth MMD."""

import pytest
import numpy as np
import pandas as pd

from uncertainty_lens.detectors import MMDShiftDetector


class TestAdaptiveMMD:
    def test_adaptive_detects_mean_shift(self):
        """自适应核应该检测到均值偏移."""
        rng = np.random.default_rng(42)
        n = 400
        df = pd.DataFrame(
            {
                "x": np.concatenate([rng.normal(0, 1, n // 2), rng.normal(3, 1, n // 2)]),
                "group": ["A"] * (n // 2) + ["B"] * (n // 2),
            }
        )
        det = MMDShiftDetector(bandwidth="adaptive", n_permutations=100, seed=42)
        result = det.analyze(df, group_col="group")
        assert result["uncertainty_scores"]["x"] > 0.5

    def test_adaptive_no_false_positive(self):
        """无偏移时不应误报."""
        rng = np.random.default_rng(123)
        n = 400
        df = pd.DataFrame(
            {
                "x": rng.normal(0, 1, n),
                "y": rng.normal(5, 2, n),
                "group": rng.choice(["A", "B"], n),
            }
        )
        det = MMDShiftDetector(bandwidth="adaptive", n_permutations=100, seed=42)
        result = det.analyze(df, group_col="group")
        # 分数应该较低
        for col in ["x", "y"]:
            assert result["uncertainty_scores"][col] < 0.6

    def test_adaptive_detects_variance_shift(self):
        """自适应核应检测方差变化（单一 median bandwidth 可能遗漏）."""
        rng = np.random.default_rng(42)
        n = 400
        df = pd.DataFrame(
            {
                "x": np.concatenate([rng.normal(0, 1, n // 2), rng.normal(0, 5, n // 2)]),
                "group": ["A"] * (n // 2) + ["B"] * (n // 2),
            }
        )
        det = MMDShiftDetector(bandwidth="adaptive", n_permutations=100, seed=42)
        result = det.analyze(df, group_col="group")
        assert result["uncertainty_scores"]["x"] > 0.3

    def test_adaptive_vs_median_on_subtle_shift(self):
        """自适应核在微妙偏移上不弱于 median."""
        rng = np.random.default_rng(42)
        n = 600
        df = pd.DataFrame(
            {
                "x": np.concatenate([rng.normal(0, 1, n // 2), rng.normal(0.8, 1, n // 2)]),
                "group": ["A"] * (n // 2) + ["B"] * (n // 2),
            }
        )
        det_adaptive = MMDShiftDetector(bandwidth="adaptive", n_permutations=150, seed=42)
        det_median = MMDShiftDetector(bandwidth="median", n_permutations=150, seed=42)

        r_adaptive = det_adaptive.analyze(df, group_col="group")
        r_median = det_median.analyze(df, group_col="group")

        # 自适应版本的分数应不低于 median 版本（允许一定随机波动）
        assert r_adaptive["uncertainty_scores"]["x"] >= r_median["uncertainty_scores"]["x"] - 0.2

    def test_adaptive_output_structure(self):
        """输出结构应与 median 版本一致."""
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame(
            {
                "x": rng.normal(0, 1, n),
                "group": rng.choice(["A", "B"], n),
            }
        )
        det = MMDShiftDetector(bandwidth="adaptive", n_permutations=50, seed=42)
        result = det.analyze(df, group_col="group")

        assert "uncertainty_scores" in result
        assert "group_shift" in result
        assert "per_feature_mmd" in result
        assert "joint_mmd" in result
