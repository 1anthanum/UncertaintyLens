"""MMDShiftDetector 测试."""

import numpy as np
import pandas as pd
import pytest

from uncertainty_lens.detectors import MMDShiftDetector


class TestMMDBasic:
    """基本功能测试."""

    def test_detects_mean_shift(self):
        """均值偏移应该被检测到."""
        rng = np.random.default_rng(42)
        n = 500
        group = np.array(["A"] * n + ["B"] * n)
        x = np.concatenate([rng.normal(0, 1, n), rng.normal(3, 1, n)])
        y = rng.normal(0, 1, 2 * n)
        df = pd.DataFrame({"x": x, "y": y, "group": group})

        det = MMDShiftDetector(n_permutations=100, seed=42)
        result = det.analyze(df, group_col="group")

        assert result["uncertainty_scores"]["x"] > 0.5
        assert bool(result["group_shift"])

    def test_no_shift_gives_low_score(self):
        """无偏移时分数应该低."""
        rng = np.random.default_rng(42)
        n = 500
        group = np.array(["A"] * n + ["B"] * n)
        x = rng.normal(0, 1, 2 * n)
        y = rng.normal(0, 1, 2 * n)
        df = pd.DataFrame({"x": x, "y": y, "group": group})

        det = MMDShiftDetector(n_permutations=100, seed=42)
        result = det.analyze(df, group_col="group")

        # 无偏移 → 分数应低
        assert result["uncertainty_scores"]["x"] < 0.5
        assert result["uncertainty_scores"]["y"] < 0.5

    def test_detects_variance_shift(self):
        """方差偏移（形状变化）应该被检测到 — KS 的弱项."""
        rng = np.random.default_rng(42)
        n = 500
        group = np.array(["A"] * n + ["B"] * n)
        # 均值相同，但方差不同
        x = np.concatenate([rng.normal(0, 1, n), rng.normal(0, 5, n)])
        df = pd.DataFrame({"x": x, "group": group})

        det = MMDShiftDetector(n_permutations=100, seed=42)
        result = det.analyze(df, group_col="group")

        assert result["uncertainty_scores"]["x"] > 0.3

    def test_joint_mmd_detects_correlation_shift(self):
        """相关结构变化只有联合 MMD 能抓到."""
        rng = np.random.default_rng(42)
        n = 500
        # A 组: x,y 正相关
        x_a = rng.normal(0, 1, n)
        y_a = 0.8 * x_a + rng.normal(0, 0.5, n)
        # B 组: x,y 负相关（边际分布相同！）
        x_b = rng.normal(0, 1, n)
        y_b = -0.8 * x_b + rng.normal(0, 0.5, n)

        group = np.array(["A"] * n + ["B"] * n)
        df = pd.DataFrame(
            {
                "x": np.concatenate([x_a, x_b]),
                "y": np.concatenate([y_a, y_b]),
                "group": group,
            }
        )

        det = MMDShiftDetector(n_permutations=100, seed=42)
        result = det.analyze(df, group_col="group")

        # 联合 MMD 应该检测到
        assert bool(result["joint_mmd"])
        has_joint_shift = any(v["shift_detected"] for v in result["joint_mmd"].values())
        assert has_joint_shift

    def test_output_structure(self):
        """输出结构完整性."""
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame(
            {
                "a": rng.normal(0, 1, n),
                "b": rng.normal(0, 1, n),
                "g": np.where(np.arange(n) < 100, "X", "Y"),
            }
        )

        det = MMDShiftDetector(n_permutations=50, seed=42)
        result = det.analyze(df, group_col="g")

        assert "uncertainty_scores" in result
        assert "group_shift" in result
        assert "per_feature_mmd" in result
        assert "joint_mmd" in result
        assert result["method"] == "mmd_permutation"


class TestMMDEdgeCases:
    """边界情况."""

    def test_no_group_col(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"x": rng.normal(0, 1, 100)})
        det = MMDShiftDetector(seed=42)
        result = det.analyze(df)
        assert result["uncertainty_scores"]["x"] == 0.0

    def test_single_group(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"x": rng.normal(0, 1, 100), "g": "A"})
        det = MMDShiftDetector(seed=42)
        result = det.analyze(df, group_col="g")
        assert result["uncertainty_scores"]["x"] == 0.0

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            MMDShiftDetector(n_permutations=5)
        with pytest.raises(ValueError):
            MMDShiftDetector(significance=0.0)

    def test_pipeline_integration(self):
        """与 pipeline 集成."""
        from uncertainty_lens import UncertaintyPipeline

        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame(
            {
                "a": np.concatenate([rng.normal(0, 1, 100), rng.normal(5, 1, 100)]),
                "b": rng.normal(0, 1, n),
                "g": np.where(np.arange(n) < 100, "X", "Y"),
            }
        )

        pipeline = UncertaintyPipeline()
        pipeline.register("mmd_shift", MMDShiftDetector(n_permutations=50, seed=42), weight=0.15)
        report = pipeline.analyze(df, group_col="g")

        assert "mmd_shift_analysis" in report
        for col_data in report["uncertainty_index"].values():
            assert "mmd_shift_score" in col_data
