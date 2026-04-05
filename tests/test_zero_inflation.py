"""ZeroInflationDetector 测试."""

import numpy as np
import pandas as pd
import pytest

from uncertainty_lens.detectors import ZeroInflationDetector


class TestZeroInflationBasic:
    """基本功能测试."""

    def test_detects_high_zero_inflation(self):
        """92% 零值应该被检测为零膨胀."""
        rng = np.random.default_rng(42)
        n = 1000
        values = np.zeros(n)
        values[:80] = rng.lognormal(5, 1, 80)  # 8% 非零

        df = pd.DataFrame({"capital_gain": values})
        det = ZeroInflationDetector(zero_threshold=0.5)
        result = det.analyze(df)

        assert "capital_gain" in result["zero_inflated_features"]
        assert result["uncertainty_scores"]["capital_gain"] > 0.3
        assert result["feature_analysis"]["capital_gain"]["is_zero_inflated"]
        assert result["feature_analysis"]["capital_gain"]["zero_fraction"] > 0.9

    def test_normal_data_not_flagged(self):
        """正态分布数据不应被标记为零膨胀."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"x": rng.normal(5, 2, 1000)})

        det = ZeroInflationDetector()
        result = det.analyze(df)

        assert len(result["zero_inflated_features"]) == 0
        assert result["uncertainty_scores"]["x"] == 0.0

    def test_moderate_zeros(self):
        """30% 零值不应被标记（低于阈值），但有小分数."""
        rng = np.random.default_rng(42)
        n = 1000
        values = rng.normal(5, 2, n)
        values[:300] = 0

        df = pd.DataFrame({"x": values})
        det = ZeroInflationDetector(zero_threshold=0.5)
        result = det.analyze(df)

        assert not result["feature_analysis"]["x"]["is_zero_inflated"]
        # 但 30% 零值还是有一点分数
        assert result["uncertainty_scores"]["x"] > 0

    def test_multiple_features(self):
        """多特征混合：一个零膨胀，一个正常."""
        rng = np.random.default_rng(42)
        n = 1000
        inflated = np.zeros(n)
        inflated[:50] = rng.exponential(100, 50)  # 5% 非零
        normal = rng.normal(0, 1, n)

        df = pd.DataFrame({"inflated": inflated, "normal": normal})
        det = ZeroInflationDetector()
        result = det.analyze(df)

        assert "inflated" in result["zero_inflated_features"]
        assert "normal" not in result["zero_inflated_features"]
        assert result["uncertainty_scores"]["inflated"] > result["uncertainty_scores"]["normal"]

    def test_nonzero_stats_computed(self):
        """非零部分的统计量应被计算."""
        rng = np.random.default_rng(42)
        n = 1000
        values = np.zeros(n)
        values[:100] = rng.lognormal(5, 2, 100)

        df = pd.DataFrame({"x": values})
        det = ZeroInflationDetector()
        result = det.analyze(df)

        stats = result["feature_analysis"]["x"]["nonzero_stats"]
        assert "mean" in stats
        assert "std" in stats
        assert "skewness" in stats
        assert "outlier_fraction" in stats

    def test_recommendation_for_severe(self):
        """>90% 零值应建议二值化."""
        values = np.zeros(1000)
        values[:50] = np.arange(1, 51)

        df = pd.DataFrame({"x": values})
        det = ZeroInflationDetector()
        result = det.analyze(df)

        rec = result["feature_analysis"]["x"]["recommendation"]
        assert "二值化" in rec


class TestZeroInflationEdgeCases:
    """边界情况."""

    def test_all_zeros(self):
        """全部为零."""
        df = pd.DataFrame({"x": np.zeros(100)})
        det = ZeroInflationDetector()
        result = det.analyze(df)

        assert result["feature_analysis"]["x"]["is_zero_inflated"]
        assert result["uncertainty_scores"]["x"] > 0.5

    def test_empty_dataframe(self):
        with pytest.raises(ValueError, match="empty"):
            ZeroInflationDetector().analyze(pd.DataFrame())

    def test_non_dataframe(self):
        with pytest.raises(TypeError):
            ZeroInflationDetector().analyze(np.array([1, 2, 3]))

    def test_invalid_threshold(self):
        with pytest.raises(ValueError):
            ZeroInflationDetector(zero_threshold=0.0)
        with pytest.raises(ValueError):
            ZeroInflationDetector(zero_threshold=1.0)

    def test_pipeline_integration(self):
        """与 pipeline 集成."""
        from uncertainty_lens import UncertaintyPipeline

        rng = np.random.default_rng(42)
        n = 500
        inflated = np.zeros(n)
        inflated[:40] = rng.lognormal(5, 1, 40)
        normal = rng.normal(0, 1, n)

        df = pd.DataFrame({"inflated": inflated, "normal": normal})

        pipeline = UncertaintyPipeline()
        pipeline.register("zero_inflation", ZeroInflationDetector(), weight=0.2)
        report = pipeline.analyze(df)

        assert "zero_inflation_analysis" in report
        # 零膨胀特征的复合分数应高于正常特征
        zi_score = report["uncertainty_index"]["inflated"]["composite_score"]
        nm_score = report["uncertainty_index"]["normal"]["composite_score"]
        assert zi_score > nm_score
