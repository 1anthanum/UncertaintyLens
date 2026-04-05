"""Tests for StreamingDetector — 在线流式不确定性检测."""

import pytest
import numpy as np
import pandas as pd

from uncertainty_lens.detectors import StreamingDetector


class TestStreamingBasic:
    def test_single_batch(self):
        """单次更新应产生分数."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"x": rng.normal(0, 1, 100), "y": rng.normal(5, 2, 100)})
        det = StreamingDetector(window_size=50)
        result = det.update(df)

        assert "uncertainty_scores" in result
        assert "x" in result["uncertainty_scores"]
        assert "y" in result["uncertainty_scores"]

    def test_incremental_update(self):
        """多次增量更新不应报错."""
        rng = np.random.default_rng(42)
        det = StreamingDetector(window_size=50)

        for i in range(5):
            chunk = pd.DataFrame({"x": rng.normal(0, 1, 50)})
            result = det.update(chunk)

        assert result["stats"]["n_total"] == 250
        assert result["uncertainty_scores"]["x"] >= 0

    def test_detects_drift(self):
        """注入均值偏移后应检测到漂移."""
        rng = np.random.default_rng(42)
        det = StreamingDetector(window_size=100, drift_threshold=30)

        # 正常阶段
        for _ in range(5):
            chunk = pd.DataFrame({"x": rng.normal(0, 1, 100)})
            det.update(chunk)

        # 偏移阶段（均值 +10）
        for _ in range(5):
            chunk = pd.DataFrame({"x": rng.normal(10, 1, 100)})
            det.update(chunk)

        result = det.get_scores()
        assert result["drift_detected"] is True

    def test_missing_values_tracked(self):
        """缺失值应被追踪并反映在分数中."""
        det = StreamingDetector(window_size=50)
        data = pd.DataFrame({"x": [1, 2, np.nan, 4, np.nan] * 20})
        result = det.update(data)

        # 40% 缺失应给较高分数
        assert result["uncertainty_scores"]["x"] > 0.1

    def test_stable_data_low_score(self):
        """稳定数据应给低分数."""
        rng = np.random.default_rng(42)
        det = StreamingDetector(window_size=200)

        for _ in range(5):
            chunk = pd.DataFrame({"x": rng.normal(0, 1, 200)})
            det.update(chunk)

        result = det.get_scores()
        assert result["uncertainty_scores"]["x"] < 0.3
        assert result["drift_detected"] is False

    def test_variance_spike_detected(self):
        """方差突变应被检测."""
        rng = np.random.default_rng(42)
        det = StreamingDetector(window_size=100)

        # 正常阶段
        for _ in range(3):
            chunk = pd.DataFrame({"x": rng.normal(0, 1, 100)})
            det.update(chunk)

        # 方差激增
        chunk = pd.DataFrame({"x": rng.normal(0, 20, 200)})
        det.update(chunk)

        result = det.get_scores()
        # 分数应因方差变化而升高
        assert result["uncertainty_scores"]["x"] > 0.1

    def test_reset(self):
        """重置后应恢复初始状态."""
        rng = np.random.default_rng(42)
        det = StreamingDetector(window_size=50)
        det.update(pd.DataFrame({"x": rng.normal(0, 1, 100)}))
        det.reset()

        result = det.get_scores()
        assert result["stats"]["n_total"] == 0
        assert result["uncertainty_scores"] == {}


class TestStreamingEdgeCases:
    def test_empty_dataframe(self):
        det = StreamingDetector(window_size=50)
        result = det.update(pd.DataFrame())
        assert result["uncertainty_scores"] == {}

    def test_non_numeric_ignored(self):
        det = StreamingDetector(window_size=50)
        df = pd.DataFrame({"text": ["a", "b", "c"] * 20, "num": range(60)})
        result = det.update(df)
        assert "num" in result["uncertainty_scores"]
        assert "text" not in result["uncertainty_scores"]

    def test_invalid_input(self):
        det = StreamingDetector(window_size=50)
        with pytest.raises(TypeError):
            det.update("not a dataframe")

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            StreamingDetector(window_size=5)
        with pytest.raises(ValueError):
            StreamingDetector(ewma_alpha=0)
        with pytest.raises(ValueError):
            StreamingDetector(drift_threshold=-1)

    def test_pipeline_compatible(self):
        """应兼容 Pipeline 的 analyze 接口."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "x": rng.normal(0, 1, 500),
                "y": rng.normal(5, 2, 500),
            }
        )
        det = StreamingDetector(window_size=100)
        result = det.analyze(df)

        assert "uncertainty_scores" in result
        assert "x" in result["uncertainty_scores"]
        assert "y" in result["uncertainty_scores"]

    def test_alerts_generated(self):
        """当存在问题时应生成警报."""
        det = StreamingDetector(window_size=50, drift_threshold=20)

        # 大量缺失
        data = pd.DataFrame({"x": [np.nan] * 40 + [1.0] * 10})
        result = det.update(data)
        # 80% 缺失应触发警报
        assert any("缺失率" in a for a in result["alerts"])
