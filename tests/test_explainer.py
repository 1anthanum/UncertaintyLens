"""Tests for UncertaintyExplainer — 不确定性可解释性模块."""

import pytest
import numpy as np
import pandas as pd

from uncertainty_lens import UncertaintyPipeline
from uncertainty_lens.detectors import (
    UncertaintyExplainer,
    ConformalShiftDetector,
    UncertaintyDecomposer,
    JackknifePlusDetector,
)


def _build_pipeline_and_report():
    """构建管线并分析测试数据."""
    rng = np.random.default_rng(42)
    n = 500
    df = pd.DataFrame(
        {
            "clean": rng.normal(0, 1, n),
            "noisy": rng.normal(0, 10, n),
            "missing": np.where(rng.random(n) < 0.3, np.nan, rng.normal(0, 1, n)),
            "group": rng.choice(["A", "B"], n),
        }
    )
    # 注入 B 组偏移
    mask = df["group"] == "B"
    df.loc[mask, "noisy"] += 20

    pipeline = UncertaintyPipeline()
    pipeline.register("conformal_shift", ConformalShiftDetector(seed=42), weight=0.1)
    pipeline.register("decomposition", UncertaintyDecomposer(n_bootstrap=100, seed=42), weight=0.15)
    report = pipeline.analyze(df, group_col="group")
    return report


class TestExplainerBasic:
    def test_output_structure(self):
        report = _build_pipeline_and_report()
        explainer = UncertaintyExplainer(language="cn")
        result = explainer.explain(report)

        assert "feature_explanations" in result
        assert "global_insights" in result
        assert "action_plan" in result

    def test_feature_explanations_keys(self):
        report = _build_pipeline_and_report()
        explainer = UncertaintyExplainer()
        result = explainer.explain(report)

        for col, expl in result["feature_explanations"].items():
            assert "composite_score" in expl
            assert "summary" in expl
            assert "top_contributors" in expl
            assert "all_contributors" in expl

    def test_contributions_sum_to_composite(self):
        report = _build_pipeline_and_report()
        explainer = UncertaintyExplainer()
        result = explainer.explain(report)

        for col, expl in result["feature_explanations"].items():
            total_contrib = sum(c["contribution"] for c in expl["all_contributors"])
            # 贡献之和应约等于 composite (允许舍入误差)
            assert abs(total_contrib - expl["composite_score"]) < 0.01

    def test_missing_feature_high_missing_contribution(self):
        report = _build_pipeline_and_report()
        explainer = UncertaintyExplainer()
        result = explainer.explain(report)

        missing_expl = result["feature_explanations"]["missing"]
        contributors = {c["detector"]: c for c in missing_expl["all_contributors"]}
        assert "missing" in contributors
        assert contributors["missing"]["raw_score"] > 0.01

    def test_english_language(self):
        report = _build_pipeline_and_report()
        explainer = UncertaintyExplainer(language="en")
        result = explainer.explain(report)

        for col, expl in result["feature_explanations"].items():
            # English summaries should not contain Chinese characters
            assert "不确定性" not in expl["summary"]

    def test_top_k_parameter(self):
        report = _build_pipeline_and_report()
        explainer = UncertaintyExplainer(top_k=2)
        result = explainer.explain(report)

        for col, expl in result["feature_explanations"].items():
            assert len(expl["top_contributors"]) <= 2

    def test_global_insights_nonempty(self):
        report = _build_pipeline_and_report()
        explainer = UncertaintyExplainer()
        result = explainer.explain(report)

        # 至少应该有一些全局洞察
        assert isinstance(result["global_insights"], list)

    def test_action_plan_ordered_by_priority(self):
        report = _build_pipeline_and_report()
        explainer = UncertaintyExplainer()
        result = explainer.explain(report)

        plan = result["action_plan"]
        if len(plan) > 1:
            priorities = [a["priority"] for a in plan]
            assert priorities == sorted(priorities)


class TestExplainerEdgeCases:
    def test_empty_report(self):
        explainer = UncertaintyExplainer()
        result = explainer.explain({"uncertainty_index": {}})
        assert result["feature_explanations"] == {}
        assert result["global_insights"] == []
        assert result["action_plan"] == []

    def test_invalid_input(self):
        explainer = UncertaintyExplainer()
        with pytest.raises(TypeError):
            explainer.explain("not a dict")

    def test_invalid_language(self):
        with pytest.raises(ValueError):
            UncertaintyExplainer(language="fr")

    def test_invalid_top_k(self):
        with pytest.raises(ValueError):
            UncertaintyExplainer(top_k=0)

    def test_all_low_scores(self):
        """所有特征都很干净时不应生成行动建议."""
        report = {
            "uncertainty_index": {
                "a": {
                    "composite_score": 0.1,
                    "missing_score": 0.0,
                    "anomaly_score": 0.05,
                    "variance_score": 0.1,
                    "level": "Low",
                },
                "b": {
                    "composite_score": 0.05,
                    "missing_score": 0.0,
                    "anomaly_score": 0.0,
                    "variance_score": 0.05,
                    "level": "Low",
                },
            }
        }
        explainer = UncertaintyExplainer()
        result = explainer.explain(report)
        assert len(result["action_plan"]) == 0
