"""
零膨胀特征检测器.

专门解决已知的检测局限：当特征有大量零值（如 92% 为零的 capital_gain）时，
标准异常检测器将零视为"多数 → 正常"，共形预测器认为"预测 0 就对了"，
导致复合分数偏低。

本检测器直接识别零膨胀模式并给出合理的不确定性分数：

1. **零比例检测** — 超过阈值（默认 50%）的零值特征被标记
2. **双组件分析** — 将数据分为零组和非零组，分别分析
3. **条件不确定性** — 非零部分的方差、偏度、离群程度
4. **分数计算** — 综合零比例 + 非零部分的不规则性

典型场景：
  - 金融: capital_gain/loss（大量零 + 长尾）
  - 电商: 退货金额、折扣金额
  - 医疗: 住院天数、手术费用
  - 传感器: 故障计数、报警次数

Usage
-----
::

    from uncertainty_lens.detectors import ZeroInflationDetector

    detector = ZeroInflationDetector(zero_threshold=0.5)
    result = detector.analyze(df)
    # result["uncertainty_scores"]       → 每特征 [0,1] 分数
    # result["zero_inflated_features"]   → 被识别的零膨胀特征
    # result["feature_analysis"]         → 每特征详细分析
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional


class ZeroInflationDetector:
    """
    零膨胀特征检测器.

    Parameters
    ----------
    zero_threshold : float
        零值比例超过此阈值才认为是零膨胀（默认 0.50 = 50%）。
    zero_tolerance : float
        接近零的容差，|x| < zero_tolerance 视为零（默认 1e-8）。
    """

    def __init__(
        self,
        zero_threshold: float = 0.50,
        zero_tolerance: float = 1e-8,
    ):
        if not 0 < zero_threshold < 1:
            raise ValueError(f"zero_threshold must be in (0, 1), got {zero_threshold}")
        if zero_tolerance < 0:
            raise ValueError(f"zero_tolerance must be >= 0, got {zero_tolerance}")

        self.zero_threshold = zero_threshold
        self.zero_tolerance = zero_tolerance
        self.results_: Optional[Dict[str, Any]] = None

    # ── public API ─────────────────────────────────────────────────────

    def analyze(
        self,
        df: pd.DataFrame,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        分析每个数值特征的零膨胀程度.

        Returns
        -------
        dict
            "uncertainty_scores"      — 每特征 [0,1] 分数
            "zero_inflated_features"  — 零膨胀特征名列表
            "feature_analysis"        — 每特征详细分析
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(df).__name__}")
        if df.empty:
            raise ValueError("DataFrame is empty")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return self._default_result([], "No numeric columns")

        feature_analysis: Dict[str, Dict[str, Any]] = {}
        uncertainty_scores: Dict[str, float] = {}
        zero_inflated: List[str] = []

        for col in numeric_cols:
            analysis = self._analyze_feature(df[col].dropna().values, col)
            feature_analysis[col] = analysis
            uncertainty_scores[col] = analysis["uncertainty_score"]
            if analysis["is_zero_inflated"]:
                zero_inflated.append(col)

        results = {
            "uncertainty_scores": uncertainty_scores,
            "zero_inflated_features": zero_inflated,
            "feature_analysis": feature_analysis,
            "method": "zero_inflation_detection",
            "zero_threshold": self.zero_threshold,
        }
        self.results_ = results
        return results

    # ── 单特征分析 ─────────────────────────────────────────────────────

    def _analyze_feature(self, values: np.ndarray, col_name: str) -> Dict[str, Any]:
        """分析单个特征的零膨胀程度."""
        n = len(values)
        if n == 0:
            return {
                "is_zero_inflated": False,
                "uncertainty_score": 0.0,
                "zero_fraction": 0.0,
                "note": "empty",
            }

        # 零值检测
        is_zero = np.abs(values) < self.zero_tolerance
        zero_fraction = float(is_zero.sum()) / n
        is_inflated = zero_fraction >= self.zero_threshold

        if not is_inflated:
            # 不是零膨胀 → 给一个基于零比例的小分数
            # 即使不是"零膨胀"，20-30%的零值也有一定不确定性
            if zero_fraction > 0.1:
                score = round(float(zero_fraction * 0.3), 4)
            else:
                score = 0.0
            return {
                "is_zero_inflated": False,
                "uncertainty_score": score,
                "zero_fraction": round(zero_fraction, 4),
                "note": "not zero-inflated",
            }

        # ── 零膨胀特征的详细分析 ──

        nonzero_values = values[~is_zero]
        n_nonzero = len(nonzero_values)

        result: Dict[str, Any] = {
            "is_zero_inflated": True,
            "zero_fraction": round(zero_fraction, 4),
            "n_total": n,
            "n_zero": int(is_zero.sum()),
            "n_nonzero": n_nonzero,
        }

        if n_nonzero < 3:
            # 几乎全是零
            result["uncertainty_score"] = round(min(1.0, zero_fraction * 1.2), 4)
            result["note"] = "nearly_all_zero"
            return result

        # 非零部分的统计特征
        nz_mean = float(np.mean(nonzero_values))
        nz_std = float(np.std(nonzero_values))
        nz_median = float(np.median(nonzero_values))
        nz_skewness = self._safe_skewness(nonzero_values)
        nz_kurtosis = self._safe_kurtosis(nonzero_values)

        # 非零部分的离群值比例（IQR 方法）
        q1, q3 = np.percentile(nonzero_values, [25, 75])
        iqr = q3 - q1
        if iqr > 0:
            outlier_mask = (nonzero_values < q1 - 1.5 * iqr) | (nonzero_values > q3 + 1.5 * iqr)
            outlier_fraction = float(outlier_mask.sum()) / n_nonzero
        else:
            outlier_fraction = 0.0

        result["nonzero_stats"] = {
            "mean": round(nz_mean, 4),
            "std": round(nz_std, 4),
            "median": round(nz_median, 4),
            "skewness": round(nz_skewness, 4),
            "kurtosis": round(nz_kurtosis, 4),
            "outlier_fraction": round(outlier_fraction, 4),
        }

        # ── 不确定性分数计算 ──
        # 三个维度加权：
        #   1. 零比例本身 (权重 0.4)：越多零 → 越不确定
        #   2. 非零部分偏度 (权重 0.3)：高偏度 → 非零部分不规则
        #   3. 非零部分离群比例 (权重 0.3)：多离群 → 不规则

        # (1) 零比例分数: sigmoid, 80% 零 → ~0.7, 95% 零 → ~0.95
        zero_score = float(1.0 / (1.0 + np.exp(-10 * (zero_fraction - 0.7))))

        # (2) 偏度分数: |skew| > 2 → 高不规则
        skew_score = float(1.0 / (1.0 + np.exp(-2 * (abs(nz_skewness) - 1.5))))

        # (3) 离群分数
        outlier_score = min(1.0, outlier_fraction * 5)

        composite = 0.4 * zero_score + 0.3 * skew_score + 0.3 * outlier_score
        composite = min(1.0, composite)

        result["uncertainty_score"] = round(composite, 4)
        result["component_scores"] = {
            "zero_proportion": round(zero_score, 4),
            "skewness": round(skew_score, 4),
            "outlier": round(outlier_score, 4),
        }

        # 建议
        if zero_fraction > 0.9:
            result["recommendation"] = (
                "极高零膨胀 — 建议先做二值化（是否为零），" "再单独建模非零部分"
            )
        elif zero_fraction > 0.7:
            result["recommendation"] = (
                "显著零膨胀 — 建议使用零膨胀模型（ZIP/ZINB）" "或 Hurdle 模型"
            )
        else:
            result["recommendation"] = (
                "中度零膨胀 — 注意标准回归模型可能不适用，" "考虑 Tobit 模型或变换"
            )

        return result

    # ── helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _safe_skewness(x: np.ndarray) -> float:
        n = len(x)
        if n < 3:
            return 0.0
        m = x.mean()
        s = x.std()
        if s < 1e-10:
            return 0.0
        return float(np.mean(((x - m) / s) ** 3))

    @staticmethod
    def _safe_kurtosis(x: np.ndarray) -> float:
        n = len(x)
        if n < 4:
            return 0.0
        m = x.mean()
        s = x.std()
        if s < 1e-10:
            return 0.0
        return float(np.mean(((x - m) / s) ** 4) - 3)

    def _default_result(self, cols: List[str], note: str) -> Dict[str, Any]:
        return {
            "uncertainty_scores": {c: 0.0 for c in cols},
            "zero_inflated_features": [],
            "feature_analysis": {},
            "method": "zero_inflation_detection",
            "note": note,
        }
