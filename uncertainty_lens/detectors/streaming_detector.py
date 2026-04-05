"""
在线流式不确定性检测器 (Streaming Uncertainty Detector).

与批处理模式不同，流式检测器逐行（或小批量）接收数据，
实时更新不确定性估计，无需每次重新扫描全部数据。

适用场景:
  - 生产环境的实时数据质量监控
  - 数据管道中的在线检测
  - 大数据集的增量处理

核心思路:
  使用 Welford 在线算法维护均值/方差的运行估计，
  配合滑动窗口检测短期漂移，EWMA 检测趋势变化。

  不检测:
  - 缺失模式（直接计数即可）
  - 异常值（使用运行 IQR 的近似）
  - 方差变化（比较最近窗口 vs 历史）
  - 分布漂移（Page-Hinkley 测试）

Usage
-----
::

    detector = StreamingDetector(window_size=500)

    for batch in data_stream:
        detector.update(batch)
        scores = detector.get_scores()
        if scores["drift_detected"]:
            alert("数据质量变化!")
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional


class StreamingDetector:
    """
    在线流式不确定性检测器.

    Parameters
    ----------
    window_size : int
        滑动窗口大小（行数）。窗口越大越平滑，越小响应越快。
        默认 500。
    ewma_alpha : float
        EWMA（指数加权移动平均）衰减系数。
        越大越关注最近数据，越小越平滑。默认 0.05。
    drift_threshold : float
        Page-Hinkley 漂移检测阈值。越大越不容易触发警报。
        默认 50.0。
    seed : int
        随机种子。
    """

    def __init__(
        self,
        window_size: int = 500,
        ewma_alpha: float = 0.05,
        drift_threshold: float = 50.0,
        seed: int = 42,
    ):
        if window_size < 10:
            raise ValueError(f"window_size must be >= 10, got {window_size}")
        if not 0 < ewma_alpha < 1:
            raise ValueError(f"ewma_alpha must be in (0, 1), got {ewma_alpha}")
        if drift_threshold <= 0:
            raise ValueError(f"drift_threshold must be > 0, got {drift_threshold}")

        self.window_size = window_size
        self.ewma_alpha = ewma_alpha
        self.drift_threshold = drift_threshold
        self.seed = seed

        # 内部状态
        self._feature_states: Dict[str, _FeatureState] = {}
        self._n_total = 0
        self._initialized = False

    def update(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        接收新数据并更新内部状态。

        Parameters
        ----------
        data : pd.DataFrame
            新到达的数据（一行或多行）。

        Returns
        -------
        dict
            当前不确定性评估快照:
            - uncertainty_scores: 每特征的实时不确定性分数
            - drift_detected: 是否检测到漂移
            - alerts: 触发的警报列表
            - stats: 运行统计量
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"data must be DataFrame, got {type(data).__name__}")

        if data.empty:
            return self.get_scores()

        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        # 初始化新特征的状态
        for col in numeric_cols:
            if col not in self._feature_states:
                self._feature_states[col] = _FeatureState(
                    name=col,
                    window_size=self.window_size,
                    ewma_alpha=self.ewma_alpha,
                    drift_threshold=self.drift_threshold,
                )

        # 逐行更新（向量化）
        for col in numeric_cols:
            values = data[col].values
            self._feature_states[col].update_batch(values)

        self._n_total += len(data)
        self._initialized = True

        return self.get_scores()

    def get_scores(self) -> Dict[str, Any]:
        """
        获取当前不确定性评估。

        Returns
        -------
        dict
            包含 uncertainty_scores, drift_detected, alerts, stats。
        """
        if not self._initialized:
            return {
                "uncertainty_scores": {},
                "drift_detected": False,
                "alerts": [],
                "stats": {"n_total": 0},
            }

        scores = {}
        alerts = []
        any_drift = False

        for col, state in self._feature_states.items():
            col_score, col_alerts = state.compute_score()
            scores[col] = round(col_score, 4)
            alerts.extend(col_alerts)
            if state.drift_detected:
                any_drift = True

        return {
            "uncertainty_scores": scores,
            "drift_detected": any_drift,
            "alerts": alerts,
            "stats": {
                "n_total": self._n_total,
                "n_features": len(self._feature_states),
                "window_size": self.window_size,
            },
        }

    def reset(self) -> None:
        """重置所有内部状态。"""
        self._feature_states.clear()
        self._n_total = 0
        self._initialized = False

    # ── Pipeline 兼容接口 ──

    def analyze(self, df: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
        """
        Pipeline 兼容接口。

        模拟流式处理：将 DataFrame 按 window_size 分块逐步输入。
        """
        self.reset()
        n = len(df)
        chunk_size = max(self.window_size, 100)

        for start in range(0, n, chunk_size):
            chunk = df.iloc[start : start + chunk_size]
            self.update(chunk)

        result = self.get_scores()
        # 添加所有特征的默认分数（非数值特征得 0 分）
        for col in df.columns:
            if col not in result["uncertainty_scores"]:
                result["uncertainty_scores"][col] = 0.0

        return result


class _FeatureState:
    """
    单个特征的在线追踪状态。

    维护:
    - Welford 运行均值/方差
    - 滑动窗口最近值
    - EWMA 趋势
    - Page-Hinkley 漂移检测
    - 缺失计数
    - 近似 IQR (P2 算法简化版)
    """

    def __init__(
        self,
        name: str,
        window_size: int,
        ewma_alpha: float,
        drift_threshold: float,
    ):
        self.name = name
        self.window_size = window_size
        self.ewma_alpha = ewma_alpha
        self.drift_threshold = drift_threshold

        # Welford 算法状态
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0  # 方差的增量累加

        # 滑动窗口
        self._window: List[float] = []

        # EWMA
        self._ewma_mean: Optional[float] = None
        self._ewma_var: Optional[float] = None

        # Page-Hinkley
        self._ph_sum = 0.0
        self._ph_min = 0.0
        self.drift_detected = False
        self._drift_point: Optional[int] = None

        # 缺失计数
        self.n_missing = 0

        # 极端值追踪 (简化: 维护 window 内的 min/max)
        self._historical_std: Optional[float] = None

    def update_batch(self, values: np.ndarray) -> None:
        """更新一批值。"""
        for v in values:
            self._update_single(v)

    def _update_single(self, value: float) -> None:
        """更新单个值（Welford 在线算法）。"""
        if np.isnan(value):
            self.n_missing += 1
            return

        self.n += 1

        # Welford 更新
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.M2 += delta * delta2

        # 滑动窗口
        self._window.append(value)
        if len(self._window) > self.window_size:
            self._window.pop(0)

        # EWMA 更新
        if self._ewma_mean is None:
            self._ewma_mean = value
            self._ewma_var = 0.0
        else:
            diff = value - self._ewma_mean
            self._ewma_mean += self.ewma_alpha * diff
            self._ewma_var = (1 - self.ewma_alpha) * (
                self._ewma_var + self.ewma_alpha * diff * diff
            )

        # Page-Hinkley 漂移检测
        if self.n > self.window_size:
            # 检测均值偏移
            self._ph_sum += value - self.mean
            self._ph_min = min(self._ph_min, self._ph_sum)
            if self._ph_sum - self._ph_min > self.drift_threshold:
                if not self.drift_detected:
                    self.drift_detected = True
                    self._drift_point = self.n

        # 保存历史标准差（在窗口满后）
        if self.n == self.window_size and self._historical_std is None:
            self._historical_std = self.variance**0.5 if self.variance > 0 else 1.0

    @property
    def variance(self) -> float:
        if self.n < 2:
            return 0.0
        return self.M2 / (self.n - 1)

    def compute_score(self) -> tuple:
        """计算当前不确定性分数和警报。"""
        alerts = []
        sub_scores = []

        total_obs = self.n + self.n_missing

        # 1. 缺失分数
        if total_obs > 0:
            missing_rate = self.n_missing / total_obs
            missing_score = min(1.0, missing_rate * 5)  # 20% 缺失 → 1.0
            sub_scores.append(("missing", missing_score, 0.3))
            if missing_rate > 0.1:
                alerts.append(f"[{self.name}] 缺失率 {missing_rate:.1%}")

        # 2. 方差变化分数（窗口 vs 历史）
        variance_score = 0.0
        if len(self._window) >= 20 and self._historical_std is not None:
            window_std = float(np.std(self._window))
            ratio = window_std / (self._historical_std + 1e-10)
            # 方差比 > 2 或 < 0.5 说明变化显著
            if ratio > 1:
                variance_score = min(1.0, (ratio - 1) / 3)  # 4× → 1.0
            else:
                variance_score = min(1.0, (1 / ratio - 1) / 3)
            sub_scores.append(("variance_change", variance_score, 0.25))
            if ratio > 3:
                alerts.append(f"[{self.name}] 方差激增 {ratio:.1f}×")

        # 3. 异常值分数（窗口内）
        anomaly_score = 0.0
        if len(self._window) >= 20:
            arr = np.array(self._window)
            q25, q75 = np.percentile(arr, [25, 75])
            iqr = q75 - q25
            if iqr > 1e-10:
                lower = q25 - 3 * iqr
                upper = q75 + 3 * iqr
                outlier_frac = np.mean((arr < lower) | (arr > upper))
                anomaly_score = min(1.0, outlier_frac * 10)  # 10% 极端值 → 1.0
            sub_scores.append(("anomaly", anomaly_score, 0.2))

        # 4. 漂移分数
        drift_score = 0.0
        if self.drift_detected:
            drift_score = 0.8
            alerts.append(f"[{self.name}] 检测到分布漂移 (Page-Hinkley)")
        elif self.n > self.window_size and self._ewma_mean is not None:
            # EWMA 均值偏离全局均值的程度
            global_std = self.variance**0.5 if self.variance > 0 else 1.0
            ewma_deviation = abs(self._ewma_mean - self.mean) / (global_std + 1e-10)
            drift_score = min(1.0, ewma_deviation / 2)
        sub_scores.append(("drift", drift_score, 0.25))

        # 加权综合
        if sub_scores:
            total_w = sum(w for _, _, w in sub_scores)
            composite = sum(s * w for _, s, w in sub_scores) / total_w
        else:
            composite = 0.0

        return composite, alerts
