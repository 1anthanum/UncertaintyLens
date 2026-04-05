"""
Maximum Mean Discrepancy (MMD) 分布偏移检测器.

相比 ConformalShiftDetector 使用的 KS 检验，MMD 有三个关键优势：

1. **多维联合检测** — KS 只能逐特征单维检验，MMD 可以检测多个特征
   的联合分布偏移（比如特征间相关结构的变化）。
2. **对任意分布偏移敏感** — KS 检验对纯位置偏移（均值不同但形状相同）
   不敏感，MMD 基于核方法可以检测均值、方差、形状等任意差异。
3. **统计一致性** — MMD 对所有连续分布具有一致检验功效（given 足够样本）。

理论基础 (Gretton et al., 2012):
  MMD²(P,Q) = E[k(x,x')] - 2·E[k(x,y)] + E[k(y,y')]
  其中 k 是核函数（默认高斯核），x~P, y~Q。
  MMD² = 0 当且仅当 P = Q（对特征核成立）。

使用置换检验 (permutation test) 而非渐近分布做假设检验，
对小样本更稳健。

Usage
-----
::

    from uncertainty_lens.detectors import MMDShiftDetector

    detector = MMDShiftDetector(n_permutations=200, seed=42)
    result = detector.analyze(df, group_col="region")
    # result["group_shift"]         → 哪些组有偏移
    # result["uncertainty_scores"]  → 每特征的偏移不确定性分数
    # result["joint_mmd"]           → 多维联合 MMD 检测结果
"""

import warnings

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple


class MMDShiftDetector:
    """
    基于 MMD 的分布偏移检测器.

    Parameters
    ----------
    n_permutations : int
        置换检验次数（默认 200）。越多越精确但越慢。
    significance : float
        显著性水平（默认 0.05）。
    bandwidth : str or float
        核带宽。"median" 使用 median heuristic，
        "adaptive" 使用多尺度自适应核（推荐，对不同尺度的偏移更敏感），
        也可以指定具体 float 值。
    seed : int
        随机种子。
    """

    def __init__(
        self,
        n_permutations: int = 200,
        significance: float = 0.05,
        bandwidth: str = "median",
        seed: int = 42,
    ):
        if n_permutations < 10:
            raise ValueError(f"n_permutations must be >= 10, got {n_permutations}")
        if not 0 < significance < 1:
            raise ValueError(f"significance must be in (0, 1), got {significance}")

        self.n_permutations = n_permutations
        self.significance = significance
        self.bandwidth = bandwidth
        self.seed = seed
        self.results_: Optional[Dict[str, Any]] = None

    # ── public API ─────────────────────────────────────────────────────

    def analyze(
        self,
        df: pd.DataFrame,
        group_col: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        运行 MMD 偏移检测.

        Parameters
        ----------
        df : DataFrame
            输入数据。
        group_col : str, optional
            分组列名。如果无分组列，返回默认分数。

        Returns
        -------
        dict
            "uncertainty_scores" — 每特征 [0,1] 分数
            "group_shift"        — 每组偏移检测结果
            "joint_mmd"          — 多维联合 MMD 结果
            "per_feature_mmd"    — 逐特征 MMD 结果
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(df).__name__}")
        if df.empty:
            raise ValueError("DataFrame is empty")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return self._default_result([], "No numeric columns")

        if group_col is None or group_col not in df.columns:
            return self._default_result(numeric_cols, "No group column provided")

        groups = df[group_col].dropna().unique()
        if len(groups) < 2:
            return self._default_result(numeric_cols, "Need at least 2 groups")

        rng = np.random.default_rng(self.seed)

        # 准备干净的数据矩阵
        df_clean = df[numeric_cols].dropna()
        group_labels = df.loc[df_clean.index, group_col]

        if len(df_clean) < 20:
            return self._default_result(numeric_cols, "Insufficient data")

        # ── 逐特征 MMD ──
        per_feature_results: Dict[str, Dict[str, Any]] = {}
        feature_scores: Dict[str, float] = {}

        for col in numeric_cols:
            x_all = df_clean[col].values
            col_results = {}

            for group_name in groups:
                mask = group_labels == group_name
                x_group = x_all[mask]
                x_rest = x_all[~mask]

                if len(x_group) < 5 or len(x_rest) < 5:
                    continue

                mmd2, p_value = self._mmd_permutation_test(
                    x_group.reshape(-1, 1),
                    x_rest.reshape(-1, 1),
                    rng,
                )
                col_results[str(group_name)] = {
                    "mmd_squared": round(float(mmd2), 6),
                    "p_value": round(float(p_value), 4),
                    "shift_detected": p_value < self.significance,
                }

            per_feature_results[col] = col_results

            # 特征分数: 取所有组中最低 p-value 的效应
            if col_results:
                min_p = min(r["p_value"] for r in col_results.values())
                max_mmd = max(r["mmd_squared"] for r in col_results.values())
                # Sigmoid 映射: p_value → score
                # p=0.01 → ~0.73, p=0.001 → ~0.95, p=0.1 → ~0.27
                # 比较保守：只有 p < 0.05 才开始给高分
                score = float(1.0 / (1.0 + np.exp(12 * (min_p - 0.03))))
                feature_scores[col] = round(min(1.0, score), 4)
            else:
                feature_scores[col] = 0.0

        # ── 多维联合 MMD ──
        joint_results: Dict[str, Dict[str, Any]] = {}

        if len(numeric_cols) >= 2:
            # 标准化后做联合 MMD
            X_all = df_clean[numeric_cols].values
            X_std = (X_all - X_all.mean(axis=0)) / (X_all.std(axis=0) + 1e-10)

            for group_name in groups:
                mask = (group_labels == group_name).values
                X_group = X_std[mask]
                X_rest = X_std[~mask]

                if len(X_group) < 5 or len(X_rest) < 5:
                    continue

                # 对大数据集子采样以控制计算量
                max_n = 2000
                if len(X_group) > max_n:
                    idx = rng.choice(len(X_group), max_n, replace=False)
                    X_group = X_group[idx]
                if len(X_rest) > max_n:
                    idx = rng.choice(len(X_rest), max_n, replace=False)
                    X_rest = X_rest[idx]

                mmd2, p_value = self._mmd_permutation_test(X_group, X_rest, rng)
                joint_results[str(group_name)] = {
                    "mmd_squared": round(float(mmd2), 6),
                    "p_value": round(float(p_value), 4),
                    "shift_detected": p_value < self.significance,
                    "n_features": len(numeric_cols),
                }

        # ── 组偏移汇总 ──
        group_shift = {}
        for group_name in groups:
            gn = str(group_name)
            shifted_features = [
                col
                for col, res in per_feature_results.items()
                if gn in res and res[gn]["shift_detected"]
            ]
            joint_shifted = gn in joint_results and joint_results[gn]["shift_detected"]
            if shifted_features or joint_shifted:
                group_shift[gn] = {
                    "shifted_features": shifted_features,
                    "joint_shift": joint_shifted,
                    "n_shifted": len(shifted_features),
                }

        results = {
            "uncertainty_scores": feature_scores,
            "group_shift": group_shift,
            "per_feature_mmd": per_feature_results,
            "joint_mmd": joint_results,
            "method": "mmd_permutation",
            "n_permutations": self.n_permutations,
            "significance": self.significance,
        }
        self.results_ = results
        return results

    # ── MMD 核心计算 ────────────────────────────────────────────────

    def _mmd_permutation_test(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[float, float]:
        """
        计算 MMD² 并用置换检验估计 p-value.

        使用高斯核 k(x,y) = exp(-||x-y||² / (2σ²))，
        σ 由 median heuristic 自动选择。
        """
        n_x, n_y = len(X), len(Y)
        XY = np.vstack([X, Y])
        n = n_x + n_y

        # 子采样以控制 O(n²) 的计算量
        max_n = 1000
        if n > max_n:
            idx_x = rng.choice(n_x, min(n_x, max_n // 2), replace=False)
            idx_y = rng.choice(n_y, min(n_y, max_n // 2), replace=False)
            X = X[idx_x]
            Y = Y[idx_y]
            n_x, n_y = len(X), len(Y)
            XY = np.vstack([X, Y])
            n = n_x + n_y

        # 计算距离矩阵
        dists = self._pairwise_sq_distances(XY)

        if self.bandwidth == "adaptive":
            # 自适应多尺度核: 使用多个 bandwidth, 取最大 MMD²
            # 这样对粗粒度和细粒度的偏移都敏感
            # (参考 Gretton et al. 2012, Section 6; Sutherland et al. 2017)
            triu_idx = np.triu_indices(n, k=1)
            nonzero_dists = dists[triu_idx]
            med = float(np.median(nonzero_dists))
            if med < 1e-10:
                med = 1.0
            # 5 个尺度: 0.25×, 0.5×, 1×, 2×, 4× median
            bandwidths = [med * scale for scale in [0.25, 0.5, 1.0, 2.0, 4.0]]
            return self._adaptive_mmd_test(dists, n_x, n_y, bandwidths, rng)
        else:
            # 单一 bandwidth
            if self.bandwidth == "median":
                triu_idx = np.triu_indices(n, k=1)
                nonzero_dists = dists[triu_idx]
                sigma2 = float(np.median(nonzero_dists))
                if sigma2 < 1e-10:
                    sigma2 = 1.0
            else:
                sigma2 = float(self.bandwidth) ** 2

            K = np.exp(-dists / (2 * sigma2))
            mmd2_observed = self._compute_mmd2(K, n_x, n_y)

            count_ge = 0
            indices = np.arange(n)
            for _ in range(self.n_permutations):
                rng.shuffle(indices)
                K_perm = K[np.ix_(indices, indices)]
                mmd2_perm = self._compute_mmd2(K_perm, n_x, n_y)
                if mmd2_perm >= mmd2_observed:
                    count_ge += 1

            p_value = (count_ge + 1) / (self.n_permutations + 1)
            return mmd2_observed, p_value

    def _adaptive_mmd_test(
        self,
        dists: np.ndarray,
        n_x: int,
        n_y: int,
        bandwidths: List[float],
        rng: np.random.Generator,
    ) -> Tuple[float, float]:
        """
        多尺度自适应 MMD 检验.

        对每个 bandwidth 计算 MMD², 取最大值作为统计量。
        置换检验在所有尺度上同步进行, 保持 family-wise 控制。
        """
        n = n_x + n_y

        # 预计算所有尺度的核矩阵
        kernels = []
        for sigma2 in bandwidths:
            K = np.exp(-dists / (2 * sigma2))
            kernels.append(K)

        # 观测到的 max-MMD²
        observed_mmds = [self._compute_mmd2(K, n_x, n_y) for K in kernels]
        mmd2_observed = max(observed_mmds)

        # 置换检验: 在每次置换中取所有尺度的 max
        count_ge = 0
        indices = np.arange(n)
        for _ in range(self.n_permutations):
            rng.shuffle(indices)
            perm_max = max(
                self._compute_mmd2(K[np.ix_(indices, indices)], n_x, n_y) for K in kernels
            )
            if perm_max >= mmd2_observed:
                count_ge += 1

        p_value = (count_ge + 1) / (self.n_permutations + 1)
        return mmd2_observed, p_value

    @staticmethod
    def _pairwise_sq_distances(X: np.ndarray) -> np.ndarray:
        """计算 ||x_i - x_j||² 矩阵."""
        sq_norms = np.sum(X**2, axis=1)
        dists = sq_norms[:, None] + sq_norms[None, :] - 2 * X @ X.T
        np.maximum(dists, 0, out=dists)  # 数值稳定性
        return dists

    @staticmethod
    def _compute_mmd2(K: np.ndarray, n_x: int, n_y: int) -> float:
        """从核矩阵计算无偏 MMD² 估计."""
        Kxx = K[:n_x, :n_x]
        Kyy = K[n_x:, n_x:]
        Kxy = K[:n_x, n_x:]

        # 无偏估计: 排除对角线
        sum_xx = float(Kxx.sum() - np.trace(Kxx))
        sum_yy = float(Kyy.sum() - np.trace(Kyy))
        sum_xy = float(Kxy.sum())

        n, m = n_x, n_y
        if n < 2 or m < 2:
            return 0.0

        mmd2 = sum_xx / (n * (n - 1)) + sum_yy / (m * (m - 1)) - 2 * sum_xy / (n * m)
        return mmd2

    # ── helpers ─────────────────────────────────────────────────────

    def _default_result(self, cols: List[str], note: str) -> Dict[str, Any]:
        return {
            "uncertainty_scores": {c: 0.0 for c in cols},
            "group_shift": {},
            "per_feature_mmd": {},
            "joint_mmd": {},
            "method": "mmd_permutation",
            "note": note,
        }
