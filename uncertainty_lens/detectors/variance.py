"""
Variance hotspot detection module.

Identifies unexpectedly high or unexplainable variance regions in data,
which represent key sources of decision uncertainty.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, Optional, List


class VarianceDetector:
    """
    Variance hotspot detector.

    Capabilities:
    1. Compute coefficient of variation (CV) per feature
    2. Variance decomposition (between-group vs. within-group) if group column provided
    3. Detect temporal variance trends
    4. Output variance-uncertainty scores
    """

    def __init__(self, cv_threshold: float = 0.5):
        self.cv_threshold = cv_threshold
        self.results_ = None

    def analyze(
        self,
        df: pd.DataFrame,
        group_col: Optional[str] = None,
        time_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        results = {
            "basic_stats": self._compute_basic_stats(df, numeric_cols),
            "cv_analysis": self._compute_cv(df, numeric_cols),
            "uncertainty_scores": {},
        }

        if group_col and group_col in df.columns:
            results["variance_decomposition"] = self._decompose_variance(
                df, numeric_cols, group_col
            )

        if time_col and time_col in df.columns:
            results["temporal_variance"] = self._analyze_temporal_variance(
                df, numeric_cols, time_col
            )

        for col in numeric_cols:
            cv = results["cv_analysis"].get(col, {}).get("cv", 0)
            unexplained_ratio = 1.0

            if "variance_decomposition" in results and col in results["variance_decomposition"]:
                unexplained_ratio = results["variance_decomposition"][col].get(
                    "within_group_ratio", 1.0
                )

            results["uncertainty_scores"][col] = self._compute_uncertainty_score(
                cv=cv, unexplained_ratio=unexplained_ratio
            )

        self.results_ = results
        return results

    def _compute_basic_stats(self, df: pd.DataFrame, cols: List[str]) -> Dict[str, Dict]:
        stats_dict = {}
        for col in cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue

            stats_dict[col] = {
                "count": int(len(series)),
                "mean": round(float(series.mean()), 4),
                "std": round(float(series.std()), 4),
                "min": round(float(series.min()), 4),
                "max": round(float(series.max()), 4),
                "skewness": round(float(series.skew()), 4),
                "kurtosis": round(float(series.kurtosis()), 4),
                "range": round(float(series.max() - series.min()), 4),
            }
        return stats_dict

    def _compute_cv(self, df: pd.DataFrame, cols: List[str]) -> Dict[str, Dict]:
        """
        Compute coefficient of variation (CV = std / mean).

        CV is dimensionless, enabling cross-feature comparison.
        CV > 0.5 typically indicates high dispersion.
        CV > 1.0 means std exceeds the mean — extremely unstable data.
        """
        cv_dict = {}
        for col in cols:
            series = df[col].dropna()
            mean = series.mean()

            if mean == 0 or len(series) < 2:
                cv_dict[col] = {"cv": float("inf"), "level": "N/A"}
                continue

            cv = float(series.std() / abs(mean))
            level = (
                "low" if cv < 0.2 else "medium" if cv < 0.5 else "high" if cv < 1.0 else "very high"
            )

            cv_dict[col] = {
                "cv": round(cv, 4),
                "level": level,
                "is_high_variance": cv > self.cv_threshold,
            }
        return cv_dict

    def _decompose_variance(
        self, df: pd.DataFrame, cols: List[str], group_col: str
    ) -> Dict[str, Dict]:
        """
        Variance decomposition: total = between-group + within-group.

        High between-group ratio -> variance is explained by known grouping -> lower uncertainty.
        High within-group ratio -> unexplained variance -> higher uncertainty.
        """
        decomp = {}
        grouped = df.groupby(group_col)

        for col in cols:
            series = df[col].dropna()
            if len(series) < 2:
                continue

            total_var = float(series.var())
            if total_var == 0:
                decomp[col] = {
                    "total_variance": 0,
                    "between_group_ratio": 0,
                    "within_group_ratio": 0,
                }
                continue

            group_means = grouped[col].mean()
            group_counts = grouped[col].count()
            grand_mean = series.mean()

            between_var = float(
                np.average(
                    (group_means - grand_mean) ** 2,
                    weights=group_counts,
                )
            )

            within_var = float(
                np.average(
                    grouped[col].var().fillna(0),
                    weights=group_counts,
                )
            )

            decomp[col] = {
                "total_variance": round(total_var, 4),
                "between_group_variance": round(between_var, 4),
                "within_group_variance": round(within_var, 4),
                "between_group_ratio": round(between_var / total_var, 4) if total_var > 0 else 0,
                "within_group_ratio": round(within_var / total_var, 4) if total_var > 0 else 0,
                "n_groups": int(grouped.ngroups),
            }

        return decomp

    def _analyze_temporal_variance(
        self, df: pd.DataFrame, cols: List[str], time_col: str
    ) -> Dict[str, Any]:
        temporal = {}

        try:
            df_sorted = df.sort_values(time_col)

            for col in cols:
                series = df_sorted[col].dropna()
                if len(series) < 20:
                    continue

                n = len(series)
                window_size = n // 4
                windows = []

                for i in range(4):
                    start = i * window_size
                    end = start + window_size if i < 3 else n
                    window_var = float(series.iloc[start:end].var())
                    windows.append(window_var)

                variance_trend = "stable"
                if windows[-1] > windows[0] * 1.5:
                    variance_trend = "increasing"
                elif windows[-1] < windows[0] * 0.5:
                    variance_trend = "decreasing"

                temporal[col] = {
                    "window_variances": [round(v, 4) for v in windows],
                    "variance_trend": variance_trend,
                }
        except Exception:
            pass

        return temporal

    def _compute_uncertainty_score(self, cv: float, unexplained_ratio: float) -> float:
        if cv == float("inf"):
            cv_score = 1.0
        else:
            cv_score = 1 / (1 + np.exp(-5 * (cv - 0.5)))

        unexplained_score = unexplained_ratio

        score = 0.5 * cv_score + 0.5 * unexplained_score
        return round(float(min(1.0, score)), 4)
