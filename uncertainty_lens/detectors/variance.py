"""
Variance hotspot detection module.

Identifies unexpectedly high or unexplainable variance regions in data,
which represent key sources of decision uncertainty.
"""

import warnings

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
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")
        if df.empty:
            raise ValueError("DataFrame is empty — nothing to analyze")
        if group_col is not None and group_col not in df.columns:
            raise ValueError(f"group_col '{group_col}' not found in DataFrame columns")
        if time_col is not None and time_col not in df.columns:
            raise ValueError(f"time_col '{time_col}' not found in DataFrame columns")

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
            cv_info = results["cv_analysis"].get(col, {})
            cv = cv_info.get("cv", 0)
            cv_method = cv_info.get("cv_method", "standard")
            unexplained_ratio = 1.0

            if "variance_decomposition" in results and col in results["variance_decomposition"]:
                decomp_entry = results["variance_decomposition"][col]
                ratio = decomp_entry.get("within_group_ratio", 1.0)
                # Handle NaN from constant-feature decomposition
                if isinstance(ratio, float) and np.isnan(ratio):
                    ratio = 0.0  # constant feature → no unexplained variance
                unexplained_ratio = ratio

            results["uncertainty_scores"][col] = self._compute_uncertainty_score(
                cv=cv, unexplained_ratio=unexplained_ratio, cv_method=cv_method
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

        For features where mean ≈ 0, CV is undefined.  We fall back to
        standard deviation directly, normalized against a reference scale
        (median absolute value of all non-zero features), and flag the
        result as ``"cv_method": "std_fallback"``.
        """
        cv_dict = {}
        for col in cols:
            series = df[col].dropna()

            if len(series) < 2:
                cv_dict[col] = {
                    "cv": float("inf"),
                    "level": "N/A",
                    "cv_method": "insufficient_data",
                }
                continue

            mean = float(series.mean())
            std = float(series.std())

            # Near-zero mean: CV is numerically unstable when |mean| << std.
            # Threshold: |mean| < 0.1 * std means CV > 10, which is
            # unreliable and dominated by the sign of the mean.
            if (std > 0 and abs(mean) < 0.1 * std) or abs(mean) < 1e-15:
                # Use std directly; classify based on absolute spread
                # relative to median absolute deviation of the series
                mad = float(np.median(np.abs(series - series.median())))
                if mad == 0:
                    mad = std if std > 0 else 1.0

                # Effective dispersion: std / MAD (scale-free)
                effective_cv = std / mad if mad > 0 else 0.0
                # MAD ≈ 0.6745 * std for normal data, so std/MAD ≈ 1.48
                # Use 1.5 as the "normal" baseline
                level = (
                    "low"
                    if effective_cv < 0.75
                    else (
                        "medium"
                        if effective_cv < 1.5
                        else "high" if effective_cv < 3.0 else "very high"
                    )
                )

                cv_dict[col] = {
                    "cv": round(effective_cv, 4),
                    "level": level,
                    "is_high_variance": effective_cv > self.cv_threshold * 3,  # adjusted threshold
                    "cv_method": "std_fallback",
                    "note": "Mean ≈ 0; using std/MAD as dispersion proxy",
                }
                continue

            cv = std / abs(mean)
            level = (
                "low" if cv < 0.2 else "medium" if cv < 0.5 else "high" if cv < 1.0 else "very high"
            )

            cv_dict[col] = {
                "cv": round(cv, 4),
                "level": level,
                "is_high_variance": cv > self.cv_threshold,
                "cv_method": "standard",
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
                    "between_group_ratio": float("nan"),
                    "within_group_ratio": float("nan"),
                    "note": "Constant feature — variance decomposition undefined",
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

                # Determine trend using Spearman rank correlation of
                # window index vs. variance (non-parametric monotonic trend)
                from scipy.stats import spearmanr

                variance_trend = "stable"
                if len(windows) >= 3:
                    rho, p_val = spearmanr(range(len(windows)), windows)
                    if p_val < 0.1:  # loose threshold for only 4 windows
                        variance_trend = "increasing" if rho > 0 else "decreasing"
                else:
                    # Fallback for very few windows
                    p_val = float("nan")
                    if windows[-1] > windows[0] * 1.5:
                        variance_trend = "increasing"
                    elif windows[-1] < windows[0] * 0.5:
                        variance_trend = "decreasing"

                temporal[col] = {
                    "window_variances": [round(v, 4) for v in windows],
                    "variance_trend": variance_trend,
                    "trend_p_value": round(float(p_val), 4) if np.isfinite(p_val) else None,
                }
        except (KeyError, TypeError) as e:
            warnings.warn(f"Temporal variance analysis skipped: {e}")

        return temporal

    def _compute_uncertainty_score(
        self, cv: float, unexplained_ratio: float, cv_method: str = "standard"
    ) -> float:
        if cv == float("inf"):
            # Insufficient data — can't determine, assign moderate uncertainty
            cv_score = 0.5
        elif cv_method == "std_fallback":
            # For zero-mean features, cv = std/MAD.
            # For normal data, std/MAD ≈ 1.48 — this is the *baseline*.
            # Only flag as high uncertainty if dispersion is much larger
            # than expected for a normal distribution.
            # Sigmoid: std/MAD = 3.0 → score ≈ 0.5 (heavy-tailed → uncertain)
            cv_score = 1 / (1 + np.exp(-3 * (cv - 3.0)))
        else:
            # Standard CV: 0.5 → score ≈ 0.5, steepness = 5
            cv_score = 1 / (1 + np.exp(-5 * (cv - 0.5)))

        unexplained_score = unexplained_ratio

        score = 0.5 * cv_score + 0.5 * unexplained_score
        return round(float(min(1.0, score)), 4)
