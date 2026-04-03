"""
Conformal distribution-shift detection module.

Uses a model-free, conformal-prediction-inspired approach to detect whether
subgroups in the data are drawn from the same distribution as the overall
population.  Large deviations signal that the group has *distributional
uncertainty* — statistics computed on the whole dataset may not generalize
to that group.

The method works as follows for each numeric feature:

1. **Calibration**: Draw a random calibration half from the full dataset and
   compute empirical quantile residuals (non-conformity scores) for each
   observation: ``|x - median| / MAD``.  These define the "conformity
   envelope" of the population.
2. **Testing**: For each group, compute the same residuals and compare
   their empirical CDF against the calibration set using a two-sample
   Kolmogorov–Smirnov test.  A small *p*-value means the group's spread
   is *not exchangeable* with the population.
3. **Scoring**: Convert the per-feature, per-group results into an
   ``uncertainty_scores`` dict (0–1 per feature) compatible with the
   pipeline's composite index.

No trained ML model is required — only ``numpy`` and ``scipy``.
"""

import warnings

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from typing import Dict, Any, Optional, List


class ConformalShiftDetector:
    """
    Distribution-shift detector based on conformal non-conformity scores.

    Parameters
    ----------
    significance : float
        Significance level for the KS test (default 0.05).
    calibration_fraction : float
        Fraction of data used for calibration (default 0.5).
    seed : int
        Random seed for calibration/test split reproducibility.
    """

    def __init__(
        self,
        significance: float = 0.05,
        calibration_fraction: float = 0.5,
        seed: int = 42,
    ):
        if not 0 < significance < 1:
            raise ValueError(f"significance must be in (0, 1), got {significance}")
        if not 0 < calibration_fraction < 1:
            raise ValueError(
                f"calibration_fraction must be in (0, 1), got {calibration_fraction}"
            )
        self.significance = significance
        self.calibration_fraction = calibration_fraction
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
        Run conformal shift analysis.

        Parameters
        ----------
        df : pd.DataFrame
        group_col : str, optional
            Column that defines subgroups.  If *None*, the detector still
            computes population-level non-conformity statistics but cannot
            detect inter-group shift; all uncertainty scores default to 0.

        Returns
        -------
        dict
            ``"uncertainty_scores"``  – per-feature float in [0, 1]
            ``"group_shift"``         – per-group, per-feature KS results
            ``"population_profile"``  – calibration-set summary stats
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")
        if df.empty:
            raise ValueError("DataFrame is empty — nothing to analyze")
        if group_col is not None and group_col not in df.columns:
            raise ValueError(
                f"group_col '{group_col}' not found in DataFrame columns"
            )

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return {
                "uncertainty_scores": {},
                "group_shift": {},
                "population_profile": {},
            }

        rng = np.random.default_rng(self.seed)

        # ---- calibration / test split ----
        n = len(df)
        cal_idx = rng.choice(n, size=int(n * self.calibration_fraction), replace=False)
        cal_mask = np.zeros(n, dtype=bool)
        cal_mask[cal_idx] = True

        df_cal = df.iloc[cal_mask]

        # ---- compute calibration non-conformity scores per feature ----
        pop_profile: Dict[str, Dict[str, Any]] = {}
        cal_scores: Dict[str, np.ndarray] = {}

        for col in numeric_cols:
            series = df_cal[col].dropna()
            if len(series) < 5:
                continue
            median = float(series.median())
            mad = float(np.median(np.abs(series - median)))
            if mad == 0:
                # Constant or near-constant: fall back to std
                mad = float(series.std())
            if mad == 0:
                mad = 1.0  # truly constant feature

            scores = np.abs(series.values - median) / mad
            cal_scores[col] = scores
            pop_profile[col] = {
                "median": round(median, 4),
                "mad": round(mad, 4),
                "cal_score_mean": round(float(scores.mean()), 4),
                "cal_score_p95": round(float(np.percentile(scores, 95)), 4),
                "n_calibration": len(scores),
            }

        # ---- per-group shift detection ----
        group_shift: Dict[str, Dict[str, Any]] = {}
        # Accumulate per-feature worst-case shift for uncertainty_scores
        feature_shift_pvalues: Dict[str, List[float]] = {c: [] for c in cal_scores}

        if group_col is not None and group_col in df.columns:
            groups = df[group_col].dropna().unique()
            for g in groups:
                g_str = str(g)
                group_shift[g_str] = {}
                g_df = df[df[group_col] == g]

                for col, c_scores in cal_scores.items():
                    g_series = g_df[col].dropna()
                    if len(g_series) < 3:
                        group_shift[g_str][col] = {
                            "ks_statistic": None,
                            "p_value": None,
                            "shifted": False,
                            "note": "Insufficient group data (< 3)",
                        }
                        continue

                    median = pop_profile[col]["median"]
                    mad = pop_profile[col]["mad"]
                    g_scores = np.abs(g_series.values - median) / mad

                    ks_stat, p_val = sp_stats.ks_2samp(c_scores, g_scores)
                    shifted = p_val < self.significance

                    group_shift[g_str][col] = {
                        "ks_statistic": round(float(ks_stat), 4),
                        "p_value": round(float(p_val), 4),
                        "shifted": shifted,
                        "group_score_mean": round(float(g_scores.mean()), 4),
                    }

                    feature_shift_pvalues[col].append(p_val)

        # ---- build uncertainty_scores ----
        uncertainty_scores: Dict[str, float] = {}
        for col in numeric_cols:
            if col not in feature_shift_pvalues or not feature_shift_pvalues[col]:
                # No group info → can't detect shift → score 0
                uncertainty_scores[col] = 0.0
                continue

            pvals = feature_shift_pvalues[col]
            n_shifted = sum(1 for p in pvals if p < self.significance)
            shift_ratio = n_shifted / len(pvals)

            # Combine the worst p-value (Fisher's method is overkill here;
            # use min-p transformed through a sigmoid for a smooth 0-1 score)
            min_p = min(pvals)
            # Sigmoid: p=0.05 → ~0.5, p→0 → 1.0, p→1 → 0.0
            p_score = 1.0 / (1.0 + np.exp(10 * (min_p - self.significance)))

            # Blend: how many groups shifted × how severe the worst shift is
            score = 0.4 * shift_ratio + 0.6 * p_score
            uncertainty_scores[col] = round(float(min(1.0, score)), 4)

        results = {
            "uncertainty_scores": uncertainty_scores,
            "group_shift": group_shift,
            "population_profile": pop_profile,
            "method": "conformal_ks",
            "significance": self.significance,
        }
        self.results_ = results
        return results
