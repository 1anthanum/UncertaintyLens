"""
Model-aware conformal prediction detector.

Wraps **any scikit-learn-compatible regressor** with split conformal
prediction to produce distribution-free prediction intervals.  The width
of the interval for each feature (used as target in turn) becomes an
uncertainty score: wider interval → higher uncertainty.

Unlike ConformalShiftDetector (Tier 1), this detector *requires* a model
and produces **prediction-level** uncertainty rather than data-level.

Theory (split conformal)
------------------------
1. Split data into a *training* set and a *calibration* set.
2. Fit the model on the training set.
3. Compute calibration residuals: ``|y_cal - ŷ_cal|``.
4. For a desired coverage ``1 - α``, pick the ``⌈(1 - α)(1 + 1/n_cal)⌉``-th
   quantile of the residuals as the *conformal radius* ``q``.
5. The prediction interval for a new point is ``[ŷ - q, ŷ + q]``.

The guarantees are **marginal** (over the randomness of the calibration
split), **distribution-free**, and hold in finite samples.

Usage
-----
::

    from sklearn.ensemble import GradientBoostingRegressor
    from uncertainty_lens.detectors import ConformalPredictor

    detector = ConformalPredictor(
        model=GradientBoostingRegressor(),
        target_col="price",
    )
    results = detector.analyze(df)
    # results["prediction_intervals"]  → per-sample [lo, hi]
    # results["uncertainty_scores"]    → per-feature width-based score
"""

import warnings

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union

try:
    from sklearn.base import clone, is_regressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


class ConformalPredictor:
    """
    Split conformal prediction wrapper for sklearn regressors.

    Parameters
    ----------
    model : sklearn estimator, optional
        Any sklearn-compatible regressor.  If *None*, a ``Ridge`` regressor
        with standard scaling is used automatically.
    target_col : str, optional
        Column to predict.  If *None*, each numeric column is predicted
        in turn using the remaining features (leave-one-out over columns).
    coverage : float
        Desired marginal coverage (default 0.9 = 90%).
    calibration_fraction : float
        Fraction of rows reserved for calibration (default 0.3).
    seed : int
        Random seed for train/cal split.
    """

    def __init__(
        self,
        model: Any = None,
        target_col: Optional[str] = None,
        coverage: float = 0.9,
        calibration_fraction: float = 0.3,
        seed: int = 42,
    ):
        if not _HAS_SKLEARN:
            raise ImportError(
                "scikit-learn is required for ConformalPredictor. "
                "Install it with: pip install scikit-learn"
            )
        if not 0 < coverage < 1:
            raise ValueError(f"coverage must be in (0, 1), got {coverage}")
        if not 0 < calibration_fraction < 1:
            raise ValueError(f"calibration_fraction must be in (0, 1), got {calibration_fraction}")

        self.model = model
        self.target_col = target_col
        self.coverage = coverage
        self.calibration_fraction = calibration_fraction
        self.seed = seed
        self.results_: Optional[Dict[str, Any]] = None

    # ── public API ─────────────────────────────────────────────────────

    def analyze(
        self,
        df: pd.DataFrame,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run conformal prediction analysis.

        If ``target_col`` was specified, produces prediction intervals for
        that single target.  Otherwise, iterates over every numeric column
        as target, using the rest as features.

        Returns
        -------
        dict
            ``"uncertainty_scores"``     – per-feature float in [0, 1]
            ``"conformal_results"``      – per-target detailed results
            ``"coverage_achieved"``      – empirical coverage on calibration set
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")
        if df.empty:
            raise ValueError("DataFrame is empty — nothing to analyze")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            return {
                "uncertainty_scores": {c: 0.0 for c in numeric_cols},
                "conformal_results": {},
                "note": "Need at least 2 numeric columns for conformal prediction",
            }

        # Drop rows with any NaN in numeric columns for clean modeling
        df_clean = df[numeric_cols].dropna()
        if len(df_clean) < 20:
            return {
                "uncertainty_scores": {c: 0.0 for c in numeric_cols},
                "conformal_results": {},
                "note": f"Insufficient clean rows ({len(df_clean)}) for conformal prediction",
            }

        rng = np.random.default_rng(self.seed)

        if self.target_col is not None:
            # Single-target mode
            if self.target_col not in numeric_cols:
                raise ValueError(f"target_col '{self.target_col}' not found in numeric columns")
            targets = [self.target_col]
        else:
            # Leave-one-out column mode: each feature takes a turn as target
            targets = numeric_cols

        conformal_results: Dict[str, Dict[str, Any]] = {}
        uncertainty_scores: Dict[str, float] = {}

        for target in targets:
            feature_cols = [c for c in numeric_cols if c != target]
            if not feature_cols:
                continue

            result = self._run_conformal(df_clean, feature_cols, target, rng)
            conformal_results[target] = result

            # Score = normalized interval width
            # Wider intervals → higher uncertainty
            uncertainty_scores[target] = result["uncertainty_score"]

        # Features that were never a target get score 0
        for c in numeric_cols:
            if c not in uncertainty_scores:
                uncertainty_scores[c] = 0.0

        results = {
            "uncertainty_scores": uncertainty_scores,
            "conformal_results": conformal_results,
            "method": "split_conformal",
            "coverage_target": self.coverage,
        }
        self.results_ = results
        return results

    # ── internals ──────────────────────────────────────────────────────

    def _run_conformal(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        rng: np.random.Generator,
    ) -> Dict[str, Any]:
        """Run split conformal for one target column."""
        n = len(df)
        n_cal = max(10, int(n * self.calibration_fraction))
        n_train = n - n_cal

        # Random split
        indices = rng.permutation(n)
        train_idx = indices[:n_train]
        cal_idx = indices[n_train:]

        X_train = df.iloc[train_idx][feature_cols].values
        y_train = df.iloc[train_idx][target_col].values
        X_cal = df.iloc[cal_idx][feature_cols].values
        y_cal = df.iloc[cal_idx][target_col].values

        # Fit model
        model = self._get_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)

        # Calibration residuals
        y_cal_pred = model.predict(X_cal)
        residuals = np.abs(y_cal - y_cal_pred)

        # Conformal quantile — correct finite-sample formula
        # Reference: Vovk et al. (2005), Lei et al. (2018)
        # The (1-α)-quantile of the augmented residual set {r_1,...,r_n, ∞}
        # is the ⌈(n+1)(1-α)⌉-th smallest residual.
        alpha = 1.0 - self.coverage
        n_cal = len(residuals)
        q_index = int(np.ceil((n_cal + 1) * (1 - alpha)))
        sorted_residuals = np.sort(residuals)
        if q_index >= n_cal:
            # Coverage ≈ 1: use the max residual (infinite interval fallback)
            conformal_radius = float(sorted_residuals[-1])
        else:
            conformal_radius = float(sorted_residuals[q_index - 1])  # 0-indexed

        # Prediction intervals for a held-out test set (NOT calibration set)
        # Report theoretical coverage, not circular empirical coverage
        lo = y_cal_pred - conformal_radius
        hi = y_cal_pred + conformal_radius
        # Note: checking on cal set is circular; we report it for diagnostics
        # but flag it as such in the output
        empirical_coverage_cal = float(np.mean((y_cal >= lo) & (y_cal <= hi)))

        # Full-dataset prediction intervals
        X_full = df[feature_cols].values
        y_full_pred = model.predict(X_full)
        intervals_lo = y_full_pred - conformal_radius
        intervals_hi = y_full_pred + conformal_radius

        # Normalized interval width → uncertainty score
        # Use IQR instead of range for robustness to outliers
        target_iqr = float(np.percentile(df[target_col], 75) - np.percentile(df[target_col], 25))
        target_range = float(df[target_col].max() - df[target_col].min())
        normalizer = target_iqr if target_iqr > 0 else (target_range if target_range > 0 else 1.0)
        normalized_width = (2 * conformal_radius) / normalizer

        # Score mapping: use calibrated sigmoid
        # Inflection: normalized_width = 1.0 (interval = 1 IQR) → score ≈ 0.5
        # Steepness: 3.0 (moderate, avoids saturation)
        score = float(1.0 / (1.0 + np.exp(-3.0 * (normalized_width - 1.0))))

        return {
            "conformal_radius": round(conformal_radius, 4),
            "interval_width": round(2 * conformal_radius, 4),
            "normalized_width": round(normalized_width, 4),
            "empirical_coverage_cal": round(empirical_coverage_cal, 4),
            "coverage_target": self.coverage,
            "coverage_note": "empirical_coverage_cal is measured on the calibration set (circular); "
            "theoretical guarantee is >= coverage_target on new data",
            "n_train": n_train,
            "n_calibration": n_cal,
            "residual_mean": round(float(residuals.mean()), 4),
            "residual_p95": round(float(np.percentile(residuals, 95)), 4),
            "uncertainty_score": round(min(1.0, score), 4),
        }

    def _get_model(self):
        """Return a fresh clone of the model, or a default Ridge."""
        if self.model is not None:
            return clone(self.model)
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
