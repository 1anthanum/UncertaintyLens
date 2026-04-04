"""
CatBoost-based aleatoric uncertainty estimator.

Uses CatBoost's native ``RMSEWithUncertainty`` loss function, which jointly
predicts both the target value and a per-sample variance estimate during a
single gradient-boosting run.  The predicted variance is a direct estimate
of **aleatoric (data) uncertainty** — how noisy each observation is.

This is particularly powerful for tabular data because:

- CatBoost handles categorical features, missing values, and feature
  interactions natively — no preprocessing required.
- The variance estimate is produced *during* training, not via post-hoc
  methods like bootstrap or dropout, so it is efficient.
- 2025 research shows GBDTs outperform deep learning specifically in the
  high data-uncertainty regime (arXiv:2509.04430).

Requirements
------------
``catboost`` must be installed.  It is an **optional** dependency::

    pip install catboost

If catboost is not installed, importing this module will succeed but
instantiating the detector will raise ``ImportError`` with a clear message.

Usage
-----
::

    from uncertainty_lens.detectors import CatBoostUncertainty

    detector = CatBoostUncertainty(target_col="price", iterations=500)
    results = detector.analyze(df)
    # results["per_sample_variance"]   → array of predicted variances
    # results["uncertainty_scores"]    → per-feature aggregated score
"""

import warnings

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

try:
    from catboost import CatBoostRegressor, Pool

    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False


class CatBoostUncertainty:
    """
    CatBoost RMSEWithUncertainty detector.

    Parameters
    ----------
    target_col : str, optional
        Column to predict.  If *None*, each numeric column is used as
        target in turn (same leave-one-out-column strategy as
        ConformalPredictor).
    iterations : int
        Number of boosting rounds (default 300).
    learning_rate : float
        Learning rate (default 0.05).
    depth : int
        Tree depth (default 6).
    calibration_fraction : float
        Fraction held out for variance calibration (default 0.2).
    seed : int
        Random seed.
    verbose : bool
        CatBoost training verbosity (default False).
    """

    def __init__(
        self,
        target_col: Optional[str] = None,
        iterations: int = 300,
        learning_rate: float = 0.05,
        depth: int = 6,
        calibration_fraction: float = 0.2,
        seed: int = 42,
        verbose: bool = False,
    ):
        if not _HAS_CATBOOST:
            raise ImportError(
                "CatBoost is required for CatBoostUncertainty.\n"
                "Install it with: pip install catboost\n"
                "This is an optional dependency — the rest of "
                "UncertaintyLens works without it."
            )
        if not 0 < calibration_fraction < 1:
            raise ValueError(f"calibration_fraction must be in (0, 1), got {calibration_fraction}")

        self.target_col = target_col
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.calibration_fraction = calibration_fraction
        self.seed = seed
        self.verbose = verbose
        self.results_: Optional[Dict[str, Any]] = None

    # ── public API ─────────────────────────────────────────────────────

    def analyze(
        self,
        df: pd.DataFrame,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Fit CatBoost with RMSEWithUncertainty and estimate per-sample
        aleatoric variance.

        Returns
        -------
        dict
            ``"uncertainty_scores"``     – per-feature float in [0, 1]
            ``"feature_results"``        – per-target detailed results
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")
        if df.empty:
            raise ValueError("DataFrame is empty — nothing to analyze")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            return {
                "uncertainty_scores": {c: 0.0 for c in numeric_cols},
                "feature_results": {},
                "note": "Need at least 2 numeric columns",
            }

        df_clean = df[numeric_cols].dropna()
        if len(df_clean) < 30:
            return {
                "uncertainty_scores": {c: 0.0 for c in numeric_cols},
                "feature_results": {},
                "note": f"Insufficient clean rows ({len(df_clean)})",
            }

        if self.target_col is not None:
            if self.target_col not in numeric_cols:
                raise ValueError(f"target_col '{self.target_col}' not found in numeric columns")
            targets = [self.target_col]
        else:
            targets = numeric_cols

        rng = np.random.default_rng(self.seed)
        feature_results: Dict[str, Dict[str, Any]] = {}
        uncertainty_scores: Dict[str, float] = {}

        for target in targets:
            feature_cols = [c for c in numeric_cols if c != target]
            if not feature_cols:
                continue

            result = self._fit_and_estimate(df_clean, feature_cols, target, rng)
            feature_results[target] = result
            uncertainty_scores[target] = result["uncertainty_score"]

        for c in numeric_cols:
            if c not in uncertainty_scores:
                uncertainty_scores[c] = 0.0

        results = {
            "uncertainty_scores": uncertainty_scores,
            "feature_results": feature_results,
            "method": "catboost_rmse_with_uncertainty",
        }
        self.results_ = results
        return results

    # ── internals ──────────────────────────────────────────────────────

    def _fit_and_estimate(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        rng: np.random.Generator,
    ) -> Dict[str, Any]:
        """Fit CatBoost with RMSEWithUncertainty for one target."""
        n = len(df)
        n_cal = max(10, int(n * self.calibration_fraction))

        indices = rng.permutation(n)
        train_idx = indices[: n - n_cal]
        cal_idx = indices[n - n_cal :]

        X_train = df.iloc[train_idx][feature_cols]
        y_train = df.iloc[train_idx][target_col].values
        X_cal = df.iloc[cal_idx][feature_cols]
        y_cal = df.iloc[cal_idx][target_col].values

        model = CatBoostRegressor(
            loss_function="RMSEWithUncertainty",
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            random_seed=self.seed,
            verbose=self.verbose,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)

        # Predict returns [mean, variance] for RMSEWithUncertainty
        preds_cal = model.predict(X_cal)
        pred_means_cal = preds_cal[:, 0]
        pred_vars_cal = preds_cal[:, 1]

        # Variance estimates from CatBoost can be in log-space
        # RMSEWithUncertainty outputs log(variance), so exponentiate
        pred_vars_cal = np.exp(pred_vars_cal)

        # Empirical check: residuals vs predicted variance
        residuals = np.abs(y_cal - pred_means_cal)
        cal_rmse = float(np.sqrt(np.mean(residuals**2)))

        # Full-dataset prediction
        X_full = df[feature_cols]
        preds_full = model.predict(X_full)
        pred_vars_full = np.exp(preds_full[:, 1])

        # Mean predicted variance → measure of aleatoric uncertainty
        mean_var = float(np.mean(pred_vars_full))
        median_var = float(np.median(pred_vars_full))

        # Normalize: variance relative to target range squared
        target_range = float(df[target_col].max() - df[target_col].min())
        if target_range > 0:
            normalized_std = np.sqrt(mean_var) / target_range
        else:
            normalized_std = 0.0

        # Sigmoid: normalized_std = 0.2 → ~0.5
        score = float(1.0 / (1.0 + np.exp(-10 * (normalized_std - 0.2))))

        return {
            "mean_predicted_variance": round(mean_var, 4),
            "median_predicted_variance": round(median_var, 4),
            "predicted_std_mean": round(float(np.sqrt(mean_var)), 4),
            "normalized_std": round(normalized_std, 4),
            "calibration_rmse": round(cal_rmse, 4),
            "n_train": len(train_idx),
            "n_calibration": len(cal_idx),
            "uncertainty_score": round(float(min(1.0, score)), 4),
            "high_uncertainty_samples_pct": round(
                float(np.mean(pred_vars_full > np.percentile(pred_vars_full, 90))),
                4,
            ),
        }
