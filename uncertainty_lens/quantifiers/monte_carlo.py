"""
Monte Carlo uncertainty quantification module.

Estimates how data uncertainty propagates into downstream statistics
by repeatedly resampling the data with perturbations (missing value
imputation variants, noise injection) and measuring the spread of results.
"""

import warnings

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Callable


class MonteCarloQuantifier:
    """
    Monte Carlo uncertainty quantifier.

    Runs repeated perturbation-resampling trials to estimate how
    sensitive a summary statistic is to data uncertainty.

    Usage:
        quantifier = MonteCarloQuantifier(n_simulations=500)
        result = quantifier.estimate(df, statistic_fn=lambda d: d["revenue"].mean())
        print(result["confidence_interval_95"])
    """

    def __init__(
        self,
        n_simulations: int = 500,
        missing_strategy: str = "sample",
        noise_scale: float = 0.05,
        random_state: Optional[int] = 42,
    ):
        self.n_simulations = n_simulations
        self.missing_strategy = missing_strategy
        self.noise_scale = noise_scale
        self.rng = np.random.default_rng(random_state)

    def estimate(
        self,
        df: pd.DataFrame,
        statistic_fn: Callable[[pd.DataFrame], float],
        columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulations and return uncertainty bounds.

        Parameters
        ----------
        df : pd.DataFrame
            Input data (may contain NaNs).
        statistic_fn : callable
            A function that takes a DataFrame and returns a scalar statistic.
        columns : list[str], optional
            Numeric columns to perturb. Defaults to all numeric columns.

        Returns
        -------
        dict with keys: point_estimate, mean, std, confidence_interval_95,
        confidence_interval_99, simulated_values, sensitivity_ratio.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")
        if df.empty:
            raise ValueError("DataFrame is empty — nothing to simulate")
        if not callable(statistic_fn):
            raise TypeError("statistic_fn must be callable")

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        point_estimate = statistic_fn(df)
        simulated_values = []
        n_failures = 0

        for _ in range(self.n_simulations):
            perturbed = self._perturb(df, columns)
            try:
                value = statistic_fn(perturbed)
                if np.isfinite(value):
                    simulated_values.append(float(value))
                else:
                    n_failures += 1
            except (ValueError, TypeError, ZeroDivisionError):
                n_failures += 1
                continue

        failure_rate = n_failures / self.n_simulations if self.n_simulations > 0 else 0
        if failure_rate > 0.2:
            warnings.warn(
                f"Monte Carlo: {n_failures}/{self.n_simulations} simulations failed "
                f"({failure_rate:.0%}). Results may be unreliable."
            )

        if len(simulated_values) < 10:
            return {
                "point_estimate": float(point_estimate),
                "error": "Too few successful simulations",
                "successful_simulations": len(simulated_values),
            }

        values = np.array(simulated_values)
        ci_95 = (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))
        ci_99 = (float(np.percentile(values, 0.5)), float(np.percentile(values, 99.5)))

        mean_val = float(np.mean(values))

        # Relative CI width: normalized uncertainty spread
        # Uses IQR of simulated values for robustness, normalized by
        # the IQR of the CI bounds rather than just the mean
        if abs(mean_val) > 1e-15:
            relative_ci_width = (ci_95[1] - ci_95[0]) / abs(mean_val)
        else:
            # Mean ≈ 0: normalize by CI width itself (gives dimensionless ~2)
            relative_ci_width = float("inf") if ci_95[1] == ci_95[0] else 2.0

        return {
            "point_estimate": float(point_estimate),
            "mean": mean_val,
            "std": float(np.std(values)),
            "confidence_interval_95": ci_95,
            "confidence_interval_99": ci_99,
            "relative_ci_width": round(relative_ci_width, 4),
            "successful_simulations": len(simulated_values),
            "failure_rate": round(n_failures / max(1, self.n_simulations), 4),
        }

    def _perturb(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Create one perturbed copy of the data."""
        perturbed = df.copy()

        for col in columns:
            if col not in perturbed.columns:
                continue

            series = perturbed[col]

            # Re-impute missing values with random draws from observed distribution
            missing_mask = series.isna()
            if missing_mask.any():
                observed = series.dropna().values
                if len(observed) > 0:
                    if self.missing_strategy == "sample":
                        fills = self.rng.choice(observed, size=int(missing_mask.sum()))
                    else:  # "mean"
                        fills = np.full(int(missing_mask.sum()), np.mean(observed))
                    perturbed.loc[missing_mask, col] = fills

            # Add small noise to observed values
            observed_mask = ~series.isna()
            if observed_mask.any() and self.noise_scale > 0:
                std = series[observed_mask].std()
                if pd.notna(std) and std > 0:
                    noise = self.rng.normal(
                        0, std * self.noise_scale, size=int(observed_mask.sum())
                    )
                    perturbed.loc[observed_mask, col] = series[observed_mask].values + noise

        return perturbed
