"""
Monte Carlo uncertainty quantification module.

Estimates how data uncertainty propagates into downstream statistics
by repeatedly resampling the data with perturbations (missing value
imputation variants, noise injection) and measuring the spread of results.
"""

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
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        point_estimate = statistic_fn(df)
        simulated_values = []

        for _ in range(self.n_simulations):
            perturbed = self._perturb(df, columns)
            try:
                value = statistic_fn(perturbed)
                if np.isfinite(value):
                    simulated_values.append(float(value))
            except Exception:
                continue

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
        sensitivity = (ci_95[1] - ci_95[0]) / abs(mean_val) if mean_val != 0 else float("inf")

        return {
            "point_estimate": float(point_estimate),
            "mean": mean_val,
            "std": float(np.std(values)),
            "confidence_interval_95": ci_95,
            "confidence_interval_99": ci_99,
            "sensitivity_ratio": round(sensitivity, 4),
            "successful_simulations": len(simulated_values),
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
                    noise = self.rng.normal(0, std * self.noise_scale, size=int(observed_mask.sum()))
                    perturbed.loc[observed_mask, col] = series[observed_mask].values + noise

        return perturbed
