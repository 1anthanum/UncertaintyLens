"""
Missing value pattern analysis module.

Detects missing patterns in data, assesses the missing mechanism (MCAR/MAR/MNAR),
and computes each feature's missing-uncertainty contribution score.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, Optional


class MissingPatternDetector:
    """
    Missing value pattern detector.

    Capabilities:
    1. Compute per-feature missing rates
    2. Detect co-missing correlation patterns
    3. Assess missing mechanism (random vs. systematic)
    4. Output missing-uncertainty scores
    """

    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.results_ = None

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")
        if df.empty:
            raise ValueError("DataFrame is empty — nothing to analyze")

        results = {
            "summary": self._compute_summary(df),
            "missing_rates": self._compute_missing_rates(df),
            "co_missing_matrix": self._compute_co_missing(df),
            "mcar_test": self._test_mcar(df),
            "uncertainty_scores": {},
        }

        for col in df.columns:
            results["uncertainty_scores"][col] = self._compute_uncertainty_score(
                missing_rate=results["missing_rates"].get(col, 0),
                is_random=results["mcar_test"]["is_mcar"],
            )

        self.results_ = results
        return results

    def _compute_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        total_cells = df.shape[0] * df.shape[1]
        total_missing = df.isna().sum().sum()

        return {
            "total_rows": df.shape[0],
            "total_columns": df.shape[1],
            "total_cells": total_cells,
            "total_missing": int(total_missing),
            "overall_missing_rate": round(total_missing / total_cells, 4) if total_cells > 0 else 0,
            "columns_with_missing": int((df.isna().sum() > 0).sum()),
            "complete_rows": int(df.dropna().shape[0]),
            "complete_row_rate": (
                round(df.dropna().shape[0] / df.shape[0], 4) if df.shape[0] > 0 else 0
            ),
        }

    def _compute_missing_rates(self, df: pd.DataFrame) -> Dict[str, float]:
        rates = {}
        for col in df.columns:
            rate = df[col].isna().mean()
            rates[col] = round(float(rate), 4)
        return rates

    def _compute_co_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_indicator = df.isna().astype(int)
        cols_with_missing = missing_indicator.columns[missing_indicator.sum() > 0]

        if len(cols_with_missing) < 2:
            return pd.DataFrame()

        co_missing = missing_indicator[cols_with_missing].corr()
        return co_missing

    def _test_mcar(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Simplified MCAR test.

        If data is MCAR, rows with missing values and rows without should show
        no significant distributional differences on other features (t-test).
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            return {"is_mcar": True, "p_values": {}, "note": "Not enough features to test"}

        p_values = {}
        non_random_count = 0

        for target_col in numeric_cols:
            if df[target_col].isna().sum() == 0:
                continue

            is_missing = df[target_col].isna()

            for other_col in numeric_cols:
                if other_col == target_col:
                    continue
                if df[other_col].isna().sum() > 0:
                    continue

                group_missing = df.loc[is_missing, other_col].dropna()
                group_present = df.loc[~is_missing, other_col].dropna()

                if len(group_missing) < 5 or len(group_present) < 5:
                    continue

                t_stat, p_val = stats.ttest_ind(group_missing, group_present, equal_var=False)
                key = f"{target_col}_vs_{other_col}"
                p_values[key] = round(float(p_val), 4)

                if p_val < self.significance_level:
                    non_random_count += 1

        total_tests = len(p_values)
        is_mcar = non_random_count < max(1, total_tests * 0.1)

        return {
            "is_mcar": is_mcar,
            "non_random_pairs": non_random_count,
            "total_tests": total_tests,
            "p_values": p_values,
            "interpretation": (
                "Missing pattern is approximately MCAR — uncertainty is relatively manageable"
                if is_mcar
                else "Missing pattern is non-random (MAR/MNAR) — systematic information loss detected"
            ),
        }

    def _compute_uncertainty_score(self, missing_rate: float, is_random: bool) -> float:
        """
        Compute per-feature missing-uncertainty score (0–1).

        Sigmoid mapping: <5% missing ~ 0, 5–20% ramps up, >20% ~ 1.
        Non-random missingness adds a 30% penalty.
        """
        base_score = 1 / (1 + np.exp(-20 * (missing_rate - 0.15)))

        if not is_random and missing_rate > 0:
            penalty = 1.3
        else:
            penalty = 1.0

        score = min(1.0, base_score * penalty)
        return round(float(score), 4)
