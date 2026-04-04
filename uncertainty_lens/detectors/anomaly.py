"""
Anomaly detection module.

Uses multiple algorithms (IQR, Isolation Forest, LOF) with ensemble voting
to detect outliers. Outliers are treated as uncertainty signals rather than
noise to be removed.
"""

import warnings

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from typing import Dict, Any, List, Optional


class AnomalyDetector:
    """
    Multi-method ensemble anomaly detector.

    Voting across three methods:
    1. IQR (interquartile range) — univariate
    2. Isolation Forest — multivariate
    3. LOF (Local Outlier Factor) — density-based

    Points flagged by multiple methods carry higher uncertainty contribution.
    """

    def __init__(
        self,
        iqr_factor: float = 1.5,
        contamination: float = 0.05,
        min_votes: int = 2,
        random_state: int = 42,
    ):
        self.iqr_factor = iqr_factor
        self.contamination = contamination
        self.min_votes = min_votes
        self.random_state = random_state
        self.results_ = None

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        numeric_df = df.select_dtypes(include=[np.number]).dropna()

        if numeric_df.empty or numeric_df.shape[0] < 10:
            return {
                "anomaly_counts": {},
                "uncertainty_scores": {},
                "note": "Insufficient data for anomaly detection",
            }

        iqr_flags = self._detect_iqr(numeric_df)
        iso_flags = self._detect_isolation_forest(numeric_df)
        lof_flags = self._detect_lof(numeric_df)

        vote_matrix = iqr_flags.astype(int) + iso_flags.astype(int) + lof_flags.astype(int)
        consensus_flags = vote_matrix >= self.min_votes

        results = {
            "method_results": {
                "iqr": {col: int(iqr_flags[col].sum()) for col in numeric_df.columns},
                "isolation_forest": int(iso_flags.any(axis=1).sum()),
                "lof": int(lof_flags.any(axis=1).sum()),
            },
            "consensus_anomalies": {
                col: int(consensus_flags[col].sum()) for col in numeric_df.columns
            },
            "vote_matrix": vote_matrix,
            "anomaly_rates": {},
            "uncertainty_scores": {},
        }

        n_rows = numeric_df.shape[0]
        for col in numeric_df.columns:
            rate = consensus_flags[col].sum() / n_rows
            results["anomaly_rates"][col] = round(float(rate), 4)
            results["uncertainty_scores"][col] = self._compute_uncertainty_score(
                anomaly_rate=rate,
                vote_distribution=vote_matrix[col].value_counts().to_dict(),
                n_rows=n_rows,
            )

        self.results_ = results
        return results

    def _detect_iqr(self, df: pd.DataFrame) -> pd.DataFrame:
        flags = pd.DataFrame(False, index=df.index, columns=df.columns)

        for col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1

            if iqr == 0:
                continue

            lower = q1 - self.iqr_factor * iqr
            upper = q3 + self.iqr_factor * iqr
            flags[col] = (df[col] < lower) | (df[col] > upper)

        return flags

    def _detect_isolation_forest(self, df: pd.DataFrame) -> pd.DataFrame:
        flags = pd.DataFrame(False, index=df.index, columns=df.columns)

        try:
            iso = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
                n_estimators=100,
            )
            predictions = iso.fit_predict(df)
            is_anomaly = predictions == -1

            for col in df.columns:
                flags[col] = is_anomaly
        except ValueError as e:
            warnings.warn(f"Isolation Forest skipped: {e}")

        return flags

    def _detect_lof(self, df: pd.DataFrame) -> pd.DataFrame:
        flags = pd.DataFrame(False, index=df.index, columns=df.columns)

        try:
            n_neighbors = min(20, df.shape[0] - 1)
            if n_neighbors < 2:
                return flags

            lof = LocalOutlierFactor(
                n_neighbors=n_neighbors,
                contamination=self.contamination,
            )
            predictions = lof.fit_predict(df)
            is_anomaly = predictions == -1

            for col in df.columns:
                flags[col] = is_anomaly
        except ValueError as e:
            warnings.warn(f"LOF skipped: {e}")

        return flags

    def _compute_uncertainty_score(
        self,
        anomaly_rate: float,
        vote_distribution: Dict,
        n_rows: int,
    ) -> float:
        # Sigmoid: anomaly_rate = 8% → score ≈ 0.5
        # Steepness 30: sharp transition (anomaly rates are typically small)
        base_score = 1 / (1 + np.exp(-30 * (anomaly_rate - 0.08)))

        # Small-sample penalty: anomaly detection is less reliable with
        # fewer observations.  log-scaling provides diminishing penalty.
        # n=100 → penalty=1.0, n=50 → ~1.15, n=20 → ~1.35
        if n_rows >= 100:
            sample_penalty = 1.0
        else:
            sample_penalty = 1.0 + 0.5 * np.log(100 / max(n_rows, 10)) / np.log(10)

        score = min(1.0, base_score * sample_penalty)
        return round(float(score), 4)
