"""
Unified analysis pipeline.

Chains the three detectors together and outputs a comprehensive
uncertainty analysis report.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from uncertainty_lens.detectors import (
    MissingPatternDetector,
    AnomalyDetector,
    VarianceDetector,
)


class UncertaintyPipeline:
    """
    Uncertainty analysis pipeline.

    Usage:
        pipeline = UncertaintyPipeline()
        report = pipeline.analyze(df, group_col="channel")
        print(report["uncertainty_index"])
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        missing_kwargs: Optional[Dict] = None,
        anomaly_kwargs: Optional[Dict] = None,
        variance_kwargs: Optional[Dict] = None,
    ):
        self.weights = weights or {
            "missing": 0.4,
            "anomaly": 0.3,
            "variance": 0.3,
        }

        self.missing_detector = MissingPatternDetector(**(missing_kwargs or {}))
        self.anomaly_detector = AnomalyDetector(**(anomaly_kwargs or {}))
        self.variance_detector = VarianceDetector(**(variance_kwargs or {}))

        self.report_ = None

    def analyze(
        self,
        df: pd.DataFrame,
        group_col: Optional[str] = None,
        time_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        missing_results = self.missing_detector.analyze(df)
        anomaly_results = self.anomaly_detector.analyze(df)
        variance_results = self.variance_detector.analyze(
            df, group_col=group_col, time_col=time_col
        )

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        uncertainty_index = {}

        for col in numeric_cols:
            m_score = missing_results["uncertainty_scores"].get(col, 0)
            a_score = anomaly_results["uncertainty_scores"].get(col, 0)
            v_score = variance_results["uncertainty_scores"].get(col, 0)

            composite = (
                self.weights["missing"] * m_score
                + self.weights["anomaly"] * a_score
                + self.weights["variance"] * v_score
            )
            uncertainty_index[col] = {
                "composite_score": round(float(composite), 4),
                "missing_score": m_score,
                "anomaly_score": a_score,
                "variance_score": v_score,
                "level": self._score_to_level(composite),
            }

        uncertainty_index = dict(
            sorted(
                uncertainty_index.items(),
                key=lambda x: x[1]["composite_score"],
                reverse=True,
            )
        )

        report = {
            "uncertainty_index": uncertainty_index,
            "missing_analysis": missing_results,
            "anomaly_analysis": anomaly_results,
            "variance_analysis": variance_results,
            "summary": self._generate_summary(uncertainty_index, df),
        }

        self.report_ = report
        return report

    def _score_to_level(self, score: float) -> str:
        if score < 0.2:
            return "Low"
        elif score < 0.4:
            return "Medium-Low"
        elif score < 0.6:
            return "Medium"
        elif score < 0.8:
            return "Medium-High"
        else:
            return "High"

    def _generate_summary(
        self, uncertainty_index: Dict, df: pd.DataFrame
    ) -> Dict[str, Any]:
        if not uncertainty_index:
            return {"message": "No numeric features to analyze"}

        scores = [v["composite_score"] for v in uncertainty_index.values()]
        avg_score = np.mean(scores)

        top_uncertain = list(uncertainty_index.items())[:3]
        bottom_uncertain = list(uncertainty_index.items())[-3:]

        return {
            "overall_uncertainty": round(float(avg_score), 4),
            "overall_level": self._score_to_level(avg_score),
            "total_features_analyzed": len(uncertainty_index),
            "high_uncertainty_features": [
                col
                for col, v in uncertainty_index.items()
                if v["composite_score"] >= 0.6
            ],
            "low_uncertainty_features": [
                col
                for col, v in uncertainty_index.items()
                if v["composite_score"] < 0.2
            ],
            "top_3_uncertain": [
                {"feature": col, **vals} for col, vals in top_uncertain
            ],
            "most_reliable": [
                {"feature": col, **vals} for col, vals in bottom_uncertain
            ],
        }
