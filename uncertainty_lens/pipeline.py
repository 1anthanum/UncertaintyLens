"""
Unified analysis pipeline.

Chains detectors together and outputs a comprehensive uncertainty analysis
report.  Supports both a simple constructor API (backward-compatible) and a
``register()`` API for adding custom detectors at runtime.

Backward-compatible usage (unchanged)::

    pipeline = UncertaintyPipeline()
    report = pipeline.analyze(df, group_col="channel")

Extensible usage::

    from my_detectors import DriftDetector

    pipeline = UncertaintyPipeline()
    pipeline.register("drift", DriftDetector(), weight=0.2)
    report = pipeline.analyze(df)

Custom detectors must satisfy the ``UncertaintyDetector`` protocol:

    - An ``analyze(df, **kwargs)`` method that returns a ``dict``
      containing at least an ``"uncertainty_scores"`` key mapping
      column names to floats in [0, 1].
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Protocol, runtime_checkable

from uncertainty_lens.detectors import (
    MissingPatternDetector,
    AnomalyDetector,
    VarianceDetector,
)

# ---------------------------------------------------------------------------
# Detector protocol – any object with this shape can be registered
# ---------------------------------------------------------------------------


@runtime_checkable
class UncertaintyDetector(Protocol):
    """
    Structural typing protocol for uncertainty detectors.

    Any class that implements ``analyze(df, **kwargs) -> dict`` with an
    ``"uncertainty_scores"`` key in the return value satisfies this protocol.
    You do **not** need to inherit from this class.
    """

    def analyze(self, df: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class UncertaintyPipeline:
    """
    Uncertainty analysis pipeline.

    Parameters
    ----------
    weights : dict, optional
        Mapping of ``{"missing": float, "anomaly": float, "variance": float}``.
        Values are normalized internally so they sum to 1.  Negative values
        raise ``ValueError``.
    missing_kwargs, anomaly_kwargs, variance_kwargs : dict, optional
        Extra keyword arguments forwarded to the built-in detectors.

    Examples
    --------
    Basic (backward-compatible)::

        pipeline = UncertaintyPipeline()
        report = pipeline.analyze(df, group_col="channel")

    With custom detector::

        pipeline = UncertaintyPipeline()
        pipeline.register("drift", DriftDetector(), weight=0.2)
        report = pipeline.analyze(df)
    """

    # ── constructor ────────────────────────────────────────────────────

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        missing_kwargs: Optional[Dict] = None,
        anomaly_kwargs: Optional[Dict] = None,
        variance_kwargs: Optional[Dict] = None,
    ):
        # ----- registry: ordered dict of {name: (detector, weight)} -----
        self._registry: Dict[str, dict] = {}

        # Register the three built-in detectors
        raw_weights = weights or {
            "missing": 0.4,
            "anomaly": 0.3,
            "variance": 0.3,
        }

        # Validate required keys for built-in detectors
        for key in ("missing", "anomaly", "variance"):
            if key not in raw_weights:
                raise ValueError(f"Missing required weight key: '{key}'")
            if raw_weights[key] < 0:
                raise ValueError(f"Weight '{key}' must be non-negative, got {raw_weights[key]}")

        total = sum(raw_weights[k] for k in ("missing", "anomaly", "variance"))
        if total == 0:
            raise ValueError("At least one weight must be greater than zero")

        # Store *raw* weights; normalization happens in analyze()
        self._registry["missing"] = {
            "detector": MissingPatternDetector(**(missing_kwargs or {})),
            "weight": raw_weights["missing"],
        }
        self._registry["anomaly"] = {
            "detector": AnomalyDetector(**(anomaly_kwargs or {})),
            "weight": raw_weights["anomaly"],
        }
        self._registry["variance"] = {
            "detector": VarianceDetector(**(variance_kwargs or {})),
            "weight": raw_weights["variance"],
        }

        self.report_: Optional[Dict[str, Any]] = None

    # ── backward-compatible weight property ────────────────────────────

    @property
    def weights(self) -> Dict[str, float]:
        """Return normalized weights for all registered detectors."""
        raw = {name: entry["weight"] for name, entry in self._registry.items()}
        total = sum(raw.values())
        if total == 0:
            return raw
        return {k: v / total for k, v in raw.items()}

    # ── registration API ───────────────────────────────────────────────

    def register(
        self,
        name: str,
        detector: Any,
        weight: float = 0.2,
    ) -> "UncertaintyPipeline":
        """
        Register a custom detector.

        Parameters
        ----------
        name : str
            Unique name for this detector (e.g. ``"drift"``).  If the name
            already exists the registration is **replaced** (useful for
            swapping built-in detectors with custom implementations).
        detector : UncertaintyDetector
            Any object with an ``analyze(df, **kwargs)`` method that returns
            a dict containing an ``"uncertainty_scores"`` key.
        weight : float
            Relative weight in the composite score (default 0.2).
            Must be non-negative.

        Returns
        -------
        self
            For method chaining.

        Raises
        ------
        TypeError
            If `detector` does not have an ``analyze`` method.
        ValueError
            If `weight` is negative.
        """
        if not hasattr(detector, "analyze") or not callable(detector.analyze):
            raise TypeError(
                f"Detector '{name}' must have a callable `analyze` method. "
                f"Got {type(detector).__name__}."
            )
        if weight < 0:
            raise ValueError(f"Weight for '{name}' must be non-negative, got {weight}")

        self._registry[name] = {"detector": detector, "weight": weight}
        return self

    def unregister(self, name: str) -> "UncertaintyPipeline":
        """
        Remove a registered detector by name.

        Raises ``KeyError`` if the name is not registered.
        """
        if name not in self._registry:
            raise KeyError(
                f"No detector registered under '{name}'. "
                f"Registered: {list(self._registry.keys())}"
            )
        del self._registry[name]
        return self

    @property
    def registered_detectors(self) -> list:
        """Return a list of ``(name, detector_class, weight)`` tuples."""
        return [
            (name, type(entry["detector"]).__name__, entry["weight"])
            for name, entry in self._registry.items()
        ]

    # ── core analysis ──────────────────────────────────────────────────

    def analyze(
        self,
        df: pd.DataFrame,
        group_col: Optional[str] = None,
        time_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run all registered detectors and produce a unified report.

        Parameters
        ----------
        df : pd.DataFrame
        group_col : str, optional
            Grouping column (passed to detectors that accept it).
        time_col : str, optional
            Time column (passed to detectors that accept it).

        Returns
        -------
        dict
            Keys: ``"uncertainty_index"``, ``"summary"``, and one
            ``"<name>_analysis"`` entry per registered detector.
        """
        # ---- input validation ----
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")
        if df.empty:
            raise ValueError("DataFrame is empty — nothing to analyze")
        if df.select_dtypes(include=[np.number]).columns.size == 0:
            raise ValueError("DataFrame has no numeric columns — nothing to analyze")
        if group_col is not None and group_col not in df.columns:
            raise ValueError(f"group_col '{group_col}' not found in DataFrame columns")

        if not self._registry:
            raise ValueError("No detectors registered — call register() first")

        # ---- run each detector ----
        analysis_results: Dict[str, Dict[str, Any]] = {}
        extra_kwargs: Dict[str, Any] = {}
        if group_col is not None:
            extra_kwargs["group_col"] = group_col
        if time_col is not None:
            extra_kwargs["time_col"] = time_col

        for name, entry in self._registry.items():
            detector = entry["detector"]

            # Pass only kwargs the detector's analyze() actually accepts
            # (avoids TypeError for detectors that don't take group_col)
            import inspect

            sig = inspect.signature(detector.analyze)
            params = sig.parameters

            call_kwargs: Dict[str, Any] = {}
            for k, v in extra_kwargs.items():
                if k in params or any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
                ):
                    call_kwargs[k] = v

            result = detector.analyze(df, **call_kwargs)
            analysis_results[name] = result

        # ---- build composite uncertainty index ----
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        normalized_weights = self.weights  # already normalized via property
        uncertainty_index: Dict[str, Dict[str, Any]] = {}

        for col in numeric_cols:
            scores: Dict[str, float] = {}
            active_weights: Dict[str, float] = {}

            for name, result in analysis_results.items():
                score = result.get("uncertainty_scores", {}).get(col, 0.0)
                scores[f"{name}_score"] = score
                # Adaptive weighting: detectors that returned score=0
                # (e.g. missing detector on clean data) get reduced weight
                # to avoid diluting the composite with uninformative zeros.
                raw_w = normalized_weights[name]
                if score > 0:
                    active_weights[name] = raw_w
                else:
                    # Zero-score detectors keep 10% weight
                    # (they confirm low risk, not zero information)
                    active_weights[name] = raw_w * 0.1

            # Re-normalize active weights
            total_active = sum(active_weights.values())
            if total_active > 0:
                composite = sum(
                    (active_weights[name] / total_active) * scores[f"{name}_score"]
                    for name in active_weights
                )
            else:
                composite = 0.0

            entry = {
                "composite_score": round(float(composite), 4),
                **scores,
                "level": self._score_to_level(composite),
            }
            uncertainty_index[col] = entry

        # Sort high → low
        uncertainty_index = dict(
            sorted(
                uncertainty_index.items(),
                key=lambda x: x[1]["composite_score"],
                reverse=True,
            )
        )

        # ---- assemble report ----
        report: Dict[str, Any] = {
            "uncertainty_index": uncertainty_index,
            "summary": self._generate_summary(uncertainty_index, df),
        }

        # Each detector gets a "<name>_analysis" key
        for name, result in analysis_results.items():
            report[f"{name}_analysis"] = result

        self.report_ = report
        return report

    # ── helpers ─────────────────────────────────────────────────────────

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

    def _generate_summary(self, uncertainty_index: Dict, df: pd.DataFrame) -> Dict[str, Any]:
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
                col for col, v in uncertainty_index.items() if v["composite_score"] >= 0.6
            ],
            "low_uncertainty_features": [
                col for col, v in uncertainty_index.items() if v["composite_score"] < 0.2
            ],
            "top_3_uncertain": [{"feature": col, **vals} for col, vals in top_uncertain],
            "most_reliable": [{"feature": col, **vals} for col, vals in bottom_uncertain],
        }
