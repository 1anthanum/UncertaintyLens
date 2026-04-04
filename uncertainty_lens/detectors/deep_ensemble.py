"""
Deep Ensemble uncertainty detector for feature learnability assessment.

Trains an ensemble of neural networks (MLPs) with different random
initializations on each numeric feature as a prediction target.  The
disagreement across ensemble members reveals two things that pure
statistical methods cannot:

1. **Predictive uncertainty** — How much do different models disagree on
   predictions?  High disagreement = the feature's relationship to other
   features is hard to learn.

2. **Feature learnability** — Can a model extract useful patterns from
   this feature using the rest of the data?  Measured by how much the
   ensemble's cross-validated performance exceeds a dummy baseline.

Method
------
For each numeric feature *f* (treated as the target):

1. Train ``n_ensemble`` MLPs, each with a different random seed, using
   the remaining numeric features as inputs.  Each MLP is trained on
   an 80/20 train/validation split (different per ensemble member).

2. Collect predictions from all ensemble members on the held-out set.

3. **Epistemic score** = mean coefficient of variation of ensemble
   predictions across samples.  High CV → models disagree → unstable
   learning signal.

4. **Learnability score** = 1 - clamp(ensemble_R² / baseline_R², 0, 1).
   If ensemble can't predict the feature from others, it's "isolated"
   (high uncertainty about its relationship to the data).

5. **Composite** = weighted blend of epistemic + learnability.

This detector answers: "If I train a model using these features, which
ones will cause unreliable predictions?"

Requirements
------------
Only ``scikit-learn`` (already a dependency).  No PyTorch/TensorFlow.
For tabular data up to ~50K rows, sklearn's MLPRegressor is sufficient.
GPU acceleration can be added later via a PyTorch backend for larger
datasets.

Usage
-----
::

    from uncertainty_lens.detectors import DeepEnsembleDetector

    pipeline = UncertaintyPipeline()
    pipeline.register("deep_ensemble", DeepEnsembleDetector(), weight=0.15)
    report = pipeline.analyze(df)

    # Access per-feature learnability
    lea = report["deep_ensemble_analysis"]["learnability"]
    for feat, info in lea.items():
        print(f"{feat}: R²={info['ensemble_r2']:.3f}, learnable={info['is_learnable']}")
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score


class DeepEnsembleDetector:
    """
    Deep Ensemble detector for feature learnability and predictive uncertainty.

    Parameters
    ----------
    n_ensemble : int
        Number of ensemble members (default 5).  More members give
        more stable uncertainty estimates but increase compute time.
    hidden_layers : tuple
        MLP architecture.  Default ``(64, 32)`` — two hidden layers.
        Kept small intentionally: we're measuring *learnability*, not
        maximizing accuracy.
    max_iter : int
        Maximum training iterations per MLP (default 200).
    test_size : float
        Fraction of data reserved for validation (default 0.2).
    seed : int
        Base random seed.  Each ensemble member uses ``seed + i``.
    learnability_threshold : float
        R² threshold above which a feature is considered "learnable"
        (default 0.1).  Below this, the feature has no meaningful
        relationship to the other features.
    """

    def __init__(
        self,
        n_ensemble: int = 5,
        hidden_layers: tuple = (64, 32),
        max_iter: int = 200,
        test_size: float = 0.2,
        seed: int = 42,
        learnability_threshold: float = 0.1,
    ):
        if n_ensemble < 2:
            raise ValueError(f"n_ensemble must be >= 2, got {n_ensemble}")

        self.n_ensemble = n_ensemble
        self.hidden_layers = hidden_layers
        self.max_iter = max_iter
        self.test_size = test_size
        self.seed = seed
        self.learnability_threshold = learnability_threshold
        self.results_: Optional[Dict[str, Any]] = None

    def analyze(
        self,
        df: pd.DataFrame,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Assess feature learnability and predictive uncertainty.

        Parameters
        ----------
        df : pd.DataFrame
            Input data.  Only numeric columns are analyzed.

        Returns
        -------
        dict
            ``"uncertainty_scores"``  – composite score per feature [0, 1]
            ``"learnability"``        – per-feature ensemble R², baseline R²,
                                        learnability flag
            ``"epistemic"``           – per-feature ensemble disagreement
            ``"recommendations"``     – actionable advice per feature
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(df).__name__}")
        if df.empty:
            raise ValueError("DataFrame is empty")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            # Need at least 2 features to predict one from the others
            return {
                "uncertainty_scores": {c: 0.5 for c in numeric_cols},
                "learnability": {},
                "epistemic": {},
                "recommendations": {},
                "config": self._config(),
            }

        # Drop rows with NaN in numeric columns for training
        df_clean = df[numeric_cols].dropna()
        if len(df_clean) < 30:
            warnings.warn(
                f"DeepEnsemble: only {len(df_clean)} complete rows "
                f"(need >=30). Returning default scores."
            )
            return {
                "uncertainty_scores": {c: 0.5 for c in numeric_cols},
                "learnability": {},
                "epistemic": {},
                "recommendations": {},
                "config": self._config(),
            }

        uncertainty_scores: Dict[str, float] = {}
        learnability: Dict[str, Dict[str, Any]] = {}
        epistemic: Dict[str, Dict[str, Any]] = {}
        recommendations: Dict[str, Dict[str, str]] = {}

        for target_col in numeric_cols:
            feature_cols = [c for c in numeric_cols if c != target_col]
            if not feature_cols:
                continue

            result = self._evaluate_feature(df_clean, target_col, feature_cols)

            learnability[target_col] = result["learnability"]
            epistemic[target_col] = result["epistemic"]

            # Composite score: blend of epistemic uncertainty and unlearnability
            epi_score = result["epistemic"]["score"]
            learn_score = result["learnability"]["uncertainty_score"]
            composite = 0.4 * epi_score + 0.6 * learn_score
            uncertainty_scores[target_col] = round(float(min(1.0, composite)), 4)

            # Recommendation
            recommendations[target_col] = self._recommend(
                target_col, result["learnability"], result["epistemic"]
            )

        results = {
            "uncertainty_scores": uncertainty_scores,
            "learnability": learnability,
            "epistemic": epistemic,
            "recommendations": recommendations,
            "config": self._config(),
        }
        self.results_ = results
        return results

    # ── internal methods ────────────────────────────────────────────────

    def _evaluate_feature(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
    ) -> Dict[str, Any]:
        """Train ensemble and evaluate one target feature."""

        X = df[feature_cols].values
        y = df[target_col].values

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        # Collect predictions from each ensemble member
        all_val_preds = []
        all_val_true = []
        ensemble_r2_scores = []

        for i in range(self.n_ensemble):
            member_seed = self.seed + i

            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled,
                y_scaled,
                test_size=self.test_size,
                random_state=member_seed,
            )

            mlp = MLPRegressor(
                hidden_layer_sizes=self.hidden_layers,
                max_iter=self.max_iter,
                random_state=member_seed,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=10,
                learning_rate_init=0.001,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress convergence warnings
                mlp.fit(X_train, y_train)

            val_pred = mlp.predict(X_val)
            r2 = r2_score(y_val, val_pred)
            ensemble_r2_scores.append(r2)

            # For epistemic assessment, predict on full validation set
            # using the same split as member 0 for comparability
            if i == 0:
                # Store the first member's val set as reference
                ref_X_val = X_val

            # Predict on reference validation set for disagreement
            ref_pred = mlp.predict(ref_X_val)
            all_val_preds.append(ref_pred)

        all_val_preds = np.array(all_val_preds)  # shape: (n_ensemble, n_val)

        # ── Epistemic: ensemble disagreement ──
        pred_std = np.std(all_val_preds, axis=0)  # std across members per sample
        pred_mean = np.mean(all_val_preds, axis=0)

        # Coefficient of variation of predictions (relative disagreement)
        with np.errstate(divide="ignore", invalid="ignore"):
            pred_cv = np.where(
                np.abs(pred_mean) > 1e-8,
                pred_std / np.abs(pred_mean),
                pred_std,  # if mean≈0, just use std
            )

        mean_disagreement = float(np.mean(pred_std))
        mean_cv = float(np.mean(pred_cv))

        # Epistemic score: sigmoid on mean disagreement
        # For standardized data, disagreement of 0.5 std is substantial
        epi_score = float(1.0 / (1.0 + np.exp(-5 * (mean_disagreement - 0.3))))

        # ── Learnability: can the ensemble predict this feature? ──
        mean_r2 = float(np.mean(ensemble_r2_scores))
        std_r2 = float(np.std(ensemble_r2_scores))

        # Baseline: dummy regressor (predicts mean)
        dummy = DummyRegressor(strategy="mean")
        X_train_0, X_val_0, y_train_0, y_val_0 = train_test_split(
            X_scaled,
            y_scaled,
            test_size=self.test_size,
            random_state=self.seed,
        )
        dummy.fit(X_train_0, y_train_0)
        baseline_r2 = float(r2_score(y_val_0, dummy.predict(X_val_0)))

        # Learnability: how much better than baseline?
        # R² improvement over baseline, clamped to [0, 1]
        r2_improvement = max(0, mean_r2 - baseline_r2)
        is_learnable = mean_r2 > self.learnability_threshold

        # Uncertainty from unlearnability:
        # If R² is low, the feature is unpredictable from others → high uncertainty
        # Sigmoid: inflection at R²=0.1, steepness 10
        learn_uncertainty = float(1.0 / (1.0 + np.exp(10 * (mean_r2 - 0.1))))

        return {
            "learnability": {
                "ensemble_r2": round(mean_r2, 4),
                "r2_std": round(std_r2, 4),
                "baseline_r2": round(baseline_r2, 4),
                "r2_improvement": round(r2_improvement, 4),
                "is_learnable": is_learnable,
                "uncertainty_score": round(learn_uncertainty, 4),
            },
            "epistemic": {
                "mean_disagreement": round(mean_disagreement, 4),
                "mean_cv": round(mean_cv, 4),
                "score": round(epi_score, 4),
            },
        }

    def _recommend(
        self,
        feature: str,
        learn_info: Dict[str, Any],
        epi_info: Dict[str, Any],
    ) -> Dict[str, str]:
        """Generate actionable recommendation for one feature."""
        r2 = learn_info["ensemble_r2"]
        is_learnable = learn_info["is_learnable"]
        disagreement = epi_info["mean_disagreement"]

        if not is_learnable and disagreement > 0.3:
            return {
                "action": "investigate_or_drop",
                "explanation": (
                    f"Feature '{feature}' is neither predictable from other "
                    f"features (R²={r2:.3f}) nor consistently estimated by "
                    f"the ensemble (disagreement={disagreement:.3f}). "
                    f"Consider whether this feature carries real signal or "
                    f"is noise. If it's a target variable, the data may be "
                    f"insufficient for reliable modeling."
                ),
            }
        elif not is_learnable:
            return {
                "action": "independent_feature",
                "explanation": (
                    f"Feature '{feature}' cannot be predicted from other "
                    f"features (R²={r2:.3f}). This means it carries "
                    f"independent information — valuable if it's a predictor, "
                    f"but concerning if missing values need imputation "
                    f"(other features won't help reconstruct it)."
                ),
            }
        elif disagreement > 0.3:
            return {
                "action": "unstable_learning",
                "explanation": (
                    f"Feature '{feature}' is learnable (R²={r2:.3f}) but "
                    f"the ensemble shows high disagreement ({disagreement:.3f}). "
                    f"The learning signal exists but is fragile — different "
                    f"model initializations capture different patterns. "
                    f"Consider more data or feature engineering."
                ),
            }
        else:
            return {
                "action": "reliable",
                "explanation": (
                    f"Feature '{feature}' has a stable, learnable "
                    f"relationship with other features (R²={r2:.3f}, "
                    f"disagreement={disagreement:.3f}). Models should "
                    f"produce consistent predictions for this feature."
                ),
            }

    def _config(self) -> Dict[str, Any]:
        return {
            "n_ensemble": self.n_ensemble,
            "hidden_layers": self.hidden_layers,
            "max_iter": self.max_iter,
            "test_size": self.test_size,
            "learnability_threshold": self.learnability_threshold,
        }
