"""
Aleatoric vs. Epistemic uncertainty decomposition module.

Decomposes each feature's uncertainty into two components using bootstrap
resampling — no trained ML model required.

**Aleatoric uncertainty** (irreducible)
    Inherent noise in the data.  Even with infinite samples, this variance
    remains.  Operationally: "you need better sensors / cleaner data
    collection to reduce this."

**Epistemic uncertainty** (reducible)
    Arises from insufficient data.  If the statistic of interest changes a
    lot across bootstrap resamples, the estimate is unstable.  Operationally:
    "you need *more* data to reduce this."

Method
------
For each numeric feature *f*:

1. Draw ``n_bootstrap`` resamples (with replacement) of size ``len(df)``.
2. In each resample, compute a summary statistic (default: mean).
3. **Epistemic component** = variance of the bootstrap statistics (how much
   the estimate moves when data changes).
4. **Aleatoric component** = mean of each resample's internal variance (how
   spread the data itself is, regardless of sampling).
5. Both are normalized into [0, 1] scores and combined into a composite
   ``uncertainty_scores`` entry that the pipeline's index can consume.

The decomposition is also performed per-group when ``group_col`` is
provided, enabling comparison of "why is this group uncertain?"
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List


class UncertaintyDecomposer:
    """
    Bootstrap-based aleatoric / epistemic uncertainty decomposer.

    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap iterations (default 200).
    seed : int
        Random seed for reproducibility.
    statistic : str
        Summary statistic to track across resamples.
        ``"mean"`` (default) or ``"median"``.
    """

    def __init__(
        self,
        n_bootstrap: int = 200,
        seed: int = 42,
        statistic: str = "mean",
    ):
        if n_bootstrap < 10:
            raise ValueError(f"n_bootstrap must be >= 10, got {n_bootstrap}")
        if statistic not in ("mean", "median"):
            raise ValueError(f"statistic must be 'mean' or 'median', got {statistic}")
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self.statistic = statistic
        self.results_: Optional[Dict[str, Any]] = None

    # ── public API ─────────────────────────────────────────────────────

    def analyze(
        self,
        df: pd.DataFrame,
        group_col: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Decompose uncertainty for each numeric feature.

        Returns
        -------
        dict
            ``"uncertainty_scores"`` – composite score per feature (0–1)
            ``"decomposition"``      – per-feature aleatoric/epistemic detail
            ``"group_decomposition"`` – per-group breakdown (if group_col given)
            ``"recommendation"``     – actionable advice per feature
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")
        if df.empty:
            raise ValueError("DataFrame is empty — nothing to analyze")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return {
                "uncertainty_scores": {},
                "decomposition": {},
                "group_decomposition": {},
                "recommendation": {},
            }

        rng = np.random.default_rng(self.seed)

        # ---- population-level decomposition ----
        decomp = self._decompose_features(df, numeric_cols, rng)

        # ---- per-group decomposition ----
        group_decomp: Dict[str, Dict[str, Any]] = {}
        if group_col is not None and group_col in df.columns:
            for g, g_df in df.groupby(group_col):
                g_numeric = [c for c in numeric_cols if c in g_df.columns]
                if len(g_df) >= 5 and g_numeric:
                    group_decomp[str(g)] = self._decompose_features(g_df, g_numeric, rng)

        # ---- build uncertainty_scores and recommendations ----
        uncertainty_scores: Dict[str, float] = {}
        recommendations: Dict[str, Dict[str, str]] = {}

        for col, d in decomp.items():
            ale = d["aleatoric_score"]
            epi = d["epistemic_score"]

            # Composite: weighted blend favoring the dominant component
            composite = 0.5 * ale + 0.5 * epi
            uncertainty_scores[col] = round(float(min(1.0, composite)), 4)

            # Actionable recommendation
            if epi > ale * 1.5 and epi > 0.3:
                action = "collect_more_data"
                explanation = (
                    f"Epistemic uncertainty dominates ({epi:.2f} vs {ale:.2f}). "
                    f"The estimate is unstable — collecting more samples would "
                    f"substantially reduce uncertainty."
                )
            elif ale > epi * 1.5 and ale > 0.3:
                action = "improve_measurement"
                explanation = (
                    f"Aleatoric uncertainty dominates ({ale:.2f} vs {epi:.2f}). "
                    f"The data itself is noisy — more samples won't help much. "
                    f"Consider improving data quality or measurement precision."
                )
            elif ale > 0.3 and epi > 0.3:
                action = "both"
                explanation = (
                    f"Both components are substantial (aleatoric={ale:.2f}, "
                    f"epistemic={epi:.2f}). Address data quality *and* sample size."
                )
            else:
                action = "none"
                explanation = (
                    f"Uncertainty is low (aleatoric={ale:.2f}, "
                    f"epistemic={epi:.2f}). No immediate action needed."
                )

            recommendations[col] = {
                "action": action,
                "explanation": explanation,
            }

        results = {
            "uncertainty_scores": uncertainty_scores,
            "decomposition": decomp,
            "group_decomposition": group_decomp,
            "recommendation": recommendations,
            "config": {
                "n_bootstrap": self.n_bootstrap,
                "statistic": self.statistic,
            },
        }
        self.results_ = results
        return results

    # ── internals ──────────────────────────────────────────────────────

    def _decompose_features(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        rng: np.random.Generator,
    ) -> Dict[str, Dict[str, Any]]:
        """Run bootstrap decomposition for a set of columns."""
        n = len(df)
        decomp: Dict[str, Dict[str, Any]] = {}

        for col in numeric_cols:
            series = df[col].dropna().values
            if len(series) < 5:
                decomp[col] = {
                    "aleatoric_raw": 0.0,
                    "epistemic_raw": 0.0,
                    "aleatoric_score": 0.0,
                    "epistemic_score": 0.0,
                    "dominant": "insufficient_data",
                    "n_valid": len(series),
                }
                continue

            stat_fn = np.mean if self.statistic == "mean" else np.median

            boot_statistics = np.empty(self.n_bootstrap)
            boot_internal_vars = np.empty(self.n_bootstrap)

            for b in range(self.n_bootstrap):
                idx = rng.choice(len(series), size=len(series), replace=True)
                resample = series[idx]
                boot_statistics[b] = stat_fn(resample)
                # Use ddof=0 consistently: within each resample we estimate
                # the full population variance (not a sample from it)
                boot_internal_vars[b] = np.var(resample, ddof=0)

            # Raw components
            # Epistemic: variance of the bootstrap statistics (ddof=0 for
            # consistency — this is the full set of bootstrap estimates)
            epistemic_raw = float(np.var(boot_statistics, ddof=0))
            aleatoric_raw = float(np.mean(boot_internal_vars))

            # Normalize BOTH to the same scale: population variance
            # This makes epi_ratio and ale_ratio directly comparable
            pop_var = float(np.var(series, ddof=0)) if len(series) > 1 else 1.0
            if pop_var == 0:
                pop_var = 1.0
            pop_std = np.sqrt(pop_var)

            # Epistemic ratio: √(var of bootstrap stats) / pop_std
            # Theoretical value for well-sampled mean: 1/√n
            epi_ratio = np.sqrt(epistemic_raw) / pop_std

            # Aleatoric ratio: √(mean internal var) / pop_std
            # For well-behaved data, this should be ≈ 1.0
            # (bootstrap internal variance ≈ population variance)
            ale_ratio = np.sqrt(aleatoric_raw) / pop_std

            # Sigmoid scoring — measure EXCESS over theoretical baseline
            #
            # Epistemic: for well-sampled data, epi_ratio ≈ 1/√n.
            # Score measures how much WORSE than this baseline the feature is.
            # If n=100, baseline = 0.1; if n=10000, baseline = 0.01
            n_valid = len(series)
            epi_baseline = 1.0 / np.sqrt(max(n_valid, 2))
            # Excess ratio: how many times worse than baseline?
            epi_excess = epi_ratio / max(epi_baseline, 1e-10)
            # Score: 1x baseline → 0.0, 3x baseline → ~0.5, 10x → ~0.95
            epi_score = float(1.0 / (1.0 + np.exp(-3 * (epi_excess - 3.0))))

            # Aleatoric: ale_ratio ≈ 1.0 is the mathematical baseline
            # (bootstrap internal var ≈ population var by construction).
            # Score measures excess above this baseline.
            # ale_ratio = 1.0 → score near 0; ale_ratio = 1.5 → ~0.5
            ale_excess = ale_ratio - 1.0  # 0 = baseline, positive = excess noise
            ale_score = float(1.0 / (1.0 + np.exp(-8 * (ale_excess - 0.2))))

            # Determine dominant component using a statistically motivated
            # threshold: one component is "dominant" if it contributes >2x
            # the other to total uncertainty
            if epi_score > 0.01 and ale_score > 0.01:
                ratio = epi_score / ale_score
                if ratio > 2.0:
                    dominant = "epistemic"
                elif ratio < 0.5:
                    dominant = "aleatoric"
                else:
                    dominant = "mixed"
            elif epi_score > ale_score:
                dominant = "epistemic"
            elif ale_score > epi_score:
                dominant = "aleatoric"
            else:
                dominant = "mixed"

            decomp[col] = {
                "aleatoric_raw": round(aleatoric_raw, 6),
                "epistemic_raw": round(epistemic_raw, 6),
                "aleatoric_score": round(min(1.0, ale_score), 4),
                "epistemic_score": round(min(1.0, epi_score), 4),
                "dominant": dominant,
                "epi_ratio": round(float(epi_ratio), 4),
                "ale_ratio": round(float(ale_ratio), 4),
                "n_valid": len(series),
            }

        return decomp
