from uncertainty_lens.detectors.missing_pattern import MissingPatternDetector
from uncertainty_lens.detectors.anomaly import AnomalyDetector
from uncertainty_lens.detectors.variance import VarianceDetector
from uncertainty_lens.detectors.conformal_shift import ConformalShiftDetector
from uncertainty_lens.detectors.decomposition import UncertaintyDecomposer
from uncertainty_lens.detectors.conformal_predictor import ConformalPredictor
from uncertainty_lens.detectors.deep_ensemble import DeepEnsembleDetector
from uncertainty_lens.detectors.jackknife_plus import JackknifePlusDetector
from uncertainty_lens.detectors.mmd_shift import MMDShiftDetector
from uncertainty_lens.detectors.zero_inflation import ZeroInflationDetector
from uncertainty_lens.detectors.uncertainty_explainer import UncertaintyExplainer
from uncertainty_lens.detectors.streaming_detector import StreamingDetector

# CatBoost is optional — import only if catboost is installed
try:
    from uncertainty_lens.detectors.catboost_uncertainty import CatBoostUncertainty
except ImportError:
    CatBoostUncertainty = None  # type: ignore[assignment,misc]

__all__ = [
    "MissingPatternDetector",
    "AnomalyDetector",
    "VarianceDetector",
    "ConformalShiftDetector",
    "UncertaintyDecomposer",
    "ConformalPredictor",
    "CatBoostUncertainty",
    "DeepEnsembleDetector",
    "JackknifePlusDetector",
    "MMDShiftDetector",
    "ZeroInflationDetector",
    "UncertaintyExplainer",
    "StreamingDetector",
]
