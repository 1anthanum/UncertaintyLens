# Changelog

All notable changes to UncertaintyLens will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-04-05

### Added

**Core Pipeline**
- `UncertaintyPipeline` with configurable weights and `register()`/`unregister()` API for custom detectors
- `UncertaintyDetector` protocol (structural typing) — any class with `analyze(df, **kwargs)` qualifies
- Adaptive composite scoring with zero-score weight dampening (10% residual weight)
- Score calibration stretch for compressed ranges (3× expansion when std < 0.1)
- Auto-attached attribution analysis on every `analyze()` call

**Detectors (10 total)**
- `MissingPatternDetector` — missing rates, co-missing correlation matrix, Little's MCAR test, MAR/MNAR mechanism classification
- `AnomalyDetector` — ensemble voting across IQR, Isolation Forest, and LOF with configurable consensus threshold (default ≥ 2 votes)
- `VarianceDetector` — coefficient of variation, between/within-group variance decomposition, temporal trend detection
- `ConformalShiftDetector` — group-wise distribution shift detection using conformal p-values
- `UncertaintyDecomposer` — aleatoric vs epistemic uncertainty decomposition via bootstrap ensembles
- `JackknifePlusDetector` — leave-one-out conformal prediction intervals with coverage guarantees
- `MMDShiftDetector` — Maximum Mean Discrepancy with adaptive multi-bandwidth kernel (5 scales: 0.25×–4× median, synchronized permutation test)
- `ZeroInflationDetector` — statistical detection of excess zero counts beyond expected rates
- `DeepEnsembleDetector` — neural network ensemble disagreement as uncertainty proxy
- `StreamingDetector` — online monitoring with Welford running statistics, EWMA trend detection, Page-Hinkley drift test, sliding window anomaly detection

**Explainability**
- `UncertaintyExplainer` — per-feature attribution decomposition (contribution = detector_pct × composite_score)
- Natural language summaries in Chinese and English
- Global insights aggregated across features per detector
- Prioritized action plans with severity-specific remediation advice
- Detector metadata with 10 detectors × 3 severity levels × 2 languages

**Visualization**
- `generate_decision_report()` — self-contained interactive HTML reports
- Uncertainty heatmap (feature × detector score matrix)
- Composite ranking bar chart
- Feature distribution histograms
- Attribution stacked bar chart (per-feature detector decomposition)
- Global radar chart (health profile across detection dimensions)
- Action plan HTML cards with severity color coding
- Feature waterfall chart (single-feature decomposition)
- Sankey information loss flow diagram
- Confidence interval and violin plots

**Testing (244 total checks)**
- 153 unit tests covering all detectors, edge cases, and error handling
- Core benchmark: 39 checks across 4 classic datasets (Iris, Wine, Housing, Census)
- Blind test: 25 checks across 3 independent datasets (Insurance, Climate, HR) — all expectations defined before seeing results, zero threshold adjustment
- Extended test: 27 checks across 6 adversarial/extreme datasets (Titanic-like, Adult-like, borderline outliers, masked shift, 80-column wide table, 50-row tiny sample, retail mixed)
- Unified test dashboard runner with HTML report generation

**Documentation**
- Complete beginner tutorial in Chinese (TUTORIAL.md)
- Professional English README with full API examples
- Inline docstrings following NumPy style

**Infrastructure**
- `pyproject.toml` with full project metadata and tool configuration
- Docker support for Hugging Face Spaces deployment
- Streamlit web interface (`app/main.py`)

## [0.1.0] - 2025-12-01

### Added
- Initial release with 3 core detectors (Missing, Anomaly, Variance)
- Basic pipeline with fixed weights
- Streamlit dashboard
- Heatmap, confidence interval, and Sankey visualizations
