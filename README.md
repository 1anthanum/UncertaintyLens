---
title: UncertaintyLens
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
short_description: Reveal what your data doesn't know.
---

# UncertaintyLens

**Reveal what your data doesn't know — and how much that ignorance costs.**

UncertaintyLens is a Python toolkit that automatically detects uncertainty in tabular data, quantifies the cost of information asymmetry, and presents results through interactive visualizations with actionable remediation advice.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Tests: 244/244](https://img.shields.io/badge/tests-244%2F244-brightgreen)
![Version: 1.0.0](https://img.shields.io/badge/version-1.0.0-blue)

> **Live Demo**: [Try it on Hugging Face Spaces](https://huggingface.co/spaces/xuyangchen/UncertaintyLens)
>
> **中文教程**: See [TUTORIAL.md](TUTORIAL.md) for a complete beginner's guide in Chinese.

## Why UncertaintyLens?

Most data quality tools stop at "you have 5% missing values." UncertaintyLens goes further: it runs 10 specialized detectors across your dataset, produces a single composite uncertainty score per feature, explains *why* each feature scored the way it did, and recommends specific actions to fix the problems.

**Key capabilities:**

- **10 detectors** covering missing patterns, anomalies, variance instability, distribution shift, zero inflation, prediction intervals, and more
- **Automatic attribution** — decomposes each feature's score into per-detector contributions (e.g., "60% from missing values, 30% from anomalies")
- **Interactive HTML reports** — self-contained files with heatmaps, radar charts, waterfall diagrams, and prioritized action plans
- **Streaming monitor** — online drift detection for real-time data pipelines using EWMA and Page-Hinkley tests
- **Extensible pipeline** — register custom detectors with one line of code
- **244 tests** across 16 datasets including blind tests and adversarial scenarios

## Quick Start

### Installation

```bash
git clone https://github.com/1anthanum/UncertaintyLens.git
cd UncertaintyLens
pip install -e .
```

### Basic Usage

```python
import pandas as pd
from uncertainty_lens import UncertaintyPipeline

df = pd.read_csv("your_data.csv")

pipeline = UncertaintyPipeline()
report = pipeline.analyze(df, group_col="category")

# Uncertainty index sorted high → low
for feature, scores in report["uncertainty_index"].items():
    print(f"{feature}: {scores['composite_score']:.3f} ({scores['level']})")
```

### Generate an HTML Report

```python
pipeline.generate_report(df=df, output_path="report.html", title="My Analysis")
```

Open `report.html` in any browser — no server or internet connection needed.

### Interactive Dashboard

```bash
streamlit run app/main.py
```

## Detectors

UncertaintyLens ships with 10 built-in detectors. The first three are enabled by default; the rest can be registered via `pipeline.register()`.

| Detector | What It Does | Default |
|----------|--------------|---------|
| **MissingPatternDetector** | Missing rates, co-missing correlation, MCAR/MAR/MNAR mechanism testing | ✅ |
| **AnomalyDetector** | Ensemble voting (IQR + Isolation Forest + LOF); ≥2 votes to flag | ✅ |
| **VarianceDetector** | CV analysis, between/within-group variance decomposition, temporal trends | ✅ |
| **ConformalShiftDetector** | Distribution shift between groups using conformal p-values | Register |
| **UncertaintyDecomposer** | Splits uncertainty into aleatoric (data noise) vs epistemic (knowledge gap) | Register |
| **JackknifePlusDetector** | Prediction interval width via leave-one-out conformal inference | Register |
| **MMDShiftDetector** | Multi-dimensional distribution drift with adaptive multi-bandwidth kernel | Register |
| **ZeroInflationDetector** | Detects columns with abnormally high zero counts | Register |
| **DeepEnsembleDetector** | Neural network ensemble disagreement as uncertainty proxy | Register |
| **StreamingDetector** | Online monitoring with Welford stats, EWMA, and Page-Hinkley drift test | Register |

### Registering Additional Detectors

```python
from uncertainty_lens.detectors import (
    ConformalShiftDetector, JackknifePlusDetector,
    MMDShiftDetector, ZeroInflationDetector,
)

pipeline = UncertaintyPipeline()
pipeline.register("conformal_shift", ConformalShiftDetector(seed=42), weight=0.1)
pipeline.register("jackknife_plus", JackknifePlusDetector(seed=42), weight=0.1)
pipeline.register("mmd_shift", MMDShiftDetector(n_permutations=200, seed=42), weight=0.1)
pipeline.register("zero_inflation", ZeroInflationDetector(), weight=0.2)

report = pipeline.analyze(df)
```

### Writing Custom Detectors

Any class with an `analyze(df, **kwargs)` method returning `{"uncertainty_scores": {col: float}}` works:

```python
class MyDetector:
    def analyze(self, df, **kwargs):
        scores = {}
        for col in df.select_dtypes(include="number").columns:
            scores[col] = ...  # your logic here, return 0.0–1.0
        return {"uncertainty_scores": scores}

pipeline.register("my_detector", MyDetector(), weight=0.2)
```

## Attribution & Explainability

Every analysis automatically includes an attribution breakdown — no extra code needed.

```python
report = pipeline.analyze(df)

explanation = report["explanation"]
for col, expl in explanation["feature_explanations"].items():
    print(expl["summary"])
    # → "'income' uncertainty is high (0.62), main causes: missing (39%), variance (37%), anomaly (24%)"

for action in explanation["action_plan"]:
    print(f"[{action['severity']}] {action['label']}: {action['action']}")
    # → [high] Missing Pattern: Consider MICE imputation or analyze missing mechanism...
```

## Streaming / Online Monitoring

For real-time data pipelines:

```python
from uncertainty_lens.detectors import StreamingDetector

detector = StreamingDetector(window_size=500)

for batch in data_stream:
    result = detector.update(batch)
    if result["drift_detected"]:
        for alert in result["alerts"]:
            print(f"⚠️ {alert}")
```

## Report Contents

The generated HTML report includes:

1. **Overview dashboard** — dataset stats, overall uncertainty level
2. **Heatmap** — feature × detector score matrix
3. **Ranking chart** — features sorted by composite score
4. **Distribution plots** — histograms per feature
5. **Attribution bar chart** — stacked decomposition per feature
6. **Radar chart** — health profile across detection dimensions
7. **Action plan cards** — prioritized remediation advice

## Testing

```bash
# Unit tests (153 tests)
PYTHONPATH=. python -m pytest tests/ -v

# Core benchmark — 4 classic datasets, 39 checks
PYTHONPATH=. python examples/benchmark_all.py

# Blind test — 3 independent datasets, 25 checks (no threshold tuning)
PYTHONPATH=. python examples/benchmark_blind.py

# Extended test — 6 adversarial/extreme datasets, 27 checks
PYTHONPATH=. python examples/benchmark_extended.py

# Run everything + generate HTML dashboard
PYTHONPATH=. python examples/test_dashboard.py
```

**Current status: 244/244 all passing** (153 unit + 39 core + 25 blind + 27 extended).

## Project Structure

```
UncertaintyLens/
├── uncertainty_lens/
│   ├── __init__.py
│   ├── pipeline.py                  # Orchestrator: chains detectors, computes composite scores
│   ├── detectors/
│   │   ├── missing_pattern.py       # Missing rates, co-missing, MCAR test
│   │   ├── anomaly.py               # IQR + Isolation Forest + LOF ensemble
│   │   ├── variance.py              # CV, variance decomposition, temporal trends
│   │   ├── conformal_shift.py       # Group distribution shift (conformal)
│   │   ├── decomposition.py         # Aleatoric vs epistemic decomposition
│   │   ├── jackknife_plus.py        # Jackknife+ prediction intervals
│   │   ├── mmd_shift.py             # Adaptive multi-bandwidth MMD
│   │   ├── zero_inflation.py        # Zero-inflated column detection
│   │   ├── deep_ensemble.py         # Neural ensemble uncertainty
│   │   ├── streaming_detector.py    # Online Welford + EWMA + Page-Hinkley
│   │   └── uncertainty_explainer.py # Attribution & action plan generator
│   ├── visualizers/
│   │   ├── report.py                # HTML report generator
│   │   ├── heatmap.py               # Uncertainty heatmap
│   │   ├── explainer_charts.py      # Attribution bar, radar, waterfall
│   │   ├── confidence.py            # Confidence intervals & violin plots
│   │   ├── decision.py              # Decision support charts
│   │   └── sankey.py                # Information loss flow diagram
│   └── quantifiers/
├── tests/                           # 153 unit tests
├── examples/                        # Benchmarks & demos
│   ├── benchmark_all.py             # Core benchmark (39 checks)
│   ├── benchmark_blind.py           # Blind test (25 checks)
│   ├── benchmark_extended.py        # Extended test (27 checks)
│   ├── test_dashboard.py            # Unified runner + HTML dashboard
│   └── generate_demo_report.py      # Demo report generator
├── app/                             # Streamlit web interface
├── TUTORIAL.md                      # Complete beginner tutorial (Chinese)
├── CHANGELOG.md                     # Version history
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

## Tech Stack

- **Analysis**: pandas, NumPy, SciPy, scikit-learn
- **Visualization**: Plotly, Matplotlib
- **Web Interface**: Streamlit
- **Testing**: pytest (244 checks across 16 datasets)

## License

MIT — see [LICENSE](LICENSE) for details.
