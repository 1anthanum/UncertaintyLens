# UncertaintyLens

**Reveal what your data doesn't know -- and how much that ignorance costs.**

UncertaintyLens is a Python toolkit that automatically detects uncertainty in tabular data, quantifies the cost of information asymmetry, and presents results through interactive visualizations.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

> **Live Demo**: [Try it on Hugging Face Spaces](https://huggingface.co/spaces/1anthanum/UncertaintyLens)

## Why UncertaintyLens?

Most data quality tools stop at "you have 5% missing values." UncertaintyLens goes further:

- **Missing Pattern Analysis** -- Tests whether missingness is random (MCAR) or systematic (MAR/MNAR), which dramatically changes how you should handle it
- **Ensemble Anomaly Detection** -- Combines IQR, Isolation Forest, and LOF with consensus voting so no single method's bias dominates
- **Variance Decomposition** -- Separates explainable variance (between-group) from unexplained variance (within-group) to pinpoint true uncertainty hotspots
- **Composite Scoring** -- Merges all three dimensions into a single 0-1 uncertainty index per feature, with configurable weights

## Quick Start

### Installation

```bash
git clone https://github.com/1anthanum/UncertaintyLens.git
cd UncertaintyLens
pip install -e .
```

### Python API

```python
import pandas as pd
from uncertainty_lens import UncertaintyPipeline

df = pd.read_csv("your_data.csv")

pipeline = UncertaintyPipeline(
    weights={"missing": 0.4, "anomaly": 0.3, "variance": 0.3}
)
report = pipeline.analyze(df, group_col="category")

# Uncertainty index sorted high -> low
for feature, scores in report["uncertainty_index"].items():
    print(f"{feature}: {scores['composite_score']:.3f} ({scores['level']})")
```

### Interactive Dashboard

```bash
streamlit run app/main.py
```

Upload a CSV or explore with built-in sample advertising data (1,000 records across 5 channels).

## Architecture

```
uncertainty_lens/
    pipeline.py              # Orchestrates detectors, computes composite scores
    detectors/
        missing_pattern.py   # Missing rate, co-missing correlation, MCAR test
        anomaly.py           # IQR + Isolation Forest + LOF ensemble
        variance.py          # CV analysis, variance decomposition, temporal trends
    quantifiers/
        monte_carlo.py       # Monte Carlo uncertainty quantification
    visualizers/
        heatmap.py           # Uncertainty heatmap + stacked bar chart
        confidence.py        # Grouped confidence intervals + violin plots
        sankey.py            # Information loss flow diagram
app/
    main.py                  # Streamlit web application
```

## Detectors

### MissingPatternDetector

Analyzes missing value patterns and tests the missing mechanism:

| Mechanism | Meaning | Implication |
|-----------|---------|-------------|
| MCAR | Missing completely at random | Safe to impute; uncertainty is manageable |
| MAR/MNAR | Missingness depends on observed/unobserved data | Systematic information loss; 30% penalty applied |

### AnomalyDetector

Ensemble voting across three complementary methods:

| Method | Type | Strength |
|--------|------|----------|
| IQR | Univariate | Simple, robust to distribution shape |
| Isolation Forest | Multivariate | Catches high-dimensional outliers |
| LOF | Density-based | Detects local anomalies in clusters |

Points need >= 2 votes to be flagged, reducing false positives.

### VarianceDetector

- **Coefficient of Variation (CV)**: Dimensionless metric for cross-feature comparison
- **Variance Decomposition**: Splits total variance into between-group and within-group components
- **Temporal Analysis**: Detects whether variance is increasing, decreasing, or stable over time

## Visualizations

| Chart | Purpose |
|-------|---------|
| Uncertainty Heatmap | Grid view of scores across features and dimensions |
| Stacked Bar | Shows what drives each feature's uncertainty |
| Sankey Diagram | Information loss flow from raw to reliable data |
| Confidence Intervals | Group-level uncertainty comparison |
| Violin Plots | Full distribution shape by group |

## Running Tests

```bash
pytest tests/ -v
```

## Tech Stack

- **Analysis**: pandas, NumPy, SciPy, scikit-learn
- **Visualization**: Plotly
- **Web Interface**: Streamlit
- **Testing**: pytest

## License

MIT
