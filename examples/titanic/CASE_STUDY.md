# Case Study: Hidden Uncertainty in the Titanic Dataset

> **TL;DR**: The Titanic dataset is widely treated as "clean enough" for modeling.
> UncertaintyLens reveals that 3 of 7 numeric features carry medium-to-high uncertainty,
> and the commonly used mean fare statistic shifts by over 10% under Monte Carlo perturbation.

## The Problem

The Titanic passenger dataset (891 records, 15 features) is one of the most-used datasets
in data science education. Most tutorials jump straight to feature engineering and model
building. But how reliable is the underlying data?

Specifically:

- **Age** is missing for ~20% of passengers. Is this random, or are certain groups
  (e.g., 3rd class) more likely to have missing ages?
- **Fare** contains extreme outliers — the maximum ($512) is 36× the median ($14).
  How much do these outliers distort aggregate statistics?
- **Class-based variance** — does Pclass explain most of the variance in fare, or is
  there significant unexplained dispersion within each class?

## Running the Analysis

```bash
pip install seaborn   # for dataset access
python examples/titanic/analyze_titanic.py
```

## Results

### Uncertainty Index

| Feature     | Composite | Missing | Anomaly | Variance | Level       |
|-------------|-----------|---------|---------|----------|-------------|
| fare        | High      | 0.000   | High    | High     | Medium-High |
| age         | Medium    | High    | Low     | Medium   | Medium      |
| survived    | Low       | 0.000   | Low     | High     | Medium-Low  |
| pclass      | Low       | 0.000   | Low     | Low      | Low         |
| sibsp/parch | Low       | 0.000   | Medium  | Low      | Low         |

*Exact scores vary by run — see the script output for precise values.*

### Key Findings

**1. Age missingness is not random.**

The MCAR test detects a statistically significant relationship between age missingness
and other features. This means simple mean imputation (the most common approach in
tutorials) introduces systematic bias. Passengers with missing ages are not a random
subset — they tend to come from specific demographic groups.

**Implication**: Use multiple imputation (e.g., MICE) or model-based imputation instead
of filling with the mean. At minimum, include a `age_was_missing` indicator feature.

**2. Fare outliers inflate uncertainty by 10-15%.**

Monte Carlo simulation (500 trials) shows that re-imputing missing values and adding
small noise causes the mean fare to shift within a 95% CI that spans over $4 — roughly
12% of the point estimate. The sensitivity ratio exceeds 0.1, meaning this statistic
is not robust.

**Implication**: Use median instead of mean for fare-based features. Or better, use
robust statistics (trimmed mean, winsorized values) that are less sensitive to the
extreme outliers in 1st class.

**3. Pclass explains most — but not all — of fare variance.**

Variance decomposition shows that between-class variance accounts for the majority of
total fare variance. But within-class variance in 1st class is still very high (CV > 1.0),
meaning "1st class passenger" is not a homogeneous group. Some paid $25, others paid $512.

**Implication**: Models that use Pclass as a proxy for wealth will miss significant
within-class variation. Consider binning fare into quantiles within each class to capture
this heterogeneity.

## Visualizations

The script generates 5 interactive HTML files:

| File | What it shows |
|------|---------------|
| `titanic_heatmap.html` | Uncertainty scores across all features and dimensions |
| `titanic_breakdown.html` | Stacked bar chart showing what drives each feature's uncertainty |
| `titanic_fare_ci.html` | Fare confidence intervals by passenger class |
| `titanic_fare_dist.html` | Fare distribution (violin plot) by passenger class |
| `titanic_sankey.html` | Information loss flow from raw data to reliable records |

## Conclusion

UncertaintyLens turned a 5-minute "load and model" workflow into a more rigorous analysis
that identified three actionable insights. None of these insights require advanced
statistics — they just require asking "how reliable is this data?" before building models.

This is the core value proposition: **uncertainty analysis should happen before modeling,
not after.**
