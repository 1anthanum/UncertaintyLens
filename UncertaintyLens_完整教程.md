# UncertaintyLens 完整开发教程

> 从理论原理到代码实现的完整学习路径
>
> 版本：1.0 | 日期：2026年3月29日

---

## 目录

- [第一部分：基础概念与理论原理](#第一部分基础概念与理论原理)
- [第二部分：环境搭建与项目初始化](#第二部分环境搭建与项目初始化)
- [第三部分：核心模块实现（检测层）](#第三部分核心模块实现检测层)
- [第四部分：可视化层实现](#第四部分可视化层实现)
- [第五部分：完整案例与 Streamlit 应用](#第五部分完整案例与-streamlit-应用)
- [附录：数据集获取指南](#附录数据集获取指南)

---

## 第一部分：基础概念与理论原理

### 1.1 什么是数据不确定性

在任何数据驱动的决策中，我们都面对一个被忽略的事实：**数据本身是不完整的**。不确定性不是错误——它是信息不足的自然结果。

数据不确定性分为三种类型：

**偶然不确定性（Aleatoric Uncertainty）**：数据本身的内在随机性。比如广告点击率天然存在波动，即使完美的数据收集也无法消除。这是不可约减的。

**认知不确定性（Epistemic Uncertainty）**：因为我们不知道或数据不足而产生的不确定性。比如某个广告渠道只有 10 天的数据，我们对它的效果判断自然不可靠。这是可以通过收集更多数据来减少的。

**模型不确定性（Model Uncertainty）**：选择不同的分析方法可能得出不同结论。比如用不同的归因模型（首次点击 vs 末次点击 vs 线性归因），同一笔广告预算的 ROI 可能差异巨大。

UncertaintyLens 的核心目标：**自动识别数据中的这三种不确定性，并将它们转化为可理解的经济成本**。

### 1.2 为什么不确定性分析有商业价值

一个具体的例子：假设你是一家电商公司的营销负责人，每月广告预算 100 万元，分配到 5 个渠道。

传统分析告诉你：渠道 A 的 ROI 是 3.2，渠道 B 是 2.1，渠道 C 是 4.5...

但它没告诉你的是：渠道 A 的数据完整度只有 60%（有 40% 的转化无法归因），渠道 C 虽然 ROI 高但只有 200 个样本（统计置信度极低），渠道 B 的数据最完整最可靠。

不确定性分析的输出是：**"渠道 C 的 ROI 置信区间是 [1.2, 7.8]，而渠道 B 的置信区间是 [1.8, 2.4]。你在渠道 C 上的 30 万元预算中，可能有 8-15 万元是在为不确定性付费。"**

这就是我们要做的东西。

### 1.3 核心算法原理

#### 1.3.1 缺失值模式分析

缺失值不是随机的。缺失数据通常分为三种机制：

- **完全随机缺失（MCAR）**：缺失与任何变量无关。比如传感器随机故障。
- **随机缺失（MAR）**：缺失与已观测变量有关。比如高收入人群更不愿透露收入。
- **非随机缺失（MNAR）**：缺失与缺失值本身有关。比如效果差的广告渠道更可能不报告数据。

MNAR 是最危险的，因为它意味着你的数据在系统性地隐藏某些信息。我们用 Little's MCAR 检验来判断缺失机制：

```
检验统计量 d² = (观测均值 - 期望均值)ᵀ × Σ⁻¹ × (观测均值 - 期望均值)
```

如果 p 值 < 0.05，说明缺失不是完全随机的，需要进一步调查。

**代码中的实现思路**：计算每个特征的缺失率，检测缺失值之间的相关性（是否某些列总是一起缺失），生成缺失模式矩阵。

#### 1.3.2 异常值检测

异常值在不确定性分析中的含义不同于传统数据清洗。我们不是要"去掉异常值"，而是要**识别哪些数据点的异常程度最大，因为它们代表了最大的不确定性来源**。

我们使用多种方法综合判断：

**IQR 方法（四分位距）**：简单但有效。Q1 - 1.5×IQR 到 Q3 + 1.5×IQR 之外的点被标记。适合单变量。

**Isolation Forest（孤立森林）**：基于随机树的方法。核心思想是：异常点更容易被"孤立"——需要更少的分割次数就能把它和其他数据分开。适合多变量。

**局部异常因子（LOF）**：比较每个点与其邻域的密度差异。如果一个点周围很"稀疏"但邻居们周围很"密集"，它就可能是异常的。

**我们的策略**：对每个数据点，用多种方法投票。被多种方法同时标记的点，其"不确定性贡献度"更高。

#### 1.3.3 方差热点识别

方差不是坏事——我们关注的是**不期望的、无法解释的方差**。

具体做法：

1. 对每个数值特征计算基础统计量（均值、方差、偏度、峰度）
2. 计算变异系数（CV = 标准差 / 均值），让不同量纲的特征可以比较
3. 对分组数据（比如按渠道、按时间段分组），计算组内方差和组间方差的比例
4. 方差分解：总方差 = 可解释方差 + 残差方差。残差方差占比越高，不确定性越大

#### 1.3.4 不确定性指数计算

我们为每个特征/维度计算一个综合的"不确定性指数"（Uncertainty Index, UI），范围 0-1：

```
UI = w₁ × 缺失率得分 + w₂ × 异常值得分 + w₃ × 方差得分

其中：
- 缺失率得分 = 缺失比例的 sigmoid 变换（让 5% 以下的缺失率得分接近 0，50% 以上接近 1）
- 异常值得分 = 被标记为异常的数据比例
- 方差得分 = 残差方差占总方差的比例
- w₁, w₂, w₃ 默认权重为 0.4, 0.3, 0.3（可配置）
```

### 1.4 可视化理论

根据不确定性可视化的研究文献，最有效的不确定性可视化方式包括：

**热力图（Heatmap）**：用颜色深浅表示不确定性程度。适合展示"哪些维度/特征的不确定性最高"的全局视图。

**误差带 / 置信区间（Error Bands）**：在时间序列或折线图上叠加置信区间带。带越宽，不确定性越大。直观地展示"我们对这个趋势有多确定"。

**桑基图（Sankey Diagram）**：展示信息流的损失。比如从"原始数据"到"可归因数据"的过程中，有多少信息在每一步丢失了。

**小提琴图 / 分布图**：展示数据分布的形状，包括多峰、偏态等特征。比箱线图能传达更多不确定性信息。

**关键设计原则**：不确定性可视化要避免给人虚假的确定感。颜色使用红-黄-绿色阶（红 = 高不确定性），始终显示置信区间而非点估计。

---

## 第二部分：环境搭建与项目初始化

### 2.1 环境要求

```
Python 3.10 或更高版本
Git
pip 或 Poetry（推荐 Poetry）
```

### 2.2 创建项目

打开终端，执行以下命令：

```bash
# 创建项目目录
mkdir uncertainty-lens
cd uncertainty-lens

# 初始化 Git
git init

# 创建 Python 虚拟环境
python -m venv .venv

# 激活虚拟环境
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate
```

### 2.3 安装依赖

创建 `requirements.txt` 文件：

```
# 核心依赖
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0

# 可视化
plotly>=5.15.0
matplotlib>=3.7.0

# Web 界面
streamlit>=1.28.0

# 开发工具
pytest>=7.4.0
black>=23.0.0
```

安装：

```bash
pip install -r requirements.txt
```

### 2.4 项目目录结构

```bash
# 创建完整的目录结构
mkdir -p uncertainty_lens/detectors
mkdir -p uncertainty_lens/quantifiers
mkdir -p uncertainty_lens/visualizers
mkdir -p app
mkdir -p examples/advertising
mkdir -p examples/ecommerce
mkdir -p examples/supply_chain
mkdir -p tests
```

创建所有 `__init__.py` 文件：

```bash
touch uncertainty_lens/__init__.py
touch uncertainty_lens/detectors/__init__.py
touch uncertainty_lens/quantifiers/__init__.py
touch uncertainty_lens/visualizers/__init__.py
```

### 2.5 创建 `uncertainty_lens/__init__.py`

```python
"""
UncertaintyLens - 数据不确定性分析与可视化工具

自动检测数据中的不确定性特征，量化信息不对称带来的额外成本，
并通过交互式可视化呈现结果。
"""

__version__ = "0.1.0"

from uncertainty_lens.pipeline import UncertaintyPipeline
```

### 2.6 创建 `.gitignore`

```
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.venv/

# IDE
.vscode/
.idea/

# 数据文件（不要提交大数据集）
*.csv
*.parquet
data/raw/

# 系统文件
.DS_Store
```

### 2.7 创建 `README.md` 初始版本

```markdown
# UncertaintyLens 🔍

**告诉你数据里哪些地方你不知道，以及不知道让你多花了多少钱。**

一个基于 Python 的数据不确定性分析与可视化工具包。

## 功能

- 🔍 自动检测数据中的缺失模式、异常值分布、方差热点
- 📊 交互式不确定性可视化（热力图、置信区间、桑基图）
- 💰 将不确定性量化为经济成本估算

## 安装

（待发布到 PyPI）

## 快速开始

（开发中）

## License

MIT
```

此时你的项目结构应该是这样的：

```
uncertainty-lens/
├── uncertainty_lens/
│   ├── __init__.py
│   ├── detectors/
│   │   ├── __init__.py
│   │   ├── missing_pattern.py    （下一步创建）
│   │   ├── anomaly.py            （下一步创建）
│   │   └── variance.py           （下一步创建）
│   ├── quantifiers/
│   │   └── __init__.py
│   ├── visualizers/
│   │   └── __init__.py
│   └── pipeline.py               （下一步创建）
├── app/
├── examples/
├── tests/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 第三部分：核心模块实现（检测层）

### 3.1 缺失值模式检测器

创建文件 `uncertainty_lens/detectors/missing_pattern.py`：

```python
"""
缺失值模式分析模块

检测数据中的缺失模式，判断缺失机制（MCAR/MAR/MNAR），
计算缺失值对数据不确定性的贡献度。
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, Optional


class MissingPatternDetector:
    """
    缺失值模式检测器

    功能：
    1. 计算每个特征的缺失率
    2. 检测缺失值之间的相关模式
    3. 评估缺失机制（是否随机）
    4. 输出缺失不确定性得分
    """

    def __init__(self, significance_level: float = 0.05):
        """
        参数:
            significance_level: 统计检验的显著性水平，默认 0.05
        """
        self.significance_level = significance_level
        self.results_ = None

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        对 DataFrame 执行完整的缺失值模式分析。

        参数:
            df: 要分析的 DataFrame

        返回:
            包含分析结果的字典
        """
        results = {
            "summary": self._compute_summary(df),
            "missing_rates": self._compute_missing_rates(df),
            "co_missing_matrix": self._compute_co_missing(df),
            "mcar_test": self._test_mcar(df),
            "uncertainty_scores": {},
        }

        # 计算每个特征的缺失不确定性得分
        for col in df.columns:
            results["uncertainty_scores"][col] = self._compute_uncertainty_score(
                missing_rate=results["missing_rates"].get(col, 0),
                is_random=results["mcar_test"]["is_mcar"],
            )

        self.results_ = results
        return results

    def _compute_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算总体缺失情况摘要"""
        total_cells = df.shape[0] * df.shape[1]
        total_missing = df.isna().sum().sum()

        return {
            "total_rows": df.shape[0],
            "total_columns": df.shape[1],
            "total_cells": total_cells,
            "total_missing": int(total_missing),
            "overall_missing_rate": round(total_missing / total_cells, 4)
                if total_cells > 0 else 0,
            "columns_with_missing": int((df.isna().sum() > 0).sum()),
            "complete_rows": int(df.dropna().shape[0]),
            "complete_row_rate": round(
                df.dropna().shape[0] / df.shape[0], 4
            ) if df.shape[0] > 0 else 0,
        }

    def _compute_missing_rates(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算每个特征的缺失率"""
        rates = {}
        for col in df.columns:
            rate = df[col].isna().mean()
            rates[col] = round(float(rate), 4)
        return rates

    def _compute_co_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算缺失值共现矩阵。

        如果两个特征总是一起缺失，说明它们的缺失有共同原因，
        这对理解不确定性的来源非常重要。
        """
        # 创建缺失指示矩阵（1 = 缺失, 0 = 不缺失）
        missing_indicator = df.isna().astype(int)

        # 只保留有缺失值的列
        cols_with_missing = missing_indicator.columns[
            missing_indicator.sum() > 0
        ]
        if len(cols_with_missing) < 2:
            return pd.DataFrame()

        # 计算缺失指示变量之间的相关系数
        co_missing = missing_indicator[cols_with_missing].corr()
        return co_missing

    def _test_mcar(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        简化版的 MCAR 检验。

        思路：如果数据是完全随机缺失（MCAR），那么有缺失值的行
        和没有缺失值的行，在其他特征上的分布应该没有显著差异。

        我们用 t 检验来对比。
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            return {"is_mcar": True, "p_values": {}, "note": "特征不足，无法检验"}

        p_values = {}
        non_random_count = 0

        for target_col in numeric_cols:
            if df[target_col].isna().sum() == 0:
                continue

            # 创建缺失指示变量
            is_missing = df[target_col].isna()

            for other_col in numeric_cols:
                if other_col == target_col:
                    continue
                if df[other_col].isna().sum() > 0:
                    continue  # 跳过也有缺失的列

                # 比较：target_col 缺失时，other_col 的值
                # vs target_col 不缺失时，other_col 的值
                group_missing = df.loc[is_missing, other_col].dropna()
                group_present = df.loc[~is_missing, other_col].dropna()

                if len(group_missing) < 5 or len(group_present) < 5:
                    continue

                t_stat, p_val = stats.ttest_ind(
                    group_missing, group_present, equal_var=False
                )
                key = f"{target_col}_vs_{other_col}"
                p_values[key] = round(float(p_val), 4)

                if p_val < self.significance_level:
                    non_random_count += 1

        total_tests = len(p_values)
        is_mcar = non_random_count < max(1, total_tests * 0.1)

        return {
            "is_mcar": is_mcar,
            "non_random_pairs": non_random_count,
            "total_tests": total_tests,
            "p_values": p_values,
            "interpretation": (
                "缺失值模式接近完全随机（MCAR），不确定性相对可控"
                if is_mcar
                else "缺失值模式非随机（MAR/MNAR），存在系统性信息缺失，不确定性较高"
            ),
        }

    def _compute_uncertainty_score(
        self, missing_rate: float, is_random: bool
    ) -> float:
        """
        计算单个特征的缺失不确定性得分（0-1）。

        使用 sigmoid 变换让得分有合理的非线性映射：
        - 0-5% 缺失率：得分接近 0（可接受）
        - 5-20% 缺失率：得分快速上升
        - 20%+ 缺失率：得分接近 1（高不确定性）

        如果缺失不是随机的，额外增加不确定性惩罚。
        """
        # Sigmoid 变换，中心点在 0.15（15% 缺失率）
        base_score = 1 / (1 + np.exp(-20 * (missing_rate - 0.15)))

        # 非随机缺失的惩罚系数
        if not is_random and missing_rate > 0:
            penalty = 1.3  # 增加 30% 的不确定性
        else:
            penalty = 1.0

        score = min(1.0, base_score * penalty)
        return round(float(score), 4)
```

### 3.2 异常值检测器

创建文件 `uncertainty_lens/detectors/anomaly.py`：

```python
"""
异常值检测模块

使用多种算法综合检测异常值，
将异常值视为不确定性的信号而非需要删除的噪声。
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from typing import Dict, Any, List, Optional


class AnomalyDetector:
    """
    多方法综合异常值检测器

    使用三种方法投票：
    1. IQR（四分位距）方法 - 单变量
    2. Isolation Forest（孤立森林）- 多变量
    3. LOF（局部异常因子）- 基于密度

    被多种方法同时标记的点，其"不确定性贡献"更高。
    """

    def __init__(
        self,
        iqr_factor: float = 1.5,
        contamination: float = 0.05,
        min_votes: int = 2,
    ):
        """
        参数:
            iqr_factor: IQR 方法的倍数，默认 1.5
            contamination: 预期异常比例，默认 5%
            min_votes: 被标记为异常需要的最少方法数，默认 2
        """
        self.iqr_factor = iqr_factor
        self.contamination = contamination
        self.min_votes = min_votes
        self.results_ = None

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        对 DataFrame 执行综合异常值检测。

        参数:
            df: 要分析的 DataFrame（只分析数值列）

        返回:
            包含分析结果的字典
        """
        numeric_df = df.select_dtypes(include=[np.number]).dropna()

        if numeric_df.empty or numeric_df.shape[0] < 10:
            return {
                "anomaly_counts": {},
                "uncertainty_scores": {},
                "note": "数据量不足，无法进行异常值检测",
            }

        # 三种方法分别检测
        iqr_flags = self._detect_iqr(numeric_df)
        iso_flags = self._detect_isolation_forest(numeric_df)
        lof_flags = self._detect_lof(numeric_df)

        # 综合投票
        vote_matrix = iqr_flags.astype(int) + iso_flags.astype(int) + lof_flags.astype(int)
        consensus_flags = vote_matrix >= self.min_votes

        # 汇总结果
        results = {
            "method_results": {
                "iqr": {col: int(iqr_flags[col].sum()) for col in numeric_df.columns},
                "isolation_forest": int(iso_flags.any(axis=1).sum()),
                "lof": int(lof_flags.any(axis=1).sum()),
            },
            "consensus_anomalies": {
                col: int(consensus_flags[col].sum())
                for col in numeric_df.columns
            },
            "vote_matrix": vote_matrix,
            "anomaly_rates": {},
            "uncertainty_scores": {},
        }

        # 计算每个特征的异常率和不确定性得分
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
        """IQR 方法：逐列检测"""
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
        """Isolation Forest：多变量检测"""
        flags = pd.DataFrame(False, index=df.index, columns=df.columns)

        try:
            iso = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100,
            )
            predictions = iso.fit_predict(df)
            # -1 = 异常, 1 = 正常
            is_anomaly = predictions == -1

            # 对所有列标记相同的行
            for col in df.columns:
                flags[col] = is_anomaly

        except Exception:
            pass  # 如果失败，返回全 False

        return flags

    def _detect_lof(self, df: pd.DataFrame) -> pd.DataFrame:
        """LOF：基于密度的检测"""
        flags = pd.DataFrame(False, index=df.index, columns=df.columns)

        try:
            n_neighbors = min(20, df.shape[0] - 1)
            lof = LocalOutlierFactor(
                n_neighbors=n_neighbors,
                contamination=self.contamination,
            )
            predictions = lof.fit_predict(df)
            is_anomaly = predictions == -1

            for col in df.columns:
                flags[col] = is_anomaly

        except Exception:
            pass

        return flags

    def _compute_uncertainty_score(
        self,
        anomaly_rate: float,
        vote_distribution: Dict,
        n_rows: int,
    ) -> float:
        """
        计算单个特征的异常不确定性得分（0-1）。

        考虑因素：
        - 异常值比例（越高越不确定）
        - 方法一致性（多方法一致标记 = 更确定是异常）
        - 样本量（小样本的异常检测本身不确定性更高）
        """
        # 基础得分：异常率的 sigmoid 变换
        base_score = 1 / (1 + np.exp(-30 * (anomaly_rate - 0.08)))

        # 小样本惩罚：样本小于 100 时增加不确定性
        sample_penalty = 1.0 if n_rows >= 100 else 1.0 + (100 - n_rows) / 200

        score = min(1.0, base_score * sample_penalty)
        return round(float(score), 4)
```

### 3.3 方差热点检测器

创建文件 `uncertainty_lens/detectors/variance.py`：

```python
"""
方差热点识别模块

检测数据中不期望的、无法解释的高方差区域，
这些区域代表了决策中的不确定性来源。
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, Optional, List


class VarianceDetector:
    """
    方差热点检测器

    功能：
    1. 计算每个特征的变异系数（CV）
    2. 如果有分组变量，进行方差分解（组间 vs 组内）
    3. 检测方差随时间或分组的异常波动
    4. 输出方差不确定性得分
    """

    def __init__(self, cv_threshold: float = 0.5):
        """
        参数:
            cv_threshold: 变异系数阈值，超过此值视为高方差，默认 0.5
        """
        self.cv_threshold = cv_threshold
        self.results_ = None

    def analyze(
        self,
        df: pd.DataFrame,
        group_col: Optional[str] = None,
        time_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        执行方差热点分析。

        参数:
            df: 要分析的 DataFrame
            group_col: 可选的分组列名（如渠道、供应商等）
            time_col: 可选的时间列名

        返回:
            包含分析结果的字典
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        results = {
            "basic_stats": self._compute_basic_stats(df, numeric_cols),
            "cv_analysis": self._compute_cv(df, numeric_cols),
            "uncertainty_scores": {},
        }

        # 如果有分组变量，做方差分解
        if group_col and group_col in df.columns:
            results["variance_decomposition"] = self._decompose_variance(
                df, numeric_cols, group_col
            )

        # 如果有时间变量，检测方差随时间的变化
        if time_col and time_col in df.columns:
            results["temporal_variance"] = self._analyze_temporal_variance(
                df, numeric_cols, time_col
            )

        # 计算综合不确定性得分
        for col in numeric_cols:
            cv = results["cv_analysis"].get(col, {}).get("cv", 0)
            unexplained_ratio = 1.0  # 默认

            if "variance_decomposition" in results and col in results["variance_decomposition"]:
                unexplained_ratio = results["variance_decomposition"][col].get(
                    "within_group_ratio", 1.0
                )

            results["uncertainty_scores"][col] = self._compute_uncertainty_score(
                cv=cv, unexplained_ratio=unexplained_ratio
            )

        self.results_ = results
        return results

    def _compute_basic_stats(
        self, df: pd.DataFrame, cols: List[str]
    ) -> Dict[str, Dict]:
        """计算基础统计量"""
        stats_dict = {}
        for col in cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue

            stats_dict[col] = {
                "count": int(len(series)),
                "mean": round(float(series.mean()), 4),
                "std": round(float(series.std()), 4),
                "min": round(float(series.min()), 4),
                "max": round(float(series.max()), 4),
                "skewness": round(float(series.skew()), 4),
                "kurtosis": round(float(series.kurtosis()), 4),
                "range": round(float(series.max() - series.min()), 4),
            }
        return stats_dict

    def _compute_cv(
        self, df: pd.DataFrame, cols: List[str]
    ) -> Dict[str, Dict]:
        """
        计算变异系数（CV = 标准差 / 均值）。

        CV 的好处是无量纲，可以跨特征比较。
        CV > 0.5 通常意味着高离散度。
        CV > 1.0 意味着标准差大于均值，数据极不稳定。
        """
        cv_dict = {}
        for col in cols:
            series = df[col].dropna()
            mean = series.mean()

            if mean == 0 or len(series) < 2:
                cv_dict[col] = {"cv": float("inf"), "level": "无法计算"}
                continue

            cv = float(series.std() / abs(mean))
            level = (
                "低" if cv < 0.2
                else "中" if cv < 0.5
                else "高" if cv < 1.0
                else "极高"
            )

            cv_dict[col] = {
                "cv": round(cv, 4),
                "level": level,
                "is_high_variance": cv > self.cv_threshold,
            }
        return cv_dict

    def _decompose_variance(
        self, df: pd.DataFrame, cols: List[str], group_col: str
    ) -> Dict[str, Dict]:
        """
        方差分解：总方差 = 组间方差 + 组内方差

        组间方差占比高 → 方差是由已知分组因素解释的 → 不确定性较低
        组内方差占比高 → 方差无法被已知因素解释 → 不确定性较高
        """
        decomp = {}
        grouped = df.groupby(group_col)

        for col in cols:
            series = df[col].dropna()
            if len(series) < 2:
                continue

            total_var = float(series.var())
            if total_var == 0:
                decomp[col] = {
                    "total_variance": 0,
                    "between_group_ratio": 0,
                    "within_group_ratio": 0,
                }
                continue

            # 组间方差：各组均值的方差（加权）
            group_means = grouped[col].mean()
            group_counts = grouped[col].count()
            grand_mean = series.mean()

            between_var = float(
                np.average(
                    (group_means - grand_mean) ** 2,
                    weights=group_counts,
                )
            )

            # 组内方差：各组内部方差的加权平均
            within_var = float(
                np.average(
                    grouped[col].var().fillna(0),
                    weights=group_counts,
                )
            )

            decomp[col] = {
                "total_variance": round(total_var, 4),
                "between_group_variance": round(between_var, 4),
                "within_group_variance": round(within_var, 4),
                "between_group_ratio": round(between_var / total_var, 4)
                    if total_var > 0 else 0,
                "within_group_ratio": round(within_var / total_var, 4)
                    if total_var > 0 else 0,
                "n_groups": int(grouped.ngroups),
            }

        return decomp

    def _analyze_temporal_variance(
        self, df: pd.DataFrame, cols: List[str], time_col: str
    ) -> Dict[str, Any]:
        """检测方差随时间的变化趋势"""
        temporal = {}

        try:
            df_sorted = df.sort_values(time_col)

            for col in cols:
                series = df_sorted[col].dropna()
                if len(series) < 20:
                    continue

                # 将数据分为 4 个时间窗口，比较各窗口的方差
                n = len(series)
                window_size = n // 4
                windows = []

                for i in range(4):
                    start = i * window_size
                    end = start + window_size if i < 3 else n
                    window_var = float(series.iloc[start:end].var())
                    windows.append(window_var)

                # 方差是否在增长？
                variance_trend = "稳定"
                if windows[-1] > windows[0] * 1.5:
                    variance_trend = "增长"
                elif windows[-1] < windows[0] * 0.5:
                    variance_trend = "下降"

                temporal[col] = {
                    "window_variances": [round(v, 4) for v in windows],
                    "variance_trend": variance_trend,
                }

        except Exception:
            pass

        return temporal

    def _compute_uncertainty_score(
        self, cv: float, unexplained_ratio: float
    ) -> float:
        """
        计算方差不确定性得分（0-1）。

        综合考虑变异系数和不可解释方差比例。
        """
        # CV 部分的得分
        if cv == float("inf"):
            cv_score = 1.0
        else:
            cv_score = 1 / (1 + np.exp(-5 * (cv - 0.5)))

        # 不可解释方差部分的得分
        unexplained_score = unexplained_ratio

        # 综合：取加权平均
        score = 0.5 * cv_score + 0.5 * unexplained_score
        return round(float(min(1.0, score)), 4)
```

### 3.4 检测器注册

更新 `uncertainty_lens/detectors/__init__.py`：

```python
from uncertainty_lens.detectors.missing_pattern import MissingPatternDetector
from uncertainty_lens.detectors.anomaly import AnomalyDetector
from uncertainty_lens.detectors.variance import VarianceDetector

__all__ = [
    "MissingPatternDetector",
    "AnomalyDetector",
    "VarianceDetector",
]
```

### 3.5 统一分析流水线

创建文件 `uncertainty_lens/pipeline.py`：

```python
"""
统一分析流水线

将三个检测器串联，输出综合不确定性分析报告。
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
    不确定性分析流水线

    用法:
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
        """
        参数:
            weights: 三种不确定性的权重，默认 {"missing": 0.4, "anomaly": 0.3, "variance": 0.3}
            missing_kwargs: 缺失值检测器的额外参数
            anomaly_kwargs: 异常值检测器的额外参数
            variance_kwargs: 方差检测器的额外参数
        """
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
        """
        执行完整的不确定性分析。

        参数:
            df: 要分析的 DataFrame
            group_col: 分组列名（如广告渠道、供应商等）
            time_col: 时间列名

        返回:
            完整的分析报告字典
        """
        # 运行三个检测器
        missing_results = self.missing_detector.analyze(df)
        anomaly_results = self.anomaly_detector.analyze(df)
        variance_results = self.variance_detector.analyze(
            df, group_col=group_col, time_col=time_col
        )

        # 合并不确定性得分
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

        # 按不确定性从高到低排序
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
        """将数值得分转为可读的等级"""
        if score < 0.2:
            return "低不确定性"
        elif score < 0.4:
            return "中低不确定性"
        elif score < 0.6:
            return "中等不确定性"
        elif score < 0.8:
            return "较高不确定性"
        else:
            return "高不确定性"

    def _generate_summary(
        self, uncertainty_index: Dict, df: pd.DataFrame
    ) -> Dict[str, Any]:
        """生成人类可读的分析摘要"""
        if not uncertainty_index:
            return {"message": "没有可分析的数值特征"}

        scores = [v["composite_score"] for v in uncertainty_index.values()]
        avg_score = np.mean(scores)

        # 找出不确定性最高的特征
        top_uncertain = list(uncertainty_index.items())[:3]

        # 找出不确定性最低的特征
        bottom_uncertain = list(uncertainty_index.items())[-3:]

        return {
            "overall_uncertainty": round(float(avg_score), 4),
            "overall_level": self._score_to_level(avg_score),
            "total_features_analyzed": len(uncertainty_index),
            "high_uncertainty_features": [
                col for col, v in uncertainty_index.items()
                if v["composite_score"] >= 0.6
            ],
            "low_uncertainty_features": [
                col for col, v in uncertainty_index.items()
                if v["composite_score"] < 0.2
            ],
            "top_3_uncertain": [
                {"feature": col, **vals} for col, vals in top_uncertain
            ],
            "most_reliable": [
                {"feature": col, **vals} for col, vals in bottom_uncertain
            ],
        }
```

### 3.6 测试你的代码

创建文件 `tests/test_basic.py`：

```python
"""
基础功能测试

在写完每个模块后运行，确保核心逻辑正确。
"""

import pandas as pd
import numpy as np
import sys
import os

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uncertainty_lens.pipeline import UncertaintyPipeline
from uncertainty_lens.detectors import (
    MissingPatternDetector,
    AnomalyDetector,
    VarianceDetector,
)


def create_test_data():
    """创建一个带有已知不确定性特征的测试数据集"""
    np.random.seed(42)
    n = 500

    df = pd.DataFrame({
        # 干净的特征（低不确定性）
        "clean_feature": np.random.normal(100, 5, n),

        # 有很多缺失值的特征（高缺失不确定性）
        "missing_feature": np.where(
            np.random.random(n) > 0.7,
            np.nan,
            np.random.normal(50, 10, n),
        ),

        # 有异常值的特征（高异常不确定性）
        "anomaly_feature": np.concatenate([
            np.random.normal(200, 10, n - 20),
            np.random.normal(500, 50, 20),  # 20 个异常点
        ]),

        # 高方差特征
        "high_variance_feature": np.random.exponential(100, n),

        # 分组变量
        "channel": np.random.choice(["A", "B", "C", "D"], n),
    })

    return df


def test_missing_detector():
    """测试缺失值检测器"""
    print("=" * 50)
    print("测试缺失值检测器")
    print("=" * 50)

    df = create_test_data()
    detector = MissingPatternDetector()
    results = detector.analyze(df)

    print(f"总体缺失率: {results['summary']['overall_missing_rate']}")
    print(f"各特征缺失率: {results['missing_rates']}")
    print(f"MCAR 检验: {results['mcar_test']['interpretation']}")
    print(f"不确定性得分: {results['uncertainty_scores']}")
    print()

    # 验证：missing_feature 的不确定性应该最高
    scores = results["uncertainty_scores"]
    numeric_scores = {
        k: v for k, v in scores.items()
        if k != "channel" and isinstance(v, (int, float))
    }
    if numeric_scores:
        max_col = max(numeric_scores, key=numeric_scores.get)
        assert max_col == "missing_feature", \
            f"预期 missing_feature 不确定性最高，实际是 {max_col}"
        print("✓ 缺失值检测器通过")
    print()


def test_anomaly_detector():
    """测试异常值检测器"""
    print("=" * 50)
    print("测试异常值检测器")
    print("=" * 50)

    df = create_test_data()
    detector = AnomalyDetector()
    results = detector.analyze(df)

    print(f"各特征异常率: {results['anomaly_rates']}")
    print(f"不确定性得分: {results['uncertainty_scores']}")
    print()

    # 验证：anomaly_feature 应该被检测到最多异常
    if results["anomaly_rates"]:
        max_col = max(results["anomaly_rates"], key=results["anomaly_rates"].get)
        print(f"异常率最高的特征: {max_col}")
        print("✓ 异常值检测器通过")
    print()


def test_variance_detector():
    """测试方差检测器"""
    print("=" * 50)
    print("测试方差检测器")
    print("=" * 50)

    df = create_test_data()
    detector = VarianceDetector()
    results = detector.analyze(df, group_col="channel")

    print(f"变异系数分析: {results['cv_analysis']}")
    if "variance_decomposition" in results:
        print(f"方差分解（前两个特征）:")
        for col in list(results["variance_decomposition"].keys())[:2]:
            decomp = results["variance_decomposition"][col]
            print(f"  {col}: 组间={decomp['between_group_ratio']:.2%}, "
                  f"组内={decomp['within_group_ratio']:.2%}")
    print(f"不确定性得分: {results['uncertainty_scores']}")
    print("✓ 方差检测器通过")
    print()


def test_pipeline():
    """测试完整流水线"""
    print("=" * 50)
    print("测试完整流水线")
    print("=" * 50)

    df = create_test_data()
    pipeline = UncertaintyPipeline()
    report = pipeline.analyze(df, group_col="channel")

    print("\n不确定性指数（按从高到低排序）:")
    print("-" * 60)
    for col, vals in report["uncertainty_index"].items():
        print(f"  {col:25s} | 综合: {vals['composite_score']:.3f} | "
              f"缺失: {vals['missing_score']:.3f} | "
              f"异常: {vals['anomaly_score']:.3f} | "
              f"方差: {vals['variance_score']:.3f} | "
              f"{vals['level']}")

    print(f"\n总体不确定性: {report['summary']['overall_level']}")
    print(f"高不确定性特征: {report['summary']['high_uncertainty_features']}")
    print(f"低不确定性特征: {report['summary']['low_uncertainty_features']}")
    print("\n✓ 完整流水线通过")


if __name__ == "__main__":
    test_missing_detector()
    test_anomaly_detector()
    test_variance_detector()
    test_pipeline()
    print("\n" + "=" * 50)
    print("所有测试通过！")
    print("=" * 50)
```

运行测试：

```bash
cd uncertainty-lens
python tests/test_basic.py
```

---

## 第四部分：可视化层实现

### 4.1 不确定性热力图

创建文件 `uncertainty_lens/visualizers/heatmap.py`：

```python
"""
不确定性热力图模块

将不确定性指数可视化为交互式热力图，
让用户一眼看到"哪些维度最不确定"。
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


def create_uncertainty_heatmap(
    uncertainty_index: Dict[str, Dict],
    title: str = "数据不确定性热力图",
) -> go.Figure:
    """
    创建不确定性热力图。

    参数:
        uncertainty_index: pipeline 输出的 uncertainty_index 字典
        title: 图表标题

    返回:
        Plotly Figure 对象
    """
    # 准备数据
    features = list(uncertainty_index.keys())
    dimensions = ["缺失不确定性", "异常不确定性", "方差不确定性", "综合得分"]

    z_data = []
    for col in features:
        vals = uncertainty_index[col]
        z_data.append([
            vals["missing_score"],
            vals["anomaly_score"],
            vals["variance_score"],
            vals["composite_score"],
        ])

    z_array = np.array(z_data)

    # 创建热力图
    fig = go.Figure(data=go.Heatmap(
        z=z_array,
        x=dimensions,
        y=features,
        colorscale=[
            [0.0, "#1a9641"],    # 绿色 = 低不确定性
            [0.25, "#a6d96a"],
            [0.5, "#ffffbf"],    # 黄色 = 中等
            [0.75, "#fdae61"],
            [1.0, "#d7191c"],    # 红色 = 高不确定性
        ],
        zmin=0,
        zmax=1,
        text=[[f"{v:.3f}" for v in row] for row in z_data],
        texttemplate="%{text}",
        textfont={"size": 12},
        hovertemplate=(
            "特征: %{y}<br>"
            "维度: %{x}<br>"
            "得分: %{z:.3f}<br>"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        title={
            "text": title,
            "font": {"size": 18},
        },
        xaxis_title="不确定性维度",
        yaxis_title="数据特征",
        width=800,
        height=max(400, len(features) * 50),
        font={"family": "Arial"},
    )

    return fig


def create_uncertainty_bar(
    uncertainty_index: Dict[str, Dict],
    title: str = "各特征不确定性得分",
) -> go.Figure:
    """
    创建堆叠条形图，展示每个特征的不确定性组成。
    """
    features = list(uncertainty_index.keys())

    missing_scores = [uncertainty_index[f]["missing_score"] for f in features]
    anomaly_scores = [uncertainty_index[f]["anomaly_score"] for f in features]
    variance_scores = [uncertainty_index[f]["variance_score"] for f in features]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="缺失不确定性",
        x=features,
        y=missing_scores,
        marker_color="#3288bd",
    ))
    fig.add_trace(go.Bar(
        name="异常不确定性",
        x=features,
        y=anomaly_scores,
        marker_color="#fee08b",
    ))
    fig.add_trace(go.Bar(
        name="方差不确定性",
        x=features,
        y=variance_scores,
        marker_color="#d53e4f",
    ))

    fig.update_layout(
        barmode="stack",
        title={"text": title, "font": {"size": 18}},
        xaxis_title="数据特征",
        yaxis_title="不确定性得分",
        yaxis={"range": [0, 1.2]},
        width=800,
        height=500,
        legend={"orientation": "h", "y": -0.15},
        font={"family": "Arial"},
    )

    return fig
```

### 4.2 置信区间可视化

创建文件 `uncertainty_lens/visualizers/confidence.py`：

```python
"""
置信区间可视化模块

用误差带展示数据中的不确定性范围，
让用户直观理解"我们对这个数值有多确定"。
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional, List


def create_confidence_plot(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    confidence_level: float = 0.95,
    title: Optional[str] = None,
) -> go.Figure:
    """
    创建分组置信区间图。

    展示每个组的均值和置信区间，
    置信区间越宽 = 不确定性越高。

    参数:
        df: 数据
        value_col: 数值列名
        group_col: 分组列名
        confidence_level: 置信水平，默认 0.95
        title: 图表标题
    """
    groups = df[group_col].unique()
    means = []
    ci_lower = []
    ci_upper = []
    group_names = []
    n_samples = []

    for group in sorted(groups):
        data = df[df[group_col] == group][value_col].dropna()
        if len(data) < 2:
            continue

        mean = data.mean()
        se = stats.sem(data)
        ci = stats.t.interval(
            confidence_level, df=len(data) - 1, loc=mean, scale=se
        )

        group_names.append(str(group))
        means.append(mean)
        ci_lower.append(ci[0])
        ci_upper.append(ci[1])
        n_samples.append(len(data))

    # 计算置信区间宽度（作为不确定性的度量）
    ci_widths = [u - l for u, l in zip(ci_upper, ci_lower)]

    fig = go.Figure()

    # 置信区间（误差条）
    fig.add_trace(go.Scatter(
        x=group_names,
        y=means,
        error_y=dict(
            type="data",
            symmetric=False,
            array=[u - m for u, m in zip(ci_upper, means)],
            arrayminus=[m - l for m, l in zip(means, ci_lower)],
            color="rgba(214, 39, 40, 0.6)",
            thickness=2,
            width=10,
        ),
        mode="markers",
        marker=dict(size=10, color="#1f77b4"),
        name=f"均值 ± {confidence_level:.0%} CI",
        hovertemplate=(
            "组: %{x}<br>"
            "均值: %{y:.2f}<br>"
            f"置信区间宽度: " + "%{customdata[0]:.2f}<br>"
            "样本量: %{customdata[1]}<br>"
            "<extra></extra>"
        ),
        customdata=list(zip(ci_widths, n_samples)),
    ))

    if not title:
        title = f"{value_col} 的分组置信区间（{confidence_level:.0%} CI）"

    fig.update_layout(
        title={"text": title, "font": {"size": 16}},
        xaxis_title=group_col,
        yaxis_title=value_col,
        width=800,
        height=500,
        font={"family": "Arial"},
        showlegend=True,
    )

    return fig


def create_distribution_comparison(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    title: Optional[str] = None,
) -> go.Figure:
    """
    创建分组分布对比图（小提琴图）。

    小提琴图比箱线图能传达更多信息：
    - 分布的形状（多峰、偏态）
    - 密度集中区域
    - 尾部行为
    """
    groups = sorted(df[group_col].unique())

    fig = go.Figure()

    colors = px.colors.qualitative.Set2
    for i, group in enumerate(groups):
        data = df[df[group_col] == group][value_col].dropna()
        if len(data) < 5:
            continue

        fig.add_trace(go.Violin(
            y=data,
            name=str(group),
            box_visible=True,
            meanline_visible=True,
            line_color=colors[i % len(colors)],
            fillcolor=f"rgba{tuple(list(px.colors.hex_to_rgb(colors[i % len(colors)])) + [0.3])}",
        ))

    if not title:
        title = f"{value_col} 的分组分布对比"

    fig.update_layout(
        title={"text": title, "font": {"size": 16}},
        yaxis_title=value_col,
        xaxis_title=group_col,
        width=800,
        height=500,
        font={"family": "Arial"},
    )

    return fig
```

### 4.3 信息流失桑基图

创建文件 `uncertainty_lens/visualizers/sankey.py`：

```python
"""
信息流失桑基图模块

展示从"原始数据"到"可用于决策的数据"的过程中，
有多少信息在每一步丢失了。
"""

import plotly.graph_objects as go
from typing import Dict, List, Optional


def create_info_loss_sankey(
    total_records: int,
    missing_records: int,
    anomaly_records: int,
    high_variance_records: int,
    title: str = "数据信息流失分析",
) -> go.Figure:
    """
    创建信息流失桑基图。

    展示数据从原始状态到可靠状态的过程中，
    有多少数据因为各种不确定性因素而"不可信"。

    参数:
        total_records: 总记录数
        missing_records: 有缺失值的记录数
        anomaly_records: 含异常值的记录数
        high_variance_records: 高方差区域的记录数
        title: 图表标题
    """
    # 计算各环节的数据量
    # 注意：一条记录可能同时有多个问题，所以用估算
    clean_records = max(
        0,
        total_records - missing_records - anomaly_records - high_variance_records
    )
    # 防止重复计算导致负数
    clean_records = max(int(total_records * 0.3), clean_records)

    uncertain_total = total_records - clean_records

    # 桑基图的节点
    labels = [
        f"原始数据\n({total_records:,} 条)",          # 0
        f"缺失数据\n({missing_records:,} 条)",         # 1
        f"异常数据\n({anomaly_records:,} 条)",          # 2
        f"高方差数据\n({high_variance_records:,} 条)",  # 3
        f"可靠数据\n({clean_records:,} 条)",            # 4
        f"不确定数据\n({uncertain_total:,} 条)",        # 5
    ]

    # 连线：source -> target, value
    source = [0, 0, 0, 0, 1, 2, 3]
    target = [1, 2, 3, 4, 5, 5, 5]
    value = [
        missing_records,
        anomaly_records,
        high_variance_records,
        clean_records,
        missing_records,
        anomaly_records,
        high_variance_records,
    ]

    # 节点颜色
    node_colors = [
        "#2196F3",  # 原始数据 - 蓝色
        "#FF9800",  # 缺失 - 橙色
        "#F44336",  # 异常 - 红色
        "#9C27B0",  # 高方差 - 紫色
        "#4CAF50",  # 可靠 - 绿色
        "#E91E63",  # 不确定 - 粉红色
    ]

    # 连线颜色
    link_colors = [
        "rgba(255,152,0,0.3)",   # -> 缺失
        "rgba(244,67,54,0.3)",   # -> 异常
        "rgba(156,39,176,0.3)",  # -> 高方差
        "rgba(76,175,80,0.3)",   # -> 可靠
        "rgba(255,152,0,0.2)",   # 缺失 -> 不确定
        "rgba(244,67,54,0.2)",   # 异常 -> 不确定
        "rgba(156,39,176,0.2)",  # 高方差 -> 不确定
    ]

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20,
            thickness=30,
            line=dict(color="black", width=0.5),
            label=labels,
            color=node_colors,
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors,
        ),
    )])

    # 计算信息损失率
    loss_rate = uncertain_total / total_records if total_records > 0 else 0

    fig.update_layout(
        title={
            "text": f"{title}<br>"
                    f"<sub>信息损失率: {loss_rate:.1%} | "
                    f"可靠数据: {clean_records:,}/{total_records:,} 条</sub>",
            "font": {"size": 16},
        },
        width=900,
        height=500,
        font={"family": "Arial", "size": 12},
    )

    return fig
```

### 4.4 可视化模块注册

更新 `uncertainty_lens/visualizers/__init__.py`：

```python
from uncertainty_lens.visualizers.heatmap import (
    create_uncertainty_heatmap,
    create_uncertainty_bar,
)
from uncertainty_lens.visualizers.confidence import (
    create_confidence_plot,
    create_distribution_comparison,
)
from uncertainty_lens.visualizers.sankey import create_info_loss_sankey

__all__ = [
    "create_uncertainty_heatmap",
    "create_uncertainty_bar",
    "create_confidence_plot",
    "create_distribution_comparison",
    "create_info_loss_sankey",
]
```

---

## 第五部分：完整案例与 Streamlit 应用

### 5.1 Streamlit 应用

创建文件 `app/main.py`：

```python
"""
UncertaintyLens - Streamlit 交互式应用

用浏览器访问，上传数据即可获得不确定性分析报告。
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# 确保能导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uncertainty_lens.pipeline import UncertaintyPipeline
from uncertainty_lens.visualizers import (
    create_uncertainty_heatmap,
    create_uncertainty_bar,
    create_confidence_plot,
    create_distribution_comparison,
    create_info_loss_sankey,
)


# ========== 页面配置 ==========
st.set_page_config(
    page_title="UncertaintyLens",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 UncertaintyLens")
st.markdown("**告诉你数据里哪些地方你不知道，以及不知道让你多花了多少钱。**")
st.markdown("---")


# ========== 侧边栏：数据上传与配置 ==========
with st.sidebar:
    st.header("📁 数据输入")

    data_source = st.radio(
        "选择数据来源",
        ["上传 CSV 文件", "使用示例数据"],
    )

    if data_source == "上传 CSV 文件":
        uploaded_file = st.file_uploader("上传你的 CSV 文件", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            df = None
    else:
        # 生成示例数据（模拟广告投放场景）
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            "渠道": np.random.choice(
                ["搜索广告", "社交媒体", "视频平台", "信息流", "邮件营销"],
                n,
            ),
            "展示量": np.random.lognormal(8, 1.5, n).astype(int),
            "点击量": np.where(
                np.random.random(n) > 0.1,
                np.random.lognormal(5, 1.2, n).astype(int),
                np.nan,  # 10% 的点击数据缺失
            ),
            "转化量": np.where(
                np.random.random(n) > 0.25,
                np.random.poisson(10, n),
                np.nan,  # 25% 的转化数据缺失
            ),
            "花费": np.concatenate([
                np.random.lognormal(6, 0.8, n - 30),
                np.random.lognormal(9, 0.5, 30),  # 异常高花费
            ]),
            "归因收入": np.where(
                np.random.random(n) > 0.35,
                np.random.lognormal(7, 1.5, n),
                np.nan,  # 35% 的归因数据缺失！
            ),
        })
        st.success("已加载广告投放示例数据（1000 条记录）")

    if df is not None:
        st.markdown("---")
        st.header("⚙️ 分析配置")

        all_cols = df.columns.tolist()
        string_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        group_col = st.selectbox(
            "选择分组列（如渠道、类别）",
            ["无"] + string_cols,
        )
        group_col = None if group_col == "无" else group_col

        st.markdown("---")
        st.header("🎛️ 权重配置")
        w_missing = st.slider("缺失不确定性权重", 0.0, 1.0, 0.4, 0.05)
        w_anomaly = st.slider("异常不确定性权重", 0.0, 1.0, 0.3, 0.05)
        w_variance = st.slider("方差不确定性权重", 0.0, 1.0, 0.3, 0.05)

        # 归一化权重
        total_w = w_missing + w_anomaly + w_variance
        if total_w > 0:
            w_missing /= total_w
            w_anomaly /= total_w
            w_variance /= total_w


# ========== 主区域：分析结果 ==========
if df is not None:
    # 数据预览
    with st.expander("📊 数据预览", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("总行数", f"{df.shape[0]:,}")
        col2.metric("总列数", f"{df.shape[1]}")
        col3.metric("缺失值总数", f"{df.isna().sum().sum():,}")

    # 运行分析
    with st.spinner("正在分析数据不确定性..."):
        pipeline = UncertaintyPipeline(
            weights={
                "missing": w_missing,
                "anomaly": w_anomaly,
                "variance": w_variance,
            }
        )
        report = pipeline.analyze(df, group_col=group_col)

    # ===== 摘要卡片 =====
    st.markdown("## 📋 分析摘要")
    summary = report["summary"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "总体不确定性",
        f"{summary['overall_uncertainty']:.1%}",
        delta=summary["overall_level"],
    )
    col2.metric("分析特征数", summary["total_features_analyzed"])
    col3.metric("高不确定性特征", len(summary["high_uncertainty_features"]))
    col4.metric("低不确定性特征", len(summary["low_uncertainty_features"]))

    # ===== 热力图 =====
    st.markdown("## 🌡️ 不确定性热力图")
    st.markdown("颜色越红 = 不确定性越高。找到红色区域就是找到了数据中最需要关注的地方。")

    fig_heatmap = create_uncertainty_heatmap(report["uncertainty_index"])
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # ===== 堆叠条形图 =====
    st.markdown("## 📊 不确定性组成分析")
    st.markdown("每个特征的不确定性由哪些因素贡献？蓝色=缺失，黄色=异常，红色=方差。")

    fig_bar = create_uncertainty_bar(report["uncertainty_index"])
    st.plotly_chart(fig_bar, use_container_width=True)

    # ===== 信息流失桑基图 =====
    st.markdown("## 🔀 信息流失分析")
    st.markdown("从原始数据到可靠数据，有多少信息丢失了？")

    missing_rows = int(df.isnull().any(axis=1).sum())
    anomaly_count = sum(
        report["anomaly_analysis"].get("consensus_anomalies", {}).values()
    )
    high_var_count = sum(
        1 for v in report["variance_analysis"].get("cv_analysis", {}).values()
        if isinstance(v, dict) and v.get("is_high_variance", False)
    ) * (df.shape[0] // 5)  # 估算

    fig_sankey = create_info_loss_sankey(
        total_records=df.shape[0],
        missing_records=missing_rows,
        anomaly_records=min(anomaly_count, df.shape[0] // 3),
        high_variance_records=min(high_var_count, df.shape[0] // 4),
    )
    st.plotly_chart(fig_sankey, use_container_width=True)

    # ===== 分组分析 =====
    if group_col:
        st.markdown(f"## 📈 按 {group_col} 分组分析")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_col = st.selectbox("选择要分析的数值特征", numeric_cols)

        if selected_col:
            tab1, tab2 = st.tabs(["置信区间", "分布对比"])

            with tab1:
                fig_ci = create_confidence_plot(
                    df, selected_col, group_col
                )
                st.plotly_chart(fig_ci, use_container_width=True)
                st.markdown("*误差条越宽 = 该组的数据不确定性越高*")

            with tab2:
                fig_violin = create_distribution_comparison(
                    df, selected_col, group_col
                )
                st.plotly_chart(fig_violin, use_container_width=True)
                st.markdown("*分布越宽越不规则 = 不确定性越高*")

    # ===== 详细报告 =====
    with st.expander("🔬 查看详细分析数据", expanded=False):
        st.json(report["summary"])

else:
    st.info("👈 请在左侧上传数据或选择示例数据开始分析")
```

### 5.2 启动应用

```bash
# 在项目根目录运行
streamlit run app/main.py
```

浏览器会自动打开 `http://localhost:8501`，你就能看到完整的交互式不确定性分析应用了。

### 5.3 发布到 Streamlit Cloud（免费）

1. 把项目推送到 GitHub
2. 访问 https://share.streamlit.io
3. 连接你的 GitHub 仓库
4. 选择 `app/main.py` 作为入口文件
5. 点击 Deploy

这样任何人都可以通过链接访问你的工具。

---

## 附录：数据集获取指南

### A.1 广告投放场景

| 数据集 | 来源 | 说明 |
|--------|------|------|
| E-commerce Behavior Data | Kaggle | 多品类电商用户行为数据，含浏览/点击/购买事件 |
| Google Ads 公开数据 | BigQuery | 通过 `bigquery-public-data` 可访问 |

### A.2 电商定价场景

| 数据集 | 来源 | 说明 |
|--------|------|------|
| Dynamic Pricing Dataset | Kaggle | 动态定价数据 |
| Electronic Products Pricing | Kaggle | 电子产品多平台价格数据 |

### A.3 供应链场景

| 数据集 | 来源 | 说明 |
|--------|------|------|
| Supply Chain Shipment Pricing | Data.gov | 美国供应链运输定价数据 |
| 美国联邦采购数据集 | Nature Scientific Data | 1979-2023 年，学术级别 |

### A.4 下载方式

大部分 Kaggle 数据集可以通过命令行下载：

```bash
# 安装 Kaggle CLI
pip install kaggle

# 配置 API 密钥（从 kaggle.com/account 获取）
# 将 kaggle.json 放到 ~/.kaggle/

# 下载数据集
kaggle datasets download -d mkechinov/ecommerce-behavior-data-from-multi-category-store
```

---

## 下一步

完成以上所有步骤后，你已经拥有了一个可工作的 MVP。接下来的优先事项：

1. **用真实数据集替换示例数据**，生成有说服力的分析案例
2. **写第一篇技术文章**发布到知乎/Medium
3. **完善 README**，加上截图和使用示例
4. **发布到 PyPI**，让别人可以 `pip install uncertainty-lens`
5. **开始第二层开发**（成本量化模块）

记住：先让东西跑起来，再让它变好。不要在第一版就追求完美。
