# UncertaintyLens 完整新手教程

> 从零开始，手把手教你使用 UncertaintyLens 分析数据中的不确定性。
> 无需机器学习背景，跟着步骤走就能出结果。

---

## 目录

1. [什么是 UncertaintyLens？（通俗版）](#1-什么是-uncertaintylens通俗版)
2. [环境准备与安装](#2-环境准备与安装)
3. [5 分钟快速上手](#3-5-分钟快速上手)
4. [理解分析报告](#4-理解分析报告)
5. [进阶：注册更多检测器](#5-进阶注册更多检测器)
6. [生成交互式 HTML 报告](#6-生成交互式-html-报告)
7. [实时流式监控](#7-实时流式监控)
8. [运行测试套件](#8-运行测试套件)
9. [常见问题 FAQ](#9-常见问题-faq)
10. [术语表](#10-术语表)

---

## 1. 什么是 UncertaintyLens？（通俗版）

想象你拿到一份 Excel 表格，里面有上千行数据。你想用这些数据做决策，但你不确定：

- 有些列有很多空值，靠谱吗？
- 某些数字是不是异常值（比如月薪 100 万）？
- 不同分组之间的数据分布差异大不大？
- 数据会不会随时间发生"漂移"（今天的规律明天就不适用了）？

**UncertaintyLens 就是帮你回答这些问题的工具。** 它会自动扫描你的数据，给每一列打出一个"不确定性分数"（0 = 完全可靠，1 = 非常不可靠），然后告诉你问题出在哪里、该怎么修。

**核心能力：**

- **10 个检测器**：分别检查缺失值、异常值、方差不稳定、分布偏移、零膨胀等问题
- **自动归因**：不只告诉你"这列有问题"，还告诉你"问题的 60% 来自缺失值，30% 来自异常值"
- **交互式报告**：一键生成 HTML 报告，带热力图、雷达图、行动建议
- **流式监控**：数据实时流入时，在线追踪质量变化

---

## 2. 环境准备与安装

### 2.1 系统要求

- Python 3.10 或更高版本
- pip 包管理器

### 2.2 安装步骤

```bash
# 第一步：克隆项目（或解压你拿到的项目文件夹）
cd UncertaintyLens

# 第二步：安装依赖
pip install -r requirements.txt

# 第三步：以"开发模式"安装本项目（这样你可以直接 import）
pip install -e .
```

**验证安装是否成功：**

```bash
python -c "from uncertainty_lens import UncertaintyPipeline; print('安装成功！')"
```

如果看到 `安装成功！`，说明一切就绪。

### 2.3 依赖清单

| 包名 | 用途 | 版本要求 |
|------|------|----------|
| pandas | 数据处理 | ≥ 2.0 |
| numpy | 数值计算 | ≥ 1.24 |
| scipy | 统计检验 | ≥ 1.10 |
| scikit-learn | 异常检测 | ≥ 1.3 |
| plotly | 交互图表 | ≥ 5.15 |
| matplotlib | 静态图表 | ≥ 3.7 |
| streamlit | Web 界面 | ≥ 1.28 |

---

## 3. 5 分钟快速上手

### 3.1 最简单的例子

```python
import pandas as pd
import numpy as np
from uncertainty_lens import UncertaintyPipeline

# ── 第一步：准备数据 ──
# 这里我们用随机数据模拟，你换成自己的 CSV 也行
rng = np.random.default_rng(42)
n = 1000

df = pd.DataFrame({
    "年龄": rng.normal(35, 10, n),                                    # 正常数据
    "月收入": np.where(rng.random(n) < 0.2, np.nan, rng.normal(8000, 3000, n)),  # 20% 缺失
    "消费金额": np.concatenate([rng.normal(500, 100, n-5), [50000, 60000, 70000, 80000, 90000]]),  # 有极端值
})

# ── 第二步：创建管线并分析 ──
pipeline = UncertaintyPipeline()
report = pipeline.analyze(df)

# ── 第三步：查看结果 ──
for col, info in report["uncertainty_index"].items():
    print(f"  {col}: 不确定性 = {info['composite_score']:.3f} ({info['level']})")
```

**预期输出（大致）：**

```
  月收入: 不确定性 = 0.543 (Medium)
  消费金额: 不确定性 = 0.312 (Medium-Low)
  年龄: 不确定性 = 0.045 (Low)
```

解读：月收入因为有 20% 缺失所以分数最高，消费金额因为有极端值所以排第二，年龄数据干净所以分数很低。

### 3.2 用你自己的 CSV 文件

```python
import pandas as pd
from uncertainty_lens import UncertaintyPipeline

# 读取你的数据
df = pd.read_csv("你的数据.csv")

# 分析
pipeline = UncertaintyPipeline()
report = pipeline.analyze(df)

# 打印摘要
summary = report["summary"]
print(f"整体不确定性: {summary['overall_uncertainty']:.3f} ({summary['overall_level']})")
print(f"高风险特征: {summary['high_uncertainty_features']}")
print(f"低风险特征: {summary['low_uncertainty_features']}")
```

### 3.3 带分组对比

如果你的数据有一列代表"分组"（比如性别、地区、渠道），可以让管线对比不同组的差异：

```python
report = pipeline.analyze(df, group_col="性别")
```

这会额外检查：男性组和女性组的数据分布是否有显著差异。如果差异大，相关特征的不确定性分数会更高。

---

## 4. 理解分析报告

`pipeline.analyze()` 返回一个字典，核心结构如下：

### 4.1 uncertainty_index（最重要）

```python
report["uncertainty_index"]
# 返回：
# {
#   "月收入": {
#       "composite_score": 0.543,     # 综合不确定性分数 (0~1)
#       "missing_score": 0.800,       # 缺失值检测器的分数
#       "anomaly_score": 0.120,       # 异常值检测器的分数
#       "variance_score": 0.250,      # 方差检测器的分数
#       "level": "Medium"             # 等级标签
#   },
#   ...
# }
```

**分数含义：**

| 分数范围 | 等级 | 含义 |
|----------|------|------|
| 0.0 ~ 0.2 | Low | 数据可靠，可以放心使用 |
| 0.2 ~ 0.4 | Medium-Low | 基本可靠，但有小问题值得留意 |
| 0.4 ~ 0.6 | Medium | 需要关注，建议做数据清洗后再使用 |
| 0.6 ~ 0.8 | Medium-High | 明显有问题，使用前必须处理 |
| 0.8 ~ 1.0 | High | 数据严重不可靠，不建议直接使用 |

### 4.2 explanation（自动归因）

管线会自动生成归因分析（"问题到底出在哪？"）：

```python
expl = report["explanation"]

# 查看某一列的详细归因
col_expl = expl["feature_explanations"]["月收入"]
print(col_expl["summary"])
# → "月收入 的不确定性较高 (0.543), 主要来源: 缺失模式检测 (73.5%), 方差波动检测 (18.2%), ..."

# 查看行动建议
for action in expl["action_plan"]:
    print(f"  [{action['severity']}] {action['label']}: {action['action']}")
# → [high] 缺失模式检测: 建议先用多重插补 (MICE) 填充缺失值，或分析缺失机制 (MCAR/MAR/MNAR)...
```

### 4.3 summary（全局概览）

```python
report["summary"]
# {
#   "overall_uncertainty": 0.300,
#   "overall_level": "Medium-Low",
#   "total_features_analyzed": 3,
#   "high_uncertainty_features": ["月收入"],
#   "low_uncertainty_features": ["年龄"],
#   ...
# }
```

---

## 5. 进阶：注册更多检测器

默认管线只用了 3 个基础检测器（缺失、异常、方差）。你可以注册更多检测器来获得更全面的分析：

### 5.1 推荐的全功能配置

```python
from uncertainty_lens import UncertaintyPipeline
from uncertainty_lens.detectors import (
    ConformalShiftDetector,     # 分布偏移检测
    UncertaintyDecomposer,     # 不确定性分解（随机 vs 认知）
    JackknifePlusDetector,     # 预测区间估计
    MMDShiftDetector,          # 多维分布漂移
    ZeroInflationDetector,     # 零值膨胀检测
    DeepEnsembleDetector,      # 深度集成不确定性
)

pipeline = UncertaintyPipeline()

# 注册额外检测器（名称、检测器对象、权重）
pipeline.register("conformal_shift", ConformalShiftDetector(seed=42), weight=0.1)
pipeline.register("decomposition", UncertaintyDecomposer(n_bootstrap=100, seed=42), weight=0.15)
pipeline.register("jackknife_plus", JackknifePlusDetector(seed=42), weight=0.1)
pipeline.register("mmd_shift", MMDShiftDetector(n_permutations=200, seed=42), weight=0.1)
pipeline.register("zero_inflation", ZeroInflationDetector(), weight=0.2)
pipeline.register("deep_ensemble", DeepEnsembleDetector(n_ensemble=5, seed=42), weight=0.1)

# 分析（和之前一样）
report = pipeline.analyze(df, group_col="性别")
```

### 5.2 各检测器简介

| 检测器 | 通俗解释 | 适用场景 |
|--------|----------|----------|
| MissingPatternDetector | 检查哪些列有空值，空值是随机的还是有规律的 | 所有数据 |
| AnomalyDetector | 找出异常的极端值（比如月薪 100 万） | 所有数值列 |
| VarianceDetector | 检查数据波动是否稳定 | 时间序列、分组数据 |
| ConformalShiftDetector | 对比不同组的数据分布差异 | 有分组列时 |
| UncertaintyDecomposer | 把不确定性拆成"数据本身的噪音"和"我们不了解的部分" | 想深入理解不确定性来源时 |
| JackknifePlusDetector | 估算预测的置信区间宽度 | 评估预测可靠性 |
| MMDShiftDetector | 多维度综合检测分布是否发生变化 | 数据漂移监控 |
| ZeroInflationDetector | 检测某列零值是否异常多 | 计数数据（如订单量、索赔次数） |
| DeepEnsembleDetector | 用多个模型"投票"来衡量不确定性 | 需要高精度评估时 |

---

## 6. 生成交互式 HTML 报告

### 6.1 一步生成

```python
from uncertainty_lens import UncertaintyPipeline
import pandas as pd

df = pd.read_csv("你的数据.csv")

pipeline = UncertaintyPipeline()
report = pipeline.analyze(df, group_col="分组列名")  # group_col 可选

# 生成报告
path = pipeline.generate_report(
    df=df,
    output_path="我的分析报告.html",
    title="数据质量分析报告"
)
print(f"报告已生成: {path}")
```

用浏览器打开 `我的分析报告.html` 即可查看。

### 6.2 报告包含什么？

报告是一个独立的 HTML 文件（无需网络），包含以下内容：

1. **概览仪表盘** — 数据集基本信息、整体不确定性等级
2. **热力图** — 每个特征 × 每个检测器的分数矩阵，颜色越红越有问题
3. **排名条形图** — 按不确定性从高到低排列的特征
4. **分布图** — 每个特征的数值分布直方图
5. **归因分解图** — 每个特征的问题来源堆叠图（"60% 来自缺失，30% 来自异常"）
6. **雷达图** — 数据集在各检测维度上的"健康度"
7. **行动计划** — 按优先级排列的修复建议卡片

### 6.3 使用现成的 Demo 脚本

```bash
cd UncertaintyLens
python examples/generate_demo_report.py
```

这会用内置的示例数据集生成一份完整报告 `uncertainty_demo_report.html`。

---

## 7. 实时流式监控

如果你的数据是实时流入的（比如传感器数据、交易流水），可以用 `StreamingDetector` 做在线监控：

```python
from uncertainty_lens.detectors import StreamingDetector
import pandas as pd
import numpy as np

detector = StreamingDetector(window_size=200)

# 模拟数据流：每次来 50 行
rng = np.random.default_rng(42)

for i in range(20):
    # 模拟：前 10 批温度均值 25°C，后 10 批均值跳到 35°C
    if i < 10:
        batch = pd.DataFrame({"温度": rng.normal(25, 2, 50), "湿度": rng.normal(60, 5, 50)})
    else:
        batch = pd.DataFrame({"温度": rng.normal(35, 2, 50), "湿度": rng.normal(60, 5, 50)})

    result = detector.update(batch)

    if result["drift_detected"]:
        print(f"⚠️ 第 {i+1} 批: 检测到漂移！")
        for alert in result["alerts"]:
            print(f"   {alert}")
    else:
        print(f"✅ 第 {i+1} 批: 正常 (总数据: {result['stats']['n_total']})")
```

---

## 8. 运行测试套件

### 8.1 运行单元测试

```bash
cd UncertaintyLens
PYTHONPATH=. python -m pytest tests/ -v
```

预期结果：153 个测试通过（5 个跳过是正常的，因为 CatBoost 是可选依赖）。

### 8.2 运行基准测试

```bash
# 核心基准（4 个经典数据集，39 项检查）
PYTHONPATH=. python examples/benchmark_all.py

# 盲测（3 个独立数据集，25 项检查）
PYTHONPATH=. python examples/benchmark_blind.py

# 扩展测试（6 个极端/对抗数据集，27 项检查）
PYTHONPATH=. python examples/benchmark_extended.py
```

### 8.3 一键运行全部 + 生成仪表盘

```bash
PYTHONPATH=. python examples/test_dashboard.py
```

这会依次运行所有 4 个测试套件（单元测试 + 3 个基准），并生成 `test_dashboard.html`。用浏览器打开即可看到完整的测试结果仪表盘。

预期结果：**244/244 全部通过**。

---

## 9. 常见问题 FAQ

### Q: 我的数据有非数值列（比如"姓名"、"城市"），会出错吗？

不会。管线会自动跳过非数值列，只分析数值列。非数值列可以作为 `group_col` 参数传入做分组对比。

### Q: composite_score 是怎么算出来的？

每个检测器给每列打一个 0~1 的分数，然后按权重加权平均。如果某个检测器对这列给了 0 分（说明没检测到问题），它的权重会自动降低，避免稀释其他检测器发现的真实问题。

### Q: 我只关心缺失值，不想跑其他检测器，可以吗？

可以。调整权重即可：

```python
pipeline = UncertaintyPipeline(weights={"missing": 1.0, "anomaly": 0.0, "variance": 0.0})
```

### Q: 数据量很大（百万行），会不会很慢？

管线针对大数据做了优化。如果确实很慢，建议：
1. 先用流式检测器 `StreamingDetector` 做快速扫描
2. 或者随机采样 10% 的数据先跑一遍，看哪些列需要关注

### Q: 报告是中文还是英文？

默认报告界面是英文，但归因分析（explanation）默认是中文。你可以切换：

```python
from uncertainty_lens.detectors import UncertaintyExplainer

explainer = UncertaintyExplainer(language="en")  # 改为英文
result = explainer.explain(report)
```

### Q: 怎么写自定义检测器？

只需实现一个 `analyze(df, **kwargs)` 方法，返回包含 `"uncertainty_scores"` 的字典：

```python
class MyDetector:
    def analyze(self, df, **kwargs):
        scores = {}
        for col in df.select_dtypes(include="number").columns:
            scores[col] = 0.5  # 你的检测逻辑
        return {"uncertainty_scores": scores}

pipeline.register("my_detector", MyDetector(), weight=0.2)
```

---

## 10. 术语表

| 术语 | 通俗解释 |
|------|----------|
| 不确定性 (Uncertainty) | 数据中"不靠谱"的程度。分数越高，用这列数据做决策的风险越大。 |
| 检测器 (Detector) | 从某个角度检查数据问题的工具。比如"缺失检测器"专门看空值，"异常检测器"专门找极端值。 |
| 管线 (Pipeline) | 把多个检测器串起来，一次性跑完所有检查。 |
| 综合分数 (Composite Score) | 把所有检测器的结果加权合并后的总分。 |
| 归因 (Attribution) | "问题的贡献分解"——告诉你这列的总分里，各个检测器分别贡献了多少。 |
| 分布偏移 (Distribution Shift) | 数据的统计特征发生了变化（比如用户群体变了）。 |
| 零膨胀 (Zero Inflation) | 某列的零值比正常情况多得多（比如大部分用户投诉次数为 0）。 |
| MMD | 最大均值差异——一种衡量两组数据分布是否相同的方法。 |
| EWMA | 指数加权移动平均——越近的数据权重越大，用于捕捉趋势变化。 |
| Page-Hinkley | 一种在线检测数据均值是否突然变化的统计方法。 |
| Welford 算法 | 一种不需要存储所有历史数据就能实时计算平均值和方差的方法。 |
| 置信区间 (Confidence Interval) | "我有 95% 的把握认为真实值在这个范围内"——范围越大说明越不确定。 |
| Bootstrap | 从数据中反复抽样来估计统计量的不确定性。类似"重复做实验看结果波动有多大"。 |

---

## 附录：项目文件结构

```
UncertaintyLens/
├── uncertainty_lens/           # 核心代码
│   ├── __init__.py            # 入口：导出 UncertaintyPipeline
│   ├── pipeline.py            # 管线：串联所有检测器
│   ├── detectors/             # 检测器合集
│   │   ├── missing_pattern.py    # 缺失模式
│   │   ├── anomaly.py            # 异常值
│   │   ├── variance.py           # 方差波动
│   │   ├── conformal_shift.py    # 分布偏移 (Conformal)
│   │   ├── decomposition.py      # 不确定性分解
│   │   ├── jackknife_plus.py     # Jackknife+ 预测区间
│   │   ├── mmd_shift.py          # MMD 多维漂移
│   │   ├── zero_inflation.py     # 零膨胀检测
│   │   ├── deep_ensemble.py      # 深度集成
│   │   ├── streaming_detector.py # 流式在线检测
│   │   └── uncertainty_explainer.py  # 归因解释器
│   ├── visualizers/           # 可视化
│   │   ├── report.py             # HTML 报告生成
│   │   ├── heatmap.py            # 热力图
│   │   ├── explainer_charts.py   # 归因图表
│   │   └── ...
│   └── quantifiers/           # 量化工具
├── tests/                     # 单元测试 (153 个)
├── examples/                  # 示例与基准
│   ├── benchmark_all.py          # 核心基准 (39 项)
│   ├── benchmark_blind.py        # 盲测 (25 项)
│   ├── benchmark_extended.py     # 扩展测试 (27 项)
│   ├── test_dashboard.py         # 一键全测 + 仪表盘
│   └── generate_demo_report.py   # 生成 Demo 报告
├── requirements.txt           # 依赖清单
├── pyproject.toml             # 项目配置
└── README.md
```

---

*教程版本：v1.0 — 对应 UncertaintyLens v1.0.0*
