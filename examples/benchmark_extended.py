"""
UncertaintyLens 扩展基准测试

4 类新场景, 覆盖合成数据无法触及的盲区:
  A. 仿真真实数据集 (Titanic-style / Adult-style)  — 自然缺失 + 混合类型
  B. 对抗性边界数据集                                — 专门挑战检测器的极限
  C. 极端工程场景                                    — 宽表/小样本/常量列/全缺失
  D. 跨域混合                                        — 时间索引 + 类别编码 + 数值

用法:
  PYTHONPATH=. python examples/benchmark_extended.py
"""

import sys, time, warnings
import numpy as np
import pandas as pd
from collections import defaultdict

from uncertainty_lens import UncertaintyPipeline
from uncertainty_lens.detectors import (
    ConformalShiftDetector,
    UncertaintyDecomposer,
    JackknifePlusDetector,
    MMDShiftDetector,
    ZeroInflationDetector,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ─── Pipeline ────────────────────────────────────────────────────────


def build_pipeline():
    p = UncertaintyPipeline(weights={"missing": 0.35, "anomaly": 0.25, "variance": 0.25})
    p.register("conformal_shift", ConformalShiftDetector(seed=42), weight=0.1)
    p.register("decomposition", UncertaintyDecomposer(n_bootstrap=200, seed=42), weight=0.15)
    p.register("jackknife_plus", JackknifePlusDetector(n_folds=10, seed=42), weight=0.1)
    p.register("mmd_shift", MMDShiftDetector(n_permutations=200, seed=42), weight=0.1)
    p.register("zero_inflation", ZeroInflationDetector(zero_threshold=0.5), weight=0.2)
    return p


# ═══════════════════════════════════════════════════════════════════
# A. 仿真真实数据集
# ═══════════════════════════════════════════════════════════════════


def generate_titanic_like(n=1200, seed=314):
    """
    Titanic 风格生存数据: 自然缺失 + 类别编码 + 幸存偏差.
    age: 20% MCAR 缺失 (模拟真实 Titanic)
    cabin_code: 77% 缺失 (头等舱记录更全)
    fare: 极端右偏 (头等舱 vs 三等舱)
    pclass: 1/2/3, 分组列
    survived: 0/1, 与 pclass 强相关 (选择偏差)
    sibsp: 零膨胀计数
    embarked_code: 5% 编码缺失
    """
    rng = np.random.default_rng(seed)
    pclass = rng.choice([1, 2, 3], n, p=[0.25, 0.25, 0.50])
    age = rng.normal(30, 14, n).clip(1, 80)
    age[rng.random(n) < 0.20] = np.nan  # 20% MCAR
    # fare 与 pclass 强相关
    fare_base = np.where(pclass == 1, 80, np.where(pclass == 2, 25, 8))
    fare = rng.lognormal(np.log(fare_base), 0.6)
    # cabin 77% 缺失, 头等舱缺失少
    cabin_code = rng.integers(1, 150, n).astype(float)
    missing_prob = np.where(pclass == 1, 0.40, np.where(pclass == 2, 0.75, 0.90))
    cabin_code[rng.random(n) < missing_prob] = np.nan
    # sibsp 零膨胀
    sibsp = np.zeros(n, dtype=float)
    has_sib = rng.random(n) < 0.35
    sibsp[has_sib] = rng.poisson(1.5, has_sib.sum()).astype(float)
    # embarked
    embarked_code = rng.choice([1, 2, 3], n, p=[0.72, 0.19, 0.09]).astype(float)
    embarked_code[rng.random(n) < 0.05] = np.nan
    # survived (选择偏差)
    surv_prob = np.where(pclass == 1, 0.63, np.where(pclass == 2, 0.47, 0.24))
    survived = (rng.random(n) < surv_prob).astype(float)

    return pd.DataFrame(
        {
            "age": age,
            "fare": fare,
            "cabin_code": cabin_code,
            "sibsp": sibsp,
            "embarked_code": embarked_code,
            "survived": survived,
            "pclass": pclass.astype(str),
        }
    )


def generate_adult_like(n=5000, seed=271):
    """
    Adult/Census Income 风格: 混合类型 + 自然偏态 + 组间偏移.
    hours_per_week: 双峰 (part-time ~20h + full-time ~40h)
    capital_gain: 95% 零膨胀
    education_years: 离散阶梯 (8,10,12,13,16)
    fnlwgt: 对数正态长尾 (人口权重)
    age: 右偏 (劳动力年龄分布)
    group: "<=50K" / ">50K"
    """
    rng = np.random.default_rng(seed)
    n_high = int(n * 0.24)
    n_low = n - n_high
    group = np.array(["<=50K"] * n_low + [">50K"] * n_high)
    rng.shuffle(group)

    age = np.where(
        group == ">50K", rng.normal(44, 10, n).clip(17, 90), rng.normal(33, 13, n).clip(17, 90)
    )
    hours = np.where(
        rng.random(n) < 0.15, rng.normal(20, 5, n).clip(1, 35), rng.normal(42, 8, n).clip(35, 99)
    )
    education_years = rng.choice(
        [8, 10, 12, 13, 14, 16], n, p=[0.05, 0.15, 0.30, 0.20, 0.15, 0.15]
    ).astype(float)
    capital_gain = np.zeros(n)
    has_gain = rng.random(n) < 0.05
    capital_gain[has_gain] = rng.lognormal(9, 1.2, has_gain.sum())
    fnlwgt = rng.lognormal(12, 0.8, n)

    return pd.DataFrame(
        {
            "age": age,
            "hours_per_week": hours,
            "education_years": education_years,
            "capital_gain": capital_gain,
            "fnlwgt": fnlwgt,
            "group": group,
        }
    )


# ═══════════════════════════════════════════════════════════════════
# B. 对抗性边界数据集
# ═══════════════════════════════════════════════════════════════════


def generate_adversarial(n=2000, seed=159):
    """
    专门设计来挑战检测器边界的数据:
    borderline_outlier: 异常值恰好在 IQR*3 边界附近 (~5% 刚好越界)
    masked_shift: 两组均值差 0.3σ (非常微弱偏移)
    perfect_corr: x 和 y 完全线性相关 (r=0.999), 但加了极少噪声
    sneaky_missing: 缺失率恰好 1% (容易被忽略)
    uniform_flat: 均匀分布, 无异常无偏移 (应该全低分)
    bimodal_symmetric: 完美对称双峰 (均值=0, 但方差大)
    """
    rng = np.random.default_rng(seed)
    # borderline outliers
    base = rng.normal(0, 1, n)
    q25, q75 = np.percentile(base, [25, 75])
    iqr = q75 - q25
    upper = q75 + 3 * iqr
    # 注入5%恰好在边界的值
    n_border = int(n * 0.05)
    base[:n_border] = upper + rng.uniform(-0.05, 0.15, n_border)
    borderline_outlier = base.copy()

    # masked shift
    group = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
    masked_shift = np.where(group == "A", rng.normal(0, 1, n), rng.normal(0.3, 1, n))  # 0.3σ 偏移

    # perfect correlation
    x = rng.normal(0, 1, n)
    perfect_corr_x = x.copy()
    perfect_corr_y = x * 2.5 + 3.0 + rng.normal(0, 0.01, n)

    # sneaky missing (1%)
    sneaky = rng.normal(50, 10, n)
    sneaky[rng.random(n) < 0.01] = np.nan

    # uniform flat
    uniform_flat = rng.uniform(0, 100, n)

    # bimodal symmetric
    bimodal = np.where(rng.random(n) < 0.5, rng.normal(-3, 1, n), rng.normal(3, 1, n))

    return pd.DataFrame(
        {
            "borderline_outlier": borderline_outlier,
            "masked_shift": masked_shift,
            "perfect_corr_x": perfect_corr_x,
            "perfect_corr_y": perfect_corr_y,
            "sneaky_missing": sneaky,
            "uniform_flat": uniform_flat,
            "bimodal_symmetric": bimodal,
            "group": group,
        }
    )


# ═══════════════════════════════════════════════════════════════════
# C. 极端工程场景
# ═══════════════════════════════════════════════════════════════════


def generate_wide_table(n=200, n_cols=80, seed=628):
    """
    宽表: 80列, 200行. 模拟基因组/传感器阵列.
    其中:
      - 5列有 30% 缺失
      - 3列是常量
      - 2列有极端异常值 (>10σ)
      - 其余正常
    """
    rng = np.random.default_rng(seed)
    data = {}
    for i in range(n_cols):
        col = f"feat_{i:03d}"
        if i < 5:  # 缺失列
            vals = rng.normal(0, 1, n)
            vals[rng.random(n) < 0.30] = np.nan
        elif i < 8:  # 常量列
            vals = np.full(n, 42.0)
        elif i < 10:  # 极端异常
            vals = rng.normal(0, 1, n)
            vals[:3] = rng.uniform(50, 100, 3)  # 50-100σ 异常
        else:
            vals = rng.normal(i * 0.1, 1 + i * 0.05, n)
        data[col] = vals
    return pd.DataFrame(data)


def generate_tiny_sample(n=50, seed=999):
    """
    极小样本: 50行, 测试统计功效下降.
    clean: 标准正态
    noisy: σ=5 正态
    missing: 20% 缺失
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "clean": rng.normal(0, 1, n),
            "noisy": rng.normal(0, 5, n),
            "missing_col": np.where(rng.random(n) < 0.2, np.nan, rng.normal(0, 1, n)),
            "group": rng.choice(["X", "Y"], n),
        }
    )


# ═══════════════════════════════════════════════════════════════════
# D. 跨域混合
# ═══════════════════════════════════════════════════════════════════


def generate_retail_mixed(n=4000, seed=503):
    """
    零售混合数据: 时间索引 + 类别编码 + 数值 + 周期性.
    day_of_week: 1-7, 周期性
    month: 1-12
    store_id: 编码, 10% 缺失
    sales: 对数正态 + 节假日尖峰 (12月)
    temperature: 正弦季节性
    promotion: 0/1 二元 (零膨胀 80%=0)
    inventory: 正常连续
    customer_count: Poisson
    region: A/B/C 分组
    """
    rng = np.random.default_rng(seed)
    month = rng.integers(1, 13, n)
    day_of_week = rng.integers(1, 8, n).astype(float)
    store_id = rng.integers(100, 120, n).astype(float)
    store_id[rng.random(n) < 0.10] = np.nan
    # sales with December spike
    base_sales = rng.lognormal(6, 0.8, n)
    december_mask = month == 12
    base_sales[december_mask] *= rng.uniform(1.5, 3.0, december_mask.sum())
    sales = base_sales
    temperature = 20 + 15 * np.sin(2 * np.pi * (month - 1) / 12) + rng.normal(0, 3, n)
    promotion = (rng.random(n) < 0.20).astype(float)
    inventory = rng.normal(500, 100, n).clip(0, None)
    customer_count = rng.poisson(80, n).astype(float)
    region = rng.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2])

    return pd.DataFrame(
        {
            "day_of_week": day_of_week,
            "month": month.astype(float),
            "store_id": store_id,
            "sales": sales,
            "temperature": temperature,
            "promotion": promotion,
            "inventory": inventory,
            "customer_count": customer_count,
            "region": region,
        }
    )


# ═══════════════════════════════════════════════════════════════════
# 检查注册
# ═══════════════════════════════════════════════════════════════════

CHECKS = []


def check(dataset, name, category):
    def decorator(fn):
        CHECKS.append((dataset, name, category, fn))
        return fn

    return decorator


# ── A. Titanic-like ──


@check("Titanic", "age 20% MCAR 缺失被检测", "missing")
def _(r):
    return r["uncertainty_index"].get("age", {}).get("missing_score", 0) > 0.01


@check("Titanic", "cabin_code 77% 大量缺失 (最高)", "missing")
def _(r):
    cab = r["uncertainty_index"].get("cabin_code", {}).get("missing_score", 0)
    age_m = r["uncertainty_index"].get("age", {}).get("missing_score", 0)
    return cab > age_m  # cabin 缺失应严重于 age


@check("Titanic", "fare 右偏长尾异常", "anomaly")
def _(r):
    return r["uncertainty_index"].get("fare", {}).get("anomaly_score", 0) > 0.05


@check("Titanic", "sibsp 零膨胀被识别", "variance")
def _(r):
    zi = r.get("zero_inflation_analysis", {})
    return "sibsp" in zi.get("zero_inflated_features", [])


@check("Titanic", "pclass 组间偏移 (生存偏差)", "shift")
def _(r):
    return bool(r.get("conformal_shift_analysis", {}).get("group_shift", {}))


@check("Titanic", "embarked 缺失被检测", "missing")
def _(r):
    return r["uncertainty_index"].get("embarked_code", {}).get("missing_score", 0) > 0.0


# ── A. Adult-like ──


@check("Adult", "capital_gain 95% 零膨胀", "variance")
def _(r):
    zi = r.get("zero_inflation_analysis", {})
    return "capital_gain" in zi.get("zero_inflated_features", [])


@check("Adult", "hours_per_week 双峰方差", "variance")
def _(r):
    return r["uncertainty_index"].get("hours_per_week", {}).get("variance_score", 0) > 0.1


@check("Adult", "fnlwgt 对数正态长尾", "anomaly")
def _(r):
    return r["uncertainty_index"].get("fnlwgt", {}).get("anomaly_score", 0) > 0.05


@check("Adult", "收入组间偏移", "shift")
def _(r):
    return bool(r.get("conformal_shift_analysis", {}).get("group_shift", {}))


@check("Adult", "education_years < fnlwgt (离散阶梯 vs 长尾)", "ranking")
def _(r):
    edu = r["uncertainty_index"].get("education_years", {}).get("composite_score", 1)
    fn = r["uncertainty_index"].get("fnlwgt", {}).get("composite_score", 0)
    return edu < fn


# ── B. Adversarial ──


@check("Adversarial", "borderline_outlier 异常被检测 (边界挑战)", "anomaly")
def _(r):
    return r["uncertainty_index"].get("borderline_outlier", {}).get("anomaly_score", 0) > 0.01


@check("Adversarial", "uniform_flat 低不确定性 (无问题基线)", "ranking")
def _(r):
    return r["uncertainty_index"].get("uniform_flat", {}).get("composite_score", 1) < 0.5


@check("Adversarial", "bimodal_symmetric 方差信号 (双峰大方差)", "variance")
def _(r):
    return r["uncertainty_index"].get("bimodal_symmetric", {}).get("variance_score", 0) > 0.1


@check("Adversarial", "sneaky_missing 1% 缺失仍被检测到", "missing")
def _(r):
    return r["uncertainty_index"].get("sneaky_missing", {}).get("missing_score", 0) > 0.0


@check("Adversarial", "perfect_corr_y < borderline (可预测 vs 异常)", "ranking")
def _(r):
    py = r["uncertainty_index"].get("perfect_corr_y", {}).get("composite_score", 1)
    bo = r["uncertainty_index"].get("borderline_outlier", {}).get("composite_score", 0)
    return py < bo


# ── C. Wide table ──


@check("WideTable", "feat_000-004 缺失列被检测", "missing")
def _(r):
    detected = sum(
        1
        for i in range(5)
        if r["uncertainty_index"].get(f"feat_{i:03d}", {}).get("missing_score", 0) > 0.01
    )
    return detected >= 4  # 至少 4/5 被检测到


@check("WideTable", "feat_008-009 极端异常被检测", "anomaly")
def _(r):
    detected = sum(
        1
        for i in [8, 9]
        if r["uncertainty_index"].get(f"feat_{i:03d}", {}).get("anomaly_score", 0) > 0.05
    )
    return detected >= 1


@check("WideTable", "处理 80 列不崩溃 (工程稳定性)", "ranking")
def _(r):
    return len(r["uncertainty_index"]) >= 70  # 排除常量列后至少 70 列有分数


# ── C. Tiny sample ──


@check("TinySample", "50行数据不崩溃", "ranking")
def _(r):
    return len(r["uncertainty_index"]) >= 2


@check("TinySample", "missing_col 缺失被检测 (小样本)", "missing")
def _(r):
    return r["uncertainty_index"].get("missing_col", {}).get("missing_score", 0) > 0.0


@check("TinySample", "noisy > clean (噪声排序正确, 小样本)", "ranking")
def _(r):
    n = r["uncertainty_index"].get("noisy", {}).get("composite_score", 0)
    c = r["uncertainty_index"].get("clean", {}).get("composite_score", 1)
    return n > c


# ── D. Retail mixed ──


@check("Retail", "store_id 10% 缺失被检测", "missing")
def _(r):
    return r["uncertainty_index"].get("store_id", {}).get("missing_score", 0) > 0.01


@check("Retail", "sales 对数正态长尾 (含节假日尖峰)", "anomaly")
def _(r):
    return r["uncertainty_index"].get("sales", {}).get("anomaly_score", 0) > 0.05


@check("Retail", "promotion 零膨胀 (80% 为零)", "variance")
def _(r):
    zi = r.get("zero_inflation_analysis", {})
    return "promotion" in zi.get("zero_inflated_features", [])


@check("Retail", "区域间分布偏移", "shift")
def _(r):
    return bool(r.get("conformal_shift_analysis", {}).get("group_shift", {}))


@check("Retail", "inventory 稳定 < sales 不确定", "ranking")
def _(r):
    inv = r["uncertainty_index"].get("inventory", {}).get("composite_score", 1)
    sal = r["uncertainty_index"].get("sales", {}).get("composite_score", 0)
    return inv < sal


# ═══════════════════════════════════════════════════════════════════
# 数据集配置
# ═══════════════════════════════════════════════════════════════════

DATASETS = {
    "Titanic": (generate_titanic_like, "pclass", "A. 仿真真实 — 自然缺失/混合类型/幸存偏差"),
    "Adult": (generate_adult_like, "group", "A. 仿真真实 — 零膨胀/双峰/收入偏移"),
    "Adversarial": (generate_adversarial, "group", "B. 对抗性 — 边界异常/微弱偏移/完美相关"),
    "WideTable": (generate_wide_table, None, "C. 极端工程 — 80列宽表/常量列/极端异常"),
    "TinySample": (generate_tiny_sample, "group", "C. 极端工程 — 50行小样本"),
    "Retail": (generate_retail_mixed, "region", "D. 跨域混合 — 时间+类别+数值+周期"),
}


# ═══════════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════════


def main():
    n_datasets = len(DATASETS)
    n_checks = len(CHECKS)
    print("=" * 65)
    print("  UncertaintyLens 扩展基准测试")
    print(f"  {n_datasets} 个新场景 · {n_checks} 项检查 · 4 类测试类别")
    print("=" * 65)

    reports = {}
    for i, (name, (gen_fn, group_col, desc)) in enumerate(DATASETS.items(), 1):
        print(f"\n[{i}/{n_datasets}] {name} — {desc}")
        df = gen_fn()
        pipeline = build_pipeline()
        t0 = time.time()
        report = pipeline.analyze(df, group_col=group_col)
        elapsed = time.time() - t0
        reports[name] = report
        n_feats = report["summary"]["total_features_analyzed"]
        level = report["summary"]["overall_level"]
        print(f"  {len(df):>6,} 行 · {n_feats} 特征 · {level} · {elapsed:.1f}s")

    # ── 检查 ──
    print(f"\n{'=' * 65}")
    print("  检查明细")
    print(f"{'=' * 65}")

    results = []
    for dataset, name, category, fn in CHECKS:
        report = reports[dataset]
        try:
            passed = fn(report)
        except Exception as e:
            passed = False
        results.append((dataset, name, category, passed))

    current_ds = None
    for dataset, name, category, passed in results:
        if dataset != current_ds:
            current_ds = dataset
            print(f"\n  [{dataset}]")
        status = "✓" if passed else "✗"
        print(f"    {status} [{category:8s}] {name}")

    # ── 分类统计 ──
    print(f"\n{'=' * 65}")
    print("  按测试类别分类准确率")
    print(f"{'=' * 65}")

    type_labels = {
        "A": "A. 仿真真实数据集",
        "B": "B. 对抗性边界",
        "C": "C. 极端工程场景",
        "D": "D. 跨域混合",
    }
    type_map = {
        "Titanic": "A",
        "Adult": "A",
        "Adversarial": "B",
        "WideTable": "C",
        "TinySample": "C",
        "Retail": "D",
    }
    type_stats = defaultdict(lambda: [0, 0])
    for ds, _, _, passed in results:
        t = type_map[ds]
        type_stats[t][1] += 1
        if passed:
            type_stats[t][0] += 1
    for t in ["A", "B", "C", "D"]:
        p, total = type_stats[t]
        pct = p / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {type_labels[t]:25s}  {p}/{total}  {bar} {pct:.0f}%")

    # ── 按检测能力 ──
    print(f"\n{'=' * 65}")
    print("  按检测能力分类准确率")
    print(f"{'=' * 65}")
    cat_names = {
        "missing": "缺失检测",
        "anomaly": "异常值检测",
        "variance": "方差检测",
        "shift": "偏移检测",
        "ranking": "排序合理性",
    }
    cat_stats = defaultdict(lambda: [0, 0])
    for _, _, cat, passed in results:
        cat_stats[cat][1] += 1
        if passed:
            cat_stats[cat][0] += 1
    for cat in ["missing", "anomaly", "variance", "shift", "ranking"]:
        p, total = cat_stats[cat]
        pct = p / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {cat_names[cat]:12s}  {p}/{total}  {bar} {pct:.0f}%")

    # ── 总计 ──
    total_passed = sum(1 for _, _, _, p in results if p)
    print(f"\n{'─' * 65}")
    print(f"  扩展基准总计: {total_passed}/{n_checks} ({total_passed/n_checks:.1%})")

    if total_passed < n_checks:
        print(f"\n  未通过:")
        for ds, name, cat, p in results:
            if not p:
                print(f"    ✗ [{ds}] [{cat}] {name}")

    return 0 if total_passed == n_checks else 1


if __name__ == "__main__":
    sys.exit(main())
