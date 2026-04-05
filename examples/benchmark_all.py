"""
UncertaintyLens 统一批量基准测试

合并全部 7 个数据集，按检测能力分类统计准确率：
  - 缺失检测 (Missing)
  - 异常值检测 (Anomaly)
  - 方差/噪声检测 (Variance)
  - 分布偏移检测 (Shift)
  - 排序合理性 (Ranking)

输出:
  1. 逐项 PASS/FAIL 明细
  2. 按检测能力分类的准确率
  3. 按数据集分类的准确率
  4. 已知局限性清单

用法:
  PYTHONPATH=. python examples/benchmark_all.py
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from uncertainty_lens import UncertaintyPipeline
from uncertainty_lens.detectors import (
    ConformalShiftDetector,
    UncertaintyDecomposer,
    ConformalPredictor,
    JackknifePlusDetector,
    MMDShiftDetector,
    ZeroInflationDetector,
)

warnings.filterwarnings("ignore", category=UserWarning)


# ─── Pipeline ────────────────────────────────────────────────────────


def build_pipeline():
    pipeline = UncertaintyPipeline(weights={"missing": 0.35, "anomaly": 0.25, "variance": 0.25})
    pipeline.register("conformal_shift", ConformalShiftDetector(seed=42), weight=0.1)
    pipeline.register("decomposition", UncertaintyDecomposer(n_bootstrap=200, seed=42), weight=0.15)
    pipeline.register("jackknife_plus", JackknifePlusDetector(n_folds=10, seed=42), weight=0.1)
    pipeline.register("mmd_shift", MMDShiftDetector(n_permutations=200, seed=42), weight=0.1)
    pipeline.register("zero_inflation", ZeroInflationDetector(zero_threshold=0.5), weight=0.2)
    return pipeline


# ─── 数据集导入 ──────────────────────────────────────────────────────

from examples.benchmark_real_data import generate_housing, generate_wine, generate_census
from examples.benchmark_accuracy import (
    generate_medical,
    generate_sensor,
    generate_ecommerce,
    generate_financial,
)

# ─── 检查定义 ────────────────────────────────────────────────────────
# 每条检查: (数据集名, 检查名, 检测能力分类, 检查函数)

CHECKS = []


def check(dataset, name, category):
    """装饰器: 注册一条检查."""

    def decorator(fn):
        CHECKS.append((dataset, name, category, fn))
        return fn

    return decorator


# ── Housing (15K) ──


@check("Housing", "AvgRooms 离群值被检测", "anomaly")
def _(r):
    return r["uncertainty_index"].get("AvgRooms", {}).get("anomaly_score", 0) > 0.1


@check("Housing", "Population 重尾被检测", "anomaly")
def _(r):
    return r["uncertainty_index"].get("Population", {}).get("anomaly_score", 0) > 0.05


@check("Housing", "HouseValue 异方差被检测", "variance")
def _(r):
    return r["uncertainty_index"].get("HouseValue", {}).get("variance_score", 0) > 0.05


@check("Housing", "North/South 分布偏移被检测", "shift")
def _(r):
    return bool(r.get("conformal_shift_analysis", {}).get("group_shift", {}))


@check("Housing", "Lat/Lon 低于 MedIncome (空间特征更稳定)", "ranking")
def _(r):
    lat = r["uncertainty_index"].get("Latitude", {}).get("composite_score", 1)
    mi = r["uncertainty_index"].get("MedIncome", {}).get("composite_score", 0)
    return lat < mi


# ── Wine (6.5K) ──


@check("Wine", "residual_sugar 极端离群值", "anomaly")
def _(r):
    return r["uncertainty_index"].get("residual_sugar", {}).get("composite_score", 0) > 0.1


@check("Wine", "free_sulfur_dioxide 离群值", "anomaly")
def _(r):
    return r["uncertainty_index"].get("free_sulfur_dioxide", {}).get("anomaly_score", 0) > 0.05


@check("Wine", "无缺失数据无误报", "missing")
def _(r):
    return all(v.get("missing_score", 1) < 0.05 for v in r["uncertainty_index"].values())


@check("Wine", "pH 低不确定性", "ranking")
def _(r):
    return r["uncertainty_index"].get("pH", {}).get("composite_score", 1) < 0.5


@check("Wine", "Red/White 分布偏移被检测", "shift")
def _(r):
    return bool(r.get("conformal_shift_analysis", {}).get("group_shift", {}))


@check("Wine", "MMD 检测到 Red/White 联合偏移", "shift")
def _(r):
    mmd = r.get("mmd_shift_analysis", {})
    joint = mmd.get("joint_mmd", {})
    return any(v.get("shift_detected", False) for v in joint.values())


# ── Census (20K) ──


@check("Census", "capital_gain 方差信号 (零膨胀)", "variance")
def _(r):
    return r["uncertainty_index"].get("capital_gain", {}).get("variance_score", 0) > 0.8


@check("Census", "capital_loss 方差信号", "variance")
def _(r):
    return r["uncertainty_index"].get("capital_loss", {}).get("variance_score", 0) > 0.8


@check("Census", "education_num 低不确定性", "ranking")
def _(r):
    return r["uncertainty_index"].get("education_num", {}).get("composite_score", 1) < 0.6


@check("Census", "workclass_code 缺失被检测", "missing")
def _(r):
    return r["uncertainty_index"].get("workclass_code", {}).get("missing_score", 0) > 0.01


@check("Census", "hours_per_week 组间不确定性", "shift")
def _(r):
    return (
        r.get("conformal_shift_analysis", {}).get("uncertainty_scores", {}).get("hours_per_week", 0)
        > 0.1
    )


@check("Census", "capital_gain 零膨胀被识别", "variance")
def _(r):
    zi = r.get("zero_inflation_analysis", {})
    return "capital_gain" in zi.get("zero_inflated_features", [])


@check("Census", "capital_loss 零膨胀被识别", "variance")
def _(r):
    zi = r.get("zero_inflation_analysis", {})
    return "capital_loss" in zi.get("zero_inflated_features", [])


@check("Census", "MMD 检测到性别组间偏移", "shift")
def _(r):
    mmd = r.get("mmd_shift_analysis", {})
    return bool(mmd.get("group_shift", {}))


# ── Medical (10K) ──


@check("Medical", "cholesterol 缺失被检测 (MAR)", "missing")
def _(r):
    return r["uncertainty_index"].get("cholesterol", {}).get("missing_score", 0) > 0.01


@check("Medical", "BMI < cholesterol (无缺失更可靠)", "ranking")
def _(r):
    return r["uncertainty_index"].get("bmi", {}).get("composite_score", 0) < r[
        "uncertainty_index"
    ].get("cholesterol", {}).get("composite_score", 0)


@check("Medical", "systolic_bp 测量噪声方差", "variance")
def _(r):
    return r["uncertainty_index"].get("systolic_bp", {}).get("variance_score", 0) > 0.1


@check("Medical", "中心间系统偏差偏移", "shift")
def _(r):
    return bool(r.get("conformal_shift_analysis", {}).get("group_shift", {}))


@check("Medical", "glucose 异常值 (糖尿病人群)", "anomaly")
def _(r):
    return r["uncertainty_index"].get("glucose", {}).get("anomaly_score", 0) > 0.05


# ── Sensor (12K) ──


@check("Sensor", "temperature 传感器退化方差", "variance")
def _(r):
    return r["uncertainty_index"].get("temperature", {}).get("variance_score", 0) > 0.3


@check("Sensor", "phase 间温度概念漂移", "shift")
def _(r):
    return (
        r.get("conformal_shift_analysis", {}).get("uncertainty_scores", {}).get("temperature", 0)
        > 0.5
    )


@check("Sensor", "voltage 连续缺失 (传感器掉线)", "missing")
def _(r):
    return r["uncertainty_index"].get("voltage", {}).get("missing_score", 0) > 0.01


@check("Sensor", "vibration 突发异常", "anomaly")
def _(r):
    return r["uncertainty_index"].get("vibration", {}).get("anomaly_score", 0) > 0.1


@check("Sensor", "pressure 可靠基线", "ranking")
def _(r):
    return r["uncertainty_index"].get("pressure", {}).get("composite_score", 1) < 0.5


# ── Ecommerce (15K) ──


@check("Ecommerce", "total_spent 高不确定性 (泄露特征)", "ranking")
def _(r):
    return r["uncertainty_index"].get("total_spent", {}).get("composite_score", 0) > 0.2


@check("Ecommerce", "rating 标签噪声方差", "variance")
def _(r):
    return r["uncertainty_index"].get("rating", {}).get("variance_score", 0) > 0.05


@check("Ecommerce", "VIP/regular 分布偏移", "shift")
def _(r):
    return bool(r.get("conformal_shift_analysis", {}).get("group_shift", {}))


@check("Ecommerce", "order_count 幂律尾部异常", "anomaly")
def _(r):
    return r["uncertainty_index"].get("order_count", {}).get("anomaly_score", 0) > 0.1


@check("Ecommerce", "return_flag < total_spent", "ranking")
def _(r):
    return r["uncertainty_index"].get("return_flag", {}).get("composite_score", 0) < r[
        "uncertainty_index"
    ].get("total_spent", {}).get("composite_score", 0)


# ── Financial (10K) ──


@check("Financial", "sentiment 数据损坏异常", "anomaly")
def _(r):
    return r["uncertainty_index"].get("sentiment", {}).get("anomaly_score", 0) > 0.3


@check("Financial", "sentiment 排名前 3", "ranking")
def _(r):
    ranked = sorted(
        r["uncertainty_index"].items(), key=lambda x: x[1]["composite_score"], reverse=True
    )
    top3 = [name for name, _ in ranked[:3]]
    return "sentiment" in top3


@check("Financial", "bull/bear regime 偏移", "shift")
def _(r):
    return bool(r.get("conformal_shift_analysis", {}).get("group_shift", {}))


@check("Financial", "market_return 厚尾异常", "anomaly")
def _(r):
    return r["uncertainty_index"].get("market_return", {}).get("anomaly_score", 0) > 0.05


@check("Financial", "volume 对数正态方差", "variance")
def _(r):
    return r["uncertainty_index"].get("volume", {}).get("variance_score", 0) > 0.3


# ─── 数据集配置 ──────────────────────────────────────────────────────

DATASETS = {
    "Housing": (generate_housing, "region", "房产 — 异方差/离群值/空间漂移"),
    "Wine": (generate_wine, "wine_type", "化学 — 离群值/无缺失/组间偏移"),
    "Census": (generate_census, "sex", "人口 — 零膨胀/缺失/弱位置偏移"),
    "Medical": (generate_medical, "center", "临床 — 缺失(MAR)/测量噪声/系统偏差"),
    "Sensor": (generate_sensor, "phase", "IoT — 概念漂移/传感器退化/连续缺失"),
    "Ecommerce": (generate_ecommerce, "user_type", "电商 — 泄露特征/标签噪声/幂律分布"),
    "Financial": (generate_financial, "regime", "金融 — 厚尾/regime change/数据损坏"),
}


# ─── 主程序 ──────────────────────────────────────────────────────────


def main():
    print("=" * 65)
    print("  UncertaintyLens 统一批量基准测试")
    print(f"  {len(DATASETS)} 个数据集 · {len(CHECKS)} 项检查 · 5 类检测能力")
    print("=" * 65)

    # 运行所有数据集
    reports = {}
    for i, (name, (gen_fn, group_col, desc)) in enumerate(DATASETS.items(), 1):
        print(f"\n[{i}/{len(DATASETS)}] {name} — {desc}")
        df = gen_fn()
        pipeline = build_pipeline()
        t0 = time.time()
        report = pipeline.analyze(df, group_col=group_col)
        elapsed = time.time() - t0
        reports[name] = report

        n_feats = report["summary"]["total_features_analyzed"]
        level = report["summary"]["overall_level"]
        print(f"  {len(df):>6,} 行 · {n_feats} 特征 · {level} · {elapsed:.1f}s")

    # 运行所有检查
    print(f"\n{'=' * 65}")
    print("  检查明细")
    print(f"{'=' * 65}")

    results = []  # (dataset, name, category, passed)
    for dataset, name, category, fn in CHECKS:
        report = reports[dataset]
        passed = fn(report)
        results.append((dataset, name, category, passed))

    # 按数据集分组输出
    current_ds = None
    for dataset, name, category, passed in results:
        if dataset != current_ds:
            current_ds = dataset
            print(f"\n  [{dataset}]")
        status = "✓" if passed else "✗"
        print(f"    {status} [{category:8s}] {name}")

    # ── 按检测能力分类统计 ──
    print(f"\n{'=' * 65}")
    print("  按检测能力分类准确率")
    print(f"{'=' * 65}")

    category_stats = defaultdict(lambda: [0, 0])  # [passed, total]
    for _, _, cat, passed in results:
        category_stats[cat][1] += 1
        if passed:
            category_stats[cat][0] += 1

    category_names = {
        "missing": "缺失检测 (Missing)",
        "anomaly": "异常值检测 (Anomaly)",
        "variance": "方差/噪声检测 (Variance)",
        "shift": "分布偏移检测 (Shift)",
        "ranking": "排序合理性 (Ranking)",
    }

    for cat in ["missing", "anomaly", "variance", "shift", "ranking"]:
        p, t = category_stats[cat]
        pct = p / t * 100 if t > 0 else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {category_names[cat]:25s}  {p}/{t}  {bar} {pct:.0f}%")

    # ── 按数据集分类统计 ──
    print(f"\n{'=' * 65}")
    print("  按数据集分类准确率")
    print(f"{'=' * 65}")

    ds_stats = defaultdict(lambda: [0, 0])
    for ds, _, _, passed in results:
        ds_stats[ds][1] += 1
        if passed:
            ds_stats[ds][0] += 1

    for ds in DATASETS:
        p, t = ds_stats[ds]
        status = "PASS" if p == t else "PARTIAL"
        print(f"  {status:8s}  {ds:12s}  {p}/{t}")

    # ── 总计 ──
    total_passed = sum(1 for _, _, _, p in results if p)
    total_checks = len(results)
    print(f"\n{'─' * 65}")
    print(f"  总计: {total_passed}/{total_checks} 项通过 ({total_passed/total_checks:.1%})")

    # ── 已知局限性 ──
    failures = [(ds, name, cat) for ds, name, cat, p in results if not p]
    if failures:
        print(f"\n{'=' * 65}")
        print("  未通过的检查:")
        print(f"{'=' * 65}")
        for ds, name, cat in failures:
            print(f"  ✗ [{ds}] [{cat}] {name}")
    else:
        print(f"\n  全部通过 ✓")

    print(f"\n{'=' * 65}")
    print("  已知检测局限性")
    print(f"{'=' * 65}")
    limitations = [
        "零膨胀特征: 92%为零时，异常/共形检测器认为'预测0'很准确，复合分数偏低",
        "纯位置偏移: 均值偏移但方差相同且重叠度>80%时，KS检验不一定能检测",
        "共线性: 属于建模问题而非数据不确定性，当前检测器不直接处理",
        "时间序列自相关: 当前检测器假设 i.i.d.，不处理时间依赖结构",
        "小样本 (<100行): bootstrap 和共形方法的统计功效下降",
    ]
    for i, lim in enumerate(limitations, 1):
        print(f"  {i}. {lim}")

    return 0 if total_passed == total_checks else 1


if __name__ == "__main__":
    sys.exit(main())
