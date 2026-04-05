"""
UncertaintyLens 盲测验证 (Blind Test)

核心原则:
  1. 所有预期在看到结果之前已定义 (写在 BLIND_CHECKS 中)
  2. 数据生成器与之前的 7 个完全不同
  3. 运行一次，不调阈值
  4. 原始通过率就是真实准确率

3 个新场景:
  - Insurance  (保险): 零膨胀理赔 + 年龄歧视偏移 + MNAR 缺失
  - Climate    (气候): 长尾极端事件 + 趋势漂移 + 传感器故障
  - HR         (人力): 薪资偏移(性别) + 绩效评分噪声 + 编码型缺失

用法:
  PYTHONPATH=. python examples/benchmark_blind.py
"""

import sys
import time
import warnings
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


# ─── Pipeline (与 benchmark_all 一致) ─────────────────────────────────


def build_pipeline():
    pipeline = UncertaintyPipeline(weights={"missing": 0.35, "anomaly": 0.25, "variance": 0.25})
    pipeline.register("conformal_shift", ConformalShiftDetector(seed=42), weight=0.1)
    pipeline.register("decomposition", UncertaintyDecomposer(n_bootstrap=200, seed=42), weight=0.15)
    pipeline.register("jackknife_plus", JackknifePlusDetector(n_folds=10, seed=42), weight=0.1)
    pipeline.register("mmd_shift", MMDShiftDetector(n_permutations=200, seed=42), weight=0.1)
    pipeline.register("zero_inflation", ZeroInflationDetector(zero_threshold=0.5), weight=0.2)
    return pipeline


# ═══════════════════════════════════════════════════════════════════════
# 数据生成器
# ═══════════════════════════════════════════════════════════════════════


def generate_insurance(n=8000, seed=123):
    """
    保险理赔数据.

    特征设计:
      claim_amount   — 零膨胀 (~85% 为零, 非零部分 lognormal 长尾)
      premium        — 正常连续, 与 age 正相关
      age            — 均匀 18-80, 组列: age_group (young/middle/senior)
      bmi            — 正态 ~27, 小方差
      income         — MNAR 缺失: 高收入者 30% 拒绝填写
      prev_claims    — 计数型, 零膨胀 (~70% 为零)
      satisfaction   — 1-5 离散打分, 有 10% 随机噪声
      risk_score     — 人工计算 = premium*0.3 + age*0.1, 无随机性
    """
    rng = np.random.default_rng(seed)

    age = rng.uniform(18, 80, n)
    age_group = np.where(age < 35, "young", np.where(age < 55, "middle", "senior"))

    bmi = rng.normal(27, 4, n).clip(15, 50)
    premium = 200 + age * 15 + bmi * 10 + rng.normal(0, 50, n)

    # 零膨胀理赔 (85% 为零)
    claim_amount = np.zeros(n)
    has_claim = rng.random(n) < 0.15
    claim_amount[has_claim] = rng.lognormal(7, 1.5, has_claim.sum())

    # MNAR 缺失: 高收入者更可能拒填
    income = rng.lognormal(10.5, 0.6, n)
    missing_prob = np.where(income > np.percentile(income, 70), 0.30, 0.05)
    income[rng.random(n) < missing_prob] = np.nan

    # 零膨胀历史理赔数 (70%)
    prev_claims = np.zeros(n, dtype=float)
    has_prev = rng.random(n) < 0.30
    prev_claims[has_prev] = rng.poisson(2, has_prev.sum()).astype(float)

    # 满意度 (1-5), 10% 噪声
    satisfaction = rng.integers(1, 6, n).astype(float)
    noise_mask = rng.random(n) < 0.10
    satisfaction[noise_mask] = rng.integers(1, 6, noise_mask.sum())

    # 确定性风险分数 (无随机性)
    risk_score = premium * 0.3 + age * 0.1

    return pd.DataFrame(
        {
            "claim_amount": claim_amount,
            "premium": premium,
            "age": age,
            "bmi": bmi,
            "income": income,
            "prev_claims": prev_claims,
            "satisfaction": satisfaction,
            "risk_score": risk_score,
            "age_group": age_group,
        }
    )


def generate_climate(n=10000, seed=456):
    """
    气候监测站数据.

    特征设计:
      temperature    — 正弦季节性 + 线性趋势漂移(+2°C in second half)
                       station A/B: B 偏高 3°C (系统偏差)
      precipitation  — 极端长尾 (gamma), 偶发暴雨 (>200mm)
      wind_speed     — Weibull 分布, 3% 极端值 (台风)
      humidity       — Beta 分布, 较稳定
      pressure       — 正态, 低方差, 可靠基线
      co2_ppm        — 缓慢上升趋势 + 高斯噪声
      solar_index    — 传感器故障: 5% 连续块缺失 + 2% 异常尖峰
      station        — 分组: A/B
    """
    rng = np.random.default_rng(seed)

    station = rng.choice(["A", "B"], n, p=[0.6, 0.4])
    time_idx = np.linspace(0, 4 * np.pi, n)

    # 温度: 季节性 + 趋势 + 站点偏差
    temp_base = 15 + 10 * np.sin(time_idx)
    trend = np.linspace(0, 2, n)  # 后半段 +2°C
    station_bias = np.where(station == "B", 3.0, 0.0)
    temperature = temp_base + trend + station_bias + rng.normal(0, 2, n)

    # 降水: gamma 长尾
    precipitation = rng.gamma(0.8, 30, n)
    # 注入暴雨事件
    storm_mask = rng.random(n) < 0.02
    precipitation[storm_mask] = rng.uniform(200, 500, storm_mask.sum())

    # 风速: Weibull
    wind_speed = rng.weibull(2, n) * 8
    typhoon_mask = rng.random(n) < 0.03
    wind_speed[typhoon_mask] = rng.uniform(30, 60, typhoon_mask.sum())

    # 湿度: Beta, 较稳定
    humidity = rng.beta(5, 3, n) * 100

    # 气压: 正态, 低方差
    pressure = rng.normal(1013, 5, n)

    # CO2: 上升趋势
    co2_ppm = 410 + np.linspace(0, 15, n) + rng.normal(0, 3, n)

    # 太阳指数: 传感器故障
    solar_index = rng.normal(100, 10, n)
    # 连续块缺失 (3段, 每段~170个)
    for start in [1000, 4000, 7500]:
        end = min(start + 170, n)
        solar_index[start:end] = np.nan
    # 2% 异常尖峰
    spike_mask = rng.random(n) < 0.02
    solar_index[spike_mask] = rng.uniform(300, 500, spike_mask.sum())

    return pd.DataFrame(
        {
            "temperature": temperature,
            "precipitation": precipitation,
            "wind_speed": wind_speed,
            "humidity": humidity,
            "pressure": pressure,
            "co2_ppm": co2_ppm,
            "solar_index": solar_index,
            "station": station,
        }
    )


def generate_hr(n=6000, seed=789):
    """
    人力资源数据.

    特征设计:
      salary         — 性别薪资差距: female 均值低 8%, 检测偏移
      performance    — 1-5 评分 + 20% 主观噪声 (标签不一致)
      tenure_years   — 指数分布, 右偏
      training_hours — 零膨胀 (~60% 为零, 很多人不培训)
      dept_code      — 编码特征, 5% 编码错误导致 NaN
      overtime_hrs   — 双峰: 大部分 <5, 少数 >30 (burn-out 群体)
      age            — 均匀 22-65
      engagement     — 0-100 分, 与 performance 弱相关
      gender         — 分组: M/F
    """
    rng = np.random.default_rng(seed)

    gender = rng.choice(["M", "F"], n, p=[0.55, 0.45])
    age = rng.uniform(22, 65, n)

    # 薪资: 性别偏移
    base_salary = 50000 + age * 500 + rng.normal(0, 8000, n)
    gender_penalty = np.where(gender == "F", -0.08, 0.0)
    salary = base_salary * (1 + gender_penalty)

    # 绩效: 20% 噪声
    true_perf = rng.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.15, 0.40, 0.25, 0.15])
    noise_mask = rng.random(n) < 0.20
    noisy_perf = true_perf.copy().astype(float)
    noisy_perf[noise_mask] = rng.choice([1, 2, 3, 4, 5], noise_mask.sum())

    # 任期: 指数分布
    tenure_years = rng.exponential(4, n).clip(0, 35)

    # 培训时数: 零膨胀 60%
    training_hours = np.zeros(n)
    trained = rng.random(n) < 0.40
    training_hours[trained] = rng.lognormal(2, 1, trained.sum())

    # 部门编码: 5% 缺失 (编码错误)
    dept_code = rng.choice([10, 20, 30, 40, 50], n).astype(float)
    dept_code[rng.random(n) < 0.05] = np.nan

    # 加班: 双峰
    overtime_hrs = np.zeros(n)
    normal_work = rng.random(n) < 0.85
    overtime_hrs[normal_work] = rng.exponential(3, normal_work.sum()).clip(0, 10)
    burnout = ~normal_work
    overtime_hrs[burnout] = rng.normal(35, 5, burnout.sum()).clip(20, 60)

    # 参与度
    engagement = rng.normal(65, 15, n).clip(0, 100)

    return pd.DataFrame(
        {
            "salary": salary,
            "performance": noisy_perf,
            "tenure_years": tenure_years,
            "training_hours": training_hours,
            "dept_code": dept_code,
            "overtime_hrs": overtime_hrs,
            "age": age,
            "engagement": engagement,
            "gender": gender,
        }
    )


# ═══════════════════════════════════════════════════════════════════════
# 盲测检查 (预先定义, 不可修改)
# ═══════════════════════════════════════════════════════════════════════

BLIND_CHECKS = []


def blind(dataset, name, category):
    """注册盲测检查."""

    def decorator(fn):
        BLIND_CHECKS.append((dataset, name, category, fn))
        return fn

    return decorator


# ── Insurance (8K) ──
# 预期理由在注释中


# claim_amount 85% 零值 → 零膨胀检测器应识别
@blind("Insurance", "claim_amount 零膨胀被识别", "variance")
def _(r):
    zi = r.get("zero_inflation_analysis", {})
    return "claim_amount" in zi.get("zero_inflated_features", [])


# prev_claims 70% 零值 → 也应被识别
@blind("Insurance", "prev_claims 零膨胀被识别", "variance")
def _(r):
    zi = r.get("zero_inflation_analysis", {})
    return "prev_claims" in zi.get("zero_inflated_features", [])


# income MNAR 缺失 → missing_score > 0
@blind("Insurance", "income MNAR 缺失被检测", "missing")
def _(r):
    return r["uncertainty_index"].get("income", {}).get("missing_score", 0) > 0.01


# claim_amount 长尾 → anomaly_score > 0
@blind("Insurance", "claim_amount 长尾异常", "anomaly")
def _(r):
    return r["uncertainty_index"].get("claim_amount", {}).get("anomaly_score", 0) > 0.05


# risk_score = f(premium, age) 无随机性 → 排序上应低于 claim_amount
@blind("Insurance", "risk_score < claim_amount (确定性 vs 随机)", "ranking")
def _(r):
    rs = r["uncertainty_index"].get("risk_score", {}).get("composite_score", 1)
    ca = r["uncertainty_index"].get("claim_amount", {}).get("composite_score", 0)
    return rs < ca


# bmi 方差小, 无缺失 → composite 较低 (< 0.5)
@blind("Insurance", "bmi 低不确定性", "ranking")
def _(r):
    return r["uncertainty_index"].get("bmi", {}).get("composite_score", 1) < 0.5


# 年龄组间应有分布差异 (young/middle/senior 的 premium 不同)
@blind("Insurance", "年龄组间分布偏移 (KS)", "shift")
def _(r):
    return bool(r.get("conformal_shift_analysis", {}).get("group_shift", {}))


# MMD 也应检测到组间偏移
@blind("Insurance", "年龄组间 MMD 偏移", "shift")
def _(r):
    mmd = r.get("mmd_shift_analysis", {})
    return bool(mmd.get("group_shift", {}))


# ── Climate (10K) ──


# precipitation 极端长尾 → 高异常分数
@blind("Climate", "precipitation 长尾极端事件", "anomaly")
def _(r):
    return r["uncertainty_index"].get("precipitation", {}).get("anomaly_score", 0) > 0.1


# wind_speed 台风极端值 → 异常
@blind("Climate", "wind_speed 台风极端值", "anomaly")
def _(r):
    return r["uncertainty_index"].get("wind_speed", {}).get("anomaly_score", 0) > 0.05


# solar_index 连续块缺失 → missing_score > 0
@blind("Climate", "solar_index 传感器故障缺失", "missing")
def _(r):
    return r["uncertainty_index"].get("solar_index", {}).get("missing_score", 0) > 0.01


# solar_index 也有异常尖峰 → anomaly_score > 0
@blind("Climate", "solar_index 异常尖峰", "anomaly")
def _(r):
    return r["uncertainty_index"].get("solar_index", {}).get("anomaly_score", 0) > 0.05


# pressure 低方差, 稳定 → 低 composite
@blind("Climate", "pressure 可靠基线", "ranking")
def _(r):
    return r["uncertainty_index"].get("pressure", {}).get("composite_score", 1) < 0.4


# station A/B 温度偏差 → 偏移检测
@blind("Climate", "A/B 站点温度偏移 (KS)", "shift")
def _(r):
    return bool(r.get("conformal_shift_analysis", {}).get("group_shift", {}))


# MMD 联合偏移 (温度 + 其他)
@blind("Climate", "A/B 站点 MMD 联合偏移", "shift")
def _(r):
    mmd = r.get("mmd_shift_analysis", {})
    joint = mmd.get("joint_mmd", {})
    return any(v.get("shift_detected", False) for v in joint.values())


# humidity Beta分布, 较稳定 → 不确定性低于 precipitation
@blind("Climate", "humidity < precipitation (稳定 vs 极端)", "ranking")
def _(r):
    hu = r["uncertainty_index"].get("humidity", {}).get("composite_score", 1)
    pr = r["uncertainty_index"].get("precipitation", {}).get("composite_score", 0)
    return hu < pr


# temperature 方差 (趋势漂移 + 站点偏差)
@blind("Climate", "temperature 方差信号", "variance")
def _(r):
    return r["uncertainty_index"].get("temperature", {}).get("variance_score", 0) > 0.05


# ── HR (6K) ──


# salary 性别偏移 → shift 检测
@blind("HR", "salary 性别薪资偏移 (KS)", "shift")
def _(r):
    shifts = r.get("conformal_shift_analysis", {}).get("group_shift", {})
    return bool(shifts)


# MMD 也应检测到性别偏移
@blind("HR", "salary 性别 MMD 偏移", "shift")
def _(r):
    mmd = r.get("mmd_shift_analysis", {})
    return bool(mmd.get("group_shift", {}))


# training_hours 60% 零值 → 零膨胀
@blind("HR", "training_hours 零膨胀被识别", "variance")
def _(r):
    zi = r.get("zero_inflation_analysis", {})
    return "training_hours" in zi.get("zero_inflated_features", [])


# dept_code 5% 缺失 → missing_score > 0
@blind("HR", "dept_code 编码缺失被检测", "missing")
def _(r):
    return r["uncertainty_index"].get("dept_code", {}).get("missing_score", 0) > 0.01


# overtime_hrs 双峰 → 异常分数 > 0
@blind("HR", "overtime_hrs 双峰异常", "anomaly")
def _(r):
    return r["uncertainty_index"].get("overtime_hrs", {}).get("anomaly_score", 0) > 0.05


# performance 20% 噪声 → 方差信号
@blind("HR", "performance 标签噪声", "variance")
def _(r):
    return r["uncertainty_index"].get("performance", {}).get("variance_score", 0) > 0.05


# tenure_years 指数分布右偏 → anomaly > 0
@blind("HR", "tenure_years 右偏分布", "anomaly")
def _(r):
    return r["uncertainty_index"].get("tenure_years", {}).get("anomaly_score", 0) > 0.05


# engagement 无特殊问题 → composite 低于 salary
@blind("HR", "engagement < salary (稳定 vs 偏移)", "ranking")
def _(r):
    eng = r["uncertainty_index"].get("engagement", {}).get("composite_score", 1)
    sal = r["uncertainty_index"].get("salary", {}).get("composite_score", 0)
    return eng < sal


# ═══════════════════════════════════════════════════════════════════════
# 数据集配置
# ═══════════════════════════════════════════════════════════════════════

BLIND_DATASETS = {
    "Insurance": (generate_insurance, "age_group", "保险 — 零膨胀理赔/MNAR缺失/年龄组偏移"),
    "Climate": (generate_climate, "station", "气候 — 极端长尾/趋势漂移/传感器故障"),
    "HR": (generate_hr, "gender", "人力 — 薪资偏移/标签噪声/零膨胀培训"),
}


# ═══════════════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════════════


def main():
    print("=" * 65)
    print("  UncertaintyLens 盲测验证 (Blind Test)")
    print(f"  {len(BLIND_DATASETS)} 个新数据集 · {len(BLIND_CHECKS)} 项预定义检查")
    print("  ⚠ 所有预期在实现前已锁定, 不可修改")
    print("=" * 65)

    # 运行所有数据集
    reports = {}
    for i, (name, (gen_fn, group_col, desc)) in enumerate(BLIND_DATASETS.items(), 1):
        print(f"\n[{i}/{len(BLIND_DATASETS)}] {name} — {desc}")
        df = gen_fn()
        pipeline = build_pipeline()
        t0 = time.time()
        report = pipeline.analyze(df, group_col=group_col)
        elapsed = time.time() - t0
        reports[name] = report

        n_feats = report["summary"]["total_features_analyzed"]
        level = report["summary"]["overall_level"]
        print(f"  {len(df):>6,} 行 · {n_feats} 特征 · {level} · {elapsed:.1f}s")

    # 运行所有盲测检查
    print(f"\n{'=' * 65}")
    print("  盲测检查明细")
    print(f"{'=' * 65}")

    results = []
    for dataset, name, category, fn in BLIND_CHECKS:
        report = reports[dataset]
        try:
            passed = fn(report)
        except Exception as e:
            passed = False
            print(f"  [ERROR] {dataset}/{name}: {e}")
        results.append((dataset, name, category, passed))

    current_ds = None
    for dataset, name, category, passed in results:
        if dataset != current_ds:
            current_ds = dataset
            print(f"\n  [{dataset}]")
        status = "✓" if passed else "✗"
        print(f"    {status} [{category:8s}] {name}")

    # ── 按检测能力分类 ──
    print(f"\n{'=' * 65}")
    print("  按检测能力分类准确率")
    print(f"{'=' * 65}")

    category_stats = defaultdict(lambda: [0, 0])
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

    # ── 按数据集 ──
    print(f"\n{'=' * 65}")
    print("  按数据集分类准确率")
    print(f"{'=' * 65}")

    ds_stats = defaultdict(lambda: [0, 0])
    for ds, _, _, passed in results:
        ds_stats[ds][1] += 1
        if passed:
            ds_stats[ds][0] += 1

    for ds in BLIND_DATASETS:
        p, t = ds_stats[ds]
        status = "PASS" if p == t else "PARTIAL"
        print(f"  {status:8s}  {ds:12s}  {p}/{t}")

    # ── 总计 ──
    total_passed = sum(1 for _, _, _, p in results if p)
    total_checks = len(results)
    print(f"\n{'─' * 65}")
    print(f"  盲测总计: {total_passed}/{total_checks} ({total_passed/total_checks:.1%})")

    if total_passed == total_checks:
        print("  🎯 全部通过 — 盲测验证成功")
    else:
        failures = [(ds, name, cat) for ds, name, cat, p in results if not p]
        print(f"\n  未通过:")
        for ds, name, cat in failures:
            print(f"    ✗ [{ds}] [{cat}] {name}")

    return 0 if total_passed == total_checks else 1


if __name__ == "__main__":
    sys.exit(main())
