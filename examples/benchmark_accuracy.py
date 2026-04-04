"""
UncertaintyLens 准确性基准 — 第二轮

在第一轮 benchmark_real_data.py 基础上，补充 4 个新场景：

  4. Medical   — 多重共线性、测量噪声、系统偏差
  5. Sensor    — 时间漂移（concept drift）、传感器退化
  6. Ecommerce — 特征泄露（leakage）、标签噪声
  7. Financial — 厚尾分布、regime change、高度相关特征

每个数据集的不确定性都是人工注入的，拥有已知的 ground truth，
可以量化检测器的准确性。
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from uncertainty_lens import UncertaintyPipeline
from uncertainty_lens.detectors import (
    ConformalShiftDetector,
    UncertaintyDecomposer,
    ConformalPredictor,
    JackknifePlusDetector,
)

warnings.filterwarnings("ignore", category=UserWarning)

OUTPUT_DIR = Path(__file__).parent / "benchmark_results"
OUTPUT_DIR.mkdir(exist_ok=True)


def build_pipeline():
    """包含全部主要检测器的标准 pipeline."""
    pipeline = UncertaintyPipeline(weights={"missing": 0.35, "anomaly": 0.25, "variance": 0.25})
    pipeline.register("conformal_shift", ConformalShiftDetector(seed=42), weight=0.1)
    pipeline.register("decomposition", UncertaintyDecomposer(n_bootstrap=200, seed=42), weight=0.15)
    pipeline.register("jackknife_plus", JackknifePlusDetector(n_folds=10, seed=42), weight=0.1)
    return pipeline


def print_report(name, report, df, expected):
    """打印分析摘要并验证预期检测结果."""
    summary = report["summary"]
    print(f"\n{'─' * 60}")
    print(f"  {name}")
    print(f"{'─' * 60}")
    print(f"  行数: {len(df):,}  |  特征数: {summary['total_features_analyzed']}")
    print(f"  整体不确定性: {summary['overall_uncertainty']:.1%} ({summary['overall_level']})")
    print(f"  高风险: {summary['high_uncertainty_features']}")
    print(f"  可靠:  {summary['low_uncertainty_features']}")

    print(f"\n  {'特征':25s} {'分数':>6s}  {'等级':12s}")
    print(f"  {'─'*25} {'─'*6}  {'─'*12}")
    for col, vals in report["uncertainty_index"].items():
        print(f"  {col:25s} {vals['composite_score']:6.3f}  {vals['level']}")

    # 验证预期发现
    print(f"\n  预期检测结果:")
    results = {}
    for check_name, check_fn in expected.items():
        result = check_fn(report)
        results[check_name] = result
        status = "✓" if result else "✗"
        print(f"    {status} {check_name}")

    all_pass = all(results.values())
    return all_pass, results


# ═══════════════════════════════════════════════════════════════════════
# 数据集 4: Medical — 模拟临床试验数据
# - 多重共线性 (BMI ≈ weight / height²)
# - 测量噪声 (血压测量不精确)
# - 系统偏差 (不同设备/中心的校准差异)
# - 缺失模式: MAR (收入高的人更不容易缺失)
# ═══════════════════════════════════════════════════════════════════════


def generate_medical(n=10000, seed=42):
    rng = np.random.default_rng(seed)

    # 基础生理特征
    age = rng.normal(55, 15, n).clip(18, 90).astype(int)
    sex = rng.choice(["M", "F"], n, p=[0.48, 0.52])
    is_male = sex == "M"

    # height / weight 有内在相关性
    height = np.where(is_male, rng.normal(175, 7, n), rng.normal(162, 6, n))
    weight = np.where(is_male, rng.normal(80, 15, n), rng.normal(65, 12, n))
    # 加入年龄效应
    weight += (age - 40) * 0.2 + rng.normal(0, 3, n)
    weight = weight.clip(40, 180)

    # BMI = weight / (height/100)² → 与 weight, height 高度共线
    bmi = weight / (height / 100) ** 2

    # 血压: 高噪声测量 (设备测量误差 ± 10)
    systolic_true = 120 + 0.3 * (age - 50) + rng.normal(0, 12, n)
    measurement_noise = rng.normal(0, 10, n)  # 设备噪声很大
    systolic_measured = systolic_true + measurement_noise
    systolic_measured = systolic_measured.clip(80, 220)

    # 舒张压: 与收缩压相关但噪声更小
    diastolic = 0.6 * systolic_true + rng.normal(10, 5, n)
    diastolic = diastolic.clip(50, 130)

    # 实验室指标: 胆固醇 (不同中心有系统偏差)
    center = rng.choice(["A", "B", "C"], n, p=[0.4, 0.35, 0.25])
    center_bias = {"A": 0, "B": 15, "C": -10}  # 系统性校准偏差
    cholesterol_true = 200 + 0.5 * (age - 50) + rng.normal(0, 35, n)
    cholesterol = cholesterol_true + np.array([center_bias[c] for c in center])

    # 血糖: 正常分布 + 少量极端值
    glucose = rng.normal(100, 20, n)
    diabetic = rng.random(n) < 0.1
    glucose[diabetic] = rng.normal(180, 40, diabetic.sum())
    glucose = glucose.clip(50, 400)

    # MAR 缺失: 收入高的人更少缺失（非随机缺失）
    income_proxy = rng.lognormal(3.5, 0.8, n)
    missing_prob = 0.15 / (1 + income_proxy / 50)  # 收入越高，缺失越少
    cholesterol_with_missing = cholesterol.copy()
    cholesterol_with_missing[rng.random(n) < missing_prob] = np.nan

    # 完全无信息的噪声特征（应该被标为低可靠性）
    noise_feature = rng.standard_t(3, n) * 10  # t 分布，厚尾

    return pd.DataFrame(
        {
            "age": age,
            "height": np.round(height, 1),
            "weight": np.round(weight, 1),
            "bmi": np.round(bmi, 1),
            "systolic_bp": np.round(systolic_measured, 0).astype(int),
            "diastolic_bp": np.round(diastolic, 0).astype(int),
            "cholesterol": np.round(cholesterol_with_missing, 0),
            "glucose": np.round(glucose, 0).astype(int),
            "noise_feature": np.round(noise_feature, 2),
            "center": center,
        }
    )


def run_medical():
    df = generate_medical()
    pipeline = build_pipeline()
    t0 = time.time()
    report = pipeline.analyze(df, group_col="center")
    elapsed = time.time() - t0

    expected = {
        # 胆固醇有缺失 → missing 检测器应该捕获
        "cholesterol 缺失被检测": lambda r: (
            r["uncertainty_index"].get("cholesterol", {}).get("missing_score", 0) > 0.01
        ),
        # BMI = weight/height² 是确定性计算，无数据不确定性。
        # 共线性是建模问题，不影响数据层面的不确定性评分。
        # BMI/height/weight 应该都排在 cholesterol（有缺失+偏差）之下。
        "BMI 低于 cholesterol（无缺失/偏差的特征更可靠）": lambda r: (
            r["uncertainty_index"].get("bmi", {}).get("composite_score", 0)
            < r["uncertainty_index"].get("cholesterol", {}).get("composite_score", 0)
        ),
        # systolic_bp 有大测量噪声 → 方差较大
        "systolic_bp 方差信号被检测": lambda r: (
            r["uncertainty_index"].get("systolic_bp", {}).get("variance_score", 0) > 0.1
        ),
        # 不同中心的 cholesterol 有系统偏差 → 分组漂移检测
        "检测到中心间分布偏移": lambda r: (
            bool(r.get("conformal_shift_analysis", {}).get("group_shift", {}))
        ),
        # glucose 有离群值（10% 糖尿病人群）
        "glucose 异常值被检测": lambda r: (
            r["uncertainty_index"].get("glucose", {}).get("anomaly_score", 0) > 0.05
        ),
    }

    all_pass, results = print_report("Medical (10,000 行)", report, df, expected)
    print(f"  耗时: {elapsed:.1f}s")
    return all_pass, results


# ═══════════════════════════════════════════════════════════════════════
# 数据集 5: Sensor — 模拟 IoT 传感器数据
# - 时间漂移 (concept drift): 前半段和后半段分布不同
# - 传感器退化: 噪声随时间增大
# - 周期性模式 + 突发异常
# - 缺失: 传感器掉线导致连续缺失
# ═══════════════════════════════════════════════════════════════════════


def generate_sensor(n=12000, seed=42):
    rng = np.random.default_rng(seed)

    t = np.arange(n)
    # 时间标签: 前半段 "phase1", 后半段 "phase2"
    phase = np.where(t < n // 2, "phase1", "phase2")

    # 温度: 正弦周期 + 线性漂移 (概念漂移)
    temp_base = 25 + 5 * np.sin(2 * np.pi * t / 1000)
    drift = np.where(t < n // 2, 0, 0.002 * (t - n // 2))  # phase2 升温
    # 传感器退化: 噪声随时间增大
    noise_scale = 0.5 + 0.001 * t  # 从 0.5 增到 ~12.5
    temp_noise = rng.normal(0, noise_scale)
    temperature = temp_base + drift + temp_noise

    # 湿度: 与温度反相关 + 独立噪声
    humidity = 80 - 0.8 * (temperature - 25) + rng.normal(0, 3, n)
    humidity = humidity.clip(10, 100)

    # 压力: 稳定，低噪声，可靠特征
    pressure = 1013 + rng.normal(0, 2, n)

    # 振动: 正常情况低，但有 2% 突发异常（设备故障）
    vibration = rng.exponential(0.5, n)
    fault_mask = rng.random(n) < 0.02
    vibration[fault_mask] = rng.uniform(10, 50, fault_mask.sum())

    # 电压: 传感器掉线导致整段 NaN (burst missing)
    voltage = 3.3 + rng.normal(0, 0.05, n)
    # 制造 3 段连续掉线（每段 ~200 个点）
    for start in [2000, 5500, 9000]:
        length = rng.integers(150, 250)
        voltage[start : start + length] = np.nan

    # 信号强度: 与温度无关，随机游走
    signal = np.cumsum(rng.normal(0, 0.1, n)) + 50
    signal = signal.clip(0, 100)

    return pd.DataFrame(
        {
            "temperature": np.round(temperature, 2),
            "humidity": np.round(humidity, 1),
            "pressure": np.round(pressure, 2),
            "vibration": np.round(vibration, 3),
            "voltage": np.round(voltage, 4),
            "signal_strength": np.round(signal, 2),
            "phase": phase,
        }
    )


def run_sensor():
    df = generate_sensor()
    pipeline = build_pipeline()
    t0 = time.time()
    report = pipeline.analyze(df, group_col="phase")
    elapsed = time.time() - t0

    expected = {
        # 温度有漂移 + 噪声增大 → 方差大
        "temperature 方差较高（传感器退化）": lambda r: (
            r["uncertainty_index"].get("temperature", {}).get("variance_score", 0) > 0.3
        ),
        # phase1 vs phase2 温度分布不同 → 偏移分数应高
        "检测到 phase 间温度偏移": lambda r: (
            r.get("conformal_shift_analysis", {})
            .get("uncertainty_scores", {})
            .get("temperature", 0)
            > 0.5
        ),
        # voltage 有连续缺失
        "voltage 缺失被检测": lambda r: (
            r["uncertainty_index"].get("voltage", {}).get("missing_score", 0) > 0.01
        ),
        # vibration 有突发异常
        "vibration 异常被检测": lambda r: (
            r["uncertainty_index"].get("vibration", {}).get("anomaly_score", 0) > 0.1
        ),
        # pressure 是可靠特征
        "pressure 相对可靠": lambda r: (
            r["uncertainty_index"].get("pressure", {}).get("composite_score", 1) < 0.5
        ),
    }

    all_pass, results = print_report("Sensor (12,000 行)", report, df, expected)
    print(f"  耗时: {elapsed:.1f}s")
    return all_pass, results


# ═══════════════════════════════════════════════════════════════════════
# 数据集 6: Ecommerce — 模拟电商行为数据
# - 特征泄露 (leakage): total_spent ≈ price × quantity
# - 标签噪声: 10% 的评分是随机的
# - 季节性分组差异 (weekday vs weekend)
# - 类别不平衡: 90% 普通用户，10% VIP
# ═══════════════════════════════════════════════════════════════════════


def generate_ecommerce(n=15000, seed=42):
    rng = np.random.default_rng(seed)

    # 用户类型
    user_type = np.where(rng.random(n) < 0.1, "VIP", "regular")
    is_vip = user_type == "VIP"

    # 价格和数量
    price = np.where(
        is_vip,
        rng.lognormal(4.5, 0.8, n),  # VIP 买更贵的
        rng.lognormal(3.5, 0.6, n),
    )
    price = price.clip(5, 5000)

    quantity = np.where(
        is_vip,
        rng.poisson(3, n) + 1,
        rng.poisson(1.5, n) + 1,
    )
    quantity = quantity.clip(1, 20)

    # 特征泄露: total_spent = price * quantity + 小噪声
    # 这个特征与 price/quantity 高度共线，应该被标记
    total_spent = price * quantity + rng.normal(0, 2, n)
    total_spent = total_spent.clip(0, None)

    # 浏览时长: 与购买行为弱相关
    browse_minutes = rng.exponential(8, n)
    browse_minutes = browse_minutes.clip(0.1, 120)

    # 评分: 基于真实满意度 + 10% 标签噪声
    true_satisfaction = 3.0 + 0.3 * np.log1p(browse_minutes) + rng.normal(0, 0.5, n)
    rating = true_satisfaction.clip(1, 5)
    # 注入 10% 随机评分（标签噪声）
    noisy_mask = rng.random(n) < 0.10
    rating[noisy_mask] = rng.uniform(1, 5, noisy_mask.sum())
    rating = np.round(rating, 1)

    # 历史订单数: 幂律分布（大量新客 + 少量老客）
    order_count = rng.pareto(1.5, n) * 2 + 1
    order_count = np.clip(order_count, 1, 500).astype(int)

    # 折扣率: VIP 享受更多折扣
    discount = np.where(
        is_vip,
        rng.beta(2, 5, n) * 0.5,  # 0-50% 折扣
        rng.beta(1, 10, n) * 0.3,  # 0-30% 折扣
    )

    # 退货率: 与满意度相关
    return_flag = (rng.random(n) < (0.3 - 0.05 * true_satisfaction.clip(1, 5))).astype(int)

    return pd.DataFrame(
        {
            "price": np.round(price, 2),
            "quantity": quantity,
            "total_spent": np.round(total_spent, 2),
            "browse_minutes": np.round(browse_minutes, 1),
            "rating": rating,
            "order_count": order_count,
            "discount": np.round(discount, 3),
            "return_flag": return_flag,
            "user_type": user_type,
        }
    )


def run_ecommerce():
    df = generate_ecommerce()
    pipeline = build_pipeline()
    t0 = time.time()
    report = pipeline.analyze(df, group_col="user_type")
    elapsed = time.time() - t0

    expected = {
        # total_spent ≈ price × quantity → 方差极大（因为是乘积）
        "total_spent 不确定性高（泄露特征）": lambda r: (
            r["uncertainty_index"].get("total_spent", {}).get("composite_score", 0) > 0.2
        ),
        # rating 有 10% 噪声 → 方差偏高
        "rating 有方差信号（标签噪声）": lambda r: (
            r["uncertainty_index"].get("rating", {}).get("variance_score", 0) > 0.05
        ),
        # VIP vs regular 有分布差异
        "检测到 VIP/regular 分布偏移": lambda r: (
            bool(r.get("conformal_shift_analysis", {}).get("group_shift", {}))
        ),
        # order_count 是幂律分布 → 异常值多
        "order_count 异常值被检测（幂律尾部）": lambda r: (
            r["uncertainty_index"].get("order_count", {}).get("anomaly_score", 0) > 0.1
        ),
        # return_flag 是二值变量 → 方差有限，应该较低
        "return_flag 不确定性不是最高": lambda r: (
            r["uncertainty_index"].get("return_flag", {}).get("composite_score", 0)
            < r["uncertainty_index"].get("total_spent", {}).get("composite_score", 0)
        ),
    }

    all_pass, results = print_report("Ecommerce (15,000 行)", report, df, expected)
    print(f"  耗时: {elapsed:.1f}s")
    return all_pass, results


# ═══════════════════════════════════════════════════════════════════════
# 数据集 7: Financial — 模拟金融市场数据
# - 厚尾分布 (收益率 t 分布)
# - Regime change: 牛市 vs 熊市
# - 高度相关特征 (多个市场指数)
# - 波动率聚类 (GARCH 效应)
# ═══════════════════════════════════════════════════════════════════════


def generate_financial(n=10000, seed=42):
    rng = np.random.default_rng(seed)

    # Regime: 60% 牛市, 40% 熊市
    regime = np.where(np.arange(n) < int(n * 0.6), "bull", "bear")
    is_bull = regime == "bull"

    # 市场收益率: 牛市正均值 + 低波动，熊市负均值 + 高波动
    market_return = np.where(
        is_bull,
        rng.standard_t(5, n) * 0.01 + 0.0005,  # 牛: μ=0.05%/天, σ适中
        rng.standard_t(3, n) * 0.025 - 0.001,  # 熊: μ=-0.1%/天, 厚尾 + 高波动
    )

    # 相关指数 (与 market_return 高度相关)
    index_a = market_return * 0.95 + rng.normal(0, 0.002, n)
    index_b = market_return * 0.88 + rng.normal(0, 0.003, n)

    # 波动率: GARCH 效应 (波动率聚类)
    volatility = np.zeros(n)
    volatility[0] = 0.01
    for i in range(1, n):
        volatility[i] = 0.00001 + 0.85 * volatility[i - 1] + 0.1 * market_return[i - 1] ** 2
    realized_vol = np.sqrt(volatility) * 100  # 转换为百分比

    # 交易量: 与波动率正相关 + 对数正态
    volume = np.exp(10 + 0.5 * realized_vol + rng.normal(0, 0.3, n))

    # 利差 (spread): 熊市扩大
    spread = np.where(
        is_bull,
        rng.exponential(0.5, n),
        rng.exponential(1.5, n),
    )
    spread = spread.clip(0, 20)

    # 情绪指标: 有 5% 数据损坏（极端异常值）
    sentiment = rng.normal(50, 10, n)
    corrupt_mask = rng.random(n) < 0.05
    sentiment[corrupt_mask] = rng.choice([-999, 999], corrupt_mask.sum())

    # VIX 代理: 平稳特征，低不确定性
    vix_proxy = realized_vol * 1.5 + rng.normal(0, 1, n)
    vix_proxy = vix_proxy.clip(5, 80)

    return pd.DataFrame(
        {
            "market_return": np.round(market_return, 6),
            "index_a": np.round(index_a, 6),
            "index_b": np.round(index_b, 6),
            "realized_vol": np.round(realized_vol, 4),
            "volume": np.round(volume, 0).astype(int),
            "spread": np.round(spread, 3),
            "sentiment": np.round(sentiment, 1),
            "vix_proxy": np.round(vix_proxy, 2),
            "regime": regime,
        }
    )


def run_financial():
    df = generate_financial()
    pipeline = build_pipeline()
    t0 = time.time()
    report = pipeline.analyze(df, group_col="regime")
    elapsed = time.time() - t0

    expected = {
        # sentiment 有 5% 极端异常值 (-999, 999) → 异常 + 方差爆表
        "sentiment 异常值被检测（数据损坏）": lambda r: (
            r["uncertainty_index"].get("sentiment", {}).get("anomaly_score", 0) > 0.3
        ),
        # sentiment 应该是最高不确定性之一
        "sentiment 不确定性排名前 3": lambda r: (
            sorted(
                r["uncertainty_index"].items(),
                key=lambda x: x[1]["composite_score"],
                reverse=True,
            )[0][0]
            == "sentiment"
            or sorted(
                r["uncertainty_index"].items(),
                key=lambda x: x[1]["composite_score"],
                reverse=True,
            )[1][0]
            == "sentiment"
            or sorted(
                r["uncertainty_index"].items(),
                key=lambda x: x[1]["composite_score"],
                reverse=True,
            )[2][0]
            == "sentiment"
        ),
        # 牛市 vs 熊市 → 明确的分布偏移
        "检测到 regime 分布偏移": lambda r: (
            bool(r.get("conformal_shift_analysis", {}).get("group_shift", {}))
        ),
        # market_return 厚尾 → 异常值检测
        "market_return 异常值被检测（厚尾）": lambda r: (
            r["uncertainty_index"].get("market_return", {}).get("anomaly_score", 0) > 0.05
        ),
        # volume 对数正态 → 方差大
        "volume 方差被检测": lambda r: (
            r["uncertainty_index"].get("volume", {}).get("variance_score", 0) > 0.3
        ),
    }

    all_pass, results = print_report("Financial (10,000 行)", report, df, expected)
    print(f"  耗时: {elapsed:.1f}s")
    return all_pass, results


# ═══════════════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════════════


def main():
    print("=" * 60)
    print("  UncertaintyLens 准确性基准 — 第二轮")
    print("  (4 个新场景: Medical / Sensor / Ecommerce / Financial)")
    print("=" * 60)

    all_results = {}
    all_checks = {}

    datasets = [
        ("Medical", run_medical),
        ("Sensor", run_sensor),
        ("Ecommerce", run_ecommerce),
        ("Financial", run_financial),
    ]

    for name, runner in datasets:
        print(f"\n[{len(all_results)+1}/{len(datasets)}] {name} ...")
        passed, checks = runner()
        all_results[name] = passed
        all_checks[name] = checks

    # 总结
    print(f"\n{'=' * 60}")
    print("  基准总结")
    print(f"{'=' * 60}")

    total_checks = 0
    passed_checks = 0
    for name, passed in all_results.items():
        status = "PASS" if passed else "PARTIAL"
        n_pass = sum(all_checks[name].values())
        n_total = len(all_checks[name])
        total_checks += n_total
        passed_checks += n_pass
        print(f"  {status:8s}  {name:15s}  ({n_pass}/{n_total} 检查通过)")

    print(f"\n  总计: {passed_checks}/{total_checks} 项检查通过")
    print(f"  准确率: {passed_checks/total_checks:.1%}")

    n_pass_ds = sum(all_results.values())
    print(f"  {n_pass_ds}/{len(all_results)} 数据集完全通过")

    return 0 if all(all_results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
