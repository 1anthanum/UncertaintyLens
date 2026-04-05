"""
UncertaintyLens 综合测试仪表盘

汇总所有测试套件并生成 HTML 仪表盘:
  1. 单元测试 (pytest)
  2. 核心基准 (benchmark_all.py — 7 数据集 39 项)
  3. 盲测验证 (benchmark_blind.py — 3 数据集 25 项)
  4. 扩展基准 (benchmark_extended.py — 6 数据集 27 项)

输出: test_dashboard.html — 可直接在浏览器中查看

用法:
  PYTHONPATH=. python examples/test_dashboard.py
"""

import subprocess
import sys
import time
import re
from pathlib import Path
from datetime import datetime


def run_cmd(cmd, timeout=600):
    """运行命令并返回 (returncode, stdout)."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT"


def parse_pytest(output):
    """解析 pytest 输出."""
    m = re.search(r"(\d+) passed", output)
    passed = int(m.group(1)) if m else 0
    m = re.search(r"(\d+) failed", output)
    failed = int(m.group(1)) if m else 0
    m = re.search(r"(\d+) skipped", output)
    skipped = int(m.group(1)) if m else 0
    m = re.search(r"in ([\d.]+)s", output)
    elapsed = float(m.group(1)) if m else 0
    return passed, failed, skipped, elapsed


def parse_benchmark(output):
    """解析基准测试输出 — 匹配 '总计' 行."""
    # 匹配 "总计: 39/39 项通过 (100.0%)" 或 "盲测总计: 25/25 (100.0%)"
    m = re.search(r"总计:\s*(\d+)/(\d+).*?\(([\d.]+%)\)", output)
    if m:
        return int(m.group(1)), int(m.group(2)), m.group(3)
    # fallback
    m = re.search(r"(\d+)/(\d+).*?(\d+\.?\d*%)", output)
    if m:
        return int(m.group(1)), int(m.group(2)), m.group(3)
    return 0, 0, "0%"


def extract_failures(output):
    """提取失败项."""
    failures = []
    for line in output.split("\n"):
        if "✗" in line:
            failures.append(line.strip())
    return failures


def main():
    start_time = time.time()
    print("=" * 65)
    print("  UncertaintyLens 综合测试仪表盘")
    print("=" * 65)

    results = {}

    # 1. Pytest
    print("\n[1/4] 运行单元测试 (pytest)...")
    rc, out = run_cmd("PYTHONPATH=. python -m pytest tests/ --tb=short -q")
    passed, failed, skipped, elapsed = parse_pytest(out)
    results["pytest"] = {
        "name": "单元测试 (pytest)",
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "total": passed + failed,
        "elapsed": elapsed,
        "success": failed == 0,
    }
    print(f"  {passed} passed, {failed} failed, {skipped} skipped ({elapsed:.1f}s)")

    # 2. Core benchmark
    print("\n[2/4] 运行核心基准 (7 数据集)...")
    rc, out = run_cmd("PYTHONPATH=. python examples/benchmark_all.py")
    bp, bt, bpct = parse_benchmark(out)
    failures_core = extract_failures(out)
    results["core"] = {
        "name": "核心基准 (benchmark_all)",
        "passed": bp,
        "total": bt,
        "pct": bpct,
        "success": bp == bt,
        "failures": failures_core,
    }
    print(f"  {bp}/{bt} ({bpct})")

    # 3. Blind test
    print("\n[3/4] 运行盲测验证 (3 数据集)...")
    rc, out = run_cmd("PYTHONPATH=. python examples/benchmark_blind.py")
    bp, bt, bpct = parse_benchmark(out)
    failures_blind = extract_failures(out)
    results["blind"] = {
        "name": "盲测验证 (benchmark_blind)",
        "passed": bp,
        "total": bt,
        "pct": bpct,
        "success": bp == bt,
        "failures": failures_blind,
    }
    print(f"  {bp}/{bt} ({bpct})")

    # 4. Extended benchmark
    print("\n[4/4] 运行扩展基准 (6 数据集)...")
    rc, out = run_cmd("PYTHONPATH=. python examples/benchmark_extended.py")
    bp, bt, bpct = parse_benchmark(out)
    failures_ext = extract_failures(out)
    results["extended"] = {
        "name": "扩展基准 (benchmark_extended)",
        "passed": bp,
        "total": bt,
        "pct": bpct,
        "success": bp == bt,
        "failures": failures_ext,
    }
    print(f"  {bp}/{bt} ({bpct})")

    total_elapsed = time.time() - start_time

    # ── 汇总 ──
    total_checks = (
        results["pytest"]["total"]
        + results["core"]["total"]
        + results["blind"]["total"]
        + results["extended"]["total"]
    )
    total_passed = (
        results["pytest"]["passed"]
        + results["core"]["passed"]
        + results["blind"]["passed"]
        + results["extended"]["passed"]
    )
    all_green = all(r["success"] for r in results.values())

    print(f"\n{'=' * 65}")
    print(f"  总计: {total_passed}/{total_checks} 通过")
    print(f"  耗时: {total_elapsed:.1f}s")
    if all_green:
        print("  状态: ✅ 全部通过")
    else:
        print("  状态: ❌ 有失败项")
    print(f"{'=' * 65}")

    # ── 生成 HTML 仪表盘 ──
    html = generate_html(results, total_passed, total_checks, total_elapsed, all_green)
    output_path = Path("test_dashboard.html")
    output_path.write_text(html, encoding="utf-8")
    print(f"\n仪表盘已生成: {output_path.resolve()}")

    return 0 if all_green else 1


def generate_html(results, total_passed, total_checks, elapsed, all_green):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    status_color = "#27ae60" if all_green else "#e74c3c"
    status_text = "ALL PASSED" if all_green else "FAILURES DETECTED"
    pct = total_passed / total_checks * 100 if total_checks > 0 else 0

    # 构建每个测试套件的卡片
    cards = []
    for key in ["pytest", "core", "blind", "extended"]:
        r = results[key]
        ok = r["success"]
        card_color = "#27ae60" if ok else "#e74c3c"
        icon = "✅" if ok else "❌"

        if key == "pytest":
            detail = f'{r["passed"]} passed, {r["failed"]} failed, {r["skipped"]} skipped'
            sub = f'{r["elapsed"]:.1f}s'
        else:
            detail = f'{r["passed"]}/{r["total"]} ({r["pct"]})'
            sub = ""
            if r.get("failures"):
                sub = "<br>".join(
                    f'<span style="color:#e74c3c">{f}</span>' for f in r["failures"][:5]
                )

        cards.append(f"""
        <div class="suite-card" style="border-top: 4px solid {card_color}">
            <div class="suite-header">
                <span class="suite-icon">{icon}</span>
                <span class="suite-name">{r["name"]}</span>
            </div>
            <div class="suite-result" style="color: {card_color}">{detail}</div>
            <div class="suite-sub">{sub}</div>
        </div>
        """)

    cards_html = "\n".join(cards)

    # 检测器列表
    detectors = [
        ("MissingPatternDetector", "缺失模式检测", "统计各特征缺失率及模式"),
        ("AnomalyDetector", "异常值检测", "IQR + Z-score 共识投票"),
        ("VarianceDetector", "方差检测", "变异系数 + 异方差分析"),
        ("ConformalShiftDetector", "分布偏移 (KS)", "逐特征 KS 检验"),
        ("UncertaintyDecomposer", "不确定性分解", "Bootstrap 认知/偶然分解"),
        ("JackknifePlusDetector", "预测区间 (CV+)", "Barber et al. 2021"),
        ("MMDShiftDetector", "分布偏移 (MMD)", "多尺度核检验，自适应带宽"),
        ("ZeroInflationDetector", "零膨胀检测", "双组件分析 + 三维评分"),
        ("UncertaintyExplainer", "可解释性归因", "检测器贡献分解 + 行动建议"),
        ("StreamingDetector", "在线流式检测", "Welford + EWMA + Page-Hinkley"),
    ]
    det_rows = "\n".join(
        f'<tr><td class="det-name">{name}</td><td>{cn}</td><td class="det-desc">{desc}</td></tr>'
        for name, cn, desc in detectors
    )

    # 数据集覆盖矩阵
    datasets_info = [
        ("Housing", "15K", "房产", "异方差/离群值/空间漂移"),
        ("Wine", "6.5K", "化学", "离群值/无缺失/组间偏移"),
        ("Census", "20K", "人口", "零膨胀/缺失/弱偏移"),
        ("Medical", "10K", "临床", "MAR缺失/测量噪声/系统偏差"),
        ("Sensor", "12K", "IoT", "概念漂移/传感器退化"),
        ("Ecommerce", "15K", "电商", "泄露特征/标签噪声/幂律"),
        ("Financial", "10K", "金融", "厚尾/regime/数据损坏"),
        ("Insurance", "8K", "保险", "零膨胀理赔/MNAR缺失"),
        ("Climate", "10K", "气候", "极端长尾/趋势漂移"),
        ("HR", "6K", "人力", "薪资偏移/标签噪声"),
        ("Titanic", "1.2K", "生存", "自然缺失/混合类型"),
        ("Adult", "5K", "收入", "双峰/零膨胀/偏移"),
        ("Adversarial", "2K", "对抗", "边界异常/微弱偏移"),
        ("WideTable", "200", "基因组", "80列宽表/常量列"),
        ("TinySample", "50", "极小", "统计功效下降测试"),
        ("Retail", "4K", "零售", "时间+类别+数值混合"),
    ]
    ds_rows = "\n".join(
        f'<tr><td class="ds-name">{name}</td><td>{size}</td><td>{domain}</td><td class="ds-desc">{desc}</td></tr>'
        for name, size, domain, desc in datasets_info
    )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UncertaintyLens — 综合测试仪表盘</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #f0f2f5;
            color: #1a1a2e;
            line-height: 1.6;
        }}
        .container {{ max-width: 960px; margin: 0 auto; padding: 20px; }}
        header {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            color: white;
            padding: 36px 0 28px;
            margin-bottom: 24px;
        }}
        header h1 {{ font-size: 26px; font-weight: 700; letter-spacing: -0.5px; }}
        .header-sub {{ font-size: 13px; opacity: 0.7; margin-top: 4px; }}
        .hero-row {{
            display: flex;
            gap: 20px;
            margin-bottom: 24px;
            align-items: stretch;
        }}
        .hero-card {{
            flex: 1;
            background: white;
            border-radius: 12px;
            padding: 24px;
            text-align: center;
            box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        }}
        .hero-value {{ font-size: 44px; font-weight: 800; }}
        .hero-label {{ font-size: 13px; color: #7f8c8d; margin-top: 4px; }}
        .hero-sub {{ font-size: 11px; color: #bdc3c7; }}
        .suites-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .suite-card {{
            background: white;
            border-radius: 10px;
            padding: 18px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        .suite-header {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
        }}
        .suite-icon {{ font-size: 18px; }}
        .suite-name {{ font-weight: 600; font-size: 13px; }}
        .suite-result {{ font-size: 20px; font-weight: 700; }}
        .suite-sub {{ font-size: 11px; color: #95a5a6; margin-top: 4px; }}
        .section {{
            background: white;
            border-radius: 10px;
            padding: 22px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        .section h2 {{
            font-size: 16px; color: #1a1a2e;
            margin-bottom: 12px; padding-bottom: 8px;
            border-bottom: 2px solid #ecf0f1;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }}
        th {{
            text-align: left;
            padding: 8px 10px;
            background: #f8f9fa;
            font-weight: 600;
            color: #555;
            border-bottom: 2px solid #ecf0f1;
        }}
        td {{
            padding: 7px 10px;
            border-bottom: 1px solid #f0f0f0;
        }}
        .det-name {{ font-family: "SF Mono", Monaco, monospace; font-size: 11px; color: #2980b9; }}
        .det-desc {{ color: #7f8c8d; }}
        .ds-name {{ font-weight: 600; color: #16213e; }}
        .ds-desc {{ color: #7f8c8d; }}
        .progress-bar {{
            width: 100%;
            height: 8px;
            background: #ecf0f1;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 8px;
        }}
        .progress-fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s;
        }}
        footer {{
            text-align: center;
            padding: 20px;
            font-size: 11px;
            color: #bdc3c7;
        }}
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1>UncertaintyLens — 综合测试仪表盘</h1>
            <div class="header-sub">生成时间: {now} · 总耗时: {elapsed:.0f}s</div>
        </div>
    </header>

    <div class="container">

        <div class="hero-row">
            <div class="hero-card">
                <div class="hero-value" style="color: {status_color}">{total_passed}/{total_checks}</div>
                <div class="hero-label">检查通过</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {pct:.0f}%; background: {status_color}"></div>
                </div>
                <div class="hero-sub" style="margin-top: 6px">{pct:.1f}%</div>
            </div>
            <div class="hero-card">
                <div class="hero-value" style="color: {status_color}">{status_text}</div>
                <div class="hero-label">整体状态</div>
            </div>
            <div class="hero-card">
                <div class="hero-value">16</div>
                <div class="hero-label">数据集</div>
                <div class="hero-sub">7 核心 + 3 盲测 + 6 扩展</div>
            </div>
            <div class="hero-card">
                <div class="hero-value">10</div>
                <div class="hero-label">检测器</div>
                <div class="hero-sub">含可解释性 + 流式</div>
            </div>
        </div>

        <div class="suites-grid">
            {cards_html}
        </div>

        <div class="section">
            <h2>检测器清单</h2>
            <table>
                <thead><tr><th>类名</th><th>功能</th><th>方法描述</th></tr></thead>
                <tbody>{det_rows}</tbody>
            </table>
        </div>

        <div class="section">
            <h2>数据集覆盖矩阵 (16 个)</h2>
            <table>
                <thead><tr><th>数据集</th><th>样本量</th><th>领域</th><th>测试重点</th></tr></thead>
                <tbody>{ds_rows}</tbody>
            </table>
        </div>

    </div>
    <footer>UncertaintyLens v1.0 — 综合测试仪表盘</footer>
</body>
</html>"""


if __name__ == "__main__":
    sys.exit(main())
