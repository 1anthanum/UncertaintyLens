"""
可解释性可视化组件 (Explainer Visualizations).

为 UncertaintyExplainer 的分析结果提供交互式图表:
  1. 归因瀑布图 (Attribution Waterfall) — 每个特征的检测器贡献分解
  2. 全局雷达图 (Global Radar) — 数据集在各检测维度上的健康度
  3. 行动计划甘特图 (Action Priority) — 按优先级排列的修复建议
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List

# ── 颜色配置 ──

DETECTOR_COLORS = {
    "missing": "#e74c3c",
    "anomaly": "#e67e22",
    "variance": "#f1c40f",
    "conformal_shift": "#3498db",
    "decomposition": "#9b59b6",
    "jackknife_plus": "#1abc9c",
    "mmd_shift": "#2980b9",
    "zero_inflation": "#d35400",
    "conformal_pred": "#16a085",
    "deep_ensemble": "#8e44ad",
}

SEVERITY_COLORS = {
    "high": "#e74c3c",
    "moderate": "#f39c12",
    "low": "#27ae60",
}


# ═══════════════════════════════════════════════════════════════════════
# 1. 归因堆叠条形图 — 每个特征的检测器贡献
# ═══════════════════════════════════════════════════════════════════════


def create_attribution_bar(
    explanation: Dict[str, Any],
    title: str = "不确定性归因分解 — 每个特征的问题来源",
    max_features: int = 15,
) -> go.Figure:
    """
    堆叠条形图: 每个特征的 composite_score 拆解为各检测器贡献。

    Parameters
    ----------
    explanation : dict
        UncertaintyExplainer.explain() 的返回结果。
    title : str
        图表标题。
    max_features : int
        最多展示的特征数（按 composite 降序）。
    """
    feat_expls = explanation.get("feature_explanations", {})
    if not feat_expls:
        fig = go.Figure()
        fig.add_annotation(text="无可用数据", showarrow=False, font_size=16)
        return fig

    # 按 composite 降序排列
    sorted_features = sorted(
        feat_expls.items(),
        key=lambda x: x[1]["composite_score"],
        reverse=True,
    )[:max_features]

    features = [f for f, _ in sorted_features]
    features.reverse()  # Plotly 从下到上渲染

    # 收集所有出现过的检测器
    all_detectors = set()
    for _, expl in sorted_features:
        for c in expl["all_contributors"]:
            all_detectors.add(c["detector"])

    # 按名字排序保证一致性
    all_detectors = sorted(all_detectors)

    fig = go.Figure()

    for det in all_detectors:
        values = []
        hover_texts = []
        for feat in features:
            expl = feat_expls[feat]
            contrib_map = {c["detector"]: c for c in expl["all_contributors"]}
            if det in contrib_map:
                c = contrib_map[det]
                values.append(c["contribution"])
                hover_texts.append(
                    f"<b>{feat}</b><br>"
                    f"检测器: {c['label']}<br>"
                    f"原始分数: {c['raw_score']:.3f}<br>"
                    f"贡献: {c['contribution']:.3f} ({c['pct']})<br>"
                    f"原因: {c['reason']}"
                )
            else:
                values.append(0)
                hover_texts.append("")

        color = DETECTOR_COLORS.get(det, "#95a5a6")
        # 获取中文标签
        label_map = {
            c["detector"]: c["label"]
            for _, expl in sorted_features
            for c in expl["all_contributors"]
        }
        label = label_map.get(det, det)

        fig.add_trace(
            go.Bar(
                y=features,
                x=values,
                name=label,
                orientation="h",
                marker_color=color,
                hovertext=hover_texts,
                hoverinfo="text",
            )
        )

    fig.update_layout(
        barmode="stack",
        title=dict(text=title, font_size=16),
        xaxis_title="不确定性贡献",
        yaxis_title="",
        height=max(350, len(features) * 35 + 100),
        margin=dict(l=120, r=30, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font_size=11,
        ),
        plot_bgcolor="white",
        xaxis=dict(gridcolor="#ecf0f1", range=[0, 1]),
    )

    return fig


# ═══════════════════════════════════════════════════════════════════════
# 2. 全局雷达图 — 数据集在各检测维度的健康度
# ═══════════════════════════════════════════════════════════════════════


def create_global_radar(
    explanation: Dict[str, Any],
    title: str = "数据质量雷达图 — 各维度健康度",
) -> go.Figure:
    """
    雷达图: 数据集在缺失/异常/方差/偏移/零膨胀等维度的平均问题程度。

    中心=0 (健康), 外圈=1 (严重问题)。
    """
    insights = explanation.get("global_insights", [])
    feat_expls = explanation.get("feature_explanations", {})

    if not feat_expls:
        fig = go.Figure()
        fig.add_annotation(text="无可用数据", showarrow=False, font_size=16)
        return fig

    # 计算每个检测维度的平均分
    detector_avgs: Dict[str, list] = {}
    for col, expl in feat_expls.items():
        for c in expl["all_contributors"]:
            det = c["detector"]
            detector_avgs.setdefault(det, []).append(c["raw_score"])

    # 取前 8 个维度
    avg_scores = {det: sum(scores) / len(scores) for det, scores in detector_avgs.items()}
    sorted_dims = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:8]

    if not sorted_dims:
        fig = go.Figure()
        fig.add_annotation(text="无可用数据", showarrow=False, font_size=16)
        return fig

    # 标签映射
    label_map = {}
    for col, expl in feat_expls.items():
        for c in expl["all_contributors"]:
            label_map[c["detector"]] = c["label"]

    categories = [label_map.get(d, d) for d, _ in sorted_dims]
    values = [s for _, s in sorted_dims]
    # 闭合雷达图
    categories.append(categories[0])
    values.append(values[0])

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            fillcolor="rgba(231, 76, 60, 0.15)",
            line=dict(color="#e74c3c", width=2),
            marker=dict(size=6),
            name="平均问题程度",
        )
    )

    # 添加 "健康基线" (0.2 以下算正常)
    baseline = [0.2] * len(categories)
    fig.add_trace(
        go.Scatterpolar(
            r=baseline,
            theta=categories,
            line=dict(color="#27ae60", width=1, dash="dash"),
            name="健康基线 (0.2)",
            fill=None,
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#ecf0f1"),
            bgcolor="white",
        ),
        title=dict(text=title, font_size=16),
        height=420,
        margin=dict(t=60, b=30),
        legend=dict(font_size=11),
    )

    return fig


# ═══════════════════════════════════════════════════════════════════════
# 3. 行动计划卡片 HTML (非 Plotly)
# ═══════════════════════════════════════════════════════════════════════


def build_action_plan_html(explanation: Dict[str, Any]) -> str:
    """
    生成行动计划的 HTML 卡片。

    返回可直接嵌入报告的 HTML 片段。
    """
    plan = explanation.get("action_plan", [])
    insights = explanation.get("global_insights", [])

    parts = []

    # ── 全局洞察卡片 ──
    if insights:
        parts.append('<div class="insight-grid">')
        for ins in insights[:4]:
            avg = ins["average_score"]
            if avg >= 0.5:
                color = "#e74c3c"
                icon = "⚠️"
            elif avg >= 0.3:
                color = "#f39c12"
                icon = "⚡"
            else:
                color = "#3498db"
                icon = "ℹ️"

            affected = ", ".join(ins["affected_features"][:3])
            more = (
                f" 等{len(ins['affected_features'])}个" if len(ins["affected_features"]) > 3 else ""
            )

            parts.append(f"""
            <div class="insight-card" style="border-left: 4px solid {color}">
                <div class="insight-header">
                    <span class="insight-icon">{icon}</span>
                    <span class="insight-label">{ins["label"]}</span>
                    <span class="insight-score" style="color: {color}">
                        均值 {avg:.2f}
                    </span>
                </div>
                <div class="insight-body">
                    影响特征: {affected}{more}
                </div>
            </div>
            """)
        parts.append("</div>")

    # ── 行动计划列表 ──
    if plan:
        parts.append('<div class="action-list">')
        for action in plan[:8]:
            sev = action["severity"]
            color = SEVERITY_COLORS.get(sev, "#95a5a6")
            features = ", ".join(action["features"][:4])
            more = f" 等{len(action['features'])}个" if len(action["features"]) > 4 else ""

            parts.append(f"""
            <div class="action-item" style="border-left: 4px solid {color}">
                <div class="action-header">
                    <span class="action-priority">#{action["priority"]}</span>
                    <span class="action-label">{action["label"]}</span>
                    <span class="action-severity" style="background: {color}; color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px;">
                        {sev}
                    </span>
                </div>
                <div class="action-features">特征: {features}{more}</div>
                <div class="action-text">{action["action"]}</div>
            </div>
            """)
        parts.append("</div>")
    else:
        parts.append(
            '<p style="color: #27ae60; font-weight: 500;">✅ 所有特征数据质量良好，无需特殊处理。</p>'
        )

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════
# 4. 单特征详情图 — 贡献瀑布 (Waterfall)
# ═══════════════════════════════════════════════════════════════════════


def create_feature_waterfall(
    feature_explanation: Dict[str, Any],
    feature_name: str = "",
) -> go.Figure:
    """
    单个特征的贡献瀑布图（Waterfall chart）。

    每个检测器的贡献逐级叠加，最终到达 composite_score。
    """
    contributors = feature_explanation.get("all_contributors", [])
    composite = feature_explanation.get("composite_score", 0)

    if not contributors:
        fig = go.Figure()
        fig.add_annotation(text="无贡献数据", showarrow=False, font_size=16)
        return fig

    # 过滤贡献 > 0.001 的检测器
    non_zero = [c for c in contributors if c["contribution"] > 0.001]
    if not non_zero:
        non_zero = contributors[:3]

    labels = [c["label"] for c in non_zero]
    values = [c["contribution"] for c in non_zero]
    colors = [DETECTOR_COLORS.get(c["detector"], "#95a5a6") for c in non_zero]

    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            x=labels + ["合计"],
            y=values + [None],
            measure=["relative"] * len(non_zero) + ["total"],
            connector=dict(line=dict(color="#bdc3c7", width=1)),
            increasing=dict(marker_color="#e74c3c"),
            decreasing=dict(marker_color="#27ae60"),
            totals=dict(marker_color="#34495e"),
            textposition="outside",
            text=[f"+{v:.3f}" for v in values] + [f"{composite:.3f}"],
        )
    )

    fig.update_layout(
        title=dict(
            text=f"'{feature_name}' 不确定性分解" if feature_name else "不确定性分解",
            font_size=14,
        ),
        yaxis_title="累计不确定性",
        height=350,
        margin=dict(t=50, b=40),
        plot_bgcolor="white",
        yaxis=dict(gridcolor="#ecf0f1"),
    )

    return fig
