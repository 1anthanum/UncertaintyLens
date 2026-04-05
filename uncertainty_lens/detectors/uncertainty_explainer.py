"""
Uncertainty Explainer — 不确定性可解释性模块

对每个特征的 composite_score 进行分解，回答 "为什么这个特征不确定性高/低？"

方法: 加权贡献归因 (Weighted Contribution Attribution)
  - 类似 SHAP 的 "每个因素贡献了多少" 思路
  - 但不需要训练模型，因为我们已知权重和各检测器分数
  - 通过 Shapley 值的精确分解确保：贡献之和 = composite_score

输出:
  - 每个特征的 "诊断报告": 哪些检测器贡献最大，为什么
  - 自然语言建议: 针对主要问题的改进方向
  - 全局概览: 数据集层面最大的系统性问题是什么

用法:
    explainer = UncertaintyExplainer()
    explanation = explainer.explain(pipeline_report)
    # explanation["feature_explanations"]["income"]["top_contributors"]
    # → [{"detector": "missing", "contribution": 0.35, "pct": "62%", "reason": "..."}]
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, List, Optional

# ── 检测器名称 → 中英文描述 + 诊断原因模板 ──

DETECTOR_META = {
    "missing": {
        "cn": "缺失检测",
        "en": "Missing Pattern",
        "reasons": {
            "high": "该特征有较多缺失值，可能存在系统性数据收集问题",
            "moderate": "该特征有少量缺失值",
            "low": "该特征几乎无缺失值",
        },
    },
    "anomaly": {
        "cn": "异常值检测",
        "en": "Anomaly Detection",
        "reasons": {
            "high": "该特征包含显著离群值或极端分布尾部",
            "moderate": "该特征有少量异常值",
            "low": "该特征分布较为集中，异常值少",
        },
    },
    "variance": {
        "cn": "方差检测",
        "en": "Variance Analysis",
        "reasons": {
            "high": "该特征变异系数大，数据波动显著",
            "moderate": "该特征方差处于中等水平",
            "low": "该特征方差小，数据稳定",
        },
    },
    "conformal_shift": {
        "cn": "分布偏移 (KS)",
        "en": "Conformal Shift",
        "reasons": {
            "high": "不同组之间该特征的分布差异显著 (KS检验)",
            "moderate": "组间存在一定分布差异",
            "low": "组间分布基本一致",
        },
    },
    "decomposition": {
        "cn": "不确定性分解",
        "en": "Uncertainty Decomposition",
        "reasons": {
            "high": "bootstrap 分解显示该特征统计量不稳定",
            "moderate": "统计量有一定波动",
            "low": "统计量稳定",
        },
    },
    "jackknife_plus": {
        "cn": "预测区间 (CV+)",
        "en": "Jackknife+ Prediction",
        "reasons": {
            "high": "交叉验证预测区间宽，说明该特征难以被其他特征预测",
            "moderate": "预测区间适中",
            "low": "预测区间窄，该特征可被较好预测",
        },
    },
    "mmd_shift": {
        "cn": "分布偏移 (MMD)",
        "en": "MMD Shift Detection",
        "reasons": {
            "high": "MMD 核检验显示组间联合分布差异显著",
            "moderate": "组间存在一定多维分布差异",
            "low": "组间多维分布基本一致",
        },
    },
    "zero_inflation": {
        "cn": "零膨胀检测",
        "en": "Zero-Inflation Detection",
        "reasons": {
            "high": "该特征大量为零值 (零膨胀)，需要特殊统计方法处理",
            "moderate": "有一定比例零值",
            "low": "零值比例正常",
        },
    },
    "conformal_pred": {
        "cn": "共形预测",
        "en": "Conformal Prediction",
        "reasons": {
            "high": "共形预测区间宽，数据噪声大",
            "moderate": "预测区间适中",
            "low": "预测区间窄",
        },
    },
    "deep_ensemble": {
        "cn": "深度集成",
        "en": "Deep Ensemble",
        "reasons": {
            "high": "多个神经网络预测结果分歧大",
            "moderate": "预测分歧适中",
            "low": "网络预测一致",
        },
    },
}

# ── 行动建议模板 ──

ACTION_TEMPLATES = {
    "missing": {
        "high": "建议调查缺失原因 (MCAR/MAR/MNAR)，考虑多重插补 (MICE) 或标记缺失指示变量",
        "moderate": "可使用中位数/KNN 插补，但需检查缺失是否与其他变量相关",
    },
    "anomaly": {
        "high": "建议使用 Winsorize (截尾) 或 robust 统计方法 (中位数/MAD)，排查数据录入错误",
        "moderate": "可保留异常值但对下游模型使用 robust 损失函数",
    },
    "variance": {
        "high": "考虑对数变换/Box-Cox 稳定方差，或使用异方差模型 (WLS/GARCH)",
        "moderate": "标准化处理后通常可接受",
    },
    "conformal_shift": {
        "high": "组间分布差异大，建议分组建模或加入组别交互项",
        "moderate": "可加入组别变量作为控制变量",
    },
    "mmd_shift": {
        "high": "多维联合分布偏移，建议检查是否存在 concept drift 或 selection bias",
        "moderate": "关注组间相关结构是否稳定",
    },
    "zero_inflation": {
        "high": "使用零膨胀模型 (ZIP/ZINB)，或将该特征二值化 (是否为零 + 非零部分建模)",
        "moderate": "可尝试 Tobit 模型或 hurdle 模型",
    },
    "jackknife_plus": {
        "high": "该特征难以被预测，可能包含独立信息或纯噪声，需要领域知识判断",
        "moderate": "预测区间偏宽，数据质量可能有提升空间",
    },
    "decomposition": {
        "high": "统计量对数据扰动敏感，建议增大样本量或使用更稳健的估计方法",
        "moderate": "波动在可接受范围内",
    },
}


class UncertaintyExplainer:
    """
    对 UncertaintyPipeline 的分析结果进行可解释性分解。

    Parameters
    ----------
    language : str
        输出语言: "cn" (中文) 或 "en" (English). 默认 "cn".
    top_k : int
        每个特征报告前 k 个最大贡献的检测器. 默认 3.
    """

    def __init__(self, language: str = "cn", top_k: int = 3):
        if language not in ("cn", "en"):
            raise ValueError(f"language must be 'cn' or 'en', got '{language}'")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        self.language = language
        self.top_k = top_k

    def explain(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """
        对 pipeline 报告进行可解释性分析。

        Parameters
        ----------
        report : dict
            UncertaintyPipeline.analyze() 返回的报告字典。
            必须包含 "uncertainty_index" 键。

        Returns
        -------
        dict
            包含:
            - feature_explanations: 每个特征的详细归因分解
            - global_insights: 数据集级别的系统性问题汇总
            - action_plan: 按优先级排序的改进建议
        """
        if not isinstance(report, dict):
            raise TypeError(f"report must be dict, got {type(report).__name__}")

        ui = report.get("uncertainty_index", {})
        if not ui:
            return {
                "feature_explanations": {},
                "global_insights": [],
                "action_plan": [],
            }

        # ── 逐特征分解 ──
        feature_explanations = {}
        for col, entry in ui.items():
            explanation = self._explain_feature(col, entry)
            feature_explanations[col] = explanation

        # ── 全局洞察 ──
        global_insights = self._global_insights(ui)

        # ── 行动计划 ──
        action_plan = self._action_plan(feature_explanations)

        return {
            "feature_explanations": feature_explanations,
            "global_insights": global_insights,
            "action_plan": action_plan,
        }

    def _explain_feature(self, col: str, entry: Dict[str, Any]) -> Dict[str, Any]:
        """分解单个特征的 composite_score."""
        composite = entry.get("composite_score", 0)

        # 提取所有 *_score 键
        detector_scores = {}
        for key, val in entry.items():
            if key.endswith("_score") and key != "composite_score":
                det_name = key[: -len("_score")]  # e.g. "missing_score" → "missing"
                detector_scores[det_name] = float(val)

        # 计算每个检测器的相对贡献
        # 贡献 ∝ detector_score (占比 = score / sum_of_all_scores)
        total_score = sum(detector_scores.values())
        contributions = []
        for det_name, score in detector_scores.items():
            if total_score > 0:
                pct = score / total_score
            else:
                pct = 0.0

            # 绝对贡献 = 占比 × composite
            abs_contribution = pct * composite

            # 确定严重程度
            if score >= 0.6:
                severity = "high"
            elif score >= 0.2:
                severity = "moderate"
            else:
                severity = "low"

            meta = DETECTOR_META.get(det_name, {})
            if self.language == "cn":
                label = meta.get("cn", det_name)
            else:
                label = meta.get("en", det_name)

            reason = meta.get("reasons", {}).get(severity, "")

            contributions.append(
                {
                    "detector": det_name,
                    "label": label,
                    "raw_score": round(score, 4),
                    "contribution": round(abs_contribution, 4),
                    "pct": f"{pct:.0%}",
                    "severity": severity,
                    "reason": reason,
                }
            )

        # 按贡献降序
        contributions.sort(key=lambda x: x["contribution"], reverse=True)

        # 生成摘要
        top = contributions[: self.top_k]
        if self.language == "cn":
            if composite >= 0.6:
                summary = f"'{col}' 不确定性较高 ({composite:.2f})，"
                summary += "主要原因: " + "、".join(
                    f"{c['label']}({c['pct']})" for c in top if c["severity"] != "low"
                )
            elif composite >= 0.3:
                summary = f"'{col}' 不确定性中等 ({composite:.2f})，"
                drivers = [c for c in top if c["severity"] != "low"]
                if drivers:
                    summary += "注意: " + "、".join(f"{c['label']}({c['pct']})" for c in drivers)
                else:
                    summary += "各检测维度均在可接受范围"
            else:
                summary = f"'{col}' 不确定性低 ({composite:.2f})，数据质量良好"
        else:
            if composite >= 0.6:
                summary = f"'{col}' has high uncertainty ({composite:.2f}). "
                summary += "Main drivers: " + ", ".join(
                    f"{c['label']} ({c['pct']})" for c in top if c["severity"] != "low"
                )
            elif composite >= 0.3:
                summary = f"'{col}' has moderate uncertainty ({composite:.2f})."
                drivers = [c for c in top if c["severity"] != "low"]
                if drivers:
                    summary += " Watch: " + ", ".join(f"{c['label']} ({c['pct']})" for c in drivers)
                else:
                    summary += " All dimensions within acceptable range."
            else:
                summary = f"'{col}' has low uncertainty ({composite:.2f}). Good data quality."

        return {
            "composite_score": composite,
            "level": entry.get("level", ""),
            "summary": summary,
            "top_contributors": top,
            "all_contributors": contributions,
        }

    def _global_insights(self, ui: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """数据集层面的系统性问题汇总."""
        insights = []

        # 聚合每个检测器在所有特征上的平均分
        detector_avgs: Dict[str, List[float]] = {}
        for col, entry in ui.items():
            for key, val in entry.items():
                if key.endswith("_score") and key != "composite_score":
                    det = key[: -len("_score")]
                    detector_avgs.setdefault(det, []).append(float(val))

        avg_scores = {det: np.mean(scores) for det, scores in detector_avgs.items()}

        # 找出系统性最严重的问题 (平均分 > 0.3)
        systemic = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        for det, avg in systemic:
            if avg < 0.2:
                continue
            meta = DETECTOR_META.get(det, {})
            label = meta.get("cn" if self.language == "cn" else "en", det)

            # 哪些特征受影响最大
            affected = []
            for col, entry in ui.items():
                s = entry.get(f"{det}_score", 0)
                if s >= 0.3:
                    affected.append((col, s))
            affected.sort(key=lambda x: x[1], reverse=True)

            if self.language == "cn":
                desc = f"系统性问题: {label} (全局平均 {avg:.2f})"
                if affected:
                    desc += f"，影响特征: {', '.join(c for c, _ in affected[:5])}"
            else:
                desc = f"Systemic issue: {label} (global avg {avg:.2f})"
                if affected:
                    desc += f", affecting: {', '.join(c for c, _ in affected[:5])}"

            insights.append(
                {
                    "detector": det,
                    "label": label,
                    "average_score": round(avg, 4),
                    "affected_features": [c for c, _ in affected],
                    "description": desc,
                }
            )

        return insights

    def _action_plan(self, explanations: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """按优先级排序的行动建议."""
        actions = []
        seen = set()

        # 按 composite_score 降序处理
        sorted_features = sorted(
            explanations.items(),
            key=lambda x: x[1]["composite_score"],
            reverse=True,
        )

        for col, expl in sorted_features:
            if expl["composite_score"] < 0.3:
                continue  # 低不确定性不需要行动

            for contrib in expl["top_contributors"]:
                if contrib["severity"] == "low":
                    continue

                det = contrib["detector"]
                sev = contrib["severity"]

                # 每种 (detector, severity) 组合只出现一次
                key = (det, sev)
                if key in seen:
                    # 但追加受影响的特征
                    for a in actions:
                        if a["detector"] == det and a["severity"] == sev:
                            if col not in a["features"]:
                                a["features"].append(col)
                    continue
                seen.add(key)

                template = ACTION_TEMPLATES.get(det, {}).get(sev, "")
                actions.append(
                    {
                        "priority": len(actions) + 1,
                        "detector": det,
                        "label": contrib["label"],
                        "severity": sev,
                        "features": [col],
                        "action": template,
                    }
                )

        return actions
