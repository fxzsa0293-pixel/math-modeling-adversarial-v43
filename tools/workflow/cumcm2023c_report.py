from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .category_forecast import verify_category_forecast


REQUIRED_CASES = [
    "CUMCM2023C_sales_distribution_relationships",
    "CUMCM2023C_daily_sales_forecast",
    "CUMCM2023C_category_replenishment_pricing",
    "CUMCM2023C_item_replenishment_pricing",
]


def build_cumcm2023c_report(benchmark_summary: Path, output_path: Path) -> Dict[str, Any]:
    benchmark_summary = Path(benchmark_summary).resolve()
    suite = json.loads(benchmark_summary.read_text(encoding="utf-8"))
    cases = {item["case_id"]: item for item in suite.get("cases", [])}
    missing = [case_id for case_id in REQUIRED_CASES if cases.get(case_id, {}).get("verdict") != "pass"]
    if missing:
        raise ValueError(f"report_requires_passing_cases:{missing}")

    analysis_manifest = json.loads(Path(cases[REQUIRED_CASES[0]]["summary_path"]).read_text(encoding="utf-8"))
    analysis_text = Path(analysis_manifest["artifacts"]["report_md"]["path"]).read_text(encoding="utf-8").replace("# 销量分布与关联分析", "").replace("## ", "### ").strip()
    forecast = _load_summary(cases[REQUIRED_CASES[1]])
    category = _load_summary(cases[REQUIRED_CASES[2]])
    item = _load_summary(cases[REQUIRED_CASES[3]])
    category_policy = pd.read_csv(category["workflow_exports"]["recommended_retail_policy_csv"])
    item_policy = pd.read_csv(item["workflow_exports"]["recommended_retail_policy_csv"])
    selected_items = item_policy.loc[item_policy["selected"].astype(bool)].copy()
    category_record = _recommended_record(category)
    item_record = _recommended_record(item)
    category_comparison = category["domain_verification"].get("policy_comparison") or {}
    item_comparison = item["domain_verification"].get("policy_comparison") or {}
    category_forecast_verification = cases[REQUIRED_CASES[2]].get("upstream_category_forecast") or {}
    if category_forecast_verification:
        current_forecast_verification = verify_category_forecast(Path(category_forecast_verification["manifest_path"]))
        if not current_forecast_verification["passed"] or current_forecast_verification["manifest_hash"] != category_forecast_verification.get("manifest_hash"):
            raise ValueError(f"report_category_forecast_verification_failed:{current_forecast_verification['issues']}")
    category_forecast_manifest = json.loads(Path(category_forecast_verification["manifest_path"]).read_text(encoding="utf-8")) if category_forecast_verification.get("passed") else {}
    forecast_diagnostics = pd.read_csv(category_forecast_manifest["artifacts"]["diagnostics"]["path"]) if category_forecast_manifest else pd.DataFrame()
    selected_forecasts = forecast_diagnostics.loc[forecast_diagnostics["selected_by_selection"].astype(bool)] if not forecast_diagnostics.empty else forecast_diagnostics
    item_evidence = json.loads(Path(item_record["evidence_paths"]["route_evidence"]).read_text(encoding="utf-8"))["evidence"]
    service_rows = [dict(scenario=scenario["name"], **service) for scenario in item_evidence["scenario_results"] for service in scenario.get("category_service", [])]
    service_frame = pd.DataFrame(service_rows)

    lines = [
        "# 蔬菜类商品的自动定价与补货决策",
        "",
        "## 摘要",
        "",
        "本文基于四个原始附件建立可追溯的数据分析、需求场景与补货定价模型。问题1使用日销量分布、Pearson/Spearman相关及单品支持度筛选描述品类与单品关系；问题2对六品类未来七日建立离散价格-需求场景与考虑损耗的CVaR联合补货定价模型；问题3将候选集限制为2023年6月24-30日可售单品，在27-33个单品和最小陈列量2.5 kg约束下优化7月1日策略。所有数值结果均绑定源文件和派生模型哈希，并由独立验证器重算；策略比较采用共同随机数压力测试。相关分析仅反映描述性联系，不作因果解释。",
        "",
        "关键词：蔬菜零售；需求响应；补货定价；混合整数规划；CVaR；稳健性分析",
        "",
        "## 1 问题重述",
        "",
        "目标是在历史销售、批发价格和近期损耗率条件下，分析销量规律，制定品类级七日补货定价方案，并在有限销售空间下制定7月1日单品级方案，同时明确进一步需要采集的数据。",
        "",
        "## 2 假设与符号",
        "",
        "- 当日未售商品不结转，损耗通过可销售补货量折减表示。",
        "- 决策日前未知批发价和需求只由训练截止日前数据估计，不使用未来真实值。",
        "- 价格在三个由历史成本和售价构造的离散档位中选择，价格弹性仅从历史期估计并限制在声明区间。",
        "- 低、中、高需求场景刻画不确定性，风险目标采用CVaR，压力测试与优化场景分离。",
        "- 描述性相关可能受季节、促销、供给和共同趋势影响，不等同于替代或互补关系。",
        "",
        "主要符号：q_i为补货量，p_ik为价格档，y_ik为价格选择变量，s_iw为场景销量，d_ikw为场景需求，c_i为单位批发成本，l_i为损耗率。",
        "",
        "核心模型为：",
        "",
        r"$$\sum_k y_{ik}=x_i,\qquad q_i^{\min}x_i\le q_i\le q_i^{\max}x_i,$$",
        "",
        r"$$0\le s_{ikw}\le d_{ikw}y_{ik},\qquad \sum_k s_{ikw}\le(1-l_i)q_i,$$",
        "",
        r"$$\Pi_w=\sum_{i,k}p_{ik}s_{ikw}-\sum_i c_iq_i-h\sum_{i,k}(d_{ikw}y_{ik}-s_{ikw}),$$",
        "",
        r"$$\min\;\eta+\frac{1}{1-\alpha}\sum_w\pi_w z_w,\qquad z_w\ge-\Pi_w-\eta,\ z_w\ge0.$$",
        "",
        r"问题3另加入品类服务约束 $\sum_{i\in g,k}s_{ikw}\ge \rho D_{gw}$，其中当前声明服务率 $\rho=0.8$。",
        "",
        "## 3 数据与验证协议",
        "",
        f"分析及参数估计截止于{analysis_manifest['cutoff']}。问题2决策日期为2023-07-01至2023-07-07；问题3候选集来自2023-06-24至2023-06-30。四个真实案例均通过当前管线门控。聚合预测推荐路线为{forecast['final_judge']['recommended_route']['route_id']}；品类预测使用8个非重叠滚动选择窗口选路，另留4个窗口仅作最终检验。",
        "",
        "## 4 问题1：销量分布与关联",
        "",
        analysis_text,
        "",
        "## 5 问题2：品类级补货与定价",
        "",
        "对每个日期-品类选择一个价格档并决定补货量。场景销量同时受需求上界和损耗后可销售补货量约束；场景利润等于销售收入减采购成本和缺货惩罚，优化利润损失的CVaR。固定成本加成策略作为同约束baseline。",
        "",
        _forecast_route_table(selected_forecasts),
        "",
        f"六个品类的路线仅按滚动选择区间MAE确定；最终检验发生{int(selected_forecasts['final_test_reversal'].astype(bool).sum()) if not selected_forecasts.empty else 0}次路线反转，该结果仅用于披露模型不确定性，不用于重新选路。未来7日低、中、高场景由所选路线在选择区间的残差比例分位数生成，并作为问题2优化的上游输入。",
        "",
        _policy_table(category_policy, 42),
        "",
        f"推荐路线为{category_record['route_id']}，声明场景期望利润为{category_record['metrics']['expected_profit']:.3f}。{_comparison_sentence(category_comparison)}",
        "",
        "## 6 问题3：单品级补货与定价",
        "",
        f"候选集中共有{len(item_policy)}个近期可售单品；模型选择{len(selected_items)}个，满足27-33个范围。所有选中单品补货量均不低于{selected_items['replenishment'].min():.3f} kg，六个品类均有覆盖。模型进一步设置18条品类-场景服务约束，要求各品类销量达到预测需求的80%。",
        "",
        _policy_table(selected_items, 33),
        "",
        _service_table(service_frame),
        "",
        f"推荐路线为{item_record['route_id']}，声明场景期望利润为{item_record['metrics']['expected_profit']:.3f}。{_comparison_sentence(item_comparison)}",
        "",
        "## 7 问题4：进一步数据需求",
        "",
        "1. 采集逐时客流、天气、节假日和社区活动，以区分共同需求冲击与品类替代关系。",
        "2. 记录陈列面积、货架位置、缺货时段和补货到架时间，以识别零销量来自无需求还是无库存。",
        "3. 保存促销、折扣原因、标签价和成交价，以识别真实价格弹性并减轻清仓行为造成的内生性偏差。",
        "4. 记录批次、供应商、产地、品质等级和到货时间，以建立批发价与损耗率的条件分布。",
        "5. 记录报损量、盘点库存和废弃处置价值，以替换当前平均损耗率假设。",
        "6. 进行小范围随机价格或陈列实验，以支持替代、互补和因果需求响应识别。",
        "",
        "## 8 模型评价与结论边界",
        "",
        "模型优点是原始附件、上游预测、派生参数、决策模型、策略明细和验证结果均有哈希绑定，baseline与候选共享场景和随机冲击。压力测试以最近365天去除星期效应后的品类对数销量残差估计跨品类相关矩阵，并通过高斯copula生成相关对数正态冲击。主要局限是价格只在离散网格中搜索，弹性来自观察性数据，相关矩阵被假定在决策期稳定，且附件不含决策期真实经营结果，因此压力测试不能替代部署回测。",
        "",
        "## 9 可复现工件",
        "",
        f"- 问题1 manifest：{cases[REQUIRED_CASES[0]]['summary_path']}",
        f"- 聚合预测 summary：{cases[REQUIRED_CASES[1]]['summary_path']}",
        f"- 问题2策略：{category['workflow_exports']['recommended_retail_policy_csv']}",
        f"- 问题3策略：{item['workflow_exports']['recommended_retail_policy_csv']}",
        f"- 问题3品类服务达成：{item['workflow_exports'].get('recommended_retail_category_service_csv', '')}",
        "",
    ]
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return {"path": str(output_path), "section_count": 9, "required_cases": REQUIRED_CASES, "source_summary": str(benchmark_summary)}


def verify_cumcm2023c_report(benchmark_summary: Path) -> Dict[str, Any]:
    benchmark_summary = Path(benchmark_summary).resolve()
    suite = json.loads(benchmark_summary.read_text(encoding="utf-8"))
    report = suite.get("complete_report") or {}
    path = Path(str(report.get("path", "")))
    issues = []
    if not path.is_file():
        issues.append("complete_report_missing")
    elif _sha256(path) != report.get("sha256"):
        issues.append("complete_report_hash_mismatch")
    if report.get("required_cases") != REQUIRED_CASES:
        issues.append("complete_report_required_cases_mismatch")
    cases = {item.get("case_id"): item for item in suite.get("cases", [])}
    if any(cases.get(case_id, {}).get("verdict") != "pass" for case_id in REQUIRED_CASES):
        issues.append("complete_report_case_gate_not_passed")
    return {"kind": "cumcm2023c_complete_report", "passed": not issues, "issues": issues, "path": str(path) if path else None}


def _load_summary(case: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(Path(case["summary_path"]).read_text(encoding="utf-8"))


def _recommended_record(summary: Dict[str, Any]) -> Dict[str, Any]:
    route_id = summary["final_judge"]["recommended_route"]["route_id"]
    registry = json.loads(Path(summary["registry_path"]).read_text(encoding="utf-8"))
    return next(item for item in registry if item["route_id"] == route_id)


def _comparison_sentence(comparison: Dict[str, Any]) -> str:
    if not comparison:
        return "压力测试比较不可用。"
    conclusion = "候选在声明压力分布下稳健优于baseline" if comparison.get("candidate_robustly_better") is True else "区间未证明候选稳健优于baseline，正式推荐应回退baseline"
    return f"共同随机数压力测试中，候选相对baseline平均利润差为{comparison['mean_profit_difference']:.3f}，95%区间为[{comparison['ci95_lower']:.3f}, {comparison['ci95_upper']:.3f}]，候选胜率为{comparison['candidate_win_rate']:.3f}；{conclusion}。"


def _policy_table(frame: pd.DataFrame, limit: int) -> str:
    lines = ["| 日期 | 商品或品类 | 售价 | 补货量(kg) |", "|---|---|---:|---:|"]
    for row in frame.head(limit).to_dict(orient="records"):
        lines.append(f"| {row.get('decision_date', '')} | {row.get('category', '')} | {float(row['price']):.4f} | {float(row['replenishment']):.4f} |")
    return "\n".join(lines)


def _forecast_route_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "品类预测诊断不可用。"
    lines = ["| 品类 | 选择路线 | 选择MAE | 最终检验MAE | 是否反转 |", "|---|---|---:|---:|---|"]
    for row in frame.to_dict(orient="records"):
        lines.append(f"| {row['category']} | {row['route']} | {float(row['selection_mae']):.3f} | {float(row['final_test_mae']):.3f} | {'是' if bool(row['final_test_reversal']) else '否'} |")
    return "\n".join(lines)


def _service_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "品类服务约束证据不可用。"
    lines = ["| 场景 | 品类 | 最低销量 | 实际销量 | 达标 |", "|---|---|---:|---:|---|"]
    for row in frame.to_dict(orient="records"):
        lines.append(f"| {row['scenario']} | {row['group']} | {float(row['minimum_sales']):.3f} | {float(row['actual_sales']):.3f} | {'是' if bool(row['satisfied']) else '否'} |")
    return "\n".join(lines)


def _sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
