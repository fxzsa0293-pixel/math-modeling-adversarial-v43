"""
V36 resource-backed C-topic adapters.

Only adds adapters for categories that have local runnable resources:
- evaluation/ranking: CUMCM2012A wine evaluation data;
- retail forecasting/operation: CUMCM2023C vegetable sales data.

Categories without local data (survey, NLP/public opinion, generic graph) are
intentionally excluded rather than filled with speculative templates.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_absolute_error


@dataclass
class ResourceBackedResult:
    case: str
    c_topic_pattern: str
    status: str
    data_path: str
    routes: List[Dict[str, Any]]
    aggregate: Dict[str, Any]
    recommended_route: str
    recommendation_evidence: List[str]
    figure_descriptions: List[str]
    limitations: List[str]


def run_v36_resource_backed_c_adapters(
    reference_roots: Iterable[Path | str] | None = None,
    output_dir: Path | str | None = None,
) -> Dict[str, Any]:
    roots = _resolve_roots(reference_roots)
    results: List[ResourceBackedResult] = []

    wine_file = _find_file(roots, "2012A_T1_processed.xls", preferred_keywords=["CUMCM2012A题", "P15-1"])
    if wine_file:
        results.append(_run_2012a_wine_evaluation(wine_file))
    else:
        results.append(_blocked("CUMCM2012A_wine_evaluation_v36", "evaluation_ranking", Path("."), "2012A_T1_processed.xls not found"))

    vegetable_root = _find_dir_with_files(roots, required_files=["附件1.xlsx", "附件2.xlsx"], preferred_keywords=["2023赛题", "C题"])
    if vegetable_root:
        results.append(_run_2023c_vegetable_retail(vegetable_root))
    else:
        results.append(_blocked("CUMCM2023C_vegetable_retail_v36", "retail_forecast_operation", Path("."), "2023C 附件1/附件2 not found"))

    suite = {
        "version": "v36",
        "purpose": "resource-backed C-topic adapters for locally available categories",
        "summary": {
            "executed_cases": sum(1 for result in results if result.status == "executed"),
            "blocked_cases": sum(1 for result in results if result.status != "executed"),
            "patterns": [result.c_topic_pattern for result in results],
            "routes_evaluated": sum(len(result.routes) for result in results),
            "excluded_due_to_missing_resources": ["survey_questionnaire", "nlp_public_opinion", "generic_graph_network"],
            "local_python_execution": True,
        },
        "results": [asdict(result) for result in results],
        "workflow_patch": _workflow_patch(),
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "v36_resource_backed_c_adapters.json").write_text(json.dumps(suite, ensure_ascii=False, indent=2), encoding="utf-8")
        (output_dir / "v36_resource_backed_c_adapters.md").write_text(format_v36_report(suite), encoding="utf-8")
    return suite


def format_v36_report(suite: Dict[str, Any]) -> str:
    lines = ["# V36 有资源支撑的 C 题专向 adapter 报告", "", f"- summary: {suite.get('summary')}", "", "## Results"]
    for result in suite.get("results", []):
        lines.extend([
            f"### {result['case']}",
            f"- pattern: {result['c_topic_pattern']}",
            f"- status: {result['status']}",
            f"- data_path: {result['data_path']}",
            f"- recommended_route: {result['recommended_route']}",
            f"- aggregate: {result['aggregate']}",
            f"- recommendation_evidence: {result['recommendation_evidence']}",
            "- route leaderboard:",
        ])
        for route in result.get("routes", []):
            lines.append(f"  - {route.get('route')}: {route.get('metrics')}")
        lines.append("- figure_descriptions:")
        for item in result.get("figure_descriptions", []):
            lines.append(f"  - {item}")
        lines.append("- limitations:")
        for item in result.get("limitations", []):
            lines.append(f"  - {item}")
        lines.append("")
    lines.append("## Workflow Patch")
    for item in suite.get("workflow_patch", []):
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def _run_2012a_wine_evaluation(data_file: Path) -> ResourceBackedResult:
    red_first = _judge_sample_scores(data_file, "T1_red_grape")
    red_second = _judge_sample_scores(data_file, "T2_red_grape")
    white_first = _judge_sample_scores(data_file, "T1_white_grape")
    white_second = _judge_sample_scores(data_file, "T2_white_grape")
    first = pd.concat([red_first.assign(kind="red"), white_first.assign(kind="white")], ignore_index=True)
    second = pd.concat([red_second.assign(kind="red"), white_second.assign(kind="white")], ignore_index=True)
    merged = first.merge(second, on=["kind", "sample_id"], suffixes=("_first", "_second"))
    merged["consensus_score"] = merged["mean_score_second"]
    score_cols_first = [col for col in merged.columns if col.startswith("judge_") and col.endswith("_first")]
    score_cols_second = [col for col in merged.columns if col.startswith("judge_") and col.endswith("_second")]
    judge_cols = score_cols_first + score_cols_second

    routes: List[Dict[str, Any]] = []
    first_metrics = _ranking_metrics(merged["consensus_score"], merged["mean_score_first"])
    routes.append({"route": "first_panel_baseline", "metrics": first_metrics})

    median_aggregate = merged[score_cols_first].median(axis=1)
    routes.append({"route": "equal_judge_average", "metrics": _ranking_metrics(merged["consensus_score"], median_aggregate)})

    judge_matrix = merged[score_cols_first].to_numpy(dtype=float)
    reliabilities = []
    for idx, col in enumerate(score_cols_first):
        others = np.delete(judge_matrix, idx, axis=1).mean(axis=1)
        corr = spearmanr(judge_matrix[:, idx], others).correlation
        reliabilities.append(max(float(corr) if np.isfinite(corr) else 0.0, 0.0))
    weights = np.asarray(reliabilities, dtype=float)
    if weights.sum() <= 1e-12:
        weights = np.ones_like(weights)
    weighted = judge_matrix @ (weights / weights.sum())
    reliability_metrics = _ranking_metrics(merged["consensus_score"], weighted)
    reliability_metrics["mean_judge_reliability"] = round(float(np.mean(reliabilities)), 6)
    routes.append({"route": "reliability_weighted_panel", "metrics": reliability_metrics})

    topsis_input = merged[["mean_score_first", "std_score_first"]].copy()
    topsis_input["stability_first"] = -topsis_input.pop("std_score_first")
    topsis_score = _topsis_score(topsis_input.to_numpy(dtype=float))
    routes.append({"route": "score_stability_topsis", "metrics": _ranking_metrics(merged["consensus_score"], topsis_score)})

    routes = sorted(routes, key=lambda route: (-route["metrics"]["spearman_to_consensus"], -route["metrics"].get("top5_overlap", 0)))
    best = routes[0]
    aggregate = {
        "n_samples": int(len(merged)),
        "best_route": best["route"],
        "best_spearman_to_consensus": best["metrics"]["spearman_to_consensus"],
        "best_kendall_to_consensus": best["metrics"]["kendall_to_consensus"],
        "baseline_spearman": first_metrics["spearman_to_consensus"],
        "top5_overlap": best["metrics"]["top5_overlap"],
    }
    return ResourceBackedResult(
        case="CUMCM2012A_wine_evaluation_ranking_v36",
        c_topic_pattern="evaluation_ranking",
        status="executed",
        data_path=str(data_file),
        routes=routes,
        aggregate=aggregate,
        recommended_route=best["route"],
        recommendation_evidence=[
            "真实读取 2012A 葡萄酒两组评委评分表，构造样品级评分矩阵。",
            "使用第二组评委均分作为 holdout 目标，比较第一组均值、第一组可靠性加权与第一组 TOPSIS 排序。",
            f"推荐 {best['route']}，Spearman={best['metrics']['spearman_to_consensus']}，Top5 overlap={best['metrics']['top5_overlap']}。",
        ],
        figure_descriptions=[
            "两组评委样品均分散点图：检查评委组间一致性。",
            "不同评价路线 Spearman/Kendall 对比图：展示排序稳定性。",
            "评委可靠性权重条形图：说明加权依据而不是拍脑袋赋权。",
            "Top5 样品排序对照表：展示模型推荐与共识排序差异。",
        ],
        limitations=[
            "该 adapter 使用第二组评委作为外部检验目标，仍不等同于真实客观酒质标签。",
            "当前只用第一组评分预测第二组排序，未引入理化指标，适合作为评价排序基线而非完整 2012A 解法。",
        ],
    )


def _run_2023c_vegetable_retail(case_dir: Path) -> ResourceBackedResult:
    item_file = case_dir / "附件1.xlsx"
    sales_file = case_dir / "附件2.xlsx"
    item_df = pd.read_excel(item_file)
    sales_df = pd.read_excel(sales_file)
    item_df["单品编码"] = item_df["单品编码"].astype(str)
    sales_df["单品编码"] = sales_df["单品编码"].astype(str)
    sales_df = sales_df[sales_df["销售类型"].astype(str).str.contains("销售", na=False)].copy()
    sales_df["销售日期"] = pd.to_datetime(sales_df["销售日期"])
    sales_df["revenue"] = sales_df["销量(千克)"] * sales_df["销售单价(元/千克)"]
    merged = sales_df.merge(item_df, on="单品编码", how="left")
    category_col = "分类名称" if "分类名称" in merged.columns else None
    if not category_col:
        return _blocked("CUMCM2023C_vegetable_retail_v36", "retail_forecast_operation", case_dir, "分类名称 column not found after merge")

    daily = merged.groupby(["销售日期", category_col], as_index=False).agg(quantity=("销量(千克)", "sum"), revenue=("revenue", "sum"))
    pivot = daily.pivot(index="销售日期", columns=category_col, values="quantity").fillna(0.0).sort_index()
    total_daily = pivot.sum(axis=1)
    horizon_rows = []
    for category in pivot.columns:
        series = pivot[category].to_numpy(dtype=float)
        if len(series) < 30:
            continue
        horizon_rows.extend(_forecast_one_series(str(category), series))
    if not horizon_rows:
        return _blocked("CUMCM2023C_vegetable_retail_v36", "retail_forecast_operation", case_dir, "no forecast rows generated")
    metrics_df = pd.DataFrame(horizon_rows)
    routes: List[Dict[str, Any]] = []
    for route_name in ["mean_7day_baseline", "naive_last_day", "moving_average_14", "weekday_profile"]:
        sub = metrics_df[metrics_df["route"] == route_name]
        routes.append({
            "route": route_name,
            "metrics": {
                "MAE": round(float(sub["MAE"].mean()), 6),
                "WAPE": round(float(sub["abs_error"].sum() / max(sub["actual"].abs().sum(), 1e-9)), 6),
                "categories": int(sub["category"].nunique()),
            },
        })

    category_stats = daily.groupby(category_col).agg(quantity=("quantity", "sum"), revenue=("revenue", "sum")).reset_index()
    category_stats["avg_price"] = category_stats["revenue"] / category_stats["quantity"].replace(0, np.nan)
    category_stats["quantity_share"] = category_stats["quantity"] / category_stats["quantity"].sum()
    category_stats["gross_margin_proxy"] = category_stats["revenue"] * category_stats["quantity_share"]
    policy = category_stats.sort_values("gross_margin_proxy", ascending=False).head(3)
    routes.append({
        "route": "top_category_revenue_mix_policy",
        "metrics": {
            "max_margin_proxy": round(float(policy["gross_margin_proxy"].sum()), 6),
            "covered_quantity_share": round(float(policy["quantity_share"].sum()), 6),
            "top_categories": policy[category_col].astype(str).tolist(),
        },
    })

    forecast_routes = [route for route in routes if "MAE" in route["metrics"]]
    forecast_routes = sorted(forecast_routes, key=lambda route: route["metrics"]["WAPE"])
    policy_route = next(route for route in routes if route["route"] == "top_category_revenue_mix_policy")
    best = forecast_routes[0]
    composite_route = {
        "route": "forecast_then_revenue_mix_policy",
        "metrics": {
            "WAPE": best["metrics"]["WAPE"],
            "MAE": best["metrics"]["MAE"],
            "max_margin_proxy": policy_route["metrics"]["max_margin_proxy"],
            "covered_quantity_share": policy_route["metrics"]["covered_quantity_share"],
            "forecast_component": best["route"],
            "policy_component": policy_route["route"],
        },
    }
    routes = forecast_routes + [policy_route, composite_route]
    aggregate = {
        "n_sales_rows": int(len(sales_df)),
        "n_days": int(pivot.shape[0]),
        "n_categories": int(pivot.shape[1]),
        "best_forecast_route": best["route"],
        "best_WAPE": best["metrics"]["WAPE"],
        "best_MAE": best["metrics"]["MAE"],
        "top_policy_categories": policy_route["metrics"]["top_categories"],
    }
    return ResourceBackedResult(
        case="CUMCM2023C_vegetable_retail_forecast_operation_v36",
        c_topic_pattern="retail_forecast_operation",
        status="executed",
        data_path=str(case_dir),
        routes=routes,
        aggregate=aggregate,
        recommended_route="forecast_then_revenue_mix_policy",
        recommendation_evidence=[
            "真实读取 2023C 蔬菜单品信息与销售流水，按分类聚合日销量。",
            f"比较 7日均值 baseline、昨日值、14日移动平均、星期画像，最佳预测为 {best['route']}，WAPE={best['metrics']['WAPE']}。",
            f"经营策略基于销售额与销量占比 proxy，优先品类为 {policy_route['metrics']['top_categories']}。",
        ],
        figure_descriptions=[
            "分类日销量时间序列图：展示不同品类需求波动与季节/星期效应。",
            "预测路线 WAPE/MAE 对比图：说明为什么推荐某条预测路线。",
            "品类销售额-销量占比气泡图：展示经营策略选择依据。",
            "预测残差按星期箱线图：检查是否存在明显星期偏差。",
        ],
        limitations=[
            "该 adapter 尚未使用附件3的批发价/损耗率/补货约束，当前为预测与品类策略基线。",
            "策略指标是 revenue mix proxy，不等同于完整利润最大化模型。",
        ],
    )


def _judge_sample_scores(data_file: Path, sheet_name: str) -> pd.DataFrame:
    raw = pd.read_excel(data_file, sheet_name=sheet_name, header=None)
    data = raw.iloc[2:].copy()
    sample = pd.to_numeric(data.iloc[:, 0], errors="coerce")
    judge_scores = data.iloc[:, 3:13].apply(pd.to_numeric, errors="coerce")
    judge_scores.columns = [f"judge_{idx + 1}" for idx in range(judge_scores.shape[1])]
    working = pd.concat([sample.rename("sample_id"), judge_scores], axis=1).dropna(subset=["sample_id"])
    grouped = working.groupby("sample_id")[judge_scores.columns].sum().reset_index()
    grouped["sample_id"] = grouped["sample_id"].astype(int)
    grouped["mean_score"] = grouped[judge_scores.columns].mean(axis=1)
    grouped["std_score"] = grouped[judge_scores.columns].std(axis=1)
    return grouped


def _ranking_metrics(consensus: pd.Series, candidate: pd.Series | np.ndarray) -> Dict[str, Any]:
    consensus_arr = np.asarray(consensus, dtype=float)
    candidate_arr = np.asarray(candidate, dtype=float)
    spearman = spearmanr(consensus_arr, candidate_arr).correlation
    kendall = kendalltau(consensus_arr, candidate_arr).correlation
    top_k = min(5, len(consensus_arr))
    consensus_top = set(np.argsort(-consensus_arr)[:top_k])
    candidate_top = set(np.argsort(-candidate_arr)[:top_k])
    return {
        "spearman_to_consensus": round(float(spearman) if np.isfinite(spearman) else 0.0, 6),
        "kendall_to_consensus": round(float(kendall) if np.isfinite(kendall) else 0.0, 6),
        "top5_overlap": round(float(len(consensus_top & candidate_top) / top_k), 6),
    }


def _topsis_score(matrix: np.ndarray) -> np.ndarray:
    matrix = np.nan_to_num(matrix.astype(float), nan=0.0)
    norm = np.sqrt((matrix ** 2).sum(axis=0))
    norm[norm == 0] = 1.0
    normalized = matrix / norm
    positive = normalized.max(axis=0)
    negative = normalized.min(axis=0)
    d_pos = np.sqrt(((normalized - positive) ** 2).sum(axis=1))
    d_neg = np.sqrt(((normalized - negative) ** 2).sum(axis=1))
    return d_neg / np.maximum(d_pos + d_neg, 1e-12)


def _forecast_one_series(category: str, series: np.ndarray, test_window: int = 28) -> List[Dict[str, Any]]:
    rows = []
    start = max(14, len(series) - test_window)
    for idx in range(start, len(series)):
        actual = float(series[idx])
        history = series[:idx]
        if len(history) < 7:
            continue
        candidates = {
            "mean_7day_baseline": float(np.mean(history[-7:])),
            "naive_last_day": float(history[-1]),
            "moving_average_14": float(np.mean(history[-14:])),
        }
        weekday_values = history[np.arange(len(history)) % 7 == idx % 7]
        candidates["weekday_profile"] = float(np.mean(weekday_values[-4:])) if len(weekday_values) else candidates["mean_7day_baseline"]
        for route, pred in candidates.items():
            rows.append({"category": category, "route": route, "actual": actual, "prediction": pred, "MAE": abs(actual - pred), "abs_error": abs(actual - pred)})
    return rows


def _blocked(case: str, pattern: str, path: Path, reason: str) -> ResourceBackedResult:
    return ResourceBackedResult(case, pattern, "blocked", str(path), [], {"reason": reason}, "none", [], [], [reason])


def _find_file(roots: List[Path], file_name: str, preferred_keywords: List[str]) -> Path | None:
    hits = []
    for root in roots:
        if root.exists():
            hits.extend(path for path in root.rglob(file_name) if path.is_file())
    if not hits:
        return None
    return sorted(hits, key=lambda path: (-sum(key in str(path) for key in preferred_keywords), len(str(path))))[0]


def _find_dir_with_files(roots: List[Path], required_files: List[str], preferred_keywords: List[str]) -> Path | None:
    candidates = []
    for root in roots:
        if not root.exists():
            continue
        for first_file in root.rglob(required_files[0]):
            if not first_file.is_file():
                continue
            candidate = first_file.parent
            if all((candidate / file_name).exists() for file_name in required_files):
                candidates.append(candidate)
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: (-sum(key in str(path) for key in preferred_keywords), len(str(path))))[0]


def _resolve_roots(reference_roots: Iterable[Path | str] | None) -> List[Path]:
    if reference_roots is not None:
        return [Path(root) for root in reference_roots]
    return [Path(__file__).resolve().parent.parent / "data" / "reference_root"]


def _workflow_patch() -> List[str]:
    return [
        "有本地数据的类别才进入 executable adapter；没有资源的问卷、NLP、通用图结构暂不补模板，防止幻觉创新。",
        "评价排序类必须同时输出 baseline 排序、加权排序、TOPSIS/熵权类排序，并报告 Spearman/Kendall/TopK overlap。",
        "经营零售类必须把预测与决策分离：先用滚动窗口验证预测，再用利润或销售额 proxy 生成策略。",
        "如果复杂评价模型没有超过单组/均值 baseline，则只能作为消融或反例，不能作为主推创新。",
        "每个新增 adapter 必须提供可复查数据路径、指标、推荐理由、图表描述和限制边界。",
    ]


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent / "run" / "v36_resource_backed_c_adapters"
    suite = run_v36_resource_backed_c_adapters(output_dir=output_dir)
    print(format_v36_report(suite))
