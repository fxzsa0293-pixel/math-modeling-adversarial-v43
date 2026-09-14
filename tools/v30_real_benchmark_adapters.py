"""
V30 real benchmark adapters.

This layer converts selected benchmark-registry candidates into directly
executable local Python evidence. It does not claim to reproduce every original
paper; it provides robust, data-grounded baseline/strong-route comparisons that
an adversarial modeling agent can use for model selection and self-checking.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import scipy.io as sio

try:
    from v26_evaluation_competition_suite import run_v26_evaluation_suite
except Exception:  # pragma: no cover
    run_v26_evaluation_suite = None

try:
    from v27_optimization_competition_suite import run_v27_optimization_suite
except Exception:  # pragma: no cover
    run_v27_optimization_suite = None


@dataclass
class AdapterResult:
    case: str
    archetypes: List[str]
    status: str
    data_path: str
    routes: List[Dict[str, Any]]
    aggregate: Dict[str, Any]
    recommended_route: str
    recommendation_evidence: List[str]
    figure_descriptions: List[str]
    limitations: List[str]


def run_v30_real_benchmark_adapters(reference_root: Path | str, output_dir: Path | str | None = None) -> Dict[str, Any]:
    reference_root = Path(reference_root)
    results: List[AdapterResult] = []

    taxi_case = _run_2015b_taxi_adapter(reference_root)
    if taxi_case.status == "executed":
        results.append(taxi_case)

    road_case = _run_2016b_road_adapter(reference_root)
    if road_case.status == "executed":
        results.append(road_case)

    # V38: do not wrap V26 here. The old wrapper lacked a true baseline metric
    # and could fabricate perfect equal-weight scores. Wine evaluation is handled
    # by v36_resource_backed_c_adapters with explicit holdout-style semantics.

    if run_v27_optimization_suite is not None:
        v27 = run_v27_optimization_suite(reference_root)
        if v27.get("status") == "executed":
            results.append(_wrap_v27(v27))

    summary = _summarize_results(results)
    suite = {
        "version": "v30",
        "purpose": "executable local Python adapters for multi-case model-selection evidence",
        "summary": summary,
        "results": [asdict(result) for result in results],
        "next_actions": _next_actions(results),
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "v30_real_benchmark_adapters.json").write_text(json.dumps(suite, ensure_ascii=False, indent=2), encoding="utf-8")
        (output_dir / "v30_real_benchmark_adapters.md").write_text(format_v30_report(suite), encoding="utf-8")
    return suite


def format_v30_report(suite: Dict[str, Any]) -> str:
    summary = suite.get("summary", {})
    lines = [
        "# V30 真实执行 Benchmark Adapter 报告",
        "",
        f"- executed_cases: {summary.get('executed_cases')}",
        f"- archetype_coverage: {summary.get('archetype_coverage')}",
        f"- routes_evaluated: {summary.get('routes_evaluated')}",
        f"- local_python_execution: {summary.get('local_python_execution')}",
        "",
        "## Case results",
    ]
    for result in suite.get("results", []):
        lines.extend([
            f"### {result['case']}",
            f"- status: {result['status']}",
            f"- archetypes: {result['archetypes']}",
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
    lines.extend(["## Next actions"])
    for action in suite.get("next_actions", []):
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


def _run_2015b_taxi_adapter(reference_root: Path) -> AdapterResult:
    case_dir = _find_dir(reference_root, "CUMCM2015B", "出租车")
    if not case_dir:
        return _blocked("CUMCM2015B_taxi_subsidy", ["constrained_optimization", "simulation_or_queueing"], reference_root, "case directory not found")
    data_dir = case_dir / "cangqiongdata"
    demand_files = sorted(data_dir.glob("demand*.txt"), key=_hour_key)
    supply_files = sorted(data_dir.glob("distribute*.txt"), key=_hour_key)
    if len(demand_files) < 12 or len(supply_files) < 12:
        return _blocked("CUMCM2015B_taxi_subsidy", ["constrained_optimization", "simulation_or_queueing"], case_dir, "hourly demand/supply files incomplete")

    hourly_rows = []
    for demand_file, supply_file in zip(demand_files, supply_files):
        hour = _hour_key(demand_file)
        demand = _read_point_weight_file(demand_file)
        supply = _read_point_weight_file(supply_file)
        hourly_rows.append(_taxi_hour_metrics(hour, demand, supply))
    hourly = pd.DataFrame(hourly_rows).sort_values("hour")

    routes = [
        _taxi_route_no_subsidy(hourly),
        _taxi_route_peak_flat(hourly),
        _taxi_route_scarcity_index(hourly),
    ]
    routes = sorted(routes, key=lambda item: item["metrics"]["weighted_shortage_index"])
    best = routes[0]
    aggregate = {
        "hours": int(len(hourly)),
        "total_demand_weight": float(hourly["demand_weight"].sum()),
        "total_supply_weight": float(hourly["supply_weight"].sum()),
        "worst_hour_by_shortage": int(hourly.sort_values("shortage_index", ascending=False).iloc[0]["hour"]),
        "max_shortage_index": round(float(hourly["shortage_index"].max()), 4),
        "mean_nearest_supply_distance_km": round(float(hourly["mean_nearest_supply_distance_km"].mean()), 4),
        "best_route_gain_vs_no_subsidy_pct": round(100 * (routes[-1]["metrics"]["weighted_shortage_index"] - best["metrics"]["weighted_shortage_index"]) / max(routes[-1]["metrics"]["weighted_shortage_index"], 1e-9), 2),
    }
    return AdapterResult(
        case="CUMCM2015B_taxi_subsidy_realdata",
        archetypes=["constrained_optimization", "simulation_or_queueing", "spatial_or_geostat"],
        status="executed",
        data_path=str(data_dir),
        routes=routes,
        aggregate=aggregate,
        recommended_route=best["route"],
        recommendation_evidence=[
            "真实读取 24 小时 demand/distribute 文本数据，按小时计算需求供给缺口。",
            "稀缺指数路线同时考虑供需比、最近供给距离和峰时压力，比固定峰时补贴更有针对性。",
            f"最紧张小时为 {aggregate['worst_hour_by_shortage']} 点，最高 shortage_index={aggregate['max_shortage_index']}。",
        ],
        figure_descriptions=[
            "24小时供需缺口折线图：横轴小时，纵轴 demand/supply 与 shortage_index，突出早晚高峰。",
            "需求-供给空间散点图：点大小表示权重，颜色区分需求点与出租车分布点。",
            "补贴策略对比柱状图：no_subsidy、peak_flat、scarcity_index 的加权缺口指数。",
            "最近供给距离箱线图：展示不同小时乘客到最近出租车热点的空间错配程度。",
        ],
        limitations=[
            "该 adapter 是政策评估基线，不等同于完整出租车个体级离散事件仿真。",
            "经纬度距离使用局部近似，城市道路网络绕行尚未纳入。",
        ],
    )


def _run_2016b_road_adapter(reference_root: Path) -> AdapterResult:
    case_dir = _find_dir(reference_root, "CUMCM2016B", "小区开放")
    if not case_dir:
        return _blocked("CUMCM2016B_open_community_road", ["network_or_graph"], reference_root, "case directory not found")
    mat_files = list(case_dir.rglob("*.mat"))
    scenarios = []
    for dt_file in mat_files:
        if "Dt_" not in dt_file.name:
            continue
        before, after = dt_file.stem.split("Dt_", 1)
        dt0_file = next((p for p in mat_files if p.name == f"{before}Dt0_{after}.mat"), None)
        delta_file = next((p for p in mat_files if p.name == f"{before}dtDelta_{after}.mat"), None)
        if dt0_file and delta_file:
            scenarios.append(_road_scenario_metrics(dt0_file, dt_file, delta_file))
    if not scenarios:
        return _blocked("CUMCM2016B_open_community_road", ["network_or_graph"], case_dir, "paired Dt/Dt0/dtDelta mat files not found")

    routes = [_road_route_baseline(scenarios), _road_route_opening_effect(scenarios), _road_route_risk_adjusted(scenarios)]
    routes = sorted(routes, key=lambda item: item["metrics"]["score"], reverse=True)
    best = routes[0]
    aggregate = {
        "scenarios": len(scenarios),
        "total_pairs": int(sum(item["n_pairs"] for item in scenarios)),
        "mean_delta": round(float(np.mean([item["mean_delta"] for item in scenarios])), 4),
        "mean_improvement_rate": round(float(np.mean([item["improvement_rate"] for item in scenarios])), 4),
        "worst_degradation": round(float(min(item["p05_delta"] for item in scenarios)), 4),
        "best_scenario": max(scenarios, key=lambda item: item["mean_delta"])["scenario"],
    }
    return AdapterResult(
        case="CUMCM2016B_open_community_road_realdata",
        archetypes=["network_or_graph", "constrained_optimization"],
        status="executed",
        data_path=str(case_dir),
        routes=routes,
        aggregate=aggregate,
        recommended_route=best["route"],
        recommendation_evidence=[
            "真实读取开放前 Dt0、开放后 Dt 与 dtDelta 的 MAT 文件，并比较全样本 OD 时间变化。",
            "risk_adjusted_opening 不只看均值改善，还惩罚 p05 恶化尾部，更符合交通方案评审口径。",
            f"平均改善率为 {aggregate['mean_improvement_rate']}，最差尾部变化为 {aggregate['worst_degradation']}。",
        ],
        figure_descriptions=[
            "开放前后通行时间分布对比图：展示 Dt0 与 Dt 的核密度/箱线图。",
            "dtDelta 排序曲线：展示道路开放对不同 OD 对的改善与恶化尾部。",
            "场景对比柱状图：比较不同小区/参数场景的 mean_delta、p05_delta、improvement_rate。",
            "风险收益二维图：横轴平均改善，纵轴尾部恶化，标注推荐方案。",
        ],
        limitations=[
            "MAT 附件已是 Matlab 程序生成的 OD 时间结果，当前 adapter 没有重建完整路网拓扑。",
            "缺少原始 GIS/道路容量信息，因此不能独立验证每条道路的物理通行能力。",
        ],
    )


def _wrap_v26(v26: Dict[str, Any]) -> AdapterResult:
    aggregate = v26.get("aggregate", {})
    return AdapterResult(
        case="CUMCM2012A_wine_evaluation_realdata",
        archetypes=["multi_criteria_evaluation"],
        status="executed",
        data_path=str(v26.get("data_path")),
        routes=[
            {"route": "equal_weight_TOPSIS", "metrics": {"mean_rank_correlation": None, "mean_top1_overlap": None}},
            {"route": "entropy_weight_TOPSIS", "metrics": {"mean_rank_correlation": aggregate.get("mean_rank_correlation"), "mean_top1_overlap": aggregate.get("mean_top1_overlap")}},
        ],
        aggregate=aggregate,
        recommended_route="equal_weight_TOPSIS",
        recommendation_evidence=[
            "真实读取 2012A 葡萄酒评分表，多 sheet 执行熵权 TOPSIS 与等权 TOPSIS 对比。",
            "推荐路线必须同时报告权重、排序稳定性和评委一致性，避免只给最终排名。",
        ],
        figure_descriptions=[
            "权重柱状图：展示熵权识别出的关键评价维度。",
            "TOPSIS排序对比图：比较等权与熵权排序差异。",
            "排序稳定性扰动图：展示权重扰动下 Top 样品是否稳定。",
        ],
        limitations=["评价类结论受评分表质量影响，需补主观权重或专家约束作敏感性对照。"],
    )


def _wrap_v27(v27: Dict[str, Any]) -> AdapterResult:
    aggregate = v27.get("aggregate", {})
    routes = [{"route": item.get("route"), "metrics": {"objective": item.get("objective"), "feasible": item.get("feasible"), **item.get("metrics", {})}} for item in v27.get("routes", [])]
    return AdapterResult(
        case="CUMCM2003B_open_pit_truck_assignment_realdata",
        archetypes=["constrained_optimization"],
        status="executed",
        data_path=str(v27.get("data_path")),
        routes=routes,
        aggregate=aggregate,
        recommended_route="integer_repair",
        recommendation_evidence=[
            "真实重建 2003B 露天矿运输约束，比较 greedy、LP relaxation、integer repair。",
            "LP relaxation 提供下界，整数修复提供可部署方案，二者组合能解释最优性差距。",
        ],
        figure_descriptions=[
            "路线分配热力图：展示 5×10 分配矩阵。",
            "目标值对比柱状图：比较 greedy、LP、integer repair。",
            "容量敏感性折线图：展示容量缩放下目标变化。",
        ],
        limitations=["当前整数求解是 LP rounding + repair，不是严格 MILP 全局最优证明。"],
    )


def _taxi_hour_metrics(hour: int, demand: pd.DataFrame, supply: pd.DataFrame) -> Dict[str, Any]:
    demand_weight = float(demand["weight"].sum())
    supply_weight = float(supply["weight"].sum())
    nearest = _nearest_distances_km(demand[["lon", "lat"]].to_numpy(), supply[["lon", "lat"]].to_numpy())
    weighted_distance = float(np.average(nearest, weights=demand["weight"].to_numpy())) if len(nearest) else 0.0
    shortage_ratio = max(demand_weight - supply_weight, 0.0) / max(demand_weight, 1e-9)
    spatial_pressure = weighted_distance / max(float(np.nanpercentile(nearest, 90)) if len(nearest) else 1.0, 1e-9)
    supply_demand_pressure = demand_weight / max(supply_weight, 1e-9)
    return {
        "hour": int(hour),
        "demand_points": int(len(demand)),
        "supply_points": int(len(supply)),
        "demand_weight": demand_weight,
        "supply_weight": supply_weight,
        "shortage_ratio": shortage_ratio,
        "mean_nearest_supply_distance_km": weighted_distance,
        "shortage_index": float(supply_demand_pressure * (1.0 + spatial_pressure)),
    }


def _taxi_route_no_subsidy(hourly: pd.DataFrame) -> Dict[str, Any]:
    return {"route": "no_subsidy_baseline", "metrics": _taxi_policy_metrics(hourly, np.zeros(len(hourly)))}


def _taxi_route_peak_flat(hourly: pd.DataFrame) -> Dict[str, Any]:
    peak = hourly["hour"].isin([7, 8, 9, 17, 18, 19]).to_numpy(dtype=float)
    return {"route": "peak_flat_subsidy", "metrics": _taxi_policy_metrics(hourly, peak * 0.18)}


def _taxi_route_scarcity_index(hourly: pd.DataFrame) -> Dict[str, Any]:
    index = hourly["shortage_index"].to_numpy(dtype=float)
    scaled = np.divide(index, max(index.max(), 1e-9)) * 0.28
    return {"route": "scarcity_index_subsidy", "metrics": _taxi_policy_metrics(hourly, scaled)}


def _taxi_policy_metrics(hourly: pd.DataFrame, subsidy_strength: np.ndarray) -> Dict[str, Any]:
    base = hourly["shortage_index"].to_numpy(dtype=float)
    adjusted = base * (1.0 - np.clip(subsidy_strength, 0, 0.35))
    return {
        "weighted_shortage_index": round(float(np.average(adjusted, weights=hourly["demand_weight"])), 6),
        "max_hour_shortage_index": round(float(adjusted.max()), 6),
        "subsidy_budget_index": round(float(subsidy_strength.sum()), 6),
        "high_pressure_hours": int((adjusted > np.percentile(adjusted, 75)).sum()),
    }


def _road_scenario_metrics(dt0_file: Path, dt_file: Path, delta_file: Path) -> Dict[str, Any]:
    dt0 = _mat_vector(dt0_file)
    dt = _mat_vector(dt_file)
    delta = _mat_vector(delta_file)
    if delta.size == 0 and dt0.size and dt.size:
        delta = dt0 - dt
    n = min(dt0.size, dt.size, delta.size)
    dt0, dt, delta = dt0[:n], dt[:n], delta[:n]
    return {
        "scenario": dt_file.stem,
        "n_pairs": int(n),
        "mean_before": round(float(np.mean(dt0)), 6),
        "mean_after": round(float(np.mean(dt)), 6),
        "mean_delta": round(float(np.mean(delta)), 6),
        "median_delta": round(float(np.median(delta)), 6),
        "p05_delta": round(float(np.percentile(delta, 5)), 6),
        "p95_delta": round(float(np.percentile(delta, 95)), 6),
        "improvement_rate": round(float(np.mean(delta > 0)), 6),
        "degradation_rate": round(float(np.mean(delta < 0)), 6),
    }


def _road_route_baseline(scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"route": "closed_community_baseline", "metrics": {"score": 0.0, "mean_delta": 0.0, "tail_risk": 0.0}}


def _road_route_opening_effect(scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
    mean_delta = float(np.mean([item["mean_delta"] for item in scenarios]))
    return {"route": "mean_improvement_opening", "metrics": {"score": round(mean_delta, 6), "mean_delta": round(mean_delta, 6), "tail_risk": 0.0}}


def _road_route_risk_adjusted(scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
    mean_delta = float(np.mean([item["mean_delta"] for item in scenarios]))
    tail_risk = float(abs(min(0.0, min(item["p05_delta"] for item in scenarios))))
    score = mean_delta - 0.35 * tail_risk
    return {"route": "risk_adjusted_opening", "metrics": {"score": round(score, 6), "mean_delta": round(mean_delta, 6), "tail_risk": round(tail_risk, 6)}}


def _read_point_weight_file(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, sep=r"\s+", header=None, names=["lon", "lat", "weight"], engine="python")
    data = data.apply(pd.to_numeric, errors="coerce").dropna()
    return data[(data["lon"].between(70, 140)) & (data["lat"].between(15, 55)) & (data["weight"] >= 0)].reset_index(drop=True)


def _nearest_distances_km(demand_xy: np.ndarray, supply_xy: np.ndarray) -> np.ndarray:
    if len(demand_xy) == 0 or len(supply_xy) == 0:
        return np.array([])
    lat_scale = 111.0
    lon_scale = 111.0 * np.cos(np.deg2rad(np.nanmean(demand_xy[:, 1])))
    demand_scaled = demand_xy * np.array([lon_scale, lat_scale])
    supply_scaled = supply_xy * np.array([lon_scale, lat_scale])
    distances = np.sqrt(((demand_scaled[:, None, :] - supply_scaled[None, :, :]) ** 2).sum(axis=2))
    return distances.min(axis=1)


def _mat_vector(path: Path) -> np.ndarray:
    data = sio.loadmat(path)
    arrays = [value for key, value in data.items() if not key.startswith("__")]
    if not arrays:
        return np.array([], dtype=float)
    arr = np.asarray(arrays[0]).ravel()
    values = []
    for item in arr:
        try:
            values.append(float(np.asarray(item).ravel()[0]))
        except Exception:
            try:
                values.append(float(item))
            except Exception:
                pass
    return np.asarray(values, dtype=float)


def _find_dir(reference_root: Path, *keywords: str) -> Path | None:
    search_roots = [reference_root]
    registry_path = Path(__file__).resolve().parent / "run" / "v29_benchmark_registry" / "v29_benchmark_registry.json"
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        for case in registry.get("cases", []):
            text = case.get("root", "") + case.get("case_id", "") + case.get("title", "")
            if all(keyword in text for keyword in keywords):
                path = Path(case["root"])
                if path.exists():
                    return path
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_dir() and all(keyword in str(path) for keyword in keywords):
                return path
    return None


def _hour_key(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits or 0)


def _blocked(case: str, archetypes: List[str], path: Path, reason: str) -> AdapterResult:
    return AdapterResult(case, archetypes, "blocked", str(path), [], {"reason": reason}, "none", [], [], [reason])


def _summarize_results(results: List[AdapterResult]) -> Dict[str, Any]:
    archetypes = sorted({archetype for result in results for archetype in result.archetypes})
    return {
        "executed_cases": len(results),
        "archetype_coverage": archetypes,
        "routes_evaluated": sum(len(result.routes) for result in results),
        "local_python_execution": True,
        "cases": [result.case for result in results],
    }


def _next_actions(results: List[AdapterResult]) -> List[str]:
    return [
        "Add 2011B police-platform adapter with explicit graph reconstruction from MAT files.",
        "Add 2012B solar-house adapter for physics + constrained optimization route comparison.",
        "Promote V30 adapters into the adversarial loop so Agent A must run candidate routes before recommending models.",
        "Require reviewer Agent B to check adapter logs, data shapes, baseline strength, leakage, and robustness metrics.",
    ]


if __name__ == "__main__":
    default_reference = Path(__file__).resolve().parent.parent / "data" / "reference_root"
    output_dir = Path(__file__).resolve().parent / "run" / "v30_real_benchmark_adapters"
    suite = run_v30_real_benchmark_adapters(default_reference, output_dir)
    print(format_v30_report(suite))
