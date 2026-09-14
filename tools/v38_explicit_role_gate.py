"""
V38 explicit-role evidence gate.

Unlike V37, this gate does not infer baseline/candidate roles from route names.
The caller must provide a route_contract mapping: case -> route -> role.
Missing role is a hard failure.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

PRIMARY_METRIC_BY_PATTERN = {
    "forecasting": ["MAE", "RMSE", "WAPE"],
    "constrained_optimization": ["objective", "weighted_shortage_index", "score"],
    "multi_criteria_evaluation": ["mean_rank_correlation", "mean_top1_overlap"],
    "simulation_or_queueing": ["weighted_shortage_index", "max_hour_shortage_index", "p95_wait"],
    "network_or_graph": ["score", "mean_delta", "tail_risk"],
    "spatial_or_geostat": ["annual_energy_index", "mean_nearest_supply_distance_km", "mean_Moran_I", "WAPE"],
    "mechanism_or_physics": ["MAE", "safety_margin_to_44C", "objective", "score"],
    "medical_environment_statistics": ["MAE", "RMSE", "R2"],
    "geometric_deformation": ["centerline_MAE", "mean_linearity_ratio"],
    "business_forecast_optimization": ["MAE", "RMSE", "max_margin_proxy"],
    "evaluation_ranking": ["spearman_to_consensus", "kendall_to_consensus", "top5_overlap"],
    "retail_forecast_operation": ["WAPE", "MAE", "max_margin_proxy"],
}

LOWER_BETTER = {
    "MAE", "RMSE", "WAPE", "MAPE_proxy", "weighted_shortage_index",
    "max_hour_shortage_index", "tail_risk", "centerline_MAE", "objective", "p95_wait",
    "mean_nearest_supply_distance_km", "constraint_violation", "runtime_seconds",
}


def format_v38_report(result: Dict[str, Any]) -> str:
    lines = [
        "# V38 显式角色证据门报告",
        "",
        f"- gate: {result.get('gate')}",
        f"- surviving_route_count: {result.get('surviving_route_count')}",
        f"- eliminated_route_count: {result.get('eliminated_route_count')}",
        "",
        "## Strict Contract",
    ]
    for item in result.get("strict_contract", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Case Audits")
    for audit in result.get("case_audits", []):
        lines.extend([
            f"### {audit['case']}",
            f"- status: {audit['status']}",
            f"- patterns: {audit['patterns']}",
            f"- data_path: {audit['data_path']}",
            f"- recommended_route: {audit['recommended_route']}",
            f"- route_ids: {audit['route_ids']}",
            f"- baseline_routes: {audit['baseline_routes']}",
            f"- audit_flags: {audit['audit_flags']}",
            "- surviving_routes:",
        ])
        for route in audit.get("surviving_routes", []):
            lines.append(f"  - {route['route']} ({route['role']}): {route['recommendation']} | {route['metrics']}")
        lines.append("- eliminated_routes:")
        for route in audit.get("eliminated_routes", []):
            lines.append(f"  - {route['route']} ({route['role']}): {route['reasons']} | {route['metrics']}")
        lines.append("")
    return "\n".join(lines) + "\n"

ALLOWED_ROLES = {"baseline", "strong_baseline", "candidate", "composite", "ablation", "negative_control"}
PROXY_METRICS = {"max_margin_proxy", "gross_margin_proxy", "revenue_mix_proxy"}


@dataclass
class V38RouteAudit:
    case: str
    route: str
    role: str
    status: str
    reasons: List[str]
    metrics: Dict[str, Any]
    comparisons: List[Dict[str, Any]]
    recommendation: str


@dataclass
class V38CaseAudit:
    case: str
    status: str
    data_path: str
    patterns: List[str]
    recommended_route: str
    routes_total: int
    routes_survived: int
    route_ids: List[str]
    baseline_routes: List[str]
    eliminated_routes: List[V38RouteAudit]
    surviving_routes: List[V38RouteAudit]
    audit_flags: List[str]


def run_v38_explicit_role_gate(
    suites: Iterable[Dict[str, Any]],
    route_contract: Mapping[str, Mapping[str, str]],
    output_dir: Path | str | None = None,
    label: str = "v38-explicit-role",
) -> Dict[str, Any]:
    case_results = []
    for suite in suites:
        case_results.extend([result for result in suite.get("results", []) if result.get("status") == "executed"])
    case_audits = [_audit_case(result, route_contract.get(result.get("case", ""), {})) for result in case_results]
    gate = _gate_decision(case_audits)
    result = {
        "version": "v38",
        "label": label,
        "purpose": "explicit-role strict evidence gate for self-contained modeling-agent reproduction",
        "gate": gate,
        "case_audits": [_case_to_dict(audit) for audit in case_audits],
        "surviving_route_count": sum(audit.routes_survived for audit in case_audits),
        "eliminated_route_count": sum(len(audit.eliminated_routes) for audit in case_audits),
        "strict_contract": _strict_contract(gate),
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "v38_explicit_role_gate.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        (output_dir / "v38_explicit_role_gate.md").write_text(format_v38_report(result), encoding="utf-8")
    return result


def _audit_case(case_result: Dict[str, Any], contract: Mapping[str, str]) -> V38CaseAudit:
    case = case_result.get("case", "unknown_case")
    raw_routes = case_result.get("routes", [])
    routes = [_with_contract_role(route, contract) for route in raw_routes]
    patterns = _patterns(case_result)
    data_path = str(case_result.get("data_path", ""))
    recommended_route = str(case_result.get("recommended_route", ""))
    route_ids = [route.get("route", "") for route in routes]
    baseline_routes = [route.get("route", "") for route in routes if route.get("role") in {"baseline", "strong_baseline"}]
    primary_metrics = _primary_metrics(patterns)

    flags: List[str] = []
    if not data_path or data_path == "None":
        flags.append("missing_data_path")
    elif not Path(data_path).exists():
        flags.append("data_path_not_found")
    if not case_result.get("limitations"):
        flags.append("missing_limitations")
    if len(routes) < 2:
        flags.append("not_enough_route_comparison")
    if not baseline_routes:
        flags.append("no_explicit_baseline")
    if not recommended_route:
        flags.append("missing_recommended_route")
    elif recommended_route not in route_ids:
        flags.append("recommended_route_not_in_routes")
    missing_roles = [route.get("route", "") for route in routes if not route.get("role")]
    invalid_roles = [route.get("route", "") for route in routes if route.get("role") and route.get("role") not in ALLOWED_ROLES]
    if missing_roles:
        flags.append("missing_route_role")
    if invalid_roles:
        flags.append("invalid_route_role")

    route_audits = [_audit_route(case, route, routes, primary_metrics, recommended_route) for route in routes]
    route_hard_reasons = {"missing_route_role", "invalid_route_role", "missing_primary_metric"}
    if any(reason in route_hard_reasons for audit in route_audits for reason in audit.reasons):
        flags.append("route_hard_evidence_failure")
    surviving = [audit for audit in route_audits if audit.status == "survived"]
    eliminated = [audit for audit in route_audits if audit.status == "eliminated"]
    status = "passed" if not _hard_flags(flags) and surviving else "failed"
    return V38CaseAudit(case, status, data_path, patterns, recommended_route, len(routes), len(surviving), route_ids, baseline_routes, eliminated, surviving, flags)


def _with_contract_role(route: Dict[str, Any], contract: Mapping[str, str]) -> Dict[str, Any]:
    copied = dict(route)
    route_id = str(copied.get("route", ""))
    if route_id in contract:
        copied["role"] = contract[route_id]
    else:
        copied.pop("role", None)
    return copied


def _audit_route(case: str, route: Dict[str, Any], routes: List[Dict[str, Any]], primary_metrics: List[str], recommended_route: str) -> V38RouteAudit:
    route_name = route.get("route", "unknown_route")
    role = route.get("role", "")
    metrics = route.get("metrics", {}) or {}
    reasons: List[str] = []
    comparisons: List[Dict[str, Any]] = []

    if not role:
        reasons.append("missing_route_role")
    elif role not in ALLOWED_ROLES:
        reasons.append("invalid_route_role")

    usable_metrics = [metric for metric in primary_metrics if metric not in PROXY_METRICS]
    primary_metric_required = role in {"baseline", "strong_baseline", "candidate", "composite"} or route_name == recommended_route
    if primary_metric_required and not any(metric in metrics and isinstance(metrics.get(metric), (int, float)) for metric in usable_metrics):
        reasons.append("missing_primary_metric")

    if any(metric in PROXY_METRICS for metric in metrics):
        reasons.append("proxy_metric_auxiliary_only")

    baseline = _best_baseline(routes, usable_metrics)
    if baseline is not None and role not in {"baseline", "strong_baseline"}:
        comparisons = _compare_all_metrics(route, baseline, usable_metrics)
        if comparisons and all(item.get("worse_than_baseline") for item in comparisons):
            reasons.append(f"dominated_by_baseline:{baseline.get('route')}")
        elif any(item.get("worse_than_baseline") for item in comparisons):
            reasons.append(f"tradeoff_vs_baseline:{baseline.get('route')}")

    hard_route_reasons = {"missing_route_role", "invalid_route_role", "missing_primary_metric"}
    if role in {"baseline", "strong_baseline"} and any(reason in hard_route_reasons for reason in reasons):
        status = "eliminated"
        recommendation = "baseline_invalid"
    elif any(reason.startswith("dominated_by_baseline") or reason in hard_route_reasons for reason in reasons) and role not in {"baseline", "strong_baseline"}:
        status = "eliminated"
        recommendation = "eliminate_or_keep_only_as_ablation"
    else:
        status = "survived"
        recommendation = "keep_as_required_baseline" if role in {"baseline", "strong_baseline"} else "keep_as_candidate"
    if route_name == recommended_route and status == "survived":
        recommendation = "keep_as_recommended_route"
    elif route_name == recommended_route and status == "eliminated":
        recommendation = "recommended_route_rejected_by_v38_gate"
    return V38RouteAudit(case, route_name, role, status, reasons, metrics, comparisons, recommendation)


def _patterns(case_result: Dict[str, Any]) -> List[str]:
    patterns = case_result.get("archetypes") or []
    if not patterns and case_result.get("c_topic_pattern"):
        patterns = [case_result["c_topic_pattern"]]
    return list(patterns)


def _primary_metrics(patterns: Iterable[str]) -> List[str]:
    metrics: List[str] = []
    for pattern in patterns:
        metrics.extend(PRIMARY_METRIC_BY_PATTERN.get(pattern, []))
    return list(dict.fromkeys(metrics)) or ["MAE", "RMSE", "WAPE", "objective", "score"]


def _metric_lower_better(metric: str) -> bool:
    return metric in LOWER_BETTER


def _best_baseline(routes: List[Dict[str, Any]], primary_metrics: List[str]) -> Dict[str, Any] | None:
    baselines = [route for route in routes if route.get("role") in {"baseline", "strong_baseline"}]
    if not baselines:
        return None
    comparable = [metric for metric in primary_metrics if any(isinstance(route.get("metrics", {}).get(metric), (int, float)) for route in baselines)]
    if not comparable:
        return None
    metric = comparable[0]
    lower = _metric_lower_better(metric)
    return sorted(baselines, key=lambda route: route.get("metrics", {}).get(metric, float("inf") if lower else -float("inf")), reverse=not lower)[0]


def _compare_all_metrics(route: Dict[str, Any], baseline: Dict[str, Any], primary_metrics: List[str]) -> List[Dict[str, Any]]:
    out = []
    for metric in primary_metrics:
        route_value = route.get("metrics", {}).get(metric)
        baseline_value = baseline.get("metrics", {}).get(metric)
        if not isinstance(route_value, (int, float)) or not isinstance(baseline_value, (int, float)):
            continue
        lower = _metric_lower_better(metric)
        worse = route_value > baseline_value if lower else route_value < baseline_value
        better = route_value < baseline_value if lower else route_value > baseline_value
        out.append({
            "metric": metric,
            "direction": "lower_better" if lower else "higher_better",
            "route_value": route_value,
            "baseline_value": baseline_value,
            "worse_than_baseline": bool(worse),
            "better_than_baseline": bool(better),
        })
    return out


def _hard_flags(flags: List[str]) -> List[str]:
    hard = {
        "missing_data_path",
        "data_path_not_found",
        "missing_limitations",
        "not_enough_route_comparison",
        "no_explicit_baseline",
        "missing_recommended_route",
        "recommended_route_not_in_routes",
        "missing_route_role",
        "invalid_route_role",
        "route_hard_evidence_failure",
    }
    return [flag for flag in flags if flag in hard]


def _gate_decision(case_audits: List[V38CaseAudit]) -> Dict[str, Any]:
    critical_flags = sorted(set(flag for audit in case_audits for flag in _hard_flags(audit.audit_flags)))
    weak_cases = [audit.case for audit in case_audits if audit.status != "passed"]
    rejected = [route.case for audit in case_audits for route in audit.eliminated_routes if route.route == audit.recommended_route]
    if rejected:
        critical_flags.append("recommended_route_rejected")
    verdict = "pass" if not critical_flags and not weak_cases else "revise"
    return {
        "verdict": verdict,
        "critical_flags": sorted(set(critical_flags)),
        "weak_cases": weak_cases,
        "rejected_recommended_cases": rejected,
        "minimum_contract_satisfied": verdict == "pass",
        "rationale": "explicit role/data/baseline contract satisfied" if verdict == "pass" else "explicit role evidence contract failed",
    }


def _strict_contract(gate: Dict[str, Any]) -> List[str]:
    status = "已满足" if gate.get("minimum_contract_satisfied") else "未满足"
    return [
        f"V38 显式角色证据门状态：{status}。",
        "route role 必须由 route_contract 显式提供，禁止依赖路线名称猜测 baseline。",
        "主指标缺失不得用默认值填充，完整路线缺失主指标即 hard fail。",
        "proxy 指标只能作为辅助说明，不能作为主门槛指标。",
        "recommended_route 必须存在且不免检。"
    ]


def _case_to_dict(audit: V38CaseAudit) -> Dict[str, Any]:
    return {
        "case": audit.case,
        "status": audit.status,
        "data_path": audit.data_path,
        "patterns": audit.patterns,
        "recommended_route": audit.recommended_route,
        "routes_total": audit.routes_total,
        "routes_survived": audit.routes_survived,
        "route_ids": audit.route_ids,
        "baseline_routes": audit.baseline_routes,
        "eliminated_routes": [asdict(route) for route in audit.eliminated_routes],
        "surviving_routes": [asdict(route) for route in audit.surviving_routes],
        "audit_flags": audit.audit_flags,
    }
