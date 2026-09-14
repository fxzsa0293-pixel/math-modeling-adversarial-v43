from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

from solver.experiment_registry import choose_primary_metric, metric_lower_better
from solver.schemas import ExperimentRecord, ProblemProfile, RouteSpec


def export_workflow_results(
    profile: ProblemProfile,
    routes: List[RouteSpec],
    records: List[ExperimentRecord],
    final: Dict[str, Any],
    workspace_paths: Dict[str, str],
    adapter_dir: Path,
    experiments_dir: Path,
) -> Dict[str, str]:
    reports_dir = Path(workspace_paths["reports"])
    results_dir = Path(workspace_paths["work/question_01/results"])
    code_dir = Path(workspace_paths["work/question_01/code"])
    logs_dir = Path(workspace_paths["work/question_01/logs"])
    figures_dir = Path(workspace_paths["work/question_01/figures"])
    for directory in (reports_dir, results_dir, code_dir, logs_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    copied_code = _copy_tree_files(adapter_dir, code_dir, suffixes={".py"})
    copied_logs = _copy_tree_files(experiments_dir, logs_dir, suffixes={".json", ".txt", ".log", ".csv"})
    comparison_csv = reports_dir / "route_comparison.csv"
    _write_route_comparison(records, final, comparison_csv)
    metrics_csv = reports_dir / "route_metrics_long.csv"
    _write_metrics_long(records, metrics_csv)
    summary_json = reports_dir / "result_summary.json"
    result_summary = _result_summary(profile, routes, records, final, copied_code, copied_logs)
    summary_json.write_text(json.dumps(result_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_md = reports_dir / "result_summary.md"
    summary_md.write_text(_format_summary_md(result_summary), encoding="utf-8")
    figure_md = reports_dir / "figure_requirements.md"
    figure_md.write_text(_figure_requirements(profile, final), encoding="utf-8")
    comparison_figure = figures_dir / "route_comparison.svg"
    _write_route_figure(records, final, comparison_figure)
    paper_section = Path(workspace_paths["paper/sections"]) / "model_results.md"
    paper_section.write_text(_paper_section(profile, records, final), encoding="utf-8")
    stable_result_copy = results_dir / "result_summary.json"
    stable_result_copy.write_text(json.dumps(result_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    exports = {
        "route_comparison_csv": str(comparison_csv),
        "route_metrics_long_csv": str(metrics_csv),
        "result_summary_json": str(summary_json),
        "result_summary_md": str(summary_md),
        "figure_requirements_md": str(figure_md),
        "route_comparison_svg": str(comparison_figure),
        "paper_results_section_md": str(paper_section),
        "stable_result_copy": str(stable_result_copy),
    }
    recommended_id = (final.get("recommended_route") or {}).get("route_id") if isinstance(final.get("recommended_route"), dict) else None
    recommended_record = next((record for record in records if record.route_id == recommended_id), None)
    if recommended_record:
        for evidence_name, export_name, filename in (
            ("retail_policy", "recommended_retail_policy_csv", "recommended_retail_policy.csv"),
            ("retail_scenarios", "recommended_retail_scenarios_csv", "recommended_retail_scenarios.csv"),
            ("retail_category_service", "recommended_retail_category_service_csv", "recommended_retail_category_service.csv"),
        ):
            source = Path(recommended_record.evidence_paths.get(evidence_name, ""))
            if source.is_file():
                destination = results_dir / filename
                shutil.copy2(source, destination)
                exports[export_name] = str(destination)
    return exports


def _copy_tree_files(source: Path, target: Path, suffixes: set[str]) -> List[str]:
    copied: List[str] = []
    if not source.exists():
        return copied
    for path in source.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            relative = path.relative_to(source)
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            copied.append(str(destination))
    return copied


def _write_route_comparison(records: List[ExperimentRecord], final: Dict[str, Any], path: Path) -> None:
    metric = str(final.get("primary_metric") or choose_primary_metric(records))
    direction = str(final.get("metric_direction") or ("lower_better" if metric_lower_better(metric) else "higher_better"))
    rows = []
    score_lookup = {item.get("route_id"): item for item in final.get("route_scores", []) if isinstance(item, dict)}
    for record in records:
        score = score_lookup.get(record.route_id, {})
        rows.append({
            "route_id": record.route_id,
            "role": record.role,
            "status": record.status,
            "primary_metric": metric,
            "metric_direction": direction,
            "metric_value": record.metrics.get(metric, ""),
            "improvement_vs_baseline": score.get("improvement_vs_baseline", ""),
            "recommendation": record.recommendation,
            "data_path": record.data_path,
            "code_path": record.code_path,
            "final_code_hash": record.final_code_hash,
            "limitations": "; ".join(record.limitations),
            "error": record.error[:300],
        })
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["route_id"])
        writer.writeheader()
        writer.writerows(rows)


def _write_metrics_long(records: List[ExperimentRecord], path: Path) -> None:
    rows = []
    for record in records:
        for name, value in sorted(record.metrics.items()):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                rows.append({"route_id": record.route_id, "role": record.role, "status": record.status, "metric": name, "value": value})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["route_id", "role", "status", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def _write_route_figure(records: List[ExperimentRecord], final: Dict[str, Any], path: Path) -> None:
    metric = str(final.get("primary_metric") or choose_primary_metric(records))
    plotted = [(record.route_id, float(record.metrics[metric])) for record in records if record.status == "executed" and isinstance(record.metrics.get(metric), (int, float))]
    width = max(720, 170 * max(len(plotted), 1)); height = 440; baseline = 330; chart_height = 240
    recommended = (final.get("recommended_route") or {}).get("route_id") if isinstance(final.get("recommended_route"), dict) else None
    values = [value for _, value in plotted]; low = min(values) if values else 0.0; high = max(values) if values else 1.0
    span = max(high - min(0.0, low), 1e-12); bar_width = min(110, (width - 120) / max(len(plotted), 1) * 0.62)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="#ffffff"/>', f'<text x="60" y="38" font-family="Arial,sans-serif" font-size="20" fill="#202124">Route comparison: {_xml(metric)}</text>', f'<line x1="60" y1="{baseline}" x2="{width - 30}" y2="{baseline}" stroke="#555"/>']
    if not plotted:
        parts.append('<text x="60" y="200" font-family="Arial,sans-serif" font-size="16">No executed route with a numeric primary metric</text>')
    for index, (label, value) in enumerate(plotted):
        x = 80 + index * ((width - 120) / len(plotted)); bar_height = max(2.0, (value - min(0.0, low)) / span * chart_height); y = baseline - bar_height
        color = "#D55E00" if label == recommended else "#3972A4"
        parts.extend([f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" fill="{color}"/>', f'<text x="{x + bar_width / 2}" y="{y - 7}" text-anchor="middle" font-family="Arial,sans-serif" font-size="12">{value:.4g}</text>', f'<text x="{x + bar_width / 2}" y="{baseline + 22}" text-anchor="middle" font-family="Arial,sans-serif" font-size="11">{_xml(label[:22])}</text>'])
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def _xml(value: str) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _result_summary(
    profile: ProblemProfile,
    routes: List[RouteSpec],
    records: List[ExperimentRecord],
    final: Dict[str, Any],
    copied_code: List[str],
    copied_logs: List[str],
) -> Dict[str, Any]:
    executed = [record for record in records if record.status == "executed"]
    failed = [record for record in records if record.status != "executed"]
    return {
        "version": "v44",
        "problem_id": profile.problem_id,
        "archetypes": profile.archetypes,
        "target_candidates": profile.target_candidates,
        "metric_candidates": profile.metric_candidates,
        "verdict": final.get("verdict"),
        "primary_metric": final.get("primary_metric"),
        "metric_direction": final.get("metric_direction"),
        "recommended_route": final.get("recommended_route"),
        "flags": final.get("flags", []),
        "next_actions": final.get("next_actions", []),
        "gate_layers": final.get("gate_layers", {}),
        "claim_level": final.get("claim_level"),
        "executed_route_count": len(executed),
        "failed_route_count": len(failed),
        "route_count": len(routes),
        "copied_code_files": copied_code,
        "copied_log_files": copied_logs,
        "traceability": {
            "all_code_hashes_present": all(bool(record.final_code_hash) for record in records),
            "all_executed_records_have_data_path": all(bool(record.data_path) for record in executed),
            "baseline_present": any(record.role in {"baseline", "strong_baseline"} and record.status == "executed" for record in records),
            "all_executed_records_have_evidence": all(bool(record.evidence_paths) for record in executed),
            "single_run_id": len({record.run_id for record in records}) <= 1,
        },
    }


def _paper_section(profile: ProblemProfile, records: List[ExperimentRecord], final: Dict[str, Any]) -> str:
    recommended = final.get("recommended_route") or {}
    route_id = recommended.get("route_id") if isinstance(recommended, dict) else ""
    record = next((item for item in records if item.route_id == route_id), None)
    metric = str(final.get("primary_metric") or "")
    task = profile.contract.task_type if profile.contract else ""
    lines = [
        "# 模型求解与结果", "", "## 求解协议", "",
        f"本问题按 {task} 任务处理。所有候选路线使用题目契约中声明的数据、指标方向和验证协议；数值结果仅在机器门控通过后进入结论。",
        "", "## 路线比较", "", "| 路线 | 角色 | 状态 | 主指标 | 数值 |", "|---|---|---|---|---:|",
    ]
    for item in records:
        lines.append(f"| {item.route_id} | {item.role} | {item.status} | {metric} | {item.metrics.get(metric, '')} |")
    lines.extend(["", "## 推荐结果", ""])
    if final.get("verdict") == "pass" and record is not None:
        lines.append(f"最终推荐 {record.route_id}。其证据等级为 {final.get('claim_level')}，主指标 {metric} 的结果为 {record.metrics.get(metric)}。")
        numeric = [(key, value) for key, value in sorted(record.metrics.items()) if isinstance(value, (int, float)) and not isinstance(value, bool) and key != metric]
        if numeric:
            lines.extend(["", "推荐路线的其余可复算指标如下：", "", "| 指标 | 数值 |", "|---|---:|"])
            lines.extend(f"| {key} | {value} |" for key, value in numeric)
    else:
        lines.append("当前机器门控未通过，因此本节不形成数值性推荐结论。")
    lines.extend(["", "## 结论边界", ""])
    for flag in final.get("flags", []) or ["当前门控未报告额外风险标志，但仍应结合题意审查模型假设。"]:
        lines.append(f"- {flag}")
    lines.extend(["", "结果来源：reports/route_comparison.csv、reports/route_metrics_long.csv 及各路线 evidence 文件。", ""])
    return "\n".join(lines)


def _format_summary_md(summary: Dict[str, Any]) -> str:
    recommended = summary.get("recommended_route") or {}
    lines = [
        "# 结果汇总",
        "",
        f"- version: {summary['version']}",
        f"- problem_id: {summary['problem_id']}",
        f"- verdict: {summary['verdict']}",
        f"- primary_metric: {summary['primary_metric']}",
        f"- metric_direction: {summary['metric_direction']}",
        f"- executed_route_count: {summary['executed_route_count']}",
        f"- failed_route_count: {summary['failed_route_count']}",
        "",
        "## Recommended Route",
        f"- route: {recommended.get('route_id') if isinstance(recommended, dict) else recommended}",
        f"- role: {recommended.get('role') if isinstance(recommended, dict) else ''}",
        f"- metric_value: {recommended.get('value') if isinstance(recommended, dict) else ''}",
        f"- improvement_vs_baseline: {recommended.get('improvement_vs_baseline') if isinstance(recommended, dict) else ''}",
        "",
        "## Flags",
    ]
    for flag in summary.get("flags", []) or ["none"]:
        lines.append(f"- {flag}")
    lines.extend(["", "## Next Actions"])
    for action in summary.get("next_actions", []) or ["none"]:
        lines.append(f"- {action}")
    lines.extend(["", "## Traceability"])
    for key, value in summary.get("traceability", {}).items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines).rstrip() + "\n"


def _figure_requirements(profile: ProblemProfile, final: Dict[str, Any]) -> str:
    archetypes = set(profile.archetypes)
    lines = [
        "# Figure Requirements",
        "",
        "These are figure descriptions, not mandatory generated images. Use them only when they support the argument.",
        "",
        "## Core Evidence Figures",
        "- Route comparison bar/table figure: show the primary metric for baselines and candidate routes, with the recommended route highlighted.",
        "- Error or residual diagnostic figure: show whether the recommended route has systematic bias or unstable regions.",
        "- Data quality figure/table: summarize missingness, sample size, and key variable distributions from the data audit.",
    ]
    if "forecasting" in archetypes or "retail_forecast_operation" in archetypes:
        lines.extend([
            "- Time-order validation figure: plot observed versus predicted values on the held-out tail segment.",
            "- Feature/lag sensitivity figure: show how selected lag or window configurations affect validation error.",
        ])
    if "evaluation_ranking" in archetypes:
        lines.extend([
            "- Ranking stability figure: compare rankings under entropy/TOPSIS/PCA/stability-weighted routes.",
            "- Weight contribution figure: show indicator weights or principal component loadings.",
        ])
    if "constrained_optimization" in archetypes:
        lines.extend([
            "- Feasible solution comparison figure: show objective value and constraint violation for baseline and optimized policies.",
            "- Sensitivity figure: show objective changes under key parameter perturbations.",
        ])
    lines.extend([
        "",
        "## Current Recommendation Context",
        f"- recommended_route: {final.get('recommended_route')}",
        f"- primary_metric: {final.get('primary_metric')}",
        f"- metric_direction: {final.get('metric_direction')}",
    ])
    return "\n".join(lines).rstrip() + "\n"
