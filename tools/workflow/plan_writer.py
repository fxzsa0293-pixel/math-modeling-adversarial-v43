from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from solver.schemas import ProblemProfile, RouteSpec


def write_solving_plan(profile: ProblemProfile, routes: List[RouteSpec], data_audit: Dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "求解计划.md"
    lines = [
        "# 求解计划",
        "",
        "## 1. Problem Profile",
        f"- problem_id: {profile.problem_id}",
        f"- brief_path: {profile.brief_path or 'not_provided'}",
        f"- data_root: {profile.data_root}",
        f"- inferred_archetypes: {profile.archetypes}",
        f"- target_candidates: {profile.target_candidates}",
        f"- metric_candidates: {profile.metric_candidates}",
        f"- contract_status: {profile.contract.status if profile.contract else 'missing'}",
        f"- contract_source: {profile.contract.source if profile.contract else 'missing'}",
        f"- bound_data_file: {profile.contract.data_file if profile.contract else 'missing'}",
        f"- bound_sheet: {profile.contract.sheet_name if profile.contract else 'missing'}",
        f"- bound_target: {profile.contract.target_column if profile.contract else 'missing'}",
        f"- validation_mode: {profile.contract.validation_mode if profile.contract else 'missing'}",
        "",
        "## 2. Data Inputs",
    ]
    readable = [report for report in data_audit.get("files", []) if report.get("readable")]
    for report in readable[:12]:
        lines.append(f"- `{Path(report['path']).name}`: rows={report.get('row_count')}, columns={report.get('column_count')}, path={report['path']}")
    if len(readable) > 12:
        lines.append(f"- additional_readable_files: {len(readable) - 12}")
    lines.extend([
        "",
        "## 3. Route Plan",
        "The route plan keeps explicit baselines, candidate methods, and unsupported ideas separated. Do not convert unsupported routes into fabricated results.",
    ])
    for route in routes:
        lines.append(f"### {route.route_id}")
        lines.append(f"- role: {route.role}")
        lines.append(f"- family: {route.family}")
        lines.append(f"- supported: {route.supported}")
        lines.append(f"- expected_metrics: {route.expected_metrics}")
        lines.append(f"- rationale: {route.rationale}")
        lines.append(f"- fallback: if this route fails or underperforms, keep it as ablation evidence and prefer a stronger executed route.")
        lines.append("")
    lines.extend([
        "## 4. Output Contract",
        "- executable adapters: `generated_adapters/` and `competition_workspace/work/question_01/code/`",
        "- experiment logs: `experiments/` and `competition_workspace/work/question_01/logs/`",
        "- numerical registry: `registry/experiment_registry.json`",
        "- route comparison: `competition_workspace/reports/route_comparison.csv`",
        "- result summary: `competition_workspace/reports/result_summary.json` and `result_summary.md`",
        "- figure requirements: `competition_workspace/reports/figure_requirements.md`",
        "",
        "## 5. Review Rules",
        "- Candidate routes are compared with a named baseline under one validation protocol; retain the baseline when no candidate has reliable gain.",
        "- Time-series or ordered data should use time-order validation unless the adapter states a defensible reason.",
        "- Paper claims should cite result summary fields, route comparison rows, or data audit entries.",
    ])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path
