from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from .experiment_registry import best_baseline, choose_primary_metric, metric_lower_better
from .schemas import ExperimentRecord, ProblemProfile, RouteSpec

DEFAULT_MIN_IMPROVEMENT = 0.01


def judge_solution(profile: ProblemProfile, routes: List[RouteSpec], records: List[ExperimentRecord], output_dir: Path, min_improvement: float = DEFAULT_MIN_IMPROVEMENT, domain_verification: Dict[str, object] | None = None) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    executed = [record for record in records if record.status == "executed"]
    failed = [record for record in records if record.status != "executed" and record.role not in {"ablation", "diagnostic"}]
    if profile.contract and profile.contract.task_type == "constrained_optimization" and profile.contract.multiobjective_model:
        metric = "pareto_count"
    else:
        metric = choose_primary_metric(executed, profile.metric_candidates)
    directions = profile.metric_directions or {}
    baseline = best_baseline(executed, metric, directions)
    route_scores = _route_scores(executed, metric, directions, baseline)
    recommended = _recommend(profile, route_scores, baseline, min_improvement)
    run_ids = sorted({record.run_id for record in executed if record.run_id})
    input_bindings = sorted(
        ({"path": record.data_path, "sha256": record.input_hash} for record in executed if record.data_path),
        key=lambda item: item["path"],
    )
    code_bindings = sorted(
        ({"path": record.code_path, "sha256": record.final_code_hash} for record in executed if record.code_path),
        key=lambda item: item["path"],
    )
    contract_hash = hashlib.sha256(json.dumps(asdict(profile.contract) if profile.contract else None, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    contract_status = profile.contract.status if profile.contract else "missing"
    evidence_issues = _evidence_issues(profile, executed)
    evidence_issues.extend(_assembly_issues(profile))
    evidence_issues.extend(_retail_manifest_issues(profile))
    evidence_issues.extend(_mechanism_manifest_issues(profile))
    route_issues = _route_binding_issues(routes, executed)
    validation_issues = _validation_issues(profile, executed)
    domain_issues = _domain_issues(profile, executed, domain_verification, contract_hash)
    retail_stress_flag = None
    if profile.contract and profile.contract.retail_decision_model and domain_verification:
        comparison = domain_verification.get("policy_comparison") or {}
        if recommended and recommended.get("role") not in {"baseline", "strong_baseline"} and comparison.get("candidate_robustly_better") is not True:
            recommended = next((item for item in route_scores if item.get("role") in {"baseline", "strong_baseline"}), recommended)
            retail_stress_flag = "retail_candidate_not_stress_robust"
    flags: List[str] = []
    if contract_status != "ready":
        flags.append(f"contract_{contract_status}")
    if not executed:
        flags.append("no_executed_route")
    if baseline is None:
        flags.append("no_executed_baseline")
    if failed:
        flags.append("has_failed_routes")
    flags.extend(evidence_issues + route_issues + validation_issues + domain_issues)
    if retail_stress_flag:
        flags.append(retail_stress_flag)

    supported_routes = [route for route in routes if route.supported]
    if contract_status in {"needs_input", "invalid", "missing"}:
        verdict = "blocked"
    elif (routes and not supported_routes) or (not executed and failed and all("unsupported_" in (record.error or "") for record in failed)):
        verdict = "unsupported"
    elif not executed or baseline is None or evidence_issues or route_issues or validation_issues or domain_issues:
        verdict = "revise"
    else:
        verdict = "pass"

    result = {
        "version": "v44",
        "problem_id": profile.problem_id,
        "run_id": run_ids[0] if len(run_ids) == 1 else None,
        "contract_hash": contract_hash,
        "input_bindings": list({item["path"]: item for item in input_bindings}.values()),
        "code_bindings": list({item["path"]: item for item in code_bindings}.values()),
        "verdict": verdict,
        "primary_metric": metric,
        "metric_direction": directions.get(metric, "lower_better" if metric_lower_better(metric, directions) else "higher_better"),
        "minimum_improvement": min_improvement,
        "recommended_route": recommended,
        "route_scores": route_scores,
        "failed_routes": [record.__dict__ for record in failed],
        "domain_verification": domain_verification,
        "flags": list(dict.fromkeys(flags)),
        "gate_layers": {
            "contract": {"status": contract_status, "pass": contract_status == "ready"},
            "execution": {"pass": bool(executed), "executed": len(executed), "failed": len(failed)},
            "evidence": {"pass": bool(executed) and not evidence_issues and not route_issues, "issues": evidence_issues + route_issues},
            "validation": {"pass": bool(executed) and not validation_issues, "issues": validation_issues},
            "domain_constraints": {"pass": bool(executed) and not domain_issues, "issues": domain_issues},
        },
        "claim_level": _claim_level(verdict, profile, domain_verification),
        "next_actions": _next_actions(flags, profile),
    }
    (output_dir / "final_judge.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "final_judge.md").write_text(_format_md(result), encoding="utf-8")
    return result


def _route_scores(executed, metric, directions, baseline):
    items = []
    for record in executed:
        selection_metric = f"selection_{metric}" if metric in {"MAE", "RMSE", "WAPE", "R2"} else metric
        value = record.metrics.get(selection_metric, record.metrics.get(metric))
        if not isinstance(value, (int, float)):
            continue
        improvement = None
        baseline_value = baseline.metrics.get(selection_metric, baseline.metrics.get(metric)) if baseline else None
        if baseline and record.route_id != baseline.route_id and isinstance(baseline_value, (int, float)):
            base_value = baseline_value
            improvement = ((base_value - value) if metric_lower_better(metric, directions) else (value - base_value)) / abs(base_value or 1.0)
        final_value = record.metrics.get(f"final_test_{metric}")
        final_baseline = baseline.metrics.get(f"final_test_{metric}") if baseline else None
        final_improvement = None
        if record.route_id != getattr(baseline, "route_id", None) and isinstance(final_value, (int, float)) and isinstance(final_baseline, (int, float)):
            final_improvement = ((final_baseline - final_value) if metric_lower_better(metric, directions) else (final_value - final_baseline)) / abs(final_baseline or 1.0)
        items.append({"route_id": record.route_id, "role": record.role, "metric": metric, "value": value, "selection_value": value, "final_test_value": final_value, "improvement_vs_baseline": None if improvement is None else round(float(improvement), 6), "final_test_improvement_vs_baseline": None if final_improvement is None else round(float(final_improvement), 6), "data_path": record.data_path, "limitations": record.limitations})
    return sorted(items, key=lambda item: item["value"], reverse=not metric_lower_better(metric, directions))


def _recommend(profile, scores, baseline, min_improvement):
    if not scores:
        return None
    if profile.contract and profile.contract.task_type == "evaluation_ranking":
        return scores[0]
    candidates = [item for item in scores if item["role"] not in {"baseline", "strong_baseline", "negative_control", "ablation", "diagnostic"}]
    selection_winner = next((item for item in candidates if (item.get("improvement_vs_baseline") or 0) >= min_improvement), None)
    if selection_winner and (selection_winner.get("final_test_improvement_vs_baseline") is None or selection_winner.get("final_test_improvement_vs_baseline") >= 0):
        return selection_winner
    if baseline:
        return next((item for item in scores if item["route_id"] == baseline.route_id), scores[0])
    return scores[0]


def _evidence_issues(profile: ProblemProfile, records: List[ExperimentRecord]) -> List[str]:
    issues = []
    expected_contract = asdict(profile.contract) if profile.contract else None
    run_ids = {record.run_id for record in records if record.run_id}
    if len(run_ids) > 1:
        issues.append("mixed_run_ids")
    for record in records:
        if not record.run_id or not record.input_hash or not record.config_hash:
            issues.append(f"incomplete_provenance:{record.route_id}")
        if not record.data_path or not Path(record.data_path).is_file() or _sha256(Path(record.data_path)) != record.input_hash:
            issues.append(f"input_hash_mismatch:{record.route_id}")
        if not record.code_path or not Path(record.code_path).is_file() or _sha256(Path(record.code_path)) != record.final_code_hash:
            issues.append(f"code_hash_mismatch:{record.route_id}")
        if "route_evidence" not in record.evidence_paths:
            issues.append(f"missing_route_evidence:{record.route_id}")
        for name, raw_path in record.evidence_paths.items():
            path = Path(raw_path)
            expected = record.artifact_hashes.get(name)
            if not path.is_file() or not expected or _sha256(path) != expected:
                issues.append(f"invalid_evidence:{record.route_id}:{name}")
            elif name == "route_evidence":
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    configuration = {"contract": payload.get("contract", {}), "route": payload.get("route", {})}
                    actual_config = hashlib.sha256(json.dumps(configuration, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
                    identity_ok = (
                        payload.get("problem_id") == profile.problem_id == record.problem_id
                        and payload.get("route_id") == record.route_id
                        and payload.get("data_path") == record.data_path
                    )
                    metrics_ok = payload.get("metrics_recomputed") == record.metrics
                    contract_ok = payload.get("contract") == expected_contract
                    if actual_config != record.config_hash or payload.get("run_id") != record.run_id or not identity_ok or not metrics_ok or not contract_ok:
                        issues.append(f"evidence_provenance_mismatch:{record.route_id}")
                except Exception:
                    issues.append(f"invalid_evidence_schema:{record.route_id}")
        if not record.evidence_paths:
            issues.append(f"missing_evidence:{record.route_id}")
    return issues


def _assembly_issues(profile: ProblemProfile) -> List[str]:
    contract = profile.contract
    if not contract or not contract.data_sources:
        return []
    path = Path(contract.data_assembly_manifest)
    if not path.is_file():
        return ["data_assembly_manifest_missing"]
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ["data_assembly_manifest_invalid"]
    issues = []
    output_path = Path(str(manifest.get("output_path", "")))
    if not output_path.is_file() or str(output_path.resolve()) != str(Path(contract.data_file).resolve()) or _sha256(output_path) != manifest.get("output_sha256"):
        issues.append("data_assembly_output_binding_mismatch")
    declared_sources = {str(item.get("alias")): str(Path(item.get("path", "")).resolve()) for item in contract.data_sources if isinstance(item, dict)}
    manifest_sources = {str(item.get("alias")): item for item in manifest.get("sources", []) if isinstance(item, dict)}
    if set(declared_sources) != set(manifest_sources):
        issues.append("data_assembly_source_set_mismatch")
    for alias, source_path in declared_sources.items():
        source = manifest_sources.get(alias, {})
        path = Path(source_path)
        if not path.is_file() or str(path.resolve()) != str(Path(str(source.get("path", ""))).resolve()) or _sha256(path) != source.get("sha256"):
            issues.append(f"data_assembly_source_hash_mismatch:{alias}")
    manifest_aggregation = manifest.get("aggregation")
    aggregation_matches = (not contract.data_aggregation and manifest_aggregation is None) or (
        bool(contract.data_aggregation)
        and isinstance(manifest_aggregation, dict)
        and manifest_aggregation.get("group_by") == contract.data_aggregation.get("group_by")
        and manifest_aggregation.get("aggregations") == contract.data_aggregation.get("aggregations")
        and manifest_aggregation.get("sort_by") == contract.data_aggregation.get("sort_by", [])
    )
    if manifest.get("base_table") != contract.base_table or len(manifest.get("joins", [])) != len(contract.data_joins) or not aggregation_matches:
        issues.append("data_assembly_contract_mismatch")
    return issues


def _retail_manifest_issues(profile: ProblemProfile) -> List[str]:
    contract = profile.contract
    if not contract or not contract.retail_decision_model or not contract.retail_decision_manifest:
        return []
    path = Path(contract.retail_decision_manifest)
    if not path.is_file():
        return ["retail_decision_manifest_missing"]
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ["retail_decision_manifest_invalid"]
    issues = []
    model_hash = hashlib.sha256(json.dumps(contract.retail_decision_model, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    if manifest.get("model_sha256") != model_hash:
        issues.append("retail_decision_model_binding_mismatch")
    sources = manifest.get("sources", [])
    if not isinstance(sources, list) or not sources:
        issues.append("retail_decision_sources_missing")
        sources = []
    for source in sources:
        source_path = Path(str(source.get("path", "")))
        if not source_path.is_file() or _sha256(source_path) != source.get("sha256"):
            issues.append(f"retail_decision_source_hash_mismatch:{source.get('alias', '')}")
    derived_path = Path(str(manifest.get("derived_input_path", "")))
    if not derived_path.is_file() or derived_path.resolve() != Path(contract.data_file).resolve() or _sha256(derived_path) != manifest.get("derived_input_sha256"):
        issues.append("retail_decision_derived_input_binding_mismatch")
    if not manifest.get("training_cutoff") or not manifest.get("decision_dates"):
        issues.append("retail_decision_time_boundary_missing")
    return issues


def _mechanism_manifest_issues(profile: ProblemProfile) -> List[str]:
    contract = profile.contract
    if not contract or not contract.mechanism_data_manifest:
        return []
    try:
        from workflow.mechanism_model_builder import verify_mechanism_data_manifest

        verification = verify_mechanism_data_manifest(Path(contract.mechanism_data_manifest), contract.mechanism_model)
        return [] if verification.get("passed") is True else [f"mechanism_data_manifest:{issue}" for issue in verification.get("issues", ["verification_failed"])]
    except Exception as exc:
        return [f"mechanism_data_manifest_verification_error:{exc}"]


def _route_binding_issues(routes: List[RouteSpec], records: List[ExperimentRecord]) -> List[str]:
    expected = {route.route_id: route for route in routes}
    issues = []
    seen = set()
    for record in records:
        route = expected.get(record.route_id)
        if route is None:
            issues.append(f"executed_route_not_declared:{record.route_id}")
            continue
        if record.route_id in seen:
            issues.append(f"duplicate_executed_route:{record.route_id}")
        seen.add(record.route_id)
        if record.role != route.role:
            issues.append(f"route_role_mismatch:{record.route_id}")
        if route.adapter_path and Path(record.code_path).resolve() != Path(route.adapter_path).resolve():
            issues.append(f"route_adapter_binding_mismatch:{record.route_id}")
        missing_metrics = [metric for metric in route.expected_metrics if not isinstance(record.metrics.get(metric), (int, float))]
        if missing_metrics:
            issues.append(f"route_metrics_missing:{record.route_id}:{','.join(missing_metrics)}")
    return issues


def _validation_issues(profile: ProblemProfile, records: List[ExperimentRecord]) -> List[str]:
    task = profile.contract.task_type if profile.contract else ""
    issues = []
    if task == "forecasting":
        for record in records:
            if record.metrics.get("validation_split") != "train_selection_final_test_recursive":
                issues.append(f"invalid_forecast_validation:{record.route_id}")
            if not all(isinstance(record.metrics.get(f"final_test_{metric}"), (int, float)) for metric in ("MAE", "RMSE", "WAPE")):
                issues.append(f"missing_final_test_metrics:{record.route_id}")
    if task == "regression":
        for record in records:
            expected = "group_holdout" if profile.contract and profile.contract.group_columns else "train_selection_final_test_random"
            if record.metrics.get("validation_split") != expected:
                issues.append(f"invalid_regression_validation:{record.route_id}")
            if not all(isinstance(record.metrics.get(f"final_test_{metric}"), (int, float)) for metric in ("MAE", "RMSE", "WAPE")):
                issues.append(f"missing_final_test_metrics:{record.route_id}")
    if task == "evaluation_ranking":
        for record in records:
            if not isinstance(record.metrics.get("ranking_stability"), (int, float)):
                issues.append(f"missing_ranking_stability:{record.route_id}")
    return issues


def _domain_issues(profile: ProblemProfile, records: List[ExperimentRecord], verification, contract_hash: str) -> List[str]:
    if not profile.contract or (profile.contract.task_type not in {"constrained_optimization", "mechanism", "simulation", "spatial"} and verification is None):
        return []
    issues = []
    if profile.contract.task_type == "constrained_optimization" and not profile.contract.multiobjective_model:
        issues.extend(
            f"constraint_violation:{record.route_id}"
            for record in records
            if not isinstance(record.metrics.get("constraint_violation"), (int, float))
            or not math.isfinite(float(record.metrics["constraint_violation"]))
            or float(record.metrics["constraint_violation"]) > 1e-7
        )
    run_ids = {record.run_id for record in records if record.run_id}
    if not verification or verification.get("passed") is not True:
        issues.append("domain_verification_failed_or_missing")
    elif verification.get("contract_hash") != contract_hash or len(run_ids) != 1 or verification.get("run_id") not in run_ids:
        issues.append("domain_verification_binding_mismatch")
    else:
        if profile.contract.domain_verifier and Path(str(verification.get("script_path", ""))).resolve() != Path(profile.contract.domain_verifier).resolve():
            issues.append("domain_verifier_script_not_contract_bound")
        for field in ("script_path", "registry_path", "stdout_path", "stderr_path"):
            path = Path(str(verification.get(field, "")))
            expected = verification.get(field.replace("path", "hash"))
            if not path.is_file() or not expected or _sha256(path) != expected:
                issues.append(f"domain_verification_artifact_mismatch:{field}")
        stdout_path = Path(str(verification.get("stdout_path", "")))
        if stdout_path.is_file():
            try:
                published = json.loads(stdout_path.read_text(encoding="utf-8"))
                for key in ("checks", "issues", "policy_comparison"):
                    if key in published and published[key] != verification.get(key):
                        issues.append(f"domain_verification_stdout_mismatch:{key}")
            except Exception:
                issues.append("domain_verification_stdout_invalid")
        registry_path = Path(str(verification.get("registry_path", "")))
        if registry_path.is_file():
            try:
                registry = json.loads(registry_path.read_text(encoding="utf-8"))
                expected_records = [record.__dict__ for record in records if record.status == "executed"]
                registry_records = [item for item in registry if item.get("status") == "executed"]
                if registry_records != expected_records:
                    issues.append("domain_verification_registry_not_current_run")
            except Exception:
                issues.append("domain_verification_registry_invalid")
    return issues


def _claim_level(verdict: str, profile: ProblemProfile, domain_verification=None) -> str:
    if verdict != "pass":
        return "no_numerical_claim"
    if profile.contract and domain_verification and domain_verification.get("passed") is True:
        return str(domain_verification.get("claim_level") or "domain_verified_feasible_result")
    return "screening_evidence_only"


def _next_actions(flags: List[str], profile: ProblemProfile) -> List[str]:
    actions = []
    if any(flag.startswith("contract_") for flag in flags):
        unresolved = profile.contract.unresolved_fields if profile.contract else []
        actions.append(f"Complete problem_contract.json fields: {unresolved}")
    if any(flag.startswith(("missing_evidence", "invalid_evidence", "incomplete_provenance")) for flag in flags):
        actions.append("Re-run routes and preserve recomputable evidence with matching hashes.")
    if any(flag.startswith("invalid_forecast_validation") for flag in flags):
        actions.append("Use one declared forecast origin and prevent held-out target values from entering features.")
    if "no_executed_baseline" in flags:
        actions.append("Implement and execute a baseline under the same validation protocol.")
    return actions or ["Inspect route-level evidence before citing numerical results in the paper."]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _format_md(result: Dict[str, object]) -> str:
    lines = ["# V44 Final Judge", "", f"- verdict: {result['verdict']}", f"- claim_level: {result['claim_level']}", f"- primary_metric: {result['primary_metric']}", f"- recommended_route: {result['recommended_route']}", "", "## Gate Layers"]
    for name, payload in result["gate_layers"].items():
        lines.append(f"- {name}: {payload}")
    lines.extend(["", "## Flags"] + [f"- {flag}" for flag in result["flags"] or ["none"]])
    lines.extend(["", "## Next Actions"] + [f"- {action}" for action in result["next_actions"]])
    return "\n".join(lines) + "\n"
