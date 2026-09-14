from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from workflow.problem_dag import verify_workflow_manifest
from contract_completeness import audit_contract
from reproducibility_audit import audit_reproducibility


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, issues: list[str], label: str) -> dict[str, Any]:
    if not path.is_file():
        issues.append(f"missing_file:{label}:{path}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        issues.append(f"invalid_json:{label}:{type(exc).__name__}")
        return {}
    if not isinstance(value, dict):
        issues.append(f"invalid_object:{label}")
        return {}
    return value


def _resolve(base: Path, value: Any) -> Path:
    path = Path(str(value or ""))
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _verify_bound_file(base: Path, item: dict[str, Any], issues: list[str], label: str) -> Path | None:
    if not isinstance(item, dict):
        issues.append(f"invalid_file_binding:{label}")
        return None
    if not str(item.get("path", "")).strip():
        issues.append(f"missing_file_binding:{label}")
        return None
    path = _resolve(base, item.get("path"))
    if not path.is_file():
        issues.append(f"missing_file:{label}:{path}")
        return None
    expected = str(item.get("sha256", "")).lower()
    if not expected:
        issues.append(f"missing_sha256:{label}")
    elif _sha256(path) != expected:
        issues.append(f"hash_mismatch:{label}")
    return path


def _same_value(actual: Any, expected: Any, tolerance: float = 1e-9) -> bool:
    try:
        left = float(actual)
        right = float(expected)
        return abs(left - right) <= tolerance * max(1.0, abs(right))
    except (TypeError, ValueError):
        return actual == expected


def _trace_claim(path: Path, evidence: dict[str, Any], expected: Any) -> bool:
    tolerance = float(evidence.get("tolerance", 1e-9))
    if "json_path" in evidence:
        try:
            current: Any = json.loads(path.read_text(encoding="utf-8"))
            for part in str(evidence["json_path"]).split("."):
                current = current[int(part)] if isinstance(current, list) else current[part]
            return _same_value(current, expected, tolerance)
        except (OSError, json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError):
            return False
    selector = evidence.get("csv_selector")
    if isinstance(selector, dict) and str(selector.get("column", "")):
        where = selector.get("where") or {}
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                matches = [row for row in csv.DictReader(handle)
                           if all(str(row.get(str(key), "")) == str(value) for key, value in where.items())]
            return len(matches) == 1 and _same_value(matches[0].get(str(selector["column"])), expected, tolerance)
        except (OSError, csv.Error, TypeError, ValueError):
            return False
    return False


def _evidence_passed(path: Path) -> bool:
    if path.suffix.lower() != ".json":
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    return (
        payload.get("passed") is True
        or payload.get("core_pass") is True
        or str(payload.get("verdict", "")).lower() in {"pass", "ready"}
    )


def _verify_answer_artifact(path: Path, question_id: str, issues: list[str]) -> None:
    """Reject empty or structurally empty answer files.

    The check is intentionally format-aware but conservative: it validates
    common JSON/CSV/text answer artifacts without pretending to understand the
    domain-specific answer schema.
    """
    label = f"question:{question_id}:answer"
    try:
        if path.stat().st_size <= 0:
            issues.append(f"answer_artifact_empty:{question_id}")
            return
        suffix = path.suffix.lower()
        if suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            if value in (None, {}, [], ""):
                issues.append(f"answer_artifact_structurally_empty:{question_id}")
        elif suffix in {".csv", ".tsv"}:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.reader(handle, delimiter="	" if suffix == ".tsv" else ","))
            if len(rows) < 2 or not any(str(cell).strip() for cell in rows[1]):
                issues.append(f"answer_artifact_structurally_empty:{question_id}")
        elif suffix in {".txt", ".md", ".tex", ".html"}:
            if not path.read_text(encoding="utf-8", errors="replace").strip():
                issues.append(f"answer_artifact_structurally_empty:{question_id}")
    except (OSError, UnicodeError, json.JSONDecodeError, csv.Error, ValueError):
        issues.append(f"answer_artifact_unreadable:{question_id}")


def _verify_claim_registry(path: Path, questions: list[dict[str, Any]], base: Path, issues: list[str]) -> None:
    registry = _load_json(path, issues, "claim_registry")
    if registry.get("kind") != "paper_claim_registry" or registry.get("passed") is not True:
        issues.append("claim_registry_not_passed")
    records = registry.get("claims")
    if not isinstance(records, list) or not records:
        issues.append("claim_registry_claims_missing")
        return
    expected: dict[tuple[str, int], dict[str, Any]] = {}
    for question in questions:
        qid = str(question.get("question_id", "")).strip()
        for index, claim in enumerate(question.get("critical_claims") or []):
            expected[(qid, index)] = claim if isinstance(claim, dict) else {}
    seen: set[tuple[str, int]] = set()
    for index, record in enumerate(records):
        label = f"claim_registry:{index}"
        if not isinstance(record, dict):
            issues.append(f"{label}:invalid")
            continue
        qid = str(record.get("question_id", "")).strip()
        try:
            claim_index = int(record.get("claim_index"))
        except (TypeError, ValueError):
            issues.append(f"{label}:claim_index_invalid")
            continue
        key = (qid, claim_index)
        if key not in expected:
            issues.append(f"{label}:unexpected_claim:{qid}:{claim_index}")
            continue
        if key in seen:
            issues.append(f"{label}:duplicate_claim:{qid}:{claim_index}")
        seen.add(key)
        claim = expected[key]
        if record.get("value") is None or not str(record.get("unit", "")).strip():
            issues.append(f"{label}:value_or_unit_missing")
        if not str(record.get("statement", "")).strip():
            issues.append(f"{label}:statement_missing")
        if claim.get("value") is not None and not _same_value(record.get("value"), claim.get("value"), float((record.get("evidence") or {}).get("tolerance", 1e-9))):
            issues.append(f"{label}:value_mismatch:{qid}:{claim_index}")
        if str(record.get("unit", "")) != str(claim.get("unit", "")):
            issues.append(f"{label}:unit_mismatch:{qid}:{claim_index}")
        evidence = record.get("evidence") or {}
        evidence_path = _verify_bound_file(base, evidence, issues, label)
        if evidence_path is None or not _trace_claim(evidence_path, evidence, record.get("value")):
            issues.append(f"{label}:not_traceable")
    for key in sorted(set(expected) - seen):
        issues.append(f"claim_registry:missing_claim:{key[0]}:{key[1]}")


def _verify_rules_manifest(path: Path, issues: list[str]) -> None:
    manifest = _load_json(path, issues, "official_rules:manifest")
    pages = manifest.get("pages")
    if not isinstance(pages, list) or not pages:
        issues.append("official_rules_manifest_pages_missing")
        return
    for index, page in enumerate(pages):
        label = f"official_rules:page:{index}"
        if not isinstance(page, dict):
            issues.append(f"official_rules_page_invalid:{index}")
            continue
        if not str(page.get("title", "")).strip() or not str(page.get("url", "")).startswith("https://"):
            issues.append(f"official_rules_page_metadata_invalid:{index}")
        _verify_bound_file(
            path.parent,
            {"path": page.get("local_file"), "sha256": page.get("sha256")},
            issues,
            label,
        )


def _verify_timing_evidence(path: Path, measured_minutes: float, issues: list[str]) -> None:
    payload = _load_json(path, issues, "time_budget:evidence_json")
    if payload.get("kind") != "competition_timing" or payload.get("passed") is not True:
        issues.append("time_budget_evidence_not_passed")
        return
    mode = str(payload.get("measurement_mode", "")).strip()
    if mode not in {"full_rehearsal", "formal_contest"}:
        issues.append("time_budget_measurement_mode_missing_or_invalid")
    try:
        recorded = float(payload["elapsed_minutes"])
    except (KeyError, TypeError, ValueError):
        issues.append("time_budget_evidence_elapsed_missing_or_invalid")
        return
    if not _same_value(recorded, measured_minutes, tolerance=1e-6):
        issues.append("time_budget_elapsed_mismatch")
    required_stages = {
        "intake", "data_audit", "baseline", "candidate_routes",
        "independent_validation", "paper", "packaging", "human_review",
    }
    stages = payload.get("stages")
    if not isinstance(stages, dict):
        issues.append("time_budget_stages_missing")
        return
    for stage in sorted(required_stages):
        record = stages.get(stage)
        if not isinstance(record, dict) or record.get("passed") is not True:
            issues.append(f"time_budget_stage_not_passed:{stage}")
            continue
        try:
            elapsed = float(record["elapsed_minutes"])
            if elapsed < 0:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            issues.append(f"time_budget_stage_elapsed_invalid:{stage}")
            elapsed = 0.0
        # A passed stage recorded as a near-zero instant is usually a script
        # marker rather than evidence that the corresponding competition work
        # was actually performed. This is a quality floor, not a claim about
        # how long a real team must spend on the stage.
        if stage in {"paper", "packaging", "human_review"} and elapsed < 0.1:
            issues.append(f"time_budget_stage_too_short:{stage}")
        sessions = record.get("sessions")
        if not isinstance(sessions, list) or not sessions:
            issues.append(f"time_budget_stage_sessions_missing:{stage}")
    try:
        stage_total = sum(float(stages[stage]["elapsed_minutes"]) for stage in required_stages)
        if not _same_value(stage_total, recorded, tolerance=1e-6):
            issues.append("time_budget_stage_total_mismatch")
    except (KeyError, TypeError, ValueError):
        pass


def _verify_fallback_evidence(path: Path, primary_route: str, fallback_route: str, issues: list[str], question_id: str) -> None:
    payload = _load_json(path, issues, f"question:{question_id}:fallback_json")
    if not _evidence_passed(path):
        issues.append(f"fallback_evidence_not_passed:{question_id}")
    if payload.get("kind") != "fallback_switch_rehearsal":
        issues.append(f"fallback_evidence_kind_invalid:{question_id}")
    if str(payload.get("primary_route", "")) != primary_route:
        issues.append(f"fallback_primary_route_mismatch:{question_id}")
    if str(payload.get("fallback_route", "")) != fallback_route:
        issues.append(f"fallback_route_mismatch:{question_id}")
    if payload.get("status") != "fallback_succeeded" or payload.get("selected_route") != fallback_route:
        issues.append(f"fallback_switch_not_succeeded:{question_id}")
    events = payload.get("events")
    if not isinstance(events, list) or len(events) < 2:
        issues.append(f"fallback_switch_events_missing:{question_id}")
        return
    primary_failed = any(
        isinstance(event, dict) and event.get("route") == primary_route and event.get("status") == "failed"
        for event in events
    )
    fallback_succeeded = any(
        isinstance(event, dict) and event.get("route") == fallback_route and event.get("status") == "succeeded"
        for event in events
    )
    if not primary_failed:
        issues.append(f"fallback_primary_failure_not_rehearsed:{question_id}")
    if not fallback_succeeded:
        issues.append(f"fallback_success_event_missing:{question_id}")
    if primary_failed and fallback_succeeded:
        primary_index = next(i for i, event in enumerate(events) if isinstance(event, dict) and event.get("route") == primary_route and event.get("status") == "failed")
        fallback_index = next(i for i, event in enumerate(events) if isinstance(event, dict) and event.get("route") == fallback_route and event.get("status") == "succeeded")
        if fallback_index <= primary_index:
            issues.append(f"fallback_event_order_invalid:{question_id}")


def audit_readiness(checklist_path: Path) -> dict[str, Any]:
    checklist_path = checklist_path.resolve()
    issues: list[str] = []
    warnings: list[str] = []
    spec = _load_json(checklist_path, issues, "checklist")
    base = checklist_path.parent

    if spec.get("version") != "v44-readiness-1":
        issues.append("invalid_checklist_version")
    if not str(spec.get("problem_id", "")).strip():
        issues.append("missing_problem_id")

    rules = spec.get("official_rules") or {}
    for key in ("confirmed", "ai_policy_confirmed", "submission_requirements_confirmed"):
        if rules.get(key) is not True:
            issues.append(f"official_rules_not_confirmed:{key}")
    for key in (
        "team_led_core_modeling",
        "ai_outputs_require_human_review",
        "no_external_problem_discussion",
        "no_problem_related_platform_browsing",
    ):
        if rules.get(key) is not True:
            issues.append(f"competition_rule_not_acknowledged:{key}")
    if not str(rules.get("source", "")).strip():
        issues.append("official_rules_source_missing")
    if not str(rules.get("checked_at", "")).strip():
        issues.append("official_rules_checked_at_missing")
    rules_manifest = _verify_bound_file(base, rules.get("evidence") or {}, issues, "official_rules:evidence")
    if rules_manifest is not None:
        _verify_rules_manifest(rules_manifest, issues)

    timing = spec.get("time_budget") or {}
    try:
        total = float(timing["total_minutes"])
        measured = float(timing["measured_end_to_end_minutes"])
        reserve = float(timing["reserve_minutes"])
        if min(total, measured, reserve) < 0 or measured + reserve > total:
            issues.append("time_budget_exceeded_or_invalid")
    except (KeyError, TypeError, ValueError):
        issues.append("time_budget_missing_or_invalid")
    timing_evidence = _verify_bound_file(base, timing.get("evidence") or {}, issues, "time_budget:evidence")
    if timing_evidence is not None:
        try:
            _verify_timing_evidence(timing_evidence, float(timing["measured_end_to_end_minutes"]), issues)
        except (KeyError, TypeError, ValueError):
            pass

    data_audit_binding = spec.get("data_audit") or {}
    data_audit_path = _verify_bound_file(base, data_audit_binding, issues, "data_audit")
    if data_audit_path is not None:
        data_audit = _load_json(data_audit_path, issues, "data_audit")
        if data_audit.get("kind") != "data_audit":
            issues.append("data_audit_kind_invalid")
        if data_audit.get("passed") is not True:
            issues.append("data_audit_not_passed")
        try:
            if int(data_audit.get("file_count", 0)) <= 0 or int(data_audit.get("readable_file_count", 0)) <= 0:
                issues.append("data_audit_no_readable_files")
        except (TypeError, ValueError):
            issues.append("data_audit_counts_invalid")
    else:
        issues.append("data_audit_evidence_missing")

    applicability_binding = spec.get("applicability_audit") or {}
    applicability_path = _verify_bound_file(base, applicability_binding, issues, "applicability_audit")
    if applicability_path is None:
        issues.append("applicability_audit_evidence_missing")
    else:
        applicability = _load_json(applicability_path, issues, "applicability_audit")
        if applicability.get("kind") != "applicability_audit":
            issues.append("applicability_audit_kind_invalid")
        if applicability.get("passed") is not True:
            issues.append("applicability_audit_not_passed")

    reproducibility_binding = spec.get("reproducibility_audit") or {}
    reproducibility_path = _verify_bound_file(base, reproducibility_binding, issues, "reproducibility_audit")
    if reproducibility_path is None:
        issues.append("reproducibility_audit_evidence_missing")
    else:
        reproducibility = _load_json(reproducibility_path, issues, "reproducibility_audit")
        if reproducibility.get("kind") != "reproducibility_audit":
            issues.append("reproducibility_audit_kind_invalid")
        if reproducibility.get("passed") is not True:
            issues.append("reproducibility_audit_not_passed")
        source_path = Path(str(reproducibility.get("source_path", "")))
        if source_path.is_file():
            try:
                live_repro = audit_reproducibility(source_path, source_path.parent / "_readiness_reproducibility_check.json")
                if live_repro.get("passed") is not True:
                    issues.extend(f"reproducibility_live_check:{item}" for item in live_repro.get("issues", []))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                issues.append(f"reproducibility_live_check_error:{type(exc).__name__}")
        else:
            issues.append("reproducibility_source_path_missing")
        for key in ("command", "rerun_command"):
            value = reproducibility.get(key)
            if not isinstance(value, list) or not value:
                issues.append(f"reproducibility_{key}_missing:{key}")
        if not isinstance(reproducibility.get("environment"), dict) or not reproducibility.get("environment"):
            issues.append("reproducibility_environment_missing")
        if reproducibility.get("rerun_passed") is not True:
            issues.append("reproducibility_rerun_not_passed")
        comparison = reproducibility.get("comparison")
        if not isinstance(comparison, dict) or comparison.get("passed") is not True or not comparison.get("checks"):
            issues.append("reproducibility_comparison_not_passed")
        if not isinstance(comparison, dict) or comparison.get("output_hashes_match") is not True:
            issues.append("reproducibility_output_hash_comparison_failed")
        outputs = reproducibility.get("outputs")
        rerun_outputs = reproducibility.get("rerun_outputs")
        if not isinstance(outputs, list) or not outputs or not isinstance(rerun_outputs, list) or not rerun_outputs:
            issues.append("reproducibility_rerun_outputs_missing")
        else:
            original_pairs = {(str(item.get("path", "")), str(item.get("sha256", "")).lower()) for item in outputs if isinstance(item, dict)}
            rerun_pairs = {(str(item.get("path", "")), str(item.get("sha256", "")).lower()) for item in rerun_outputs if isinstance(item, dict)}
            if original_pairs != rerun_pairs:
                issues.append("reproducibility_rerun_output_hash_mismatch")

    workflow_binding = spec.get("workflow_summary") or {}
    workflow_path = _verify_bound_file(base, workflow_binding, issues, "workflow_summary")
    if workflow_path is None:
        workflow_path = _resolve(base, workflow_binding.get("path") if isinstance(workflow_binding, dict) else "")
        workflow = {}
    else:
        workflow = _load_json(workflow_path, issues, "workflow_summary")
    if workflow and workflow.get("passed") is not True:
        issues.append("workflow_not_passed")
    workflow_manifest_binding = spec.get("workflow_manifest") or {}
    workflow_manifest_path = _verify_bound_file(base, workflow_manifest_binding, issues, "workflow_manifest")
    if workflow_manifest_path is not None:
        manifest_verification = verify_workflow_manifest(workflow_manifest_path)
        issues.extend(f"workflow_manifest:{item}" for item in manifest_verification.get("issues", []))
        if manifest_verification.get("passed") is not True and not manifest_verification.get("issues"):
            issues.append("workflow_manifest_not_passed")
    nodes = {str(node.get("node_id")): node for node in workflow.get("nodes", []) if isinstance(node, dict)}

    questions = spec.get("questions")
    if not isinstance(questions, list) or not questions:
        issues.append("questions_missing")
        questions = []
    seen: set[str] = set()
    for question in questions:
        if not isinstance(question, dict):
            issues.append("question_entry_invalid")
            continue
        qid = str(question.get("question_id", "")).strip()
        label = f"question:{qid or 'unknown'}"
        if not qid or qid in seen:
            issues.append(f"question_id_missing_or_duplicate:{qid}")
        seen.add(qid)
        if question.get("status") != "answered":
            issues.append(f"question_not_answered:{qid}")
        node_id = str(question.get("workflow_node_id", ""))
        node = nodes.get(node_id) or {}
        if node_id not in nodes or node.get("passed") is not True:
            issues.append(f"question_workflow_node_not_passed:{qid}:{node_id}")
        answer_path = _verify_bound_file(base, question.get("answer_artifact") or {}, issues, f"{label}:answer")
        if answer_path is not None:
            _verify_answer_artifact(answer_path, qid, issues)
        node_outputs = {
            str(_resolve(workflow_path.parent, value))
            for value in (node.get("output_artifacts") or {}).values()
            if str(value).strip()
        }
        if answer_path is not None and str(answer_path) not in node_outputs:
            issues.append(f"question_answer_not_node_output:{qid}")
        validation_binding = question.get("independent_validation") or {}
        validation_path = _verify_bound_file(base, validation_binding, issues, f"{label}:independent_validation")
        if validation_path is None:
            issues.append(f"independent_validation_evidence_missing:{qid}")
        else:
            validation_payload = _load_json(validation_path, issues, f"{label}:independent_validation")
            if validation_payload.get("passed") is not True and validation_payload.get("core_pass") is not True:
                issues.append(f"independent_validation_not_passed:{qid}")
            if not any(key in validation_payload for key in ("checks", "verification", "recomputed", "independent")):
                issues.append(f"independent_validation_structure_missing:{qid}")
        contract_binding = question.get("contract") or {}
        contract_path = _verify_bound_file(base, contract_binding, issues, f"{label}:contract")
        if contract_path is None:
            contract_path = _resolve(base, contract_binding.get("path") if isinstance(contract_binding, dict) else "")
            contract = {}
        else:
            contract = _load_json(contract_path, issues, f"{label}:contract")
            node_contract = str(node.get("contract_path", "")).strip()
            if not node_contract or contract_path != _resolve(workflow_path.parent, node_contract):
                issues.append(f"question_contract_not_node_contract:{qid}")
        if contract:
            if contract.get("status") != "ready" or contract.get("unresolved_fields"):
                issues.append(f"question_contract_not_ready:{qid}")
            # A status flag alone is insufficient: final readiness requires
            # explicit units, assumptions, constraints, ambiguity resolution,
            # and human semantic confirmation for every question.
            contract_audit = audit_contract(contract_path)
            for contract_issue in contract_audit.get("issues", []):
                issues.append(f"question_contract_incomplete:{qid}:{contract_issue}")
        claims = question.get("critical_claims")
        if not isinstance(claims, list) or not claims:
            issues.append(f"critical_claims_missing:{qid}")
        else:
            for index, claim in enumerate(claims):
                claim_label = f"{label}:claim:{index}"
                if not isinstance(claim, dict) or not str(claim.get("statement", "")).strip():
                    issues.append(f"claim_statement_missing:{qid}:{index}")
                    continue
                if claim.get("value") is None:
                    issues.append(f"claim_value_missing:{qid}:{index}")
                if not str(claim.get("unit", "")).strip():
                    issues.append(f"claim_unit_missing:{qid}:{index}")
                claim_evidence = claim.get("evidence") or {}
                claim_path = _verify_bound_file(base, claim_evidence, issues, claim_label)
                if claim_path is not None and not _trace_claim(claim_path, claim_evidence, claim.get("value")):
                    issues.append(f"claim_value_not_traceable:{qid}:{index}")

        fallback = question.get("fallback") or {}
        if not str(fallback.get("primary_route", "")).strip():
            issues.append(f"primary_route_missing:{qid}")
        if not str(fallback.get("fallback_route", "")).strip():
            issues.append(f"fallback_route_missing:{qid}")
        if fallback.get("primary_route") == fallback.get("fallback_route"):
            issues.append(f"fallback_not_distinct:{qid}")
        if fallback.get("tested") is not True:
            issues.append(f"fallback_not_tested:{qid}")
        fallback_evidence = _verify_bound_file(base, fallback.get("evidence") or {}, issues, f"{label}:fallback")
        if fallback_evidence is not None:
            _verify_fallback_evidence(fallback_evidence, str(fallback.get("primary_route", "")), str(fallback.get("fallback_route", "")), issues, qid)

    expected_question_ids = spec.get("expected_question_ids")
    if not isinstance(expected_question_ids, list) or not expected_question_ids:
        issues.append("expected_question_ids_missing")
    else:
        expected = {str(item).strip() for item in expected_question_ids if str(item).strip()}
        if len(expected) != len(expected_question_ids):
            issues.append("expected_question_ids_invalid_or_duplicate")
        missing = sorted(expected - seen)
        unexpected = sorted(seen - expected)
        issues.extend(f"expected_question_missing:{item}" for item in missing)
        issues.extend(f"unexpected_question:{item}" for item in unexpected)

    claim_registry_binding = spec.get("claim_registry") or {}
    claim_registry_path = _verify_bound_file(base, claim_registry_binding, issues, "claim_registry")
    if claim_registry_path is None:
        issues.append("claim_registry_evidence_missing")
    else:
        _verify_claim_registry(claim_registry_path, questions, base, issues)

    risks = spec.get("risks")
    if not isinstance(risks, list) or not risks:
        issues.append("risk_register_missing")
        risks = []
    for index, risk in enumerate(risks):
        if not isinstance(risk, dict):
            issues.append(f"risk_entry_invalid:{index}")
            continue
        severity = str(risk.get("severity", "")).lower()
        status = str(risk.get("status", "")).lower()
        if severity not in {"low", "medium", "high"}:
            issues.append(f"risk_severity_invalid:{index}")
        if status not in {"mitigated", "accepted"}:
            issues.append(f"risk_unresolved:{index}")
        if severity == "high" and status != "mitigated":
            issues.append(f"high_risk_not_mitigated:{index}")
        if not str(risk.get("description", "")).strip() or not str(risk.get("mitigation", "")).strip():
            issues.append(f"risk_description_or_mitigation_missing:{index}")
        if status == "mitigated":
            risk_evidence = _verify_bound_file(base, risk.get("evidence") or {}, issues, f"risk:{index}")
            if risk_evidence is not None:
                risk_payload = _load_json(risk_evidence, issues, f"risk:{index}:json")
                if not _evidence_passed(risk_evidence):
                    issues.append(f"risk_evidence_not_passed:{index}")
                if not any(key in risk_payload for key in ("checks", "sensitivity", "diagnostics", "verification", "analysis")):
                    issues.append(f"risk_evidence_structure_missing:{index}")

    deliverables = spec.get("deliverables")
    if not isinstance(deliverables, list) or not deliverables:
        issues.append("deliverables_missing")
        deliverables = []
    deliverable_names: set[str] = set()
    for index, deliverable in enumerate(deliverables):
        if not isinstance(deliverable, dict):
            issues.append(f"deliverable_invalid:{index}")
        else:
            name = str(deliverable.get("name", "")).strip()
            if not name or name in deliverable_names:
                issues.append(f"deliverable_name_missing_or_duplicate:{index}")
            deliverable_names.add(name)
            if deliverable.get("required") is True:
                artifact = _verify_bound_file(base, deliverable, issues, f"deliverable:{index}")
                suffix = artifact.suffix.lower() if artifact is not None else ""
                if artifact is not None and name == "paper" and suffix not in {".pdf", ".doc", ".docx"}:
                    issues.append("deliverable_format_invalid:paper")
                if artifact is not None and name == "support_archive" and suffix not in {".zip", ".rar"}:
                    issues.append("deliverable_format_invalid:support_archive")
                if artifact is not None and name == "ai_usage_detail_pdf" and suffix != ".pdf":
                    issues.append("deliverable_format_invalid:ai_usage_detail_pdf")
                if artifact is not None and deliverable.get("max_bytes") is not None:
                    try:
                        if artifact.stat().st_size > int(deliverable["max_bytes"]):
                            issues.append(f"deliverable_size_exceeded:{name}")
                    except (TypeError, ValueError):
                        issues.append(f"deliverable_max_bytes_invalid:{name}")

    ai_usage = spec.get("ai_usage") or {}
    if ai_usage.get("used") is not True:
        issues.append("ai_usage_not_declared")
    if ai_usage.get("declaration_in_paper") is not True:
        issues.append("ai_declaration_in_paper_missing")
    if ai_usage.get("details_documented") is not True:
        issues.append("ai_usage_details_not_documented")
    if ai_usage.get("all_outputs_human_verified") is not True:
        issues.append("ai_outputs_not_fully_human_verified")
    for required_name in ("paper", "support_archive", "ai_usage_detail_pdf"):
        if required_name not in deliverable_names:
            issues.append(f"required_deliverable_not_listed:{required_name}")

    submission_binding = spec.get("submission_package_audit") or {}
    submission_path = _verify_bound_file(base, submission_binding, issues, "submission_package_audit")
    if submission_path is not None:
        submission = _load_json(submission_path, issues, "submission_package_audit_json")
        if submission.get("kind") != "submission_package_audit" or submission.get("passed") is not True:
            issues.append("submission_package_audit_not_passed")
        by_name = {str(item.get("name")): item for item in deliverables if isinstance(item, dict)}
        paper_record = submission.get("paper") or {}
        support_record = submission.get("support_archive") or {}
        if str(paper_record.get("sha256", "")) != str((by_name.get("paper") or {}).get("sha256", "")):
            issues.append("submission_paper_hash_mismatch")
        if str(support_record.get("sha256", "")) != str((by_name.get("support_archive") or {}).get("sha256", "")):
            issues.append("submission_support_hash_mismatch")

    conduct = spec.get("competition_conduct") or {}
    for key in (
        "team_led_core_modeling_confirmed",
        "no_external_problem_discussion_confirmed",
        "no_problem_related_platform_browsing_confirmed",
        "all_ai_content_reviewed_confirmed",
    ):
        if conduct.get(key) is not True:
            issues.append(f"competition_conduct_not_confirmed:{key}")
    if not str(conduct.get("attested_by", "")).strip():
        issues.append("competition_conduct_attester_missing")
    if not str(conduct.get("attested_at", "")).strip():
        issues.append("competition_conduct_time_missing")
    if conduct.get("prohibited_activity_detected") is True:
        issues.append("competition_conduct_prohibited_activity_detected")
    if conduct.get("image_recognition_used") is True:
        issues.append("competition_conduct_image_recognition_used")

    review = spec.get("human_review") or {}
    if review.get("paper_content_confirmed") is not True:
        issues.append("human_paper_content_review_missing")
    if review.get("layout_confirmed") is not True:
        issues.append("human_layout_review_missing")
    if review.get("attachment_format_confirmed") is not True:
        issues.append("human_attachment_format_review_missing")
    for key in (
        "anonymous_confirmed",
        "paper_page_limit_confirmed",
        "paper_first_page_abstract_confirmed",
        "appendix_file_list_confirmed",
        "source_code_runnable_confirmed",
        "ai_declaration_before_references_confirmed",
    ):
        if review.get(key) is not True:
            issues.append(f"human_submission_check_missing:{key}")
    if not str(review.get("reviewer", "")).strip():
        issues.append("human_reviewer_missing")
    if not str(review.get("checked_at", "")).strip():
        issues.append("human_review_time_missing")
    if spec.get("image_recognition_used") is True:
        issues.append("image_recognition_was_used")

    unique_issues = sorted(set(issues))
    return {
        "version": "v44-readiness-1", "kind": "competition_readiness_audit",
        "problem_id": spec.get("problem_id", ""), "ready": not unique_issues,
        "verdict": "READY" if not unique_issues else "NOT_READY",
        "issues": unique_issues, "warnings": sorted(set(warnings)),
        "question_count": len(questions), "workflow_summary": str(workflow_path),
        "checklist_path": str(checklist_path),
        "checklist_sha256": _sha256(checklist_path) if checklist_path.is_file() else "",
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = ["# Competition readiness audit", "", f"- verdict: **{result['verdict']}**",
             f"- problem_id: {result['problem_id']}", f"- question_count: {result['question_count']}",
             "", "## Blocking issues", ""]
    lines.extend([f"- {item}" for item in result["issues"]] or ["- none"])
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {item}" for item in result["warnings"]] or ["- none"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit contest workflow readiness without image recognition.")
    parser.add_argument("checklist", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = audit_readiness(args.checklist)
    output_dir = (args.output_dir or args.checklist.parent / "readiness_audit").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "competition_readiness.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "competition_readiness.md").write_text(_markdown(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
