from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from competition_readiness import audit_readiness
from workflow.problem_dag import WorkflowNode, build_workflow_manifest


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _ready_case(tmp_path: Path) -> Path:
    contract = _write(tmp_path / "contract.json", {
        "status": "ready", "unresolved_fields": [], "expected_outputs": ["answer"],
        "units": {"answer": "dimensionless"},
        "semantic_review": {
            "units_confirmed": True, "constraints_confirmed": True,
            "assumptions_confirmed": True, "ambiguities_resolved": True,
            "human_confirmed": True, "assumptions": [], "ambiguities": [],
            "passed": True, "reviewers": ["team"], "reviewed_at": "2026-09-06T12:00:00+08:00",
        },
    })
    answer = _write(tmp_path / "answer.csv", "x,y\n1,2\n")
    evidence = _write(tmp_path / "audit.json", {"passed": True, "value": 2})
    claim_registry = _write(tmp_path / "claim_registry.json", {
        "kind": "paper_claim_registry", "passed": True,
        "claims": [{"question_id": "q1", "claim_index": 0,
                    "statement": "answer", "value": 2, "unit": "dimensionless",
                    "evidence": {"path": evidence.name, "sha256": _hash(evidence), "json_path": "value"}}],
    })
    fallback = _write(tmp_path / "fallback.json", {
        "kind": "fallback_switch_rehearsal", "passed": True,
        "primary_route": "candidate", "fallback_route": "baseline",
        "status": "fallback_succeeded", "selected_route": "baseline",
        "events": [
            {"route": "candidate", "status": "failed"},
            {"route": "baseline", "status": "succeeded"},
        ],
    })
    risk = _write(tmp_path / "sensitivity.json", {"passed": True, "sensitivity": {"parameter": "x", "range": [0.9, 1.1]}})
    validation = _write(tmp_path / "independent_validation.json", {
        "passed": True, "checks": [{"recomputed": True}], "verification": "independent",
    })
    timing = _write(tmp_path / "timing.json", {
        "kind": "competition_timing", "measurement_mode": "full_rehearsal", "passed": True, "elapsed_minutes": 70,
        "stages": {name: {"passed": True, "elapsed_minutes": 8.75, "sessions": [{"elapsed_minutes": 8.75, "passed": True}]} for name in (
            "intake", "data_audit", "baseline", "candidate_routes",
            "independent_validation", "paper", "packaging", "human_review",
        )},
    })
    data_audit = _write(tmp_path / "data_audit.json", {
        "kind": "data_audit", "version": "v44", "passed": True,
        "file_count": 1, "readable_file_count": 1, "warnings": [],
    })
    applicability = _write(tmp_path / "applicability.json", {
        "kind": "applicability_audit", "passed": True,
        "task_type": "optimization", "scope_boundary": "declared linear constraints only",
    })
    reproducibility = _write(tmp_path / "reproducibility.json", {
        "kind": "reproducibility_audit", "passed": True,
        "source_path": str((tmp_path / "reproducibility.json").resolve()),
        "command": ["python", "solve.py"], "rerun_command": ["python", "solve.py"],
        "environment": {"python": "3.11"}, "rerun_passed": True,
        "comparison": {"passed": True, "checks": ["output_hash"], "output_hashes_match": True},
        "inputs": [{"path": answer.name, "sha256": _hash(answer)}],
        "outputs": [{"path": answer.name, "sha256": _hash(answer)}],
        "rerun_outputs": [{"path": answer.name, "sha256": _hash(answer)}],
    })
    paper = _write(tmp_path / "paper.pdf", "test artifact")
    support = _write(tmp_path / "support.zip", "test archive")
    ai_detail = _write(tmp_path / "AI工具使用详情.pdf", "test detail")
    submission = _write(tmp_path / "submission_audit.json", {
        "kind": "submission_package_audit", "passed": True,
        "paper": {"sha256": _hash(paper)}, "support_archive": {"sha256": _hash(support)},
    })
    rules_page = _write(tmp_path / "official_rules.html", "official contest rules")
    rules = _write(tmp_path / "official_rules.json", {"pages": [{
        "title": "rules", "url": "https://example.test/rules",
        "local_file": rules_page.name, "sha256": _hash(rules_page),
    }]})
    manifest_info = build_workflow_manifest("toy", [WorkflowNode(
        "solve_q1", "optimization", contract_path=str(contract),
        output_artifacts={"result_summary": str(answer)}, verification={"passed": True},
    )], tmp_path / "workflow_manifest.json")
    workflow = _write(tmp_path / "workflow_summary.json", {
        "passed": True,
        "manifest": manifest_info,
        "nodes": [{"node_id": "solve_q1", "passed": True, "contract_path": str(contract),
                   "output_artifacts": {"result_summary": str(answer)}}],
    })
    checklist = {
        "version": "v44-readiness-1", "problem_id": "toy",
        "official_rules": {
            "confirmed": True, "ai_policy_confirmed": True,
            "submission_requirements_confirmed": True,
            "source": "official rules", "checked_at": "2026-09-06T12:00:00+08:00",
            "evidence": {"path": rules.name, "sha256": _hash(rules)},
            "team_led_core_modeling": True, "ai_outputs_require_human_review": True,
            "no_external_problem_discussion": True, "no_problem_related_platform_browsing": True,
        },
        "time_budget": {"total_minutes": 100, "measured_end_to_end_minutes": 70, "reserve_minutes": 20,
                        "evidence": {"path": timing.name, "sha256": _hash(timing)}},
        "data_audit": {"path": data_audit.name, "sha256": _hash(data_audit)},
        "applicability_audit": {"path": applicability.name, "sha256": _hash(applicability)},
        "reproducibility_audit": {"path": reproducibility.name, "sha256": _hash(reproducibility)},
        "claim_registry": {"path": claim_registry.name, "sha256": _hash(claim_registry)},
        "workflow_summary": {"path": workflow.name, "sha256": _hash(workflow)},
        "workflow_manifest": {"path": "workflow_manifest.json", "sha256": _hash(tmp_path / "workflow_manifest.json")},
        "expected_question_ids": ["q1"],
        "questions": [{
            "question_id": "q1", "status": "answered", "workflow_node_id": "solve_q1",
            "contract": {"path": contract.name, "sha256": _hash(contract)},
            "answer_artifact": {"path": answer.name, "sha256": _hash(answer)},
            "independent_validation": {"path": validation.name, "sha256": _hash(validation)},
            "critical_claims": [{"statement": "answer", "value": 2, "unit": "dimensionless",
                                 "evidence": {"path": evidence.name, "sha256": _hash(evidence),
                                              "json_path": "value"}}],
            "fallback": {"primary_route": "candidate", "fallback_route": "baseline", "tested": True,
                         "evidence": {"path": fallback.name, "sha256": _hash(fallback)}},
        }],
        "risks": [{"severity": "high", "status": "mitigated", "description": "parameter ambiguity",
                   "mitigation": "sensitivity analysis", "evidence": {"path": risk.name, "sha256": _hash(risk)}}],
        "deliverables": [
            {"name": "paper", "required": True, "path": paper.name, "sha256": _hash(paper), "max_bytes": 1000},
            {"name": "support_archive", "required": True, "path": support.name, "sha256": _hash(support), "max_bytes": 1000},
            {"name": "ai_usage_detail_pdf", "required": True, "path": ai_detail.name, "sha256": _hash(ai_detail), "max_bytes": 1000},
        ],
        "submission_package_audit": {"path": submission.name, "sha256": _hash(submission)},
        "ai_usage": {"used": True, "declaration_in_paper": True, "details_documented": True,
                     "all_outputs_human_verified": True},
        "competition_conduct": {
            "team_led_core_modeling_confirmed": True,
            "no_external_problem_discussion_confirmed": True,
            "no_problem_related_platform_browsing_confirmed": True,
            "all_ai_content_reviewed_confirmed": True,
            "attested_by": "team", "attested_at": "2026-09-06T12:00:00+08:00",
        },
        "human_review": {"paper_content_confirmed": True, "layout_confirmed": True,
                         "attachment_format_confirmed": True, "reviewer": "human",
                         "checked_at": "2026-09-06T12:00:00+08:00",
                         "anonymous_confirmed": True, "paper_page_limit_confirmed": True,
                         "paper_first_page_abstract_confirmed": True, "appendix_file_list_confirmed": True,
                         "source_code_runnable_confirmed": True,
                         "ai_declaration_before_references_confirmed": True},
        "image_recognition_used": False,
    }
    return _write(tmp_path / "competition_readiness.json", checklist)


def test_complete_readiness_case_passes(tmp_path: Path) -> None:
    result = audit_readiness(_ready_case(tmp_path))
    assert result["ready"] is True
    assert result["verdict"] == "READY"
    assert result["issues"] == []


def test_missing_fallback_and_excess_time_block_readiness(tmp_path: Path) -> None:
    path = _ready_case(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["questions"][0]["fallback"]["tested"] = False
    payload["time_budget"]["measured_end_to_end_minutes"] = 90
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    result = audit_readiness(path)
    assert result["ready"] is False
    assert "fallback_not_tested:q1" in result["issues"]
    assert "time_budget_exceeded_or_invalid" in result["issues"]


def test_hash_mutation_and_unresolved_high_risk_are_rejected(tmp_path: Path) -> None:
    path = _ready_case(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    (tmp_path / "answer.csv").write_text("mutated", encoding="utf-8")
    payload["risks"][0]["status"] = "accepted"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    result = audit_readiness(path)
    assert result["ready"] is False
    assert "hash_mismatch:question:q1:answer" in result["issues"]
    assert "high_risk_not_mitigated:0" in result["issues"]


def test_omitted_expected_question_is_rejected(tmp_path: Path) -> None:
    path = _ready_case(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["expected_question_ids"].append("q2")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    result = audit_readiness(path)
    assert result["ready"] is False
    assert "expected_question_missing:q2" in result["issues"]


def test_claim_value_must_match_bound_evidence(tmp_path: Path) -> None:
    path = _ready_case(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["questions"][0]["critical_claims"][0]["value"] = 3
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    result = audit_readiness(path)
    assert result["ready"] is False
    assert "claim_value_not_traceable:q1:0" in result["issues"]


def test_csv_claim_selector_is_traceable(tmp_path: Path) -> None:
    path = _ready_case(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    table = _write(tmp_path / "metrics.csv", "scenario,profit\ncase1,123.5\ncase2,100\n")
    payload["questions"][0]["critical_claims"][0] = {
        "statement": "case1 profit", "value": 123.5, "unit": "yuan",
        "evidence": {"path": table.name, "sha256": _hash(table),
                     "csv_selector": {"where": {"scenario": "case1"}, "column": "profit"}},
    }
    registry_path = tmp_path / "claim_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["claims"][0] = {
        "question_id": "q1", "claim_index": 0, "statement": "case1 profit",
        "value": 123.5, "unit": "yuan",
        "evidence": {"path": table.name, "sha256": _hash(table),
                     "csv_selector": {"where": {"scenario": "case1"}, "column": "profit"}},
    }
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    payload["claim_registry"]["sha256"] = _hash(registry_path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    result = audit_readiness(path)
    assert result["ready"] is True


def test_empty_bindings_report_clear_issue_codes(tmp_path: Path) -> None:
    path = _ready_case(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["official_rules"]["evidence"] = {}
    payload["workflow_summary"] = {}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    result = audit_readiness(path)
    assert "missing_file_binding:official_rules:evidence" in result["issues"]
    assert "missing_file_binding:workflow_summary" in result["issues"]


def test_official_compliance_and_size_limit_are_hard_gates(tmp_path: Path) -> None:
    path = _ready_case(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["official_rules"]["no_problem_related_platform_browsing"] = False
    payload["deliverables"][0]["max_bytes"] = 1
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    result = audit_readiness(path)
    assert "competition_rule_not_acknowledged:no_problem_related_platform_browsing" in result["issues"]
    assert "deliverable_size_exceeded:paper" in result["issues"]


def test_rules_manifest_detects_archived_page_mutation(tmp_path: Path) -> None:
    path = _ready_case(tmp_path)
    (tmp_path / "official_rules.html").write_text("mutated", encoding="utf-8")
    result = audit_readiness(path)
    assert "hash_mismatch:official_rules:page:0" in result["issues"]


def test_conduct_must_be_attested_after_execution(tmp_path: Path) -> None:
    path = _ready_case(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["competition_conduct"]["no_external_problem_discussion_confirmed"] = False
    payload["competition_conduct"]["attested_by"] = ""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    result = audit_readiness(path)
    assert "competition_conduct_not_confirmed:no_external_problem_discussion_confirmed" in result["issues"]
    assert "competition_conduct_attester_missing" in result["issues"]


def test_timing_must_cover_full_workflow_and_match_total(tmp_path: Path) -> None:
    path = _ready_case(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    timing_path = tmp_path / "timing.json"
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    del timing["stages"]["packaging"]
    timing["elapsed_minutes"] = 60
    timing_path.write_text(json.dumps(timing), encoding="utf-8")
    payload["time_budget"]["evidence"]["sha256"] = _hash(timing_path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    result = audit_readiness(path)
    assert "time_budget_elapsed_mismatch" in result["issues"]
    assert "time_budget_stage_not_passed:packaging" in result["issues"]


def test_mitigated_risk_requires_passed_evidence(tmp_path: Path) -> None:
    path = _ready_case(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    risk_path = tmp_path / "sensitivity.json"
    risk_path.write_text(json.dumps({"passed": False}), encoding="utf-8")
    payload["risks"][0]["evidence"]["sha256"] = _hash(risk_path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    result = audit_readiness(path)
    assert "risk_evidence_not_passed:0" in result["issues"]


def test_mitigated_risk_requires_diagnostic_structure(tmp_path: Path) -> None:
    path = _ready_case(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    risk_path = tmp_path / "sensitivity.json"
    risk_path.write_text(json.dumps({"passed": True}), encoding="utf-8")
    payload["risks"][0]["evidence"]["sha256"] = _hash(risk_path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    result = audit_readiness(path)
    assert "risk_evidence_structure_missing:0" in result["issues"]


def test_fallback_route_names_must_match_evidence(tmp_path: Path) -> None:
    path = _ready_case(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["questions"][0]["fallback"]["fallback_route"] = "other"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    result = audit_readiness(path)
    assert "fallback_route_mismatch:q1" in result["issues"]


def test_fallback_evidence_must_rehearse_primary_failure(tmp_path: Path) -> None:
    path = _ready_case(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    fallback_path = tmp_path / "fallback.json"
    fallback = json.loads(fallback_path.read_text(encoding="utf-8"))
    fallback["events"] = [{"route": "candidate", "status": "succeeded"}]
    fallback_path.write_text(json.dumps(fallback), encoding="utf-8")
    payload["questions"][0]["fallback"]["evidence"]["sha256"] = _hash(fallback_path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    result = audit_readiness(path)
    assert "fallback_switch_events_missing:q1" in result["issues"]


def test_answer_and_contract_must_belong_to_workflow_node(tmp_path: Path) -> None:
    path = _ready_case(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    other_answer = _write(tmp_path / "other_answer.csv", "x,y\n1,2\n")
    other_contract = _write(tmp_path / "other_contract.json", {"status": "ready", "unresolved_fields": []})
    payload["questions"][0]["answer_artifact"] = {"path": other_answer.name, "sha256": _hash(other_answer)}
    payload["questions"][0]["contract"] = {"path": other_contract.name, "sha256": _hash(other_contract)}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    result = audit_readiness(path)
    assert "question_answer_not_node_output:q1" in result["issues"]
    assert "question_contract_not_node_contract:q1" in result["issues"]


def test_answer_artifact_cannot_be_header_only(tmp_path: Path) -> None:
    path = _ready_case(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    answer_path = tmp_path / "answer.csv"
    answer_path.write_text("answer,unit\n", encoding="utf-8")
    payload["questions"][0]["answer_artifact"]["sha256"] = _hash(answer_path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    result = audit_readiness(path)
    assert "answer_artifact_structurally_empty:q1" in result["issues"]


def test_claim_unit_is_required(tmp_path: Path) -> None:
    path = _ready_case(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["questions"][0]["critical_claims"][0]["unit"]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    assert "claim_unit_missing:q1:0" in audit_readiness(path)["issues"]


def test_semantic_contract_confirmation_is_required(tmp_path: Path) -> None:
    path = _ready_case(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    contract_path = tmp_path / "contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract.pop("semantic_review")
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    payload["questions"][0]["contract"]["sha256"] = _hash(contract_path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    result = audit_readiness(path)
    assert "question_contract_incomplete:q1:semantic_review_missing" in result["issues"]


def test_data_audit_evidence_is_required(tmp_path: Path) -> None:
    path = _ready_case(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("data_audit")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    assert "data_audit_evidence_missing" in audit_readiness(path)["issues"]


def test_independent_validation_evidence_is_required(tmp_path: Path) -> None:
    path = _ready_case(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["questions"][0].pop("independent_validation")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    assert "independent_validation_evidence_missing:q1" in audit_readiness(path)["issues"]


def test_applicability_audit_evidence_is_required(tmp_path: Path) -> None:
    path = _ready_case(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("applicability_audit")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    assert "applicability_audit_evidence_missing" in audit_readiness(path)["issues"]


def test_reproducibility_audit_evidence_is_required(tmp_path: Path) -> None:
    path = _ready_case(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("reproducibility_audit")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    assert "reproducibility_audit_evidence_missing" in audit_readiness(path)["issues"]


def test_image_recognition_is_a_hard_block(tmp_path: Path) -> None:
    path = _ready_case(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["image_recognition_used"] = True
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    result = audit_readiness(path)
    assert "image_recognition_was_used" in result["issues"]
    assert result["ready"] is False


def test_prohibited_activity_is_a_hard_block(tmp_path: Path) -> None:
    path = _ready_case(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["competition_conduct"]["prohibited_activity_detected"] = True
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    result = audit_readiness(path)
    assert "competition_conduct_prohibited_activity_detected" in result["issues"]


def test_timing_rejects_placeholder_paper_stage(tmp_path: Path) -> None:
    path = _ready_case(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    timing_path = tmp_path / "timing.json"
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    timing["stages"]["paper"]["elapsed_minutes"] = 0.001
    timing["stages"]["paper"]["sessions"][0]["elapsed_minutes"] = 0.001
    timing["elapsed_minutes"] = 62.251
    timing_path.write_text(json.dumps(timing), encoding="utf-8")
    payload["time_budget"]["evidence"]["sha256"] = _hash(timing_path)
    payload["time_budget"]["measured_end_to_end_minutes"] = 62.251
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    assert "time_budget_stage_too_short:paper" in audit_readiness(path)["issues"]


def test_claim_registry_must_cover_each_critical_claim(tmp_path: Path) -> None:
    path = _ready_case(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    registry_path = tmp_path / "claim_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["claims"] = []
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    payload["claim_registry"]["sha256"] = _hash(registry_path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    result = audit_readiness(path)
    assert "claim_registry_claims_missing" in result["issues"]


def test_workflow_manifest_revalidation_rejects_mutated_output(tmp_path: Path) -> None:
    path = _ready_case(tmp_path)
    (tmp_path / "answer.csv").write_text("mutated", encoding="utf-8")
    result = audit_readiness(path)
    assert any(item.startswith("workflow_manifest:workflow_artifact_hash_mismatch") for item in result["issues"])


def test_submission_audit_must_pass_and_bind_deliverable_hashes(tmp_path: Path) -> None:
    path = _ready_case(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    audit_path = tmp_path / "submission_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["passed"] = False
    audit["paper"]["sha256"] = "wrong"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    payload["submission_package_audit"]["sha256"] = _hash(audit_path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    result = audit_readiness(path)
    assert "submission_package_audit_not_passed" in result["issues"]
    assert "submission_paper_hash_mismatch" in result["issues"]
