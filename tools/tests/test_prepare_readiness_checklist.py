from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prepare_readiness_checklist import prepare_checklist


def test_prepare_checklist_binds_solver_nodes_without_claiming_readiness(tmp_path: Path) -> None:
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({"status": "ready", "unresolved_fields": []}), encoding="utf-8")
    result = tmp_path / "result.json"
    result.write_text(json.dumps({"value": 1}), encoding="utf-8")
    workflow = tmp_path / "workflow_summary.json"
    workflow.write_text(json.dumps({
        "kind": "executable_problem_workflow", "problem_id": "toy", "passed": True,
        "nodes": [
            {"node_id": "solve", "question_id": "question_01", "kind": "optimization",
             "passed": True, "contract_path": str(contract),
             "output_artifacts": {"result_summary": str(result)}},
            {"node_id": "paper", "question_id": "question_02", "kind": "paper", "passed": True},
        ],
    }), encoding="utf-8")
    output = tmp_path / "readiness.json"
    prepared = prepare_checklist(workflow, output, total_minutes=100, reserve_minutes=20)
    assert prepared["expected_question_ids"] == ["question_01"]
    assert len(prepared["questions"]) == 1
    assert prepared["questions"][0]["status"] == "answered"
    assert prepared["questions"][0]["critical_claims"] == []
    assert prepared["questions"][0]["fallback"]["tested"] is False
    assert prepared["official_rules"]["confirmed"] is False
    assert output.is_file()


def test_prepare_checklist_marks_missing_answer_as_needs_work(tmp_path: Path) -> None:
    workflow = tmp_path / "workflow_summary.json"
    workflow.write_text(json.dumps({
        "kind": "executable_problem_workflow", "problem_id": "toy", "passed": False,
        "nodes": [{"node_id": "solve", "question_id": "q1", "kind": "prediction",
                   "passed": False, "contract_path": "", "output_artifacts": {}}],
    }), encoding="utf-8")
    prepared = prepare_checklist(workflow, tmp_path / "readiness.json")
    assert prepared["questions"][0]["status"] == "needs_work"
    assert prepared["questions"][0]["answer_artifact"] == {}


def test_prepare_checklist_binds_explicit_evidence_paths(tmp_path: Path) -> None:
    workflow = tmp_path / "workflow_summary.json"
    workflow.write_text(json.dumps({
        "kind": "executable_problem_workflow", "problem_id": "toy",
        "nodes": [],
    }), encoding="utf-8")
    evidence = {}
    for name in ("data_audit", "applicability", "reproducibility", "claims"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"kind": name}), encoding="utf-8")
        evidence[name] = path
    output = tmp_path / "readiness.json"
    prepared = prepare_checklist(
        workflow, output, data_audit=evidence["data_audit"],
        applicability_audit=evidence["applicability"],
        reproducibility_audit=evidence["reproducibility"],
        claim_registry=evidence["claims"],
    )
    for field, key in (("data_audit", "data_audit"),
                       ("applicability_audit", "applicability"),
                       ("reproducibility_audit", "reproducibility"),
                       ("claim_registry", "claims")):
        binding = prepared[field]
        assert binding["path"] == evidence[key].name
        assert len(binding["sha256"]) == 64
