from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path, base: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        display = str(resolved.relative_to(base.resolve()))
    except ValueError:
        display = str(resolved)
    return {"path": display, "sha256": _sha256(resolved)}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("workflow_summary_not_object")
    return value


def prepare_checklist(
    workflow_summary: Path,
    output_path: Path,
    *,
    official_rules_manifest: Path | None = None,
    data_audit: Path | None = None,
    applicability_audit: Path | None = None,
    reproducibility_audit: Path | None = None,
    claim_registry: Path | None = None,
    total_minutes: int = 4320,
    reserve_minutes: int = 720,
) -> dict[str, Any]:
    workflow_summary = workflow_summary.resolve()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workflow = _load(workflow_summary)
    if workflow.get("kind") != "executable_problem_workflow":
        raise ValueError("workflow_summary_kind_invalid")
    questions = []
    expected_ids = []
    data_audit_path: Path | None = None
    for node in workflow.get("nodes", []):
        if not isinstance(node, dict) or node.get("kind") == "paper":
            continue
        node_id = str(node.get("node_id", "")).strip()
        question_id = str(node.get("question_id") or node_id).strip()
        contract = Path(str(node.get("contract_path", "")))
        outputs = node.get("output_artifacts") or {}
        answer_value = outputs.get("result_summary") or outputs.get("result") or ""
        answer = Path(str(answer_value))
        # Solver summaries expose the audit directory indirectly through the
        # contract path. Bind the nearest generated data_audit.json when it
        # exists, while leaving the draft incomplete if it cannot be found.
        if data_audit_path is None and contract.is_file():
            for parent in [contract.parent, *contract.parents]:
                candidate = parent / "work" / "data_audit" / "data_audit.json"
                if candidate.is_file():
                    data_audit_path = candidate
                    break
        expected_ids.append(question_id)
        questions.append({
            "question_id": question_id,
            "status": "answered" if node.get("passed") is True and answer.is_file() else "needs_work",
            "workflow_node_id": node_id,
            "contract": _binding(contract, output_path.parent) if contract.is_file() else {},
            "answer_artifact": _binding(answer, output_path.parent) if answer.is_file() else {},
            "independent_validation": {},
            "critical_claims": [],
            "fallback": {
                "primary_route": "", "fallback_route": "", "tested": False, "evidence": {},
            },
        })

    rules_binding = {}
    if official_rules_manifest is not None and official_rules_manifest.is_file():
        rules_binding = _binding(official_rules_manifest, output_path.parent)
    manifest_value = (workflow.get("manifest") or {}).get("manifest_path", "")
    manifest_path = Path(str(manifest_value))
    selected_data_audit = data_audit if data_audit and data_audit.is_file() else data_audit_path
    checklist = {
        "version": "v44-readiness-1",
        "problem_id": str(workflow.get("problem_id", "")),
        "official_rules": {
            "confirmed": False, "ai_policy_confirmed": False,
            "submission_requirements_confirmed": False, "source": "", "checked_at": "",
            "evidence": rules_binding, "team_led_core_modeling": False,
            "ai_outputs_require_human_review": False, "no_external_problem_discussion": False,
            "no_problem_related_platform_browsing": False,
        },
        "time_budget": {
            "total_minutes": total_minutes, "measured_end_to_end_minutes": 0,
            "reserve_minutes": reserve_minutes, "evidence": {},
        },
        "data_audit": _binding(selected_data_audit, output_path.parent) if selected_data_audit and selected_data_audit.is_file() else {},
        "applicability_audit": _binding(applicability_audit, output_path.parent) if applicability_audit and applicability_audit.is_file() else {},
        "reproducibility_audit": _binding(reproducibility_audit, output_path.parent) if reproducibility_audit and reproducibility_audit.is_file() else {},
        "workflow_summary": _binding(workflow_summary, output_path.parent),
        "workflow_manifest": _binding(manifest_path, output_path.parent) if manifest_path.is_file() else {},
        "expected_question_ids": expected_ids,
        "questions": questions,
        "claim_registry": _binding(claim_registry, output_path.parent) if claim_registry and claim_registry.is_file() else {},
        "risks": [],
        "deliverables": [],
        "submission_package_audit": {},
        "ai_usage": {
            "used": True, "declaration_in_paper": False, "details_documented": False,
            "all_outputs_human_verified": False,
        },
        "competition_conduct": {
            "team_led_core_modeling_confirmed": False,
            "no_external_problem_discussion_confirmed": False,
            "no_problem_related_platform_browsing_confirmed": False,
            "all_ai_content_reviewed_confirmed": False,
            "attested_by": "", "attested_at": "",
        },
        "human_review": {
            "paper_content_confirmed": False, "layout_confirmed": False,
            "attachment_format_confirmed": False, "reviewer": "", "checked_at": "",
            "anonymous_confirmed": False, "paper_page_limit_confirmed": False,
            "paper_first_page_abstract_confirmed": False, "appendix_file_list_confirmed": False,
            "source_code_runnable_confirmed": False,
            "ai_declaration_before_references_confirmed": False,
        },
        "image_recognition_used": False,
    }
    output_path.write_text(json.dumps(checklist, ensure_ascii=False, indent=2), encoding="utf-8")
    return checklist


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a conservative readiness checklist from a V44 workflow.")
    parser.add_argument("workflow_summary", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--official-rules-manifest", type=Path)
    parser.add_argument("--data-audit", type=Path)
    parser.add_argument("--applicability-audit", type=Path)
    parser.add_argument("--reproducibility-audit", type=Path)
    parser.add_argument("--claim-registry", type=Path)
    parser.add_argument("--total-minutes", type=int, default=4320)
    parser.add_argument("--reserve-minutes", type=int, default=720)
    args = parser.parse_args()
    result = prepare_checklist(
        args.workflow_summary, args.output_path,
        official_rules_manifest=args.official_rules_manifest,
        data_audit=args.data_audit,
        applicability_audit=args.applicability_audit,
        reproducibility_audit=args.reproducibility_audit,
        claim_registry=args.claim_registry,
        total_minutes=args.total_minutes, reserve_minutes=args.reserve_minutes,
    )
    print(json.dumps({
        "output_path": str(args.output_path.resolve()),
        "problem_id": result["problem_id"],
        "question_count": len(result["questions"]),
        "ready_by_construction": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
