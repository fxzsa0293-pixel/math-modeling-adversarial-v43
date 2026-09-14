"""Audit semantic completeness of a question contract before numerical work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def audit_contract(path: Path) -> dict[str, Any]:
    path = path.resolve()
    issues: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"kind": "contract_completeness_audit", "passed": False, "issues": [f"contract_unreadable:{type(exc).__name__}"], "path": str(path)}
    if not isinstance(payload, dict):
        return {"kind": "contract_completeness_audit", "passed": False, "issues": ["contract_not_object"], "path": str(path)}
    if payload.get("status") != "ready":
        issues.append("contract_status_not_ready")
    if payload.get("unresolved_fields"):
        issues.append("contract_has_unresolved_fields")
    if not isinstance(payload.get("expected_outputs"), list) or not payload.get("expected_outputs"):
        issues.append("expected_outputs_missing")
    semantic = payload.get("semantic_review")
    if not isinstance(semantic, dict):
        issues.append("semantic_review_missing")
        semantic = {}
    required = {
        "units_confirmed": "units_not_confirmed",
        "constraints_confirmed": "constraints_not_confirmed",
        "assumptions_confirmed": "assumptions_not_confirmed",
        "ambiguities_resolved": "ambiguities_not_resolved",
        "human_confirmed": "human_semantic_confirmation_missing",
    }
    for key, issue in required.items():
        if semantic.get(key) is not True:
            issues.append(issue)
    if not isinstance(semantic.get("assumptions"), list):
        issues.append("assumptions_list_missing")
    if not isinstance(semantic.get("ambiguities"), list):
        issues.append("ambiguities_list_missing")
    if semantic.get("passed") is not True:
        issues.append("semantic_review_not_passed")
    if not isinstance(semantic.get("reviewers"), list) or not semantic.get("reviewers"):
        issues.append("semantic_review_reviewers_missing")
    if not str(semantic.get("reviewed_at", "")).strip():
        issues.append("semantic_review_time_missing")
    if not isinstance(payload.get("units"), dict):
        issues.append("units_schema_invalid")
    result = {
        "version": "v44-contract-audit-1",
        "kind": "contract_completeness_audit",
        "path": str(path),
        "passed": not issues,
        "verdict": "PASS" if not issues else "INCOMPLETE",
        "issues": sorted(set(issues)),
        "problem_id": payload.get("problem_id", ""),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit semantic completeness of a problem contract.")
    parser.add_argument("contract", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_contract(args.contract)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
