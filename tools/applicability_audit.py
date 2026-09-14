"""Audit whether the declared solver scope is appropriate for a contract.

This is a conservative guard against applying a convenient generic adapter to
an unsupported problem.  It does not infer a mathematical model from prose.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SUPPORTED_TASK_TYPES = {
    "forecasting", "regression", "evaluation_ranking", "optimization",
    "constrained_optimization", "multiobjective", "robust_optimization",
    "mechanism", "simulation", "spatial", "retail_decision",
}


def audit_applicability(contract_path: Path, output_path: Path) -> dict[str, Any]:
    contract_path = contract_path.resolve()
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("contract_not_object")
    task_type = str(payload.get("task_type", "")).strip()
    issues: list[str] = []
    warnings: list[str] = []
    support_level = "native_supported" if task_type in SUPPORTED_TASK_TYPES else "specialist_required"
    if not task_type:
        issues.append("task_type_missing")
    elif task_type not in SUPPORTED_TASK_TYPES:
        if not payload.get("specialist_routes") or not str(payload.get("domain_verifier", "")).strip():
            issues.append("unsupported_task_type_without_specialist_route")
        else:
            warnings.append("specialist_route_requires_explicit_code_authorization")
    scope_boundary = str(payload.get("scope_boundary", "")).strip()
    if not scope_boundary:
        issues.append("scope_boundary_missing")
    routes = payload.get("routes")
    if routes is not None:
        if not isinstance(routes, list) or not routes:
            issues.append("route_plan_missing_or_invalid")
        else:
            for index, route in enumerate(routes):
                if not isinstance(route, dict):
                    issues.append(f"route_invalid:{index}")
                elif route.get("supported") is not True:
                    issues.append(f"unsupported_route_declared:{index}")
    result = {
        "version": "v44-applicability-audit-1",
        "kind": "applicability_audit",
        "contract_path": str(contract_path),
        "problem_id": payload.get("problem_id", ""),
        "task_type": task_type,
        "support_level": support_level,
        "scope_boundary": scope_boundary,
        "passed": not issues,
        "verdict": "PASS" if not issues else "NOT_APPLICABLE",
        "issues": sorted(set(issues)),
        "warnings": sorted(set(warnings)),
    }
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit solver applicability for a problem contract.")
    parser.add_argument("contract", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit_applicability(args.contract, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
