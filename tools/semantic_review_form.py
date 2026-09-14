"""Create and apply a human semantic-review form for a question contract.

The form is intentionally separate from numerical solving.  It gives the
team a small, auditable handoff: agree on units, constraints, assumptions,
ambiguities and outputs first, then write the confirmed review back into a
contract copy.  It never uses image recognition or infers missing semantics.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"not_an_object:{path}")
    return value


def create_review_form(contract_path: Path, output_path: Path) -> dict[str, Any]:
    contract_path = contract_path.resolve()
    contract = _load(contract_path)
    semantic = contract.get("semantic_review") if isinstance(contract.get("semantic_review"), dict) else {}
    form = {
        "version": "v44-semantic-review-1",
        "kind": "semantic_review_form",
        "contract_path": str(contract_path),
        "problem_id": contract.get("problem_id", ""),
        "question_id": contract.get("question_id", contract.get("problem_id", "")),
        "expected_outputs": contract.get("expected_outputs", []),
        "units": contract.get("units", {}),
        "constraints": contract.get("constraints", []),
        "assumptions": semantic.get("assumptions", contract.get("assumptions", [])),
        "ambiguities": semantic.get("ambiguities", contract.get("ambiguities", [])),
        "confirmations": {
            "units_confirmed": False,
            "constraints_confirmed": False,
            "assumptions_confirmed": False,
            "ambiguities_resolved": False,
            "human_confirmed": False,
        },
        "reviewers": [],
        "reviewed_at": "",
        "notes": "Fill this form after jointly reading the official statement; do not infer unresolved semantics from model output.",
    }
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(form, ensure_ascii=False, indent=2), encoding="utf-8")
    return form


def apply_review_form(contract_path: Path, form_path: Path, output_path: Path) -> dict[str, Any]:
    contract = _load(contract_path)
    form = _load(form_path)
    confirmations = form.get("confirmations")
    if not isinstance(confirmations, dict):
        raise ValueError("confirmations_missing")
    required = ("units_confirmed", "constraints_confirmed", "assumptions_confirmed", "ambiguities_resolved", "human_confirmed")
    issues = [f"confirmation_missing:{key}" for key in required if confirmations.get(key) is not True]
    if not isinstance(form.get("expected_outputs"), list) or not form.get("expected_outputs"):
        issues.append("expected_outputs_missing")
    if not isinstance(form.get("units"), dict):
        issues.append("units_schema_invalid")
    if not isinstance(form.get("assumptions"), list) or not isinstance(form.get("ambiguities"), list):
        issues.append("assumptions_or_ambiguities_list_missing")
    reviewers = form.get("reviewers")
    if not isinstance(reviewers, list) or not reviewers or not all(str(item).strip() for item in reviewers):
        issues.append("reviewers_missing")
    if not str(form.get("reviewed_at", "")).strip():
        issues.append("reviewed_at_missing")
    semantic = {
        **{key: confirmations.get(key) is True for key in required},
        "assumptions": form.get("assumptions", []),
        "ambiguities": form.get("ambiguities", []),
        "reviewers": reviewers if isinstance(reviewers, list) else [],
        "reviewed_at": str(form.get("reviewed_at", "")),
        "form_path": str(form_path.resolve()),
        "passed": not issues,
        "issues": sorted(set(issues)),
    }
    contract["semantic_review"] = semantic
    if not issues:
        contract["status"] = "ready"
        contract["unresolved_fields"] = []
    else:
        contract["status"] = "needs_input"
        contract.setdefault("unresolved_fields", [])
        contract["unresolved_fields"] = sorted(set(contract["unresolved_fields"]) | set(issues))
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"kind": "semantic_review_application", "passed": not issues, "issues": sorted(set(issues)), "contract": contract}


def main() -> int:
    parser = argparse.ArgumentParser(description="Create or apply a human semantic-review form.")
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("contract", type=Path)
    create.add_argument("output", type=Path)
    apply = sub.add_parser("apply")
    apply.add_argument("contract", type=Path)
    apply.add_argument("form", type=Path)
    apply.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "create":
        result = create_review_form(args.contract, args.output)
        print(json.dumps({"output": str(args.output.resolve()), "kind": result["kind"]}, ensure_ascii=False))
        return 0
    result = apply_review_form(args.contract, args.form, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
