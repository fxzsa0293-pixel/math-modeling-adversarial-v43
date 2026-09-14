"""Build a traceable paper-claim registry from a readiness checklist.

The registry is deliberately mechanical: it copies each declared critical
claim and its evidence binding, then marks the result passed only when every
claim has a statement, value, unit, and re-readable evidence location.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_registry(checklist_path: Path, output_path: Path) -> dict[str, Any]:
    payload = json.loads(checklist_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    issues: list[str] = []
    for question in payload.get("questions", []):
        qid = str(question.get("question_id", "")).strip()
        for index, claim in enumerate(question.get("critical_claims") or []):
            record = {
                "question_id": qid,
                "claim_index": index,
                "statement": claim.get("statement", ""),
                "value": claim.get("value"),
                "unit": claim.get("unit", ""),
                "evidence": claim.get("evidence") or {},
            }
            records.append(record)
            if not qid or not str(record["statement"]).strip() or record["value"] is None:
                issues.append(f"claim_incomplete:{qid}:{index}")
            if not str(record["unit"]).strip():
                issues.append(f"claim_unit_missing:{qid}:{index}")
            evidence = record["evidence"]
            if not str(evidence.get("path", "")).strip() or not (str(evidence.get("json_path", "")).strip() or isinstance(evidence.get("csv_selector"), dict)):
                issues.append(f"claim_evidence_binding_missing:{qid}:{index}")
    result = {
        "version": "v44-paper-claim-registry-1",
        "kind": "paper_claim_registry",
        "source_checklist": str(checklist_path.resolve()),
        "passed": bool(records) and not issues,
        "verdict": "PASS" if records and not issues else "INCOMPLETE",
        "claims": records,
        "issues": sorted(set(issues)),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a traceable paper claim registry.")
    parser.add_argument("checklist", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = build_registry(args.checklist, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
