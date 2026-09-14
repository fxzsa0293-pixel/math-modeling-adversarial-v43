from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_claim_registry import build_registry


def test_registry_copies_complete_claims(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({"value": 3}), encoding="utf-8")
    checklist = tmp_path / "checklist.json"
    checklist.write_text(json.dumps({"questions": [{"question_id": "q1", "critical_claims": [{
        "statement": "value", "value": 3, "unit": "kg",
        "evidence": {"path": evidence.name, "json_path": "value"},
    }]}]}), encoding="utf-8")
    result = build_registry(checklist, tmp_path / "registry.json")
    assert result["passed"] is True
    assert result["claims"][0]["value"] == 3


def test_registry_rejects_claim_without_evidence_binding(tmp_path: Path) -> None:
    checklist = tmp_path / "checklist.json"
    checklist.write_text(json.dumps({"questions": [{"question_id": "q1", "critical_claims": [{
        "statement": "value", "value": 3, "unit": "kg",
    }]}]}), encoding="utf-8")
    result = build_registry(checklist, tmp_path / "registry.json")
    assert result["passed"] is False
    assert "claim_evidence_binding_missing:q1:0" in result["issues"]
