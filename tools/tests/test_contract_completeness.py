import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contract_completeness import audit_contract


def test_incomplete_contract_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    path.write_text(json.dumps({"problem_id": "q1", "status": "ready", "unresolved_fields": []}), encoding="utf-8")
    result = audit_contract(path)
    assert result["passed"] is False
    assert "semantic_review_missing" in result["issues"]
    assert "expected_outputs_missing" in result["issues"]


def test_complete_contract_passes(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    path.write_text(json.dumps({
        "problem_id": "q1", "status": "ready", "unresolved_fields": [],
        "expected_outputs": ["answer.json"], "units": {"x": "kg"},
        "semantic_review": {
            "units_confirmed": True, "constraints_confirmed": True,
            "assumptions_confirmed": True, "ambiguities_resolved": True,
            "human_confirmed": True, "assumptions": [], "ambiguities": [],
            "passed": True, "reviewers": ["team"], "reviewed_at": "2026-09-06T12:00:00+08:00",
        },
    }), encoding="utf-8")
    result = audit_contract(path)
    assert result["passed"] is True
