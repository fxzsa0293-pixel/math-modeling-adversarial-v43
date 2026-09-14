from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from applicability_audit import audit_applicability


def test_supported_task_requires_scope_boundary(tmp_path: Path) -> None:
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({"problem_id": "q1", "task_type": "optimization", "scope_boundary": "linear model"}), encoding="utf-8")
    result = audit_applicability(contract, tmp_path / "audit.json")
    assert result["passed"] is True
    assert result["support_level"] == "native_supported"


def test_unknown_task_without_specialist_is_rejected(tmp_path: Path) -> None:
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({"problem_id": "q1", "task_type": "pde", "scope_boundary": ""}), encoding="utf-8")
    result = audit_applicability(contract, tmp_path / "audit.json")
    assert result["passed"] is False
    assert "unsupported_task_type_without_specialist_route" in result["issues"]
