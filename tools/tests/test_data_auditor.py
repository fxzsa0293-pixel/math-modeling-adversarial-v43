from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workflow.data_auditor import audit_data_root


def test_data_audit_marks_readable_input_as_passed(tmp_path: Path) -> None:
    (tmp_path / "table.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    result = audit_data_root(tmp_path, tmp_path / "audit")
    assert result["kind"] == "data_audit"
    assert result["passed"] is True
    saved = json.loads((tmp_path / "audit" / "data_audit.json").read_text(encoding="utf-8"))
    assert saved["passed"] is True


def test_data_audit_rejects_empty_root(tmp_path: Path) -> None:
    result = audit_data_root(tmp_path, tmp_path / "audit")
    assert result["passed"] is False
    assert "no_supported_data_file" in result["warnings"]
