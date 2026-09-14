from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproducibility_audit import audit_reproducibility


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reproducibility_audit_accepts_rerun_evidence(tmp_path: Path) -> None:
    source = tmp_path / "input.csv"
    output = tmp_path / "output.json"
    source.write_text("x,y\n1,2\n", encoding="utf-8")
    output.write_text(json.dumps({"value": 2}), encoding="utf-8")
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({
        "kind": "reproducibility_audit", "passed": True,
        "command": ["python", "solve.py"], "rerun_command": ["python", "solve.py"],
        "environment": {"python": "3.11"}, "rerun_passed": True,
        "comparison": {"passed": True, "checks": ["output_hash"], "output_hashes_match": True},
        "inputs": [{"path": source.name, "sha256": _hash(source)}],
        "outputs": [{"path": output.name, "sha256": _hash(output)}],
        "rerun_outputs": [{"path": output.name, "sha256": _hash(output)}],
    }), encoding="utf-8")
    result = audit_reproducibility(evidence, tmp_path / "audit.json")
    assert result["passed"] is True


def test_reproducibility_audit_rejects_missing_rerun(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({"kind": "reproducibility_audit", "passed": True}), encoding="utf-8")
    result = audit_reproducibility(evidence, tmp_path / "audit.json")
    assert result["passed"] is False
    assert "rerun_not_passed" in result["issues"]
