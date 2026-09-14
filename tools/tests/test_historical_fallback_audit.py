import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from historical_fallback_audit import audit_fallbacks


def test_fallback_audit_requires_two_executed_routes(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"cases": [{"case_id": "one", "summary_path": str(tmp_path / "missing.json")}] }), encoding="utf-8")
    result = audit_fallbacks(summary, tmp_path / "out")
    assert result["passed"] is False
    assert result["eligible_case_count"] == 0


def test_fallback_audit_accepts_same_registry_pair(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    run_dir = case_dir / "run"
    registry_dir = run_dir / "registry"
    evidence_dir = run_dir / "experiments" / "evidence"
    registry_dir.mkdir(parents=True)
    evidence_dir.mkdir(parents=True)
    for route in ("primary", "fallback"):
        (evidence_dir / f"{route}.json").write_text("{}", encoding="utf-8")
    registry = []
    for route in ("primary", "fallback"):
        registry.append({
            "route_id": route, "role": "candidate", "status": "executed",
            "evidence_paths": {"route_evidence": str(evidence_dir / f"{route}.json")},
        })
    (registry_dir / "experiment_registry.json").write_text(json.dumps(registry), encoding="utf-8")
    run_summary = run_dir / "summary.json"
    run_summary.write_text(json.dumps({"recommended_route": {"route_id": "primary"}}), encoding="utf-8")
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"cases": [{"case_id": "one", "summary_path": str(run_summary)}]}), encoding="utf-8")
    result = audit_fallbacks(summary, tmp_path / "out")
    assert result["passed"] is True
    assert result["eligible_case_count"] == 1
