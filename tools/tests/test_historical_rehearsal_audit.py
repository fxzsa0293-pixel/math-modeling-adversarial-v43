import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from historical_rehearsal_audit import audit_rehearsal


def _summary(path: Path, *, passed: bool = True) -> None:
    path.write_text(json.dumps({
        "suite": "demo",
        "passed": passed,
        "all_passed": passed,
        "passed_cases": 1 if passed else 0,
        "case_count": 1,
        "complete_report": {
            "path": str(path.parent / "report.md"),
            "verification": {"passed": passed},
        },
        "workflow_dag": {
            "manifest_path": str(path.parent / "workflow.json"),
            "passed": passed,
        },
    }), encoding="utf-8")
    (path.parent / "report.md").write_text("report", encoding="utf-8")
    (path.parent / "workflow.json").write_text("{}", encoding="utf-8")


def test_historical_audit_never_claims_formal_ready(tmp_path: Path) -> None:
    primary = tmp_path / "primary.json"
    cross = tmp_path / "cross.json"
    _summary(primary)
    _summary(cross)
    result = audit_rehearsal(primary, cross, tmp_path / "out")
    assert result["verdict"] == "TECHNICAL_REHEARSAL_PARTIAL"
    assert result["ready_for_formal_contest"] is False
    assert "eight_stage_timing" in result["missing_competition_evidence"]
    assert "real_data_multi_case_benchmark" in result["completed_capabilities"]


def test_historical_audit_records_failed_suites(tmp_path: Path) -> None:
    primary = tmp_path / "primary.json"
    cross = tmp_path / "cross.json"
    _summary(primary, passed=False)
    _summary(cross, passed=True)
    result = audit_rehearsal(primary, cross, tmp_path / "out")
    assert "real_data_multi_case_benchmark" in result["missing_competition_evidence"]
    assert "cross_archetype_benchmark" in result["completed_capabilities"]


def test_historical_audit_records_failure_injected_switch(tmp_path: Path) -> None:
    primary = tmp_path / "primary.json"
    cross = tmp_path / "cross.json"
    _summary(primary)
    _summary(cross)
    switch = tmp_path / "switch.json"
    switch.write_text(json.dumps({"passed": True, "status": "fallback_succeeded"}), encoding="utf-8")
    result = audit_rehearsal(primary, cross, tmp_path / "out", switch_rehearsal=switch)
    assert "failure_injected_fallback_switch" in result["completed_capabilities"]


def test_historical_audit_records_timed_rehearsal(tmp_path: Path) -> None:
    primary = tmp_path / "primary.json"
    cross = tmp_path / "cross.json"
    _summary(primary)
    _summary(cross)
    timed = tmp_path / "timed.json"
    timed.write_text(json.dumps({"passed": True, "timing": {"passed": True}}), encoding="utf-8")
    result = audit_rehearsal(primary, cross, tmp_path / "out", timed_rehearsal=timed)
    assert "eight_stage_timed_rehearsal" in result["completed_capabilities"]


def test_historical_audit_records_package_audit(tmp_path: Path) -> None:
    primary = tmp_path / "primary.json"
    cross = tmp_path / "cross.json"
    _summary(primary)
    _summary(cross)
    package = tmp_path / "package.json"
    package.write_text(json.dumps({"passed": True}), encoding="utf-8")
    result = audit_rehearsal(primary, cross, tmp_path / "out", package_audit=package)
    assert "historical_submission_package_audit" in result["completed_capabilities"]
