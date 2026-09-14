import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fallback_switch import run_with_fallback


def test_primary_failure_switches_to_fallback_and_records_events(tmp_path: Path) -> None:
    evidence = tmp_path / "switch.json"

    def primary():
        raise RuntimeError("injected_failure")

    result = run_with_fallback("primary", "fallback", primary, lambda: {"answer": 7}, evidence)
    assert result["passed"] is True
    assert result["status"] == "fallback_succeeded"
    assert result["selected_route"] == "fallback"
    assert [event["status"] for event in result["events"]] == ["failed", "succeeded"]
    saved = json.loads(evidence.read_text(encoding="utf-8"))
    assert saved["primary_route"] == "primary"


def test_both_failures_are_not_marked_passed() -> None:
    result = run_with_fallback("primary", "fallback", lambda: 1 / 0, lambda: 1 / 0)
    assert result["passed"] is False
    assert result["status"] == "all_routes_failed"


def test_routes_must_be_distinct() -> None:
    try:
        run_with_fallback("same", "same", lambda: 1, lambda: 2)
    except ValueError as exc:
        assert str(exc) == "distinct_primary_and_fallback_routes_required"
    else:
        raise AssertionError("expected distinct-route validation")
