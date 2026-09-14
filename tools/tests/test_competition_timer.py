from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import competition_timer


def test_timer_requires_all_stages_to_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    current = datetime(2026, 9, 6, tzinfo=timezone.utc)
    monkeypatch.setattr(competition_timer, "_now", lambda: current)
    path = tmp_path / "timing.json"
    competition_timer.start(path, "toy")
    for stage in competition_timer.STAGES:
        competition_timer.begin_stage(path, stage)
        current += timedelta(minutes=2)
        competition_timer.end_stage(path, stage, passed=True)
    result = competition_timer.finalize(path)
    assert result["passed"] is True
    assert result["elapsed_minutes"] == pytest.approx(16.0)
    assert all(result["stages"][stage]["sessions"] for stage in competition_timer.STAGES)


def test_timer_rejects_overlapping_or_unknown_stage(tmp_path: Path) -> None:
    path = tmp_path / "timing.json"
    competition_timer.start(path, "toy")
    competition_timer.begin_stage(path, "intake")
    with pytest.raises(ValueError, match="stage_already_active"):
        competition_timer.begin_stage(path, "data_audit")
    with pytest.raises(ValueError, match="unknown_stage"):
        competition_timer.begin_stage(path, "other")


def test_failed_stage_prevents_final_pass(tmp_path: Path) -> None:
    path = tmp_path / "timing.json"
    competition_timer.start(path, "toy")
    for stage in competition_timer.STAGES:
        competition_timer.begin_stage(path, stage)
        competition_timer.end_stage(path, stage, passed=stage != "packaging")
    assert competition_timer.finalize(path)["passed"] is False
