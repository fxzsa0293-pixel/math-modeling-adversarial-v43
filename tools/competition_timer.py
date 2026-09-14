from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGES = (
    "intake", "data_audit", "baseline", "candidate_routes",
    "independent_validation", "paper", "packaging", "human_review",
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.isoformat()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def start(path: Path, problem_id: str) -> dict[str, Any]:
    if path.exists():
        raise ValueError("timing_file_already_exists")
    now = _now()
    payload = {
        "version": "v44-readiness-1", "kind": "competition_timing",
        "measurement_mode": "full_rehearsal",
        "problem_id": problem_id, "started_at": _iso(now), "completed_at": "",
        "active_stage": "", "active_started_at": "", "elapsed_minutes": 0.0,
        "passed": False,
        "stages": {stage: {"passed": False, "elapsed_minutes": 0.0, "sessions": []} for stage in STAGES},
    }
    _write(path, payload)
    return payload


def begin_stage(path: Path, stage: str) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"unknown_stage:{stage}")
    payload = _load(path)
    if payload.get("kind") != "competition_timing":
        raise ValueError("timing_file_invalid")
    if payload.get("active_stage"):
        raise ValueError(f"stage_already_active:{payload['active_stage']}")
    payload["active_stage"] = stage
    payload["active_started_at"] = _iso(_now())
    _write(path, payload)
    return payload


def end_stage(path: Path, stage: str, passed: bool) -> dict[str, Any]:
    payload = _load(path)
    if payload.get("active_stage") != stage or not payload.get("active_started_at"):
        raise ValueError(f"stage_not_active:{stage}")
    ended = _now()
    started = datetime.fromisoformat(payload["active_started_at"])
    minutes = max(0.0, (ended - started).total_seconds() / 60.0)
    record = payload["stages"][stage]
    record["sessions"].append({
        "started_at": payload["active_started_at"], "ended_at": _iso(ended),
        "elapsed_minutes": minutes, "passed": bool(passed),
    })
    record["elapsed_minutes"] = sum(float(item["elapsed_minutes"]) for item in record["sessions"])
    record["passed"] = bool(passed)
    payload["active_stage"] = ""
    payload["active_started_at"] = ""
    _update(payload)
    _write(path, payload)
    return payload


def finalize(path: Path) -> dict[str, Any]:
    payload = _load(path)
    if payload.get("active_stage"):
        raise ValueError(f"stage_still_active:{payload['active_stage']}")
    _update(payload)
    payload["completed_at"] = _iso(_now())
    payload["passed"] = all(payload["stages"][stage].get("passed") is True for stage in STAGES)
    _write(path, payload)
    return payload


def _update(payload: dict[str, Any]) -> None:
    payload["elapsed_minutes"] = sum(float(payload["stages"][stage]["elapsed_minutes"]) for stage in STAGES)


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Record auditable contest-workflow stage timings.")
    parser.add_argument("timing_file", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    start_parser = sub.add_parser("start")
    start_parser.add_argument("--problem-id", required=True)
    begin_parser = sub.add_parser("begin")
    begin_parser.add_argument("stage", choices=STAGES)
    end_parser = sub.add_parser("end")
    end_parser.add_argument("stage", choices=STAGES)
    end_parser.add_argument("--passed", action="store_true")
    sub.add_parser("finalize")
    args = parser.parse_args()
    path = args.timing_file.resolve()
    if args.command == "start":
        result = start(path, args.problem_id)
    elif args.command == "begin":
        result = begin_stage(path, args.stage)
    elif args.command == "end":
        result = end_stage(path, args.stage, args.passed)
    else:
        result = finalize(path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
