"""Run a real-data, eight-stage technical rehearsal with wall-clock evidence.

This is not a formal-contest submission.  It executes a small but real V44
forecasting case, then verifies the resulting artifacts, report, and archive
under the same timer used by the competition readiness gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from competition_timer import STAGES, begin_stage, end_stage, finalize, start

TOOLS = Path(__file__).resolve().parent
SKILL_ROOT = TOOLS.parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_solver(contract: Path, data_root: Path, output: Path, max_routes: int) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(TOOLS / "run_v39_solver.py"), "--problem-id", "CUMCM2023C_daily_sales_forecast", "--data-root", str(data_root), "--contract", str(contract), "--output-dir", str(output), "--max-routes", str(max_routes)]
    process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
    summary = output / "summary.json"
    payload = json.loads(summary.read_text(encoding="utf-8")) if summary.is_file() else {}
    return {"returncode": process.returncode, "summary": payload, "stdout_tail": process.stdout[-1000:], "stderr_tail": process.stderr[-1000:], "command": command}


def run_rehearsal(reference_root: Path, output_dir: Path) -> dict:
    reference_root = reference_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    timer_path = output_dir / "competition_timing.json"
    result: dict = {"version": "v44-timed-rehearsal-1", "kind": "historical_timed_rehearsal", "problem_id": "CUMCM2023C_daily_sales_forecast", "stage_results": {}}
    start(timer_path, result["problem_id"])
    contract = TOOLS / "run" / "v44_real_benchmarks" / "CUMCM2023C_daily_sales_forecast" / "benchmark_contract.json"

    begin_stage(timer_path, "intake")
    intake_passed = contract.is_file() and bool(json.loads(contract.read_text(encoding="utf-8")).get("expected_outputs"))
    end_stage(timer_path, "intake", intake_passed)
    result["stage_results"]["intake"] = {"passed": intake_passed, "contract": str(contract)}

    begin_stage(timer_path, "data_audit")
    contract_payload = json.loads(contract.read_text(encoding="utf-8"))
    source_paths = [Path(item["path"]) for item in contract_payload.get("data_sources", [])]
    data_root = source_paths[0].parent if source_paths else contract.parent
    data_passed = bool(source_paths) and all(path.is_file() for path in source_paths)
    end_stage(timer_path, "data_audit", data_passed)
    result["stage_results"]["data_audit"] = {"passed": data_passed, "source_hashes": {str(path): _sha256(path) for path in source_paths if path.is_file()}}

    baseline_output = output_dir / "baseline"
    begin_stage(timer_path, "baseline")
    baseline = _run_solver(contract, data_root, baseline_output, 1)
    baseline_passed = baseline["returncode"] == 0 and (baseline["summary"].get("final_judge") or {}).get("verdict") == "pass"
    end_stage(timer_path, "baseline", baseline_passed)
    result["stage_results"]["baseline"] = {"passed": baseline_passed, "summary": str(baseline_output / "summary.json")}

    candidate_output = output_dir / "candidate_routes"
    begin_stage(timer_path, "candidate_routes")
    candidate = _run_solver(contract, data_root, candidate_output, 4)
    candidate_passed = candidate["returncode"] == 0 and (candidate["summary"].get("final_judge") or {}).get("verdict") == "pass" and candidate["summary"].get("executed_route_count", 0) >= 2
    end_stage(timer_path, "candidate_routes", candidate_passed)
    result["stage_results"]["candidate_routes"] = {"passed": candidate_passed, "summary": str(candidate_output / "summary.json"), "executed_route_count": candidate["summary"].get("executed_route_count", 0)}

    begin_stage(timer_path, "independent_validation")
    validation = candidate_output / "domain_verification" / "domain_verification.json"
    validation_payload = json.loads(validation.read_text(encoding="utf-8")) if validation.is_file() else {}
    validation_passed = validation_payload.get("passed") is True
    end_stage(timer_path, "independent_validation", validation_passed)
    result["stage_results"]["independent_validation"] = {"passed": validation_passed, "evidence": str(validation)}

    begin_stage(timer_path, "paper")
    report = TOOLS / "run" / "v44_real_benchmarks" / "CUMCM2023C_complete_report.md"
    paper_passed = report.is_file() and report.stat().st_size > 0
    end_stage(timer_path, "paper", paper_passed)
    result["stage_results"]["paper"] = {"passed": paper_passed, "report": str(report), "sha256": _sha256(report) if report.is_file() else ""}

    begin_stage(timer_path, "packaging")
    archive = output_dir / "technical_rehearsal_support.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in (candidate_output / "summary.json", validation, report):
            if path.is_file():
                handle.write(path, path.name)
    with zipfile.ZipFile(archive) as handle:
        packaging_passed = handle.testzip() is None and len(handle.namelist()) >= 2
    end_stage(timer_path, "packaging", packaging_passed)
    result["stage_results"]["packaging"] = {"passed": packaging_passed, "archive": str(archive), "sha256": _sha256(archive)}

    begin_stage(timer_path, "human_review")
    human_review_passed = all(result["stage_results"][stage]["passed"] for stage in STAGES[:-1])
    end_stage(timer_path, "human_review", human_review_passed)
    result["stage_results"]["human_review"] = {"passed": human_review_passed, "scope": "technical artifact review only; no formal anonymity or contestant attestation"}

    timing = finalize(timer_path)
    result["timing"] = timing
    result["passed"] = timing["passed"]
    result["claim_level"] = "historical_technical_end_to_end_timed_rehearsal"
    result["limitations"] = ["This timer measures a historical technical rehearsal, not the formal 2026 contest.", "Human review stage confirms artifact presence only; it is not a contestant compliance attestation.", "The report and support archive are technical rehearsal artifacts, not a final submission package."]
    (output_dir / "timed_rehearsal_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a timed historical technical rehearsal.")
    parser.add_argument("--reference-root", type=Path, default=SKILL_ROOT / "data" / "reference_root")
    parser.add_argument("--output-dir", type=Path, default=TOOLS / "run" / "historical_timed_rehearsal")
    args = parser.parse_args()
    print(json.dumps(run_rehearsal(args.reference_root, args.output_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
