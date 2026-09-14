"""Audit historical real-data rehearsal evidence without claiming contest readiness.

This audit is deliberately narrower than ``competition_readiness.py``.  It
answers: which reusable technical stages have actually been exercised, and
which competition-stage artifacts are still absent?  It never fabricates
elapsed time or human attestations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        return {"path": str(path), "exists": False, "sha256": ""}
    return {"path": str(path), "exists": True, "sha256": _sha256(path), "bytes": path.stat().st_size}


def audit_rehearsal(benchmark_summary: Path, cross_summary: Path, output_dir: Path, fallback_audit: Path | None = None, switch_rehearsal: Path | None = None, timed_rehearsal: Path | None = None, package_audit: Path | None = None) -> dict[str, Any]:
    benchmark_summary = benchmark_summary.resolve()
    cross_summary = cross_summary.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    primary = _load(benchmark_summary)
    cross = _load(cross_summary)
    evidence: dict[str, Any] = {
        "primary_benchmark_summary": _file_record(benchmark_summary),
        "cross_archetype_summary": _file_record(cross_summary),
    }
    completed: list[str] = []
    missing: list[str] = []
    warnings: list[str] = []

    if primary.get("all_passed") is True and int(primary.get("passed_cases", 0)) == int(primary.get("case_count", -1)):
        completed.append("real_data_multi_case_benchmark")
    else:
        missing.append("real_data_multi_case_benchmark")
    if cross.get("passed") is True and int(cross.get("passed_cases", 0)) == int(cross.get("case_count", -1)):
        completed.append("cross_archetype_benchmark")
    else:
        missing.append("cross_archetype_benchmark")

    report_path = Path(str((primary.get("complete_report") or {}).get("path", "")))
    if report_path.is_file() and (primary.get("complete_report") or {}).get("verification", {}).get("passed") is True:
        completed.append("complete_markdown_report")
        evidence["complete_markdown_report"] = _file_record(report_path)
    else:
        missing.append("complete_markdown_report")

    workflow_path = Path(str((primary.get("workflow_dag") or {}).get("manifest_path", "")))
    if workflow_path.is_file() and (primary.get("workflow_dag") or {}).get("passed") is True:
        completed.append("workflow_dag_and_manifest")
        evidence["workflow_manifest"] = _file_record(workflow_path)
    else:
        missing.append("workflow_dag_and_manifest")

    if fallback_audit is not None:
        fallback_payload = _load(fallback_audit.resolve())
        evidence["historical_fallback_audit"] = _file_record(fallback_audit.resolve())
        if fallback_payload.get("passed") is True:
            completed.append("tested_primary_fallback_pairs")
        else:
            missing.append("tested_primary_fallback_pairs")
    else:
        missing.append("tested_primary_fallback_pairs")

    if switch_rehearsal is not None:
        switch_payload = _load(switch_rehearsal.resolve())
        evidence["failure_injected_switch"] = _file_record(switch_rehearsal.resolve())
        if switch_payload.get("passed") is True and switch_payload.get("status") == "fallback_succeeded":
            completed.append("failure_injected_fallback_switch")
        else:
            missing.append("failure_injected_fallback_switch")
    else:
        missing.append("failure_injected_fallback_switch")

    if timed_rehearsal is not None:
        timed_payload = _load(timed_rehearsal.resolve())
        evidence["timed_rehearsal"] = _file_record(timed_rehearsal.resolve())
        timing = timed_payload.get("timing") or {}
        if timed_payload.get("passed") is True and timing.get("passed") is True:
            completed.append("eight_stage_timed_rehearsal")
        else:
            missing.append("eight_stage_timed_rehearsal")
    else:
        missing.append("eight_stage_timed_rehearsal")

    if package_audit is not None:
        package_payload = _load(package_audit.resolve())
        evidence["submission_package_audit"] = _file_record(package_audit.resolve())
        if package_payload.get("passed") is True:
            completed.append("historical_submission_package_audit")
        else:
            missing.append("historical_submission_package_audit")
    else:
        missing.append("historical_submission_package_audit")

    # These are intentionally evidence-based.  Existing benchmark timestamps
    # are model-run durations, not a complete eight-stage contest timer.
    for label in (
        "eight_stage_timing",
        "final_paper_submission_pdf",
        "support_archive",
        "ai_usage_detail_pdf",
        "human_layout_and_anonymity_review",
        "contest_conduct_attestation",
    ):
        missing.append(label)
    warnings.append("Historical benchmark evidence is not a formal-contest readiness decision.")
    warnings.append("Benchmark elapsed_seconds are solver runtimes, not end-to-end stage timings.")
    warnings.append("No image recognition or OCR evidence is inferred by this audit.")

    result = {
        "version": "v44-rehearsal-audit-1",
        "kind": "historical_rehearsal_audit",
        "ready_for_formal_contest": False,
        "verdict": "TECHNICAL_REHEARSAL_PARTIAL",
        "completed_capabilities": sorted(set(completed)),
        "missing_competition_evidence": sorted(set(missing)),
        "warnings": sorted(set(warnings)),
        "primary_suite": {
            "suite": primary.get("suite", ""),
            "case_count": primary.get("case_count", 0),
            "passed_cases": primary.get("passed_cases", 0),
        },
        "cross_suite": {
            "suite": cross.get("suite", ""),
            "case_count": cross.get("case_count", 0),
            "passed_cases": cross.get("passed_cases", 0),
            "scope_boundary": (cross.get("case") or {}).get("scope_boundary", ""),
        },
        "evidence": evidence,
    }
    output = output_dir / "historical_rehearsal_audit.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit historical benchmark rehearsal evidence.")
    parser.add_argument("--benchmark-summary", type=Path, required=True)
    parser.add_argument("--cross-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fallback-audit", type=Path)
    parser.add_argument("--switch-rehearsal", type=Path)
    parser.add_argument("--timed-rehearsal", type=Path)
    parser.add_argument("--package-audit", type=Path)
    args = parser.parse_args()
    result = audit_rehearsal(args.benchmark_summary, args.cross_summary, args.output_dir, args.fallback_audit, args.switch_rehearsal, args.timed_rehearsal, args.package_audit)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
