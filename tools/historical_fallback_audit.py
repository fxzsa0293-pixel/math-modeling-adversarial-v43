"""Build evidence for independently executed primary/fallback route pairs.

The audit is intentionally conservative: it proves that both routes were
executed against the same benchmark registry and that their evidence files
exist.  It does not claim that a live failure-triggered switch was rehearsed.
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


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _record_for_route(registry: list[dict[str, Any]], route_id: str) -> dict[str, Any] | None:
    for record in registry:
        if str(record.get("route_id", "")) == route_id:
            return record
    return None


def _choose_pair(records: list[dict[str, Any]], recommended: str) -> tuple[str, str] | None:
    executed = [r for r in records if r.get("status") == "executed" and str(r.get("route_id", "")).strip()]
    if len(executed) < 2:
        return None
    primary = recommended if any(str(r.get("route_id")) == recommended for r in executed) else str(executed[0]["route_id"])
    preferred = [r for r in executed if str(r.get("route_id")) != primary and str(r.get("role", "")) == "baseline"]
    fallback_record = (preferred or [r for r in executed if str(r.get("route_id")) != primary])[0]
    return primary, str(fallback_record["route_id"])


def _case_evidence(case: dict[str, Any]) -> dict[str, Any]:
    summary_path = Path(str(case.get("summary_path", ""))).resolve()
    summary = _load(summary_path) if summary_path.is_file() else {}
    registry_path = summary_path.parent / "registry" / "experiment_registry.json"
    registry_payload = _load(registry_path) if registry_path.is_file() else []
    records = registry_payload if isinstance(registry_payload, list) else []
    recommended = str((summary.get("recommended_route") or {}).get("route_id", ""))
    pair = _choose_pair(records, recommended)
    result: dict[str, Any] = {
        "case_id": str(case.get("case_id", "")),
        "summary_path": str(summary_path),
        "registry_path": str(registry_path),
        "primary_route": pair[0] if pair else recommended,
        "fallback_route": pair[1] if pair else "",
        "same_registry": bool(pair and registry_path.is_file()),
        "primary_executed": False,
        "fallback_executed": False,
        "route_evidence": {},
        "passed": False,
        "claim_level": "both_routes_executed_same_registry",
        "limitations": ["This proves prior execution of both routes, not a failure-triggered live switch."],
    }
    if not pair:
        result["limitations"].append("At least two executed routes were not found.")
        return result
    for label, route_id in (("primary", pair[0]), ("fallback", pair[1])):
        record = _record_for_route(records, route_id) or {}
        evidence_path = Path(str((record.get("evidence_paths") or {}).get("route_evidence", ""))).resolve()
        exists = evidence_path.is_file()
        result[f"{label}_executed"] = record.get("status") == "executed"
        result["route_evidence"][label] = {
            "route_id": route_id,
            "status": record.get("status", ""),
            "path": str(evidence_path),
            "exists": exists,
            "sha256": _sha256(evidence_path) if exists else "",
            "metrics": record.get("metrics", {}),
        }
    result["passed"] = bool(result["same_registry"] and result["primary_executed"] and result["fallback_executed"] and all(item.get("exists") for item in result["route_evidence"].values()))
    return result


def audit_fallbacks(benchmark_summary: Path, output_dir: Path) -> dict[str, Any]:
    benchmark_summary = benchmark_summary.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _load(benchmark_summary)
    cases = payload.get("cases", []) if isinstance(payload, dict) else []
    case_evidence = [_case_evidence(case) for case in cases if isinstance(case, dict)]
    eligible = [item for item in case_evidence if item.get("primary_route") and item.get("fallback_route")]
    result = {
        "version": "v44-fallback-audit-1",
        "kind": "historical_fallback_audit",
        "source_summary": {"path": str(benchmark_summary), "sha256": _sha256(benchmark_summary) if benchmark_summary.is_file() else ""},
        "passed": bool(eligible) and all(item.get("passed") is True for item in eligible),
        "eligible_case_count": len(eligible),
        "skipped_case_ids": [item.get("case_id") for item in case_evidence if item not in eligible],
        "claim_level": "both_routes_executed_same_registry",
        "cases": case_evidence,
        "limitations": ["No case here simulates an injected primary-route failure followed by an automatic switch."],
    }
    (output_dir / "historical_fallback_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit executed primary/fallback route pairs.")
    parser.add_argument("--benchmark-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit_fallbacks(args.benchmark_summary, args.output_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
