"""Rehearse failure-triggered selection using existing verified route artifacts.

The primary callback is intentionally failed.  The fallback callback reads a
previously produced route-evidence JSON, so this tests switching semantics
without fabricating a numerical model result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fallback_switch import run_with_fallback


def run_rehearsal(fallback_evidence: Path, output_path: Path) -> dict:
    fallback_evidence = fallback_evidence.resolve()
    output_path = output_path.resolve()
    if not fallback_evidence.is_file():
        raise FileNotFoundError(fallback_evidence)

    def primary():
        raise RuntimeError("injected_primary_failure_for_rehearsal")

    def fallback():
        payload = json.loads(fallback_evidence.read_text(encoding="utf-8"))
        return {
            "evidence_path": str(fallback_evidence),
            "route_id": payload.get("route_id"),
            "metrics_recomputed": payload.get("metrics_recomputed", {}),
        }

    result = run_with_fallback(
        "injected_primary_route",
        str(json.loads(fallback_evidence.read_text(encoding="utf-8")).get("route_id", "fallback_route")),
        primary,
        fallback,
        output_path,
    )
    result.update({
        "claim_level": "failure_injected_switch_to_existing_fallback_artifact",
        "fallback_evidence_path": str(fallback_evidence),
        "fallback_evidence_exists": fallback_evidence.is_file(),
    })
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a failure-injected fallback rehearsal.")
    parser.add_argument("--fallback-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_rehearsal(args.fallback_evidence, args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
