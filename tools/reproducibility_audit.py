"""Validate evidence that a modeling run can be reproduced.

The audit requires a recorded command, environment, input/output hashes and a
successful rerun comparison. It does not execute arbitrary commands itself.
The team must run the declared command in the declared environment and record
the result before final submission.
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


def _binding(path: Path, base: Path, issues: list[str], label: str) -> None:
    if not isinstance(path, Path) or not path.is_file():
        issues.append(f"{label}_missing")


def audit_reproducibility(path: Path, output: Path) -> dict[str, Any]:
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("reproducibility_not_object")
    issues: list[str] = []
    if payload.get("kind") != "reproducibility_audit":
        issues.append("reproducibility_kind_invalid")
    if payload.get("passed") is not True:
        issues.append("reproducibility_not_passed")
    for key in ("command", "rerun_command"):
        value = payload.get(key)
        if not isinstance(value, list) or not value or not all(str(item).strip() for item in value):
            issues.append(f"{key}_missing_or_invalid")
    if not isinstance(payload.get("environment"), dict) or not payload.get("environment"):
        issues.append("environment_missing_or_invalid")
    if payload.get("rerun_passed") is not True:
        issues.append("rerun_not_passed")
    comparison = payload.get("comparison")
    if not isinstance(comparison, dict) or comparison.get("passed") is not True or not comparison.get("checks"):
        issues.append("rerun_comparison_missing_or_not_passed")
    if not isinstance(comparison, dict) or comparison.get("output_hashes_match") is not True:
        issues.append("rerun_output_hash_comparison_missing_or_failed")
    base = path.parent
    for key in ("inputs", "outputs"):
        bindings = payload.get(key)
        if not isinstance(bindings, list) or not bindings:
            issues.append(f"{key}_bindings_missing")
            continue
        for index, item in enumerate(bindings):
            if not isinstance(item, dict) or not str(item.get("path", "")).strip() or not str(item.get("sha256", "")).strip():
                issues.append(f"{key}_binding_invalid:{index}")
                continue
            raw = Path(str(item["path"]))
            target = raw.resolve() if raw.is_absolute() else (base / raw).resolve()
            if not target.is_file():
                issues.append(f"{key}_file_missing:{index}")
            elif _sha256(target) != str(item["sha256"]).lower():
                issues.append(f"{key}_hash_mismatch:{index}")
    rerun_outputs = payload.get("rerun_outputs")
    outputs = payload.get("outputs")
    if not isinstance(rerun_outputs, list) or not rerun_outputs:
        issues.append("rerun_outputs_bindings_missing")
    elif isinstance(outputs, list):
        expected = {(str(item.get("path", "")), str(item.get("sha256", "")).lower()) for item in outputs if isinstance(item, dict)}
        actual = {(str(item.get("path", "")), str(item.get("sha256", "")).lower()) for item in rerun_outputs if isinstance(item, dict)}
        if expected != actual:
            issues.append("rerun_output_bindings_do_not_match_original")
    result = {
        "version": "v44-reproducibility-audit-1",
        "kind": "reproducibility_audit",
        "source_path": str(path),
        "passed": not issues,
        "verdict": "PASS" if not issues else "INCOMPLETE",
        "issues": sorted(set(issues)),
    }
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit reproducibility evidence.")
    parser.add_argument("evidence", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit_reproducibility(args.evidence, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
