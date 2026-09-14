from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from .schemas import ProblemContract


def run_domain_verifier(
    script: Path,
    contract: ProblemContract,
    registry_path: Path,
    output_dir: Path,
    timeout_seconds: int = 120,
) -> Dict[str, Any]:
    script = Path(script).resolve()
    registry_path = Path(registry_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / "contract_for_verifier.json"
    contract_path.write_text(json.dumps(asdict(contract), ensure_ascii=False, indent=2), encoding="utf-8")
    command = [sys.executable, str(script), "--contract", str(contract_path), "--registry", str(registry_path)]
    verifier_env = os.environ.copy()
    for name in ("DASHSCOPE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        verifier_env.pop(name, None)
    try:
        proc = subprocess.run(
            command,
            cwd=script.parent,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            env=verifier_env,
            timeout=timeout_seconds,
        )
        stdout, stderr, exit_code = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = _timeout_text(exc.stdout)
        stderr = _timeout_text(exc.stderr)
        exit_code = None
    stdout_path = output_dir / "domain_verifier.stdout.txt"
    stderr_path = output_dir / "domain_verifier.stderr.txt"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    payload: Dict[str, Any] = {}
    error = ""
    if exit_code == 0:
        try:
            payload = json.loads(stdout[stdout.index("{"):])
            if (
                not isinstance(payload.get("passed"), bool)
                or not isinstance(payload.get("issues"), list)
                or not isinstance(payload.get("checks"), list)
            ):
                raise ValueError("verifier output requires boolean passed plus issues and checks lists")
        except Exception as exc:
            error = f"invalid_verifier_output:{exc}"
    else:
        error = f"verifier_exit_code:{exit_code}"
    records = json.loads(registry_path.read_text(encoding="utf-8"))
    run_ids = {record.get("run_id") for record in records if record.get("status") == "executed"}
    result = {
        "passed": bool(exit_code == 0 and not error and payload.get("passed")),
        "issues": list(payload.get("issues", [])) + ([error] if error else []),
        "checks": payload.get("checks", []),
        "run_id": next(iter(run_ids)) if len(run_ids) == 1 else None,
        "contract_hash": _json_hash(asdict(contract)),
        "script_path": str(script),
        "script_hash": _sha256(script),
        "registry_path": str(registry_path),
        "registry_hash": _sha256(registry_path),
        "stdout_path": str(stdout_path),
        "stdout_hash": _sha256(stdout_path),
        "stderr_path": str(stderr_path),
        "stderr_hash": _sha256(stderr_path),
        "exit_code": exit_code,
        "command": command,
    }
    (output_dir / "domain_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _timeout_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
