"""Create a conservative, text-layer-only contest intake manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


QUESTION_RE = re.compile(r"(?:问题|第)\s*([1-9][0-9]*)\s*(?:问|题)?")
STRUCTURED = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".txt", ".md"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pdf_text(path: Path) -> tuple[str, list[str]]:
    pdftotext = shutil.which("pdftotext")
    if not pdftotext:
        return "", ["pdftotext_missing"]
    try:
        result = subprocess.run([pdftotext, "-layout", str(path), "-"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    except (OSError, subprocess.SubprocessError):
        return "", ["pdf_text_extraction_failed"]
    if result.returncode != 0:
        return "", ["pdf_text_extraction_failed"]
    return result.stdout, []


def _text_for(path: Path) -> tuple[str, list[str]]:
    if path.suffix.lower() == ".pdf":
        return _pdf_text(path)
    if path.suffix.lower() in {".csv", ".tsv", ".json", ".txt", ".md"}:
        try:
            return path.read_text(encoding="utf-8", errors="replace"), []
        except OSError:
            return "", ["text_read_failed"]
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return "", ["structured_spreadsheet_requires_parser"]
    return "", ["binary_or_unparsed_file"]


def build_intake(input_dir: Path, output_dir: Path, problem_id: str) -> dict[str, Any]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    files: list[dict[str, Any]] = []
    question_ids: set[str] = set()
    for path in sorted(item for item in input_dir.rglob("*") if item.is_file()):
        text, warnings = _text_for(path)
        matches = sorted({f"question_{int(number):02d}" for number in QUESTION_RE.findall(text)})
        question_ids.update(matches)
        files.append({
            "relative_path": str(path.relative_to(input_dir)).replace("\\", "/"),
            "path": str(path),
            "suffix": path.suffix.lower(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "text_layer_read": bool(text),
            "question_ids_detected": matches,
            "warnings": warnings,
        })
    questions = sorted(question_ids)
    result: dict[str, Any] = {
        "version": "v44-intake-1",
        "kind": "competition_intake_manifest",
        "problem_id": problem_id,
        "input_dir": str(input_dir),
        "files": files,
        "expected_question_ids": questions,
        "question_count": len(questions),
        "contracts": [
            {"question_id": question_id, "status": "needs_input", "human_confirmed": False, "notes": ["Draft created from text-layer detection only; fill semantics, units, constraints and outputs manually."]}
            for question_id in questions
        ],
        "image_recognition_used": False,
        "human_review_required": True,
        "warnings": sorted({warning for item in files for warning in item["warnings"]}),
    }
    target = output_dir / "competition_intake_manifest.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a text-layer-only contest intake manifest.")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--problem-id", required=True)
    args = parser.parse_args()
    print(json.dumps(build_intake(args.input_dir, args.output_dir, args.problem_id), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
