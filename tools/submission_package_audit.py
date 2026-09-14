from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any

MAX_BYTES = 20 * 1024 * 1024
SOURCE_SUFFIXES = {".py", ".r", ".m", ".ipynb", ".jl", ".cpp", ".c", ".java", ".sas", ".sps"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paper_text(path: Path) -> tuple[list[str], list[str]]:
    pdfinfo = shutil.which("pdfinfo")
    pdftotext = shutil.which("pdftotext")
    if not pdfinfo or not pdftotext:
        return [], ["poppler_tools_missing"]
    try:
        info = subprocess.run([pdfinfo, str(path)], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        match = re.search(r"^Pages:\s+(\d+)$", info.stdout, flags=re.MULTILINE)
        if info.returncode != 0 or not match:
            return [], ["paper_pdf_info_failed"]
        page_count = int(match.group(1))
        extracted = subprocess.run([pdftotext, "-layout", str(path), "-"], capture_output=True,
                                   text=True, encoding="utf-8", errors="replace", timeout=120)
        if extracted.returncode != 0:
            return [], ["paper_text_extraction_failed"]
        pages = extracted.stdout.split("\f")
        if pages and not pages[-1].strip():
            pages.pop()
        pages = (pages + [""] * page_count)[:page_count]
    except (OSError, subprocess.SubprocessError, ValueError):
        return [], ["paper_pdf_unreadable"]
    issues: list[str] = []
    if not pages or not any(text.strip() for text in pages):
        issues.append("paper_text_layer_missing")
    return pages, issues


def audit_submission_package(
    paper: Path,
    support_archive: Path,
    *,
    forbidden_terms: list[str] | None = None,
    expected_support_files: list[str] | None = None,
) -> dict[str, Any]:
    paper = paper.resolve()
    support_archive = support_archive.resolve()
    issues: list[str] = []
    warnings: list[str] = []
    forbidden_terms = [term.strip() for term in (forbidden_terms or []) if term.strip()]
    expected_support_files = [name.replace("\\", "/") for name in (expected_support_files or [])]

    if not paper.is_file():
        issues.append("paper_missing")
        pages: list[str] = []
    else:
        if paper.suffix.lower() != ".pdf":
            issues.append("paper_not_pdf")
        if paper.stat().st_size > MAX_BYTES:
            issues.append("paper_exceeds_20mb")
        pages, paper_issues = _paper_text(paper)
        issues.extend(paper_issues)

    full_text = "\n".join(pages)
    # PDF text extraction may insert spacing between CJK glyphs; use a
    # whitespace-collapsed copy for structural headings while preserving the
    # original text for identity-term checks.
    first_page = pages[0] if pages else ""
    normalized_text = re.sub(r"\s+", "", full_text)
    normalized_first_page = re.sub(r"\s+", "", first_page)
    if pages and "摘要" not in first_page:
        issues.append("paper_first_page_not_abstract_by_text")
    if pages and ("承诺书" in first_page or "编号专用页" in first_page):
        issues.append("paper_first_page_contains_excluded_front_matter")
    if "目录" in full_text[: max(1, len(full_text) // 4)]:
        issues.append("paper_table_of_contents_detected")

    ai_index = normalized_text.find("AI工具使用声明")
    reference_index = normalized_text.find("参考文献")
    if ai_index < 0:
        issues.append("paper_ai_declaration_missing")
    if reference_index < 0:
        issues.append("paper_references_heading_missing")
    if ai_index >= 0 and reference_index >= 0 and ai_index > reference_index:
        issues.append("paper_ai_declaration_not_before_references")

    appendix_page = next((index for index, text in enumerate(pages) if "附录" in text), None)
    body_pages = appendix_page if appendix_page is not None else len(pages)
    if body_pages > 30:
        issues.append("paper_body_exceeds_30_pages")
    if appendix_page is None:
        warnings.append("appendix_heading_not_detected")
    for term in forbidden_terms:
        if term.casefold() in full_text.casefold():
            issues.append(f"paper_forbidden_identity_term:{term}")

    archive_names: list[str] = []
    if not support_archive.is_file():
        issues.append("support_archive_missing")
    else:
        if support_archive.stat().st_size > MAX_BYTES:
            issues.append("support_archive_exceeds_20mb")
        if support_archive.suffix.lower() != ".zip":
            issues.append("support_archive_not_machine_auditable_zip")
        else:
            try:
                with zipfile.ZipFile(support_archive) as archive:
                    bad = archive.testzip()
                    if bad:
                        issues.append(f"support_archive_corrupt_member:{bad}")
                    archive_names = [name.replace("\\", "/") for name in archive.namelist() if not name.endswith("/")]
            except (OSError, zipfile.BadZipFile) as exc:
                issues.append(f"support_archive_unreadable:{type(exc).__name__}")

    basenames = {Path(name).name for name in archive_names}
    if "AI工具使用详情.pdf" not in basenames:
        issues.append("ai_usage_detail_pdf_missing_from_support")
    if not any(Path(name).suffix.lower() in SOURCE_SUFFIXES for name in archive_names):
        issues.append("runnable_source_not_detected_in_support")
    for expected in expected_support_files:
        if expected not in archive_names:
            issues.append(f"expected_support_file_missing:{expected}")
        elif Path(expected).name not in full_text:
            issues.append(f"support_file_not_listed_in_paper_appendix:{expected}")
    for term in forbidden_terms:
        if any(term.casefold() in name.casefold() for name in archive_names):
            issues.append(f"support_forbidden_identity_term:{term}")

    unique_issues = sorted(set(issues))
    return {
        "version": "v44-readiness-1", "kind": "submission_package_audit",
        "passed": not unique_issues, "verdict": "PASS" if not unique_issues else "FAIL",
        "issues": unique_issues, "warnings": sorted(set(warnings)),
        "paper": {"path": str(paper), "sha256": _sha256(paper) if paper.is_file() else "",
                  "bytes": paper.stat().st_size if paper.is_file() else None,
                  "page_count": len(pages), "body_page_count_by_heading": body_pages},
        "support_archive": {"path": str(support_archive),
                            "sha256": _sha256(support_archive) if support_archive.is_file() else "",
                            "bytes": support_archive.stat().st_size if support_archive.is_file() else None,
                            "file_count": len(archive_names), "files": archive_names},
        "image_recognition_used": False, "visual_layout_verified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit CUMCM submission files without OCR or image recognition.")
    parser.add_argument("paper", type=Path)
    parser.add_argument("support_archive", type=Path)
    parser.add_argument("--forbidden-term", action="append", default=[])
    parser.add_argument("--expected-support-file", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_submission_package(args.paper, args.support_archive,
                                      forbidden_terms=args.forbidden_term,
                                      expected_support_files=args.expected_support_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
