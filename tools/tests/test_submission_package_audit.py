from __future__ import annotations

import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import submission_package_audit


def _pdf(path: Path) -> Path:
    path.write_bytes(b"%PDF-1.4\n% test fixture; extraction is mocked\n")
    return path


def _patch_text(monkeypatch, texts: list[str]) -> None:
    monkeypatch.setattr(submission_package_audit, "_paper_text", lambda path: (texts, []))


def _support(path: Path, include_ai: bool = True) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("code/solve.py", "print(1)")
        if include_ai:
            archive.writestr("AI工具使用详情.pdf", "details")
    return path


def test_valid_text_and_zip_package_passes(tmp_path: Path, monkeypatch) -> None:
    paper = _pdf(tmp_path / "paper.pdf")
    _patch_text(monkeypatch, ["摘要\n正文", "AI工具使用声明\n参考文献\n附录\nsolve.py\nAI工具使用详情.pdf"])
    support = _support(tmp_path / "support.zip")
    result = submission_package_audit.audit_submission_package(
        paper, support, expected_support_files=["code/solve.py", "AI工具使用详情.pdf"]
    )
    assert result["passed"] is True
    assert result["visual_layout_verified"] is False


def test_cjk_heading_spacing_is_normalized(tmp_path: Path, monkeypatch) -> None:
    paper = _pdf(tmp_path / "paper.pdf")
    _patch_text(monkeypatch, ["摘 要\n正文", "AI 工具使用声明\n参考 文献\n附 录\nsolve.py"])
    support = _support(tmp_path / "support.zip")
    result = submission_package_audit.audit_submission_package(
        paper, support, expected_support_files=["code/solve.py"]
    )
    assert "paper_ai_declaration_missing" not in result["issues"]
    assert "paper_references_heading_missing" not in result["issues"]


def test_missing_ai_detail_and_identity_term_fail(tmp_path: Path, monkeypatch) -> None:
    paper = _pdf(tmp_path / "paper.pdf")
    _patch_text(monkeypatch, ["摘要\n某大学\n参考文献\n附录"])
    support = _support(tmp_path / "support.zip", include_ai=False)
    result = submission_package_audit.audit_submission_package(paper, support, forbidden_terms=["某大学"])
    assert "paper_ai_declaration_missing" in result["issues"]
    assert "paper_forbidden_identity_term:某大学" in result["issues"]
    assert "ai_usage_detail_pdf_missing_from_support" in result["issues"]


def test_body_page_limit_and_appendix_file_list_are_checked(tmp_path: Path, monkeypatch) -> None:
    paper = _pdf(tmp_path / "paper.pdf")
    _patch_text(monkeypatch, ["摘要"] + ["正文"] * 30 + ["AI工具使用声明\n参考文献\n附录"])
    support = _support(tmp_path / "support.zip")
    result = submission_package_audit.audit_submission_package(paper, support, expected_support_files=["code/solve.py"])
    assert "paper_body_exceeds_30_pages" in result["issues"]
    assert "support_file_not_listed_in_paper_appendix:code/solve.py" in result["issues"]


def test_rar_is_not_claimed_machine_auditable(tmp_path: Path, monkeypatch) -> None:
    paper = _pdf(tmp_path / "paper.pdf")
    _patch_text(monkeypatch, ["摘要\nAI工具使用声明\n参考文献\n附录"])
    rar = tmp_path / "support.rar"
    rar.write_bytes(b"not parsed")
    result = submission_package_audit.audit_submission_package(paper, rar)
    assert "support_archive_not_machine_auditable_zip" in result["issues"]
