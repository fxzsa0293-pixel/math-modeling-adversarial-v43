import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from competition_intake import build_intake


def test_intake_hashes_files_and_detects_questions_without_images(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "题目.txt").write_text("问题一：说明。\n问题2：计算。\n", encoding="utf-8")
    (input_dir / "附件.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    result = build_intake(input_dir, tmp_path / "out", "demo")
    assert result["expected_question_ids"] == ["question_02"]
    assert result["image_recognition_used"] is False
    assert result["contracts"][0]["status"] == "needs_input"
    assert all(item["sha256"] for item in result["files"])
    saved = json.loads((tmp_path / "out" / "competition_intake_manifest.json").read_text(encoding="utf-8"))
    assert saved["problem_id"] == "demo"


def test_intake_records_unparsed_binary_warning(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "figure.png").write_bytes(b"not an image parser input")
    result = build_intake(input_dir, tmp_path / "out", "demo")
    assert "binary_or_unparsed_file" in result["warnings"]
    assert result["question_count"] == 0


def test_intake_marks_spreadsheets_for_structured_parser(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "附件.xlsx").write_bytes(b"xlsx-bytes")
    result = build_intake(input_dir, tmp_path / "out", "demo")
    assert "structured_spreadsheet_requires_parser" in result["warnings"]
    assert result["files"][0]["text_layer_read"] is False
