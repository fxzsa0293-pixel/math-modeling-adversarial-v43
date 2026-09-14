from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from semantic_review_form import apply_review_form, create_review_form


def _contract(path: Path) -> None:
    path.write_text(json.dumps({
        "problem_id": "demo", "question_id": "q1", "status": "needs_input",
        "unresolved_fields": ["semantic_review"],
        "expected_outputs": ["answer"], "units": {"answer": "kg"},
        "constraints": [{"name": "nonnegative"}],
    }), encoding="utf-8")


def test_create_and_apply_semantic_review(tmp_path: Path) -> None:
    contract = tmp_path / "contract.json"
    _contract(contract)
    form_path = tmp_path / "review.json"
    form = create_review_form(contract, form_path)
    assert form["kind"] == "semantic_review_form"
    form["confirmations"] = {key: True for key in form["confirmations"]}
    form["reviewers"] = ["team"]
    form["reviewed_at"] = "2026-09-06T12:00:00+08:00"
    form_path.write_text(json.dumps(form), encoding="utf-8")
    result = apply_review_form(contract, form_path, tmp_path / "ready.json")
    assert result["passed"] is True
    assert result["contract"]["status"] == "ready"
    assert result["contract"]["semantic_review"]["human_confirmed"] is True


def test_apply_rejects_unconfirmed_form(tmp_path: Path) -> None:
    contract = tmp_path / "contract.json"
    _contract(contract)
    form_path = tmp_path / "review.json"
    create_review_form(contract, form_path)
    result = apply_review_form(contract, form_path, tmp_path / "blocked.json")
    assert result["passed"] is False
    assert "confirmation_missing:human_confirmed" in result["issues"]
