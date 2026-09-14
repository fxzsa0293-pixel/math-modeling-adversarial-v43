"""Smoke and negative tests for the self-contained V38 skill package."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = ROOT.parent / "data" / "reference_root"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v38_explicit_role_gate import run_v38_explicit_role_gate


def test_v38_self_contained_runner_passes() -> None:
    if not REFERENCE_ROOT.exists() or not any(REFERENCE_ROOT.rglob("*")):
        pytest.skip("local historical reference corpus is not installed")
    root = ROOT
    proc = subprocess.run(
        [sys.executable, "run.py", "--self-contained"],
        cwd=root,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=240,
    )
    assert proc.returncode == 0, proc.stderr
    summary = json.loads((root / "run" / "v38_self_contained" / "summary.json").read_text(encoding="utf-8"))
    assert summary["gate"]["verdict"] == "pass"
    assert summary["case_count"] == 11
    assert summary["entrypoint"] == "run.py"


def test_v38_recommended_routes_exist_and_paths_valid() -> None:
    if not REFERENCE_ROOT.exists() or not any(REFERENCE_ROOT.rglob("*")):
        pytest.skip("local historical reference corpus is not installed")
    root = ROOT
    result = json.loads((root / "run" / "v38_self_contained" / "v38" / "v38_explicit_role_gate.json").read_text(encoding="utf-8"))
    for audit in result["case_audits"]:
        assert audit["recommended_route"] in audit["route_ids"], audit["case"]
        assert "data_path_not_found" not in audit["audit_flags"], audit["case"]
        assert "missing_limitations" not in audit["audit_flags"], audit["case"]
        assert "missing_route_role" not in audit["audit_flags"], audit["case"]


def _toy_suite(route_name: str = "candidate", recommended: str = "candidate") -> dict:
    return {
        "results": [
            {
                "case": "toy_case",
                "status": "executed",
                "archetypes": ["forecasting"],
                "data_path": __file__,
                "routes": [
                    {"route": "mean_fake_baseline_name", "metrics": {"MAE": 1.0}},
                    {"route": route_name, "metrics": {"MAE": 2.0}},
                ],
                "recommended_route": recommended,
                "limitations": ["toy limitation"],
            }
        ]
    }


def test_missing_role_is_hard_fail_even_if_name_contains_baseline() -> None:
    result = run_v38_explicit_role_gate([_toy_suite()], route_contract={})
    assert result["gate"]["verdict"] == "revise"
    assert "missing_route_role" in result["gate"]["critical_flags"]


def test_recommended_route_not_in_routes_is_hard_fail() -> None:
    contract = {"toy_case": {"mean_fake_baseline_name": "baseline", "candidate": "candidate"}}
    result = run_v38_explicit_role_gate([_toy_suite(recommended="not_present")], route_contract=contract)
    assert result["gate"]["verdict"] == "revise"
    assert "recommended_route_not_in_routes" in result["gate"]["critical_flags"]


def test_missing_primary_metric_is_not_filled_with_default() -> None:
    suite = {
        "results": [
            {
                "case": "toy_case",
                "status": "executed",
                "archetypes": ["multi_criteria_evaluation"],
                "data_path": __file__,
                "routes": [
                    {"route": "equal_weight", "metrics": {"role": "old_fake"}},
                    {"route": "entropy", "metrics": {"mean_rank_correlation": 0.9}},
                ],
                "recommended_route": "entropy",
                "limitations": ["toy limitation"],
            }
        ]
    }
    contract = {"toy_case": {"equal_weight": "baseline", "entropy": "candidate"}}
    result = run_v38_explicit_role_gate([suite], route_contract=contract)
    assert result["gate"]["verdict"] == "revise"
    audit = result["case_audits"][0]
    eliminated = {route["route"]: route for route in audit["eliminated_routes"]}
    assert "equal_weight" in eliminated
    assert "missing_primary_metric" in eliminated["equal_weight"]["reasons"]


def test_self_prediction_route_removed_from_2012a_v36() -> None:
    if not REFERENCE_ROOT.exists() or not any(REFERENCE_ROOT.rglob("*")):
        pytest.skip("local historical reference corpus is not installed")
    root = ROOT
    result = json.loads((root / "run" / "v38_self_contained" / "v38" / "v38_explicit_role_gate.json").read_text(encoding="utf-8"))
    wine = next(audit for audit in result["case_audits"] if audit["case"] == "CUMCM2012A_wine_evaluation_ranking_v36")
    assert "second_panel_baseline" not in wine["route_ids"]
