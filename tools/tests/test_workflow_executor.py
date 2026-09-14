from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workflow.problem_dag import verify_workflow_manifest
from workflow.workflow_executor import execute_workflow, verify_assembled_report
from workflow.mechanism_model_builder import build_cumcm2018a_mechanism_data, verify_mechanism_data_manifest


def _forecast_contract(path: Path, data_file: Path, *, status: str = "ready") -> None:
    payload = {
        "version": "v44", "problem_id": "workflow_forecast",
        "task_type": "forecasting", "status": status,
        "data_file": str(data_file), "sheet_name": None,
        "target_column": "sales", "time_column": "date",
        "feature_columns": [], "known_future_columns": [],
        "metric_directions": {"MAE": "lower_better", "RMSE": "lower_better", "WAPE": "lower_better"},
        "expected_outputs": ["forecast", "paper_results_section"],
        "forecast_horizon": 4, "seasonal_period": 4,
        "validation_mode": "fixed_origin_holdout",
        "source": "provided",
        "unresolved_fields": [] if status == "ready" else ["target_requires_confirmation"],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _workflow(path: Path, contract: Path) -> None:
    payload = {
        "version": "v44", "problem_id": "multi_question_toy",
        "nodes": [
            {
                "node_id": "forecast", "kind": "prediction",
                "contract_path": str(contract), "max_routes": 4,
                "outputs": {"paper_section": "workflow_exports.paper_results_section_md"},
            },
            {
                "node_id": "paper", "kind": "paper", "dependencies": ["forecast"],
                "inputs": {"forecast_section": {"from_node": "forecast", "output": "paper_section"}},
                "outputs": {"report": "report_path", "report_manifest": "manifest_path"},
            },
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_executable_workflow_runs_nodes_and_dynamic_workspaces(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    frame = pd.DataFrame({"date": range(80), "sales": [100 + i + (i % 4) * 3 for i in range(80)]})
    data_file = data_root / "sales.csv"
    frame.to_csv(data_file, index=False)
    contract = tmp_path / "forecast_contract.json"
    workflow = tmp_path / "workflow.json"
    _forecast_contract(contract, data_file)
    _workflow(workflow, contract)

    result = execute_workflow(workflow, tmp_path / "run", max_routes=4, timeout_seconds=120)

    assert result["passed"] is True
    assert result["topological_order"] == ["forecast", "paper"]
    assert {node["question_id"] for node in result["nodes"]} == {"question_01", "question_02"}
    stable_result = tmp_path / "run" / "competition_workspace" / "work" / "question_01" / "results" / "result_summary.json"
    assert stable_result.is_file()
    report_manifest = Path(result["nodes"][1]["output_artifacts"]["report_manifest"])
    assert verify_assembled_report(report_manifest)["passed"] is True
    manifest_path = Path(result["manifest"]["manifest_path"])
    assert verify_workflow_manifest(manifest_path)["passed"] is True

    judge_path = tmp_path / "run" / "nodes" / "forecast" / "judge" / "final_judge.json"
    original_judge = judge_path.read_text(encoding="utf-8")
    judge_path.write_text(original_judge + " ", encoding="utf-8")
    assert verify_workflow_manifest(manifest_path)["passed"] is False
    judge_path.write_text(original_judge, encoding="utf-8")
    assert verify_workflow_manifest(manifest_path)["passed"] is True

    report_path = Path(result["nodes"][1]["output_artifacts"]["report"])
    report_path.write_text(report_path.read_text(encoding="utf-8") + "mutation", encoding="utf-8")
    assert verify_assembled_report(report_manifest)["passed"] is False
    assert verify_workflow_manifest(manifest_path)["passed"] is False


def test_executable_workflow_blocks_downstream_after_failed_gate(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    data_file = data_root / "sales.csv"
    pd.DataFrame({"date": range(30), "sales": range(30)}).to_csv(data_file, index=False)
    contract = tmp_path / "blocked_contract.json"
    workflow = tmp_path / "workflow.json"
    _forecast_contract(contract, data_file, status="needs_input")
    _workflow(workflow, contract)

    result = execute_workflow(workflow, tmp_path / "run", max_routes=2, timeout_seconds=60)

    assert result["passed"] is False
    assert result["nodes"][0]["passed"] is False
    assert "solver_verdict:blocked" in result["nodes"][0]["verification"]["issues"]
    assert result["nodes"][1]["passed"] is False
    assert "upstream_execution_not_passed" in result["nodes"][1]["verification"]["issues"]
    assert not (tmp_path / "run" / "nodes" / "paper" / "complete_report.md").exists()


def test_executable_workflow_injects_upstream_artifact_into_downstream_contract(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    sales = data_root / "sales.csv"
    pd.DataFrame({"date": range(80), "sales": [50 + i + (i % 4) for i in range(80)]}).to_csv(sales, index=False)
    forecast_contract = tmp_path / "forecast.json"
    _forecast_contract(forecast_contract, sales)
    evaluation_contract = tmp_path / "evaluation.json"
    evaluation_contract.write_text(json.dumps({
        "version": "v44", "problem_id": "workflow_evaluation",
        "task_type": "evaluation_ranking", "status": "ready",
        "data_file": str(sales), "sheet_name": None,
        "indicator_directions": {"metric_value": "lower_better", "improvement_vs_baseline": "higher_better"},
        "metric_directions": {"ranking_stability": "higher_better"},
        "expected_outputs": ["ranking", "paper_results_section"],
        "validation_mode": "stability_audit", "source": "provided", "unresolved_fields": [],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    workflow = tmp_path / "workflow.json"
    workflow.write_text(json.dumps({
        "version": "v44", "problem_id": "artifact_injection",
        "nodes": [
            {
                "node_id": "forecast", "kind": "prediction", "contract_path": str(forecast_contract),
                "max_routes": 8, "outputs": {"metrics": "workflow_exports.route_comparison_csv"},
            },
            {
                "node_id": "evaluate", "kind": "analysis", "dependencies": ["forecast"],
                "contract_path": str(evaluation_contract), "max_routes": 2,
                "inputs": {"metrics": {"from_node": "forecast", "output": "metrics", "contract_field": "data_file"}},
                "outputs": {"result": "workflow_exports.result_summary_json"},
            },
        ],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    result = execute_workflow(workflow, tmp_path / "run", max_routes=8, timeout_seconds=120)

    assert result["passed"] is True
    downstream = result["nodes"][1]
    upstream_metrics = result["nodes"][0]["output_artifacts"]["metrics"]
    effective_contract = json.loads((tmp_path / "run" / "nodes" / "evaluate" / "effective_contract.json").read_text(encoding="utf-8"))
    assert Path(effective_contract["data_file"]).resolve() == Path(upstream_metrics).resolve()
    assert Path(downstream["input_artifacts"]["metrics"]).resolve() == Path(upstream_metrics).resolve()
    assert downstream["execution"]["verdict"] == "pass"


def test_executable_workflow_reuses_only_fully_verified_dag(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    data_file = data_root / "sales.csv"
    frame = pd.DataFrame({"date": range(80), "sales": [100 + i + (i % 4) for i in range(80)]})
    frame.to_csv(data_file, index=False)
    contract = tmp_path / "forecast_contract.json"
    workflow = tmp_path / "workflow.json"
    _forecast_contract(contract, data_file)
    _workflow(workflow, contract)
    output = tmp_path / "run"

    first = execute_workflow(workflow, output, max_routes=3, timeout_seconds=120)
    first_summary = json.loads((output / "nodes" / "forecast" / "summary.json").read_text(encoding="utf-8"))
    first_run_id = first_summary["final_judge"]["run_id"]
    second = execute_workflow(workflow, output, max_routes=3, timeout_seconds=120)

    assert first["passed"] is True
    assert second["passed"] is True and second["reused"] is True
    assert all(node["execution"]["reused"] is True for node in second["nodes"])
    unchanged_summary = json.loads((output / "nodes" / "forecast" / "summary.json").read_text(encoding="utf-8"))
    assert unchanged_summary["final_judge"]["run_id"] == first_run_id

    changed_config = execute_workflow(workflow, output, max_routes=4, timeout_seconds=120)
    changed_summary = json.loads((output / "nodes" / "forecast" / "summary.json").read_text(encoding="utf-8"))
    assert changed_config.get("reused") is not True
    assert changed_summary["final_judge"]["run_id"] != first_run_id
    config_run_id = changed_summary["final_judge"]["run_id"]

    frame.loc[len(frame)] = [80, 181]
    frame.to_csv(data_file, index=False)
    third = execute_workflow(workflow, output, max_routes=4, timeout_seconds=120)
    refreshed_summary = json.loads((output / "nodes" / "forecast" / "summary.json").read_text(encoding="utf-8"))
    assert third["passed"] is True and third.get("reused") is not True
    assert refreshed_summary["final_judge"]["run_id"] != config_run_id


def test_cumcm2018a_mechanism_manifest_recomputes_source_and_rejects_mutation(tmp_path: Path) -> None:
    source = tmp_path / "CUMCM2018A_data.xlsx"
    materials = pd.DataFrame([
        ["title", None, None, None, None],
        ["layer", "density", "heat", "conductivity", "thickness"],
        ["I", 300, 1377, 0.082, 0.6],
        ["II", 862, 2100, 0.37, "0.6-25"],
        ["III", 74.2, 1726, 0.045, 3.6],
        ["IV", 1.18, 1005, 0.028, "0.6-6.4"],
    ])
    temperature = pd.DataFrame([["title", None], ["time", "temp"]] + [[i, 37 + i / 10] for i in range(12)])
    with pd.ExcelWriter(source) as writer:
        materials.to_excel(writer, sheet_name="Appendix 1", header=False, index=False)
        temperature.to_excel(writer, sheet_name="Appendix 2", header=False, index=False)
    built = build_cumcm2018a_mechanism_data(source, tmp_path / "derived")
    manifest = Path(built["manifest_path"])
    assert verify_mechanism_data_manifest(manifest, built["model"])["passed"] is True

    derived = Path(built["data_file"])
    derived.write_bytes(derived.read_bytes() + b"0,0\n")
    verification = verify_mechanism_data_manifest(manifest, built["model"])
    assert verification["passed"] is False
    assert "mechanism_temperature_hash_mismatch" in verification["issues"]
