"""Tests for V43 automatic solver search and credibility constraints."""
from __future__ import annotations

import json
import hashlib
import subprocess
import sys
import importlib.util
import pytest
from pathlib import Path

import numpy as np
import pandas as pd
from dataclasses import asdict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solver.experiment_runner import _validate_and_recompute, run_experiments
from solver.final_judge import judge_solution
from solver.problem_parser import parse_problem
from solver.schemas import ExperimentRecord, ProblemProfile
from solver.schemas import DataAsset, ProblemContract
from solver.problem_contract import load_or_infer_contract, validate_contract
from solver.adapter_generator import _adapter_source, generate_adapters
from solver.schemas import RouteSpec
from solver.route_planner import plan_routes
from solver.domain_verifier import run_domain_verifier
from solver.native_optimization import solve_declared_model, solve_network_model
from solver.native_mechanism import solve_mechanism_model, verify_native_mechanism
from solver.native_spatial import run_native_spatial_experiments, solve_spatial_model, verify_native_spatial


def test_v41_solver_runs_search_routes(tmp_path: Path) -> None:
    data_root = tmp_path / "problem_data"
    data_root.mkdir()
    frame = pd.DataFrame({
        "week": list(range(1, 61)),
        "price": [10 + i * 0.3 for i in range(60)],
        "promotion": [1 if i % 7 in (0, 1) else 0 for i in range(60)],
        "sales": [100 + i * 2 + (i % 3) * 5 + (8 if i % 7 in (0, 1) else 0) for i in range(60)],
    })
    frame.to_csv(data_root / "sales.csv", index=False, encoding="utf-8")
    brief = tmp_path / "brief.txt"
    brief.write_text("Retail forecasting problem: predict future sales using historical price and promotion data.", encoding="utf-8")
    out = tmp_path / "run"
    proc = subprocess.run(
        [sys.executable, "run_v39_solver.py", "--problem-id", "toy_retail", "--data-root", str(data_root), "--brief", str(brief), "--output-dir", str(out), "--max-routes", "12"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=240,
    )
    assert proc.returncode == 0, proc.stderr
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["version"] == "v44"
    workspace = out / "competition_workspace"
    assert (workspace / "work" / "data_audit" / "data_audit.json").exists()
    assert (workspace / "work" / "data_audit" / "数据体检报告.md").exists()
    assert (workspace / "work" / "plan" / "求解计划.md").exists()
    assert (workspace / "reports" / "route_comparison.csv").exists()
    assert (workspace / "reports" / "result_summary.json").exists()
    assert (workspace / "reports" / "figure_requirements.md").exists()
    registry = json.loads((out / "registry" / "experiment_registry.json").read_text(encoding="utf-8"))
    metrics_by_route = {record["route_id"]: record["metrics"] for record in registry if record["status"] == "executed"}
    assert "train_selection_final_test_recursive" in {metrics.get("validation_split") for metrics in metrics_by_route.values()}
    assert all("selection_MAE" in metrics and "final_test_MAE" in metrics for metrics in metrics_by_route.values())
    assert {metrics.get("target") for metrics in metrics_by_route.values()} == {"sales"}
    assert "tuned_ridge_lag_search" in metrics_by_route
    assert "random_forest_feature_search" in metrics_by_route
    assert "gradient_boosting_feature_search" in metrics_by_route
    assert "elastic_net_feature_search" in metrics_by_route
    assert "extra_trees_feature_search" in metrics_by_route
    assert "model_family_ensemble" in metrics_by_route
    assert "residual_hybrid_forecast" in metrics_by_route
    assert metrics_by_route["tuned_ridge_lag_search"].get("selected_config")
    assert metrics_by_route["random_forest_feature_search"].get("selected_config")
    assert metrics_by_route["gradient_boosting_feature_search"].get("selected_config")
    assert all(record.get("original_code_hash") and record.get("final_code_hash") for record in registry)


def test_run_py_v44_default_output_dir_is_not_v38(tmp_path: Path) -> None:
    data_root = tmp_path / "problem_data"
    data_root.mkdir()
    pd.DataFrame({"x1": range(20), "target": range(5, 25)}).to_csv(data_root / "table.csv", index=False)
    proc = subprocess.run(
        [sys.executable, "run.py", "--v39-solver", "--problem-id", "toy_generic", "--data-root", str(data_root), "--max-routes", "4"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    summary = json.loads((ROOT / "run" / "v44_auto_solver" / "summary.json").read_text(encoding="utf-8"))
    assert summary["entrypoint"] == "run_v39_solver.py"
    assert "v44_auto_solver" in summary["output_dir"]


def test_parser_prefers_larger_relevant_table_and_encoding_fallback(tmp_path: Path) -> None:
    data_root = tmp_path / "problem_data"
    data_root.mkdir()
    pd.DataFrame({"id": range(9), "value": range(9)}).to_csv(data_root / "tiny.csv", index=False, encoding="utf-8")
    pd.DataFrame({"date": range(60), "sales": range(100, 160), "price": range(60)}).to_csv(data_root / "sales_data.csv", index=False, encoding="gb18030")
    profile = parse_problem("retail_forecast", data_root)
    assert "sales" in profile.target_candidates
    assert Path(profile.assets[0].path).name == "sales_data.csv"
    assert profile.assets[0].row_count == 60


def test_target_unknown_blocks_arbitrary_regression(tmp_path: Path) -> None:
    data_root = tmp_path / "problem_data"
    data_root.mkdir()
    pd.DataFrame({"feature_a": range(30), "feature_b": range(30, 60)}).to_csv(data_root / "features.csv", index=False)
    out = tmp_path / "run"
    proc = subprocess.run(
        [sys.executable, "run_v39_solver.py", "--problem-id", "coords_only", "--data-root", str(data_root), "--output-dir", str(out), "--max-routes", "4"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    judge = json.loads((out / "judge" / "final_judge.json").read_text(encoding="utf-8"))
    assert judge["verdict"] == "blocked"
    assert judge["gate_layers"]["contract"]["pass"] is False


def test_timeout_is_recorded(tmp_path: Path) -> None:
    adapter = tmp_path / "slow.py"
    adapter.write_text("import time\ntime.sleep(5)\n", encoding="utf-8")
    records = run_experiments([adapter], tmp_path / "out", timeout_seconds=1, max_repairs=0)
    assert records[0].status == "failed"
    assert "timeout_after_1s" in records[0].error
    assert records[0].repair_log


def test_v41_evaluation_routes_include_pca_and_stability(tmp_path: Path) -> None:
    data_root = tmp_path / "problem_data"
    data_root.mkdir()
    pd.DataFrame({
        "score_quality": [90, 80, 70, 60, 50, 40, 30, 20, 10, 5],
        "cost": [10, 20, 30, 40, 50, 60, 70, 80, 90, 95],
        "stability": [88, 82, 75, 63, 52, 45, 33, 22, 12, 6],
    }).to_csv(data_root / "ranking.csv", index=False)
    contract = tmp_path / "problem_contract.json"
    contract.write_text(json.dumps({
        "version": "v44",
        "problem_id": "evaluation_ranking_demo",
        "task_type": "evaluation_ranking",
        "status": "ready",
        "data_file": str((data_root / "ranking.csv").resolve()),
        "indicator_directions": {"score_quality": "higher_better", "cost": "lower_better", "stability": "higher_better"},
        "source": "provided",
    }), encoding="utf-8")
    out = tmp_path / "run"
    proc = subprocess.run(
        [sys.executable, "run_v39_solver.py", "--problem-id", "evaluation_ranking_demo", "--data-root", str(data_root), "--contract", str(contract), "--output-dir", str(out), "--max-routes", "8"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    registry = json.loads((out / "registry" / "experiment_registry.json").read_text(encoding="utf-8"))
    executed = {record["route_id"] for record in registry if record["status"] == "executed"}
    assert "pca_factor_ranking" in executed
    assert "stability_weighted_ranking" in executed


def test_final_judge_rejects_tiny_gain() -> None:
    contract = ProblemContract("v44", "tiny_gain", "forecasting", "ready", target_column="target", time_column="time")
    profile = ProblemProfile("tiny_gain", "", "", [], ["forecasting"], [], ["target"], ["MAE"], {"MAE": "lower_better"}, [], contract)
    records = [
        ExperimentRecord("tiny_gain", "baseline", "baseline", "executed", "", "", {"MAE": 100.0, "validation_split": "fixed_origin_recursive"}),
        ExperimentRecord("tiny_gain", "candidate", "candidate", "executed", "", "", {"MAE": 99.5, "validation_split": "fixed_origin_recursive"}),
    ]
    result = judge_solution(profile, [], records, Path.cwd() / "run" / "test_tiny_gain", min_improvement=0.01)
    assert result["verdict"] == "revise"
    assert result["recommended_route"]["route_id"] == "baseline"
    assert any(flag.startswith("missing_evidence") for flag in result["flags"])


def test_contract_target_is_invariant_to_column_order(tmp_path: Path) -> None:
    brief = "Retail forecasting problem: predict future sales using historical price."
    paths = []
    for name, columns in (("a.csv", ["week", "price", "sales"]), ("b.csv", ["sales", "week", "price"])):
        path = tmp_path / name
        pd.DataFrame({column: range(30) for column in columns}).to_csv(path, index=False)
        paths.append(DataAsset(str(path), ".csv", path.stat().st_size, columns=columns, row_count=30, column_count=3, readable=True))
    for asset in paths:
        contract = load_or_infer_contract("retail", [asset], ["forecasting"], ["price", "sales"], brief)
        assert contract.target_column == "sales"
        assert contract.status == "ready"


def test_tampered_evidence_is_rejected(tmp_path: Path) -> None:
    data = tmp_path / "data.csv"
    data.write_text("x,target\n1,2\n", encoding="utf-8")
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}", encoding="utf-8")
    contract = ProblemContract("v44", "tamper", "regression", "ready", data_file=str(data), target_column="target")
    profile = ProblemProfile("tamper", "", str(tmp_path), [], ["generic_tabular_modeling"], [], ["target"], ["MAE"], {"MAE": "lower_better"}, [], contract)
    record = ExperimentRecord("tamper", "baseline", "baseline", "executed", "", str(data), {"MAE": 1.0}, run_id="run", input_hash="x", config_hash="y", evidence_paths={"route_evidence": str(evidence)}, artifact_hashes={"route_evidence": "wrong"})
    result = judge_solution(profile, [], [record], tmp_path / "judge")
    assert result["verdict"] == "revise"
    assert any(flag.startswith("invalid_evidence") for flag in result["flags"])


def test_recursive_forecast_does_not_read_held_out_targets(tmp_path: Path) -> None:
    path = tmp_path / "sales.csv"
    asset = DataAsset(str(path), ".csv", 1, columns=["week", "price", "sales"], row_count=30, column_count=3, readable=True)
    contract = ProblemContract("v44", "leakage", "forecasting", "ready", data_file=str(path), target_column="sales", time_column="week", forecast_horizon=5, validation_mode="fixed_origin_holdout")
    profile = ProblemProfile("leakage", "", str(tmp_path), [], ["forecasting"], [asset], ["sales"], ["MAE"], {"MAE": "lower_better"}, [], contract)
    route = RouteSpec("regularized_lag_model", "candidate", "forecasting", "", ["MAE"])
    namespace = {"__name__": "audit_only"}
    exec(_adapter_source(profile, route), namespace)
    base = pd.DataFrame({"week": range(30), "price": [10 + i / 10 for i in range(30)], "sales": [100 + i for i in range(30)]})
    changed = base.copy()
    changed.loc[25:, "sales"] = [1000, 2000, 3000, 4000, 5000]
    changed.loc[25:, "price"] = [100, 200, 300, 400, 500]
    _, first = namespace["run_regression_route"](route.route_id, route.role, route.family, base)
    _, second = namespace["run_regression_route"](route.route_id, route.role, route.family, changed)
    assert first["predicted"] == second["predicted"]


def test_topsis_respects_lower_better_cost_direction(tmp_path: Path) -> None:
    path = tmp_path / "ranking.csv"
    asset = DataAsset(str(path), ".csv", 1, columns=["quality", "cost"], row_count=5, column_count=2, readable=True)
    contract = ProblemContract("v44", "ranking", "evaluation_ranking", "ready", data_file=str(path), indicator_directions={"quality": "higher_better", "cost": "lower_better"})
    profile = ProblemProfile("ranking", "", str(tmp_path), [], ["evaluation_ranking"], [asset], [], ["ranking_stability"], {"ranking_stability": "higher_better"}, [], contract)
    route = RouteSpec("topsis_ranking", "candidate", "evaluation", "", ["ranking_stability"])
    namespace = {"__name__": "audit_only"}
    exec(_adapter_source(profile, route), namespace)
    frame = pd.DataFrame({"quality": [10, 10, 8, 6, 4], "cost": [1, 10, 5, 5, 5]})
    _, evidence = namespace["run_evaluation_route"](route.route_id, route.role, frame)
    assert evidence["scores"][0] > evidence["scores"][1]


def test_auto_evaluation_requires_direction_confirmation(tmp_path: Path) -> None:
    path = tmp_path / "ranking.csv"
    asset = DataAsset(str(path), ".csv", 1, columns=["quality", "cost"], row_count=20, column_count=2, readable=True)
    contract = load_or_infer_contract("ranking", [asset], ["evaluation_ranking"], [], "综合评价并排名")
    assert contract.status == "needs_input"
    assert "indicator_directions_confirmation" in contract.unresolved_fields


def test_multiple_data_files_require_explicit_selection(tmp_path: Path) -> None:
    assets = []
    for name in ("a.csv", "b.csv"):
        path = tmp_path / name
        assets.append(DataAsset(str(path), ".csv", 1, columns=["x", "target"], row_count=20, column_count=2, readable=True))
    contract = load_or_infer_contract("regression", assets, ["generic_tabular_modeling"], ["target"], "predict target")
    assert contract.status == "needs_input"
    assert "data_file_selection" in contract.unresolved_fields


def test_forecast_training_and_inference_features_match() -> None:
    contract = ProblemContract("v44", "feature_parity", "forecasting", "ready", data_file="x.csv", target_column="sales", time_column="week", feature_columns=["price"], known_future_columns=["price"], forecast_horizon=2, validation_mode="fixed_origin_holdout")
    profile = ProblemProfile("feature_parity", "", "", [], ["forecasting"], [], ["sales"], ["MAE"], {"MAE": "lower_better"}, [], contract)
    namespace = {"__name__": "audit_only"}
    exec(_adapter_source(profile, RouteSpec("regularized_lag_model", "candidate", "forecasting", "", ["MAE"])), namespace)
    frame = pd.DataFrame({"week": range(12), "price": [10 + value for value in range(12)], "sales": [100 + 2 * value for value in range(12)]})
    X, _, features, engineered, spec = namespace["engineer_features"](frame, "sales", True)
    medians = np.zeros(len(features))
    row = namespace["forecast_feature_row"](frame.iloc[8], 8, frame["sales"].iloc[:8].tolist(), "sales", features, medians, spec)
    expected = engineered.loc[8, features].to_numpy(dtype=float)
    assert list(row[0]) == list(expected)


def test_contract_task_type_overrides_keyword_archetype() -> None:
    contract = ProblemContract("v44", "contract_router", "regression", "ready", data_file="x.csv", target_column="target", validation_mode="random_holdout")
    profile = ProblemProfile("contract_router", "", "", [], ["forecasting", "retail_forecast_operation"], [], ["target"], ["MAE"], {"MAE": "lower_better"}, [], contract)
    routes = plan_routes(profile, [])
    assert routes[0].route_id == "simple_statistical_baseline"
    assert all(route.family != "forecasting" for route in routes)


def test_runner_rejects_observed_values_not_in_bound_data(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    pd.DataFrame({"x": [1, 2, 3], "target": [10, 20, 30]}).to_csv(path, index=False)
    contract = ProblemContract("v44", "evidence", "regression", "ready", data_file=str(path), target_column="target", validation_mode="random_holdout")
    route = RouteSpec("simple_statistical_baseline", "baseline", "baseline", "", ["MAE"])
    import hashlib
    payload = {"problem_id": "evidence", "route_id": route.route_id, "role": route.role, "status": "executed", "data_path": str(path), "metrics": {"MAE": 0}, "evidence": {"observed": [999.0], "predicted": [999.0], "sample_rows": [0], "target_column": "target"}, "contract": contract.__dict__, "route": route.__dict__, "input_hash_at_load": hashlib.sha256(path.read_bytes()).hexdigest()}
    import pytest
    with pytest.raises(ValueError, match="observed_values_do_not_match_bound_data"):
        _validate_and_recompute(payload)


def test_contract_rejects_target_as_known_future(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    pd.DataFrame({"time": range(10), "target": range(10)}).to_csv(path, index=False)
    asset = DataAsset(str(path), ".csv", path.stat().st_size, columns=["time", "target"], row_count=10, column_count=2, readable=True)
    contract_path = tmp_path / "contract.json"
    payload = ProblemContract("v44", "bad_future", "forecasting", "ready", data_file=str(path), target_column="target", time_column="time", feature_columns=["target"], known_future_columns=["target"], forecast_horizon=2, validation_mode="fixed_origin_holdout").__dict__
    contract_path.write_text(json.dumps(payload), encoding="utf-8")
    contract = load_or_infer_contract("bad_future", [asset], ["forecasting"], ["target"], "", contract_path)
    assert contract.status == "invalid"
    assert "target_cannot_be_known_future" in contract.unresolved_fields


def test_multi_agent_gate_requires_current_bindings_and_verifier(tmp_path: Path) -> None:
    script_path = ROOT.parent / "scripts" / "run_multi_agent.py"
    spec = importlib.util.spec_from_file_location("run_multi_agent", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    data = tmp_path / "data.csv"
    code = tmp_path / "route.py"
    data.write_text("x\n1\n", encoding="utf-8")
    code.write_text("print('ok')\n", encoding="utf-8")
    gate_path = tmp_path / "gate.json"
    bound_gate = {
        "verdict": "pass",
        "problem_id": "p",
        "run_id": "r",
        "contract_hash": "c",
        "input_bindings": [{"path": str(data), "sha256": module.sha256(data)}],
        "code_bindings": [{"path": str(code), "sha256": module.sha256(code)}],
        "gate_layers": {"evidence": {"pass": True}},
    }
    gate_path.write_text(json.dumps(bound_gate), encoding="utf-8")
    assert module.validate_machine_gate(bound_gate, gate_path) == (True, [])
    code.write_text("print('changed')\n", encoding="utf-8")
    passed, issues = module.validate_machine_gate(bound_gate, gate_path)
    assert not passed
    assert any(issue.startswith("gate_binding_mismatch:code_bindings") for issue in issues)
    assert module.validate_machine_gate({"verdict": "pass"}, gate_path)[0] is False


def test_empty_execution_cannot_show_vacuous_gate_pass(tmp_path: Path) -> None:
    contract = ProblemContract("v44", "empty", "regression", "ready", data_file="missing.csv", target_column="target", validation_mode="random_holdout")
    profile = ProblemProfile("empty", "", "", [], [], [], ["target"], ["MAE"], {"MAE": "lower_better"}, [], contract)
    result = judge_solution(profile, [], [], tmp_path / "judge")
    assert result["verdict"] == "revise"
    assert all(layer["pass"] is False for name, layer in result["gate_layers"].items() if name != "contract")


def test_regression_uses_disjoint_selection_and_final_test(tmp_path: Path) -> None:
    path = tmp_path / "regression.csv"
    pd.DataFrame({"x": range(50), "target": [3 * value + 2 for value in range(50)]}).to_csv(path, index=False)
    asset = DataAsset(str(path), ".csv", path.stat().st_size, columns=["x", "target"], row_count=50, column_count=2, readable=True)
    contract = ProblemContract("v44", "regression_split", "regression", "ready", data_file=str(path), target_column="target", feature_columns=["x"], validation_mode="random_holdout")
    profile = ProblemProfile("regression_split", "", str(tmp_path), [], ["generic_tabular_modeling"], [asset], ["target"], ["MAE"], {"MAE": "lower_better"}, [], contract)
    adapter = tmp_path / "ridge.py"
    adapter.write_text(_adapter_source(profile, RouteSpec("regularized_regression_cv", "candidate", "tabular", "", ["MAE"])), encoding="utf-8")
    record = run_experiments([adapter], tmp_path / "experiments")[0]
    assert record.status == "executed", record.error
    assert record.metrics["validation_split"] == "train_selection_final_test_random"
    assert "selection_MAE" in record.metrics and "final_test_MAE" in record.metrics
    evidence = json.loads(Path(record.evidence_paths["route_evidence"]).read_text(encoding="utf-8"))["evidence"]
    assert set(evidence["sample_rows"]).isdisjoint(evidence["final_test_sample_rows"])


def test_final_test_reversal_prevents_candidate_recommendation() -> None:
    contract = ProblemContract("v44", "reversal", "regression", "ready", data_file="x.csv", target_column="target", validation_mode="random_holdout")
    profile = ProblemProfile("reversal", "", "", [], [], [], ["target"], ["MAE"], {"MAE": "lower_better"}, [], contract)
    baseline = ExperimentRecord("reversal", "baseline", "baseline", "executed", "", "", {"MAE": 10.0, "selection_MAE": 10.0, "final_test_MAE": 10.0})
    candidate = ExperimentRecord("reversal", "candidate", "candidate", "executed", "", "", {"MAE": 5.0, "selection_MAE": 5.0, "final_test_MAE": 12.0})
    from solver.final_judge import _recommend, _route_scores
    scores = _route_scores([baseline, candidate], "MAE", profile.metric_directions, baseline)
    assert _recommend(profile, scores, baseline, 0.01)["route_id"] == "baseline"


def test_final_test_does_not_select_second_best_candidate() -> None:
    contract = ProblemContract("v44", "no_test_selection", "regression", "ready", data_file="x.csv", target_column="target", validation_mode="random_holdout")
    profile = ProblemProfile("no_test_selection", "", "", [], [], [], ["target"], ["MAE"], {"MAE": "lower_better"}, [], contract)
    records = [
        ExperimentRecord("no_test_selection", "baseline", "baseline", "executed", "", "", {"selection_MAE": 10.0, "final_test_MAE": 10.0}),
        ExperimentRecord("no_test_selection", "winner", "candidate", "executed", "", "", {"selection_MAE": 4.0, "final_test_MAE": 12.0}),
        ExperimentRecord("no_test_selection", "runner_up", "candidate", "executed", "", "", {"selection_MAE": 5.0, "final_test_MAE": 6.0}),
    ]
    from solver.final_judge import _recommend, _route_scores
    scores = _route_scores(records, "MAE", profile.metric_directions, records[0])
    assert _recommend(profile, scores, records[0], 0.01)["route_id"] == "baseline"


def test_parser_records_full_row_count_beyond_sample_limit(tmp_path: Path) -> None:
    path = tmp_path / "large.csv"
    pd.DataFrame({"x": range(750), "target": range(750)}).to_csv(path, index=False)
    profile = parse_problem("large", tmp_path, None)
    assert profile.assets[0].row_count == 750


def test_forecast_rejects_duplicate_time_values() -> None:
    contract = ProblemContract("v44", "duplicate_time", "forecasting", "ready", data_file="x.csv", target_column="sales", time_column="date", forecast_horizon=2, validation_mode="fixed_origin_holdout")
    profile = ProblemProfile("duplicate_time", "", "", [], [], [], ["sales"], ["MAE"], {"MAE": "lower_better"}, [], contract)
    namespace = {"__name__": "audit_only"}
    exec(_adapter_source(profile, RouteSpec("regularized_lag_model", "candidate", "forecasting", "", ["MAE"])), namespace)
    frame = pd.DataFrame({"date": ["2026-01-01", "2026-01-01"] + [f"2026-01-{day:02d}" for day in range(2, 12)], "sales": range(12)})
    import pytest
    with pytest.raises(RuntimeError, match="forecast_time_column_contains_duplicates"):
        namespace["run_regression_route"]("regularized_lag_model", "candidate", "forecasting", frame)


def test_group_holdout_keeps_entities_disjoint(tmp_path: Path) -> None:
    path = tmp_path / "grouped.csv"
    frame = pd.DataFrame({"group": [f"g{group}" for group in range(10) for _ in range(6)], "x": range(60), "target": [value * 2 for value in range(60)]})
    frame.to_csv(path, index=False)
    contract = ProblemContract("v44", "grouped", "regression", "ready", data_file=str(path), target_column="target", group_columns=["group"], feature_columns=["x"], validation_mode="group_holdout")
    profile = ProblemProfile("grouped", "", str(tmp_path), [], [], [], ["target"], ["MAE"], {"MAE": "lower_better"}, [], contract)
    adapter = tmp_path / "grouped.py"
    adapter.write_text(_adapter_source(profile, RouteSpec("regularized_regression_cv", "candidate", "tabular", "", ["MAE"])), encoding="utf-8")
    record = run_experiments([adapter], tmp_path / "experiments")[0]
    assert record.status == "executed", record.error
    assert record.metrics["validation_split"] == "group_holdout"
    evidence = json.loads(Path(record.evidence_paths["route_evidence"]).read_text(encoding="utf-8"))["evidence"]
    groups = [set(evidence[name]) for name in ("train_groups", "selection_groups", "final_test_groups")]
    assert groups[0].isdisjoint(groups[1]) and groups[0].isdisjoint(groups[2]) and groups[1].isdisjoint(groups[2])


def test_ranking_scores_are_independently_recomputed(tmp_path: Path) -> None:
    path = tmp_path / "ranking.csv"
    frame = pd.DataFrame({"quality": [9, 8, 7, 6, 5, 4], "cost": [1, 2, 4, 3, 6, 8]})
    frame.to_csv(path, index=False)
    contract = ProblemContract("v44", "ranking_recompute", "evaluation_ranking", "ready", data_file=str(path), indicator_directions={"quality": "higher_better", "cost": "lower_better"}, validation_mode="stability_audit")
    profile = ProblemProfile("ranking_recompute", "", str(tmp_path), [], [], [], [], ["ranking_stability"], {"ranking_stability": "higher_better"}, [], contract)
    namespace = {"__name__": "audit_only"}
    route = RouteSpec("topsis_ranking", "candidate", "evaluation", "", ["ranking_stability"])
    exec(_adapter_source(profile, route), namespace)
    metrics, evidence = namespace["run_evaluation_route"](route.route_id, route.role, frame)
    evidence["scores"][0] += 0.1
    import hashlib
    payload = {"problem_id": contract.problem_id, "route_id": route.route_id, "role": route.role, "status": "executed", "data_path": str(path), "metrics": metrics, "evidence": evidence, "contract": contract.__dict__, "route": route.__dict__, "input_hash_at_load": hashlib.sha256(path.read_bytes()).hexdigest()}
    import pytest
    with pytest.raises(ValueError, match="ranking_scores_not_recomputable"):
        _validate_and_recompute(payload)


def test_regression_with_missing_values_keeps_recomputable_evidence(tmp_path: Path) -> None:
    path = tmp_path / "missing.csv"
    values = [float(value) if value % 6 else np.nan for value in range(60)]
    pd.DataFrame({"x": values, "z": range(60), "target": [2 * value + 1 for value in range(60)]}).to_csv(path, index=False)
    contract = ProblemContract("v44", "missing_values", "regression", "ready", data_file=str(path), target_column="target", feature_columns=["x", "z"], validation_mode="random_holdout")
    profile = ProblemProfile("missing_values", "", str(tmp_path), [], [], [], ["target"], ["MAE"], {"MAE": "lower_better"}, [], contract)
    adapter = tmp_path / "ridge.py"
    adapter.write_text(_adapter_source(profile, RouteSpec("tuned_ridge_lag_search", "candidate", "tabular", "", ["MAE"])), encoding="utf-8")
    record = run_experiments([adapter], tmp_path / "experiments")[0]
    assert record.status == "executed", record.error
    assert np.isfinite(record.metrics["selection_MAE"]) and np.isfinite(record.metrics["final_test_MAE"])


def test_specialist_contract_plans_declared_routes(tmp_path: Path) -> None:
    data = tmp_path / "data.csv"
    adapter = tmp_path / "baseline.py"
    verifier = tmp_path / "verify.py"
    data.write_text("x\n1\n", encoding="utf-8")
    adapter.write_text("print('adapter')\n", encoding="utf-8")
    verifier.write_text("print('verifier')\n", encoding="utf-8")
    asset = DataAsset(str(data), ".csv", data.stat().st_size, columns=["x"], row_count=1, column_count=1, readable=True)
    route = {"route_id": "feasible_baseline", "role": "baseline", "family": "optimization", "adapter_path": str(adapter), "expected_metrics": ["objective", "constraint_violation"]}
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(ProblemContract("v44", "specialist", "constrained_optimization", "ready", data_file=str(data), constraints=[{"name": "capacity"}], metric_directions={"objective": "higher_better"}, specialist_routes=[route], domain_verifier=str(verifier)).__dict__), encoding="utf-8")
    contract = load_or_infer_contract("specialist", [asset], ["constrained_optimization"], [], "", contract_path)
    assert contract.status == "ready", contract.unresolved_fields
    profile = ProblemProfile("specialist", "", str(tmp_path), [], [], [asset], [], ["objective", "constraint_violation"], contract.metric_directions, [], contract)
    routes = plan_routes(profile, [])
    assert len(routes) == 1 and routes[0].adapter_path == str(adapter.resolve())


def test_specialist_code_outside_audited_roots_is_rejected(tmp_path: Path) -> None:
    data = tmp_path / "data.csv"
    data.write_text("x\n1\n", encoding="utf-8")
    outside = tmp_path.parent / "outside_adapter.py"
    outside.write_text("print('adapter')\n", encoding="utf-8")
    verifier = tmp_path / "verify.py"
    verifier.write_text("print('verifier')\n", encoding="utf-8")
    asset = DataAsset(str(data), ".csv", data.stat().st_size, columns=["x"], row_count=1, column_count=1, readable=True)
    route = {"route_id": "baseline", "role": "baseline", "family": "optimization", "adapter_path": str(outside), "expected_metrics": ["objective", "constraint_violation"]}
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(ProblemContract("v44", "outside", "constrained_optimization", "ready", data_file=str(data), constraints=[{"name": "capacity"}], metric_directions={"objective": "higher_better"}, specialist_routes=[route], domain_verifier=str(verifier)).__dict__), encoding="utf-8")
    contract = load_or_infer_contract("outside", [asset], ["constrained_optimization"], [], "", contract_path)
    assert contract.status == "invalid"
    assert "specialist_adapter_outside_audited_workspace:baseline" in contract.unresolved_fields


def test_optimization_evidence_recomputes_objective_and_violations(tmp_path: Path) -> None:
    data = tmp_path / "data.csv"
    data.write_text("x\n1\n", encoding="utf-8")
    contract = ProblemContract("v44", "optimization", "constrained_optimization", "ready", data_file=str(data), constraints=[{"name": "capacity"}], metric_directions={"objective": "higher_better"})
    route = RouteSpec("candidate", "candidate", "optimization", "", ["objective", "constraint_violation"])
    import hashlib
    payload = {"problem_id": "optimization", "route_id": "candidate", "role": "candidate", "status": "executed", "data_path": str(data), "metrics": {"objective": -1, "constraint_violation": 99}, "evidence": {"decision_variables": {"x": 2.0}, "objective_components": [{"name": "profit", "value": 12.5}, {"name": "penalty", "value": -2.5}], "constraint_checks": [{"name": "capacity", "lhs": 2.0, "sense": "<=", "rhs": 3.0, "tolerance": 1e-8}]}, "contract": contract.__dict__, "route": route.__dict__, "input_hash_at_load": hashlib.sha256(data.read_bytes()).hexdigest()}
    metrics, _ = _validate_and_recompute(payload)
    assert metrics["objective"] == 10.0
    assert metrics["constraint_violation"] == 0.0


def test_authorized_specialist_optimization_runs_end_to_end(tmp_path: Path) -> None:
    data_root = tmp_path / "problem"
    code_root = data_root / "reviewed_code"
    code_root.mkdir(parents=True)
    data = data_root / "inputs.csv"
    data.write_text("capacity,value\n3,5\n", encoding="utf-8")

    def write_adapter(path: Path, route_id: str, role: str, value: float) -> None:
        route = {
            "route_id": route_id, "role": role, "family": "optimization",
            "adapter_path": str(path), "expected_metrics": ["objective", "constraint_violation"],
        }
        contract = {
            "version": "v44", "problem_id": "knapsack", "task_type": "constrained_optimization",
            "status": "ready", "data_file": str(data), "sheet_name": None, "target_column": "",
            "time_column": "", "group_columns": [], "feature_columns": [], "known_future_columns": [],
            "indicator_directions": {}, "metric_directions": {"objective": "higher_better"}, "units": {},
            "constraints": [{"name": "capacity"}], "expected_outputs": [], "forecast_horizon": 1,
            "validation_mode": "", "specialist_routes": [], "domain_verifier": "",
            "allowed_code_roots": [], "source": "provided", "unresolved_fields": [], "notes": [],
        }
        path.write_text(
            "import hashlib, json, os\n"
            f"data_path = {str(data)!r}\nroute = {route!r}\n"
            "contract = json.load(open(os.path.join(os.path.dirname(data_path), 'problem_contract.json'), encoding='utf-8'))\n"
            "payload = {'problem_id': 'knapsack', 'route_id': route['route_id'], 'role': route['role'], 'status': 'executed', 'data_path': data_path, 'input_hash_at_load': hashlib.sha256(open(data_path, 'rb').read()).hexdigest(), 'metrics': {}, 'evidence': {'decision_variables': {'x': 2.0}, 'objective_components': [{'name': 'value', 'value': " + repr(value) + "}], 'constraint_checks': [{'name': 'capacity', 'lhs': 2.0, 'sense': '<=', 'rhs': 3.0}]}, 'contract': contract, 'route': route, 'environment': {}, 'limitations': []}\n"
            "print(json.dumps(payload))\n",
            encoding="utf-8",
        )

    baseline = code_root / "baseline.py"
    candidate = code_root / "candidate.py"
    write_adapter(baseline, "feasible_baseline", "baseline", 5.0)
    write_adapter(candidate, "improved_candidate", "candidate", 8.0)
    verifier = code_root / "verify.py"
    verifier.write_text(
        "import argparse, json\nparser = argparse.ArgumentParser(); parser.add_argument('--contract'); parser.add_argument('--registry'); args = parser.parse_args()\nrecords = json.load(open(args.registry, encoding='utf-8'))\npassed = len(records) == 2 and all(r['metrics']['constraint_violation'] == 0 for r in records)\nprint(json.dumps({'passed': passed, 'issues': [] if passed else ['infeasible_or_missing_routes'], 'checks': [{'name': 'all_routes_feasible', 'passed': passed}]}))\n",
        encoding="utf-8",
    )
    routes = [
        {"route_id": "feasible_baseline", "role": "baseline", "family": "optimization", "adapter_path": str(baseline), "expected_metrics": ["objective", "constraint_violation"]},
        {"route_id": "improved_candidate", "role": "candidate", "family": "optimization", "adapter_path": str(candidate), "expected_metrics": ["objective", "constraint_violation"]},
    ]
    contract = ProblemContract(
        "v44", "knapsack", "constrained_optimization", "ready", data_file=str(data),
        constraints=[{"name": "capacity"}], metric_directions={"objective": "higher_better"},
        specialist_routes=routes, domain_verifier=str(verifier), allowed_code_roots=[str(code_root)], source="provided",
    )
    contract_path = data_root / "problem_contract.json"
    contract_path.write_text(json.dumps(contract.__dict__), encoding="utf-8")
    out = tmp_path / "run"
    proc = subprocess.run(
        [sys.executable, "run_v39_solver.py", "--problem-id", "knapsack", "--data-root", str(data_root), "--contract", str(contract_path), "--output-dir", str(out), "--allow-specialist-code"],
        cwd=ROOT, text=True, encoding="utf-8", capture_output=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    final = json.loads((out / "judge" / "final_judge.json").read_text(encoding="utf-8"))
    assert final["verdict"] == "pass", final
    assert final["claim_level"] == "domain_verified_feasible_result"
    assert final["recommended_route"]["route_id"] == "improved_candidate"
    assert final["domain_verification"]["passed"] is True


def test_declared_milp_runs_without_specialist_code(tmp_path: Path) -> None:
    data_root = tmp_path / "problem"
    data_root.mkdir()
    data = data_root / "inputs.csv"
    data.write_text("placeholder\n1\n", encoding="utf-8")
    model = {
        "variables": [
            {"name": "x", "kind": "binary", "lower": 0, "upper": 1},
            {"name": "y", "kind": "binary", "lower": 0, "upper": 1},
        ],
        "objective": {"sense": "max", "coefficients": {"x": 5, "y": 4}},
        "linear_constraints": [{"name": "capacity", "coefficients": {"x": 3, "y": 2}, "sense": "<=", "rhs": 3}],
    }
    contract = ProblemContract(
        "v44", "native_knapsack", "constrained_optimization", "ready", data_file=str(data),
        constraints=[{"name": "capacity"}], metric_directions={"objective": "higher_better"},
        optimization_model=model, source="provided",
    )
    contract_path = data_root / "problem_contract.json"
    contract_path.write_text(json.dumps(contract.__dict__), encoding="utf-8")
    out = tmp_path / "run"
    proc = subprocess.run(
        [sys.executable, "run_v39_solver.py", "--problem-id", "native_knapsack", "--data-root", str(data_root), "--contract", str(contract_path), "--output-dir", str(out)],
        cwd=ROOT, text=True, encoding="utf-8", capture_output=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    final = json.loads((out / "judge" / "final_judge.json").read_text(encoding="utf-8"))
    assert final["verdict"] == "pass", final
    assert final["claim_level"] == "domain_verified_optimal_result"
    assert final["recommended_route"]["route_id"] == "declared_milp_solver"
    assert final["recommended_route"]["value"] == 5.0


def test_native_feasible_baseline_handles_coupling_constraints() -> None:
    contract = {
        "optimization_model": {
            "variables": [{"name": "x", "kind": "integer", "lower": 0, "upper": 10}],
            "objective": {"sense": "min", "coefficients": {"x": 1}},
            "linear_constraints": [{"name": "minimum", "coefficients": {"x": 1}, "sense": ">=", "rhs": 5}],
        }
    }
    result = solve_declared_model(contract, {"route_id": "lower_bound_feasible_baseline"})
    assert result["metrics"]["constraint_violation"] == 0
    assert result["evidence"]["decision_variables"]["x"] >= 5


def test_native_unconstrained_linear_model_is_supported() -> None:
    contract = {
        "optimization_model": {
            "variables": [{"name": "x", "kind": "continuous", "lower": 0, "upper": 10}],
            "objective": {"sense": "max", "coefficients": {"x": 1}},
            "linear_constraints": [],
        }
    }
    result = solve_declared_model(contract, {"route_id": "declared_milp_solver"})
    assert result["metrics"]["objective"] == 10
    assert result["metrics"]["optimality_proven"] is True


def test_native_infeasible_model_has_structured_status() -> None:
    contract = {
        "optimization_model": {
            "variables": [{"name": "x", "kind": "continuous", "lower": 0, "upper": 10}],
            "objective": {"sense": "min", "coefficients": {"x": 1}},
            "linear_constraints": [
                {"name": "lower", "coefficients": {"x": 1}, "sense": ">=", "rhs": 2},
                {"name": "upper", "coefficients": {"x": 1}, "sense": "<=", "rhs": 1},
            ],
        }
    }
    result = solve_declared_model(contract, {"route_id": "declared_milp_solver"})
    assert result["metrics"]["model_status"] == "infeasible"
    assert result["metrics"]["has_feasible_solution"] is False
    assert result["metrics"]["objective"] is None
    assert result["evidence"]["decision_variables"] == {}


def test_native_unbounded_model_has_structured_status() -> None:
    contract = {
        "optimization_model": {
            "variables": [{"name": "x", "kind": "continuous", "lower": 0}],
            "objective": {"sense": "max", "coefficients": {"x": 1}},
            "linear_constraints": [],
        }
    }
    result = solve_declared_model(contract, {"route_id": "declared_milp_solver"})
    assert result["metrics"]["model_status"] == "unbounded"
    assert result["metrics"]["has_feasible_solution"] is False
    assert result["metrics"]["optimality_proven"] is False


def test_native_shortest_path_preserves_network_semantics() -> None:
    network = {
        "kind": "shortest_path",
        "nodes": ["s", "a", "b", "t"],
        "edges": [
            {"id": "s_a", "from": "s", "to": "a", "cost": 1},
            {"id": "a_t", "from": "a", "to": "t", "cost": 2},
            {"id": "s_b", "from": "s", "to": "b", "cost": 2},
            {"id": "b_t", "from": "b", "to": "t", "cost": 5},
        ],
        "source": "s",
        "sink": "t",
    }
    result = solve_network_model({"network_model": network}, {"route_id": "declared_milp_solver"})
    assert result["metrics"]["objective"] == 3
    assert result["metrics"]["network_feasible"] is True
    assert result["evidence"]["network"]["node_balance"] == {"s": 1.0, "a": 0.0, "b": 0.0, "t": -1.0}


def test_native_min_cost_flow_and_max_flow_emit_auditable_evidence() -> None:
    min_cost = {
        "kind": "min_cost_flow",
        "nodes": ["factory", "hub", "market"],
        "edges": [
            {"id": "f_h", "from": "factory", "to": "hub", "cost": 1, "capacity": 5},
            {"id": "h_m", "from": "hub", "to": "market", "cost": 2, "capacity": 5},
            {"id": "f_m", "from": "factory", "to": "market", "cost": 5, "capacity": 5},
        ],
        "supplies": {"factory": 5, "hub": 0, "market": -5},
    }
    result = solve_network_model({"network_model": min_cost}, {"route_id": "declared_milp_solver"})
    assert result["metrics"]["objective"] == 15
    assert result["metrics"]["network_feasible"] is True


def _mechanism_contract(tmp_path: Path, kind: str = "first_order_relaxation") -> ProblemContract:
    from scipy.integrate import solve_ivp
    times = np.linspace(0, 10, 31)
    if kind == "first_order_relaxation":
        states = solve_ivp(lambda t, y: [0.7 * (5.0 - y[0])], (0, 10), [1.0], t_eval=times).y[0]
        model = {"kind": kind, "time_column": "time", "state_columns": ["y"], "initial_state": {"y": 1.0}, "parameters": {"k": {"lower": 0.01, "upper": 2.0}, "equilibrium": {"lower": 2.0, "upper": 8.0}}, "multistart": 4}
    elif kind == "logistic_growth":
        states = solve_ivp(lambda t, y: [0.5 * y[0] * (1 - y[0] / 10)], (0, 10), [1.0], t_eval=times).y[0]
        model = {"kind": kind, "time_column": "time", "state_columns": ["y"], "initial_state": {"y": 1.0}, "parameters": {"r": {"lower": 0.01, "upper": 2.0}, "K": {"lower": 2.0, "upper": 20.0}}, "multistart": 4}
    else:
        result = solve_ivp(lambda t, y: [-0.8*y[0]*y[1], 0.8*y[0]*y[1]-0.3*y[1], 0.3*y[1]], (0, 10), [0.99, 0.01, 0.0], t_eval=times)
        states = result.y.T
        model = {"kind": "sir", "time_column": "time", "state_columns": ["S", "I", "R"], "initial_state": {"S": 0.99, "I": 0.01, "R": 0.0}, "parameters": {"beta": {"lower": 0.01, "upper": 2.0}, "gamma": {"lower": 0.01, "upper": 1.0}}, "multistart": 4}
    path = tmp_path / f"{kind}.csv"
    frame = pd.DataFrame({"time": times})
    for index, name in enumerate(model["state_columns"]): frame[name] = states[:, index] if getattr(states, "ndim", 1) == 2 else states
    frame.to_csv(path, index=False)
    return ProblemContract("v44", f"mechanism_{kind}", "mechanism", "ready", data_file=str(path), time_column="time", mechanism_model=model, metric_directions={"MAE": "lower_better", "RMSE": "lower_better"})


def test_native_mechanism_fits_builtin_odes_and_reports_identifiability(tmp_path: Path) -> None:
    for kind in ("first_order_relaxation", "logistic_growth", "sir"):
        contract = _mechanism_contract(tmp_path, kind)
        result = solve_mechanism_model(asdict(contract), {"route_id": "declared_ode_multistart"})
        assert result["metrics"]["validation_split"] == "train_selection_final_test_ode"
        assert result["evidence"]["fit_info"]["multistart_count"] == 5
        assert np.isfinite(result["evidence"]["predicted_final_test"]).all()
        if kind == "sir": assert np.allclose(np.sum(result["evidence"]["predicted_final_test"], axis=1), 1.0, atol=1e-6)


def test_native_mechanism_tampered_trajectory_fails_verifier(tmp_path: Path) -> None:
    contract = _mechanism_contract(tmp_path)
    route = RouteSpec("declared_ode_multistart", "candidate", "mechanism", "", ["RMSE", "MAE"])
    result = solve_mechanism_model(asdict(contract), route.__dict__)
    evidence_path = tmp_path / "route.json"
    payload = {"run_id": "run", "problem_id": contract.problem_id, "route_id": route.route_id, "data_path": contract.data_file, "contract": asdict(contract), "route": route.__dict__, "metrics_recomputed": result["metrics"], "evidence": result["evidence"]}
    payload["evidence"]["predicted_final_test"][0][0] += 1.0
    evidence_path.write_text(json.dumps(payload), encoding="utf-8")
    record = ExperimentRecord(contract.problem_id, route.route_id, route.role, "executed", str(Path(__file__).resolve()), contract.data_file, result["metrics"], run_id="run", input_hash=hashlib.sha256(Path(contract.data_file).read_bytes()).hexdigest(), config_hash="x", evidence_paths={"route_evidence": str(evidence_path)})
    verification = verify_native_mechanism(contract, [record], tmp_path / "registry.json", tmp_path / "verify")
    assert verification["passed"] is False


def test_native_simulation_mm1_is_reproducible_and_reports_ci(tmp_path: Path) -> None:
    from solver.native_simulation import solve_simulation_model
    contract = ProblemContract(
        "v44", "queue", "simulation", "ready", data_file=str(tmp_path / "inputs.csv"),
        simulation_model={"kind": "mm1_queue", "arrival_rate": 0.8, "service_rate": 1.2, "horizon": 250.0, "warmup": 25.0, "replications": 8, "seed": 17},
        metric_directions={"mean_wait": "lower_better", "utilization": "higher_better", "ci_half_width": "lower_better"},
    )
    route = {"route_id": "simulation_mm1_monte_carlo"}
    first = solve_simulation_model(asdict(contract), route); second = solve_simulation_model(asdict(contract), route)
    assert first == second
    assert first["metrics"]["replications"] == 8
    assert first["metrics"]["ci_half_width"] >= 0
    assert len(first["evidence"]["replication_seeds"]) == 8
    assert abs(first["metrics"]["utilization"] - 2 / 3) < 0.2


def test_native_simulation_verifier_rejects_tampered_replication(tmp_path: Path) -> None:
    from solver.native_simulation import run_native_simulation_experiments, verify_native_simulation
    data = tmp_path / "inputs.csv"; data.write_text("x\n1\n", encoding="utf-8")
    contract = ProblemContract("v44", "queue_tamper", "simulation", "ready", data_file=str(data), simulation_model={"kind": "mm1_queue", "arrival_rate": 0.5, "service_rate": 1.0, "horizon": 100.0, "warmup": 10.0, "replications": 4, "seed": 3}, metric_directions={"mean_wait": "lower_better", "utilization": "higher_better"})
    route = RouteSpec("simulation_mm1_monte_carlo", "candidate", "simulation", "", ["mean_wait", "utilization", "ci_half_width"])
    records = run_native_simulation_experiments(contract, [route], tmp_path / "experiments")
    evidence_path = Path(records[0].evidence_paths["route_evidence"]); payload = json.loads(evidence_path.read_text(encoding="utf-8")); payload["evidence"]["replication_metrics"][0]["mean_wait"] += 1.0; evidence_path.write_text(json.dumps(payload), encoding="utf-8")
    verification = verify_native_simulation(contract, records, tmp_path / "registry.json", tmp_path / "verify")
    assert verification["passed"] is False


def test_simulation_cli_produces_domain_verified_result(tmp_path: Path) -> None:
    data_root = tmp_path / "queue_data"; data_root.mkdir()
    data = data_root / "inputs.csv"; data.write_text("placeholder\n1\n", encoding="utf-8")
    model = {"kind": "mm1_queue", "arrival_rate": 0.6, "service_rate": 1.0, "horizon": 80.0, "warmup": 8.0, "replications": 4, "seed": 11}
    contract = ProblemContract("v44", "queue_cli", "simulation", "ready", data_file=str(data), simulation_model=model, metric_directions={"mean_wait": "lower_better", "ci_half_width": "lower_better", "utilization": "higher_better"}, source="provided")
    contract_path = data_root / "problem_contract.json"; contract_path.write_text(json.dumps(asdict(contract)), encoding="utf-8")
    out = tmp_path / "run"
    proc = subprocess.run([sys.executable, "run_v39_solver.py", "--problem-id", "queue_cli", "--data-root", str(data_root), "--contract", str(contract_path), "--output-dir", str(out)], cwd=ROOT, text=True, encoding="utf-8", capture_output=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["executed_route_count"] == 2
    assert summary["final_judge"]["verdict"] == "pass"
    assert summary["final_judge"]["claim_level"] == "domain_verified_queue_monte_carlo"
    assert summary["domain_verification"]["passed"] is True


def test_native_simulation_mmc_matches_erlang_c_reference(tmp_path: Path) -> None:
    from solver.native_simulation import solve_simulation_model
    model = {"kind": "mmc_queue", "arrival_rate": 2.0, "service_rate": 1.0, "servers": 3, "horizon": 20000.0, "warmup": 2000.0, "replications": 8, "seed": 71}
    contract = ProblemContract("v44", "mmc", "simulation", "ready", data_file=str(tmp_path / "inputs.csv"), simulation_model=model)
    baseline = solve_simulation_model(asdict(contract), {"route_id": "simulation_analytical_baseline"})
    simulated = solve_simulation_model(asdict(contract), {"route_id": "simulation_queue_monte_carlo"})
    assert abs(simulated["metrics"]["mean_wait"] - baseline["metrics"]["mean_wait"]) < 0.06
    assert abs(simulated["metrics"]["utilization"] - 2 / 3) < 0.025
    assert simulated["metrics"]["rejection_rate"] == 0.0


def test_native_simulation_finite_capacity_reports_blocking(tmp_path: Path) -> None:
    from solver.native_simulation import solve_simulation_model
    base = {"kind": "mmc_queue", "arrival_rate": 5.0, "service_rate": 1.0, "servers": 2, "horizon": 10000.0, "warmup": 1000.0, "replications": 6, "seed": 13}
    small = ProblemContract("v44", "mmck-small", "simulation", "ready", data_file=str(tmp_path / "inputs.csv"), simulation_model={**base, "capacity": 3})
    large = ProblemContract("v44", "mmck-large", "simulation", "ready", data_file=str(tmp_path / "inputs.csv"), simulation_model={**base, "capacity": 8})
    small_mc = solve_simulation_model(asdict(small), {"route_id": "simulation_queue_monte_carlo"})
    small_closed = solve_simulation_model(asdict(small), {"route_id": "simulation_analytical_baseline"})
    large_closed = solve_simulation_model(asdict(large), {"route_id": "simulation_analytical_baseline"})
    assert small_mc["metrics"]["rejection_rate"] > 0.2
    assert abs(small_mc["metrics"]["rejection_rate"] - small_closed["metrics"]["rejection_rate"]) < 0.03
    assert large_closed["metrics"]["rejection_rate"] < small_closed["metrics"]["rejection_rate"]


def test_native_simulation_rejects_invalid_mmc_contract(tmp_path: Path) -> None:
    data = tmp_path / "inputs.csv"; data.write_text("x\n1\n", encoding="utf-8")
    invalid = ProblemContract("v44", "bad-mmc", "simulation", "ready", data_file=str(data), simulation_model={"kind": "mmc_queue", "arrival_rate": 2.0, "service_rate": 1.0, "servers": 2, "horizon": 10.0, "warmup": 1.0, "replications": 2})
    checked = validate_contract(invalid, [DataAsset(str(data), "csv", 1, ["x"])])
    assert checked.status == "invalid"
    assert "simulation_mmc_requires_arrival_rate_below_total_service_rate" in checked.unresolved_fields

    invalid.simulation_model["arrival_rate"] = 1.0; invalid.simulation_model["capacity"] = 1
    checked = validate_contract(invalid, [DataAsset(str(data), "csv", 1, ["x"])])
    assert checked.status == "invalid"
    assert "simulation_capacity_invalid" in checked.unresolved_fields


def test_native_simulation_verifier_rejects_tampered_rejection_rate(tmp_path: Path) -> None:
    from solver.native_simulation import run_native_simulation_experiments, verify_native_simulation
    data = tmp_path / "inputs.csv"; data.write_text("x\n1\n", encoding="utf-8")
    model = {"kind": "mmc_queue", "arrival_rate": 4.0, "service_rate": 1.0, "servers": 2, "capacity": 3, "horizon": 200.0, "warmup": 20.0, "replications": 3, "seed": 9}
    contract = ProblemContract("v44", "mmck-tamper", "simulation", "ready", data_file=str(data), simulation_model=model)
    route = RouteSpec("simulation_queue_monte_carlo", "candidate", "simulation", "", ["mean_wait", "rejection_rate"])
    records = run_native_simulation_experiments(contract, [route], tmp_path / "experiments")
    evidence_path = Path(records[0].evidence_paths["route_evidence"]); payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    payload["evidence"]["replication_metrics"][0]["rejection_rate"] = 0.0
    evidence_path.write_text(json.dumps(payload), encoding="utf-8")
    assert verify_native_simulation(contract, records, tmp_path / "registry.json", tmp_path / "verify")["passed"] is False


def test_native_simulation_empirical_distributions_are_reproducible(tmp_path: Path) -> None:
    from solver.native_simulation import solve_simulation_model
    model = {"kind": "mmc_queue", "arrival_rate": 1.0, "service_rate": 1.0, "servers": 2, "horizon": 500.0, "warmup": 50.0, "replications": 4, "seed": 23, "arrival_distribution": {"kind": "empirical", "values": [0.25, 0.5, 1.5], "weights": [1, 2, 1]}, "service_distribution": {"kind": "empirical", "values": [0.5, 1.0, 2.0]}}
    contract = ProblemContract("v44", "empirical-queue", "simulation", "ready", data_file=str(tmp_path / "inputs.csv"), simulation_model=model)
    first = solve_simulation_model(asdict(contract), {"route_id": "simulation_queue_monte_carlo"})
    second = solve_simulation_model(asdict(contract), {"route_id": "simulation_queue_monte_carlo"})
    assert first == second
    assert first["evidence"]["analytical_reference"] is None
    assert first["metrics"]["method"] == "event_driven_queue"


def test_simulation_empirical_route_uses_deterministic_mean_baseline(tmp_path: Path) -> None:
    data = tmp_path / "inputs.csv"; data.write_text("x\n1\n", encoding="utf-8")
    model = {"kind": "mmc_queue", "arrival_rate": 1.0, "service_rate": 1.0, "servers": 2, "horizon": 100.0, "warmup": 10.0, "replications": 2, "arrival_distribution": {"kind": "deterministic"}, "service_distribution": {"kind": "empirical", "values": [0.5, 1.5]}}
    contract = ProblemContract("v44", "distribution-route", "simulation", "ready", data_file=str(data), simulation_model=model)
    profile = ProblemProfile("distribution-route", "", str(tmp_path), [], [], [DataAsset(str(data), ".csv", data.stat().st_size, ["x"], 1, 1, True)], [], [], {}, [], contract)
    routes = plan_routes(profile, [])
    assert routes[0].route_id == "simulation_mean_deterministic_baseline"
    assert routes[1].route_id == "simulation_queue_monte_carlo"


def _spatial_contract(tmp_path: Path, block_size: float = 1.0) -> ProblemContract:
    data = tmp_path / "surface.csv"
    pd.DataFrame([
        {"x": x, "y": y, "value": x + 2 * y}
        for x in range(6) for y in range(6)
    ]).to_csv(data, index=False)
    return ProblemContract(
        "v44", "surface", "spatial", "ready", data_file=str(data),
        spatial_model={"kind": "idw", "x_column": "x", "y_column": "y", "target_column": "value", "power": 2.0, "block_size": block_size, "seed": 19},
        metric_directions={"MAE": "lower_better", "RMSE": "lower_better"},
    )


def test_native_spatial_idw_is_reproducible_and_block_disjoint(tmp_path: Path) -> None:
    contract = _spatial_contract(tmp_path)
    first = solve_spatial_model(asdict(contract), {"route_id": "spatial_idw"})
    second = solve_spatial_model(asdict(contract), {"route_id": "spatial_idw"})
    assert first == second
    evidence = first["evidence"]
    groups = [set(evidence[name]) for name in ("train_block_keys", "selection_block_keys", "final_test_block_keys")]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert first["metrics"]["spatial_block_count"] == 36


def test_native_spatial_verifier_rejects_tampered_prediction(tmp_path: Path) -> None:
    contract = _spatial_contract(tmp_path)
    route = RouteSpec("spatial_idw", "candidate", "spatial", "", ["MAE", "RMSE"])
    records = run_native_spatial_experiments(contract, [route], tmp_path / "experiments")
    evidence_path = Path(records[0].evidence_paths["route_evidence"])
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    payload["evidence"]["final_test_predicted"][0] += 1.0
    evidence_path.write_text(json.dumps(payload), encoding="utf-8")
    verification = verify_native_spatial(contract, records, tmp_path / "registry.json", tmp_path / "verify")
    assert verification["passed"] is False


def test_native_spatial_geographic_uses_haversine_and_moran_diagnostic(tmp_path: Path) -> None:
    from solver.native_spatial import _distance
    assert _distance(np.asarray([0.0, 0.0]), np.asarray([[0.0, 1.0]]), "geographic_wgs84")[0] == pytest.approx(111.195, rel=1e-3)
    data = tmp_path / "geographic.csv"
    pd.DataFrame([{"lon": 116.0 + x * 0.05, "lat": 39.0 + y * 0.05, "value": x + 2 * y} for x in range(6) for y in range(6)]).to_csv(data, index=False)
    contract = ProblemContract("v44", "geo", "spatial", "ready", data_file=str(data), spatial_model={"kind": "idw", "x_column": "lon", "y_column": "lat", "target_column": "value", "coordinate_system": "geographic_wgs84", "block_size": 3.0, "power": 2.0, "seed": 7, "moran_neighbors": 3}, metric_directions={"MAE": "lower_better", "RMSE": "lower_better"})
    result = solve_spatial_model(asdict(contract), {"route_id": "spatial_idw"})
    assert result["evidence"]["distance_metric"] == "haversine_km"
    assert np.isfinite(result["metrics"]["final_test_residual_morans_i"])


def test_native_spatial_verifier_rejects_tampered_distance_metadata(tmp_path: Path) -> None:
    contract = _spatial_contract(tmp_path)
    route = RouteSpec("spatial_idw", "candidate", "spatial", "", ["MAE", "RMSE"])
    records = run_native_spatial_experiments(contract, [route], tmp_path / "experiments")
    evidence_path = Path(records[0].evidence_paths["route_evidence"]); payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    payload["evidence"]["distance_metric"] = "haversine_km"
    evidence_path.write_text(json.dumps(payload), encoding="utf-8")
    verification = verify_native_spatial(contract, records, tmp_path / "registry.json", tmp_path / "verify")
    assert verification["passed"] is False


def test_native_spatial_rejects_insufficient_blocks(tmp_path: Path) -> None:
    contract = _spatial_contract(tmp_path, block_size=100.0)
    with pytest.raises(ValueError, match="3_spatial_blocks"):
        solve_spatial_model(asdict(contract), {"route_id": "spatial_idw"})


def _multiobjective_contract(tmp_path: Path) -> ProblemContract:
    data = tmp_path / "multiobjective.csv"
    data.write_text("placeholder\n1\n", encoding="utf-8")
    base_model = {
        "variables": [{"name": "x", "kind": "binary"}, {"name": "y", "kind": "binary"}],
        "linear_constraints": [{"name": "one_choice", "coefficients": {"x": 1, "y": 1}, "sense": "<=", "rhs": 1}],
    }
    model = {
        "base_model": base_model,
        "objectives": [
            {"name": "benefit_x", "sense": "max", "coefficients": {"x": 1}},
            {"name": "benefit_y", "sense": "max", "coefficients": {"y": 1}},
        ],
        "epsilon_grid_size": 5,
    }
    return ProblemContract("v44", "multiobjective", "constrained_optimization", "ready", data_file=str(data), multiobjective_model=model, metric_directions={"pareto_count": "higher_better"}, source="provided")


def test_native_multiobjective_finds_and_verifies_pareto_front(tmp_path: Path) -> None:
    from solver.native_multiobjective import run_native_multiobjective_experiments, solve_multiobjective_model, verify_native_multiobjective
    contract = _multiobjective_contract(tmp_path)
    result = solve_multiobjective_model(asdict(contract), {"route_id": "multiobjective_epsilon_constraint"})
    assert result["metrics"]["pareto_count"] == 2
    values = {tuple(solution["objective_values"]) for solution in result["evidence"]["solutions"]}
    assert values == {(1.0, 0.0), (0.0, 1.0)}
    routes = [RouteSpec("multiobjective_single_objective_baseline", "baseline", "multiobjective", "", ["pareto_count"]), RouteSpec("multiobjective_epsilon_constraint", "candidate", "multiobjective", "", ["pareto_count"])]
    records = run_native_multiobjective_experiments(contract, routes, tmp_path / "experiments")
    verification = verify_native_multiobjective(contract, records, tmp_path / "registry.json", tmp_path / "verify")
    assert verification["passed"] is True
    assert verification["claim_level"] == "domain_verified_sampled_pareto_set"


def test_native_multiobjective_verifier_rejects_tampered_objective_value(tmp_path: Path) -> None:
    from solver.native_multiobjective import run_native_multiobjective_experiments, verify_native_multiobjective
    contract = _multiobjective_contract(tmp_path)
    route = RouteSpec("multiobjective_epsilon_constraint", "candidate", "multiobjective", "", ["pareto_count"])
    records = run_native_multiobjective_experiments(contract, [route], tmp_path / "experiments")
    evidence_path = Path(records[0].evidence_paths["route_evidence"])
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    payload["evidence"]["solutions"][0]["objective_values"][0] += 2.0
    evidence_path.write_text(json.dumps(payload), encoding="utf-8")
    verification = verify_native_multiobjective(contract, records, tmp_path / "registry.json", tmp_path / "verify")
    assert verification["passed"] is False


def test_native_multiobjective_contract_rejects_more_than_two_objectives(tmp_path: Path) -> None:
    contract = _multiobjective_contract(tmp_path)
    contract.multiobjective_model["objectives"].append({"name": "total", "sense": "max", "coefficients": {"x": 1, "y": 1}})
    checked = validate_contract(contract, [DataAsset(contract.data_file, "csv", 1, ["placeholder"])])
    assert checked.status == "invalid"
    assert "multiobjective_native_requires_exactly_two_objectives" in checked.unresolved_fields


def test_native_multiobjective_verifier_rejects_fractional_binary_solution(tmp_path: Path) -> None:
    from solver.native_multiobjective import run_native_multiobjective_experiments, verify_native_multiobjective
    contract = _multiobjective_contract(tmp_path)
    route = RouteSpec("multiobjective_epsilon_constraint", "candidate", "multiobjective", "", ["pareto_count"])
    records = run_native_multiobjective_experiments(contract, [route], tmp_path / "experiments")
    evidence_path = Path(records[0].evidence_paths["route_evidence"]); payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    solution = payload["evidence"]["solutions"][0]; solution["decision_values"] = {"x": 0.5, "y": 0.5}; solution["objective_values"] = [0.5, 0.5]
    evidence_path.write_text(json.dumps(payload), encoding="utf-8")
    verification = verify_native_multiobjective(contract, records, tmp_path / "registry.json", tmp_path / "verify")
    assert verification["passed"] is False
    assert verification["checks"][0]["integrality"] is False


def test_multiobjective_cli_produces_domain_verified_result(tmp_path: Path) -> None:
    data_root = tmp_path / "multiobjective_data"
    data_root.mkdir()
    data = data_root / "inputs.csv"
    data.write_text("placeholder\n1\n", encoding="utf-8")
    contract = _multiobjective_contract(tmp_path)
    contract.data_file = str(data)
    contract_path = data_root / "problem_contract.json"
    contract_path.write_text(json.dumps(asdict(contract)), encoding="utf-8")
    out = tmp_path / "run"
    proc = subprocess.run([sys.executable, "run_v39_solver.py", "--problem-id", "multiobjective", "--data-root", str(data_root), "--contract", str(contract_path), "--output-dir", str(out)], cwd=ROOT, text=True, encoding="utf-8", capture_output=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["executed_route_count"] == 2
    assert summary["final_judge"]["verdict"] == "pass"
    assert summary["final_judge"]["claim_level"] == "domain_verified_sampled_pareto_set"
    assert summary["domain_verification"]["passed"] is True


def _joined_forecasting_contract(main: Path, lookup: Path) -> ProblemContract:
    return ProblemContract(
        "v44", "joined_forecast", "forecasting", "ready", data_file=str(main),
        data_sources=[{"alias": "observations", "path": str(main)}, {"alias": "regions", "path": str(lookup)}],
        base_table="observations",
        data_joins=[{"left": "observations", "right": "regions", "on": "region_id", "how": "left", "validate": "many_to_one", "max_unmatched_left_rate": 0.0}],
        target_column="sales", time_column="week", feature_columns=["region_id", "regions__income"],
        known_future_columns=["region_id", "regions__income"], forecast_horizon=8, validation_mode="fixed_origin_holdout",
        metric_directions={"MAE": "lower_better", "RMSE": "lower_better", "WAPE": "lower_better"}, source="provided",
    )


def test_multitable_join_runs_end_to_end_and_binds_manifest(tmp_path: Path) -> None:
    data_root = tmp_path / "joined_data"; data_root.mkdir()
    main = data_root / "observations.csv"; lookup = data_root / "regions.csv"
    pd.DataFrame({"week": range(1, 49), "region_id": [1, 2] * 24, "sales": [100 + 2 * week + 10 * (week % 2) for week in range(1, 49)]}).to_csv(main, index=False)
    pd.DataFrame({"region_id": [1, 2], "income": [50.0, 80.0]}).to_csv(lookup, index=False)
    contract = _joined_forecasting_contract(main, lookup)
    contract_path = data_root / "problem_contract.json"; contract_path.write_text(json.dumps(asdict(contract)), encoding="utf-8")
    out = tmp_path / "run"
    proc = subprocess.run([sys.executable, "run_v39_solver.py", "--problem-id", "joined_forecast", "--data-root", str(data_root), "--contract", str(contract_path), "--output-dir", str(out), "--max-routes", "3"], cwd=ROOT, text=True, encoding="utf-8", capture_output=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8")); assembly = summary["data_assembly"]
    assert summary["contract_status"] == "ready"
    assert summary["executed_route_count"] == 3
    assert assembly["output_rows"] == 48 and "regions__income" in assembly["output_columns"]
    assert Path(assembly["manifest_path"]).is_file() and len(assembly["output_sha256"]) == 64
    used_contract = json.loads(Path(summary["contract_path"]).read_text(encoding="utf-8"))
    assert used_contract["data_file"] == assembly["output_path"] and used_contract["sheet_name"] is None
    for key in ("route_metrics_long_csv", "route_comparison_svg", "paper_results_section_md"):
        artifact = Path(summary["workflow_exports"][key]); assert artifact.is_file() and artifact.stat().st_size > 0


def test_multitable_final_gate_rejects_changed_source_after_assembly(tmp_path: Path) -> None:
    from workflow.data_assembler import assemble_contract_data
    main = tmp_path / "main.csv"; lookup = tmp_path / "lookup.csv"
    pd.DataFrame({"week": range(1, 25), "region_id": [1, 2] * 12, "sales": range(24)}).to_csv(main, index=False)
    pd.DataFrame({"region_id": [1, 2], "income": [50, 80]}).to_csv(lookup, index=False)
    contract = _joined_forecasting_contract(main, lookup); contract.forecast_horizon = 4
    assets = [DataAsset(str(main), ".csv", main.stat().st_size, ["week", "region_id", "sales"], 24, 3, True), DataAsset(str(lookup), ".csv", lookup.stat().st_size, ["region_id", "income"], 2, 2, True)]
    contract = validate_contract(contract, assets); assembly = assemble_contract_data(contract, tmp_path / "assembly")
    assembled = Path(assembly["output_path"]); assets.append(DataAsset(str(assembled), ".csv", assembled.stat().st_size, assembly["output_columns"], assembly["output_rows"], len(assembly["output_columns"]), True))
    profile = ProblemProfile(contract.problem_id, "", str(tmp_path), [], ["forecasting"], assets, [], ["MAE"], contract.metric_directions, [], contract)
    route = RouteSpec("baseline", "baseline", "forecasting", "", ["MAE"])
    lookup.write_text("region_id,income\n1,999\n2,80\n", encoding="utf-8")
    result = judge_solution(profile, [route], [], tmp_path / "judge")
    assert "data_assembly_source_hash_mismatch:regions" in result["flags"]


def test_multitable_join_rejects_cardinality_violation(tmp_path: Path) -> None:
    from workflow.data_assembler import assemble_contract_data
    main = tmp_path / "main.csv"; lookup = tmp_path / "lookup.csv"
    pd.DataFrame({"id": [1, 2], "y": [3, 4]}).to_csv(main, index=False)
    pd.DataFrame({"id": [1, 1, 2], "value": [5, 6, 7]}).to_csv(lookup, index=False)
    contract = ProblemContract("v44", "bad-cardinality", "regression", "ready", data_file=str(main), data_sources=[{"alias": "main", "path": str(main)}, {"alias": "lookup", "path": str(lookup)}], base_table="main", data_joins=[{"left": "main", "right": "lookup", "on": "id", "how": "left", "validate": "many_to_one"}], target_column="y")
    with pytest.raises(Exception, match="not a many-to-one merge"):
        assemble_contract_data(contract, tmp_path / "assembly")


def test_multitable_join_rejects_unmatched_threshold(tmp_path: Path) -> None:
    from workflow.data_assembler import assemble_contract_data
    main = tmp_path / "main.csv"; lookup = tmp_path / "lookup.csv"
    pd.DataFrame({"id": [1, 2, 3], "y": [3, 4, 5]}).to_csv(main, index=False)
    pd.DataFrame({"id": [1, 2], "value": [5, 6]}).to_csv(lookup, index=False)
    contract = ProblemContract("v44", "bad-match", "regression", "ready", data_file=str(main), data_sources=[{"alias": "main", "path": str(main)}, {"alias": "lookup", "path": str(lookup)}], base_table="main", data_joins=[{"left": "main", "right": "lookup", "on": "id", "how": "left", "validate": "many_to_one", "max_unmatched_left_rate": 0.0}], target_column="y")
    with pytest.raises(ValueError, match="join_unmatched_left_rate_exceeded"):
        assemble_contract_data(contract, tmp_path / "assembly")


def test_multitable_aggregation_produces_time_series_and_manifest(tmp_path: Path) -> None:
    from workflow.data_assembler import assemble_contract_data
    sales = tmp_path / "sales.csv"; products = tmp_path / "products.csv"
    pd.DataFrame({"date": ["2024-01-01", "2024-01-01", "2024-01-02"], "product_id": [1, 2, 1], "quantity": [2.0, 3.0, 4.0]}).to_csv(sales, index=False)
    pd.DataFrame({"product_id": [1, 2], "category": ["A", "B"]}).to_csv(products, index=False)
    contract = ProblemContract(
        "v44", "aggregate", "forecasting", "ready", data_file=str(sales),
        data_sources=[{"alias": "sales", "path": str(sales)}, {"alias": "products", "path": str(products)}], base_table="sales",
        data_joins=[{"left": "sales", "right": "products", "on": "product_id", "how": "left", "validate": "many_to_one", "max_unmatched_left_rate": 0.0}],
        data_aggregation={"group_by": ["date"], "aggregations": {"daily_sales": {"column": "quantity", "function": "sum"}, "product_count": {"column": "product_id", "function": "nunique"}}, "sort_by": ["date"]},
        target_column="daily_sales", time_column="date", feature_columns=["product_count"], known_future_columns=["product_count"], forecast_horizon=1, validation_mode="fixed_origin_holdout",
    )
    assets = [DataAsset(str(sales), ".csv", sales.stat().st_size, ["date", "product_id", "quantity"], 3, 3, True), DataAsset(str(products), ".csv", products.stat().st_size, ["product_id", "category"], 2, 2, True)]
    assert validate_contract(contract, assets).status == "ready"
    result = assemble_contract_data(contract, tmp_path / "assembly")
    frame = pd.read_csv(result["output_path"])
    assert frame.to_dict(orient="records") == [{"date": "2024-01-01", "daily_sales": 5.0, "product_count": 2}, {"date": "2024-01-02", "daily_sales": 4.0, "product_count": 1}]
    assert result["aggregation"]["rows_before"] == 3 and result["aggregation"]["rows_after"] == 2


def _robust_contract(tmp_path: Path, risk_measure: str = "worst_case") -> ProblemContract:
    data = tmp_path / "robust_inputs.csv"; data.write_text("placeholder\n1\n", encoding="utf-8")
    scenario_constraints = [{"name": "demand", "coefficients": {"x": 1, "y": 1}, "sense": ">=", "rhs": 10}]
    model = {
        "objective_sense": "min", "risk_measure": risk_measure, "alpha": 0.8,
        "variables": [{"name": "x", "kind": "continuous", "lower": 0, "upper": 10}, {"name": "y", "kind": "continuous", "lower": 0, "upper": 10}],
        "scenarios": [
            {"name": "normal", "probability": 0.9, "objective_coefficients": {"x": 1, "y": 4}, "linear_constraints": scenario_constraints},
            {"name": "stress", "probability": 0.1, "objective_coefficients": {"x": 10, "y": 1}, "linear_constraints": scenario_constraints},
        ],
    }
    return ProblemContract("v44", "robust", "constrained_optimization", "ready", data_file=str(data), robust_optimization_model=model, metric_directions={"objective": "lower_better", "constraint_violation": "lower_better"}, source="provided")


def test_native_robust_worst_case_improves_expected_baseline(tmp_path: Path) -> None:
    from solver.native_robust import solve_robust_model
    contract = _robust_contract(tmp_path)
    baseline = solve_robust_model(asdict(contract), {"route_id": "robust_expected_baseline"})
    candidate = solve_robust_model(asdict(contract), {"route_id": "robust_declared_risk_solver"})
    assert baseline["evidence"]["decision_variables"] == {"x": 10.0, "y": 0.0}
    assert baseline["metrics"]["objective"] == pytest.approx(100.0)
    assert candidate["evidence"]["decision_variables"] == pytest.approx({"x": 2.5, "y": 7.5})
    assert candidate["metrics"]["objective"] == pytest.approx(32.5)
    assert candidate["metrics"]["constraint_violation"] == 0.0


def test_native_robust_cvar_compiles_and_recomputes(tmp_path: Path) -> None:
    from solver.native_robust import solve_robust_model
    contract = _robust_contract(tmp_path, risk_measure="cvar")
    result = solve_robust_model(asdict(contract), {"route_id": "robust_declared_risk_solver"})
    assert result["metrics"]["model_status"] == "optimal"
    assert result["metrics"]["objective"] == pytest.approx(32.5)
    assert result["evidence"]["declared_risk_measure"] == "cvar"


def test_native_robust_verifier_rejects_tampered_scenario_cost(tmp_path: Path) -> None:
    from solver.native_robust import run_native_robust_experiments, verify_native_robust
    contract = _robust_contract(tmp_path)
    route = RouteSpec("robust_declared_risk_solver", "candidate", "robust_optimization", "", ["objective", "constraint_violation"])
    records = run_native_robust_experiments(contract, [route], tmp_path / "experiments")
    evidence_path = Path(records[0].evidence_paths["route_evidence"]); payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    payload["evidence"]["scenario_results"][0]["cost"] += 1.0
    evidence_path.write_text(json.dumps(payload), encoding="utf-8")
    assert verify_native_robust(contract, records, tmp_path / "registry.json", tmp_path / "verify")["passed"] is False


def test_robust_cli_produces_domain_verified_result(tmp_path: Path) -> None:
    data_root = tmp_path / "robust_data"; data_root.mkdir()
    contract = _robust_contract(data_root); contract_path = data_root / "problem_contract.json"
    contract_path.write_text(json.dumps(asdict(contract)), encoding="utf-8"); out = tmp_path / "run"
    proc = subprocess.run([sys.executable, "run_v39_solver.py", "--problem-id", "robust", "--data-root", str(data_root), "--contract", str(contract_path), "--output-dir", str(out)], cwd=ROOT, text=True, encoding="utf-8", capture_output=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["executed_route_count"] == 2
    assert summary["final_judge"]["verdict"] == "pass"
    assert summary["final_judge"]["recommended_route"]["route_id"] == "robust_declared_risk_solver"
    assert summary["final_judge"]["claim_level"] == "domain_verified_scenario_robust_result"
    paper = Path(summary["workflow_exports"]["paper_results_section_md"]).read_text(encoding="utf-8")
    assert "robust_declared_risk_solver" in paper and "domain_verified_scenario_robust_result" in paper


def _retail_contract(tmp_path: Path, risk_measure: str = "expected") -> ProblemContract:
    data = tmp_path / "retail_inputs.csv"
    data.write_text("placeholder\n1\n", encoding="utf-8")
    model = {
        "risk_measure": risk_measure,
        "alpha": 0.8,
        "budget": 160.0,
        "min_selected": 1,
        "max_selected": 2,
        "shortage_penalty": 0.2,
        "baseline_markup": 0.2,
        "scenarios": [{"name": "normal", "probability": 0.7}, {"name": "high", "probability": 0.3}],
        "items": [
            {"name": "leafy", "unit_cost": 4.0, "loss_rate": 0.1, "min_replenishment": 5.0, "max_replenishment": 25.0, "price_options": [
                {"price": 5.0, "demand": {"normal": 20.0, "high": 24.0}},
                {"price": 8.0, "demand": {"normal": 15.0, "high": 18.0}},
            ]},
            {"name": "root", "unit_cost": 3.0, "loss_rate": 0.05, "min_replenishment": 4.0, "max_replenishment": 20.0, "price_options": [
                {"price": 4.0, "demand": {"normal": 12.0, "high": 15.0}},
                {"price": 6.5, "demand": {"normal": 10.0, "high": 13.0}},
            ]},
        ],
    }
    return ProblemContract("v44", "retail", "constrained_optimization", "ready", data_file=str(data), retail_decision_model=model, metric_directions={"objective": "lower_better", "expected_profit": "higher_better", "constraint_violation": "lower_better"}, source="provided")


@pytest.mark.parametrize("risk_measure", ["expected", "worst_case", "cvar"])
def test_native_retail_joint_price_replenishment_is_feasible(tmp_path: Path, risk_measure: str) -> None:
    from solver.native_retail import solve_retail_model
    contract = _retail_contract(tmp_path, risk_measure)
    baseline = solve_retail_model(asdict(contract), {"route_id": "retail_fixed_markup_baseline"})
    candidate = solve_retail_model(asdict(contract), {"route_id": "retail_joint_price_replenishment"})
    assert baseline["metrics"]["constraint_violation"] == pytest.approx(0.0)
    assert candidate["metrics"]["constraint_violation"] == pytest.approx(0.0)
    assert candidate["metrics"]["objective"] <= baseline["metrics"]["objective"] + 1e-7
    assert candidate["metrics"]["expected_profit"] > baseline["metrics"]["expected_profit"]
    assert any(item["price_option"] == 1 for item in candidate["evidence"]["decisions"] if item["selected"])


def test_native_retail_verifier_rejects_tampered_profit(tmp_path: Path) -> None:
    from solver.native_retail import run_native_retail_experiments, verify_native_retail
    contract = _retail_contract(tmp_path)
    route = RouteSpec("retail_joint_price_replenishment", "candidate", "retail_decision", "", ["objective", "expected_profit", "constraint_violation"])
    records = run_native_retail_experiments(contract, [route], tmp_path / "experiments")
    evidence_path = Path(records[0].evidence_paths["route_evidence"])
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    payload["evidence"]["scenario_results"][0]["profit"] += 10.0
    evidence_path.write_text(json.dumps(payload), encoding="utf-8")
    verification = verify_native_retail(contract, records, tmp_path / "registry.json", tmp_path / "verify")
    assert verification["passed"] is False


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (lambda model: model.update({"stress_test": {"seed": 7, "replications": 2, "demand_cv": 0.2}}), "retail_stress_replications_invalid"),
        (lambda model: model.update({"stress_test": {"seed": "7", "replications": 100, "demand_cv": 0.2}}), "retail_stress_seed_invalid"),
        (lambda model: model.update({"stress_test": {"seed": 7, "replications": 100, "demand_cv": -0.1}}), "retail_stress_demand_cv_invalid"),
    ],
)
def test_retail_contract_rejects_invalid_stress_schema(tmp_path: Path, mutation, expected_error: str) -> None:
    contract = _retail_contract(tmp_path)
    mutation(contract.retail_decision_model)
    checked = validate_contract(contract, [DataAsset(contract.data_file, "csv", 1, ["placeholder"])])
    assert checked.status == "invalid"
    assert expected_error in checked.unresolved_fields


def test_retail_contract_rejects_invalid_elasticity_range(tmp_path: Path) -> None:
    contract = _retail_contract(tmp_path)
    contract.retail_decision_model["items"][0].update({"base_demand": 10.0, "reference_price": 5.0, "elasticity": -1.0, "stress_elasticity_range": [-0.2, -2.0]})
    checked = validate_contract(contract, [DataAsset(contract.data_file, "csv", 1, ["placeholder"])])
    assert checked.status == "invalid"
    assert "retail_stress_elasticity_range_invalid:leafy" in checked.unresolved_fields


def test_native_retail_verifier_rejects_tampered_stress_evidence(tmp_path: Path) -> None:
    from solver.native_retail import run_native_retail_experiments, verify_native_retail
    contract = _retail_contract(tmp_path)
    contract.retail_decision_model["stress_test"] = {"seed": 17, "replications": 50, "demand_cv": 0.2}
    route = RouteSpec("retail_joint_price_replenishment", "candidate", "retail_decision", "", ["objective", "expected_profit", "constraint_violation"])
    records = run_native_retail_experiments(contract, [route], tmp_path / "experiments")
    evidence_path = Path(records[0].evidence_paths["route_evidence"])
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    payload["evidence"]["stress_test"]["mean_profit"] += 1.0
    evidence_path.write_text(json.dumps(payload), encoding="utf-8")
    verification = verify_native_retail(contract, records, tmp_path / "registry.json", tmp_path / "verify")
    assert verification["passed"] is False
    assert verification["checks"][0]["stress_test_recomputed"] is False


def test_retail_cli_produces_domain_verified_policy(tmp_path: Path) -> None:
    data_root = tmp_path / "retail_data"
    data_root.mkdir()
    contract = _retail_contract(data_root)
    contract_path = data_root / "problem_contract.json"
    contract_path.write_text(json.dumps(asdict(contract)), encoding="utf-8")
    out = tmp_path / "run"
    proc = subprocess.run([sys.executable, "run_v39_solver.py", "--problem-id", "retail", "--data-root", str(data_root), "--contract", str(contract_path), "--output-dir", str(out)], cwd=ROOT, text=True, encoding="utf-8", capture_output=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["executed_route_count"] == 2
    assert summary["final_judge"]["verdict"] == "pass"
    assert summary["final_judge"]["recommended_route"]["route_id"] == "retail_joint_price_replenishment"
    assert summary["final_judge"]["claim_level"] == "domain_verified_retail_policy"
    comparison = summary["domain_verification"]["policy_comparison"]
    assert comparison["common_random_numbers"] is True and comparison["replications"] == 500
    assert comparison["candidate_robustly_better"] is True
    policy = pd.read_csv(summary["workflow_exports"]["recommended_retail_policy_csv"])
    scenarios = pd.read_csv(summary["workflow_exports"]["recommended_retail_scenarios_csv"])
    assert len(policy) == 2 and len(scenarios) == 4
    assert set(policy.columns) >= {"category", "price", "replenishment"}
    service = pd.read_csv(summary["workflow_exports"]["recommended_retail_category_service_csv"])
    assert list(service.columns) == ["scenario", "group", "minimum_sales", "actual_sales", "satisfied"]


def test_retail_manifest_gate_rejects_changed_source(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("x\n1\n", encoding="utf-8")
    contract = _retail_contract(tmp_path)
    model_hash = hashlib.sha256(json.dumps(contract.retail_decision_model, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    manifest = tmp_path / "retail_manifest.json"
    data_path = Path(contract.data_file)
    manifest.write_text(json.dumps({"model_sha256": model_hash, "training_cutoff": "2023-06-30", "decision_dates": ["2023-07-01"], "derived_input_path": str(data_path), "derived_input_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(), "sources": [{"alias": "source", "path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}]}), encoding="utf-8")
    contract.retail_decision_manifest = str(manifest)
    asset = DataAsset(contract.data_file, ".csv", Path(contract.data_file).stat().st_size, ["placeholder"], 1, 1, True)
    profile = ProblemProfile(contract.problem_id, "", str(tmp_path), [], ["constrained_optimization"], [asset], [], ["objective"], contract.metric_directions, [], contract)
    routes = plan_routes(profile, [], max_routes=2)
    from solver.experiment_registry import write_registry
    from solver.native_retail import run_native_retail_experiments, verify_native_retail
    records = run_native_retail_experiments(contract, routes, tmp_path / "experiments")
    registry = write_registry(records, tmp_path / "registry")
    verification = verify_native_retail(contract, records, registry, tmp_path / "verify")
    source.write_text("x\n999\n", encoding="utf-8")
    result = judge_solution(profile, routes, records, tmp_path / "judge", domain_verification=verification)
    assert result["verdict"] == "revise"
    assert "retail_decision_source_hash_mismatch:source" in result["flags"]


def test_retail_manifest_gate_rejects_changed_derived_input(tmp_path: Path) -> None:
    contract = _retail_contract(tmp_path)
    data_path = Path(contract.data_file)
    source = tmp_path / "source.csv"
    source.write_text("x\n1\n", encoding="utf-8")
    manifest = tmp_path / "retail_manifest.json"
    manifest.write_text(json.dumps({
        "model_sha256": hashlib.sha256(json.dumps(contract.retail_decision_model, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest(),
        "training_cutoff": "2023-06-30",
        "decision_dates": ["2023-07-01"],
        "derived_input_path": str(data_path),
        "derived_input_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "sources": [{"alias": "source", "path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}],
    }), encoding="utf-8")
    contract.retail_decision_manifest = str(manifest)
    asset = DataAsset(contract.data_file, ".csv", data_path.stat().st_size, ["placeholder"], 1, 1, True)
    profile = ProblemProfile(contract.problem_id, "", str(tmp_path), [], ["constrained_optimization"], [asset], [], ["objective"], contract.metric_directions, [], contract)
    routes = plan_routes(profile, [], max_routes=2)
    from solver.experiment_registry import write_registry
    from solver.native_retail import run_native_retail_experiments, verify_native_retail
    records = run_native_retail_experiments(contract, routes, tmp_path / "experiments")
    registry = write_registry(records, tmp_path / "registry")
    verification = verify_native_retail(contract, records, registry, tmp_path / "verify")
    data_path.write_text("placeholder\nchanged\n", encoding="utf-8")
    result = judge_solution(profile, routes, records, tmp_path / "judge", domain_verification=verification)
    assert result["verdict"] == "revise"
    assert "retail_decision_derived_input_binding_mismatch" in result["flags"]


def test_native_retail_verifier_rejects_group_constraint_tampering(tmp_path: Path) -> None:
    from solver.native_retail import run_native_retail_experiments, verify_native_retail
    contract = _retail_contract(tmp_path)
    contract.retail_decision_model["items"][0]["group"] = "leaf"
    contract.retail_decision_model["items"][1]["group"] = "root"
    contract.retail_decision_model["group_selection_bounds"] = {"leaf": {"min": 1, "max": 1}, "root": {"min": 1, "max": 1}}
    route = RouteSpec("retail_joint_price_replenishment", "candidate", "retail_decision", "", ["objective", "expected_profit", "constraint_violation"])
    records = run_native_retail_experiments(contract, [route], tmp_path / "experiments")
    evidence_path = Path(records[0].evidence_paths["route_evidence"])
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    payload["evidence"]["decisions"][0]["selected"] = False
    evidence_path.write_text(json.dumps(payload), encoding="utf-8")
    verification = verify_native_retail(contract, records, tmp_path / "registry.json", tmp_path / "verify")
    assert verification["passed"] is False
    assert verification["checks"][0]["business_quantities_recomputed"] is False


def test_native_retail_enforces_and_verifies_category_service_requirement(tmp_path: Path) -> None:
    from solver.native_retail import run_native_retail_experiments, solve_retail_model, verify_native_retail
    contract = _retail_contract(tmp_path)
    contract.retail_decision_model["items"][0]["group"] = "vegetables"
    contract.retail_decision_model["items"][1]["group"] = "vegetables"
    contract.retail_decision_model["category_service_requirements"] = [
        {"group": "vegetables", "scenario": "normal", "minimum_sales": 25.0},
        {"group": "vegetables", "scenario": "high", "minimum_sales": 30.0},
    ]
    result = solve_retail_model(asdict(contract), {"route_id": "retail_joint_price_replenishment"})
    assert result["metrics"]["has_feasible_solution"] is True
    for scenario in result["evidence"]["scenario_results"]:
        required = 25.0 if scenario["name"] == "normal" else 30.0
        assert sum(item["sales"] for item in scenario["items"]) >= required - 1e-7
    route = RouteSpec("retail_joint_price_replenishment", "candidate", "retail_decision", "", ["objective", "expected_profit", "constraint_violation"])
    records = run_native_retail_experiments(contract, [route], tmp_path / "experiments")
    assert verify_native_retail(contract, records, tmp_path / "registry.json", tmp_path / "verify")["passed"] is True
    evidence_path = Path(records[0].evidence_paths["route_evidence"]); payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    payload["evidence"]["scenario_results"][0]["items"][0]["sales"] = 0.0
    evidence_path.write_text(json.dumps(payload), encoding="utf-8")
    assert verify_native_retail(contract, records, tmp_path / "registry.json", tmp_path / "verify_tampered")["passed"] is False


def test_retail_contract_rejects_invalid_category_service_requirement(tmp_path: Path) -> None:
    contract = _retail_contract(tmp_path)
    contract.retail_decision_model["category_service_requirements"] = [{"group": "missing", "scenario": "normal", "minimum_sales": 1.0}]
    checked = validate_contract(contract, [DataAsset(contract.data_file, "csv", 1, ["placeholder"])])
    assert checked.status == "invalid"
    assert "retail_category_service_requirement_invalid:0" in checked.unresolved_fields


def test_retail_contract_rejects_invalid_demand_correlation(tmp_path: Path) -> None:
    contract = _retail_contract(tmp_path)
    contract.retail_decision_model["items"][0]["group"] = "leaf"
    contract.retail_decision_model["items"][1]["group"] = "root"
    contract.retail_decision_model["stress_test"] = {"seed": 7, "replications": 100, "demand_cv": 0.2, "demand_correlation": {"groups": ["leaf", "root"], "matrix": [[1.0, 1.2], [1.2, 1.0]]}}
    checked = validate_contract(contract, [DataAsset(contract.data_file, "csv", 1, ["placeholder"])])
    assert checked.status == "invalid"
    assert "retail_stress_demand_correlation_invalid" in checked.unresolved_fields


def test_native_retail_uses_correlated_demand_shocks(tmp_path: Path) -> None:
    from solver.native_retail import run_native_retail_experiments, verify_native_retail
    contract = _retail_contract(tmp_path)
    contract.retail_decision_model["items"][0]["group"] = "leaf"
    contract.retail_decision_model["items"][1]["group"] = "root"
    contract.retail_decision_model["stress_test"] = {"seed": 7, "replications": 100, "demand_cv": 0.2, "demand_correlation": {"groups": ["leaf", "root"], "matrix": [[1.0, 0.8], [0.8, 1.0]]}}
    routes = [
        RouteSpec("retail_fixed_markup_baseline", "baseline", "retail_decision", "", ["objective"]),
        RouteSpec("retail_joint_price_replenishment", "candidate", "retail_decision", "", ["objective"]),
    ]
    records = run_native_retail_experiments(contract, routes, tmp_path / "experiments")
    verification = verify_native_retail(contract, records, tmp_path / "registry.json", tmp_path / "verify")
    assert verification["passed"] is True
    assert verification["policy_comparison"]["correlated_demand_shocks"] is True
    for record in records:
        evidence = json.loads(Path(record.evidence_paths["route_evidence"]).read_text(encoding="utf-8"))["evidence"]
        assert evidence["stress_test"]["correlated_demand_shocks"] is True


def test_cumcm2023c_report_refuses_failed_case(tmp_path: Path) -> None:
    from workflow.cumcm2023c_report import REQUIRED_CASES, build_cumcm2023c_report
    summary = {"cases": [{"case_id": case_id, "verdict": "pass"} for case_id in REQUIRED_CASES]}
    summary["cases"][2]["verdict"] = "revise"
    summary_path = tmp_path / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="report_requires_passing_cases"):
        build_cumcm2023c_report(summary_path, tmp_path / "report.md")


def test_cumcm2023c_report_verifier_rejects_mutation(tmp_path: Path) -> None:
    from workflow.cumcm2023c_report import REQUIRED_CASES, verify_cumcm2023c_report
    report_path = tmp_path / "report.md"
    report_path.write_text("verified report", encoding="utf-8")
    summary = {
        "cases": [{"case_id": case_id, "verdict": "pass"} for case_id in REQUIRED_CASES],
        "complete_report": {"path": str(report_path), "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(), "required_cases": REQUIRED_CASES},
    }
    summary_path = tmp_path / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    assert verify_cumcm2023c_report(summary_path)["passed"] is True
    report_path.write_text("mutated report", encoding="utf-8")
    verification = verify_cumcm2023c_report(summary_path)
    assert verification["passed"] is False
    assert "complete_report_hash_mismatch" in verification["issues"]


def _category_forecast_sources(tmp_path: Path) -> tuple[Path, Path]:
    products = tmp_path / "products.xlsx"
    sales = tmp_path / "sales.xlsx"
    categories = [f"category_{index}" for index in range(6)]
    codes = [str(index + 1) for index in range(6)]
    pd.DataFrame({"单品编码": codes, "分类名称": categories}).to_excel(products, index=False, sheet_name="Sheet1")
    dates = pd.date_range("2022-11-01", "2023-06-30", freq="D")
    rows = []
    for category_index, code in enumerate(codes):
        for index, date in enumerate(dates):
            demand = 20.0 + category_index * 3.0 + (index % 7) * 2.0 + 0.01 * index
            rows.append({"销售日期": date, "单品编码": code, "销量(千克)": demand})
    pd.DataFrame(rows).to_excel(sales, index=False, sheet_name="Sheet1")
    return products, sales


def test_category_forecast_selects_on_holdout_and_recomputes(tmp_path: Path) -> None:
    from workflow.category_forecast import build_category_forecast, verify_category_forecast
    products, sales = _category_forecast_sources(tmp_path)
    result = build_category_forecast(products, sales, tmp_path / "forecast")
    verification = verify_category_forecast(Path(result["manifest_path"]))
    assert verification["passed"] is True
    forecast = pd.read_csv(result["forecast_path"])
    diagnostics = pd.read_csv(result["diagnostics_path"])
    assert len(forecast) == 42 and forecast["category"].nunique() == 6
    assert diagnostics.groupby("category")["selected_by_selection"].sum().eq(1).all()
    selected = diagnostics.loc[diagnostics["selected_by_selection"].astype(bool)]
    minimum = diagnostics.groupby("category")["selection_mae"].min()
    assert all(row.selection_mae == pytest.approx(minimum[row.category]) for row in selected.itertuples())


def test_category_forecast_verifier_rejects_artifact_tampering(tmp_path: Path) -> None:
    from workflow.category_forecast import build_category_forecast, verify_category_forecast
    products, sales = _category_forecast_sources(tmp_path)
    result = build_category_forecast(products, sales, tmp_path / "forecast")
    forecast_path = Path(result["forecast_path"])
    forecast = pd.read_csv(forecast_path)
    forecast.loc[0, "base_forecast"] += 100.0
    forecast.to_csv(forecast_path, index=False)
    verification = verify_category_forecast(Path(result["manifest_path"]))
    assert verification["passed"] is False
    assert "category_forecast_artifact_hash_mismatch:forecast" in verification["issues"]


def test_problem_workflow_dag_binds_dependencies_and_artifacts(tmp_path: Path) -> None:
    from workflow.problem_dag import WorkflowNode, build_workflow_manifest, verify_workflow_manifest
    forecast = tmp_path / "forecast.csv"; forecast.write_text("value\n10\n", encoding="utf-8")
    policy = tmp_path / "policy.csv"; policy.write_text("quantity\n8\n", encoding="utf-8")
    report = tmp_path / "report.md"; report.write_text("result", encoding="utf-8")
    forecast_contract = tmp_path / "forecast_contract.json"; forecast_contract.write_text("{}", encoding="utf-8")
    nodes = [
        WorkflowNode("forecast", "prediction", contract_path=str(forecast_contract), output_artifacts={"forecast": str(forecast)}, verification={"passed": True}),
        WorkflowNode("policy", "optimization", dependencies=["forecast"], input_artifacts={"forecast": str(forecast)}, output_artifacts={"policy": str(policy)}, verification={"passed": True}),
        WorkflowNode("paper", "paper", dependencies=["policy"], input_artifacts={"policy": str(policy)}, output_artifacts={"report": str(report)}, verification={"passed": True}),
    ]
    built = build_workflow_manifest("toy", nodes, tmp_path / "workflow_manifest.json")
    assert built["passed"] is True
    assert verify_workflow_manifest(Path(built["manifest_path"]))["passed"] is True
    forecast_contract.write_text('{"changed": true}', encoding="utf-8")
    verification = verify_workflow_manifest(Path(built["manifest_path"]))
    assert verification["passed"] is False
    assert "workflow_contract_hash_mismatch:forecast" in verification["issues"]


def test_problem_workflow_dag_rejects_cycles_and_undeclared_artifact_edges(tmp_path: Path) -> None:
    from workflow.problem_dag import WorkflowNode, build_workflow_manifest, validate_workflow_nodes
    with pytest.raises(ValueError, match="workflow_dependency_cycle"):
        validate_workflow_nodes([WorkflowNode("a", "analysis", ["b"]), WorkflowNode("b", "prediction", ["a"])])
    artifact = tmp_path / "artifact.csv"; artifact.write_text("x\n1\n", encoding="utf-8")
    other = tmp_path / "other.txt"; other.write_text("ok", encoding="utf-8")
    built = build_workflow_manifest("toy", [
        WorkflowNode("producer", "analysis", output_artifacts={"artifact": str(artifact)}, verification={"passed": True}),
        WorkflowNode("consumer", "paper", input_artifacts={"artifact": str(artifact)}, output_artifacts={"other": str(other)}, verification={"passed": True}),
    ], tmp_path / "workflow_manifest.json")
    assert built["passed"] is False


def test_retail_final_judge_falls_back_when_stress_comparison_fails(tmp_path: Path) -> None:
    contract = _retail_contract(tmp_path)
    asset = DataAsset(contract.data_file, ".csv", Path(contract.data_file).stat().st_size, ["placeholder"], 1, 1, True)
    profile = ProblemProfile(contract.problem_id, "", str(tmp_path), [], ["constrained_optimization"], [asset], [], ["objective"], contract.metric_directions, [], contract)
    routes = plan_routes(profile, [], max_routes=2)
    from solver.experiment_registry import write_registry
    from solver.native_retail import run_native_retail_experiments, verify_native_retail
    records = run_native_retail_experiments(contract, routes, tmp_path / "experiments")
    registry = write_registry(records, tmp_path / "registry")
    verification = verify_native_retail(contract, records, registry, tmp_path / "verify")
    verification["policy_comparison"]["candidate_robustly_better"] = False
    result = judge_solution(profile, routes, records, tmp_path / "judge", domain_verification=verification)
    assert result["recommended_route"]["route_id"] == "retail_fixed_markup_baseline"
    assert "retail_candidate_not_stress_robust" in result["flags"]
    assert result["verdict"] == "revise"
    assert "domain_verification_stdout_mismatch:policy_comparison" in result["flags"]


def test_retail_analysis_outputs_and_rejects_tampering(tmp_path: Path) -> None:
    from workflow.retail_analysis import analyze_cumcm2023c_sales, verify_retail_analysis
    products = tmp_path / "products.xlsx"
    sales = tmp_path / "sales.xlsx"
    pd.DataFrame({"单品编码": ["1", "2"], "单品名称": ["A", "B"], "分类名称": ["叶类", "根类"]}).to_excel(products, index=False, sheet_name="Sheet1")
    dates = pd.date_range("2023-01-01", periods=80, freq="D")
    pd.DataFrame({"销售日期": list(dates) * 2, "单品编码": ["1"] * 80 + ["2"] * 80, "销量(千克)": list(np.arange(80) + 1.0) + list(np.arange(80) * 0.5 + 2.0), "销售单价(元/千克)": [5.0] * 80 + [7.0] * 80}).to_excel(sales, index=False, sheet_name="Sheet1")
    result = analyze_cumcm2023c_sales(products, sales, tmp_path / "analysis", cutoff="2023-03-31", min_item_days=30, top_edges=10)
    verification = verify_retail_analysis(Path(result["manifest_path"]))
    assert verification["passed"] is True
    assert verification["checks"]["eligible_item_count_recomputed"] is True
    edge_path = Path(result["artifacts"]["item_correlation_edges_csv"]["path"])
    edge_path.write_text("tampered", encoding="utf-8")
    assert verify_retail_analysis(Path(result["manifest_path"]))["passed"] is False


def test_native_predictive_verifier_rebuilds_default_forecast_routes(tmp_path: Path) -> None:
    from solver.experiment_registry import write_registry
    from solver.native_predictive_verifier import verify_native_predictive
    data = tmp_path / "series.csv"
    pd.DataFrame({"week": range(1, 61), "sales": [100 + 0.8 * index + 12 * np.sin(index * 2 * np.pi / 7) for index in range(60)]}).to_csv(data, index=False)
    contract = ProblemContract("v44", "verified-forecast", "forecasting", "ready", data_file=str(data), target_column="sales", time_column="week", forecast_horizon=6, validation_mode="fixed_origin_holdout", metric_directions={"MAE": "lower_better", "RMSE": "lower_better", "WAPE": "lower_better"})
    asset = DataAsset(str(data), ".csv", data.stat().st_size, ["week", "sales"], 60, 2, True)
    profile = ProblemProfile(contract.problem_id, "", str(tmp_path), [], ["forecasting"], [asset], ["sales"], ["MAE", "RMSE", "WAPE"], contract.metric_directions, [], contract)
    routes = [RouteSpec("mean_or_last_value_baseline", "baseline", "baseline", "", ["MAE", "RMSE", "WAPE"]), RouteSpec("rolling_mean_model", "candidate", "forecasting", "", ["MAE", "RMSE", "WAPE"]), RouteSpec("seasonal_naive_forecast", "candidate", "forecasting", "", ["MAE", "RMSE", "WAPE"]), RouteSpec("regularized_lag_model", "candidate", "forecasting", "", ["MAE", "RMSE", "WAPE"])]
    adapters = generate_adapters(profile, routes, tmp_path / "adapters")
    records = run_experiments(adapters, tmp_path / "experiments", timeout_seconds=60)
    registry = write_registry(records, tmp_path / "registry")
    result = verify_native_predictive(contract, records, registry, tmp_path / "verify")
    assert result["passed"] is True
    assert result["claim_level"] == "domain_verified_predictive_result"
    assert len(result["checks"]) == 4 and all(item["predictions_recomputed"] for item in result["checks"])


def test_predictive_adapter_preserves_unicode_evidence_identity(tmp_path: Path) -> None:
    data = tmp_path / "daily_sales.csv"
    pd.DataFrame({"日期": range(1, 31), "日销量": [100 + index for index in range(30)]}).to_csv(data, index=False)
    contract = ProblemContract("v44", "unicode-forecast", "forecasting", "ready", data_file=str(data), target_column="日销量", time_column="日期", forecast_horizon=4, validation_mode="fixed_origin_holdout", metric_directions={"MAE": "lower_better", "RMSE": "lower_better", "WAPE": "lower_better"})
    profile = ProblemProfile(contract.problem_id, "", str(tmp_path), [], ["forecasting"], [DataAsset(str(data), ".csv", data.stat().st_size, ["日期", "日销量"], 30, 2, True)], ["日销量"], ["MAE", "RMSE", "WAPE"], contract.metric_directions, [], contract)
    route = RouteSpec("seasonal_naive_forecast", "candidate", "forecasting", "", ["MAE", "RMSE", "WAPE"])
    records = run_experiments(generate_adapters(profile, [route], tmp_path / "adapters"), tmp_path / "experiments", timeout_seconds=60)
    assert records[0].status == "executed"
    evidence = json.loads(Path(records[0].evidence_paths["route_evidence"]).read_text(encoding="utf-8"))
    assert evidence["evidence"]["target_column"] == "日销量"


def test_native_predictive_verifier_rejects_tampered_ridge_prediction(tmp_path: Path) -> None:
    from solver.experiment_registry import write_registry
    from solver.native_predictive_verifier import verify_native_predictive
    data = tmp_path / "series.csv"; pd.DataFrame({"week": range(1, 41), "sales": [index + index % 4 for index in range(40)]}).to_csv(data, index=False)
    contract = ProblemContract("v44", "tampered-forecast", "forecasting", "ready", data_file=str(data), target_column="sales", time_column="week", forecast_horizon=4, validation_mode="fixed_origin_holdout", metric_directions={"MAE": "lower_better", "RMSE": "lower_better", "WAPE": "lower_better"})
    profile = ProblemProfile(contract.problem_id, "", str(tmp_path), [], ["forecasting"], [DataAsset(str(data), ".csv", data.stat().st_size, ["week", "sales"], 40, 2, True)], ["sales"], ["MAE"], contract.metric_directions, [], contract)
    route = RouteSpec("regularized_lag_model", "candidate", "forecasting", "", ["MAE", "RMSE", "WAPE"])
    records = run_experiments(generate_adapters(profile, [route], tmp_path / "adapters"), tmp_path / "experiments", timeout_seconds=60)
    evidence_path = Path(records[0].evidence_paths["route_evidence"]); payload = json.loads(evidence_path.read_text(encoding="utf-8")); payload["evidence"]["predicted"][0] += 2.0; evidence_path.write_text(json.dumps(payload), encoding="utf-8")
    registry = write_registry(records, tmp_path / "registry")
    assert verify_native_predictive(contract, records, registry, tmp_path / "verify")["passed"] is False


def test_cumcm2023c_pdf_verifier_rejects_mutation(tmp_path: Path) -> None:
    from workflow.cumcm2023c_pdf import verify_cumcm2023c_pdf

    source = tmp_path / "summary.json"
    source.write_text("{}", encoding="utf-8")
    tex = tmp_path / "paper.tex"
    tex.write_text("paper", encoding="utf-8")
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-test")
    log = tmp_path / "build.log"
    log.write_text("ok", encoding="utf-8")

    def binding(path: Path) -> dict[str, str]:
        return {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    manifest = {
        "sources": [binding(source)],
        "tex": binding(tex),
        "pdf": binding(pdf),
        "log": binding(log),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert verify_cumcm2023c_pdf(manifest_path)["passed"] is True

    pdf.write_bytes(b"%PDF-mutated")
    verification = verify_cumcm2023c_pdf(manifest_path)
    assert verification["passed"] is False
    assert "submission_pdf_hash_mismatch" in verification["issues"]


@pytest.mark.parametrize("grouped", [False, True])
def test_native_predictive_verifier_rebuilds_regression_routes(tmp_path: Path, grouped: bool) -> None:
    from solver.experiment_registry import write_registry
    from solver.native_predictive_verifier import verify_native_predictive
    size = 90
    data = tmp_path / "regression.csv"
    pd.DataFrame({"entity": [index % 15 for index in range(size)], "x1": np.arange(size), "x2": np.arange(size) % 7, "target": [3 + 1.5 * index + 2 * (index % 7) for index in range(size)]}).to_csv(data, index=False)
    groups = ["entity"] if grouped else []
    contract = ProblemContract("v44", f"regression-{grouped}", "regression", "ready", data_file=str(data), target_column="target", feature_columns=["x1", "x2"], group_columns=groups, validation_mode="group_holdout" if grouped else "random_holdout", metric_directions={"MAE": "lower_better", "RMSE": "lower_better", "R2": "higher_better"})
    profile = ProblemProfile(contract.problem_id, "", str(tmp_path), [], [], [DataAsset(str(data), ".csv", data.stat().st_size, ["entity", "x1", "x2", "target"], size, 4, True)], ["target"], ["MAE", "RMSE", "R2"], contract.metric_directions, [], contract)
    routes = [RouteSpec("simple_statistical_baseline", "baseline", "baseline", "", ["MAE", "RMSE", "R2"]), RouteSpec("regularized_regression_cv", "candidate", "tabular", "", ["MAE", "RMSE", "R2"])]
    records = run_experiments(generate_adapters(profile, routes, tmp_path / "adapters"), tmp_path / "experiments", timeout_seconds=60)
    registry = write_registry(records, tmp_path / "registry"); verification = verify_native_predictive(contract, records, registry, tmp_path / "verify")
    assert verification["passed"] is True
    assert verification["claim_level"] == "domain_verified_predictive_result"
    assert all(item["predictions_recomputed"] and item["split_rows_recomputed"] for item in verification["checks"])
