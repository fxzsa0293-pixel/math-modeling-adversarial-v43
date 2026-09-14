from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .schemas import ExperimentRecord, ProblemContract, RouteSpec


def _risk_value(costs: list[float], probabilities: list[float], risk: str, alpha: float) -> float:
    if risk == "expected":
        return float(np.dot(costs, probabilities))
    if risk == "worst_case":
        return float(max(costs))
    return float(min(eta + sum(probability * max(0.0, cost - eta) for cost, probability in zip(costs, probabilities)) / (1.0 - alpha) for eta in costs))


def solve_robust_model(contract: Dict[str, Any], route: Dict[str, Any]) -> Dict[str, Any]:
    model = contract["robust_optimization_model"]; variables = model["variables"]; scenarios = model["scenarios"]
    names = [str(item["name"]) for item in variables]; n = len(names)
    probabilities = [float(item["probability"]) for item in scenarios]
    scenario_c = [np.asarray([float(item["objective_coefficients"].get(name, 0.0)) for name in names]) for item in scenarios]
    declared_risk = str(model.get("risk_measure", "worst_case")); alpha = float(model.get("alpha", 0.95))
    optimized_risk = "expected" if route["route_id"] == "robust_expected_baseline" else declared_risk
    auxiliary_names: list[str] = []
    if optimized_risk == "expected":
        c = sum(probability * vector for probability, vector in zip(probabilities, scenario_c)); total = n
    elif optimized_risk == "worst_case":
        auxiliary_names = ["__worst_cost"]; total = n + 1; c = np.zeros(total); c[n] = 1.0
    else:
        auxiliary_names = ["__cvar_eta"] + [f"__cvar_excess_{index}" for index in range(len(scenarios))]
        total = n + 1 + len(scenarios); c = np.zeros(total); c[n] = 1.0
        for index, probability in enumerate(probabilities): c[n + 1 + index] = probability / (1.0 - alpha)
    lower = [float(item.get("lower", 0.0)) for item in variables]
    upper = [1.0 if item.get("kind") == "binary" else float(item.get("upper", math.inf)) for item in variables]
    integrality = [0 if item.get("kind", "continuous") == "continuous" else 1 for item in variables]
    if optimized_risk == "worst_case": lower += [-math.inf]; upper += [math.inf]; integrality += [0]
    elif optimized_risk == "cvar": lower += [-math.inf] + [0.0] * len(scenarios); upper += [math.inf] * (1 + len(scenarios)); integrality += [0] * (1 + len(scenarios))
    rows = []; lbs = []; ubs = []
    for scenario in scenarios:
        for constraint in scenario["linear_constraints"]:
            row = [float(constraint["coefficients"].get(name, 0.0)) for name in names] + [0.0] * (total - n)
            rhs = float(constraint["rhs"]); rows.append(row)
            if constraint["sense"] == "<=": lbs.append(-math.inf); ubs.append(rhs)
            elif constraint["sense"] == ">=": lbs.append(rhs); ubs.append(math.inf)
            else: lbs.append(rhs); ubs.append(rhs)
    if optimized_risk == "worst_case":
        for vector in scenario_c:
            row = list(vector) + [-1.0]; rows.append(row); lbs.append(-math.inf); ubs.append(0.0)
    elif optimized_risk == "cvar":
        for index, vector in enumerate(scenario_c):
            row = list(vector) + [-1.0] + [0.0] * len(scenarios); row[n + 1 + index] = -1.0
            rows.append(row); lbs.append(-math.inf); ubs.append(0.0)
    constraints = LinearConstraint(np.asarray(rows), np.asarray(lbs), np.asarray(ubs)) if rows else None
    result = milp(c=np.asarray(c), integrality=np.asarray(integrality), bounds=Bounds(np.asarray(lower), np.asarray(upper)), constraints=constraints, options={"time_limit": 30.0})
    if result.x is None:
        return {"metrics": {"objective": None, "constraint_violation": None, "model_status": "infeasible", "has_feasible_solution": False, "optimality_proven": False, "solver_status": int(result.status)}, "evidence": {"decision_variables": {}, "scenario_results": []}}
    decisions = {name: float(value) for name, value in zip(names, result.x[:n])}
    costs = [float(np.dot(vector, result.x[:n])) for vector in scenario_c]
    checks = []; violations = []
    for scenario, cost in zip(scenarios, costs):
        scenario_checks = []
        for constraint in scenario["linear_constraints"]:
            lhs = sum(float(coef) * decisions[name] for name, coef in constraint["coefficients"].items()); rhs = float(constraint["rhs"]); sense = constraint["sense"]
            violation = max(0.0, lhs - rhs) if sense == "<=" else max(0.0, rhs - lhs) if sense == ">=" else abs(lhs - rhs)
            violations.append(violation); scenario_checks.append({"name": constraint["name"], "lhs": lhs, "sense": sense, "rhs": rhs, "violation": violation})
        checks.append({"name": scenario["name"], "probability": float(scenario["probability"]), "cost": cost, "constraint_checks": scenario_checks})
    objective = _risk_value(costs, probabilities, declared_risk, alpha)
    status = "optimal" if int(result.status) == 0 and bool(result.success) else "feasible_not_proven_optimal"
    return {"metrics": {"objective": objective, "expected_cost": _risk_value(costs, probabilities, "expected", alpha), "worst_case_cost": max(costs), "constraint_violation": sum(violations), "scenario_count": len(scenarios), "risk_measure": declared_risk, "optimized_risk_measure": optimized_risk, "model_status": status, "has_feasible_solution": True, "optimality_proven": bool(status == "optimal" and optimized_risk == declared_risk), "solver_status": int(result.status), "mip_gap": float(getattr(result, "mip_gap", 0.0) or 0.0)}, "evidence": {"decision_variables": decisions, "scenario_results": checks, "declared_risk_measure": declared_risk, "optimized_risk_measure": optimized_risk, "alpha": alpha, "auxiliary_variables": {name: float(value) for name, value in zip(auxiliary_names, result.x[n:])}}}


def run_native_robust_experiments(contract: ProblemContract, routes: Iterable[RouteSpec], output_dir: Path) -> list[ExperimentRecord]:
    output_dir.mkdir(parents=True, exist_ok=True); evidence_dir = output_dir / "evidence"; evidence_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex; code_path = Path(__file__).resolve(); input_path = Path(contract.data_file).resolve(); records = []
    for route in routes:
        try:
            solved = solve_robust_model(asdict(contract), route.__dict__); evidence_path = evidence_dir / f"{route.route_id}.json"
            payload = {"run_id": run_id, "problem_id": contract.problem_id, "route_id": route.route_id, "data_path": str(input_path), "contract": asdict(contract), "route": route.__dict__, "metrics_recomputed": solved["metrics"], "evidence": solved["evidence"]}
            evidence_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            record = ExperimentRecord(contract.problem_id, route.route_id, route.role, "executed", str(code_path), str(input_path), solved["metrics"], original_code_hash=_sha256(code_path), final_code_hash=_sha256(code_path), run_id=run_id, input_hash=_sha256(input_path), config_hash=_json_hash({"contract": asdict(contract), "route": route.__dict__}), artifact_hashes={"route_evidence": _sha256(evidence_path)}, evidence_paths={"route_evidence": str(evidence_path)}, environment={"scipy": __import__("scipy").__version__})
        except Exception as exc:
            record = ExperimentRecord(contract.problem_id, route.route_id, route.role, "failed", str(code_path), str(input_path), {}, str(exc), original_code_hash=_sha256(code_path), final_code_hash=_sha256(code_path), run_id=run_id, input_hash=_sha256(input_path))
        records.append(record); (output_dir / f"{route.route_id}.json").write_text(json.dumps(record.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
    return records


def verify_native_robust(contract: ProblemContract, records: Iterable[ExperimentRecord], registry_path: Path, output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True); records = list(records); checks = []; issues = []; model = contract.robust_optimization_model
    registry_path = Path(registry_path)
    if not registry_path.is_file():
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text(json.dumps([record.__dict__ for record in records], ensure_ascii=False), encoding="utf-8")
    names = [str(item["name"]) for item in model["variables"]]; probabilities = [float(item["probability"]) for item in model["scenarios"]]; risk = str(model.get("risk_measure", "worst_case")); alpha = float(model.get("alpha", 0.95))
    for record in records:
        if record.status != "executed" or record.role in {"ablation", "diagnostic"}: continue
        try:
            payload = json.loads(Path(record.evidence_paths["route_evidence"]).read_text(encoding="utf-8")); evidence = payload["evidence"]; decisions = {name: float(evidence["decision_variables"][name]) for name in names}
            model_matches = payload.get("contract", {}).get("robust_optimization_model") == model
            route_matches = payload.get("route_id") == record.route_id and payload.get("route", {}).get("route_id") == record.route_id
            bounds_ok = all(float(item.get("lower", 0.0)) - 1e-7 <= decisions[item["name"]] <= float(item.get("upper", 1.0 if item.get("kind") == "binary" else math.inf)) + 1e-7 for item in model["variables"]); integer_ok = all(abs(decisions[item["name"]] - round(decisions[item["name"]])) <= 1e-7 for item in model["variables"] if item.get("kind", "continuous") in {"integer", "binary"})
            costs = []; violation = 0.0
            for scenario in model["scenarios"]:
                costs.append(sum(float(coef) * decisions[name] for name, coef in scenario["objective_coefficients"].items()))
                for constraint in scenario["linear_constraints"]:
                    lhs = sum(float(coef) * decisions[name] for name, coef in constraint["coefficients"].items()); rhs = float(constraint["rhs"]); sense = constraint["sense"]
                    violation += max(0.0, lhs - rhs) if sense == "<=" else max(0.0, rhs - lhs) if sense == ">=" else abs(lhs - rhs)
            objective = _risk_value(costs, probabilities, risk, alpha)
            objective_ok = math.isclose(float(record.metrics["objective"]), objective, rel_tol=1e-8, abs_tol=1e-8); violation_ok = math.isclose(float(record.metrics["constraint_violation"]), violation, rel_tol=1e-8, abs_tol=1e-8)
            stored = evidence["scenario_results"]; scenario_ok = len(stored) == len(costs)
            for stored_scenario, scenario, cost in zip(stored, model["scenarios"], costs):
                scenario_ok &= stored_scenario["name"] == scenario["name"] and math.isclose(float(stored_scenario["cost"]), cost, rel_tol=1e-8, abs_tol=1e-8)
                stored_checks = stored_scenario.get("constraint_checks", [])
                scenario_ok &= len(stored_checks) == len(scenario["linear_constraints"])
                for stored_check, constraint in zip(stored_checks, scenario["linear_constraints"]):
                    lhs = sum(float(coef) * decisions[name] for name, coef in constraint["coefficients"].items())
                    scenario_ok &= stored_check.get("name") == constraint["name"] and stored_check.get("sense") == constraint["sense"] and math.isclose(float(stored_check.get("lhs")), lhs, rel_tol=1e-8, abs_tol=1e-8) and math.isclose(float(stored_check.get("rhs")), float(constraint["rhs"]), rel_tol=1e-8, abs_tol=1e-8)
            rerun = solve_robust_model(asdict(contract), {"route_id": record.route_id})
            optimum_ok = math.isclose(float(record.metrics["objective"]), float(rerun["metrics"]["objective"]), rel_tol=1e-8, abs_tol=1e-8)
            item = {"route_id": record.route_id, "model_matches": model_matches, "route_matches": route_matches, "bounds": bounds_ok, "integrality": integer_ok, "all_scenarios_feasible": violation <= 1e-7, "objective_recomputed": objective_ok, "constraint_violation_recomputed": violation_ok, "scenario_evidence_recomputed": scenario_ok, "optimal_objective_recomputed": optimum_ok}; checks.append(item)
            if not all(value for key, value in item.items() if key != "route_id"): issues.append(f"native_robust_verification_failed:{record.route_id}")
        except Exception as exc: issues.append(f"native_robust_verification_error:{record.route_id}:{exc}")
    stdout = output_dir / "native_robust_verifier.stdout.txt"; stderr = output_dir / "native_robust_verifier.stderr.txt"
    stdout.write_text(json.dumps({"checks": checks}, ensure_ascii=False), encoding="utf-8"); stderr.write_text("", encoding="utf-8"); run_ids = {record.run_id for record in records if record.run_id}
    result = {"kind": "native_robust_optimization", "passed": not issues and bool(checks), "issues": issues, "checks": checks, "claim_level": "domain_verified_scenario_robust_result" if not issues else "no_numerical_claim", "run_id": next(iter(run_ids)) if len(run_ids) == 1 else None, "contract_hash": _json_hash(asdict(contract)), "script_path": str(Path(__file__).resolve()), "script_hash": _sha256(Path(__file__).resolve()), "registry_path": str(registry_path.resolve()), "registry_hash": _sha256(registry_path), "stdout_path": str(stdout), "stdout_hash": _sha256(stdout), "stderr_path": str(stderr), "stderr_hash": _sha256(stderr), "exit_code": 0 if not issues else 1}
    (output_dir / "domain_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"); return result


def _sha256(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def _json_hash(payload: Any) -> str: return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
