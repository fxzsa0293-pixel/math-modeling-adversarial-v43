from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .schemas import ExperimentRecord, ProblemContract, RouteSpec


def _objective_value(objective: Dict[str, Any], decisions: Dict[str, float]) -> float:
    return float(sum(float(value) * decisions.get(str(name), 0.0) for name, value in objective["coefficients"].items()))


def _minimization_vector(objective: Dict[str, Any], names: List[str]) -> np.ndarray:
    sign = 1.0 if objective["sense"] == "min" else -1.0
    return np.asarray([sign * float(objective["coefficients"].get(name, 0.0)) for name in names], dtype=float)


def _solve(base: Dict[str, Any], objective: Dict[str, Any], extra_constraints: List[Dict[str, Any]]) -> Dict[str, Any]:
    variables = base["variables"]
    names = [str(item["name"]) for item in variables]
    rows: List[List[float]] = []
    lower_constraints: List[float] = []
    upper_constraints: List[float] = []
    for item in list(base.get("linear_constraints", [])) + extra_constraints:
        rows.append([float(item.get("coefficients", {}).get(name, 0.0)) for name in names])
        rhs = float(item["rhs"])
        if item["sense"] == "<=":
            lower_constraints.append(-math.inf); upper_constraints.append(rhs)
        elif item["sense"] == ">=":
            lower_constraints.append(rhs); upper_constraints.append(math.inf)
        else:
            lower_constraints.append(rhs); upper_constraints.append(rhs)
    constraints = LinearConstraint(np.asarray(rows), np.asarray(lower_constraints), np.asarray(upper_constraints)) if rows else None
    lower = np.asarray([float(item.get("lower", 0.0)) for item in variables])
    upper = np.asarray([1.0 if item.get("kind") == "binary" else float(item.get("upper", math.inf)) for item in variables])
    integrality = np.asarray([0 if item.get("kind", "continuous") == "continuous" else 1 for item in variables])
    result = milp(c=_minimization_vector(objective, names), integrality=integrality, bounds=Bounds(lower, upper), constraints=constraints, options={"time_limit": 30.0})
    if result.x is None:
        return {"status": int(result.status), "model_status": "infeasible" if int(result.status) == 2 else "no_solution", "decisions": {}}
    decisions = {name: float(value) for name, value in zip(names, result.x)}
    return {"status": int(result.status), "model_status": "optimal" if int(result.status) == 0 and bool(result.success) else "feasible_not_proven_optimal", "decisions": decisions}


def _epsilon_constraint(objective: Dict[str, Any], epsilon: float) -> Dict[str, Any]:
    return {"name": f"epsilon:{objective['name']}", "coefficients": dict(objective["coefficients"]), "sense": "<=" if objective["sense"] == "min" else ">=", "rhs": float(epsilon)}


def _dominates(a: List[float], b: List[float], objectives: List[Dict[str, Any]]) -> bool:
    no_worse = True
    strictly_better = False
    for av, bv, objective in zip(a, b, objectives):
        if objective["sense"] == "min":
            no_worse &= av <= bv + 1e-9; strictly_better |= av < bv - 1e-9
        else:
            no_worse &= av >= bv - 1e-9; strictly_better |= av > bv + 1e-9
    return bool(no_worse and strictly_better)


def _pareto(solutions: List[Dict[str, Any]], objectives: List[Dict[str, Any]], variable_names: List[str]) -> List[Dict[str, Any]]:
    result = []
    for index, candidate in enumerate(solutions):
        if any(_dominates(other["objective_values"], candidate["objective_values"], objectives) for j, other in enumerate(solutions) if j != index):
            continue
        candidate_vector = [float(candidate["decision_values"][name]) for name in variable_names]
        if not any(np.allclose(candidate_vector, [float(existing["decision_values"][name]) for name in variable_names], rtol=1e-8, atol=1e-8) for existing in result):
            result.append(candidate)
    return result


def solve_multiobjective_model(contract: Dict[str, Any], route: Dict[str, Any]) -> Dict[str, Any]:
    model = contract["multiobjective_model"]
    base = model["base_model"]
    objectives = model["objectives"]
    anchors = []
    for objective in objectives:
        solved = _solve(base, objective, [])
        if not solved["decisions"]:
            raise ValueError(f"multiobjective_anchor_failed:{objective['name']}")
        decisions = solved["decisions"]
        anchors.append({"decision_values": decisions, "objective_values": [_objective_value(item, decisions) for item in objectives], "status": solved["model_status"], "anchor": objective["name"]})
    if route["route_id"] == "multiobjective_single_objective_baseline":
        solutions = anchors[:1]
    else:
        solutions = list(anchors)
        grid_size = int(model.get("epsilon_grid_size", 7))
        for constrained_index, constrained in enumerate(objectives[1:], start=1):
            values = [item["objective_values"][constrained_index] for item in anchors]
            low, high = min(values), max(values)
            for epsilon in np.linspace(low, high, grid_size):
                solved = _solve(base, objectives[0], [_epsilon_constraint(constrained, float(epsilon))])
                if solved["decisions"]:
                    decisions = solved["decisions"]
                    solutions.append({"decision_values": decisions, "objective_values": [_objective_value(item, decisions) for item in objectives], "status": solved["model_status"], "epsilon": float(epsilon), "constrained_objective": constrained["name"]})
    variable_names = [str(item["name"]) for item in base["variables"]]
    pareto = _pareto(solutions, objectives, variable_names)
    feasible = all(item["status"] in {"optimal", "feasible_not_proven_optimal"} for item in pareto)
    metrics = {"pareto_count": len(pareto), "candidate_solution_count": len(solutions), "feasible": bool(feasible), "anchor_count": len(anchors), "objective_count": len(objectives), "epsilon_grid_size": int(model.get("epsilon_grid_size", 7)), "validation_split": "declared_constraint_pareto_front"}
    return {"metrics": metrics, "evidence": {"objectives": objectives, "solutions": pareto, "anchors": anchors, "base_model": base}}


def run_native_multiobjective_experiments(contract: ProblemContract, routes: Iterable[RouteSpec], output_dir: Path) -> list[ExperimentRecord]:
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir = output_dir / "evidence"; evidence_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex; code_path = Path(__file__).resolve(); input_path = Path(contract.data_file).resolve() if contract.data_file else code_path
    records = []
    for route in routes:
        try:
            solved = solve_multiobjective_model(asdict(contract), route.__dict__)
            evidence_path = evidence_dir / f"{route.route_id}.json"
            payload = {"run_id": run_id, "problem_id": contract.problem_id, "route_id": route.route_id, "data_path": str(input_path), "contract": asdict(contract), "route": route.__dict__, "environment": {"numpy": np.__version__}, "metrics_recomputed": solved["metrics"], "evidence": solved["evidence"]}
            evidence_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            record = ExperimentRecord(contract.problem_id, route.route_id, route.role, "executed", str(code_path), str(input_path), solved["metrics"], original_code_hash=_sha256(code_path), final_code_hash=_sha256(code_path), run_id=run_id, input_hash=_sha256(input_path), config_hash=_json_hash({"contract": asdict(contract), "route": route.__dict__}), artifact_hashes={"route_evidence": _sha256(evidence_path)}, evidence_paths={"route_evidence": str(evidence_path)}, environment={"numpy": np.__version__})
        except Exception as exc:
            record = ExperimentRecord(contract.problem_id, route.route_id, route.role, "failed", str(code_path), str(input_path), {}, str(exc), original_code_hash=_sha256(code_path), final_code_hash=_sha256(code_path), run_id=run_id, input_hash=_sha256(input_path))
        records.append(record); (output_dir / f"{route.route_id}.json").write_text(json.dumps(record.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
    return records


def verify_native_multiobjective(contract: ProblemContract, records: Iterable[ExperimentRecord], registry_path: Path, output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True); records = list(records); issues = []; checks = []; registry_path = Path(registry_path)
    if not registry_path.is_file():
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text(json.dumps([record.__dict__ for record in records], ensure_ascii=False), encoding="utf-8")
    for record in records:
        if record.status != "executed" or record.role in {"ablation", "diagnostic"}: continue
        try:
            payload = json.loads(Path(record.evidence_paths["route_evidence"]).read_text(encoding="utf-8")); evidence = payload["evidence"]; objectives = evidence["objectives"]; base = evidence["base_model"]; solutions = evidence["solutions"]
            feasible_ok = True; pareto_ok = True; values_ok = True; integrality_ok = True
            for solution in solutions:
                decisions = {str(name): float(value) for name, value in solution["decision_values"].items()}
                for item in base["variables"]:
                    value = decisions.get(str(item["name"]))
                    if value is None or value < float(item.get("lower", 0.0)) - 1e-7 or value > float(item.get("upper", 1.0 if item.get("kind") == "binary" else math.inf)) + 1e-7:
                        feasible_ok = False
                    if value is not None and item.get("kind", "continuous") in {"integer", "binary"} and not math.isclose(value, round(value), abs_tol=1e-7):
                        integrality_ok = False
                for item in base.get("linear_constraints", []):
                    lhs = sum(float(coef) * decisions.get(str(name), 0.0) for name, coef in item["coefficients"].items()); rhs = float(item["rhs"])
                    feasible_ok &= (lhs <= rhs + 1e-7 if item["sense"] == "<=" else lhs >= rhs - 1e-7 if item["sense"] == ">=" else abs(lhs - rhs) <= 1e-7)
                recomputed = [_objective_value(item, decisions) for item in objectives]
                values_ok &= np.allclose(recomputed, solution["objective_values"], rtol=1e-8, atol=1e-8)
            for i, first in enumerate(solutions):
                for j, second in enumerate(solutions):
                    if i != j and _dominates(first["objective_values"], second["objective_values"], objectives): pareto_ok = False
            metrics_ok = int(record.metrics.get("pareto_count", -1)) == len(solutions) and bool(record.metrics.get("feasible")) == bool(feasible_ok and integrality_ok)
            checks.append({"route_id": record.route_id, "feasible": feasible_ok, "integrality": integrality_ok, "objective_values_recomputed": values_ok, "non_dominated": pareto_ok, "metrics_ok": metrics_ok, "pareto_count": len(solutions)})
            if not feasible_ok or not integrality_ok or not values_ok or not pareto_ok or not metrics_ok: issues.append(f"native_multiobjective_verification_failed:{record.route_id}")
        except Exception as exc: issues.append(f"native_multiobjective_verification_error:{record.route_id}:{exc}")
    stdout = output_dir / "native_multiobjective_verifier.stdout.txt"; stderr = output_dir / "native_multiobjective_verifier.stderr.txt"; stdout.write_text(json.dumps({"checks": checks}, ensure_ascii=False), encoding="utf-8"); stderr.write_text("", encoding="utf-8")
    result = {"kind": "native_multiobjective", "passed": not issues and bool(checks), "issues": issues, "checks": checks, "claim_level": "domain_verified_sampled_pareto_set" if not issues and any(item["route_id"] == "multiobjective_epsilon_constraint" for item in checks) else "domain_verified_multiobjective_baseline", "run_id": next(iter({record.run_id for record in records if record.run_id}), None), "contract_hash": _json_hash(asdict(contract)), "script_path": str(Path(__file__).resolve()), "script_hash": _sha256(Path(__file__).resolve()), "registry_path": str(registry_path.resolve()), "registry_hash": _sha256(registry_path), "stdout_path": str(stdout), "stdout_hash": _sha256(stdout), "stderr_path": str(stderr), "stderr_hash": _sha256(stderr), "exit_code": 0 if not issues else 1}
    (output_dir / "domain_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"); return result


def _sha256(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def _json_hash(payload: Any) -> str: return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
