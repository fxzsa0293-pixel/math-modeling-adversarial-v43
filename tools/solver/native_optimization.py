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


def solve_declared_model(contract: Dict[str, Any], route: Dict[str, Any]) -> Dict[str, Any]:
    model = contract["optimization_model"]
    variables = model["variables"]
    names = [item["name"] for item in variables]
    objective = model["objective"]
    sign = 1.0 if objective["sense"] == "min" else -1.0
    c = np.asarray([sign * float(objective["coefficients"].get(name, 0.0)) for name in names])
    lower = np.asarray([float(item.get("lower", 0.0)) for item in variables])
    upper = np.asarray([1.0 if item.get("kind") == "binary" else float(item.get("upper", math.inf)) for item in variables])
    rows, lb, ub = [], [], []
    checks = []
    for item in model["linear_constraints"]:
        rows.append([float(item["coefficients"].get(name, 0.0)) for name in names])
        rhs = float(item["rhs"])
        if item["sense"] == "<=":
            lb.append(-math.inf); ub.append(rhs)
        elif item["sense"] == ">=":
            lb.append(rhs); ub.append(math.inf)
        else:
            lb.append(rhs); ub.append(rhs)
    integrality = np.zeros(len(names), dtype=int)
    if route["route_id"] != "linear_relaxation_diagnostic":
        integrality = np.asarray([0 if item.get("kind", "continuous") == "continuous" else 1 for item in variables])
    constraints = LinearConstraint(np.asarray(rows), np.asarray(lb), np.asarray(ub)) if rows else None
    if route["route_id"] == "lower_bound_feasible_baseline":
        # A feasibility solve is a valid baseline even when variable lower bounds
        # do not satisfy coupling constraints such as x + y >= 10.
        feasibility = milp(
            c=np.zeros(len(names)),
            integrality=integrality,
            bounds=Bounds(lower, upper),
            constraints=constraints,
            options={"time_limit": 30.0},
        )
        solver_status = int(feasibility.status)
        solver_result = feasibility
        model_status = _model_status(feasibility, feasibility.x is not None)
    else:
        result = milp(c=c, integrality=integrality, bounds=Bounds(lower, upper), constraints=constraints, options={"time_limit": 30.0})
        solver_status = int(result.status)
        solver_result = result
        model_status = _model_status(result, result.x is not None)
    values = np.asarray(solver_result.x, dtype=float) if solver_result.x is not None else None
    decisions = {name: float(value) for name, value in zip(names, values)} if values is not None else {}
    objective_value = sum(float(objective["coefficients"].get(name, 0.0)) * decisions[name] for name in names) if values is not None else None
    for item, row in zip(model["linear_constraints"], rows):
        checks.append({
            "name": item["name"], "lhs": float(np.dot(row, values)) if values is not None else None, "sense": item["sense"],
            "rhs": float(item["rhs"]), "tolerance": 1e-7,
        })
    violations = []
    for item in checks:
        if item["lhs"] is None:
            continue
        if item["sense"] == "<=":
            violations.append(max(0.0, item["lhs"] - item["rhs"] - item["tolerance"]))
        elif item["sense"] == ">=":
            violations.append(max(0.0, item["rhs"] - item["lhs"] - item["tolerance"]))
        else:
            violations.append(max(0.0, abs(item["lhs"] - item["rhs"]) - item["tolerance"]))
    dual_bound = getattr(solver_result, "mip_dual_bound", None)
    objective_bound = None
    if route["route_id"] != "lower_bound_feasible_baseline" and dual_bound is not None:
        objective_bound = float(sign * dual_bound)
    return {
        "metrics": {
            "objective": objective_value,
            "constraint_violation": sum(violations) if values is not None else None,
            "constraint_count": len(checks),
            "solver_status": solver_status,
            "solver_message": str(solver_result.message),
            "model_status": model_status,
            "has_feasible_solution": bool(values is not None),
            "optimality_proven": bool(route["route_id"] != "lower_bound_feasible_baseline" and model_status == "optimal"),
            "mip_gap": float(getattr(solver_result, "mip_gap", 0.0) or 0.0),
            "objective_bound": objective_bound,
        },
        "evidence": {
            "decision_variables": decisions,
            "objective_components": [{"name": "declared_linear_objective", "value": objective_value}],
            "constraint_checks": checks,
        },
    }


def solve_network_model(contract: Dict[str, Any], route: Dict[str, Any]) -> Dict[str, Any]:
    """Compile a standard network model to MILP, retaining semantic evidence."""
    network = contract["network_model"]
    kind = network["kind"]
    edges = network["edges"]
    names = [str(edge["id"]) for edge in edges]
    variables = []
    for edge in edges:
        variables.append({"name": str(edge["id"]), "kind": "binary" if kind in {"shortest_path", "assignment"} else "continuous", "lower": float(edge.get("lower", 0.0)), "upper": float(edge.get("capacity", 1.0 if kind in {"shortest_path", "assignment"} else math.inf))})
    if kind == "max_flow":
        variables.append({"name": "__network_flow_value", "kind": "continuous", "lower": 0.0})
    coefficients = {str(edge["id"]): float(edge.get("cost", 0.0)) for edge in edges}
    if kind == "max_flow":
        coefficients = {"__network_flow_value": 1.0}
    constraints = []
    nodes = [str(node) for node in network["nodes"]]
    if kind == "shortest_path":
        supplies = {str(network["source"]): 1.0, str(network["sink"]): -1.0}
    elif kind == "max_flow":
        supplies = {str(network["source"]): 0.0, str(network["sink"]): 0.0}
    else:
        supplies = {str(node): float(value) for node, value in network["supplies"].items()}
    for node in nodes:
        row = {}
        for edge in edges:
            edge_id = str(edge["id"])
            if str(edge["from"]) == node:
                row[edge_id] = row.get(edge_id, 0.0) + 1.0
            if str(edge["to"]) == node:
                row[edge_id] = row.get(edge_id, 0.0) - 1.0
        if kind == "max_flow" and node in {str(network["source"]), str(network["sink"])}:
            row["__network_flow_value"] = -1.0 if node == str(network["source"]) else 1.0
        if kind in {"shortest_path", "min_cost_flow", "assignment", "transportation"}:
            constraints.append({"name": f"flow_balance:{node}", "coefficients": row, "sense": "==", "rhs": supplies.get(node, 0.0)})
        elif kind == "max_flow" and node not in {str(network["source"]), str(network["sink"])}:
            constraints.append({"name": f"flow_balance:{node}", "coefficients": row, "sense": "==", "rhs": 0.0})
        elif kind == "max_flow":
            constraints.append({"name": f"flow_balance:{node}", "coefficients": row, "sense": "==", "rhs": 0.0})
    compiled = {"optimization_model": {"variables": variables, "objective": {"sense": "max" if kind == "max_flow" else "min", "coefficients": coefficients}, "linear_constraints": constraints}}
    result = solve_declared_model(compiled, route)
    flow = {edge_id: result["evidence"]["decision_variables"].get(edge_id) for edge_id in names}
    result["evidence"]["network"] = {"kind": kind, "edge_flow": flow, "node_balance": _network_balances(network, flow), "compiled_model": compiled["optimization_model"]}
    if kind == "max_flow":
        result["evidence"]["network"]["flow_value"] = result["evidence"]["decision_variables"].get("__network_flow_value")
    result["metrics"]["network_kind"] = kind
    result["metrics"]["network_feasible"] = bool(result["metrics"].get("has_feasible_solution") and _network_semantic_issues(network, result["evidence"]["network"])[0] == [])
    return result


def _network_balances(network: Dict[str, Any], flow: Dict[str, Any]) -> Dict[str, float]:
    balances = {str(node): 0.0 for node in network["nodes"]}
    for edge in network["edges"]:
        value = float(flow.get(str(edge["id"])) or 0.0)
        balances[str(edge["from"])] += value
        balances[str(edge["to"])] -= value
    return balances


def _network_semantic_issues(network: Dict[str, Any], evidence: Dict[str, Any]) -> tuple[list[str], Dict[str, Any]]:
    """Independently check the original network declaration and its edge flows."""
    kind = network["kind"]
    flow = evidence.get("edge_flow", {})
    issues: list[str] = []
    edges = network["edges"]
    for edge in edges:
        edge_id = str(edge["id"])
        value = flow.get(edge_id)
        if value is None:
            issues.append(f"missing_edge_flow:{edge_id}")
            continue
        value = float(value)
        lower = float(edge.get("lower", 0.0))
        upper = float(edge.get("capacity", math.inf))
        if value < lower - 1e-7 or value > upper + 1e-7:
            issues.append(f"edge_bound_violation:{edge_id}")
        if kind in {"shortest_path", "assignment"} and abs(value - round(value)) > 1e-7:
            issues.append(f"edge_integrality_violation:{edge_id}")
    balances = evidence.get("node_balance", {})
    expected: Dict[str, float] = {str(node): 0.0 for node in network["nodes"]}
    if kind == "shortest_path":
        expected[str(network["source"])] = 1.0
        expected[str(network["sink"])] = -1.0
    elif kind in {"min_cost_flow", "assignment", "transportation"}:
        expected.update({str(node): float(value) for node, value in network["supplies"].items()})
    elif kind == "max_flow":
        value = evidence.get("flow_value")
        if value is None:
            issues.append("missing_flow_value")
            value = 0.0
        expected[str(network["source"])] = float(value)
        expected[str(network["sink"])] = -float(value)
    for node, target in expected.items():
        if abs(float(balances.get(node, 0.0)) - target) > 1e-7:
            issues.append(f"node_balance_violation:{node}")
    if kind == "shortest_path" and not issues:
        selected = {str(edge["id"]): edge for edge in edges if float(flow.get(str(edge["id"]), 0.0)) > 0.5}
        current = str(network["source"])
        visited = set()
        while current != str(network["sink"]):
            outgoing = [edge for edge in selected.values() if str(edge["from"]) == current]
            if len(outgoing) != 1 or current in visited:
                issues.append("shortest_path_disconnected_or_cyclic")
                break
            visited.add(current)
            current = str(outgoing[0]["to"])
        if current == str(network["sink"]):
            used = {edge_id for edge_id, edge in selected.items() if edge_id in selected}
            if len(used) != len(visited):
                issues.append("shortest_path_contains_extra_edges")
    return issues, {"expected_balance": expected}


def _model_status(result: Any, has_solution: bool) -> str:
    """Map SciPy MILP terminal states to stable evidence labels."""
    status = int(getattr(result, "status", -1))
    message = str(getattr(result, "message", "")).lower()
    if status == 0 and bool(getattr(result, "success", False)):
        return "optimal"
    if status == 2:
        return "infeasible"
    if status == 3:
        return "unbounded"
    if status == 1 or "time limit" in message:
        return "feasible_not_proven_optimal" if has_solution else "time_limit_no_solution"
    return "numerical_failure"


def verify_native_optimization(
    contract: ProblemContract, records: Iterable[ExperimentRecord], registry_path: Path, output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = list(records)
    issues = []
    checks = []
    for record in records:
        # Diagnostics are recorded for transparency but are not domain claims.
        if record.status != "executed" or record.role in {"ablation", "diagnostic"}:
            continue
        try:
            payload = json.loads(Path(record.evidence_paths["route_evidence"]).read_text(encoding="utf-8"))
            actual = payload["metrics_recomputed"]
            recomputed = _recompute_evidence(payload["contract"], payload["evidence"], payload.get("route", {}).get("route_id"))
            network_issues = []
            if payload["contract"].get("network_model"):
                network_issues, _ = _network_semantic_issues(payload["contract"]["network_model"], payload["evidence"].get("network", {}))
            objective_ok = recomputed["objective"] is not None and math.isclose(float(actual.get("objective")), recomputed["objective"], rel_tol=1e-7, abs_tol=1e-7)
            violation_ok = recomputed["constraint_violation"] is not None and math.isclose(float(actual.get("constraint_violation")), recomputed["constraint_violation"], rel_tol=1e-7, abs_tol=1e-7)
            integer_ok = recomputed["integer_ok"]
            feasible = recomputed["constraint_violation"] is not None and recomputed["constraint_violation"] <= 1e-7 and integer_ok and not network_issues and bool(record.metrics.get("has_feasible_solution", True))
            optimality_proven = record.metrics.get("optimality_proven")
            model_status = record.metrics.get("model_status", "unknown")
            solver_status_ok = record.role == "baseline" or record.metrics.get("solver_status") == 0
            gap = record.metrics.get("mip_gap")
            gap_ok = record.role == "baseline" or (isinstance(gap, (int, float)) and math.isfinite(float(gap)) and float(gap) <= 1e-7)
            bound = record.metrics.get("objective_bound")
            bound_ok = record.role == "baseline" or bound is None or math.isclose(float(bound), recomputed["objective"], rel_tol=1e-7, abs_tol=1e-7)
            optimality_ok = record.role == "baseline" or (model_status == "optimal" and optimality_proven is True and solver_status_ok and gap_ok and bound_ok)
            status_ok = model_status in {"optimal", "feasible_not_proven_optimal"}
            checks.append({"route_id": record.route_id, "model_status": model_status, "objective_recomputed": objective_ok, "constraints_recomputed": violation_ok, "integer_variables_valid": integer_ok, "network_semantic_issues": network_issues, "feasible": feasible, "optimality_proven": optimality_proven, "solver_status_ok": solver_status_ok, "mip_gap_ok": gap_ok, "objective_bound_ok": bound_ok, "optimality_ok": optimality_ok, "status_ok": status_ok})
            if model_status not in {"optimal", "feasible_not_proven_optimal"}:
                issues.append(f"native_optimization_model_status:{record.route_id}:{model_status}")
            if not objective_ok or not violation_ok or not feasible or not optimality_ok:
                issues.append(f"native_optimization_verification_failed:{record.route_id}")
        except Exception as exc:
            issues.append(f"native_optimization_verification_error:{record.route_id}:{exc}")
    run_ids = {record.run_id for record in records if record.run_id}
    stdout_path = output_dir / "native_optimization_verifier.stdout.txt"
    stderr_path = output_dir / "native_optimization_verifier.stderr.txt"
    stdout_path.write_text(json.dumps({"checks": checks}, ensure_ascii=False), encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    result = {
        "kind": "native_optimization", "passed": not issues, "issues": issues, "checks": checks,
        "claim_level": "domain_verified_optimal_result" if not issues and any(item.get("optimality_ok") and item.get("route_id") != "lower_bound_feasible_baseline" for item in checks) else ("domain_verified_feasible_result" if not issues else "no_numerical_claim"),
        "run_id": next(iter(run_ids)) if len(run_ids) == 1 else None,
        "contract_hash": _json_hash(asdict(contract)),
        "script_path": str(Path(__file__).resolve()), "script_hash": _sha256(Path(__file__).resolve()),
        "registry_path": str(registry_path.resolve()), "registry_hash": _sha256(registry_path),
        "stdout_path": str(stdout_path), "stdout_hash": _sha256(stdout_path),
        "stderr_path": str(stderr_path), "stderr_hash": _sha256(stderr_path), "exit_code": 0 if not issues else 1,
    }
    (output_dir / "domain_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _recompute_evidence(contract: Dict[str, Any], evidence: Dict[str, Any], route_id: str | None = None) -> Dict[str, Any]:
    model = contract.get("optimization_model") or evidence.get("network", {}).get("compiled_model")
    if not model:
        return {"objective": None, "constraint_violation": None, "integer_ok": False}
    variables = model["variables"]
    decisions = evidence.get("decision_variables", {})
    if any(item["name"] not in decisions for item in variables):
        return {"objective": None, "constraint_violation": None, "integer_ok": False}
    values = {item["name"]: float(decisions[item["name"]]) for item in variables}
    objective = model["objective"]
    objective_value = sum(float(coef) * values[name] for name, coef in objective["coefficients"].items())
    violations = []
    for item in model["linear_constraints"]:
        lhs = sum(float(coef) * values[name] for name, coef in item["coefficients"].items())
        rhs = float(item["rhs"])
        if item["sense"] == "<=":
            violations.append(max(0.0, lhs - rhs - 1e-7))
        elif item["sense"] == ">=":
            violations.append(max(0.0, rhs - lhs - 1e-7))
        else:
            violations.append(max(0.0, abs(lhs - rhs) - 1e-7))
    integer_ok = all(
        abs(values[item["name"]] - round(values[item["name"]])) <= 1e-7
        for item in variables if item.get("kind", "continuous") in {"integer", "binary"}
    )
    bounds_ok = all(
        float(item.get("lower", 0.0)) - 1e-7 <= values[item["name"]] <= float(item.get("upper", 1.0 if item.get("kind") == "binary" else math.inf)) + 1e-7
        for item in variables
    )
    # The relaxation diagnostic deliberately drops integrality; it still must obey bounds.
    integrality_ok = True if route_id == "linear_relaxation_diagnostic" else integer_ok
    return {"objective": objective_value, "constraint_violation": sum(violations), "integer_ok": integrality_ok and bounds_ok}


def run_native_experiments(contract: ProblemContract, routes: Iterable[RouteSpec], output_dir: Path) -> list[ExperimentRecord]:
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    code_path = Path(__file__).resolve()
    input_path = Path(contract.data_file).resolve()
    records = []
    for route in routes:
        try:
            solved = solve_network_model(asdict(contract), route.__dict__) if contract.network_model else solve_declared_model(asdict(contract), route.__dict__)
            evidence_path = evidence_dir / f"{route.route_id}.json"
            evidence_payload = {
                "run_id": run_id, "problem_id": contract.problem_id, "route_id": route.route_id,
                "data_path": str(input_path), "contract": asdict(contract), "route": route.__dict__,
                "environment": {"scipy": __import__("scipy").__version__},
                "metrics_recomputed": solved["metrics"], "evidence": solved["evidence"],
            }
            evidence_path.write_text(json.dumps(evidence_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            configuration = {"contract": asdict(contract), "route": route.__dict__}
            record = ExperimentRecord(
                contract.problem_id, route.route_id, route.role, "executed", str(code_path), str(input_path),
                solved["metrics"], original_code_hash=_sha256(code_path), final_code_hash=_sha256(code_path),
                run_id=run_id, input_hash=_sha256(input_path),
                config_hash=hashlib.sha256(json.dumps(configuration, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest(),
                artifact_hashes={"route_evidence": _sha256(evidence_path)}, evidence_paths={"route_evidence": str(evidence_path)},
                environment={"scipy": __import__("scipy").__version__},
            )
        except Exception as exc:
            record = ExperimentRecord(contract.problem_id, route.route_id, route.role, "failed", str(code_path), str(input_path), {}, str(exc), original_code_hash=_sha256(code_path), final_code_hash=_sha256(code_path), run_id=run_id, input_hash=_sha256(input_path))
        records.append(record)
        (output_dir / f"{route.route_id}.json").write_text(json.dumps(record.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
    return records


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
