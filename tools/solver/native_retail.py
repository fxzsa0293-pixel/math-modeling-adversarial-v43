from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from .schemas import ExperimentRecord, ProblemContract, RouteSpec


TOLERANCE = 1e-7


def _risk_value(losses: list[float], probabilities: list[float], risk: str, alpha: float) -> float:
    if risk == "expected":
        return float(np.dot(losses, probabilities))
    if risk == "worst_case":
        return float(max(losses))
    return float(min(eta + sum(probability * max(0.0, loss - eta) for loss, probability in zip(losses, probabilities)) / (1.0 - alpha) for eta in losses))


def _fixed_option(item: Dict[str, Any], markup: float) -> int:
    target = float(item["unit_cost"]) * (1.0 + markup)
    return min(range(len(item["price_options"])), key=lambda index: (abs(float(item["price_options"][index]["price"]) - target), float(item["price_options"][index]["price"])))


def _compile(model: Dict[str, Any], route_id: str) -> Dict[str, Any]:
    items = model["items"]
    scenarios = model["scenarios"]
    scenario_names = [str(item["name"]) for item in scenarios]
    probabilities = [float(item["probability"]) for item in scenarios]
    risk = str(model.get("risk_measure", "expected"))
    alpha = float(model.get("alpha", 0.95))
    shortage_penalty = float(model.get("shortage_penalty", 0.0))
    baseline = route_id == "retail_fixed_markup_baseline"
    markup = float(model.get("baseline_markup", 0.2))

    names: list[str] = []
    lower: list[float] = []
    upper: list[float] = []
    integrality: list[int] = []
    index: dict[tuple[Any, ...], int] = {}

    def add(key: tuple[Any, ...], name: str, lb: float, ub: float, integer: bool = False) -> int:
        position = len(names)
        index[key] = position
        names.append(name)
        lower.append(lb)
        upper.append(ub)
        integrality.append(1 if integer else 0)
        return position

    for i, item in enumerate(items):
        add(("select", i), f"select[{i}]", 0.0, 1.0, True)
        add(("replenishment", i), f"replenishment[{i}]", 0.0, float(item["max_replenishment"]))
        fixed = _fixed_option(item, markup) if baseline else None
        for k, _ in enumerate(item["price_options"]):
            enabled = not baseline or k == fixed
            add(("price", i, k), f"price[{i},{k}]", 0.0, 1.0 if enabled else 0.0, True)
    for s, _ in enumerate(scenarios):
        for i, item in enumerate(items):
            for k, option in enumerate(item["price_options"]):
                add(("sales", s, i, k), f"sales[{s},{i},{k}]", 0.0, float(option["demand"][scenario_names[s]]))

    auxiliary: dict[str, int] = {}
    if risk == "worst_case":
        auxiliary["worst_loss"] = add(("worst_loss",), "worst_loss", -math.inf, math.inf)
    elif risk == "cvar":
        auxiliary["eta"] = add(("eta",), "cvar_eta", -math.inf, math.inf)
        for s, _ in enumerate(scenarios):
            auxiliary[f"excess_{s}"] = add(("excess", s), f"cvar_excess[{s}]", 0.0, math.inf)

    scenario_loss = [np.zeros(len(names), dtype=float) for _ in scenarios]
    for s, scenario in enumerate(scenarios):
        scenario_name = str(scenario["name"])
        for i, item in enumerate(items):
            cost = float(item["unit_cost"])
            scenario_loss[s][index[("replenishment", i)]] += cost
            for k, option in enumerate(item["price_options"]):
                demand = float(option["demand"][scenario_name])
                price = float(option["price"])
                scenario_loss[s][index[("price", i, k)]] += shortage_penalty * demand
                scenario_loss[s][index[("sales", s, i, k)]] -= price + shortage_penalty

    c = np.zeros(len(names), dtype=float)
    if risk == "expected":
        for probability, coefficients in zip(probabilities, scenario_loss):
            c += probability * coefficients
    elif risk == "worst_case":
        c[auxiliary["worst_loss"]] = 1.0
    else:
        c[auxiliary["eta"]] = 1.0
        for s, probability in enumerate(probabilities):
            c[auxiliary[f"excess_{s}"]] = probability / (1.0 - alpha)

    rows: list[np.ndarray] = []
    lbs: list[float] = []
    ubs: list[float] = []

    def constrain(coefficients: Dict[int, float], lb: float, ub: float) -> None:
        row = np.zeros(len(names), dtype=float)
        for position, value in coefficients.items():
            row[position] = value
        rows.append(row)
        lbs.append(lb)
        ubs.append(ub)

    for i, item in enumerate(items):
        price_row = {index[("select", i)]: -1.0}
        for k, _ in enumerate(item["price_options"]):
            price_row[index[("price", i, k)]] = 1.0
        constrain(price_row, 0.0, 0.0)
        constrain({index[("replenishment", i)]: 1.0, index[("select", i)]: -float(item["max_replenishment"])}, -math.inf, 0.0)
        constrain({index[("replenishment", i)]: -1.0, index[("select", i)]: float(item["min_replenishment"])}, -math.inf, 0.0)
        usable_fraction = 1.0 - float(item.get("loss_rate", 0.0))
        for s, scenario in enumerate(scenarios):
            availability = {index[("replenishment", i)]: -usable_fraction}
            for k, option in enumerate(item["price_options"]):
                sales_position = index[("sales", s, i, k)]
                availability[sales_position] = 1.0
                demand = float(option["demand"][str(scenario["name"])])
                constrain({sales_position: 1.0, index[("price", i, k)]: -demand}, -math.inf, 0.0)
            constrain(availability, -math.inf, 0.0)

    selection = {index[("select", i)]: 1.0 for i in range(len(items))}
    constrain(selection, float(model.get("min_selected", 0)), float(model.get("max_selected", len(items))))
    for group, bounds in model.get("group_selection_bounds", {}).items():
        group_selection = {index[("select", i)]: 1.0 for i, item in enumerate(items) if str(item.get("group")) == str(group)}
        constrain(group_selection, float(bounds.get("min", 0)), float(bounds.get("max", len(group_selection))))
    scenario_lookup = {str(scenario["name"]): s for s, scenario in enumerate(scenarios)}
    for requirement in model.get("category_service_requirements", []):
        group = str(requirement["group"])
        s = scenario_lookup[str(requirement["scenario"])]
        aggregate_sales = {
            index[("sales", s, i, k)]: 1.0
            for i, item in enumerate(items) if str(item.get("group")) == group
            for k, _ in enumerate(item["price_options"])
        }
        constrain(aggregate_sales, float(requirement["minimum_sales"]), math.inf)
    budget = {index[("replenishment", i)]: float(item["unit_cost"]) for i, item in enumerate(items)}
    constrain(budget, -math.inf, float(model.get("budget", math.inf)))
    if risk == "worst_case":
        for coefficients in scenario_loss:
            row = {position: float(value) for position, value in enumerate(coefficients) if value}
            row[auxiliary["worst_loss"]] = -1.0
            constrain(row, -math.inf, 0.0)
    elif risk == "cvar":
        for s, coefficients in enumerate(scenario_loss):
            row = {position: float(value) for position, value in enumerate(coefficients) if value}
            row[auxiliary["eta"]] = -1.0
            row[auxiliary[f"excess_{s}"]] = -1.0
            constrain(row, -math.inf, 0.0)

    return {
        "names": names,
        "index": index,
        "lower": np.asarray(lower),
        "upper": np.asarray(upper),
        "integrality": np.asarray(integrality),
        "objective": c,
        "rows": np.asarray(rows),
        "lbs": np.asarray(lbs),
        "ubs": np.asarray(ubs),
        "scenario_loss": scenario_loss,
        "probabilities": probabilities,
        "risk": risk,
        "alpha": alpha,
        "fixed_options": {i: _fixed_option(item, markup) for i, item in enumerate(items)} if baseline else {},
    }


def solve_retail_model(contract: Dict[str, Any], route: Dict[str, Any]) -> Dict[str, Any]:
    model = contract["retail_decision_model"]
    compiled = _compile(model, str(route["route_id"]))
    constraints = LinearConstraint(compiled["rows"], compiled["lbs"], compiled["ubs"])
    result = milp(c=compiled["objective"], integrality=compiled["integrality"], bounds=Bounds(compiled["lower"], compiled["upper"]), constraints=constraints, options={"time_limit": 30.0})
    if result.x is None:
        return {"metrics": {"objective": None, "expected_profit": None, "constraint_violation": None, "model_status": "infeasible", "has_feasible_solution": False, "optimality_proven": False}, "evidence": {"decisions": [], "scenario_results": []}}
    values = np.asarray(result.x, dtype=float)
    items = model["items"]
    scenarios = model["scenarios"]
    scenario_names = [str(item["name"]) for item in scenarios]
    decisions = []
    for i, item in enumerate(items):
        selected = bool(values[compiled["index"][("select", i)]] > 0.5)
        chosen = next((k for k, _ in enumerate(item["price_options"]) if values[compiled["index"][("price", i, k)]] > 0.5), None)
        decisions.append({"name": str(item["name"]), "selected": selected, "price_option": chosen, "price": float(item["price_options"][chosen]["price"]) if chosen is not None else None, "replenishment": float(values[compiled["index"][("replenishment", i)]])})
    scenario_results = []
    losses = []
    for s, scenario in enumerate(scenarios):
        item_results = []
        revenue = 0.0
        procurement_cost = 0.0
        shortage_cost = 0.0
        for i, item in enumerate(items):
            replenishment = float(values[compiled["index"][("replenishment", i)]])
            procurement_cost += float(item["unit_cost"]) * replenishment
            chosen = decisions[i]["price_option"]
            demand = float(item["price_options"][chosen]["demand"][scenario_names[s]]) if chosen is not None else 0.0
            sales = sum(float(values[compiled["index"][("sales", s, i, k)]]) for k, _ in enumerate(item["price_options"]))
            price = float(decisions[i]["price"] or 0.0)
            shortage = max(0.0, demand - sales)
            revenue += price * sales
            shortage_cost += float(model.get("shortage_penalty", 0.0)) * shortage
            item_results.append({"name": str(item["name"]), "demand": demand, "sales": sales, "shortage": shortage, "usable_replenishment": (1.0 - float(item.get("loss_rate", 0.0))) * replenishment})
        category_service = []
        sales_by_name = {item["name"]: float(item["sales"]) for item in item_results}
        for requirement in model.get("category_service_requirements", []):
            if str(requirement["scenario"]) != scenario_names[s]:
                continue
            actual_sales = sum(sales_by_name[str(item["name"])] for item in items if str(item.get("group")) == str(requirement["group"]))
            minimum_sales = float(requirement["minimum_sales"])
            category_service.append({"group": str(requirement["group"]), "minimum_sales": minimum_sales, "actual_sales": actual_sales, "satisfied": actual_sales + TOLERANCE >= minimum_sales})
        profit = revenue - procurement_cost - shortage_cost
        losses.append(-profit)
        scenario_results.append({"name": scenario_names[s], "probability": float(scenario["probability"]), "revenue": revenue, "procurement_cost": procurement_cost, "shortage_cost": shortage_cost, "profit": profit, "items": item_results, "category_service": category_service})
    objective = _risk_value(losses, compiled["probabilities"], compiled["risk"], compiled["alpha"])
    stress = _stress_policy(model, decisions)
    row_values = compiled["rows"] @ values
    violation = float(np.maximum(compiled["lbs"] - row_values, 0.0).sum() + np.maximum(row_values - compiled["ubs"], 0.0).sum())
    status = "optimal" if int(result.status) == 0 and bool(result.success) else "feasible_not_proven_optimal"
    return {
        "metrics": {"objective": objective, "expected_profit": -float(np.dot(losses, compiled["probabilities"])), "worst_case_profit": -max(losses), "stress_mean_profit": stress["mean_profit"], "stress_profit_p05": stress["profit_p05"], "stress_mean_shortage_rate": stress["mean_shortage_rate"], "constraint_violation": violation, "selected_item_count": sum(bool(item["selected"]) for item in decisions), "risk_measure": compiled["risk"], "model_status": status, "has_feasible_solution": True, "optimality_proven": status == "optimal", "solver_status": int(result.status), "mip_gap": float(getattr(result, "mip_gap", 0.0) or 0.0)},
        "evidence": {"decisions": decisions, "scenario_results": scenario_results, "stress_test": stress, "risk_measure": compiled["risk"], "alpha": compiled["alpha"], "fixed_price_options": compiled["fixed_options"]},
    }


def _stress_policy(model: Dict[str, Any], decisions: list[Dict[str, Any]]) -> Dict[str, Any]:
    settings = model.get("stress_test", {})
    replications = int(settings.get("replications", 500))
    seed = int(settings.get("seed", 20230701))
    demand_cv = float(settings.get("demand_cv", 0.2))
    sigma = math.sqrt(math.log1p(demand_cv * demand_cv))
    rng = np.random.default_rng(seed)
    correlation = settings.get("demand_correlation")
    correlated = isinstance(correlation, dict) and bool(correlation.get("groups"))
    if correlated:
        groups = [str(value) for value in correlation["groups"]]
        matrix = np.asarray(correlation["matrix"], dtype=float)
        normals = rng.multivariate_normal(np.zeros(len(groups)), matrix, size=replications)
        group_shocks = np.exp(sigma * normals - 0.5 * sigma * sigma)
        group_index = {group: position for position, group in enumerate(groups)}
        shocks = np.column_stack([group_shocks[:, group_index[str(item.get("category", item.get("group")))]] for item in model["items"]])
    else:
        shocks = rng.lognormal(mean=-0.5 * sigma * sigma, sigma=sigma, size=(replications, len(model["items"])))
    elasticity_uniforms = rng.random((replications, len(model["items"])))
    probabilities = {str(item["name"]): float(item["probability"]) for item in model["scenarios"]}
    profits = np.zeros(replications, dtype=float)
    shortages = np.zeros(replications, dtype=float)
    demands = np.zeros(replications, dtype=float)
    decision_map = {str(item["name"]): item for item in decisions}
    for i, item in enumerate(model["items"]):
        decision = decision_map[str(item["name"])]
        replenishment = float(decision["replenishment"])
        profits -= float(item["unit_cost"]) * replenishment
        option_index = decision["price_option"]
        if option_index is None:
            continue
        option = item["price_options"][int(option_index)]
        if all(key in item for key in ("base_demand", "reference_price", "stress_elasticity_range")):
            elasticity_low, elasticity_high = [float(value) for value in item["stress_elasticity_range"]]
            sampled_elasticity = elasticity_low + (elasticity_high - elasticity_low) * elasticity_uniforms[:, i]
            scenario_factor = sum(probabilities[name] * float(option["demand"][name]) for name in probabilities) / max(float(item["base_demand"]) * (float(option["price"]) / float(item["reference_price"])) ** float(item.get("elasticity", -1.0)), 1e-12)
            simulated_demand = float(item["base_demand"]) * scenario_factor * (float(option["price"]) / float(item["reference_price"])) ** sampled_elasticity * shocks[:, i]
        else:
            central_demand = sum(probabilities[name] * float(value) for name, value in option["demand"].items())
            simulated_demand = central_demand * shocks[:, i]
        usable = (1.0 - float(item.get("loss_rate", 0.0))) * replenishment
        sales = np.minimum(simulated_demand, usable)
        shortage = np.maximum(simulated_demand - sales, 0.0)
        profits += float(option["price"]) * sales - float(model.get("shortage_penalty", 0.0)) * shortage
        shortages += shortage
        demands += simulated_demand
    return {"seed": seed, "replications": replications, "demand_cv": demand_cv, "correlated_demand_shocks": correlated, "elasticity_uncertainty": any("stress_elasticity_range" in item for item in model["items"]), "mean_profit": float(np.mean(profits)), "profit_p05": float(np.quantile(profits, 0.05)), "mean_shortage_rate": float(np.mean(shortages / np.maximum(demands, 1e-12))), "profit_samples": profits.tolist()}


def _recompute(model: Dict[str, Any], evidence: Dict[str, Any], route_id: str) -> Dict[str, Any]:
    decisions = evidence.get("decisions", [])
    if len(decisions) != len(model["items"]):
        return {"valid": False}
    decision_map = {item.get("name"): item for item in decisions}
    probabilities = [float(item["probability"]) for item in model["scenarios"]]
    scenario_map = {item.get("name"): item for item in evidence.get("scenario_results", [])}
    violation = 0.0
    total_cost = 0.0
    selected_count = 0
    losses = []
    scenario_checks = []
    for item in model["items"]:
        stored = decision_map.get(item["name"], {})
        selected = bool(stored.get("selected"))
        option_index = stored.get("price_option")
        replenishment = float(stored.get("replenishment", -1.0))
        selected_count += selected
        total_cost += float(item["unit_cost"]) * replenishment
        violation += max(0.0, -replenishment)
        violation += max(0.0, replenishment - float(item["max_replenishment"]) * int(selected))
        violation += max(0.0, float(item["min_replenishment"]) * int(selected) - replenishment)
        if selected != isinstance(option_index, int) or (isinstance(option_index, int) and not 0 <= option_index < len(item["price_options"])):
            violation += 1.0
        if route_id == "retail_fixed_markup_baseline" and selected and option_index != _fixed_option(item, float(model.get("baseline_markup", 0.2))):
            violation += 1.0
    violation += max(0.0, total_cost - float(model.get("budget", math.inf)))
    violation += max(0.0, float(model.get("min_selected", 0)) - selected_count)
    violation += max(0.0, selected_count - float(model.get("max_selected", len(model["items"]))))
    for group, bounds in model.get("group_selection_bounds", {}).items():
        group_count = sum(bool(decision_map.get(item["name"], {}).get("selected")) for item in model["items"] if str(item.get("group")) == str(group))
        violation += max(0.0, float(bounds.get("min", 0)) - group_count)
        violation += max(0.0, group_count - float(bounds.get("max", len(model["items"]))))
    for scenario in model["scenarios"]:
        stored_scenario = scenario_map.get(scenario["name"], {})
        stored_items = {item.get("name"): item for item in stored_scenario.get("items", [])}
        revenue = 0.0
        cost = total_cost
        shortage_cost = 0.0
        for item in model["items"]:
            decision = decision_map.get(item["name"], {})
            option_index = decision.get("price_option")
            replenishment = float(decision.get("replenishment", 0.0))
            item_evidence = stored_items.get(item["name"], {})
            sales = float(item_evidence.get("sales", -1.0))
            demand = float(item["price_options"][option_index]["demand"][scenario["name"]]) if isinstance(option_index, int) and 0 <= option_index < len(item["price_options"]) else 0.0
            price = float(item["price_options"][option_index]["price"]) if isinstance(option_index, int) and 0 <= option_index < len(item["price_options"]) else 0.0
            usable = (1.0 - float(item.get("loss_rate", 0.0))) * replenishment
            violation += max(0.0, -sales) + max(0.0, sales - demand) + max(0.0, sales - usable)
            shortage = max(0.0, demand - sales)
            revenue += price * sales
            shortage_cost += float(model.get("shortage_penalty", 0.0)) * shortage
            if not all(math.isclose(float(item_evidence.get(field, math.inf)), expected, rel_tol=1e-7, abs_tol=1e-7) for field, expected in (("demand", demand), ("shortage", shortage), ("usable_replenishment", usable))):
                violation += 1.0
        for requirement in model.get("category_service_requirements", []):
            if str(requirement["scenario"]) != str(scenario["name"]):
                continue
            group_sales = sum(
                float(stored_items.get(item["name"], {}).get("sales", 0.0))
                for item in model["items"] if str(item.get("group")) == str(requirement["group"])
            )
            violation += max(0.0, float(requirement["minimum_sales"]) - group_sales)
            stored_service = {str(item.get("group")): item for item in stored_scenario.get("category_service", [])}.get(str(requirement["group"]), {})
            if not math.isclose(float(stored_service.get("minimum_sales", math.inf)), float(requirement["minimum_sales"]), rel_tol=1e-7, abs_tol=1e-7) or not math.isclose(float(stored_service.get("actual_sales", math.inf)), group_sales, rel_tol=1e-7, abs_tol=1e-7) or stored_service.get("satisfied") is not (group_sales + TOLERANCE >= float(requirement["minimum_sales"])):
                violation += 1.0
        profit = revenue - cost - shortage_cost
        losses.append(-profit)
        scenario_checks.append(all(math.isclose(float(stored_scenario.get(field, math.inf)), expected, rel_tol=1e-7, abs_tol=1e-7) for field, expected in (("revenue", revenue), ("procurement_cost", cost), ("shortage_cost", shortage_cost), ("profit", profit))))
    risk = str(model.get("risk_measure", "expected"))
    alpha = float(model.get("alpha", 0.95))
    return {"valid": violation <= TOLERANCE and all(scenario_checks), "constraint_violation": violation, "objective": _risk_value(losses, probabilities, risk, alpha), "expected_profit": -float(np.dot(losses, probabilities)), "selected_item_count": selected_count}


def run_native_retail_experiments(contract: ProblemContract, routes: Iterable[RouteSpec], output_dir: Path) -> list[ExperimentRecord]:
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    code_path = Path(__file__).resolve()
    input_path = Path(contract.data_file).resolve()
    records = []
    for route in routes:
        try:
            solved = solve_retail_model(asdict(contract), route.__dict__)
            evidence_path = evidence_dir / f"{route.route_id}.json"
            payload = {"run_id": run_id, "problem_id": contract.problem_id, "route_id": route.route_id, "data_path": str(input_path), "contract": asdict(contract), "route": route.__dict__, "metrics_recomputed": solved["metrics"], "evidence": solved["evidence"]}
            evidence_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            policy_path = evidence_dir / f"{route.route_id}_policy.csv"
            scenario_path = evidence_dir / f"{route.route_id}_scenario_results.csv"
            service_path = evidence_dir / f"{route.route_id}_category_service.csv"
            _write_policy_tables(solved["evidence"], policy_path, scenario_path, service_path)
            evidence_paths = {"route_evidence": str(evidence_path), "retail_policy": str(policy_path), "retail_scenarios": str(scenario_path), "retail_category_service": str(service_path)}
            record = ExperimentRecord(contract.problem_id, route.route_id, route.role, "executed", str(code_path), str(input_path), solved["metrics"], original_code_hash=_sha256(code_path), final_code_hash=_sha256(code_path), run_id=run_id, input_hash=_sha256(input_path), config_hash=_json_hash({"contract": asdict(contract), "route": route.__dict__}), artifact_hashes={name: _sha256(Path(path)) for name, path in evidence_paths.items()}, evidence_paths=evidence_paths, environment={"scipy": __import__("scipy").__version__})
        except Exception as exc:
            record = ExperimentRecord(contract.problem_id, route.route_id, route.role, "failed", str(code_path), str(input_path), {}, str(exc), original_code_hash=_sha256(code_path), final_code_hash=_sha256(code_path), run_id=run_id, input_hash=_sha256(input_path))
        records.append(record)
        (output_dir / f"{route.route_id}.json").write_text(json.dumps(record.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
    return records


def _write_policy_tables(evidence: Dict[str, Any], policy_path: Path, scenario_path: Path, service_path: Path) -> None:
    policy_rows = []
    for item in evidence["decisions"]:
        raw_name = str(item["name"])
        decision_date, category = raw_name.split("|", 1) if "|" in raw_name else ("", raw_name)
        policy_rows.append({"decision_date": decision_date, "category": category, "selected": bool(item["selected"]), "price": item["price"], "replenishment": item["replenishment"], "price_option": item["price_option"]})
    scenario_rows = []
    for scenario in evidence["scenario_results"]:
        for item in scenario["items"]:
            raw_name = str(item["name"])
            decision_date, category = raw_name.split("|", 1) if "|" in raw_name else ("", raw_name)
            scenario_rows.append({"scenario": scenario["name"], "probability": scenario["probability"], "scenario_profit": scenario["profit"], "decision_date": decision_date, "category": category, "demand": item["demand"], "sales": item["sales"], "shortage": item["shortage"], "usable_replenishment": item["usable_replenishment"]})
    pd.DataFrame(policy_rows).to_csv(policy_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(scenario_rows).to_csv(scenario_path, index=False, encoding="utf-8-sig")
    service_rows = [dict(scenario=scenario["name"], **service) for scenario in evidence["scenario_results"] for service in scenario.get("category_service", [])]
    pd.DataFrame(service_rows, columns=["scenario", "group", "minimum_sales", "actual_sales", "satisfied"]).to_csv(service_path, index=False, encoding="utf-8-sig")


def verify_native_retail(contract: ProblemContract, records: Iterable[ExperimentRecord], registry_path: Path, output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = list(records)
    checks = []
    issues = []
    stress_samples: Dict[str, np.ndarray] = {}
    registry_path = Path(registry_path)
    if not registry_path.is_file():
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text(json.dumps([record.__dict__ for record in records], ensure_ascii=False, indent=2), encoding="utf-8")
    for record in records:
        if record.status != "executed" or record.role in {"ablation", "diagnostic"}:
            continue
        try:
            payload = json.loads(Path(record.evidence_paths["route_evidence"]).read_text(encoding="utf-8"))
            recomputed = _recompute(contract.retail_decision_model, payload["evidence"], record.route_id)
            rerun = solve_retail_model(asdict(contract), {"route_id": record.route_id})
            stress = _stress_policy(contract.retail_decision_model, payload["evidence"]["decisions"])
            stored_stress = payload["evidence"].get("stress_test", {})
            stress_samples[record.route_id] = np.asarray(stress["profit_samples"], dtype=float)
            item = {
                "route_id": record.route_id,
                "model_matches": payload.get("contract", {}).get("retail_decision_model") == contract.retail_decision_model,
                "route_matches": payload.get("route_id") == record.route_id and payload.get("route", {}).get("route_id") == record.route_id,
                "business_quantities_recomputed": recomputed.get("valid") is True,
                "objective_recomputed": math.isclose(float(record.metrics["objective"]), float(recomputed.get("objective")), rel_tol=1e-7, abs_tol=1e-7),
                "expected_profit_recomputed": math.isclose(float(record.metrics["expected_profit"]), float(recomputed.get("expected_profit")), rel_tol=1e-7, abs_tol=1e-7),
                "stress_test_recomputed": all(math.isclose(float(stored_stress.get(field)), float(stress[field]), rel_tol=1e-10, abs_tol=1e-10) for field in ("mean_profit", "profit_p05", "mean_shortage_rate")) and stored_stress.get("seed") == stress["seed"] and stored_stress.get("replications") == stress["replications"] and stored_stress.get("correlated_demand_shocks") == stress["correlated_demand_shocks"],
                "optimal_objective_recomputed": math.isclose(float(record.metrics["objective"]), float(rerun["metrics"]["objective"]), rel_tol=1e-7, abs_tol=1e-7),
                "optimality_proven": record.metrics.get("optimality_proven") is True,
            }
            checks.append(item)
            if not all(value for key, value in item.items() if key != "route_id"):
                issues.append(f"native_retail_verification_failed:{record.route_id}")
        except Exception as exc:
            issues.append(f"native_retail_verification_error:{record.route_id}:{exc}")
    policy_comparison = None
    baseline_samples = stress_samples.get("retail_fixed_markup_baseline")
    candidate_samples = stress_samples.get("retail_joint_price_replenishment")
    if baseline_samples is not None and candidate_samples is not None and len(baseline_samples) == len(candidate_samples):
        difference = candidate_samples - baseline_samples
        standard_error = float(np.std(difference, ddof=1) / math.sqrt(len(difference))) if len(difference) > 1 else 0.0
        mean_difference = float(np.mean(difference))
        ci_lower = mean_difference - 1.96 * standard_error
        policy_comparison = {"common_random_numbers": True, "correlated_demand_shocks": bool(contract.retail_decision_model.get("stress_test", {}).get("demand_correlation")), "elasticity_uncertainty": any("stress_elasticity_range" in item for item in contract.retail_decision_model["items"]), "replications": len(difference), "mean_profit_difference": mean_difference, "ci95_lower": ci_lower, "ci95_upper": mean_difference + 1.96 * standard_error, "candidate_win_rate": float(np.mean(difference > 0.0)), "candidate_robustly_better": ci_lower > 0.0}
    stdout = output_dir / "native_retail_verifier.stdout.txt"
    stderr = output_dir / "native_retail_verifier.stderr.txt"
    stdout.write_text(json.dumps({"checks": checks, "policy_comparison": policy_comparison}, ensure_ascii=False), encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    run_ids = {record.run_id for record in records if record.run_id}
    result = {"kind": "native_retail_decision", "passed": not issues and bool(checks), "issues": issues, "checks": checks, "policy_comparison": policy_comparison, "claim_level": "domain_verified_retail_policy" if not issues and checks else "no_numerical_claim", "run_id": next(iter(run_ids)) if len(run_ids) == 1 else None, "contract_hash": _json_hash(asdict(contract)), "script_path": str(Path(__file__).resolve()), "script_hash": _sha256(Path(__file__).resolve()), "registry_path": str(registry_path.resolve()), "registry_hash": _sha256(registry_path), "stdout_path": str(stdout), "stdout_hash": _sha256(stdout), "stderr_path": str(stderr), "stderr_hash": _sha256(stderr), "exit_code": 0 if not issues else 1}
    (output_dir / "domain_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
