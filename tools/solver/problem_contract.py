from __future__ import annotations

import json
import math
import re
import numpy as np
from pathlib import Path
from typing import Iterable, List

from .schemas import DataAsset, ProblemContract

CONTRACT_VERSION = "v44"
TASK_TYPES = {"forecasting", "regression", "evaluation_ranking", "constrained_optimization", "mechanism", "simulation", "spatial"}
VALIDATION_MODES = {
    "forecasting": {"fixed_origin_holdout", "rolling_origin"},
    "regression": {"random_holdout", "group_holdout"},
    "evaluation_ranking": {"stability_audit"},
}
DEFAULT_VALIDATION_MODE = {
    "forecasting": "fixed_origin_holdout",
    "regression": "random_holdout",
    "evaluation_ranking": "stability_audit",
}
TARGET_PATTERNS = [
    re.compile(r"(?:predict|forecast|estimate)\s+(?:future\s+)?([A-Za-z_][A-Za-z0-9_]*)", re.I),
    re.compile(r"(?:预测|估计|拟合)(?:未来)?[的 ]*([\u4e00-\u9fffA-Za-z_][\u4e00-\u9fffA-Za-z0-9_]*)"),
]
LOWER_HINTS = ("cost", "price", "loss", "risk", "time", "成本", "价格", "损耗", "风险", "时间")
ID_HINTS = ("id", "编号", "序号", "code", "index")


def load_or_infer_contract(
    problem_id: str,
    assets: List[DataAsset],
    archetypes: Iterable[str],
    target_candidates: List[str],
    brief_text: str,
    contract_path: Path | None = None,
) -> ProblemContract:
    if contract_path:
        payload = json.loads(Path(contract_path).read_text(encoding="utf-8"))
        contract = ProblemContract(**payload)
        contract.source = "provided"
        return validate_contract(contract, assets)

    task_type = _task_type(archetypes)
    data_file = assets[0].path if assets else ""
    columns = assets[0].columns if assets else []
    target = _target_from_brief(brief_text, columns)
    if not target and len(target_candidates) == 1:
        target = target_candidates[0]
    if not target and "target" in columns:
        target = "target"

    time_column = next((col for col in columns if any(term in str(col).lower() for term in ("date", "time", "week", "month", "year", "日期", "时间", "周", "月", "年"))), "")
    directions = {}
    if task_type == "evaluation_ranking":
        for column in columns:
            lowered = str(column).lower()
            if any(hint in lowered for hint in ID_HINTS):
                continue
            directions[str(column)] = "lower_better" if any(hint in lowered for hint in LOWER_HINTS) else "higher_better"

    unresolved = []
    if not data_file:
        unresolved.append("data_file")
    if len(assets) > 1:
        unresolved.append("data_file_selection")
    if assets and len(assets[0].sheet_names) > 1:
        unresolved.append("sheet_name")
    if task_type in {"forecasting", "regression"} and not target:
        unresolved.append("target_column")
    if task_type == "forecasting" and not time_column:
        unresolved.append("time_column")
    if task_type == "evaluation_ranking" and not directions:
        unresolved.append("indicator_directions")
    elif task_type == "evaluation_ranking":
        unresolved.append("indicator_directions_confirmation")
    contract = ProblemContract(
        version=CONTRACT_VERSION,
        problem_id=problem_id,
        task_type=task_type,
        status="needs_input" if unresolved else "ready",
        data_file=data_file,
        sheet_name=assets[0].sheet_names[0] if assets and len(assets[0].sheet_names) == 1 else None,
        target_column=target,
        time_column=time_column,
        indicator_directions=directions,
        forecast_horizon=max(2, int((assets[0].row_count or 8) * 0.3)) if task_type == "forecasting" else 1,
        validation_mode="fixed_origin_holdout" if task_type == "forecasting" else ("random_holdout" if task_type == "regression" else "stability_audit"),
        unresolved_fields=unresolved,
        notes=["Inferred fields are candidates. Provide --contract when domain interpretation or multiple tables/sheets matter."],
    )
    return validate_contract(contract, assets)


def validate_contract(contract: ProblemContract, assets: List[DataAsset]) -> ProblemContract:
    errors = []
    missing = []
    multi_source = bool(contract.data_sources)
    if multi_source:
        errors.extend(_validate_data_sources(contract, assets))
    for raw_root in contract.allowed_code_roots:
        if not isinstance(raw_root, str) or not Path(raw_root).is_dir():
            errors.append(f"invalid_allowed_code_root:{raw_root}")
    if contract.version != CONTRACT_VERSION:
        errors.append(f"unsupported_contract_version:{contract.version}")
    if contract.task_type not in TASK_TYPES:
        errors.append(f"unsupported_task_type:{contract.task_type}")
    if contract.task_type in {"forecasting", "regression"} and not contract.target_column:
        missing.append("target_column")
    if contract.task_type == "forecasting" and not contract.time_column:
        missing.append("time_column")
    if contract.task_type == "evaluation_ranking" and not contract.indicator_directions:
        missing.append("indicator_directions")
    if contract.task_type == "constrained_optimization" and not contract.constraints and not contract.optimization_model and not contract.robust_optimization_model and not contract.retail_decision_model and not contract.network_model and not contract.multiobjective_model:
        missing.append("constraints")
    if contract.optimization_model:
        errors.extend(_validate_optimization_model(contract))
    native_robust = contract.task_type == "constrained_optimization" and bool(contract.robust_optimization_model)
    if contract.robust_optimization_model:
        errors.extend(_validate_robust_optimization_model(contract))
    native_retail = contract.task_type == "constrained_optimization" and bool(contract.retail_decision_model)
    if contract.retail_decision_model:
        errors.extend(_validate_retail_decision_model(contract))
    native_multiobjective = contract.task_type == "constrained_optimization" and bool(contract.multiobjective_model)
    if contract.multiobjective_model:
        errors.extend(_validate_multiobjective_model(contract))
    native_optimization = contract.task_type == "constrained_optimization" and bool(contract.optimization_model)
    native_network = contract.task_type == "constrained_optimization" and bool(contract.network_model)
    if contract.network_model:
        errors.extend(_validate_network_model(contract))
    native_mechanism = contract.task_type == "mechanism" and bool(contract.mechanism_model)
    if contract.mechanism_model:
        errors.extend(_validate_mechanism_model(contract))
        model = contract.mechanism_model
        matching = [asset for asset in assets if Path(asset.path).resolve() == Path(contract.data_file).resolve()] if contract.data_file else []
        columns = matching[0].sheet_columns.get(contract.sheet_name, matching[0].columns) if matching and contract.sheet_name else (matching[0].columns if matching else [])
        for column in ([str(model.get("time_column") or contract.time_column)] + [str(value) for value in model.get("state_columns", [])]):
            if column and column not in columns:
                errors.append(f"mechanism_column_not_found:{column}")
    native_simulation = contract.task_type == "simulation" and bool(contract.simulation_model)
    if contract.simulation_model:
        errors.extend(_validate_simulation_model(contract))
    native_spatial = contract.task_type == "spatial" and bool(contract.spatial_model)
    if contract.spatial_model:
        errors.extend(_validate_spatial_model(contract))
        model = contract.spatial_model
        matching = [asset for asset in assets if Path(asset.path).resolve() == Path(contract.data_file).resolve()] if contract.data_file else []
        columns = matching[0].sheet_columns.get(contract.sheet_name, matching[0].columns) if matching and contract.sheet_name else (matching[0].columns if matching else [])
        for column in [model.get("x_column"), model.get("y_column"), model.get("target_column")]:
            if column and column not in columns:
                errors.append(f"spatial_column_not_found:{column}")
    if contract.task_type in {"constrained_optimization", "mechanism", "simulation", "spatial"} and not native_optimization and not native_robust and not native_retail and not native_network and not native_mechanism and not native_simulation and not native_spatial and not native_multiobjective:
        if not contract.specialist_routes:
            missing.append("specialist_routes")
        if not contract.domain_verifier:
            missing.append("domain_verifier")
    route_ids = set()
    baseline_count = 0
    for index, route in enumerate(contract.specialist_routes):
        if not isinstance(route, dict):
            errors.append(f"invalid_specialist_route:{index}")
            continue
        required = {"route_id", "role", "family", "adapter_path", "expected_metrics"}
        if required - route.keys():
            errors.append(f"incomplete_specialist_route:{index}")
            continue
        route_id = str(route["route_id"])
        if route_id in route_ids:
            errors.append(f"duplicate_specialist_route:{route_id}")
        route_ids.add(route_id)
        baseline_count += route["role"] in {"baseline", "strong_baseline"}
        if route["role"] not in {"baseline", "strong_baseline", "candidate", "composite", "ablation", "negative_control"}:
            errors.append(f"invalid_specialist_role:{route_id}")
        adapter_path = Path(route["adapter_path"]).resolve()
        if not adapter_path.is_file():
            errors.append(f"specialist_adapter_not_found:{route_id}")
        elif not _within_audited_workspace(adapter_path, contract, assets):
            errors.append(f"specialist_adapter_outside_audited_workspace:{route_id}")
    if contract.specialist_routes and baseline_count == 0:
        errors.append("specialist_routes_require_baseline")
    if contract.domain_verifier:
        verifier_path = Path(contract.domain_verifier).resolve()
        if not verifier_path.is_file():
            errors.append("domain_verifier_not_found")
        elif not _within_audited_workspace(verifier_path, contract, assets):
            errors.append("domain_verifier_outside_audited_workspace")
    if not contract.validation_mode and contract.task_type in DEFAULT_VALIDATION_MODE:
        contract.validation_mode = DEFAULT_VALIDATION_MODE[contract.task_type]
    allowed_modes = VALIDATION_MODES.get(contract.task_type)
    if allowed_modes and contract.validation_mode not in allowed_modes:
        errors.append(f"invalid_validation_mode:{contract.validation_mode}")
    if contract.task_type == "regression" and contract.group_columns and contract.validation_mode != "group_holdout":
        errors.append("group_columns_require_group_holdout")
    if contract.task_type == "forecasting" and contract.group_columns:
        errors.append("grouped_forecasting_requires_specialist_adapter")
    if contract.target_column and contract.target_column in contract.known_future_columns:
        errors.append("target_cannot_be_known_future")
    if contract.time_column and contract.time_column in contract.known_future_columns:
        errors.append("time_column_cannot_be_known_future")
    invalid_directions = [name for name, direction in contract.indicator_directions.items() if direction not in {"higher_better", "lower_better"}]
    if invalid_directions:
        errors.append(f"invalid_indicator_directions:{invalid_directions}")
    invalid_metric_directions = [name for name, direction in contract.metric_directions.items() if direction not in {"higher_better", "lower_better"}]
    if invalid_metric_directions:
        errors.append(f"invalid_metric_directions:{invalid_metric_directions}")
    specialist_metrics = {str(metric) for route in contract.specialist_routes if isinstance(route, dict) for metric in route.get("expected_metrics", [])}
    if contract.specialist_routes:
        for metric in specialist_metrics:
            if metric not in contract.metric_directions and metric not in {"constraint_violation", "runtime_seconds"}:
                missing.append(f"metric_direction:{metric}")
    matching = [asset for asset in assets if Path(asset.path).resolve() == Path(contract.data_file).resolve()] if contract.data_file else []
    source_paths = {str(Path(item["path"]).resolve()) for item in contract.data_sources if isinstance(item, dict) and item.get("path")}
    if multi_source and (not matching or str(Path(contract.data_file).resolve()) in source_paths):
        projected_columns = _projected_join_columns(contract, assets)
        if projected_columns is not None:
            matching = [DataAsset("<assembled>", ".csv", 0, columns=projected_columns, readable=True)]
    if not matching:
        errors.append("data_file_not_audited")
    else:
        columns = matching[0].sheet_columns.get(contract.sheet_name, matching[0].columns) if contract.sheet_name else matching[0].columns
        for label, column in (("target_column", contract.target_column), ("time_column", contract.time_column)):
            if column and column not in columns:
                errors.append(f"{label}_not_found:{column}")
        for column in contract.feature_columns + contract.known_future_columns + contract.group_columns + list(contract.indicator_directions):
            if column not in columns:
                errors.append(f"contract_column_not_found:{column}")
        if any(column not in contract.feature_columns for column in contract.known_future_columns):
            errors.append("known_future_columns_must_be_feature_columns")
        if contract.sheet_name and matching[0].sheet_names and contract.sheet_name not in matching[0].sheet_names:
            errors.append(f"sheet_not_found:{contract.sheet_name}")
        if len(matching[0].sheet_names) > 1 and not contract.sheet_name:
            missing.append("sheet_name")
    if contract.forecast_horizon < 1:
        errors.append("forecast_horizon_must_be_positive")
    if not isinstance(contract.seasonal_period, int) or contract.seasonal_period < 1:
        errors.append("seasonal_period_must_be_positive_integer")
    if errors:
        contract.status = "invalid"
        contract.unresolved_fields = list(dict.fromkeys(contract.unresolved_fields + errors))
    elif missing or contract.unresolved_fields:
        contract.status = "needs_input"
        contract.unresolved_fields = list(dict.fromkeys(contract.unresolved_fields + missing))
    else:
        contract.status = "ready"
    return contract


def _validate_data_sources(contract: ProblemContract, assets: List[DataAsset]) -> List[str]:
    errors: List[str] = []
    if not isinstance(contract.data_sources, list) or not contract.data_sources:
        return ["data_sources_invalid"]
    aliases = []
    asset_map = {str(Path(asset.path).resolve()): asset for asset in assets}
    for index, source in enumerate(contract.data_sources):
        if not isinstance(source, dict) or not isinstance(source.get("alias"), str) or not source.get("alias") or not isinstance(source.get("path"), str):
            errors.append(f"data_source_schema_invalid:{index}")
            continue
        alias = str(source["alias"]); aliases.append(alias)
        path = str(Path(source["path"]).resolve())
        asset = asset_map.get(path)
        if asset is None or not asset.readable:
            errors.append(f"data_source_not_audited:{alias}")
        elif source.get("sheet_name") is not None and asset.sheet_names and source["sheet_name"] not in asset.sheet_names:
            errors.append(f"data_source_sheet_not_found:{alias}:{source['sheet_name']}")
        if asset is not None and source.get("columns") is not None:
            available = asset.sheet_columns.get(source.get("sheet_name"), asset.columns)
            columns = source.get("columns")
            if not isinstance(columns, list) or not columns or not all(isinstance(value, str) and value in available for value in columns):
                errors.append(f"data_source_columns_invalid:{alias}")
    if len(aliases) != len(set(aliases)):
        errors.append("data_source_aliases_duplicate")
    alias_set = set(aliases)
    if not contract.base_table or contract.base_table not in alias_set:
        errors.append("base_table_invalid")
    if not isinstance(contract.data_joins, list):
        return errors + ["data_joins_invalid"]
    included = {contract.base_table}; joined_right = set()
    for index, join in enumerate(contract.data_joins):
        if not isinstance(join, dict):
            errors.append(f"data_join_schema_invalid:{index}"); continue
        left = join.get("left"); right = join.get("right")
        if left not in included or right not in alias_set or right in included or right in joined_right:
            errors.append(f"data_join_order_invalid:{index}")
        if join.get("how") not in {"left", "inner", "right", "outer"}:
            errors.append(f"data_join_how_invalid:{index}")
        if join.get("validate") not in {"one_to_one", "one_to_many", "many_to_one", "many_to_many"}:
            errors.append(f"data_join_cardinality_invalid:{index}")
        left_on = join.get("left_on") or join.get("on"); right_on = join.get("right_on") or join.get("on")
        left_keys = [left_on] if isinstance(left_on, str) else left_on
        right_keys = [right_on] if isinstance(right_on, str) else right_on
        if not isinstance(left_keys, list) or not left_keys or not all(isinstance(value, str) and value for value in left_keys) or not isinstance(right_keys, list) or len(right_keys) != len(left_keys) or not all(isinstance(value, str) and value for value in right_keys):
            errors.append(f"data_join_keys_invalid:{index}")
        right_columns = join.get("right_columns")
        if right_columns is not None and (not isinstance(right_columns, list) or not all(isinstance(value, str) for value in right_columns)):
            errors.append(f"data_join_right_columns_invalid:{index}")
        for rate_name in ("max_unmatched_left_rate", "max_unmatched_right_rate"):
            rate = join.get(rate_name, 1.0)
            if not isinstance(rate, (int, float)) or not math.isfinite(float(rate)) or not 0.0 <= float(rate) <= 1.0:
                errors.append(f"data_join_rate_invalid:{index}:{rate_name}")
        included.add(str(right)); joined_right.add(str(right))
    if alias_set - included:
        errors.append(f"data_sources_unjoined:{sorted(alias_set - included)}")
    projected = _projected_join_columns(contract, assets, apply_aggregation=False)
    if projected is None:
        errors.append("data_join_columns_unresolved")
    elif contract.data_aggregation:
        aggregation = contract.data_aggregation
        group_by = aggregation.get("group_by") if isinstance(aggregation, dict) else None
        aggregations = aggregation.get("aggregations") if isinstance(aggregation, dict) else None
        if not isinstance(group_by, list) or not group_by or not all(isinstance(value, str) and value in projected for value in group_by):
            errors.append("data_aggregation_group_by_invalid")
        if not isinstance(aggregations, dict) or not aggregations:
            errors.append("data_aggregations_invalid")
        else:
            for output, spec in aggregations.items():
                if not isinstance(output, str) or not output or not isinstance(spec, dict) or spec.get("column") not in projected or spec.get("function") not in {"sum", "mean", "min", "max", "median", "count", "nunique", "std"}:
                    errors.append(f"data_aggregation_invalid:{output}")
        output_columns = list(group_by or []) + list(aggregations or {})
        sort_by = aggregation.get("sort_by", []) if isinstance(aggregation, dict) else []
        if not isinstance(sort_by, list) or any(value not in output_columns for value in sort_by):
            errors.append("data_aggregation_sort_by_invalid")
    return errors


def _projected_join_columns(contract: ProblemContract, assets: List[DataAsset], apply_aggregation: bool = True) -> List[str] | None:
    asset_map = {str(Path(asset.path).resolve()): asset for asset in assets}
    source_columns: dict[str, List[str]] = {}
    for source in contract.data_sources:
        if not isinstance(source, dict) or not source.get("path") or not source.get("alias"):
            return None
        asset = asset_map.get(str(Path(source["path"]).resolve()))
        if not asset:
            return None
        available = list(asset.sheet_columns.get(source.get("sheet_name"), asset.columns))
        source_columns[str(source["alias"])] = list(source.get("columns") or available)
    if contract.base_table not in source_columns:
        return None
    columns = list(source_columns[contract.base_table])
    for index, join in enumerate(contract.data_joins):
        if not isinstance(join, dict) or str(join.get("right")) not in source_columns:
            return None
        left_on = join.get("left_on") or join.get("on"); right_on = join.get("right_on") or join.get("on")
        left_keys = [left_on] if isinstance(left_on, str) else left_on; right_keys = [right_on] if isinstance(right_on, str) else right_on
        if not isinstance(left_keys, list) or not isinstance(right_keys, list) or len(left_keys) != len(right_keys):
            return None
        if any(key not in columns for key in left_keys) or any(key not in source_columns[str(join["right"])] for key in right_keys):
            return None
        prefix = str(join.get("right_prefix", f"{join['right']}__"))
        right_output = join.get("right_columns") if join.get("right_columns") is not None else source_columns[str(join["right"])]
        if any(value not in source_columns[str(join["right"])] for value in right_output):
            return None
        columns.extend(f"{prefix}{column}" for column in right_output if column not in right_keys)
    if apply_aggregation and contract.data_aggregation and isinstance(contract.data_aggregation.get("group_by"), list) and isinstance(contract.data_aggregation.get("aggregations"), dict):
        return [str(value) for value in contract.data_aggregation["group_by"]] + [str(value) for value in contract.data_aggregation["aggregations"]]
    return columns


def _validate_optimization_model(contract: ProblemContract) -> List[str]:
    model = contract.optimization_model
    if not isinstance(model, dict):
        return ["optimization_model_not_object"]
    variables = model.get("variables")
    objective = model.get("objective")
    linear_constraints = model.get("linear_constraints")
    errors = []
    if not isinstance(variables, list) or not variables:
        return ["optimization_variables_missing"]
    names = []
    for variable in variables:
        if not isinstance(variable, dict) or not isinstance(variable.get("name"), str):
            errors.append("optimization_variable_schema_invalid")
            continue
        names.append(variable["name"])
        if variable.get("kind", "continuous") not in {"continuous", "integer", "binary"}:
            errors.append(f"optimization_variable_kind_invalid:{variable['name']}")
        for bound_name in ("lower", "upper"):
            if bound_name in variable and (not isinstance(variable[bound_name], (int, float)) or not math.isfinite(float(variable[bound_name]))):
                errors.append(f"optimization_{bound_name}_invalid:{variable['name']}")
        lower = variable.get("lower", 0.0)
        upper = variable.get("upper", 1.0 if variable.get("kind") == "binary" else math.inf)
        if isinstance(lower, (int, float)) and isinstance(upper, (int, float)) and lower > upper:
            errors.append(f"optimization_bounds_reversed:{variable['name']}")
        if variable.get("kind") == "binary" and (lower < 0 or upper > 1):
            errors.append(f"optimization_binary_bounds_invalid:{variable['name']}")
    if len(names) != len(set(names)):
        errors.append("optimization_variable_names_duplicate")
    if not isinstance(objective, dict) or objective.get("sense") not in {"min", "max"} or not isinstance(objective.get("coefficients"), dict):
        errors.append("optimization_objective_invalid")
    elif any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in objective["coefficients"].values()):
        errors.append("optimization_objective_coefficients_invalid")
    if not isinstance(linear_constraints, list):
        errors.append("optimization_linear_constraints_missing")
    else:
        for item in linear_constraints:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str) or item.get("sense") not in {"<=", ">=", "=="} or not isinstance(item.get("coefficients"), dict) or not isinstance(item.get("rhs"), (int, float)):
                errors.append("optimization_linear_constraint_invalid")
            elif (not math.isfinite(float(item["rhs"])) or any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in item["coefficients"].values())):
                errors.append(f"optimization_linear_constraint_values_invalid:{item['name']}")
    declared_names = set(names)
    if isinstance(objective, dict) and isinstance(objective.get("coefficients"), dict):
        if not set(objective["coefficients"]).issubset(declared_names):
            errors.append("optimization_objective_unknown_variable")
    if isinstance(linear_constraints, list):
        for item in linear_constraints:
            if isinstance(item, dict) and isinstance(item.get("coefficients"), dict) and not set(item["coefficients"]).issubset(declared_names):
                errors.append(f"optimization_constraint_unknown_variable:{item.get('name', '')}")
    declared = {str(item.get("name")) for item in contract.constraints if isinstance(item, dict) and item.get("name")}
    model_names = {str(item.get("name")) for item in linear_constraints if isinstance(item, dict)} if isinstance(linear_constraints, list) else set()
    if declared and declared != model_names:
        errors.append("optimization_constraint_names_must_match_contract")
    return errors


def _validate_robust_optimization_model(contract: ProblemContract) -> List[str]:
    model = contract.robust_optimization_model
    if not isinstance(model, dict):
        return ["robust_optimization_model_not_object"]
    errors: List[str] = []
    if model.get("objective_sense", "min") != "min":
        errors.append("robust_native_supports_minimization_only")
    variables = model.get("variables")
    scenarios = model.get("scenarios")
    if not isinstance(variables, list) or not variables:
        return ["robust_optimization_variables_missing"]
    names = []
    for variable in variables:
        if not isinstance(variable, dict) or not isinstance(variable.get("name"), str) or not variable.get("name") or variable.get("kind", "continuous") not in {"continuous", "integer", "binary"}:
            errors.append("robust_optimization_variable_invalid"); continue
        names.append(variable["name"]); kind = variable.get("kind", "continuous")
        lower = variable.get("lower", 0.0); upper = variable.get("upper", 1.0 if kind == "binary" else math.inf)
        if not isinstance(lower, (int, float)) or not math.isfinite(float(lower)) or not isinstance(upper, (int, float)) or math.isnan(float(upper)) or float(lower) > float(upper):
            errors.append(f"robust_optimization_bounds_invalid:{variable.get('name', '')}")
    if len(names) != len(set(names)):
        errors.append("robust_optimization_variable_names_duplicate")
    declared = set(names)
    if not isinstance(scenarios, list) or len(scenarios) < 2:
        errors.append("robust_optimization_requires_two_scenarios")
    else:
        scenario_names = []; probability_sum = 0.0
        for index, scenario in enumerate(scenarios):
            if not isinstance(scenario, dict) or not isinstance(scenario.get("name"), str) or not scenario.get("name") or not isinstance(scenario.get("probability"), (int, float)) or float(scenario.get("probability", 0)) <= 0 or not isinstance(scenario.get("objective_coefficients"), dict) or not isinstance(scenario.get("linear_constraints"), list):
                errors.append(f"robust_scenario_schema_invalid:{index}"); continue
            scenario_names.append(scenario["name"]); probability_sum += float(scenario["probability"])
            coefficients = scenario["objective_coefficients"]
            if not set(coefficients).issubset(declared) or any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in coefficients.values()):
                errors.append(f"robust_scenario_objective_invalid:{scenario['name']}")
            for constraint in scenario["linear_constraints"]:
                if not isinstance(constraint, dict) or not isinstance(constraint.get("name"), str) or constraint.get("sense") not in {"<=", ">=", "=="} or not isinstance(constraint.get("coefficients"), dict) or not isinstance(constraint.get("rhs"), (int, float)):
                    errors.append(f"robust_scenario_constraint_invalid:{scenario['name']}"); continue
                if not set(constraint["coefficients"]).issubset(declared) or not math.isfinite(float(constraint["rhs"])) or any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in constraint["coefficients"].values()):
                    errors.append(f"robust_scenario_constraint_values_invalid:{scenario['name']}:{constraint.get('name', '')}")
        if len(scenario_names) != len(set(scenario_names)):
            errors.append("robust_scenario_names_duplicate")
        if not math.isclose(probability_sum, 1.0, rel_tol=1e-8, abs_tol=1e-8):
            errors.append("robust_scenario_probabilities_must_sum_to_one")
    risk = model.get("risk_measure", "worst_case")
    if risk not in {"expected", "worst_case", "cvar"}:
        errors.append("robust_risk_measure_invalid")
    alpha = model.get("alpha", 0.95)
    if risk == "cvar" and (not isinstance(alpha, (int, float)) or not math.isfinite(float(alpha)) or not 0.0 < float(alpha) < 1.0):
        errors.append("robust_cvar_alpha_invalid")
    return errors


def _validate_retail_decision_model(contract: ProblemContract) -> List[str]:
    model = contract.retail_decision_model
    if not isinstance(model, dict):
        return ["retail_decision_model_not_object"]
    errors: List[str] = []
    scenarios = model.get("scenarios")
    items = model.get("items")
    if not isinstance(scenarios, list) or len(scenarios) < 2:
        return ["retail_requires_two_scenarios"]
    scenario_names = []
    probability_sum = 0.0
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict) or not isinstance(scenario.get("name"), str) or not scenario.get("name") or not isinstance(scenario.get("probability"), (int, float)) or float(scenario.get("probability", 0)) <= 0:
            errors.append(f"retail_scenario_invalid:{index}")
            continue
        scenario_names.append(scenario["name"])
        probability_sum += float(scenario["probability"])
    if len(scenario_names) != len(set(scenario_names)):
        errors.append("retail_scenario_names_duplicate")
    if not math.isclose(probability_sum, 1.0, rel_tol=1e-8, abs_tol=1e-8):
        errors.append("retail_scenario_probabilities_must_sum_to_one")
    if not isinstance(items, list) or not items:
        return errors + ["retail_items_missing"]
    item_names = []
    for index, item in enumerate(items):
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item.get("name"):
            errors.append(f"retail_item_invalid:{index}")
            continue
        name = item["name"]
        item_names.append(name)
        numeric_fields = ("unit_cost", "min_replenishment", "max_replenishment")
        if any(not isinstance(item.get(field), (int, float)) or not math.isfinite(float(item[field])) or float(item[field]) < 0 for field in numeric_fields):
            errors.append(f"retail_item_numeric_invalid:{name}")
            continue
        if float(item["min_replenishment"]) > float(item["max_replenishment"]):
            errors.append(f"retail_item_bounds_reversed:{name}")
        loss = item.get("loss_rate", 0.0)
        if not isinstance(loss, (int, float)) or not 0 <= float(loss) < 1:
            errors.append(f"retail_loss_rate_invalid:{name}")
        options = item.get("price_options")
        if not isinstance(options, list) or not options:
            errors.append(f"retail_price_options_missing:{name}")
            continue
        prices = []
        for option_index, option in enumerate(options):
            if not isinstance(option, dict) or not isinstance(option.get("price"), (int, float)) or not math.isfinite(float(option["price"])) or float(option["price"]) <= 0 or not isinstance(option.get("demand"), dict):
                errors.append(f"retail_price_option_invalid:{name}:{option_index}")
                continue
            prices.append(float(option["price"]))
            demand = option["demand"]
            if set(demand) != set(scenario_names) or any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0 for value in demand.values()):
                errors.append(f"retail_demand_scenarios_invalid:{name}:{option_index}")
        if len(prices) != len(set(prices)):
            errors.append(f"retail_price_options_duplicate:{name}")
        if "stress_elasticity_range" in item:
            stress_range = item["stress_elasticity_range"]
            if not isinstance(stress_range, list) or len(stress_range) != 2 or any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in stress_range) or float(stress_range[0]) > float(stress_range[1]):
                errors.append(f"retail_stress_elasticity_range_invalid:{name}")
            elif not all(isinstance(item.get(field), (int, float)) and math.isfinite(float(item[field])) and float(item[field]) > 0 for field in ("base_demand", "reference_price")) or not isinstance(item.get("elasticity"), (int, float)) or not math.isfinite(float(item["elasticity"])):
                errors.append(f"retail_stress_elasticity_inputs_invalid:{name}")
    if len(item_names) != len(set(item_names)):
        errors.append("retail_item_names_duplicate")
    for field in ("budget", "shortage_penalty", "baseline_markup"):
        value = model.get(field, 0.0)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
            errors.append(f"retail_{field}_invalid")
    minimum = model.get("min_selected", 0)
    maximum = model.get("max_selected", len(items))
    if not isinstance(minimum, int) or not isinstance(maximum, int) or not 0 <= minimum <= maximum <= len(items):
        errors.append("retail_selection_bounds_invalid")
    groups = {str(item.get("group")) for item in items if isinstance(item, dict) and item.get("group") is not None}
    group_bounds = model.get("group_selection_bounds", {})
    if not isinstance(group_bounds, dict):
        errors.append("retail_group_selection_bounds_invalid")
    else:
        for group, bounds in group_bounds.items():
            group_size = sum(str(item.get("group")) == str(group) for item in items if isinstance(item, dict))
            if str(group) not in groups or not isinstance(bounds, dict) or not isinstance(bounds.get("min", 0), int) or not isinstance(bounds.get("max", group_size), int) or not 0 <= bounds.get("min", 0) <= bounds.get("max", group_size) <= group_size:
                errors.append(f"retail_group_selection_bound_invalid:{group}")
    service_requirements = model.get("category_service_requirements", [])
    if not isinstance(service_requirements, list):
        errors.append("retail_category_service_requirements_invalid")
    else:
        seen_requirements = set()
        for index, requirement in enumerate(service_requirements):
            if not isinstance(requirement, dict):
                errors.append(f"retail_category_service_requirement_invalid:{index}")
                continue
            key = (str(requirement.get("group")), str(requirement.get("scenario")))
            minimum_sales = requirement.get("minimum_sales")
            if key[0] not in groups or key[1] not in scenario_names or not isinstance(minimum_sales, (int, float)) or not math.isfinite(float(minimum_sales)) or float(minimum_sales) < 0:
                errors.append(f"retail_category_service_requirement_invalid:{index}")
            if key in seen_requirements:
                errors.append(f"retail_category_service_requirement_duplicate:{key[0]}:{key[1]}")
            seen_requirements.add(key)
    risk = model.get("risk_measure", "expected")
    if risk not in {"expected", "worst_case", "cvar"}:
        errors.append("retail_risk_measure_invalid")
    alpha = model.get("alpha", 0.95)
    if risk == "cvar" and (not isinstance(alpha, (int, float)) or not 0 < float(alpha) < 1):
        errors.append("retail_cvar_alpha_invalid")
    stress = model.get("stress_test", {})
    if not isinstance(stress, dict):
        errors.append("retail_stress_test_invalid")
    elif stress:
        if not isinstance(stress.get("seed"), int):
            errors.append("retail_stress_seed_invalid")
        if not isinstance(stress.get("replications"), int) or not 30 <= stress["replications"] <= 100000:
            errors.append("retail_stress_replications_invalid")
        if not isinstance(stress.get("demand_cv"), (int, float)) or not math.isfinite(float(stress["demand_cv"])) or not 0 <= float(stress["demand_cv"]) <= 5:
            errors.append("retail_stress_demand_cv_invalid")
        correlation = stress.get("demand_correlation")
        if correlation is not None:
            valid = isinstance(correlation, dict) and isinstance(correlation.get("groups"), list) and isinstance(correlation.get("matrix"), list)
            if valid:
                correlation_groups = [str(value) for value in correlation["groups"]]
                item_groups = {str(item.get("category", item.get("group"))) for item in items if isinstance(item, dict)}
                try:
                    matrix = np.asarray(correlation["matrix"], dtype=float)
                    valid = len(correlation_groups) == len(set(correlation_groups)) and set(correlation_groups) == item_groups and matrix.shape == (len(correlation_groups), len(correlation_groups)) and np.isfinite(matrix).all() and np.allclose(matrix, matrix.T, rtol=1e-8, atol=1e-8) and np.allclose(np.diag(matrix), 1.0, rtol=1e-8, atol=1e-8) and float(np.linalg.eigvalsh(matrix).min()) >= -1e-8 and np.max(np.abs(matrix)) <= 1.0 + 1e-8
                except Exception:
                    valid = False
            if not valid:
                errors.append("retail_stress_demand_correlation_invalid")
    return errors


def _validate_spatial_model(contract: ProblemContract) -> List[str]:
    model = contract.spatial_model
    if not isinstance(model, dict):
        return ["spatial_model_not_object"]
    errors = []
    if model.get("kind") not in {"idw"}:
        errors.append(f"spatial_kind_invalid:{model.get('kind')}")
    if model.get("coordinate_system", "projected") not in {"projected", "geographic_wgs84"}:
        errors.append(f"spatial_coordinate_system_invalid:{model.get('coordinate_system')}")
    for name in ("x_column", "y_column", "target_column"):
        if not isinstance(model.get(name), str) or not model.get(name):
            errors.append(f"spatial_{name}_required")
    for name, default in (("power", 2.0), ("block_size", 1.0)):
        value = model.get(name, default)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0:
            errors.append(f"spatial_{name}_must_be_positive")
    neighbors = model.get("moran_neighbors", 4)
    if not isinstance(neighbors, int) or neighbors < 1:
        errors.append("spatial_moran_neighbors_invalid")
    return errors


def _validate_multiobjective_model(contract: ProblemContract) -> List[str]:
    model = contract.multiobjective_model
    if not isinstance(model, dict):
        return ["multiobjective_model_not_object"]
    errors: List[str] = []
    base = model.get("base_model")
    objectives = model.get("objectives")
    variable_names: set[str] = set()
    if not isinstance(base, dict):
        errors.append("multiobjective_base_model_required")
    elif not isinstance(base.get("variables"), list) or not isinstance(base.get("linear_constraints"), list):
        errors.append("multiobjective_base_model_schema_invalid")
    else:
        variables = base["variables"]
        if not variables:
            errors.append("multiobjective_variables_required")
        names = []
        for item in variables:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item.get("name") or item.get("kind", "continuous") not in {"continuous", "integer", "binary"}:
                errors.append("multiobjective_variable_schema_invalid")
                continue
            names.append(item["name"]); kind = item.get("kind", "continuous")
            lower = item.get("lower", 0.0); upper = item.get("upper", 1.0 if kind == "binary" else math.inf)
            if not isinstance(lower, (int, float)) or not math.isfinite(float(lower)) or not isinstance(upper, (int, float)) or math.isnan(float(upper)) or float(lower) > float(upper):
                errors.append(f"multiobjective_variable_bounds_invalid:{item['name']}")
            if kind == "binary" and (float(lower) < 0.0 or float(upper) > 1.0):
                errors.append(f"multiobjective_binary_bounds_invalid:{item['name']}")
        if len(names) != len(set(names)):
            errors.append("multiobjective_variable_names_duplicate")
        variable_names = set(names)
        for item in base["linear_constraints"]:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item.get("name") or item.get("sense") not in {"<=", ">=", "=="} or not isinstance(item.get("coefficients"), dict) or not isinstance(item.get("rhs"), (int, float)):
                errors.append("multiobjective_linear_constraint_invalid")
                continue
            if not math.isfinite(float(item["rhs"])) or any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in item["coefficients"].values()):
                errors.append(f"multiobjective_constraint_values_invalid:{item['name']}")
            if not set(item["coefficients"]).issubset(variable_names):
                errors.append(f"multiobjective_constraint_unknown_variable:{item['name']}")
    if not isinstance(objectives, list) or len(objectives) != 2:
        errors.append("multiobjective_native_requires_exactly_two_objectives")
    else:
        names = []
        for item in objectives:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str) or item.get("sense") not in {"min", "max"} or not isinstance(item.get("coefficients"), dict):
                errors.append("multiobjective_objective_schema_invalid")
                continue
            names.append(item["name"])
            if not set(item["coefficients"]).issubset(variable_names):
                errors.append(f"multiobjective_unknown_variable:{item['name']}")
            if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in item["coefficients"].values()):
                errors.append(f"multiobjective_coefficients_invalid:{item['name']}")
            elif not any(abs(float(value)) > 0.0 for value in item["coefficients"].values()):
                errors.append(f"multiobjective_objective_zero:{item['name']}")
        if len(names) != len(set(names)):
            errors.append("multiobjective_objective_names_duplicate")
    grid = model.get("epsilon_grid_size", 7)
    if not isinstance(grid, int) or grid < 2 or grid > 101:
        errors.append("multiobjective_epsilon_grid_size_invalid")
    return errors


def _validate_simulation_model(contract: ProblemContract) -> List[str]:
    model = contract.simulation_model
    if not isinstance(model, dict):
        return ["simulation_model_not_object"]
    errors = []
    if model.get("kind") not in {"mm1_queue", "mmc_queue"}:
        errors.append(f"simulation_kind_invalid:{model.get('kind')}")
    for name in ("arrival_rate", "service_rate"):
        value = model.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0:
            errors.append(f"simulation_{name}_must_be_positive")
    for name in ("arrival_distribution", "service_distribution"):
        distribution = model.get(name, {"kind": "exponential"})
        if not isinstance(distribution, dict) or distribution.get("kind", "exponential") not in {"exponential", "deterministic", "empirical"}:
            errors.append(f"simulation_{name}_invalid")
            continue
        if distribution.get("kind") == "empirical":
            values = distribution.get("values")
            weights = distribution.get("weights")
            if not isinstance(values, list) or not values or any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0 for value in values):
                errors.append(f"simulation_{name}_values_invalid")
            if weights is not None and (not isinstance(weights, list) or len(weights) != len(values or []) or any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0 for value in weights) or sum(float(value) for value in weights) <= 0):
                errors.append(f"simulation_{name}_weights_invalid")
    if model.get("kind") == "mm1_queue" and isinstance(model.get("arrival_rate"), (int, float)) and isinstance(model.get("service_rate"), (int, float)) and float(model["arrival_rate"]) >= float(model["service_rate"]):
        errors.append("simulation_mm1_requires_arrival_rate_below_service_rate")
    if model.get("kind") == "mmc_queue":
        servers = model.get("servers")
        if not isinstance(servers, int) or servers < 2:
            errors.append("simulation_servers_invalid")
        capacity = model.get("capacity")
        if capacity is not None and (not isinstance(capacity, int) or capacity < servers):
            errors.append("simulation_capacity_invalid")
        if (
            capacity is None
            and isinstance(servers, int)
            and isinstance(model.get("arrival_rate"), (int, float))
            and isinstance(model.get("service_rate"), (int, float))
            and float(model["arrival_rate"]) >= servers * float(model["service_rate"])
        ):
            errors.append("simulation_mmc_requires_arrival_rate_below_total_service_rate")
    for name, minimum in (("horizon", 1.0), ("warmup", 0.0)):
        value = model.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < minimum:
            errors.append(f"simulation_{name}_invalid")
    if isinstance(model.get("horizon"), (int, float)) and isinstance(model.get("warmup"), (int, float)) and float(model["warmup"]) >= float(model["horizon"]):
        errors.append("simulation_warmup_must_be_less_than_horizon")
    for name, minimum in (("replications", 2), ("seed", 0)):
        value = model.get(name, 20 if name == "replications" else 42)
        if not isinstance(value, int) or value < minimum:
            errors.append(f"simulation_{name}_invalid")
    return errors


def _validate_mechanism_model(contract: ProblemContract) -> List[str]:
    model = contract.mechanism_model
    if not isinstance(model, dict):
        return ["mechanism_model_not_object"]
    errors = []
    kind = model.get("kind")
    allowed = {"first_order_relaxation", "logistic_growth", "sir"}
    if kind not in allowed:
        errors.append(f"mechanism_kind_invalid:{kind}")
    time_column = str(model.get("time_column") or contract.time_column)
    if not time_column:
        errors.append("mechanism_time_column_required")
    state_columns = model.get("state_columns")
    if not isinstance(state_columns, list) or not state_columns or not all(isinstance(value, str) and value for value in state_columns):
        errors.append("mechanism_state_columns_required")
    elif kind in {"first_order_relaxation", "logistic_growth"} and len(state_columns) != 1:
        errors.append("mechanism_single_state_required")
    elif kind == "sir" and state_columns != ["S", "I", "R"]:
        errors.append("mechanism_sir_state_columns_must_be_S_I_R")
    parameters = model.get("parameters")
    required = {
        "first_order_relaxation": {"k", "equilibrium"},
        "logistic_growth": {"r", "K"},
        "sir": {"beta", "gamma"},
    }.get(kind, set())
    if not isinstance(parameters, dict) or set(parameters) != required:
        errors.append("mechanism_parameter_set_invalid")
    else:
        for name, bounds in parameters.items():
            if not isinstance(bounds, dict) or not all(isinstance(bounds.get(key), (int, float)) and math.isfinite(float(bounds[key])) for key in ("lower", "upper")):
                errors.append(f"mechanism_parameter_bounds_invalid:{name}")
            elif float(bounds["lower"]) > float(bounds["upper"]):
                errors.append(f"mechanism_parameter_bounds_reversed:{name}")
    initial = model.get("initial_state")
    if not isinstance(initial, dict) or not isinstance(state_columns, list) or set(initial) != set(state_columns):
        errors.append("mechanism_initial_state_invalid")
    elif any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in initial.values()):
        errors.append("mechanism_initial_state_non_finite")
    if kind == "logistic_growth" and isinstance(parameters, dict) and isinstance(parameters.get("K"), dict) and float(parameters["K"].get("lower", 0)) <= 0:
        errors.append("mechanism_logistic_K_must_be_positive")
    if kind == "sir" and isinstance(parameters, dict) and any(float(parameters[name].get("lower", 0)) < 0 for name in ("beta", "gamma") if isinstance(parameters.get(name), dict)):
        errors.append("mechanism_sir_rates_must_be_nonnegative")
    return errors


def _validate_network_model(contract: ProblemContract) -> List[str]:
    model = contract.network_model
    errors = []
    if not isinstance(model, dict):
        return ["network_model_not_object"]
    kind = model.get("kind")
    if kind not in {"shortest_path", "max_flow", "min_cost_flow", "assignment", "transportation"}:
        errors.append(f"network_kind_invalid:{kind}")
    nodes = model.get("nodes")
    edges = model.get("edges")
    if not isinstance(nodes, list) or not nodes:
        errors.append("network_nodes_missing")
    if not isinstance(edges, list) or not edges:
        errors.append("network_edges_missing")
    node_set = {str(node) for node in nodes} if isinstance(nodes, list) else set()
    edge_ids = set()
    for edge in edges or []:
        if not isinstance(edge, dict) or not all(key in edge for key in ("id", "from", "to")):
            errors.append("network_edge_schema_invalid")
            continue
        edge_id = str(edge["id"])
        if edge_id in edge_ids:
            errors.append(f"network_duplicate_edge:{edge_id}")
        edge_ids.add(edge_id)
        if str(edge["from"]) not in node_set or str(edge["to"]) not in node_set:
            errors.append(f"network_edge_endpoint_missing:{edge_id}")
        for name in ("cost", "capacity", "lower"):
            if name in edge and (not isinstance(edge[name], (int, float)) or not math.isfinite(float(edge[name]))):
                errors.append(f"network_{name}_invalid:{edge_id}")
        if float(edge.get("capacity", math.inf)) < float(edge.get("lower", 0.0)):
            errors.append(f"network_capacity_below_lower:{edge_id}")
    if kind in {"shortest_path", "max_flow"} and (str(model.get("source")) not in node_set or str(model.get("sink")) not in node_set):
        errors.append("network_source_sink_required")
    if kind in {"min_cost_flow", "transportation", "assignment"}:
        supplies = model.get("supplies")
        if not isinstance(supplies, dict) or not supplies:
            errors.append("network_supplies_required")
        elif any(str(node) not in node_set for node in supplies):
            errors.append("network_supply_node_missing")
        elif abs(sum(float(value) for value in supplies.values())) > 1e-7:
            errors.append("network_supply_sum_must_be_zero")
    return errors


def _within_audited_workspace(path: Path, contract: ProblemContract, assets: List[DataAsset]) -> bool:
    roots = {Path(asset.path).resolve().parent for asset in assets}
    if contract.data_file:
        roots.add(Path(contract.data_file).resolve().parent)
    for raw_root in contract.allowed_code_roots:
        root = Path(raw_root).resolve()
        roots.add(root)
    try:
        return any(path.is_relative_to(root) for root in roots)
    except ValueError:
        return False


def write_contract(path: Path, contract: ProblemContract) -> Path:
    from dataclasses import asdict

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(contract), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _task_type(archetypes: Iterable[str]) -> str:
    values = set(archetypes)
    if "forecasting" in values or "retail_forecast_operation" in values:
        return "forecasting"
    if "evaluation_ranking" in values:
        return "evaluation_ranking"
    if "constrained_optimization" in values:
        return "constrained_optimization"
    if "mechanism_or_physics" in values:
        return "mechanism"
    if "simulation_or_queueing" in values:
        return "simulation"
    if "spatial_or_geostat" in values:
        return "spatial"
    return "regression"


def _target_from_brief(brief: str, columns: Iterable[str]) -> str:
    if not brief:
        return ""
    column_lookup = {str(column).lower(): str(column) for column in columns}
    for pattern in TARGET_PATTERNS:
        for match in pattern.finditer(brief):
            token = match.group(1).strip().lower()
            if token in column_lookup:
                return column_lookup[token]
    lowered = brief.lower()
    explicit = [original for key, original in column_lookup.items() if key in lowered]
    target_like = [col for col in explicit if any(term in col.lower() for term in ("target", "label", "sales", "demand", "profit", "score", "销量", "需求", "收益", "评分"))]
    return target_like[0] if len(target_like) == 1 else ""
