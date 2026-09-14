from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .schemas import ExperimentRecord, ProblemContract

VERIFIED_FORECAST_ROUTES = {"mean_or_last_value_baseline", "rolling_mean_model", "seasonal_naive_forecast", "regularized_lag_model"}
VERIFIED_REGRESSION_ROUTES = {"simple_statistical_baseline", "regularized_regression_cv"}


def _read(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    for encoding in ("utf-8", "gb18030", "gbk", "utf-16"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=encoding)
        except Exception:
            continue
    raise ValueError("predictive_verifier_data_read_failed")


def _prepare(contract: ProblemContract) -> tuple[pd.DataFrame, int, int, str]:
    frame = _read(Path(contract.data_file)); target = contract.target_column
    required = list(dict.fromkeys([target, contract.time_column] + list(contract.known_future_columns)))
    model_frame = frame[required].dropna(subset=[target]).copy(); model_frame["_source_row"] = model_frame.index.to_numpy(dtype=int)
    time_values = model_frame[contract.time_column]
    if not pd.api.types.is_numeric_dtype(time_values):
        parsed = pd.to_datetime(time_values, errors="coerce")
        if parsed.isna().any(): raise ValueError("forecast_time_column_contains_unparseable_values")
        model_frame["_parsed_time"] = parsed; sort_column = "_parsed_time"
    else: sort_column = contract.time_column
    if model_frame[sort_column].duplicated().any(): raise ValueError("forecast_time_column_contains_duplicates")
    model_frame = model_frame.sort_values(sort_column, kind="stable").drop(columns=["_parsed_time"], errors="ignore").reset_index(drop=True)
    horizon = int(contract.forecast_horizon); raw_cut = len(model_frame) - horizon; selection_size = max(2, min(horizon, raw_cut // 4)); selection_cut = raw_cut - selection_size
    return model_frame, selection_cut, raw_cut, target


def _engineer(frame: pd.DataFrame, target: str, group_columns: list[str]) -> tuple[np.ndarray, np.ndarray, list[str], Dict[str, Any], np.ndarray]:
    engineered = frame.copy(); engineered["_row_index"] = np.arange(len(engineered), dtype=float)
    excluded = {target, "_source_row", *group_columns}
    numeric = [column for column in engineered.columns if column not in excluded and pd.api.types.is_numeric_dtype(engineered[column])]
    spec = {"numeric_columns": list(numeric), "squared_columns": list(numeric[:5]), "interaction_columns": list(numeric[:2])}
    for lag in (1, 2, 3, 7): engineered[f"{target}_lag_{lag}"] = engineered[target].shift(lag)
    engineered[f"{target}_roll3"] = engineered[target].shift(1).rolling(3, min_periods=1).mean()
    engineered[f"{target}_roll7"] = engineered[target].shift(1).rolling(7, min_periods=1).mean()
    for column in numeric[:5]: engineered[f"{column}_sq"] = engineered[column] ** 2
    if len(numeric) >= 2: engineered[f"{numeric[0]}_x_{numeric[1]}"] = engineered[numeric[0]] * engineered[numeric[1]]
    engineered = engineered.dropna(subset=[target]).copy()
    features = [column for column in engineered.columns if column not in excluded and pd.api.types.is_numeric_dtype(engineered[column])]
    X = engineered[features].to_numpy(dtype=float); y = engineered[target].to_numpy(dtype=float)
    medians = np.nanmedian(X, axis=0); medians = np.where(np.isfinite(medians), medians, 0.0); X = np.where(np.isfinite(X), X, medians)
    return X, y, features, spec, medians


def _feature_row(row: pd.Series, row_index: int, history: list[float], target: str, features: list[str], medians: np.ndarray, spec: Dict[str, Any]) -> np.ndarray:
    values: Dict[str, float] = {"_row_index": float(row_index)}
    for column in spec["numeric_columns"]:
        if column != "_row_index": values[column] = float(row[column]) if column in row.index and pd.notna(row[column]) else np.nan
    for lag in (1, 2, 3, 7): values[f"{target}_lag_{lag}"] = history[-lag] if len(history) >= lag else np.nan
    values[f"{target}_roll3"] = float(np.mean(history[-3:])) if history else np.nan; values[f"{target}_roll7"] = float(np.mean(history[-7:])) if history else np.nan
    for column in spec["squared_columns"]: values[f"{column}_sq"] = values.get(column, np.nan) ** 2
    interaction = spec["interaction_columns"]
    if len(interaction) >= 2: values[f"{interaction[0]}_x_{interaction[1]}"] = values.get(interaction[0], np.nan) * values.get(interaction[1], np.nan)
    vector = np.asarray([values.get(feature, np.nan) for feature in features], dtype=float)
    return np.where(np.isfinite(vector), vector, medians).reshape(1, -1)


def _recursive(model: Ridge, frame: pd.DataFrame, start: int, target: str, features: list[str], medians: np.ndarray, spec: Dict[str, Any]) -> np.ndarray:
    history = frame[target].iloc[:start].astype(float).tolist(); result = []
    for index in range(start, len(frame)):
        prediction = float(model.predict(_feature_row(frame.iloc[index], index, history, target, features, medians, spec))[0]); result.append(prediction); history.append(prediction)
    return np.asarray(result)


def _rebuild_forecast(contract: ProblemContract, route_id: str) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    frame, selection_cut, raw_cut, target = _prepare(contract); y = frame[target].to_numpy(dtype=float)
    if route_id == "mean_or_last_value_baseline":
        selection = np.repeat(y[selection_cut - 1], raw_cut - selection_cut); final = np.repeat(y[raw_cut - 1], len(frame) - raw_cut)
    elif route_id == "rolling_mean_model":
        selection = np.repeat(float(np.mean(y[max(0, selection_cut - 7):selection_cut])), raw_cut - selection_cut); final = np.repeat(float(np.mean(y[max(0, raw_cut - 7):raw_cut])), len(frame) - raw_cut)
    elif route_id == "seasonal_naive_forecast":
        period = int(contract.seasonal_period)
        selection_history = y[:selection_cut].astype(float).tolist(); selection_values = []
        for _ in range(raw_cut - selection_cut):
            value = selection_history[-period]; selection_values.append(value); selection_history.append(value)
        final_history = y[:raw_cut].astype(float).tolist(); final_values = []
        for _ in range(len(frame) - raw_cut):
            value = final_history[-period]; final_values.append(value); final_history.append(value)
        selection = np.asarray(selection_values); final = np.asarray(final_values)
    elif route_id == "regularized_lag_model":
        train = frame.iloc[:selection_cut].copy(); X, train_y, features, spec, medians = _engineer(train, target, contract.group_columns); model = Ridge(alpha=1.0).fit(X, train_y)
        selection = _recursive(model, frame.iloc[:raw_cut], selection_cut, target, features, medians, spec)
        development = frame.iloc[:raw_cut].copy(); X_dev, y_dev, final_features, final_spec, final_medians = _engineer(development, target, contract.group_columns); final_model = Ridge(alpha=1.0).fit(X_dev, y_dev)
        final = _recursive(final_model, frame, raw_cut, target, final_features, final_medians, final_spec)
    else: raise ValueError(f"predictive_route_not_independently_supported:{route_id}")
    selection_rows = frame["_source_row"].iloc[selection_cut:raw_cut].astype(int).tolist(); final_rows = frame["_source_row"].iloc[raw_cut:].astype(int).tolist()
    return selection, final, selection_rows, final_rows


def _rebuild_regression(contract: ProblemContract, route_id: str) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    frame = _read(Path(contract.data_file)); target = contract.target_column
    required = list(dict.fromkeys([target] + list(contract.feature_columns) + list(contract.group_columns)))
    model_frame = frame[required].dropna(subset=[target]).copy() if contract.feature_columns else frame.dropna(subset=[target]).copy()
    model_frame["_source_row"] = model_frame.index.to_numpy(dtype=int)
    engineered = model_frame.copy(); engineered["_row_index"] = np.arange(len(engineered), dtype=float)
    excluded = {target, "_source_row", *contract.group_columns}
    numeric = [column for column in engineered.columns if column not in excluded and pd.api.types.is_numeric_dtype(engineered[column])]
    for column in numeric[:5]: engineered[f"{column}_sq"] = engineered[column] ** 2
    if len(numeric) >= 2: engineered[f"{numeric[0]}_x_{numeric[1]}"] = engineered[numeric[0]] * engineered[numeric[1]]
    engineered = engineered.dropna(subset=[target]).copy(); features = [column for column in engineered.columns if column not in excluded and pd.api.types.is_numeric_dtype(engineered[column])]
    X = engineered[features].to_numpy(dtype=float); y = engineered[target].to_numpy(dtype=float); row_ids = engineered["_source_row"].to_numpy(dtype=int)
    if contract.group_columns:
        groups = engineered[contract.group_columns].astype(str).agg("||".join, axis=1).to_numpy()
        outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42); development_index, test_index = next(outer.split(X, y, groups))
        X_development, X_test = X[development_index], X[test_index]; y_development, y_test = y[development_index], y[test_index]; development_rows, test_rows = row_ids[development_index], row_ids[test_index]; development_groups = groups[development_index]
        inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=7); train_index, selection_index = next(inner.split(X_development, y_development, development_groups))
        X_train, X_selection = X_development[train_index], X_development[selection_index]; y_train, y_selection = y_development[train_index], y_development[selection_index]; selection_rows = development_rows[selection_index]
    else:
        X_development, X_test, y_development, y_test, development_rows, test_rows = train_test_split(X, y, row_ids, test_size=0.2, random_state=42)
        X_train, X_selection, y_train, y_selection, _, selection_rows = train_test_split(X_development, y_development, development_rows, test_size=0.25, random_state=7)
    if route_id == "simple_statistical_baseline":
        selection = np.repeat(float(np.mean(y_train)), len(y_selection)); final = np.repeat(float(np.mean(np.concatenate([y_train, y_selection]))), len(y_test))
    elif route_id == "regularized_regression_cv":
        medians = np.nanmedian(X_train, axis=0); medians = np.where(np.isfinite(medians), medians, 0.0)
        X_train = np.where(np.isfinite(X_train), X_train, medians); X_selection = np.where(np.isfinite(X_selection), X_selection, medians); X_test = np.where(np.isfinite(X_test), X_test, medians)
        selection = Ridge(alpha=1.0).fit(X_train, y_train).predict(X_selection)
        final = Ridge(alpha=1.0).fit(np.vstack([X_train, X_selection]), np.concatenate([y_train, y_selection])).predict(X_test)
    else: raise ValueError(f"predictive_route_not_independently_supported:{route_id}")
    return np.asarray(selection), np.asarray(final), [int(value) for value in selection_rows], [int(value) for value in test_rows]


def verify_native_predictive(contract: ProblemContract, records: Iterable[ExperimentRecord], registry_path: Path, output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True); records = list(records); checks = []; issues = []; verified_count = 0; unsupported = []
    for record in records:
        if record.status != "executed" or record.role in {"ablation", "diagnostic"}: continue
        supported = (contract.task_type == "forecasting" and record.route_id in VERIFIED_FORECAST_ROUTES) or (contract.task_type == "regression" and record.route_id in VERIFIED_REGRESSION_ROUTES)
        if not supported:
            unsupported.append(record.route_id); continue
        try:
            payload = json.loads(Path(record.evidence_paths["route_evidence"]).read_text(encoding="utf-8")); evidence = payload["evidence"]
            selection, final, selection_rows, final_rows = (_rebuild_forecast(contract, record.route_id) if contract.task_type == "forecasting" else _rebuild_regression(contract, record.route_id))
            predictions_ok = np.allclose(selection, np.asarray(evidence["predicted"], dtype=float), rtol=1e-9, atol=1e-9) and np.allclose(final, np.asarray(evidence["final_test_predicted"], dtype=float), rtol=1e-9, atol=1e-9)
            rows_ok = selection_rows == evidence["sample_rows"] and final_rows == evidence["final_test_sample_rows"]
            item = {"route_id": record.route_id, "predictions_recomputed": bool(predictions_ok), "split_rows_recomputed": bool(rows_ok)}; checks.append(item); verified_count += 1
            if not predictions_ok or not rows_ok: issues.append(f"native_predictive_verification_failed:{record.route_id}")
        except Exception as exc: issues.append(f"native_predictive_verification_error:{record.route_id}:{exc}")
    registry_path = Path(registry_path); stdout = output_dir / "native_predictive_verifier.stdout.txt"; stderr = output_dir / "native_predictive_verifier.stderr.txt"
    stdout.write_text(json.dumps({"checks": checks, "unsupported_routes": unsupported}, ensure_ascii=False), encoding="utf-8"); stderr.write_text("", encoding="utf-8"); run_ids = {record.run_id for record in records if record.run_id}
    all_claim_routes_verified = verified_count > 0 and not unsupported and not issues
    result = {"kind": "native_predictive", "passed": not issues and verified_count > 0, "issues": issues, "checks": checks, "unsupported_routes": unsupported, "claim_level": "domain_verified_predictive_result" if all_claim_routes_verified else "screening_evidence_only", "run_id": next(iter(run_ids)) if len(run_ids) == 1 else None, "contract_hash": _json_hash(asdict(contract)), "script_path": str(Path(__file__).resolve()), "script_hash": _sha256(Path(__file__).resolve()), "registry_path": str(registry_path.resolve()), "registry_hash": _sha256(registry_path), "stdout_path": str(stdout), "stdout_hash": _sha256(stdout), "stderr_path": str(stderr), "stderr_hash": _sha256(stderr), "exit_code": 0 if not issues else 1}
    (output_dir / "domain_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"); return result


def _sha256(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def _json_hash(payload: Any) -> str: return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
