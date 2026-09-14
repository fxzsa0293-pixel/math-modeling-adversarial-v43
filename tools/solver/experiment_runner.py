from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import numpy as np

from .schemas import ExperimentRecord


def run_experiments(adapter_paths: Iterable[Path], output_dir: Path, timeout_seconds: int = 120, max_repairs: int = 1) -> List[ExperimentRecord]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records: List[ExperimentRecord] = []
    run_id = uuid.uuid4().hex
    for adapter in adapter_paths:
        adapter = Path(adapter).resolve()
        original_hash = _sha256(adapter)
        record = _run_one(adapter, output_dir, run_id, timeout_seconds, original_hash, [])
        if record.status != "executed" and max_repairs > 0 and _repairable(record.error):
            repair_log = [{"attempt": 1, "strategy": "read_excel_engine_guard", "before_hash": original_hash, "reason": record.error[:500]}]
            _append_repair_guard(adapter)
            repair_log[0]["after_hash"] = _sha256(adapter)
            repaired = _run_one(adapter, output_dir, run_id, timeout_seconds, original_hash, repair_log)
            repaired.repair_attempts = 1
            record = repaired
        if record.status != "executed" and not record.repair_log:
            record.repair_log = [{"attempt": 0, "strategy": "repair_limited", "reason": "No safe automatic repair matched this failure."}]
        records.append(record)
        (output_dir / f"{record.route_id}.json").write_text(json.dumps(record.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
    return records


def _run_one(adapter: Path, output_dir: Path, run_id: str, timeout_seconds: int, original_hash: str, repair_log: List[dict]) -> ExperimentRecord:
    command = [sys.executable, str(adapter)]
    log_dir = output_dir / "raw_logs" / run_id
    evidence_dir = output_dir / "evidence" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{adapter.stem}.stdout.txt"
    stderr_path = log_dir / f"{adapter.stem}.stderr.txt"
    for stale in (stdout_path, stderr_path):
        if stale.exists():
            stale.unlink()
    try:
        child_env = os.environ.copy()
        child_env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.run(
            command,
            cwd=adapter.parent,
            env=child_env,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text(_timeout_text(exc.stdout), encoding="utf-8")
        stderr_path.write_text(_timeout_text(exc.stderr), encoding="utf-8")
        record = ExperimentRecord(
            "unknown", adapter.stem, "unknown", "failed", str(adapter), "", {},
            f"timeout_after_{timeout_seconds}s: {exc}", 0, [], "", original_hash, _sha256(adapter), repair_log,
        )
        return _with_provenance(record, run_id, command, None, stdout_path, stderr_path)
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        record = ExperimentRecord(
            "unknown", adapter.stem, "unknown", "failed", str(adapter), "", {},
            proc.stderr[-3000:] or proc.stdout[-3000:], 0, [], "", original_hash, _sha256(adapter), repair_log,
        )
        return _with_provenance(record, run_id, command, proc.returncode, stdout_path, stderr_path)
    try:
        payload = json.loads(proc.stdout[proc.stdout.index("{"):])
        metrics, evidence = _validate_and_recompute(payload)
        evidence_path = evidence_dir / f"{payload.get('route_id', adapter.stem)}.json"
        evidence_payload = {
            "run_id": run_id,
            "problem_id": payload.get("problem_id"),
            "route_id": payload.get("route_id"),
            "data_path": payload.get("data_path"),
            "contract": payload.get("contract", {}),
            "route": payload.get("route", {}),
            "environment": payload.get("environment", {}),
            "metrics_recomputed": metrics,
            "evidence": evidence,
        }
        evidence_path.write_text(json.dumps(evidence_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        record = ExperimentRecord(
            payload.get("problem_id", "unknown"),
            payload.get("route_id", adapter.stem),
            payload.get("role", "unknown"),
            payload.get("status", "executed"),
            str(adapter),
            payload.get("data_path", ""),
            metrics,
            "",
            0,
            payload.get("limitations", []),
            payload.get("recommendation", ""),
            original_hash,
            _sha256(adapter),
            repair_log,
        )
        record.input_hash = _sha256(Path(record.data_path)) if record.data_path and Path(record.data_path).is_file() else ""
        configuration = {"contract": payload.get("contract", {}), "route": payload.get("route", {})}
        record.config_hash = hashlib.sha256(json.dumps(configuration, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        record.evidence_paths = {"route_evidence": str(evidence_path)}
        record.artifact_hashes = {"route_evidence": _sha256(evidence_path)}
        record.environment = {str(key): str(value) for key, value in payload.get("environment", {}).items()}
        return _with_provenance(record, run_id, command, proc.returncode, stdout_path, stderr_path)
    except Exception as exc:
        record = ExperimentRecord(
            "unknown", adapter.stem, "unknown", "failed", str(adapter), "", {},
            f"json_parse_failed: {exc}\n{proc.stdout[-3000:]}", 0, [], "", original_hash, _sha256(adapter), repair_log,
        )
        return _with_provenance(record, run_id, command, proc.returncode, stdout_path, stderr_path)


def _validate_and_recompute(payload: dict) -> tuple[dict, dict]:
    required = {"problem_id", "route_id", "role", "status", "data_path", "metrics", "evidence", "contract", "route", "input_hash_at_load"}
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"missing_output_fields: {missing}")
    metrics = dict(payload["metrics"])
    evidence = payload["evidence"]
    contract = payload["contract"]
    route = payload["route"]
    data_path = Path(payload["data_path"]).resolve()
    if payload["status"] != "executed":
        raise ValueError("successful_process_must_report_executed")
    if route.get("route_id") != payload["route_id"] or route.get("role") != payload["role"]:
        raise ValueError("route_identity_mismatch")
    if contract.get("problem_id") != payload["problem_id"] or contract.get("status") != "ready":
        raise ValueError("contract_identity_or_status_mismatch")
    if not data_path.is_file() or Path(contract.get("data_file", "")).resolve() != data_path:
        raise ValueError("contract_data_path_mismatch")
    if payload["input_hash_at_load"] != _sha256(data_path):
        raise ValueError("input_changed_during_execution")
    if "observed" in evidence or "predicted" in evidence:
        observed = evidence.get("observed", [])
        predicted = evidence.get("predicted", [])
        if not observed or len(observed) != len(predicted):
            raise ValueError("prediction_evidence_length_mismatch")
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in observed + predicted):
            raise ValueError("prediction_evidence_non_finite")
        sample_rows = evidence.get("sample_rows", [])
        target = evidence.get("target_column")
        if target != contract.get("target_column") or len(sample_rows) != len(observed):
            raise ValueError("prediction_sample_identity_missing")
        source = _read_bound_table(data_path, contract.get("sheet_name"))
        if target not in source.columns or any(not isinstance(row, int) or row < 0 or row >= len(source) for row in sample_rows):
            raise ValueError("prediction_sample_identity_invalid")
        source_observed = pd.to_numeric(source.iloc[sample_rows][target], errors="coerce").tolist()
        if any(pd.isna(value) for value in source_observed) or any(not math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-9) for actual, expected in zip(observed, source_observed)):
            raise ValueError("observed_values_do_not_match_bound_data")
        selection_metrics = _regression_metrics(observed, predicted)
        metrics.update({f"selection_{name}": value for name, value in selection_metrics.items()})
        metrics.update(selection_metrics)
        final_observed = evidence.get("final_test_observed", [])
        final_predicted = evidence.get("final_test_predicted", [])
        final_rows = evidence.get("final_test_sample_rows", [])
        if not final_observed or len(final_observed) != len(final_predicted) or len(final_rows) != len(final_observed):
            raise ValueError("final_test_evidence_length_mismatch")
        if set(sample_rows) & set(final_rows):
            raise ValueError("selection_final_test_overlap")
        if contract.get("group_columns"):
            train_groups = set(evidence.get("train_groups", []))
            selection_groups = set(evidence.get("selection_groups", []))
            final_groups = set(evidence.get("final_test_groups", []))
            if not train_groups or not selection_groups or not final_groups or train_groups & selection_groups or train_groups & final_groups or selection_groups & final_groups:
                raise ValueError("group_holdout_overlap_or_missing")
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in final_observed + final_predicted):
            raise ValueError("final_test_evidence_non_finite")
        if any(not isinstance(row, int) or row < 0 or row >= len(source) for row in final_rows):
            raise ValueError("final_test_sample_identity_invalid")
        final_source_observed = pd.to_numeric(source.iloc[final_rows][target], errors="coerce").tolist()
        if any(pd.isna(value) for value in final_source_observed) or any(not math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-9) for actual, expected in zip(final_observed, final_source_observed)):
            raise ValueError("final_test_observed_values_do_not_match_bound_data")
        metrics.update({f"final_test_{name}": value for name, value in _regression_metrics(final_observed, final_predicted).items()})
    elif "scores" in evidence:
        scores = evidence.get("scores", [])
        if not scores or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in scores):
            raise ValueError("ranking_evidence_invalid")
        if not evidence.get("indicator_directions"):
            raise ValueError("ranking_direction_evidence_missing")
        if evidence.get("indicator_directions") != contract.get("indicator_directions"):
            raise ValueError("ranking_directions_contract_mismatch")
        columns = evidence.get("indicator_columns", [])
        values = evidence.get("indicator_values", [])
        sample_rows = evidence.get("sample_rows", [])
        if columns != list(contract.get("indicator_directions", {})) or len(values) != len(scores) or len(sample_rows) != len(scores):
            raise ValueError("ranking_sample_identity_missing")
        source = _read_bound_table(data_path, contract.get("sheet_name"))
        selected = source[columns].apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="all")
        selected = selected.fillna(selected.median(numeric_only=True)).fillna(0)
        if list(selected.index) != sample_rows or not _matrix_close(values, selected.to_numpy(dtype=float).tolist()):
            raise ValueError("ranking_values_do_not_match_bound_data")
        route_scores = _ranking_scores(payload["route_id"], payload["role"], _normalize_indicators(selected, contract["indicator_directions"]))
        if not _vector_close(scores, route_scores):
            raise ValueError("ranking_scores_not_recomputable")
        perturbations = evidence.get("perturbation_scores", [])
        if len(perturbations) < 20 or any(len(item) != len(scores) for item in perturbations):
            raise ValueError("ranking_perturbation_evidence_invalid")
        seed = evidence.get("perturbation_seed")
        sigma = evidence.get("perturbation_relative_sigma")
        if not isinstance(seed, int) or not isinstance(sigma, (int, float)) or sigma <= 0:
            raise ValueError("ranking_perturbation_protocol_invalid")
        rng = np.random.default_rng(seed)
        column_scale = selected.std(axis=0).replace(0, 1).fillna(1).to_numpy(dtype=float)
        recomputed_perturbations = []
        for _ in perturbations:
            perturbed = selected + rng.normal(0, sigma, size=selected.shape) * column_scale
            recomputed_perturbations.append(_ranking_scores(payload["route_id"], payload["role"], _normalize_indicators(perturbed, contract["indicator_directions"])))
        if any(not _vector_close(expected, actual) for expected, actual in zip(perturbations, recomputed_perturbations)):
            raise ValueError("ranking_perturbations_not_recomputable")
        correlations = [_spearman(scores, item) for item in recomputed_perturbations]
        stability = sum(correlations) / len(correlations)
        variance = sum((value - stability) ** 2 for value in correlations) / len(correlations)
        metrics["ranking_stability"] = round(stability, 6)
        metrics["ranking_stability_std"] = round(math.sqrt(variance), 6)
        metrics["perturbation_repeats"] = len(perturbations)
    elif "constraint_checks" in evidence or "objective_components" in evidence:
        checks = evidence.get("constraint_checks", [])
        components = evidence.get("objective_components", [])
        decisions = evidence.get("decision_variables", {})
        if not isinstance(checks, list) or not isinstance(components, list) or not components or not isinstance(decisions, dict) or not decisions:
            raise ValueError("optimization_evidence_incomplete")
        declared_names = {str(item.get("name")) for item in contract.get("constraints", []) if isinstance(item, dict) and item.get("name")}
        check_names = {str(item.get("name")) for item in checks if isinstance(item, dict)}
        if declared_names and not declared_names.issubset(check_names):
            raise ValueError("declared_constraints_not_checked")
        violations = []
        for item in checks:
            if not isinstance(item, dict) or not {"name", "lhs", "sense", "rhs"}.issubset(item):
                raise ValueError("constraint_check_schema_invalid")
            lhs, rhs = item["lhs"], item["rhs"]
            tolerance = item.get("tolerance", 1e-8)
            if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (lhs, rhs, tolerance)) or tolerance < 0:
                raise ValueError("constraint_check_value_invalid")
            if item["sense"] == "<=":
                violation = max(0.0, float(lhs) - float(rhs) - float(tolerance))
            elif item["sense"] == ">=":
                violation = max(0.0, float(rhs) - float(lhs) - float(tolerance))
            elif item["sense"] == "==":
                violation = max(0.0, abs(float(lhs) - float(rhs)) - float(tolerance))
            else:
                raise ValueError("constraint_check_sense_invalid")
            violations.append(violation)
        objective_values = []
        for item in components:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not isinstance(item.get("value"), (int, float)) or not math.isfinite(item["value"]):
                raise ValueError("objective_component_invalid")
            objective_values.append(float(item["value"]))
        if any(not isinstance(name, str) or not isinstance(value, (int, float)) or not math.isfinite(value) for name, value in decisions.items()):
            raise ValueError("decision_variable_invalid")
        metrics["objective"] = round(sum(objective_values), 6)
        metrics["constraint_violation"] = round(sum(violations), 12)
        metrics["constraint_count"] = len(checks)
    else:
        raise ValueError("unsupported_evidence_payload")
    if not all(not isinstance(value, float) or math.isfinite(value) for value in metrics.values()):
        raise ValueError("non_finite_metric")
    return metrics, evidence


def _read_bound_table(path: Path, sheet_name=None) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name or 0)
    last_error = None
    for encoding in ("utf-8", "gb18030", "gbk", "utf-16"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=encoding)
        except Exception as exc:
            last_error = exc
    raise last_error or ValueError("bound_table_unreadable")


def _regression_metrics(observed, predicted) -> dict:
    errors = [abs(actual - estimate) for actual, estimate in zip(observed, predicted)]
    result = {
        "MAE": round(sum(errors) / len(errors), 6),
        "RMSE": round(math.sqrt(sum((actual - estimate) ** 2 for actual, estimate in zip(observed, predicted)) / len(errors)), 6),
        "WAPE": round(sum(errors) / (sum(abs(value) for value in observed) or 1.0), 6),
    }
    observed_mean = sum(observed) / len(observed)
    denominator = sum((value - observed_mean) ** 2 for value in observed)
    if denominator:
        result["R2"] = round(1.0 - sum((actual - estimate) ** 2 for actual, estimate in zip(observed, predicted)) / denominator, 6)
    return result


def _matrix_close(left, right) -> bool:
    if len(left) != len(right):
        return False
    return all(len(a) == len(b) and all(math.isclose(float(x), float(y), rel_tol=1e-9, abs_tol=1e-9) for x, y in zip(a, b)) for a, b in zip(left, right))


def _vector_close(left, right) -> bool:
    return len(left) == len(right) and all(math.isclose(float(x), float(y), rel_tol=1e-8, abs_tol=1e-8) for x, y in zip(left, right))


def _normalize_indicators(frame: pd.DataFrame, directions: dict) -> pd.DataFrame:
    oriented = frame.copy()
    for column in oriented.columns:
        if directions.get(str(column)) == "lower_better":
            oriented[column] = -oriented[column]
    scale = (oriented.max(axis=0) - oriented.min(axis=0)).replace(0, 1)
    return (oriented - oriented.min(axis=0)) / scale


def _ranking_scores(route_id: str, role: str, normalized: pd.DataFrame) -> list[float]:
    if role == "baseline":
        scores = normalized.mean(axis=1)
    elif route_id == "entropy_weight_ranking":
        shifted = normalized + 1e-9
        probability = shifted / shifted.sum(axis=0).replace(0, 1)
        entropy = -(probability * np.log(probability + 1e-12)).sum(axis=0) / np.log(max(len(probability), 2))
        weights = np.asarray((1 - entropy) / max(float((1 - entropy).sum()), 1e-12))
        scores = shifted.dot(weights)
    elif route_id == "topsis_ranking":
        norm = normalized / np.sqrt((normalized ** 2).sum(axis=0)).replace(0, 1)
        positive, negative = norm.max(axis=0), norm.min(axis=0)
        positive_distance = np.sqrt(((norm - positive) ** 2).sum(axis=1))
        negative_distance = np.sqrt(((norm - negative) ** 2).sum(axis=1))
        scores = negative_distance / (positive_distance + negative_distance + 1e-12)
    elif route_id == "pca_factor_ranking":
        centered = normalized - normalized.mean(axis=0)
        _, _, vectors = np.linalg.svd(centered.to_numpy(dtype=float), full_matrices=False)
        weights = vectors[0]
        if weights.sum() < 0:
            weights = -weights
        scores = centered.to_numpy(dtype=float).dot(weights)
    elif route_id == "stability_weighted_ranking":
        std = normalized.std(axis=0).replace(0, np.nan)
        weights = np.asarray((1 / std).replace([np.inf, -np.inf], np.nan).fillna(0))
        if float(weights.sum()) <= 1e-12:
            weights = np.ones(normalized.shape[1])
        weights = weights / weights.sum()
        scores = normalized.dot(weights)
    else:
        raise ValueError(f"unsupported_ranking_route_for_recompute:{route_id}")
    return [float(value) for value in np.asarray(scores)]


def _spearman(left, right) -> float:
    def ranks(values):
        order = sorted(range(len(values)), key=lambda index: values[index])
        result = [0.0] * len(values)
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and values[order[end]] == values[order[start]]:
                end += 1
            rank = (start + end - 1) / 2.0
            for position in order[start:end]:
                result[position] = rank
            start = end
        return result
    x, y = ranks(left), ranks(right)
    x_mean, y_mean = sum(x) / len(x), sum(y) / len(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a - x_mean) ** 2 for a in x) * sum((b - y_mean) ** 2 for b in y))
    return numerator / denominator if denominator else 0.0


def _with_provenance(record: ExperimentRecord, run_id: str, command: List[str], exit_code: int | None, stdout_path: Path, stderr_path: Path) -> ExperimentRecord:
    record.run_id = run_id
    record.command = command
    record.exit_code = exit_code
    record.stdout_path = str(stdout_path)
    record.stderr_path = str(stderr_path)
    for name, path in (("stdout", stdout_path), ("stderr", stderr_path)):
        if path.is_file():
            record.evidence_paths.setdefault(name, str(path))
            record.artifact_hashes.setdefault(name, _sha256(path))
    return record


def _timeout_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _repairable(error: str) -> bool:
    lowered = (error or "").lower()
    return "read_excel" in lowered or "excel" in lowered


def _append_repair_guard(adapter: Path) -> None:
    text = adapter.read_text(encoding="utf-8")
    if "# V40_REPAIR_GUARD" not in text:
        text = text.replace("pd.read_excel(path)", "pd.read_excel(path, engine=None)")
        text += "\n# V40_REPAIR_GUARD: limited safe repair applied; hashes are recorded in registry.\n"
        adapter.write_text(text, encoding="utf-8")
