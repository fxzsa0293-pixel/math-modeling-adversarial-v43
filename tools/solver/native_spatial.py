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

from .schemas import ExperimentRecord, ProblemContract, RouteSpec


def _read_data(path: Path, sheet: str | None) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet or 0)
    for encoding in ("utf-8", "gb18030", "gbk", "utf-16"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=encoding)
        except Exception:
            continue
    raise ValueError("spatial_data_read_failed")


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    residual = actual - predicted
    return {"MAE": float(np.mean(np.abs(residual))), "RMSE": float(np.sqrt(np.mean(residual ** 2)))}


def _blocks(x: np.ndarray, y: np.ndarray, block_size: float) -> np.ndarray:
    return np.column_stack((np.floor((x - np.min(x)) / block_size).astype(int), np.floor((y - np.min(y)) / block_size).astype(int)))


def _block_keys(blocks: np.ndarray) -> np.ndarray:
    return np.asarray([f"{int(row[0])}:{int(row[1])}" for row in blocks], dtype=object)


def _split_blocks(unique: list[str], seed: int) -> tuple[set[str], set[str], set[str]]:
    shuffled = list(unique)
    np.random.default_rng(seed).shuffle(shuffled)
    train_end = max(1, int(len(shuffled) * 0.6))
    selection_end = max(train_end + 1, int(len(shuffled) * 0.8))
    if selection_end >= len(shuffled):
        selection_end = len(shuffled) - 1
    return (set(shuffled[:train_end]), set(shuffled[train_end:selection_end]), set(shuffled[selection_end:]))


def _distance(origin: np.ndarray, points: np.ndarray, coordinate_system: str) -> np.ndarray:
    if coordinate_system == "projected":
        return np.sqrt(np.sum((points - origin) ** 2, axis=1))
    lon1 = np.radians(float(origin[0])); lat1 = np.radians(float(origin[1]))
    lon2 = np.radians(points[:, 0]); lat2 = np.radians(points[:, 1])
    dlon = lon2 - lon1; dlat = lat2 - lat1
    value = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(value, 0.0, 1.0)))


def _block_coordinates(xy: np.ndarray, coordinate_system: str) -> np.ndarray:
    if coordinate_system == "projected":
        return xy
    lon0 = float(np.mean(xy[:, 0])); lat0 = float(np.mean(xy[:, 1]))
    x_km = 6371.0088 * np.radians(xy[:, 0] - lon0) * math.cos(math.radians(lat0))
    y_km = 6371.0088 * np.radians(xy[:, 1] - lat0)
    return np.column_stack((x_km, y_km))


def _predict_idw(train_xy: np.ndarray, train_y: np.ndarray, query_xy: np.ndarray, power: float, coordinate_system: str = "projected") -> np.ndarray:
    predictions = []
    for point in query_xy:
        distances = _distance(point, train_xy, coordinate_system)
        exact = np.flatnonzero(distances <= 1e-12)
        if len(exact):
            predictions.append(float(np.mean(train_y[exact])))
            continue
        weights = 1.0 / np.maximum(distances, 1e-12) ** power
        predictions.append(float(np.sum(weights * train_y) / np.sum(weights)))
    return np.asarray(predictions, dtype=float)


def _morans_i(coordinates: np.ndarray, residuals: np.ndarray, coordinate_system: str, neighbors: int) -> float:
    count = len(residuals)
    if count < 3 or float(np.sum((residuals - np.mean(residuals)) ** 2)) <= 1e-20:
        return 0.0
    weights = np.zeros((count, count), dtype=float)
    for index, point in enumerate(coordinates):
        distances = _distance(point, coordinates, coordinate_system)
        order = [item for item in np.argsort(distances) if item != index][:min(neighbors, count - 1)]
        for other in order:
            weights[index, other] = 1.0 / max(float(distances[other]), 1e-12)
    row_sums = weights.sum(axis=1)
    valid = row_sums > 0
    weights[valid] /= row_sums[valid, None]
    centered = residuals - np.mean(residuals)
    denominator = float(np.dot(centered, centered)); total_weight = float(weights.sum())
    return float(count / total_weight * np.sum(weights * np.outer(centered, centered)) / denominator) if total_weight > 0 else 0.0


def solve_spatial_model(contract: Dict[str, Any], route: Dict[str, Any]) -> Dict[str, Any]:
    model = contract["spatial_model"]; frame = _read_data(Path(contract["data_file"]), contract.get("sheet_name"))
    columns = [model["x_column"], model["y_column"], model["target_column"]]
    frame = frame[columns].apply(pd.to_numeric, errors="coerce").dropna().reset_index()
    if len(frame) < 12:
        raise ValueError("spatial_requires_at_least_12_complete_rows")
    x = frame[model["x_column"]].to_numpy(float); y = frame[model["y_column"]].to_numpy(float); target = frame[model["target_column"]].to_numpy(float)
    coordinate_system = str(model.get("coordinate_system", "projected")); coordinates = np.column_stack((x, y))
    if coordinate_system == "geographic_wgs84" and (np.any(x < -180) or np.any(x > 180) or np.any(y < -90) or np.any(y > 90)):
        raise ValueError("spatial_geographic_coordinates_out_of_range")
    block_xy = _block_coordinates(coordinates, coordinate_system)
    keys = _block_keys(_blocks(block_xy[:, 0], block_xy[:, 1], float(model.get("block_size", 1.0))))
    unique = sorted(set(keys.tolist()))
    if len(unique) < 3:
        raise ValueError("spatial_requires_at_least_3_spatial_blocks")
    n_blocks = len(unique); seed = int(model.get("seed", 0)); train_blocks, selection_blocks, final_blocks = _split_blocks(unique, seed)
    if not selection_blocks or not final_blocks:
        raise ValueError("spatial_block_split_failed")
    masks = [np.isin(keys, list(group)) for group in (train_blocks, selection_blocks, final_blocks)]
    train_xy = coordinates[masks[0]]; train_y = target[masks[0]]
    if route["route_id"] == "spatial_global_mean_baseline":
        selection_pred = np.repeat(float(np.mean(train_y)), int(masks[1].sum())); final_pred = np.repeat(float(np.mean(train_y)), int(masks[2].sum())); method = "global_mean"
    else:
        power = float(model.get("power", 2.0)); selection_pred = _predict_idw(train_xy, train_y, coordinates[masks[1]], power, coordinate_system); final_pred = _predict_idw(train_xy, train_y, coordinates[masks[2]], power, coordinate_system); method = "idw"
    selection_metrics = _metrics(target[masks[1]], selection_pred); final_metrics = _metrics(target[masks[2]], final_pred)
    moran_neighbors = int(model.get("moran_neighbors", 4)); final_moran = _morans_i(coordinates[masks[2]], target[masks[2]] - final_pred, coordinate_system, moran_neighbors)
    metrics = {**selection_metrics, **{f"final_test_{name}": value for name, value in final_metrics.items()}, "final_test_residual_morans_i": final_moran, "validation_split": "spatial_block_train_selection_final_test", "spatial_block_count": n_blocks, "train_rows": int(masks[0].sum()), "selection_size": int(masks[1].sum()), "test_size": int(masks[2].sum()), "method": method}
    evidence = {"coordinates": coordinates.tolist(), "block_coordinates": block_xy.tolist(), "observed": target.tolist(), "spatial_block_keys": keys.tolist(), "train_rows": frame["index"].iloc[np.flatnonzero(masks[0])].tolist(), "selection_rows": frame["index"].iloc[np.flatnonzero(masks[1])].tolist(), "final_test_rows": frame["index"].iloc[np.flatnonzero(masks[2])].tolist(), "train_predicted": np.repeat(float(np.mean(train_y)), int(masks[0].sum())).tolist(), "selection_predicted": selection_pred.tolist(), "final_test_predicted": final_pred.tolist(), "train_block_keys": sorted(train_blocks), "selection_block_keys": sorted(selection_blocks), "final_test_block_keys": sorted(final_blocks), "split_seed": seed, "power": float(model.get("power", 2.0)), "block_size": float(model.get("block_size", 1.0)), "coordinate_system": coordinate_system, "distance_metric": "haversine_km" if coordinate_system == "geographic_wgs84" else "euclidean_in_declared_coordinate_units", "moran_neighbors": moran_neighbors}
    return {"metrics": metrics, "evidence": evidence}


def run_native_spatial_experiments(contract: ProblemContract, routes: Iterable[RouteSpec], output_dir: Path) -> list[ExperimentRecord]:
    output_dir.mkdir(parents=True, exist_ok=True); evidence_dir = output_dir / "evidence"; evidence_dir.mkdir(parents=True, exist_ok=True); run_id = uuid.uuid4().hex; code_path = Path(__file__).resolve(); input_path = Path(contract.data_file).resolve(); records = []
    for route in routes:
        try:
            solved = solve_spatial_model(asdict(contract), route.__dict__); evidence_path = evidence_dir / f"{route.route_id}.json"; payload = {"run_id": run_id, "problem_id": contract.problem_id, "route_id": route.route_id, "data_path": str(input_path), "contract": asdict(contract), "route": route.__dict__, "environment": {"numpy": np.__version__}, "metrics_recomputed": solved["metrics"], "evidence": solved["evidence"]}; evidence_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"); record = ExperimentRecord(contract.problem_id, route.route_id, route.role, "executed", str(code_path), str(input_path), solved["metrics"], original_code_hash=_sha256(code_path), final_code_hash=_sha256(code_path), run_id=run_id, input_hash=_sha256(input_path), config_hash=_json_hash({"contract": asdict(contract), "route": route.__dict__}), artifact_hashes={"route_evidence": _sha256(evidence_path)}, evidence_paths={"route_evidence": str(evidence_path)}, environment={"numpy": np.__version__})
        except Exception as exc:
            record = ExperimentRecord(contract.problem_id, route.route_id, route.role, "failed", str(code_path), str(input_path), {}, str(exc), original_code_hash=_sha256(code_path), final_code_hash=_sha256(code_path), run_id=run_id, input_hash=_sha256(input_path))
        records.append(record); (output_dir / f"{route.route_id}.json").write_text(json.dumps(record.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
    return records


def verify_native_spatial(contract: ProblemContract, records: Iterable[ExperimentRecord], registry_path: Path, output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True); records = list(records); issues = []; checks = []; registry_path = Path(registry_path)
    if not registry_path.is_file():
        registry_path.parent.mkdir(parents=True, exist_ok=True); registry_path.write_text(json.dumps([record.__dict__ for record in records], ensure_ascii=False), encoding="utf-8")
    for record in records:
        if record.status != "executed" or record.role in {"ablation", "diagnostic"}: continue
        try:
            payload = json.loads(Path(record.evidence_paths["route_evidence"]).read_text(encoding="utf-8")); model = payload["contract"]["spatial_model"]; evidence = payload["evidence"]; coordinates = np.asarray(evidence["coordinates"], float); observed = np.asarray(evidence["observed"], float); coordinate_system = str(model.get("coordinate_system", "projected")); block_xy = _block_coordinates(coordinates, coordinate_system); expected_keys = _block_keys(_blocks(block_xy[:, 0], block_xy[:, 1], float(model.get("block_size", 1.0)))); keys = np.asarray(evidence["spatial_block_keys"], object); keys_ok = np.array_equal(keys, expected_keys); expected_train, expected_selection, expected_final = _split_blocks(sorted(set(expected_keys.tolist())), int(model.get("seed", 0))); declared_train = set(evidence["train_block_keys"]); declared_selection = set(evidence["selection_block_keys"]); declared_final = set(evidence["final_test_block_keys"]); split_ok = (declared_train, declared_selection, declared_final) == (expected_train, expected_selection, expected_final); train = np.isin(expected_keys, list(expected_train)); selection = np.isin(expected_keys, list(expected_selection)); final = np.isin(expected_keys, list(expected_final)); train_y = observed[train]; train_xy = coordinates[train]
            if record.route_id == "spatial_global_mean_baseline":
                selection_pred = np.repeat(float(np.mean(train_y)), int(selection.sum())); final_pred = np.repeat(float(np.mean(train_y)), int(final.sum()))
            else:
                selection_pred = _predict_idw(train_xy, train_y, coordinates[selection], float(model.get("power", 2.0)), coordinate_system); final_pred = _predict_idw(train_xy, train_y, coordinates[final], float(model.get("power", 2.0)), coordinate_system)
            stored_selection = np.asarray(evidence["selection_predicted"], float); stored_final = np.asarray(evidence["final_test_predicted"], float); reproducible = np.allclose(stored_selection, selection_pred, rtol=1e-12, atol=1e-12) and np.allclose(stored_final, final_pred, rtol=1e-12, atol=1e-12); block_disjoint = not (expected_train & expected_selection or expected_train & expected_final or expected_selection & expected_final); final_metrics = _metrics(observed[final], final_pred); moran = _morans_i(coordinates[final], observed[final] - final_pred, coordinate_system, int(model.get("moran_neighbors", 4))); metrics_ok = all(math.isclose(float(record.metrics[f"final_test_{name}"]), value, rel_tol=1e-12, abs_tol=1e-12) for name, value in final_metrics.items()) and math.isclose(float(record.metrics["final_test_residual_morans_i"]), moran, rel_tol=1e-12, abs_tol=1e-12); distance_ok = evidence.get("distance_metric") == ("haversine_km" if coordinate_system == "geographic_wgs84" else "euclidean_in_declared_coordinate_units"); item = {"route_id": record.route_id, "reproducible": reproducible, "keys_recomputed": keys_ok, "split_ok": split_ok, "block_disjoint": block_disjoint, "distance_metric_ok": distance_ok, "metrics_ok": metrics_ok, "spatial_block_count": len(set(keys.tolist()))}; checks.append(item)
            if not reproducible or not keys_ok or not split_ok or not block_disjoint or not distance_ok or not metrics_ok: issues.append(f"native_spatial_verification_failed:{record.route_id}")
        except Exception as exc: issues.append(f"native_spatial_verification_error:{record.route_id}:{exc}")
    stdout = output_dir / "native_spatial_verifier.stdout.txt"; stderr = output_dir / "native_spatial_verifier.stderr.txt"; stdout.write_text(json.dumps({"checks": checks}, ensure_ascii=False), encoding="utf-8"); stderr.write_text("", encoding="utf-8"); run_ids = {record.run_id for record in records if record.run_id}; result = {"kind": "native_spatial", "passed": not issues and bool(checks), "issues": issues, "checks": checks, "claim_level": "domain_verified_spatial_idw" if not issues and any(item["route_id"] == "spatial_idw" for item in checks) else "domain_verified_spatial_baseline", "run_id": next(iter(run_ids)) if len(run_ids) == 1 else None, "contract_hash": _json_hash(asdict(contract)), "script_path": str(Path(__file__).resolve()), "script_hash": _sha256(Path(__file__).resolve()), "registry_path": str(registry_path.resolve()), "registry_hash": _sha256(registry_path), "stdout_path": str(stdout), "stdout_hash": _sha256(stdout), "stderr_path": str(stderr), "stderr_hash": _sha256(stderr), "exit_code": 0 if not issues else 1}; (output_dir / "domain_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"); return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
