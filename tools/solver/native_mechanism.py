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
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from .schemas import ExperimentRecord, ProblemContract, RouteSpec


def _read_data(path: Path, sheet: str | None) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet or 0)
    for encoding in ("utf-8", "gb18030", "gbk", "utf-16"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=encoding)
        except Exception:
            continue
    raise ValueError("mechanism_data_read_failed")


def _rhs(kind: str, params: Dict[str, float]):
    def rhs(t: float, state: np.ndarray) -> np.ndarray:
        if kind == "first_order_relaxation":
            return np.asarray([params["k"] * (params["equilibrium"] - state[0])])
        if kind == "logistic_growth":
            return np.asarray([params["r"] * state[0] * (1.0 - state[0] / params["K"])])
        s, i, r = state
        return np.asarray([-params["beta"] * s * i, params["beta"] * s * i - params["gamma"] * i, params["gamma"] * i])
    return rhs


def _simulate(model: Dict[str, Any], times: np.ndarray, params: Dict[str, float]) -> tuple[np.ndarray, Dict[str, Any]]:
    kind = model["kind"]
    columns = model["state_columns"]
    initial = np.asarray([float(model["initial_state"][name]) for name in columns])
    shifted = np.asarray(times, dtype=float) - float(times[0])
    if np.any(np.diff(shifted) < 0):
        raise ValueError("mechanism_time_must_be_sorted")
    result = solve_ivp(_rhs(kind, params), (float(shifted[0]), float(shifted[-1])), initial, t_eval=shifted, rtol=float(model.get("rtol", 1e-8)), atol=float(model.get("atol", 1e-10)), method=str(model.get("method", "RK45")))
    if not result.success or result.y.shape[1] != len(times):
        raise ValueError(f"mechanism_integration_failed:{result.message}")
    return result.y.T, {"success": bool(result.success), "message": str(result.message), "nfev": int(result.nfev)}


def _metric(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    residual = actual - predicted
    return {"MAE": float(np.mean(np.abs(residual))), "RMSE": float(np.sqrt(np.mean(residual ** 2))), "WAPE": float(np.sum(np.abs(residual)) / max(np.sum(np.abs(actual)), 1e-12))}


def _fit(model: Dict[str, Any], times: np.ndarray, observed: np.ndarray, seed: int = 42) -> tuple[Dict[str, float], Dict[str, Any]]:
    names = list(model["parameters"])
    lower = np.asarray([float(model["parameters"][name]["lower"]) for name in names])
    upper = np.asarray([float(model["parameters"][name]["upper"]) for name in names])
    rng = np.random.default_rng(seed)
    starts = [lower + (upper - lower) * rng.random(len(names)) for _ in range(int(model.get("multistart", 8)))]
    starts.append((lower + upper) / 2.0)
    fits = []
    for start in starts:
        def residual(values: np.ndarray) -> np.ndarray:
            params = dict(zip(names, values))
            try:
                prediction, _ = _simulate(model, times, params)
                return (prediction - observed).ravel()
            except Exception:
                return np.full(observed.size, 1e12)
        fit = least_squares(residual, start, bounds=(lower, upper), max_nfev=int(model.get("max_nfev", 2000)), xtol=1e-12, ftol=1e-12, gtol=1e-12)
        fits.append((float(np.sum(fit.fun ** 2)), fit))
    fits.sort(key=lambda item: item[0])
    objective, best = fits[0]
    params = {name: float(value) for name, value in zip(names, best.x)}
    spread = float(np.std([item[0] for item in fits]))
    at_bound = [name for name, value, lo, hi in zip(names, best.x, lower, upper) if abs(value - lo) <= 1e-7 * max(1.0, abs(lo)) or abs(value - hi) <= 1e-7 * max(1.0, abs(hi))]
    return params, {"multistart_count": len(fits), "best_sse": objective, "objective_spread": spread, "parameter_at_bound": at_bound, "converged": bool(best.success), "optimizer_message": str(best.message)}


def _sensitivity(model: Dict[str, Any], times: np.ndarray, params: Dict[str, float]) -> Dict[str, Any]:
    names = list(params)
    base, _ = _simulate(model, times, params)
    columns = []
    for name in names:
        step = 1e-5 * max(1.0, abs(params[name]))
        changed = dict(params)
        changed[name] += step
        perturbed, _ = _simulate(model, times, changed)
        columns.append(((perturbed - base) / step).ravel())
    jac = np.column_stack(columns) if columns else np.empty((base.size, 0))
    rank = int(np.linalg.matrix_rank(jac)) if jac.size else 0
    condition = float(np.linalg.cond(jac)) if jac.size and min(jac.shape) else math.inf
    status = "locally_identifiable" if rank == len(names) and condition < 1e8 else ("weakly_identifiable" if rank == len(names) else "not_identifiable")
    return {"parameter_names": names, "jacobian_shape": list(jac.shape), "rank": rank, "condition_number": condition, "status": status}


def solve_mechanism_model(contract: Dict[str, Any], route: Dict[str, Any]) -> Dict[str, Any]:
    model = contract["mechanism_model"]
    frame = _read_data(Path(contract["data_file"]), contract.get("sheet_name"))
    time_column = str(model.get("time_column") or contract.get("time_column"))
    states = list(model["state_columns"])
    frame = frame[[time_column] + states].apply(pd.to_numeric, errors="coerce").dropna().sort_values(time_column).reset_index()
    if len(frame) < 9:
        raise ValueError("mechanism_requires_at_least_9_complete_rows")
    times = frame[time_column].to_numpy(dtype=float)
    observed = frame[states].to_numpy(dtype=float)
    n = len(frame); train_end = max(3, int(n * 0.6)); selection_end = max(train_end + 2, int(n * 0.8))
    if selection_end >= n:
        selection_end = n - 1
    if route["route_id"] == "mechanism_constant_baseline":
        params = {}
        train_pred = np.repeat(observed[0:1], train_end, axis=0)
        selection_pred = np.repeat(observed[0:1], selection_end - train_end, axis=0)
        final_pred = np.repeat(observed[0:1], n - selection_end, axis=0)
        fit_info = {"baseline": "initial_state_constant"}
        sensitivity = {"status": "not_applicable"}
    else:
        params, fit_info = _fit(model, times[:train_end], observed[:train_end], int(model.get("seed", 42)))
        # Integrate once from the declared initial condition; later splits are
        # slices of the same trajectory, never reinitialized forecasts.
        full_pred, integration = _simulate(model, times, params)
        train_pred = full_pred[:train_end]
        selection_pred = full_pred[train_end:selection_end]
        final_pred = full_pred[selection_end:]
        fit_info["integration"] = integration
        sensitivity = _sensitivity(model, times[:train_end], params)
    metrics = {"MAE": _metric(observed[train_end:selection_end], selection_pred)["MAE"], "RMSE": _metric(observed[train_end:selection_end], selection_pred)["RMSE"], "WAPE": _metric(observed[train_end:selection_end], selection_pred)["WAPE"], "validation_split": "train_selection_final_test_ode", "selection_size": int(selection_end - train_end), "test_size": int(n - selection_end), "rows_used": int(n), "state_columns": states, "model_kind": model["kind"], "identifiability_status": sensitivity.get("status", "not_applicable")}
    metrics.update({f"final_test_{key}": value for key, value in _metric(observed[selection_end:], final_pred).items()})
    evidence = {"times": times.tolist(), "observed": observed.tolist(), "predicted_train": train_pred.tolist(), "predicted_selection": selection_pred.tolist(), "predicted_final_test": final_pred.tolist(), "train_rows": frame["index"].iloc[:train_end].tolist(), "selection_rows": frame["index"].iloc[train_end:selection_end].tolist(), "final_test_rows": frame["index"].iloc[selection_end:].tolist(), "parameters": params, "parameter_bounds": model["parameters"], "initial_state": model["initial_state"], "fit_info": fit_info, "sensitivity": sensitivity, "solver_tolerances": {"rtol": model.get("rtol", 1e-8), "atol": model.get("atol", 1e-10)}}
    return {"metrics": metrics, "evidence": evidence}


def run_native_mechanism_experiments(contract: ProblemContract, routes: Iterable[RouteSpec], output_dir: Path) -> list[ExperimentRecord]:
    output_dir.mkdir(parents=True, exist_ok=True); evidence_dir = output_dir / "evidence"; evidence_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex; code_path = Path(__file__).resolve(); input_path = Path(contract.data_file).resolve(); records = []
    for route in routes:
        try:
            solved = solve_mechanism_model(asdict(contract), route.__dict__)
            evidence_path = evidence_dir / f"{route.route_id}.json"
            payload = {"run_id": run_id, "problem_id": contract.problem_id, "route_id": route.route_id, "data_path": str(input_path), "contract": asdict(contract), "route": route.__dict__, "environment": {"scipy": __import__("scipy").__version__}, "metrics_recomputed": solved["metrics"], "evidence": solved["evidence"]}
            evidence_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            config_hash = hashlib.sha256(json.dumps({"contract": asdict(contract), "route": route.__dict__}, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
            record = ExperimentRecord(contract.problem_id, route.route_id, route.role, "executed", str(code_path), str(input_path), solved["metrics"], original_code_hash=_sha256(code_path), final_code_hash=_sha256(code_path), run_id=run_id, input_hash=_sha256(input_path), config_hash=config_hash, artifact_hashes={"route_evidence": _sha256(evidence_path)}, evidence_paths={"route_evidence": str(evidence_path)}, environment={"scipy": __import__("scipy").__version__})
        except Exception as exc:
            record = ExperimentRecord(contract.problem_id, route.route_id, route.role, "failed", str(code_path), str(input_path), {}, str(exc), original_code_hash=_sha256(code_path), final_code_hash=_sha256(code_path), run_id=run_id, input_hash=_sha256(input_path))
        records.append(record); (output_dir / f"{route.route_id}.json").write_text(json.dumps(record.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
    return records


def verify_native_mechanism(contract: ProblemContract, records: Iterable[ExperimentRecord], registry_path: Path, output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True); issues = []; checks = []
    registry_path = Path(registry_path)
    if not registry_path.is_file():
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text(json.dumps([record.__dict__ for record in records], ensure_ascii=False), encoding="utf-8")
    for record in records:
        if record.status != "executed" or record.role in {"ablation", "diagnostic"}: continue
        try:
            payload = json.loads(Path(record.evidence_paths["route_evidence"]).read_text(encoding="utf-8")); model = payload["contract"]["mechanism_model"]; evidence = payload["evidence"]
            times = np.asarray(evidence["times"], dtype=float); observed = np.asarray(evidence["observed"], dtype=float); params = {key: float(value) for key, value in evidence["parameters"].items()}
            if params:
                predicted, integration = _simulate(model, times, params)
            else:
                predicted = np.repeat(observed[0:1], len(times), axis=0); integration = {"success": True}
            stored = np.asarray(evidence["predicted_selection"] + evidence["predicted_final_test"], dtype=float)
            start = len(evidence["predicted_train"]); recomputed = predicted[start:]
            reproducible = stored.shape == recomputed.shape and np.allclose(stored, recomputed, rtol=1e-7, atol=1e-7)
            bounds_ok = all(float(spec["lower"]) - 1e-8 <= value <= float(spec["upper"]) + 1e-8 for name, value in params.items() for spec in [model["parameters"][name]])
            finite_ok = bool(np.isfinite(predicted).all())
            conservation_ok = True if model["kind"] != "sir" else bool(np.allclose(predicted.sum(axis=1), predicted[0].sum(), atol=1e-6))
            selection_end = start + len(evidence["predicted_selection"]); selection_metrics = _metric(observed[start:selection_end], predicted[start:selection_end]); final_metrics = _metric(observed[selection_end:], predicted[selection_end:])
            metrics_ok = all(math.isclose(float(record.metrics[f"final_test_{name}"]), final_metrics[name], rel_tol=1e-7, abs_tol=1e-7) for name in final_metrics)
            item = {"route_id": record.route_id, "reproducible": reproducible, "bounds_ok": bounds_ok, "finite_ok": finite_ok, "conservation_ok": conservation_ok, "metrics_ok": metrics_ok, "integration_success": integration.get("success", False), "identifiability": evidence.get("sensitivity", {})}
            checks.append(item)
            if not all(item[key] for key in ("reproducible", "bounds_ok", "finite_ok", "conservation_ok", "metrics_ok", "integration_success")): issues.append(f"native_mechanism_verification_failed:{record.route_id}")
        except Exception as exc: issues.append(f"native_mechanism_verification_error:{record.route_id}:{exc}")
    stdout = output_dir / "native_mechanism_verifier.stdout.txt"; stderr = output_dir / "native_mechanism_verifier.stderr.txt"; stdout.write_text(json.dumps({"checks": checks}, ensure_ascii=False), encoding="utf-8"); stderr.write_text("", encoding="utf-8")
    run_ids = {record.run_id for record in records if record.run_id}; result = {"kind": "native_mechanism", "passed": not issues and bool(checks), "issues": issues, "checks": checks, "claim_level": "domain_verified_ode_fit" if not issues and any(item["route_id"] == "declared_ode_multistart" for item in checks) else "domain_verified_baseline", "run_id": next(iter(run_ids)) if len(run_ids) == 1 else None, "contract_hash": _json_hash(asdict(contract)), "script_path": str(Path(__file__).resolve()), "script_hash": _sha256(Path(__file__).resolve()), "registry_path": str(registry_path.resolve()), "registry_hash": _sha256(registry_path), "stdout_path": str(stdout), "stdout_hash": _sha256(stdout), "stderr_path": str(stderr), "stderr_hash": _sha256(stderr), "exit_code": 0 if not issues else 1}
    (output_dir / "domain_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"); return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
