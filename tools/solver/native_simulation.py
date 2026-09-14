from __future__ import annotations

import hashlib
import heapq
import json
import math
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np

from .schemas import ExperimentRecord, ProblemContract, RouteSpec

SIMULATION_METRICS = ("mean_wait", "utilization", "throughput", "completed", "arrivals", "accepted", "rejected", "rejection_rate")


def _queue_parameters(model: Dict[str, Any]) -> tuple[int, int | None]:
    if model["kind"] == "mm1_queue":
        return 1, None
    return int(model["servers"]), int(model["capacity"]) if model.get("capacity") is not None else None


def _sample_duration(rng: np.random.Generator, rate: float, distribution: Dict[str, Any]) -> float:
    kind = distribution.get("kind", "exponential")
    if kind == "exponential":
        return float(rng.exponential(1.0 / rate))
    if kind == "deterministic":
        return 1.0 / rate
    values = np.asarray(distribution["values"], dtype=float)
    weights = distribution.get("weights")
    probabilities = None if weights is None else np.asarray(weights, dtype=float) / float(np.sum(weights))
    return float(rng.choice(values, p=probabilities))


def _simulate_queue(model: Dict[str, Any], seed: int, force_deterministic: bool = False) -> Dict[str, float]:
    lam = float(model["arrival_rate"]); mu = float(model["service_rate"])
    horizon = float(model["horizon"]); warmup = float(model["warmup"])
    servers, capacity = _queue_parameters(model)
    arrival_distribution = {"kind": "deterministic"} if force_deterministic else model.get("arrival_distribution", {"kind": "exponential"})
    service_distribution = {"kind": "deterministic"} if force_deterministic else model.get("service_distribution", {"kind": "exponential"})
    rng = np.random.default_rng(int(seed))
    server_free = [0.0] * servers; heapq.heapify(server_free)
    departures: list[float] = []
    jobs: list[tuple[float, float, float]] = []
    arrival = 0.0; measured_arrivals = 0; measured_accepted = 0; measured_rejected = 0

    while True:
        arrival += _sample_duration(rng, lam, arrival_distribution)
        if arrival > horizon:
            break
        while departures and departures[0] <= arrival:
            heapq.heappop(departures)
        measured = arrival >= warmup
        if measured:
            measured_arrivals += 1
        if capacity is not None and len(departures) >= capacity:
            if measured:
                measured_rejected += 1
            continue
        service = _sample_duration(rng, mu, service_distribution)
        available = heapq.heappop(server_free)
        start = max(arrival, available); finish = start + service
        heapq.heappush(server_free, finish); heapq.heappush(departures, finish)
        jobs.append((arrival, start, finish))
        if measured:
            measured_accepted += 1

    window = horizon - warmup
    measured_jobs = [job for job in jobs if warmup <= job[0] <= horizon]
    waits = [start - arrived for arrived, start, _ in measured_jobs]
    completed = sum(1 for _, _, finish in jobs if warmup < finish <= horizon)
    busy_time = sum(max(0.0, min(finish, horizon) - max(start, warmup)) for _, start, finish in jobs)
    return {
        "mean_wait": float(np.mean(waits)) if waits else 0.0,
        "utilization": float(busy_time / (window * servers)),
        "throughput": float(completed / window),
        "completed": float(completed), "arrivals": float(measured_arrivals),
        "accepted": float(measured_accepted), "rejected": float(measured_rejected),
        "rejection_rate": float(measured_rejected / measured_arrivals) if measured_arrivals else 0.0,
    }


def _simulate_mm1(model: Dict[str, Any], seed: int) -> Dict[str, float]:
    return _simulate_queue(model, seed)


def _summary(values: np.ndarray) -> Dict[str, float]:
    mean = float(np.mean(values)); std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return {"mean": mean, "std": std, "ci_half_width": float(1.96 * std / math.sqrt(max(len(values), 1))), "replications": int(len(values))}


def _analytical_queue(model: Dict[str, Any]) -> Dict[str, float]:
    lam = float(model["arrival_rate"]); mu = float(model["service_rate"]); servers, capacity = _queue_parameters(model)
    offered = lam / mu
    if capacity is not None:
        weights = [1.0]
        for n in range(1, capacity + 1):
            weights.append(weights[-1] * lam / (min(n, servers) * mu))
        probabilities = np.asarray(weights, dtype=float) / sum(weights)
        blocking = float(probabilities[-1]); throughput = lam * (1.0 - blocking)
        expected_queue = float(sum(max(0, n - servers) * probabilities[n] for n in range(capacity + 1)))
        return {"mean_wait": expected_queue / throughput if throughput > 0 else math.inf, "utilization": throughput / (servers * mu), "throughput": throughput, "rejection_rate": blocking}
    rho = offered / servers
    if rho >= 1.0:
        return {"mean_wait": math.inf, "utilization": rho, "throughput": lam, "rejection_rate": 0.0}
    normalizer = sum(offered ** n / math.factorial(n) for n in range(servers)) + offered ** servers / (math.factorial(servers) * (1.0 - rho))
    p0 = 1.0 / normalizer
    erlang_c = offered ** servers * p0 / (math.factorial(servers) * (1.0 - rho))
    return {"mean_wait": erlang_c / (servers * mu - lam), "utilization": rho, "throughput": lam, "rejection_rate": 0.0}


def _analytical_mm1(model: Dict[str, Any]) -> Dict[str, float]:
    return _analytical_queue(model)


def _aggregate_replications(replications_data: list[Dict[str, float]]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    summaries = {name: _summary(np.asarray([item[name] for item in replications_data], dtype=float)) for name in SIMULATION_METRICS}
    metrics: Dict[str, Any] = {name: summary["mean"] for name, summary in summaries.items()}
    metrics.update({"ci_half_width": summaries["mean_wait"]["ci_half_width"], "wait_std": summaries["mean_wait"]["std"], "utilization_ci_half_width": summaries["utilization"]["ci_half_width"], "rejection_rate_ci_half_width": summaries["rejection_rate"]["ci_half_width"], "replications": len(replications_data)})
    return metrics, summaries


def solve_simulation_model(contract: Dict[str, Any], route: Dict[str, Any]) -> Dict[str, Any]:
    model = contract["simulation_model"]; seed = int(model.get("seed", 42)); replications = int(model.get("replications", 20))
    servers, capacity = _queue_parameters(model)
    if route["route_id"] == "simulation_analytical_baseline":
        values = _analytical_queue(model); queue_name = f"M/M/{servers}" + (f"/{capacity}" if capacity is not None else "")
        metrics = {**values, "ci_half_width": 0.0, "replications": 0, "validation_split": "analytical_queue", "method": f"{queue_name} closed form"}
        return {"metrics": metrics, "evidence": {"analytical": values, "replication_seeds": [], "replication_metrics": []}}
    deterministic_baseline = route["route_id"] == "simulation_mean_deterministic_baseline"
    seeds = [seed + index for index in range(replications)]
    replications_data = [_simulate_queue(model, item, force_deterministic=deterministic_baseline) for item in seeds]
    metrics, summaries = _aggregate_replications(replications_data)
    metrics.update({"warmup": float(model["warmup"]), "horizon": float(model["horizon"]), "servers": servers, "capacity": capacity, "validation_split": "independent_replications", "method": "mean_deterministic_queue" if deterministic_baseline else "event_driven_queue"})
    exponential = all(model.get(name, {"kind": "exponential"}).get("kind", "exponential") == "exponential" for name in ("arrival_distribution", "service_distribution"))
    evidence = {"replication_seeds": seeds, "replication_metrics": replications_data, "metric_summaries": summaries, "wait_summary": summaries["mean_wait"], "utilization_summary": summaries["utilization"], "analytical_reference": _analytical_queue(model) if exponential else None, "force_deterministic": deterministic_baseline, "common_random_numbers": False, "warmup_diagnostics": {"warmup": float(model["warmup"]), "window": float(model["horizon"] - model["warmup"])}}
    return {"metrics": metrics, "evidence": evidence}


def run_native_simulation_experiments(contract: ProblemContract, routes: Iterable[RouteSpec], output_dir: Path) -> list[ExperimentRecord]:
    output_dir.mkdir(parents=True, exist_ok=True); evidence_dir = output_dir / "evidence"; evidence_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex; code_path = Path(__file__).resolve(); input_path = Path(contract.data_file).resolve(); records = []
    for route in routes:
        try:
            solved = solve_simulation_model(asdict(contract), route.__dict__); evidence_path = evidence_dir / f"{route.route_id}.json"
            payload = {"run_id": run_id, "problem_id": contract.problem_id, "route_id": route.route_id, "data_path": str(input_path), "contract": asdict(contract), "route": route.__dict__, "environment": {"numpy": np.__version__}, "metrics_recomputed": solved["metrics"], "evidence": solved["evidence"]}
            evidence_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            record = ExperimentRecord(contract.problem_id, route.route_id, route.role, "executed", str(code_path), str(input_path), solved["metrics"], original_code_hash=_sha256(code_path), final_code_hash=_sha256(code_path), run_id=run_id, input_hash=_sha256(input_path), config_hash=_json_hash({"contract": asdict(contract), "route": route.__dict__}), artifact_hashes={"route_evidence": _sha256(evidence_path)}, evidence_paths={"route_evidence": str(evidence_path)}, environment={"numpy": np.__version__})
        except Exception as exc:
            record = ExperimentRecord(contract.problem_id, route.route_id, route.role, "failed", str(code_path), str(input_path), {}, str(exc), original_code_hash=_sha256(code_path), final_code_hash=_sha256(code_path), run_id=run_id, input_hash=_sha256(input_path))
        records.append(record); (output_dir / f"{route.route_id}.json").write_text(json.dumps(record.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
    return records


def verify_native_simulation(contract: ProblemContract, records: Iterable[ExperimentRecord], registry_path: Path, output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True); records = list(records); issues = []; checks = []; registry_path = Path(registry_path)
    if not registry_path.is_file():
        registry_path.parent.mkdir(parents=True, exist_ok=True); registry_path.write_text(json.dumps([record.__dict__ for record in records], ensure_ascii=False), encoding="utf-8")
    declared_model = asdict(contract)["simulation_model"]
    for record in records:
        if record.status != "executed" or record.role in {"ablation", "diagnostic"}: continue
        try:
            payload = json.loads(Path(record.evidence_paths["route_evidence"]).read_text(encoding="utf-8")); model = payload["contract"]["simulation_model"]; evidence = payload["evidence"]; model_matches = model == declared_model
            if record.route_id == "simulation_analytical_baseline":
                expected = _analytical_queue(declared_model); actual = evidence["analytical"]
                reproducible = set(actual) == set(expected) and all(math.isclose(float(actual[name]), value, rel_tol=1e-12, abs_tol=1e-12) for name, value in expected.items())
                metrics_ok = all(math.isclose(float(record.metrics[name]), value, rel_tol=1e-12, abs_tol=1e-12) for name, value in expected.items())
                item = {"route_id": record.route_id, "model_matches": model_matches, "reproducible": reproducible, "metrics_ok": metrics_ok, "ci_valid": True}
            else:
                expected_seeds = [int(declared_model.get("seed", 42)) + index for index in range(int(declared_model.get("replications", 20)))]; seeds = [int(value) for value in evidence["replication_seeds"]]
                force_deterministic = record.route_id == "simulation_mean_deterministic_baseline"
                rerun = [_simulate_queue(declared_model, item, force_deterministic=force_deterministic) for item in expected_seeds]; stored = evidence["replication_metrics"]
                reproducible = len(rerun) == len(stored) and all(all(math.isclose(float(a[name]), float(b[name]), rel_tol=1e-12, abs_tol=1e-12) for name in SIMULATION_METRICS) for a, b in zip(rerun, stored))
                expected_metrics, expected_summaries = _aggregate_replications(rerun); metric_names = list(SIMULATION_METRICS) + ["ci_half_width", "wait_std", "utilization_ci_half_width", "rejection_rate_ci_half_width"]
                metrics_ok = all(math.isclose(float(record.metrics[name]), float(expected_metrics[name]), rel_tol=1e-12, abs_tol=1e-12) for name in metric_names)
                item = {"route_id": record.route_id, "model_matches": model_matches, "seeds_ok": seeds == expected_seeds, "deterministic_mode_ok": evidence.get("force_deterministic") is force_deterministic, "reproducible": reproducible, "metrics_ok": metrics_ok, "summaries_ok": evidence.get("metric_summaries") == expected_summaries, "ci_valid": expected_metrics["ci_half_width"] >= 0, "replications": len(seeds)}
            checks.append(item); required = ("model_matches", "reproducible", "metrics_ok", "ci_valid")
            if not all(item.get(key, False) for key in required) or item.get("seeds_ok") is False or item.get("summaries_ok") is False or item.get("deterministic_mode_ok") is False:
                issues.append(f"native_simulation_verification_failed:{record.route_id}")
        except Exception as exc:
            issues.append(f"native_simulation_verification_error:{record.route_id}:{exc}")
    stdout = output_dir / "native_simulation_verifier.stdout.txt"; stderr = output_dir / "native_simulation_verifier.stderr.txt"
    stdout.write_text(json.dumps({"checks": checks}, ensure_ascii=False), encoding="utf-8"); stderr.write_text("", encoding="utf-8")
    run_ids = {record.run_id for record in records if record.run_id}
    result = {"kind": "native_simulation", "passed": not issues and bool(checks), "issues": issues, "checks": checks, "claim_level": "domain_verified_queue_monte_carlo" if not issues and any(item["route_id"] != "simulation_analytical_baseline" for item in checks) else "domain_verified_analytical_baseline", "run_id": next(iter(run_ids)) if len(run_ids) == 1 else None, "contract_hash": _json_hash(asdict(contract)), "script_path": str(Path(__file__).resolve()), "script_hash": _sha256(Path(__file__).resolve()), "registry_path": str(registry_path.resolve()), "registry_hash": _sha256(registry_path), "stdout_path": str(stdout), "stdout_hash": _sha256(stdout), "stderr_path": str(stderr), "stderr_hash": _sha256(stderr), "exit_code": 0 if not issues else 1}
    (output_dir / "domain_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"); return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
