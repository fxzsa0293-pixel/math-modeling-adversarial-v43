from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Mapping

from .schemas import ExperimentRecord

DEFAULT_METRIC_DIRECTIONS = {
    "MAE": "lower_better",
    "RMSE": "lower_better",
    "WAPE": "lower_better",
    "MAPE": "lower_better",
    "objective": "lower_better",
    "expected_profit": "higher_better",
    "constraint_violation": "lower_better",
    "runtime_seconds": "lower_better",
    "ranking_stability": "higher_better",
    "R2": "higher_better",
    "score": "higher_better",
}
LOWER_BETTER = {metric for metric, direction in DEFAULT_METRIC_DIRECTIONS.items() if direction == "lower_better"}


def write_registry(records: Iterable[ExperimentRecord], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = [record.__dict__ for record in records]
    path = output_dir / "experiment_registry.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md = output_dir / "experiment_registry.md"
    lines = ["# V44 Experiment Registry", ""]
    for record in payload:
        lines.append(f"## {record['route_id']}")
        lines.append(f"- status: {record['status']}")
        lines.append(f"- role: {record['role']}")
        lines.append(f"- data_path: {record['data_path']}")
        lines.append(f"- metrics: {record['metrics']}")
        lines.append(f"- original_code_hash: {record.get('original_code_hash', '')}")
        lines.append(f"- final_code_hash: {record.get('final_code_hash', '')}")
        lines.append(f"- run_id: {record.get('run_id', '')}")
        lines.append(f"- input_hash: {record.get('input_hash', '')}")
        lines.append(f"- config_hash: {record.get('config_hash', '')}")
        lines.append(f"- evidence_paths: {record.get('evidence_paths', {})}")
        lines.append(f"- environment: {record.get('environment', {})}")
        if record.get("repair_log"):
            lines.append(f"- repair_log: {record['repair_log']}")
        if record.get("error"):
            lines.append(f"- error: {record['error'][:500]}")
        lines.append("")
    md.write_text("\n".join(lines), encoding="utf-8")
    return path


def metric_lower_better(metric: str, directions: Mapping[str, str] | None = None) -> bool:
    direction = (directions or DEFAULT_METRIC_DIRECTIONS).get(metric, DEFAULT_METRIC_DIRECTIONS.get(metric, "higher_better"))
    return direction == "lower_better"


def best_baseline(records: List[ExperimentRecord], metric: str, directions: Mapping[str, str] | None = None) -> ExperimentRecord | None:
    baselines = [record for record in records if record.status == "executed" and record.role in {"baseline", "strong_baseline"} and isinstance(record.metrics.get(metric), (int, float))]
    if not baselines:
        return None
    reverse = not metric_lower_better(metric, directions)
    return sorted(baselines, key=lambda record: record.metrics[metric], reverse=reverse)[0]


def choose_primary_metric(records: List[ExperimentRecord], metric_candidates: Iterable[str] | None = None) -> str:
    candidates = list(metric_candidates or [])
    # Optimization quality is the objective; feasibility is a gate/secondary metric.
    if any(record.metrics.get("objective") is not None for record in records):
        candidates = ["objective", "constraint_violation"] + candidates
    candidates.extend(["MAE", "RMSE", "WAPE", "ranking_stability", "constraint_violation", "objective", "R2", "score"])
    for metric in dict.fromkeys(candidates):
        if any(isinstance(record.metrics.get(metric), (int, float)) for record in records):
            return metric
    return "score"
