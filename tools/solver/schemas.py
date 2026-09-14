from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DataAsset:
    path: str
    suffix: str
    size_bytes: int
    columns: List[str] = field(default_factory=list)
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    readable: bool = False
    error: str = ""
    quality_score: float = 0.0
    target_score: float = 0.0
    sheet_names: List[str] = field(default_factory=list)
    sheet_columns: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class ProblemContract:
    version: str
    problem_id: str
    task_type: str
    status: str
    data_file: str = ""
    sheet_name: Optional[str] = None
    data_sources: List[Dict[str, Any]] = field(default_factory=list)
    base_table: str = ""
    data_joins: List[Dict[str, Any]] = field(default_factory=list)
    data_aggregation: Dict[str, Any] = field(default_factory=dict)
    data_assembly_manifest: str = ""
    target_column: str = ""
    time_column: str = ""
    group_columns: List[str] = field(default_factory=list)
    feature_columns: List[str] = field(default_factory=list)
    known_future_columns: List[str] = field(default_factory=list)
    indicator_directions: Dict[str, str] = field(default_factory=dict)
    metric_directions: Dict[str, str] = field(default_factory=dict)
    units: Dict[str, str] = field(default_factory=dict)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    optimization_model: Dict[str, Any] = field(default_factory=dict)
    robust_optimization_model: Dict[str, Any] = field(default_factory=dict)
    retail_decision_model: Dict[str, Any] = field(default_factory=dict)
    retail_decision_manifest: str = ""
    multiobjective_model: Dict[str, Any] = field(default_factory=dict)
    network_model: Dict[str, Any] = field(default_factory=dict)
    mechanism_model: Dict[str, Any] = field(default_factory=dict)
    mechanism_data_manifest: str = ""
    simulation_model: Dict[str, Any] = field(default_factory=dict)
    spatial_model: Dict[str, Any] = field(default_factory=dict)
    expected_outputs: List[str] = field(default_factory=list)
    forecast_horizon: int = 1
    seasonal_period: int = 7
    validation_mode: str = ""
    specialist_routes: List[Dict[str, Any]] = field(default_factory=list)
    domain_verifier: str = ""
    allowed_code_roots: List[str] = field(default_factory=list)
    source: str = "inferred"
    unresolved_fields: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class ProblemProfile:
    problem_id: str
    brief_path: str
    data_root: str
    keywords: List[str]
    archetypes: List[str]
    assets: List[DataAsset]
    target_candidates: List[str]
    metric_candidates: List[str]
    metric_directions: Dict[str, str]
    notes: List[str]
    contract: Optional[ProblemContract] = None


@dataclass
class CorpusHit:
    case_id: str
    source_path: str
    score: float
    matched_terms: List[str]
    suggested_methods: List[str]


@dataclass
class RouteSpec:
    route_id: str
    role: str
    family: str
    rationale: str
    expected_metrics: List[str]
    source: str = "heuristic"
    supported: bool = True
    adapter_path: str = ""


@dataclass
class ExperimentRecord:
    problem_id: str
    route_id: str
    role: str
    status: str
    code_path: str
    data_path: str
    metrics: Dict[str, Any]
    error: str = ""
    repair_attempts: int = 0
    limitations: List[str] = field(default_factory=list)
    recommendation: str = ""
    original_code_hash: str = ""
    final_code_hash: str = ""
    repair_log: List[Dict[str, Any]] = field(default_factory=list)
    run_id: str = ""
    input_hash: str = ""
    config_hash: str = ""
    artifact_hashes: Dict[str, str] = field(default_factory=dict)
    evidence_paths: Dict[str, str] = field(default_factory=dict)
    command: List[str] = field(default_factory=list)
    exit_code: Optional[int] = None
    stdout_path: str = ""
    stderr_path: str = ""
    environment: Dict[str, str] = field(default_factory=dict)


def to_jsonable(payload: Any) -> Any:
    if hasattr(payload, "__dataclass_fields__"):
        return asdict(payload)
    if isinstance(payload, list):
        return [to_jsonable(item) for item in payload]
    if isinstance(payload, dict):
        return {key: to_jsonable(value) for key, value in payload.items()}
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def read_text_safe(path: Path, max_chars: int = 20000) -> str:
    if not path or not path.exists() or path.is_dir():
        return ""
    for encoding in ("utf-8", "gb18030", "gbk", "utf-16"):
        try:
            return path.read_text(encoding=encoding, errors="ignore")[:max_chars]
        except Exception:
            continue
    return ""
