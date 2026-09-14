from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List


NODE_KINDS = {"analysis", "prediction", "optimization", "mechanism", "simulation", "sensitivity", "paper"}


@dataclass
class WorkflowNode:
    node_id: str
    kind: str
    dependencies: List[str] = field(default_factory=list)
    contract_path: str = ""
    input_artifacts: Dict[str, str] = field(default_factory=dict)
    output_artifacts: Dict[str, str] = field(default_factory=dict)
    verification: Dict[str, Any] = field(default_factory=dict)
    execution: Dict[str, Any] = field(default_factory=dict)


def build_workflow_manifest(problem_id: str, nodes: Iterable[WorkflowNode], output_path: Path) -> Dict[str, Any]:
    nodes = list(nodes)
    order = validate_workflow_nodes(nodes)
    by_id = {node.node_id: node for node in nodes}
    completed = set()
    node_records = []
    producer_by_path: Dict[str, str] = {}
    duplicate_output_paths = set()
    for node in nodes:
        for raw_path in node.output_artifacts.values():
            path = str(Path(raw_path).resolve())
            if path in producer_by_path and producer_by_path[path] != node.node_id:
                duplicate_output_paths.add(path)
            else:
                producer_by_path[path] = node.node_id
    for node_id in order:
        node = by_id[node_id]
        issues = []
        if any(dependency not in completed for dependency in node.dependencies):
            issues.append("upstream_not_completed")
        if node.verification.get("passed") is not True:
            issues.append("node_verification_not_passed")
        inputs = _bind_artifacts(node.input_artifacts, issues, "input")
        outputs = _bind_artifacts(node.output_artifacts, issues, "output")
        if any(binding["path"] in duplicate_output_paths for binding in outputs.values()):
            issues.append("duplicate_output_artifact_producer")
        contract_binding = None
        if node.contract_path:
            contract = Path(node.contract_path).resolve()
            if not contract.is_file():
                issues.append("contract_missing")
            else:
                contract_binding = {"path": str(contract), "sha256": _sha256(contract)}
        dependency_set = set(node.dependencies)
        for binding in inputs.values():
            producer = producer_by_path.get(binding["path"])
            if producer is not None and producer not in dependency_set:
                issues.append(f"undeclared_artifact_dependency:{producer}")
        status = "complete" if not issues else "blocked"
        if status == "complete":
            completed.add(node_id)
        node_records.append({**asdict(node), "status": status, "issues": issues, "contract_binding": contract_binding, "input_bindings": inputs, "output_bindings": outputs})
    manifest = {"version": "v44", "kind": "problem_workflow_dag", "problem_id": problem_id, "topological_order": order, "passed": len(completed) == len(nodes), "nodes": node_records}
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"manifest_path": str(output_path), "manifest_sha256": _sha256(output_path), "passed": manifest["passed"], "node_count": len(nodes)}


def verify_workflow_manifest(manifest_path: Path) -> Dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    issues = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        nodes = [WorkflowNode(node["node_id"], node["kind"], node.get("dependencies", []), node.get("contract_path", ""), node.get("input_artifacts", {}), node.get("output_artifacts", {}), node.get("verification", {}), node.get("execution", {})) for node in manifest["nodes"]]
        order = validate_workflow_nodes(nodes)
        if order != manifest.get("topological_order"):
            issues.append("workflow_topological_order_mismatch")
        if manifest.get("passed") is not True or any(node.get("status") != "complete" or node.get("issues") for node in manifest["nodes"]):
            issues.append("workflow_contains_incomplete_node")
        for node in manifest["nodes"]:
            for binding_kind in ("input_bindings", "output_bindings"):
                for alias, binding in node.get(binding_kind, {}).items():
                    path = Path(binding.get("path", ""))
                    if not path.is_file() or _sha256(path) != binding.get("sha256"):
                        issues.append(f"workflow_artifact_hash_mismatch:{node['node_id']}:{binding_kind}:{alias}")
            contract_binding = node.get("contract_binding")
            contract_path = Path(contract_binding.get("path", "")) if isinstance(contract_binding, dict) else None
            if node.get("contract_path") and (contract_path is None or not contract_path.is_file() or _sha256(contract_path) != contract_binding.get("sha256")):
                issues.append(f"workflow_contract_hash_mismatch:{node['node_id']}")
            run_manifest = node.get("execution", {}).get("run_manifest_path")
            if run_manifest:
                run_verification = verify_artifact_manifest(Path(run_manifest))
                issues.extend(f"workflow_node_run_manifest:{node['node_id']}:{issue}" for issue in run_verification["issues"])
            report_manifest = node.get("execution", {}).get("report_manifest_path")
            if report_manifest:
                report_verification = verify_artifact_manifest(Path(report_manifest))
                issues.extend(f"workflow_node_report_manifest:{node['node_id']}:{issue}" for issue in report_verification["issues"])
    except Exception as exc:
        issues.append(f"workflow_manifest_verification_error:{exc}")
    return {"kind": "problem_workflow_dag", "passed": not issues, "issues": issues, "manifest_path": str(manifest_path), "manifest_sha256": _sha256(manifest_path) if manifest_path.is_file() else None}


def build_artifact_manifest(kind: str, paths: Iterable[Path], output_path: Path) -> Path:
    output_path = Path(output_path).resolve()
    unique = sorted({Path(path).resolve() for path in paths if Path(path).is_file() and Path(path).resolve() != output_path}, key=str)
    payload = {
        "version": "v44", "kind": kind,
        "artifacts": [{"path": str(path), "sha256": _sha256(path)} for path in unique],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def verify_artifact_manifest(manifest_path: Path) -> Dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    issues = []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        kind = payload.get("kind")
        if kind == "assembled_modeling_report":
            artifacts = list(payload.get("sources") or []) + ([payload["report"]] if isinstance(payload.get("report"), dict) else [])
        elif kind in {"solver_node_run", "workflow_runtime", "cross_archetype_benchmark"}:
            artifacts = payload.get("artifacts")
        else:
            artifacts = None
            issues.append(f"artifact_manifest_kind_invalid:{kind}")
        if not isinstance(artifacts, list) or not artifacts:
            issues.append("artifact_manifest_empty_or_invalid")
        for index, binding in enumerate(artifacts or []):
            path = Path(str(binding.get("path", "")))
            if not path.is_file() or _sha256(path) != binding.get("sha256"):
                issues.append(f"artifact_hash_mismatch:{index}")
    except Exception as exc:
        issues.append(f"artifact_manifest_verification_error:{exc}")
    return {"passed": not issues, "issues": issues, "manifest_path": str(manifest_path)}


def validate_workflow_nodes(nodes: Iterable[WorkflowNode]) -> List[str]:
    nodes = list(nodes)
    ids = [node.node_id for node in nodes]
    if any(not node_id for node_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("workflow_node_ids_invalid_or_duplicate")
    known = set(ids)
    for node in nodes:
        if node.kind not in NODE_KINDS:
            raise ValueError(f"workflow_node_kind_invalid:{node.node_id}:{node.kind}")
        if node.node_id in node.dependencies or not set(node.dependencies).issubset(known):
            raise ValueError(f"workflow_node_dependencies_invalid:{node.node_id}")
        if len(node.dependencies) != len(set(node.dependencies)):
            raise ValueError(f"workflow_node_dependencies_duplicate:{node.node_id}")
    indegree = {node_id: 0 for node_id in ids}
    downstream = {node_id: [] for node_id in ids}
    for node in nodes:
        indegree[node.node_id] = len(node.dependencies)
        for dependency in node.dependencies:
            downstream[dependency].append(node.node_id)
    ready = sorted(node_id for node_id, degree in indegree.items() if degree == 0)
    order = []
    while ready:
        node_id = ready.pop(0)
        order.append(node_id)
        for child in sorted(downstream[node_id]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort()
    if len(order) != len(ids):
        raise ValueError("workflow_dependency_cycle")
    return order


def _bind_artifacts(artifacts: Dict[str, str], issues: List[str], kind: str) -> Dict[str, Dict[str, str]]:
    if not isinstance(artifacts, dict) or len(artifacts) != len(set(artifacts)):
        issues.append(f"{kind}_artifacts_invalid")
        return {}
    bindings = {}
    for alias, raw_path in artifacts.items():
        path = Path(raw_path).resolve()
        if not path.is_file():
            issues.append(f"{kind}_artifact_missing:{alias}")
            continue
        bindings[str(alias)] = {"path": str(path), "sha256": _sha256(path)}
    return bindings


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
