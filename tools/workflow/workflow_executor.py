from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from run_v39_solver import run_solver
from solver.schemas import write_json
from workflow.problem_dag import (
    WorkflowNode, build_artifact_manifest, build_workflow_manifest,
    validate_workflow_nodes, verify_artifact_manifest, verify_workflow_manifest,
)
from workflow.workspace_protocol import create_multi_question_workspace


SUPPORTED_ACTIONS = {"solver", "paper"}
SOLVER_NODE_KINDS = {"analysis", "prediction", "optimization", "mechanism", "simulation", "sensitivity"}
CONTRACT_ARTIFACT_FIELDS = {"data_file", "data_assembly_manifest", "retail_decision_manifest", "mechanism_data_manifest", "domain_verifier"}
NODE_ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")


def execute_workflow(
    workflow_path: Path,
    output_dir: Path,
    *,
    default_data_root: Path | None = None,
    default_corpus_root: Path | None = None,
    max_routes: int = 8,
    timeout_seconds: int = 60,
    allow_specialist_code: bool = False,
    resume: bool = True,
) -> Dict[str, Any]:
    workflow_path = Path(workflow_path).resolve()
    output_dir = Path(output_dir).resolve()
    spec = _load_and_validate_spec(workflow_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    execution_config_path = output_dir / "workflow_execution_config.json"
    write_json(execution_config_path, {
        "version": "v44", "max_routes": max_routes, "timeout_seconds": timeout_seconds,
        "allow_specialist_code": allow_specialist_code,
        "default_data_root": str(Path(default_data_root).resolve()) if default_data_root else "",
        "default_corpus_root": str(Path(default_corpus_root).resolve()) if default_corpus_root else "",
    })
    tools_root = Path(__file__).resolve().parents[1]
    runtime_manifest_path = build_artifact_manifest(
        "workflow_runtime",
        [tools_root / "run_v39_solver.py"]
        + list((tools_root / "solver").glob("*.py"))
        + list((tools_root / "workflow").glob("*.py")),
        output_dir / "workflow_runtime_manifest.json",
    )
    node_specs = spec["nodes"]
    order = validate_workflow_nodes([_spec_node(item, workflow_path.parent) for item in node_specs])
    if resume:
        reusable = _reuse_complete_workflow(workflow_path, output_dir, spec["problem_id"], order)
        if reusable is not None:
            return reusable
    by_id = {item["node_id"]: item for item in node_specs}
    question_ids = {node_id: f"question_{index:02d}" for index, node_id in enumerate(order, 1)}
    workspaces = create_multi_question_workspace(output_dir, question_ids.values())
    results: Dict[str, Dict[str, Any]] = {}
    manifest_nodes: List[WorkflowNode] = []

    for node_id in order:
        item = by_id[node_id]
        action = item.get("action") or ("paper" if item["kind"] == "paper" else "solver")
        question_id = question_ids[node_id]
        node_dir = output_dir / "nodes" / node_id
        node_dir.mkdir(parents=True, exist_ok=True)
        input_artifacts: Dict[str, str] = {
            "workflow_spec": str(workflow_path),
            "workflow_execution_config": str(execution_config_path),
            "workflow_runtime_manifest": str(runtime_manifest_path),
        }
        if item.get("contract_path"):
            input_artifacts["source_contract"] = str(_resolve_path(workflow_path.parent, item["contract_path"]))
        output_artifacts: Dict[str, str] = {}
        execution: Dict[str, Any] = {"action": action, "question_id": question_id}
        issues: List[str] = []
        upstream_ok = all(results.get(dep, {}).get("passed") is True for dep in item.get("dependencies", []))
        if not upstream_ok:
            issues.append("upstream_execution_not_passed")
        try:
            resolved_inputs = _resolve_inputs(item, results)
            input_artifacts.update({alias: value["path"] for alias, value in resolved_inputs.items()})
            if not issues and action == "solver":
                result, contract_path = _execute_solver_node(
                    item, workflow_path.parent, node_dir, workspaces[question_id], resolved_inputs,
                    default_data_root, default_corpus_root, max_routes, timeout_seconds, allow_specialist_code,
                )
                passed = result["final_judge"].get("verdict") == "pass"
                if not passed:
                    issues.append(f"solver_verdict:{result['final_judge'].get('verdict')}")
                output_artifacts = _resolve_outputs(item, result, node_dir / "summary.json")
                _add_solver_input_bindings(input_artifacts, result)
                run_manifest = build_artifact_manifest(
                    "solver_node_run",
                    _solver_run_artifacts(result, node_dir),
                    node_dir / "solver_run_manifest.json",
                )
                output_artifacts["_solver_run_manifest"] = str(run_manifest)
                execution.update({
                    "summary_path": str(node_dir / "summary.json"),
                    "verdict": result["final_judge"].get("verdict"),
                    "run_manifest_path": str(run_manifest),
                })
            elif not issues and action == "paper":
                report = _execute_paper_node(item, node_dir, resolved_inputs)
                contract_path = ""
                passed = report["verification"]["passed"]
                output_artifacts = _resolve_outputs(item, report, Path(report["report_path"]))
                output_artifacts["_report_manifest"] = report["manifest_path"]
                execution.update({"report_path": report["report_path"], "report_manifest_path": report["manifest_path"]})
            else:
                contract_path = ""
                passed = False
        except Exception as exc:
            contract_path = ""
            passed = False
            issues.append(f"node_execution_error:{type(exc).__name__}:{exc}")

        verification = {"passed": passed and not issues, "issues": issues}
        node_result = {
            "node_id": node_id, "kind": item["kind"], "question_id": question_id,
            "passed": verification["passed"], "verification": verification,
            "input_artifacts": input_artifacts, "output_artifacts": output_artifacts,
            "contract_path": contract_path, "execution": execution,
        }
        write_json(node_dir / "node_execution.json", node_result)
        results[node_id] = node_result
        manifest_nodes.append(WorkflowNode(
            node_id=node_id, kind=item["kind"], dependencies=item.get("dependencies", []),
            contract_path=contract_path, input_artifacts=input_artifacts, output_artifacts=output_artifacts,
            verification=verification, execution=execution,
        ))

    built = build_workflow_manifest(spec["problem_id"], manifest_nodes, output_dir / "workflow_manifest.json")
    verification = verify_workflow_manifest(Path(built["manifest_path"]))
    summary = {
        "version": "v44", "kind": "executable_problem_workflow",
        "problem_id": spec["problem_id"], "workflow_path": str(workflow_path),
        "workflow_sha256": _sha256(workflow_path), "topological_order": order,
        "passed": built["passed"] and verification["passed"],
        "nodes": [results[node_id] for node_id in order],
        "manifest": built, "verification": verification,
        "workspace_path": str(output_dir / "competition_workspace"),
    }
    write_json(output_dir / "workflow_summary.json", summary)
    return summary


def _reuse_complete_workflow(workflow_path: Path, output_dir: Path, problem_id: str, order: List[str]) -> Dict[str, Any] | None:
    manifest_path = output_dir / "workflow_manifest.json"
    if not manifest_path.is_file():
        return None
    verification = verify_workflow_manifest(manifest_path)
    if not verification["passed"]:
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if manifest.get("problem_id") != problem_id or manifest.get("topological_order") != order:
        return None
    records = {node.get("node_id"): node for node in manifest.get("nodes", [])}
    if set(records) != set(order):
        return None
    nodes = []
    for node_id in order:
        record = records[node_id]
        execution = dict(record.get("execution", {}))
        execution["reused"] = True
        nodes.append({
            "node_id": node_id, "kind": record.get("kind"),
            "question_id": execution.get("question_id"), "passed": True,
            "verification": record.get("verification", {"passed": True, "issues": []}),
            "input_artifacts": record.get("input_artifacts", {}),
            "output_artifacts": record.get("output_artifacts", {}),
            "contract_path": record.get("contract_path", ""), "execution": execution,
        })
    summary = {
        "version": "v44", "kind": "executable_problem_workflow",
        "problem_id": problem_id, "workflow_path": str(workflow_path),
        "workflow_sha256": _sha256(workflow_path), "topological_order": order,
        "passed": True, "reused": True, "nodes": nodes,
        "manifest": {
            "manifest_path": str(manifest_path), "manifest_sha256": _sha256(manifest_path),
            "passed": True, "node_count": len(nodes),
        },
        "verification": verification,
        "workspace_path": str(output_dir / "competition_workspace"),
    }
    write_json(output_dir / "workflow_summary.json", summary)
    return summary


def _load_and_validate_spec(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != "v44" or not isinstance(payload.get("problem_id"), str) or not payload["problem_id"]:
        raise ValueError("workflow_header_invalid")
    nodes = payload.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("workflow_nodes_missing")
    for item in nodes:
        if not isinstance(item, dict) or not NODE_ID_PATTERN.fullmatch(str(item.get("node_id", ""))):
            raise ValueError("workflow_node_schema_invalid")
        action = item.get("action") or ("paper" if item.get("kind") == "paper" else "solver")
        if action not in SUPPORTED_ACTIONS or (action == "paper") != (item.get("kind") == "paper"):
            raise ValueError(f"workflow_action_invalid:{item.get('node_id')}")
        if action == "solver" and item.get("kind") not in SOLVER_NODE_KINDS:
            raise ValueError(f"workflow_solver_kind_invalid:{item.get('node_id')}")
        if action == "solver" and not item.get("contract_path"):
            raise ValueError(f"workflow_contract_missing:{item.get('node_id')}")
        for binding in (item.get("inputs") or {}).values():
            if not isinstance(binding, dict) or not binding.get("from_node") or not binding.get("output"):
                raise ValueError(f"workflow_input_binding_invalid:{item.get('node_id')}")
            field = binding.get("contract_field")
            if field and field not in CONTRACT_ARTIFACT_FIELDS:
                raise ValueError(f"workflow_contract_field_not_allowed:{field}")
    return payload


def _spec_node(item: Dict[str, Any], base: Path) -> WorkflowNode:
    contract = str(_resolve_path(base, item["contract_path"])) if item.get("contract_path") else ""
    return WorkflowNode(item["node_id"], item["kind"], list(item.get("dependencies", [])), contract_path=contract)


def _resolve_inputs(item: Dict[str, Any], results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    resolved = {}
    dependencies = set(item.get("dependencies", []))
    for alias, binding in (item.get("inputs") or {}).items():
        source_id = binding["from_node"]
        if source_id not in dependencies:
            raise ValueError(f"input_source_not_direct_dependency:{alias}:{source_id}")
        source = results.get(source_id)
        if not source or source.get("passed") is not True:
            raise ValueError(f"input_source_not_passed:{alias}:{source_id}")
        path = source.get("output_artifacts", {}).get(binding["output"])
        if not path or not Path(path).is_file():
            raise ValueError(f"input_output_missing:{alias}:{source_id}:{binding['output']}")
        resolved[alias] = {"path": str(Path(path).resolve()), "contract_field": str(binding.get("contract_field", ""))}
    return resolved


def _execute_solver_node(
    item: Dict[str, Any], base: Path, node_dir: Path, workspace: Dict[str, str],
    inputs: Dict[str, Dict[str, str]], default_data_root: Path | None, default_corpus_root: Path | None,
    max_routes: int, timeout_seconds: int, allow_specialist_code: bool,
) -> tuple[Dict[str, Any], str]:
    source_contract = _resolve_path(base, item["contract_path"])
    contract = json.loads(source_contract.read_text(encoding="utf-8"))
    for binding in inputs.values():
        if binding["contract_field"]:
            contract[binding["contract_field"]] = binding["path"]
    effective_contract = node_dir / "effective_contract.json"
    write_json(effective_contract, contract)
    injected_data = next((binding["path"] for binding in inputs.values() if binding["contract_field"] == "data_file"), None)
    data_root = _resolve_path(base, item.get("data_root")) if item.get("data_root") else (Path(injected_data).parent if injected_data else default_data_root)
    if data_root is None:
        data_file = Path(str(contract.get("data_file", "")))
        data_root = data_file.parent if data_file.is_file() else source_contract.parent
    brief = _resolve_path(base, item["brief"]) if item.get("brief") else None
    corpus = _resolve_path(base, item["corpus_root"]) if item.get("corpus_root") else default_corpus_root
    args = argparse.Namespace(
        problem_id=str(contract.get("problem_id") or item["node_id"]), data_root=Path(data_root),
        brief=brief, contract=effective_contract, corpus_root=corpus, output_dir=node_dir,
        max_routes=int(item.get("max_routes", max_routes)),
        timeout_seconds=int(item.get("timeout_seconds", timeout_seconds)),
        allow_specialist_code=bool(item.get("allow_specialist_code", allow_specialist_code)),
    )
    summary = run_solver(args, workspace_paths=workspace)
    return summary, str(Path(summary["contract_path"]).resolve())


def _add_solver_input_bindings(input_artifacts: Dict[str, str], summary: Dict[str, Any]) -> None:
    final = summary.get("final_judge", {})
    bound = []
    for group in (final.get("input_bindings", []), final.get("code_bindings", [])):
        for binding in group or []:
            path = Path(str(binding.get("path", "")))
            if path.is_file():
                bound.append(path.resolve())
    verification = summary.get("domain_verification") or {}
    script = Path(str(verification.get("script_path", "")))
    if script.is_file():
        bound.append(script.resolve())
    contract = summary.get("profile", {}).get("contract", {}) or {}
    for key in ("data_assembly_manifest", "retail_decision_manifest", "mechanism_data_manifest"):
        manifest_path = Path(str(contract.get(key, "")))
        if manifest_path.is_file():
            bound.append(manifest_path.resolve())
            bound.extend(_manifest_bound_paths(manifest_path))
    existing = {str(Path(path).resolve()) for path in input_artifacts.values()}
    for index, path in enumerate(sorted(set(bound), key=str)):
        if str(path) not in existing:
            input_artifacts[f"solver_bound_input_{index:02d}"] = str(path)


def _manifest_bound_paths(manifest_path: Path) -> List[Path]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    paths: List[Path] = []

    def visit(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                visit(child, key)
        elif key.endswith("path") and isinstance(value, str):
            path = Path(value)
            if path.is_file():
                paths.append(path.resolve())

    visit(payload)
    return paths


def _solver_run_artifacts(summary: Dict[str, Any], node_dir: Path) -> List[Path]:
    paths: List[Path] = [node_dir / "summary.json", Path(summary["contract_path"]), Path(summary["registry_path"])]
    paths.extend([node_dir / "judge" / "final_judge.json", node_dir / "judge" / "final_judge.md"])
    for raw_path in summary.get("workflow_exports", {}).values():
        paths.append(Path(str(raw_path)))
    for key in ("data_audit_path", "solving_plan_path"):
        if summary.get(key):
            paths.append(Path(str(summary[key])))
    verification = summary.get("domain_verification") or {}
    for key, value in verification.items():
        if key.endswith("_path") and isinstance(value, str):
            paths.append(Path(value))
    registry_path = Path(summary["registry_path"])
    if registry_path.is_file():
        for record in json.loads(registry_path.read_text(encoding="utf-8")):
            for raw_path in (record.get("evidence_paths") or {}).values():
                paths.append(Path(str(raw_path)))
            if record.get("code_path"):
                paths.append(Path(str(record["code_path"])))
    return paths


def _execute_paper_node(item: Dict[str, Any], node_dir: Path, inputs: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    if not inputs:
        raise ValueError("paper_requires_input_sections")
    sections = []
    bindings = []
    for alias, binding in inputs.items():
        path = Path(binding["path"])
        sections.append(path.read_text(encoding="utf-8"))
        bindings.append({"alias": alias, "path": str(path), "sha256": _sha256(path)})
    report_path = node_dir / str(item.get("report_name", "complete_report.md"))
    report_path.write_text(("\n\n".join(sections)).rstrip() + "\n", encoding="utf-8")
    manifest = {"version": "v44", "kind": "assembled_modeling_report", "sources": bindings, "report": {"path": str(report_path), "sha256": _sha256(report_path)}}
    manifest_path = node_dir / "report_manifest.json"
    write_json(manifest_path, manifest)
    return {"report_path": str(report_path), "manifest_path": str(manifest_path), "verification": verify_assembled_report(manifest_path)}


def verify_assembled_report(manifest_path: Path) -> Dict[str, Any]:
    issues = []
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        bindings = list(manifest.get("sources", [])) + [manifest.get("report", {})]
        if not manifest.get("sources"):
            issues.append("report_sources_missing")
        for index, binding in enumerate(bindings):
            path = Path(str(binding.get("path", "")))
            if not path.is_file() or _sha256(path) != binding.get("sha256"):
                issues.append(f"report_artifact_hash_mismatch:{index}")
    except Exception as exc:
        issues.append(f"report_verification_error:{exc}")
    return {"passed": not issues, "issues": issues, "manifest_path": str(Path(manifest_path).resolve())}


def _resolve_outputs(item: Dict[str, Any], payload: Dict[str, Any], default_path: Path) -> Dict[str, str]:
    declarations = item.get("outputs") or {"primary": "$default"}
    outputs = {}
    for alias, selector in declarations.items():
        value = str(default_path) if selector == "$default" else _select(payload, str(selector))
        path = Path(str(value)).resolve()
        if not path.is_file():
            raise ValueError(f"declared_output_not_file:{alias}:{selector}")
        outputs[str(alias)] = str(path)
    return outputs


def _select(payload: Dict[str, Any], selector: str) -> Any:
    value: Any = payload
    for part in selector.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"output_selector_missing:{selector}")
        value = value[part]
    return value


def _resolve_path(base: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return (path if path.is_absolute() else base / path).resolve()


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
