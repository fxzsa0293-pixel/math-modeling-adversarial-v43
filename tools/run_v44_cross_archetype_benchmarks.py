from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from solver.schemas import ProblemContract
from workflow.mechanism_model_builder import build_cumcm2018a_mechanism_data, verify_mechanism_data_manifest
from workflow.problem_dag import build_artifact_manifest, verify_artifact_manifest, verify_workflow_manifest
from workflow.workflow_executor import execute_workflow


TOOLS = Path(__file__).resolve().parent
SKILL_ROOT = TOOLS.parent


def find_cumcm2018a(reference_root: Path) -> Path:
    matches = [path.resolve() for path in Path(reference_root).rglob("CUMCM2018A_data.xlsx") if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected_one_CUMCM2018A_data_workbook:found={len(matches)}")
    return matches[0]


def run_cross_archetype_benchmarks(reference_root: Path, output_dir: Path) -> Dict[str, Any]:
    reference_root = Path(reference_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source = find_cumcm2018a(reference_root)
    build = build_cumcm2018a_mechanism_data(source, output_dir / "derived" / "CUMCM2018A")
    contract = ProblemContract(
        version="v44", problem_id="CUMCM2018A_lumped_thermal_mechanism",
        task_type="mechanism", status="ready", data_file=build["data_file"],
        time_column="time_s", mechanism_model=build["model"],
        mechanism_data_manifest=build["manifest_path"],
        metric_directions={"MAE": "lower_better", "RMSE": "lower_better", "WAPE": "lower_better"},
        expected_outputs=["held_out_temperature_trajectory", "parameter_fit", "identifiability", "paper_results_section"],
        source="provided",
        notes=[
            "Real CUMCM2018A Appendix 1 and Appendix 2.",
            "The declared first-order relaxation is a lumped thermal baseline, not the full multilayer heat-conduction PDE required for final clothing-thickness design.",
        ],
    )
    contract_path = output_dir / "contracts" / "CUMCM2018A_mechanism.json"
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(json.dumps(asdict(contract), ensure_ascii=False, indent=2), encoding="utf-8")
    workflow_path = output_dir / "CUMCM2018A_workflow.json"
    workflow_path.write_text(json.dumps({
        "version": "v44", "problem_id": "CUMCM2018A_cross_archetype",
        "nodes": [
            {
                "node_id": "thermal_mechanism", "kind": "mechanism",
                "contract_path": str(contract_path), "data_root": str(Path(build["data_file"]).parent),
                "max_routes": 2,
                "outputs": {
                    "result_summary": "workflow_exports.result_summary_json",
                    "paper_section": "workflow_exports.paper_results_section_md",
                    "route_metrics": "workflow_exports.route_metrics_long_csv",
                },
            },
            {
                "node_id": "paper", "kind": "paper", "dependencies": ["thermal_mechanism"],
                "inputs": {"thermal_section": {"from_node": "thermal_mechanism", "output": "paper_section"}},
                "report_name": "CUMCM2018A_lumped_mechanism_report.md",
                "outputs": {"report": "report_path", "report_manifest": "manifest_path"},
            },
        ],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    workflow = execute_workflow(workflow_path, output_dir / "run", max_routes=2, timeout_seconds=300)
    solver_summary_path = output_dir / "run" / "nodes" / "thermal_mechanism" / "summary.json"
    solver_summary = json.loads(solver_summary_path.read_text(encoding="utf-8"))
    registry = json.loads(Path(solver_summary["registry_path"]).read_text(encoding="utf-8"))
    metrics = {record["route_id"]: record["metrics"] for record in registry if record.get("status") == "executed"}
    candidate = metrics.get("declared_ode_multistart", {})
    baseline = metrics.get("mechanism_constant_baseline", {})
    domain = solver_summary.get("domain_verification") or {}
    manifest_path = build_artifact_manifest(
        "cross_archetype_benchmark",
        [source, Path(build["data_file"]), Path(build["materials_file"]), Path(build["manifest_path"]),
         contract_path, workflow_path, Path(workflow["manifest"]["manifest_path"]), solver_summary_path,
         Path(workflow["nodes"][1]["output_artifacts"]["report"])],
        output_dir / "benchmark_artifact_manifest.json",
    )
    passed = all([
        build["verification"]["passed"], workflow["passed"], domain.get("passed") is True,
        verify_workflow_manifest(Path(workflow["manifest"]["manifest_path"]))["passed"],
        verify_artifact_manifest(manifest_path)["passed"], bool(candidate), bool(baseline),
    ])
    summary = {
        "version": "v44", "suite": "cross_archetype_real_data",
        "passed": passed, "case_count": 1, "passed_cases": int(passed),
        "case": {
            "case_id": "CUMCM2018A_lumped_thermal_mechanism", "source_path": str(source),
            "source_sha256": _sha256(source), "source_verification": build["verification"],
            "solver_verdict": solver_summary["final_judge"]["verdict"],
            "claim_level": solver_summary["final_judge"]["claim_level"],
            "recommended_route": solver_summary["final_judge"]["recommended_route"],
            "baseline_final_test_MAE": baseline.get("final_test_MAE"),
            "candidate_final_test_MAE": candidate.get("final_test_MAE"),
            "candidate_identifiability": candidate.get("identifiability_status"),
            "domain_verification_passed": domain.get("passed"),
            "scope_boundary": "Verified lumped first-order thermal fit only; multilayer PDE and thickness optimization remain unsupported by this case.",
        },
        "workflow": workflow["manifest"], "artifact_manifest_path": str(manifest_path),
    }
    summary_path = output_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V44 real-data benchmarks outside the CUMCM2023C retail archetype.")
    parser.add_argument("--reference-root", type=Path, default=SKILL_ROOT / "data" / "reference_root")
    parser.add_argument("--output-dir", type=Path, default=TOOLS / "run" / "v44_cross_archetype_benchmarks")
    args = parser.parse_args()
    print(json.dumps(run_cross_archetype_benchmarks(args.reference_root, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
