from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from solver.adapter_generator import generate_adapters
from solver.corpus_retriever import retrieve_corpus
from solver.experiment_registry import write_registry
from solver.experiment_runner import run_experiments
from solver.final_judge import _format_md, judge_solution
from solver.domain_verifier import run_domain_verifier
from solver.native_optimization import run_native_experiments, verify_native_optimization
from solver.native_mechanism import run_native_mechanism_experiments, verify_native_mechanism
from solver.native_simulation import run_native_simulation_experiments, verify_native_simulation
from solver.native_spatial import run_native_spatial_experiments, verify_native_spatial
from solver.native_multiobjective import run_native_multiobjective_experiments, verify_native_multiobjective
from solver.native_robust import run_native_robust_experiments, verify_native_robust
from solver.native_retail import run_native_retail_experiments, verify_native_retail
from solver.native_predictive_verifier import verify_native_predictive
from solver.problem_parser import parse_problem
from solver.route_planner import plan_routes
from solver.schemas import write_json
from solver.problem_contract import write_contract
from solver.problem_contract import validate_contract
from solver.schemas import DataAsset
from workflow.data_auditor import audit_data_root
from workflow.data_assembler import assemble_contract_data
from workflow.plan_writer import write_solving_plan
from workflow.result_exporter import export_workflow_results
from workflow.workspace_protocol import create_workspace

TOOLS = Path(__file__).resolve().parent
SKILL_ROOT = TOOLS.parent
DEFAULT_DATA_ROOT = SKILL_ROOT / "data" / "reference_root"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V44 evidence-gated automatic modeling solver pipeline.")
    parser.add_argument("--problem-id", default="demo_problem", help="Stable id for this modeling problem.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Problem data directory or reference corpus root.")
    parser.add_argument("--brief", type=Path, default=None, help="Optional problem statement text/markdown file.")
    parser.add_argument("--contract", type=Path, default=None, help="Optional V44 problem_contract.json. Required when automatic binding is ambiguous.")
    parser.add_argument("--corpus-root", type=Path, default=None, help="Optional historical paper/code corpus root.")
    parser.add_argument("--output-dir", type=Path, default=TOOLS / "run" / "v44_auto_solver", help="Output directory.")
    parser.add_argument("--max-routes", type=int, default=12, help="Maximum generated routes.")
    parser.add_argument("--timeout-seconds", type=int, default=60, help="Per-adapter execution timeout.")
    parser.add_argument("--allow-specialist-code", action="store_true", help="Execute contract-declared specialist adapters and verifier. Review these scripts before enabling.")
    return parser.parse_args()


def run_solver(args: argparse.Namespace, workspace_paths: dict[str, str] | None = None) -> dict:
    if not args.data_root.exists():
        raise FileNotFoundError(f"data_root not found: {args.data_root}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace_paths = workspace_paths or create_workspace(output_dir)
    profile = parse_problem(args.problem_id, args.data_root, args.brief, contract_path=args.contract)
    data_assembly = None
    if profile.contract.status == "ready" and profile.contract.data_sources:
        try:
            data_assembly = assemble_contract_data(profile.contract, Path(workspace_paths["work/data_audit"]) / "assembly")
            assembled_asset = DataAsset(
                path=profile.contract.data_file, suffix=".csv", size_bytes=Path(profile.contract.data_file).stat().st_size,
                columns=list(data_assembly["output_columns"]), row_count=int(data_assembly["output_rows"]),
                column_count=len(data_assembly["output_columns"]), readable=True, quality_score=10.0,
            )
            profile.assets.append(assembled_asset)
            profile.contract = validate_contract(profile.contract, profile.assets)
        except Exception as exc:
            profile.contract.status = "invalid"
            profile.contract.unresolved_fields = list(dict.fromkeys(profile.contract.unresolved_fields + [f"data_assembly_failed:{exc}"]))
            data_assembly = {"error": str(exc)}
    contract_path = write_contract(Path(workspace_paths["problem"]) / "problem_contract.json", profile.contract)
    data_audit = audit_data_root(args.data_root, Path(workspace_paths["work/data_audit"]), profile.contract)
    hits = retrieve_corpus(profile, args.corpus_root or args.data_root)
    routes = plan_routes(profile, hits, max_routes=args.max_routes)
    plan_path = write_solving_plan(profile, routes, data_audit, Path(workspace_paths["work/plan"]))
    adapter_dir = output_dir / "generated_adapters"
    specialist_authorized = not profile.contract.specialist_routes or args.allow_specialist_code
    native_optimization = profile.contract.task_type == "constrained_optimization" and bool(profile.contract.optimization_model or profile.contract.network_model)
    native_mechanism = profile.contract.task_type == "mechanism" and bool(profile.contract.mechanism_model)
    native_simulation = profile.contract.task_type == "simulation" and bool(profile.contract.simulation_model)
    native_spatial = profile.contract.task_type == "spatial" and bool(profile.contract.spatial_model)
    native_multiobjective = profile.contract.task_type == "constrained_optimization" and bool(profile.contract.multiobjective_model)
    native_robust = profile.contract.task_type == "constrained_optimization" and bool(profile.contract.robust_optimization_model)
    native_retail = profile.contract.task_type == "constrained_optimization" and bool(profile.contract.retail_decision_model)
    if profile.contract.status == "ready" and native_multiobjective:
        adapter_dir.mkdir(parents=True, exist_ok=True)
        adapters = []
        records = run_native_multiobjective_experiments(profile.contract, routes, output_dir / "experiments")
    elif profile.contract.status == "ready" and native_retail:
        adapter_dir.mkdir(parents=True, exist_ok=True)
        adapters = []
        records = run_native_retail_experiments(profile.contract, routes, output_dir / "experiments")
    elif profile.contract.status == "ready" and native_robust:
        adapter_dir.mkdir(parents=True, exist_ok=True)
        adapters = []
        records = run_native_robust_experiments(profile.contract, routes, output_dir / "experiments")
    elif profile.contract.status == "ready" and native_optimization:
        adapter_dir.mkdir(parents=True, exist_ok=True)
        adapters = []
        records = run_native_experiments(profile.contract, routes, output_dir / "experiments")
    elif profile.contract.status == "ready" and native_mechanism:
        adapter_dir.mkdir(parents=True, exist_ok=True)
        adapters = []
        records = run_native_mechanism_experiments(profile.contract, routes, output_dir / "experiments")
    elif profile.contract.status == "ready" and native_simulation:
        adapter_dir.mkdir(parents=True, exist_ok=True)
        adapters = []
        records = run_native_simulation_experiments(profile.contract, routes, output_dir / "experiments")
    elif profile.contract.status == "ready" and native_spatial:
        adapter_dir.mkdir(parents=True, exist_ok=True)
        adapters = []
        records = run_native_spatial_experiments(profile.contract, routes, output_dir / "experiments")
    elif profile.contract.status == "ready" and specialist_authorized:
        adapters = generate_adapters(profile, routes, adapter_dir)
        # Contract-declared specialist code is reviewed input, never an auto-repair target.
        records = run_experiments(
            adapters,
            output_dir / "experiments",
            timeout_seconds=args.timeout_seconds,
            max_repairs=0 if profile.contract.specialist_routes else 1,
        )
    else:
        adapter_dir.mkdir(parents=True, exist_ok=True)
        adapters = []
        records = []
    registry_path = write_registry(records, output_dir / "registry")
    domain_verification = None
    if native_multiobjective and records:
        domain_verification = verify_native_multiobjective(profile.contract, records, registry_path, output_dir / "domain_verification")
    elif native_retail and records:
        domain_verification = verify_native_retail(profile.contract, records, registry_path, output_dir / "domain_verification")
    elif native_robust and records:
        domain_verification = verify_native_robust(profile.contract, records, registry_path, output_dir / "domain_verification")
    elif native_optimization and records:
        domain_verification = verify_native_optimization(profile.contract, records, registry_path, output_dir / "domain_verification")
    elif native_mechanism and records:
        domain_verification = verify_native_mechanism(profile.contract, records, registry_path, output_dir / "domain_verification")
    elif native_simulation and records:
        domain_verification = verify_native_simulation(profile.contract, records, registry_path, output_dir / "domain_verification")
    elif native_spatial and records:
        domain_verification = verify_native_spatial(profile.contract, records, registry_path, output_dir / "domain_verification")
    elif profile.contract.task_type in {"forecasting", "regression"} and records:
        domain_verification = verify_native_predictive(profile.contract, records, registry_path, output_dir / "domain_verification")
    elif profile.contract.domain_verifier and records:
        domain_verification = run_domain_verifier(
            Path(profile.contract.domain_verifier),
            profile.contract,
            registry_path,
            output_dir / "domain_verification",
            timeout_seconds=args.timeout_seconds,
        )
    final = judge_solution(profile, routes, records, output_dir / "judge", domain_verification=domain_verification)
    if profile.contract.specialist_routes and not specialist_authorized:
        final["flags"].append("specialist_code_not_authorized")
        final["next_actions"] = ["Review the contract-declared scripts, then rerun with --allow-specialist-code."]
        write_json(output_dir / "judge" / "final_judge.json", final)
        (output_dir / "judge" / "final_judge.md").write_text(_format_md(final), encoding="utf-8")
    workflow_exports = export_workflow_results(
        profile,
        routes,
        records,
        final,
        workspace_paths,
        adapter_dir,
        output_dir / "experiments",
    )

    summary = {
        "version": "v44",
        "problem_id": args.problem_id,
        "data_root": str(args.data_root),
        "brief": str(args.brief or ""),
        "contract_path": str(contract_path),
        "contract_status": profile.contract.status,
        "specialist_code_authorized": specialist_authorized,
        "profile": asdict(profile),
        "corpus_hits": [asdict(hit) for hit in hits],
        "routes": [asdict(route) for route in routes],
        "adapter_count": len(adapters),
        "executed_route_count": sum(1 for record in records if record.status == "executed"),
        "failed_route_count": sum(1 for record in records if record.status != "executed"),
        "registry_path": str(registry_path),
        "final_judge": final,
        "domain_verification": domain_verification,
        "workspace_paths": workspace_paths,
        "data_audit_path": str(Path(workspace_paths["work/data_audit"]) / "data_audit.json"),
        "data_assembly": data_assembly,
        "solving_plan_path": str(plan_path),
        "workflow_exports": workflow_exports,
        "output_dir": str(output_dir),
        "entrypoint": "run_v39_solver.py",
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    summary = run_solver(parse_args())
    final = summary["final_judge"]
    print(json.dumps({
        "version": summary["version"],
        "problem_id": summary["problem_id"],
        "adapter_count": summary["adapter_count"],
        "executed_route_count": summary["executed_route_count"],
        "failed_route_count": summary["failed_route_count"],
        "verdict": final["verdict"],
        "recommended_route": final["recommended_route"],
        "output_dir": summary["output_dir"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
