"""Unified CLI for the V38 benchmark and V44 solver.

Usage:
    python run.py --self-contained
    python run.py --data-root path/to/reference_root
    python run.py --v39-solver --data-root path/to/problem_data
    python run.py --auto-solver --data-root path/to/problem_data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from v30_real_benchmark_adapters import run_v30_real_benchmark_adapters
from v32_forecasting_adapter import run_v32_forecasting_adapter
from v34_mechanism_spatial_adapters import run_v34_mechanism_spatial_adapters
from v35_c_topic_specialization import run_v35_c_topic_specialization
from v36_resource_backed_c_adapters import run_v36_resource_backed_c_adapters
from v38_explicit_role_gate import run_v38_explicit_role_gate
from run_v39_solver import main as run_v39_main

TOOLS = Path(__file__).resolve().parent
SKILL_ROOT = TOOLS.parent
DEFAULT_DATA_ROOT = SKILL_ROOT / "data" / "reference_root"

ROUTE_CONTRACT: Dict[str, Dict[str, str]] = {
    "CUMCM2015B_taxi_subsidy_realdata": {
        "no_subsidy_baseline": "baseline",
        "peak_flat_subsidy": "candidate",
        "scarcity_index_subsidy": "candidate",
    },
    "CUMCM2016B_open_community_road_realdata": {
        "closed_community_baseline": "baseline",
        "mean_improvement_opening": "candidate",
        "risk_adjusted_opening": "candidate",
    },
    "CUMCM2003B_open_pit_truck_assignment_realdata": {
        "greedy_repair": "baseline",
        "relaxed_lp": "candidate",
        "integer_repair": "candidate",
    },
    "CUMCM2016D_wind_power_forecasting_v32": {
        "naive_persistence": "baseline",
        "seasonal_naive_96": "strong_baseline",
        "rolling_mean_96points": "strong_baseline",
        "ewma_alpha_02": "candidate",
        "ridge_lag_wind": "candidate",
        "ridge_poly_lag_wind": "candidate",
    },
    "CUMCM2018A_high_temperature_clothing_mechanism_v34": {
        "constant_skin_baseline": "baseline",
        "linear_warmup_baseline": "strong_baseline",
        "first_order_RC_mechanism": "candidate",
    },
    "CUMCM2012B_solar_house_spatial_multiobjective_v34": {
        "east_orientation_baseline": "baseline",
        "north_orientation_negative_control": "negative_control",
        "west_orientation_candidate": "candidate",
        "south_orientation_strong": "candidate",
        "pareto_component_selection": "candidate",
    },
    "CUMCM2012C_stroke_environment_statistics_v35": {
        "monthly_mean_baseline": "baseline",
        "month_trend_baseline": "strong_baseline",
        "ridge_environment_cv": "candidate",
    },
    "CUMCM2013C_ancient_tower_deformation_v35": {
        "centroid_vertical_baseline": "baseline",
        "PCA_centerline_deformation": "candidate",
    },
    "CUMCM2014C_pig_operation_forecast_optimization_v35": {
        "mean_price_baseline": "baseline",
        "naive_last_price": "strong_baseline",
        "moving_average_3": "candidate",
        "moving_average_5": "candidate",
        "moving_average_10": "candidate",
        "margin_max_operation_policy": "ablation",
        "forecast_then_margin_policy": "composite",
    },
    "CUMCM2012A_wine_evaluation_ranking_v36": {
        "first_panel_baseline": "baseline",
        "equal_judge_average": "candidate",
        "reliability_weighted_panel": "candidate",
        "score_stability_topsis": "candidate",
    },
    "CUMCM2023C_vegetable_retail_forecast_operation_v36": {
        "mean_7day_baseline": "baseline",
        "naive_last_day": "strong_baseline",
        "moving_average_14": "candidate",
        "weekday_profile": "candidate",
        "top_category_revenue_mix_policy": "ablation",
        "forecast_then_revenue_mix_policy": "composite",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V38 benchmark gate or V44 automatic solver pipeline.")
    parser.add_argument("--v39-solver", action="store_true", help="Compatibility alias for --auto-solver.")
    parser.add_argument("--auto-solver", action="store_true", help="Run V44 automatic solver workflow instead of V38 benchmark gate.")
    parser.add_argument("--workflow", type=Path, default=None, help="Execute a V44 multi-question workflow JSON.")
    parser.add_argument("--no-workflow-resume", action="store_true", help="Force workflow execution even if the prior complete DAG still verifies.")
    parser.add_argument("--problem-id", default="demo_problem", help="Stable problem id for V44 solver mode.")
    parser.add_argument("--brief", type=Path, default=None, help="Optional problem statement for V44 solver mode.")
    parser.add_argument("--contract", type=Path, default=None, help="Optional V44 problem contract JSON.")
    parser.add_argument("--corpus-root", type=Path, default=None, help="Optional corpus root for V44 solver mode.")
    parser.add_argument("--max-routes", type=int, default=8, help="Maximum routes in V44 solver mode.")
    parser.add_argument("--timeout-seconds", type=int, default=60, help="Per-adapter timeout in V44 solver mode.")
    parser.add_argument("--allow-specialist-code", action="store_true", help="Execute reviewed specialist scripts declared by the contract.")
    parser.add_argument("--self-contained", action="store_true", help="Use bundled data/reference_root.")
    parser.add_argument("--data-root", type=Path, default=None, help="Custom reference_root. Overrides bundled data.")
    parser.add_argument("--output-dir", type=Path, default=TOOLS / "run" / "v38_self_contained", help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workflow:
        from workflow.workflow_executor import execute_workflow

        workflow_output = args.output_dir
        if workflow_output == TOOLS / "run" / "v38_self_contained":
            workflow_output = TOOLS / "run" / "v44_workflow"
        result = execute_workflow(
            args.workflow, workflow_output,
            default_data_root=args.data_root, default_corpus_root=args.corpus_root,
            max_routes=args.max_routes, timeout_seconds=args.timeout_seconds,
            allow_specialist_code=args.allow_specialist_code,
            resume=not args.no_workflow_resume,
        )
        print(json.dumps({
            "version": result["version"], "problem_id": result["problem_id"],
            "passed": result["passed"], "topological_order": result["topological_order"],
            "manifest_path": result["manifest"]["manifest_path"],
            "output_dir": str(workflow_output.resolve()),
        }, ensure_ascii=False, indent=2))
        return
    if args.v39_solver or args.auto_solver:
        import sys
        solver_output_dir = args.output_dir
        if solver_output_dir == TOOLS / "run" / "v38_self_contained":
            solver_output_dir = TOOLS / "run" / "v44_auto_solver"
        argv = ["run_v39_solver.py", "--problem-id", args.problem_id, "--data-root", str(args.data_root or DEFAULT_DATA_ROOT), "--output-dir", str(solver_output_dir)]
        if args.brief:
            argv.extend(["--brief", str(args.brief)])
        if args.contract:
            argv.extend(["--contract", str(args.contract)])
        if args.corpus_root:
            argv.extend(["--corpus-root", str(args.corpus_root)])
        argv.extend(["--max-routes", str(args.max_routes)])
        argv.extend(["--timeout-seconds", str(args.timeout_seconds)])
        if args.allow_specialist_code:
            argv.append("--allow-specialist-code")
        old_argv = sys.argv
        try:
            sys.argv = argv
            run_v39_main()
        finally:
            sys.argv = old_argv
        return
    data_root = args.data_root or DEFAULT_DATA_ROOT
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    v30 = run_v30_real_benchmark_adapters(data_root, output_dir / "v30")
    v32 = run_v32_forecasting_adapter([data_root], output_dir / "v32")
    v34 = run_v34_mechanism_spatial_adapters([data_root], output_dir / "v34")
    v35 = run_v35_c_topic_specialization([data_root], output_dir / "v35")
    v36 = run_v36_resource_backed_c_adapters([data_root], output_dir / "v36")
    gate = run_v38_explicit_role_gate([v30, v32, v34, v35, v36], ROUTE_CONTRACT, output_dir / "v38")

    summary = {
        "version": "v38",
        "data_root": str(data_root),
        "case_count": len(gate.get("case_audits", [])),
        "surviving_route_count": gate.get("surviving_route_count"),
        "eliminated_route_count": gate.get("eliminated_route_count"),
        "gate": gate.get("gate"),
        "output_dir": str(output_dir),
        "entrypoint": "run.py",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
