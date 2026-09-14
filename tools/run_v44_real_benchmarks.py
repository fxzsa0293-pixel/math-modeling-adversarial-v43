from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from solver.schemas import ProblemContract
from workflow.retail_model_builder import build_cumcm2023c_item_retail_model, build_cumcm2023c_retail_model
from workflow.retail_analysis import analyze_cumcm2023c_sales, verify_retail_analysis
from workflow.cumcm2023c_report import build_cumcm2023c_report, verify_cumcm2023c_report
from workflow.category_forecast import build_category_forecast, verify_category_forecast
from workflow.problem_dag import WorkflowNode, build_workflow_manifest, verify_workflow_manifest

TOOLS = Path(__file__).resolve().parent
SKILL_ROOT = TOOLS.parent


def _find_2023c(root: Path) -> tuple[Path, Path, Path, Path]:
    candidates = list(root.rglob("附件2.xlsx"))
    for sales in candidates:
        product = sales.with_name("附件1.xlsx")
        wholesale = sales.with_name("附件3.xlsx")
        loss = sales.with_name("附件4.xlsx")
        if product.is_file() and wholesale.is_file() and loss.is_file() and "2023" in str(sales) and ("C题" in str(sales) or "2023C" in str(sales)):
            return product.resolve(), sales.resolve(), wholesale.resolve(), loss.resolve()
    raise FileNotFoundError("CUMCM2023C attachments 1 through 4 were not found")


def _vegetable_contract(product: Path, sales: Path) -> ProblemContract:
    return ProblemContract(
        version="v44", problem_id="CUMCM2023C_daily_sales_forecast", task_type="forecasting", status="ready",
        data_file=str(sales),
        data_sources=[{"alias": "sales", "path": str(sales), "sheet_name": "Sheet1", "columns": ["销售日期", "单品编码", "销量(千克)"]}, {"alias": "products", "path": str(product), "sheet_name": "Sheet1", "columns": ["单品编码"]}],
        base_table="sales",
        data_joins=[{"left": "sales", "right": "products", "on": "单品编码", "right_columns": [], "how": "left", "validate": "many_to_one", "max_unmatched_left_rate": 0.0}],
        data_aggregation={"group_by": ["销售日期"], "aggregations": {"日销量": {"column": "销量(千克)", "function": "sum"}, "在售单品数": {"column": "单品编码", "function": "nunique"}}, "sort_by": ["销售日期"]},
        target_column="日销量", time_column="销售日期", forecast_horizon=7, validation_mode="fixed_origin_holdout",
        metric_directions={"MAE": "lower_better", "RMSE": "lower_better", "WAPE": "lower_better"},
        expected_outputs=["seven_day_daily_sales_forecast", "route_comparison", "paper_results_section"], source="provided",
        notes=["Real CUMCM2023C attachments; this benchmark covers daily aggregate forecasting, not category-level replenishment or pricing."],
    )


def _retail_contract(build: dict) -> ProblemContract:
    return ProblemContract(
        version="v44", problem_id="CUMCM2023C_category_replenishment_pricing", task_type="constrained_optimization", status="ready",
        data_file=str(build["data_file"]), retail_decision_model=build["model"], retail_decision_manifest=str(build["manifest_path"]),
        metric_directions={"objective": "lower_better", "expected_profit": "higher_better", "constraint_violation": "lower_better"},
        expected_outputs=["seven_day_category_replenishment", "seven_day_category_pricing", "scenario_profit", "paper_results_section"],
        source="provided", notes=["Derived only from CUMCM2023C attachments 1-4 through 2023-06-30; decisions cover 2023-07-01 through 2023-07-07."],
    )


def _item_retail_contract(build: dict) -> ProblemContract:
    return ProblemContract(
        version="v44", problem_id="CUMCM2023C_item_replenishment_pricing", task_type="constrained_optimization", status="ready",
        data_file=str(build["data_file"]), retail_decision_model=build["model"], retail_decision_manifest=str(build["manifest_path"]),
        metric_directions={"objective": "lower_better", "expected_profit": "higher_better", "constraint_violation": "lower_better"},
        expected_outputs=["july_1_item_assortment", "item_replenishment", "item_pricing", "category_coverage", "paper_results_section"],
        source="provided", notes=["Candidates are restricted to items sold during 2023-06-24 through 2023-06-30; 27-33 selected items, minimum display quantity 2.5 kg, all six categories represented."],
    )


def _run_case(case_id: str, contract: ProblemContract, data_root: Path, case_dir: Path, max_routes: int) -> dict:
    case_dir.mkdir(parents=True, exist_ok=True)
    contract_path = case_dir / "benchmark_contract.json"
    contract_path.write_text(json.dumps(asdict(contract), ensure_ascii=False, indent=2), encoding="utf-8")
    run_dir = case_dir / "run"
    started = time.perf_counter()
    process = subprocess.run([sys.executable, str(TOOLS / "run_v39_solver.py"), "--problem-id", case_id, "--data-root", str(data_root), "--contract", str(contract_path), "--output-dir", str(run_dir), "--max-routes", str(max_routes)], text=True, encoding="utf-8", capture_output=True, timeout=900)
    elapsed = time.perf_counter() - started
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    return {"case_id": case_id, "status": "executed" if process.returncode == 0 else "failed", "verdict": summary.get("final_judge", {}).get("verdict"), "claim_level": summary.get("final_judge", {}).get("claim_level"), "executed_routes": summary.get("executed_route_count", 0), "elapsed_seconds": elapsed, "summary_path": str(summary_path), "stdout": process.stdout[-2000:], "stderr": process.stderr[-2000:]}


def run_benchmarks(reference_root: Path, output_dir: Path, max_routes: int = 4) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    try:
        product, sales, wholesale, loss = _find_2023c(reference_root)
        analysis_dir = output_dir / "CUMCM2023C_sales_distribution_relationships"
        started = time.perf_counter()
        analysis = analyze_cumcm2023c_sales(product, sales, analysis_dir)
        verification = verify_retail_analysis(Path(analysis["manifest_path"]))
        cases.append({"case_id": "CUMCM2023C_sales_distribution_relationships", "status": "executed", "verdict": "pass" if verification["passed"] else "revise", "claim_level": verification["claim_level"], "executed_routes": 1, "elapsed_seconds": time.perf_counter() - started, "summary_path": analysis["manifest_path"], "verification": verification})
        cases.append(_run_case("CUMCM2023C_daily_sales_forecast", _vegetable_contract(product, sales), sales.parent, output_dir / "CUMCM2023C_daily_sales_forecast", max_routes))
        retail_dir = output_dir / "CUMCM2023C_category_replenishment_pricing"
        category_forecast = build_category_forecast(product, sales, retail_dir / "category_forecast")
        category_forecast_verification = verify_category_forecast(Path(category_forecast["manifest_path"]))
        if not category_forecast_verification["passed"]:
            raise ValueError(f"category_forecast_verification_failed:{category_forecast_verification['issues']}")
        build = build_cumcm2023c_retail_model(product, sales, wholesale, loss, retail_dir / "derived_inputs", category_forecast_manifest=Path(category_forecast["manifest_path"]))
        cases.append(_run_case("CUMCM2023C_category_replenishment_pricing", _retail_contract(build), Path(build["data_file"]).parent, retail_dir, max_routes))
        cases[-1]["upstream_category_forecast"] = category_forecast_verification
        item_dir = output_dir / "CUMCM2023C_item_replenishment_pricing"
        item_build = build_cumcm2023c_item_retail_model(product, sales, wholesale, loss, item_dir / "derived_inputs", category_forecast_manifest=Path(category_forecast["manifest_path"]))
        cases.append(_run_case("CUMCM2023C_item_replenishment_pricing", _item_retail_contract(item_build), Path(item_build["data_file"]).parent, item_dir, max_routes))
    except Exception as exc:
        cases.append({"case_id": "CUMCM2023C_pipeline", "status": "blocked", "error": str(exc)})
    report = {"version": "v44", "suite": "current_pipeline_real_data", "case_count": len(cases), "passed_cases": sum(item.get("verdict") == "pass" for item in cases), "all_passed": bool(cases) and all(item.get("verdict") == "pass" for item in cases), "cases": cases}
    summary_path = output_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if report["all_passed"] and len(cases) == 4:
        try:
            report["complete_report"] = build_cumcm2023c_report(summary_path, output_dir / "CUMCM2023C_complete_report.md")
            report_path = Path(report["complete_report"]["path"])
            report["complete_report"]["sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
            summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            report["complete_report"]["verification"] = verify_cumcm2023c_report(summary_path)
            if not report["complete_report"]["verification"]["passed"]:
                raise ValueError(f"complete_report_verification_failed:{report['complete_report']['verification']['issues']}")
            analysis_case, forecast_case, category_case, item_case = cases
            forecast_manifest_path = category_case["upstream_category_forecast"]["manifest_path"]
            nodes = [
                WorkflowNode("problem_1_analysis", "analysis", output_artifacts={"analysis_manifest": analysis_case["summary_path"]}, verification=analysis_case["verification"]),
                WorkflowNode("aggregate_forecast", "prediction", contract_path=str((output_dir / "CUMCM2023C_daily_sales_forecast" / "benchmark_contract.json").resolve()), output_artifacts={"forecast_summary": forecast_case["summary_path"]}, verification={"passed": forecast_case["verdict"] == "pass"}),
                WorkflowNode("category_forecast", "prediction", output_artifacts={"category_forecast_manifest": forecast_manifest_path}, verification=category_case["upstream_category_forecast"]),
                WorkflowNode("problem_2_optimization", "optimization", dependencies=["category_forecast"], contract_path=str((output_dir / "CUMCM2023C_category_replenishment_pricing" / "benchmark_contract.json").resolve()), input_artifacts={"category_forecast_manifest": forecast_manifest_path}, output_artifacts={"problem_2_summary": category_case["summary_path"]}, verification={"passed": category_case["verdict"] == "pass"}),
                WorkflowNode("problem_3_optimization", "optimization", dependencies=["category_forecast"], contract_path=str((output_dir / "CUMCM2023C_item_replenishment_pricing" / "benchmark_contract.json").resolve()), input_artifacts={"category_forecast_manifest": forecast_manifest_path}, output_artifacts={"problem_3_summary": item_case["summary_path"]}, verification={"passed": item_case["verdict"] == "pass"}),
                WorkflowNode("complete_report", "paper", dependencies=["problem_1_analysis", "aggregate_forecast", "problem_2_optimization", "problem_3_optimization"], input_artifacts={"analysis_manifest": analysis_case["summary_path"], "forecast_summary": forecast_case["summary_path"], "problem_2_summary": category_case["summary_path"], "problem_3_summary": item_case["summary_path"]}, output_artifacts={"report": str(report_path)}, verification=report["complete_report"]["verification"]),
            ]
            report["workflow_dag"] = build_workflow_manifest("CUMCM2023C", nodes, output_dir / "CUMCM2023C_workflow_manifest.json")
            report["workflow_dag"]["verification"] = verify_workflow_manifest(Path(report["workflow_dag"]["manifest_path"]))
            if not report["workflow_dag"]["passed"] or not report["workflow_dag"]["verification"]["passed"]:
                raise ValueError(f"workflow_dag_verification_failed:{report['workflow_dag']['verification']['issues']}")
        except Exception as exc:
            report["all_passed"] = False
            report["report_error"] = str(exc)
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real-data benchmarks through the current V44 contract and gate pipeline.")
    parser.add_argument("--reference-root", type=Path, default=SKILL_ROOT / "data" / "reference_root")
    parser.add_argument("--output-dir", type=Path, default=TOOLS / "run" / "v44_real_benchmarks")
    parser.add_argument("--max-routes", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run_benchmarks(args.reference_root.resolve(), args.output_dir.resolve(), args.max_routes), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
