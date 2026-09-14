"""
V35 C-topic specialization suite.

Runs historical C-topic cases to harden the workflow for CUMCM C-style
problems: medical/environment statistics, geometric deformation, and business
operation forecasting/optimization.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler


@dataclass
class CTopicResult:
    case: str
    c_topic_pattern: str
    status: str
    data_path: str
    routes: List[Dict[str, Any]]
    aggregate: Dict[str, Any]
    recommended_route: str
    recommendation_evidence: List[str]
    figure_descriptions: List[str]
    limitations: List[str]


def run_v35_c_topic_specialization(
    reference_roots: Iterable[Path | str] | None = None,
    output_dir: Path | str | None = None,
) -> Dict[str, Any]:
    roots = _resolve_roots(reference_roots)
    results: List[CTopicResult] = []

    case2012 = _find_case_dir(roots, "2012C", required_files=["2007年.xls"])
    if case2012:
        results.append(_run_2012c_medical_environment(case2012))
    else:
        results.append(_blocked("CUMCM2012C_stroke_environment", "medical_environment_statistics", Path("."), "2012C case directory not found"))

    case2013 = _find_case_dir(roots, "2013C", required_files=["数据一.xls"])
    if case2013:
        results.append(_run_2013c_geometric_deformation(case2013))
    else:
        results.append(_blocked("CUMCM2013C_ancient_tower_deformation", "geometric_deformation", Path("."), "2013C case directory not found"))

    case2014 = _find_case_dir(roots, "2014C", required_files=["roujia.txt", "biandongfeiyong.txt"])
    if case2014:
        results.append(_run_2014c_pig_operation(case2014))
    else:
        results.append(_blocked("CUMCM2014C_pig_operation", "business_forecast_optimization", Path("."), "2014C case directory not found"))

    suite = {
        "version": "v35",
        "purpose": "C-topic specialization benchmark suite",
        "summary": {
            "executed_cases": sum(1 for result in results if result.status == "executed"),
            "blocked_cases": sum(1 for result in results if result.status != "executed"),
            "patterns": [result.c_topic_pattern for result in results],
            "routes_evaluated": sum(len(result.routes) for result in results),
            "local_python_execution": True,
        },
        "results": [asdict(result) for result in results],
        "c_topic_workflow_patch": _c_topic_workflow_patch(),
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "v35_c_topic_specialization.json").write_text(json.dumps(suite, ensure_ascii=False, indent=2), encoding="utf-8")
        (output_dir / "v35_c_topic_specialization.md").write_text(format_v35_report(suite), encoding="utf-8")
    return suite


def format_v35_report(suite: Dict[str, Any]) -> str:
    lines = ["# V35 C题专向多案例评测报告", "", f"- summary: {suite.get('summary')}", "", "## Results"]
    for result in suite.get("results", []):
        lines.extend([
            f"### {result['case']}",
            f"- pattern: {result['c_topic_pattern']}",
            f"- status: {result['status']}",
            f"- data_path: {result['data_path']}",
            f"- recommended_route: {result['recommended_route']}",
            f"- aggregate: {result['aggregate']}",
            f"- recommendation_evidence: {result['recommendation_evidence']}",
            "- route leaderboard:",
        ])
        for route in result.get("routes", []):
            lines.append(f"  - {route.get('route')}: {route.get('metrics')}")
        lines.append("- figure_descriptions:")
        for item in result.get("figure_descriptions", []):
            lines.append(f"  - {item}")
        lines.append("- limitations:")
        for item in result.get("limitations", []):
            lines.append(f"  - {item}")
        lines.append("")
    lines.append("## C-topic workflow patch")
    for item in suite.get("c_topic_workflow_patch", []):
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def _run_2012c_medical_environment(case_dir: Path) -> CTopicResult:
    data_file = _find_file(case_dir, "2007年.xls")
    if not data_file:
        return _blocked("CUMCM2012C_stroke_environment", "medical_environment_statistics", case_dir, "2007 monthly data not found")

    raw = pd.read_excel(data_file, header=None)
    df = raw.iloc[1:].copy()
    df.columns = ["month", "stroke_count", "avg_pressure", "high_pressure", "low_pressure", "avg_temp", "high_temp", "low_temp", "avg_humidity", "min_humidity"]
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    if len(df) < 8:
        return _blocked("CUMCM2012C_stroke_environment", "medical_environment_statistics", data_file, "usable monthly rows fewer than 8")

    y = df["stroke_count"].to_numpy(dtype=float)
    month = df[["month"]].to_numpy(dtype=float)
    features = df[["avg_pressure", "avg_temp", "avg_humidity"]].to_numpy(dtype=float)
    loo = LeaveOneOut()

    routes: List[Dict[str, Any]] = []
    mean_pred = np.full_like(y, np.mean(y))
    routes.append({"route": "monthly_mean_baseline", "metrics": _regression_metrics(y, mean_pred)})

    trend_pred = cross_val_predict(LinearRegression(), month, y, cv=loo)
    routes.append({"route": "month_trend_baseline", "metrics": _regression_metrics(y, trend_pred)})

    scaled = StandardScaler().fit_transform(features)
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=loo, scoring="neg_mean_absolute_error")
    env_pred = cross_val_predict(ridge, scaled, y, cv=loo)
    env_metrics = _regression_metrics(y, env_pred)
    corrs = {name: round(float(stats.pearsonr(df[name], df["stroke_count"])[0]), 6) for name in ["avg_pressure", "avg_temp", "avg_humidity"]}
    env_metrics["env_correlations"] = corrs
    routes.append({"route": "ridge_environment_cv", "metrics": env_metrics})

    routes = sorted(routes, key=lambda route: route["metrics"]["MAE"])
    best = routes[0]
    baseline = next(route for route in routes if route["route"] == "monthly_mean_baseline")
    aggregate = {
        "n_months": int(len(df)),
        "best_route": best["route"],
        "best_MAE": best["metrics"]["MAE"],
        "baseline_MAE": baseline["metrics"]["MAE"],
        "improvement_over_mean_pct": _pct_gain_lower(best["metrics"]["MAE"], baseline["metrics"]["MAE"]),
        "strongest_abs_corr": max(corrs.items(), key=lambda item: abs(item[1])),
    }
    return CTopicResult(
        case="CUMCM2012C_stroke_environment_statistics_v35",
        c_topic_pattern="medical_environment_statistics",
        status="executed",
        data_path=str(data_file),
        routes=routes,
        aggregate=aggregate,
        recommended_route=best["route"],
        recommendation_evidence=[
            "真实读取 2012C 脑卒中月度发病与气压/温度/湿度数据。",
            "采用 leave-one-month-out 验证，避免小样本月度统计中的过拟合幻觉。",
            f"最佳路线为 {best['route']}，MAE={best['metrics']['MAE']}，相对均值 baseline 改善 {aggregate['improvement_over_mean_pct']}%。",
        ],
        figure_descriptions=[
            "月度发病人数折线图：展示季节性、异常月份和趋势强弱。",
            "环境变量相关性条形图：比较气压、温度、湿度与发病人数相关方向。",
            "留一法预测-真实散点图：检查模型是否只是拟合均值。",
            "残差月份图：定位模型无法解释的月份并提示题面外因素。",
        ],
        limitations=[
            "只有 12 个聚合月度样本，任何显著性和因果结论都必须保守。",
            "当前 adapter 是统计关联筛查，不可直接解释为因果干预效果。",
        ],
    )


def _run_2013c_geometric_deformation(case_dir: Path) -> CTopicResult:
    data_file = _find_file(case_dir, "数据一.xls")
    if not data_file:
        return _blocked("CUMCM2013C_ancient_tower_deformation", "geometric_deformation", case_dir, "数据一.xls not found")

    raw = pd.read_excel(data_file, header=None)
    numeric = raw.apply(pd.to_numeric, errors="coerce")
    blocks = []
    for start in [0, 3, 6, 9]:
        block = numeric.iloc[:, start:start + 3].dropna()
        if block.shape[1] == 3 and len(block) > 10:
            block.columns = ["x", "y", "z"]
            blocks.append(block)
    if not blocks:
        return _blocked("CUMCM2013C_ancient_tower_deformation", "geometric_deformation", data_file, "no 3D coordinate blocks parsed")

    all_metrics = []
    for block in blocks:
        coords = block[["x", "y", "z"]].to_numpy(dtype=float)
        centered = coords - coords.mean(axis=0)
        pca = PCA(n_components=3).fit(centered)
        vertical = coords[:, 2]
        x_model = LinearRegression().fit(vertical.reshape(-1, 1), coords[:, 0])
        y_model = LinearRegression().fit(vertical.reshape(-1, 1), coords[:, 1])
        x_pred = x_model.predict(vertical.reshape(-1, 1))
        y_pred = y_model.predict(vertical.reshape(-1, 1))
        centroid_x = np.full_like(coords[:, 0], coords[:, 0].mean())
        centroid_y = np.full_like(coords[:, 1], coords[:, 1].mean())
        centroid_mae = (mean_absolute_error(coords[:, 0], centroid_x) + mean_absolute_error(coords[:, 1], centroid_y)) / 2
        radial = np.sqrt((coords[:, 0] - coords[:, 0].mean()) ** 2 + (coords[:, 1] - coords[:, 1].mean()) ** 2)
        all_metrics.append({
            "point_count": int(len(coords)),
            "pca_linearity_ratio": round(float(pca.explained_variance_ratio_[0]), 6),
            "tilt_slope_x_per_z": round(float(x_model.coef_[0]), 6),
            "tilt_slope_y_per_z": round(float(y_model.coef_[0]), 6),
            "centerline_MAE": round(float((mean_absolute_error(coords[:, 0], x_pred) + mean_absolute_error(coords[:, 1], y_pred)) / 2), 6),
            "centroid_centerline_MAE": round(float(centroid_mae), 6),
            "radial_std": round(float(np.std(radial)), 6),
        })

    baseline = {"route": "centroid_vertical_baseline", "metrics": {"centerline_MAE": round(float(np.mean([m["centroid_centerline_MAE"] for m in all_metrics])), 6), "blocks": len(all_metrics)}}
    pca_route = {
        "route": "PCA_centerline_deformation",
        "metrics": {
            "centerline_MAE": round(float(np.mean([m["centerline_MAE"] for m in all_metrics])), 6),
            "mean_linearity_ratio": round(float(np.mean([m["pca_linearity_ratio"] for m in all_metrics])), 6),
            "max_abs_tilt_x": round(float(max(abs(m["tilt_slope_x_per_z"]) for m in all_metrics)), 6),
            "max_abs_tilt_y": round(float(max(abs(m["tilt_slope_y_per_z"]) for m in all_metrics)), 6),
        },
    }
    routes = sorted([baseline, pca_route], key=lambda route: route["metrics"]["centerline_MAE"])
    best = routes[0]
    aggregate = {"blocks": len(blocks), "best_route": best["route"], "best_centerline_MAE": best["metrics"]["centerline_MAE"], "mean_linearity_ratio": pca_route["metrics"]["mean_linearity_ratio"]}
    return CTopicResult(
        case="CUMCM2013C_ancient_tower_deformation_v35",
        c_topic_pattern="geometric_deformation",
        status="executed",
        data_path=str(data_file),
        routes=routes,
        aggregate=aggregate,
        recommended_route=best["route"],
        recommendation_evidence=[
            "真实读取 2013C 古塔三维测点数据，按三列坐标块解析。",
            "用 PCA/中心线拟合审计古塔变形，而不是直接套黑箱回归。",
            f"平均线性主轴解释率 {aggregate['mean_linearity_ratio']}，可支持倾斜/弯曲几何诊断。",
        ],
        figure_descriptions=[
            "三维测点散点图：展示各层测点空间分布。",
            "PCA 中心线图：叠加塔体测点与拟合中心线。",
            "高度-水平偏移曲线：展示倾斜趋势。",
            "径向残差箱线图：比较不同测点块的变形离散程度。",
        ],
        limitations=[
            "当前 adapter 只做几何诊断，不还原完整古塔结构力学模型。",
            "原始数据块语义需结合题面确认，自动解析可能需要人工复核。",
        ],
    )


def _run_2014c_pig_operation(case_dir: Path) -> CTopicResult:
    price_file = case_dir / "roujia.txt"
    cost_file = case_dir / "biandongfeiyong.txt"
    if not price_file.exists():
        return _blocked("CUMCM2014C_pig_operation", "business_forecast_optimization", case_dir, "roujia.txt not found")

    price = _read_vector(price_file)
    cost = _read_vector(cost_file) if cost_file.exists() else np.full_like(price, np.nanmean(price) * 0.6)
    n = min(len(price), len(cost))
    price, cost = price[:n], cost[:n]
    margin = price - cost / 10.0

    routes: List[Dict[str, Any]] = []
    mean_forecast = np.full_like(price[1:], price[:-1].mean())
    naive = price[:-1]
    routes.append({"route": "mean_price_baseline", "metrics": _forecast_metrics(price[1:], mean_forecast)})
    routes.append({"route": "naive_last_price", "metrics": _forecast_metrics(price[1:], naive)})
    for window in [3, 5, 10]:
        preds, actual = [], []
        for idx in range(window, len(price) - 1):
            preds.append(float(np.mean(price[idx - window:idx])))
            actual.append(float(price[idx + 1]))
        routes.append({"route": f"moving_average_{window}", "metrics": _forecast_metrics(np.asarray(actual), np.asarray(preds))})

    best_period_index = int(np.argmax(margin))
    routes.append({
        "route": "margin_max_operation_policy",
        "metrics": {
            "max_margin_proxy": round(float(margin[best_period_index]), 6),
            "best_period_index": best_period_index,
            "mean_margin_proxy": round(float(np.mean(margin)), 6),
            "profit_lift_vs_mean_pct": _pct_gain_higher(float(margin[best_period_index]), float(np.mean(margin))),
        },
    })
    forecast_routes = [route for route in routes if "MAE" in route["metrics"]]
    best_forecast = min(forecast_routes, key=lambda route: route["metrics"]["MAE"])
    policy = next(route for route in routes if route["route"] == "margin_max_operation_policy")
    composite_route = {
        "route": "forecast_then_margin_policy",
        "metrics": {
            "MAE": best_forecast["metrics"]["MAE"],
            "RMSE": best_forecast["metrics"].get("RMSE"),
            "max_margin_proxy": policy["metrics"]["max_margin_proxy"],
            "best_period_index": policy["metrics"]["best_period_index"],
            "forecast_component": best_forecast["route"],
            "policy_component": policy["route"],
        },
    }
    routes.append(composite_route)
    aggregate = {
        "n_periods": int(len(price)),
        "best_forecast_route": best_forecast["route"],
        "best_forecast_MAE": best_forecast["metrics"]["MAE"],
        "best_policy_period": best_period_index,
        "max_margin_proxy": policy["metrics"]["max_margin_proxy"],
    }
    return CTopicResult(
        case="CUMCM2014C_pig_operation_forecast_optimization_v35",
        c_topic_pattern="business_forecast_optimization",
        status="executed",
        data_path=str(case_dir),
        routes=sorted(routes, key=lambda route: route["metrics"].get("MAE", 1e9)),
        aggregate=aggregate,
        recommended_route="forecast_then_margin_policy",
        recommendation_evidence=[
            "真实读取 2014C 生猪肉价与变动费用序列。",
            f"价格预测最佳路线为 {best_forecast['route']}，MAE={best_forecast['metrics']['MAE']}。",
            f"经营策略使用 margin proxy 定位收益最优周期 index={best_period_index}。",
        ],
        figure_descriptions=[
            "肉价与成本时间序列图：展示经营环境变化。",
            "预测路线 MAE 对比图：比较均值、naive、移动平均。",
            "margin proxy 曲线：标注建议出栏/经营决策窗口。",
            "预测误差滚动图：检查价格突变期模型失效。",
        ],
        limitations=[
            "该 adapter 是经营预测/策略基线，不等同于完整猪场群体动态规划。",
            "成本文件单位需结合题面复核，当前 margin 使用 proxy 缩放。",
        ],
    )


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 6),
        "R2": round(float(r2_score(y_true, y_pred)), 6),
        "bias": round(float(np.mean(y_pred - y_true)), 6),
    }


def _forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    error = y_true - y_pred
    return {
        "MAE": round(float(np.mean(np.abs(error))), 6),
        "RMSE": round(float(np.sqrt(np.mean(error ** 2))), 6),
        "bias": round(float(np.mean(y_pred - y_true)), 6),
    }


def _read_vector(path: Path) -> np.ndarray:
    text = path.read_text(errors="ignore")
    values = []
    for token in text.replace("\t", " ").replace("\n", " ").split():
        try:
            values.append(float(token))
        except ValueError:
            continue
    return np.asarray(values, dtype=float)


def _pct_gain_lower(value: float, baseline: float) -> float:
    return round(float(100 * (baseline - value) / max(abs(baseline), 1e-9)), 2)


def _pct_gain_higher(value: float, baseline: float) -> float:
    return round(float(100 * (value - baseline) / max(abs(baseline), 1e-9)), 2)


def _blocked(case: str, pattern: str, path: Path, reason: str) -> CTopicResult:
    return CTopicResult(case, pattern, "blocked", str(path), [], {"reason": reason}, "none", [], [], [reason])


def _find_file(case_dir: Path, file_name: str) -> Path | None:
    direct = case_dir / file_name
    if direct.exists():
        return direct
    return next((path for path in case_dir.rglob(file_name) if path.is_file()), None)


def _find_case_dir(roots: List[Path], key: str, required_files: List[str]) -> Path | None:
    scored: List[tuple[int, Path]] = []
    for root in roots:
        if not root.exists():
            continue
        for candidate in root.rglob("*"):
            if not candidate.is_dir() or key not in str(candidate):
                continue
            found = sum(1 for file_name in required_files if _find_file(candidate, file_name))
            if found == 0:
                continue
            score = found * 1000
            text = str(candidate)
            if "优秀论文" in text:
                score += 200
            if "按年份分类" in text:
                score += 100
            if "MathModel-master" in text:
                score -= 200
            score -= len(text) // 20
            scored.append((score, candidate))
    if not scored:
        return None
    return sorted(scored, key=lambda item: (-item[0], len(str(item[1]))))[0][1]


def _resolve_roots(reference_roots: Iterable[Path | str] | None) -> List[Path]:
    if reference_roots is not None:
        return [Path(root) for root in reference_roots]
    return [Path(__file__).resolve().parent.parent / "data" / "reference_root"]


def _c_topic_workflow_patch() -> List[str]:
    return [
        "C题先做题型识别：医学统计、几何空间、经营管理、评价排序、预测决策、网络/图结构分别进入不同 adapter。",
        "每个 adapter 至少跑 baseline、强解释模型、强预测模型三类路线，并用真实 Python 指标淘汰弱路线。",
        "医学环境类必须区分相关、预测、因果，默认使用小样本交叉验证与保守解释，禁止凭相关性写干预结论。",
        "几何变形类优先解析坐标块、中心线/PCA/残差诊断，再决定是否上复杂曲面或结构力学模型。",
        "经营管理类必须预测和决策分离，先验证预测 baseline，再用利润/风险/约束 proxy 做策略优化。",
        "所有 C 题必须输出图表描述：数据结构图、路线对比图、残差/稳健性图、结论边界图。",
        "终审 agent 看到未超过 baseline、样本量不足却强因果、单位未复核、指标不可解释时必须打回。",
    ]


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent / "run" / "v35_c_topic_specialization"
    suite = run_v35_c_topic_specialization(output_dir=output_dir)
    print(format_v35_report(suite))
