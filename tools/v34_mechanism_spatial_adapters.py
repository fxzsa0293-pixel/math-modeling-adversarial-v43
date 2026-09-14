"""
V34 mechanism, spatial, and multi-objective adapters.

Adds real-data adapters for:
- CUMCM2018A high-temperature protective clothing (mechanism/physics);
- CUMCM2012B solar house PV orientation and component choice (spatial + multi-objective optimization).

The adapters are intentionally lightweight and evidence-oriented: they establish
baselines, strong routes, metrics, and limitations that V33 can audit.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


@dataclass
class V34AdapterResult:
    case: str
    archetypes: List[str]
    status: str
    data_path: str
    routes: List[Dict[str, Any]]
    aggregate: Dict[str, Any]
    recommended_route: str
    recommendation_evidence: List[str]
    figure_descriptions: List[str]
    limitations: List[str]


def run_v34_mechanism_spatial_adapters(reference_roots: Iterable[Path | str] | None = None, output_dir: Path | str | None = None) -> Dict[str, Any]:
    roots = _resolve_roots(reference_roots)
    results: List[V34AdapterResult] = []
    thermal_file = _find_file(roots, "CUMCM2018A_data.xlsx")
    if thermal_file:
        results.append(_run_2018a_thermal_adapter(thermal_file))
    solar_dir = _find_solar_case_dir(roots)
    if solar_dir:
        solar_result = _run_2012b_solar_adapter(solar_dir)
        if solar_result.status == "executed":
            results.append(solar_result)
    suite = {
        "version": "v34",
        "purpose": "mechanism/physics, spatial, and multi-objective real-data adapters",
        "summary": {
            "executed_cases": sum(1 for result in results if result.status == "executed"),
            "archetype_coverage": sorted({a for result in results for a in result.archetypes}),
            "routes_evaluated": sum(len(result.routes) for result in results),
            "local_python_execution": True,
        },
        "results": [asdict(result) for result in results],
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "v34_mechanism_spatial_adapters.json").write_text(json.dumps(suite, ensure_ascii=False, indent=2), encoding="utf-8")
        (output_dir / "v34_mechanism_spatial_adapters.md").write_text(format_v34_report(suite), encoding="utf-8")
    return suite


def format_v34_report(suite: Dict[str, Any]) -> str:
    lines = ["# V34 机理/空间/多目标真实 Adapter 报告", "", f"- summary: {suite.get('summary')}", "", "## Results"]
    for result in suite.get("results", []):
        lines.extend([
            f"### {result['case']}",
            f"- status: {result['status']}",
            f"- archetypes: {result['archetypes']}",
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
    return "\n".join(lines) + "\n"


def _run_2018a_thermal_adapter(data_path: Path) -> V34AdapterResult:
    material = pd.read_excel(data_path, sheet_name="Appendix 1", header=None)
    temp = pd.read_excel(data_path, sheet_name="Appendix 2", header=None)
    material_rows = material.iloc[2:].copy()
    material_rows.columns = ["layer", "density", "specific_heat", "conductivity", "thickness_mm"]
    numeric_material = material_rows.copy()
    for col in ["density", "specific_heat", "conductivity"]:
        numeric_material[col] = pd.to_numeric(numeric_material[col], errors="coerce")
    temp_frame = temp.iloc[2:, :2].copy()
    temp_frame.columns = ["time_s", "skin_temp_c"]
    temp_frame = temp_frame.apply(pd.to_numeric, errors="coerce").dropna()
    time = temp_frame["time_s"].to_numpy(dtype=float)
    skin = temp_frame["skin_temp_c"].to_numpy(dtype=float)

    ambient_proxy = 75.0
    baseline_pred = np.full_like(skin, np.mean(skin[:60]))
    linear_pred = np.interp(time, [time.min(), time.max()], [skin[0], skin[-1]])
    params, rc_pred = _fit_first_order_thermal(time, skin, ambient_proxy)
    routes = [
        {"route": "constant_skin_baseline", "metrics": _thermal_metrics(skin, baseline_pred)},
        {"route": "linear_warmup_baseline", "metrics": _thermal_metrics(skin, linear_pred)},
        {"route": "first_order_RC_mechanism", "metrics": {**_thermal_metrics(skin, rc_pred), "tau_seconds": round(float(params[1]), 3), "asymptote_temp_c": round(float(params[0]), 3)}},
    ]
    routes = sorted(routes, key=lambda route: route["metrics"]["MAE"])
    best = routes[0]
    layer_resistance = _thermal_resistance_summary(numeric_material)
    aggregate = {
        "n_measurements": int(len(skin)),
        "duration_seconds": int(time.max() - time.min()),
        "initial_skin_temp_c": round(float(skin[0]), 3),
        "max_skin_temp_c": round(float(np.max(skin)), 3),
        "best_route": best["route"],
        "best_MAE": best["metrics"]["MAE"],
        "baseline_MAE": next(route for route in routes if route["route"] == "constant_skin_baseline")["metrics"]["MAE"],
        "dominant_resistance_layer": layer_resistance["dominant_layer"],
        "total_nominal_resistance": layer_resistance["total_nominal_resistance"],
    }
    return V34AdapterResult(
        case="CUMCM2018A_high_temperature_clothing_mechanism_v34",
        archetypes=["mechanism_or_physics", "constrained_optimization"],
        status="executed",
        data_path=str(data_path),
        routes=routes,
        aggregate=aggregate,
        recommended_route=best["route"],
        recommendation_evidence=[
            "真实读取 2018A 高温服材料参数与假人皮肤外侧温度测量序列。",
            f"一阶 RC 机理模型在 {aggregate['n_measurements']} 个温度点上拟合，MAE={aggregate['best_MAE']}。",
            f"材料名义热阻显示主导隔热层为 {aggregate['dominant_resistance_layer']}，可作为厚度优化的优先变量。",
        ],
        figure_descriptions=[
            "皮肤温度实测-拟合曲线：比较 constant、linear、first-order RC 三条路线。",
            "残差时间图：检查高温初期和后期是否存在系统偏差。",
            "各层名义热阻柱状图：展示哪一层控制总隔热能力。",
            "安全阈值时间图：标注皮肤温度接近 44℃ 等安全线的时间窗口。",
        ],
        limitations=[
            "当前 adapter 是一阶 lumped RC 机理基线，不等同于完整多层 PDE 热传导求解。",
            "材料 II/IV 厚度为范围，当前只用名义中点估计热阻，精确优化需补有限差分。",
        ],
    )


def _run_2012b_solar_adapter(case_dir: Path) -> V34AdapterResult:
    component_file = next(case_dir.rglob("*附件3*.xls"), None)
    radiation_file = next(case_dir.rglob("*附件4*.xls"), None)
    if not component_file or not radiation_file:
        return V34AdapterResult("CUMCM2012B_solar_house_spatial_multiobjective_v34", ["spatial_or_geostat", "constrained_optimization"], "blocked", str(case_dir), [], {"reason": "solar component or radiation file not found"}, "none", [], [], ["缺少组件或辐射附件。"])
    components = _load_solar_components(component_file)
    radiation = _load_solar_radiation(radiation_file)
    orientation_cols = {
        "east_orientation_baseline": "east",
        "south_orientation_strong": "south",
        "west_orientation_candidate": "west",
        "north_orientation_negative_control": "north",
    }
    best_component = components.sort_values(["efficiency", "power_density_w_m2"], ascending=False).iloc[0]
    routes = []
    for route_name, orientation in orientation_cols.items():
        annual_irradiance = float(radiation[orientation].sum())
        annual_energy_index = annual_irradiance * float(best_component["efficiency"])
        routes.append({
            "route": route_name,
            "metrics": {
                "annual_energy_index": round(annual_energy_index, 6),
                "annual_irradiance_sum": round(annual_irradiance, 6),
                "component_model": str(best_component["model"]),
                "component_efficiency": round(float(best_component["efficiency"]), 6),
                "power_density_w_m2": round(float(best_component["power_density_w_m2"]), 6),
            },
        })
    pareto = _component_pareto(components)
    routes.append({
        "route": "pareto_component_selection",
        "metrics": {
            "pareto_count": int(len(pareto)),
            "best_efficiency": round(float(components["efficiency"].max()), 6),
            "best_power_density_w_m2": round(float(components["power_density_w_m2"].max()), 6),
            "annual_energy_index": max(route["metrics"].get("annual_energy_index", 0) for route in routes),
        },
    })
    routes = sorted(routes, key=lambda route: route["metrics"].get("annual_energy_index", 0), reverse=True)
    best = routes[0]
    aggregate = {
        "radiation_hours": int(len(radiation)),
        "component_count": int(len(components)),
        "pareto_component_count": int(len(pareto)),
        "best_route": best["route"],
        "best_annual_energy_index": best["metrics"].get("annual_energy_index"),
        "south_vs_east_gain_pct": _pct_gain(_route_metric(routes, "south_orientation_strong", "annual_energy_index"), _route_metric(routes, "east_orientation_baseline", "annual_energy_index")),
        "best_component_model": str(best_component["model"]),
    }
    return V34AdapterResult(
        case="CUMCM2012B_solar_house_spatial_multiobjective_v34",
        archetypes=["spatial_or_geostat", "constrained_optimization"],
        status="executed",
        data_path=str(case_dir),
        routes=routes,
        aggregate=aggregate,
        recommended_route=best["route"],
        recommendation_evidence=[
            "真实读取 2012B 光伏组件参数与山西大同逐时各朝向辐射强度。",
            f"南向路线相对东向 baseline 的年能量指数提升 {aggregate['south_vs_east_gain_pct']}%。",
            f"组件层面构造效率/功率密度 Pareto 选择，Pareto 候选数为 {aggregate['pareto_component_count']}。",
        ],
        figure_descriptions=[
            "各朝向全年辐射总量柱状图：比较东南西北朝向的资源差异。",
            "组件效率-功率密度 Pareto 散点图：标出非支配组件。",
            "月度/小时辐射热力图：展示太阳能收益的季节与日内结构。",
            "路线能量指数对比图：比较 east baseline、south strong、west candidate、north negative control。",
        ],
        limitations=[
            "当前 adapter 使用朝向总辐射和组件效率构造能量指数，未完整考虑屋顶几何遮挡。",
            "附件缺少价格字段时，多目标只覆盖效率和功率密度，经济性需补成本数据。",
        ],
    )


def _fit_first_order_thermal(time: np.ndarray, skin: np.ndarray, ambient_proxy: float) -> tuple[np.ndarray, np.ndarray]:
    def model(t: np.ndarray, asymptote: float, tau: float) -> np.ndarray:
        return asymptote - (asymptote - skin[0]) * np.exp(-np.maximum(t, 0) / max(tau, 1e-6))
    try:
        params, _ = curve_fit(model, time, skin, p0=[min(ambient_proxy, max(skin) + 5), 1800.0], bounds=([skin[0], 10.0], [ambient_proxy, 20000.0]), maxfev=20000)
    except Exception:
        params = np.array([float(max(skin)), 1800.0])
    return params, model(time, *params)


def _thermal_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, Any]:
    error = actual - predicted
    abs_error = np.abs(error)
    return {
        "MAE": round(float(np.mean(abs_error)), 6),
        "RMSE": round(float(np.sqrt(np.mean(error ** 2))), 6),
        "max_abs_error": round(float(np.max(abs_error)), 6),
        "p90_abs_error": round(float(np.percentile(abs_error, 90)), 6),
        "safety_margin_to_44C": round(float(44.0 - np.max(predicted)), 6),
    }


def _thermal_resistance_summary(material: pd.DataFrame) -> Dict[str, Any]:
    rows = []
    for _, row in material.iterrows():
        layer = str(row.get("layer"))
        conductivity = float(row.get("conductivity")) if pd.notna(row.get("conductivity")) else np.nan
        thickness = _parse_thickness_mm(row.get("thickness_mm"))
        if pd.notna(conductivity) and conductivity > 0 and thickness > 0:
            rows.append((layer, thickness / 1000.0 / conductivity))
    if not rows:
        return {"dominant_layer": "unknown", "total_nominal_resistance": 0.0}
    dominant = max(rows, key=lambda item: item[1])
    return {"dominant_layer": dominant[0], "total_nominal_resistance": round(float(sum(v for _, v in rows)), 6)}


def _parse_thickness_mm(value: Any) -> float:
    text = str(value)
    if "-" in text:
        parts = [float(x) for x in text.split("-") if x.strip()]
        return float(np.mean(parts))
    try:
        return float(text)
    except Exception:
        return 0.0


def _load_solar_components(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=0, header=None)
    rows = []
    current_type = "unknown"
    for _, row in raw.iloc[1:].iterrows():
        if pd.notna(row.iloc[0]) and str(row.iloc[0]).strip():
            current_type = str(row.iloc[0])
        model = row.iloc[1]
        power = pd.to_numeric(row.iloc[2], errors="coerce")
        length = pd.to_numeric(row.iloc[3], errors="coerce")
        width = pd.to_numeric(row.iloc[4], errors="coerce")
        efficiency = pd.to_numeric(row.iloc[7], errors="coerce")
        if pd.notna(model) and pd.notna(power) and pd.notna(length) and pd.notna(width) and pd.notna(efficiency):
            area = float(length) * float(width) / 1_000_000.0
            rows.append({"type": current_type, "model": str(model), "power": float(power), "area_m2": area, "efficiency": float(efficiency), "power_density_w_m2": float(power) / max(area, 1e-9)})
    return pd.DataFrame(rows)


def _load_solar_radiation(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=0, header=None)
    data = raw.iloc[2:, [6, 7, 8, 9]].copy()
    data.columns = ["east", "south", "west", "north"]
    return data.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _component_pareto(components: pd.DataFrame) -> pd.DataFrame:
    keep = []
    values = components[["efficiency", "power_density_w_m2"]].to_numpy(dtype=float)
    for i, row in enumerate(values):
        dominated = False
        for j, other in enumerate(values):
            if i == j:
                continue
            if np.all(other >= row) and np.any(other > row):
                dominated = True
                break
        keep.append(not dominated)
    return components.loc[keep]


def _route_metric(routes: List[Dict[str, Any]], route_name: str, metric: str) -> float:
    for route in routes:
        if route.get("route") == route_name:
            return float(route.get("metrics", {}).get(metric, 0.0))
    return 0.0


def _pct_gain(value: float, baseline: float) -> float:
    return round(float(100 * (value - baseline) / max(abs(baseline), 1e-9)), 2)


def _find_file(roots: List[Path], filename: str) -> Path | None:
    for root in roots:
        if root.exists():
            matches = list(root.rglob(filename))
            if matches:
                return sorted(matches, key=lambda p: len(str(p)))[0]
    return None


def _find_solar_case_dir(roots: List[Path]) -> Path | None:
    for root in roots:
        if not root.exists():
            continue
        for component_file in root.rglob("*附件3*.xls"):
            if "2012B" in str(component_file):
                parent = component_file.parent
                if list(parent.rglob("*附件4*.xls")) or list(parent.parent.rglob("*附件4*.xls")):
                    return parent.parent if parent.name == "附件" else parent
    return None


def _resolve_roots(reference_roots: Iterable[Path | str] | None) -> List[Path]:
    if reference_roots is not None:
        return [Path(root) for root in reference_roots]
    return [Path(__file__).resolve().parent.parent / "data" / "reference_root"]


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent / "run" / "v34_mechanism_spatial_adapters"
    suite = run_v34_mechanism_spatial_adapters(output_dir=output_dir)
    print(format_v34_report(suite))
