"""
V32 forecasting benchmark adapter.

Adds a fast real-data forecasting adapter on CUMCM2016D wind-power data with
rolling-origin validation. The point is honest route selection: advanced models
must beat naive persistence before being recommended.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


@dataclass
class ForecastingResult:
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


def run_v32_forecasting_adapter(reference_roots: Iterable[Path | str] | None = None, output_dir: Path | str | None = None) -> Dict[str, Any]:
    roots = _resolve_roots(reference_roots)
    wind_file = _find_wind_file(roots)
    if not wind_file:
        suite = {"version": "v32", "summary": {"executed_cases": 0, "reason": "201501.xls wind-power data not found"}, "results": []}
    else:
        frame = _load_wind_month(wind_file)
        result = _run_wind_forecasting(frame, wind_file)
        suite = {
            "version": "v32",
            "purpose": "forecasting adapter with rolling-origin validation",
            "summary": {
                "executed_cases": 1 if result.status == "executed" else 0,
                "archetype_coverage": ["forecasting"] if result.status == "executed" else [],
                "routes_evaluated": len(result.routes),
                "local_python_execution": result.status == "executed",
            },
            "results": [asdict(result)],
        }
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "v32_forecasting_adapter.json").write_text(json.dumps(suite, ensure_ascii=False, indent=2), encoding="utf-8")
        (output_dir / "v32_forecasting_adapter.md").write_text(format_v32_report(suite), encoding="utf-8")
    return suite


def format_v32_report(suite: Dict[str, Any]) -> str:
    lines = ["# V32 预测类真实执行 Adapter 报告", "", f"- summary: {suite.get('summary')}", "", "## Results"]
    for result in suite.get("results", []):
        lines.extend([
            f"### {result['case']}",
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
    return "\n".join(lines) + "\n"


def _run_wind_forecasting(frame: pd.DataFrame, data_path: Path) -> ForecastingResult:
    if frame.empty or len(frame) < 120:
        return ForecastingResult("CUMCM2016D_wind_power_forecasting_v32", ["forecasting"], "blocked", str(data_path), [], {"reason": "not enough time-series points"}, "none", [], [], ["数据点不足，无法做 rolling-origin validation。"])

    frame = frame.sort_values(["day", "quarter"]).reset_index(drop=True)
    y = frame["power"].to_numpy(dtype=float)
    wind = frame["wind_speed"].to_numpy(dtype=float)
    hour = np.arange(len(y)) % 96
    min_train = 96
    max_window = 7 * 96
    route_predictions = {
        "naive_persistence": [],
        "seasonal_naive_96": [],
        "rolling_mean_96points": [],
        "ewma_alpha_02": [],
        "ridge_lag_wind": [],
        "ridge_poly_lag_wind": [],
    }
    targets = []

    for idx in range(min_train, len(y) - 1):
        target = y[idx + 1]
        train_start = max(0, idx - max_window + 1)
        train_y = y[train_start:idx + 1]
        train_wind = wind[train_start:idx + 1]
        train_hour = np.arange(train_start, idx + 1) % 96
        targets.append(target)

        route_predictions["naive_persistence"].append(float(y[idx]))
        route_predictions["seasonal_naive_96"].append(float(y[idx - 95] if idx >= 95 else y[idx]))
        route_predictions["rolling_mean_96points"].append(float(np.mean(y[max(0, idx - 95):idx + 1])))
        route_predictions["ewma_alpha_02"].append(_ewma(y[max(0, idx - 95):idx + 1], 0.2))

        X_train, y_train = _lag_features(train_y, train_wind, train_hour)
        x_now = _single_feature(y, wind, hour, idx)
        if len(y_train) >= 48:
            ridge_pred = _ridge_predict(X_train, y_train, np.asarray([x_now], dtype=float), alpha=1.0)
            poly_pred = _ridge_predict(_poly_features(X_train), y_train, _poly_features(np.asarray([x_now], dtype=float)), alpha=5.0)
            route_predictions["ridge_lag_wind"].append(ridge_pred)
            route_predictions["ridge_poly_lag_wind"].append(poly_pred)
        else:
            route_predictions["ridge_lag_wind"].append(float(y[idx]))
            route_predictions["ridge_poly_lag_wind"].append(float(y[idx]))

    targets_array = np.asarray(targets, dtype=float)
    routes = []
    for route_name, preds in route_predictions.items():
        metrics = _forecast_metrics(targets_array, np.asarray(preds, dtype=float))
        metrics["rolling_origin_folds"] = int(len(targets_array))
        routes.append({"route": route_name, "metrics": metrics})
    routes = sorted(routes, key=lambda item: (item["metrics"]["MAE"], item["metrics"]["RMSE"]))
    best = routes[0]
    naive = next(route for route in routes if route["route"] == "naive_persistence")
    improvement = (naive["metrics"]["MAE"] - best["metrics"]["MAE"]) / max(naive["metrics"]["MAE"], 1e-9)
    aggregate = {
        "n_points": int(len(y)),
        "rolling_origin_folds": int(len(targets_array)),
        "best_route": best["route"],
        "best_MAE": best["metrics"]["MAE"],
        "naive_MAE": naive["metrics"]["MAE"],
        "improvement_over_naive_pct": round(float(100 * improvement), 2),
        "target_mean": round(float(np.mean(y)), 4),
        "target_std": round(float(np.std(y)), 4),
    }
    advanced_warning = "本次真实 rolling-origin 对比中，naive persistence 已是最佳；因此预测主路线应优先采用 naive/物理解释基线，而不是强行上复杂模型。" if best["route"] == "naive_persistence" else f"最佳路线 {best['route']} 打败 naive，可作为预测主路线候选。"
    return ForecastingResult(
        case="CUMCM2016D_wind_power_forecasting_v32",
        archetypes=["forecasting"],
        status="executed",
        data_path=str(data_path),
        routes=routes,
        aggregate=aggregate,
        recommended_route=best["route"],
        recommendation_evidence=[
            "真实读取 2016D 风电 201501.xls，多 sheet 合并为 15 分钟序列。",
            f"使用 rolling-origin validation，评估 {aggregate['rolling_origin_folds']} 个一步预测折。",
            f"最佳路线 {best['route']} 的 MAE={best['metrics']['MAE']}，相对 naive 改善 {aggregate['improvement_over_naive_pct']}%。",
            advanced_warning,
        ],
        figure_descriptions=[
            "真实功率与最佳预测曲线局部对比图：展示连续几天的一步预测贴合情况。",
            "路线 MAE/RMSE 对比柱状图：比较 naive、seasonal naive、rolling mean、EWMA、Ridge 与多项式 Ridge。",
            "rolling-origin 误差时间图：显示误差是否集中在特定日内时段或异常天气段。",
            "残差-风速散点图：检验强风/低风速区间是否存在系统偏差。",
            "日内小时误差热力图：横轴 96 个 15 分钟时段，纵轴日期，颜色为绝对误差。",
        ],
        limitations=[
            "当前 adapter 只使用 201501 单月数据，跨月/跨季节泛化仍需多月 benchmark。",
            "本快版使用轻量模型，避免把 benchmark 执行时间拖垮；复杂模型应在专项 adapter 中单独扩展。",
            "一步预测已可验证路线选择，但多步滚动预测还需单独评估误差累积。",
        ],
    )


def _ridge_predict(X_train: np.ndarray, y_train: np.ndarray, X_now: np.ndarray, alpha: float) -> float:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    now_scaled = scaler.transform(X_now)
    model = Ridge(alpha=alpha)
    model.fit(X_scaled, y_train)
    return float(model.predict(now_scaled)[0])


def _forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    error = y_true - y_pred
    abs_error = np.abs(error)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    denom = np.maximum(np.abs(y_true), 1e-6)
    return {
        "MAE": round(float(mae), 6),
        "RMSE": round(float(rmse), 6),
        "MAPE_proxy": round(float(np.mean(abs_error / denom)), 6),
        "WAPE": round(float(np.sum(abs_error) / max(np.sum(np.abs(y_true)), 1e-9)), 6),
        "p90_abs_error": round(float(np.percentile(abs_error, 90)), 6),
        "bias": round(float(np.mean(y_pred - y_true)), 6),
    }


def _lag_features(train_y: np.ndarray, train_wind: np.ndarray, train_hour: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, target = [], []
    for i in range(2, len(train_y) - 1):
        rows.append([
            train_y[i],
            train_y[i - 1],
            train_y[i - 2],
            float(np.mean(train_y[max(0, i - 7):i + 1])),
            train_wind[i],
            train_wind[i - 1],
            np.sin(2 * np.pi * train_hour[i] / 96),
            np.cos(2 * np.pi * train_hour[i] / 96),
        ])
        target.append(train_y[i + 1])
    return np.asarray(rows, dtype=float), np.asarray(target, dtype=float)


def _single_feature(y: np.ndarray, wind: np.ndarray, hour: np.ndarray, idx: int) -> List[float]:
    return [
        float(y[idx]),
        float(y[idx - 1]),
        float(y[idx - 2]),
        float(np.mean(y[max(0, idx - 7):idx + 1])),
        float(wind[idx]),
        float(wind[idx - 1]),
        float(np.sin(2 * np.pi * hour[idx] / 96)),
        float(np.cos(2 * np.pi * hour[idx] / 96)),
    ]


def _poly_features(X: np.ndarray) -> np.ndarray:
    selected = X[:, [0, 3, 4]]
    return np.column_stack([X, selected ** 2, selected[:, 0] * selected[:, 2]])


def _ewma(values: np.ndarray, alpha: float) -> float:
    weights = (1.0 - alpha) ** np.arange(len(values) - 1, -1, -1)
    return float(np.sum(weights * values) / np.sum(weights))


def _load_wind_month(path: Path) -> pd.DataFrame:
    rows = []
    excel = pd.ExcelFile(path)
    for day_index, sheet in enumerate(excel.sheet_names, start=1):
        raw = pd.read_excel(path, sheet_name=sheet, header=None)
        for row_index in range(1, len(raw)):
            row = raw.iloc[row_index]
            date_value = row.iloc[0] if len(row) > 0 else None
            for block in range(4):
                base = 1 + block * 4
                if base + 1 >= len(row):
                    continue
                power = pd.to_numeric(row.iloc[base], errors="coerce")
                wind = pd.to_numeric(row.iloc[base + 1], errors="coerce")
                if pd.notna(power) and pd.notna(wind):
                    rows.append({"day": day_index, "date": str(date_value), "quarter": len(rows), "power": float(power), "wind_speed": float(wind)})
    return pd.DataFrame(rows)


def _find_wind_file(roots: List[Path]) -> Path | None:
    for root in roots:
        if not root.exists():
            continue
        candidates = list(root.rglob("201501.xls"))
        if candidates:
            return sorted(candidates, key=lambda p: len(str(p)))[0]
    return None


def _resolve_roots(reference_roots: Iterable[Path | str] | None) -> List[Path]:
    if reference_roots is not None:
        return [Path(root) for root in reference_roots]
    return [Path(__file__).resolve().parent.parent / "data" / "reference_root"]


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent / "run" / "v32_forecasting_adapter"
    suite = run_v32_forecasting_adapter(output_dir=output_dir)
    print(format_v32_report(suite))
