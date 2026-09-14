from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from .table_cache import read_excel_cached


ROUTES = ("last_value", "rolling_mean_7", "seasonal_naive_7", "weekday_median_8")


def build_category_forecast(
    product_path: Path,
    sales_path: Path,
    output_dir: Path,
    training_cutoff: str = "2023-06-30",
    horizon: int = 7,
) -> Dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    product_path = Path(product_path).resolve()
    sales_path = Path(sales_path).resolve()
    result = _compute(product_path, sales_path, training_cutoff, horizon)
    forecast_path = output_dir / "category_forecast.csv"
    diagnostics_path = output_dir / "category_forecast_diagnostics.csv"
    result["forecast"].to_csv(forecast_path, index=False, encoding="utf-8")
    result["diagnostics"].to_csv(diagnostics_path, index=False, encoding="utf-8")
    manifest = {
        "version": "v44",
        "kind": "category_rolling_origin_forecast",
        "training_cutoff": training_cutoff,
        "horizon": horizon,
        "selection_protocol": "eight_nonoverlapping_7_day_origins_before_final_test",
        "final_test_protocol": "four_nonoverlapping_7_day_origins_not_used_for_route_selection",
        "scenario_calibration": "selection_residual_ratio_quantiles_0.2_0.5_0.8",
        "sources": [
            {"alias": "products", "path": str(product_path), "sha256": _sha256(product_path)},
            {"alias": "sales", "path": str(sales_path), "sha256": _sha256(sales_path)},
        ],
        "artifacts": {
            "forecast": {"path": str(forecast_path), "sha256": _sha256(forecast_path)},
            "diagnostics": {"path": str(diagnostics_path), "sha256": _sha256(diagnostics_path)},
        },
        "category_count": int(result["forecast"]["category"].nunique()),
        "selected_routes": result["selected_routes"],
    }
    manifest_path = output_dir / "category_forecast_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"forecast_path": str(forecast_path), "diagnostics_path": str(diagnostics_path), "manifest_path": str(manifest_path), "manifest": manifest}


def verify_category_forecast(manifest_path: Path) -> Dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    issues = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sources = {item["alias"]: item for item in manifest["sources"]}
        for alias, source in sources.items():
            path = Path(source["path"])
            if not path.is_file() or _sha256(path) != source["sha256"]:
                issues.append(f"category_forecast_source_hash_mismatch:{alias}")
        for name, artifact in manifest["artifacts"].items():
            path = Path(artifact["path"])
            if not path.is_file() or _sha256(path) != artifact["sha256"]:
                issues.append(f"category_forecast_artifact_hash_mismatch:{name}")
        if not issues:
            recomputed = _compute(Path(sources["products"]["path"]), Path(sources["sales"]["path"]), manifest["training_cutoff"], int(manifest["horizon"]))
            stored_forecast = pd.read_csv(manifest["artifacts"]["forecast"]["path"])
            stored_diagnostics = pd.read_csv(manifest["artifacts"]["diagnostics"]["path"])
            if not _frames_equal(stored_forecast, recomputed["forecast"]):
                issues.append("category_forecast_values_not_recomputed")
            if not _frames_equal(stored_diagnostics, recomputed["diagnostics"]):
                issues.append("category_forecast_diagnostics_not_recomputed")
            if manifest.get("selected_routes") != recomputed["selected_routes"]:
                issues.append("category_forecast_route_selection_not_recomputed")
    except Exception as exc:
        issues.append(f"category_forecast_verification_error:{exc}")
    passed = not issues
    return {"kind": "category_rolling_origin_forecast", "passed": passed, "issues": issues, "claim_level": "domain_verified_category_forecast" if passed else "no_numerical_claim", "manifest_path": str(manifest_path), "manifest_hash": _sha256(manifest_path) if manifest_path.is_file() else None}


def _compute(product_path: Path, sales_path: Path, training_cutoff: str, horizon: int) -> Dict[str, Any]:
    products = read_excel_cached(product_path, "Sheet1", ["单品编码", "分类名称"])
    sales = read_excel_cached(sales_path, "Sheet1", ["销售日期", "单品编码", "销量(千克)"])
    products["单品编码"] = products["单品编码"].astype(str)
    sales["单品编码"] = sales["单品编码"].astype(str)
    sales["销售日期"] = pd.to_datetime(sales["销售日期"])
    cutoff = pd.Timestamp(training_cutoff)
    future_dates = pd.date_range(cutoff + pd.offsets.Day(1), periods=horizon, freq="D")
    merged = sales.loc[sales["销售日期"] <= cutoff].merge(products, on="单品编码", how="left", validate="many_to_one")
    if merged["分类名称"].isna().any():
        raise ValueError("category_forecast_product_mapping_incomplete")
    daily = merged.groupby(["销售日期", "分类名称"], as_index=False)["销量(千克)"].sum()
    forecasts = []
    diagnostics = []
    selected_routes: Dict[str, str] = {}
    for category in sorted(daily["分类名称"].astype(str).unique()):
        values = daily.loc[daily["分类名称"].astype(str) == category].set_index("销售日期")["销量(千克)"]
        index = pd.date_range(values.index.min(), cutoff, freq="D")
        series = values.reindex(index, fill_value=0.0).astype(float)
        if len(series) < 12 * horizon + 56:
            raise ValueError(f"category_forecast_history_too_short:{category}")
        final_start = len(series) - 4 * horizon
        selection_start = final_start - 8 * horizon
        route_rows = []
        selection_predictions: Dict[str, list[float]] = {}
        for route in ROUTES:
            selection_actual, selection_pred = _rolling_predictions(series, selection_start, final_start, horizon, route)
            final_actual, final_pred = _rolling_predictions(series, final_start, len(series), horizon, route)
            selection_predictions[route] = selection_pred.tolist()
            route_rows.append({"category": category, "route": route, "selection_mae": _mae(selection_actual, selection_pred), "final_test_mae": _mae(final_actual, final_pred), "selection_rows": len(selection_actual), "final_test_rows": len(final_actual)})
        chosen = min(route_rows, key=lambda row: (row["selection_mae"], ROUTES.index(row["route"])))
        selected_routes[category] = chosen["route"]
        best_final = min(row["final_test_mae"] for row in route_rows)
        chosen["selected_by_selection"] = True
        chosen["final_test_reversal"] = chosen["final_test_mae"] > best_final + 1e-12
        for row in route_rows:
            row.setdefault("selected_by_selection", False)
            row.setdefault("final_test_reversal", False)
            diagnostics.append(row)
        future = _forecast(series, future_dates, chosen["route"])
        selection_actual = series.iloc[selection_start:final_start].to_numpy(dtype=float)
        residual_ratio = selection_actual / np.maximum(np.asarray(selection_predictions[chosen["route"]], dtype=float), 1e-6)
        factors = np.maximum.accumulate(np.clip(np.quantile(residual_ratio, [0.2, 0.5, 0.8]), [0.5, 0.75, 1.0], [1.0, 1.25, 1.6]))
        for date, prediction in zip(future_dates, future):
            forecasts.append({"decision_date": str(date.date()), "category": category, "selected_route": chosen["route"], "base_forecast": max(float(prediction), 0.0), "low_factor": float(factors[0]), "central_factor": float(factors[1]), "high_factor": float(factors[2]), "selection_mae": chosen["selection_mae"], "final_test_mae": chosen["final_test_mae"], "final_test_reversal": chosen["final_test_reversal"]})
    return {"forecast": pd.DataFrame(forecasts), "diagnostics": pd.DataFrame(diagnostics), "selected_routes": selected_routes}


def _rolling_predictions(series: pd.Series, start: int, stop: int, horizon: int, route: str) -> tuple[np.ndarray, np.ndarray]:
    actual = []
    predicted = []
    for origin in range(start, stop, horizon):
        block_stop = min(origin + horizon, stop)
        dates = series.index[origin:block_stop]
        forecast = _forecast(series.iloc[:origin], dates, route)
        actual.extend(series.iloc[origin:block_stop].tolist())
        predicted.extend(forecast.tolist())
    return np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)


def _forecast(history: pd.Series, dates: pd.DatetimeIndex, route: str) -> np.ndarray:
    values = history.to_numpy(dtype=float)
    if route == "last_value":
        return np.repeat(values[-1], len(dates))
    if route == "rolling_mean_7":
        return np.repeat(float(np.mean(values[-7:])), len(dates))
    if route == "seasonal_naive_7":
        return np.asarray([values[-7 + (index % 7)] for index in range(len(dates))], dtype=float)
    if route == "weekday_median_8":
        lookup = {weekday: float(np.median(history.loc[history.index.dayofweek == weekday].tail(8))) for weekday in range(7)}
        return np.asarray([lookup[date.dayofweek] for date in dates], dtype=float)
    raise ValueError(f"category_forecast_route_unknown:{route}")


def _mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def _frames_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if list(left.columns) != list(right.columns) or left.shape != right.shape:
        return False
    for column in left.columns:
        if pd.api.types.is_numeric_dtype(right[column]):
            if not np.allclose(pd.to_numeric(left[column]), pd.to_numeric(right[column]), rtol=1e-10, atol=1e-10, equal_nan=True):
                return False
        elif left[column].astype(str).tolist() != right[column].astype(str).tolist():
            return False
    return True


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
