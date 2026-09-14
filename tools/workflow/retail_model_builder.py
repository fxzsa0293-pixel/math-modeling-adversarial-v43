from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from .table_cache import read_excel_cached
from .category_forecast import verify_category_forecast


def build_cumcm2023c_retail_model(
    product_path: Path,
    sales_path: Path,
    wholesale_path: Path,
    loss_path: Path,
    output_dir: Path,
    training_cutoff: str = "2023-06-30",
    decision_start: str = "2023-07-01",
    horizon: int = 7,
    category_forecast_manifest: Path | None = None,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "products": Path(product_path).resolve(),
        "sales": Path(sales_path).resolve(),
        "wholesale": Path(wholesale_path).resolve(),
        "loss": Path(loss_path).resolve(),
    }
    products = read_excel_cached(paths["products"], "Sheet1", ["单品编码", "单品名称", "分类名称"])
    sales = read_excel_cached(paths["sales"], "Sheet1", ["销售日期", "单品编码", "销量(千克)", "销售单价(元/千克)"])
    wholesale = read_excel_cached(paths["wholesale"], "Sheet1", ["日期", "单品编码", "批发价格(元/千克)"])
    loss = read_excel_cached(paths["loss"], "Sheet1", ["单品编码", "损耗率(%)"])

    products["单品编码"] = products["单品编码"].astype(str)
    for frame in (sales, wholesale, loss):
        frame["单品编码"] = frame["单品编码"].astype(str)
    cutoff = pd.Timestamp(training_cutoff)
    decision_dates = pd.date_range(decision_start, periods=horizon, freq="D")
    forecast_rows = None
    forecast_binding = None
    if category_forecast_manifest is not None:
        forecast_manifest_path = Path(category_forecast_manifest).resolve()
        forecast_verification = verify_category_forecast(forecast_manifest_path)
        if not forecast_verification["passed"]:
            raise ValueError(f"category_forecast_verification_failed:{forecast_verification['issues']}")
        forecast_manifest = json.loads(forecast_manifest_path.read_text(encoding="utf-8"))
        if forecast_manifest.get("training_cutoff") != training_cutoff or int(forecast_manifest.get("horizon", 0)) != horizon:
            raise ValueError("category_forecast_time_boundary_mismatch")
        forecast_artifact = forecast_manifest["artifacts"]["forecast"]
        forecast_path = Path(forecast_artifact["path"]).resolve()
        if _sha256(forecast_path) != forecast_artifact["sha256"]:
            raise ValueError("category_forecast_binding_invalid")
        forecast_rows = pd.read_csv(forecast_path, dtype={"category": str})
        forecast_binding = {"manifest_path": str(forecast_manifest_path), "manifest_sha256": _sha256(forecast_manifest_path), "forecast_path": str(forecast_path), "forecast_sha256": _sha256(forecast_path)}
    sales["销售日期"] = pd.to_datetime(sales["销售日期"])
    wholesale["日期"] = pd.to_datetime(wholesale["日期"])
    sales = sales.loc[sales["销售日期"] <= cutoff].merge(products, on="单品编码", how="left", validate="many_to_one")
    wholesale = wholesale.loc[wholesale["日期"] <= cutoff].merge(products, on="单品编码", how="left", validate="many_to_one")
    loss = loss.merge(products, on="单品编码", how="left", validate="many_to_one")
    if sales["分类名称"].isna().any() or wholesale["分类名称"].isna().any() or loss["分类名称"].isna().any():
        raise ValueError("retail_source_product_mapping_incomplete")

    sales = sales.loc[(sales["销量(千克)"] > 0) & (sales["销售单价(元/千克)"] > 0)].copy()
    sales["销售额"] = sales["销量(千克)"] * sales["销售单价(元/千克)"]
    daily = sales.groupby(["销售日期", "分类名称"], as_index=False).agg(销量=("销量(千克)", "sum"), 销售额=("销售额", "sum"))
    daily["均价"] = daily["销售额"] / daily["销量"]
    daily = daily.sort_values(["分类名称", "销售日期"], kind="stable")
    wholesale_daily = wholesale.groupby(["日期", "分类名称"], as_index=False).agg(批发价=("批发价格(元/千克)", "mean"))
    loss_by_category = loss.groupby("分类名称")["损耗率(%)"].mean().to_dict()
    categories = sorted(str(value) for value in products["分类名称"].dropna().unique())
    if len(categories) != 6:
        raise ValueError(f"retail_expected_six_categories:{len(categories)}")
    demand_correlation = _estimate_category_correlation(daily, categories)

    scenario_names = ["low", "central", "high"]
    scenario_probabilities = [0.2, 0.6, 0.2]
    model_items = []
    rows = []
    diagnostics = []
    total_reference_cost = 0.0
    for category in categories:
        history = daily.loc[daily["分类名称"] == category].set_index("销售日期").sort_index()
        complete_index = pd.date_range(history.index.min(), cutoff, freq="D")
        history = history.reindex(complete_index)
        history["销量"] = history["销量"].fillna(0.0)
        history["均价"] = history["均价"].ffill().bfill()
        raw_elasticity, elasticity, fit_rows = _estimate_elasticity(history)
        ratios = (history["销量"] / history["销量"].shift(7).replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna().tail(180)
        historical_factors = np.quantile(ratios, [0.2, 0.5, 0.8]) if len(ratios) >= 14 else np.asarray([0.8, 1.0, 1.2])
        historical_factors = np.maximum.accumulate(np.clip(historical_factors, [0.5, 0.75, 1.0], [1.0, 1.25, 1.6]))
        category_cost = wholesale_daily.loc[(wholesale_daily["分类名称"] == category) & (wholesale_daily["日期"] > cutoff - pd.Timedelta(days=14)), "批发价"].mean()
        if not math.isfinite(float(category_cost)):
            category_cost = wholesale_daily.loc[wholesale_daily["分类名称"] == category, "批发价"].tail(30).mean()
        reference_price = float(history["均价"].tail(14).median())
        loss_rate = float(np.clip(float(loss_by_category[category]) / 100.0, 0.0, 0.95))
        price_candidates = _price_grid(float(category_cost), reference_price)
        diagnostics.append({"category": category, "raw_elasticity": raw_elasticity, "used_elasticity": elasticity, "elasticity_fit_rows": fit_rows, "demand_source": "rolling_origin_category_forecast" if forecast_rows is not None else "seasonal_naive_fallback", "unit_cost": float(category_cost), "reference_price": reference_price, "loss_rate": loss_rate})
        for decision_date in decision_dates:
            if forecast_rows is not None:
                matched = forecast_rows.loc[(forecast_rows["category"].astype(str) == category) & (forecast_rows["decision_date"] == str(decision_date.date()))]
                if len(matched) != 1:
                    raise ValueError(f"category_forecast_row_missing_or_duplicate:{decision_date.date()}:{category}")
                forecast_row = matched.iloc[0]
                base_demand = float(forecast_row["base_forecast"])
                factors = np.asarray([forecast_row["low_factor"], forecast_row["central_factor"], forecast_row["high_factor"]], dtype=float)
                forecast_route = str(forecast_row["selected_route"])
            else:
                reference_date = decision_date - pd.Timedelta(days=7)
                base_demand = float(history["销量"].get(reference_date, history["销量"].tail(7).median()))
                factors = historical_factors
                forecast_route = "seasonal_naive_fallback"
            options = []
            for price in price_candidates:
                adjusted = max(0.0, base_demand * (price / reference_price) ** elasticity)
                demand = {name: float(adjusted * factor) for name, factor in zip(scenario_names, factors)}
                options.append({"price": price, "demand": demand})
            high_demand = max(option["demand"]["high"] for option in options)
            max_replenishment = max(0.1, high_demand / max(1.0 - loss_rate, 1e-6) * 1.05)
            item_name = f"{decision_date.date()}|{category}"
            elasticity_range = [max(-3.0, elasticity - 0.75), min(-0.1, elasticity + 0.25)]
            model_items.append({"name": item_name, "group": str(decision_date.date()), "category": category, "unit_cost": float(category_cost), "loss_rate": loss_rate, "base_demand": base_demand, "reference_price": reference_price, "elasticity": elasticity, "stress_elasticity_range": elasticity_range, "min_replenishment": 0.0, "max_replenishment": max_replenishment, "price_options": options})
            total_reference_cost += float(category_cost) * max_replenishment
            rows.append({"decision_date": str(decision_date.date()), "category": category, "forecast_route": forecast_route, "unit_cost": float(category_cost), "loss_rate": loss_rate, "base_demand": base_demand, "elasticity": elasticity, "reference_price": reference_price, "low_factor": float(factors[0]), "central_factor": float(factors[1]), "high_factor": float(factors[2])})

    model = {
        "risk_measure": "cvar",
        "alpha": 0.8,
        "budget": total_reference_cost,
        "min_selected": len(model_items),
        "max_selected": len(model_items),
        "shortage_penalty": 1.0,
        "stress_test": {"seed": 20230701, "replications": 1000, "demand_cv": 0.2, "demand_correlation": demand_correlation},
        "baseline_markup": 0.2,
        "scenarios": [{"name": name, "probability": probability} for name, probability in zip(scenario_names, scenario_probabilities)],
        "items": model_items,
    }
    inputs_path = output_dir / "retail_model_inputs.csv"
    pd.DataFrame(rows).to_csv(inputs_path, index=False, encoding="utf-8")
    manifest = {
        "version": "v44",
        "kind": "cumcm2023c_category_replenishment_pricing",
        "training_cutoff": str(cutoff.date()),
        "decision_dates": [str(value.date()) for value in decision_dates],
        "sources": [{"alias": alias, "path": str(path), "sha256": _sha256(path)} for alias, path in paths.items()],
        "derived_input_path": str(inputs_path.resolve()),
        "derived_input_sha256": _sha256(inputs_path),
        "category_count": len(categories),
        "decision_item_count": len(model_items),
        "diagnostics": diagnostics,
        "upstream_category_forecast": forecast_binding,
        "model_sha256": _json_hash(model),
    }
    if forecast_binding is not None:
        manifest["sources"].extend([
            {"alias": "category_forecast_manifest", "path": forecast_binding["manifest_path"], "sha256": forecast_binding["manifest_sha256"]},
            {"alias": "category_forecast", "path": forecast_binding["forecast_path"], "sha256": forecast_binding["forecast_sha256"]},
        ])
    manifest_path = output_dir / "retail_decision_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"model": model, "data_file": str(inputs_path.resolve()), "manifest_path": str(manifest_path.resolve()), "manifest": manifest}


def build_cumcm2023c_item_retail_model(
    product_path: Path,
    sales_path: Path,
    wholesale_path: Path,
    loss_path: Path,
    output_dir: Path,
    training_cutoff: str = "2023-06-30",
    decision_date: str = "2023-07-01",
    category_forecast_manifest: Path | None = None,
    category_service_level: float = 0.8,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"products": Path(product_path).resolve(), "sales": Path(sales_path).resolve(), "wholesale": Path(wholesale_path).resolve(), "loss": Path(loss_path).resolve()}
    products = read_excel_cached(paths["products"], "Sheet1", ["单品编码", "单品名称", "分类名称"])
    sales = read_excel_cached(paths["sales"], "Sheet1", ["销售日期", "单品编码", "销量(千克)", "销售单价(元/千克)"])
    wholesale = read_excel_cached(paths["wholesale"], "Sheet1", ["日期", "单品编码", "批发价格(元/千克)"])
    loss = read_excel_cached(paths["loss"], "Sheet1", ["单品编码", "损耗率(%)"])
    for frame in (products, sales, wholesale, loss):
        frame["单品编码"] = frame["单品编码"].astype(str)
    cutoff = pd.Timestamp(training_cutoff)
    decision = pd.Timestamp(decision_date)
    forecast_binding = None
    forecast_row_by_category: Dict[str, pd.Series] = {}
    if category_forecast_manifest is not None:
        forecast_manifest_path = Path(category_forecast_manifest).resolve()
        forecast_verification = verify_category_forecast(forecast_manifest_path)
        if not forecast_verification["passed"]:
            raise ValueError(f"category_forecast_verification_failed:{forecast_verification['issues']}")
        forecast_manifest = json.loads(forecast_manifest_path.read_text(encoding="utf-8"))
        forecast_path = Path(forecast_manifest["artifacts"]["forecast"]["path"]).resolve()
        forecast_frame = pd.read_csv(forecast_path)
        forecast_frame = forecast_frame.loc[forecast_frame["decision_date"] == str(decision.date())]
        forecast_row_by_category = {str(row["category"]): row for _, row in forecast_frame.iterrows()}
        forecast_binding = {"manifest_path": str(forecast_manifest_path), "manifest_sha256": _sha256(forecast_manifest_path), "forecast_path": str(forecast_path), "forecast_sha256": _sha256(forecast_path)}
    if not 0.0 <= category_service_level <= 1.0:
        raise ValueError("category_service_level_out_of_range")
    recent_start = cutoff - pd.Timedelta(days=6)
    sales["销售日期"] = pd.to_datetime(sales["销售日期"])
    wholesale["日期"] = pd.to_datetime(wholesale["日期"])
    sales = sales.loc[sales["销售日期"] <= cutoff].merge(products, on="单品编码", how="left", validate="many_to_one")
    wholesale = wholesale.loc[wholesale["日期"] <= cutoff]
    loss_map = loss.groupby("单品编码")["损耗率(%)"].mean().to_dict()
    eligible_codes = sorted(sales.loc[(sales["销售日期"] >= recent_start) & (sales["销量(千克)"] > 0), "单品编码"].unique())
    if len(eligible_codes) < 33:
        raise ValueError(f"retail_item_candidates_below_33:{len(eligible_codes)}")
    valid_sales = sales.loc[(sales["销量(千克)"] > 0) & (sales["销售单价(元/千克)"] > 0)].assign(_revenue=lambda frame: frame["销量(千克)"] * frame["销售单价(元/千克)"])
    category_daily = valid_sales.groupby(["销售日期", "分类名称"], as_index=False).agg(销量=("销量(千克)", "sum"))
    daily_item = valid_sales.groupby(["销售日期", "单品编码"], as_index=False).agg(销量=("销量(千克)", "sum"), 销售额=("_revenue", "sum"))
    daily_item["均价"] = daily_item["销售额"] / daily_item["销量"]
    metadata = products.set_index("单品编码").to_dict(orient="index")
    scenarios = [{"name": "low", "probability": 0.2}, {"name": "central", "probability": 0.6}, {"name": "high", "probability": 0.2}]
    factors = {"low": 0.75, "central": 1.0, "high": 1.30}
    items = []
    rows = []
    category_candidates: Dict[str, int] = {}
    reference_budget = 0.0
    for code in eligible_codes:
        meta = metadata[code]
        category = str(meta["分类名称"])
        item_name = str(meta["单品名称"])
        category_candidates[category] = category_candidates.get(category, 0) + 1
        history = daily_item.loc[daily_item["单品编码"] == code].set_index("销售日期").sort_index()
        index = pd.date_range(max(history.index.min(), cutoff - pd.Timedelta(days=179)), cutoff, freq="D")
        history = history.reindex(index)
        history["销量"] = history["销量"].fillna(0.0)
        history["均价"] = history["均价"].ffill().bfill()
        positive = history.loc[(history["销量"] > 0) & history["均价"].notna()]
        raw_elasticity, elasticity, fit_rows = _estimate_elasticity(positive) if len(positive) else (None, -1.0, 0)
        same_weekday = history.loc[history.index.dayofweek == decision.dayofweek, "销量"].tail(8)
        base_demand = float(max(same_weekday.median() if len(same_weekday) else history["销量"].tail(7).median(), 0.1))
        recent_cost = wholesale.loc[(wholesale["单品编码"] == code) & (wholesale["日期"] >= cutoff - pd.Timedelta(days=13)), "批发价格(元/千克)"].mean()
        if not math.isfinite(float(recent_cost)):
            recent_cost = wholesale.loc[wholesale["单品编码"] == code, "批发价格(元/千克)"].tail(10).mean()
        if not math.isfinite(float(recent_cost)):
            continue
        reference_price = float(positive["均价"].tail(14).median()) if len(positive) else float(recent_cost) * 1.3
        loss_rate = float(np.clip(float(loss_map.get(code, 10.0)) / 100.0, 0.0, 0.95))
        options = []
        for price in _price_grid(float(recent_cost), reference_price):
            adjusted = max(0.0, base_demand * (price / reference_price) ** elasticity)
            options.append({"price": price, "demand": {name: float(adjusted * factor) for name, factor in factors.items()}})
        high_demand = max(option["demand"]["high"] for option in options)
        max_replenishment = max(2.5, high_demand / max(1.0 - loss_rate, 1e-6) * 1.10)
        decision_name = f"{decision.date()}|{item_name}({code})"
        elasticity_range = [max(-3.0, elasticity - 0.75), min(-0.1, elasticity + 0.25)]
        items.append({"name": decision_name, "group": category, "category": category, "item_code": code, "unit_cost": float(recent_cost), "loss_rate": loss_rate, "base_demand": base_demand, "reference_price": reference_price, "elasticity": elasticity, "stress_elasticity_range": elasticity_range, "min_replenishment": 2.5, "max_replenishment": max_replenishment, "price_options": options})
        reference_budget += float(recent_cost) * max_replenishment
        rows.append({"decision_date": str(decision.date()), "item_code": code, "item_name": item_name, "category": category, "unit_cost": float(recent_cost), "loss_rate": loss_rate, "base_demand": base_demand, "raw_elasticity": raw_elasticity, "used_elasticity": elasticity, "elasticity_fit_rows": fit_rows, "reference_price": reference_price})
    if len(items) < 33:
        raise ValueError(f"retail_item_candidates_with_cost_below_33:{len(items)}")
    group_bounds = {category: {"min": 1, "max": sum(item["category"] == category for item in items)} for category in sorted({item["category"] for item in items})}
    demand_correlation = _estimate_category_correlation(category_daily, sorted(group_bounds))
    service_requirements = []
    for category in group_bounds:
        forecast_row = forecast_row_by_category.get(category)
        if forecast_binding is not None and forecast_row is None:
            raise ValueError(f"category_forecast_missing_for_item_service:{category}")
        if forecast_row is not None:
            for scenario in scenarios:
                factor = float(forecast_row[f"{scenario['name']}_factor"])
                service_requirements.append({"group": category, "scenario": scenario["name"], "minimum_sales": float(forecast_row["base_forecast"]) * factor * category_service_level})
    model = {"risk_measure": "cvar", "alpha": 0.8, "budget": reference_budget, "min_selected": 27, "max_selected": 33, "group_selection_bounds": group_bounds, "category_service_requirements": service_requirements, "shortage_penalty": 5.0, "baseline_markup": 0.2, "stress_test": {"seed": 20230701, "replications": 1000, "demand_cv": 0.25, "demand_correlation": demand_correlation}, "scenarios": scenarios, "items": items}
    inputs_path = output_dir / "retail_item_model_inputs.csv"
    pd.DataFrame(rows).to_csv(inputs_path, index=False, encoding="utf-8")
    manifest = {"version": "v44", "kind": "cumcm2023c_item_replenishment_pricing", "training_cutoff": str(cutoff.date()), "decision_dates": [str(decision.date())], "eligible_window": [str(recent_start.date()), str(cutoff.date())], "sources": [{"alias": alias, "path": str(path), "sha256": _sha256(path)} for alias, path in paths.items()], "derived_input_path": str(inputs_path.resolve()), "derived_input_sha256": _sha256(inputs_path), "candidate_item_count": len(items), "selection_bounds": [27, 33], "minimum_display_kg": 2.5, "category_candidate_counts": category_candidates, "category_service_level": category_service_level, "category_service_requirement_count": len(service_requirements), "upstream_category_forecast": forecast_binding, "model_sha256": _json_hash(model)}
    if forecast_binding is not None:
        manifest["sources"].extend([
            {"alias": "category_forecast_manifest", "path": forecast_binding["manifest_path"], "sha256": forecast_binding["manifest_sha256"]},
            {"alias": "category_forecast", "path": forecast_binding["forecast_path"], "sha256": forecast_binding["forecast_sha256"]},
        ])
    manifest_path = output_dir / "retail_item_decision_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"model": model, "data_file": str(inputs_path.resolve()), "manifest_path": str(manifest_path.resolve()), "manifest": manifest}


def _estimate_elasticity(history: pd.DataFrame) -> tuple[float | None, float, int]:
    frame = history.loc[(history["销量"] > 0) & (history["均价"] > 0), ["销量", "均价"]].tail(365).copy()
    if len(frame) < 30 or frame["均价"].nunique() < 3:
        return None, -1.0, len(frame)
    weekday = pd.get_dummies(frame.index.dayofweek, prefix="weekday", drop_first=True, dtype=float)
    trend = np.linspace(0.0, 1.0, len(frame))
    matrix = np.column_stack([np.ones(len(frame)), np.log(frame["均价"].to_numpy()), trend, weekday.to_numpy()])
    coefficients, *_ = np.linalg.lstsq(matrix, np.log(frame["销量"].to_numpy()), rcond=None)
    raw = float(coefficients[1])
    used = float(np.clip(raw, -3.0, -0.1)) if math.isfinite(raw) and raw < 0 else -1.0
    return raw if math.isfinite(raw) else None, used, len(frame)


def _estimate_category_correlation(daily: pd.DataFrame, categories: list[str]) -> Dict[str, Any]:
    frame = daily.copy()
    frame["销售日期"] = pd.to_datetime(frame["销售日期"])
    wide = frame.pivot_table(index="销售日期", columns="分类名称", values="销量", aggfunc="sum").reindex(columns=categories).fillna(0.0).sort_index().tail(365)
    logged = np.log1p(wide)
    residual = logged.copy()
    for weekday in range(7):
        mask = residual.index.dayofweek == weekday
        if mask.any():
            residual.loc[mask] = residual.loc[mask] - logged.loc[mask].mean(axis=0)
    raw = residual.corr().fillna(0.0).to_numpy(dtype=float)
    np.fill_diagonal(raw, 1.0)
    eigenvalues, eigenvectors = np.linalg.eigh((raw + raw.T) / 2.0)
    clipped = np.maximum(eigenvalues, 1e-6)
    positive = eigenvectors @ np.diag(clipped) @ eigenvectors.T
    scale = np.sqrt(np.diag(positive))
    correlation = positive / np.outer(scale, scale)
    correlation = np.clip((correlation + correlation.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    return {"groups": categories, "matrix": correlation.tolist(), "estimation": "365_day_log1p_sales_residual_after_weekday_mean_removal_nearest_psd"}


def _price_grid(unit_cost: float, reference_price: float) -> list[float]:
    raw = [max(unit_cost * 1.10, reference_price * 0.90), max(unit_cost * 1.25, reference_price), max(unit_cost * 1.45, reference_price * 1.10)]
    values = []
    for value in raw:
        rounded = round(float(value), 4)
        if rounded not in values:
            values.append(rounded)
    while len(values) < 3:
        values.append(round(values[-1] * 1.05, 4))
    return values


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
