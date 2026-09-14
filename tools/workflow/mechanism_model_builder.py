from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def build_cumcm2018a_mechanism_data(source: Path, output_dir: Path) -> Dict[str, Any]:
    source = Path(source).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    temperature, materials = _extract_cumcm2018a(source)
    temperature_path = output_dir / "skin_temperature.csv"
    materials_path = output_dir / "material_layers.csv"
    temperature.to_csv(temperature_path, index=False, encoding="utf-8")
    materials.to_csv(materials_path, index=False, encoding="utf-8")
    model = {
        "kind": "first_order_relaxation",
        "time_column": "time_s",
        "state_columns": ["skin_temp_c"],
        "initial_state": {"skin_temp_c": float(temperature.iloc[0]["skin_temp_c"])},
        "parameters": {
            "k": {"lower": 1e-6, "upper": 0.05},
            "equilibrium": {"lower": 37.0, "upper": 80.0},
        },
        "multistart": 8,
        "seed": 2018,
        "rtol": 1e-8,
        "atol": 1e-10,
    }
    manifest = {
        "version": "v44", "kind": "cumcm2018a_mechanism_data",
        "source": {"path": str(source), "sha256": _sha256(source), "sheets": ["Appendix 1", "Appendix 2"]},
        "temperature": {"path": str(temperature_path), "sha256": _sha256(temperature_path), "rows": len(temperature), "columns": list(temperature.columns)},
        "materials": {"path": str(materials_path), "sha256": _sha256(materials_path), "rows": len(materials), "columns": list(materials.columns)},
        "parse_protocol": {"header_rows_skipped": 2, "thickness_range_policy": "store_bounds_and_midpoint"},
        "model_sha256": _json_hash(model),
    }
    manifest_path = output_dir / "mechanism_data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    verification = verify_mechanism_data_manifest(manifest_path, model)
    if not verification["passed"]:
        raise ValueError(f"mechanism_data_verification_failed:{verification['issues']}")
    return {
        "data_file": str(temperature_path), "materials_file": str(materials_path),
        "manifest_path": str(manifest_path), "model": model, "verification": verification,
    }


def verify_mechanism_data_manifest(manifest_path: Path, mechanism_model: Dict[str, Any]) -> Dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    issues = []
    checks: Dict[str, Any] = {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("kind") != "cumcm2018a_mechanism_data":
            issues.append("mechanism_manifest_kind_invalid")
        source = Path(str(manifest.get("source", {}).get("path", "")))
        temperature_path = Path(str(manifest.get("temperature", {}).get("path", "")))
        materials_path = Path(str(manifest.get("materials", {}).get("path", "")))
        for name, path in (("source", source), ("temperature", temperature_path), ("materials", materials_path)):
            binding = manifest.get(name, {})
            valid = path.is_file() and _sha256(path) == binding.get("sha256")
            checks[f"{name}_hash_valid"] = valid
            if not valid:
                issues.append(f"mechanism_{name}_hash_mismatch")
        if not issues:
            expected_temperature, expected_materials = _extract_cumcm2018a(source)
            actual_temperature = pd.read_csv(temperature_path)
            actual_materials = pd.read_csv(materials_path)
            temperature_equal = _frames_equal(actual_temperature, expected_temperature)
            materials_equal = _frames_equal(actual_materials, expected_materials)
            checks.update({"temperature_recomputed": temperature_equal, "materials_recomputed": materials_equal})
            if not temperature_equal:
                issues.append("mechanism_temperature_recompute_mismatch")
            if not materials_equal:
                issues.append("mechanism_materials_recompute_mismatch")
        model_matches = manifest.get("model_sha256") == _json_hash(mechanism_model)
        checks["model_matches"] = model_matches
        if not model_matches:
            issues.append("mechanism_model_binding_mismatch")
    except Exception as exc:
        issues.append(f"mechanism_manifest_verification_error:{exc}")
    return {"kind": "cumcm2018a_mechanism_data", "passed": not issues, "issues": issues, "checks": checks, "manifest_path": str(manifest_path)}


def _extract_cumcm2018a(source: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    temperature_raw = pd.read_excel(source, sheet_name="Appendix 2", header=None).iloc[2:, :2].copy()
    temperature_raw.columns = ["time_s", "skin_temp_c"]
    temperature = temperature_raw.apply(pd.to_numeric, errors="coerce").dropna().sort_values("time_s").reset_index(drop=True)
    if len(temperature) < 9 or temperature["time_s"].duplicated().any() or not np.all(np.diff(temperature["time_s"]) > 0):
        raise ValueError("cumcm2018a_temperature_series_invalid")

    material_raw = pd.read_excel(source, sheet_name="Appendix 1", header=None).iloc[2:, :5].copy()
    material_raw.columns = ["layer", "density_kg_m3", "specific_heat_j_kg_c", "conductivity_w_m_c", "thickness_mm_raw"]
    rows = []
    for row in material_raw.itertuples(index=False):
        thickness_min, thickness_max = _thickness_bounds(row.thickness_mm_raw)
        nominal = (thickness_min + thickness_max) / 2.0
        conductivity = float(row.conductivity_w_m_c)
        rows.append({
            "layer": str(row.layer), "density_kg_m3": float(row.density_kg_m3),
            "specific_heat_j_kg_c": float(row.specific_heat_j_kg_c),
            "conductivity_w_m_c": conductivity, "thickness_min_mm": thickness_min,
            "thickness_max_mm": thickness_max, "thickness_nominal_mm": nominal,
            "nominal_resistance_m2_c_w": nominal / 1000.0 / conductivity,
        })
    materials = pd.DataFrame(rows)
    if len(materials) != 4 or not np.isfinite(materials.select_dtypes(include="number").to_numpy()).all():
        raise ValueError("cumcm2018a_material_table_invalid")
    return temperature, materials


def _thickness_bounds(value: Any) -> tuple[float, float]:
    text = str(value).strip().replace("–", "-").replace("—", "-")
    parts = [float(part) for part in text.split("-")]
    if len(parts) == 1:
        return parts[0], parts[0]
    if len(parts) == 2 and 0 < parts[0] <= parts[1]:
        return parts[0], parts[1]
    raise ValueError(f"cumcm2018a_thickness_invalid:{value}")


def _frames_equal(actual: pd.DataFrame, expected: pd.DataFrame) -> bool:
    if list(actual.columns) != list(expected.columns) or actual.shape != expected.shape:
        return False
    for column in expected.columns:
        if pd.api.types.is_numeric_dtype(expected[column]):
            if not np.allclose(pd.to_numeric(actual[column]), expected[column], rtol=1e-12, atol=1e-12):
                return False
        elif actual[column].astype(str).tolist() != expected[column].astype(str).tolist():
            return False
    return True


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
