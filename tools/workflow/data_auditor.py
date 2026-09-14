from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List
from solver.schemas import ProblemContract

import pandas as pd

SUPPORTED_SUFFIXES = {".csv", ".tsv", ".txt", ".xlsx", ".xls"}
ENCODINGS = ("utf-8", "gb18030", "gbk", "utf-16")


def audit_data_root(data_root: Path, output_dir: Path, contract: ProblemContract | None = None) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(data_root).resolve()
    files = [path for path in data_root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES]
    file_reports = [_audit_file(path) for path in sorted(files, key=lambda item: str(item))]
    summary = {
        "version": "v44",
        "kind": "data_audit",
        "data_root": str(data_root),
        "file_count": len(file_reports),
        "readable_file_count": sum(1 for report in file_reports if report.get("readable")),
        "total_rows_observed": sum(_safe_int(report.get("row_count")) for report in file_reports),
        "files": file_reports,
        "warnings": _warnings(file_reports),
        "binding": {
            "contract_status": contract.status if contract else "missing",
            "data_file": contract.data_file if contract else "",
            "sheet_name": contract.sheet_name if contract else None,
            "target_column": contract.target_column if contract else "",
            "time_column": contract.time_column if contract else "",
        },
    }
    summary["passed"] = bool(summary["readable_file_count"] > 0 and not any(
        str(warning).startswith("unreadable_file:") for warning in summary["warnings"]
    ))
    (output_dir / "data_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "数据体检报告.md").write_text(_format_markdown(summary), encoding="utf-8")
    return summary


def _audit_file(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    base: Dict[str, Any] = {
        "path": str(path.resolve()),
        "suffix": suffix,
        "size_bytes": path.stat().st_size,
        "readable": False,
    }
    try:
        if suffix in {".csv", ".tsv", ".txt"}:
            base.update(_audit_delimited(path, suffix))
        else:
            base.update(_audit_excel(path))
        base["readable"] = True
    except Exception as exc:
        base["error"] = str(exc)[:500]
    return base


def _audit_delimited(path: Path, suffix: str) -> Dict[str, Any]:
    sep = "\t" if suffix == ".tsv" else None
    last_error: Exception | None = None
    for encoding in ENCODINGS:
        try:
            frame = pd.read_csv(path, sep=sep, engine="python", encoding=encoding)
            return {
                "encoding": encoding,
                "row_count": int(len(frame)),
                "column_count": int(len(frame.columns)),
                "columns": [str(column) for column in frame.columns],
                "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
                "missing_values": {str(column): int(value) for column, value in frame.isna().sum().items()},
                "numeric_describe": _numeric_describe(frame),
                "delimiter": "tab" if sep == "\t" else _sniff_delimiter(path, encoding),
            }
        except Exception as exc:
            last_error = exc
    raise last_error or RuntimeError("failed to read delimited file")


def _audit_excel(path: Path) -> Dict[str, Any]:
    workbook_meta = _openpyxl_meta(path)
    excel = pd.ExcelFile(path)
    sheets: List[Dict[str, Any]] = []
    for sheet_name in excel.sheet_names:
        frame = pd.read_excel(path, sheet_name=sheet_name)
        sheet_meta = workbook_meta.get(sheet_name, {})
        sheets.append({
            "sheet_name": str(sheet_name),
            "row_count": int(len(frame)),
            "column_count": int(len(frame.columns)),
            "columns": [str(column) for column in frame.columns],
            "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
            "missing_values": {str(column): int(value) for column, value in frame.isna().sum().items()},
            "numeric_describe": _numeric_describe(frame),
            "hidden": bool(sheet_meta.get("hidden", False)),
            "merged_cell_count": int(sheet_meta.get("merged_cell_count", 0)),
        })
    first = sheets[0] if sheets else {}
    return {
        "sheet_count": len(sheets),
        "sheet_names": [sheet["sheet_name"] for sheet in sheets],
        "row_count": first.get("row_count", 0),
        "column_count": first.get("column_count", 0),
        "columns": first.get("columns", []),
        "sheets": sheets,
    }


def _openpyxl_meta(path: Path) -> Dict[str, Dict[str, Any]]:
    try:
        import openpyxl

        workbook = openpyxl.load_workbook(path, read_only=False, data_only=True)
        meta = {}
        for sheet in workbook.worksheets:
            meta[sheet.title] = {
                "hidden": sheet.sheet_state != "visible",
                "merged_cell_count": len(sheet.merged_cells.ranges),
            }
        workbook.close()
        return meta
    except Exception:
        return {}


def _numeric_describe(frame: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    numeric = frame.select_dtypes(include="number")
    if numeric.empty:
        return {}
    desc = numeric.describe().transpose()
    fields = ["count", "mean", "std", "min", "max"]
    return {
        str(index): {field: _to_float(row.get(field)) for field in fields if field in row}
        for index, row in desc.iterrows()
    }


def _sniff_delimiter(path: Path, encoding: str) -> str:
    try:
        sample = path.read_text(encoding=encoding, errors="ignore")[:4096]
        dialect = csv.Sniffer().sniff(sample)
        return dialect.delimiter
    except Exception:
        return "auto"


def _to_float(value: Any) -> float:
    try:
        return round(float(value), 6)
    except Exception:
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _warnings(file_reports: List[Dict[str, Any]]) -> List[str]:
    warnings: List[str] = []
    if not file_reports:
        warnings.append("no_supported_data_file")
    if not any(report.get("readable") for report in file_reports):
        warnings.append("no_readable_data_file")
    for report in file_reports:
        if not report.get("readable"):
            warnings.append(f"unreadable_file: {report.get('path')}")
        if report.get("sheet_count", 0) and any(sheet.get("hidden") for sheet in report.get("sheets", [])):
            warnings.append(f"hidden_sheet_detected: {report.get('path')}")
        if report.get("sheet_count", 0) and any(sheet.get("merged_cell_count", 0) for sheet in report.get("sheets", [])):
            warnings.append(f"merged_cells_detected: {report.get('path')}")
    return warnings


def _format_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        "# 数据体检报告",
        "",
        f"- version: {summary['version']}",
        f"- data_root: {summary['data_root']}",
        f"- file_count: {summary['file_count']}",
        f"- readable_file_count: {summary['readable_file_count']}",
        f"- total_rows_observed: {summary['total_rows_observed']}",
        f"- binding: {summary.get('binding', {})}",
        "",
        "## Warnings",
    ]
    for warning in summary["warnings"] or ["none"]:
        lines.append(f"- {warning}")
    lines.extend(["", "## Files"])
    for report in summary["files"]:
        lines.append(f"### {Path(report['path']).name}")
        lines.append(f"- path: {report['path']}")
        lines.append(f"- readable: {report.get('readable')}")
        if report.get("error"):
            lines.append(f"- error: {report['error']}")
        if report.get("sheet_count"):
            lines.append(f"- sheet_count: {report.get('sheet_count')}")
            lines.append(f"- sheet_names: {report.get('sheet_names')}")
        lines.append(f"- rows: {report.get('row_count', '')}")
        lines.append(f"- columns: {report.get('column_count', '')}")
        lines.append(f"- column_names: {report.get('columns', [])}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
