from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from solver.schemas import ProblemContract

VALID_CARDINALITIES = {"one_to_one", "one_to_many", "many_to_one", "many_to_many"}
VALID_JOIN_TYPES = {"left", "inner", "right", "outer"}


def assemble_contract_data(contract: ProblemContract, output_dir: Path) -> Dict[str, Any] | None:
    if not contract.data_sources:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    sources: Dict[str, pd.DataFrame] = {}
    source_manifest = []
    for spec in contract.data_sources:
        alias = str(spec["alias"]); path = Path(spec["path"]).resolve()
        frame = _read_table(path, spec.get("sheet_name"), spec.get("columns"))
        frame.columns = [str(column) for column in frame.columns]
        sources[alias] = frame
        source_manifest.append({"alias": alias, "path": str(path), "sheet_name": spec.get("sheet_name"), "sha256": _sha256(path), "rows": len(frame), "columns": list(frame.columns)})

    assembled = sources[contract.base_table].copy()
    included = {contract.base_table}
    join_manifest = []
    for index, join in enumerate(contract.data_joins):
        left = str(join["left"]); right = str(join["right"])
        if left not in included:
            raise ValueError(f"join_left_not_yet_assembled:{index}:{left}")
        right_frame = sources[right].copy()
        left_on = _columns(join.get("left_on") or join.get("on"))
        right_on = _columns(join.get("right_on") or join.get("on"))
        if assembled[left_on].isna().any(axis=None) or right_frame[right_on].isna().any(axis=None):
            raise ValueError(f"join_null_key:{index}")
        selected_right = join.get("right_columns")
        if selected_right is not None:
            right_frame = right_frame[list(dict.fromkeys(right_on + [str(value) for value in selected_right]))]
        prefix = str(join.get("right_prefix", f"{right}__"))
        rename = {column: f"{prefix}{column}" for column in right_frame.columns if column not in right_on}
        right_frame = right_frame.rename(columns=rename)
        left_keys = assembled[left_on].drop_duplicates()
        right_keys = right_frame[right_on].drop_duplicates()
        key_probe = left_keys.merge(right_keys, left_on=left_on, right_on=right_on, how="outer", indicator=True)
        unmatched_left = int((key_probe["_merge"] == "left_only").sum())
        unmatched_right = int((key_probe["_merge"] == "right_only").sum())
        left_rate = unmatched_left / max(len(left_keys), 1); right_rate = unmatched_right / max(len(right_keys), 1)
        if left_rate > float(join.get("max_unmatched_left_rate", 1.0)) + 1e-12:
            raise ValueError(f"join_unmatched_left_rate_exceeded:{index}:{left_rate}")
        if right_rate > float(join.get("max_unmatched_right_rate", 1.0)) + 1e-12:
            raise ValueError(f"join_unmatched_right_rate_exceeded:{index}:{right_rate}")
        before = len(assembled)
        assembled = assembled.merge(right_frame, left_on=left_on, right_on=right_on, how=str(join["how"]), validate=str(join["validate"]), sort=False)
        for right_key, left_key in zip(right_on, left_on):
            if right_key != left_key and right_key in assembled.columns:
                assembled = assembled.drop(columns=[right_key])
        included.add(right)
        join_manifest.append({"index": index, "left": left, "right": right, "how": join["how"], "validate": join["validate"], "left_on": left_on, "right_on": right_on, "rows_before": before, "rows_after": len(assembled), "unmatched_left_keys": unmatched_left, "unmatched_right_keys": unmatched_right, "unmatched_left_rate": left_rate, "unmatched_right_rate": right_rate, "renamed_columns": rename})

    aggregation_manifest = None
    if contract.data_aggregation:
        aggregation = contract.data_aggregation
        group_by = [str(value) for value in aggregation["group_by"]]
        named = {str(output): pd.NamedAgg(column=str(spec["column"]), aggfunc=str(spec["function"])) for output, spec in aggregation["aggregations"].items()}
        rows_before = len(assembled)
        assembled = assembled.groupby(group_by, dropna=False, sort=False).agg(**named).reset_index()
        sort_by = [str(value) for value in aggregation.get("sort_by", [])]
        if sort_by:
            assembled = assembled.sort_values(sort_by, kind="stable").reset_index(drop=True)
        aggregation_manifest = {"group_by": group_by, "aggregations": aggregation["aggregations"], "sort_by": sort_by, "rows_before": rows_before, "rows_after": len(assembled)}

    output_path = output_dir / "assembled_dataset.csv"
    assembled.to_csv(output_path, index=False, encoding="utf-8")
    manifest = {"version": "v44", "base_table": contract.base_table, "sources": source_manifest, "joins": join_manifest, "aggregation": aggregation_manifest, "output_path": str(output_path.resolve()), "output_sha256": _sha256(output_path), "output_rows": len(assembled), "output_columns": list(assembled.columns)}
    manifest_path = output_dir / "assembly_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    contract.data_file = str(output_path.resolve()); contract.sheet_name = None; contract.data_assembly_manifest = str(manifest_path.resolve())
    return {**manifest, "manifest_path": str(manifest_path.resolve()), "manifest_sha256": _sha256(manifest_path)}


def _columns(value: Any) -> list[str]:
    if isinstance(value, str) and value:
        return [value]
    if isinstance(value, list) and value and all(isinstance(item, str) and item for item in value):
        return list(value)
    raise ValueError("join_keys_invalid")


def _read_table(path: Path, sheet_name: Any = None, columns: Any = None) -> pd.DataFrame:
    usecols = [str(value) for value in columns] if isinstance(columns, list) and columns else None
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name if sheet_name is not None else 0, usecols=usecols)
    sep = "\t" if path.suffix.lower() == ".tsv" else None
    last_error = None
    for encoding in ("utf-8", "gb18030", "gbk", "utf-16"):
        try:
            return pd.read_csv(path, sep=sep, engine="python", encoding=encoding, usecols=usecols)
        except Exception as exc:
            last_error = exc
    raise last_error or RuntimeError(f"cannot_read_source:{path}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
