from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from .schemas import DataAsset, ProblemProfile, read_text_safe
from .problem_contract import load_or_infer_contract

DATA_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xls", ".txt"}
TARGET_TERMS = ["target", "label", "score", "rank", "sales", "sale", "demand", "price", "power", "cost", "profit", "revenue", "y", "需求", "销量", "销售", "价格", "功率", "评分", "发病", "成本", "收益", "产量", "温度"]
ID_TERMS = ["id", "编号", "序号", "代码", "code", "坐标", "经度", "纬度", "index"]
ARCHETYPE_TERMS = {
    "forecasting": {"预测", "forecast", "时间", "日期", "销量", "销售", "价格", "功率", "需求", "序列"},
    "evaluation_ranking": {"评价", "排名", "评分", "score", "rank", "topsis", "熵权", "层次分析", "权重"},
    "constrained_optimization": {"优化", "约束", "调度", "分配", "路径", "选址", "成本", "收益", "整数规划"},
    "medical_environment_statistics": {"医学", "发病", "环境", "风险", "统计", "临床", "疾病"},
    "spatial_or_geostat": {"空间", "经纬", "坐标", "区域", "地理", "距离", "位置"},
    "mechanism_or_physics": {"温度", "传热", "物理", "机理", "微分", "材料", "热"},
    "simulation_or_queueing": {"仿真", "模拟", "排队", "队列", "随机过程", "monte carlo", "simulation", "queue", "replication"},
    "retail_forecast_operation": {"销售", "零售", "补货", "定价", "损耗", "蔬菜", "品类"},
}
METRIC_DIRECTIONS = {
    "MAE": "lower_better",
    "RMSE": "lower_better",
    "WAPE": "lower_better",
    "MAPE": "lower_better",
    "ranking_stability": "higher_better",
    "agreement_with_equal_weight": "diagnostic",
    "objective": "lower_better",
    "constraint_violation": "lower_better",
    "runtime_seconds": "lower_better",
    "R2": "higher_better",
    "score": "higher_better",
    "mean_wait": "lower_better",
    "ci_half_width": "lower_better",
    "utilization": "higher_better",
    "throughput": "higher_better",
}


def parse_problem(
    problem_id: str,
    data_root: Path,
    brief_path: Path | None = None,
    max_assets: int = 80,
    contract_path: Path | None = None,
) -> ProblemProfile:
    data_root = Path(data_root).resolve()
    brief_text = read_text_safe(Path(brief_path)) if brief_path else ""
    raw_assets = [_inspect_asset(path, brief_text, problem_id) for path in _candidate_files(data_root)]
    readable = [asset for asset in raw_assets if asset.readable]
    unreadable = [asset for asset in raw_assets if not asset.readable]
    assets = sorted(readable, key=lambda asset: asset.quality_score, reverse=True)[:max_assets]
    assets.extend(unreadable[: max(0, max_assets - len(assets))])
    corpus_text = " ".join([brief_text, problem_id] + [" ".join(asset.columns) for asset in assets])
    keywords = _extract_keywords(corpus_text)
    archetypes = _infer_archetypes(corpus_text, assets)
    targets = _target_candidates(assets)
    metrics = _metric_candidates(archetypes)
    notes = []
    if not readable:
        notes.append("no_readable_data: V44 will not fabricate executable modeling results.")
    if not brief_text:
        notes.append("missing_brief: archetype inference relies mainly on column names and problem_id.")
    if assets and assets[0].row_count is not None and assets[0].row_count < 20:
        notes.append("small_sample_warning: top-ranked table is only smoke evidence, not competition-grade evidence.")
    contract = load_or_infer_contract(problem_id, assets, archetypes, targets, brief_text, contract_path)
    if contract.task_type == "simulation":
        metrics = ["mean_wait", "ci_half_width", "utilization"]
    if contract.task_type == "spatial":
        metrics = ["MAE", "RMSE", "spatial_block_count"]
    if contract.specialist_routes:
        metrics = list(dict.fromkeys(str(metric) for route in contract.specialist_routes for metric in route.get("expected_metrics", [])))
    metric_directions = {metric: METRIC_DIRECTIONS.get(metric, "higher_better") for metric in metrics}
    metric_directions.update(contract.metric_directions)
    if contract.status != "ready":
        notes.append(f"contract_{contract.status}: {', '.join(contract.unresolved_fields)}")
    return ProblemProfile(
        problem_id=problem_id,
        brief_path=str(brief_path or ""),
        data_root=str(data_root),
        keywords=keywords,
        archetypes=archetypes,
        assets=assets,
        target_candidates=targets,
        metric_candidates=metrics,
        metric_directions=metric_directions,
        notes=notes,
        contract=contract,
    )


def _candidate_files(data_root: Path) -> List[Path]:
    files = [path for path in data_root.rglob("*") if path.is_file() and path.suffix.lower() in DATA_SUFFIXES]
    return sorted(files, key=lambda path: (-path.stat().st_size, str(path)))


def _inspect_asset(path: Path, brief_text: str, problem_id: str) -> DataAsset:
    asset = DataAsset(path=str(path.resolve()), suffix=path.suffix.lower(), size_bytes=path.stat().st_size)
    try:
        frame = _read_table_sample(path)
        if path.suffix.lower() in {".xlsx", ".xls"}:
            excel = pd.ExcelFile(path)
            asset.sheet_names = [str(name) for name in excel.sheet_names]
            for sheet_name in excel.sheet_names:
                sheet = pd.read_excel(path, sheet_name=sheet_name, nrows=0)
                asset.sheet_columns[str(sheet_name)] = [str(col) for col in sheet.columns]
        asset.columns = [str(col) for col in frame.columns]
        asset.row_count = _table_row_count(path)
        asset.column_count = int(len(frame.columns))
        asset.readable = True
        asset.target_score = _target_score(asset.columns)
        asset.quality_score = _quality_score(asset, brief_text, problem_id)
    except Exception as exc:
        asset.error = str(exc)[:200]
    return asset


def _read_table_sample(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv", ".txt"}:
        sep = "\t" if suffix == ".tsv" else None
        last_error = None
        for encoding in ("utf-8", "gb18030", "gbk", "utf-16"):
            try:
                return pd.read_csv(path, sep=sep, engine="python", nrows=500, encoding=encoding)
            except Exception as exc:
                last_error = exc
        raise last_error or RuntimeError("csv read failed")
    return pd.read_excel(path, nrows=500)


def _table_row_count(path: Path) -> int:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv", ".txt"}:
        sep = "\t" if suffix == ".tsv" else None
        last_error = None
        for encoding in ("utf-8", "gb18030", "gbk", "utf-16"):
            try:
                return int(len(pd.read_csv(path, sep=sep, engine="python", encoding=encoding, usecols=[0])))
            except Exception as exc:
                last_error = exc
        raise last_error or RuntimeError("csv row count failed")
    return int(len(pd.read_excel(path, usecols=[0])))


def _quality_score(asset: DataAsset, brief_text: str, problem_id: str) -> float:
    rows = asset.row_count or 0
    cols = asset.column_count or 0
    row_score = min(math.log10(max(rows, 1)) / 3.0, 1.0) * 4.0
    col_score = min(cols / 12.0, 1.0) * 2.0
    target_score = asset.target_score * 2.0
    text = (brief_text + " " + problem_id).lower()
    path_text = asset.path.lower()
    keyword_score = sum(0.4 for term in _extract_keywords(text)[:12] if term.lower() in path_text or any(term.lower() in col.lower() for col in asset.columns))
    penalty = 2.5 if rows < 20 else 0.0
    return round(row_score + col_score + target_score + keyword_score - penalty, 6)


def _target_score(columns: Iterable[str]) -> float:
    score = 0.0
    for column in columns:
        lowered = str(column).lower()
        if any(term.lower() in lowered for term in TARGET_TERMS):
            score += 1.0
        if any(term.lower() in lowered for term in ID_TERMS):
            score -= 0.5
    return max(0.0, min(score, 3.0))


def _extract_keywords(text: str) -> List[str]:
    chinese = re.findall(r"[\u4e00-\u9fff]{2,8}", text)
    ascii_words = re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", text.lower())
    stop = {"data", "sheet", "unnamed", "index", "column", "problem"}
    terms = [term for term in chinese + ascii_words if term.lower() not in stop]
    counts = {}
    for term in terms:
        counts[term] = counts.get(term, 0) + 1
    return [term for term, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:30]]


def _infer_archetypes(text: str, assets: Iterable[DataAsset]) -> List[str]:
    hits = []
    lowered = text.lower()
    for archetype, terms in ARCHETYPE_TERMS.items():
        score = sum(1 for term in terms if term.lower() in lowered)
        if score:
            hits.append((archetype, score))
    columns = " ".join(" ".join(asset.columns) for asset in assets).lower()
    if any(term in columns for term in ["date", "time", "日期", "时间"]):
        hits.append(("forecasting", 2))
    if any(term in columns for term in ["score", "评分", "评价", "rank"]):
        hits.append(("evaluation_ranking", 2))
    if not hits:
        hits.append(("generic_tabular_modeling", 1))
    merged = {}
    for name, score in hits:
        merged[name] = max(score, merged.get(name, 0))
    return [name for name, _ in sorted(merged.items(), key=lambda item: -item[1])[:4]]


def _target_candidates(assets: Iterable[DataAsset]) -> List[str]:
    candidates = []
    for asset in assets:
        for column in asset.columns:
            lowered = str(column).lower()
            if any(term.lower() in lowered for term in TARGET_TERMS) and not any(term.lower() in lowered for term in ID_TERMS):
                candidates.append(str(column))
    return list(dict.fromkeys(candidates))[:10]


def _metric_candidates(archetypes: Iterable[str]) -> List[str]:
    metrics = []
    for archetype in archetypes:
        if "forecast" in archetype or "retail" in archetype:
            metrics.extend(["MAE", "RMSE", "WAPE"])
        elif "evaluation" in archetype:
            metrics.extend(["ranking_stability"])
        elif "optimization" in archetype:
            metrics.extend(["objective", "constraint_violation", "runtime_seconds"])
        else:
            metrics.extend(["MAE", "RMSE", "R2", "score"])
    return list(dict.fromkeys(metrics))[:8]
