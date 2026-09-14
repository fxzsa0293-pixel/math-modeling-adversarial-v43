"""
V26 evaluation archetype competition-ready suite.

Runs a real multi-criteria evaluation case (CUMCM 2012A wine evaluation) from
local reference materials and builds evidence for entropy weighting, TOPSIS,
equal-weight baselines, rank stability, and judge-consistency auditing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


EVALUATION_CASE_HINT = "CUMCM2012A题：葡萄酒的评价（源码+数据）"


def run_v26_evaluation_suite(reference_root: Path | str) -> Dict[str, Any]:
    reference_root = _resolve_reference_root(Path(reference_root))
    data_path = _find_wine_data(reference_root)
    if not data_path:
        return {"status": "blocked", "reason": "2012A wine evaluation data not found", "reference_root": str(reference_root)}
    datasets = _load_score_datasets(data_path)
    case_results = []
    for dataset in datasets:
        case_results.append(_evaluate_dataset(dataset))
    aggregate = _aggregate_case_results(case_results)
    return {
        "status": "executed",
        "archetype": "multi_criteria_evaluation",
        "case": "CUMCM2012A_wine_evaluation",
        "data_path": str(data_path),
        "datasets": case_results,
        "aggregate": aggregate,
        "evidence_contract_status": "competition_ready" if aggregate.get("datasets_evaluated", 0) >= 4 else "needs_more_cases",
    }


def format_v26_evaluation_report(suite: Dict[str, Any]) -> str:
    lines = [
        "# V26 综合评价类真实数据证据包报告",
        "",
        f"- status: {suite.get('status')}",
        f"- archetype: {suite.get('archetype')}",
        f"- case: {suite.get('case')}",
        f"- data_path: {suite.get('data_path')}",
        f"- evidence_contract_status: {suite.get('evidence_contract_status')}",
        "",
        "## Aggregate evidence",
        f"- {suite.get('aggregate', {})}",
        "",
        "## Dataset leaderboards",
    ]
    for item in suite.get("datasets", []):
        lines.append(f"- {item.get('sheet')}: samples={item.get('n_samples')}, criteria={item.get('n_criteria')}")
        lines.append(f"  - top_entropy_topsis: {item.get('top_entropy_topsis')}")
        lines.append(f"  - top_equal_topsis: {item.get('top_equal_topsis')}")
        lines.append(f"  - rank_stability: {item.get('rank_stability')}")
        lines.append(f"  - judge_consistency: {item.get('judge_consistency')}")
        lines.append(f"  - weight_summary: {item.get('weight_summary')}")
    lines.extend([
        "",
        "## Paper-ready claims",
    ])
    for claim in _paper_claims(suite):
        lines.append(f"- {claim}")
    lines.extend([
        "",
        "## Figure descriptions",
        "- 权重柱状图：展示各评价指标的熵权分布，解释核心评价维度。",
        "- TOPSIS排序对比图：比较熵权TOPSIS与等权TOPSIS排名差异。",
        "- 排名稳定性箱线/扰动图：展示权重扰动下Top样品是否稳定。",
        "- 评委一致性热力图：展示评分者之间相关性与潜在异常评委。",
    ])
    return "\n".join(lines) + "\n"


def update_v25_registry_with_v26(archetype_registry: Dict[str, Any], v26_suite: Dict[str, Any]) -> Dict[str, Any]:
    updated = dict(archetype_registry)
    registry = {key: dict(value) for key, value in updated.get("registry", {}).items()}
    item = dict(registry.get("multi_criteria_evaluation", {}))
    evidence = list(item.get("evidence", []))
    if v26_suite.get("status") == "executed":
        evidence.append({
            "source": "v26_evaluation_competition_pack",
            "case": v26_suite.get("case"),
            "primary_status": v26_suite.get("evidence_contract_status"),
            "datasets_evaluated": v26_suite.get("aggregate", {}).get("datasets_evaluated"),
            "mean_top1_overlap": v26_suite.get("aggregate", {}).get("mean_top1_overlap"),
            "mean_rank_correlation": v26_suite.get("aggregate", {}).get("mean_rank_correlation"),
        })
        item["status"] = "competition_ready"
        item["evidence"] = evidence
        item["coverage"] = {**item.get("coverage", {}), "has_specialized_pack": True, "evidence_items": len(evidence)}
        item["missing_to_competition_ready"] = []
        item["next_builder_tasks"] = ["add AHP subjective-weight variant", "add grey-relational comparison", "add multi-case evaluation benchmark"]
    registry["multi_criteria_evaluation"] = item
    updated["registry"] = registry
    summary = dict(updated.get("summary", {}))
    summary["competition_ready"] = sum(1 for value in registry.values() if value.get("status") == "competition_ready")
    summary["prototype_ready"] = sum(1 for value in registry.values() if value.get("status") == "prototype_ready")
    summary["contract_only"] = sum(1 for value in registry.values() if value.get("status") == "contract_only")
    updated["summary"] = summary
    return updated


def _find_wine_data(reference_root: Path) -> Path | None:
    if not reference_root.exists():
        return None
    candidates = list(reference_root.rglob("2012A_T1_processed.xls"))
    if candidates:
        candidates = sorted(candidates, key=lambda p: ("优秀论文呢" not in str(p), len(str(p))))
        return candidates[0]
    return None


def _load_score_datasets(data_path: Path) -> List[Dict[str, Any]]:
    datasets = []
    excel = pd.ExcelFile(data_path)
    for sheet in excel.sheet_names:
        raw = pd.read_excel(data_path, sheet_name=sheet, header=None)
        parsed = _parse_score_sheet(raw)
        if parsed.get("matrix") is not None:
            parsed["sheet"] = sheet
            datasets.append(parsed)
    return datasets


def _parse_score_sheet(raw: pd.DataFrame) -> Dict[str, Any]:
    data = raw.iloc[2:].copy()
    sample_ids = pd.to_numeric(data.iloc[:, 0], errors="coerce")
    judge_scores = data.iloc[:, 3:13].apply(pd.to_numeric, errors="coerce")
    valid = sample_ids.notna() & (judge_scores.isna().mean(axis=1) < 0.5)
    data = data.loc[valid]
    sample_ids = sample_ids.loc[valid].astype(int)
    judge_scores = judge_scores.loc[valid].fillna(judge_scores.median(numeric_only=True)).fillna(0)
    if data.empty:
        return {"matrix": None, "reason": "empty parsed score sheet"}
    criterion_names = []
    criterion_scores = []
    for row_index, sample_id in enumerate(sample_ids):
        label = str(data.iloc[row_index, 2])
        if label.lower() == "nan":
            label = str(data.iloc[row_index, 1])
        if label in criterion_names:
            label = f"{label}_{len([x for x in criterion_names if x.startswith(label)]) + 1}"
        criterion_names.append(label)
    samples = sorted(sample_ids.unique())
    matrix = []
    judge_matrix_rows = []
    for sample in samples:
        mask = sample_ids == sample
        sample_rows = judge_scores.loc[mask]
        matrix.append(sample_rows.mean(axis=1).to_numpy(dtype=float))
        judge_matrix_rows.append(sample_rows.mean(axis=0).to_numpy(dtype=float))
    matrix = np.vstack(matrix)
    judge_matrix = np.vstack(judge_matrix_rows)
    min_cols = min(matrix.shape[1], len(criterion_names))
    return {
        "matrix": matrix[:, :min_cols],
        "criteria": criterion_names[:min_cols],
        "samples": [int(x) for x in samples],
        "judge_matrix": judge_matrix,
    }


def _evaluate_dataset(dataset: Dict[str, Any]) -> Dict[str, Any]:
    matrix = np.asarray(dataset["matrix"], dtype=float)
    entropy_weights = _entropy_weights(matrix)
    equal_weights = np.ones(matrix.shape[1]) / matrix.shape[1]
    entropy_scores = _topsis_scores(matrix, entropy_weights)
    equal_scores = _topsis_scores(matrix, equal_weights)
    entropy_rank = _rank_desc(entropy_scores)
    equal_rank = _rank_desc(equal_scores)
    stability = _rank_stability(matrix, entropy_weights, entropy_rank)
    consistency = _judge_consistency(dataset.get("judge_matrix"))
    return {
        "sheet": dataset.get("sheet"),
        "n_samples": int(matrix.shape[0]),
        "n_criteria": int(matrix.shape[1]),
        "top_entropy_topsis": _top_list(dataset["samples"], entropy_scores, entropy_rank),
        "top_equal_topsis": _top_list(dataset["samples"], equal_scores, equal_rank),
        "rank_stability": stability,
        "judge_consistency": consistency,
        "weight_summary": {
            "min": float(np.min(entropy_weights)),
            "max": float(np.max(entropy_weights)),
            "top_criteria": _top_weighted_criteria(dataset["criteria"], entropy_weights),
        },
    }


def _entropy_weights(matrix: np.ndarray) -> np.ndarray:
    shifted = matrix - np.nanmin(matrix, axis=0)
    denom = np.nanmax(shifted, axis=0)
    normalized = np.divide(shifted, denom, out=np.zeros_like(shifted), where=denom > 1e-12) + 1e-12
    p = normalized / normalized.sum(axis=0, keepdims=True)
    k = 1 / np.log(matrix.shape[0])
    entropy = -k * np.sum(p * np.log(p), axis=0)
    diversity = 1 - entropy
    if np.all(diversity <= 1e-12):
        return np.ones(matrix.shape[1]) / matrix.shape[1]
    return diversity / diversity.sum()


def _topsis_scores(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(matrix, axis=0)
    normalized = np.divide(matrix, norm, out=np.zeros_like(matrix), where=norm > 1e-12)
    weighted = normalized * weights
    ideal = weighted.max(axis=0)
    nadir = weighted.min(axis=0)
    d_pos = np.linalg.norm(weighted - ideal, axis=1)
    d_neg = np.linalg.norm(weighted - nadir, axis=1)
    return d_neg / (d_pos + d_neg + 1e-12)


def _rank_desc(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-scores)


def _top_list(samples: List[int], scores: np.ndarray, rank: np.ndarray, n: int = 5) -> List[Dict[str, Any]]:
    return [{"sample": int(samples[idx]), "score": float(scores[idx])} for idx in rank[:n]]


def _rank_stability(matrix: np.ndarray, weights: np.ndarray, base_rank: np.ndarray) -> Dict[str, Any]:
    rng = np.random.default_rng(42)
    top1 = int(base_rank[0])
    top3 = set(int(x) for x in base_rank[:3])
    top1_hits = 0
    top3_jaccards = []
    correlations = []
    base_positions = np.empty_like(base_rank)
    base_positions[base_rank] = np.arange(len(base_rank))
    for _ in range(200):
        noise = rng.normal(1.0, 0.12, size=len(weights))
        perturbed = np.clip(weights * noise, 1e-9, None)
        perturbed = perturbed / perturbed.sum()
        scores = _topsis_scores(matrix, perturbed)
        rank = _rank_desc(scores)
        if int(rank[0]) == top1:
            top1_hits += 1
        top3_new = set(int(x) for x in rank[:3])
        top3_jaccards.append(len(top3 & top3_new) / len(top3 | top3_new))
        positions = np.empty_like(rank)
        positions[rank] = np.arange(len(rank))
        correlations.append(float(np.corrcoef(base_positions, positions)[0, 1]))
    return {
        "top1_retention": top1_hits / 200,
        "mean_top3_jaccard": float(np.mean(top3_jaccards)),
        "mean_rank_correlation": float(np.mean(correlations)),
    }


def _judge_consistency(judge_matrix: Any) -> Dict[str, Any]:
    if judge_matrix is None:
        return {"status": "missing"}
    arr = np.asarray(judge_matrix, dtype=float)
    if arr.shape[1] < 2:
        return {"status": "insufficient_judges"}
    corr = np.corrcoef(arr, rowvar=False)
    upper = corr[np.triu_indices_from(corr, k=1)]
    return {
        "mean_pairwise_corr": float(np.nanmean(upper)),
        "min_pairwise_corr": float(np.nanmin(upper)),
        "low_consistency_pairs": int(np.sum(upper < 0.3)),
    }


def _top_weighted_criteria(criteria: List[str], weights: np.ndarray, n: int = 5) -> List[Dict[str, Any]]:
    order = np.argsort(-weights)
    return [{"criterion": str(criteria[idx]), "weight": float(weights[idx])} for idx in order[:n]]


def _aggregate_case_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {"datasets_evaluated": 0}
    return {
        "datasets_evaluated": len(results),
        "mean_top1_overlap": float(np.mean([item["rank_stability"]["top1_retention"] for item in results])),
        "mean_top3_jaccard": float(np.mean([item["rank_stability"]["mean_top3_jaccard"] for item in results])),
        "mean_rank_correlation": float(np.mean([item["rank_stability"]["mean_rank_correlation"] for item in results])),
        "mean_judge_corr": float(np.mean([item["judge_consistency"].get("mean_pairwise_corr", np.nan) for item in results])),
    }


def _paper_claims(suite: Dict[str, Any]) -> List[str]:
    if suite.get("status") != "executed":
        return ["Evaluation archetype evidence is blocked until a real indicator matrix is available."]
    agg = suite.get("aggregate", {})
    return [
        "Entropy-weight TOPSIS can be used as the strong objective-weighting path for multi-criteria evaluation.",
        "Equal-weight TOPSIS provides a transparent baseline that the entropy-weight route must be compared against.",
        f"Weight perturbation gives rank-stability evidence: mean top-1 retention={agg.get('mean_top1_overlap'):.3f}.",
        f"Judge-consistency audit is available: mean pairwise judge correlation={agg.get('mean_judge_corr'):.3f}.",
    ]


def _resolve_reference_root(candidate: Path) -> Path:
    if candidate.exists():
        return candidate
    toolkit_root = Path(__file__).resolve().parent
    for parent in [toolkit_root, *toolkit_root.parents]:
        if not parent.exists():
            continue
        for child in parent.iterdir():
            if child.is_dir() and child.name.startswith("05_"):
                return child
    return candidate
