from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from .table_cache import read_excel_cached


def analyze_cumcm2023c_sales(product_path: Path, sales_path: Path, output_dir: Path, cutoff: str = "2023-06-30", min_item_days: int = 60, top_edges: int = 100) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    product_path = Path(product_path).resolve()
    sales_path = Path(sales_path).resolve()
    products = read_excel_cached(product_path, "Sheet1", ["单品编码", "单品名称", "分类名称"])
    sales = read_excel_cached(sales_path, "Sheet1", ["销售日期", "单品编码", "销量(千克)", "销售单价(元/千克)"])
    products["单品编码"] = products["单品编码"].astype(str)
    sales["单品编码"] = sales["单品编码"].astype(str)
    sales["销售日期"] = pd.to_datetime(sales["销售日期"])
    sales = sales.loc[(sales["销售日期"] <= pd.Timestamp(cutoff)) & (sales["销量(千克)"] >= 0)].merge(products, on="单品编码", how="left", validate="many_to_one")
    if sales["分类名称"].isna().any():
        raise ValueError("analysis_product_mapping_incomplete")
    category_daily = sales.groupby(["销售日期", "分类名称"], as_index=False)["销量(千克)"].sum()
    category_wide = category_daily.pivot(index="销售日期", columns="分类名称", values="销量(千克)").fillna(0.0).sort_index()
    distribution = category_wide.describe(percentiles=[0.25, 0.5, 0.75]).T.reset_index().rename(columns={"分类名称": "category", "count": "days", "mean": "mean", "std": "std", "min": "min", "25%": "q25", "50%": "median", "75%": "q75", "max": "max"})
    distribution["cv"] = distribution["std"] / distribution["mean"].replace(0.0, np.nan)
    pearson = category_wide.corr(method="pearson")
    spearman = category_wide.corr(method="spearman")
    weekday = category_daily.assign(weekday=category_daily["销售日期"].dt.dayofweek).groupby(["分类名称", "weekday"], as_index=False)["销量(千克)"].agg(["mean", "median", "std"]).reset_index()

    item_daily = sales.groupby(["销售日期", "单品编码"], as_index=False)["销量(千克)"].sum()
    support = item_daily.groupby("单品编码")["销售日期"].nunique()
    eligible = support.loc[support >= min_item_days].index
    item_wide = item_daily.loc[item_daily["单品编码"].isin(eligible)].pivot(index="销售日期", columns="单品编码", values="销量(千克)").fillna(0.0).sort_index()
    item_corr = item_wide.corr(method="spearman")
    item_meta = products.set_index("单品编码").to_dict(orient="index")
    edges = []
    columns = list(item_corr.columns)
    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1:]:
            correlation = float(item_corr.loc[left, right])
            if np.isfinite(correlation):
                edges.append({"left_code": left, "left_name": item_meta[left]["单品名称"], "left_category": item_meta[left]["分类名称"], "right_code": right, "right_name": item_meta[right]["单品名称"], "right_category": item_meta[right]["分类名称"], "spearman": correlation, "absolute_spearman": abs(correlation)})
    edges = sorted(edges, key=lambda item: (-item["absolute_spearman"], item["left_code"], item["right_code"]))[:top_edges]

    paths = {
        "category_distribution_csv": output_dir / "category_distribution.csv",
        "category_pearson_csv": output_dir / "category_pearson.csv",
        "category_spearman_csv": output_dir / "category_spearman.csv",
        "category_weekday_csv": output_dir / "category_weekday.csv",
        "item_correlation_edges_csv": output_dir / "item_correlation_edges.csv",
    }
    distribution.to_csv(paths["category_distribution_csv"], index=False, encoding="utf-8-sig")
    pearson.to_csv(paths["category_pearson_csv"], encoding="utf-8-sig")
    spearman.to_csv(paths["category_spearman_csv"], encoding="utf-8-sig")
    weekday.to_csv(paths["category_weekday_csv"], index=False, encoding="utf-8-sig")
    pd.DataFrame(edges).to_csv(paths["item_correlation_edges_csv"], index=False, encoding="utf-8-sig")
    strongest_category_pairs = []
    for left_index, left in enumerate(pearson.columns):
        for right in pearson.columns[left_index + 1:]:
            strongest_category_pairs.append({"left": left, "right": right, "pearson": float(pearson.loc[left, right]), "spearman": float(spearman.loc[left, right])})
    strongest_category_pairs.sort(key=lambda item: -abs(item["spearman"]))
    report = output_dir / "sales_distribution_and_relationships.md"
    lines = ["# 销量分布与关联分析", "", f"分析截止日期：{cutoff}。品类相关性基于按日销量序列；单品相关性仅纳入至少 {min_item_days} 个有销量日期的单品。相关系数是描述性关联，不代表替代、互补或因果关系。", "", "## 品类分布", "", "| 品类 | 日均销量 | 中位数 | 标准差 | 变异系数 |", "|---|---:|---:|---:|---:|"]
    for row in distribution.to_dict(orient="records"):
        lines.append(f"| {row['category']} | {row['mean']:.3f} | {row['median']:.3f} | {row['std']:.3f} | {row['cv']:.3f} |")
    lines.extend(["", "## 品类关联", "", "| 品类 A | 品类 B | Pearson | Spearman |", "|---|---|---:|---:|"])
    for row in strongest_category_pairs:
        lines.append(f"| {row['left']} | {row['right']} | {row['pearson']:.3f} | {row['spearman']:.3f} |")
    lines.extend(["", "## 结论边界", "", "- 日销量共同受到季节、节假日、价格、促销和供给可得性影响，零阶相关不能直接判定替代或互补。", "- 单品销量稀疏，未达到支持度阈值的单品不进入成对相关排名。", "- 完整表格见同目录 CSV；后续定价模型应通过价格响应或受控实验识别需求关系。", ""])
    report.write_text("\n".join(lines), encoding="utf-8")
    artifacts = {name: {"path": str(path.resolve()), "sha256": _sha256(path)} for name, path in paths.items()}
    artifacts["report_md"] = {"path": str(report.resolve()), "sha256": _sha256(report)}
    manifest = {"version": "v44", "kind": "cumcm2023c_sales_distribution_relationships", "cutoff": cutoff, "min_item_days": min_item_days, "top_edges": top_edges, "sources": [{"alias": "products", "path": str(product_path), "sha256": _sha256(product_path)}, {"alias": "sales", "path": str(sales_path), "sha256": _sha256(sales_path)}], "row_count": len(sales), "category_count": len(category_wide.columns), "eligible_item_count": len(eligible), "artifacts": artifacts}
    manifest_path = output_dir / "analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"manifest_path": str(manifest_path.resolve()), "manifest": manifest, "artifacts": artifacts}


def verify_retail_analysis(manifest_path: Path) -> Dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    issues = []
    source_map = {item["alias"]: item for item in manifest.get("sources", [])}
    for alias in ("products", "sales"):
        source = source_map.get(alias, {})
        path = Path(str(source.get("path", "")))
        if not path.is_file() or _sha256(path) != source.get("sha256"):
            issues.append(f"analysis_source_hash_mismatch:{alias}")
    for name, artifact in manifest.get("artifacts", {}).items():
        path = Path(str(artifact.get("path", "")))
        if not path.is_file() or _sha256(path) != artifact.get("sha256"):
            issues.append(f"analysis_artifact_hash_mismatch:{name}")
    checks: Dict[str, Any] = {}
    if not issues:
        products = read_excel_cached(Path(source_map["products"]["path"]), "Sheet1", ["单品编码", "单品名称", "分类名称"])
        sales = read_excel_cached(Path(source_map["sales"]["path"]), "Sheet1", ["销售日期", "单品编码", "销量(千克)", "销售单价(元/千克)"])
        products["单品编码"] = products["单品编码"].astype(str)
        sales["单品编码"] = sales["单品编码"].astype(str)
        sales["销售日期"] = pd.to_datetime(sales["销售日期"])
        sales = sales.loc[(sales["销售日期"] <= pd.Timestamp(manifest["cutoff"])) & (sales["销量(千克)"] >= 0)].merge(products, on="单品编码", how="left", validate="many_to_one")
        support = sales.groupby("单品编码")["销售日期"].nunique()
        checks = {"row_count_recomputed": len(sales) == int(manifest["row_count"]), "category_count_recomputed": sales["分类名称"].nunique() == int(manifest["category_count"]), "eligible_item_count_recomputed": int((support >= int(manifest["min_item_days"])).sum()) == int(manifest["eligible_item_count"]), "artifact_count": len(manifest.get("artifacts", {}))}
        if not all(value for key, value in checks.items() if key != "artifact_count"):
            issues.append("analysis_statistics_recompute_failed")
    return {"kind": "retail_descriptive_analysis", "passed": not issues, "issues": issues, "checks": checks, "claim_level": "domain_verified_descriptive_relationships" if not issues else "no_numerical_claim", "manifest_path": str(manifest_path), "manifest_hash": _sha256(manifest_path)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
