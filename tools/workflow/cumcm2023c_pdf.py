from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .cumcm2023c_report import verify_cumcm2023c_report
from .problem_dag import verify_workflow_manifest


def build_cumcm2023c_pdf(benchmark_summary: Path, output_dir: Path, xelatex: str = "xelatex") -> Dict[str, Any]:
    benchmark_summary = Path(benchmark_summary).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    suite = json.loads(benchmark_summary.read_text(encoding="utf-8"))
    report_verification = verify_cumcm2023c_report(benchmark_summary)
    dag_path = Path(str(suite.get("workflow_dag", {}).get("manifest_path", "")))
    dag_verification = (
        verify_workflow_manifest(dag_path)
        if dag_path.is_file()
        else {"passed": False, "issues": ["workflow_dag_missing"]}
    )
    if (
        not suite.get("all_passed")
        or not report_verification["passed"]
        or not dag_verification["passed"]
    ):
        raise ValueError("pdf_requires_verified_complete_report")
    cases = {case["case_id"]: case for case in suite["cases"]}
    category = _read_json(cases["CUMCM2023C_category_replenishment_pricing"]["summary_path"])
    item = _read_json(cases["CUMCM2023C_item_replenishment_pricing"]["summary_path"])
    forecast_manifest = _read_json(cases["CUMCM2023C_category_replenishment_pricing"]["upstream_category_forecast"]["manifest_path"])
    forecast = pd.read_csv(forecast_manifest["artifacts"]["diagnostics"]["path"])
    forecast = forecast.loc[forecast["selected_by_selection"].astype(bool)]
    category_policy = pd.read_csv(category["workflow_exports"]["recommended_retail_policy_csv"])
    item_policy = pd.read_csv(item["workflow_exports"]["recommended_retail_policy_csv"])
    item_policy = item_policy.loc[item_policy["selected"].astype(bool)]
    service = pd.read_csv(item["workflow_exports"]["recommended_retail_category_service_csv"])
    service["ratio"] = service["actual_sales"] / service["minimum_sales"]
    category_profit = _recommended_profit(category)
    item_profit = _recommended_profit(item)
    tex_path = output_dir / "CUMCM2023C_submission.tex"
    pdf_path = output_dir / "CUMCM2023C_submission.pdf"
    tex_path.write_text(_render_tex(forecast, category_policy, item_policy, service, category, item, category_profit, item_profit), encoding="utf-8")
    executable = shutil.which(xelatex) or xelatex
    logs = []
    for _ in range(3):
        process = subprocess.run([executable, "-interaction=nonstopmode", "-halt-on-error", "-output-directory", str(output_dir), str(tex_path)], text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=180)
        logs.append(process.stdout + process.stderr)
        if process.returncode != 0:
            raise RuntimeError(f"xelatex_failed:{process.returncode}:{logs[-1][-2000:]}")
    log_path = output_dir / "xelatex.log.txt"
    log_path.write_text("\n\n--- PASS ---\n\n".join(logs), encoding="utf-8")
    manifest = {
        "version": "v44",
        "kind": "cumcm2023c_submission_pdf",
        "sources": [
            {"path": str(benchmark_summary), "sha256": _sha256(benchmark_summary)},
            {"path": suite["complete_report"]["path"], "sha256": suite["complete_report"]["sha256"]},
            {"path": str(dag_path), "sha256": _sha256(dag_path)},
        ],
        "tex": _binding(tex_path), "pdf": _binding(pdf_path), "log": _binding(log_path),
        "build": {"engine": "xelatex", "passes": 3},
    }
    manifest_path = output_dir / "CUMCM2023C_submission_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"pdf_path": str(pdf_path), "tex_path": str(tex_path), "manifest_path": str(manifest_path), "pdf_sha256": manifest["pdf"]["sha256"]}


def verify_cumcm2023c_pdf(manifest_path: Path) -> Dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    issues = []
    try:
        manifest = _read_json(manifest_path)
        for name in ("tex", "pdf", "log"):
            path = Path(manifest[name]["path"])
            if not path.is_file() or _sha256(path) != manifest[name]["sha256"]:
                issues.append(f"submission_{name}_hash_mismatch")
        for index, source in enumerate(manifest["sources"]):
            path = Path(source["path"])
            if not path.is_file() or _sha256(path) != source["sha256"]:
                issues.append(f"submission_source_hash_mismatch:{index}")
        if not issues and not Path(manifest["pdf"]["path"]).read_bytes().startswith(b"%PDF"):
            issues.append("submission_pdf_signature_invalid")
        if not issues and manifest.get("kind") == "cumcm2023c_submission_pdf":
            benchmark_path = Path(manifest["sources"][0]["path"])
            report_verification = verify_cumcm2023c_report(benchmark_path)
            issues.extend(f"submission_report:{issue}" for issue in report_verification.get("issues", []) if not report_verification.get("passed"))
            dag_path = Path(manifest["sources"][2]["path"])
            dag_verification = verify_workflow_manifest(dag_path)
            issues.extend(f"submission_dag:{issue}" for issue in dag_verification.get("issues", []) if not dag_verification.get("passed"))
            passes = manifest.get("build", {}).get("passes")
            log_text = Path(manifest["log"]["path"]).read_text(encoding="utf-8", errors="replace")
            if not isinstance(passes, int) or passes < 2 or log_text.count("--- PASS ---") + 1 != passes:
                issues.append("submission_build_pass_count_invalid")
            warnings = ("Overfull", "Underfull", "Missing character", "longtable Warning", "Rerun")
            if any(token in log_text for token in warnings):
                issues.append("submission_latex_log_warning")
            pdf_bytes = Path(manifest["pdf"]["path"]).read_bytes()
            if len(pdf_bytes) < 1024 or not pdf_bytes.rstrip().endswith(b"%%EOF"):
                issues.append("submission_pdf_structure_invalid")
    except Exception as exc:
        issues.append(f"submission_pdf_verification_error:{exc}")
    return {"kind": "cumcm2023c_submission_pdf", "passed": not issues, "issues": issues, "manifest_path": str(manifest_path)}


def _render_tex(forecast: pd.DataFrame, category_policy: pd.DataFrame, item_policy: pd.DataFrame, service: pd.DataFrame, category: Dict[str, Any], item: Dict[str, Any], category_profit: float, item_profit: float) -> str:
    forecast_rows = "\n".join(f"{_escape(r.category)} & {_escape(r.route)} & {r.selection_mae:.2f} & {r.final_test_mae:.2f} & {'是' if r.final_test_reversal else '否'} \\\\" for r in forecast.itertuples())
    policy_rows = "\n".join(f"{_escape(str(r.decision_date)[5:])} & {_escape(r.category)} & {r.price:.2f} & {r.replenishment:.2f} \\\\" for r in category_policy.itertuples())
    item_rows = "\n".join(f"{_escape(str(r.category)[:20])} & {r.price:.2f} & {r.replenishment:.2f} \\\\" for r in item_policy.itertuples())
    selection_coords = " ".join(f"({i + 1},{v:.4f})" for i, v in enumerate(forecast["selection_mae"]))
    final_coords = " ".join(f"({i + 1},{v:.4f})" for i, v in enumerate(forecast["final_test_mae"]))
    category_cmp = category["domain_verification"]["policy_comparison"]
    item_cmp = item["domain_verification"]["policy_comparison"]
    reversal_count = int(forecast["final_test_reversal"].astype(bool).sum())
    service_min = float(service["ratio"].min())
    return rf"""\documentclass[UTF8,a4paper,10pt]{{ctexart}}
\usepackage[margin=1.65cm]{{geometry}}
\usepackage{{amsmath,amssymb,booktabs,longtable,array,xcolor,pgfplots,hyperref,fancyhdr}}
\pgfplotsset{{compat=1.18}}
\definecolor{{blue}}{{HTML}}{{2878B5}}\definecolor{{orange}}{{HTML}}{{F28E2B}}
\hypersetup{{colorlinks=true,linkcolor=blue,urlcolor=blue}}
\pagestyle{{fancy}}\fancyhf{{}}\lhead{{CUMCM 2023 C}}\rhead{{可审计建模报告}}\cfoot{{\thepage}}
\setlength{{\parindent}}{{2em}}\setlength{{\parskip}}{{0.25em}}
\title{{\bfseries 蔬菜类商品的自动定价与补货决策}}\author{{证据门控建模工作流}}\date{{}}
\begin{{document}}\maketitle
\begin{{abstract}}
基于四个真实附件，建立销量关系分析、滚动验证预测、离散定价与补货混合整数规划。问题2由六品类滚动选出的预测路线产生未来7日需求场景；问题3在27--33个单品、最小陈列量2.5 kg及18条品类--场景服务约束下联合优化。全部数值工件均绑定哈希，并由独立验证器重算。
\end{{abstract}}
\textbf{{关键词：}}蔬菜零售；滚动预测；混合整数规划；CVaR；压力测试
\section{{问题与数据}}
训练及参数估计截止于2023年6月30日。问题2给出7月1--7日六品类策略；问题3给出7月1日单品策略。相关分析仅为描述性联系，不解释为替代、互补或因果。
\section{{模型}}
设 $x_i$ 为是否选择商品，$y_{{ik}}$ 为价格档，$q_i$ 为补货量，$s_{{ikw}}$ 为场景销量。
\begin{{align}}
&\sum_k y_{{ik}}=x_i,\quad q_i^{{\min}}x_i\le q_i\le q_i^{{\max}}x_i,\\
&0\le s_{{ikw}}\le d_{{ikw}}y_{{ik}},\quad \sum_k s_{{ikw}}\le(1-l_i)q_i,\\
&\Pi_w=\sum_{{i,k}}p_{{ik}}s_{{ikw}}-\sum_i c_iq_i-h\sum_{{i,k}}(d_{{ikw}}y_{{ik}}-s_{{ikw}}).
\end{{align}}
风险目标为 $\min \eta+(1-\alpha)^{{-1}}\sum_w\pi_wz_w$，其中 $z_w\ge-\Pi_w-\eta$。问题3另有 $\sum_{{i\in g,k}}s_{{ikw}}\ge0.8D_{{gw}}$。
\section{{品类预测}}
每个品类用8个不重叠7日窗口按选择MAE选路，再用4个窗口作最终检验；最终检验不得反向调参。六品类中有 {reversal_count} 个出现最终检验路线反转，作为模型不确定性披露。
\begin{{table}}[ht]\centering\small\begin{{tabular}}{{llrrc}}\toprule 品类&路线&选择MAE&最终MAE&反转\\\midrule
{forecast_rows}
\bottomrule\end{{tabular}}\caption{{六品类滚动验证选路结果}}\end{{table}}
\begin{{figure}}[ht]\centering
\begin{{tikzpicture}}\begin{{axis}}[ybar,bar width=7pt,width=.88\linewidth,height=5.2cm,ylabel={{MAE}},xtick={{1,2,3,4,5,6}},legend style={{at={{(0.5,-0.2)}},anchor=north,legend columns=2}}]
\addplot[fill=blue] coordinates {{{selection_coords}}};\addplot[fill=orange] coordinates {{{final_coords}}};\legend{{选择区间,最终检验}}\end{{axis}}\end{{tikzpicture}}
\caption{{各品类所选路线的选择与最终检验误差，横轴顺序同表1}}\end{{figure}}
\section{{问题2结果}}
联合策略声明场景期望利润为 {category_profit:.2f}。相关需求冲击下，候选相对固定加成基线的配对利润差95\%区间为 $[{category_cmp['ci95_lower']:.2f},{category_cmp['ci95_upper']:.2f}]$。
\begin{{longtable}}{{llrr}}\toprule 日期&品类&售价&补货量/kg\\\midrule\endhead
{policy_rows}
\bottomrule\end{{longtable}}
\section{{问题3结果}}
模型选中 {len(item_policy)} 个单品，声明场景期望利润为 {item_profit:.2f}；18条品类--场景服务约束全部达标，实际销量/最低销量的最小比为 {service_min:.4f}。候选相对基线压力利润差95\%区间为 $[{item_cmp['ci95_lower']:.2f},{item_cmp['ci95_upper']:.2f}]$。
\begin{{longtable}}{{lrr}}\toprule 单品&售价&补货量/kg\\\midrule\endhead
{item_rows}
\bottomrule\end{{longtable}}
\section{{稳健性与边界}}
压力测试使用1000次共同随机数，同时扰动需求和价格弹性。跨品类相关矩阵由最近365天、去除星期效应后的对数销量残差估计，并投影至半正定矩阵。该模拟只证明策略在声明分布下稳健，不能替代真实决策期回测。观察性价格弹性也不构成因果效应。
\section{{可复现性}}
源附件、品类预测、派生输入、模型、路线证据、服务约束、报告和工作流DAG均使用SHA-256绑定；机器门控失败时不输出数值推荐。
\begin{{thebibliography}}{{9}}
\bibitem{{rockafellar}} Rockafellar R T, Uryasev S. Optimization of conditional value-at-risk. Journal of Risk, 2000.
\bibitem{{hyndman}} Hyndman R J, Athanasopoulos G. Forecasting: Principles and Practice. OTexts, 2021.
\end{{thebibliography}}
\end{{document}}
"""


def _recommended_profit(summary: Dict[str, Any]) -> float:
    route = summary["final_judge"]["recommended_route"]["route_id"]
    registry = _read_json(summary["registry_path"])
    return float(next(record for record in registry if record["route_id"] == route)["metrics"]["expected_profit"])


def _binding(path: Path) -> Dict[str, str]: return {"path": str(path), "sha256": _sha256(path)}
def _read_json(path: Any) -> Any: return json.loads(Path(path).read_text(encoding="utf-8"))
def _escape(value: Any) -> str: return str(value).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
def _sha256(path: Path) -> str: return hashlib.sha256(Path(path).read_bytes()).hexdigest()
