"""Build and audit a clearly labelled historical rehearsal submission package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compile(tex: Path, work: Path) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    local_tex = work / tex.name
    shutil.copy2(tex, local_tex)
    for _ in range(2):
        process = subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", local_tex.name], cwd=work, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        (work / "xelatex.stdout.txt").write_text(process.stdout, encoding="utf-8")
        (work / "xelatex.stderr.txt").write_text(process.stderr, encoding="utf-8")
        if process.returncode != 0:
            raise RuntimeError(f"xelatex_failed:{process.returncode}")
    pdf = work / local_tex.with_suffix(".pdf").name
    if not pdf.is_file():
        raise FileNotFoundError(pdf)
    return pdf


def build(output_dir: Path) -> dict:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    source_tex = TOOLS / "output" / "pdf" / "CUMCM2023C_submission.tex"
    source = source_tex.read_text(encoding="utf-8")
    source = source.replace("\\title{\\bfseries 蔬菜类商品的自动定价与补货决策}", "\\title{\\bfseries 历史演练：蔬菜类商品的自动定价与补货决策}")
    declaration = (
        "\\section*{AI工具使用声明}\n"
        "本文件是历史技术演练工件，不是正式比赛提交稿。本演练使用 AI 辅助进行代码组织、审计器实现和文字校对；数据口径、模型结构、路线选择、数值结果、独立验证和最终判断由人工检查并以可复算工件为准。\n"
        "\\section*{附录：支撑材料文件清单}\n"
        "本演练支撑包包含：timed\\_rehearsal\\_summary.json；historical\\_fallback\\_audit.json；failure\\_injected\\_switch.json；AI工具使用详情.pdf；run\\_historical\\_timed\\_rehearsal.py。\n"
    )
    source = source.replace("\\begin{thebibliography}{9}", declaration + "\\begin{thebibliography}{9}")
    tex = output_dir / "historical_rehearsal_paper.tex"
    tex.write_text(source, encoding="utf-8")
    detail_tex = output_dir / "ai_usage_detail.tex"
    detail_tex.write_text(
        "\\documentclass[UTF8,a4paper,11pt]{ctexart}\n\\usepackage[margin=2cm]{geometry}\n\\begin{document}\n"
        "\\section*{AI工具使用详情}\n"
        "工件性质：历史技术演练，不是正式比赛提交材料。\n\n"
        "工具：Codex/大语言模型辅助环境。\n\n"
        "用途：审查现有代码结构；提出测试与审计器实现建议；生成局部程序草稿；协助整理报告文字。\n\n"
        "人工工作：确认题目契约、数据字段、模型约束、路线选择、结果解释、独立验证和最终是否通过。AI 输出未直接作为未经核验的数值结论。\n\n"
        "核验方式：读取 JSON/CSV 证据，重新计算哈希，运行独立验证器，检查计时器、报告和 ZIP 完整性；不使用图片识别或 OCR。\n\n"
        "限制：本详情不替代正式比赛要求的队员声明；正式比赛须按当届规定填写真实工具、版本、用途、采用内容和人工修改记录。\n"
        "\\end{document}\n", encoding="utf-8")
    paper_pdf = _compile(tex, output_dir / "paper_build")
    detail_pdf = _compile(detail_tex, output_dir / "detail_build")
    paper_target = output_dir / "历史技术演练论文.pdf"
    detail_target = output_dir / "AI工具使用详情.pdf"
    shutil.copy2(paper_pdf, paper_target)
    shutil.copy2(detail_pdf, detail_target)

    support = output_dir / "历史技术演练支撑材料.zip"
    files = [
        TOOLS / "run" / "historical_timed_rehearsal" / "timed_rehearsal_summary.json",
        TOOLS / "run" / "historical_fallback_audit" / "historical_fallback_audit.json",
        TOOLS / "run" / "historical_fallback_audit" / "failure_injected_switch.json",
        TOOLS / "run_historical_timed_rehearsal.py",
        TOOLS / "fallback_switch.py",
        detail_target,
    ]
    with zipfile.ZipFile(support, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            if path.is_file():
                archive.write(path, path.name)
    result = {
        "kind": "historical_rehearsal_submission_package",
        "paper": {"path": str(paper_target), "sha256": _sha256(paper_target)},
        "support_archive": {"path": str(support), "sha256": _sha256(support)},
        "ai_usage_detail_pdf": {"path": str(detail_target), "sha256": _sha256(detail_target)},
        "claim_level": "historical_rehearsal_package_only",
        "not_formal_contest_submission": True,
    }
    (output_dir / "package_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=TOOLS / "run" / "historical_submission_package")
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
