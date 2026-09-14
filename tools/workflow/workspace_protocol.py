from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

WORKSPACE_DIRS = [
    "problem",
    "data",
    "work/data_audit",
    "work/plan",
    "work/question_01/code",
    "work/question_01/results",
    "work/question_01/figures",
    "work/question_01/logs",
    "paper/sections",
    "paper/build",
    "reports",
]


def create_workspace(output_dir: Path) -> Dict[str, str]:
    workspace = output_dir / "competition_workspace"
    paths: Dict[str, str] = {"workspace": str(workspace)}
    for relative in WORKSPACE_DIRS:
        path = workspace / relative
        path.mkdir(parents=True, exist_ok=True)
        paths[relative] = str(path)
    readme = workspace / "README.md"
    if not readme.exists():
        readme.write_text(_readme_text(), encoding="utf-8")
    return paths


def create_multi_question_workspace(output_dir: Path, question_ids: Iterable[str]) -> Dict[str, Dict[str, str]]:
    """Create one shared competition workspace with isolated per-question paths."""
    workspace = Path(output_dir).resolve() / "competition_workspace"
    shared = {
        "workspace": str(workspace),
        "paper/sections": str(workspace / "paper" / "sections"),
        "paper/build": str(workspace / "paper" / "build"),
    }
    for path in (workspace / "paper" / "sections", workspace / "paper" / "build"):
        path.mkdir(parents=True, exist_ok=True)
    mappings: Dict[str, Dict[str, str]] = {}
    for question_id in question_ids:
        question = workspace / "work" / question_id
        problem = workspace / "problem" / question_id
        reports = workspace / "reports" / question_id
        mapping = {
            **shared,
            "problem": str(problem),
            "data": str(workspace / "data"),
            "work/data_audit": str(question / "data_audit"),
            "work/plan": str(question / "plan"),
            "work/question_01/code": str(question / "code"),
            "work/question_01/results": str(question / "results"),
            "work/question_01/figures": str(question / "figures"),
            "work/question_01/logs": str(question / "logs"),
            "paper/sections": str(workspace / "paper" / "sections" / question_id),
            "reports": str(reports),
        }
        for raw_path in set(mapping.values()):
            Path(raw_path).mkdir(parents=True, exist_ok=True)
        mappings[question_id] = mapping
    readme = workspace / "README.md"
    if not readme.exists():
        readme.write_text(_readme_text(), encoding="utf-8")
    return mappings


def _readme_text() -> str:
    return """# V44 Competition Workspace Protocol

This workspace is designed for auditable math-modeling competition work.
It borrows the stable pipeline skeleton from Mrite-style workflows without adopting rigid fully-autonomous rules.

## Directory Contract

- `problem/`: problem statement and manual notes.
- `data/`: optional copied or linked competition attachments.
- `work/data_audit/`: full data inspection reports.
- `work/plan/`: solving plan anchored to data, routes, outputs, and fallbacks.
- `work/question_01/code/`: generated or curated executable code for the current question/batch.
- `work/question_01/results/`: stable numerical outputs for paper writing.
- `work/question_01/figures/`: figure descriptions or generated figures.
- `work/question_01/logs/`: run logs and repair notes.
- `paper/sections/`: optional modular LaTeX or Markdown sections.
- `paper/build/`: optional PDF/LaTeX build artifacts.
- `reports/`: route comparisons, final judge, and result summaries.

## Non-Negotiable Evidence Rules

1. Numerical conclusions must trace to code output, explicit derivation, or a named data source.
2. Candidate routes must be compared against explicit baselines when the task supports it.
3. Strategic ambiguity may be escalated to the user; technical micro-decisions should be executed and logged.
4. Figure plans describe what should be shown and why; figure count is not forced.
5. `problem/problem_contract.json` is the binding source for files, sheets, target, time order, indicator directions, and validation mode.
6. A language-model judgment cannot override a failed machine gate or missing recomputable evidence.
"""
