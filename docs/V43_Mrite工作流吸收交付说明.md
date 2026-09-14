# V43 Mrite 工作流吸收交付说明

## 这版解决什么

V43 不是继续盲目堆模型，而是把 V42 的真实 Python 模型搜索装进一套竞赛工作区协议中。它吸收 Mrite-main 的优点：先体检数据、先写计划、逐路线执行、结构化保存结果、图表需求前置、结果可追溯。

## 新增模块

- `tools/workflow/workspace_protocol.py`：创建 `competition_workspace/` 标准目录。
- `tools/workflow/data_auditor.py`：输出 `data_audit.json` 与 `数据体检报告.md`。
- `tools/workflow/plan_writer.py`：输出 `求解计划.md`。
- `tools/workflow/result_exporter.py`：输出 `route_comparison.csv`、`result_summary.json/md`、`figure_requirements.md`。

## 新增输出

```text
competition_workspace/
  README.md
  work/data_audit/data_audit.json
  work/data_audit/数据体检报告.md
  work/plan/求解计划.md
  work/question_01/code/
  work/question_01/results/result_summary.json
  work/question_01/logs/
  reports/route_comparison.csv
  reports/result_summary.json
  reports/result_summary.md
  reports/figure_requirements.md
```

## 与 Mrite-main 的取舍

吸收：

- 数据体检先行。
- 求解计划作为锚点。
- 每问/每批任务有独立代码、结果、日志目录。
- 数值结果结构化保存。
- 图表需求提前规划。

不照搬：

- 不强制无人值守；战略性分歧仍可问用户。
- 不强制固定图数量或公式数量。
- 不限定只能使用某一种绘图库。
- 不强制全中文文件名，避免跨平台路径与编译问题。

## 验证

```text
python -m pytest .\tools\tests -q
13 passed

python .\tools\run.py --self-contained --output-dir .\tools\run\v43_v38_self_contained
case_count=11
gate.verdict=pass
```

## 真实评价

V43 已经是一个较可靠的“自动建模探索底座”：它能跑真实数据、比较路线、拒绝无目标硬回归、记录代码 hash、输出可交接工作区。它还不是全题型完整求解器；后续提升国赛上限的关键仍是给不同题型补强专用 adapter、让模型 API 根据历史资料生成更强路线，并把这些路线纳入同一套 registry/judge/workflow。
