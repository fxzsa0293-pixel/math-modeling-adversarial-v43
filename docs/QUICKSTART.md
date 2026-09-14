# 三分钟 Quickstart

本示例用于验证安装和最短求解链路，不代表正式赛题已经完成语义确认。它会读取仓库内的 60 行零售预测样例，自动推断初始契约，运行 baseline 与候选路线，并生成可追踪产物。

## 1. 创建隔离环境

在仓库根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r tools\requirements.txt
```

开发或运行测试时，将最后一条改为 `tools\requirements-dev.txt`。不要把仓库环境与一个已有但依赖冲突的 Conda 环境混用。

## 2. 运行最小示例

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python tools\run.py `
  --auto-solver `
  --problem-id quickstart_retail `
  --data-root examples\quickstart `
  --brief examples\quickstart\brief.md `
  --output-dir run\quickstart `
  --max-routes 6
```

macOS/Linux 对应命令：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r tools/requirements.txt
PYTHONIOENCODING=utf-8 .venv/bin/python tools/run.py \
  --auto-solver \
  --problem-id quickstart_retail \
  --data-root examples/quickstart \
  --brief examples/quickstart/brief.md \
  --output-dir run/quickstart \
  --max-routes 6
```

## 3. 检查结果

运行结束后，优先查看：

| 文件 | 用途 |
| --- | --- |
| `run/quickstart/summary.json` | 版本、契约状态、路径和总摘要 |
| `run/quickstart/judge/final_judge.json` | 分层门控、推荐路线、失败原因和下一步 |
| `run/quickstart/registry/experiment_registry.json` | 每条路线的配置、状态、指标和证据哈希 |
| `run/quickstart/competition_workspace/reports/route_comparison.csv` | 路线横向比较 |
| `run/quickstart/competition_workspace/reports/result_summary.json` | 面向后续论文和工作流的结果摘要 |
| `run/quickstart/competition_workspace/work/data_audit/数据体检报告.md` | 数据结构与质量检查 |

成功执行与正式可提交是两回事。`final_judge.json` 可能给出 `pass`、`revise` 或 `blocked`；只有解决所有门控问题、完成人工语义复核，并通过比赛就绪审计后，才能进入提交阶段。

## 4. 换成自己的题目

先让工具基于题面与附件生成初始 `problem_contract.json`，再用人工复核表确认会改变题意的字段：

```powershell
.\.venv\Scripts\python tools\semantic_review_form.py create `
  run\my_problem\competition_workspace\problem\problem_contract.json `
  run\my_problem\semantic_review.json

# 填写 semantic_review.json 后应用复核结果
.\.venv\Scripts\python tools\semantic_review_form.py apply `
  run\my_problem\competition_workspace\problem\problem_contract.json `
  run\my_problem\semantic_review.json `
  run\my_problem\problem_contract.confirmed.json

.\.venv\Scripts\python tools\contract_completeness.py `
  run\my_problem\problem_contract.confirmed.json `
  --output run\my_problem\contract_audit.json
```

确认版契约应通过 `--contract` 传回求解器。完整字段、任务类型和停止条件见 [`references/problem_contract.md`](../references/problem_contract.md)。
