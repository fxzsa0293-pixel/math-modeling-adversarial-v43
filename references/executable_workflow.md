# V44 Executable Multi-Question Workflow

工作流文件把一道多问赛题声明为有向无环图。每个求解节点使用独立题目契约并运行完整的 V44 solver；paper 节点只汇编已经通过机器门控的上游章节。

## 最小示例

```json
{
  "version": "v44",
  "problem_id": "contest_problem",
  "nodes": [
    {
      "node_id": "forecast",
      "kind": "prediction",
      "contract_path": "contracts/question_01.json",
      "outputs": {
        "forecast_metrics": "workflow_exports.route_metrics_long_csv",
        "paper_section": "workflow_exports.paper_results_section_md"
      }
    },
    {
      "node_id": "evaluation",
      "kind": "analysis",
      "dependencies": ["forecast"],
      "contract_path": "contracts/question_02.json",
      "inputs": {
        "forecast_metrics": {
          "from_node": "forecast",
          "output": "forecast_metrics",
          "contract_field": "data_file"
        }
      },
      "outputs": {"paper_section": "workflow_exports.paper_results_section_md"}
    },
    {
      "node_id": "paper",
      "kind": "paper",
      "dependencies": ["forecast", "evaluation"],
      "inputs": {
        "question_01": {"from_node": "forecast", "output": "paper_section"},
        "question_02": {"from_node": "evaluation", "output": "paper_section"}
      },
      "outputs": {"report": "report_path", "report_manifest": "manifest_path"}
    }
  ]
}
```

运行：

```powershell
python run.py --workflow "D:\problem\workflow.json" --output-dir "D:\problem\run"
```

## 节点协议

- `node_id`：字母开头的稳定 ID，只允许字母、数字、下划线和连字符。
- `kind`：`analysis`、`prediction`、`optimization`、`mechanism`、`simulation`、`sensitivity` 或 `paper`。
- `dependencies`：直接上游节点 ID。存在循环、未知依赖或重复依赖时拒绝执行。
- `contract_path`：非 paper 节点必须提供；相对路径以 workflow 文件目录为基准。
- `data_root`、`brief`、`corpus_root`：可选节点级路径，均以 workflow 文件目录为基准。
- `outputs`：别名到节点结果的点号 selector。`$default` 表示节点默认主工件。
- `inputs`：别名到上游输出的绑定。`from_node` 必须是直接依赖，`output` 必须是上游声明的输出别名。
- `contract_field`：可选，只允许将工件路径注入 `data_file`、`data_assembly_manifest`、`retail_decision_manifest` 或 `domain_verifier`。其它契约语义不得由上游隐式覆盖。

## 执行与失败语义

1. 节点按稳定拓扑顺序执行，并映射到动态 `work/question_01`、`work/question_02` 等目录。
2. 非 paper 节点必须获得 `final_judge.verdict=pass` 才算完成。
3. 任一上游未通过时，下游节点记录 `upstream_execution_not_passed`，不得执行或产生推荐。
4. paper 节点只接受已通过节点的文本工件，并生成绑定所有来源哈希的 report manifest。
5. 最终 `workflow_manifest.json` 绑定 workflow 文件、有效题目契约以及所有节点输入输出；任何绑定工件变化都会使验证失败。
6. 再次运行时，只有旧 DAG、执行配置、运行时代码和递归证据全部仍有效才复用完整工作流。使用 `--no-workflow-resume` 可强制全部节点重新执行。

该执行器不会运行 workflow 中提供的任意 shell 命令。复杂专用代码仍必须由题目契约声明，并经 `--allow-specialist-code` 显式授权。
