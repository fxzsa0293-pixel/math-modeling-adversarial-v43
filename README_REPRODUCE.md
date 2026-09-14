# V44 求解器复现说明

## 环境

```powershell
cd tools
pip install -r requirements.txt
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
```

## 复现 V38 benchmark

公开仓库不附带该 benchmark 使用的第三方历史资料。请从合法来源取得对应赛题、附件和必要输入，并放入被 Git 忽略的 `data/reference_root/`，然后运行：

```powershell
python run.py --self-contained
```

预期：`gate.verdict=pass`，`case_count=11`。

## 运行 V44 自动求解工作流

```powershell
python run.py --auto-solver --problem-id demo --data-root "..\data\reference_root" --max-routes 8 --timeout-seconds 30
```

## 运行当前管线真实数据 benchmark

以下真实数据 benchmark 同样需要用户自行合法取得对应附件；公开仓库仅保留执行与审计代码。

```powershell
python run_v44_real_benchmarks.py --max-routes 4
```

当前 suite 使用 CUMCM2023C 四个附件，执行四个真实案例：问题 1 的品类/单品分布与描述性关联、1085 天聚合销量预测、问题 2 的六品类七日补货定价、问题 3 的 27-33 个单品补货定价。零售策略包含 fixed-markup baseline、CVaR 联合优化、源文件与派生模型哈希绑定、逐场景复算和 common-random-number 压力测试。四个案例全部通过后，工作流还会生成问题 4 数据建议、九节完整 Markdown 报告和经 XeLaTeX 编译及版式验收的 PDF。结果写入 `tools/run/v44_real_benchmarks/benchmark_summary.json`。

## 执行多问题 DAG

在 `tools` 目录中运行：

```powershell
python run.py --workflow "D:\problem\workflow.json" --output-dir "D:\problem\run"
```

工作流 schema、跨节点工件注入和失败语义见 `references/executable_workflow.md`。输出包含 `workflow_summary.json`、`workflow_manifest.json`，以及动态的 `competition_workspace/work/question_01`、`question_02` 等目录。

再次执行同一工作流时，只有 workflow、执行参数、运行时代码、题目契约及全部递归工件仍通过验证才会复用结果。传入 `--no-workflow-resume` 可强制重新执行全部节点。

## 运行跨题型真实 benchmark

该命令需要用户本地准备对应的 CUMCM2018A 附件，附件不随公开仓库分发。

```powershell
python run_v44_cross_archetype_benchmarks.py
```

当前跨题型 suite 使用 CUMCM2018A 高温防护服真实 Excel 附件，验证附件解析、集总热模型、时间保留验证、独立重积分、局部可识别性、动态 DAG 和报告绑定。该案例不声称已完成多层 PDE 和服装厚度优化。

## 审计历史演练边界

在已有两个真实 benchmark 均运行后，可生成技术演练审计：

```powershell
python historical_rehearsal_audit.py `
  --benchmark-summary run/v44_real_benchmarks/benchmark_summary.json `
  --cross-summary run/v44_cross_archetype_benchmarks/benchmark_summary.json `
  --fallback-audit run/historical_fallback_audit/historical_fallback_audit.json `
  --switch-rehearsal run/historical_fallback_audit/failure_injected_switch.json `
  --timed-rehearsal run/historical_timed_rehearsal/timed_rehearsal_summary.json `
  --package-audit run/historical_submission_package/submission_package_audit.json `
  --output-dir run/historical_rehearsal_audit
```

输出 `historical_rehearsal_audit.json` 只证明已执行的技术环节和证据哈希；它明确列出尚未具备的八阶段比赛计时、真实 fallback、最终提交论文、支撑压缩包、AI 工具详情和人工合规复核，永远不会把历史 benchmark 判为正式比赛 READY。

兼容旧入口：

```powershell
python run.py --v39-solver --problem-id demo --data-root "..\data\reference_root" --max-routes 8 --timeout-seconds 30
```

默认输出目录：

```text
tools/run/v44_auto_solver/
```

## 重点输出

- `summary.json`：整体结果。
- `generated_adapters/`：每条路线生成的 Python 代码。
- `experiments/*.json`：每条路线执行记录。
- `registry/experiment_registry.json`：实验注册表、代码 hash、失败日志。
- `judge/final_judge.json`：终审推荐与打回原因。
- `competition_workspace/work/data_audit/data_audit.json`：数据体检结构化结果。
- `competition_workspace/work/data_audit/数据体检报告.md`：可读数据体检报告。
- `competition_workspace/work/plan/求解计划.md`：输入、输出、路线与 fallback。
- `competition_workspace/reports/route_comparison.csv`：路线比较表。
- `competition_workspace/reports/result_summary.json`：稳定结果汇总。
- `competition_workspace/reports/figure_requirements.md`：图表需求描述。
- `competition_workspace/reports/route_metrics_long.csv`：全部数值指标长表。
- `competition_workspace/work/question_01/figures/route_comparison.svg`：实际路线比较图。
- `competition_workspace/paper/sections/model_results.md`：受机器门控约束的论文结果章节。
- `competition_workspace/work/question_01/results/recommended_retail_policy.csv`：推荐零售路线的逐日/逐商品价格与补货量。
- `competition_workspace/work/question_01/results/recommended_retail_scenarios.csv`：推荐路线逐场景需求、销量与缺货明细。
- `tools/run/v44_real_benchmarks/CUMCM2023C_complete_report.md`：四个真实案例全部通过后生成的九节完整中间稿。
- `tools/output/pdf/CUMCM2023C_submission.pdf`：经验证的 XeLaTeX 提交工件。
- `tools/output/pdf/CUMCM2023C_submission_manifest.json`：绑定 PDF、TeX、构建日志、benchmark summary、完整报告和工作流 DAG 的哈希。

## 验证要求

运行 `python -m pytest tests -q`。测试数量会随可靠性反例增加，不应将固定测试数当作正确性证明。V38 benchmark 仅用于历史兼容；V44 的契约、证据重算、三段式验证和最终门控必须单独通过。

## 比赛提交前就绪审计

单个求解节点通过后，按 references/competition_readiness.md 填写比赛清单，再运行：

    python competition_readiness.py D:\problem\competition_readiness.json --output-dir D:\problem\readiness_audit

该命令验证逐问完成、关键数字证据、题目契约、实测 fallback、风险处置、比赛时间余量、官方规则确认和最终交付物哈希。它不识别图片；论文版式和附件格式必须由人工确认后记录在清单中。
最终清单还必须绑定可复现性证据：记录原始命令、重跑命令、环境、输入/输出哈希，并由审计器重新检查重跑输出哈希一致性。

## 正式赛题开赛初始化

正式赛题发布后，先对只读题目目录运行：

```powershell
python competition_intake.py D:\contest\input --problem-id CUMCM2026 --output-dir D:\contest\intake
```

该命令只读取 PDF 文字层和可解析的结构化文本；Excel/CSV 等附件只做文件清点和哈希绑定，Excel 内容必须由后续结构化数据审计器读取。检测到的图片、未解析二进制或待解析电子表格会进入 warnings；它不会进行图片识别，也不会自动确认题意。每个问题契约初始为 `needs_input`，必须由队员补齐语义、单位、约束、假设和输出后再进入求解。

契约补齐后可单独运行语义完整性审计：

```powershell
python semantic_review_form.py create D:\contest\run\problem\question_01\problem_contract.json D:\contest\run\question_01\semantic_review.json
# 由队员填写 semantic_review.json 后再执行：
python semantic_review_form.py apply D:\contest\run\problem\question_01\problem_contract.json D:\contest\run\question_01\semantic_review.json D:\contest\run\problem\question_01\problem_contract.json
python contract_completeness.py D:\contest\run\problem\question_01\problem_contract.json --output D:\contest\run\question_01\contract_audit.json
python applicability_audit.py D:\contest\run\problem\question_01\problem_contract.json D:\contest\run\question_01\applicability_audit.json
```

该审计要求 `status=ready`、没有 `unresolved_fields`、存在预期工件、单位字典，以及队员对单位、约束、假设和歧义的人工确认。

工作流完成后，可先生成保守的比赛就绪清单草稿并绑定已有证据：

```powershell
python prepare_readiness_checklist.py `
  D:\contest\run\workflow_summary.json `
  D:\contest\competition_readiness.json `
  --data-audit D:\contest\run\work\data_audit\data_audit.json `
  --applicability-audit D:\contest\run\question_01\applicability_audit.json `
  --reproducibility-audit D:\contest\run\reproducibility_audit.json `
  --claim-registry D:\contest\run\paper_claim_registry.json
```

该生成器会写入相对路径和 SHA-256，但不会自动填写官方规则确认、人工语义确认、风险处置、独立验证、fallback、计时或人工终审。补齐每问的 `critical_claims` 后，可生成论文关键结论登记表：

```powershell
python paper_claim_registry.py D:\contest\competition_readiness.json D:\contest\run\paper_claim_registry.json
python reproducibility_audit.py D:\contest\run\reproducibility_evidence.json D:\contest\run\reproducibility_audit.json
python competition_readiness.py D:\contest\competition_readiness.json --output-dir D:\contest\readiness_audit
```

正式完成只以 `D:\contest\readiness_audit\competition_readiness.json` 中 `verdict == "READY"` 为准。历史 benchmark 和演练输出必须保持 `TECHNICAL_REHEARSAL_PARTIAL`，不得代替当届真题的真实证据。

## 使用建议

V44 的通用 adapter 覆盖单表及声明式多表装配后的预测、回归和评价排名；原生模块还覆盖声明式优化、双目标采样 Pareto、场景鲁棒优化、受限 ODE、排队仿真和 IDW。超出这些 schema 的题目仍必须提供 specialist adapter 与领域验算器。若它给出 `pass`，仍要检查契约是否完整表达赛题语义。

复杂题在审查 `adapter_path` 和 `domain_verifier` 后，使用 `--allow-specialist-code` 显式执行。专用脚本必须位于绑定数据目录或契约的 `allowed_code_roots` 中；执行器不会尝试修改它们。
