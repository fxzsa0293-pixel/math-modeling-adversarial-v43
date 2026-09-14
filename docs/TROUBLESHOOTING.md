# 常见问题与排错

## Skill 没有触发

- 确认目录为 `$HOME/.codex/skills/math-modeling-adversarial-v43`，并且该目录内直接存在 `SKILL.md`。
- 重新启动 Codex 或新建任务，使 Skill 索引重新加载。
- 用 `$math-modeling-adversarial-v43` 显式调用以排除自动路由问题。
- 运行官方结构校验：`python <skill-creator>/scripts/quick_validate.py <skill-folder>`。

## NumPy 或 Pandas 导入失败

若出现 `numpy.dtype size changed`、DLL load failed 或 ABI incompatibility，通常是 NumPy 与 Pandas 来自不同环境或二进制版本不匹配。删除本仓库的 `.venv` 后重新创建隔离环境，并只在该环境安装 `tools/requirements.txt`。不要在损坏的系统 Python 或 Conda base 环境上继续叠加安装。

## 契约停在 needs_input

这是保护机制，不一定是程序错误。常见原因包括：

- 多个数据文件或多个 sheet 尚未指定；
- 预测目标、时间列或 forecast horizon 不明确；
- 评价指标的正向、负向未确认；
- 多表 join key、基数或聚合口径缺失；
- 硬约束、单位或题目歧义会改变结论。

根据 `unresolved_fields` 补全契约，完成人工语义复核，然后运行 `tools/contract_completeness.py`。不要仅把 `status` 手动改成 `ready`。

## final_judge 为 blocked 或 revise

打开 `judge/final_judge.json`，依次检查 `gate_layers`、`flags` 和 `next_actions`。常见问题是契约未确认、证据缺失、领域验证失败、候选路线没有稳定优于 baseline，或 specialist code 尚未授权。`revise` 表示流程跑完但证据不足以接受结论；`blocked` 表示关键前提尚未满足。

## specialist route 没有执行

契约声明的第三方或题目专用脚本默认不执行。先人工审查 `adapter_path`、`domain_verifier` 与 `allowed_code_roots`，确认可信后才添加 `--allow-specialist-code`。该参数是执行授权，不是对脚本正确性的背书。

## 找不到数据、列或 sheet

- 正式契约中的文件路径建议使用绝对路径。
- 检查 `data_audit.json` 中的实际文件、sheet、列名与读取错误。
- Windows 路径在 JSON 中需要写成 `D:\\problem\\data.xlsx`。
- 多表数据按 `references/problem_contract.md` 声明 `data_sources`、`data_joins` 和可选 `data_aggregation`。

## 历史 benchmark 很慢

`run_v44_real_benchmarks.py`、跨题型 benchmark 和完整测试集都不是三分钟 Quickstart 的必要步骤。首次安装先跑 `examples/quickstart`；修改求解器或准备发布时，再运行完整测试与历史回归。

## GitHub 为什么没有附带历史论文和赛题

公开仓库不分发权利状态未核实的第三方资料。请从官方或其他合法来源自行取得，并放入本地 `data/reference_root/` 或其他数据目录；该目录被 Git 忽略。Codex 首先读取 `SKILL.md`，再按任务需要读取说明和本地资料，不会一次性加载整个语料库。
