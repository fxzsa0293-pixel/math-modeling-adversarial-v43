# 队友交接检查清单：V44 求解器

## 必看文件

- [ ] `SKILL.md`：整体定位和 V38/V44 两种模式
- [ ] `README_REPRODUCE.md`：运行命令
- [ ] `tools/run.py`：统一入口，推荐使用 `--auto-solver`
- [ ] `tools/run_v39_solver.py`：自动求解入口，文件名保留旧称但实际为 V44
- [ ] `tools/solver/`：题意解析、检索、路线、代码生成、运行、注册、终审模块
- [ ] `tools/workflow/`：工作区协议、数据体检、求解计划、结构化结果导出
- [ ] `tools/tests/`：V44 可靠性测试；依赖本地历史语料的测试在清洁克隆中跳过
- [ ] `data/README.md`：公开资料策略；历史资料由使用者合法取得后放入本地 `data/reference_root/`

## 快速检查

```powershell
cd tools
$env:PYTHONIOENCODING='utf-8'
python run.py --auto-solver --problem-id quickstart_retail --data-root "..\examples\quickstart" --brief "..\examples\quickstart\brief.md" --max-routes 5 --timeout-seconds 30
# 准备好本地历史语料后再运行：python run.py --self-contained
python -m pytest .\tests -q
```

## 成功标准

- [ ] V38 输出 `gate.verdict=pass`
- [ ] V44 生成 `generated_adapters/`
- [ ] V44 生成 `registry/experiment_registry.json`
- [ ] V44 生成 `judge/final_judge.json`
- [ ] V44 生成 `competition_workspace/work/data_audit/data_audit.json`
- [ ] V44 生成 `competition_workspace/work/plan/求解计划.md`
- [ ] V44 生成 `competition_workspace/reports/route_comparison.csv`
- [ ] V44 生成 `competition_workspace/reports/result_summary.json`
- [ ] `python -m pytest tests -q` 全部通过，且无跳过的 V44 可靠性测试

- [ ] 按 references/competition_readiness.md 填写清单并运行 competition_readiness.py
- [ ] 就绪审计输出 verdict=READY，且逐问答案、关键数字、风险、fallback、时间预算和提交物均有证据
- [ ] 每问绑定独立验证、适用性审计和可复现性重跑证据；重跑输出哈希与原始输出一致
- [ ] 官方规则与 AI 使用政策已按当届文件人工确认
- [ ] 不使用图片识别时，论文内容、视觉版式和附件格式由人工完成最终确认

## 后续接入模型 API 的位置

- [ ] 替换或增强 `solver/problem_parser.py`
- [ ] 替换或增强 `solver/corpus_retriever.py`
- [ ] 替换或增强 `solver/route_planner.py`
- [ ] 替换或增强 `solver/adapter_generator.py`
- [ ] 保留 `solver/experiment_runner.py`、`experiment_registry.py`、`final_judge.py` 的证据记录约束
- [ ] 保留 `workflow/` 的交接产物协议

保留不动的底层原则：实验必须进入 registry，推荐必须经过 judge/evidence gate，论文数值必须能追溯到代码输出。
