---
name: math-modeling-adversarial-v43
description: Run evidence-gated mathematical-modeling contest workflows with an explicit problem contract, executable baselines and candidate routes, independent critics, and machine-verifiable results. Use for auditable multi-route modeling; not as a substitute for missing domain constraints or an unimplemented specialist solver.
metadata:
  short-description: Evidence-gated modeling workflow
---

# 数学建模对抗 Agent V44

## 定位

V44 将建模探索组织成“题目契约、真实执行、可重算证据、分层门控、独立审查”的工作流。数值结论必须能追溯到绑定的数据、代码和逐样本或逐约束结果；语言模型裁判不能覆盖机器门控失败。

## 必须先建立题目契约

运行前检查 `problem_contract.json`。自动推断只适合字段含义清楚的单表任务；遇到多文件、多 sheet、目标含糊、指标方向或约束不完整时，先按 [题目契约](references/problem_contract.md) 补全契约。契约状态不是 `ready` 时，只允许数据审计和路线规划，不得引用数值结果。

```powershell
cd tools
$env:PYTHONIOENCODING='utf-8'
python run.py --auto-solver --problem-id demo --data-root "D:\problem_data" --brief "D:\problem_data\brief.md" --contract "D:\problem_data\problem_contract.json"
```

输出中的 `competition_workspace/problem/problem_contract.json` 是本次运行实际使用的契约。检查 `judge/final_judge.json` 的五层门控：`contract`、`execution`、`evidence`、`validation`、`domain_constraints`。预测和回归任务使用独立的 selection 与 final test；路线只按 selection 选择，final test 用于确认推荐没有反转。`claim_level=screening_evidence_only` 表示通用 adapter 的结果仍需结合题意做领域解释。

正式比赛中不得仅凭自动生成的契约开算。使用 `tools/semantic_review_form.py create` 生成人工语义复核表，由队员确认预期输出、单位、约束、假设和歧义后，再用 `apply` 写回契约。随后运行 `tools/contract_completeness.py`；只有审计通过的 `ready` 契约才能进入数值求解。自动流程不得代填人工确认、复核人或时间。

## 自动求解边界

- 预测路线采用 train/selection/final-test 时间顺序验证，baseline 与候选共享信息集；两个保留区间的真实目标不能进入相应预测特征。可用 `seasonal_period` 声明周期，启用递归季节朴素基线；周期必须来自题意、时间频率或训练期诊断，不能根据 final test 调参。
- 回归预处理只在训练集拟合；存在 `group_columns` 时执行实体互斥的 group holdout。
- 评价路线必须声明每个指标的效益型或成本型方向；与等权法的一致性只作为诊断，主要输出扰动稳定性。
- 声明式线性/MILP、双目标线性/MILP（单目标锚点与 epsilon-constraint 采样 Pareto 解集）、网络流（最短路、最大流、最小费用流、运输、分配）、受限内置 ODE（松弛、Logistic、SIR）和空间 IDW 可走原生求解路径；三目标及以上优化和其它复杂约束题没有专用 adapter 时返回 `unsupported`，不得套用通用回归模板。
- 多附件或多 sheet 任务可在契约中声明 `data_sources`、`base_table` 与顺序 `data_joins`。装配阶段执行键完整性、join 基数和未匹配率检查，生成哈希绑定的 CSV 与 manifest，再交给下游路线。
- 场景鲁棒线性/MILP 可声明 `robust_optimization_model`，原生支持期望损失、最坏情形和离散场景 CVaR；所有场景约束与成本均由独立验证器复算。原生风险目标按最小化损失定义，最大化收益须先转换为负损失。
- 蔬菜零售等离散定价与补货任务可声明 `retail_decision_model`，联合决定选品、价格档和补货量，支持损耗、预算、总选品数、分组覆盖、品类-场景最低服务量、需求场景、缺货惩罚以及 expected/worst-case/CVaR。固定加成 baseline 与候选共用场景，独立验证器重算业务量、服务约束、MILP 最优值和 common-random-number 压力测试；压力测试同时扰动需求量与价格弹性，候选利润差 95% 区间下界不大于 0 时推荐回退 baseline。由真实附件派生模型时用 `retail_decision_manifest` 绑定源文件、训练截止日、决策日期、上游预测、派生输入和模型哈希。
- 原生 ODE 路线执行有界多起点参数拟合、train/selection/final-test 时间切分、独立重积分、守恒检查、敏感性秩/条件数和参数边界诊断；收敛不等于可识别，结论会保留为相应 claim level。
- CUMCM2018A 真实跨原型 suite 会重新解析并哈希绑定原始 Excel 中的 5401 点温度序列和四层材料参数，再通过可执行 DAG 运行集总一阶热模型。该 suite 只证明带独立重积分和可识别性诊断的集总热模型链路，不代表已实现赛题所需的完整多层热传导 PDE 与厚度优化。
- 原生 simulation 路线支持声明式 M/M/1、M/M/c 和有限容量 M/M/c/K；到达与服务时间可为 exponential、deterministic 或 empirical。纯指数模型提供 Erlang-C 或 birth-death 闭式基线，非指数模型使用同均值确定性基线；全部路线执行固定 seeds、warm-up、95% 置信区间和独立复模拟。时变过程与策略配对比较仍需专用 adapter。
- 原生 spatial 路线支持声明式 IDW 和全局均值基线，使用固定 seed 的空间 block train/selection/final-test 划分，并由独立验证器重算分块、距离、预测、final-test 指标和残差 Moran's I。`coordinate_system=projected` 使用声明单位下欧氏距离，`geographic_wgs84` 使用 Haversine 公里距离。
- 优化、机理和空间题可在契约的 `specialist_routes` 与 `domain_verifier` 中接入专用代码；检查脚本后显式传入 `--allow-specialist-code` 才会执行。
- baseline 若没有被候选可靠超过，可以成为正式推荐。
- 每次运行还导出全指标长表、SVG 路线比较图和门控约束下的 `paper/sections/model_results.md`；门控失败时章节不得形成数值性推荐。
- 多问题任务按 [可执行多问题工作流](references/executable_workflow.md) 声明 analysis/prediction/optimization/simulation/sensitivity/paper 节点 DAG，并用 `python run.py --workflow workflow.json` 执行。每个求解节点拥有动态问题工作区并运行完整机器门控；节点只有在自身验证通过、所有依赖完成且输入/输出与契约哈希仍匹配时才能完成。循环依赖、未声明的 artifact 边、失败上游和重复输出生产者均被拒绝。
- CUMCM2023C 当前真实 suite 覆盖问题 1 的描述性分布/关联、聚合预测、问题 2 的六品类七日补货定价、问题 3 的 27-33 个单品决策，并生成问题 4 数据建议、九节完整 Markdown 报告和经 XeLaTeX 编译、哈希验证及逐页版式检查的 PDF。该提交工件证明的是 2023 C 专用链路；跨题型自动论文生成、完整敏感性图表和竞赛级文献检索/引文核验仍未通用化。描述性相关不得解释为替代、互补或因果关系。

## 独立多 Agent 审查

用户要求独立求解者、质询者、修正者与裁判时，阅读 [多 Agent 工作流](references/multi_agent_workflow.md)，运行 `scripts/run_multi_agent.py`。裁判前必须同时提供 `--machine-gate` 和位于题目工作区内的 `--verification-script`。gate 必须绑定当前问题、运行、契约、输入和代码哈希；任一绑定失效都不能得到 `PASS`。

## 验证

```powershell
python -m pip install -r tools/requirements-dev.txt
python -m pytest tools/tests -q
```

## 比赛提交前总审计

单题 final_judge 通过不等于整场比赛已经可提交。进入最终提交阶段时，使用 tools/competition_readiness.py 检查逐问答案、关键数字证据、风险、实测 fallback、时间余量、官方规则确认、提交物哈希和人工检查状态。该审计不使用图片识别；视觉质量必须由人工确认。
使用 `tools/prepare_readiness_checklist.py` 从工作流摘要生成保守清单草稿，并通过 `--data-audit`、`--applicability-audit`、`--reproducibility-audit` 和 `--claim-registry` 显式绑定证据。生成器只填可机械验证的路径与哈希，不会声称已经就绪。
每问必须绑定非空答案工件和包含独立复算、检查或验证记录的 JSON。数据审计必须证明附件可读并绑定原始文件；适用性审计必须声明 `scope_boundary`，不支持的题型必须接入 specialist route 和领域验证器。论文关键结论用 `tools/paper_claim_registry.py` 逐条绑定问题编号、陈述、数值、单位、JSON path 或 CSV selector 与证据哈希。
正式 fallback 证据必须是故障后真实切换记录，且包含主路线失败、备用路线随后成功和最终选中备用路线。时间证据必须来自 `full_rehearsal` 或 `formal_contest`，覆盖 intake、contract、data、modeling、validation、paper、packaging、human_review 八阶段；论文、打包和人工复核不能使用占位耗时。已处置风险必须绑定诊断、敏感性分析、检查或独立验证证据。
可复现性证据必须记录原始命令、重跑命令、环境、输入/输出哈希、成功的重跑比较及一致的重跑输出绑定。最终审计会重新执行可复现性审计，不能只信任缓存的 `passed=true`。空文件、空 JSON、只有表头的 CSV/TSV，以及空 Markdown/TeX/HTML 均不算答案工件。
历史真实案例若只用于验证技术链路，可使用 tools/historical_rehearsal_audit.py 汇总 benchmark、DAG、报告和哈希证据。该审计输出 TECHNICAL_REHEARSAL_PARTIAL，不替代正式题目的 competition_readiness，也不会替代八阶段真实计时、fallback 运行、提交包和人工合规确认。
对已有多路线 benchmark，可先运行 tools/historical_fallback_audit.py，检查主路线与备用路线是否在同一实验注册表中真实执行并保留证据文件。其 claim_level 仅为 both_routes_executed_same_registry；若要声称“故障后自动切换”，还必须另行做故障注入演练。
正式赛题发布后，先使用 tools/competition_intake.py 对题目目录做文字层和结构化文件清点、哈希绑定及问题编号草稿。它明确不做图片识别，检测到未解析二进制时只产生 warning，并把每个问题契约留在 needs_input，不能替代人工题意确认。
契约补齐后用 tools/contract_completeness.py 检查预期输出、单位字典和语义人工确认；缺少单位、约束、假设或歧义确认时，不得把契约视为可进入数值求解的 ready。
tools/fallback_switch.py 与 tools/run_historical_failover_rehearsal.py 提供了可审计的故障注入切换演练：主回调被明确置失败，备用回调读取已验证路线证据。该演练证明切换语义和日志完整性，不替代正式题目下的完整重算。

正式完成的唯一机器判据是最终生成的 `competition_readiness.json` 满足 `verdict == "READY"`。在当届真题、逐问真实答案、独立验证、真实 fallback、完整计时、风险证据、复跑证据、论文与支撑材料、AI 声明、竞赛行为确认及人工视觉/匿名审查齐备前，必须保持 `NOT_READY`；历史演练不得升级为正式比赛就绪。

## 2026 国赛合规边界

2026 年竞赛期间允许使用 AI 辅助，但核心建模与分析必须由参赛队主导，AI 参与内容须逐项人工审查核实。不得与队外人员讨论赛题，也不得在交流平台浏览、发布或讨论赛题相关内容；因此赛中禁止检索或读取当届赛题讨论、共享解答、直播、论坛、群聊、GitHub/CSDN 等赛题相关内容。可读取官方赛题/规则和与当届赛题讨论无关、按论文规范引用的公开资料。论文参考文献前必须放 AI 工具使用声明，支撑材料必须包含 AI工具使用详情.pdf。规则证据可由程序归档，但确认与遵守状态只能由参赛队填写。

旧 `--v39-solver` 入口和 V38 benchmark 保留用于兼容与历史回归，但不能作为 V44 新能力的正确性证明。
