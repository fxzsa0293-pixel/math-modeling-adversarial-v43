# 比赛就绪审计

competition_readiness.py 检查整个比赛交付是否就绪，比单个模型的 final_judge.json 更严格。它只读取结构化清单和文件字节，不使用图片识别。

## 运行

在技能根目录执行：

    python tools/competition_readiness.py D:\problem\competition_readiness.json --output-dir D:\problem\readiness_audit

只有退出码为 0 且 verdict=READY 才表示清单中声明的交付满足门槛。NOT_READY 会列出缺口，不代表已有模型结果全部无效。

已运行多问题工作流后，可先生成保守草稿：

    python tools/prepare_readiness_checklist.py D:\problem\run\workflow_summary.json D:\problem\competition_readiness.json

草稿只自动绑定工作流、逐问契约和结果哈希。关键结论、fallback、风险、计时、AI 使用记录及人工确认保持未完成，必须按真实过程填写。

完整演练或正式比赛开始时，用 competition_timer.py 记录八阶段真实用时：

    python tools/competition_timer.py timing.json start --problem-id contest_problem
    python tools/competition_timer.py timing.json begin intake
    python tools/competition_timer.py timing.json end intake --passed
    python tools/competition_timer.py timing.json finalize

每个阶段可有多个 session，但同一时间只能有一个活动阶段。只有八阶段最后状态均 passed，计时证据才通过。

## 清单要求

- version 固定为 v44-readiness-1，problem_id 必填。
- official_rules 记录当届官方规则、AI 政策、提交要求、来源、确认时间和规则文件哈希。
- official_rules 还必须确认核心建模由参赛队主导、AI 输出须人工核验、不得与队外讨论赛题、不得浏览赛题相关交流平台。
- time_budget 记录总时限、实测端到端用时、预留时间和计时日志哈希。
- data_audit 必须绑定通过的数据审查 JSON 及 SHA-256；文件必须可读取且不能遗留不可读数据文件。
- applicability_audit 必须绑定通过的题型适用性审计；不支持的题型必须明确进入 specialist route 或被拒绝，不能套用通用模板。
- reproducibility_audit 必须绑定通过的可复现性证据；必须记录运行命令、重跑命令、环境、输入/输出哈希和成功的重跑比较。
- 计时日志必须是 kind=competition_timing、measurement_mode 为 full_rehearsal 或 formal_contest 且 passed=true，并覆盖 intake、data_audit、baseline、candidate_routes、independent_validation、paper、packaging、human_review 八个阶段；每阶段必须有实际 session，阶段总和、日志总用时和清单总用时必须一致。论文、打包和人工复核阶段若均低于 0.1 分钟，会被视为占位计时而拒绝。
- workflow_summary 使用 path 和 sha256 绑定通过的多问题工作流汇总。
- workflow_manifest 使用 path 和 sha256 绑定底层 DAG manifest；审计器会重新验证其中全部输入、输出、契约和节点运行 manifest 的哈希。
- expected_question_ids 必须列出题面全部问题编号，并与 questions 精确一致。
- 每一问必须绑定通过的节点、带哈希的 ready 契约、答案工件和至少一个关键结论证据。
- 每一问必须绑定独立验证 JSON；验证必须通过，并包含 checks、verification、recomputed 或 independent 结构字段，不能只依赖求解节点的 passed 标志。
- 必须绑定通过的 paper_claim_registry；注册表逐条覆盖每问关键声明，记录论文表述、数值、单位和可重新读取的证据路径。
- 关键结论证据还必须提供 json_path，或提供包含 where 与 column 的 csv_selector；审计器会重新读取该位置并核对声明值，可用 tolerance 声明数值容差。
- 每个关键结论必须写 unit；无量纲指标明确写 dimensionless。每问的答案和契约必须属于清单所指的工作流节点，不能绑定任意外部文件。
- 每一问必须记录不同于主路线的备用路线，并提供真实运行证据。
- fallback 证据必须是通过的 fallback_switch_rehearsal JSON，且其中 primary_route 与 fallback_route 必须和清单一致；必须记录主路线失败事件、备用路线成功事件、最终 selected_route，以及事件顺序。
- 高风险必须缓解并绑定证据；中低风险可以明确接受，但不能隐藏。
- 标为 mitigated 的风险证据必须是 JSON，且 passed/core_pass 为真或 verdict 为 pass/ready，并包含 checks、sensitivity、diagnostics、verification 或 analysis 结构之一。
- 所有必交文件必须存在且 SHA-256 匹配。
- 使用 AI 时，paper、support_archive 和 ai_usage_detail_pdf 必须列入交付物；可用 max_bytes 验证官方大小上限。
- ai_usage 必须确认论文声明、详情文件和全部 AI 输出的人工核验状态。
- competition_conduct 必须在本次执行后由参赛队确认：核心建模由队伍主导、无队外赛题讨论、无赛题相关交流平台浏览、全部 AI 内容已人工审查，并记录确认人和时间。
- 若清单标记检测到禁止行为或图片识别，最终审计直接判定 NOT_READY，而不是仅产生 warning。
- 论文内容、视觉版式和附件格式必须由人工确认。

READY 只证明清单中的交付满足这些可验证门槛，不证明全局最优、题意解释唯一或必然获奖。

## 提交包机械检查

使用 submission_package_audit.py 检查 PDF 文字层和 ZIP 内容，不使用 OCR 或图片识别：

    python tools/submission_package_audit.py paper.pdf support.zip --expected-support-file code/solve.py --output submission_audit.json

它检查 20MB 上限、第一页摘要、排除承诺书/编号页、目录、正文页数、AI 声明在参考文献之前、匿名敏感词、AI工具使用详情.pdf、源代码和附录文件列表。RAR 无标准库可靠解析，因此机械审计要求 ZIP。视觉排版始终保持 visual_layout_verified=false，必须人工确认。

将生成的 submission_audit.json 以 path 和 sha256 写入总清单的 submission_package_audit。总审计会要求该检查通过，并核对其中的论文和支撑包哈希与 deliverables 完全一致。
