# V38 显式角色自包含 Skill 修复交付说明

## 修复结论

V38 已针对 `V37_建模Agent缺陷目录.md` 中的核心缺陷完成修复，并通过自包含复现与负向测试。当前版本更适合作为队友交接用的 Codex skill：解压后有数据、有入口、有测试、有证据门槛。

## 修复内容

- D01：baseline 不再由路线名称推断，改为 `ROUTE_CONTRACT` 显式声明；
- D02：禁用 V30 中会产生虚假默认 `1.0` 指标的 2012A wrapper；
- D03：移除 V36 2012A 的 `second_panel_baseline` 自预测泄漏路线；
- D04：明确 V38 是 benchmark evidence gate，不伪装成完整自动对抗求解闭环；
- D06/D07：删除破损 `orchestrator.py`，统一入口为 `tools/run.py`；
- D08：删除旧 patch 脚本，避免队友误执行；
- D10：运行时默认使用包内 `data/reference_root`，不再硬编码 `F:` / `G:` 路径；
- D22：增加负向测试，覆盖缺角色、缺主指标、推荐路线不存在、自预测路线残留。

## 验证结果

```text
python run.py --self-contained
case_count=11
surviving_route_count=35
eliminated_route_count=10
gate.verdict=pass
```

```text
python -m pytest .\tests -q
6 passed
```

## 口径说明

V38 比 V37 更严格也更诚实：案例数从 12 变成 11，不是能力退化，而是删除了一个会污染证据的旧 wrapper。当前结果适合支撑“模型路线探索与推荐”，但仍不应宣传为“一键自动生成国一论文”。
