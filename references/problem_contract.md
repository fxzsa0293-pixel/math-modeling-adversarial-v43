# V44 Problem Contract

`problem_contract.json` 将题意绑定到实际数据。路径使用绝对路径。

```json
{
  "version": "v44",
  "problem_id": "retail_q1",
  "task_type": "forecasting",
  "status": "ready",
  "data_file": "D:\\problem\\sales.xlsx",
  "sheet_name": "daily_sales",
  "target_column": "sales",
  "time_column": "date",
  "group_columns": [],
  "feature_columns": ["price", "promotion"],
  "known_future_columns": ["promotion"],
  "indicator_directions": {},
  "units": {"sales": "kg/day", "price": "CNY/kg"},
  "constraints": [],
  "expected_outputs": ["held-out predictions", "MAE", "RMSE", "WAPE"],
  "forecast_horizon": 7,
  "seasonal_period": 7,
  "validation_mode": "fixed_origin_holdout",
  "source": "provided",
  "unresolved_fields": [],
  "notes": []
}
```

## Task Types

- `forecasting`: 必须给出 `target_column`、`time_column` 和预测步长。
- `seasonal_period` 为正整数，表示递归季节朴素路线的周期长度，例如日频周周期为 7。它必须在查看 final test 前由题意、采样频率或训练期诊断确定。
- 预测任务只有列入 `known_future_columns` 的外生特征可以读取预测期取值；它们必须同时列入 `feature_columns`。价格、天气等未知未来变量应先单独预测、构造场景，或只使用其历史滞后值。
- `regression`: 必须给出 `target_column`；分组样本应填写 `group_columns` 并使用 `group_holdout`。通用流程会保持 train、selection、final test 的组互斥。
- `evaluation_ranking`: 必须用 `indicator_directions` 将每个指标声明为 `higher_better` 或 `lower_better`；不要包含 ID。
- `constrained_optimization`: 可声明 `optimization_model` 或 `network_model` 走原生 LP/MILP/网络求解；未声明时需要专用 adapter 和领域验算器。
- 多目标约束优化可声明 `multiobjective_model`，其中 `base_model` 使用同样的变量、边界和线性约束，`objectives` 必须恰好声明两个非零线性目标；原生路线求解单目标锚点并用固定 epsilon 网格枚举候选，再独立重算边界、整数性、约束、目标值和非支配关系。输出是经验证的采样 Pareto 解集，不代表穷尽了完整 Pareto 前沿。三目标及以上须使用 specialist adapter。
- 多表任务使用 `data_sources=[{alias,path,sheet_name?}]`、`base_table` 和 `data_joins`。每项 join 必须声明 `left`、`right`、`on` 或 `left_on/right_on`、`how`、`validate`；`validate` 取 `one_to_one`、`one_to_many`、`many_to_one` 或 `many_to_many`。可用 `max_unmatched_left_rate` 与 `max_unmatched_right_rate` 限制未匹配键比例，范围为 0 到 1。右表非键列默认命名为 `<alias>__<column>`。
- 鲁棒优化使用 `robust_optimization_model`，包含 `variables`、至少两个带概率的 `scenarios` 和 `risk_measure`。每个场景声明 `objective_coefficients` 与 `linear_constraints`，概率和必须为 1。`risk_measure` 可为 `expected`、`worst_case` 或 `cvar`；CVaR 还需 `alpha`。原生实现只接受最小化损失，收益最大化应以负收益作为损失。
- 零售补货定价使用 `retail_decision_model`。模型包含场景概率、商品列表、每件商品的成本/损耗/补货上下限和离散 `price_options`；每个价格选项必须给出全部场景需求。可用 `min_selected/max_selected` 控制总选品数，用 `group_selection_bounds` 控制品类覆盖，用 `category_service_requirements` 声明各品类在各场景下的最低总销量，用 `budget`、`shortage_penalty`、`risk_measure` 和 `stress_test` 声明经营约束及稳健性协议。`stress_test.demand_correlation` 可给出与商品 `category`（缺省时为 `group`）完全对应的半正定相关矩阵。`retail_decision_manifest` 用于绑定由附件计算得到的模型参数和上游预测；手工声明的通用模型可以不提供 manifest，但不得声称参数来自真实附件。
- `mechanism`: 可声明受限 `mechanism_model` 走原生 ODE 拟合，目前支持 `first_order_relaxation`、`logistic_growth`、`sir`；其它方程需要专用 adapter 和领域验算器。
- 由原始附件派生机理时序时，可用 `mechanism_data_manifest` 绑定原始文件、派生序列、辅助参数表、解析协议和 `mechanism_model` 哈希。当前内置重算器覆盖 CUMCM2018A 附件格式；其它附件格式需要对应的专用来源验证器。
- `spatial`: 原生支持声明式 `spatial_model.kind=idw`，使用固定 seed 的空间 block train/selection/final-test 划分、IDW 与全局均值基线，并执行独立预测/指标/划分重算；其它空间模型仍需要专用 adapter 和领域验算器。
- `simulation`: 可声明 `simulation_model.kind=mm1_queue`，或用 `kind=mmc_queue` 并提供 `servers`；可选 `capacity` 表示系统总容量 K。`arrival_distribution` 与 `service_distribution` 支持 `exponential`、`deterministic`、`empirical`，经验分布通过正值 `values` 和可选 `weights` 声明。纯指数模型使用 Erlang-C 或 birth-death 解析基线，其它分布使用同均值确定性基线。无限容量模型必须满足 `arrival_rate < servers * service_rate`。
- `spatial`: `coordinate_system` 必须为 `projected` 或 `geographic_wgs84`。前者的 `block_size` 使用投影坐标单位，后者使用公里；经纬度字段顺序为 x=longitude、y=latitude。原生证据包含距离类型和 final-test residual Moran's I。
- 装配后需要聚合时，声明 `data_aggregation.group_by`、命名 `aggregations` 和可选 `sort_by`。每个聚合项包含 `column` 与 `function`，函数支持 `sum`、`mean`、`min`、`max`、`median`、`count`、`nunique`、`std`。

## 专用路线契约

复杂题应声明每条可执行路线、其角色和度量方向。`adapter_path` 与 `domain_verifier` 默认必须位于绑定数据文件所在目录；复用经过审查的公共代码时，显式列入 `allowed_code_roots`。专用脚本只会在传入 `--allow-specialist-code` 后执行，并且不会被自动修补。

```json
{
  "task_type": "constrained_optimization",
  "data_file": "D:\\problem\\inputs.csv",
  "constraints": [{"name": "capacity"}],
  "metric_directions": {"objective": "higher_better"},
  "allowed_code_roots": ["D:\\problem\\reviewed_code"],
  "specialist_routes": [{
    "route_id": "feasible_baseline",
    "role": "baseline",
    "family": "optimization",
    "adapter_path": "D:\\problem\\reviewed_code\\feasible_baseline.py",
    "expected_metrics": ["objective", "constraint_violation"]
  }],
  "domain_verifier": "D:\\problem\\reviewed_code\\verify_solution.py"
}
```

每个优化 adapter 必须输出 `decision_variables`、`objective_components` 和覆盖全部已声明约束的 `constraint_checks`。验算器通过标准输出一个 JSON 对象，且必须提供 `passed` 布尔值、`issues` 数组和 `checks` 数组。

自动模式发现多个可读数据文件或一个工作簿含多个 sheet，且契约未明确装配关系时保持 `needs_input`。此时应声明一个绑定文件和 sheet，或填写 `data_sources`、`base_table`、`data_joins` 及可选 `data_aggregation`；只有超出声明式装配能力的变换才需要专用加载器。不能让通用适配器猜测关联键、基数或聚合口径。

## Stop Conditions

以下情况把 `status` 保持为 `needs_input`：目标、sheet、关联键、指标方向、时间顺序或硬约束会改变问题含义且无法从题面唯一确定。以下情况使用 `invalid`：绑定文件不在审计范围、列或 sheet 不存在、版本不支持。
