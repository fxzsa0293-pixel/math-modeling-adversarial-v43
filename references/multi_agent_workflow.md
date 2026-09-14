# Multi-Agent Workflow

Use this mode when the user asks for genuine independent model roles, adversarial review, or an auditable solver-reviewer-judge loop. It requires `DASHSCOPE_API_KEY` and uses DashScope's OpenAI-compatible endpoint by default.

## Boundary

The language-model agents produce route proposals, criticisms, revisions, and judgments. They must not fabricate numerical evidence. A route is eligible only when real local code produces recomputable results and any required constraint-audit artifact. The judge receives selected structured evidence excerpts and a machine gate; a file name or a brief summary is not proof.

## Roles

- `solver`: proposes several distinct routes, assumptions, and evidence requirements.
- `critic_method`: audits mathematical formulation, objective linearization, and optimality claims.
- `critic_data`: audits provenance, joins, units, assumptions, reproducibility, and code/result linkage.
- `critic_operations`: audits feasibility and domain constraints.
- `reviser`: addresses each finding as `ACCEPT`, `PARTIAL_ACCEPT`, or `REJECT`; creates the executable revision plan.
- `judge`: returns `PASS`, `REVISE`, or `REJECT` based on artifacts, not prose quality.

The three critics run in parallel but are separate API calls and have no access to each other's messages. The reviser and judge get only the stored handoff material.

## Run

Create a concise `brief.md` in the problem workspace. It should identify the problem, data path, requested subproblem, known assumptions, and current route status.

```powershell
$py = 'C:\Users\yuki\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py 'C:\Users\yuki\.codex\skills\math-modeling-adversarial-v43\scripts\run_multi_agent.py' `
  --brief 'G:\数学建模资料\2024C题\求解\对抗流程\q1_brief.md' `
  --artifact-root 'G:\数学建模资料\2024C题' `
  --run-dir 'G:\数学建模资料\2024C题\求解\对抗流程\agent_runs\q1_round_01' `
  --verification-script 'G:\数学建模资料\2024C题\求解\verify_q1.py' `
  --machine-gate 'G:\数学建模资料\2024C题\求解\run\judge\final_judge.json' `
  --rounds 2
```

Both paths are required for `PASS` and must resolve inside `--artifact-root`. The verification script is an existing, human-selected Python file; the workflow never executes code copied from an agent response. The machine gate must include a problem id, run id, contract hash, passing gate layers, and current SHA-256 bindings for every input and executed route file. `PASS` additionally requires a zero exit code from the verification script. Missing or stale bindings force `REVISE`.

Use an API key through the environment; do not write it into a script, transcript, or output file. Configure another OpenAI-compatible endpoint only with `--endpoint` and `--model`.

## Crop-Planning Route Pool

For deterministic crop-planning questions with fixed yield, cost, price, and demand assumptions, register at least one baseline and several materially different routes. Do not run all routes mechanically; eliminate routes whose assumptions conflict with the problem.

| Route ID | Formulation | Purpose | Required evidence |
| --- | --- | --- | --- |
| `aggregate_lp_upper_bound` | Aggregate LP by year, land type, crop, season | Profit benchmark / upper bound before plot rotation | Demand derivation, balance constraints, LP status |
| `plot_milp_exact` | Plot-year-season area plus use binaries | Exact formulation of rotation, legume windows, modes, fragmentation | Solver gap/status, full constraint audit |
| `benders_or_type_assignment` | Crop-structure master problem plus plot-assignment subproblem | Reduce integer scale while retaining implementable plans | Master/subproblem logs, repair feasibility |
| `rolling_horizon_milp` | Repeated 3-year MILP with state transfer | Make three-year legume windows tractable | State handoff, terminal-window audit |
| `goal_programming_milp` | Lexicographic / epsilon-constraint profit, oversupply, fragmentation objectives | Management-friendly tradeoff plan | Priority or epsilon rationale, Pareto comparison |
| `robust_demand_milp` | Budgeted/interval demand uncertainty | Sensitivity to sales proxy uncertainty | Uncertainty set, nominal/worst-case comparison |
| `lagrangian_or_repair_heuristic` | Greedy/constructive solution plus exact repair | Fast feasible baseline for comparison and warm start | Deterministic seed/order, full audit |
| `metaheuristic_with_repair` | GA/SA/ALNS with hard-feasibility repair | Explore difficult discrete layouts if exact methods time out | Multiple seeds, repair audit, no unsupported optimality claim |

For 2024 CUMCM C Question 1, a recommended first competition set is:

1. `aggregate_lp_upper_bound`
2. `lagrangian_or_repair_heuristic`
3. `plot_milp_exact` with time limit and documented optimality gap
4. `rolling_horizon_milp`
5. `goal_programming_milp`
6. `robust_demand_milp` as sensitivity analysis rather than a replacement for the two required scenarios

## Gate Conditions

A judge should return `REVISE` when the machine gate fails or any selected route lacks one of the following:

- a stated demand construction rule and sensitivity note;
- crop/land/season parameter matching verification;
- water-irrigated and greenhouse mode feasibility checks;
- adjacent-season/year replanting and rolling three-year legume audit;
- production, normal sales, surplus, revenue, and cost reconciliation;
- a comparable baseline and the actual solver/heuristic execution status;
- a route-specific limitation and an explicit statement of whether its result is optimal, bounded, or heuristic.

For non-agricultural problems, replace the crop-specific evidence list with the task contract and its domain verifier. The operations critic is domain-neutral and must not import crop rotation rules into unrelated questions.
