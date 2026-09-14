from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from .schemas import CorpusHit, ProblemProfile, RouteSpec


def plan_routes(profile: ProblemProfile, hits: Iterable[CorpusHit], max_routes: int = 12) -> List[RouteSpec]:
    if profile.contract and profile.contract.specialist_routes:
        routes = [
            RouteSpec(
                str(item["route_id"]),
                str(item["role"]),
                str(item["family"]),
                str(item.get("rationale", "Contract-provided specialist route.")),
                [str(metric) for metric in item["expected_metrics"]],
                source="contract",
                supported=True,
                adapter_path=str(Path(item["adapter_path"]).resolve()),
            )
            for item in profile.contract.specialist_routes
        ]
        return _dedupe(routes)[:max_routes]
    if profile.contract and profile.contract.task_type == "constrained_optimization" and profile.contract.multiobjective_model:
        return [
            RouteSpec("multiobjective_single_objective_baseline", "baseline", "multiobjective", "First declared objective optimum as a reference point.", ["pareto_count"]),
            RouteSpec("multiobjective_epsilon_constraint", "candidate", "multiobjective", "Epsilon-constraint enumeration with independent Pareto filtering.", ["pareto_count"]),
        ][:max_routes]
    if profile.contract and profile.contract.task_type == "constrained_optimization" and profile.contract.retail_decision_model:
        return [
            RouteSpec("retail_fixed_markup_baseline", "baseline", "retail_decision", "Nearest declared price option to the fixed markup policy, with optimized replenishment.", ["objective", "expected_profit", "constraint_violation"]),
            RouteSpec("retail_joint_price_replenishment", "candidate", "retail_decision", "Joint assortment, discrete price, and replenishment optimization under the declared scenario risk measure.", ["objective", "expected_profit", "constraint_violation"]),
        ][:max_routes]
    if profile.contract and profile.contract.task_type == "constrained_optimization" and profile.contract.robust_optimization_model:
        return [
            RouteSpec("robust_expected_baseline", "baseline", "robust_optimization", "Expected-cost solution evaluated under the declared risk measure.", ["objective", "constraint_violation"]),
            RouteSpec("robust_declared_risk_solver", "candidate", "robust_optimization", "Scenario-feasible optimization under the declared expected, worst-case, or CVaR risk measure.", ["objective", "constraint_violation"]),
        ][:max_routes]
    if profile.contract and profile.contract.task_type == "constrained_optimization" and (profile.contract.optimization_model or profile.contract.network_model):
        return [
            RouteSpec("lower_bound_feasible_baseline", "baseline", "optimization", "Declared lower-bound feasible baseline.", ["objective", "constraint_violation"]),
            RouteSpec("declared_milp_solver", "candidate", "optimization", "Native MILP solve under the declared model.", ["objective", "constraint_violation"]),
            RouteSpec("linear_relaxation_diagnostic", "ablation", "optimization", "Continuous relaxation diagnostic, not a competing baseline.", ["objective", "constraint_violation"]),
        ][:max_routes]
    if profile.contract and profile.contract.task_type == "mechanism" and profile.contract.mechanism_model:
        return [
            RouteSpec("mechanism_constant_baseline", "baseline", "mechanism", "Constant initial-state trajectory under the declared time split.", ["RMSE", "MAE"]),
            RouteSpec("declared_ode_multistart", "candidate", "mechanism", "Bounded multi-start ODE parameter estimation.", ["RMSE", "MAE"]),
        ][:max_routes]
    if profile.contract and profile.contract.task_type == "simulation" and profile.contract.simulation_model:
        model = profile.contract.simulation_model
        exponential = all(model.get(name, {"kind": "exponential"}).get("kind", "exponential") == "exponential" for name in ("arrival_distribution", "service_distribution"))
        return [
            RouteSpec("simulation_analytical_baseline" if exponential else "simulation_mean_deterministic_baseline", "baseline", "simulation", "Closed-form queue baseline." if exponential else "Deterministic queue at the declared mean interarrival and service times.", ["mean_wait", "utilization", "rejection_rate"]),
            RouteSpec("simulation_queue_monte_carlo", "candidate", "simulation", "Independent-seed event-driven queue Monte Carlo with confidence intervals.", ["mean_wait", "utilization", "rejection_rate", "ci_half_width"]),
        ][:max_routes]
    if profile.contract and profile.contract.task_type == "spatial" and profile.contract.spatial_model:
        return [
            RouteSpec("spatial_global_mean_baseline", "baseline", "spatial", "Global mean baseline under spatial block holdout.", ["MAE", "RMSE"]),
            RouteSpec("spatial_idw", "candidate", "spatial", "Inverse-distance weighting with spatial block holdout.", ["MAE", "RMSE"]),
        ][:max_routes]
    routes: List[RouteSpec] = [_baseline_for(profile)]
    task_type = profile.contract.task_type if profile.contract else ""
    contract_ready = bool(profile.contract and profile.contract.status == "ready")
    has_target = bool(contract_ready and profile.contract.target_column)
    if task_type == "forecasting":
        if has_target:
            routes.extend([
                RouteSpec("rolling_mean_model", "candidate", "forecasting", "Time-ordered rolling mean forecast.", ["MAE", "RMSE", "WAPE"]),
                RouteSpec("seasonal_naive_forecast", "candidate", "forecasting", "Recursive seasonal-naive forecast using the declared seasonal period.", ["MAE", "RMSE", "WAPE"]),
                RouteSpec("regularized_lag_model", "candidate", "forecasting", "Ridge lag-feature model with time-order validation.", ["MAE", "RMSE", "WAPE"]),
                RouteSpec("tuned_ridge_lag_search", "candidate", "forecasting", "Searches ridge alpha values on engineered lag/time features.", ["MAE", "RMSE", "WAPE"]),
                RouteSpec("random_forest_feature_search", "candidate", "forecasting", "Searches random-forest depth/leaf settings on engineered features.", ["MAE", "RMSE", "WAPE"]),
                RouteSpec("gradient_boosting_feature_search", "candidate", "forecasting", "Searches gradient boosting settings for nonlinear trend capture.", ["MAE", "RMSE", "WAPE"]),
                RouteSpec("elastic_net_feature_search", "candidate", "forecasting", "Searches sparse linear models on engineered features.", ["MAE", "RMSE", "WAPE"]),
                RouteSpec("extra_trees_feature_search", "candidate", "forecasting", "Searches extremely randomized trees for robust nonlinear signal.", ["MAE", "RMSE", "WAPE"]),
                RouteSpec("model_family_ensemble", "composite", "hybrid", "Averages ridge, random forest, and boosting predictions.", ["MAE", "RMSE", "WAPE"]),
                RouteSpec("residual_hybrid_forecast", "composite", "hybrid", "Combines ridge trend with residual tree correction.", ["MAE", "RMSE", "WAPE"]),
            ])
        else:
            routes.append(RouteSpec("forecast_target_required", "candidate", "forecasting", "Forecasting requires an explicit target-like column.", ["MAE"], supported=False))
    if task_type == "evaluation_ranking":
        routes.extend([
            RouteSpec("entropy_weight_ranking", "candidate", "evaluation", "Entropy-weight ranking using declared indicator directions and dispersion weights.", ["ranking_stability"]),
            RouteSpec("topsis_ranking", "candidate", "evaluation", "TOPSIS ranking after declared cost/benefit orientation.", ["ranking_stability"]),
            RouteSpec("pca_factor_ranking", "candidate", "evaluation", "PCA diagnostic ranking with explicit sign orientation.", ["ranking_stability"]),
            RouteSpec("stability_weighted_ranking", "candidate", "evaluation", "Weights indicators by inverse instability and reports perturbation stability.", ["ranking_stability"]),
        ])
    if task_type == "constrained_optimization":
        routes.extend([
            RouteSpec("greedy_repair_policy", "candidate", "optimization", "Requires explicit constraints and feasibility checker; not faked by generic template.", ["objective", "constraint_violation"], supported=False),
            RouteSpec("relaxed_lp_rounding", "candidate", "optimization", "Requires decision variables and constraint matrix; not faked by generic template.", ["objective", "constraint_violation"], supported=False),
        ])
    if task_type == "regression" and has_target:
        routes.extend([
            RouteSpec("regularized_regression_cv", "candidate", "tabular", "Regularized regression with engineered numeric features.", ["MAE", "RMSE", "R2"]),
            RouteSpec("tree_ensemble_screening", "candidate", "tabular", "Random forest screening for nonlinear feature contribution.", ["MAE", "RMSE", "R2"]),
            RouteSpec("tuned_ridge_lag_search", "candidate", "tabular", "Light alpha search for robust linear baseline improvement.", ["MAE", "RMSE", "R2"]),
            RouteSpec("random_forest_feature_search", "candidate", "tabular", "Light random forest hyperparameter search.", ["MAE", "RMSE", "R2"]),
            RouteSpec("gradient_boosting_feature_search", "candidate", "tabular", "Light gradient boosting search for nonlinear tabular signal.", ["MAE", "RMSE", "R2"]),
            RouteSpec("elastic_net_feature_search", "candidate", "tabular", "Sparse linear feature selection route.", ["MAE", "RMSE", "R2"]),
            RouteSpec("extra_trees_feature_search", "candidate", "tabular", "Extremely randomized trees route for nonlinear tabular signal.", ["MAE", "RMSE", "R2"]),
            RouteSpec("model_family_ensemble", "composite", "hybrid", "Averages strong model-family predictions for robustness.", ["MAE", "RMSE", "R2"]),
        ])
    if task_type in {"spatial", "mechanism"}:
        routes.append(RouteSpec("mechanism_or_spatial_score", "candidate", "mechanism_spatial", "Requires domain equations or spatial distance definitions; not faked by generic template.", ["score", "sensitivity"], supported=False))
    if task_type in {"forecasting", "regression"} and has_target:
        routes.extend([
            RouteSpec("robust_feature_ablation", "ablation", "diagnostic", "Median/feature ablation diagnostic, not a main route.", profile.metric_candidates[:2] or ["score"]),
            RouteSpec("innovation_hybrid_route", "composite", "hybrid", "General hybrid entry for model stacking or residual correction.", profile.metric_candidates[:3] or ["score"]),
        ])
    _enrich_from_hits(routes, hits, has_target and task_type in {"forecasting", "regression"})
    return _dedupe(routes)[:max_routes]


def _baseline_for(profile: ProblemProfile) -> RouteSpec:
    task_type = profile.contract.task_type if profile.contract else ""
    ready_target = bool(profile.contract and profile.contract.status == "ready" and profile.contract.target_column)
    if task_type == "forecasting":
        return RouteSpec("mean_or_last_value_baseline", "baseline", "baseline", "Last-value baseline under the contract forecast protocol.", ["MAE", "RMSE", "WAPE"], supported=ready_target)
    if task_type == "constrained_optimization":
        return RouteSpec("simple_feasible_baseline", "baseline", "optimization", "Optimization baseline requires explicit constraints; not faked by generic template.", ["objective", "constraint_violation"], supported=False)
    if task_type == "evaluation_ranking":
        return RouteSpec("equal_weight_baseline", "baseline", "evaluation", "Equal-weight ranking baseline using declared indicator directions.", ["ranking_stability"])
    if task_type in {"mechanism", "spatial"}:
        return RouteSpec("domain_baseline_required", "baseline", "domain_specific", "A domain baseline and verifier must be implemented before execution.", ["score"], supported=False)
    if task_type == "regression":
        return RouteSpec("simple_statistical_baseline", "baseline", "baseline", "Mean-value baseline for generic tabular prediction.", ["MAE", "RMSE", "R2"], supported=ready_target)
    return RouteSpec("contract_required", "baseline", "unsupported", "A valid task contract is required before executable routes are planned.", ["score"], supported=False)


def _enrich_from_hits(routes: List[RouteSpec], hits: Iterable[CorpusHit], has_target: bool) -> None:
    methods = []
    for hit in hits:
        methods.extend(hit.suggested_methods)
    if methods and has_target:
        routes.append(RouteSpec("corpus_inspired_hybrid", "candidate", "hybrid", "Hybrid route suggested by retrieved historical methods: " + "; ".join(list(dict.fromkeys(methods))[:4]), ["MAE", "RMSE", "R2"], "corpus"))


def _dedupe(routes: Iterable[RouteSpec]) -> List[RouteSpec]:
    out = []
    seen = set()
    for route in routes:
        if route.route_id in seen:
            continue
        seen.add(route.route_id)
        out.append(route)
    return out
