from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from .schemas import ProblemProfile, RouteSpec

SUPPORTED_FAMILIES = {"baseline", "forecasting", "tabular", "evaluation", "diagnostic", "hybrid"}

ADAPTER_HEADER = '''from __future__ import annotations

import json
import hashlib
import math
import time
from importlib.metadata import version as package_version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.base import clone
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

'''


def generate_adapters(profile: ProblemProfile, routes: Iterable[RouteSpec], output_dir: Path) -> List[Path]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for route in routes:
        if route.adapter_path:
            generated.append(Path(route.adapter_path).resolve())
            continue
        path = output_dir / f"{route.route_id}.py"
        path.write_text(_adapter_source(profile, route), encoding="utf-8")
        generated.append(path)
    return generated


def _adapter_source(profile: ProblemProfile, route: RouteSpec) -> str:
    contract = profile.contract
    readable_assets = [contract.data_file] if contract and contract.data_file else []
    supported = route.family in SUPPORTED_FAMILIES and route.supported
    payload = {
        "problem_id": profile.problem_id,
        "target_candidates": profile.target_candidates,
        "archetypes": profile.archetypes,
        "metric_candidates": profile.metric_candidates,
        "metric_directions": profile.metric_directions,
        "contract": contract.__dict__ if contract else {},
    }
    return ADAPTER_HEADER + f'''
ASSETS = {json.dumps(readable_assets, ensure_ascii=False, indent=2)}
PROFILE = {repr(payload)}
ROUTE = {repr(route.__dict__)}
SUPPORTED = {repr(supported)}


def read_table(path):
    if path.suffix.lower() in {{'.xlsx', '.xls'}}:
        return pd.read_excel(path, sheet_name=PROFILE.get('contract', {{}}).get('sheet_name') or 0)
    last_error = None
    for encoding in ('utf-8', 'gb18030', 'gbk', 'utf-16'):
        try:
            return pd.read_csv(path, sep=None, engine='python', encoding=encoding)
        except Exception as exc:
            last_error = exc
    raise last_error or RuntimeError('table read failed')


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_best_table():
    best = None
    best_score = -1
    for raw in ASSETS:
        path = Path(raw)
        try:
            frame = read_table(path)
        except Exception:
            continue
        if len(frame) < 8 or frame.shape[1] < 2:
            continue
        numeric = numeric_frame(frame)
        score = len(numeric) + numeric.shape[1] * 20 + target_column_score(numeric.columns) * 100
        if score > best_score:
            best = (path, frame)
            best_score = score
    if best is None:
        raise RuntimeError('no readable table with enough rows and columns')
    return best


def numeric_frame(frame):
    converted = frame.copy()
    for column in converted.columns:
        converted[column] = pd.to_numeric(converted[column], errors='coerce')
    converted = converted.dropna(axis=1, how='all').dropna(axis=0, how='all')
    return converted


def target_column_score(columns):
    terms = ['target', 'label', 'score', 'rank', 'sales', 'sale', 'demand', 'price', 'power', 'cost', 'profit', 'revenue']
    bad = ['id', 'code', 'index', 'longitude', 'latitude']
    score = 0
    for column in columns:
        lowered = str(column).lower()
        if any(term.lower() in lowered for term in terms) and not any(term.lower() in lowered for term in bad):
            score += 1
    return score


def choose_target(frame):
    contract = PROFILE.get('contract') or {{}}
    target = contract.get('target_column')
    if contract.get('status') != 'ready' or not target:
        raise RuntimeError('contract_needs_input: explicit target binding required')
    if target not in frame.columns:
        raise RuntimeError(f'contract_target_not_found: {{target}}')
    if not pd.api.types.is_numeric_dtype(frame[target]):
        raise RuntimeError(f'contract_target_not_numeric: {{target}}')
    return target


def regression_metrics(y_true, y_pred):
    if not np.all(np.isfinite(y_pred)) or not np.all(np.isfinite(y_true)):
        raise RuntimeError('non_finite_prediction_or_target')
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    denom = float(np.sum(np.abs(y_true))) or 1.0
    out = {{'MAE': round(mae, 6), 'RMSE': round(rmse, 6), 'WAPE': round(float(np.sum(np.abs(y_true - y_pred)) / denom), 6)}}
    try:
        r2 = float(r2_score(y_true, y_pred))
        if np.isfinite(r2):
            out['R2'] = round(r2, 6)
    except Exception:
        pass
    return out


def prefixed_metrics(prefix, y_true, y_pred):
    return {{f'{{prefix}}_{{name}}': value for name, value in regression_metrics(y_true, y_pred).items()}}


def time_order_split(X, y):
    cut = max(2, int(len(y) * 0.7))
    if len(y) - cut < 2:
        cut = len(y) - 2
    return X[:cut], X[cut:], y[:cut], y[cut:]


def random_split(X, y, row_ids):
    test_size = 0.3 if len(y) >= 20 else max(2, int(len(y) * 0.25)) / len(y)
    return train_test_split(X, y, row_ids, test_size=test_size, random_state=42)



def engineer_features(model_frame, target, is_forecast):
    engineered = model_frame.copy()
    engineered['_row_index'] = np.arange(len(engineered), dtype=float)
    excluded = {{target, '_source_row', *(PROFILE.get('contract', {{}}).get('group_columns') or [])}}
    numeric_cols = [column for column in engineered.columns if column not in excluded and pd.api.types.is_numeric_dtype(engineered[column])]
    feature_spec = {{'numeric_columns': list(numeric_cols), 'squared_columns': list(numeric_cols[:5]), 'interaction_columns': list(numeric_cols[:2])}}
    if is_forecast:
        for lag in (1, 2, 3, 7):
            engineered[f'{{target}}_lag_{{lag}}'] = engineered[target].shift(lag)
        engineered[f'{{target}}_roll3'] = engineered[target].shift(1).rolling(3, min_periods=1).mean()
        engineered[f'{{target}}_roll7'] = engineered[target].shift(1).rolling(7, min_periods=1).mean()
    for column in numeric_cols[:5]:
        engineered[f'{{column}}_sq'] = engineered[column] ** 2
    if len(numeric_cols) >= 2:
        engineered[f'{{numeric_cols[0]}}_x_{{numeric_cols[1]}}'] = engineered[numeric_cols[0]] * engineered[numeric_cols[1]]
    engineered = engineered.dropna(subset=[target]).copy()
    features = [column for column in engineered.columns if column not in excluded and pd.api.types.is_numeric_dtype(engineered[column])]
    if not features:
        raise RuntimeError('no numeric features after feature engineering')
    y = engineered[target].to_numpy(dtype=float)
    X = engineered[features].to_numpy(dtype=float)
    return X, y, features, engineered, feature_spec


def impute_from_train(X_train, X_test=None):
    medians = np.nanmedian(X_train, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    train = np.where(np.isfinite(X_train), X_train, medians)
    if X_test is None:
        return train, medians
    test = np.where(np.isfinite(X_test), X_test, medians)
    return train, test, medians


def forecast_feature_row(row, row_index, history, target, features, medians, feature_spec):
    values = {{}}
    values['_row_index'] = float(row_index)
    for column in feature_spec['numeric_columns']:
        if column == '_row_index':
            continue
        values[column] = float(row[column]) if column in row.index and pd.notna(row[column]) else np.nan
    for lag in (1, 2, 3, 7):
        values[f'{{target}}_lag_{{lag}}'] = history[-lag] if len(history) >= lag else np.nan
    values[f'{{target}}_roll3'] = float(np.mean(history[-3:])) if history else np.nan
    values[f'{{target}}_roll7'] = float(np.mean(history[-7:])) if history else np.nan
    for column in feature_spec['squared_columns']:
        value = values.get(column, np.nan)
        values[f'{{column}}_sq'] = value ** 2
    interaction = feature_spec['interaction_columns']
    if len(interaction) >= 2:
        values[f'{{interaction[0]}}_x_{{interaction[1]}}'] = values.get(interaction[0], np.nan) * values.get(interaction[1], np.nan)
    vector = np.asarray([values.get(feature, np.nan) for feature in features], dtype=float)
    return np.where(np.isfinite(vector), vector, medians).reshape(1, -1)


def recursive_predict(models, frame, start, target, features, medians, feature_spec, combine):
    history = frame[target].iloc[:start].astype(float).tolist()
    predictions = []
    for index in range(start, len(frame)):
        row = forecast_feature_row(frame.iloc[index], index, history, target, features, medians, feature_spec)
        values = [float(model.predict(row)[0]) for model in models]
        prediction = float(combine(values))
        predictions.append(prediction)
        history.append(prediction)
    return np.asarray(predictions)


def refit_for_final(models, model_frame, raw_cut, target, is_forecast, combine, X_train=None, y_train=None, X_selection=None, y_selection=None, X_test=None):
    if is_forecast:
        development = model_frame.iloc[:raw_cut].copy()
        X_development, y_development, final_features, _, final_spec = engineer_features(development, target, True)
        X_development, final_medians = impute_from_train(X_development)
        fitted = [clone(model).fit(X_development, y_development) for model in models]
        return recursive_predict(fitted, model_frame, raw_cut, target, final_features, final_medians, final_spec, combine)
    X_development = np.vstack([X_train, X_selection])
    y_development = np.concatenate([y_train, y_selection])
    fitted = [clone(model).fit(X_development, y_development) for model in models]
    return combine([model.predict(X_test) for model in fitted])


def inner_split(X_train, y_train, is_forecast):
    if len(y_train) < 8:
        return X_train, X_train, y_train, y_train
    if is_forecast:
        cut = max(2, int(len(y_train) * 0.75))
        if len(y_train) - cut < 2:
            cut = len(y_train) - 2
        return X_train[:cut], X_train[cut:], y_train[:cut], y_train[cut:]
    return train_test_split(X_train, y_train, test_size=0.25, random_state=7)


def select_best_regressor(factory_configs, X_train, y_train, is_forecast):
    inner_X_train, inner_X_val, inner_y_train, inner_y_val = inner_split(X_train, y_train, is_forecast)
    inner_X_train, inner_X_val, _ = impute_from_train(inner_X_train, inner_X_val)
    best_model = None
    best_score = float('inf')
    best_config = None
    for name, factory in factory_configs:
        model = factory()
        model.fit(inner_X_train, inner_y_train)
        pred = model.predict(inner_X_val)
        score = mean_absolute_error(inner_y_val, pred)
        if score < best_score:
            best_model = factory()
            best_score = score
            best_config = name
    fitted_X_train, medians = impute_from_train(X_train)
    best_model.fit(fitted_X_train, y_train)
    return best_model, best_config, medians


def select_route_model(factory_configs, X_train, y_train, is_forecast, train_frame, target):
    if not is_forecast:
        return select_best_regressor(factory_configs, X_train, y_train, False)
    inner_cut = max(8, int(len(train_frame) * 0.75))
    if len(train_frame) - inner_cut < 2:
        inner_cut = len(train_frame) - 2
    if inner_cut < 8:
        model = factory_configs[0][1]()
        model.fit(X_train, y_train)
        return model, factory_configs[0][0], None
    inner_frame = train_frame.iloc[:inner_cut].copy()
    inner_X, inner_y, inner_features, _, inner_feature_spec = engineer_features(inner_frame, target, True)
    inner_X, inner_medians = impute_from_train(inner_X)
    best_factory = None
    best_name = None
    best_score = float('inf')
    expected = train_frame[target].iloc[inner_cut:].to_numpy(dtype=float)
    for name, factory in factory_configs:
        model = factory()
        model.fit(inner_X, inner_y)
        predicted = recursive_predict([model], train_frame, inner_cut, target, inner_features, inner_medians, inner_feature_spec, np.mean)
        score = mean_absolute_error(expected, predicted)
        if score < best_score:
            best_factory = factory
            best_name = name
            best_score = score
    best_model = best_factory()
    best_model.fit(X_train, y_train)
    return best_model, best_name, None

def run_regression_route(route_id, role, family, frame):
    target = choose_target(frame)
    contract = PROFILE.get('contract') or {{}}
    task_type = contract.get('task_type')
    is_forecast = task_type == 'forecasting'
    if family == 'forecasting' and not is_forecast:
        raise RuntimeError(f'route_contract_mismatch: {{family}} route for {{task_type}} task')
    if family == 'tabular' and task_type != 'regression':
        raise RuntimeError(f'route_contract_mismatch: {{family}} route for {{task_type}} task')
    declared_features = contract.get('feature_columns') or []
    usable_features = (contract.get('known_future_columns') or []) if is_forecast else declared_features
    required_columns = [target] + ([contract.get('time_column')] if contract.get('time_column') else []) + usable_features + (contract.get('group_columns') or [])
    required_columns = list(dict.fromkeys(column for column in required_columns if column))
    restrict_columns = is_forecast or bool(declared_features)
    X_selection = X_test = None
    model_frame = frame[required_columns].dropna(subset=[target]).copy() if restrict_columns else frame.dropna(subset=[target]).copy()
    model_frame['_source_row'] = model_frame.index.to_numpy(dtype=int)
    if is_forecast:
        time_column = contract.get('time_column')
        if not time_column or time_column not in model_frame.columns:
            raise RuntimeError('contract_needs_input: forecasting requires a bound time_column')
        time_values = model_frame[time_column]
        if not pd.api.types.is_numeric_dtype(time_values):
            parsed_time = pd.to_datetime(time_values, errors='coerce')
            if parsed_time.isna().any():
                raise RuntimeError('forecast_time_column_contains_unparseable_values')
            model_frame['_parsed_time'] = parsed_time
            sort_column = '_parsed_time'
        else:
            sort_column = time_column
        if model_frame[sort_column].duplicated().any():
            raise RuntimeError('forecast_time_column_contains_duplicates')
        model_frame = model_frame.sort_values(sort_column, kind='stable').drop(columns=['_parsed_time'], errors='ignore').reset_index(drop=True)
        horizon = int(contract.get('forecast_horizon') or 1)
        if horizon >= len(model_frame) - 8:
            raise RuntimeError('contract_forecast_horizon_leaves_insufficient_training_data')
        raw_cut = len(model_frame) - horizon
        selection_size = max(2, min(horizon, raw_cut // 4))
        selection_cut = raw_cut - selection_size
        if selection_cut < 8:
            raise RuntimeError('contract_forecast_horizon_leaves_insufficient_selection_training_data')
        train_frame = model_frame.iloc[:selection_cut].copy()
        X_train, y_train, features, engineered, feature_spec = engineer_features(train_frame, target, True)
        X_train, medians = impute_from_train(X_train)
        y_selection = model_frame[target].iloc[selection_cut:raw_cut].to_numpy(dtype=float)
        y_test = model_frame[target].iloc[raw_cut:].to_numpy(dtype=float)
        test_row_ids = model_frame['_source_row'].iloc[raw_cut:].to_numpy(dtype=int)
        selection_row_ids = model_frame['_source_row'].iloc[selection_cut:raw_cut].to_numpy(dtype=int)
        X_test = None
    else:
        X, y, features, engineered, feature_spec = engineer_features(model_frame, target, False)
        row_ids = engineered['_source_row'].to_numpy(dtype=int)
        group_columns = contract.get('group_columns') or []
        if group_columns:
            groups = engineered[group_columns].astype(str).agg('||'.join, axis=1).to_numpy()
            if len(np.unique(groups)) < 3:
                raise RuntimeError('group_holdout_requires_at_least_three_groups')
            outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            development_index, test_index = next(outer.split(X, y, groups))
            X_development, X_test = X[development_index], X[test_index]
            y_development, y_test = y[development_index], y[test_index]
            development_rows, test_row_ids = row_ids[development_index], row_ids[test_index]
            development_groups = groups[development_index]
            inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=7)
            train_index, selection_index = next(inner.split(X_development, y_development, development_groups))
            X_train, X_selection = X_development[train_index], X_development[selection_index]
            y_train, y_selection = y_development[train_index], y_development[selection_index]
            selection_row_ids = development_rows[selection_index]
            validation_protocol = 'group_holdout'
        else:
            X_development, X_test, y_development, y_test, development_rows, test_row_ids = train_test_split(X, y, row_ids, test_size=0.2, random_state=42)
            X_train, X_selection, y_train, y_selection, _, selection_row_ids = train_test_split(X_development, y_development, development_rows, test_size=0.25, random_state=7)
            validation_protocol = 'train_selection_final_test_random'
    if len(model_frame) < 8:
        raise RuntimeError('not enough rows after cleaning')
    selected_config = None
    if role == 'baseline':
        if is_forecast:
            selection_pred = np.repeat(y_train[-1], len(y_selection))
            pred = np.repeat(y_selection[-1], len(y_test))
            selected_config = 'last_value'
        else:
            selection_pred = np.repeat(np.mean(y_train), len(y_selection))
            pred = np.repeat(np.mean(np.concatenate([y_train, y_selection])), len(y_test))
            selected_config = 'mean_value'
    elif route_id == 'rolling_mean_model':
        selection_pred = np.repeat(pd.Series(y_train).tail(min(7, len(y_train))).mean(), len(y_selection))
        pred = np.repeat(model_frame[target].iloc[:raw_cut].tail(min(7, raw_cut)).mean(), len(y_test))
        selected_config = 'tail_mean_7'
    elif route_id == 'seasonal_naive_forecast':
        period = int(contract.get('seasonal_period') or 7)
        if period < 1 or len(y_train) < period:
            raise RuntimeError('seasonal_naive_requires_sufficient_history')
        selection_history = y_train.astype(float).tolist()
        selection_values = []
        for _ in range(len(y_selection)):
            value = selection_history[-period]
            selection_values.append(value)
            selection_history.append(value)
        selection_pred = np.asarray(selection_values, dtype=float)
        final_history = model_frame[target].iloc[:raw_cut].astype(float).tolist()
        final_values = []
        for _ in range(len(y_test)):
            value = final_history[-period]
            final_values.append(value)
            final_history.append(value)
        pred = np.asarray(final_values, dtype=float)
        selected_config = f'seasonal_naive_period_{{period}}'
    elif route_id in {'regularized_lag_model', 'regularized_regression_cv'}:
        model = Ridge(alpha=1.0)
        if not is_forecast:
            X_train, X_selection, medians = impute_from_train(X_train, X_selection)
            X_test = np.where(np.isfinite(X_test), X_test, medians)
        model.fit(X_train, y_train)
        selection_pred = recursive_predict([model], model_frame.iloc[:raw_cut], selection_cut, target, features, medians, feature_spec, np.mean) if is_forecast else model.predict(X_selection)
        pred = refit_for_final([model], model_frame, raw_cut if is_forecast else None, target, is_forecast, lambda values: values[0], X_train, y_train, X_selection, y_selection, X_test)
        selected_config = 'ridge_alpha_1'
    elif route_id == 'tuned_ridge_lag_search':
        model, selected_config, selected_medians = select_route_model([(f'ridge_alpha_{{alpha}}', lambda alpha=alpha: Ridge(alpha=alpha)) for alpha in (0.01, 0.1, 1.0, 10.0, 100.0)], X_train, y_train, is_forecast, train_frame if is_forecast else None, target)
        if not is_forecast:
            medians = selected_medians
            X_train = np.where(np.isfinite(X_train), X_train, medians)
            X_selection = np.where(np.isfinite(X_selection), X_selection, medians)
            X_test = np.where(np.isfinite(X_test), X_test, medians)
        selection_pred = recursive_predict([model], model_frame.iloc[:raw_cut], selection_cut, target, features, medians, feature_spec, np.mean) if is_forecast else model.predict(X_selection)
        pred = refit_for_final([model], model_frame, raw_cut if is_forecast else None, target, is_forecast, lambda values: values[0], X_train, y_train, X_selection, y_selection, X_test)
    elif route_id in {'tree_ensemble_screening', 'random_forest_feature_search'}:
        configs = []
        for depth in (3, 5, None):
            for leaf in (1, 2, 4):
                configs.append((f'rf_depth_{{depth}}_leaf_{{leaf}}', lambda depth=depth, leaf=leaf: RandomForestRegressor(n_estimators=80, max_depth=depth, min_samples_leaf=leaf, random_state=42)))
        model, selected_config, selected_medians = select_route_model(configs, X_train, y_train, is_forecast, train_frame if is_forecast else None, target)
        if not is_forecast:
            medians = selected_medians
            X_train = np.where(np.isfinite(X_train), X_train, medians)
            X_selection = np.where(np.isfinite(X_selection), X_selection, medians)
            X_test = np.where(np.isfinite(X_test), X_test, medians)
        selection_pred = recursive_predict([model], model_frame.iloc[:raw_cut], selection_cut, target, features, medians, feature_spec, np.mean) if is_forecast else model.predict(X_selection)
        pred = refit_for_final([model], model_frame, raw_cut if is_forecast else None, target, is_forecast, lambda values: values[0], X_train, y_train, X_selection, y_selection, X_test)
    elif route_id == 'gradient_boosting_feature_search':
        configs = []
        for depth in (2, 3):
            for lr in (0.03, 0.06, 0.1):
                configs.append((f'gbr_depth_{{depth}}_lr_{{lr}}', lambda depth=depth, lr=lr: GradientBoostingRegressor(n_estimators=100, learning_rate=lr, max_depth=depth, random_state=42)))
        model, selected_config, selected_medians = select_route_model(configs, X_train, y_train, is_forecast, train_frame if is_forecast else None, target)
        if not is_forecast:
            medians = selected_medians
            X_train = np.where(np.isfinite(X_train), X_train, medians)
            X_selection = np.where(np.isfinite(X_selection), X_selection, medians)
            X_test = np.where(np.isfinite(X_test), X_test, medians)
        selection_pred = recursive_predict([model], model_frame.iloc[:raw_cut], selection_cut, target, features, medians, feature_spec, np.mean) if is_forecast else model.predict(X_selection)
        pred = refit_for_final([model], model_frame, raw_cut if is_forecast else None, target, is_forecast, lambda values: values[0], X_train, y_train, X_selection, y_selection, X_test)
    elif route_id == 'elastic_net_feature_search':
        configs = []
        for alpha in (0.001, 0.01, 0.1, 1.0):
            for l1_ratio in (0.15, 0.5, 0.85):
                configs.append((f'elastic_alpha_{{alpha}}_l1_{{l1_ratio}}', lambda alpha=alpha, l1_ratio=l1_ratio: ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, random_state=42)))
        model, selected_config, selected_medians = select_route_model(configs, X_train, y_train, is_forecast, train_frame if is_forecast else None, target)
        if not is_forecast:
            medians = selected_medians
            X_train = np.where(np.isfinite(X_train), X_train, medians)
            X_selection = np.where(np.isfinite(X_selection), X_selection, medians)
            X_test = np.where(np.isfinite(X_test), X_test, medians)
        selection_pred = recursive_predict([model], model_frame.iloc[:raw_cut], selection_cut, target, features, medians, feature_spec, np.mean) if is_forecast else model.predict(X_selection)
        pred = refit_for_final([model], model_frame, raw_cut if is_forecast else None, target, is_forecast, lambda values: values[0], X_train, y_train, X_selection, y_selection, X_test)
    elif route_id == 'extra_trees_feature_search':
        configs = []
        for depth in (4, 8, None):
            for leaf in (1, 3):
                configs.append((f'extratrees_depth_{{depth}}_leaf_{{leaf}}', lambda depth=depth, leaf=leaf: ExtraTreesRegressor(n_estimators=120, max_depth=depth, min_samples_leaf=leaf, random_state=42)))
        model, selected_config, selected_medians = select_route_model(configs, X_train, y_train, is_forecast, train_frame if is_forecast else None, target)
        if not is_forecast:
            medians = selected_medians
            X_train = np.where(np.isfinite(X_train), X_train, medians)
            X_selection = np.where(np.isfinite(X_selection), X_selection, medians)
            X_test = np.where(np.isfinite(X_test), X_test, medians)
        selection_pred = recursive_predict([model], model_frame.iloc[:raw_cut], selection_cut, target, features, medians, feature_spec, np.mean) if is_forecast else model.predict(X_selection)
        pred = refit_for_final([model], model_frame, raw_cut if is_forecast else None, target, is_forecast, lambda values: values[0], X_train, y_train, X_selection, y_selection, X_test)
    elif route_id == 'model_family_ensemble':
        ridge = Ridge(alpha=1.0)
        forest = RandomForestRegressor(n_estimators=80, max_depth=5, min_samples_leaf=2, random_state=42)
        boost = GradientBoostingRegressor(n_estimators=100, learning_rate=0.06, max_depth=2, random_state=42)
        if not is_forecast:
            X_train, X_selection, medians = impute_from_train(X_train, X_selection)
            X_test = np.where(np.isfinite(X_test), X_test, medians)
        for model in (ridge, forest, boost):
            model.fit(X_train, y_train)
        selection_pred = recursive_predict([ridge, forest, boost], model_frame.iloc[:raw_cut], selection_cut, target, features, medians, feature_spec, np.mean) if is_forecast else (ridge.predict(X_selection) + forest.predict(X_selection) + boost.predict(X_selection)) / 3.0
        pred = refit_for_final([ridge, forest, boost], model_frame, raw_cut if is_forecast else None, target, is_forecast, lambda values: np.mean(values, axis=0), X_train, y_train, X_selection, y_selection, X_test)
        selected_config = 'ridge_rf_gbr_average'
    elif route_id in {'residual_hybrid_forecast', 'innovation_hybrid_route', 'corpus_inspired_hybrid'}:
        base = Ridge(alpha=1.0)
        if not is_forecast:
            X_train, X_selection, medians = impute_from_train(X_train, X_selection)
            X_test = np.where(np.isfinite(X_test), X_test, medians)
        base.fit(X_train, y_train)
        residual = y_train - base.predict(X_train)
        correction = GradientBoostingRegressor(n_estimators=80, learning_rate=0.05, max_depth=2, random_state=42)
        correction.fit(X_train, residual)
        if is_forecast:
            class ResidualModel:
                def predict(self, X):
                    return base.predict(X) + correction.predict(X)
            selection_pred = recursive_predict([ResidualModel()], model_frame.iloc[:raw_cut], selection_cut, target, features, medians, feature_spec, np.mean)
            development = model_frame.iloc[:raw_cut].copy()
            final_X, final_y, final_features, _, final_spec = engineer_features(development, target, True)
            final_X, final_medians = impute_from_train(final_X)
            final_base = clone(base).fit(final_X, final_y)
            final_correction = clone(correction).fit(final_X, final_y - final_base.predict(final_X))
            class FinalResidualModel:
                def predict(self, X):
                    return final_base.predict(X) + final_correction.predict(X)
            pred = recursive_predict([FinalResidualModel()], model_frame, raw_cut, target, final_features, final_medians, final_spec, np.mean)
        else:
            selection_pred = base.predict(X_selection) + correction.predict(X_selection)
            development_X = np.vstack([X_train, X_selection])
            development_y = np.concatenate([y_train, y_selection])
            final_base = clone(base).fit(development_X, development_y)
            final_correction = clone(correction).fit(development_X, development_y - final_base.predict(development_X))
            pred = final_base.predict(X_test) + final_correction.predict(X_test)
        selected_config = 'ridge_plus_gbr_residual'
    elif route_id == 'robust_feature_ablation':
        selection_pred = np.repeat(np.median(y_train), len(y_selection))
        pred = np.repeat(np.median(np.concatenate([y_train, y_selection])), len(y_test))
        selected_config = 'median_ablation'
    else:
        raise RuntimeError(f'unsupported_route_template: {{route_id}}')
    metrics = prefixed_metrics('selection', y_selection, selection_pred)
    metrics.update(prefixed_metrics('final_test', y_test, pred))
    metrics.update({{'MAE': metrics['selection_MAE'], 'RMSE': metrics['selection_RMSE'], 'WAPE': metrics['selection_WAPE'], 'rows_used': int(len(model_frame)), 'features_used': int(len(features)), 'target': str(target), 'validation_split': 'train_selection_final_test_recursive' if is_forecast else validation_protocol, 'selection_size': int(len(y_selection)), 'test_size': int(len(y_test)), 'forecast_horizon': int(contract.get('forecast_horizon') or 1) if is_forecast else None, 'selected_config': selected_config}})
    evidence = {{'observed': [float(v) for v in y_selection], 'predicted': [float(v) for v in selection_pred], 'sample_rows': [int(v) for v in selection_row_ids], 'final_test_observed': [float(v) for v in y_test], 'final_test_predicted': [float(v) for v in pred], 'final_test_sample_rows': [int(v) for v in test_row_ids], 'target_column': str(target)}}
    if not is_forecast and contract.get('group_columns'):
        group_columns = contract['group_columns']
        groups_by_row = model_frame.set_index('_source_row')[group_columns].astype(str).agg('||'.join, axis=1)
        evidence['selection_groups'] = groups_by_row.loc[selection_row_ids].tolist()
        evidence['final_test_groups'] = groups_by_row.loc[test_row_ids].tolist()
        all_groups = groups_by_row
        evidence['train_groups'] = sorted(set(all_groups) - set(evidence['selection_groups']) - set(evidence['final_test_groups']))
    return metrics, evidence


def normalize_indicators(frame, directions):
    oriented = frame.copy()
    for column in oriented.columns:
        if directions.get(str(column)) == 'lower_better':
            oriented[column] = -oriented[column]
    scale = (oriented.max(axis=0) - oriented.min(axis=0)).replace(0, 1)
    return (oriented - oriented.min(axis=0)) / scale


def ranking_scores(route_id, role, normalized):
    weights = None
    if role == 'baseline':
        scores = normalized.mean(axis=1)
        weights = np.repeat(1 / normalized.shape[1], normalized.shape[1])
    elif route_id == 'entropy_weight_ranking':
        shifted = normalized + 1e-9
        prob = shifted / shifted.sum(axis=0).replace(0, 1)
        entropy = -(prob * np.log(prob + 1e-12)).sum(axis=0) / np.log(max(len(prob), 2))
        weights = np.asarray((1 - entropy) / max(float((1 - entropy).sum()), 1e-12))
        scores = shifted.dot(weights)
    elif route_id == 'topsis_ranking':
        norm = normalized / np.sqrt((normalized ** 2).sum(axis=0)).replace(0, 1)
        pos = norm.max(axis=0)
        neg = norm.min(axis=0)
        d_pos = np.sqrt(((norm - pos) ** 2).sum(axis=1))
        d_neg = np.sqrt(((norm - neg) ** 2).sum(axis=1))
        scores = d_neg / (d_pos + d_neg + 1e-12)
    elif route_id == 'pca_factor_ranking':
        centered = normalized - normalized.mean(axis=0)
        _, _, vt = np.linalg.svd(centered.to_numpy(dtype=float), full_matrices=False)
        weights = vt[0]
        if weights.sum() < 0:
            weights = -weights
        scores = centered.to_numpy(dtype=float).dot(weights)
    elif route_id == 'stability_weighted_ranking':
        std = normalized.std(axis=0).replace(0, np.nan)
        weights = np.asarray((1 / std).replace([np.inf, -np.inf], np.nan).fillna(0))
        if float(weights.sum()) <= 1e-12:
            weights = np.ones(normalized.shape[1])
        weights = weights / weights.sum()
        scores = normalized.dot(weights)
    else:
        raise RuntimeError(f'unsupported_evaluation_route: {{route_id}}')
    return np.asarray(scores, dtype=float), None if weights is None else np.asarray(weights, dtype=float)


def run_evaluation_route(route_id, role, frame):
    numeric = frame.select_dtypes(include=[np.number]).dropna(axis=1, how='all').dropna(axis=0, how='all')
    if numeric.shape[0] < 5 or numeric.shape[1] < 2:
        raise RuntimeError('not enough numeric matrix for evaluation route')
    directions = (PROFILE.get('contract') or {{}}).get('indicator_directions') or {{}}
    selected = [column for column in numeric.columns if column in directions]
    if not selected:
        raise RuntimeError('contract_needs_input: evaluation requires indicator_directions')
    filled = numeric[selected].fillna(numeric[selected].median(numeric_only=True)).fillna(0)
    normalized = normalize_indicators(filled, directions)
    scores, weights = ranking_scores(route_id, role, normalized)
    baseline = normalized.mean(axis=1)
    spearman = spearmanr(scores, baseline).correlation
    kendall = kendalltau(scores, baseline).correlation
    top_n = max(1, min(5, len(scores) // 5 or 1))
    top_scores = set(np.argsort(np.asarray(scores))[-top_n:])
    top_base = set(np.argsort(np.asarray(baseline))[-top_n:])
    rng = np.random.default_rng(42)
    column_scale = filled.std(axis=0).replace(0, 1).fillna(1).to_numpy(dtype=float)
    perturbation_scores = []
    stability_values = []
    for _ in range(30):
        perturbed = filled + rng.normal(0, 0.02, size=filled.shape) * column_scale
        perturbed_normalized = normalize_indicators(perturbed, directions)
        perturbed_scores, _ = ranking_scores(route_id, role, perturbed_normalized)
        perturbation_scores.append([float(v) for v in perturbed_scores])
        correlation = spearmanr(scores, perturbed_scores).correlation
        stability_values.append(float(0 if np.isnan(correlation) else correlation))
    stability = float(np.mean(stability_values)) if stability_values else 0.0
    metrics = {{'ranking_stability': round(stability, 6), 'ranking_stability_std': round(float(np.std(stability_values)), 6), 'agreement_with_equal_weight': round(float(0 if np.isnan(spearman) else spearman), 6), 'kendall_with_equal_weight': round(float(0 if np.isnan(kendall) else kendall), 6), 'top_k_overlap_with_equal_weight': round(len(top_scores & top_base) / top_n, 6), 'rows_used': int(len(filled)), 'features_used': int(filled.shape[1]), 'perturbation_repeats': len(perturbation_scores)}}
    evidence = {{'scores': [float(v) for v in scores], 'indicator_directions': directions, 'indicator_columns': [str(v) for v in filled.columns], 'indicator_values': [[float(v) for v in row] for row in filled.to_numpy(dtype=float)], 'sample_rows': [int(v) for v in filled.index], 'weights': None if weights is None else [float(v) for v in weights], 'perturbation_seed': 42, 'perturbation_relative_sigma': 0.02, 'perturbation_scores': perturbation_scores}}
    return metrics, evidence


def run():
    started = time.time()
    if not SUPPORTED:
        raise RuntimeError(f"unsupported_route_family: {{ROUTE['family']}}")
    data_path, raw = load_best_table()
    input_hash_at_load = file_sha256(data_path)
    frame = raw
    family = ROUTE['family']
    role = ROUTE['role']
    route_id = ROUTE['route_id']
    if family == 'evaluation':
        metrics, evidence = run_evaluation_route(route_id, role, frame)
    elif family in {'baseline', 'forecasting', 'tabular', 'diagnostic', 'hybrid'}:
        metrics, evidence = run_regression_route(route_id, role, family, frame)
    else:
        raise RuntimeError(f'unsupported_route_family: {{family}}')
    metrics['runtime_seconds'] = round(time.time() - started, 6)
    return {{
        'problem_id': PROFILE['problem_id'],
        'route_id': route_id,
        'role': role,
        'status': 'executed',
        'data_path': str(data_path),
        'input_hash_at_load': input_hash_at_load,
        'metrics': metrics,
        'evidence': evidence,
        'contract': PROFILE.get('contract', {{}}),
        'route': ROUTE,
        'environment': {{name: package_version(name) for name in ('numpy', 'pandas', 'scipy', 'scikit-learn')}},
        'limitations': [
            'V44 generic adapter is a first-pass executable evidence screen, not a final domain-specific model.',
            'Routes without an implemented template are failed instead of being silently mapped to Ridge.',
        ],
        'recommendation': 'candidate evidence generated by executable V44 adapter'
    }}


if __name__ == '__main__':
    print(json.dumps(run(), ensure_ascii=False, indent=2))
'''
