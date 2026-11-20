"""Sparse-history radius calibration and horizon diagnostics for Q2.

Inputs are already causally generated features and normalized Euclidean errors.
This module never reads recordings, generates hidden per-frame online feedback,
fits a spatial scaler, or promises exchangeability-based simultaneous coverage.
AI-assisted implementation: OpenAI Codex, 2026-11-20; tests verify interfaces.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import lightgbm as lgb


LEVELS = (.5, .8, .9, .95)
REQUIRED_META = ('environment', 'episode', 'window_id', 'hand')
EXPERIMENT_EXTRAS = ('scenario', 'horizon_s', 'cap', 'delay_s', 'phase', 'policy', 'protocol')


def _levels(levels):
    values = np.asarray(levels, dtype=float)
    if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
            or np.any(values <= 0) or np.any(values >= 1) or np.any(np.diff(values) <= 0)):
        raise ValueError('Coverage levels must be strictly increasing in (0,1)')
    return values


def _metadata(meta, n=None, experiment_columns=None):
    meta = pd.DataFrame(meta).reset_index(drop=True)
    if not len(meta) or (n is not None and len(meta) != n):
        raise ValueError('Metadata must have one row per nonempty input sample')
    missing = set(REQUIRED_META)-set(meta.columns)
    if missing:
        raise ValueError(f'Missing metadata columns: {sorted(missing)}')
    experiment_columns = tuple(experiment_columns or
                               ('window_id',)+tuple(c for c in EXPERIMENT_EXTRAS if c in meta))
    if 'window_id' not in experiment_columns:
        raise ValueError('Experiment key must include window_id')
    keys = list(dict.fromkeys(['environment', 'episode', 'hand', *experiment_columns]))
    if any(c not in meta for c in keys) or meta[keys].isna().any().any():
        raise ValueError('Metadata grouping keys must exist and contain no missing values')
    return meta, experiment_columns


def hierarchical_weights(meta, experiment_columns=None):
    """Equal environment -> episode -> window experiment -> hand -> time rows.

    Request scenarios/phases are part of each experiment key. For complete
    two-hand windows this is equal time/hand weight inside every window. A key
    with more rows, or an environment with longer records, cannot dominate.
    """
    meta, experiment_columns = _metadata(meta, experiment_columns=experiment_columns)
    exp_keys = list(dict.fromkeys(['environment', 'episode', *experiment_columns]))
    grouping = meta[list(dict.fromkeys(exp_keys+['hand']))].copy()
    grouping['_experiment'] = pd.factorize(pd.MultiIndex.from_frame(grouping[exp_keys]))[0]
    environments = grouping.environment.nunique()
    episodes = grouping.groupby('environment', sort=False).episode.transform('nunique').to_numpy()
    experiments = grouping.groupby(['environment', 'episode'], sort=False)._experiment.transform('nunique').to_numpy()
    hands = grouping.groupby('_experiment', sort=False).hand.transform('nunique').to_numpy()
    rows = grouping.groupby(['_experiment', 'hand'], sort=False).hand.transform('size').to_numpy()
    weight = 1./(environments*episodes*experiments*hands*rows)
    if not np.isclose(weight.sum(), 1., atol=1e-10):
        raise ValueError('Hierarchical weights do not sum to one')
    return weight


def weighted_quantile(values, level, weights):
    """Left-continuous inverse empirical CDF; no finite-sample CP claim."""
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    if (values.ndim != 1 or len(values) == 0 or weights.shape != values.shape
            or not np.isfinite(values).all() or not np.isfinite(weights).all()
            or np.any(weights < 0) or weights.sum() <= 0 or not 0 <= level <= 1):
        raise ValueError('Invalid weighted quantile arguments')
    positive = weights > 0
    values, weights = values[positive], weights[positive]
    order = np.argsort(values, kind='stable')
    cdf = np.cumsum(weights[order])/weights.sum()
    index = min(np.searchsorted(cdf, level, side='left'), len(order)-1)
    return float(values[order[index]])


def _inputs(features, errors=None, n_features=None):
    x = np.asarray(features, float)
    if x.ndim != 2 or not len(x) or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError('Features must be a finite nonempty two-dimensional matrix')
    if n_features is not None and x.shape[1] != n_features:
        raise ValueError('Feature count differs from fitted model')
    if errors is None:
        return x
    e = np.asarray(errors, float)
    if e.shape != (len(x),) or not np.isfinite(e).all() or np.any(e < 0):
        raise ValueError('Errors must be finite nonnegative normalized Euclidean distances')
    return x, e


class RadiusCalibrator:
    """Conditional quantile or weighted static radius with independent correction.

    `fit` sees only its fitting environments. `calibrate` requires disjoint
    environments and adds a weighted residual quantile for each level. This is
    an empirical calibration adjustment, not a distribution-free guarantee.
    Upstream predictors/scalers must also exclude held environments: these
    arrays alone cannot establish that provenance.
    """
    def __init__(self, method='conditional', levels=LEVELS, parameters=None,
                 experiment_columns=None):
        if method not in ('conditional', 'static'):
            raise ValueError('method must be conditional or static')
        self.method = method
        self.levels = _levels(levels)
        self.parameters = dict(parameters or {})
        if set(self.parameters) & {'objective', 'alpha', 'device', 'device_type'}:
            raise ValueError('Objective, quantile level and CPU device are fixed by this module')
        self.experiment_columns = experiment_columns
        self.models = []
        self.correction = np.zeros(len(self.levels))
        self.fitted = False

    def fit(self, features, errors, meta):
        x, e = _inputs(features, errors)
        meta, columns = _metadata(meta, len(x), self.experiment_columns)
        weights = hierarchical_weights(meta, columns)
        self.experiment_columns = columns
        self.n_features = x.shape[1]
        self.feature_names = list(features.columns) if isinstance(features, pd.DataFrame) else None
        self.fit_environments = sorted(meta.environment.astype(str).unique())
        self.static = np.array([weighted_quantile(e, q, weights) for q in self.levels])
        self.models = []
        self.correction = np.zeros(len(self.levels))
        self.calibration_info = None
        self.fit_info = dict(rows=len(x), environments=len(self.fit_environments),
                             episodes=int(meta[['environment', 'episode']].drop_duplicates().shape[0]),
                             experiment_columns=list(columns),
                             weights='equal environment/episode/window-experiment/hand/time',
                             residual_provenance='supplied by caller; upstream fold isolation must be audited')
        if self.method == 'conditional':
            params = dict(n_estimators=300, num_leaves=15, min_child_samples=100,
                          learning_rate=.03, reg_lambda=10., n_jobs=2,
                          random_state=20261120, deterministic=True,
                          force_col_wise=True, verbosity=-1)
            params.update(self.parameters)
            for q in self.levels:
                model = lgb.LGBMRegressor(objective='quantile', alpha=float(q),
                                          device_type='cpu', **params)
                model.fit(x, np.log1p(e), sample_weight=weights*len(weights))
                self.models.append(model.booster_)
        self.fitted = True
        return self

    def predict(self, features, apply_correction=True):
        if not self.fitted:
            raise RuntimeError('Fit the calibrator before prediction')
        x = _inputs(features, n_features=self.n_features)
        if (self.feature_names is not None and isinstance(features, pd.DataFrame)
                and list(features.columns) != self.feature_names):
            raise ValueError('Feature names/order differ from fitted model')
        if self.method == 'static':
            radii = np.tile(self.static, (len(x), 1))
        else:
            predictions = np.column_stack([m.predict(x, num_threads=self.parameters.get('n_jobs', 2))
                                           for m in self.models])
            with np.errstate(over='raise', invalid='raise'):
                radii = np.expm1(predictions)
        # Level ordering never asserts monotonicity over physical time.
        radii = np.maximum.accumulate(np.maximum(radii, 0.), axis=1)
        if apply_correction:
            radii = np.maximum.accumulate(np.maximum(radii+self.correction, 0.), axis=1)
        if not np.isfinite(radii).all():
            raise ValueError('Nonfinite radius prediction')
        return radii

    def calibrate(self, features, errors, meta):
        if not self.fitted:
            raise RuntimeError('Fit before independent calibration')
        x, e = _inputs(features, errors, self.n_features)
        meta, columns = _metadata(meta, len(x), self.experiment_columns)
        environments = sorted(meta.environment.astype(str).unique())
        if set(environments) & set(self.fit_environments):
            raise ValueError('Independent calibration environments overlap fitting environments')
        base = self.predict(features, apply_correction=False)
        weights = hierarchical_weights(meta, columns)
        self.correction = np.array([weighted_quantile(e-base[:, j], q, weights)
                                    for j, q in enumerate(self.levels)])
        self.calibration_info = dict(rows=len(x), environments=environments,
                                      correction=self.correction.tolist(),
                                      interpretation='independent-environment empirical residual quantile; no distribution-free guarantee')
        return self

    def save(self, directory):
        if not self.fitted:
            raise RuntimeError('Cannot save an unfitted calibrator')
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        for j, model in enumerate(self.models):
            model.save_model(str(directory/f'quantile_{j}.txt'))
        record = dict(method=self.method, levels=self.levels.tolist(), parameters=self.parameters,
                      experiment_columns=list(self.experiment_columns), n_features=self.n_features,
                      feature_names=self.feature_names, fit_environments=self.fit_environments,
                      static=self.static.tolist(), correction=self.correction.tolist(),
                      fit_info=self.fit_info, calibration_info=self.calibration_info,
                      lightgbm=lgb.__version__, fitted=True)
        (directory/'calibrator.json').write_text(json.dumps(record, indent=2)+'\n')

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        record = json.loads((directory/'calibrator.json').read_text())
        result = cls(record['method'], record['levels'], record['parameters'], record['experiment_columns'])
        for field in ('n_features', 'feature_names', 'fit_environments', 'fit_info', 'calibration_info'):
            setattr(result, field, record[field])
        result.static = np.array(record['static'])
        result.correction = np.array(record['correction'])
        if result.method == 'conditional':
            result.models = [lgb.Booster(model_file=str(directory/f'quantile_{j}.txt'))
                             for j in range(len(result.levels))]
        result.fitted = True
        return result


class AgeBinnedRadius:
    """Training-weighted static quantiles within fixed observation-age bins.

    Rare bins fall back to the global static quantile. Thresholds are fixed
    before evaluation, not learned from held-out trajectories.
    """
    def __init__(self, levels=LEVELS, upper_bounds=(1e-9, .05, .1, .2, .5, 1., 2., np.inf),
                 minimum_environments=3, experiment_columns=None):
        self.levels = _levels(levels)
        self.upper_bounds = np.asarray(upper_bounds, float)
        if (self.upper_bounds.ndim != 1 or not len(self.upper_bounds)
                or np.any(np.isnan(self.upper_bounds)) or np.any(self.upper_bounds <= 0)
                or np.any(np.diff(self.upper_bounds) <= 0) or not np.isposinf(self.upper_bounds[-1])):
            raise ValueError('Age bins must increase and terminate at positive infinity')
        self.minimum_environments = int(minimum_environments)
        if self.minimum_environments < 1:
            raise ValueError('At least one fitting environment required')
        self.experiment_columns = experiment_columns

    def fit(self, ages, errors, meta):
        ages, errors = np.asarray(ages, float), np.asarray(errors, float)
        _inputs(ages[:, None], errors)
        if np.any(ages < 0):
            raise ValueError('Negative observation age')
        meta, columns = _metadata(meta, len(ages), self.experiment_columns)
        self.experiment_columns = columns
        w = hierarchical_weights(meta, columns)
        self.global_radius = np.array([weighted_quantile(errors, q, w) for q in self.levels])
        bins = np.searchsorted(self.upper_bounds, ages, side='left')
        self.radii = np.tile(self.global_radius, (len(self.upper_bounds), 1))
        self.bin_info = []
        for i, bound in enumerate(self.upper_bounds):
            take = bins == i
            count = meta.loc[take].environment.nunique()
            if count >= self.minimum_environments:
                weights = hierarchical_weights(meta.loc[take], columns)
                self.radii[i] = [weighted_quantile(errors[take], q, weights) for q in self.levels]
            self.bin_info.append(dict(bin=i, upper_s=float(bound) if np.isfinite(bound) else None,
                                     rows=int(take.sum()), environments=int(count),
                                     global_fallback=bool(count < self.minimum_environments)))
        self.radii = np.maximum.accumulate(self.radii, axis=1)
        self.fit_environments = sorted(meta.environment.astype(str).unique())
        return self

    def predict(self, ages):
        if not hasattr(self, 'radii'):
            raise RuntimeError('Fit age-binned radii first')
        ages = np.asarray(ages, float)
        if ages.ndim != 1 or not np.isfinite(ages).all() or np.any(ages < 0):
            raise ValueError('Ages must be finite and nonnegative')
        return self.radii[np.searchsorted(self.upper_bounds, ages, side='left')].copy()

    def save(self, directory):
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        record = dict(levels=self.levels.tolist(), upper_bounds=[float(b) if np.isfinite(b) else None for b in self.upper_bounds],
                      radii=self.radii.tolist(), global_radius=self.global_radius.tolist(),
                      bin_info=self.bin_info, fit_environments=self.fit_environments,
                      minimum_environments=self.minimum_environments,
                      experiment_columns=list(self.experiment_columns))
        (directory/'age_binned.json').write_text(json.dumps(record, indent=2)+'\n')


def oof_predictions(features, errors, meta, folds, method='conditional', levels=LEVELS,
                    parameters=None, experiment_columns=None):
    """Whole-environment held-out radii, with optional inner calibration groups.

    Folds use training_environments and held_environments. An optional
    calibration_environments must be a subset of training_environments and is
    removed from fitting. Every provided environment must be held exactly once.
    The caller must supply correctly nested point errors/features for the claim
    it makes; this function does not reconstruct point-model training ancestry.
    """
    x, e = _inputs(features, errors)
    meta, columns = _metadata(meta, len(x), experiment_columns)
    level_array = _levels(levels)
    env = meta.environment.astype(str).to_numpy()
    available = set(env)
    result = np.full((len(x), len(level_array)), np.nan)
    assignment = np.full(len(x), -1, int)
    records = []
    for number, spec in enumerate(folds):
        training = set(map(str, spec['training_environments']))
        held = set(map(str, spec['held_environments']))
        calibration = set(map(str, spec.get('calibration_environments', [])))
        if (not training or not held or training & held or training | held != available
                or not calibration <= training or not training-calibration):
            raise ValueError('Invalid disjoint environment fold specification')
        take_fit = np.isin(env, sorted(training-calibration))
        take_held = np.isin(env, sorted(held))
        if np.any(assignment[take_held] != -1):
            raise ValueError('An environment is held out more than once')
        model = RadiusCalibrator(method, level_array, parameters, columns)
        model.fit(x[take_fit], e[take_fit], meta.loc[take_fit])
        if calibration:
            take_cal = np.isin(env, sorted(calibration))
            model.calibrate(x[take_cal], e[take_cal], meta.loc[take_cal])
        result[take_held] = model.predict(x[take_held])
        assignment[take_held] = number
        records.append(dict(fold=spec.get('fold', number), fitting_environments=sorted(training-calibration),
                            calibration_environments=sorted(calibration), held_environments=sorted(held),
                            fit_rows=int(take_fit.sum()), held_rows=int(take_held.sum())))
    if np.any(assignment < 0) or not np.isfinite(result).all():
        raise ValueError('Fold specification does not cover all rows exactly once')
    return dict(radii=result, fold_assignment=assignment, folds=records)


def evaluate_radii(meta, errors, radii, levels=LEVELS, group_by=(), experiment_columns=None):
    """Pointwise coverage/size/pinball; optionally separate current-unobserved rows."""
    e, r, levels = np.asarray(errors, float), np.asarray(radii, float), _levels(levels)
    meta, columns = _metadata(meta, len(e), experiment_columns)
    if (e.shape != (len(meta),) or r.shape != (len(meta), len(levels))
            or not np.isfinite(e).all() or not np.isfinite(r).all()
            or np.any(e < 0) or np.any(r < 0)):
        raise ValueError('Invalid normalized errors or radii')
    if any(c not in meta for c in group_by):
        raise ValueError('Requested evaluation group not in metadata')
    groups = meta.groupby(list(group_by), dropna=False, observed=True, sort=True).indices if group_by else {(): np.arange(len(meta))}
    rows = []
    for key, indices in groups.items():
        key = key if isinstance(key, tuple) else (key,)
        subsets = {'all': np.asarray(indices)}
        if 'current_direct' in meta:
            direct = meta.iloc[indices].current_direct
            if not direct.isin([True, False, 0, 1]).all():
                raise ValueError('current_direct must contain boolean/0/1 values')
            subsets['not_current_direct'] = np.asarray(indices)[~direct.to_numpy(bool)]
        for subset, selected in subsets.items():
            base = dict(zip(group_by, key)); base['subset'] = subset
            base.update(rows=len(selected), environments=meta.iloc[selected].environment.nunique())
            weights = hierarchical_weights(meta.iloc[selected], columns) if len(selected) else None
            for j, q in enumerate(levels):
                difference = e[selected]-r[selected, j]
                rows.append(dict(**base, nominal=float(q),
                                 coverage=float(np.dot(weights, difference <= 0)) if len(selected) else np.nan,
                                 mean_radius=float(np.dot(weights, r[selected, j])) if len(selected) else np.nan,
                                 pinball=float(np.dot(weights, np.maximum(q*difference, (q-1)*difference))) if len(selected) else np.nan))
    return pd.DataFrame(rows)


def suggest_horizon(horizons, radii, tolerances=(.5, 1., 2.)):
    """Longest prefix whose time-envelope radius is within each tolerance.

    Radii have shape (..., H) for one chosen nominal level. Output shape is
    (..., number_of_tolerances). Zero means no supported positive horizon.
    Inf radii stop suggestions; NaN radii are rejected. No target errors enter.
    """
    h, r, tolerances = np.asarray(horizons, float), np.asarray(radii, float), np.asarray(tolerances, float)
    if (h.ndim != 1 or not len(h) or not np.isfinite(h).all() or np.any(h <= 0)
            or np.any(np.diff(h) <= 0) or r.ndim < 1 or r.shape[-1] != len(h)
            or np.isnan(r).any() or np.any(r < 0) or tolerances.ndim != 1
            or not len(tolerances) or not np.isfinite(tolerances).all() or np.any(tolerances <= 0)):
        raise ValueError('Invalid horizon grid, radii or positive tolerances')
    envelope = np.maximum.accumulate(r, axis=-1)
    accepted = envelope[..., :, None] <= tolerances
    return np.max(np.where(accepted, h.reshape((1,)*(r.ndim-1)+(len(h), 1)), 0.), axis=-2)


def evaluate_paths(meta, errors, radii, valid=None, experiment_columns=None):
    """Separately evaluate pointwise coverage and entire observed path coverage.

    One row is one hand path, with shape (paths,times). Joint window coverage
    requires every supplied hand/time to be valid; partial or empty paths are
    not counted as fully covered. No independence/simultaneous guarantee follows.
    """
    e, r = np.asarray(errors, float), np.asarray(radii, float)
    if e.ndim != 2 or e.shape != r.shape or e.shape[1] == 0:
        raise ValueError('Paths must be matching nonempty (paths,times) matrices')
    meta, columns = _metadata(meta, len(e), experiment_columns)
    valid = np.ones(e.shape, bool) if valid is None else np.asarray(valid, bool)
    if valid.shape != e.shape or not np.isfinite(e[valid]).all() or not np.isfinite(r[valid]).all() or np.any(e[valid] < 0) or np.any(r[valid] < 0):
        raise ValueError('Qualified path entries must have finite nonnegative errors/radii')
    keys = list(dict.fromkeys(['environment', 'episode', *columns]))
    rows = []
    for key, indices in meta.groupby(keys, sort=False, observed=True, dropna=False).indices.items():
        key = key if isinstance(key, tuple) else (key,)
        v = valid[indices]
        diff = e[indices]-r[indices]
        count = int(v.sum())
        hand_counts = v.sum(axis=1)
        nonempty_hands = hand_counts > 0
        hand_coverage = np.divide(((diff <= 0) & v).sum(axis=1), hand_counts,
                                  out=np.zeros(len(indices), float), where=nonempty_hands)
        rows.append(dict(zip(keys, key)) | dict(
            hand=0, hand_paths=len(indices), qualified_points=count,
            complete_reference=bool(v.all()),
            pointwise_coverage=float(hand_coverage[nonempty_hands].mean()) if count else np.nan,
            entire_path_covered=bool((diff <= 0).all()) if v.all() else np.nan,
            max_error=float(e[indices][v].max()) if count else np.nan))
    per_window = pd.DataFrame(rows)
    weight = hierarchical_weights(per_window, columns)
    complete = per_window.complete_reference.to_numpy(bool)
    nonempty = per_window.qualified_points.to_numpy() > 0
    summary = dict(windows=len(per_window), complete_reference_windows=int(complete.sum()),
                   complete_reference_weight=float(weight[complete].sum()),
                   pointwise_coverage=float(np.average(per_window.loc[nonempty, 'pointwise_coverage'], weights=weight[nonempty])) if nonempty.any() else np.nan,
                   entire_path_coverage=float(np.average(per_window.loc[complete, 'entire_path_covered'].astype(float), weights=weight[complete])) if complete.any() else np.nan,
                   interpretation='empirical path diagnostic; pointwise nominal coverage is not simultaneous coverage')
    return dict(per_window=per_window, summary=summary)


def evaluate_paths_long(meta, errors, radii, levels=LEVELS, experiment_columns=None):
    """Whole-window diagnostics from long rows without padding irregular grids.

    Metadata must contain unique time_s/hand within each experiment, plus
    expected_hand_times supplied by the fixed window ledger. Counts are checked
    independently; an omitted sample must not silently improve path coverage.
    """
    e, r, levels = np.asarray(errors, float), np.asarray(radii, float), _levels(levels)
    meta, columns = _metadata(meta, len(e), experiment_columns)
    if (e.shape != (len(meta),) or r.shape != (len(meta), len(levels))
            or not np.isfinite(e).all() or not np.isfinite(r).all()
            or np.any(e < 0) or np.any(r < 0)):
        raise ValueError('Invalid long-form errors/radii')
    keys = list(dict.fromkeys(['environment', 'episode',
                              *[c for c in ('split', 'dataset') if c in meta], *columns]))
    if not {'time_s', 'expected_hand_times'} <= set(meta):
        raise ValueError('Long-form path evaluation needs times and independent expected counts')
    if meta.duplicated(keys+['hand', 'time_s']).any():
        raise ValueError('Repeated hand/time in a path experiment')
    data = meta.copy()
    data['_error'] = e
    for j in range(len(levels)):
        data[f'_covered_{j}'] = e <= r[:, j]
        data[f'_radius_{j}'] = r[:, j]
    groups = data.groupby(keys, sort=False, observed=True, dropna=False)
    aggregate = groups.agg(hand_times=('_error', 'size'), max_error=('_error', 'max'),
                           expected_hand_times=('expected_hand_times', 'first'),
                           expected_count_variants=('expected_hand_times', 'nunique'))
    if (aggregate.expected_count_variants != 1).any() or (aggregate.expected_hand_times <= 0).any():
        raise ValueError('Expected count must be positive and constant within each experiment')
    aggregate['complete_reference'] = aggregate.hand_times == aggregate.expected_hand_times
    per_window = []
    for j, q in enumerate(levels):
        f = aggregate.join(groups[f'_covered_{j}'].agg(pointwise_coverage='mean', entire_path_covered='all'))
        f = f.join(groups[f'_radius_{j}'].mean().rename('mean_radius'))
        f['entire_path_covered'] = f.entire_path_covered.astype(float)
        f.loc[~f.complete_reference, 'entire_path_covered'] = np.nan
        f = f.reset_index(); f['nominal'] = q; f['hand'] = 0
        per_window.append(f)
    return pd.concat(per_window, ignore_index=True)


def evaluate_suggestions(meta, horizons, errors, suggestions, tolerances=(.5, 1., 2.), valid=None,
                         experiment_columns=None):
    """Audit selected durations against held references, counting zero suggestions.

    Rows are independent window/hand paths. A selected interval is assessable
    only if all its reference grid points are qualified. A zero suggestion is
    an abstention, never an automatically successful prediction.
    """
    h, e, s, tol = map(lambda a: np.asarray(a, float), (horizons, errors, suggestions, tolerances))
    if (h.ndim != 1 or not len(h) or np.any(h <= 0) or np.any(np.diff(h) <= 0)
            or not np.isfinite(h).all() or e.ndim != 2 or e.shape[1] != len(h)
            or tol.ndim != 1 or not len(tol) or not np.isfinite(tol).all() or np.any(tol <= 0)
            or s.shape != (len(e), len(tol)) or not np.isfinite(s).all() or np.any(s < 0)
            or np.any(s > h[-1]+1e-9)):
        raise ValueError('Invalid suggestion evaluation shapes or ranges')
    supported = np.isclose(s[..., None], np.r_[0., h], atol=1e-9, rtol=0).any(axis=-1)
    if not supported.all():
        raise ValueError('Suggested horizons must be zero or an evaluated grid horizon')
    meta, columns = _metadata(meta, len(e), experiment_columns)
    valid = np.ones(e.shape, bool) if valid is None else np.asarray(valid, bool)
    if valid.shape != e.shape or not np.isfinite(e[valid]).all() or np.any(e[valid] < 0):
        raise ValueError('Qualified suggestion references must be finite nonnegative errors')
    weight = hierarchical_weights(meta, columns)
    rows = []
    for j, epsilon in enumerate(tol):
        chosen = h[None, :] <= s[:, j, None]+1e-9
        positive = chosen.any(axis=1)
        complete = positive & np.all(valid | ~chosen, axis=1)
        maximum = np.max(np.where(chosen & valid, e, -np.inf), axis=1)
        covered = maximum <= epsilon
        denom = weight[complete].sum()
        rows.append(dict(tolerance=float(epsilon), positive_suggestions=int(positive.sum()),
                         assessable_suggestions=int(complete.sum()),
                         suggestion_rate=float(weight[positive].sum()),
                         assessable_rate=float(denom),
                         mean_suggested_horizon=float(np.dot(weight, s[:, j])),
                         selected_interval_within_tolerance=float(np.dot(weight[complete], covered[complete])/denom) if denom else np.nan))
    return pd.DataFrame(rows)
