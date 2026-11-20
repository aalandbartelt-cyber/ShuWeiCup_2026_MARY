#!/usr/bin/env python3
"""Fit/evaluate Q2 development radii from caller-supplied per-time residual tables.

No competition data are read implicitly. Tables are CSV/parquet with metadata,
normalized_error and explicit feature columns (default x_*). Upstream residual
and normalization provenance must be supplied/audited by the point-model run.
This development entry point rejects official test rows and all non-train rows
in fitting/calibration tables. A frozen-test runner should load saved models.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from shuwei_c.q2_calibration import (RadiusCalibrator, AgeBinnedRadius, oof_predictions,
                                    evaluate_radii, evaluate_paths_long,
                                    hierarchical_weights, suggest_horizon)


def read_table(path):
    path = Path(path)
    return pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)


def check_partition(frame, role):
    if 'split' not in frame:
        raise ValueError(f'{role} table must identify split to enforce development-only use')
    splits = set(frame.split.astype(str))
    allowed = {'train'} if role in ('fit', 'calibration') else {'train', 'train_oof', 'validation'}
    if not splits <= allowed:
        raise ValueError(f'Unexpected {role} split(s): {sorted(splits)}')


def load_learning_split(root, split, baseline, expected_counts):
    folder = root/split
    manifest = json.loads((folder/'manifest.json').read_text())
    if manifest['split'] != split or manifest.get('test_used') or manifest['sample_stride'] != 1:
        raise ValueError('Probe requires complete non-test development learning data')
    features, errors, frames = [], [], []
    index = manifest['expert_names'].index(baseline)
    for file in sorted(folder.glob('*.parquet')):
        meta = pd.read_parquet(file)
        info = json.loads(file.with_suffix('.json').read_text())
        if info['feature_names'] != manifest['feature_names'] or info['expert_names'] != manifest['expert_names']:
            raise ValueError('Inconsistent feature/expert layout in learning cache')
        with np.load(file.with_suffix('.npz')) as arrays:
            raw = arrays['features']
            if len(raw) != len(meta) or len(raw) != info['rows']:
                raise ValueError('Learning arrays do not align to metadata')
            error = np.linalg.norm(arrays['experts'][:, :, index].astype(float)-arrays['target'].astype(float), axis=1)
            features.append(raw); errors.append(error)
        meta['expected_hand_times'] = meta.window_id.map(expected_counts)
        if meta.expected_hand_times.isna().any():
            raise ValueError('Learning window missing from independent ledger')
        frames.append(meta)
    meta = pd.concat(frames, ignore_index=True)
    if len(meta) != manifest['rows'] or set(meta.split) != {split}:
        raise ValueError('Learning cache split/row count mismatch')
    print(json.dumps(dict(stage='loaded', split=split, rows=len(meta), baseline=baseline)), flush=True)
    return dict(meta=meta, features=np.concatenate(features), errors=np.concatenate(errors),
                feature_names=manifest['feature_names'], manifest=manifest)


def condition_scores(meta, errors, radii, levels, split, method):
    rows = []
    conditions = ['environment', 'horizon_s', 'cap', 'delay_s']
    for dataset in ['ALL']+sorted(meta.dataset.unique()):
        take = np.ones(len(meta), bool) if dataset == 'ALL' else (meta.dataset == dataset).to_numpy()
        f = evaluate_radii(meta.loc[take], errors[take], radii[take], levels=levels, group_by=conditions)
        f['dataset'] = dataset; f['split'] = split; f['method'] = method; rows.append(f)
    return pd.concat(rows, ignore_index=True)


def path_condition_scores(paths, split, method):
    rows = []
    conditions = ['environment', 'horizon_s', 'cap', 'delay_s', 'nominal']
    for dataset in ['ALL']+sorted(paths.dataset.unique()):
        f = paths if dataset == 'ALL' else paths[paths.dataset == dataset]
        for key, group in f.groupby(conditions, sort=True, observed=True):
            w = hierarchical_weights(group)
            complete = group.complete_reference.to_numpy(bool)
            weight = w[complete].sum()
            rows.append(dict(zip(conditions, key)) | dict(
                dataset=dataset, split=split, method=method, windows=len(group),
                complete_windows=int(complete.sum()),
                pointwise_coverage=float(np.dot(w, group.pointwise_coverage)),
                entire_path_coverage=float(np.dot(w[complete], group.loc[complete, 'entire_path_covered'])/weight) if weight else np.nan,
                mean_radius=float(np.dot(w, group.mean_radius))))
    return pd.DataFrame(rows)


def no_request_suggestions(meta, errors, radii, levels):
    """Only cap=0: every future-feature row uses the same origin prefix.

    Returned-observation paths cannot be mined retrospectively for origin-time
    suggested durations. Here both-hand tolerance uses max radius/error at each
    future grid time, and each suggested interval is checked empirically.
    """
    if .9 not in levels:
        return pd.DataFrame()
    level = list(levels).index(.9)
    work = meta.copy(); work['_error'] = errors; work['_radius'] = radii[:, level]
    work = work[work.cap == 0]
    keys = ['environment', 'episode', 'dataset', 'window_id', 'scenario', 'horizon_s', 'origin_s', 'phase']
    rows = []
    for key, group in work.groupby(keys, sort=False, observed=True):
        origin = float(group.origin_s.iloc[0])
        if group.current_direct.any() or group.expected_hand_times.nunique() != 1 or len(group) != group.expected_hand_times.iloc[0]:
            raise ValueError('No-request suggestion needs complete, wholly hidden target paths')
        future = group.groupby('time_s', sort=True).agg(radius=('_radius', 'max'), error=('_error', 'max'), hands=('hand', 'nunique'))
        if not (future.hands == 2).all():
            raise ValueError('Joint suggestions require both hand predictions at every time')
        horizons = future.index.to_numpy()-origin
        suggested = suggest_horizon(horizons, future.radius.to_numpy())
        for epsilon, duration in zip((.5, 1., 2.), suggested):
            chosen = horizons <= duration+1e-9
            rows.append(dict(zip(keys, key)) | dict(hand=0, tolerance=epsilon,
                        suggested_s=float(duration), supported=bool(chosen.any()),
                        interval_within_tolerance=bool((future.error.to_numpy()[chosen] <= epsilon).all()) if chosen.any() else np.nan,
                        maximum_selected_error=float(future.error.to_numpy()[chosen].max()) if chosen.any() else np.nan,
                        interpretation='no-new-observation prefix forecast; empirical two-hand interval assessment'))
    return pd.DataFrame(rows)


def summarize_suggestions(frame, split, method):
    if frame.empty:
        return frame
    rows = []
    conditions = ['environment', 'horizon_s', 'tolerance']
    for dataset in ['ALL']+sorted(frame.dataset.unique()):
        f = frame if dataset == 'ALL' else frame[frame.dataset == dataset]
        for key, group in f.groupby(conditions, sort=True, observed=True):
            weights = hierarchical_weights(group)
            supported = group.supported.to_numpy(bool)
            available = weights[supported].sum()
            rows.append(dict(zip(conditions, key)) | dict(dataset=dataset, split=split, method=method,
                windows=len(group), suggestion_rate=float(available),
                mean_suggested_s=float(np.dot(weights, group.suggested_s)),
                selected_interval_within_tolerance=float(np.dot(weights[supported], group.loc[supported, 'interval_within_tolerance'].astype(float))/available) if available else np.nan))
    return pd.DataFrame(rows)


def run_baseline_probe(args):
    from shuwei_c.q2_features import transform_features
    root = args.learning_data
    protocol = args.protocol_dir
    sources = [Path(__file__), Path('src/shuwei_c/q2_calibration.py'), Path('src/shuwei_c/q2_features.py'),
               protocol/'folds.json', protocol/'training_noise.json', protocol/'development_windows.csv',
               root/'train/manifest.json', root/'validation/manifest.json']
    parameters = dict(n_estimators=150, num_leaves=15, min_child_samples=100,
                      learning_rate=.04, reg_lambda=10., n_jobs=2)
    if args.parameters:
        parameters.update(json.loads(args.parameters.read_text())); sources.append(args.parameters)
    identity = dict(baseline=args.baseline, parameters=parameters, levels=args.levels,
                    source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    args.out.mkdir(parents=True, exist_ok=True)
    identity_path = args.out/'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Existing baseline probe uses different inputs; use a new directory')
    identity_path.write_text(json.dumps(identity, indent=2)+'\n')
    counts = pd.read_csv(protocol/'development_windows.csv').set_index('window_id').target_times*2
    train = load_learning_split(root, 'train', args.baseline, counts)
    validation = load_learning_split(root, 'validation', args.baseline, counts)
    folds = json.loads((protocol/'folds.json').read_text())
    full_noise = json.loads((protocol/'training_noise.json').read_text())
    partitions = [(f'fold{f["fold"]}', 'train_oof', train, f) for f in folds]
    partitions.append(('validation', 'validation', validation,
                       dict(training_environments=sorted(train['meta'].environment.unique()),
                            held_environments=sorted(validation['meta'].environment.unique()), noise=full_noise)))
    started = time.perf_counter(); completed = []
    train_env = train['meta'].environment.to_numpy()
    for name, split, target, spec in partitions:
        directory = args.out/name; directory.mkdir(exist_ok=True)
        marker = directory/'complete.json'
        if marker.exists():
            completed.append(json.loads(marker.read_text())); continue
        fitting = np.isin(train_env, spec['training_environments'])
        held = np.isin(target['meta'].environment.to_numpy(), spec['held_environments'])
        if set(spec['training_environments']) & set(spec['held_environments']):
            raise ValueError('Overlapping baseline calibration fold environments')
        fit_meta, held_meta = train['meta'].loc[fitting].copy(), target['meta'].loc[held].copy()
        scales_fit = fit_meta.dataset.map({d: v['hand_scale'] for d, v in spec['noise'].items()}).to_numpy()
        scales_held = held_meta.dataset.map({d: v['hand_scale'] for d, v in spec['noise'].items()}).to_numpy()
        xfit = transform_features(train['features'][fitting], train['feature_names'], scales_fit)
        xheld = transform_features(target['features'][held], target['feature_names'], scales_held)
        efit, eheld = train['errors'][fitting]/scales_fit, target['errors'][held]/scales_held
        for method in ('static', 'age_binned', 'conditional'):
            print(json.dumps(dict(stage='fit', partition=name, method=method, fit_rows=len(efit), held_rows=len(eheld), elapsed_s=time.perf_counter()-started)), flush=True)
            if method == 'age_binned':
                model = AgeBinnedRadius(levels=args.levels).fit(fit_meta.age_s.to_numpy(), efit, fit_meta)
                radii = model.predict(held_meta.age_s.to_numpy())
            else:
                model = RadiusCalibrator(method, levels=args.levels, parameters=parameters).fit(xfit, efit, fit_meta)
                radii = model.predict(xheld)
            model.save(directory/method)
            output = held_meta.copy(); output['normalized_error'] = eheld; output['error'] = target['errors'][held]
            output['scale'] = scales_held; output['calibration_partition'] = name
            for j, q in enumerate(args.levels):
                output[f'radius_{q:g}'] = radii[:, j]
            output.to_parquet(directory/f'{method}_radii.parquet', index=False)
            scores = condition_scores(held_meta, eheld, radii, args.levels, split, method)
            scores.to_csv(directory/f'{method}_point_metrics.csv', index=False)
            paths = evaluate_paths_long(held_meta, eheld, radii, levels=args.levels)
            paths.to_parquet(directory/f'{method}_paths.parquet', index=False)
            path_condition_scores(paths, split, method).to_csv(directory/f'{method}_path_metrics.csv', index=False)
            suggestions = no_request_suggestions(held_meta, eheld, radii, args.levels)
            suggestions.to_parquet(directory/f'{method}_suggestions.parquet', index=False)
            summarize_suggestions(suggestions, split, method).to_csv(directory/f'{method}_suggestion_metrics.csv', index=False)
            print(json.dumps(dict(stage='evaluated', partition=name, method=method, elapsed_s=time.perf_counter()-started)), flush=True)
        record = dict(partition=name, split=split, fitting_environments=spec['training_environments'],
                      held_environments=spec['held_environments'], noise=spec['noise'],
                      fit_rows=len(efit), held_rows=len(eheld), elapsed_s=time.perf_counter()-started)
        marker.write_text(json.dumps(record, indent=2)+'\n'); completed.append(record)
    for kind, fields in [('point', ['coverage', 'mean_radius', 'pinball']),
                         ('path', ['pointwise_coverage', 'entire_path_coverage', 'mean_radius']),
                         ('suggestion', ['suggestion_rate', 'mean_suggested_s', 'selected_interval_within_tolerance'])]:
        frames = []
        for name, _, _, _ in partitions:
            for method in ('static', 'age_binned', 'conditional'):
                file = args.out/name/f'{method}_{kind}_metrics.csv'
                if file.stat().st_size > 1:
                    frames.append(pd.read_csv(file))
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(args.out/f'{kind}_environment_metrics.csv', index=False)
        keys = [c for c in ('split', 'dataset', 'horizon_s', 'cap', 'delay_s', 'nominal', 'tolerance', 'subset', 'method') if c in combined]
        summary = combined.groupby(keys, observed=True, dropna=False)[fields].mean().reset_index()
        summary.to_csv(args.out/f'{kind}_summary.csv', index=False)
    manifest = dict(**identity, status='baseline sparse-calibration probe; not a selected final Q2 model',
                    partitions=completed, rows_train=len(train['meta']), rows_validation=len(validation['meta']),
                    elapsed_s=time.perf_counter()-started, test_used=False,
                    independent_correction_used=False,
                    fitting='fold-specific hand scales; fixed deterministic motion predictor; train-only radius fit',
                    suggestions='cap=0 only; known origin prefix held fixed; max over two hands; no simultaneous coverage guarantee',
                    limitations=['Validation used only for evaluation of these fixed parameters; later model selection must be disclosed.',
                                 'Pointwise and full-window coverage are distinct.',
                                 'No claim of physical truth or natural-occlusion positional accuracy.'])
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(stage='complete', elapsed_s=manifest['elapsed_s']), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path)
    parser.add_argument('--evaluate', type=Path)
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--features', nargs='+', help='Explicit feature columns; otherwise all x_* columns')
    parser.add_argument('--error-column', default='normalized_error')
    parser.add_argument('--method', choices=['static', 'conditional'], default='conditional')
    parser.add_argument('--folds', type=Path, help='Optional whole-environment folds for training residual diagnostics')
    parser.add_argument('--parameters', type=Path, help='JSON LightGBM parameter overrides, fixed before evaluation')
    parser.add_argument('--provenance', type=Path, help='Point-model residual provenance JSON, copied to manifest')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--learning-data', type=Path, help='Run fixed motion-baseline calibration from the root learning cache')
    parser.add_argument('--baseline', default='damped_fast_0.2')
    parser.add_argument('--protocol-dir', type=Path, default=Path('artifacts/c_q2/protocol'))
    parser.add_argument('--levels', type=float, nargs='+', default=[.5, .8, .9, .95])
    args = parser.parse_args()
    if args.learning_data:
        if (args.out/'manifest.json').exists():
            raise ValueError('Completed baseline probe already exists')
        run_baseline_probe(args)
        return
    if args.train is None or args.evaluate is None:
        raise ValueError('Supply --learning-data, or both --train and --evaluate')
    if args.folds and args.calibration:
        raise ValueError('For OOF inner calibration, place calibration_environments in each fold rather than supplying --calibration')
    if (args.out/'manifest.json').exists():
        raise ValueError('Completed analysis already exists; use a new output directory')
    started = time.perf_counter()
    train, evaluation = read_table(args.train), read_table(args.evaluate)
    check_partition(train, 'fit'); check_partition(evaluation, 'evaluation')
    feature_names = args.features or [c for c in train if c.startswith('x_')]
    if not feature_names:
        raise ValueError('No feature columns provided')
    if set(train.environment.astype(str)) & set(evaluation.environment.astype(str)):
        raise ValueError('Evaluation environments overlap fitting environments')
    parameters = json.loads(args.parameters.read_text()) if args.parameters else None
    model = RadiusCalibrator(args.method, levels=args.levels, parameters=parameters).fit(
        train[feature_names], train[args.error_column].to_numpy(), train)
    inputs = [args.train, args.evaluate]
    if args.calibration:
        calibration = read_table(args.calibration); check_partition(calibration, 'calibration')
        if set(calibration.environment.astype(str)) & set(evaluation.environment.astype(str)):
            raise ValueError('Evaluation environments overlap calibration environments')
        model.calibrate(calibration[feature_names], calibration[args.error_column].to_numpy(), calibration)
        inputs.append(args.calibration)
    args.out.mkdir(parents=True, exist_ok=True)
    model.save(args.out/'model')
    radii = model.predict(evaluation[feature_names])
    output = evaluation.drop(columns=feature_names).copy()
    for j, q in enumerate(model.levels):
        output[f'radius_{q:g}'] = radii[:, j]
    output.to_parquet(args.out/'evaluation_radii.parquet', index=False)
    group_by = [c for c in ('split', 'dataset', 'horizon_s', 'cap', 'delay_s', 'phase', 'policy', 'protocol') if c in evaluation]
    scores = evaluate_radii(evaluation, evaluation[args.error_column].to_numpy(), radii,
                            group_by=group_by)
    scores.to_csv(args.out/'evaluation_metrics.csv', index=False)
    fold_records = None
    if args.folds:
        folds = json.loads(args.folds.read_text())
        oof = oof_predictions(train[feature_names], train[args.error_column].to_numpy(),
                              train, folds, method=args.method, levels=args.levels, parameters=parameters)
        training = train.drop(columns=feature_names).copy()
        training['calibration_fold'] = oof['fold_assignment']
        for j, q in enumerate(model.levels):
            training[f'radius_{q:g}'] = oof['radii'][:, j]
        training.to_parquet(args.out/'training_radius_oof.parquet', index=False)
        fold_records = oof['folds']; inputs.append(args.folds)
    if args.parameters:
        inputs.append(args.parameters)
    if args.provenance:
        inputs.append(args.provenance)
    source = Path(__file__).resolve().parents[1]/'src/shuwei_c/q2_calibration.py'
    inputs.extend([Path(__file__), source])
    manifest = dict(status='development calibration complete; not final Q2 evaluation',
                    method=args.method, features=feature_names, error_column=args.error_column,
                    fit=model.fit_info, correction=model.calibration_info, folds=fold_records,
                    evaluation_rows=len(evaluation), evaluation_environments=sorted(evaluation.environment.astype(str).unique()),
                    elapsed_s=time.perf_counter()-started, test_used=False,
                    upstream_provenance=json.loads(args.provenance.read_text()) if args.provenance else 'not supplied; no upstream nesting guarantee asserted',
                    input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                    interpretation='empirical pointwise radii; path coverage and suggested intervals require separate evaluation')
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(status=manifest['status'], rows=len(evaluation), elapsed_s=manifest['elapsed_s']), indent=2))


if __name__ == '__main__':
    main()
