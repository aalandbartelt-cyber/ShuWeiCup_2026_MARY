#!/usr/bin/env python3
"""Initial causal observation-value learner and closed-loop environment validation."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import lightgbm as lgb
import numpy as np
import pandas as pd
from shuwei_c.data import Dataset, balanced_weights
from shuwei_c.q2_policy import ValuePolicy
from shuwei_c.q2_replay import Window, replay, Periodic
from shuwei_c.q2_predictors import Kinematic
from shuwei_c.q2_metrics import summarize_windows


def policy_weights(frame):
    keys = ['environment', 'episode', 'window_id', 'cap', 'delay_s']
    counts = frame.groupby(keys, observed=True, sort=False).size().rename('n').reset_index()
    counts['weight'] = balanced_weights(counts.environment, counts.episode)/counts.n
    return frame.merge(counts[keys+['weight']], on=keys, how='left', sort=False).weight.to_numpy()


def transform_value(frame, feature_names, noise, cost_weight):
    scale = frame.dataset.map({k:v['hand_scale'] for k,v in noise.items()}).to_numpy()
    x = frame[feature_names].to_numpy(float)
    for j, name in enumerate(feature_names):
        if '_length_' in name or '_speed_' in name:
            x[:, j] = np.log1p(np.expm1(x[:, j])/scale)
        elif name == 'x_cost_weight':
            x[:, j] = cost_weight
    target = frame.error_gain.to_numpy()/scale-cost_weight*frame.cost_change.to_numpy()
    return x.astype(np.float32), target


def evaluate(data, windows, noise, model, cost_weight, include_baseline=True):
    result, request_rows = [], []
    for episode, group in windows.groupby('episode', sort=False):
        rec = data.record(episode); numerical = rec.hands()
        for row in group.itertuples(index=False):
            w = Window(row.window_id, episode, rec.dataset, rec.environment, rec.split,
                       int(row.origin_index), int(row.stop), row.origin_s, row.horizon_s)
            for cap in (1, 2):
                policies = [('observation_value', ValuePolicy(model, rec.dataset,
                              noise[rec.dataset]['hand_scale'], cost_weight))]
                if include_baseline:
                    policies.append(('periodic', Periodic(cap, .5)))
                for name, policy in policies:
                    replayed = replay(rec, w, Kinematic('damped', damping=.2), policy, cap, .1, numerical=numerical)
                    metrics = dict(replayed['metrics'])
                    metrics['normalized_error'] = metrics['mean_error']/noise[rec.dataset]['hand_scale']
                    metrics['unobserved_normalized_error'] = metrics['unobserved_error']/noise[rec.dataset]['hand_scale']
                    result.append(dict(window_id=w.window_id, episode=episode, environment=rec.environment,
                                       dataset=rec.dataset, split=rec.split, horizon_s=w.horizon,
                                       cap=cap, delay_s=.1, cost_weight=cost_weight, policy=name,
                                       method='damped_0.2', **metrics))
                    for request in replayed['requests']:
                        request_rows.append(dict(window_id=w.window_id, episode=episode, policy=name,
                                                 cap=cap, cost_weight=cost_weight, **request))
    return pd.DataFrame(result), pd.DataFrame(request_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('artifacts/c_q2/value_data'))
    parser.add_argument('--out', type=Path, default=Path('artifacts/c_q2/value_initial'))
    parser.add_argument('--cost-weights', nargs='+', type=float, default=[0., .03, .1, .3])
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads((args.data/'manifest.json').read_text())
    if source_manifest['horizons'] != [.5, 1.] or source_manifest['caps'] != [1, 2] or source_manifest['delays'] != [.1]:
        raise ValueError('This initial runner is restricted to its declared development conditions')
    frames = [pd.read_parquet(p) for p in sorted(args.data.glob('*.parquet'))]
    training = pd.concat(frames, ignore_index=True)
    feature_names = [c for c in training if c.startswith('x_')]
    data = Dataset(); windows = pd.read_csv('artifacts/c_q2/protocol/development_windows.csv')
    windows = windows[windows.horizon_s.isin([.5, 1.])]
    folds = json.loads(Path('artifacts/c_q2/protocol/folds.json').read_text())
    full_noise = json.loads(Path('artifacts/c_q2/protocol/training_noise.json').read_text())
    files = [Path(__file__), Path('src/shuwei_c/q2_policy.py'), Path('src/shuwei_c/q2_replay.py'),
             Path('src/shuwei_c/q2_features.py'), Path('src/shuwei_c/q2_predictors.py'),
             args.data/'manifest.json', Path('artifacts/c_q2/protocol/folds.json')]
    identity = dict(input_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                    cost_weights=args.cost_weights, feature_names=feature_names, test_used=False,
                    policy_iteration=1, point_predictor='damped_0.2',
                    fit_target='conditional mean paired reduction of Euclidean loss minus observation cost')
    ip = args.out/'identity.json'
    if ip.exists() and json.loads(ip.read_text()) != identity:
        raise ValueError('Existing policy experiment differs')
    ip.write_text(json.dumps(identity, indent=2)+'\n')
    for p in files:
        relative = p.relative_to(Path.cwd()) if p.is_absolute() else p
        dst = args.out/'source_snapshot'/relative; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dst)
    partitions = [(f'fold{f["fold"]}', f['training_environments'], f['held_environments'], f['noise'], 'train') for f in folds]
    partitions.append(('validation', sorted(training.environment.unique()),
                       sorted(windows[windows.split=='validation'].environment.unique()), full_noise, 'validation'))
    all_results = []; started = time.perf_counter()
    for partition, fitting, held, noise, split in partitions:
        root = args.out/partition; root.mkdir(exist_ok=True)
        if (root/'complete.json').exists():
            all_results.append(pd.read_csv(root/'windows.csv')); continue
        fit = training[training.environment.isin(fitting)]
        if set(fit.environment) & set(held):
            raise ValueError('Leaked policy environments')
        held_windows = windows[(windows.split == split) & windows.environment.isin(held)]
        predictions, requests = [], []
        for cost_weight in args.cost_weights:
            x, y = transform_value(fit, feature_names, noise, cost_weight)
            weight = policy_weights(fit)
            model = lgb.LGBMRegressor(n_estimators=180, num_leaves=15, min_child_samples=50,
                                      learning_rate=.03, reg_lambda=10., n_jobs=2,
                                      objective='regression', random_state=20261120,
                                      deterministic=True, force_col_wise=True, verbosity=-1)
            model.fit(x, y, sample_weight=weight*len(weight))
            model.booster_.save_model(str(root/f'lambda{cost_weight:g}.txt'))
            result, request_log = evaluate(data, held_windows, noise, model.booster_, cost_weight)
            result['partition'] = partition; result['split'] = 'train_oof' if split == 'train' else split
            request_log['partition'] = partition
            predictions.append(result); requests.append(request_log)
            print(json.dumps(dict(partition=partition, cost_weight=cost_weight, rows=len(result),
                                  elapsed_s=time.perf_counter()-started)), flush=True)
        result = pd.concat(predictions, ignore_index=True); all_results.append(result)
        result.to_csv(root/'windows.csv', index=False)
        pd.concat(requests, ignore_index=True).to_csv(root/'requests.csv', index=False)
        (root/'complete.json').write_text(json.dumps(dict(fit_environments=fitting, held_environments=held,
             rows=len(result), noise=noise, elapsed_s=time.perf_counter()-started), indent=2)+'\n')
    frame = pd.concat(all_results, ignore_index=True); frame.to_csv(args.out/'windows.csv', index=False)
    summary = summarize_windows(frame, ['split','horizon_s','cap','delay_s','cost_weight','policy','method'])
    summary.to_csv(args.out/'summary.csv', index=False)
    (args.out/'manifest.json').write_text(json.dumps(dict(**identity, elapsed_s=time.perf_counter()-started,
        status='initial policy development; full grid, refinement, stronger predictor and calibration remain'), indent=2)+'\n')


if __name__ == '__main__':
    main()
