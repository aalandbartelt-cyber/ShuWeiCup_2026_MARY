#!/usr/bin/env python3
"""Matched Q2 numerical baselines on fixed train/validation windows only.

Same disclosed trajectory is shared by all predictors for a fixed schedule.
Prefix Kalman states are causal cached states, never smoothed estimates.
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
from shuwei_c.data import Dataset
from shuwei_c.motion import filter_record
from shuwei_c.q2_replay import Window, replay, Periodic
from shuwei_c.q2_predictors import Kinematic, Kalman
from shuwei_c.q2_metrics import summarize_windows, prediction_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'validation'], default='validation')
    parser.add_argument('--phases', type=float, nargs='+', default=[.5])
    parser.add_argument('--delays', type=float, nargs='+')
    parser.add_argument('--limit-windows', type=int)
    parser.add_argument('--out', type=Path, default=Path('artifacts/c_q2/baselines'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path('configs/c_q2.json').read_text())
    noise = json.loads(Path('artifacts/c_q2/protocol/training_noise.json').read_text())
    frame = pd.read_csv('artifacts/c_q2/protocol/development_windows.csv')
    frame = frame[frame.split == args.split]
    if args.limit_windows:
        frame = frame.head(args.limit_windows)
    data = Dataset(); started = time.perf_counter(); results = []
    delays = args.delays or [config['main_delay_s']]+config['sensitivity_delays_s']
    prefix = args.split
    by_record = args.out/f'{prefix}_records'; by_record.mkdir(exist_ok=True)
    sources = [Path(__file__), Path('configs/c_q2.json'), Path('src/shuwei_c/q2_replay.py'),
               Path('src/shuwei_c/q2_predictors.py'), Path('src/shuwei_c/q2_metrics.py'),
               Path('src/shuwei_c/data.py'), Path('src/shuwei_c/motion.py'),
               Path('artifacts/c_q2/protocol/development_windows.csv'),
               Path('artifacts/c_q2/protocol/training_noise.json')]
    signature = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    identity = dict(source_hashes=signature, split=args.split, phases=args.phases,
                    delays=delays, limit_windows=args.limit_windows)
    identity_path = args.out/f'{prefix}_identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Existing baseline run differs; use a new output directory')
    identity_path.write_text(json.dumps(identity, indent=2)+'\n')
    for episode, group in frame.groupby('episode', sort=False):
        target = by_record/f'{episode}.csv'
        marker = target.with_suffix('.json')
        expected_rows = len(group)*len(delays)*(1+3*len(args.phases))*(3+len(config['damping_times_s']))
        if target.exists() and marker.exists():
            record = json.loads(marker.read_text())
            if hashlib.sha256(target.read_bytes()).hexdigest() != record['sha256']:
                raise ValueError('Completed record cache hash changed')
            cached = pd.read_csv(target)
            keys = ['window_id','cap','delay_s','phase','method']
            if len(cached) != expected_rows or cached.duplicated(keys).any() or set(cached.window_id) != set(group.window_id):
                raise ValueError('Completed record cache keys/counts disagree')
            results.extend(cached.to_dict('records'))
            continue
        rec = data.record(episode); t, p, bad = rec.hands(); c = p.mean(axis=2)
        states, covariances = filter_record(rec, noise[rec.dataset])
        local = []
        for row in group.itertuples(index=False):
            window = Window(row.window_id, episode, rec.dataset, rec.environment,
                            rec.split, int(row.origin_index), int(row.stop),
                            row.origin_s, row.horizon_s)
            for cap in config['request_caps']:
                for delay in delays:
                    for phase in args.phases if cap else [args.phases[0]]:
                        contexts = []
                        output = replay(rec, window, Kinematic(), Periodic(cap, phase), cap, delay,
                                        observer=lambda ctx, pred: contexts.append(ctx), numerical=(t, p, bad))
                        predictors = {'constant': Kinematic(), 'linear': Kinematic('linear'),
                                      'kalman': Kalman.from_prefix(noise[rec.dataset],
                                          states[window.origin_index], covariances[window.origin_index], window.origin)}
                        for tau in config['damping_times_s']:
                            predictors[f'damped_{tau:g}'] = Kinematic('damped', damping=tau)
                        reference = c[window.origin_index+1:window.stop]
                        direct = [r['current_direct'] for r in output['steps']]
                        for name, predictor in predictors.items():
                            pred = np.array([predictor(ctx.history, ctx.now) for ctx in contexts])
                            metrics = dict(output['metrics'])
                            metrics.update(prediction_metrics(pred, reference, direct, noise[rec.dataset]['hand_scale']))
                            local.append(dict(window_id=window.window_id, split=rec.split, dataset=rec.dataset,
                                              environment=rec.environment, episode=episode, origin_s=window.origin,
                                              horizon_s=window.horizon, cap=cap, delay_s=delay,
                                              phase=phase, method=name, policy='periodic',
                                              protocol='all_modalities_hidden', **metrics))
        if len(local) != expected_rows:
            raise ValueError('Incomplete record result')
        temporary = target.with_suffix('.tmp')
        pd.DataFrame(local).to_csv(temporary, index=False); temporary.replace(target)
        complete = dict(rows=len(local), windows=len(group), sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        marker_tmp = marker.with_suffix('.tmp')
        marker_tmp.write_text(json.dumps(complete, indent=2)+'\n'); marker_tmp.replace(marker)
        results.extend(local)
        print(json.dumps(dict(split=rec.split, records_done=len(list(by_record.glob('*.csv'))),
                              windows=len(group), rows=len(local), elapsed_s=time.perf_counter()-started)), flush=True)
    result = pd.DataFrame(results)
    result.to_csv(args.out/f'{prefix}_windows.csv', index=False)
    summary = summarize_windows(result, ['split', 'horizon_s', 'cap', 'delay_s', 'phase', 'method', 'policy', 'protocol'])
    summary.to_csv(args.out/f'{prefix}_summary.csv', index=False)
    manifest = dict(**identity, test_used=False, elapsed_s=time.perf_counter()-started,
                    rows=len(result), windows=result.window_id.nunique(),
                    status='baseline development; no learned model or final calibration yet')
    (args.out/f'{prefix}_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
