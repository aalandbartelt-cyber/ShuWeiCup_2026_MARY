#!/usr/bin/env python3
"""Paired observation-vs-wait labels on training records, with visible features.

Raw-coordinate gains and scale-one features are cached. Fold-specific scales
are applied by the policy trainer, so held environments do not fit a scaler.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from shuwei_c.data import Dataset
from shuwei_c.q2_replay import Window, NoRequests, Periodic, replay
from shuwei_c.q2_predictors import Kinematic
from shuwei_c.q2_policy import DecisionRecorder, ValueFeatures, counterfactual_pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('artifacts/c_q2/value_data'))
    parser.add_argument('--horizons', nargs='+', type=float, default=[.5, 1.])
    parser.add_argument('--caps', nargs='+', type=int, default=[1, 2])
    parser.add_argument('--delays', nargs='+', type=float, default=[.1])
    parser.add_argument('--decisions', type=int, default=6)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    files = [Path(__file__), Path('src/shuwei_c/q2_policy.py'), Path('src/shuwei_c/q2_replay.py'),
             Path('src/shuwei_c/q2_features.py'), Path('src/shuwei_c/q2_predictors.py'),
             Path('artifacts/c_q2/protocol/development_windows.csv')]
    identity = dict(horizons=args.horizons, caps=args.caps, delays=args.delays,
                    decisions=args.decisions, initial_policy='no-request and periodic .5 state exploration',
                    continuation='remaining-budget periodic .5', predictor='last-two-qualified damped velocity, tau=.2',
                    input_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, test_used=False)
    ip = args.out/'identity.json'
    if ip.exists() and json.loads(ip.read_text()) != identity:
        raise ValueError('Existing value cache differs; use new output directory')
    ip.write_text(json.dumps(identity, indent=2)+'\n')
    for p in files:
        relative = p.relative_to(Path.cwd()) if p.is_absolute() else p
        target = args.out/'source_snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, target)
    windows = pd.read_csv('artifacts/c_q2/protocol/development_windows.csv')
    windows = windows[(windows.split == 'train') & windows.horizon_s.isin(args.horizons)]
    data = Dataset(); started = time.perf_counter(); completed = []
    for episode, group in windows.groupby('episode', sort=False):
        marker = args.out/f'{episode}.json'
        if marker.exists():
            info = json.loads(marker.read_text())
            output = args.out/f'{episode}.parquet'
            if hashlib.sha256(output.read_bytes()).hexdigest() != info['sha256']:
                raise ValueError('Completed value cache changed')
            completed.append(info); continue
        rec = data.record(episode); numerical = rec.hands(); rows = []
        for row in group.itertuples(index=False):
            window = Window(row.window_id, episode, rec.dataset, rec.environment, rec.split,
                            int(row.origin_index), int(row.stop), row.origin_s, row.horizon_s)
            for cap in args.caps:
                for delay in args.delays:
                    states = {}
                    for base in (NoRequests(), Periodic(cap, .5)):
                        recorder = DecisionRecorder(base)
                        replay(rec, window, Kinematic('damped', damping=.2), recorder, cap, delay, numerical=numerical)
                        for context, prefix in recorder.states:
                            states[(context.now, prefix)] = (context, prefix)
                    values = list(states.values())
                    if len(values) > args.decisions:
                        values = [values[i] for i in np.unique(np.rint(np.linspace(0, len(values)-1, args.decisions)).astype(int))]
                    features = ValueFeatures(rec.dataset, 1.)
                    for context, prefix in values:
                        x, names = features(context)
                        label = counterfactual_pair(rec, window, context, prefix, cap, delay, 1., numerical=numerical)
                        item = dict(episode=episode, environment=rec.environment, dataset=rec.dataset,
                                    split='train', window_id=window.window_id, horizon_s=window.horizon,
                                    cap=cap, delay_s=delay, time_s=context.now,
                                    issued_prefix=json.dumps(prefix), **label)
                        item.update({'x_'+name: float(value) for name, value in zip(names, x)})
                        rows.append(item)
        frame = pd.DataFrame(rows); output = args.out/f'{episode}.parquet'; temporary = output.with_suffix('.tmp')
        frame.to_parquet(temporary, index=False); temporary.replace(output)
        info = dict(episode=episode, rows=len(rows), windows=len(group),
                    sha256=hashlib.sha256(output.read_bytes()).hexdigest(), elapsed_s=time.perf_counter()-started)
        marker.write_text(json.dumps(info, indent=2)+'\n'); completed.append(info)
        print(json.dumps(info), flush=True)
    manifest = dict(**identity, rows=sum(r['rows'] for r in completed), records=len(completed),
                    elapsed_s=time.perf_counter()-started,
                    target='mean Euclidean error difference over the same complete window; cost/wait changes separately')
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
