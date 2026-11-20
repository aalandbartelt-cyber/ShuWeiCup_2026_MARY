#!/usr/bin/env python3
"""Cache causal sparse training features; reference labels are joined afterwards."""
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
from shuwei_c.q2_replay import Window, replay, Periodic
from shuwei_c.q2_predictors import Kinematic
from shuwei_c.q2_features import CausalExperts, EXPERT_NAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'validation'], required=True)
    parser.add_argument('--sample-stride', type=int, default=1)
    parser.add_argument('--out', type=Path, default=Path('artifacts/c_q2/learning_data'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path('configs/c_q2.json').read_text())
    windows = pd.read_csv('artifacts/c_q2/protocol/development_windows.csv')
    windows = windows[windows.split == args.split]
    folder = args.out/args.split; folder.mkdir(exist_ok=True)
    sources = [Path(__file__), Path('src/shuwei_c/q2_features.py'), Path('src/shuwei_c/q2_replay.py'), Path('configs/c_q2.json')]
    identity = dict(source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    split=args.split, sample_stride=args.sample_stride, training_phases='deterministic window/scenario hash',
                    validation_phase=.5, stress_in_training=True, test_used=False)
    ip = folder/'identity.json'
    if ip.exists() and json.loads(ip.read_text()) != identity:
        raise ValueError('Cache identity changed; use new output directory')
    ip.write_text(json.dumps(identity, indent=2)+'\n')
    data = Dataset(); started = time.perf_counter(); completed = []
    for episode, group in windows.groupby('episode', sort=False):
        marker = folder/f'{episode}.json'
        if marker.exists():
            completed.append(json.loads(marker.read_text())); continue
        rec = data.record(episode); t, points, bad = rec.hands(); centres = points.mean(axis=2)
        features, experts, targets, anchors, metadata = [], [], [], [], []
        feature_names = None
        for row in group.itertuples(index=False):
            window = Window(row.window_id, episode, rec.dataset, rec.environment, rec.split,
                            int(row.origin_index), int(row.stop), row.origin_s, row.horizon_s)
            for cap in config['request_caps']:
                for delay in ([.1] if cap == 0 else [.1, 0., .3]):
                    scenario = f'b{cap}_d{delay:g}'
                    seed = int(hashlib.sha256((window.window_id+scenario).encode()).hexdigest()[:8], 16)
                    phase = float(np.random.default_rng(seed).choice(config['periodic_phases'])) if rec.split == 'train' else .5
                    collector = []; extractor = CausalExperts(rec.dataset)
                    def observe(ctx, prediction):
                        i = len(collector)
                        if i % args.sample_stride == 0:
                            collector.append((ctx.now, extractor.at(ctx.history, ctx.now)))
                        else:
                            collector.append(None)
                    output = replay(rec, window, Kinematic(), Periodic(cap, phase), cap, delay, observer=observe)
                    for k, value in enumerate(collector):
                        if value is None:
                            continue
                        now, pack = value
                        # This join is outside the observer/predictor boundary.
                        target_index = window.origin_index+1+k
                        target = centres[target_index]-pack['anchor']
                        features.append(pack['features']); experts.append(pack['experts'])
                        targets.append(target.astype(np.float32)); anchors.append(pack['anchor'].astype(np.float32))
                        feature_names = pack['feature_names']
                        for hand in range(2):
                            metadata.append(dict(episode=episode, environment=rec.environment, dataset=rec.dataset,
                                                 split=rec.split, window_id=window.window_id, horizon_s=window.horizon,
                                                 origin_s=window.origin, time_s=now, hand=hand, cap=cap, delay_s=delay,
                                                 phase=phase, scenario=scenario, age_s=pack['age'][hand],
                                                 current_direct=output['steps'][k]['current_direct']))
        arrays = dict(features=np.concatenate(features), experts=np.concatenate(experts),
                      target=np.concatenate(targets), anchor=np.concatenate(anchors))
        np.savez_compressed(folder/f'{episode}.npz', **arrays)
        pd.DataFrame(metadata).to_parquet(folder/f'{episode}.parquet', index=False)
        info = dict(episode=episode, dataset=rec.dataset, rows=len(metadata), windows=len(group),
                    feature_names=feature_names, expert_names=EXPERT_NAMES, elapsed_s=time.perf_counter()-started)
        marker.write_text(json.dumps(info, indent=2)+'\n'); completed.append(info)
        print(json.dumps({k: info[k] for k in ('dataset', 'rows', 'windows', 'elapsed_s')}), flush=True)
    manifest = dict(**identity, records=len(completed), rows=sum(x['rows'] for x in completed),
                    elapsed_s=time.perf_counter()-started, feature_names=completed[0]['feature_names'], expert_names=EXPERT_NAMES)
    (folder/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
