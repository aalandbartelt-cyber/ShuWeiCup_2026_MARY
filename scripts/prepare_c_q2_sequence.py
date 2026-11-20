#!/usr/bin/env python3
"""Match masked sequence states to every existing numeric-learning row.

Repeated histories are stored once per record; current-query gaps are stored
per row. This keeps sequence caches small without dropping any numeric rows.
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
from shuwei_c.q2_replay import Window, replay, Periodic
from shuwei_c.q2_predictors import Kinematic
from shuwei_c.q2_features import CausalExperts
from shuwei_c.q2_sequence import history_sequence, STEPS, STEP_S, SEQUENCE_FIELDS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'validation'], required=True)
    parser.add_argument('--learning-data', type=Path, default=Path('artifacts/c_q2/learning_data'))
    parser.add_argument('--out', type=Path, default=Path('artifacts/c_q2/sequence/data'))
    args = parser.parse_args()
    source_folder = args.learning_data/args.split
    if not (source_folder/'manifest.json').exists():
        raise ValueError('Wait for complete numeric learning cache')
    out = args.out/args.split
    out.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), Path('src/shuwei_c/q2_sequence.py'), Path('src/shuwei_c/q2_replay.py'),
               Path('src/shuwei_c/q2_features.py'), Path('src/shuwei_c/data.py')]
    identity = dict(split=args.split, steps=STEPS, step_s=STEP_S, fields=SEQUENCE_FIELDS,
                    numeric_manifest=sha(source_folder/'manifest.json'),
                    windows=sha('artifacts/c_q2/protocol/development_windows.csv'),
                    source_hashes={str(p): sha(p) for p in sources}, test_used=False)
    identity_path = out/'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Sequence identity changed; choose a new output directory')
    identity_path.write_text(json.dumps(identity, indent=2)+'\n')
    data = Dataset()
    windows = pd.read_csv('artifacts/c_q2/protocol/development_windows.csv').set_index('window_id')
    started = time.perf_counter()
    records = []
    for path in sorted(source_folder.glob('*.parquet')):
        episode = path.stem
        target, marker = out/f'{episode}.npz', out/f'{episode}.json'
        if marker.exists():
            info = json.loads(marker.read_text())
            if info['sequence_sha256'] != sha(target) or info['metadata_sha256'] != sha(path) or info['numeric_npz_sha256'] != sha(path.with_suffix('.npz')):
                raise ValueError('Sequence cached inputs or payload changed')
            records.append(info)
            continue
        meta = pd.read_parquet(path)
        rec = data.record(episode)
        numerical = rec.hands()
        if rec.split != args.split or rec.split not in ('train', 'validation'):
            raise ValueError('Unexpected split')
        index = np.full(len(meta), -1, np.int64)
        query_gap = np.full(len(meta), np.nan, np.float32)
        sequence_states, state_lookup, anchors = [], {}, []
        anchor_mismatch = 0.
        with np.load(path.with_suffix('.npz')) as stored:
            expected_anchor = stored['anchor']
        for (window_id, scenario), group in meta.groupby(['window_id', 'scenario'], sort=False):
            w = windows.loc[window_id]
            first = group.iloc[0]
            window = Window(window_id, episode, rec.dataset, rec.environment, rec.split,
                            int(w.origin_index), int(w.stop), float(w.origin_s), float(w.horizon_s))
            row_lookup = {(float(r.time_s), int(r.hand)): int(i) for i, r in group.iterrows()}
            extractor = CausalExperts(rec.dataset)

            def observe(ctx, prediction):
                nonlocal anchor_mismatch
                if (ctx.now, 0) not in row_lookup:
                    return
                source_times = ctx.history.time[ctx.history.time > window.origin+1e-9]
                key = (window.origin_index, *source_times.tolist())
                if key not in state_lookup:
                    extractor.prepare(ctx.history)
                    anchor = extractor.anchor
                    state_lookup[key] = len(sequence_states)
                    sequence_states.append(history_sequence(ctx.history, anchor))
                    anchors.append(anchor.copy())
                else:
                    anchor = anchors[state_lookup[key]]
                for hand in range(2):
                    row = row_lookup[(ctx.now, hand)]
                    index[row] = 2*state_lookup[key]+hand
                    query_gap[row] = max(0., ctx.now-ctx.history.time[-1])
                    anchor_mismatch = max(anchor_mismatch, float(np.max(np.abs(anchor[hand].astype(np.float32)-expected_anchor[row]))))

            replay(rec, window, Kinematic(), Periodic(int(first.cap), float(first.phase)),
                   int(first.cap), float(first.delay_s), observer=observe, numerical=numerical)
        if np.any(index < 0) or not np.isfinite(query_gap).all() or anchor_mismatch > 1e-7:
            raise ValueError(f'Sequence join mismatch: {episode} {anchor_mismatch}')
        values = np.concatenate([s['values'] for s in sequence_states])
        mask = np.concatenate([s['mask'] for s in sequence_states])
        elapsed = np.concatenate([s['elapsed'] for s in sequence_states])
        step_dt = np.concatenate([s['step_dt'] for s in sequence_states])
        temporary = target.with_suffix('.tmp.npz')
        np.savez_compressed(temporary, values=values, mask=mask, elapsed=elapsed, step_dt=step_dt,
                            sequence_index=index, query_gap=query_gap)
        temporary.replace(target)
        info = dict(episode=episode, dataset=rec.dataset, split=rec.split,
                    rows=len(meta), hand_sequence_states=len(values),
                    anchor_max_difference=anchor_mismatch,
                    metadata_sha256=sha(path), numeric_npz_sha256=sha(path.with_suffix('.npz')),
                    sequence_sha256=sha(target), elapsed_s=time.perf_counter()-started)
        marker.write_text(json.dumps(info, indent=2)+'\n')
        records.append(info)
        print(json.dumps(info), flush=True)
    manifest = dict(**identity, records=len(records), rows=sum(r['rows'] for r in records),
                    hand_sequence_states=sum(r['hand_sequence_states'] for r in records),
                    elapsed_s=time.perf_counter()-started)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
