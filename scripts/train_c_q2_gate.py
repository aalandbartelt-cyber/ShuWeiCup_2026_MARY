#!/usr/bin/env python3
"""Sparse Q2 gate screening/refitting, without reading official test records."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import torch
from shuwei_c.q2_learning import load_pack, subset, fit_gate, predict_gate, score_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('artifacts/c_q2/learning_data'))
    parser.add_argument('--out', type=Path, default=Path('artifacts/c_q2/gate_screen'))
    parser.add_argument('--kinds', nargs='+', default=['mixture', 'residual'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[20261120])
    parser.add_argument('--epochs', type=int, default=80)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'manifest.json').exists():
        raise ValueError('Completed experiment; choose a new output directory')
    train = load_pack(args.data/'train'); validation = load_pack(args.data/'validation')
    folds = json.loads(Path('artifacts/c_q2/protocol/folds.json').read_text())
    noise = json.loads(Path('artifacts/c_q2/protocol/training_noise.json').read_text())
    inner = folds[0]
    keep = ~train['meta'].environment.isin(inner['held_environments']).to_numpy()
    fitting, held = subset(train, keep), subset(train, ~keep)
    torch.cuda.set_per_process_memory_fraction(.45)
    sources = [Path(__file__), Path('src/shuwei_c/q2_learning.py'), Path('src/shuwei_c/q2_features.py')]
    identity = dict(data=str(args.data), source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    data_manifests={split: hashlib.sha256((args.data/split/'manifest.json').read_bytes()).hexdigest() for split in ('train','validation')},
                    seeds=args.seeds, kinds=args.kinds, max_epochs=args.epochs,
                    inner_environments=inner['held_environments'], test_used=False,
                    status='development screen; whole-pipeline OOF still required')
    (args.out/'identity.json').write_text(json.dumps(identity, indent=2)+'\n')
    summaries = []; started = time.perf_counter()
    for kind in args.kinds:
        for seed in args.seeds:
            key = f'{kind}_seed{seed}'
            marker = args.out/f'{key}.json'
            if marker.exists():
                summaries.append(pd.read_csv(args.out/f'{key}_summary.csv')); continue
            selected = fit_gate(fitting, inner['noise'], held, kind=kind, seed=seed, epochs=args.epochs)
            torch.save(selected, args.out/f'{key}_inner.pt')
            model = fit_gate(train, noise, kind=kind, seed=seed, epochs=selected['epochs'])
            torch.save(model, args.out/f'{key}.pt')
            prediction = predict_gate(validation, model)
            frame, summary = score_predictions(validation, prediction, noise)
            frame['method'] = kind; frame['seed'] = seed
            for k, coordinate in enumerate('xyz'):
                frame['prediction_'+coordinate] = prediction[:, k]+validation['anchor'][:, k]
            frame.to_parquet(args.out/f'{key}_validation.parquet', index=False)
            summary['method'] = kind; summary['seed'] = seed
            summary.to_csv(args.out/f'{key}_summary.csv', index=False); summaries.append(summary)
            record = dict(method=kind, seed=seed, epochs=selected['epochs'],
                          inner_best=min(r['inner_validation_error'] for r in selected['trace']),
                          elapsed_s=time.perf_counter()-started)
            marker.write_text(json.dumps(record, indent=2)+'\n'); print(json.dumps(record), flush=True)
    pd.concat(summaries, ignore_index=True).to_csv(args.out/'comparison.csv', index=False)
    (args.out/'manifest.json').write_text(json.dumps(dict(**identity, elapsed_s=time.perf_counter()-started), indent=2)+'\n')


if __name__ == '__main__':
    main()
